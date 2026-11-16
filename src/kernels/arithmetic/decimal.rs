// Copyright 2025 Peter Garfield Bower
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Arithmetic kernels for decimal arrays with overflow detection.
//!
//! All operations use checked arithmetic and return `KernelError::Overflow`
//! when the result exceeds the integer width. Scale reconciliation and
//! result-precision computation are handled here so the caller passes
//! `DecimalArray` pairs and receives a correctly-typed `DecimalArray` result.

use crate::enums::error::KernelError;
use crate::enums::operators::ArithmeticOperator;
use crate::kernels::bitmask::merge_bitmasks_to_new;
use crate::traits::type_unions::Integer;
use crate::{Bitmask, DecimalArray, MaskedArray, Vec64};

/// Maximum precision per backing integer width.
pub(crate) fn max_precision<T: Integer + 'static>() -> u8 {
    use std::any::TypeId;
    let tid = TypeId::of::<T>();
    if tid == TypeId::of::<i32>() {
        9
    } else if tid == TypeId::of::<i64>() {
        18
    } else {
        38
    }
}

/// Compute `10^exp` as T using checked multiplication.
///
/// Returns `None` if the result overflows the backing integer width.
fn checked_pow10<T: Integer>(exp: u32) -> Option<T> {
    let ten = T::from_usize(10);
    let mut acc = T::one();
    for _ in 0..exp {
        acc = acc.checked_mul(&ten)?;
    }
    Some(acc)
}

/// Element-wise checked binary arithmetic on two `DecimalArray<T>` operands.
///
/// Reconciles scale for add/subtract, computes result scale for
/// multiply/divide, and detects overflow across all operations.
/// The merged null mask from both operands propagates through the result.
/// See [`decimal_binary_into`] for the scale rules.
pub fn decimal_binary<T: Integer + 'static>(
    lhs: &DecimalArray<T>,
    rhs: &DecimalArray<T>,
    op: ArithmeticOperator,
) -> Result<DecimalArray<T>, KernelError> {
    let len = lhs.len();
    if len != rhs.len() {
        return Err(KernelError::LengthMismatch(format!(
            "decimal_binary: lhs {} vs rhs {}",
            len,
            rhs.len()
        )));
    }

    let merged_mask = merge_bitmasks_to_new(
        lhs.null_mask.as_ref(),
        rhs.null_mask.as_ref(),
        len,
    );

    // Division and remainder mark zero divisors null, which requires an output
    // mask even when neither operand has one.
    let zero_divisor_nulls = matches!(
        op,
        ArithmeticOperator::Divide | ArithmeticOperator::FloorDiv | ArithmeticOperator::Remainder
    );
    let mut out_mask = match &merged_mask {
        Some(m) => Some(m.clone()),
        None if zero_divisor_nulls => Some(Bitmask::new_set_all(len, true)),
        None => None,
    };

    let mut out = Vec64::<T>::with_capacity(len);
    unsafe { out.set_len(len) };

    let (precision, scale) = decimal_binary_into(
        lhs.data.as_slice(),
        (lhs.precision, lhs.scale),
        rhs.data.as_slice(),
        (rhs.precision, rhs.scale),
        op,
        merged_mask.as_ref(),
        &mut out,
        out_mask.as_mut(),
    )?;

    // The result keeps an output mask only when it records a null.
    if merged_mask.is_none() && out_mask.as_ref().is_some_and(|m| m.all_set()) {
        out_mask = None;
    }

    Ok(DecimalArray::new(out, out_mask, precision, scale))
}

/// Element-wise checked binary arithmetic on decimal storage slices, writing
/// into the caller-provided `out` slice.
///
/// `lhs_type` and `rhs_type` hold the `(precision, scale)` of each operand.
/// Operands of different widths are widened into the output width `T` through
/// `From`, which allows, for example, a `Decimal32` operand against a
/// `Decimal64` operand to write `i64` output. Returns the `(precision, scale)`
/// of the result.
///
/// ## Scale rules
/// - Add/Subtract: operands are rescaled to the finer scale. Result
///   precision is `max(p1, p2) + 1`, capped at the width maximum.
/// - Multiply: result scale = s1 + s2, result precision = p1 + p2,
///   capped at the width maximum.
/// - Divide: the numerator is scaled up so integer division preserves
///   the desired number of fractional digits. Result scale is max of
///   the two input scales. `FloorDiv` follows the same rule.
/// - Remainder: operands are rescaled to the finer scale.
/// - Power returns `KernelError::UnsupportedType`.
///
/// ## Null handling
/// - `mask` is the merged null mask of both operands. Rows that it marks null
///   write zero.
/// - When `out_mask` is supplied, its first `len` positions are written: the
///   bits of `mask`, or valid without one, with zero divisors marked null.
/// - A zero divisor without an `out_mask` returns `KernelError::InvalidArguments`.
///
/// ## Errors
/// Overflow returns `KernelError::Overflow`. Operands and `out` of different
/// lengths, and masks shorter than the operands, return
/// `KernelError::LengthMismatch`. On error, `out` holds a partial result.
pub fn decimal_binary_into<L, R, T>(
    lhs: &[L],
    lhs_type: (u8, i8),
    rhs: &[R],
    rhs_type: (u8, i8),
    op: ArithmeticOperator,
    mask: Option<&Bitmask>,
    out: &mut [T],
    mut out_mask: Option<&mut Bitmask>,
) -> Result<(u8, i8), KernelError>
where
    L: Copy,
    R: Copy,
    T: Integer + From<L> + From<R>,
{
    let len = lhs.len();
    if len != rhs.len() || len != out.len() {
        return Err(KernelError::LengthMismatch(format!(
            "decimal_binary_into: lhs {} vs rhs {} vs out {}",
            len,
            rhs.len(),
            out.len()
        )));
    }
    if mask.is_some_and(|m| m.len() < len) || out_mask.as_deref().is_some_and(|m| m.len() < len) {
        return Err(KernelError::LengthMismatch(format!(
            "decimal_binary_into: a null mask is shorter than the {len} operand rows"
        )));
    }

    let (lp, ls) = lhs_type;
    let (rp, rs) = rhs_type;
    let max_p = max_precision::<T>();
    let finer_scale = ls.max(rs);

    // Result type, and the power-of-ten factors applied to each operand before
    // the operation.
    let (result_type, lhs_exp, rhs_exp) = match op {
        ArithmeticOperator::Add | ArithmeticOperator::Subtract => (
            ((lp.max(rp) + 1).min(max_p), finer_scale),
            finer_scale as i32 - ls as i32,
            finer_scale as i32 - rs as i32,
        ),
        ArithmeticOperator::Remainder => (
            (lp.max(rp).min(max_p), finer_scale),
            finer_scale as i32 - ls as i32,
            finer_scale as i32 - rs as i32,
        ),
        ArithmeticOperator::Multiply => (((lp + rp).min(max_p), ls + rs), 0, 0),
        ArithmeticOperator::Divide | ArithmeticOperator::FloorDiv => (
            (lp.max(rp).min(max_p), finer_scale),
            finer_scale as i32 - ls as i32 + rs as i32,
            0,
        ),
        ArithmeticOperator::Power => {
            return Err(KernelError::UnsupportedType(
                "Power is not supported for decimal arrays - consider converting to float64 first"
                    .to_string(),
            ));
        }
    };
    let lhs_factor = u32::try_from(lhs_exp)
        .ok()
        .and_then(checked_pow10::<T>)
        .ok_or_else(|| overflow_error("lhs scale factor"))?;
    let rhs_factor = u32::try_from(rhs_exp)
        .ok()
        .and_then(checked_pow10::<T>)
        .ok_or_else(|| overflow_error("rhs scale factor"))?;

    // The output mask receives the bits of `mask` in bulk, or all valid without
    // one, and the loop clears the bit of each zero divisor.
    if let Some(out_mask) = out_mask.as_deref_mut() {
        match mask {
            Some(mask) => {
                let whole_bytes = len / 8;
                out_mask.bits.as_mut_slice()[..whole_bytes]
                    .copy_from_slice(&mask.bits.as_slice()[..whole_bytes]);
                for i in whole_bytes * 8..len {
                    out_mask.set(i, mask.get(i));
                }
            }
            None => out_mask.set_range(0, len, true),
        }
    }

    // Each operator runs its own loop over the rows that `mask` leaves valid,
    // with `$value` producing the result of row `$i`, or `None` for a zero
    // divisor.
    macro_rules! run {
        (|$i:ident| $value:expr) => {{
            for $i in 0..len {
                if mask.is_some_and(|m| !m.get($i)) {
                    out[$i] = T::zero();
                    continue;
                }
                match $value {
                    Some(v) => out[$i] = v,
                    None => {
                        out[$i] = T::zero();
                        match out_mask.as_deref_mut() {
                            Some(m) => m.set($i, false),
                            None => {
                                return Err(KernelError::InvalidArguments(
                                    "decimal_binary_into: a zero divisor requires an output mask"
                                        .to_string(),
                                ));
                            }
                        }
                    }
                }
            }
        }};
    }
    // Operands widen to `T`, and multiply by their scale factor when the
    // operator rescales them.
    macro_rules! lhs_at {
        ($i:expr) => {
            <T as From<L>>::from(lhs[$i])
                .checked_mul(&lhs_factor)
                .ok_or_else(|| overflow_error("lhs rescale"))?
        };
    }
    macro_rules! rhs_at {
        ($i:expr) => {
            <T as From<R>>::from(rhs[$i])
                .checked_mul(&rhs_factor)
                .ok_or_else(|| overflow_error("rhs rescale"))?
        };
    }
    let lhs_raw = |i: usize| <T as From<L>>::from(lhs[i]);
    let rhs_raw = |i: usize| <T as From<R>>::from(rhs[i]);
    let same_scale = lhs_factor == T::one() && rhs_factor == T::one();

    match op {
        ArithmeticOperator::Add if same_scale => run!(|i| Some(
            lhs_raw(i).checked_add(&rhs_raw(i)).ok_or_else(|| overflow_error("add"))?
        )),
        ArithmeticOperator::Add => run!(|i| Some(
            lhs_at!(i).checked_add(&rhs_at!(i)).ok_or_else(|| overflow_error("add"))?
        )),
        ArithmeticOperator::Subtract if same_scale => run!(|i| Some(
            lhs_raw(i).checked_sub(&rhs_raw(i)).ok_or_else(|| overflow_error("subtract"))?
        )),
        ArithmeticOperator::Subtract => run!(|i| Some(
            lhs_at!(i).checked_sub(&rhs_at!(i)).ok_or_else(|| overflow_error("subtract"))?
        )),
        // Multiply does not rescale its operands
        ArithmeticOperator::Multiply => run!(|i| Some(
            lhs_raw(i).checked_mul(&rhs_raw(i)).ok_or_else(|| overflow_error("multiply"))?
        )),
        // A zero divisor produces null before the numerator is scaled up,
        // and a scale-up overflow on that row does not return an error. The
        // divisor is not rescaled.
        ArithmeticOperator::Divide | ArithmeticOperator::FloorDiv => run!(|i| {
            let r = rhs_raw(i);
            if r == T::zero() {
                None
            } else {
                Some(lhs_at!(i).checked_div(&r).ok_or_else(|| overflow_error("divide"))?)
            }
        }),
        // The remainder is derived from the checked quotient, because `PrimInt`
        // does not include a checked remainder. The quotient overflows only for
        // the minimum value divided by -1, where the remainder is 0. A zero
        // divisor produces null before the numerator is rescaled, as for
        // division.
        ArithmeticOperator::Remainder if same_scale => run!(|i| {
            let (l, r) = (lhs_raw(i), rhs_raw(i));
            if r == T::zero() { None } else { Some(l.checked_div(&r).map_or(T::zero(), |q| l - q * r)) }
        }),
        ArithmeticOperator::Remainder => run!(|i| {
            let r = rhs_at!(i);
            if r == T::zero() {
                None
            } else {
                let l = lhs_at!(i);
                Some(l.checked_div(&r).map_or(T::zero(), |q| l - q * r))
            }
        }),
        ArithmeticOperator::Power => unreachable!("Power returns before the element loop"),
    }

    Ok(result_type)
}

/// Element-wise negate on a decimal array. Overflow is only possible when
/// negating the minimum value of the backing integer, for e.g. i32::MIN.
pub fn decimal_negate<T: Integer + 'static>(
    arr: &DecimalArray<T>,
) -> Result<DecimalArray<T>, KernelError> {
    let len = arr.len();
    let mut out = Vec64::<T>::with_capacity(len);
    unsafe { out.set_len(len) };

    for i in 0..len {
        if arr.is_null(i) {
            out[i] = T::zero();
            continue;
        }
        out[i] = T::zero()
            .checked_sub(&arr.data[i])
            .ok_or_else(|| overflow_error("negate"))?;
    }

    Ok(DecimalArray::new(out, arr.null_mask.clone(), arr.precision, arr.scale))
}

/// Element-wise absolute value on a decimal array.
pub fn decimal_abs<T: Integer + 'static>(
    arr: &DecimalArray<T>,
) -> Result<DecimalArray<T>, KernelError> {
    let len = arr.len();
    let mut out = Vec64::<T>::with_capacity(len);
    unsafe { out.set_len(len) };

    for i in 0..len {
        if arr.is_null(i) {
            out[i] = T::zero();
            continue;
        }
        let v = arr.data[i];
        if v < T::zero() {
            out[i] = T::zero()
                .checked_sub(&v)
                .ok_or_else(|| overflow_error("abs"))?;
        } else {
            out[i] = v;
        }
    }

    Ok(DecimalArray::new(out, arr.null_mask.clone(), arr.precision, arr.scale))
}

/// Rescale a decimal array to a new scale, multiplying or dividing the raw
/// values by 10^|new_scale - old_scale|.
///
/// Scaling up multiplies raw values by the power-of-ten difference using
/// checked arithmetic. Scaling down divides and truncates.
pub fn decimal_rescale<T: Integer + 'static>(
    arr: &DecimalArray<T>,
    new_scale: i8,
) -> Result<DecimalArray<T>, KernelError> {
    if new_scale == arr.scale {
        return Ok(arr.clone());
    }

    let len = arr.len();
    let diff = (new_scale as i32 - arr.scale as i32).unsigned_abs();
    let factor = checked_pow10::<T>(diff)
        .ok_or_else(|| overflow_error("rescale factor"))?;

    let mut out = Vec64::<T>::with_capacity(len);
    unsafe { out.set_len(len) };

    if new_scale > arr.scale {
        // Scale up: multiply by factor
        for i in 0..len {
            if arr.is_null(i) {
                out[i] = T::zero();
                continue;
            }
            out[i] = arr.data[i]
                .checked_mul(&factor)
                .ok_or_else(|| overflow_error("rescale up"))?;
        }
    } else {
        // Scale down: truncating division by factor
        for i in 0..len {
            if arr.is_null(i) {
                out[i] = T::zero();
                continue;
            }
            out[i] = arr.data[i] / factor;
        }
    }

    Ok(DecimalArray::new(out, arr.null_mask.clone(), arr.precision, new_scale))
}

/// Element-wise comparison of two decimal arrays, returning a `Bitmask`.
/// Operands with different scales are rescaled to the finer scale before
/// comparing.
pub fn decimal_compare<T: Integer + 'static>(
    lhs: &DecimalArray<T>,
    rhs: &DecimalArray<T>,
    op: crate::enums::operators::ComparisonOperator,
) -> Result<Bitmask, KernelError> {
    use crate::enums::operators::ComparisonOperator::*;

    let len = lhs.len();
    if len != rhs.len() {
        return Err(KernelError::LengthMismatch(format!(
            "decimal_compare: lhs {} vs rhs {}",
            len,
            rhs.len()
        )));
    }

    let ls = lhs.scale;
    let rs = rhs.scale;
    let scale_diff = (ls - rs).unsigned_abs() as u32;
    let need_rescale = ls != rs;
    let scale_factor = if need_rescale {
        checked_pow10::<T>(scale_diff)
            .ok_or_else(|| overflow_error("comparison rescale factor"))?
    } else {
        T::one()
    };

    let mut result = Bitmask::new_set_all(len, false);

    for i in 0..len {
        // Null on either side produces false for all comparisons
        if lhs.is_null(i) || rhs.is_null(i) {
            continue;
        }
        let (l, r) = if !need_rescale {
            (lhs.data[i], rhs.data[i])
        } else if ls < rs {
            let scaled_l = lhs.data[i]
                .checked_mul(&scale_factor)
                .ok_or_else(|| overflow_error("comparison rescale"))?;
            (scaled_l, rhs.data[i])
        } else {
            let scaled_r = rhs.data[i]
                .checked_mul(&scale_factor)
                .ok_or_else(|| overflow_error("comparison rescale"))?;
            (lhs.data[i], scaled_r)
        };

        let cmp_result = match op {
            Equals => l == r,
            NotEquals => l != r,
            LessThan => l < r,
            LessThanOrEqualTo => l <= r,
            GreaterThan => l > r,
            GreaterThanOrEqualTo => l >= r,
            _ => {
                return Err(KernelError::UnsupportedType(format!(
                    "Comparison operator {:?} not supported for decimal arrays",
                    op
                )));
            }
        };
        result.set(i, cmp_result);
    }

    Ok(result)
}

/// Convert an integer array to a DecimalArray at the given scale by
/// multiplying each value by 10^scale. Used for Integer + Decimal
/// auto-promotion.
pub fn integer_to_decimal<T: Integer + 'static>(
    data: &[T],
    null_mask: Option<&Bitmask>,
    precision: u8,
    scale: i8,
) -> Result<DecimalArray<T>, KernelError> {
    let len = data.len();
    let factor = if scale > 0 {
        checked_pow10::<T>(scale as u32)
            .ok_or_else(|| overflow_error("integer-to-decimal scale factor"))?
    } else {
        T::one()
    };

    let mut out = Vec64::<T>::with_capacity(len);
    unsafe { out.set_len(len) };

    for i in 0..len {
        if null_mask.map(|m| !m.get(i)).unwrap_or(false) {
            out[i] = T::zero();
            continue;
        }
        out[i] = data[i]
            .checked_mul(&factor)
            .ok_or_else(|| overflow_error("integer-to-decimal conversion"))?;
    }

    Ok(DecimalArray::new(out, null_mask.cloned(), precision, scale))
}

// ---------------------------------------------------------------------------
// Internal utilities
// ---------------------------------------------------------------------------

fn overflow_error(context: &str) -> KernelError {
    KernelError::Overflow(format!("Decimal arithmetic overflow in {}", context))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enums::operators::ArithmeticOperator::*;

    fn dec32(vals: &[i32], p: u8, s: i8) -> DecimalArray<i32> {
        DecimalArray::<i32>::from_slice(vals, p, s)
    }

    fn dec64(vals: &[i64], p: u8, s: i8) -> DecimalArray<i64> {
        DecimalArray::<i64>::from_slice(vals, p, s)
    }

    fn dec128(vals: &[i128], p: u8, s: i8) -> DecimalArray<i128> {
        DecimalArray::<i128>::from_slice(vals, p, s)
    }

    // -----------------------------------------------------------------------
    // Same-scale add/subtract
    // -----------------------------------------------------------------------

    #[test]
    fn add_same_scale_i32() {
        let a = dec32(&[12345, 67890], 9, 2);
        let b = dec32(&[11111, 22222], 9, 2);
        let result = decimal_binary(&a, &b, Add).unwrap();
        assert_eq!(result.data.as_slice(), &[23456, 90112]);
        assert_eq!(result.scale, 2);
    }

    #[test]
    fn subtract_same_scale_i64() {
        let a = dec64(&[50000, 30000], 18, 4);
        let b = dec64(&[10000, 20000], 18, 4);
        let result = decimal_binary(&a, &b, Subtract).unwrap();
        assert_eq!(result.data.as_slice(), &[40000, 10000]);
        assert_eq!(result.scale, 4);
    }

    // -----------------------------------------------------------------------
    // Different-scale add/subtract
    // -----------------------------------------------------------------------

    #[test]
    fn add_different_scale() {
        // 100.0 at scale=1 + 10.00 at scale=2 = 110.00 at scale=2
        let a = dec64(&[1000], 18, 1); // raw 1000, scale 1 => 100.0
        let b = dec64(&[1000], 18, 2); // raw 1000, scale 2 => 10.00
        let result = decimal_binary(&a, &b, Add).unwrap();
        // lhs rescaled: 1000 * 10 = 10000 at scale 2 => 100.00
        // 10000 + 1000 = 11000 at scale 2 => 110.00
        assert_eq!(result.data.as_slice(), &[11000]);
        assert_eq!(result.scale, 2);
    }

    #[test]
    fn subtract_different_scale() {
        let a = dec32(&[1000], 9, 1); // 100.0
        let b = dec32(&[500], 9, 2);  // 5.00
        let result = decimal_binary(&a, &b, Subtract).unwrap();
        // lhs rescaled: 1000 * 10 = 10000 at scale 2 => 100.00
        // 10000 - 500 = 9500 => 95.00
        assert_eq!(result.data.as_slice(), &[9500]);
        assert_eq!(result.scale, 2);
    }

    // -----------------------------------------------------------------------
    // Multiply
    // -----------------------------------------------------------------------

    #[test]
    fn multiply_basic() {
        // 12.34 * 5.67 = 69.9678
        let a = dec64(&[1234], 18, 2);
        let b = dec64(&[567], 18, 2);
        let result = decimal_binary(&a, &b, Multiply).unwrap();
        assert_eq!(result.data.as_slice(), &[699678]);
        assert_eq!(result.scale, 4); // 2 + 2
    }

    // -----------------------------------------------------------------------
    // Divide
    // -----------------------------------------------------------------------

    #[test]
    fn divide_same_scale() {
        // 100.00 / 4.00 = 25.00
        let a = dec64(&[10000], 18, 2);
        let b = dec64(&[400], 18, 2);
        let result = decimal_binary(&a, &b, Divide).unwrap();
        // result_scale = max(2, 2) = 2
        // extra_scale = 2 - 2 + 2 = 2
        // scaled_num = 10000 * 100 = 1000000
        // 1000000 / 400 = 2500 at scale 2 => 25.00
        assert_eq!(result.data.as_slice(), &[2500]);
        assert_eq!(result.scale, 2);
    }

    #[test]
    fn divide_by_zero_produces_null() {
        let a = dec32(&[100, 200], 9, 2);
        let b = dec32(&[0, 50], 9, 2);
        let result = decimal_binary(&a, &b, Divide).unwrap();
        assert_eq!(result.data[0], 0);
        assert!(result.is_null(0));
        assert!(!result.is_null(1));
    }

    // -----------------------------------------------------------------------
    // Remainder
    // -----------------------------------------------------------------------

    #[test]
    fn remainder_same_scale() {
        // 10.00 % 3.00 = 1.00
        let a = dec64(&[1000], 18, 2);
        let b = dec64(&[300], 18, 2);
        let result = decimal_binary(&a, &b, Remainder).unwrap();
        assert_eq!(result.data.as_slice(), &[100]); // 1000 % 300 = 100
        assert_eq!(result.scale, 2);
    }

    #[test]
    fn remainder_by_zero_produces_null() {
        let a = dec32(&[100], 9, 2);
        let b = dec32(&[0], 9, 2);
        let result = decimal_binary(&a, &b, Remainder).unwrap();
        assert!(result.is_null(0));
    }

    // -----------------------------------------------------------------------
    // Negate and Abs
    // -----------------------------------------------------------------------

    #[test]
    fn negate_basic() {
        let a = dec32(&[100, -200, 0], 9, 2);
        let result = decimal_negate(&a).unwrap();
        assert_eq!(result.data.as_slice(), &[-100, 200, 0]);
        assert_eq!(result.scale, 2);
        assert_eq!(result.precision, 9);
    }

    #[test]
    fn abs_basic() {
        let a = dec64(&[-500, 300, 0, -1], 18, 4);
        let result = decimal_abs(&a).unwrap();
        assert_eq!(result.data.as_slice(), &[500, 300, 0, 1]);
    }

    // -----------------------------------------------------------------------
    // Overflow detection
    // -----------------------------------------------------------------------

    #[test]
    fn add_overflow_returns_error() {
        let a = dec32(&[i32::MAX], 9, 0);
        let b = dec32(&[1], 9, 0);
        let result = decimal_binary(&a, &b, Add);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("overflow"), "got: {err}");
    }

    #[test]
    fn multiply_overflow_returns_error() {
        let a = dec32(&[i32::MAX], 9, 0);
        let b = dec32(&[2], 9, 0);
        let result = decimal_binary(&a, &b, Multiply);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Null propagation
    // -----------------------------------------------------------------------

    #[test]
    fn null_propagation() {
        let mut a = DecimalArray::<i32>::with_capacity(3, true, 9, 2);
        a.push(100);
        a.push_null();
        a.push(300);

        let b = dec32(&[10, 20, 30], 9, 2);
        let result = decimal_binary(&a, &b, Add).unwrap();
        assert_eq!(result.get(0), Some(110));
        assert_eq!(result.get(1), None);
        assert_eq!(result.get(2), Some(330));
    }

    // -----------------------------------------------------------------------
    // Rescale
    // -----------------------------------------------------------------------

    #[test]
    fn rescale_up() {
        let a = dec64(&[100, 200], 18, 2);
        let result = decimal_rescale(&a, 4).unwrap();
        assert_eq!(result.data.as_slice(), &[10000, 20000]);
        assert_eq!(result.scale, 4);
    }

    #[test]
    fn rescale_down_truncates() {
        let a = dec64(&[12345, 67899], 18, 4);
        let result = decimal_rescale(&a, 2).unwrap();
        assert_eq!(result.data.as_slice(), &[123, 678]);
        assert_eq!(result.scale, 2);
    }

    #[test]
    fn rescale_noop() {
        let a = dec32(&[100], 9, 2);
        let result = decimal_rescale(&a, 2).unwrap();
        assert_eq!(result.data.as_slice(), &[100]);
    }

    // -----------------------------------------------------------------------
    // Comparison
    // -----------------------------------------------------------------------

    #[test]
    fn compare_same_scale() {
        use crate::enums::operators::ComparisonOperator::*;

        let a = dec32(&[100, 200, 300], 9, 2);
        let b = dec32(&[200, 200, 100], 9, 2);

        let eq = decimal_compare(&a, &b, Equals).unwrap();
        assert!(!eq.get(0));
        assert!(eq.get(1));
        assert!(!eq.get(2));

        let lt = decimal_compare(&a, &b, LessThan).unwrap();
        assert!(lt.get(0));
        assert!(!lt.get(1));
        assert!(!lt.get(2));

        let gt = decimal_compare(&a, &b, GreaterThan).unwrap();
        assert!(!gt.get(0));
        assert!(!gt.get(1));
        assert!(gt.get(2));
    }

    #[test]
    fn compare_different_scale() {
        use crate::enums::operators::ComparisonOperator::*;

        // 10.0 at scale=1 vs 10.00 at scale=2 should be equal
        let a = dec64(&[100], 18, 1);  // 10.0
        let b = dec64(&[1000], 18, 2); // 10.00
        let eq = decimal_compare(&a, &b, Equals).unwrap();
        assert!(eq.get(0));
    }

    #[test]
    fn compare_null_produces_false() {
        use crate::enums::operators::ComparisonOperator::*;

        let mut a = DecimalArray::<i32>::with_capacity(2, true, 9, 2);
        a.push(100);
        a.push_null();
        let b = dec32(&[100, 100], 9, 2);
        let eq = decimal_compare(&a, &b, Equals).unwrap();
        assert!(eq.get(0));
        assert!(!eq.get(1)); // null comparison is false
    }

    // -----------------------------------------------------------------------
    // i128 operations
    // -----------------------------------------------------------------------

    #[test]
    fn add_i128() {
        let a = dec128(&[100_000_000_000i128, 200_000_000_000], 38, 6);
        let b = dec128(&[50_000_000_000, 100_000_000_000], 38, 6);
        let result = decimal_binary(&a, &b, Add).unwrap();
        assert_eq!(result.data.as_slice(), &[150_000_000_000i128, 300_000_000_000]);
    }

    #[test]
    fn multiply_i128() {
        let a = dec128(&[1000i128], 38, 2);
        let b = dec128(&[2000i128], 38, 3);
        let result = decimal_binary(&a, &b, Multiply).unwrap();
        assert_eq!(result.data.as_slice(), &[2_000_000i128]);
        assert_eq!(result.scale, 5); // 2 + 3
    }

    // -----------------------------------------------------------------------
    // Integer to decimal conversion
    // -----------------------------------------------------------------------

    #[test]
    fn integer_to_decimal_basic() {
        let data = [10i64, 20, 30];
        let result = integer_to_decimal(&data, None, 18, 2).unwrap();
        assert_eq!(result.data.as_slice(), &[1000, 2000, 3000]);
        assert_eq!(result.scale, 2);
    }

    // -----------------------------------------------------------------------
    // Power is unsupported
    // -----------------------------------------------------------------------

    #[test]
    fn power_returns_error() {
        let a = dec32(&[100], 9, 2);
        let b = dec32(&[2], 9, 0);
        let result = decimal_binary(&a, &b, Power);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // decimal_binary_into
    // -----------------------------------------------------------------------

    fn into_result<L: Copy, R: Copy, T: Integer + From<L> + From<R>>(
        lhs: &[L],
        lhs_type: (u8, i8),
        rhs: &[R],
        rhs_type: (u8, i8),
        op: ArithmeticOperator,
        mask: Option<&Bitmask>,
    ) -> (Vec<T>, Bitmask, (u8, i8)) {
        let mut out = vec![T::zero(); lhs.len()];
        let mut out_mask = Bitmask::new_set_all(lhs.len(), false);
        let result_type = decimal_binary_into(
            lhs,
            lhs_type,
            rhs,
            rhs_type,
            op,
            mask,
            &mut out,
            Some(&mut out_mask),
        )
        .unwrap();
        (out, out_mask, result_type)
    }

    #[test]
    fn into_matches_expected_values_for_every_operator() {
        // 123.45, null, -7.00 and 10.00 at scale 2 against 0.250, 0.010, 0.000
        // and -0.003 at scale 3
        let lhs = [12345i64, 0, -700, 1000];
        let mask = Bitmask::from_bools(&[true, false, true, true]);
        let rhs = [250i64, 10, 0, -3];
        let cases: [(ArithmeticOperator, (u8, i8), [Option<i64>; 4]); 6] = [
            // 123.700, -7.000, 9.997
            (Add, (13, 3), [Some(123700), None, Some(-7000), Some(9997)]),
            // 123.200, -7.000, 10.003
            (Subtract, (13, 3), [Some(123200), None, Some(-7000), Some(10003)]),
            // 30.86250, 0.00000, -0.03000
            (Multiply, (18, 5), [Some(3086250), None, Some(0), Some(-3000)]),
            // 493.800, null for the zero divisor, -3333.333 truncated
            (Divide, (12, 3), [Some(493800), None, None, Some(-3333333)]),
            (FloorDiv, (12, 3), [Some(493800), None, None, Some(-3333333)]),
            // 0.200, null for the zero divisor, 0.001 with the sign of the dividend
            (Remainder, (12, 3), [Some(200), None, None, Some(1)]),
        ];
        for (op, result_type, expected) in cases {
            let (out, out_mask, actual_type) =
                into_result::<i64, i64, i64>(&lhs, (12, 2), &rhs, (10, 3), op, Some(&mask));
            assert_eq!(actual_type, result_type, "{op:?}");
            let actual: Vec<Option<i64>> =
                (0..4).map(|i| out_mask.get(i).then(|| out[i])).collect();
            assert_eq!(actual, expected, "{op:?}");
        }
    }

    #[test]
    fn into_widens_mixed_width_operands() {
        // Decimal32 15.0, -2.5 and 0.9 at scale 1 against Decimal64 10^10, 0.04
        // and -0.03 at scale 2, written as i64
        let lhs = [150i32, -25, 9];
        let rhs = [1_000_000_000_000i64, 4, -3];
        let cases: [(ArithmeticOperator, (u8, i8), [i64; 3]); 5] = [
            (Add, (18, 2), [1_000_000_001_500, -246, 87]),
            (Subtract, (18, 2), [-999_999_998_500, -254, 93]),
            (Multiply, (18, 3), [150_000_000_000_000, -100, -27]),
            // 0.00, -62.50, -30.00
            (Divide, (18, 2), [0, -6250, -3000]),
            // 15.00, -0.02, 0.00
            (Remainder, (18, 2), [1500, -2, 0]),
        ];
        for (op, result_type, expected) in cases {
            let (out, out_mask, actual_type) =
                into_result::<i32, i64, i64>(&lhs, (9, 1), &rhs, (18, 2), op, None);
            assert!(out_mask.all_set(), "{op:?}");
            assert_eq!(out, expected, "{op:?}");
            assert_eq!(actual_type, result_type, "{op:?}");
        }
    }

    #[test]
    fn into_null_rows_without_output_mask_write_zero() {
        let mask = Bitmask::from_bools(&[true, false, true]);
        let mut out = [7i32; 3];
        let result_type = decimal_binary_into(
            &[100i32, 200, 300],
            (9, 2),
            &[1i32, 2, 3],
            (9, 2),
            Add,
            Some(&mask),
            &mut out,
            None,
        )
        .unwrap();
        assert_eq!(out, [101, 0, 303]);
        assert_eq!(result_type, (9, 2));
    }

    #[test]
    fn into_short_masks_return_length_mismatch() {
        let short = Bitmask::new_set_all(1, true);
        let mut out = [0i32; 2];
        let result = decimal_binary_into(
            &[1i32, 2], (9, 0), &[3i32, 4], (9, 0), Add, Some(&short), &mut out, None,
        );
        assert!(matches!(result, Err(KernelError::LengthMismatch(_))));
        let mut short_out = Bitmask::new_set_all(1, true);
        let result = decimal_binary_into(
            &[1i32, 2], (9, 0), &[3i32, 4], (9, 0), Add, None, &mut out, Some(&mut short_out),
        );
        assert!(matches!(result, Err(KernelError::LengthMismatch(_))));
    }

    #[test]
    fn into_output_mask_follows_input_mask_across_byte_boundaries() {
        // 19 rows cover two whole mask bytes and a partial third
        let valid: Vec<bool> = (0..19).map(|i| i % 3 != 0).collect();
        let mask = Bitmask::from_bools(&valid);
        let lhs: Vec<i64> = (0..19).collect();
        let rhs = vec![2i64; 19];
        let (out, out_mask, _) =
            into_result::<i64, i64, i64>(&lhs, (18, 0), &rhs, (18, 0), Multiply, Some(&mask));
        for i in 0..19 {
            assert_eq!(out_mask.get(i), valid[i], "row {i}");
            assert_eq!(out[i], if valid[i] { 2 * i as i64 } else { 0 }, "row {i}");
        }
    }

    #[test]
    fn into_zero_divisor_without_output_mask_returns_error() {
        let mut out = [0i32; 1];
        let result =
            decimal_binary_into(&[100i32], (9, 2), &[0i32], (9, 2), Divide, None, &mut out, None);
        assert!(matches!(result, Err(KernelError::InvalidArguments(_))));
    }

    #[test]
    fn into_without_nulls_accepts_no_output_mask() {
        let mut out = [0i64; 2];
        let result_type = decimal_binary_into(
            &[100i64, 200],
            (18, 2),
            &[3i32, 4],
            (9, 0),
            Multiply,
            None,
            &mut out,
            None,
        )
        .unwrap();
        assert_eq!(out, [300, 800]);
        assert_eq!(result_type, (18, 2));
    }

    #[test]
    fn into_output_length_mismatch_returns_error() {
        let mut out = [0i32; 1];
        let result =
            decimal_binary_into(&[1i32, 2], (9, 0), &[3i32, 4], (9, 0), Add, None, &mut out, None);
        assert!(matches!(result, Err(KernelError::LengthMismatch(_))));
    }

    #[test]
    fn divide_by_zero_with_overflowing_numerator_produces_null() {
        // The numerator scale-up of 10^2 overflows i32 on the first row, and the
        // zero divisor produces null there
        let a = dec32(&[i32::MAX, 300], 9, 0);
        let b = dec32(&[0, 15], 9, 1);
        let result = decimal_binary(&a, &b, Divide).unwrap();
        assert!(result.is_null(0));
        // 300 / 1.5 = 200.0 at scale 1
        assert_eq!(result.get(1), Some(2000));
    }

    #[test]
    fn remainder_by_zero_with_overflowing_numerator_produces_null() {
        // The numerator rescale of 10^2 overflows i32 on the first row, and the
        // zero divisor produces null there
        let a = dec32(&[i32::MAX, 300], 9, 0);
        let b = dec32(&[0, 70], 9, 2);
        let result = decimal_binary(&a, &b, Remainder).unwrap();
        assert!(result.is_null(0));
        // 300 % 0.70 = 0.40 at scale 2
        assert_eq!(result.get(1), Some(40));
    }

    #[test]
    fn into_minimum_divided_by_minus_one_returns_overflow() {
        let mut out = [0i32; 1];
        for op in [Divide, FloorDiv] {
            let result = decimal_binary_into(
                &[i32::MIN],
                (9, 0),
                &[-1i32],
                (9, 0),
                op,
                None,
                &mut out,
                None,
            );
            assert!(matches!(result, Err(KernelError::Overflow(_))), "{op:?}");
        }
    }

    #[test]
    fn remainder_of_minimum_by_minus_one_is_zero() {
        let a = dec32(&[i32::MIN, 7], 9, 0);
        let b = dec32(&[-1, -1], 9, 0);
        let result = decimal_binary(&a, &b, Remainder).unwrap();
        assert_eq!(result.data.as_slice(), &[0, 0]);
        assert!(result.null_mask.is_none());
    }

    // -----------------------------------------------------------------------
    // Empty arrays
    // -----------------------------------------------------------------------

    #[test]
    fn empty_arrays() {
        let a = dec32(&[], 9, 2);
        let b = dec32(&[], 9, 2);
        let result = decimal_binary(&a, &b, Add).unwrap();
        assert_eq!(result.len(), 0);
    }
}
