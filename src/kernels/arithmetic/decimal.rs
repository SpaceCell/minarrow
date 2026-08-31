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
fn max_precision<T: Integer + 'static>() -> u8 {
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
///
/// ## Scale rules
/// - Add/Subtract: operands are rescaled to the finer scale. Result
///   precision is capped at the width maximum.
/// - Multiply: result scale = s1 + s2, result precision = p1 + p2,
///   capped at the width maximum.
/// - Divide: the numerator is scaled up so integer division preserves
///   the desired number of fractional digits. Result scale is max of
///   the two input scales.
/// - Negate and Abs are unary and handled by separate functions.
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

    match op {
        ArithmeticOperator::Add | ArithmeticOperator::Subtract => {
            decimal_add_sub(lhs, rhs, op, merged_mask)
        }
        ArithmeticOperator::Multiply => decimal_multiply(lhs, rhs, merged_mask),
        ArithmeticOperator::Divide | ArithmeticOperator::FloorDiv => {
            decimal_divide(lhs, rhs, merged_mask)
        }
        ArithmeticOperator::Remainder => decimal_remainder(lhs, rhs, merged_mask),
        ArithmeticOperator::Power => Err(KernelError::UnsupportedType(
            "Power is not supported for decimal arrays - consider converting to float64 first"
                .to_string(),
        )),
    }
}

/// Add or subtract two decimal arrays, reconciling scale when they differ.
fn decimal_add_sub<T: Integer + 'static>(
    lhs: &DecimalArray<T>,
    rhs: &DecimalArray<T>,
    op: ArithmeticOperator,
    merged_mask: Option<Bitmask>,
) -> Result<DecimalArray<T>, KernelError> {
    let len = lhs.len();
    let ls = lhs.scale;
    let rs = rhs.scale;
    let result_scale = ls.max(rs);
    let result_precision = (lhs.precision.max(rhs.precision) + 1).min(max_precision::<T>());

    let mut out = Vec64::<T>::with_capacity(len);
    unsafe { out.set_len(len) };

    let is_add = matches!(op, ArithmeticOperator::Add);

    if ls == rs {
        // Same scale - operate on raw values
        for i in 0..len {
            if is_masked_null(&merged_mask, i) {
                out[i] = T::zero();
                continue;
            }
            let l = lhs.data[i];
            let r = rhs.data[i];
            let result = if is_add {
                l.checked_add(&r)
            } else {
                l.checked_sub(&r)
            };
            out[i] = result.ok_or_else(|| overflow_error("add/subtract"))?;
        }
    } else {
        // Different scale - rescale the coarser operand to the finer scale
        let scale_diff = (ls - rs).unsigned_abs() as u32;
        let scale_factor = checked_pow10::<T>(scale_diff)
            .ok_or_else(|| overflow_error("scale factor computation"))?;

        for i in 0..len {
            if is_masked_null(&merged_mask, i) {
                out[i] = T::zero();
                continue;
            }
            let (l, r) = if ls < rs {
                // lhs has coarser scale, rescale lhs up
                let scaled_l = lhs.data[i]
                    .checked_mul(&scale_factor)
                    .ok_or_else(|| overflow_error("rescale for add/subtract"))?;
                (scaled_l, rhs.data[i])
            } else {
                // rhs has coarser scale, rescale rhs up
                let scaled_r = rhs.data[i]
                    .checked_mul(&scale_factor)
                    .ok_or_else(|| overflow_error("rescale for add/subtract"))?;
                (lhs.data[i], scaled_r)
            };
            let result = if is_add {
                l.checked_add(&r)
            } else {
                l.checked_sub(&r)
            };
            out[i] = result.ok_or_else(|| overflow_error("add/subtract"))?;
        }
    }

    Ok(DecimalArray::new(out, merged_mask, result_precision, result_scale))
}

/// Multiply two decimal arrays. Result scale = s1 + s2, result
/// precision = p1 + p2, capped at the width maximum.
fn decimal_multiply<T: Integer + 'static>(
    lhs: &DecimalArray<T>,
    rhs: &DecimalArray<T>,
    merged_mask: Option<Bitmask>,
) -> Result<DecimalArray<T>, KernelError> {
    let len = lhs.len();
    let result_scale = lhs.scale + rhs.scale;
    let result_precision = (lhs.precision + rhs.precision).min(max_precision::<T>());

    let mut out = Vec64::<T>::with_capacity(len);
    unsafe { out.set_len(len) };

    for i in 0..len {
        if is_masked_null(&merged_mask, i) {
            out[i] = T::zero();
            continue;
        }
        out[i] = lhs.data[i]
            .checked_mul(&rhs.data[i])
            .ok_or_else(|| overflow_error("multiply"))?;
    }

    Ok(DecimalArray::new(out, merged_mask, result_precision, result_scale))
}

/// Divide two decimal arrays. The numerator is scaled up so integer division
/// preserves fractional digits. Result scale = max of the two input scales.
fn decimal_divide<T: Integer + 'static>(
    lhs: &DecimalArray<T>,
    rhs: &DecimalArray<T>,
    merged_mask: Option<Bitmask>,
) -> Result<DecimalArray<T>, KernelError> {
    let len = lhs.len();
    let result_scale = lhs.scale.max(rhs.scale);
    let result_precision = lhs.precision.max(rhs.precision).min(max_precision::<T>());

    // Scale up the numerator so that integer division produces digits at
    // the desired result scale: numerator * 10^(result_scale - lhs.scale + rhs.scale)
    let extra_scale = (result_scale - lhs.scale + rhs.scale) as u32;
    let scale_up = if extra_scale > 0 {
        checked_pow10::<T>(extra_scale)
            .ok_or_else(|| overflow_error("division scale-up factor"))?
    } else {
        T::one()
    };

    let mut out = Vec64::<T>::with_capacity(len);
    unsafe { out.set_len(len) };
    let mut out_mask = merged_mask.clone();

    for i in 0..len {
        if is_masked_null(&out_mask, i) {
            out[i] = T::zero();
            continue;
        }
        let divisor = rhs.data[i];
        if divisor == T::zero() {
            // Division by zero produces null
            out[i] = T::zero();
            ensure_mask_null(&mut out_mask, len, i);
            continue;
        }
        let scaled_num = lhs.data[i]
            .checked_mul(&scale_up)
            .ok_or_else(|| overflow_error("division numerator scale-up"))?;
        out[i] = scaled_num / divisor;
    }

    Ok(DecimalArray::new(out, out_mask, result_precision, result_scale))
}

/// Remainder of two decimal arrays. Operands must have the same scale,
/// or the coarser operand is rescaled first.
fn decimal_remainder<T: Integer + 'static>(
    lhs: &DecimalArray<T>,
    rhs: &DecimalArray<T>,
    merged_mask: Option<Bitmask>,
) -> Result<DecimalArray<T>, KernelError> {
    let len = lhs.len();
    let ls = lhs.scale;
    let rs = rhs.scale;
    let result_scale = ls.max(rs);
    let result_precision = lhs.precision.max(rhs.precision).min(max_precision::<T>());

    let scale_diff = (ls - rs).unsigned_abs() as u32;
    let scale_factor = if scale_diff > 0 {
        checked_pow10::<T>(scale_diff)
            .ok_or_else(|| overflow_error("remainder scale factor"))?
    } else {
        T::one()
    };

    let mut out = Vec64::<T>::with_capacity(len);
    unsafe { out.set_len(len) };
    let mut out_mask = merged_mask.clone();

    for i in 0..len {
        if is_masked_null(&out_mask, i) {
            out[i] = T::zero();
            continue;
        }
        let (l, r) = if ls == rs {
            (lhs.data[i], rhs.data[i])
        } else if ls < rs {
            let scaled_l = lhs.data[i]
                .checked_mul(&scale_factor)
                .ok_or_else(|| overflow_error("remainder rescale"))?;
            (scaled_l, rhs.data[i])
        } else {
            let scaled_r = rhs.data[i]
                .checked_mul(&scale_factor)
                .ok_or_else(|| overflow_error("remainder rescale"))?;
            (lhs.data[i], scaled_r)
        };
        if r == T::zero() {
            out[i] = T::zero();
            ensure_mask_null(&mut out_mask, len, i);
            continue;
        }
        out[i] = l % r;
    }

    Ok(DecimalArray::new(out, out_mask, result_precision, result_scale))
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

#[inline(always)]
fn is_masked_null(mask: &Option<Bitmask>, i: usize) -> bool {
    mask.as_ref().map(|m| !m.get(i)).unwrap_or(false)
}

/// Ensure the mask exists and mark position `i` as null.
fn ensure_mask_null(mask: &mut Option<Bitmask>, len: usize, i: usize) {
    match mask {
        Some(m) => m.set(i, false),
        None => {
            let mut m = Bitmask::new_set_all(len, true);
            m.set(i, false);
            *mask = Some(m);
        }
    }
}

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
