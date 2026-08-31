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

#[cfg(feature = "scalar_type")]
use crate::Scalar;
#[cfg(feature = "decimal")]
use crate::DecimalArray;
use crate::enums::error::MinarrowError;
use crate::kernels::routing::broadcast::maybe_broadcast_scalar_array;
use crate::{Array, ArrayV, Bitmask, TextArray};
use crate::{NumericArray, Vec64};

use crate::kernels::arithmetic::{
    dispatch::{
        apply_float_f32, apply_float_f64, apply_int_i32, apply_int_i64, apply_int_u32,
        apply_int_u64,
    },
    string_ops::apply_str_str,
};
#[cfg(feature = "decimal")]
use crate::kernels::arithmetic::decimal::{decimal_binary, integer_to_decimal};

use crate::enums::{error::KernelError, operators::ArithmeticOperator};

/// Perform arithmetic operations on two scalars
#[cfg(feature = "scalar_type")]
pub fn scalar_arithmetic(
    lhs: Scalar,
    rhs: Scalar,
    op: ArithmeticOperator,
) -> Result<Scalar, MinarrowError> {
    use ArithmeticOperator::*;
    use Scalar;

    let result = match (lhs, rhs, op) {
        // Int32 operations
        (Scalar::Int32(l), Scalar::Int32(r), Add) => Scalar::Int32(l + r),
        (Scalar::Int32(l), Scalar::Int32(r), Subtract) => Scalar::Int32(l - r),
        (Scalar::Int32(l), Scalar::Int32(r), Multiply) => Scalar::Int32(l * r),
        // Integer scalar division is true division and returns a Float64
        // scalar, so 7 / 2 is 3.5 and division by zero follows IEEE 754.
        (Scalar::Int32(l), Scalar::Int32(r), Divide) => Scalar::Float64(l as f64 / r as f64),

        // Int64 operations
        (Scalar::Int64(l), Scalar::Int64(r), Add) => Scalar::Int64(l + r),
        (Scalar::Int64(l), Scalar::Int64(r), Subtract) => Scalar::Int64(l - r),
        (Scalar::Int64(l), Scalar::Int64(r), Multiply) => Scalar::Int64(l * r),
        (Scalar::Int64(l), Scalar::Int64(r), Divide) => Scalar::Float64(l as f64 / r as f64),

        // Float32 operations
        (Scalar::Float32(l), Scalar::Float32(r), Add) => Scalar::Float32(l + r),
        (Scalar::Float32(l), Scalar::Float32(r), Subtract) => Scalar::Float32(l - r),
        (Scalar::Float32(l), Scalar::Float32(r), Multiply) => Scalar::Float32(l * r),
        (Scalar::Float32(l), Scalar::Float32(r), Divide) => Scalar::Float32(l / r),

        // Float64 operations
        (Scalar::Float64(l), Scalar::Float64(r), Add) => Scalar::Float64(l + r),
        (Scalar::Float64(l), Scalar::Float64(r), Subtract) => Scalar::Float64(l - r),
        (Scalar::Float64(l), Scalar::Float64(r), Multiply) => Scalar::Float64(l * r),
        (Scalar::Float64(l), Scalar::Float64(r), Divide) => Scalar::Float64(l / r),

        // Mixed type promotion (Int + Float = Float)
        (Scalar::Int32(l), Scalar::Float32(r), op) => {
            return scalar_arithmetic(Scalar::Float32(l as f32), Scalar::Float32(r), op);
        }
        (Scalar::Float32(l), Scalar::Int32(r), op) => {
            return scalar_arithmetic(Scalar::Float32(l), Scalar::Float32(r as f32), op);
        }
        (Scalar::Int64(l), Scalar::Float64(r), op) => {
            return scalar_arithmetic(Scalar::Float64(l as f64), Scalar::Float64(r), op);
        }
        (Scalar::Float64(l), Scalar::Int64(r), op) => {
            return scalar_arithmetic(Scalar::Float64(l), Scalar::Float64(r as f64), op);
        }

        // Extended numeric types - Int8
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::Int8(l), Scalar::Int8(r), Add) => Scalar::Int8(l + r),
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::Int8(l), Scalar::Int8(r), Subtract) => Scalar::Int8(l - r),
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::Int8(l), Scalar::Int8(r), Multiply) => Scalar::Int8(l * r),
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::Int8(l), Scalar::Int8(r), Divide) => Scalar::Float64(l as f64 / r as f64),

        // Int16
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::Int16(l), Scalar::Int16(r), Add) => Scalar::Int16(l + r),
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::Int16(l), Scalar::Int16(r), Subtract) => Scalar::Int16(l - r),
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::Int16(l), Scalar::Int16(r), Multiply) => Scalar::Int16(l * r),
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::Int16(l), Scalar::Int16(r), Divide) => Scalar::Float64(l as f64 / r as f64),

        // UInt8
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::UInt8(l), Scalar::UInt8(r), Add) => Scalar::UInt8(l + r),
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::UInt8(l), Scalar::UInt8(r), Subtract) => Scalar::UInt8(l - r),
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::UInt8(l), Scalar::UInt8(r), Multiply) => Scalar::UInt8(l * r),
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::UInt8(l), Scalar::UInt8(r), Divide) => Scalar::Float64(l as f64 / r as f64),

        // UInt16
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::UInt16(l), Scalar::UInt16(r), Add) => Scalar::UInt16(l + r),
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::UInt16(l), Scalar::UInt16(r), Subtract) => Scalar::UInt16(l - r),
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::UInt16(l), Scalar::UInt16(r), Multiply) => Scalar::UInt16(l * r),
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::UInt16(l), Scalar::UInt16(r), Divide) => Scalar::Float64(l as f64 / r as f64),

        // UInt32
        (Scalar::UInt32(l), Scalar::UInt32(r), Add) => Scalar::UInt32(l + r),
        (Scalar::UInt32(l), Scalar::UInt32(r), Subtract) => Scalar::UInt32(l - r),
        (Scalar::UInt32(l), Scalar::UInt32(r), Multiply) => Scalar::UInt32(l * r),
        (Scalar::UInt32(l), Scalar::UInt32(r), Divide) => Scalar::Float64(l as f64 / r as f64),

        // UInt64
        (Scalar::UInt64(l), Scalar::UInt64(r), Add) => Scalar::UInt64(l + r),
        (Scalar::UInt64(l), Scalar::UInt64(r), Subtract) => Scalar::UInt64(l - r),
        (Scalar::UInt64(l), Scalar::UInt64(r), Multiply) => Scalar::UInt64(l * r),
        (Scalar::UInt64(l), Scalar::UInt64(r), Divide) => Scalar::Float64(l as f64 / r as f64),
        // String concatenation
        (Scalar::String32(l), Scalar::String32(r), Add) => Scalar::String32(format!("{}{}", l, r)),
        #[cfg(feature = "large_string")]
        (Scalar::String64(l), Scalar::String64(r), Add) => Scalar::String64(format!("{}{}", l, r)),

        // DateTime types
        #[cfg(feature = "datetime")]
        (Scalar::Datetime32(l), Scalar::Datetime32(r), Add) => Scalar::Datetime32(l + r),
        #[cfg(feature = "datetime")]
        (Scalar::Datetime64(l), Scalar::Datetime64(r), Add) => Scalar::Datetime64(l + r),
        #[cfg(feature = "datetime")]
        (Scalar::Datetime32(l), Scalar::Datetime32(r), Subtract) => Scalar::Datetime32(l - r),
        #[cfg(feature = "datetime")]
        (Scalar::Datetime64(l), Scalar::Datetime64(r), Subtract) => Scalar::Datetime64(l - r),

        // Cross-type promotions for extended numeric types with standard types
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::Int8(l), Scalar::Int32(r), op) => {
            return scalar_arithmetic(Scalar::Int32(l as i32), Scalar::Int32(r), op);
        }
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::Int32(l), Scalar::Int8(r), op) => {
            return scalar_arithmetic(Scalar::Int32(l), Scalar::Int32(r as i32), op);
        }
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::Int16(l), Scalar::Int32(r), op) => {
            return scalar_arithmetic(Scalar::Int32(l as i32), Scalar::Int32(r), op);
        }
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::Int32(l), Scalar::Int16(r), op) => {
            return scalar_arithmetic(Scalar::Int32(l), Scalar::Int32(r as i32), op);
        }
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::UInt8(l), Scalar::UInt32(r), op) => {
            return scalar_arithmetic(Scalar::UInt32(l as u32), Scalar::UInt32(r), op);
        }
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::UInt32(l), Scalar::UInt8(r), op) => {
            return scalar_arithmetic(Scalar::UInt32(l), Scalar::UInt32(r as u32), op);
        }
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::UInt16(l), Scalar::UInt32(r), op) => {
            return scalar_arithmetic(Scalar::UInt32(l as u32), Scalar::UInt32(r), op);
        }
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::UInt32(l), Scalar::UInt16(r), op) => {
            return scalar_arithmetic(Scalar::UInt32(l), Scalar::UInt32(r as u32), op);
        }

        // Boolean operations (only addition makes sense - logical OR)
        (Scalar::Boolean(l), Scalar::Boolean(r), Add) => Scalar::Boolean(l || r),

        // String with different string types
        #[cfg(feature = "large_string")]
        (Scalar::String32(l), Scalar::String64(r), Add) => Scalar::String64(format!("{}{}", l, r)),
        #[cfg(feature = "large_string")]
        (Scalar::String64(l), Scalar::String32(r), Add) => Scalar::String64(format!("{}{}", l, r)),

        // Decimal scalar operations (same width, same scale)
        #[cfg(feature = "decimal")]
        (Scalar::Decimal32(l, ls), Scalar::Decimal32(r, rs), Add) if ls == rs => {
            Scalar::Decimal32(l.checked_add(r).ok_or_else(|| MinarrowError::KernelError(
                Some("Decimal32 overflow in addition".to_string()),
            ))?, ls)
        }
        #[cfg(feature = "decimal")]
        (Scalar::Decimal32(l, ls), Scalar::Decimal32(r, rs), Subtract) if ls == rs => {
            Scalar::Decimal32(l.checked_sub(r).ok_or_else(|| MinarrowError::KernelError(
                Some("Decimal32 overflow in subtraction".to_string()),
            ))?, ls)
        }
        #[cfg(feature = "decimal")]
        (Scalar::Decimal32(l, ls), Scalar::Decimal32(r, _rs), Multiply) => {
            Scalar::Decimal32(l.checked_mul(r).ok_or_else(|| MinarrowError::KernelError(
                Some("Decimal32 overflow in multiplication".to_string()),
            ))?, ls + _rs)
        }
        #[cfg(feature = "decimal")]
        (Scalar::Decimal64(l, ls), Scalar::Decimal64(r, rs), Add) if ls == rs => {
            Scalar::Decimal64(l.checked_add(r).ok_or_else(|| MinarrowError::KernelError(
                Some("Decimal64 overflow in addition".to_string()),
            ))?, ls)
        }
        #[cfg(feature = "decimal")]
        (Scalar::Decimal64(l, ls), Scalar::Decimal64(r, rs), Subtract) if ls == rs => {
            Scalar::Decimal64(l.checked_sub(r).ok_or_else(|| MinarrowError::KernelError(
                Some("Decimal64 overflow in subtraction".to_string()),
            ))?, ls)
        }
        #[cfg(feature = "decimal")]
        (Scalar::Decimal64(l, ls), Scalar::Decimal64(r, _rs), Multiply) => {
            Scalar::Decimal64(l.checked_mul(r).ok_or_else(|| MinarrowError::KernelError(
                Some("Decimal64 overflow in multiplication".to_string()),
            ))?, ls + _rs)
        }
        #[cfg(feature = "decimal")]
        (Scalar::Decimal128(l, ls), Scalar::Decimal128(r, rs), Add) if ls == rs => {
            Scalar::Decimal128(l.checked_add(r).ok_or_else(|| MinarrowError::KernelError(
                Some("Decimal128 overflow in addition".to_string()),
            ))?, ls)
        }
        #[cfg(feature = "decimal")]
        (Scalar::Decimal128(l, ls), Scalar::Decimal128(r, rs), Subtract) if ls == rs => {
            Scalar::Decimal128(l.checked_sub(r).ok_or_else(|| MinarrowError::KernelError(
                Some("Decimal128 overflow in subtraction".to_string()),
            ))?, ls)
        }
        #[cfg(feature = "decimal")]
        (Scalar::Decimal128(l, ls), Scalar::Decimal128(r, _rs), Multiply) => {
            Scalar::Decimal128(l.checked_mul(r).ok_or_else(|| MinarrowError::KernelError(
                Some("Decimal128 overflow in multiplication".to_string()),
            ))?, ls + _rs)
        }

        // Decimal + Float -> Float64 scalar promotion
        #[cfg(feature = "decimal")]
        (Scalar::Decimal32(l, s), Scalar::Float64(r), op) | (Scalar::Float64(r), Scalar::Decimal32(l, s), op) => {
            let l_f64 = l as f64 / 10f64.powi(s as i32);
            return scalar_arithmetic(Scalar::Float64(l_f64), Scalar::Float64(r), op);
        }
        #[cfg(feature = "decimal")]
        (Scalar::Decimal64(l, s), Scalar::Float64(r), op) | (Scalar::Float64(r), Scalar::Decimal64(l, s), op) => {
            let l_f64 = l as f64 / 10f64.powi(s as i32);
            return scalar_arithmetic(Scalar::Float64(l_f64), Scalar::Float64(r), op);
        }
        #[cfg(feature = "decimal")]
        (Scalar::Decimal128(l, s), Scalar::Float64(r), op) | (Scalar::Float64(r), Scalar::Decimal128(l, s), op) => {
            use num_traits::ToPrimitive;
            let l_f64 = l.to_f64().unwrap() / 10f64.powi(s as i32);
            return scalar_arithmetic(Scalar::Float64(l_f64), Scalar::Float64(r), op);
        }

        // Null handling
        (Scalar::Null, _, _) | (_, Scalar::Null, _) => {
            return Err(MinarrowError::NullError {
                message: Some("Arithmetic operations with null values not supported".to_string()),
            });
        }

        // Catch-all for unsupported scalar type combinations
        (l, r, op) => {
            return Err(MinarrowError::NotImplemented {
                feature: format!(
                    "Scalar arithmetic operation {:?} between {:?} and {:?}. \
                     Consider casting to a common type first.",
                    op, l, r
                ),
            });
        }
    };

    Ok(result)
}

/// Public entry-point used by the execution engine.
#[inline]
pub fn resolve_binary_arithmetic(
    op: ArithmeticOperator,
    lhs: impl Into<ArrayV>,
    rhs: impl Into<ArrayV>,
    null_mask: Option<&Bitmask>,
) -> Result<Array, MinarrowError> {
    let (lhs_cast, rhs_cast) = maybe_broadcast_scalar_array(lhs.into(), rhs.into())?;
    Ok(arithmetic_dispatch(op, lhs_cast, rhs_cast, null_mask)?)
}

/// Ensures identical physical type and equal length, then applies the chosen kernel.
fn arithmetic_dispatch(
    op: ArithmeticOperator,
    lhs: impl Into<ArrayV>,
    rhs: impl Into<ArrayV>,
    null_mask: Option<&Bitmask>,
) -> Result<Array, KernelError> {
    let lhs = lhs.into();
    let rhs = rhs.into();

    // Length check for all binary ops
    if lhs.len() != rhs.len() {
        return Err(KernelError::LengthMismatch(format!(
            "arithmetic_dispatch => Length mismatch: LHS {} RHS {}",
            lhs.len(),
            rhs.len()
        )));
    }

    // Helper macros for upcasting
    macro_rules! promote_to_float64 {
        ($l:expr, $r:expr) => {
            Array::NumericArray(NumericArray::Float64(
                apply_float_f64(
                    &($l).iter().map(|&x| x as f64).collect::<Vec64<_>>(),
                    &($r).iter().map(|&x| x as f64).collect::<Vec64<_>>(),
                    op,
                    null_mask,
                )?
                .into(),
            ))
        };
    }
    macro_rules! promote_to_float32 {
        ($l:expr, $r:expr) => {
            Array::NumericArray(NumericArray::Float32(
                apply_float_f32(
                    &($l).iter().map(|&x| x as f32).collect::<Vec64<_>>(),
                    &($r).iter().map(|&x| x as f32).collect::<Vec64<_>>(),
                    op,
                    null_mask,
                )?
                .into(),
            ))
        };
    }

    // Extract sliced data based on ArrayView offset and len
    let lhs_offset = lhs.offset;
    let lhs_len = lhs.len();
    let rhs_offset = rhs.offset;
    let rhs_len = rhs.len();

    // Dispatch based on array types
    match (&lhs.array, &rhs.array) {
        // Numeric operations - same types
        (
            Array::NumericArray(NumericArray::Int32(l)),
            Array::NumericArray(NumericArray::Int32(r)),
        ) => {
            let lhs_slice = &l.data.as_slice()[lhs_offset..lhs_offset + lhs_len];
            let rhs_slice = &r.data.as_slice()[rhs_offset..rhs_offset + rhs_len];
            Ok(Array::NumericArray(NumericArray::Int32(
                apply_int_i32(lhs_slice, rhs_slice, op, null_mask)?.into(),
            )))
        }
        (
            Array::NumericArray(NumericArray::Int64(l)),
            Array::NumericArray(NumericArray::Int64(r)),
        ) => {
            let lhs_slice = &l.data.as_slice()[lhs_offset..lhs_offset + lhs_len];
            let rhs_slice = &r.data.as_slice()[rhs_offset..rhs_offset + rhs_len];
            Ok(Array::NumericArray(NumericArray::Int64(
                apply_int_i64(lhs_slice, rhs_slice, op, null_mask)?.into(),
            )))
        }
        (
            Array::NumericArray(NumericArray::UInt32(l)),
            Array::NumericArray(NumericArray::UInt32(r)),
        ) => {
            let lhs_slice = &l.data.as_slice()[lhs_offset..lhs_offset + lhs_len];
            let rhs_slice = &r.data.as_slice()[rhs_offset..rhs_offset + rhs_len];
            Ok(Array::NumericArray(NumericArray::UInt32(
                apply_int_u32(lhs_slice, rhs_slice, op, null_mask)?.into(),
            )))
        }
        (
            Array::NumericArray(NumericArray::UInt64(l)),
            Array::NumericArray(NumericArray::UInt64(r)),
        ) => {
            let lhs_slice = &l.data.as_slice()[lhs_offset..lhs_offset + lhs_len];
            let rhs_slice = &r.data.as_slice()[rhs_offset..rhs_offset + rhs_len];
            Ok(Array::NumericArray(NumericArray::UInt64(
                apply_int_u64(lhs_slice, rhs_slice, op, null_mask)?.into(),
            )))
        }
        (
            Array::NumericArray(NumericArray::Float32(l)),
            Array::NumericArray(NumericArray::Float32(r)),
        ) => {
            let lhs_slice = &l.data.as_slice()[lhs_offset..lhs_offset + lhs_len];
            let rhs_slice = &r.data.as_slice()[rhs_offset..rhs_offset + rhs_len];
            Ok(Array::NumericArray(NumericArray::Float32(
                apply_float_f32(lhs_slice, rhs_slice, op, null_mask)?.into(),
            )))
        }
        (
            Array::NumericArray(NumericArray::Float64(l)),
            Array::NumericArray(NumericArray::Float64(r)),
        ) => {
            let lhs_slice = &l.data.as_slice()[lhs_offset..lhs_offset + lhs_len];
            let rhs_slice = &r.data.as_slice()[rhs_offset..rhs_offset + rhs_len];
            Ok(Array::NumericArray(NumericArray::Float64(
                apply_float_f64(lhs_slice, rhs_slice, op, null_mask)?.into(),
            )))
        }

        // Mixed numeric types - promote to higher precision
        (
            Array::NumericArray(NumericArray::Int32(l)),
            Array::NumericArray(NumericArray::Float64(r)),
        ) => {
            let lhs_slice = &l.data.as_slice()[lhs_offset..lhs_offset + lhs_len];
            let rhs_slice = &r.data.as_slice()[rhs_offset..rhs_offset + rhs_len];
            Ok(promote_to_float64!(lhs_slice, rhs_slice))
        }
        (
            Array::NumericArray(NumericArray::Float64(l)),
            Array::NumericArray(NumericArray::Int32(r)),
        ) => {
            let lhs_slice = &l.data.as_slice()[lhs_offset..lhs_offset + lhs_len];
            let rhs_slice = &r.data.as_slice()[rhs_offset..rhs_offset + rhs_len];
            Ok(promote_to_float64!(lhs_slice, rhs_slice))
        }
        (
            Array::NumericArray(NumericArray::Int32(l)),
            Array::NumericArray(NumericArray::Float32(r)),
        ) => {
            let lhs_slice = &l.data.as_slice()[lhs_offset..lhs_offset + lhs_len];
            let rhs_slice = &r.data.as_slice()[rhs_offset..rhs_offset + rhs_len];
            Ok(promote_to_float32!(lhs_slice, rhs_slice))
        }
        (
            Array::NumericArray(NumericArray::Float32(l)),
            Array::NumericArray(NumericArray::Int32(r)),
        ) => {
            let lhs_slice = &l.data.as_slice()[lhs_offset..lhs_offset + lhs_len];
            let rhs_slice = &r.data.as_slice()[rhs_offset..rhs_offset + rhs_len];
            Ok(promote_to_float32!(lhs_slice, rhs_slice))
        }

        // String operations for concatenation
        (Array::TextArray(TextArray::String32(l)), Array::TextArray(TextArray::String32(r))) => {
            if matches!(op, ArithmeticOperator::Add) {
                Ok(Array::TextArray(TextArray::String32(
                    apply_str_str(l, r)?.into(),
                )))
            } else {
                Err(KernelError::UnsupportedType(format!(
                    "Arithmetic operation {:?} not supported for strings",
                    op
                )))
            }
        }
        #[cfg(feature = "large_string")]
        (Array::TextArray(TextArray::String64(l)), Array::TextArray(TextArray::String64(r))) => {
            if matches!(op, ArithmeticOperator::Add) {
                Ok(Array::TextArray(TextArray::String64(
                    apply_str_str(l, r)?.into(),
                )))
            } else {
                Err(KernelError::UnsupportedType(format!(
                    "Arithmetic operation {:?} not supported for strings",
                    op
                )))
            }
        }

        // -----------------------------------------------------------------
        // Decimal same-type dispatch
        // -----------------------------------------------------------------

        #[cfg(feature = "decimal")]
        (
            Array::NumericArray(NumericArray::Decimal32(l)),
            Array::NumericArray(NumericArray::Decimal32(r)),
        ) => {
            let result = decimal_binary(l.as_ref(), r.as_ref(), op)?;
            Ok(Array::NumericArray(NumericArray::Decimal32(result.into())))
        }
        #[cfg(feature = "decimal")]
        (
            Array::NumericArray(NumericArray::Decimal64(l)),
            Array::NumericArray(NumericArray::Decimal64(r)),
        ) => {
            let result = decimal_binary(l.as_ref(), r.as_ref(), op)?;
            Ok(Array::NumericArray(NumericArray::Decimal64(result.into())))
        }
        #[cfg(feature = "decimal")]
        (
            Array::NumericArray(NumericArray::Decimal128(l)),
            Array::NumericArray(NumericArray::Decimal128(r)),
        ) => {
            let result = decimal_binary(l.as_ref(), r.as_ref(), op)?;
            Ok(Array::NumericArray(NumericArray::Decimal128(result.into())))
        }

        // -----------------------------------------------------------------
        // Decimal width promotion (narrower widened to wider)
        // -----------------------------------------------------------------

        #[cfg(feature = "decimal")]
        (
            Array::NumericArray(NumericArray::Decimal32(l)),
            Array::NumericArray(NumericArray::Decimal64(r)),
        ) => {
            let widened = DecimalArray::<i64>::from(l.as_ref().clone());
            let result = decimal_binary(&widened, r.as_ref(), op)?;
            Ok(Array::NumericArray(NumericArray::Decimal64(result.into())))
        }
        #[cfg(feature = "decimal")]
        (
            Array::NumericArray(NumericArray::Decimal64(l)),
            Array::NumericArray(NumericArray::Decimal32(r)),
        ) => {
            let widened = DecimalArray::<i64>::from(r.as_ref().clone());
            let result = decimal_binary(l.as_ref(), &widened, op)?;
            Ok(Array::NumericArray(NumericArray::Decimal64(result.into())))
        }
        #[cfg(feature = "decimal")]
        (
            Array::NumericArray(NumericArray::Decimal32(l)),
            Array::NumericArray(NumericArray::Decimal128(r)),
        ) => {
            let widened: DecimalArray<i128> =
                DecimalArray::<i64>::from(l.as_ref().clone()).into();
            let result = decimal_binary(&widened, r.as_ref(), op)?;
            Ok(Array::NumericArray(NumericArray::Decimal128(result.into())))
        }
        #[cfg(feature = "decimal")]
        (
            Array::NumericArray(NumericArray::Decimal128(l)),
            Array::NumericArray(NumericArray::Decimal32(r)),
        ) => {
            let widened: DecimalArray<i128> =
                DecimalArray::<i64>::from(r.as_ref().clone()).into();
            let result = decimal_binary(l.as_ref(), &widened, op)?;
            Ok(Array::NumericArray(NumericArray::Decimal128(result.into())))
        }
        #[cfg(feature = "decimal")]
        (
            Array::NumericArray(NumericArray::Decimal64(l)),
            Array::NumericArray(NumericArray::Decimal128(r)),
        ) => {
            let widened = DecimalArray::<i128>::from(l.as_ref().clone());
            let result = decimal_binary(&widened, r.as_ref(), op)?;
            Ok(Array::NumericArray(NumericArray::Decimal128(result.into())))
        }
        #[cfg(feature = "decimal")]
        (
            Array::NumericArray(NumericArray::Decimal128(l)),
            Array::NumericArray(NumericArray::Decimal64(r)),
        ) => {
            let widened = DecimalArray::<i128>::from(r.as_ref().clone());
            let result = decimal_binary(l.as_ref(), &widened, op)?;
            Ok(Array::NumericArray(NumericArray::Decimal128(result.into())))
        }

        // -----------------------------------------------------------------
        // Integer + Decimal auto-promotion (integer promoted to decimal)
        // -----------------------------------------------------------------

        #[cfg(feature = "decimal")]
        (
            Array::NumericArray(NumericArray::Int32(l)),
            Array::NumericArray(NumericArray::Decimal32(r)),
        ) => {
            let lhs_slice = &l.data.as_slice()[lhs_offset..lhs_offset + lhs_len];
            let promoted = integer_to_decimal(lhs_slice, l.null_mask.as_ref(), r.precision, r.scale)?;
            let result = decimal_binary(&promoted, r.as_ref(), op)?;
            Ok(Array::NumericArray(NumericArray::Decimal32(result.into())))
        }
        #[cfg(feature = "decimal")]
        (
            Array::NumericArray(NumericArray::Decimal32(l)),
            Array::NumericArray(NumericArray::Int32(r)),
        ) => {
            let rhs_slice = &r.data.as_slice()[rhs_offset..rhs_offset + rhs_len];
            let promoted = integer_to_decimal(rhs_slice, r.null_mask.as_ref(), l.precision, l.scale)?;
            let result = decimal_binary(l.as_ref(), &promoted, op)?;
            Ok(Array::NumericArray(NumericArray::Decimal32(result.into())))
        }
        #[cfg(feature = "decimal")]
        (
            Array::NumericArray(NumericArray::Int64(l)),
            Array::NumericArray(NumericArray::Decimal64(r)),
        ) => {
            let lhs_slice = &l.data.as_slice()[lhs_offset..lhs_offset + lhs_len];
            let promoted = integer_to_decimal(lhs_slice, l.null_mask.as_ref(), r.precision, r.scale)?;
            let result = decimal_binary(&promoted, r.as_ref(), op)?;
            Ok(Array::NumericArray(NumericArray::Decimal64(result.into())))
        }
        #[cfg(feature = "decimal")]
        (
            Array::NumericArray(NumericArray::Decimal64(l)),
            Array::NumericArray(NumericArray::Int64(r)),
        ) => {
            let rhs_slice = &r.data.as_slice()[rhs_offset..rhs_offset + rhs_len];
            let promoted = integer_to_decimal(rhs_slice, r.null_mask.as_ref(), l.precision, l.scale)?;
            let result = decimal_binary(l.as_ref(), &promoted, op)?;
            Ok(Array::NumericArray(NumericArray::Decimal64(result.into())))
        }

        // Int32/Int64 + Decimal128: widen the integer to i128, promote to decimal
        #[cfg(feature = "decimal")]
        (
            Array::NumericArray(NumericArray::Int32(l)),
            Array::NumericArray(NumericArray::Decimal128(r)),
        ) => {
            let lhs_slice = &l.data.as_slice()[lhs_offset..lhs_offset + lhs_len];
            let data_i128: Vec64<i128> = lhs_slice.iter().map(|&v| v as i128).collect();
            let promoted = integer_to_decimal(data_i128.as_slice(), l.null_mask.as_ref(), r.precision, r.scale)?;
            let result = decimal_binary(&promoted, r.as_ref(), op)?;
            Ok(Array::NumericArray(NumericArray::Decimal128(result.into())))
        }
        #[cfg(feature = "decimal")]
        (
            Array::NumericArray(NumericArray::Decimal128(l)),
            Array::NumericArray(NumericArray::Int32(r)),
        ) => {
            let rhs_slice = &r.data.as_slice()[rhs_offset..rhs_offset + rhs_len];
            let data_i128: Vec64<i128> = rhs_slice.iter().map(|&v| v as i128).collect();
            let promoted = integer_to_decimal(data_i128.as_slice(), r.null_mask.as_ref(), l.precision, l.scale)?;
            let result = decimal_binary(l.as_ref(), &promoted, op)?;
            Ok(Array::NumericArray(NumericArray::Decimal128(result.into())))
        }
        #[cfg(feature = "decimal")]
        (
            Array::NumericArray(NumericArray::Int64(l)),
            Array::NumericArray(NumericArray::Decimal128(r)),
        ) => {
            let lhs_slice = &l.data.as_slice()[lhs_offset..lhs_offset + lhs_len];
            let data_i128: Vec64<i128> = lhs_slice.iter().map(|&v| v as i128).collect();
            let promoted = integer_to_decimal(data_i128.as_slice(), l.null_mask.as_ref(), r.precision, r.scale)?;
            let result = decimal_binary(&promoted, r.as_ref(), op)?;
            Ok(Array::NumericArray(NumericArray::Decimal128(result.into())))
        }
        #[cfg(feature = "decimal")]
        (
            Array::NumericArray(NumericArray::Decimal128(l)),
            Array::NumericArray(NumericArray::Int64(r)),
        ) => {
            let rhs_slice = &r.data.as_slice()[rhs_offset..rhs_offset + rhs_len];
            let data_i128: Vec64<i128> = rhs_slice.iter().map(|&v| v as i128).collect();
            let promoted = integer_to_decimal(data_i128.as_slice(), r.null_mask.as_ref(), l.precision, l.scale)?;
            let result = decimal_binary(l.as_ref(), &promoted, op)?;
            Ok(Array::NumericArray(NumericArray::Decimal128(result.into())))
        }

        // -----------------------------------------------------------------
        // Float + Decimal auto-promotion (decimal demoted to Float64)
        // -----------------------------------------------------------------

        #[cfg(feature = "decimal")]
        (
            Array::NumericArray(NumericArray::Float64(l)),
            Array::NumericArray(NumericArray::Decimal32(r)),
        ) => {
            let lhs_slice = &l.data.as_slice()[lhs_offset..lhs_offset + lhs_len];
            let rhs_f64: Vec64<f64> = r.data.as_slice().iter()
                .map(|&v| v as f64 / 10f64.powi(r.scale as i32))
                .collect();
            Ok(Array::NumericArray(NumericArray::Float64(
                apply_float_f64(lhs_slice, &rhs_f64, op, null_mask)?.into(),
            )))
        }
        #[cfg(feature = "decimal")]
        (
            Array::NumericArray(NumericArray::Decimal32(l)),
            Array::NumericArray(NumericArray::Float64(r)),
        ) => {
            let rhs_slice = &r.data.as_slice()[rhs_offset..rhs_offset + rhs_len];
            let lhs_f64: Vec64<f64> = l.data.as_slice().iter()
                .map(|&v| v as f64 / 10f64.powi(l.scale as i32))
                .collect();
            Ok(Array::NumericArray(NumericArray::Float64(
                apply_float_f64(&lhs_f64, rhs_slice, op, null_mask)?.into(),
            )))
        }
        #[cfg(feature = "decimal")]
        (
            Array::NumericArray(NumericArray::Float64(l)),
            Array::NumericArray(NumericArray::Decimal64(r)),
        ) => {
            let lhs_slice = &l.data.as_slice()[lhs_offset..lhs_offset + lhs_len];
            let rhs_f64: Vec64<f64> = r.data.as_slice().iter()
                .map(|&v| v as f64 / 10f64.powi(r.scale as i32))
                .collect();
            Ok(Array::NumericArray(NumericArray::Float64(
                apply_float_f64(lhs_slice, &rhs_f64, op, null_mask)?.into(),
            )))
        }
        #[cfg(feature = "decimal")]
        (
            Array::NumericArray(NumericArray::Decimal64(l)),
            Array::NumericArray(NumericArray::Float64(r)),
        ) => {
            let rhs_slice = &r.data.as_slice()[rhs_offset..rhs_offset + rhs_len];
            let lhs_f64: Vec64<f64> = l.data.as_slice().iter()
                .map(|&v| v as f64 / 10f64.powi(l.scale as i32))
                .collect();
            Ok(Array::NumericArray(NumericArray::Float64(
                apply_float_f64(&lhs_f64, rhs_slice, op, null_mask)?.into(),
            )))
        }
        #[cfg(feature = "decimal")]
        (
            Array::NumericArray(NumericArray::Float64(l)),
            Array::NumericArray(NumericArray::Decimal128(r)),
        ) => {
            use num_traits::ToPrimitive;
            let lhs_slice = &l.data.as_slice()[lhs_offset..lhs_offset + lhs_len];
            let rhs_f64: Vec64<f64> = r.data.as_slice().iter()
                .map(|v| v.to_f64().unwrap() / 10f64.powi(r.scale as i32))
                .collect();
            Ok(Array::NumericArray(NumericArray::Float64(
                apply_float_f64(lhs_slice, &rhs_f64, op, null_mask)?.into(),
            )))
        }
        #[cfg(feature = "decimal")]
        (
            Array::NumericArray(NumericArray::Decimal128(l)),
            Array::NumericArray(NumericArray::Float64(r)),
        ) => {
            use num_traits::ToPrimitive;
            let rhs_slice = &r.data.as_slice()[rhs_offset..rhs_offset + rhs_len];
            let lhs_f64: Vec64<f64> = l.data.as_slice().iter()
                .map(|v| v.to_f64().unwrap() / 10f64.powi(l.scale as i32))
                .collect();
            Ok(Array::NumericArray(NumericArray::Float64(
                apply_float_f64(&lhs_f64, rhs_slice, op, null_mask)?.into(),
            )))
        }

        // Float32 + Decimal -> Float64 (promote both to f64)
        #[cfg(feature = "decimal")]
        (
            Array::NumericArray(NumericArray::Float32(l)),
            Array::NumericArray(NumericArray::Decimal32(r)),
        ) => {
            let lhs_f64: Vec64<f64> = l.data.as_slice()[lhs_offset..lhs_offset + lhs_len]
                .iter().map(|&v| v as f64).collect();
            let rhs_f64: Vec64<f64> = r.data.as_slice().iter()
                .map(|&v| v as f64 / 10f64.powi(r.scale as i32))
                .collect();
            Ok(Array::NumericArray(NumericArray::Float64(
                apply_float_f64(&lhs_f64, &rhs_f64, op, null_mask)?.into(),
            )))
        }
        #[cfg(feature = "decimal")]
        (
            Array::NumericArray(NumericArray::Decimal32(l)),
            Array::NumericArray(NumericArray::Float32(r)),
        ) => {
            let lhs_f64: Vec64<f64> = l.data.as_slice().iter()
                .map(|&v| v as f64 / 10f64.powi(l.scale as i32))
                .collect();
            let rhs_f64: Vec64<f64> = r.data.as_slice()[rhs_offset..rhs_offset + rhs_len]
                .iter().map(|&v| v as f64).collect();
            Ok(Array::NumericArray(NumericArray::Float64(
                apply_float_f64(&lhs_f64, &rhs_f64, op, null_mask)?.into(),
            )))
        }
        #[cfg(feature = "decimal")]
        (
            Array::NumericArray(NumericArray::Float32(l)),
            Array::NumericArray(NumericArray::Decimal64(r)),
        ) => {
            let lhs_f64: Vec64<f64> = l.data.as_slice()[lhs_offset..lhs_offset + lhs_len]
                .iter().map(|&v| v as f64).collect();
            let rhs_f64: Vec64<f64> = r.data.as_slice().iter()
                .map(|&v| v as f64 / 10f64.powi(r.scale as i32))
                .collect();
            Ok(Array::NumericArray(NumericArray::Float64(
                apply_float_f64(&lhs_f64, &rhs_f64, op, null_mask)?.into(),
            )))
        }
        #[cfg(feature = "decimal")]
        (
            Array::NumericArray(NumericArray::Decimal64(l)),
            Array::NumericArray(NumericArray::Float32(r)),
        ) => {
            let lhs_f64: Vec64<f64> = l.data.as_slice().iter()
                .map(|&v| v as f64 / 10f64.powi(l.scale as i32))
                .collect();
            let rhs_f64: Vec64<f64> = r.data.as_slice()[rhs_offset..rhs_offset + rhs_len]
                .iter().map(|&v| v as f64).collect();
            Ok(Array::NumericArray(NumericArray::Float64(
                apply_float_f64(&lhs_f64, &rhs_f64, op, null_mask)?.into(),
            )))
        }
        #[cfg(feature = "decimal")]
        (
            Array::NumericArray(NumericArray::Float32(l)),
            Array::NumericArray(NumericArray::Decimal128(r)),
        ) => {
            use num_traits::ToPrimitive;
            let lhs_f64: Vec64<f64> = l.data.as_slice()[lhs_offset..lhs_offset + lhs_len]
                .iter().map(|&v| v as f64).collect();
            let rhs_f64: Vec64<f64> = r.data.as_slice().iter()
                .map(|v| v.to_f64().unwrap() / 10f64.powi(r.scale as i32))
                .collect();
            Ok(Array::NumericArray(NumericArray::Float64(
                apply_float_f64(&lhs_f64, &rhs_f64, op, null_mask)?.into(),
            )))
        }
        #[cfg(feature = "decimal")]
        (
            Array::NumericArray(NumericArray::Decimal128(l)),
            Array::NumericArray(NumericArray::Float32(r)),
        ) => {
            use num_traits::ToPrimitive;
            let lhs_f64: Vec64<f64> = l.data.as_slice().iter()
                .map(|v| v.to_f64().unwrap() / 10f64.powi(l.scale as i32))
                .collect();
            let rhs_f64: Vec64<f64> = r.data.as_slice()[rhs_offset..rhs_offset + rhs_len]
                .iter().map(|&v| v as f64).collect();
            Ok(Array::NumericArray(NumericArray::Float64(
                apply_float_f64(&lhs_f64, &rhs_f64, op, null_mask)?.into(),
            )))
        }

        // Unsupported combinations
        _ => Err(KernelError::UnsupportedType(
            "Unsupported array type combination for arithmetic operations".to_string(),
        )),
    }
}

// ---------------------------------------------------------------------------
// Decimal dispatch integration tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "decimal")]
mod decimal_dispatch_tests {
    use super::*;
    use crate::{Array, DecimalArray, FloatArray, IntegerArray, MaskedArray, NumericArray};

    fn arr_dec32(vals: &[i32], p: u8, s: i8) -> Array {
        Array::NumericArray(NumericArray::Decimal32(
            DecimalArray::<i32>::from_slice(vals, p, s).into(),
        ))
    }

    fn arr_dec64(vals: &[i64], p: u8, s: i8) -> Array {
        Array::NumericArray(NumericArray::Decimal64(
            DecimalArray::<i64>::from_slice(vals, p, s).into(),
        ))
    }

    fn arr_dec128(vals: &[i128], p: u8, s: i8) -> Array {
        Array::NumericArray(NumericArray::Decimal128(
            DecimalArray::<i128>::from_slice(vals, p, s).into(),
        ))
    }

    fn arr_i32(vals: &[i32]) -> Array {
        Array::from_int32(IntegerArray::from_slice(vals))
    }

    fn arr_i64(vals: &[i64]) -> Array {
        Array::from_int64(IntegerArray::from_slice(vals))
    }

    fn arr_f64(vals: &[f64]) -> Array {
        Array::from_float64(FloatArray::from_slice(vals))
    }

    // -----------------------------------------------------------------------
    // Same-type decimal arithmetic through dispatch
    // -----------------------------------------------------------------------

    #[test]
    fn dispatch_decimal32_add() {
        let result = resolve_binary_arithmetic(
            ArithmeticOperator::Add,
            arr_dec32(&[100, 200, 300], 9, 2),
            arr_dec32(&[10, 20, 30], 9, 2),
            None,
        )
        .unwrap();

        match result {
            Array::NumericArray(NumericArray::Decimal32(a)) => {
                assert_eq!(a.data.as_slice(), &[110, 220, 330]);
                assert_eq!(a.scale, 2);
            }
            _ => panic!("expected Decimal32 result"),
        }
    }

    #[test]
    fn dispatch_decimal64_subtract() {
        let result = resolve_binary_arithmetic(
            ArithmeticOperator::Subtract,
            arr_dec64(&[5000, 3000], 18, 4),
            arr_dec64(&[1000, 2000], 18, 4),
            None,
        )
        .unwrap();

        match result {
            Array::NumericArray(NumericArray::Decimal64(a)) => {
                assert_eq!(a.data.as_slice(), &[4000, 1000]);
                assert_eq!(a.scale, 4);
            }
            _ => panic!("expected Decimal64 result"),
        }
    }

    #[test]
    fn dispatch_decimal128_multiply() {
        let result = resolve_binary_arithmetic(
            ArithmeticOperator::Multiply,
            arr_dec128(&[1000i128], 38, 2),
            arr_dec128(&[200i128], 38, 3),
            None,
        )
        .unwrap();

        match result {
            Array::NumericArray(NumericArray::Decimal128(a)) => {
                assert_eq!(a.data.as_slice(), &[200_000i128]);
                assert_eq!(a.scale, 5); // 2 + 3
            }
            _ => panic!("expected Decimal128 result"),
        }
    }

    // -----------------------------------------------------------------------
    // Cross-scale decimal arithmetic
    // -----------------------------------------------------------------------

    #[test]
    fn dispatch_decimal_add_different_scale() {
        // 10.0 (scale=1) + 1.00 (scale=2) = 11.00 (scale=2)
        let result = resolve_binary_arithmetic(
            ArithmeticOperator::Add,
            arr_dec64(&[100], 18, 1),
            arr_dec64(&[100], 18, 2),
            None,
        )
        .unwrap();

        match result {
            Array::NumericArray(NumericArray::Decimal64(a)) => {
                assert_eq!(a.data.as_slice(), &[1100]); // 100*10 + 100
                assert_eq!(a.scale, 2);
            }
            _ => panic!("expected Decimal64 result"),
        }
    }

    // -----------------------------------------------------------------------
    // Decimal width promotion
    // -----------------------------------------------------------------------

    #[test]
    fn dispatch_decimal32_plus_decimal64_widens() {
        let result = resolve_binary_arithmetic(
            ArithmeticOperator::Add,
            arr_dec32(&[100], 9, 2),
            arr_dec64(&[200], 18, 2),
            None,
        )
        .unwrap();

        match result {
            Array::NumericArray(NumericArray::Decimal64(a)) => {
                assert_eq!(a.data.as_slice(), &[300i64]);
                assert_eq!(a.scale, 2);
            }
            _ => panic!("expected Decimal64 result"),
        }
    }

    #[test]
    fn dispatch_decimal64_plus_decimal128_widens() {
        let result = resolve_binary_arithmetic(
            ArithmeticOperator::Add,
            arr_dec64(&[500], 18, 2),
            arr_dec128(&[600i128], 38, 2),
            None,
        )
        .unwrap();

        match result {
            Array::NumericArray(NumericArray::Decimal128(a)) => {
                assert_eq!(a.data.as_slice(), &[1100i128]);
            }
            _ => panic!("expected Decimal128 result"),
        }
    }

    // -----------------------------------------------------------------------
    // Integer + Decimal auto-promotion
    // -----------------------------------------------------------------------

    #[test]
    fn dispatch_int32_plus_decimal32() {
        // integer 5 + decimal 1.00 (raw=100, scale=2)
        // integer promoted: 5 * 100 = 500 at scale 2
        // 500 + 100 = 600 at scale 2 => 6.00
        let result = resolve_binary_arithmetic(
            ArithmeticOperator::Add,
            arr_i32(&[5]),
            arr_dec32(&[100], 9, 2),
            None,
        )
        .unwrap();

        match result {
            Array::NumericArray(NumericArray::Decimal32(a)) => {
                assert_eq!(a.data.as_slice(), &[600]);
                assert_eq!(a.scale, 2);
            }
            _ => panic!("expected Decimal32 result"),
        }
    }

    #[test]
    fn dispatch_decimal64_minus_int64() {
        // decimal 10.00 (raw=1000, scale=2) - integer 3
        // integer promoted: 3 * 100 = 300 at scale 2
        // 1000 - 300 = 700 at scale 2 => 7.00
        let result = resolve_binary_arithmetic(
            ArithmeticOperator::Subtract,
            arr_dec64(&[1000], 18, 2),
            arr_i64(&[3]),
            None,
        )
        .unwrap();

        match result {
            Array::NumericArray(NumericArray::Decimal64(a)) => {
                assert_eq!(a.data.as_slice(), &[700]);
                assert_eq!(a.scale, 2);
            }
            _ => panic!("expected Decimal64 result"),
        }
    }

    // -----------------------------------------------------------------------
    // Float + Decimal auto-promotion
    // -----------------------------------------------------------------------

    #[test]
    fn dispatch_float64_plus_decimal32() {
        // 2.5 + 1.00 (raw=100, scale=2) = 3.5
        let result = resolve_binary_arithmetic(
            ArithmeticOperator::Add,
            arr_f64(&[2.5]),
            arr_dec32(&[100], 9, 2),
            None,
        )
        .unwrap();

        match result {
            Array::NumericArray(NumericArray::Float64(a)) => {
                assert!((a.data[0] - 3.5).abs() < 1e-12);
            }
            _ => panic!("expected Float64 result, got: {:?}", result),
        }
    }

    #[test]
    fn dispatch_decimal64_times_float64() {
        // 10.00 (raw=1000, scale=2) * 2.0 = 20.0
        let result = resolve_binary_arithmetic(
            ArithmeticOperator::Multiply,
            arr_dec64(&[1000], 18, 2),
            arr_f64(&[2.0]),
            None,
        )
        .unwrap();

        match result {
            Array::NumericArray(NumericArray::Float64(a)) => {
                assert!((a.data[0] - 20.0).abs() < 1e-12);
            }
            _ => panic!("expected Float64 result"),
        }
    }

    // -----------------------------------------------------------------------
    // Overflow detection through dispatch
    // -----------------------------------------------------------------------

    #[test]
    fn dispatch_decimal_overflow_returns_error() {
        let result = resolve_binary_arithmetic(
            ArithmeticOperator::Add,
            arr_dec32(&[i32::MAX], 9, 0),
            arr_dec32(&[1], 9, 0),
            None,
        );
        assert!(result.is_err(), "expected overflow error");
    }

    // -----------------------------------------------------------------------
    // Broadcasting (length-1 decimal)
    // -----------------------------------------------------------------------

    #[test]
    fn dispatch_decimal_broadcast_scalar() {
        // [1.00, 2.00, 3.00] + [0.50] => [1.50, 2.50, 3.50]
        let result = resolve_binary_arithmetic(
            ArithmeticOperator::Add,
            arr_dec64(&[100, 200, 300], 18, 2),
            arr_dec64(&[50], 18, 2),
            None,
        )
        .unwrap();

        match result {
            Array::NumericArray(NumericArray::Decimal64(a)) => {
                assert_eq!(a.data.as_slice(), &[150, 250, 350]);
                assert_eq!(a.len(), 3);
            }
            _ => panic!("expected Decimal64 result"),
        }
    }

    // -----------------------------------------------------------------------
    // Division and remainder
    // -----------------------------------------------------------------------

    #[test]
    fn dispatch_decimal_divide() {
        let result = resolve_binary_arithmetic(
            ArithmeticOperator::Divide,
            arr_dec64(&[10000], 18, 2), // 100.00
            arr_dec64(&[400], 18, 2),   // 4.00
            None,
        )
        .unwrap();

        match result {
            Array::NumericArray(NumericArray::Decimal64(a)) => {
                assert_eq!(a.data.as_slice(), &[2500]); // 25.00
                assert_eq!(a.scale, 2);
            }
            _ => panic!("expected Decimal64 result"),
        }
    }

    #[test]
    fn dispatch_decimal_remainder() {
        let result = resolve_binary_arithmetic(
            ArithmeticOperator::Remainder,
            arr_dec64(&[1000], 18, 2), // 10.00
            arr_dec64(&[300], 18, 2),  // 3.00
            None,
        )
        .unwrap();

        match result {
            Array::NumericArray(NumericArray::Decimal64(a)) => {
                assert_eq!(a.data.as_slice(), &[100]); // 1.00
            }
            _ => panic!("expected Decimal64 result"),
        }
    }
}
