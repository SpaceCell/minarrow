// Copyright 2025-2026 Peter Garfield Bower. All Rights Reserved.
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
        (Scalar::Int32(l), Scalar::Int32(r), Divide) => Scalar::Int32(l / r),

        // Int64 operations
        (Scalar::Int64(l), Scalar::Int64(r), Add) => Scalar::Int64(l + r),
        (Scalar::Int64(l), Scalar::Int64(r), Subtract) => Scalar::Int64(l - r),
        (Scalar::Int64(l), Scalar::Int64(r), Multiply) => Scalar::Int64(l * r),
        (Scalar::Int64(l), Scalar::Int64(r), Divide) => Scalar::Int64(l / r),

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
        (Scalar::Int8(l), Scalar::Int8(r), Divide) => Scalar::Int8(l / r),

        // Int16
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::Int16(l), Scalar::Int16(r), Add) => Scalar::Int16(l + r),
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::Int16(l), Scalar::Int16(r), Subtract) => Scalar::Int16(l - r),
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::Int16(l), Scalar::Int16(r), Multiply) => Scalar::Int16(l * r),
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::Int16(l), Scalar::Int16(r), Divide) => Scalar::Int16(l / r),

        // UInt8
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::UInt8(l), Scalar::UInt8(r), Add) => Scalar::UInt8(l + r),
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::UInt8(l), Scalar::UInt8(r), Subtract) => Scalar::UInt8(l - r),
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::UInt8(l), Scalar::UInt8(r), Multiply) => Scalar::UInt8(l * r),
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::UInt8(l), Scalar::UInt8(r), Divide) => Scalar::UInt8(l / r),

        // UInt16
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::UInt16(l), Scalar::UInt16(r), Add) => Scalar::UInt16(l + r),
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::UInt16(l), Scalar::UInt16(r), Subtract) => Scalar::UInt16(l - r),
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::UInt16(l), Scalar::UInt16(r), Multiply) => Scalar::UInt16(l * r),
        #[cfg(feature = "extended_numeric_types")]
        (Scalar::UInt16(l), Scalar::UInt16(r), Divide) => Scalar::UInt16(l / r),

        // UInt32
        (Scalar::UInt32(l), Scalar::UInt32(r), Add) => Scalar::UInt32(l + r),
        (Scalar::UInt32(l), Scalar::UInt32(r), Subtract) => Scalar::UInt32(l - r),
        (Scalar::UInt32(l), Scalar::UInt32(r), Multiply) => Scalar::UInt32(l * r),
        (Scalar::UInt32(l), Scalar::UInt32(r), Divide) => Scalar::UInt32(l / r),

        // UInt64
        (Scalar::UInt64(l), Scalar::UInt64(r), Add) => Scalar::UInt64(l + r),
        (Scalar::UInt64(l), Scalar::UInt64(r), Subtract) => Scalar::UInt64(l - r),
        (Scalar::UInt64(l), Scalar::UInt64(r), Multiply) => Scalar::UInt64(l * r),
        (Scalar::UInt64(l), Scalar::UInt64(r), Divide) => Scalar::UInt64(l / r),
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

        // Unsupported combinations
        _ => Err(KernelError::UnsupportedType(
            "Unsupported array type combination for arithmetic operations".to_string(),
        )),
    }
}
