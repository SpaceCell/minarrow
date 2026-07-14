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

//! NdArray broadcasting operations.
//!
//! Elementwise arithmetic between two arrays of the same
//! logical shape, with single-element expansion when one operand holds
//! one value, plus the array-scalar forms in both operand orders.
//!
//! The implementation walks logical elements in column-major order, so
//! non-contiguous operands such as a Matrix-imported array compute
//! correctly. Results are compact column-major. Division follows IEEE
//! 754, and NaN missing values propagate through every operation.

use crate::enums::error::MinarrowError;
use crate::enums::operators::ArithmeticOperator;
use crate::structs::ndarray::NdArray;
use crate::traits::type_unions::Float;
use crate::Vec64;

/// Resolve an elementwise binary operation over two arrays.
///
/// Operands of the same logical shape combine element by element. An
/// operand holding a single value expands across the other. Any other
/// shape pairing is an error.
pub(crate) fn resolve_ndarray_arithmetic<T: Float>(
    op: ArithmeticOperator,
    lhs: &NdArray<T>,
    rhs: &NdArray<T>,
) -> Result<NdArray<T>, MinarrowError> {
    if lhs.shape() == rhs.shape() {
        let flat: Vec64<T> = lhs
            .into_iter()
            .zip(rhs.into_iter())
            .map(|(a, b)| apply_op(op, a, b))
            .collect();
        let mut result = NdArray::from_slice(&flat, lhs.shape());
        result.name = lhs.name.clone();
        return Ok(result);
    }
    if rhs.len() == 1 {
        let scalar = rhs.get(&vec![0; rhs.ndim()]);
        return map_ndarray(lhs, |a| apply_op(op, a, scalar));
    }
    if lhs.len() == 1 {
        let scalar = lhs.get(&vec![0; lhs.ndim()]);
        return map_ndarray(rhs, |b| apply_op(op, scalar, b));
    }
    Err(MinarrowError::ShapeError {
        message: format!(
            "broadcast: shapes {:?} and {:?} are incompatible",
            lhs.shape(),
            rhs.shape()
        ),
    })
}

/// Apply one arithmetic operator to a pair of elements.
#[inline]
fn apply_op<T: Float>(op: ArithmeticOperator, a: T, b: T) -> T {
    match op {
        ArithmeticOperator::Add => a + b,
        ArithmeticOperator::Subtract => a - b,
        ArithmeticOperator::Multiply => a * b,
        ArithmeticOperator::Divide => a / b,
        ArithmeticOperator::Remainder => a % b,
        _ => unreachable!("broadcast ndarray supports add, subtract, multiply, divide, remainder"),
    }
}

/// Map a unary closure over every logical element, producing a compact
/// array with the source's shape and name.
fn map_ndarray<T: Float>(
    arr: &NdArray<T>,
    f: impl Fn(T) -> T,
) -> Result<NdArray<T>, MinarrowError> {
    let flat: Vec64<T> = arr.into_iter().map(f).collect();
    let mut result = NdArray::from_slice(&flat, arr.shape());
    result.name = arr.name.clone();
    Ok(result)
}

/// Resolve an elementwise operation between an array and a scalar.
pub(crate) fn resolve_ndarray_scalar_arithmetic<T: Float>(
    op: ArithmeticOperator,
    lhs: &NdArray<T>,
    scalar: T,
) -> Result<NdArray<T>, MinarrowError> {
    map_ndarray(lhs, |a| apply_op(op, a, scalar))
}

/// Resolve an elementwise operation with the scalar on the left.
pub(crate) fn resolve_scalar_ndarray_arithmetic<T: Float>(
    op: ArithmeticOperator,
    scalar: T,
    rhs: &NdArray<T>,
) -> Result<NdArray<T>, MinarrowError> {
    map_ndarray(rhs, |b| apply_op(op, scalar, b))
}

/// Broadcast addition: `lhs + rhs` with single-element expansion.
pub fn broadcast_ndarray_add<T: Float>(
    lhs: &NdArray<T>,
    rhs: &NdArray<T>,
) -> Result<NdArray<T>, MinarrowError> {
    resolve_ndarray_arithmetic(ArithmeticOperator::Add, lhs, rhs)
}

/// Broadcast subtraction: `lhs - rhs` with single-element expansion.
pub fn broadcast_ndarray_sub<T: Float>(
    lhs: &NdArray<T>,
    rhs: &NdArray<T>,
) -> Result<NdArray<T>, MinarrowError> {
    resolve_ndarray_arithmetic(ArithmeticOperator::Subtract, lhs, rhs)
}

/// Broadcast multiplication: `lhs * rhs` with single-element expansion.
pub fn broadcast_ndarray_mul<T: Float>(
    lhs: &NdArray<T>,
    rhs: &NdArray<T>,
) -> Result<NdArray<T>, MinarrowError> {
    resolve_ndarray_arithmetic(ArithmeticOperator::Multiply, lhs, rhs)
}

/// Broadcast division: `lhs / rhs` with single-element expansion.
/// Follows IEEE 754, yielding infinity or NaN on division by zero.
pub fn broadcast_ndarray_div<T: Float>(
    lhs: &NdArray<T>,
    rhs: &NdArray<T>,
) -> Result<NdArray<T>, MinarrowError> {
    resolve_ndarray_arithmetic(ArithmeticOperator::Divide, lhs, rhs)
}

/// Broadcast remainder: `lhs % rhs` with single-element expansion.
/// Follows IEEE 754, yielding NaN on a zero divisor.
pub fn broadcast_ndarray_rem<T: Float>(
    lhs: &NdArray<T>,
    rhs: &NdArray<T>,
) -> Result<NdArray<T>, MinarrowError> {
    resolve_ndarray_arithmetic(ArithmeticOperator::Remainder, lhs, rhs)
}

/// Broadcast scalar addition: `lhs + scalar` over every element.
pub fn broadcast_ndarray_scalar_add<T: Float>(
    lhs: &NdArray<T>,
    scalar: T,
) -> Result<NdArray<T>, MinarrowError> {
    map_ndarray(lhs, |a| a + scalar)
}

/// Broadcast scalar subtraction: `lhs - scalar` over every element.
pub fn broadcast_ndarray_scalar_sub<T: Float>(
    lhs: &NdArray<T>,
    scalar: T,
) -> Result<NdArray<T>, MinarrowError> {
    map_ndarray(lhs, |a| a - scalar)
}

/// Broadcast scalar multiplication: `lhs * scalar` over every element.
pub fn broadcast_ndarray_scalar_mul<T: Float>(
    lhs: &NdArray<T>,
    scalar: T,
) -> Result<NdArray<T>, MinarrowError> {
    map_ndarray(lhs, |a| a * scalar)
}

/// Broadcast scalar division: `lhs / scalar` over every element.
/// Follows IEEE 754, yielding infinity or NaN on division by zero.
pub fn broadcast_ndarray_scalar_div<T: Float>(
    lhs: &NdArray<T>,
    scalar: T,
) -> Result<NdArray<T>, MinarrowError> {
    map_ndarray(lhs, |a| a / scalar)
}

/// Broadcast scalar addition with the scalar on the left: `scalar + rhs`.
pub fn broadcast_scalar_ndarray_add<T: Float>(
    scalar: T,
    rhs: &NdArray<T>,
) -> Result<NdArray<T>, MinarrowError> {
    map_ndarray(rhs, |b| scalar + b)
}

/// Broadcast scalar subtraction with the scalar on the left: `scalar - rhs`.
pub fn broadcast_scalar_ndarray_sub<T: Float>(
    scalar: T,
    rhs: &NdArray<T>,
) -> Result<NdArray<T>, MinarrowError> {
    map_ndarray(rhs, |b| scalar - b)
}

/// Broadcast scalar multiplication with the scalar on the left: `scalar * rhs`.
pub fn broadcast_scalar_ndarray_mul<T: Float>(
    scalar: T,
    rhs: &NdArray<T>,
) -> Result<NdArray<T>, MinarrowError> {
    map_ndarray(rhs, |b| scalar * b)
}

/// Broadcast scalar division with the scalar on the left: `scalar / rhs`.
/// Follows IEEE 754, yielding infinity or NaN on division by zero.
pub fn broadcast_scalar_ndarray_div<T: Float>(
    scalar: T,
    rhs: &NdArray<T>,
) -> Result<NdArray<T>, MinarrowError> {
    map_ndarray(rhs, |b| scalar / b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Buffer;

    #[test]
    fn add_same_shape_2d() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let b = NdArray::from_slice(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[3, 2]);
        let c = broadcast_ndarray_add(&a, &b).unwrap();
        assert_eq!(c.shape(), &[3, 2]);
        assert_eq!(c.get(&[0, 0]), 11.0);
        assert_eq!(c.get(&[2, 1]), 66.0);
    }

    #[test]
    fn sub_mul_div_values() {
        let a = NdArray::from_slice(&[10.0, 20.0, 30.0], &[3]);
        let b = NdArray::from_slice(&[2.0, 4.0, 5.0], &[3]);
        let sub = broadcast_ndarray_sub(&a, &b).unwrap();
        assert_eq!((&sub).into_iter().collect::<Vec<f64>>(), vec![8.0, 16.0, 25.0]);
        let mul = broadcast_ndarray_mul(&a, &b).unwrap();
        assert_eq!((&mul).into_iter().collect::<Vec<f64>>(), vec![20.0, 80.0, 150.0]);
        let div = broadcast_ndarray_div(&a, &b).unwrap();
        assert_eq!((&div).into_iter().collect::<Vec<f64>>(), vec![5.0, 5.0, 6.0]);
    }

    #[test]
    fn single_element_expands() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let one = NdArray::from_slice(&[10.0], &[1]);
        let c = broadcast_ndarray_add(&a, &one).unwrap();
        assert_eq!((&c).into_iter().collect::<Vec<f64>>(), vec![11.0, 12.0, 13.0]);
        // The reversed order expands the same way. Subtraction keeps
        // operand order: [10] - [1,2,3] = [9,8,7].
        let d = broadcast_ndarray_sub(&one, &a).unwrap();
        assert_eq!((&d).into_iter().collect::<Vec<f64>>(), vec![9.0, 8.0, 7.0]);
    }

    #[test]
    fn shape_mismatch_errors() {
        let a = NdArray::<f64>::new(&[3, 2]);
        let b = NdArray::<f64>::new(&[2, 3]);
        assert!(broadcast_ndarray_add(&a, &b).is_err());
    }

    #[test]
    fn scalar_forms_both_orders() {
        let a = NdArray::from_slice(&[10.0, 20.0, 30.0], &[3]);
        let c = broadcast_ndarray_scalar_sub(&a, 1.0).unwrap();
        assert_eq!((&c).into_iter().collect::<Vec<f64>>(), vec![9.0, 19.0, 29.0]);
        let d = broadcast_scalar_ndarray_sub(100.0, &a).unwrap();
        assert_eq!((&d).into_iter().collect::<Vec<f64>>(), vec![90.0, 80.0, 70.0]);
        let e = broadcast_scalar_ndarray_div(60.0, &a).unwrap();
        assert_eq!((&e).into_iter().collect::<Vec<f64>>(), vec![6.0, 3.0, 2.0]);
    }

    #[test]
    fn non_contiguous_operand() {
        // A custom-stride buffer places column 1 at offset 4 with a gap at
        // index 3. The logical walk skips the gap.
        let buf: Vec64<f64> = vec![1.0, 2.0, 3.0, 99.0, 4.0, 5.0, 6.0].into_iter().collect();
        let a = NdArray::from_buffer(
            Buffer::from_vec64(buf),
            &[3, 2],
            &[1, 4],
        );
        let b = NdArray::from_slice(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &[3, 2]);
        let c = broadcast_ndarray_add(&a, &b).unwrap();
        assert_eq!(
            (&c).into_iter().collect::<Vec<f64>>(),
            vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        );
        assert!(c.is_contiguous());
    }

    #[test]
    fn f32_elements() {
        let a = NdArray::<f32>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        let c = broadcast_ndarray_scalar_mul(&a, 2.0f32).unwrap();
        assert_eq!(c.get(&[1, 1]), 8.0f32);
    }

    #[test]
    fn nan_propagates() {
        let a = NdArray::from_slice(&[1.0, f64::NAN, 3.0], &[3]);
        let b = NdArray::from_slice(&[1.0, 1.0, 1.0], &[3]);
        let c = broadcast_ndarray_add(&a, &b).unwrap();
        assert_eq!(c.get(&[0]), 2.0);
        assert!(c.get(&[1]).is_nan());
        assert_eq!(c.get(&[2]), 4.0);
    }

    #[test]
    fn preserves_name() {
        let mut a = NdArray::from_slice(&[1.0, 2.0], &[2]);
        a.set_name("prices");
        let b = NdArray::from_slice(&[1.0, 1.0], &[2]);
        let c = broadcast_ndarray_add(&a, &b).unwrap();
        assert_eq!(c.name.as_deref(), Some("prices"));
    }

    #[test]
    fn native_operators() {
        let a = NdArray::from_slice(&[10.0, 20.0, 30.0], &[3]);
        let b = NdArray::from_slice(&[2.0, 4.0, 5.0], &[3]);
        let sum = (a.clone() + b.clone()).unwrap();
        assert_eq!((&sum).into_iter().collect::<Vec<f64>>(), vec![12.0, 24.0, 35.0]);
        let diff = (a.clone() - b.clone()).unwrap();
        assert_eq!((&diff).into_iter().collect::<Vec<f64>>(), vec![8.0, 16.0, 25.0]);
        let prod = (a.clone() * b.clone()).unwrap();
        assert_eq!((&prod).into_iter().collect::<Vec<f64>>(), vec![20.0, 80.0, 150.0]);
        let quot = (a.clone() / b.clone()).unwrap();
        assert_eq!((&quot).into_iter().collect::<Vec<f64>>(), vec![5.0, 5.0, 6.0]);
        let rem = (a % b).unwrap();
        assert_eq!((&rem).into_iter().collect::<Vec<f64>>(), vec![0.0, 0.0, 0.0]);
    }

    #[cfg(feature = "views")]
    #[test]
    fn native_operators_on_views() {
        use std::sync::Arc;
        let a = Arc::new(NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let b = Arc::new(NdArray::from_slice(&[1.0, 1.0, 1.0, 1.0], &[2, 2]));
        let sum = (a.as_view() + b.as_view()).unwrap();
        assert_eq!(sum.get(&[1, 1]), 5.0);
    }

    #[cfg(feature = "value_type")]
    #[test]
    fn value_pair_and_scalar() {
        use std::sync::Arc;
        use crate::{Scalar, Value};

        let a = Value::NdArray(Arc::new(NdArray::from_slice(&[1.0, 2.0, 3.0], &[3])));
        let b = Value::NdArray(Arc::new(NdArray::from_slice(&[10.0, 10.0, 10.0], &[3])));
        let sum = (a.clone() + b).unwrap();
        let Value::NdArray(nd) = sum else {
            panic!("expected Value::NdArray");
        };
        assert_eq!((&*nd).into_iter().collect::<Vec<f64>>(), vec![11.0, 12.0, 13.0]);

        let scaled = (a * Value::Scalar(Scalar::Float64(3.0))).unwrap();
        let Value::NdArray(nd) = scaled else {
            panic!("expected Value::NdArray");
        };
        assert_eq!((&*nd).into_iter().collect::<Vec<f64>>(), vec![3.0, 6.0, 9.0]);
    }

    #[cfg(feature = "value_type")]
    #[test]
    fn value_rejects_mixed_container() {
        use std::sync::Arc;
        use crate::{Array, FloatArray, Value};

        let nd = Value::NdArray(Arc::new(NdArray::from_slice(&[1.0, 2.0], &[2])));
        let arr = Value::Array(Arc::new(Array::from_float64(FloatArray::from_slice(&[
            1.0, 2.0,
        ]))));
        assert!((nd + arr).is_err());
    }
}
