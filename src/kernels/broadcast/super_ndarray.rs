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

//! SuperNdArray broadcasting operations.
//!
//! Elementary elementwise arithmetic between two chunked arrays whose
//! batches align, plus the array-scalar forms in both operand orders.
//! Every operation preserves the chunk boundaries of the left operand.
//!
//! Two chunked arrays combine batch by batch, which requires identical
//! chunk boundaries. Operands chunked differently return an error with
//! guidance to `rechunk` or `consolidate` first, rather than silently
//! materialising.

use crate::enums::error::MinarrowError;
use crate::enums::operators::ArithmeticOperator;
use crate::kernels::broadcast::ndarray::{
    broadcast_ndarray_scalar_add, broadcast_ndarray_scalar_div, broadcast_ndarray_scalar_mul,
    broadcast_ndarray_scalar_sub, broadcast_scalar_ndarray_add, broadcast_scalar_ndarray_div,
    broadcast_scalar_ndarray_mul, broadcast_scalar_ndarray_sub, resolve_ndarray_arithmetic,
    resolve_ndarray_scalar_arithmetic, resolve_scalar_ndarray_arithmetic,
};
use crate::structs::chunked::super_ndarray::SuperNdArray;
use crate::structs::ndarray::NdArray;
use crate::traits::type_unions::Float;

/// Resolve an elementwise binary operation over two chunked arrays.
///
/// Batches combine pairwise, so both operands must share the same chunk
/// boundaries as well as the same logical shape.
pub(crate) fn resolve_super_ndarray_arithmetic<T: Float>(
    op: ArithmeticOperator,
    lhs: &SuperNdArray<T>,
    rhs: &SuperNdArray<T>,
) -> Result<SuperNdArray<T>, MinarrowError> {
    if lhs.ndim() != rhs.ndim() || lhs.inner_shape() != rhs.inner_shape() {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "broadcast: shapes {:?} and {:?} are incompatible",
                lhs.shape(),
                rhs.shape()
            ),
        });
    }
    let boundaries_match = lhs.n_batches() == rhs.n_batches()
        && lhs
            .iter_batches()
            .zip(rhs.iter_batches())
            .all(|(a, b)| a.shape()[0] == b.shape()[0]);
    if !boundaries_match {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "broadcast: chunk boundaries differ ({} vs {} batches), rechunk or consolidate first",
                lhs.n_batches(),
                rhs.n_batches()
            ),
        });
    }

    let batches: Result<Vec<NdArray<T>>, MinarrowError> = lhs
        .iter_batches()
        .zip(rhs.iter_batches())
        .map(|(a, b)| resolve_ndarray_arithmetic(op, a, b))
        .collect();
    Ok(SuperNdArray::from_batches(batches?, lhs.name.clone()))
}

/// Map a unary closure over every batch, preserving chunk boundaries.
fn map_super_ndarray<T: Float>(
    arr: &SuperNdArray<T>,
    f: impl Fn(&NdArray<T>) -> Result<NdArray<T>, MinarrowError>,
) -> Result<SuperNdArray<T>, MinarrowError> {
    let batches: Result<Vec<NdArray<T>>, MinarrowError> =
        arr.iter_batches().map(|b| f(b)).collect();
    Ok(SuperNdArray::from_batches(batches?, arr.name.clone()))
}

/// Resolve an elementwise operation between a chunked array and a scalar,
/// preserving chunk boundaries.
pub(crate) fn resolve_super_ndarray_scalar_arithmetic<T: Float>(
    op: ArithmeticOperator,
    lhs: &SuperNdArray<T>,
    scalar: T,
) -> Result<SuperNdArray<T>, MinarrowError> {
    map_super_ndarray(lhs, |b| resolve_ndarray_scalar_arithmetic(op, b, scalar))
}

/// Resolve an elementwise operation with the scalar on the left,
/// preserving chunk boundaries.
pub(crate) fn resolve_scalar_super_ndarray_arithmetic<T: Float>(
    op: ArithmeticOperator,
    scalar: T,
    rhs: &SuperNdArray<T>,
) -> Result<SuperNdArray<T>, MinarrowError> {
    map_super_ndarray(rhs, |b| resolve_scalar_ndarray_arithmetic(op, scalar, b))
}

/// Broadcast addition: `lhs + rhs` batch by batch.
pub fn broadcast_super_ndarray_add<T: Float>(
    lhs: &SuperNdArray<T>,
    rhs: &SuperNdArray<T>,
) -> Result<SuperNdArray<T>, MinarrowError> {
    resolve_super_ndarray_arithmetic(ArithmeticOperator::Add, lhs, rhs)
}

/// Broadcast subtraction: `lhs - rhs` batch by batch.
pub fn broadcast_super_ndarray_sub<T: Float>(
    lhs: &SuperNdArray<T>,
    rhs: &SuperNdArray<T>,
) -> Result<SuperNdArray<T>, MinarrowError> {
    resolve_super_ndarray_arithmetic(ArithmeticOperator::Subtract, lhs, rhs)
}

/// Broadcast multiplication: `lhs * rhs` batch by batch.
pub fn broadcast_super_ndarray_mul<T: Float>(
    lhs: &SuperNdArray<T>,
    rhs: &SuperNdArray<T>,
) -> Result<SuperNdArray<T>, MinarrowError> {
    resolve_super_ndarray_arithmetic(ArithmeticOperator::Multiply, lhs, rhs)
}

/// Broadcast division: `lhs / rhs` batch by batch.
/// Follows IEEE 754, yielding infinity or NaN on division by zero.
pub fn broadcast_super_ndarray_div<T: Float>(
    lhs: &SuperNdArray<T>,
    rhs: &SuperNdArray<T>,
) -> Result<SuperNdArray<T>, MinarrowError> {
    resolve_super_ndarray_arithmetic(ArithmeticOperator::Divide, lhs, rhs)
}

/// Broadcast remainder: `lhs % rhs` batch by batch.
/// Follows IEEE 754, yielding NaN on a zero divisor.
pub fn broadcast_super_ndarray_rem<T: Float>(
    lhs: &SuperNdArray<T>,
    rhs: &SuperNdArray<T>,
) -> Result<SuperNdArray<T>, MinarrowError> {
    resolve_super_ndarray_arithmetic(ArithmeticOperator::Remainder, lhs, rhs)
}

/// Broadcast scalar addition: `lhs + scalar` over every element.
pub fn broadcast_super_ndarray_scalar_add<T: Float>(
    lhs: &SuperNdArray<T>,
    scalar: T,
) -> Result<SuperNdArray<T>, MinarrowError> {
    map_super_ndarray(lhs, |b| {
        broadcast_ndarray_scalar_add(b, scalar)
    })
}

/// Broadcast scalar subtraction: `lhs - scalar` over every element.
pub fn broadcast_super_ndarray_scalar_sub<T: Float>(
    lhs: &SuperNdArray<T>,
    scalar: T,
) -> Result<SuperNdArray<T>, MinarrowError> {
    map_super_ndarray(lhs, |b| {
        broadcast_ndarray_scalar_sub(b, scalar)
    })
}

/// Broadcast scalar multiplication: `lhs * scalar` over every element.
pub fn broadcast_super_ndarray_scalar_mul<T: Float>(
    lhs: &SuperNdArray<T>,
    scalar: T,
) -> Result<SuperNdArray<T>, MinarrowError> {
    map_super_ndarray(lhs, |b| {
        broadcast_ndarray_scalar_mul(b, scalar)
    })
}

/// Broadcast scalar division: `lhs / scalar` over every element.
/// Follows IEEE 754, yielding infinity or NaN on division by zero.
pub fn broadcast_super_ndarray_scalar_div<T: Float>(
    lhs: &SuperNdArray<T>,
    scalar: T,
) -> Result<SuperNdArray<T>, MinarrowError> {
    map_super_ndarray(lhs, |b| {
        broadcast_ndarray_scalar_div(b, scalar)
    })
}

/// Broadcast scalar addition with the scalar on the left: `scalar + rhs`.
pub fn broadcast_scalar_super_ndarray_add<T: Float>(
    scalar: T,
    rhs: &SuperNdArray<T>,
) -> Result<SuperNdArray<T>, MinarrowError> {
    map_super_ndarray(rhs, |b| {
        broadcast_scalar_ndarray_add(scalar, b)
    })
}

/// Broadcast scalar subtraction with the scalar on the left: `scalar - rhs`.
pub fn broadcast_scalar_super_ndarray_sub<T: Float>(
    scalar: T,
    rhs: &SuperNdArray<T>,
) -> Result<SuperNdArray<T>, MinarrowError> {
    map_super_ndarray(rhs, |b| {
        broadcast_scalar_ndarray_sub(scalar, b)
    })
}

/// Broadcast scalar multiplication with the scalar on the left: `scalar * rhs`.
pub fn broadcast_scalar_super_ndarray_mul<T: Float>(
    scalar: T,
    rhs: &SuperNdArray<T>,
) -> Result<SuperNdArray<T>, MinarrowError> {
    map_super_ndarray(rhs, |b| {
        broadcast_scalar_ndarray_mul(scalar, b)
    })
}

/// Broadcast scalar division with the scalar on the left: `scalar / rhs`.
/// Follows IEEE 754, yielding infinity or NaN on division by zero.
pub fn broadcast_scalar_super_ndarray_div<T: Float>(
    scalar: T,
    rhs: &SuperNdArray<T>,
) -> Result<SuperNdArray<T>, MinarrowError> {
    map_super_ndarray(rhs, |b| {
        broadcast_scalar_ndarray_div(scalar, b)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunked(values: &[&[f64]], inner: &[usize], name: &str) -> SuperNdArray<f64> {
        let batches: Vec<NdArray<f64>> = values
            .iter()
            .map(|v| {
                let n = v.len() / inner.iter().product::<usize>().max(1);
                let mut shape = vec![n];
                shape.extend_from_slice(inner);
                NdArray::from_slice(v, &shape)
            })
            .collect();
        SuperNdArray::from_batches(batches, name)
    }

    #[test]
    fn add_matching_chunks() {
        let a = chunked(&[&[1.0, 2.0], &[3.0, 4.0, 5.0]], &[], "a");
        let b = chunked(&[&[10.0, 10.0], &[10.0, 10.0, 10.0]], &[], "b");
        let c = broadcast_super_ndarray_add(&a, &b).unwrap();
        assert_eq!(c.n_batches(), 2);
        let vals: Vec<f64> = (&c).into_iter().collect();
        assert_eq!(vals, vec![11.0, 12.0, 13.0, 14.0, 15.0]);
        assert_eq!(c.name, "a");
    }

    #[test]
    fn mismatched_chunk_boundaries_error() {
        let a = chunked(&[&[1.0, 2.0], &[3.0, 4.0]], &[], "a");
        let b = chunked(&[&[1.0, 2.0, 3.0, 4.0]], &[], "b");
        let err = broadcast_super_ndarray_add(&a, &b).unwrap_err();
        assert!(format!("{}", err).contains("rechunk"));
    }

    #[test]
    fn inner_shape_mismatch_errors() {
        let a = chunked(&[&[1.0, 2.0, 3.0, 4.0]], &[2], "a");
        let b = chunked(&[&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], &[3], "b");
        assert!(broadcast_super_ndarray_mul(&a, &b).is_err());
    }

    #[test]
    fn scalar_preserves_chunking() {
        let a = chunked(&[&[1.0, 2.0], &[3.0]], &[], "stream");
        let c = broadcast_super_ndarray_scalar_mul(&a, 10.0).unwrap();
        assert_eq!(c.n_batches(), 2);
        assert_eq!(c.batch(0).unwrap().shape(), &[2]);
        assert_eq!(c.batch(1).unwrap().shape(), &[1]);
        let vals: Vec<f64> = (&c).into_iter().collect();
        assert_eq!(vals, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn scalar_left_order() {
        let a = chunked(&[&[2.0, 4.0], &[5.0]], &[], "a");
        let c = broadcast_scalar_super_ndarray_div(20.0, &a).unwrap();
        let vals: Vec<f64> = (&c).into_iter().collect();
        assert_eq!(vals, vec![10.0, 5.0, 4.0]);
    }

    #[test]
    fn add_2d_batches() {
        let a = chunked(&[&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]], &[2], "a");
        let b = chunked(&[&[1.0, 1.0, 1.0, 1.0], &[1.0, 1.0, 1.0, 1.0]], &[2], "b");
        let c = broadcast_super_ndarray_add(&a, &b).unwrap();
        assert_eq!(c.shape(), vec![4, 2]);
        assert_eq!(c.get(&[0, 0]), 2.0);
        assert_eq!(c.get(&[3, 1]), 9.0);
    }

    #[test]
    fn f32_scalar() {
        let a = SuperNdArray::from_batches(
            vec![NdArray::<f32>::from_slice(&[1.0f32, 2.0], &[2])],
            "f32",
        );
        let c = broadcast_super_ndarray_scalar_add(&a, 1.0f32).unwrap();
        let vals: Vec<f32> = (&c).into_iter().collect();
        assert_eq!(vals, vec![2.0f32, 3.0]);
    }

    #[test]
    fn native_operators() {
        let a = chunked(&[&[1.0, 2.0], &[3.0]], &[], "a");
        let b = chunked(&[&[10.0, 10.0], &[10.0]], &[], "b");
        let sum = (a.clone() + b.clone()).unwrap();
        assert_eq!((&sum).into_iter().collect::<Vec<f64>>(), vec![11.0, 12.0, 13.0]);
        let prod = (a * b).unwrap();
        assert_eq!((&prod).into_iter().collect::<Vec<f64>>(), vec![10.0, 20.0, 30.0]);
    }

    #[cfg(feature = "value_type")]
    #[test]
    fn value_pair_preserves_chunking() {
        use std::sync::Arc;
        use crate::Value;

        let a = Value::SuperNdArray(Arc::new(chunked(&[&[1.0, 2.0], &[3.0]], &[], "a")));
        let b = Value::SuperNdArray(Arc::new(chunked(&[&[1.0, 1.0], &[1.0]], &[], "b")));
        let sum = (a + b).unwrap();
        let Value::SuperNdArray(snd) = sum else {
            panic!("expected Value::SuperNdArray");
        };
        assert_eq!(snd.n_batches(), 2);
        assert_eq!((&*snd).into_iter().collect::<Vec<f64>>(), vec![2.0, 3.0, 4.0]);
    }
}
