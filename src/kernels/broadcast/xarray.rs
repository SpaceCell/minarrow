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

//! XArray broadcasting operations.
//!
//! Elementary elementwise arithmetic between two labelled arrays whose
//! axes match by name and coordinates. The left operand's axes carry
//! through to the result.
//!
//! Two chunked operands combine batch by batch, preserving their chunk
//! boundaries. Every other storage pairing materialises to owned arrays
//! first, since view and chunked layouts cannot combine in place.

use crate::enums::error::MinarrowError;
use crate::enums::operators::ArithmeticOperator;
use crate::kernels::broadcast::ndarray::resolve_ndarray_arithmetic;
#[cfg(feature = "chunked")]
use crate::kernels::broadcast::super_ndarray::resolve_super_ndarray_arithmetic;
use crate::structs::ndarray::NdArray;
use crate::structs::xarray::{NdArrayE, XArray};
use crate::traits::type_unions::Float;

/// Resolve an elementwise binary operation over two labelled arrays.
///
/// Axes must match by name and coordinates. The result carries the left
/// operand's axes.
pub(crate) fn resolve_xarray_arithmetic<T: Float>(
    op: ArithmeticOperator,
    lhs: &XArray<T>,
    rhs: &XArray<T>,
) -> Result<XArray<T>, MinarrowError> {
    if lhs.axes() != rhs.axes() {
        return Err(MinarrowError::IncompatibleTypeError {
            from: "XArray",
            to: "XArray",
            message: Some(format!(
                "broadcast: axes {:?} and {:?} do not match",
                lhs.dim_names(),
                rhs.dim_names()
            )),
        });
    }

    // Chunked pairs combine batch by batch, preserving boundaries.
    #[cfg(feature = "chunked")]
    if let (NdArrayE::Chunked(a), NdArrayE::Chunked(b)) = (lhs.storage(), rhs.storage()) {
        let out = resolve_super_ndarray_arithmetic(op, a, b)?;
        return Ok(XArray::from_storage(
            NdArrayE::Chunked(out),
            lhs.axes().to_vec(),
        ));
    }

    // Remaining pairings materialise to owned arrays. `to_owned` is a
    // refcount bump when the storage is already owned.
    let l = lhs.to_owned();
    let r = rhs.to_owned();
    let (NdArrayE::Owned(a), NdArrayE::Owned(b)) = (l.storage(), r.storage()) else {
        unreachable!("to_owned yields owned storage");
    };
    let out = resolve_ndarray_arithmetic(op, a, b)?;
    Ok(XArray::from_storage(
        NdArrayE::Owned(out),
        lhs.axes().to_vec(),
    ))
}

/// Broadcast addition: `lhs + rhs` with matching axes.
pub fn broadcast_xarray_add<T: Float>(
    lhs: &XArray<T>,
    rhs: &XArray<T>,
) -> Result<XArray<T>, MinarrowError> {
    resolve_xarray_arithmetic(ArithmeticOperator::Add, lhs, rhs)
}

/// Broadcast subtraction: `lhs - rhs` with matching axes.
pub fn broadcast_xarray_sub<T: Float>(
    lhs: &XArray<T>,
    rhs: &XArray<T>,
) -> Result<XArray<T>, MinarrowError> {
    resolve_xarray_arithmetic(ArithmeticOperator::Subtract, lhs, rhs)
}

/// Broadcast multiplication: `lhs * rhs` with matching axes.
pub fn broadcast_xarray_mul<T: Float>(
    lhs: &XArray<T>,
    rhs: &XArray<T>,
) -> Result<XArray<T>, MinarrowError> {
    resolve_xarray_arithmetic(ArithmeticOperator::Multiply, lhs, rhs)
}

/// Broadcast division: `lhs / rhs` with matching axes.
/// Follows IEEE 754, yielding infinity or NaN on division by zero.
pub fn broadcast_xarray_div<T: Float>(
    lhs: &XArray<T>,
    rhs: &XArray<T>,
) -> Result<XArray<T>, MinarrowError> {
    resolve_xarray_arithmetic(ArithmeticOperator::Divide, lhs, rhs)
}

/// Broadcast remainder: `lhs % rhs` with matching axes.
/// Follows IEEE 754, yielding NaN on a zero divisor.
pub fn broadcast_xarray_rem<T: Float>(
    lhs: &XArray<T>,
    rhs: &XArray<T>,
) -> Result<XArray<T>, MinarrowError> {
    resolve_xarray_arithmetic(ArithmeticOperator::Remainder, lhs, rhs)
}

/// Resolve an elementwise operation between a labelled array and a scalar.
/// The array's axes carry through unchanged.
pub(crate) fn resolve_xarray_scalar_arithmetic<T: Float>(
    op: ArithmeticOperator,
    lhs: &XArray<T>,
    scalar: T,
) -> Result<XArray<T>, MinarrowError> {
    let l = lhs.to_owned();
    let NdArrayE::Owned(a) = l.storage() else {
        unreachable!("to_owned yields owned storage");
    };
    let single = NdArray::from_slice(&[scalar], &[1]);
    let out = resolve_ndarray_arithmetic(op, a, &single)?;
    Ok(XArray::from_storage(
        NdArrayE::Owned(out),
        lhs.axes().to_vec(),
    ))
}

/// Resolve an elementwise operation with the scalar on the left.
/// The array's axes carry through unchanged.
pub(crate) fn resolve_scalar_xarray_arithmetic<T: Float>(
    op: ArithmeticOperator,
    scalar: T,
    rhs: &XArray<T>,
) -> Result<XArray<T>, MinarrowError> {
    let r = rhs.to_owned();
    let NdArrayE::Owned(b) = r.storage() else {
        unreachable!("to_owned yields owned storage");
    };
    let single = NdArray::from_slice(&[scalar], &[1]);
    let out = resolve_ndarray_arithmetic(op, &single, b)?;
    Ok(XArray::from_storage(
        NdArrayE::Owned(out),
        rhs.axes().to_vec(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_matching_axes() {
        let a = XArray::new(
            NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
            &["obs", "feat"],
        );
        let b = XArray::new(
            NdArray::from_slice(&[10.0, 10.0, 10.0, 10.0], &[2, 2]),
            &["obs", "feat"],
        );
        let c = broadcast_xarray_add(&a, &b).unwrap();
        assert_eq!(c.dim_names(), vec!["obs", "feat"]);
        assert_eq!(c.get(&[0, 0]), 11.0);
        assert_eq!(c.get(&[1, 1]), 14.0);
    }

    #[test]
    fn axes_mismatch_errors() {
        let a = XArray::new(NdArray::from_slice(&[1.0, 2.0], &[2]), &["time"]);
        let b = XArray::new(NdArray::from_slice(&[1.0, 2.0], &[2]), &["depth"]);
        assert!(broadcast_xarray_mul(&a, &b).is_err());
    }

    #[cfg(feature = "chunked")]
    #[test]
    fn chunked_pair_preserves_batches() {
        use crate::structs::chunked::super_ndarray::SuperNdArray;

        let a = XArray::from_batches(
            SuperNdArray::from_batches(
                vec![
                    NdArray::from_slice(&[1.0, 2.0], &[2]),
                    NdArray::from_slice(&[3.0], &[1]),
                ],
                "a",
            ),
            &["time"],
        );
        let b = XArray::from_batches(
            SuperNdArray::from_batches(
                vec![
                    NdArray::from_slice(&[10.0, 10.0], &[2]),
                    NdArray::from_slice(&[10.0], &[1]),
                ],
                "b",
            ),
            &["time"],
        );
        let c = broadcast_xarray_add(&a, &b).unwrap();
        assert!(!c.is_owned());
        let vals: Vec<f64> = (&c).into_iter().collect();
        assert_eq!(vals, vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn native_operators() {
        let a = XArray::new(NdArray::from_slice(&[2.0, 4.0], &[2]), &["time"]);
        let b = XArray::new(NdArray::from_slice(&[1.0, 2.0], &[2]), &["time"]);
        let sum = (a.clone() + b.clone()).unwrap();
        assert_eq!(sum.get(&[0]), 3.0);
        assert_eq!(sum.get(&[1]), 6.0);
        let quot = (a / b).unwrap();
        assert_eq!(quot.get(&[0]), 2.0);
        assert_eq!(quot.get(&[1]), 2.0);
    }

    #[cfg(feature = "value_type")]
    #[test]
    fn value_pair_and_scalar() {
        use std::sync::Arc;

        use crate::{Scalar, Value};

        let a = Value::XArray(Arc::new(XArray::new(
            NdArray::from_slice(&[1.0, 2.0], &[2]),
            &["time"],
        )));
        let scaled = (a * Value::Scalar(Scalar::Float64(10.0))).unwrap();
        let Value::XArray(xa) = scaled else {
            panic!("expected Value::XArray");
        };
        assert_eq!(xa.get(&[0]), 10.0);
        assert_eq!(xa.get(&[1]), 20.0);
        assert_eq!(xa.dim_names(), vec!["time"]);
    }
}
