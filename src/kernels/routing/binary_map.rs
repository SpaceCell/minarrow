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

//! # Binary Map Module
//!
//! Generic binary function application with broadcasting support.

use crate::Numeric;
use crate::enums::error::KernelError;
use crate::kernels::routing::broadcast::maybe_broadcast_scalar_array;
use crate::{Array, ArrayV, Vec64};

/// Apply a binary function element-wise with broadcasting.
///
/// Both inputs are cast to type `T`, the function is applied element-wise,
/// and the result is returned as an Array of type `T`.
///
/// # Type Parameter
/// - `T`: The numeric type to cast inputs to and produce output as.
///   Must implement `Numeric` for casting support, and `Vec64<T>` must
///   be convertible to `Array`.
///
/// # Broadcasting
/// If one input has length 1 and the other has length N, the length-1
/// input is broadcast to match. This enables scalar-array operations
/// when combined with `From<Scalar> for ArrayV`.
///
/// # Example
/// ```ignore
/// use minarrow::{Array, Scalar, kernels::routing::binary_map};
///
/// let arr = Array::from(Vec64::from(&[1.0f64, 2.0, 3.0]));
///
/// // Array + Scalar (broadcasts 10.0 to [10.0, 10.0, 10.0])
/// let result = binary_map::<f64, _>(arr.clone(), Scalar::Float64(10.0), |a, b| a + b)?;
///
/// // Two arrays
/// let arr2 = Array::from(Vec64::from(&[10.0f64, 20.0, 30.0]));
/// let result = binary_map::<f64, _>(arr, arr2, |a, b| a * b)?;
/// ```
pub fn binary_map<T, F>(
    lhs: impl Into<ArrayV>,
    rhs: impl Into<ArrayV>,
    f: F,
) -> Result<Array, KernelError>
where
    T: Numeric,
    Vec64<T>: Into<Array>,
    F: Fn(T, T) -> T,
{
    let (lhs, rhs) = maybe_broadcast_scalar_array(lhs.into(), rhs.into())?;

    let lhs_vec = lhs.to_typed_vec::<T>()?;
    let rhs_vec = rhs.to_typed_vec::<T>()?;

    let result: Vec64<T> = lhs_vec
        .iter()
        .zip(rhs_vec.iter())
        .map(|(a, b)| f(*a, *b))
        .collect();

    Ok(result.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NumericArray;

    fn make_f64_array(values: &[f64]) -> Array {
        Vec64::from(values).into()
    }

    fn make_i32_array(values: &[i32]) -> Array {
        Vec64::from(values).into()
    }

    #[test]
    fn test_binary_map_two_arrays() {
        let arr1 = make_f64_array(&[1.0, 2.0, 3.0]);
        let arr2 = make_f64_array(&[10.0, 20.0, 30.0]);

        let result = binary_map::<f64, _>(arr1, arr2, |a, b| a + b).unwrap();

        match result {
            Array::NumericArray(NumericArray::Float64(a)) => {
                assert_eq!(a.data.as_slice(), &[11.0, 22.0, 33.0]);
            }
            _ => panic!("Expected Float64 array"),
        }
    }

    #[test]
    fn test_binary_map_broadcast_scalar() {
        let arr = make_f64_array(&[1.0, 2.0, 3.0]);
        let scalar_arr = make_f64_array(&[10.0]); // length-1 array simulates scalar

        let result = binary_map::<f64, _>(arr, scalar_arr, |a, b| a * b).unwrap();

        match result {
            Array::NumericArray(NumericArray::Float64(a)) => {
                assert_eq!(a.data.as_slice(), &[10.0, 20.0, 30.0]);
            }
            _ => panic!("Expected Float64 array"),
        }
    }

    #[test]
    fn test_binary_map_type_cast() {
        // i32 arrays cast to f64 for computation
        let arr1 = make_i32_array(&[1, 2, 3]);
        let arr2 = make_i32_array(&[10, 20, 30]);

        let result = binary_map::<f64, _>(arr1, arr2, |a, b| a + b).unwrap();

        match result {
            Array::NumericArray(NumericArray::Float64(a)) => {
                assert_eq!(a.data.as_slice(), &[11.0, 22.0, 33.0]);
            }
            _ => panic!("Expected Float64 array"),
        }
    }

    #[cfg(feature = "scalar_type")]
    #[test]
    fn test_binary_map_with_scalar() {
        use crate::Scalar;

        let arr = make_f64_array(&[1.0, 2.0, 3.0]);
        let scalar = Scalar::Float64(10.0);

        let result = binary_map::<f64, _>(arr, scalar, |a, b| a + b).unwrap();

        match result {
            Array::NumericArray(NumericArray::Float64(a)) => {
                assert_eq!(a.data.as_slice(), &[11.0, 12.0, 13.0]);
            }
            _ => panic!("Expected Float64 array"),
        }
    }
}
