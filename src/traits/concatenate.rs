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

//! # Concatenate Trait Module
//!
//! Provides uniform concatenation across Minarrow types.
//!
//! ## Overview
//! The `Concatenate` trait enables combining two instances of the same type:
//! - **Scalars**: Scalar + Scalar -> Array (with type promotion for numerics)
//! - **Arrays**: Array + Array -> Array (same type or upcast)
//! - **Tables**: Table + Table -> Table (vertical concat with field validation)
//! - **Cubes**: Cube + Cube -> Cube (with shape validation)
//! - **Matrix**: Matrix + Matrix -> Matrix (with shape validation)
//! - **Views**: Similar rules to their concrete types
//! - **Tuples**: Element-wise concatenation (recursive, inner values must be compatible)
//! - **Bitmasks**: Concatenate mask vectors
//!
//! ## Important: Consuming Semantics
//! **The `concat` method consumes both inputs for maximum efficiency.**
//! This means you cannot use the original arrays after concatenating them.
//!
//! If you need to preserve the original arrays, clone them first:
//! ```rust
//! # use minarrow::{Vec64, Concatenate};
//! let arr1 = Vec64::from(vec![1, 2, 3]);
//! let arr2 = Vec64::from(vec![4, 5, 6]);
//!
//! // If you need to keep arr1:
//! let result = arr1.clone().concat(arr2).unwrap();
//! // Now arr1 is still usable, but arr2 has been consumed
//! ```
//!
//! ## Rules
//! 1. Only concatenates within the same logical type (e.g., Array -> Array)
//! 2. Numeric arrays support type promotion (e.g., i32 + i64 -> i64)
//! 3. Structured types (e.g., Table, Cube) validate shape/schema compatibility
//! 4. Tuple concatenation is element-wise and recursive
//!
//! ## Example
//! ```rust
//! # use minarrow::{Array, IntegerArray, Concatenate};
//! let arr1 = Array::from_int32(IntegerArray::from_slice(&[1, 2, 3]));
//! let arr2 = Array::from_int32(IntegerArray::from_slice(&[4, 5, 6]));
//! let result = arr1.concat(arr2).unwrap();  // Both arr1 and arr2 are consumed
//! // result: Array([1, 2, 3, 4, 5, 6])
//! ```

use crate::enums::error::MinarrowError;
use ::vec64::Vec64;

/// Concatenate trait for combining two instances of the same type.
///
/// # Consuming Semantics
/// **This trait consumes both `self` and `other` for maximum efficiency.**
/// The first array's buffer is reused and the second array's data is appended.
/// If you need to preserve the original arrays, clone them before calling `concat`.
///
/// Implementors must ensure:
/// - Type compatibility (must be same or compatible types)
/// - Shape validation where applicable (e.g., tables, cubes, matrices)
/// - Field/schema compatibility for structured types
///
/// Returns `Result<Self, MinarrowError>` where `Self` is the concatenated result.
pub trait Concatenate {
    /// Concatenates `self` with `other`, **consuming both** and returning a new instance.
    ///
    /// # Ownership
    /// Both `self` and `other` are moved and cannot be used afterward.
    /// To preserve an array, clone it first: `arr1.clone().concat(arr2)`.
    ///
    /// # Errors
    /// - `TypeError`: Incompatible types that cannot be concatenated
    /// - `ShapeError`: Shape mismatch (for tables, cubes, matrices)
    /// - `IncompatibleTypeError`: Schema/field mismatch (for structured types)
    ///
    /// # Example
    /// ```rust
    /// # use minarrow::{Vec64, Concatenate};
    /// let v1 = Vec64::from(vec![1, 2, 3]);
    /// let v2 = Vec64::from(vec![4, 5, 6]);
    /// let result = v1.concat(v2).unwrap();  // v1 and v2 are now consumed
    /// assert_eq!(result.as_slice(), &[1, 2, 3, 4, 5, 6]);
    /// ```
    fn concat(self, other: Self) -> Result<Self, MinarrowError>
    where
        Self: Sized;
}

impl<T> Concatenate for Vec64<T> {
    fn concat(
        mut self,
        other: Self,
    ) -> core::result::Result<Self, crate::enums::error::MinarrowError> {
        // Consume other and extend self with its elements
        self.extend(other.into_iter());
        Ok(self)
    }
}

#[cfg(test)]
mod concatenate_tests {
    use super::*;
    use crate::vec64;

    #[test]
    fn test_vec64_concatenate() {
        let v1 = vec64![1, 2, 3];
        let v2 = vec64![4, 5, 6];
        let result = v1.concat(v2).unwrap();
        assert_eq!(result.as_slice(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_vec64_concatenate_empty() {
        let v1: Vec64<i32> = vec64![];
        let v2 = vec64![1, 2];
        let result = v1.concat(v2).unwrap();
        assert_eq!(result.as_slice(), &[1, 2]);
    }

    #[test]
    fn test_vec64_concatenate_both_empty() {
        let v1: Vec64<i32> = vec64![];
        let v2: Vec64<i32> = vec64![];
        let result = v1.concat(v2).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_vec64_concatenate_preserves_first() {
        let v1 = vec64![1, 2, 3];
        let v2 = vec64![4, 5, 6];
        // Clone v1 to preserve it
        let result = v1.clone().concat(v2).unwrap();
        assert_eq!(result.as_slice(), &[1, 2, 3, 4, 5, 6]);
        // v1 is still usable
        assert_eq!(v1.as_slice(), &[1, 2, 3]);
    }
}
