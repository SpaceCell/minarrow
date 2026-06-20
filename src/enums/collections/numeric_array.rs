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

//! # **NumericArray Module** - *High-Level Numerical Array Type for Unified Signature Dispatch*
//!
//! NumericArray unifies all integer and floating-point arrays
//! into a single enum for standardised numeric operations.
//!   
//! ## Features
//! - direct variant access
//! - zero-cost casts when the type is known
//! - lossless conversions between integer and float types.
//! - simplifies function signatures by accepting `impl Into<NumericArray>`
//! - centralises dispatch
//! - preserves SIMD-aligned buffers across all numeric variants.

use std::{
    fmt::{Display, Formatter},
    sync::Arc,
};

use crate::{Bitmask, FloatArray, IntegerArray, MaskedArray, Vec64};
use crate::{BooleanArray, StringArray};
use crate::{
    enums::{error::MinarrowError, shape_dim::ShapeDim},
    traits::{concatenate::Concatenate, shape::Shape},
};

/// # NumericArray
///
/// Unified numerical array container
///
/// ## Purpose
/// Exists to unify numerical operations,
/// simplify API's and streamline user ergonomics.
///
/// ## Usage:
/// - It is accessible from `Array` using `.num()`,
/// and provides typed variant access via for e.g.,
/// `.i64()`, so one can drill down to the required
/// granularity via `myarr.num().i64()`
/// - This streamlines function implementations,
/// and, despite the additional `enum` layer,
/// matching lanes in many real-world scenarios.
/// This is because one can for e.g., unify a
/// function signature with `impl Into<NumericArray>`,
/// and all of the subtypes, plus `Array` and `NumericalArray`,
/// all qualify.
/// - Additionally, you can then use one `Integer` implementation
/// on the enum dispatch arm for all `Integer` variants, or,
/// in many cases, for the entire numeric arm when they are the same.
///
/// ### Typecasting behaviour
/// - If the enum already holds the given type *(which should be known at compile-time)*,
/// then using accessors like `.i32()` is zero-cost, as it transfers ownership.
/// - If you want to keep the original, of course use `.clone()` beforehand.
/// - If you use an accessor to a different base type, e.g., `.f32()` when it's a
/// `.int32()` already in the enum, it will convert it. Therefore, be mindful
/// of performance when this occurs.
///
/// ## Also see:
/// - Under [crate::traits::type_unions] , we additionally
/// include minimal `Integer`, `Float`, `Numeric` and `Primitive` traits that
/// for which the base Rust primitive types already qualify.
/// These are loose wrappers over the `num-traits` crate to help improve
/// type ergonomics when traits are required, but without requiring
/// any downcasting.
#[repr(C, align(64))]
#[derive(PartialEq, Clone, Debug, Default)]
pub enum NumericArray {
    #[cfg(feature = "extended_numeric_types")]
    Int8(Arc<IntegerArray<i8>>),
    #[cfg(feature = "extended_numeric_types")]
    Int16(Arc<IntegerArray<i16>>),
    Int32(Arc<IntegerArray<i32>>),
    Int64(Arc<IntegerArray<i64>>),
    #[cfg(feature = "extended_numeric_types")]
    UInt8(Arc<IntegerArray<u8>>),
    #[cfg(feature = "extended_numeric_types")]
    UInt16(Arc<IntegerArray<u16>>),
    UInt32(Arc<IntegerArray<u32>>),
    UInt64(Arc<IntegerArray<u64>>),
    Float32(Arc<FloatArray<f32>>),
    Float64(Arc<FloatArray<f64>>),
    #[default]
    Null, // Default Marker for mem::take
}

impl NumericArray {
    /// Returns the logical length of the numeric array.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(arr) => arr.len(),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(arr) => arr.len(),
            NumericArray::Int32(arr) => arr.len(),
            NumericArray::Int64(arr) => arr.len(),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(arr) => arr.len(),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(arr) => arr.len(),
            NumericArray::UInt32(arr) => arr.len(),
            NumericArray::UInt64(arr) => arr.len(),
            NumericArray::Float32(arr) => arr.len(),
            NumericArray::Float64(arr) => arr.len(),
            NumericArray::Null => 0,
        }
    }

    /// Removes the rows in `[start, end)`, shifting later rows left.
    /// A shared inner array is cloned first i.e. copy-on-write.
    ///
    /// # Panics
    /// Panics if `start > end` or `end > len`.
    pub fn delete_range(&mut self, start: usize, end: usize) {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(arr) => arr.delete_range(start, end),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(arr) => arr.delete_range(start, end),
            NumericArray::Int32(arr) => arr.delete_range(start, end),
            NumericArray::Int64(arr) => arr.delete_range(start, end),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(arr) => arr.delete_range(start, end),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(arr) => arr.delete_range(start, end),
            NumericArray::UInt32(arr) => arr.delete_range(start, end),
            NumericArray::UInt64(arr) => arr.delete_range(start, end),
            NumericArray::Float32(arr) => arr.delete_range(start, end),
            NumericArray::Float64(arr) => arr.delete_range(start, end),
            NumericArray::Null => {
                assert!(
                    start == 0 && end == 0,
                    "NumericArray::Null: delete_range out of bounds"
                );
            }
        }
    }

    /// Returns the underlying null mask, if any.
    #[inline]
    pub fn null_mask(&self) -> Option<&Bitmask> {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(arr) => arr.null_mask.as_ref(),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(arr) => arr.null_mask.as_ref(),
            NumericArray::Int32(arr) => arr.null_mask.as_ref(),
            NumericArray::Int64(arr) => arr.null_mask.as_ref(),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(arr) => arr.null_mask.as_ref(),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(arr) => arr.null_mask.as_ref(),
            NumericArray::UInt32(arr) => arr.null_mask.as_ref(),
            NumericArray::UInt64(arr) => arr.null_mask.as_ref(),
            NumericArray::Float32(arr) => arr.null_mask.as_ref(),
            NumericArray::Float64(arr) => arr.null_mask.as_ref(),
            NumericArray::Null => None,
        }
    }

    /// Returns true when the variant holds at least one null.
    ///
    /// Delegates to each inner array's `MaskedArray::has_nulls`; `Null` is
    /// treated as empty (no elements means no nulls).
    #[inline]
    pub fn has_nulls(&self) -> bool {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(arr) => arr.has_nulls(),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(arr) => arr.has_nulls(),
            NumericArray::Int32(arr) => arr.has_nulls(),
            NumericArray::Int64(arr) => arr.has_nulls(),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(arr) => arr.has_nulls(),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(arr) => arr.has_nulls(),
            NumericArray::UInt32(arr) => arr.has_nulls(),
            NumericArray::UInt64(arr) => arr.has_nulls(),
            NumericArray::Float32(arr) => arr.has_nulls(),
            NumericArray::Float64(arr) => arr.has_nulls(),
            NumericArray::Null => false,
        }
    }

    /// Appends all values (and null mask if present) from `other` into `self`.
    ///
    /// Panics if the two arrays are of different variants or incompatible types.
    ///
    /// This function uses copy-on-write semantics for arrays wrapped in `Arc`.
    /// If `self` is the only owner of its data, appends are performed in place without copying.
    /// If the array data is shared (`Arc` reference count > 1), the data is first cloned
    /// (so the mutation does not affect other owners), and the append is then performed on the unique copy.
    ///
    /// This ensures that calling `append_array` never mutates data referenced elsewhere,
    /// but also avoids unnecessary cloning when the data is uniquely owned.
    pub fn append_array(&mut self, other: &Self) {
        match (self, other) {
            #[cfg(feature = "extended_numeric_types")]
            (NumericArray::Int8(a), NumericArray::Int8(b)) => Arc::make_mut(a).append_array(b),
            #[cfg(feature = "extended_numeric_types")]
            (NumericArray::Int16(a), NumericArray::Int16(b)) => Arc::make_mut(a).append_array(b),
            (NumericArray::Int32(a), NumericArray::Int32(b)) => Arc::make_mut(a).append_array(b),
            (NumericArray::Int64(a), NumericArray::Int64(b)) => Arc::make_mut(a).append_array(b),

            #[cfg(feature = "extended_numeric_types")]
            (NumericArray::UInt8(a), NumericArray::UInt8(b)) => Arc::make_mut(a).append_array(b),
            #[cfg(feature = "extended_numeric_types")]
            (NumericArray::UInt16(a), NumericArray::UInt16(b)) => Arc::make_mut(a).append_array(b),
            (NumericArray::UInt32(a), NumericArray::UInt32(b)) => Arc::make_mut(a).append_array(b),
            (NumericArray::UInt64(a), NumericArray::UInt64(b)) => Arc::make_mut(a).append_array(b),

            (NumericArray::Float32(a), NumericArray::Float32(b)) => {
                Arc::make_mut(a).append_array(b)
            }
            (NumericArray::Float64(a), NumericArray::Float64(b)) => {
                Arc::make_mut(a).append_array(b)
            }

            (NumericArray::Null, NumericArray::Null) => (),
            (lhs, rhs) => panic!("Cannot append {:?} into {:?}", rhs, lhs),
        }
    }

    pub fn append_range(&mut self, other: &Self, offset: usize, len: usize) -> Result<(), MinarrowError> {
        match (self, other) {
            #[cfg(feature = "extended_numeric_types")]
            (NumericArray::Int8(a), NumericArray::Int8(b)) => Arc::make_mut(a).append_range(b, offset, len),
            #[cfg(feature = "extended_numeric_types")]
            (NumericArray::Int16(a), NumericArray::Int16(b)) => Arc::make_mut(a).append_range(b, offset, len),
            (NumericArray::Int32(a), NumericArray::Int32(b)) => Arc::make_mut(a).append_range(b, offset, len),
            (NumericArray::Int64(a), NumericArray::Int64(b)) => Arc::make_mut(a).append_range(b, offset, len),
            #[cfg(feature = "extended_numeric_types")]
            (NumericArray::UInt8(a), NumericArray::UInt8(b)) => Arc::make_mut(a).append_range(b, offset, len),
            #[cfg(feature = "extended_numeric_types")]
            (NumericArray::UInt16(a), NumericArray::UInt16(b)) => Arc::make_mut(a).append_range(b, offset, len),
            (NumericArray::UInt32(a), NumericArray::UInt32(b)) => Arc::make_mut(a).append_range(b, offset, len),
            (NumericArray::UInt64(a), NumericArray::UInt64(b)) => Arc::make_mut(a).append_range(b, offset, len),
            (NumericArray::Float32(a), NumericArray::Float32(b)) => Arc::make_mut(a).append_range(b, offset, len),
            (NumericArray::Float64(a), NumericArray::Float64(b)) => Arc::make_mut(a).append_range(b, offset, len),
            (NumericArray::Null, NumericArray::Null) => Ok(()),
            (lhs, rhs) => Err(MinarrowError::TypeError {
                from: "NumericArray",
                to: "NumericArray",
                message: Some(format!("Cannot append_range {:?} into {:?}", rhs, lhs)),
            }),
        }
    }

    /// Inserts all values (and null mask if present) from `other` into `self` at the specified index.
    ///
    /// This is an **O(n)** operation.
    ///
    /// Returns an error if the two arrays are of different variants or incompatible types,
    /// or if the index is out of bounds.
    ///
    /// This function uses copy-on-write semantics for arrays wrapped in `Arc`.
    pub fn insert_rows(&mut self, index: usize, other: &Self) -> Result<(), MinarrowError> {
        match (self, other) {
            #[cfg(feature = "extended_numeric_types")]
            (NumericArray::Int8(a), NumericArray::Int8(b)) => {
                Arc::make_mut(a).insert_rows(index, b)
            }
            #[cfg(feature = "extended_numeric_types")]
            (NumericArray::Int16(a), NumericArray::Int16(b)) => {
                Arc::make_mut(a).insert_rows(index, b)
            }
            (NumericArray::Int32(a), NumericArray::Int32(b)) => {
                Arc::make_mut(a).insert_rows(index, b)
            }
            (NumericArray::Int64(a), NumericArray::Int64(b)) => {
                Arc::make_mut(a).insert_rows(index, b)
            }

            #[cfg(feature = "extended_numeric_types")]
            (NumericArray::UInt8(a), NumericArray::UInt8(b)) => {
                Arc::make_mut(a).insert_rows(index, b)
            }
            #[cfg(feature = "extended_numeric_types")]
            (NumericArray::UInt16(a), NumericArray::UInt16(b)) => {
                Arc::make_mut(a).insert_rows(index, b)
            }
            (NumericArray::UInt32(a), NumericArray::UInt32(b)) => {
                Arc::make_mut(a).insert_rows(index, b)
            }
            (NumericArray::UInt64(a), NumericArray::UInt64(b)) => {
                Arc::make_mut(a).insert_rows(index, b)
            }

            (NumericArray::Float32(a), NumericArray::Float32(b)) => {
                Arc::make_mut(a).insert_rows(index, b)
            }
            (NumericArray::Float64(a), NumericArray::Float64(b)) => {
                Arc::make_mut(a).insert_rows(index, b)
            }

            (NumericArray::Null, NumericArray::Null) => Ok(()),
            (lhs, rhs) => Err(MinarrowError::TypeError {
                from: "NumericArray",
                to: "NumericArray",
                message: Some(format!(
                    "Cannot insert {} into {}: incompatible types",
                    rhs, lhs
                )),
            }),
        }
    }

    /// Splits the NumericArray at the specified index, consuming self and returning two arrays.
    pub fn split(self, index: usize) -> Result<(Self, Self), MinarrowError> {
        use std::sync::Arc;

        match self {
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(a) => {
                let (left, right) = Arc::try_unwrap(a)
                    .unwrap_or_else(|arc| (*arc).clone())
                    .split(index)?;
                Ok((
                    NumericArray::Int8(Arc::new(left)),
                    NumericArray::Int8(Arc::new(right)),
                ))
            }
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(a) => {
                let (left, right) = Arc::try_unwrap(a)
                    .unwrap_or_else(|arc| (*arc).clone())
                    .split(index)?;
                Ok((
                    NumericArray::UInt8(Arc::new(left)),
                    NumericArray::UInt8(Arc::new(right)),
                ))
            }
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(a) => {
                let (left, right) = Arc::try_unwrap(a)
                    .unwrap_or_else(|arc| (*arc).clone())
                    .split(index)?;
                Ok((
                    NumericArray::Int16(Arc::new(left)),
                    NumericArray::Int16(Arc::new(right)),
                ))
            }
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(a) => {
                let (left, right) = Arc::try_unwrap(a)
                    .unwrap_or_else(|arc| (*arc).clone())
                    .split(index)?;
                Ok((
                    NumericArray::UInt16(Arc::new(left)),
                    NumericArray::UInt16(Arc::new(right)),
                ))
            }
            NumericArray::Int32(a) => {
                let (left, right) = Arc::try_unwrap(a)
                    .unwrap_or_else(|arc| (*arc).clone())
                    .split(index)?;
                Ok((
                    NumericArray::Int32(Arc::new(left)),
                    NumericArray::Int32(Arc::new(right)),
                ))
            }
            NumericArray::Int64(a) => {
                let (left, right) = Arc::try_unwrap(a)
                    .unwrap_or_else(|arc| (*arc).clone())
                    .split(index)?;
                Ok((
                    NumericArray::Int64(Arc::new(left)),
                    NumericArray::Int64(Arc::new(right)),
                ))
            }
            NumericArray::UInt32(a) => {
                let (left, right) = Arc::try_unwrap(a)
                    .unwrap_or_else(|arc| (*arc).clone())
                    .split(index)?;
                Ok((
                    NumericArray::UInt32(Arc::new(left)),
                    NumericArray::UInt32(Arc::new(right)),
                ))
            }
            NumericArray::UInt64(a) => {
                let (left, right) = Arc::try_unwrap(a)
                    .unwrap_or_else(|arc| (*arc).clone())
                    .split(index)?;
                Ok((
                    NumericArray::UInt64(Arc::new(left)),
                    NumericArray::UInt64(Arc::new(right)),
                ))
            }
            NumericArray::Float32(a) => {
                let (left, right) = Arc::try_unwrap(a)
                    .unwrap_or_else(|arc| (*arc).clone())
                    .split(index)?;
                Ok((
                    NumericArray::Float32(Arc::new(left)),
                    NumericArray::Float32(Arc::new(right)),
                ))
            }
            NumericArray::Float64(a) => {
                let (left, right) = Arc::try_unwrap(a)
                    .unwrap_or_else(|arc| (*arc).clone())
                    .split(index)?;
                Ok((
                    NumericArray::Float64(Arc::new(left)),
                    NumericArray::Float64(Arc::new(right)),
                ))
            }
            NumericArray::Null => Err(MinarrowError::IndexError(
                "Cannot split Null array".to_string(),
            )),
        }
    }

    /// Returns the inner array as `Arc<IntegerArray<i32>>`, converting when the variant differs.
    ///
    /// - The matching variant returns as a shared handle without copying data.
    /// - Panics on failure. Consider the try variant for a safe alternative.
    #[inline]
    pub fn i32(&self) -> Arc<IntegerArray<i32>> {
        self.try_i32().unwrap()
    }

    /// Convert to IntegerArray<i32> using From/TryFrom as appropriate per conversion.
    ///
    /// The matching variant returns as a shared handle without copying data.
    pub fn try_i32(&self) -> Result<Arc<IntegerArray<i32>>, MinarrowError> {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(a) => Ok(Arc::new(IntegerArray::<i32>::from(&**a))),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(a) => Ok(Arc::new(IntegerArray::<i32>::from(&**a))),
            NumericArray::Int32(a) => Ok(a.clone()),
            NumericArray::Int64(a) => Ok(Arc::new(IntegerArray::<i32>::try_from(&**a)?)),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(a) => Ok(Arc::new(IntegerArray::<i32>::from(&**a))),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(a) => Ok(Arc::new(IntegerArray::<i32>::from(&**a))),
            NumericArray::UInt32(a) => Ok(Arc::new(IntegerArray::<i32>::try_from(&**a)?)),
            NumericArray::UInt64(a) => Ok(Arc::new(IntegerArray::<i32>::try_from(&**a)?)),
            NumericArray::Float32(a) => Ok(Arc::new(IntegerArray::<i32>::try_from(&**a)?)),
            NumericArray::Float64(a) => Ok(Arc::new(IntegerArray::<i32>::try_from(&**a)?)),
            NumericArray::Null => Err(MinarrowError::NullError { message: None }),
        }
    }

    /// Returns the inner array as `Arc<IntegerArray<i64>>`, converting when the variant differs.
    ///
    /// - The matching variant returns as a shared handle without copying data.
    /// - Panics on failure. Consider the try variant for a safe alternative.
    #[inline]
    pub fn i64(&self) -> Arc<IntegerArray<i64>> {
        self.try_i64().unwrap()
    }

    /// Convert to IntegerArray<i64> using From/TryFrom as appropriate per conversion.
    ///
    /// The matching variant returns as a shared handle without copying data.
    pub fn try_i64(&self) -> Result<Arc<IntegerArray<i64>>, MinarrowError> {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(a) => Ok(Arc::new(IntegerArray::<i64>::from(&**a))),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(a) => Ok(Arc::new(IntegerArray::<i64>::from(&**a))),
            NumericArray::Int32(a) => Ok(Arc::new(IntegerArray::<i64>::from(&**a))),
            NumericArray::Int64(a) => Ok(a.clone()),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(a) => Ok(Arc::new(IntegerArray::<i64>::from(&**a))),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(a) => Ok(Arc::new(IntegerArray::<i64>::from(&**a))),
            NumericArray::UInt32(a) => Ok(Arc::new(IntegerArray::<i64>::from(&**a))),
            NumericArray::UInt64(a) => Ok(Arc::new(IntegerArray::<i64>::try_from(&**a)?)),
            NumericArray::Float32(a) => Ok(Arc::new(IntegerArray::<i64>::try_from(&**a)?)),
            NumericArray::Float64(a) => Ok(Arc::new(IntegerArray::<i64>::try_from(&**a)?)),
            NumericArray::Null => Err(MinarrowError::NullError { message: None }),
        }
    }

    /// Returns the inner array as `Arc<IntegerArray<u32>>`, converting when the variant differs.
    ///
    /// - The matching variant returns as a shared handle without copying data.
    /// - Panics on failure. Consider the try variant for a safe alternative.
    #[inline]
    pub fn u32(&self) -> Arc<IntegerArray<u32>> {
        self.try_u32().unwrap()
    }

    /// Convert to IntegerArray<u32> using From/TryFrom as appropriate per conversion.
    ///
    /// The matching variant returns as a shared handle without copying data.
    pub fn try_u32(&self) -> Result<Arc<IntegerArray<u32>>, MinarrowError> {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(a) => Ok(Arc::new(IntegerArray::<u32>::from(&**a))),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(a) => Ok(Arc::new(IntegerArray::<u32>::from(&**a))),
            NumericArray::Int32(a) => Ok(Arc::new(IntegerArray::<u32>::try_from(&**a)?)),
            NumericArray::Int64(a) => Ok(Arc::new(IntegerArray::<u32>::try_from(&**a)?)),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(a) => Ok(Arc::new(IntegerArray::<u32>::from(&**a))),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(a) => Ok(Arc::new(IntegerArray::<u32>::from(&**a))),
            NumericArray::UInt32(a) => Ok(a.clone()),
            NumericArray::UInt64(a) => Ok(Arc::new(IntegerArray::<u32>::try_from(&**a)?)),
            NumericArray::Float32(a) => Ok(Arc::new(IntegerArray::<u32>::try_from(&**a)?)),
            NumericArray::Float64(a) => Ok(Arc::new(IntegerArray::<u32>::try_from(&**a)?)),
            NumericArray::Null => Err(MinarrowError::NullError { message: None }),
        }
    }

    /// Returns the inner array as `Arc<IntegerArray<u64>>`, converting when the variant differs.
    ///
    /// - The matching variant returns as a shared handle without copying data.
    /// - Panics on failure. Consider the try variant for a safe alternative.
    #[inline]
    pub fn u64(&self) -> Arc<IntegerArray<u64>> {
        self.try_u64().unwrap()
    }

    /// Convert to IntegerArray<u64> using From/TryFrom as appropriate per conversion.
    ///
    /// The matching variant returns as a shared handle without copying data.
    pub fn try_u64(&self) -> Result<Arc<IntegerArray<u64>>, MinarrowError> {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(a) => Ok(Arc::new(IntegerArray::<u64>::from(&**a))),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(a) => Ok(Arc::new(IntegerArray::<u64>::from(&**a))),
            NumericArray::Int32(a) => Ok(Arc::new(IntegerArray::<u64>::from(&**a))),
            NumericArray::Int64(a) => Ok(Arc::new(IntegerArray::<u64>::try_from(&**a)?)),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(a) => Ok(Arc::new(IntegerArray::<u64>::from(&**a))),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(a) => Ok(Arc::new(IntegerArray::<u64>::from(&**a))),
            NumericArray::UInt32(a) => Ok(Arc::new(IntegerArray::<u64>::from(&**a))),
            NumericArray::UInt64(a) => Ok(a.clone()),
            NumericArray::Float32(a) => Ok(Arc::new(IntegerArray::<u64>::try_from(&**a)?)),
            NumericArray::Float64(a) => Ok(Arc::new(IntegerArray::<u64>::try_from(&**a)?)),
            NumericArray::Null => Err(MinarrowError::NullError { message: None }),
        }
    }

    /// Returns the inner array as `Arc<FloatArray<f32>>`, converting when the variant differs.
    ///
    /// - The matching variant returns as a shared handle without copying data.
    /// - Panics on failure. Consider the try variant for a safe alternative.
    #[inline]
    pub fn f32(&self) -> Arc<FloatArray<f32>> {
        self.try_f32().unwrap()
    }

    /// Convert to FloatArray<f32> using From.
    ///
    /// The matching variant returns as a shared handle without copying data.
    pub fn try_f32(&self) -> Result<Arc<FloatArray<f32>>, MinarrowError> {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(a) => Ok(Arc::new(FloatArray::<f32>::from(&**a))),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(a) => Ok(Arc::new(FloatArray::<f32>::from(&**a))),
            NumericArray::Int32(a) => Ok(Arc::new(FloatArray::<f32>::from(&**a))),
            NumericArray::Int64(a) => Ok(Arc::new(FloatArray::<f32>::from(&**a))),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(a) => Ok(Arc::new(FloatArray::<f32>::from(&**a))),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(a) => Ok(Arc::new(FloatArray::<f32>::from(&**a))),
            NumericArray::UInt32(a) => Ok(Arc::new(FloatArray::<f32>::from(&**a))),
            NumericArray::UInt64(a) => Ok(Arc::new(FloatArray::<f32>::from(&**a))),
            NumericArray::Float32(a) => Ok(a.clone()),
            NumericArray::Float64(a) => Ok(Arc::new(FloatArray::<f32>::from(&**a))),
            NumericArray::Null => Err(MinarrowError::NullError { message: None }),
        }
    }

    /// Cast this NumericArray to Float64, staying wrapped as NumericArray.
    ///
    /// If already Float64, returns self unchanged. Otherwise casts element
    /// data to f64, preserving the null mask. Uses `Arc::try_unwrap` so that
    /// if this is the sole owner of the backing Arc, the old data is consumed
    /// and freed rather than cloned.
    pub fn cow_into_f64(self) -> Self {
        macro_rules! cast_arc {
            ($arc:expr) => {
                match Arc::try_unwrap($arc) {
                    Ok(owned) => {
                        let data: Vec64<f64> =
                            owned.data.as_slice().iter().map(|&v| v as f64).collect();
                        NumericArray::Float64(Arc::new(FloatArray::new(data, owned.null_mask)))
                    }
                    Err(shared) => {
                        let data: Vec64<f64> =
                            shared.data.as_slice().iter().map(|&v| v as f64).collect();
                        NumericArray::Float64(Arc::new(FloatArray::new(
                            data,
                            shared.null_mask.clone(),
                        )))
                    }
                }
            };
        }

        match self {
            NumericArray::Float64(_) => self,
            NumericArray::Float32(arc) => cast_arc!(arc),
            NumericArray::Int32(arc) => cast_arc!(arc),
            NumericArray::Int64(arc) => cast_arc!(arc),
            NumericArray::UInt32(arc) => cast_arc!(arc),
            NumericArray::UInt64(arc) => cast_arc!(arc),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(arc) => cast_arc!(arc),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(arc) => cast_arc!(arc),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(arc) => cast_arc!(arc),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(arc) => cast_arc!(arc),
            NumericArray::Null => {
                NumericArray::Float64(Arc::new(FloatArray::new(Vec64::new(), None)))
            }
        }
    }

    /// Returns the inner array as `Arc<FloatArray<f64>>`, converting when the variant differs.
    ///
    /// - The matching variant returns as a shared handle without copying data.
    /// - Panics on failure. Consider the try variant for a safe alternative.
    #[inline]
    pub fn f64(&self) -> Arc<FloatArray<f64>> {
        self.try_f64().unwrap()
    }

    /// Convert to FloatArray<f64> using From.
    ///
    /// The matching variant returns as a shared handle without copying data.
    pub fn try_f64(&self) -> Result<Arc<FloatArray<f64>>, MinarrowError> {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(a) => Ok(Arc::new(FloatArray::<f64>::from(&**a))),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(a) => Ok(Arc::new(FloatArray::<f64>::from(&**a))),
            NumericArray::Int32(a) => Ok(Arc::new(FloatArray::<f64>::from(&**a))),
            NumericArray::Int64(a) => Ok(Arc::new(FloatArray::<f64>::from(&**a))),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(a) => Ok(Arc::new(FloatArray::<f64>::from(&**a))),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(a) => Ok(Arc::new(FloatArray::<f64>::from(&**a))),
            NumericArray::UInt32(a) => Ok(Arc::new(FloatArray::<f64>::from(&**a))),
            NumericArray::UInt64(a) => Ok(Arc::new(FloatArray::<f64>::from(&**a))),
            NumericArray::Float32(a) => Ok(Arc::new(FloatArray::<f64>::from(&**a))),
            NumericArray::Float64(a) => Ok(a.clone()),
            NumericArray::Null => Err(MinarrowError::NullError { message: None }),
        }
    }

    /// Returns the array as `Arc<BooleanArray<u8>>`.
    ///
    /// - All non-zero values become `true`, but the null mask is preserved.
    /// - Panics on failure. Consider the try variant for a safe alternative.
    #[inline]
    pub fn bool(&self) -> Arc<BooleanArray<u8>> {
        self.try_bool().unwrap()
    }

    /// Converts to BooleanArray<u8>.
    ///
    /// All non-zero values become `true`, but the null mask is preserved.
    pub fn try_bool(&self) -> Result<Arc<BooleanArray<u8>>, MinarrowError> {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(a) => Ok(Arc::new(BooleanArray::<u8>::from(&**a))),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(a) => Ok(Arc::new(BooleanArray::<u8>::from(&**a))),
            NumericArray::Int32(a) => Ok(Arc::new(BooleanArray::<u8>::from(&**a))),
            NumericArray::Int64(a) => Ok(Arc::new(BooleanArray::<u8>::from(&**a))),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(a) => Ok(Arc::new(BooleanArray::<u8>::from(&**a))),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(a) => Ok(Arc::new(BooleanArray::<u8>::from(&**a))),
            NumericArray::UInt32(a) => Ok(Arc::new(BooleanArray::<u8>::from(&**a))),
            NumericArray::UInt64(a) => Ok(Arc::new(BooleanArray::<u8>::from(&**a))),
            NumericArray::Float32(a) => Ok(Arc::new(BooleanArray::<u8>::from(&**a))),
            NumericArray::Float64(a) => Ok(Arc::new(BooleanArray::<u8>::from(&**a))),
            NumericArray::Null => Err(MinarrowError::NullError { message: None }),
        }
    }

    /// Returns the array as `Arc<StringArray<u32>>`, formatting each value as string.
    ///
    /// - Preserves Null mask.
    /// - Panics on failure. Consider the try variant for a safe alternative.
    #[inline]
    pub fn str(&self) -> Arc<StringArray<u32>> {
        self.try_str().unwrap()
    }

    /// Converts to StringArray<u32> by formatting each value as string.
    ///
    /// Preserves Null mask.
    pub fn try_str(&self) -> Result<Arc<StringArray<u32>>, MinarrowError> {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(a) => Ok(Arc::new(StringArray::<u32>::from(&**a))),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(a) => Ok(Arc::new(StringArray::<u32>::from(&**a))),
            NumericArray::Int32(a) => Ok(Arc::new(StringArray::<u32>::from(&**a))),
            NumericArray::Int64(a) => Ok(Arc::new(StringArray::<u32>::from(&**a))),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(a) => Ok(Arc::new(StringArray::<u32>::from(&**a))),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(a) => Ok(Arc::new(StringArray::<u32>::from(&**a))),
            NumericArray::UInt32(a) => Ok(Arc::new(StringArray::<u32>::from(&**a))),
            NumericArray::UInt64(a) => Ok(Arc::new(StringArray::<u32>::from(&**a))),
            NumericArray::Float32(a) => Ok(Arc::new(StringArray::<u32>::from(&**a))),
            NumericArray::Float64(a) => Ok(Arc::new(StringArray::<u32>::from(&**a))),
            NumericArray::Null => Err(MinarrowError::NullError { message: None }),
        }
    }
}

impl Display for NumericArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(arr) => write_numeric_array_with_header(f, "Int8", arr.as_ref()),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(arr) => write_numeric_array_with_header(f, "Int16", arr.as_ref()),
            NumericArray::Int32(arr) => write_numeric_array_with_header(f, "Int32", arr.as_ref()),
            NumericArray::Int64(arr) => write_numeric_array_with_header(f, "Int64", arr.as_ref()),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(arr) => write_numeric_array_with_header(f, "UInt8", arr.as_ref()),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(arr) => write_numeric_array_with_header(f, "UInt16", arr.as_ref()),
            NumericArray::UInt32(arr) => write_numeric_array_with_header(f, "UInt32", arr.as_ref()),
            NumericArray::UInt64(arr) => write_numeric_array_with_header(f, "UInt64", arr.as_ref()),
            NumericArray::Float32(arr) => {
                write_numeric_array_with_header(f, "Float32", arr.as_ref())
            }
            NumericArray::Float64(arr) => {
                write_numeric_array_with_header(f, "Float64", arr.as_ref())
            }
            NumericArray::Null => writeln!(f, "NullNumericArray [0 values]"),
        }
    }
}

/// Writes the standard header, then delegates to the contained array's Display.
fn write_numeric_array_with_header(
    f: &mut Formatter<'_>,
    dtype_name: &str,
    arr: &(impl MaskedArray + Display + ?Sized),
) -> std::fmt::Result {
    writeln!(
        f,
        "NumericArray [{dtype_name}] [{} values] (null count: {})",
        arr.len(),
        arr.null_count()
    )?;
    // Delegate row formatting
    Display::fmt(arr, f)
}

impl Shape for NumericArray {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank1(self.len())
    }
}

// TODO: Add cross-type casting
impl Concatenate for NumericArray {
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        match (self, other) {
            #[cfg(feature = "extended_numeric_types")]
            (NumericArray::Int8(a), NumericArray::Int8(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(NumericArray::Int8(Arc::new(a.concat(b)?)))
            }
            #[cfg(feature = "extended_numeric_types")]
            (NumericArray::Int16(a), NumericArray::Int16(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(NumericArray::Int16(Arc::new(a.concat(b)?)))
            }
            (NumericArray::Int32(a), NumericArray::Int32(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(NumericArray::Int32(Arc::new(a.concat(b)?)))
            }
            (NumericArray::Int64(a), NumericArray::Int64(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(NumericArray::Int64(Arc::new(a.concat(b)?)))
            }
            #[cfg(feature = "extended_numeric_types")]
            (NumericArray::UInt8(a), NumericArray::UInt8(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(NumericArray::UInt8(Arc::new(a.concat(b)?)))
            }
            #[cfg(feature = "extended_numeric_types")]
            (NumericArray::UInt16(a), NumericArray::UInt16(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(NumericArray::UInt16(Arc::new(a.concat(b)?)))
            }
            (NumericArray::UInt32(a), NumericArray::UInt32(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(NumericArray::UInt32(Arc::new(a.concat(b)?)))
            }
            (NumericArray::UInt64(a), NumericArray::UInt64(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(NumericArray::UInt64(Arc::new(a.concat(b)?)))
            }
            (NumericArray::Float32(a), NumericArray::Float32(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(NumericArray::Float32(Arc::new(a.concat(b)?)))
            }
            (NumericArray::Float64(a), NumericArray::Float64(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(NumericArray::Float64(Arc::new(a.concat(b)?)))
            }
            (NumericArray::Null, NumericArray::Null) => Ok(NumericArray::Null),
            (lhs, rhs) => Err(MinarrowError::IncompatibleTypeError {
                from: "NumericArray",
                to: "NumericArray",
                message: Some(format!(
                    "Cannot concatenate mismatched NumericArray variants: {:?} and {:?}",
                    variant_name(&lhs),
                    variant_name(&rhs)
                )),
            }),
        }
    }
}

/// Helper function to get the variant name for error messages
fn variant_name(arr: &NumericArray) -> &'static str {
    match arr {
        #[cfg(feature = "extended_numeric_types")]
        NumericArray::Int8(_) => "Int8",
        #[cfg(feature = "extended_numeric_types")]
        NumericArray::Int16(_) => "Int16",
        NumericArray::Int32(_) => "Int32",
        NumericArray::Int64(_) => "Int64",
        #[cfg(feature = "extended_numeric_types")]
        NumericArray::UInt8(_) => "UInt8",
        #[cfg(feature = "extended_numeric_types")]
        NumericArray::UInt16(_) => "UInt16",
        NumericArray::UInt32(_) => "UInt32",
        NumericArray::UInt64(_) => "UInt64",
        NumericArray::Float32(_) => "Float32",
        NumericArray::Float64(_) => "Float64",
        NumericArray::Null => "Null",
    }
}
