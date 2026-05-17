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

//! # **TextArray Module** - *High-Level Text Array Type for Unified Signature Dispatch*
//!
//! TextArray unifies all string and categorical arrays into
//! a single enum for standardised text operations.
//!   
//! ## Features
//! - direct variant access
//! - zero-cost casts when the type is known
//! - lossless conversions between string and categorical types.  
//! - simplifies function signatures by accepting `impl Into<TextArray>`
//! - centralises dispatch
//! - preserves SIMD-aligned buffers across all text variants.

use std::fmt::{Display, Formatter};
use std::sync::Arc;

use crate::enums::error::MinarrowError;
use crate::enums::shape_dim::ShapeDim;
use crate::traits::{concatenate::Concatenate, shape::Shape};
use crate::{Bitmask, CategoricalArray, MaskedArray, StringArray};

/// # TextArray
///
/// Unified Text array container
///
/// ## Purpose
/// Exists to unify string and categorical operations, simplify API's and streamline user ergonomics.
///
/// ## Usage:
/// - It is accessible from `Array` using `.str()`,
/// and provides typed variant access via for e.g.,
/// `.str64()`, so one can drill down to the required
/// granularity via `myarr.str().str64()`
/// - This streamlines function implementations,
/// and, despite the additional `enum` layer,
/// matching lanes in many real-world scenarios.
/// This is because one can for e.g., unify a
/// function signature with `impl Into<TextArray>`,
/// and all of the subtypes, plus `Array` and `TextArray`,
/// all qualify.
/// - Additionally, you can then use one `Text` implementation
/// on the enum dispatch arm for all `Text` variants, or,
/// in many cases, for the entire text arm when they are the same.
/// This is mostly useful for the `NumericArray` enum where unifying
/// floats and integers is a very common pattern, however there may be
/// are cases where it's useful for `Categorical` and `String` data too.
/// For e.g., you can do a match on `TextArray` then just handle those
/// two cases explicitly, or for e.g., accept `impl Into<TextArray>`,
/// then simply case it to `String` for some string ops when feeling
/// barbarian vibes.
///
/// ### Typecasting behaviour
/// - If the enum already holds the given type *(which should be known at compile-time)*,
/// then using accessors like `.str32()` is zero-cost, as it transfers ownership.
/// - If you want to keep the original, of course use `.clone()` beforehand.
/// - If you use an accessor to a different base type, e.g., `.cat32()` when it's a
/// `.str32()` already in the enum, it will convert it. Therefore, be mindful
/// of performance when this occurs.
#[repr(C, align(64))]
#[derive(PartialEq, Clone, Debug, Default)]
pub enum TextArray {
    String32(Arc<StringArray<u32>>),
    #[cfg(feature = "large_string")]
    String64(Arc<StringArray<u64>>),
    #[cfg(feature = "default_categorical_8")]
    Categorical8(Arc<CategoricalArray<u8>>),
    #[cfg(feature = "extended_categorical")]
    Categorical16(Arc<CategoricalArray<u16>>),
    #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
    Categorical32(Arc<CategoricalArray<u32>>),
    #[cfg(feature = "extended_categorical")]
    Categorical64(Arc<CategoricalArray<u64>>),
    #[default]
    Null, // Default Marker for mem::take
}

impl TextArray {
    /// Returns the logical length of the text array.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            TextArray::String32(arr) => arr.len(),
            #[cfg(feature = "large_string")]
            TextArray::String64(arr) => arr.len(),
            #[cfg(feature = "default_categorical_8")]
            TextArray::Categorical8(arr) => arr.len(),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical16(arr) => arr.len(),
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            TextArray::Categorical32(arr) => arr.len(),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical64(arr) => arr.len(),
            TextArray::Null => 0,
        }
    }

    /// Returns the underlying null mask, if any.
    #[inline]
    pub fn null_mask(&self) -> Option<&Bitmask> {
        match self {
            TextArray::String32(arr) => arr.null_mask.as_ref(),
            #[cfg(feature = "large_string")]
            TextArray::String64(arr) => arr.null_mask.as_ref(),
            #[cfg(feature = "default_categorical_8")]
            TextArray::Categorical8(arr) => arr.null_mask.as_ref(),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical16(arr) => arr.null_mask.as_ref(),
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            TextArray::Categorical32(arr) => arr.null_mask.as_ref(),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical64(arr) => arr.null_mask.as_ref(),
            TextArray::Null => None,
        }
    }

    /// Returns true when the variant holds at least one null.
    ///
    /// Delegates to each inner array's `MaskedArray::has_nulls`; `Null` is
    /// treated as empty (no elements means no nulls).
    #[inline]
    pub fn has_nulls(&self) -> bool {
        match self {
            TextArray::String32(arr) => arr.has_nulls(),
            #[cfg(feature = "large_string")]
            TextArray::String64(arr) => arr.has_nulls(),
            #[cfg(feature = "default_categorical_8")]
            TextArray::Categorical8(arr) => arr.has_nulls(),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical16(arr) => arr.has_nulls(),
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            TextArray::Categorical32(arr) => arr.has_nulls(),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical64(arr) => arr.has_nulls(),
            TextArray::Null => false,
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
            (TextArray::String32(a), TextArray::String32(b)) => Arc::make_mut(a).append_array(b),
            #[cfg(feature = "large_string")]
            (TextArray::String64(a), TextArray::String64(b)) => Arc::make_mut(a).append_array(b),
            #[cfg(feature = "default_categorical_8")]
            (TextArray::Categorical8(a), TextArray::Categorical8(b)) => {
                Arc::make_mut(a).append_array(b)
            }
            #[cfg(feature = "extended_categorical")]
            (TextArray::Categorical16(a), TextArray::Categorical16(b)) => {
                Arc::make_mut(a).append_array(b)
            }
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            (TextArray::Categorical32(a), TextArray::Categorical32(b)) => {
                Arc::make_mut(a).append_array(b)
            }
            #[cfg(feature = "extended_categorical")]
            (TextArray::Categorical64(a), TextArray::Categorical64(b)) => {
                Arc::make_mut(a).append_array(b)
            }
            (TextArray::Null, TextArray::Null) => (),
            (lhs, rhs) => panic!("Cannot append {:?} into {:?}", rhs, lhs),
        }
    }

    pub fn append_range(&mut self, other: &Self, offset: usize, len: usize) -> Result<(), MinarrowError> {
        match (self, other) {
            (TextArray::String32(a), TextArray::String32(b)) => Arc::make_mut(a).append_range(b, offset, len),
            #[cfg(feature = "large_string")]
            (TextArray::String64(a), TextArray::String64(b)) => Arc::make_mut(a).append_range(b, offset, len),
            #[cfg(feature = "default_categorical_8")]
            (TextArray::Categorical8(a), TextArray::Categorical8(b)) => Arc::make_mut(a).append_range(b, offset, len),
            #[cfg(feature = "extended_categorical")]
            (TextArray::Categorical16(a), TextArray::Categorical16(b)) => Arc::make_mut(a).append_range(b, offset, len),
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            (TextArray::Categorical32(a), TextArray::Categorical32(b)) => Arc::make_mut(a).append_range(b, offset, len),
            #[cfg(feature = "extended_categorical")]
            (TextArray::Categorical64(a), TextArray::Categorical64(b)) => Arc::make_mut(a).append_range(b, offset, len),
            (TextArray::Null, TextArray::Null) => Ok(()),
            (lhs, rhs) => Err(MinarrowError::TypeError {
                from: "TextArray",
                to: "TextArray",
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
            (TextArray::String32(a), TextArray::String32(b)) => {
                Arc::make_mut(a).insert_rows(index, b)
            }
            #[cfg(feature = "large_string")]
            (TextArray::String64(a), TextArray::String64(b)) => {
                Arc::make_mut(a).insert_rows(index, b)
            }
            #[cfg(feature = "default_categorical_8")]
            (TextArray::Categorical8(a), TextArray::Categorical8(b)) => {
                Arc::make_mut(a).insert_rows(index, b)
            }
            #[cfg(feature = "extended_categorical")]
            (TextArray::Categorical16(a), TextArray::Categorical16(b)) => {
                Arc::make_mut(a).insert_rows(index, b)
            }
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            (TextArray::Categorical32(a), TextArray::Categorical32(b)) => {
                Arc::make_mut(a).insert_rows(index, b)
            }
            #[cfg(feature = "extended_categorical")]
            (TextArray::Categorical64(a), TextArray::Categorical64(b)) => {
                Arc::make_mut(a).insert_rows(index, b)
            }
            (TextArray::Null, TextArray::Null) => Ok(()),
            (lhs, rhs) => Err(MinarrowError::TypeError {
                from: "TextArray",
                to: "TextArray",
                message: Some(format!(
                    "Cannot insert {} into {}: incompatible types",
                    rhs, lhs
                )),
            }),
        }
    }

    /// Splits the TextArray at the specified index, consuming self and returning two arrays.
    pub fn split(self, index: usize) -> Result<(Self, Self), MinarrowError> {
        use std::sync::Arc;

        match self {
            TextArray::String32(a) => {
                let (left, right) = Arc::try_unwrap(a)
                    .unwrap_or_else(|arc| (*arc).clone())
                    .split(index)?;
                Ok((
                    TextArray::String32(Arc::new(left)),
                    TextArray::String32(Arc::new(right)),
                ))
            }
            #[cfg(feature = "large_string")]
            TextArray::String64(a) => {
                let (left, right) = Arc::try_unwrap(a)
                    .unwrap_or_else(|arc| (*arc).clone())
                    .split(index)?;
                Ok((
                    TextArray::String64(Arc::new(left)),
                    TextArray::String64(Arc::new(right)),
                ))
            }
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            TextArray::Categorical32(a) => {
                let (left, right) = Arc::try_unwrap(a)
                    .unwrap_or_else(|arc| (*arc).clone())
                    .split(index)?;
                Ok((
                    TextArray::Categorical32(Arc::new(left)),
                    TextArray::Categorical32(Arc::new(right)),
                ))
            }
            #[cfg(feature = "default_categorical_8")]
            TextArray::Categorical8(a) => {
                let (left, right) = Arc::try_unwrap(a)
                    .unwrap_or_else(|arc| (*arc).clone())
                    .split(index)?;
                Ok((
                    TextArray::Categorical8(Arc::new(left)),
                    TextArray::Categorical8(Arc::new(right)),
                ))
            }
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical16(a) => {
                let (left, right) = Arc::try_unwrap(a)
                    .unwrap_or_else(|arc| (*arc).clone())
                    .split(index)?;
                Ok((
                    TextArray::Categorical16(Arc::new(left)),
                    TextArray::Categorical16(Arc::new(right)),
                ))
            }
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical64(a) => {
                let (left, right) = Arc::try_unwrap(a)
                    .unwrap_or_else(|arc| (*arc).clone())
                    .split(index)?;
                Ok((
                    TextArray::Categorical64(Arc::new(left)),
                    TextArray::Categorical64(Arc::new(right)),
                ))
            }
            TextArray::Null => Err(MinarrowError::IndexError(
                "Cannot split Null array".to_string(),
            )),
        }
    }

    /// Returns a reference to the inner `StringArray<u32>` if the variant matches.
    /// No conversion or cloning is performed.
    pub fn str32_ref(&self) -> Result<&StringArray<u32>, MinarrowError> {
        match self {
            TextArray::String32(arr) => Ok(arr),
            TextArray::Null => Err(MinarrowError::NullError { message: None }),
            _ => Err(MinarrowError::TypeError {
                from: "TextArray",
                to: "StringArray<u32>",
                message: None,
            }),
        }
    }

    /// Returns a reference to the inner `StringArray<u64>` if the variant matches.
    /// No conversion or cloning is performed.
    #[cfg(feature = "large_string")]
    pub fn str64_ref(&self) -> Result<&StringArray<u64>, MinarrowError> {
        match self {
            TextArray::String64(arr) => Ok(arr),
            TextArray::Null => Err(MinarrowError::NullError { message: None }),
            _ => Err(MinarrowError::TypeError {
                from: "TextArray",
                to: "StringArray<u64>",
                message: None,
            }),
        }
    }

    /// Returns a reference to the inner `CategoricalArray<u32>` if the variant matches.
    /// No conversion or cloning is performed.
    #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
    pub fn cat32_ref(&self) -> Result<&CategoricalArray<u32>, MinarrowError> {
        match self {
            TextArray::Categorical32(arr) => Ok(arr),
            TextArray::Null => Err(MinarrowError::NullError { message: None }),
            _ => Err(MinarrowError::TypeError {
                from: "TextArray",
                to: "CategoricalArray<u32>",
                message: None,
            }),
        }
    }

    /// Returns a reference to the inner `CategoricalArray<u64>` if the variant matches.
    /// No conversion or cloning is performed.
    #[cfg(feature = "extended_categorical")]
    pub fn cat64_ref(&self) -> Result<&CategoricalArray<u64>, MinarrowError> {
        match self {
            TextArray::Categorical64(arr) => Ok(arr),
            TextArray::Null => Err(MinarrowError::NullError { message: None }),
            _ => Err(MinarrowError::TypeError {
                from: "TextArray",
                to: "CategoricalArray<u64>",
                message: None,
            }),
        }
    }

    /// Returns a reference to the inner `CategoricalArray<u8>` if the variant matches.
    /// No conversion or cloning is performed.
    #[cfg(feature = "default_categorical_8")]
    pub fn cat8_ref(&self) -> Result<&CategoricalArray<u8>, MinarrowError> {
        match self {
            TextArray::Categorical8(arr) => Ok(arr),
            TextArray::Null => Err(MinarrowError::NullError { message: None }),
            _ => Err(MinarrowError::TypeError {
                from: "TextArray",
                to: "CategoricalArray<u8>",
                message: None,
            }),
        }
    }

    /// Returns a reference to the inner `CategoricalArray<u16>` if the variant matches.
    /// No conversion or cloning is performed.
    #[cfg(feature = "extended_categorical")]
    pub fn cat16_ref(&self) -> Result<&CategoricalArray<u16>, MinarrowError> {
        match self {
            TextArray::Categorical16(arr) => Ok(arr),
            TextArray::Null => Err(MinarrowError::NullError { message: None }),
            _ => Err(MinarrowError::TypeError {
                from: "TextArray",
                to: "CategoricalArray<u16>",
                message: None,
            }),
        }
    }

    /// Casts to StringArray<u32>
    ///
    /// - Converts via TryFrom,
    /// - Uses *CloneOnWrite (COW)* when it's already a `String32`.
    pub fn str32(self) -> Result<StringArray<u32>, MinarrowError> {
        match self {
            TextArray::String32(arr) => match Arc::try_unwrap(arr) {
                Ok(inner) => Ok(inner),
                Err(shared) => Ok((*shared).clone()),
            },
            #[cfg(feature = "large_string")]
            TextArray::String64(arr) => Ok(StringArray::<u32>::try_from(&*arr)?),
            #[cfg(feature = "default_categorical_8")]
            TextArray::Categorical8(arr) => Ok(StringArray::<u32>::try_from(&*arr)?),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical16(arr) => Ok(StringArray::<u32>::try_from(&*arr)?),
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            TextArray::Categorical32(arr) => Ok(StringArray::<u32>::try_from(&*arr)?),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical64(arr) => Ok(StringArray::<u32>::try_from(&*arr)?),
            TextArray::Null => Err(MinarrowError::NullError { message: None }),
        }
    }

    /// Casts to StringArray<u64>
    ///
    /// - Converts via `From` or `TryFrom`, depending on the inner type
    /// - Uses *CloneOnWrite (COW)* when it's already a `String64`.
    #[cfg(feature = "large_string")]
    pub fn str64(self) -> Result<StringArray<u64>, MinarrowError> {
        match self {
            TextArray::String64(arr) => match Arc::try_unwrap(arr) {
                Ok(inner) => Ok(inner),
                Err(shared) => Ok((*shared).clone()),
            },
            TextArray::String32(arr) => Ok(StringArray::<u64>::from(&*arr)),
            #[cfg(feature = "default_categorical_8")]
            TextArray::Categorical8(arr) => Ok(StringArray::<u64>::try_from(&*arr)?),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical16(arr) => Ok(StringArray::<u64>::try_from(&*arr)?),
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            TextArray::Categorical32(arr) => Ok(StringArray::<u64>::try_from(&*arr)?),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical64(arr) => Ok(StringArray::<u64>::try_from(&*arr)?),
            TextArray::Null => Err(MinarrowError::NullError { message: None }),
        }
    }

    /// Casts to CategoricalArray<u32>
    ///
    /// - Converts via `From` or `TryFrom`, depending on the inner type
    /// - Uses *CloneOnWrite (COW)* when it's already a `Categorical32`.
    #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
    pub fn cat32(self) -> Result<CategoricalArray<u32>, MinarrowError> {
        match self {
            TextArray::Categorical32(arr) => match Arc::try_unwrap(arr) {
                Ok(inner) => Ok(inner),
                Err(shared) => Ok((*shared).clone()),
            },
            TextArray::String32(arr) => Ok(CategoricalArray::<u32>::try_from(&*arr)?),
            #[cfg(feature = "large_string")]
            TextArray::String64(arr) => Ok(CategoricalArray::<u32>::try_from(&*arr)?),
            #[cfg(feature = "default_categorical_8")]
            TextArray::Categorical8(arr) => Ok(CategoricalArray::<u32>::from(&*arr)),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical16(arr) => Ok(CategoricalArray::<u32>::from(&*arr)),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical64(arr) => Ok(CategoricalArray::<u32>::try_from(&*arr)?),
            TextArray::Null => Err(MinarrowError::NullError { message: None }),
        }
    }

    /// Casts to CategoricalArray<u64>
    ///
    /// - Converts via `From` or `TryFrom`, depending on the inner type
    /// - Uses *CloneOnWrite (COW)* when it's already a `Categorical32`.
    #[cfg(feature = "extended_categorical")]
    pub fn cat64(self) -> Result<CategoricalArray<u64>, MinarrowError> {
        match self {
            TextArray::Categorical64(arr) => match Arc::try_unwrap(arr) {
                Ok(inner) => Ok(inner),
                Err(shared) => Ok((*shared).clone()),
            },
            TextArray::String32(arr) => Ok(CategoricalArray::<u64>::try_from(&*arr)?),
            #[cfg(feature = "large_string")]
            TextArray::String64(arr) => Ok(CategoricalArray::<u64>::try_from(&*arr)?),
            #[cfg(feature = "default_categorical_8")]
            TextArray::Categorical8(arr) => Ok(CategoricalArray::<u64>::from(&*arr)),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical16(arr) => Ok(CategoricalArray::<u64>::from(&*arr)),
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            TextArray::Categorical32(arr) => Ok(CategoricalArray::<u64>::from(&*arr)),
            TextArray::Null => Err(MinarrowError::NullError { message: None }),
        }
    }

    /// Casts to CategoricalArray<u8>.
    ///
    /// - Converts via `From` or `TryFrom`, depending on the inner type
    /// - Uses *CloneOnWrite (COW)* when it's already a `Categorical8`.
    #[cfg(feature = "default_categorical_8")]
    pub fn cat8(self) -> Result<CategoricalArray<u8>, MinarrowError> {
        match self {
            TextArray::Categorical8(arr) => match Arc::try_unwrap(arr) {
                Ok(inner) => Ok(inner),
                Err(shared) => Ok((*shared).clone()),
            },
            TextArray::String32(arr) => Ok(CategoricalArray::<u8>::try_from(&*arr)?),
            #[cfg(feature = "large_string")]
            TextArray::String64(arr) => Ok(CategoricalArray::<u8>::try_from(&*arr)?),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical16(arr) => Ok(CategoricalArray::<u8>::try_from(&*arr)?),
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            TextArray::Categorical32(arr) => Ok(CategoricalArray::<u8>::try_from(&*arr)?),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical64(arr) => Ok(CategoricalArray::<u8>::try_from(&*arr)?),
            TextArray::Null => Err(MinarrowError::NullError { message: None }),
        }
    }

    /// Casts to CategoricalArray<u16>.
    ///
    /// - Converts via `From` or `TryFrom`, depending on the inner type
    /// - Uses *CloneOnWrite (COW)* when it's already a `Categorical16`.
    #[cfg(feature = "extended_categorical")]
    pub fn cat16(self) -> Result<CategoricalArray<u16>, MinarrowError> {
        match self {
            TextArray::Categorical16(arr) => match Arc::try_unwrap(arr) {
                Ok(inner) => Ok(inner),
                Err(shared) => Ok((*shared).clone()),
            },
            TextArray::String32(arr) => Ok(CategoricalArray::<u16>::try_from(&*arr)?),
            #[cfg(feature = "large_string")]
            TextArray::String64(arr) => Ok(CategoricalArray::<u16>::try_from(&*arr)?),
            #[cfg(feature = "default_categorical_8")]
            TextArray::Categorical8(arr) => Ok(CategoricalArray::<u16>::from(&*arr)),
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            TextArray::Categorical32(arr) => Ok(CategoricalArray::<u16>::try_from(&*arr)?),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical64(arr) => Ok(CategoricalArray::<u16>::try_from(&*arr)?),
            TextArray::Null => Err(MinarrowError::NullError { message: None }),
        }
    }
}

impl Display for TextArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TextArray::String32(arr) => write_text_array_with_header(f, "String32", arr.as_ref()),
            #[cfg(feature = "large_string")]
            TextArray::String64(arr) => write_text_array_with_header(f, "String64", arr.as_ref()),
            #[cfg(feature = "default_categorical_8")]
            TextArray::Categorical8(arr) => {
                write_text_array_with_header(f, "Categorical8", arr.as_ref())
            }
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical16(arr) => {
                write_text_array_with_header(f, "Categorical16", arr.as_ref())
            }
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            TextArray::Categorical32(arr) => {
                write_text_array_with_header(f, "Categorical32", arr.as_ref())
            }
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical64(arr) => {
                write_text_array_with_header(f, "Categorical64", arr.as_ref())
            }
            TextArray::Null => writeln!(f, "TextArray::Null [0 values]"),
        }
    }
}

/// Writes the header, then delegates row printing to the array's Display.
fn write_text_array_with_header(
    f: &mut Formatter<'_>,
    dtype: &str,
    arr: &(impl MaskedArray + Display + ?Sized),
) -> std::fmt::Result {
    writeln!(
        f,
        "TextArray [{dtype}] [{} values] (null count: {})",
        arr.len(),
        arr.null_count()
    )?;
    Display::fmt(arr, f)
}

impl Shape for TextArray {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank1(self.len())
    }
}

impl Concatenate for TextArray {
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        match (self, other) {
            (TextArray::String32(a), TextArray::String32(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(TextArray::String32(Arc::new(a.concat(b)?)))
            }
            #[cfg(feature = "large_string")]
            (TextArray::String64(a), TextArray::String64(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(TextArray::String64(Arc::new(a.concat(b)?)))
            }
            #[cfg(feature = "default_categorical_8")]
            (TextArray::Categorical8(a), TextArray::Categorical8(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(TextArray::Categorical8(Arc::new(a.concat(b)?)))
            }
            #[cfg(feature = "extended_categorical")]
            (TextArray::Categorical16(a), TextArray::Categorical16(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(TextArray::Categorical16(Arc::new(a.concat(b)?)))
            }
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            (TextArray::Categorical32(a), TextArray::Categorical32(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(TextArray::Categorical32(Arc::new(a.concat(b)?)))
            }
            #[cfg(feature = "extended_categorical")]
            (TextArray::Categorical64(a), TextArray::Categorical64(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(TextArray::Categorical64(Arc::new(a.concat(b)?)))
            }
            (TextArray::Null, TextArray::Null) => Ok(TextArray::Null),
            (lhs, rhs) => Err(MinarrowError::IncompatibleTypeError {
                from: "TextArray",
                to: "TextArray",
                message: Some(format!(
                    "Cannot concatenate mismatched TextArray variants: {:?} and {:?}",
                    text_variant_name(&lhs),
                    text_variant_name(&rhs)
                )),
            }),
        }
    }
}

/// Helper function to get the variant name for error messages
fn text_variant_name(arr: &TextArray) -> &'static str {
    match arr {
        TextArray::String32(_) => "String32",
        #[cfg(feature = "large_string")]
        TextArray::String64(_) => "String64",
        #[cfg(feature = "default_categorical_8")]
        TextArray::Categorical8(_) => "Categorical8",
        #[cfg(feature = "extended_categorical")]
        TextArray::Categorical16(_) => "Categorical16",
        #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
        TextArray::Categorical32(_) => "Categorical32",
        #[cfg(feature = "extended_categorical")]
        TextArray::Categorical64(_) => "Categorical64",
        TextArray::Null => "Null",
    }
}
