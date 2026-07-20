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

//! # **ByteSize Trait** - *Estimate Memory Footprint*
//!
//! Provides memory size estimation for all Minarrow types.
//!
//! ## Purpose
//! - Returns estimated (or exact) byte size of a type in memory
//! - Useful for memory tracking, allocation planning, and monitoring
//! - Simple calculation where possible (e.g., size_of::<T>() * n * m)
//! - Includes data buffers, null masks, and nested structures
//!
//! The trait carries two metrics. `est_bytes` reports the memory footprint
//! of the allocations, so it counts buffer capacity. `logical_bytes` reports
//! the data itself, so it counts buffer length and excludes any capacity
//! slack left behind by buffer growth. Memory budgeting uses `est_bytes`,
//! while throughput accounting and wire-size planning use `logical_bytes`.
//!
//! ## Usage
//! ```rust
//! use minarrow::{IntegerArray, ByteSize, MaskedArray};
//!
//! let arr = IntegerArray::<i64>::from_slice(&[1, 2, 3, 4, 5]);
//! let bytes = arr.est_bytes();
//! // Returns data buffer size: 5 * 8 = 40 bytes (plus small overhead)
//! let logical = arr.logical_bytes();
//! // Returns the data payload: 5 * 8 = 40 bytes
//! ```

use std::mem::size_of;

/// Trait for estimating the memory footprint of a type.
///
/// Returns the estimated number of bytes occupied by the object in memory,
/// including all owned data buffers, masks, and nested structures.
///
/// For types with directly calculatable sizes (e.g., `n * size_of::<T>()`),
/// this returns the exact value. For complex types, this provides a best estimate.
pub trait ByteSize {
    /// Returns the estimated byte size of this object in memory.
    ///
    /// This includes:
    /// - Data buffers (values, offsets, indices)
    /// - Null masks (bitmaps)
    /// - Dictionary data (for categorical types)
    /// - Nested structures (for recursive types)
    ///
    /// Does not include:
    /// - Stack size of the struct itself (only heap allocations)
    /// - Arc pointer overhead (counted once per allocation, not per reference)
    fn est_bytes(&self) -> usize;

    /// Returns the exact logical byte size of the data.
    ///
    /// The figure counts the bytes the values occupy in Arrow buffer terms.
    /// Value buffers count `len * element_width`, offsets count
    /// `(len + 1) * offset_width`, null masks count `ceil(len / 8)`
    /// and dictionaries count their summed string contents. Capacity slack
    /// left behind by buffer growth is excluded, which is the difference
    /// from [`est_bytes`](ByteSize::est_bytes). That method reports the
    /// memory footprint of the allocations, where this one reports the
    /// data itself, so callers use it for throughput denominators and
    /// wire-size planning where the figure must match the payload.
    ///
    /// Views report the window they cover rather than the backing array.
    ///
    /// ### Warning
    /// The non-Arrow numerical container types (`Matrix`, `NdArray` and their chunked
    /// and view forms, plus `XArray`) do not yet define logical byte
    /// accounting currently and panic with `unimplemented!` when called.
    fn logical_bytes(&self) -> usize;
}

// Base Buffer Type Implementations

use crate::{Bitmask, Buffer, Vec64};

/// ByteSize for Vec64<T> - 64-byte aligned vector
impl<T> ByteSize for Vec64<T> {
    #[inline]
    fn est_bytes(&self) -> usize {
        // Capacity in elements * size per element
        self.capacity() * size_of::<T>()
    }

    #[inline]
    fn logical_bytes(&self) -> usize {
        // Populated elements * size per element
        self.len() * size_of::<T>()
    }
}

/// ByteSize for Buffer<T> - unified owned/shared buffer
impl<T> ByteSize for Buffer<T> {
    #[inline]
    fn est_bytes(&self) -> usize {
        // Capacity in elements * size per element
        self.capacity() * size_of::<T>()
    }

    #[inline]
    fn logical_bytes(&self) -> usize {
        // A shared buffer reports its window length, so the figure covers
        // the elements the buffer presents rather than the backing region.
        self.len() * size_of::<T>()
    }
}

/// ByteSize for Bitmask - bit-packed bitmask
impl ByteSize for Bitmask {
    #[inline]
    fn est_bytes(&self) -> usize {
        // The capacity of the backing byte buffer, which is already
        // byte-granular
        self.bits.est_bytes()
    }

    #[inline]
    fn logical_bytes(&self) -> usize {
        // Bit-packed is the owned bit count rounded up to whole bytes
        (self.len() + 7) / 8
    }
}

// Concrete Array Type Implementations

use crate::{BooleanArray, CategoricalArray, FloatArray, IntegerArray, StringArray};

/// ByteSize for IntegerArray<T>
impl<T> ByteSize for IntegerArray<T> {
    #[inline]
    fn est_bytes(&self) -> usize {
        let data_bytes = self.data.est_bytes();
        let mask_bytes = self.null_mask.as_ref().map_or(0, |m| m.est_bytes());
        data_bytes + mask_bytes
    }

    #[inline]
    fn logical_bytes(&self) -> usize {
        self.data.logical_bytes() + self.null_mask.as_ref().map_or(0, |m| m.logical_bytes())
    }
}

/// ByteSize for FloatArray<T>
impl<T> ByteSize for FloatArray<T> {
    #[inline]
    fn est_bytes(&self) -> usize {
        let data_bytes = self.data.est_bytes();
        let mask_bytes = self.null_mask.as_ref().map_or(0, |m| m.est_bytes());
        data_bytes + mask_bytes
    }

    #[inline]
    fn logical_bytes(&self) -> usize {
        self.data.logical_bytes() + self.null_mask.as_ref().map_or(0, |m| m.logical_bytes())
    }
}

/// ByteSize for StringArray<T>
impl<T> ByteSize for StringArray<T> {
    #[inline]
    fn est_bytes(&self) -> usize {
        let data_bytes = self.data.est_bytes();
        let offsets_bytes = self.offsets.est_bytes();
        let mask_bytes = self.null_mask.as_ref().map_or(0, |m| m.est_bytes());
        data_bytes + offsets_bytes + mask_bytes
    }

    #[inline]
    fn logical_bytes(&self) -> usize {
        // The string payload plus its offsets buffer plus any null mask bytes
        self.data.logical_bytes()
            + self.offsets.logical_bytes()
            + self.null_mask.as_ref().map_or(0, |m| m.logical_bytes())
    }
}

/// ByteSize for CategoricalArray<T>
impl<T: crate::traits::type_unions::Integer> ByteSize for CategoricalArray<T> {
    #[inline]
    fn est_bytes(&self) -> usize {
        let data_bytes = self.data.est_bytes();
        // The dictionary allocates in two parts. The Vec64 stores the
        // String structs at 24 bytes each, and every String owns its
        // character buffer as a separate allocation, so both count.
        // Under shared_dict the published prefix stands in for the
        // struct count, as the sharing group owns the allocation.
        #[cfg(not(feature = "shared_dict"))]
        let struct_bytes = self.unique_values.capacity() * std::mem::size_of::<String>();
        #[cfg(feature = "shared_dict")]
        let struct_bytes = self.unique_values().len() * std::mem::size_of::<String>();
        let character_bytes: usize = self.unique_values().iter().map(|s| s.capacity()).sum();
        let mask_bytes = self.null_mask.as_ref().map_or(0, |m| m.est_bytes());
        data_bytes + struct_bytes + character_bytes + mask_bytes
    }

    #[inline]
    fn logical_bytes(&self) -> usize {
        // The index buffer plus the dictionary contents at their summed
        // string lengths.
        let dict_bytes: usize = self.unique_values().iter().map(|s| s.len()).sum();
        self.data.logical_bytes()
            + dict_bytes
            + self.null_mask.as_ref().map_or(0, |m| m.logical_bytes())
    }
}

/// ByteSize for BooleanArray<T>
impl<T> ByteSize for BooleanArray<T> {
    #[inline]
    fn est_bytes(&self) -> usize {
        let data_bytes = self.data.est_bytes();
        let mask_bytes = self.null_mask.as_ref().map_or(0, |m| m.est_bytes());
        data_bytes + mask_bytes
    }

    #[inline]
    fn logical_bytes(&self) -> usize {
        // Bit-packed values plus any null mask bytes
        self.data.logical_bytes() + self.null_mask.as_ref().map_or(0, |m| m.logical_bytes())
    }
}

/// ByteSize for DatetimeArray<T> (when datetime feature is enabled)
#[cfg(feature = "datetime")]
use crate::DatetimeArray;

#[cfg(feature = "datetime")]
impl<T> ByteSize for DatetimeArray<T> {
    #[inline]
    fn est_bytes(&self) -> usize {
        let data_bytes = self.data.est_bytes();
        let mask_bytes = self.null_mask.as_ref().map_or(0, |m| m.est_bytes());
        data_bytes + mask_bytes
    }

    #[inline]
    fn logical_bytes(&self) -> usize {
        self.data.logical_bytes() + self.null_mask.as_ref().map_or(0, |m| m.logical_bytes())
    }
}

// Mid-Level Enum Implementations

use crate::{NumericArray, TextArray};

/// ByteSize for NumericArray enum
impl ByteSize for NumericArray {
    fn est_bytes(&self) -> usize {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(arr) => arr.est_bytes(),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(arr) => arr.est_bytes(),
            NumericArray::Int32(arr) => arr.est_bytes(),
            NumericArray::Int64(arr) => arr.est_bytes(),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(arr) => arr.est_bytes(),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(arr) => arr.est_bytes(),
            NumericArray::UInt32(arr) => arr.est_bytes(),
            NumericArray::UInt64(arr) => arr.est_bytes(),
            NumericArray::Float32(arr) => arr.est_bytes(),
            NumericArray::Float64(arr) => arr.est_bytes(),
            NumericArray::Null => 0,
        }
    }

    fn logical_bytes(&self) -> usize {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(arr) => arr.logical_bytes(),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(arr) => arr.logical_bytes(),
            NumericArray::Int32(arr) => arr.logical_bytes(),
            NumericArray::Int64(arr) => arr.logical_bytes(),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(arr) => arr.logical_bytes(),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(arr) => arr.logical_bytes(),
            NumericArray::UInt32(arr) => arr.logical_bytes(),
            NumericArray::UInt64(arr) => arr.logical_bytes(),
            NumericArray::Float32(arr) => arr.logical_bytes(),
            NumericArray::Float64(arr) => arr.logical_bytes(),
            NumericArray::Null => 0,
        }
    }
}

/// ByteSize for TextArray enum
impl ByteSize for TextArray {
    fn est_bytes(&self) -> usize {
        match self {
            TextArray::String32(arr) => arr.est_bytes(),
            #[cfg(feature = "large_string")]
            TextArray::String64(arr) => arr.est_bytes(),
            #[cfg(feature = "default_categorical_8")]
            TextArray::Categorical8(arr) => arr.est_bytes(),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical16(arr) => arr.est_bytes(),
            #[cfg(any(
                not(feature = "default_categorical_8"),
                feature = "extended_categorical"
            ))]
            TextArray::Categorical32(arr) => arr.est_bytes(),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical64(arr) => arr.est_bytes(),
            TextArray::Null => 0,
        }
    }

    fn logical_bytes(&self) -> usize {
        match self {
            TextArray::String32(arr) => arr.logical_bytes(),
            #[cfg(feature = "large_string")]
            TextArray::String64(arr) => arr.logical_bytes(),
            #[cfg(feature = "default_categorical_8")]
            TextArray::Categorical8(arr) => arr.logical_bytes(),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical16(arr) => arr.logical_bytes(),
            #[cfg(any(
                not(feature = "default_categorical_8"),
                feature = "extended_categorical"
            ))]
            TextArray::Categorical32(arr) => arr.logical_bytes(),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical64(arr) => arr.logical_bytes(),
            TextArray::Null => 0,
        }
    }
}

#[cfg(feature = "datetime")]
use crate::TemporalArray;

/// ByteSize for TemporalArray enum (when datetime feature is enabled)
#[cfg(feature = "datetime")]
impl ByteSize for TemporalArray {
    fn est_bytes(&self) -> usize {
        match self {
            TemporalArray::Datetime32(arr) => arr.est_bytes(),
            TemporalArray::Datetime64(arr) => arr.est_bytes(),
            TemporalArray::Null => 0,
        }
    }

    fn logical_bytes(&self) -> usize {
        match self {
            TemporalArray::Datetime32(arr) => arr.logical_bytes(),
            TemporalArray::Datetime64(arr) => arr.logical_bytes(),
            TemporalArray::Null => 0,
        }
    }
}

// Top-Level Array Enum Implementation

use crate::Array;

/// ByteSize for Array enum
impl ByteSize for Array {
    fn est_bytes(&self) -> usize {
        match self {
            Array::NumericArray(arr) => arr.est_bytes(),
            Array::TextArray(arr) => arr.est_bytes(),
            #[cfg(feature = "datetime")]
            Array::TemporalArray(arr) => arr.est_bytes(),
            Array::BooleanArray(arr) => arr.est_bytes(),
            Array::Null => 0,
        }
    }

    fn logical_bytes(&self) -> usize {
        match self {
            Array::NumericArray(arr) => arr.logical_bytes(),
            Array::TextArray(arr) => arr.logical_bytes(),
            #[cfg(feature = "datetime")]
            Array::TemporalArray(arr) => arr.logical_bytes(),
            Array::BooleanArray(arr) => arr.logical_bytes(),
            Array::Null => 0,
        }
    }
}

// High-Level Structure Implementations

use crate::{Field, FieldArray, Table};

/// ByteSize for Field - metadata only, minimal size
impl ByteSize for Field {
    #[inline]
    fn est_bytes(&self) -> usize {
        // Field is mostly metadata (name, dtype, etc.)
        // Name string allocation
        self.name.capacity()
    }

    #[inline]
    fn logical_bytes(&self) -> usize {
        // The field counts the name plus every metadata key and value.
        // The dtype and nullable markers are fixed descriptors, so they
        // contribute no variable bytes.
        self.name.len()
            + self
                .metadata
                .iter()
                .map(|(k, v)| k.len() + v.len())
                .sum::<usize>()
    }
}

/// ByteSize for FieldArray - field metadata + array data
impl ByteSize for FieldArray {
    #[inline]
    fn est_bytes(&self) -> usize {
        self.field.est_bytes() + self.array.est_bytes()
    }

    #[inline]
    fn logical_bytes(&self) -> usize {
        self.field.logical_bytes() + self.array.logical_bytes()
    }
}

/// ByteSize for Table - sum of all column arrays
impl ByteSize for Table {
    fn est_bytes(&self) -> usize {
        self.cols.iter().map(|col| col.est_bytes()).sum()
    }

    fn logical_bytes(&self) -> usize {
        // The table name plus each column, where every column counts its
        // field schema strings through FieldArray
        self.name.len() + self.cols.iter().map(|col| col.logical_bytes()).sum::<usize>()
    }
}

// View Type Implementations

#[cfg(feature = "views")]
use crate::{ArrayV, TableV};

/// ByteSize for ArrayV - proportional estimate from underlying array
#[cfg(feature = "views")]
impl ByteSize for ArrayV {
    fn est_bytes(&self) -> usize {
        let full_len = self.array.len();
        let full_bytes = self.array.est_bytes();
        if full_len > 0 {
            (full_bytes * self.len()) / full_len
        } else {
            0
        }
    }

    /// The view reports the exact bytes of its window.
    /// Fixed-width windows count `len * element_width`, string
    /// windows read the payload span from the offsets buffer, bit-packed
    /// windows round up to whole bytes and a categorical window counts the
    /// dictionary contents in full because the indices reference the
    /// complete dictionary.
    fn logical_bytes(&self) -> usize {
        let len = self.len();
        let offset = self.offset;
        match &self.array {
            Array::NumericArray(inner) => match inner {
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(arr) => {
                    len * size_of::<i8>()
                        + arr.null_mask.as_ref().map_or(0, |_| (len + 7) / 8)
                }
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(arr) => {
                    len * size_of::<i16>()
                        + arr.null_mask.as_ref().map_or(0, |_| (len + 7) / 8)
                }
                NumericArray::Int32(arr) => {
                    len * size_of::<i32>()
                        + arr.null_mask.as_ref().map_or(0, |_| (len + 7) / 8)
                }
                NumericArray::Int64(arr) => {
                    len * size_of::<i64>()
                        + arr.null_mask.as_ref().map_or(0, |_| (len + 7) / 8)
                }
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(arr) => {
                    len * size_of::<u8>()
                        + arr.null_mask.as_ref().map_or(0, |_| (len + 7) / 8)
                }
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(arr) => {
                    len * size_of::<u16>()
                        + arr.null_mask.as_ref().map_or(0, |_| (len + 7) / 8)
                }
                NumericArray::UInt32(arr) => {
                    len * size_of::<u32>()
                        + arr.null_mask.as_ref().map_or(0, |_| (len + 7) / 8)
                }
                NumericArray::UInt64(arr) => {
                    len * size_of::<u64>()
                        + arr.null_mask.as_ref().map_or(0, |_| (len + 7) / 8)
                }
                NumericArray::Float32(arr) => {
                    len * size_of::<f32>()
                        + arr.null_mask.as_ref().map_or(0, |_| (len + 7) / 8)
                }
                NumericArray::Float64(arr) => {
                    len * size_of::<f64>()
                        + arr.null_mask.as_ref().map_or(0, |_| (len + 7) / 8)
                }
                NumericArray::Null => 0,
            },
            Array::TextArray(inner) => match inner {
                TextArray::String32(arr) => {
                    let payload = arr.offsets[offset + len] as usize
                        - arr.offsets[offset] as usize;
                    (len + 1) * size_of::<u32>()
                        + payload
                        + arr.null_mask.as_ref().map_or(0, |_| (len + 7) / 8)
                }
                #[cfg(feature = "large_string")]
                TextArray::String64(arr) => {
                    let payload = arr.offsets[offset + len] as usize
                        - arr.offsets[offset] as usize;
                    (len + 1) * size_of::<u64>()
                        + payload
                        + arr.null_mask.as_ref().map_or(0, |_| (len + 7) / 8)
                }
                #[cfg(feature = "default_categorical_8")]
                TextArray::Categorical8(arr) => {
                    let dict_bytes: usize =
                        arr.unique_values().iter().map(|s| s.len()).sum();
                    len * size_of::<u8>()
                        + dict_bytes
                        + arr.null_mask.as_ref().map_or(0, |_| (len + 7) / 8)
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(arr) => {
                    let dict_bytes: usize =
                        arr.unique_values().iter().map(|s| s.len()).sum();
                    len * size_of::<u16>()
                        + dict_bytes
                        + arr.null_mask.as_ref().map_or(0, |_| (len + 7) / 8)
                }
                #[cfg(any(
                    not(feature = "default_categorical_8"),
                    feature = "extended_categorical"
                ))]
                TextArray::Categorical32(arr) => {
                    let dict_bytes: usize =
                        arr.unique_values().iter().map(|s| s.len()).sum();
                    len * size_of::<u32>()
                        + dict_bytes
                        + arr.null_mask.as_ref().map_or(0, |_| (len + 7) / 8)
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(arr) => {
                    let dict_bytes: usize =
                        arr.unique_values().iter().map(|s| s.len()).sum();
                    len * size_of::<u64>()
                        + dict_bytes
                        + arr.null_mask.as_ref().map_or(0, |_| (len + 7) / 8)
                }
                TextArray::Null => 0,
            },
            #[cfg(feature = "datetime")]
            Array::TemporalArray(inner) => match inner {
                TemporalArray::Datetime32(arr) => {
                    len * size_of::<i32>()
                        + arr.null_mask.as_ref().map_or(0, |_| (len + 7) / 8)
                }
                TemporalArray::Datetime64(arr) => {
                    len * size_of::<i64>()
                        + arr.null_mask.as_ref().map_or(0, |_| (len + 7) / 8)
                }
                TemporalArray::Null => 0,
            },
            Array::BooleanArray(arr) => {
                (len + 7) / 8 + arr.null_mask.as_ref().map_or(0, |_| (len + 7) / 8)
            }
            Array::Null => 0,
        }
    }
}

/// ByteSize for TableV - sum of column view estimates
#[cfg(feature = "views")]
impl ByteSize for TableV {
    fn est_bytes(&self) -> usize {
        self.cols.iter().map(|col| col.est_bytes()).sum()
    }

    fn logical_bytes(&self) -> usize {
        // The view counts its name plus the field schema strings and the
        // window bytes of the active columns, so a column selection narrows
        // the figure the same way it narrows every other access method
        self.name.len()
            + self
                .active_col_indices()
                .into_iter()
                .map(|i| self.fields[i].logical_bytes() + self.cols[i].logical_bytes())
                .sum::<usize>()
    }
}

#[cfg(all(feature = "chunked", feature = "views"))]
use crate::{SuperArrayV, SuperTableV};

/// ByteSize for SuperArrayV - sum of slice estimates
#[cfg(all(feature = "chunked", feature = "views"))]
impl ByteSize for SuperArrayV {
    fn est_bytes(&self) -> usize {
        self.slices.iter().map(|slice| slice.est_bytes()).sum()
    }

    fn logical_bytes(&self) -> usize {
        self.slices.iter().map(|slice| slice.logical_bytes()).sum()
    }
}

/// ByteSize for SuperTableV - sum of slice estimates
#[cfg(all(feature = "chunked", feature = "views"))]
impl ByteSize for SuperTableV {
    fn est_bytes(&self) -> usize {
        self.slices.iter().map(|slice| slice.est_bytes()).sum()
    }

    fn logical_bytes(&self) -> usize {
        self.slices.iter().map(|slice| slice.logical_bytes()).sum()
    }
}

/// ByteSize for Matrix (when matrix feature is enabled)
#[cfg(feature = "matrix")]
use crate::Matrix;

#[cfg(feature = "matrix")]
impl ByteSize for Matrix {
    fn est_bytes(&self) -> usize {
        // Matrix contains data buffer for n_rows * n_cols elements
        self.data.est_bytes()
    }

    fn logical_bytes(&self) -> usize {
        unimplemented!("Matrix has not yet implemented logical bytes.")
    }
}

/// ByteSize for NdArray (when ndarray feature is enabled)
#[cfg(feature = "ndarray")]
use crate::structs::ndarray::NdArray;

#[cfg(feature = "ndarray")]
impl<T> ByteSize for NdArray<T> {
    fn est_bytes(&self) -> usize {
        // Physical backing buffer, including any stride padding.
        self.data.est_bytes()
    }

    fn logical_bytes(&self) -> usize {
        unimplemented!("NdArray has not yet implemented logical bytes.")
    }
}

/// ByteSize for NdArrayV - proportional estimate from the backing array
#[cfg(all(feature = "ndarray", feature = "views"))]
use crate::NdArrayV;

#[cfg(all(feature = "ndarray", feature = "views"))]
impl<T: crate::Float> ByteSize for NdArrayV<T> {
    fn est_bytes(&self) -> usize {
        let full_len = self.source.len();
        let full_bytes = self.source.est_bytes();
        if full_len > 0 {
            (full_bytes * self.len()) / full_len
        } else {
            0
        }
    }

    fn logical_bytes(&self) -> usize {
        unimplemented!("NdArrayV has not yet implemented logical bytes.")
    }
}

/// ByteSize for SuperNdArray - sum of batch estimates
#[cfg(all(feature = "ndarray", feature = "chunked"))]
use crate::SuperNdArray;

#[cfg(all(feature = "ndarray", feature = "chunked"))]
impl<T> ByteSize for SuperNdArray<T> {
    fn est_bytes(&self) -> usize {
        self.batches.iter().map(|batch| batch.est_bytes()).sum()
    }

    fn logical_bytes(&self) -> usize {
        unimplemented!("SuperNdArray has not yet implemented logical bytes.")
    }
}

/// ByteSize for SuperNdArrayV - sum of slice estimates
#[cfg(all(feature = "ndarray", feature = "chunked", feature = "views"))]
use crate::SuperNdArrayV;

#[cfg(all(feature = "ndarray", feature = "chunked", feature = "views"))]
impl<T: crate::Float> ByteSize for SuperNdArrayV<T> {
    fn est_bytes(&self) -> usize {
        self.slices.iter().map(|slice| slice.est_bytes()).sum()
    }

    fn logical_bytes(&self) -> usize {
        unimplemented!("SuperNdArrayV has not yet implemented logical bytes.")
    }
}

/// ByteSize for XArray - storage plus coordinate arrays
#[cfg(feature = "xarray")]
use crate::XArray;

#[cfg(feature = "xarray")]
impl<T: crate::Float> ByteSize for XArray<T> {
    fn est_bytes(&self) -> usize {
        use crate::structs::xarray::NdArrayE;
        let data_bytes = match self.storage() {
            NdArrayE::Owned(nd) => nd.est_bytes(),
            #[cfg(feature = "views")]
            NdArrayE::View(v) => v.est_bytes(),
        };
        let coord_bytes: usize = self
            .axes()
            .iter()
            .map(|axis| {
                axis.name.capacity()
                    + axis.coords.as_ref().map(|c| c.est_bytes()).unwrap_or(0)
            })
            .sum();
        data_bytes + coord_bytes
    }

    fn logical_bytes(&self) -> usize {
        unimplemented!("XArray has not yet implemented logical bytes.")
    }
}

/// ByteSize for Cube (when cube feature is enabled)
#[cfg(feature = "cube")]
use crate::Cube;

#[cfg(feature = "cube")]
impl ByteSize for Cube {
    fn est_bytes(&self) -> usize {
        // Cube contains multiple tables
        self.tables.iter().map(|tbl| tbl.est_bytes()).sum()
    }

    fn logical_bytes(&self) -> usize {
        self.tables.iter().map(|tbl| tbl.logical_bytes()).sum()
    }
}

/// ByteSize for SuperArray (when chunked feature is enabled)
#[cfg(feature = "chunked")]
use crate::SuperArray;

#[cfg(feature = "chunked")]
impl ByteSize for SuperArray {
    fn est_bytes(&self) -> usize {
        // Sum of all chunk arrays
        self.chunks().iter().map(|chunk| chunk.est_bytes()).sum()
    }

    fn logical_bytes(&self) -> usize {
        self.chunks().iter().map(|chunk| chunk.logical_bytes()).sum()
    }
}

/// ByteSize for SuperTable (when chunked feature is enabled)
#[cfg(feature = "chunked")]
use crate::SuperTable;

#[cfg(feature = "chunked")]
impl ByteSize for SuperTable {
    fn est_bytes(&self) -> usize {
        // Sum of all batch tables
        self.batches.iter().map(|batch| batch.est_bytes()).sum()
    }

    fn logical_bytes(&self) -> usize {
        self.batches.iter().map(|batch| batch.logical_bytes()).sum()
    }
}

// Value Enum Implementation

#[cfg(feature = "value_type")]
use crate::Value;

#[cfg(feature = "value_type")]
#[cfg(feature = "scalar_type")]
use crate::Scalar;

/// ByteSize for Scalar (when scalar_type feature is enabled)
#[cfg(feature = "value_type")]
#[cfg(feature = "scalar_type")]
impl ByteSize for Scalar {
    #[inline]
    fn est_bytes(&self) -> usize {
        // Scalars are stack-allocated, minimal heap usage
        // Only String32/String64 use heap
        match self {
            Scalar::String32(s) => s.capacity(),
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => s.capacity(),
            _ => 0, // Other scalars are inline
        }
    }

    /// A scalar reports the width its value occupies as a single Arrow
    /// element, so a boolean rounds up to the one bit-packed byte and a
    /// string reports its populated length.
    fn logical_bytes(&self) -> usize {
        match self {
            Scalar::Null => 0,
            Scalar::Boolean(_) => 1,
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int8(_) => size_of::<i8>(),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int16(_) => size_of::<i16>(),
            Scalar::Int32(_) => size_of::<i32>(),
            Scalar::Int64(_) => size_of::<i64>(),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt8(_) => size_of::<u8>(),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt16(_) => size_of::<u16>(),
            Scalar::UInt32(_) => size_of::<u32>(),
            Scalar::UInt64(_) => size_of::<u64>(),
            Scalar::Float32(_) => size_of::<f32>(),
            Scalar::Float64(_) => size_of::<f64>(),
            Scalar::String32(s) => s.len(),
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => s.len(),
            #[cfg(feature = "datetime")]
            Scalar::Datetime32(_) => size_of::<i32>(),
            #[cfg(feature = "datetime")]
            Scalar::Datetime64(_) => size_of::<i64>(),
            // The variant carries no value
            #[cfg(feature = "datetime")]
            Scalar::Interval => 0,
        }
    }
}

/// ByteSize for Value enum - delegates to inner types
#[cfg(feature = "value_type")]
impl ByteSize for Value {
    fn est_bytes(&self) -> usize {
        match self {
            #[cfg(feature = "scalar_type")]
            Value::Scalar(s) => s.est_bytes(),
            Value::Array(arr) => arr.est_bytes(),
            #[cfg(feature = "views")]
            Value::ArrayView(av) => av.est_bytes(),
            Value::Table(tbl) => tbl.est_bytes(),
            #[cfg(feature = "views")]
            Value::TableView(tv) => tv.est_bytes(),
            #[cfg(feature = "chunked")]
            Value::SuperArray(sa) => sa.est_bytes(),
            #[cfg(all(feature = "chunked", feature = "views"))]
            Value::SuperArrayView(sav) => sav.est_bytes(),
            #[cfg(feature = "chunked")]
            Value::SuperTable(st) => st.est_bytes(),
            #[cfg(all(feature = "chunked", feature = "views"))]
            Value::SuperTableView(stv) => stv.est_bytes(),
            Value::FieldArray(fa) => fa.est_bytes(),
            #[cfg(feature = "matrix")]
            Value::Matrix(m) => m.est_bytes(),
            #[cfg(feature = "ndarray")]
            Value::NdArray(nd) => nd.est_bytes(),
            #[cfg(all(feature = "ndarray", feature = "views"))]
            Value::NdArrayView(v) => v.est_bytes(),
            #[cfg(all(feature = "ndarray", feature = "chunked"))]
            Value::SuperNdArray(snd) => snd.est_bytes(),
            #[cfg(all(feature = "ndarray", feature = "chunked", feature = "views"))]
            Value::SuperNdArrayView(sv) => sv.est_bytes(),
            #[cfg(feature = "xarray")]
            Value::XArray(xa) => xa.est_bytes(),
            #[cfg(feature = "cube")]
            Value::Cube(c) => c.est_bytes(),
            Value::VecValue(vec) => {
                // Recursively sum all contained values
                vec.iter().map(|v| v.est_bytes()).sum::<usize>()
                    + vec.capacity() * size_of::<Value>() // Vec capacity overhead
            }
            Value::BoxValue(boxed) => boxed.est_bytes(),
            Value::ArcValue(arc) => arc.est_bytes(),
            Value::Tuple2(tuple) => tuple.0.est_bytes() + tuple.1.est_bytes(),
            Value::Tuple3(tuple) => tuple.0.est_bytes() + tuple.1.est_bytes() + tuple.2.est_bytes(),
            Value::Tuple4(tuple) => {
                tuple.0.est_bytes()
                    + tuple.1.est_bytes()
                    + tuple.2.est_bytes()
                    + tuple.3.est_bytes()
            }
            Value::Tuple5(tuple) => {
                tuple.0.est_bytes()
                    + tuple.1.est_bytes()
                    + tuple.2.est_bytes()
                    + tuple.3.est_bytes()
                    + tuple.4.est_bytes()
            }
            Value::Tuple6(tuple) => {
                tuple.0.est_bytes()
                    + tuple.1.est_bytes()
                    + tuple.2.est_bytes()
                    + tuple.3.est_bytes()
                    + tuple.4.est_bytes()
                    + tuple.5.est_bytes()
            }
            Value::Custom(_) => {
                // Cannot introspect custom types, return minimal estimate
                size_of::<std::sync::Arc<dyn crate::traits::custom_value::CustomValue>>()
            }
        }
    }

    /// The tabular variants delegate to their inner type. The numerical
    /// container variants (`Matrix`, `NdArray` and their chunked and view
    /// forms, plus `XArray`) and `Custom` do not define logical byte
    /// accounting and panic with `unimplemented!` when called.
    fn logical_bytes(&self) -> usize {
        match self {
            #[cfg(feature = "scalar_type")]
            Value::Scalar(s) => s.logical_bytes(),
            Value::Array(arr) => arr.logical_bytes(),
            #[cfg(feature = "views")]
            Value::ArrayView(av) => av.logical_bytes(),
            Value::Table(tbl) => tbl.logical_bytes(),
            #[cfg(feature = "views")]
            Value::TableView(tv) => tv.logical_bytes(),
            #[cfg(feature = "chunked")]
            Value::SuperArray(sa) => sa.logical_bytes(),
            #[cfg(all(feature = "chunked", feature = "views"))]
            Value::SuperArrayView(sav) => sav.logical_bytes(),
            #[cfg(feature = "chunked")]
            Value::SuperTable(st) => st.logical_bytes(),
            #[cfg(all(feature = "chunked", feature = "views"))]
            Value::SuperTableView(stv) => stv.logical_bytes(),
            Value::FieldArray(fa) => fa.logical_bytes(),
            #[cfg(feature = "matrix")]
            Value::Matrix(_) => {
                unimplemented!("Matrix does not define logical byte accounting")
            }
            #[cfg(feature = "ndarray")]
            Value::NdArray(_) => {
                unimplemented!("NdArray does not define logical byte accounting")
            }
            #[cfg(all(feature = "ndarray", feature = "views"))]
            Value::NdArrayView(_) => {
                unimplemented!("NdArrayV does not define logical byte accounting")
            }
            #[cfg(all(feature = "ndarray", feature = "chunked"))]
            Value::SuperNdArray(_) => {
                unimplemented!("SuperNdArray does not define logical byte accounting")
            }
            #[cfg(all(feature = "ndarray", feature = "chunked", feature = "views"))]
            Value::SuperNdArrayView(_) => {
                unimplemented!("SuperNdArrayV does not define logical byte accounting")
            }
            #[cfg(feature = "xarray")]
            Value::XArray(_) => {
                unimplemented!("XArray does not define logical byte accounting")
            }
            #[cfg(feature = "cube")]
            Value::Cube(c) => c.logical_bytes(),
            Value::VecValue(vec) => vec.iter().map(|v| v.logical_bytes()).sum::<usize>(),
            Value::BoxValue(boxed) => boxed.logical_bytes(),
            Value::ArcValue(arc) => arc.logical_bytes(),
            Value::Tuple2(tuple) => tuple.0.logical_bytes() + tuple.1.logical_bytes(),
            Value::Tuple3(tuple) => {
                tuple.0.logical_bytes() + tuple.1.logical_bytes() + tuple.2.logical_bytes()
            }
            Value::Tuple4(tuple) => {
                tuple.0.logical_bytes()
                    + tuple.1.logical_bytes()
                    + tuple.2.logical_bytes()
                    + tuple.3.logical_bytes()
            }
            Value::Tuple5(tuple) => {
                tuple.0.logical_bytes()
                    + tuple.1.logical_bytes()
                    + tuple.2.logical_bytes()
                    + tuple.3.logical_bytes()
                    + tuple.4.logical_bytes()
            }
            Value::Tuple6(tuple) => {
                tuple.0.logical_bytes()
                    + tuple.1.logical_bytes()
                    + tuple.2.logical_bytes()
                    + tuple.3.logical_bytes()
                    + tuple.4.logical_bytes()
                    + tuple.5.logical_bytes()
            }
            Value::Custom(_) => {
                unimplemented!("CustomValue does not define logical byte accounting")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_array_counts_data_and_mask() {
        let mut arr = IntegerArray::<i64>::from_slice(&[1, 2, 3, 4, 5]);
        assert_eq!(arr.logical_bytes(), 40);
        arr.null_mask = Some(Bitmask::new_set_all(5, true));
        assert_eq!(arr.logical_bytes(), 41);
    }

    #[test]
    fn buffer_capacity_slack_is_excluded() {
        let mut v: Vec64<i32> = Vec64::with_capacity(100);
        v.push(1);
        v.push(2);
        v.push(3);
        assert_eq!(v.logical_bytes(), 12);
        assert_eq!(v.est_bytes(), 400);
    }

    #[test]
    fn bitmask_rounds_bits_to_bytes() {
        let mask = Bitmask::new_set_all(10, true);
        assert_eq!(mask.logical_bytes(), 2);
    }

    #[test]
    fn string_array_excludes_capacity_slack() {
        let strings: Vec64<String> = (0..1000).map(|i| format!("row_{}", i)).collect();
        let refs: Vec64<&str> = strings.iter().map(String::as_str).collect();
        let payload: usize = strings.iter().map(|s| s.len()).sum();
        let arr = StringArray::<u32>::from_vec64(refs, None);
        assert_eq!(arr.logical_bytes(), payload + 1001 * size_of::<u32>());
        assert!(arr.est_bytes() >= arr.logical_bytes());
    }

    #[test]
    fn categorical_array_counts_indices_and_dictionary_contents() {
        let arr = CategoricalArray::<u32>::from_slices(
            &[0, 1, 2, 1, 0, 2],
            &["red".to_string(), "green".to_string(), "blue".to_string()],
        );
        assert_eq!(arr.logical_bytes(), 6 * size_of::<u32>() + 12);
    }

    #[test]
    fn boolean_array_rounds_bits_to_bytes() {
        let arr = BooleanArray::<()>::from_slice(&[true; 10]);
        assert_eq!(arr.logical_bytes(), 2);
    }

    #[cfg(feature = "views")]
    #[test]
    fn string_view_reports_exact_window_bytes() {
        use crate::{ArrayV, vec64};
        let refs: Vec64<&str> = vec64!["a", "bb", "ccc", "dddd", "eeeee"];
        let arr = StringArray::<u32>::from_vec64(refs, None);
        let view = ArrayV::new(crate::Array::from_string32(arr), 1, 3);
        // Window covers "bb", "ccc" and "dddd", so the payload spans 9
        // bytes and the offsets window holds 4 entries
        assert_eq!(view.logical_bytes(), 4 * size_of::<u32>() + 9);
    }

    #[cfg(feature = "views")]
    #[test]
    fn full_width_table_view_matches_owned_table() {
        use crate::{FieldArray, Table, arr_i32, arr_str32, vec64};
        let ids = vec64![1i32, 2, 3, 4, 5];
        let refs: Vec64<&str> = vec64!["a", "bb", "ccc", "dddd", "eeeee"];
        let table = Table::new(
            "t".to_string(),
            Some(vec![
                FieldArray::from_arr("ids", arr_i32!(ids)),
                FieldArray::from_arr("labels", arr_str32!(refs)),
            ]),
        );
        assert_eq!(table.slice(0, 5).logical_bytes(), table.logical_bytes());
        // A mid-table window counts the table name "t", the field names
        // "ids" and "labels", 3 ids, a 4-entry offsets window and the
        // "bb" + "ccc" + "dddd" payload
        assert_eq!(
            table.slice(1, 3).logical_bytes(),
            1 + 3 + 6 + 3 * size_of::<i32>() + 4 * size_of::<u32>() + 9
        );
    }

    #[cfg(all(feature = "value_type", feature = "scalar_type"))]
    #[test]
    fn scalar_reports_element_width() {
        assert_eq!(Scalar::Int64(7).logical_bytes(), 8);
        assert_eq!(Scalar::String32("abc".into()).logical_bytes(), 3);
        assert_eq!(Scalar::Boolean(true).logical_bytes(), 1);
        assert_eq!(Scalar::Null.logical_bytes(), 0);
    }
}
