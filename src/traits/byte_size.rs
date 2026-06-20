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
//! ## Usage
//! ```rust
//! use minarrow::{IntegerArray, ByteSize, MaskedArray};
//!
//! let arr = IntegerArray::<i64>::from_slice(&[1, 2, 3, 4, 5]);
//! let bytes = arr.est_bytes();
//! // Returns data buffer size: 5 * 8 = 40 bytes (plus small overhead)
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
}

/// ByteSize for Buffer<T> - unified owned/shared buffer
impl<T> ByteSize for Buffer<T> {
    #[inline]
    fn est_bytes(&self) -> usize {
        // Capacity in elements * size per element
        self.capacity() * size_of::<T>()
    }
}

/// ByteSize for Bitmask - bit-packed bitmask
impl ByteSize for Bitmask {
    #[inline]
    fn est_bytes(&self) -> usize {
        // Bit-packed: (capacity + 7) / 8 bytes
        self.bits.est_bytes()
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
}

/// ByteSize for FloatArray<T>
impl<T> ByteSize for FloatArray<T> {
    #[inline]
    fn est_bytes(&self) -> usize {
        let data_bytes = self.data.est_bytes();
        let mask_bytes = self.null_mask.as_ref().map_or(0, |m| m.est_bytes());
        data_bytes + mask_bytes
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
}

/// ByteSize for CategoricalArray<T>
impl<T: crate::traits::type_unions::Integer> ByteSize for CategoricalArray<T> {
    #[inline]
    fn est_bytes(&self) -> usize {
        let data_bytes = self.data.est_bytes();
        // Approximate the dictionary container at
        // `values_count * size_of::<String>()`
        let unique_values_bytes = self.unique_values().len() * std::mem::size_of::<String>();
        let mask_bytes = self.null_mask.as_ref().map_or(0, |m| m.est_bytes());
        data_bytes + unique_values_bytes + mask_bytes
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
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            TextArray::Categorical32(arr) => arr.est_bytes(),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical64(arr) => arr.est_bytes(),
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
}

/// ByteSize for FieldArray - field metadata + array data
impl ByteSize for FieldArray {
    #[inline]
    fn est_bytes(&self) -> usize {
        self.field.est_bytes() + self.array.est_bytes()
    }
}

/// ByteSize for Table - sum of all column arrays
impl ByteSize for Table {
    fn est_bytes(&self) -> usize {
        self.cols.iter().map(|col| col.est_bytes()).sum()
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
}

/// ByteSize for TableV - sum of column view estimates
#[cfg(feature = "views")]
impl ByteSize for TableV {
    fn est_bytes(&self) -> usize {
        self.cols.iter().map(|col| col.est_bytes()).sum()
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
}

/// ByteSize for SuperTableV - sum of slice estimates
#[cfg(all(feature = "chunked", feature = "views"))]
impl ByteSize for SuperTableV {
    fn est_bytes(&self) -> usize {
        self.slices.iter().map(|slice| slice.est_bytes()).sum()
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
}
