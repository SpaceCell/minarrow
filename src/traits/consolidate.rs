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

//! # Consolidate Trait Module
//!
//! Provides uniform consolidation of chunked types into contiguous storage.
//!
//! ## Overview
//! The `Consolidate` trait materialises chunked/segmented data into a single
//! contiguous buffer, enabling efficient operations and compatibility
//! with APIs that require contiguous memory.
//!
//! - **SuperArray** -> **Array**: Merges all chunks into one contiguous array
//! - **SuperTable** -> **Table**: Merges all batches into one contiguous table
//! - **SuperArrayView** -> **Array**: Copies view chunks into owned array
//! - **SuperTableView** -> **Table**: Copies view batches into owned table
//!
//! ## Use Case
//! Sometimes processing returns chunked results to retain zero-copy and/or avoid buffer thrashing.
//! Call `.consolidate()` when you need contiguous memory.
//!
//! ## Example
//! ```ignore
//! use minarrow::{SuperArray, Consolidate};
//!
//! let chunked = SuperArray::from_chunks(vec![chunk1, chunk2, chunk3]);
//! // chunked has 3 separate memory regions
//!
//! let contiguous = chunked.consolidate();
//! // contiguous is a single Array with one buffer
//! ```

use crate::structs::bitmask::Bitmask;

/// Trait for consolidating chunked types into contiguous storage.
///
/// # Output Type
/// The `Output` associated type defines what the consolidated result is:
/// - `SuperArray::Output = Array`
/// - `SuperTable::Output = Table`
///
/// # When to Use
/// - After parallel batch processing that returns chunked results
/// - Before operations requiring contiguous memory (e.g., FFI, certain operations)
/// - When you need to serialise to formats requiring single buffers
///
/// # Naming
/// The consolidated result uses the name from the source. Call `.rename()` or
/// equivalent if you need a different name.
pub trait Consolidate {
    /// The type produced after consolidation.
    type Output;

    /// Consolidates chunked data into contiguous storage.
    ///
    /// Consumes `self` and returns a consolidated `Output`.
    fn consolidate(self) -> Self::Output;
}

// Helper Functions for Consolidation

/// Extends a result null mask from a source mask's range.
///
/// Handles all four cases of (result has mask, source has mask) combinations:
/// - Both have masks: extend from source range
/// - Result has mask, source doesn't: mark new bits as valid
/// - Source has mask, result doesn't: create new mask with previous bits valid
/// - Neither has mask: no-op
pub fn extend_null_mask(
    result_mask: &mut Option<Bitmask>,
    result_len: usize,
    source_mask: Option<&Bitmask>,
    offset: usize,
    len: usize,
) {
    match (result_mask.as_mut(), source_mask) {
        (Some(mask), Some(src)) => {
            mask.extend((offset..offset + len).map(|i| src.get(i)));
        }
        (Some(mask), None) => {
            // Source has no nulls, set all bits valid
            for _ in 0..len {
                mask.set(mask.len(), true);
            }
        }
        (None, Some(src)) => {
            // Create mask, all previous values valid, then copy from source
            let mut mask = Bitmask::new_set_all(result_len, true);
            mask.extend((offset..offset + len).map(|i| src.get(i)));
            *result_mask = Some(mask);
        }
        (None, None) => {}
    }
}

// Top-level Consolidate impl over `ArrayVT` view tuples.
//
// Exhaustively matches on the first chunk's `Array` variant, gathers
// every chunk into the matching typed AVT view tuple, then calls
// `.consolidate()` on it. The typed impl lives on the typed array's
// module and owns the per-width buffer-extension logic, reading from
// each window without materialising an owned typed sub-array.

#[cfg(feature = "chunked")]
impl<'a> Consolidate for Vec<crate::aliases::ArrayVT<'a>> {
    type Output = crate::enums::array::Array;

    fn consolidate(self) -> Self::Output {
        use crate::enums::array::Array;
        use crate::enums::collections::numeric_array::NumericArray;
        use crate::enums::collections::text_array::TextArray;
        #[cfg(feature = "datetime")]
        use crate::enums::collections::temporal_array::TemporalArray;
        use std::sync::Arc;

        assert!(!self.is_empty(), "consolidate() called on empty Vec<ArrayVT>");

        // Gather typed AVT tuples for the matching variant. Every chunk
        // must share the same width as the first; mismatches are a
        // schema violation by the caller and panic.
        macro_rules! gather_numeric {
            ($variant:ident, $T:ty) => {{
                let views: Vec<crate::aliases::IntegerAVT<'a, $T>> = self
                    .iter()
                    .map(|(arr, off, len)| match arr {
                        Array::NumericArray(NumericArray::$variant(a)) => (a.as_ref(), *off, *len),
                        _ => panic!(
                            "inconsistent NumericArray variants in chunk vector"
                        ),
                    })
                    .collect();
                Array::NumericArray(NumericArray::$variant(Arc::new(views.consolidate())))
            }};
        }

        macro_rules! gather_float {
            ($variant:ident, $T:ty) => {{
                let views: Vec<crate::aliases::FloatAVT<'a, $T>> = self
                    .iter()
                    .map(|(arr, off, len)| match arr {
                        Array::NumericArray(NumericArray::$variant(a)) => (a.as_ref(), *off, *len),
                        _ => panic!(
                            "inconsistent NumericArray variants in chunk vector"
                        ),
                    })
                    .collect();
                Array::NumericArray(NumericArray::$variant(Arc::new(views.consolidate())))
            }};
        }

        macro_rules! gather_string {
            ($variant:ident, $T:ty) => {{
                let views: Vec<crate::aliases::StringAVT<'a, $T>> = self
                    .iter()
                    .map(|(arr, off, len)| match arr {
                        Array::TextArray(TextArray::$variant(a)) => (a.as_ref(), *off, *len),
                        _ => panic!(
                            "inconsistent TextArray variants in chunk vector"
                        ),
                    })
                    .collect();
                Array::TextArray(TextArray::$variant(Arc::new(views.consolidate())))
            }};
        }

        macro_rules! gather_categorical {
            ($variant:ident, $T:ty) => {{
                let views: Vec<crate::aliases::CategoricalAVT<'a, $T>> = self
                    .iter()
                    .map(|(arr, off, len)| match arr {
                        Array::TextArray(TextArray::$variant(a)) => (a.as_ref(), *off, *len),
                        _ => panic!(
                            "inconsistent TextArray variants in chunk vector"
                        ),
                    })
                    .collect();
                Array::TextArray(TextArray::$variant(Arc::new(views.consolidate())))
            }};
        }

        #[cfg(feature = "datetime")]
        macro_rules! gather_datetime {
            ($variant:ident, $T:ty) => {{
                let views: Vec<crate::aliases::DatetimeAVT<'a, $T>> = self
                    .iter()
                    .map(|(arr, off, len)| match arr {
                        Array::TemporalArray(TemporalArray::$variant(a)) => {
                            (a.as_ref(), *off, *len)
                        }
                        _ => panic!(
                            "inconsistent TemporalArray variants in chunk vector"
                        ),
                    })
                    .collect();
                Array::TemporalArray(TemporalArray::$variant(Arc::new(views.consolidate())))
            }};
        }

        match self[0].0 {
            Array::NumericArray(NumericArray::Int32(_)) => gather_numeric!(Int32, i32),
            Array::NumericArray(NumericArray::Int64(_)) => gather_numeric!(Int64, i64),
            Array::NumericArray(NumericArray::UInt32(_)) => gather_numeric!(UInt32, u32),
            Array::NumericArray(NumericArray::UInt64(_)) => gather_numeric!(UInt64, u64),
            Array::NumericArray(NumericArray::Float32(_)) => gather_float!(Float32, f32),
            Array::NumericArray(NumericArray::Float64(_)) => gather_float!(Float64, f64),
            #[cfg(feature = "extended_numeric_types")]
            Array::NumericArray(NumericArray::Int8(_)) => gather_numeric!(Int8, i8),
            #[cfg(feature = "extended_numeric_types")]
            Array::NumericArray(NumericArray::Int16(_)) => gather_numeric!(Int16, i16),
            #[cfg(feature = "extended_numeric_types")]
            Array::NumericArray(NumericArray::UInt8(_)) => gather_numeric!(UInt8, u8),
            #[cfg(feature = "extended_numeric_types")]
            Array::NumericArray(NumericArray::UInt16(_)) => gather_numeric!(UInt16, u16),
            Array::NumericArray(NumericArray::Null) => Array::Null,

            Array::TextArray(TextArray::String32(_)) => gather_string!(String32, u32),
            #[cfg(feature = "large_string")]
            Array::TextArray(TextArray::String64(_)) => gather_string!(String64, u64),
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            Array::TextArray(TextArray::Categorical32(_)) => gather_categorical!(Categorical32, u32),
            #[cfg(feature = "default_categorical_8")]
            Array::TextArray(TextArray::Categorical8(_)) => gather_categorical!(Categorical8, u8),
            #[cfg(feature = "extended_categorical")]
            Array::TextArray(TextArray::Categorical16(_)) => gather_categorical!(Categorical16, u16),
            #[cfg(feature = "extended_categorical")]
            Array::TextArray(TextArray::Categorical64(_)) => gather_categorical!(Categorical64, u64),
            Array::TextArray(TextArray::Null) => Array::Null,

            Array::BooleanArray(_) => {
                let views: Vec<crate::aliases::BooleanAVT<'a, ()>> = self
                    .iter()
                    .map(|(arr, off, len)| match arr {
                        Array::BooleanArray(a) => (a.as_ref(), *off, *len),
                        _ => panic!("inconsistent Array variants in chunk vector"),
                    })
                    .collect();
                Array::BooleanArray(Arc::new(views.consolidate()))
            }

            #[cfg(feature = "datetime")]
            Array::TemporalArray(TemporalArray::Datetime32(_)) => gather_datetime!(Datetime32, i32),
            #[cfg(feature = "datetime")]
            Array::TemporalArray(TemporalArray::Datetime64(_)) => gather_datetime!(Datetime64, i64),
            #[cfg(feature = "datetime")]
            Array::TemporalArray(TemporalArray::Null) => Array::Null,

            Array::Null => Array::Null,
        }
    }
}

