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

//! # **SuperArrayView Module** - *Chunked, Windowed View over 1:M Arrays*
//!
//! `SuperArrayV` is a **borrowed, chunked view** over a single logical array,
//! exposing an arbitrary `[offset .. offset + len)` window that may span
//! multiple underlying chunks. It presents those chunks as one continuous
//! logical range without copying the underlying memory.
//!
//! ## Role
//! - Represents a *mini-batch* window for one `Array` (or one entry inside a
//!   `SuperArray`), useful for streaming, batching, or region-wise processing.
//! - Similar in shape to `SuperArray`, but semantically a **view** over a single
//!   column’s region rather than a bag of heterogeneous columns.
//!
//! ## Interop
//! - Constructed by higher-level chunked containers (`SuperArray::slice`).
//! - Can be materialised to a contiguous `Array` via [`SuperArrayV::consolidate`].
//! - Preserves schema via an `Arc<Field>`; no schema cloning per row.
//!
//! ## Features
//! - Zero-copy iteration over chunks: [`chunks`](SuperArrayV::chunks) / [`iter`](SuperArrayV::iter).
//! - Row-wise logical iteration: [`iter_rows`](SuperArrayV::iter_rows) and
//!   random access via [`row_slice`](SuperArrayV::row_slice) / [`get_value`](SuperArrayV::get_value).
//! - Sub-windowing: [`slice`](SuperArrayV::slice) returns another borrowed view.
//!
//! ## Performance Notes
//! - Iterating rows may touch non-contiguous memory if the window crosses chunk
//!   boundaries. For hot loops, prefer contiguous runs or materialise with
//!   `consolidate()`.
//!
//! ## Invariants
//! - `len` is the logical row count of this view.
//! - `slices` are ordered, non-overlapping, and cover at most `len` rows.
//! - `field` is the schema for the underlying array and is shared by all slices.
use std::sync::Arc;

use crate::{
    Array, ArrayV, ArrayVT, Field, SuperArray,
    enums::error::MinarrowError,
    enums::shape_dim::ShapeDim,
    traits::{concatenate::Concatenate, consolidate::Consolidate, shape::Shape},
};

/// # SuperArrayView
///
/// Borrowed view over an arbitrary `[offset .. offset+len)` window of a `ChunkedArray`.
/// The window may span multiple internal chunks, presenting them as a unified logical view.
///
/// ## Purpose
/// A mini-batch of **one** array (or one `SuperArray` entry). Handy when you’ve
/// cached null counts / stats over regions and want to operate on those regions
/// without materialising the whole column.
///
/// ## Fields
/// - `slices`: constituent `ArrayView` pieces spanning the window.
/// - `len`: total logical row count for this view.
/// - `field`: schema field associated with the array (shared).
///
/// ## Notes
/// - Use [`chunks`](Self::chunks) / [`iter`](Self::iter) to walk chunk pieces,
///   or [`iter_rows`](Self::iter_rows) to traverse row-by-row across chunks.
/// - For hot paths, prefer contiguous (memory) windows; otherwise consider
///   [`consolidate`](Self::consolidate) to materialise a single buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct SuperArrayV {
    pub slices: Vec<ArrayV>,
    pub len: usize,
    pub field: Arc<Field>,
}

impl SuperArrayV {
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn n_slices(&self) -> usize {
        self.slices.len()
    }

    /// Iterator over the underlying `ArraySlice`s
    #[inline]
    pub fn chunks(&self) -> impl Iterator<Item = &ArrayV> {
        self.slices.iter()
    }

    /// Returns a sub-window of this chunked array view over `[offset .. offset+len)`.
    ///
    /// Produces a new `ChunkedArrayView` with updated slice metadata.
    /// The field metadata is preserved as-is. Underlying data is not cloned.
    pub fn slice(&self, mut offset: usize, mut len: usize) -> Self {
        assert!(offset + len <= self.len, "slice out of bounds");

        let mut slices = Vec::new();
        for array_view in &self.slices {
            let base_len = array_view.len();
            let base_offset = array_view.offset;
            if offset >= base_len {
                offset -= base_len;
                continue;
            }

            let take = (base_len - offset).min(len);
            slices.push(ArrayV::new(
                array_view.array.clone(),
                base_offset + offset,
                take,
            ));

            len -= take;
            if len == 0 {
                break;
            }
            offset = 0;
        }

        Self {
            slices,
            len: self.len,
            field: self.field.clone(),
        }
    }

    /// Returns the 1-element `Array` value at the logical index.
    pub fn get_value(&self, mut idx: usize) -> Array {
        for slice in &self.slices {
            if idx < slice.len() {
                return slice.array.slice_clone(slice.offset + idx, 1);
            }
            idx -= slice.len();
        }
        panic!("index out of bounds");
    }

    /// Iterate over all ArraySliceâ€™s in this slice.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = ArrayV> + '_ {
        self.slices.iter().cloned()
    }

    /// Returns an iterator over all rows as 1-element `ArrayView`s.
    ///
    /// Allows walking across potentially chunked memory logically row-by-row.
    #[inline]
    pub fn iter_rows(&self) -> impl Iterator<Item = ArrayVT<'_>> + '_ {
        self.slices
            .iter()
            .flat_map(|slice| {
                let base_offset = slice.offset;
                (0..slice.len()).map(move |i| (&slice.array, base_offset + i, 1))
            })
            .take(self.len)
    }

    /// Maps a logical row index into the corresponding (slice_index, intra_row_offset) pair.
    #[inline]
    fn locate(&self, row: usize) -> (usize, usize) {
        assert!(row < self.len, "row out of bounds");
        let mut acc = 0;
        for (chunk_idx, slice) in self.slices.iter().enumerate() {
            if row < acc + slice.len() {
                return (chunk_idx, row - acc);
            }
            acc += slice.len();
        }
        unreachable!()
    }

    /// Returns a zero-copy 1-row `ArrayWindow` at the given logical row index.
    pub fn row_slice(&self, row: usize) -> ArrayV {
        let (ci, ri) = self.locate(row);
        let (array, base_offset, _) = self.slices[ci].as_tuple();
        ArrayV::new(array, base_offset + ri, 1)
    }

    /// Returns the total number of elements in the array
    pub fn len(&self) -> usize {
        self.len
    }
}

impl Shape for SuperArrayV {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank1(self.len())
    }
}

impl Consolidate for SuperArrayV {
    type Output = Array;

    /// Consolidates all view chunks into a single contiguous `Array`.
    ///
    /// # Optimisations
    /// 1. Single slice covering full array: returns cheap Arc clone (no copy).
    /// 2. Multiple consecutive slices on same buffer covering full array: Arc clone.
    /// 3. Otherwise: directly extends from source data slices (single copy per element).
    fn consolidate(self) -> Array {
        if self.slices.is_empty() {
            panic!("consolidate() called on empty SuperArrayV");
        }

        // Single slice optimisation
        if self.slices.len() == 1 {
            let slice = &self.slices[0];
            if slice.offset == 0 && slice.len() == slice.array.len() {
                return slice.array.clone();
            }
        }

        // Multiple slices - check if consecutive on same underlying buffer
        let first = &self.slices[0];
        let first_array_len = first.array.len();
        let mut expected_offset = first.offset + first.len();
        let mut all_same_buffer = true;
        let mut is_consecutive = true;

        for slice in self.slices.iter().skip(1) {
            // Check same underlying array by comparing total length
            if slice.array.len() != first_array_len {
                all_same_buffer = false;
                break;
            }
            if slice.offset != expected_offset {
                is_consecutive = false;
                break;
            }
            expected_offset = slice.offset + slice.len();
        }

        if all_same_buffer && is_consecutive && self.slices.len() > 1 {
            let last_slice = &self.slices[self.slices.len() - 1];
            let combined_end = last_slice.offset + last_slice.len();

            if first.offset == 0 && combined_end == first_array_len {
                return first.array.clone();
            }
        }

        // Build `Vec<ArrayVT>` view tuples and delegate to its
        // `Consolidate` impl, which exhaustively dispatches on the first
        // chunk's `Array` variant and reads each window directly from
        // the source buffers via the typed array's impl - no
        // intermediate slice-clone.
        let views: Vec<crate::aliases::ArrayVT<'_>> = self
            .slices
            .iter()
            .map(|s| (&s.array, s.offset, s.len()))
            .collect();
        views.consolidate()
    }
}

impl Concatenate for SuperArrayV {
    /// Concatenates two super array views by materialising both to owned arrays,
    /// concatenating them, and wrapping the result back in a view.
    ///
    /// # Notes
    /// - This operation copies data from both views to create owned arrays.
    /// - The resulting view contains a single slice wrapping the concatenated array.
    /// - The field metadata from the first view is preserved.
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        // Preserve field before consuming self
        let field = self.field.clone();

        // Materialise both views to owned arrays
        let self_array = self.consolidate();
        let other_array = other.consolidate();

        // Concatenate the owned arrays
        let concatenated = self_array.concat(other_array)?;
        let len = concatenated.len();

        // Wrap the result in a new view with a single slice
        Ok(SuperArrayV {
            slices: vec![ArrayV::from(concatenated)],
            len,
            field,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::arrow_dtype::ArrowType;
    use crate::{FieldArray, NumericArray};

    // Test helper - creates a FieldArray for i32
    fn fa(name: &str, vals: &[i32]) -> FieldArray {
        let arr = Array::from_int32(crate::IntegerArray::<i32>::from_slice(vals));
        let field = Field::new(name, ArrowType::Int32, false, None);
        FieldArray::new(field, arr)
    }

    #[test]
    fn test_is_empty_and_n_pieces() {
        let f = Arc::new(Field::new("col", ArrowType::Int32, false, None));
        let empty = SuperArrayV {
            slices: Vec::new(),
            len: 0,
            field: f.clone(),
        };
        assert!(empty.is_empty());
        assert_eq!(empty.n_slices(), 0);

        let arr = Array::from_int32(crate::IntegerArray::<i32>::from_slice(&[1, 2, 3]));
        let non_empty = SuperArrayV {
            slices: Vec::from(vec![ArrayV::new(arr, 0, 3)]),
            len: 3,
            field: f.clone(),
        };
        assert!(!non_empty.is_empty());
        assert_eq!(non_empty.n_slices(), 1);
    }

    #[test]
    fn test_to_array_materialises_correctly() {
        let fa1 = fa("x", &[1, 2, 3]);
        let fa2 = fa("x", &[4, 5]);
        let ca = SuperArray::from_chunks(Vec::from(vec![fa1.clone(), fa2.clone()]));
        let slice = ca.slice(0, 5);

        let arr = slice.consolidate();
        if let Array::NumericArray(NumericArray::Int32(ints)) = arr {
            assert_eq!(ints.data.as_slice(), &[1, 2, 3, 4, 5]);
        } else {
            panic!("unexpected type");
        }
    }

    #[test]
    fn test_slice_subslice() {
        let fa1 = fa("x", &[1, 2, 3]);
        let fa2 = fa("x", &[4, 5, 6, 7]);
        let ca = SuperArray::from_chunks(Vec::from(vec![fa1.clone(), fa2.clone()]));
        let slice = ca.slice(1, 5); // [2,3,4,5,6]
        let sub = slice.slice(1, 3); // [3,4,5]
        let arr = sub.consolidate();
        if let Array::NumericArray(NumericArray::Int32(ints)) = arr {
            assert_eq!(ints.data.as_slice(), &[3, 4, 5]);
        } else {
            panic!("unexpected type");
        }
    }

    #[test]
    fn test_chunks_and_iter() {
        let fa1 = fa("y", &[10, 20]);
        let fa2 = fa("y", &[30]);
        let ca = SuperArray::from_chunks(Vec::from(vec![fa1.clone(), fa2.clone()]));
        let slice = ca.slice(0, 3);

        let collected: Vec<_> = slice.chunks().map(|c| c.as_tuple()).collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].2, 2);
        assert_eq!(collected[1].2, 1);

        let collected2: Vec<_> = slice.iter().map(|c| c.as_tuple()).collect();
        assert_eq!(collected2.len(), 2);
        assert_eq!(collected2[0].2, 2);
    }

    #[test]
    fn test_get_array_and_row_slice() {
        let fa1 = fa("z", &[7, 8]);
        let fa2 = fa("z", &[9]);
        let ca = SuperArray::from_chunks(Vec::from(vec![fa1.clone(), fa2.clone()]));
        let slice = ca.slice(0, 3);

        let arr = slice.get_value(1); // Should be 8
        if let Array::NumericArray(NumericArray::Int32(ints)) = arr {
            assert_eq!(ints.data.as_slice(), &[8]);
        } else {
            panic!("unexpected type");
        }

        let arr2 = slice.get_value(2); // Should be 9
        if let Array::NumericArray(NumericArray::Int32(ints)) = arr2 {
            assert_eq!(ints.data.as_slice(), &[9]);
        } else {
            panic!("unexpected type");
        }

        let row = slice.row_slice(2).as_tuple();
        assert_eq!(row.2, 1);
        let arr3 = row.0.slice_clone(row.1, row.2);
        if let Array::NumericArray(NumericArray::Int32(ints)) = arr3 {
            assert_eq!(ints.data.as_slice(), &[9]);
        } else {
            panic!("unexpected type");
        }
    }

    #[test]
    fn test_iter_rows_unified() {
        let fa1 = fa("w", &[1, 2]);
        let fa2 = fa("w", &[3]);
        let ca = SuperArray::from_chunks(Vec::from(vec![fa1.clone(), fa2.clone()]));
        let slice = ca.slice(0, 3);

        let rows: Vec<_> = slice.iter_rows().collect();
        assert_eq!(rows.len(), 3);

        let vals: Vec<i32> = rows
            .iter()
            .map(|s| {
                let s = s;
                if let Array::NumericArray(NumericArray::Int32(ints)) = s.0.slice_clone(s.1, s.2) {
                    ints.data[0]
                } else {
                    panic!("not i32")
                }
            })
            .collect();
        assert_eq!(vals, vec![1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_get_array_oob_panics() {
        let fa1 = fa("a", &[1]);
        let ca = SuperArray::from_chunks(Vec::from(vec![fa1.clone()]));
        let slice = ca.slice(0, 1);
        // Should panic
        slice.get_value(5);
    }

    #[test]
    fn test_field_propagation() {
        let fa1 = fa("field", &[1, 2, 3]);
        let ca = SuperArray::from_chunks(Vec::from(vec![fa1.clone()]));
        let slice = ca.slice(0, 3);
        assert_eq!(slice.field.name, "field");
        let subslice = slice.slice(1, 2);
        assert_eq!(subslice.field.name, "field");
    }

    // Boolean Array Consolidation Tests

    fn fa_bool(name: &str, vals: &[bool]) -> FieldArray {
        let arr = Array::BooleanArray(Arc::new(crate::BooleanArray::from_slice(vals)));
        let field = Field::new(name, ArrowType::Boolean, false, None);
        FieldArray::new(field, arr)
    }

    #[test]
    fn test_consolidate_boolean_single_chunk() {
        let fa1 = fa_bool("b", &[true, false, true, false]);
        let ca = SuperArray::from_chunks(vec![fa1]);
        let slice = ca.slice(0, 4);
        let arr = slice.consolidate();

        // Use arr.len() which correctly returns logical element count
        assert_eq!(arr.len(), 4);
        if let Array::BooleanArray(bools) = arr {
            // Note: bools.len() returns byte count due to Deref<Target=[u8]>
            // Use bools.data.len() for logical bit count
            assert_eq!(bools.data.len(), 4);
            assert_eq!(bools.data.get(0), true);
            assert_eq!(bools.data.get(1), false);
            assert_eq!(bools.data.get(2), true);
            assert_eq!(bools.data.get(3), false);
        } else {
            panic!("Expected BooleanArray");
        }
    }

    #[test]
    fn test_consolidate_boolean_multiple_chunks() {
        let fa1 = fa_bool("b", &[true, true]);
        let fa2 = fa_bool("b", &[false, false, true]);
        let ca = SuperArray::from_chunks(vec![fa1, fa2]);
        let slice = ca.slice(0, 5);
        let arr = slice.consolidate();

        assert_eq!(arr.len(), 5);
        if let Array::BooleanArray(bools) = arr {
            assert_eq!(bools.data.len(), 5);
            assert_eq!(bools.data.get(0), true);
            assert_eq!(bools.data.get(1), true);
            assert_eq!(bools.data.get(2), false);
            assert_eq!(bools.data.get(3), false);
            assert_eq!(bools.data.get(4), true);
        } else {
            panic!("Expected BooleanArray");
        }
    }

    #[test]
    fn test_consolidate_boolean_with_offset() {
        let fa1 = fa_bool("b", &[true, false, true, false, true]);
        let ca = SuperArray::from_chunks(vec![fa1]);
        let slice = ca.slice(1, 3); // [false, true, false]
        let arr = slice.consolidate();

        assert_eq!(arr.len(), 3);
        if let Array::BooleanArray(bools) = arr {
            assert_eq!(bools.data.len(), 3);
            assert_eq!(bools.data.get(0), false);
            assert_eq!(bools.data.get(1), true);
            assert_eq!(bools.data.get(2), false);
        } else {
            panic!("Expected BooleanArray");
        }
    }

    // String Array Consolidation Tests

    fn fa_string(name: &str, vals: &[&str]) -> FieldArray {
        let arr = Array::from_string32(crate::StringArray::<u32>::from_slice(vals));
        let field = Field::new(name, ArrowType::String, false, None);
        FieldArray::new(field, arr)
    }

    #[test]
    fn test_consolidate_string_single_chunk() {
        let fa1 = fa_string("s", &["hello", "world", "test"]);
        let ca = SuperArray::from_chunks(vec![fa1]);
        let slice = ca.slice(0, 3);
        let arr = slice.consolidate();

        // Use arr.len() which correctly returns string count
        assert_eq!(arr.len(), 3);
        if let Array::TextArray(crate::TextArray::String32(strings)) = arr {
            // Note: strings.len() returns data buffer length due to Deref<Target=[u8]>
            // Use offsets.len()-1 for actual string count
            assert_eq!(strings.offsets.len() - 1, 3);
            assert_eq!(strings.get_str(0), Some("hello"));
            assert_eq!(strings.get_str(1), Some("world"));
            assert_eq!(strings.get_str(2), Some("test"));
        } else {
            panic!("Expected String32 Array");
        }
    }

    #[test]
    fn test_consolidate_string_multiple_chunks() {
        let fa1 = fa_string("s", &["alpha", "beta"]);
        let fa2 = fa_string("s", &["gamma", "delta"]);
        let ca = SuperArray::from_chunks(vec![fa1, fa2]);
        let slice = ca.slice(0, 4);
        let arr = slice.consolidate();

        assert_eq!(arr.len(), 4);
        if let Array::TextArray(crate::TextArray::String32(strings)) = arr {
            assert_eq!(strings.offsets.len() - 1, 4);
            assert_eq!(strings.get_str(0), Some("alpha"));
            assert_eq!(strings.get_str(1), Some("beta"));
            assert_eq!(strings.get_str(2), Some("gamma"));
            assert_eq!(strings.get_str(3), Some("delta"));
        } else {
            panic!("Expected String32 Array");
        }
    }

    #[test]
    fn test_consolidate_string_with_offset() {
        let fa1 = fa_string("s", &["one", "two", "three", "four"]);
        let ca = SuperArray::from_chunks(vec![fa1]);
        let slice = ca.slice(1, 2); // ["two", "three"]
        let arr = slice.consolidate();

        assert_eq!(arr.len(), 2);
        if let Array::TextArray(crate::TextArray::String32(strings)) = arr {
            assert_eq!(strings.offsets.len() - 1, 2);
            assert_eq!(strings.get_str(0), Some("two"));
            assert_eq!(strings.get_str(1), Some("three"));
        } else {
            panic!("Expected String32 Array");
        }
    }

    #[test]
    fn test_consolidate_string_cross_chunk_slice() {
        let fa1 = fa_string("s", &["a", "bb"]);
        let fa2 = fa_string("s", &["ccc", "dddd"]);
        let ca = SuperArray::from_chunks(vec![fa1, fa2]);
        let slice = ca.slice(1, 2); // ["bb", "ccc"] spanning chunks
        let arr = slice.consolidate();

        assert_eq!(arr.len(), 2);
        if let Array::TextArray(crate::TextArray::String32(strings)) = arr {
            assert_eq!(strings.offsets.len() - 1, 2);
            assert_eq!(strings.get_str(0), Some("bb"));
            assert_eq!(strings.get_str(1), Some("ccc"));
        } else {
            panic!("Expected String32 Array");
        }
    }

    // Numeric Array Consolidation Tests (Int32, Int64, Float32, Float64)

    fn fa_i64(name: &str, vals: &[i64]) -> FieldArray {
        let arr = Array::from_int64(crate::IntegerArray::<i64>::from_slice(vals));
        let field = Field::new(name, ArrowType::Int64, false, None);
        FieldArray::new(field, arr)
    }

    fn fa_f32(name: &str, vals: &[f32]) -> FieldArray {
        let arr = Array::from_float32(crate::FloatArray::<f32>::from_slice(vals));
        let field = Field::new(name, ArrowType::Float32, false, None);
        FieldArray::new(field, arr)
    }

    fn fa_f64(name: &str, vals: &[f64]) -> FieldArray {
        let arr = Array::from_float64(crate::FloatArray::<f64>::from_slice(vals));
        let field = Field::new(name, ArrowType::Float64, false, None);
        FieldArray::new(field, arr)
    }

    #[test]
    fn test_consolidate_int32_single_chunk() {
        let fa1 = fa("i", &[10, 20, 30, 40]);
        let ca = SuperArray::from_chunks(vec![fa1]);
        let slice = ca.slice(0, 4);
        let arr = slice.consolidate();

        assert_eq!(arr.len(), 4);
        if let Array::NumericArray(NumericArray::Int32(ints)) = arr {
            assert_eq!(ints.data.as_slice(), &[10, 20, 30, 40]);
        } else {
            panic!("Expected Int32 Array");
        }
    }

    #[test]
    fn test_consolidate_int32_multiple_chunks() {
        let fa1 = fa("i", &[1, 2]);
        let fa2 = fa("i", &[3, 4, 5]);
        let ca = SuperArray::from_chunks(vec![fa1, fa2]);
        let slice = ca.slice(0, 5);
        let arr = slice.consolidate();

        assert_eq!(arr.len(), 5);
        if let Array::NumericArray(NumericArray::Int32(ints)) = arr {
            assert_eq!(ints.data.as_slice(), &[1, 2, 3, 4, 5]);
        } else {
            panic!("Expected Int32 Array");
        }
    }

    #[test]
    fn test_consolidate_int32_with_offset() {
        let fa1 = fa("i", &[100, 200, 300, 400, 500]);
        let ca = SuperArray::from_chunks(vec![fa1]);
        let slice = ca.slice(1, 3); // [200, 300, 400]
        let arr = slice.consolidate();

        assert_eq!(arr.len(), 3);
        if let Array::NumericArray(NumericArray::Int32(ints)) = arr {
            assert_eq!(ints.data.as_slice(), &[200, 300, 400]);
        } else {
            panic!("Expected Int32 Array");
        }
    }

    #[test]
    fn test_consolidate_int64_multiple_chunks() {
        let fa1 = fa_i64("i", &[1_000_000_000i64, 2_000_000_000]);
        let fa2 = fa_i64("i", &[3_000_000_000i64]);
        let ca = SuperArray::from_chunks(vec![fa1, fa2]);
        let slice = ca.slice(0, 3);
        let arr = slice.consolidate();

        assert_eq!(arr.len(), 3);
        if let Array::NumericArray(NumericArray::Int64(ints)) = arr {
            assert_eq!(
                ints.data.as_slice(),
                &[1_000_000_000i64, 2_000_000_000, 3_000_000_000]
            );
        } else {
            panic!("Expected Int64 Array");
        }
    }

    #[test]
    fn test_consolidate_float32_multiple_chunks() {
        let fa1 = fa_f32("f", &[1.5f32, 2.5]);
        let fa2 = fa_f32("f", &[3.5f32, 4.5]);
        let ca = SuperArray::from_chunks(vec![fa1, fa2]);
        let slice = ca.slice(0, 4);
        let arr = slice.consolidate();

        assert_eq!(arr.len(), 4);
        if let Array::NumericArray(NumericArray::Float32(floats)) = arr {
            assert_eq!(floats.data.as_slice(), &[1.5f32, 2.5, 3.5, 4.5]);
        } else {
            panic!("Expected Float32 Array");
        }
    }

    #[test]
    fn test_consolidate_float64_with_offset() {
        let fa1 = fa_f64("f", &[0.1, 0.2, 0.3, 0.4, 0.5]);
        let ca = SuperArray::from_chunks(vec![fa1]);
        let slice = ca.slice(2, 2); // [0.3, 0.4]
        let arr = slice.consolidate();

        assert_eq!(arr.len(), 2);
        if let Array::NumericArray(NumericArray::Float64(floats)) = arr {
            assert_eq!(floats.data.as_slice(), &[0.3, 0.4]);
        } else {
            panic!("Expected Float64 Array");
        }
    }

    #[test]
    fn test_consolidate_float64_cross_chunk_slice() {
        let fa1 = fa_f64("f", &[1.1, 2.2]);
        let fa2 = fa_f64("f", &[3.3, 4.4]);
        let ca = SuperArray::from_chunks(vec![fa1, fa2]);
        let slice = ca.slice(1, 2); // [2.2, 3.3] spanning chunks
        let arr = slice.consolidate();

        assert_eq!(arr.len(), 2);
        if let Array::NumericArray(NumericArray::Float64(floats)) = arr {
            assert_eq!(floats.data.as_slice(), &[2.2, 3.3]);
        } else {
            panic!("Expected Float64 Array");
        }
    }

    // Temporal Array Consolidation Tests

    #[cfg(feature = "datetime")]
    fn fa_datetime64(name: &str, vals: &[i64]) -> FieldArray {
        use crate::enums::time_units::TimeUnit;
        use crate::traits::masked_array::MaskedArray;
        let mut arr = crate::DatetimeArray::<i64>::with_capacity(
            vals.len(),
            false,
            Some(TimeUnit::Milliseconds),
        );
        for &v in vals {
            arr.push(v);
        }
        let arr = Array::from_datetime_i64(arr);
        let field = Field::new(
            name,
            ArrowType::Timestamp(TimeUnit::Milliseconds, None),
            false,
            None,
        );
        FieldArray::new(field, arr)
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn test_consolidate_datetime64_single_chunk() {
        let fa1 = fa_datetime64("ts", &[1000, 2000, 3000, 4000]);
        let ca = SuperArray::from_chunks(vec![fa1]);
        let slice = ca.slice(0, 4);
        let arr = slice.consolidate();

        assert_eq!(arr.len(), 4);
        if let Array::TemporalArray(crate::TemporalArray::Datetime64(dt)) = arr {
            assert_eq!(dt.data.as_slice(), &[1000i64, 2000, 3000, 4000]);
        } else {
            panic!("Expected Datetime64 Array");
        }
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn test_consolidate_datetime64_multiple_chunks() {
        let fa1 = fa_datetime64("ts", &[100, 200]);
        let fa2 = fa_datetime64("ts", &[300, 400, 500]);
        let ca = SuperArray::from_chunks(vec![fa1, fa2]);
        let slice = ca.slice(0, 5);
        let arr = slice.consolidate();

        assert_eq!(arr.len(), 5);
        if let Array::TemporalArray(crate::TemporalArray::Datetime64(dt)) = arr {
            assert_eq!(dt.data.as_slice(), &[100i64, 200, 300, 400, 500]);
        } else {
            panic!("Expected Datetime64 Array");
        }
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn test_consolidate_datetime64_with_offset() {
        let fa1 = fa_datetime64("ts", &[10, 20, 30, 40, 50]);
        let ca = SuperArray::from_chunks(vec![fa1]);
        let slice = ca.slice(1, 3); // [20, 30, 40]
        let arr = slice.consolidate();

        assert_eq!(arr.len(), 3);
        if let Array::TemporalArray(crate::TemporalArray::Datetime64(dt)) = arr {
            assert_eq!(dt.data.as_slice(), &[20i64, 30, 40]);
        } else {
            panic!("Expected Datetime64 Array");
        }
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn test_consolidate_datetime64_cross_chunk_slice() {
        let fa1 = fa_datetime64("ts", &[1, 2]);
        let fa2 = fa_datetime64("ts", &[3, 4]);
        let ca = SuperArray::from_chunks(vec![fa1, fa2]);
        let slice = ca.slice(1, 2); // [2, 3] spanning chunks
        let arr = slice.consolidate();

        assert_eq!(arr.len(), 2);
        if let Array::TemporalArray(crate::TemporalArray::Datetime64(dt)) = arr {
            assert_eq!(dt.data.as_slice(), &[2i64, 3]);
        } else {
            panic!("Expected Datetime64 Array");
        }
    }

    // Categorical Array Consolidation Tests

    #[cfg(any(
        not(feature = "default_categorical_8"),
        feature = "extended_categorical"
    ))]
    fn fa_categorical(name: &str, vals: &[&str]) -> FieldArray {
        use crate::ffi::arrow_dtype::CategoricalIndexType;
        let string_arr = crate::StringArray::<u32>::from_slice(vals);
        let cat_arr = string_arr.to_categorical_array();
        let arr = Array::from_categorical32(cat_arr);
        let field = Field::new(
            name,
            ArrowType::Dictionary(CategoricalIndexType::UInt32),
            false,
            None,
        );
        FieldArray::new(field, arr)
    }

    #[cfg(any(
        not(feature = "default_categorical_8"),
        feature = "extended_categorical"
    ))]
    #[test]
    fn test_consolidate_categorical_single_chunk() {
        let fa1 = fa_categorical("cat", &["a", "b", "a", "c"]);
        let ca = SuperArray::from_chunks(vec![fa1]);
        let slice = ca.slice(0, 4);
        let arr = slice.consolidate();

        assert_eq!(arr.len(), 4);
        if let Array::TextArray(crate::TextArray::Categorical32(cat)) = arr {
            assert_eq!(cat.len(), 4);
            // Values should decode correctly
            assert_eq!(cat.get_str(0), Some("a"));
            assert_eq!(cat.get_str(1), Some("b"));
            assert_eq!(cat.get_str(2), Some("a"));
            assert_eq!(cat.get_str(3), Some("c"));
        } else {
            panic!("Expected Categorical32 Array");
        }
    }

    #[cfg(any(
        not(feature = "default_categorical_8"),
        feature = "extended_categorical"
    ))]
    #[test]
    fn test_consolidate_categorical_with_offset() {
        let fa1 = fa_categorical("cat", &["x", "y", "z", "w", "v"]);
        let ca = SuperArray::from_chunks(vec![fa1]);
        let slice = ca.slice(1, 3); // ["y", "z", "w"]
        let arr = slice.consolidate();

        assert_eq!(arr.len(), 3);
        if let Array::TextArray(crate::TextArray::Categorical32(cat)) = arr {
            assert_eq!(cat.len(), 3);
            assert_eq!(cat.get_str(0), Some("y"));
            assert_eq!(cat.get_str(1), Some("z"));
            assert_eq!(cat.get_str(2), Some("w"));
        } else {
            panic!("Expected Categorical32 Array");
        }
    }

    #[cfg(any(
        not(feature = "default_categorical_8"),
        feature = "extended_categorical"
    ))]
    #[test]
    fn test_consolidate_categorical_same_dict_multiple_chunks() {
        use crate::ffi::arrow_dtype::CategoricalIndexType;
        // Create two chunks from the same source so they share dictionary
        let string_arr = crate::StringArray::<u32>::from_slice(&["red", "green", "blue", "red"]);
        let cat_arr = Arc::new(string_arr.to_categorical_array());

        // Create two FieldArrays pointing to slices of the same categorical
        let field = Field::new(
            "color",
            ArrowType::Dictionary(CategoricalIndexType::UInt32),
            false,
            None,
        );

        // First chunk: first 2 elements
        let arr1 = Array::TextArray(crate::TextArray::Categorical32(cat_arr.clone()));
        let fa1 = FieldArray::new(field.clone(), arr1);

        // Second chunk: last 2 elements (same array, will be sliced)
        let arr2 = Array::TextArray(crate::TextArray::Categorical32(cat_arr.clone()));
        let fa2 = FieldArray::new(field.clone(), arr2);

        let ca = SuperArray::from_chunks(vec![fa1, fa2]);
        // Slice spanning both chunks
        let slice = ca.slice(1, 4); // ["green", "blue", "red", "red"]
        let arr = slice.consolidate();

        assert_eq!(arr.len(), 4);
        if let Array::TextArray(crate::TextArray::Categorical32(cat)) = arr {
            assert_eq!(cat.get_str(0), Some("green"));
            assert_eq!(cat.get_str(1), Some("blue"));
            assert_eq!(cat.get_str(2), Some("red"));
            assert_eq!(cat.get_str(3), Some("red"));
        } else {
            panic!("Expected Categorical32 Array");
        }
    }
}

/// SuperArray -> SuperArrayV conversion
impl From<SuperArray> for SuperArrayV {
    fn from(super_array: SuperArray) -> Self {
        let len = super_array.len();

        // Get field from SuperArray or synthesise from first chunk
        let field = if let Some(f) = super_array.field.clone() {
            f
        } else if let Some(chunk) = super_array.chunks.first() {
            Arc::new(Field::new(
                "data",
                chunk.arrow_type(),
                chunk.is_nullable(),
                None,
            ))
        } else {
            panic!("Cannot convert empty SuperArray with no field to SuperArrayV")
        };

        let slices: Vec<ArrayV> = super_array.chunks.into_iter().map(ArrayV::from).collect();

        SuperArrayV { slices, len, field }
    }
}
