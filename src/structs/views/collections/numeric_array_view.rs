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

//! # **NumericArrayView Module** - *Windowed View over a NumericArray*
//!
//! `NumericArrayV` is a **read-only, windowed view** over a [`NumericArray`].
//! It groups all integer and float variants and exposes a zero-copy slice
//! `[offset .. offset + len)` for fast, indexable access.
//!
//! ## Role
//! - Lets APIs accept either a full `NumericArray` or a pre-sliced view.
//! - Avoids deep copies while enabling per-window operations and previews.
//! - Can cache per-window null counts to speed up repeated scans.
//!
//! ## Behaviour
//! - Works across numeric variants (ints + floats) behind `NumericArray`.
//! - Provides convenience accessors like [`get_f64`](NumericArrayV::get_f64) that
//!   upcast to `f64` for uniform downstream handling.
//! - Slicing returns another borrowed view; data buffers are not cloned.
//!
//! ## Threading
//! - Thread-safe for sharing across threads (uses `OnceLock` for null count caching).
//! - Safe to share via `Arc` for parallel processing.
//!
//! ## Interop
//! - Convert to an owned `NumericArray` of the window via
//!   [`to_numeric_array`](NumericArrayV::to_numeric_array).
//! - Lift to `Array` with [`inner_array`](NumericArrayV::inner_array) when you need
//!   enum-level APIs.
//!
//! ## Invariants
//! - `offset + len <= array.len()`
//! - `len` is the logical row count of this view.

use std::fmt::{self, Debug, Display, Formatter};
use std::sync::OnceLock;

use crate::enums::error::MinarrowError;
use crate::enums::shape_dim::ShapeDim;
use crate::structs::views::bitmask_view::BitmaskV;
use crate::traits::concatenate::Concatenate;
use crate::traits::print::MAX_PREVIEW;
use crate::traits::shape::Shape;
use crate::{Array, ArrayV, Bitmask, FieldArray, MaskedArray, NumericArray};

/// # NumericArrayView
///
/// Read-only, zero-copy view over a `[offset .. offset + len)` window of a
/// [`NumericArray`].
///
/// ## Purpose
/// - Return an indexable subrange without cloning buffers.
/// - Optionally cache per-window null counts for faster repeated passes.
///
/// ## Behaviour
/// - Groups integer and float variants under one enum.
/// - Upcasts via [`get_f64`](Self::get_f64) for uniform handling.
/// - Further slicing yields another borrowed view.
///
/// ## Fields
/// - `array`: backing [`NumericArray`] (enum over numeric types).
/// - `offset`: starting index into the backing array.
/// - `len`: logical number of elements in the view.
/// - `null_count`: cached `Option<usize>` for this window (internal).
///
/// ## Notes
/// - Not thread-safe due to `Cell`. Create per-thread views with [`slice`](Self::slice).
/// - Use [`to_numeric_array`](Self::to_numeric_array) to materialise the window.
#[derive(Clone, PartialEq)]
pub struct NumericArrayV {
    pub array: NumericArray,
    pub offset: usize,
    len: usize,
    null_count: OnceLock<usize>,
}

impl NumericArrayV {
    /// Creates a new `NumericArrayView` with the given offset and length.
    pub fn new(array: NumericArray, offset: usize, len: usize) -> Self {
        assert!(
            offset + len <= array.len(),
            "NumericArrayView: window out of bounds (offset + len = {}, array.len = {})",
            offset + len,
            array.len()
        );
        Self {
            array,
            offset,
            len,
            null_count: OnceLock::new(),
        }
    }

    /// Creates a new `NumericArrayView` with a precomputed null count.
    pub fn new_nc(
        array: NumericArray,
        offset: usize,
        len: usize,
        null_count: usize,
    ) -> Self {
        assert!(
            offset + len <= array.len(),
            "NumericArrayView: window out of bounds (offset + len = {}, array.len = {})",
            offset + len,
            array.len()
        );
        let lock = OnceLock::new();
        let _ = lock.set(null_count); // Pre-initialise with the provided count
        Self {
            array,
            offset,
            len,
            null_count: lock,
        }
    }

    /// Returns `true` if the view is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the full backing array wrapped as an `Array` enum, ignoring the view's offset and length.
    ///
    /// Use this to access inner array methods. The returned array is the unwindowed original.
    #[inline]
    pub fn inner_array(&self) -> Array {
        Array::NumericArray(self.array.clone()) // Arc clone for data buffer
    }

    /// Returns the value at logical index `i` as `f64`, or `None` if out of bounds or null.
    ///
    /// Converts any numeric types to `f64`, simplifying usage by avoiding explicit
    /// enum matches in caller code.
    ///
    /// # Notes
    /// - Returns `None` if `i` is out of bounds or the value is null.
    /// - Upcasts integer and float types to `f64` for uniform downstream handling.
    #[inline]
    pub fn get_f64(&self, i: usize) -> Option<f64> {
        if i >= self.len {
            return None;
        }
        let phys_idx = self.offset + i;
        match &self.array {
            NumericArray::Int32(arr) => arr.get(phys_idx).map(|v| v as f64),
            NumericArray::Int64(arr) => arr.get(phys_idx).map(|v| v as f64),
            NumericArray::UInt32(arr) => arr.get(phys_idx).map(|v| v as f64),
            NumericArray::UInt64(arr) => arr.get(phys_idx).map(|v| v as f64),
            NumericArray::Float32(arr) => arr.get(phys_idx).map(|v| v as f64),
            NumericArray::Float64(arr) => arr.get(phys_idx),
            NumericArray::Null => None,
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(arr) => arr.get(phys_idx).map(|v| v as f64),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(arr) => arr.get(phys_idx).map(|v| v as f64),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(arr) => arr.get(phys_idx).map(|v| v as f64),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(arr) => arr.get(phys_idx).map(|v| v as f64),
        }
    }

    /// Unchecked, returns None for nulls, skips bounds check.
    ///
    /// Converts any numeric types to `f64`, simplifying usage by avoiding explicit
    /// enum matches in caller code.
    #[inline]
    pub unsafe fn get_f64_unchecked(&self, i: usize) -> Option<f64> {
        let phys_idx = self.offset + i;
        match &self.array {
            NumericArray::Int32(arr) => unsafe { arr.get_unchecked(phys_idx) }.map(|v| v as f64),
            NumericArray::Int64(arr) => unsafe { arr.get_unchecked(phys_idx) }.map(|v| v as f64),
            NumericArray::UInt32(arr) => unsafe { arr.get_unchecked(phys_idx) }.map(|v| v as f64),
            NumericArray::UInt64(arr) => unsafe { arr.get_unchecked(phys_idx) }.map(|v| v as f64),
            NumericArray::Float32(arr) => unsafe { arr.get_unchecked(phys_idx) }.map(|v| v as f64),
            NumericArray::Float64(arr) => unsafe { arr.get_unchecked(phys_idx) },
            NumericArray::Null => None,
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(arr) => unsafe { arr.get_unchecked(phys_idx) }.map(|v| v as f64),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(arr) => unsafe { arr.get_unchecked(phys_idx) }.map(|v| v as f64),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(arr) => unsafe { arr.get_unchecked(phys_idx) }.map(|v| v as f64),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(arr) => unsafe { arr.get_unchecked(phys_idx) }.map(|v| v as f64),
        }
    }

    /// Returns a windowed view into a sub-range of this view.
    #[inline]
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        assert!(
            offset + len <= self.len,
            "NumericArrayView::slice: out of bounds"
        );
        Self {
            array: self.array.clone(),
            offset: self.offset + offset,
            len,
            null_count: OnceLock::new(),
        }
    }

    /// Materialise as an owned `NumericArray` for the window.
    ///
    /// If the view covers the entire backing array, returns a cheap Arc clone
    /// of the original variant. Otherwise deep-copies the window via
    /// `slice_clone` through `inner_array`.
    pub fn to_numeric_array(&self) -> NumericArray {
        if self.offset == 0 && self.len == self.array.len() {
            return self.array.clone();
        }
        self.inner_array().slice_clone(self.offset, self.len).num()
    }

    /// Returns the end index of the view.
    #[inline]
    pub fn end(&self) -> usize {
        self.offset + self.len
    }

    /// Returns the view as a tuple `(array, offset, len)`.
    ///
    /// Note: This clones the Arc-wrapped NumericArray.
    #[inline]
    pub fn as_tuple(&self) -> (NumericArray, usize, usize) {
        (self.array.clone(), self.offset, self.len)
    }

    /// Returns a reference tuple: `(&NumericArray, offset, len)`.
    ///
    /// This avoids cloning the Arc and returns a reference with a lifetime
    /// tied to this NumericArrayV.
    #[inline]
    pub fn as_tuple_ref(&self) -> (&NumericArray, usize, usize) {
        (&self.array, self.offset, self.len)
    }

    /// Returns the length of the window
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the number of nulls in the view.
    #[inline]
    pub fn null_count(&self) -> usize {
        *self
            .null_count
            .get_or_init(|| match self.array.null_mask() {
                Some(mask) => mask.view(self.offset, self.len).count_zeros(),
                None => 0,
            })
    }

    /// Returns true when the windowed view holds at least one null.
    ///
    /// Reads through `null_count`, so the cached value is trusted when set
    /// and the full popcount is only paid on the first call that observes
    /// this view.
    #[inline]
    pub fn has_nulls(&self) -> bool {
        self.null_count() > 0
    }

    /// Returns the null mask as a windowed `BitmaskView`.
    #[inline]
    pub fn null_mask_view(&self) -> Option<BitmaskV<'_>> {
        self.array
            .null_mask()
            .map(|mask| mask.view(self.offset, self.len))
    }

    /// Sets the cached null count for the view.
    ///
    /// Returns Ok(()) if the value was set, or Err(count) if it was already initialised.
    /// This is thread-safe and can only succeed once per NumericArrayV instance.
    #[inline]
    pub fn set_null_count(&self, count: usize) -> Result<(), usize> {
        self.null_count.set(count).map_err(|_| count)
    }

    /// Guarantees the backing array is Float64, then returns the f64 slice,
    /// null mask, and null count for this view's window.
    ///
    /// **If already Float64, this is a pass-through.** Otherwise the full backing
    /// NumericArray is cast to Float64 via [`NumericArray::cow_into_f64`],
    /// preserving the window offset and length.
    ///
    /// When multiple views share the same backing array, the first view to
    /// call this will trigger the cast. If it holds the sole Arc reference,
    /// the old data is consumed in place. If other references exist, the data
    /// is cloned, leaving the shared original untouched. Subsequent views
    /// that still reference the original will cast independently when they
    /// reach this call, so it generally is best avoided in such contexts as it would
    /// clone for every independent window view.
    pub fn guarantee_f64(&mut self) -> (&[f64], Option<&Bitmask>, Option<usize>) {
        if !matches!(&self.array, NumericArray::Float64(_)) {
            // Take the old array out, leaving Null as placeholder
            let old = std::mem::take(&mut self.array);
            self.array = old.cow_into_f64();
        }
        // Safe: the branch above guarantees Float64 at this point
        let NumericArray::Float64(arr) = &self.array else { unreachable!() };
        let slice = &arr.data.as_slice()[self.offset..self.offset + self.len];
        let mask = arr.null_mask.as_ref();
        let nc = if mask.is_some() {
            Some(self.null_count())
        } else {
            None
        };
        (slice, mask, nc)
    }

}

impl From<NumericArray> for NumericArrayV {
    fn from(array: NumericArray) -> Self {
        let len = array.len();
        NumericArrayV {
            array,
            offset: 0,
            len,
            null_count: OnceLock::new(),
        }
    }
}

impl From<FieldArray> for NumericArrayV {
    fn from(field_array: FieldArray) -> Self {
        match field_array.array {
            Array::NumericArray(arr) => {
                let len = arr.len();
                NumericArrayV {
                    array: arr,
                    offset: 0,
                    len,
                    null_count: OnceLock::new(),
                }
            }
            _ => panic!("FieldArray does not contain a NumericArray"),
        }
    }
}

impl From<Array> for NumericArrayV {
    fn from(array: Array) -> Self {
        match array {
            Array::NumericArray(arr) => {
                let len = arr.len();

                NumericArrayV {
                    array: arr,
                    offset: 0,
                    len,
                    null_count: OnceLock::new(),
                }
            }
            _ => panic!("Array is not a NumericArray"),
        }
    }
}

impl From<ArrayV> for NumericArrayV {
    fn from(view: ArrayV) -> Self {
        let (array, offset, len) = view.as_tuple();
        match array {
            Array::NumericArray(inner) => Self {
                array: inner,
                offset,
                len,
                null_count: OnceLock::new(),
            },
            _ => panic!("From<ArrayView>: expected NumericArray variant"),
        }
    }
}

impl Debug for NumericArrayV {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("NumericArrayView")
            .field("offset", &self.offset)
            .field("len", &self.len)
            .field("array", &self.array)
            .field("cached_null_count", &self.null_count.get())
            .finish()
    }
}

impl Shape for NumericArrayV {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank1(self.len())
    }
}

impl Concatenate for NumericArrayV {
    /// Concatenates two numeric array views by materialising both to owned numeric arrays,
    /// concatenating them, and wrapping the result back in a view.
    ///
    /// # Notes
    /// - This operation copies data from both views to create owned numeric arrays.
    /// - The resulting view has offset=0 and length equal to the combined length.
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        // Materialise both views to owned numeric arrays
        let self_array = self.to_numeric_array();
        let other_array = other.to_numeric_array();

        // Concatenate the owned numeric arrays
        let concatenated = self_array.concat(other_array)?;

        // Wrap the result in a new view
        Ok(NumericArrayV::from(concatenated))
    }
}

impl Display for NumericArrayV {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let dtype = match &self.array {
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
        };

        writeln!(
            f,
            "NumericArrayView<{dtype}> [{} rows] (offset: {}, nulls: {})",
            self.len(),
            self.offset,
            self.null_count()
        )?;

        let max = self.len().min(MAX_PREVIEW);
        for i in 0..max {
            match self.get_f64(i) {
                Some(v) => writeln!(f, "  {v}")?,
                None => writeln!(f, "  ·")?,
            }
        }

        if self.len() > MAX_PREVIEW {
            writeln!(f, "  ... ({} more)", self.len() - MAX_PREVIEW)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{Array, Bitmask, IntegerArray, NumericArray, vec64};

    #[test]
    fn test_numeric_array_view_basic_indexing_and_slice() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(100);
        arr.push(200);
        arr.push(300);
        arr.push(400);

        let numeric = NumericArray::Int32(Arc::new(arr));
        let view = NumericArrayV::new(numeric.clone(), 1, 2);

        assert_eq!(view.len(), 2);
        assert_eq!(view.offset, 1);

        // Valid indices
        assert_eq!(view.get_f64(0), Some(200.0));
        assert_eq!(view.get_f64(1), Some(300.0));
        assert_eq!(view.get_f64(2), None);

        // Slicing the view produces the correct sub-window
        let sub = view.slice(1, 1);
        assert_eq!(sub.len(), 1);
        assert_eq!(sub.get_f64(0), Some(300.0));
        assert_eq!(sub.get_f64(1), None);
    }

    #[test]
    fn test_numeric_array_view_null_count_and_cache() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(1);
        arr.push(2);
        arr.push(3);
        arr.push(4);

        // Null mask: only index 2 is null
        let mut mask = Bitmask::new_set_all(4, true);
        mask.set(2, false);
        arr.null_mask = Some(mask);

        let numeric = NumericArray::Int32(Arc::new(arr));
        let view = NumericArrayV::new(numeric.clone(), 0, 4);
        assert_eq!(view.null_count(), 1, "Null count should detect one null");
        // Should use cached value next time
        assert_eq!(view.null_count(), 1);

        // Subwindow which excludes the null
        let view2 = view.slice(0, 2);
        assert_eq!(view2.null_count(), 0);
        // Subwindow which includes only the null
        let view3 = view.slice(2, 2);
        assert_eq!(view3.null_count(), 1);
    }

    #[test]
    fn test_numeric_array_view_with_supplied_null_count() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(5);
        arr.push(6);

        let numeric = NumericArray::Int32(Arc::new(arr));
        let view = NumericArrayV::new_nc(numeric.clone(), 0, 2, 99);
        // Should always report the supplied cached value
        assert_eq!(view.null_count(), 99);
        // Trying to set again should fail since it's already initialised
        assert!(view.set_null_count(101).is_err());
        // Still returns original value
        assert_eq!(view.null_count(), 99);
    }

    #[test]
    fn test_numeric_array_view_to_numeric_array_and_as_tuple() {
        let mut arr = IntegerArray::<i32>::default();
        for v in 10..20 {
            arr.push(v);
        }
        let numeric = NumericArray::Int32(Arc::new(arr));
        let view = NumericArrayV::new(numeric.clone(), 4, 3);
        let arr2 = view.to_numeric_array();
        // Copy should be [14, 15, 16]
        if let NumericArray::Int32(a2) = arr2 {
            assert_eq!(a2.data, vec64![14, 15, 16]);
        } else {
            panic!("Unexpected variant");
        }

        // as_tuple returns correct metadata
        let tup = view.as_tuple();
        assert_eq!(&tup.0, &numeric);
        assert_eq!(tup.1, 4);
        assert_eq!(tup.2, 3);
    }

    #[test]
    fn test_numeric_array_view_null_mask_view() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(2);
        arr.push(4);
        arr.push(6);

        let mut mask = Bitmask::new_set_all(3, true);
        mask.set(0, false);
        arr.null_mask = Some(mask);

        let numeric = NumericArray::Int32(Arc::new(arr));
        let view = NumericArrayV::new(numeric, 1, 2);
        let mask_view = view.null_mask_view().expect("Should have mask");
        assert_eq!(mask_view.len(), 2);
        // Should map to bits 1 and 2 of original mask
        assert!(mask_view.get(0));
        assert!(mask_view.get(1));
    }

    #[test]
    fn test_numeric_array_view_from_numeric_array_and_array() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(1);
        arr.push(2);

        let numeric = NumericArray::Int32(Arc::new(arr));
        let view_from_numeric = NumericArrayV::from(numeric.clone());
        assert_eq!(view_from_numeric.len(), 2);
        assert_eq!(view_from_numeric.get_f64(0), Some(1.0));

        let array = Array::NumericArray(numeric);
        let view_from_array = NumericArrayV::from(array);
        assert_eq!(view_from_array.len(), 2);
        assert_eq!(view_from_array.get_f64(1), Some(2.0));
    }

    #[test]
    #[should_panic(expected = "Array is not a NumericArray")]
    fn test_numeric_array_view_from_array_panics_on_wrong_variant() {
        let array = Array::Null;
        let _view = NumericArrayV::from(array);
    }

    #[test]
    fn test_to_numeric_array_full_coverage_shares_arc() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(1);
        arr.push(2);
        arr.push(3);
        let src = Arc::new(arr);
        let numeric = NumericArray::Int32(src.clone());

        let view = NumericArrayV::new(numeric, 0, 3);
        let out = view.to_numeric_array();

        match out {
            NumericArray::Int32(out_arc) => assert!(
                Arc::ptr_eq(&src, &out_arc),
                "full-coverage to_numeric_array should share the underlying Arc"
            ),
            _ => panic!("expected Int32 variant"),
        }
    }

    #[test]
    fn test_to_numeric_array_windowed_copies() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(1);
        arr.push(2);
        arr.push(3);
        let src = Arc::new(arr);
        let numeric = NumericArray::Int32(src.clone());

        let view = NumericArrayV::new(numeric, 1, 2);
        let out = view.to_numeric_array();

        match out {
            NumericArray::Int32(out_arc) => assert!(
                !Arc::ptr_eq(&src, &out_arc),
                "windowed to_numeric_array must allocate a fresh buffer"
            ),
            _ => panic!("expected Int32 variant"),
        }
    }
}
