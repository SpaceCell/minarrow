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

//! # **BooleanArrayView Module** - *Windowed View over a BooleanArray*
//!
//! `BooleanArrayV` is a **read-only, zero-copy view** into a `[offset .. offset + len)`
//! slice of a [`BooleanArray`].
//!
//! ## Purpose
//! - Provides windowed access to bit-packed boolean data without copying buffers.
//! - Allows null counts to be cached per view for faster repeated operations.
//!
//! ## Behaviour
//! - Enables lightweight sub-ranges for use in downstream computations, filtering, or previews.
//! - Creating a view does **not** allocate; slicing is zero-copy.
//!
//! ## Threading
//! - Thread-safe for sharing across threads (uses `OnceLock` for null count caching).
//! - Safe to share via `Arc` for parallel processing.
//!
//! ## Interop
//! - Convert back to a full `BooleanArray` via [`to_boolean_array`](BooleanArrayV::to_boolean_array).
//! - Promote to `Array` via [`inner_array`](BooleanArrayV::inner_array) for unified API calls.
//!
//! ## Invariants
//! - `offset + len <= array.len()`
//! - `len` is the logical number of boolean elements in the view.

use std::fmt::{self, Debug, Display, Formatter};
use std::sync::{Arc, OnceLock};

use crate::enums::error::MinarrowError;
use crate::enums::shape_dim::ShapeDim;
use crate::traits::concatenate::Concatenate;
use crate::traits::print::MAX_PREVIEW;
use crate::traits::shape::Shape;
use crate::{Array, ArrayV, BitmaskV, BooleanArray, MaskedArray};

/// # BooleanArrayView
///
/// Borrowed, indexable view into a `[offset .. offset + len)` window of a
/// [`BooleanArray`].
///
/// ## Purpose
/// - Zero-copy access to contiguous bit-packed boolean values.
/// - Allows null counts for the view to be cached to speed up scans.
///
/// ## Fields
/// - `array`: the backing [`BooleanArray`] wrapped in `Arc`.
/// - `offset`: starting index of the view.
/// - `len`: number of logical elements in the view.
/// - `null_count`: cached null count for this range.
///
/// ## Notes
/// - Use [`slice`](Self::slice) to derive smaller views from this one.
/// - Use [`to_boolean_array`](Self::to_boolean_array) to materialise the data.
#[derive(Clone, PartialEq)]
pub struct BooleanArrayV {
    /// The **outer array** that this view is derived from - we retain a reference to it. 
    /// Importantly, this is the ***full array*** - not the *view*, and thus should not be 
    /// accessed as though it were the view subset.
    pub array: Arc<BooleanArray<()>>,
    /// The index offset from 0 that for where this view starts from the outer array
    pub offset: usize,
    /// The length of the array view
    len: usize,
    /// How many nulls are in the BooleanArrayView
    /// At construction, this is None, unless constructed via new_nc. When one uses '.null_count()',
    /// the first time it will calculate it (quickly) using Bitmask popcount, and then from that 
    /// point onwards the null count is a cached value. 
    null_count: OnceLock<usize>,
}

impl BooleanArrayV {
    /// Creates a new `BooleanArrayView` with the given offset and length.
    pub fn new(array: Arc<BooleanArray<()>>, offset: usize, len: usize) -> Self {
        assert!(
            offset + len <= array.len(),
            "BooleanArrayView: window out of bounds (offset + len = {}, array.len = {})",
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

    /// Creates a new `BooleanArrayView` with a precomputed null count.
    pub fn new_nc(
        array: Arc<BooleanArray<()>>,
        offset: usize,
        len: usize,
        null_count: usize,
    ) -> Self {
        assert!(
            offset + len <= array.len(),
            "BooleanArrayView: window out of bounds (offset + len = {}, array.len = {})",
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

    /// Returns the value at logical index `i` as an `Option<bool>`.
    #[inline]
    pub fn get_bool(&self, i: usize) -> Option<bool> {
        if i >= self.len {
            return None;
        }
        self.array.get(self.offset + i)
    }

    /// Returns a sliced view into a subrange of this view.
    #[inline]
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        assert!(
            offset + len <= self.len,
            "BooleanArrayView::slice: out of bounds"
        );
        Self {
            array: self.array.clone(),
            offset: self.offset + offset,
            len,
            null_count: OnceLock::new(),
        }
    }

    /// Returns the length of the window.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the full backing array wrapped as an `Array` enum, ignoring the view's offset and length.
    ///
    /// Use this to access inner array methods. The returned array is the unwindowed original.
    #[inline]
    pub fn inner_array(&self) -> Array {
        Array::BooleanArray(self.array.clone()) // Arc clone for data buffer
    }

    /// Returns an owned `BooleanArray` for the window.
    ///
    /// If the view covers the entire backing array, returns a cheap Arc clone
    /// of the original allocation. Otherwise deep-copies the window via
    /// `slice_clone` through `inner_array`.
    pub fn to_boolean_array(&self) -> Arc<BooleanArray<()>> {
        if self.offset == 0 && self.len == self.array.len() {
            return self.array.clone();
        }
        self.inner_array().slice_clone(self.offset, self.len).bool()
    }

    /// Returns the end index of the view.
    #[inline]
    pub fn end(&self) -> usize {
        self.offset + self.len
    }

    /// Returns the view as a tuple `(array, offset, len)`.
    #[inline]
    pub fn as_tuple(&self) -> (Arc<BooleanArray<()>>, usize, usize) {
        (self.array.clone(), self.offset, self.len)
    }

    /// Returns the number of nulls in the view.
    ///
    /// Caches it after the first calculation.
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
    /// This is thread-safe and can only succeed once per BooleanArrayV instance.
    #[inline]
    pub fn set_null_count(&self, count: usize) -> Result<(), usize> {
        self.null_count.set(count).map_err(|_| count)
    }
}

impl From<Arc<BooleanArray<()>>> for BooleanArrayV {
    fn from(array: Arc<BooleanArray<()>>) -> Self {
        let len = array.len();
        BooleanArrayV {
            array,
            offset: 0,
            len,
            null_count: OnceLock::new(),
        }
    }
}

impl From<Array> for BooleanArrayV {
    fn from(array: Array) -> Self {
        match array {
            Array::BooleanArray(arr) => {
                let len = arr.len();
                BooleanArrayV {
                    array: arr,
                    offset: 0,
                    len,
                    null_count: OnceLock::new(),
                }
            }
            _ => panic!("Array is not a BooleanArray"),
        }
    }
}

impl From<ArrayV> for BooleanArrayV {
    fn from(view: ArrayV) -> Self {
        let (array, offset, len) = view.as_tuple();
        match array {
            Array::BooleanArray(inner) => Self {
                array: inner,
                offset,
                len,
                null_count: OnceLock::new(),
            },
            _ => panic!("From<ArrayView>: expected BooleanArray variant"),
        }
    }
}

impl Debug for BooleanArrayV {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("BooleanArrayView")
            .field("offset", &self.offset)
            .field("len", &self.len)
            .field("array", &self.array)
            .field("cached_null_count", &self.null_count.get())
            .finish()
    }
}

impl Display for BooleanArrayV {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "BooleanArrayView [{} values] (offset: {}, nulls: {})",
            self.len(),
            self.offset,
            self.null_count()
        )?;

        let max = self.len().min(MAX_PREVIEW);
        for i in 0..max {
            match self.get_bool(i) {
                Some(v) => writeln!(f, "  {v}")?,
                None => writeln!(f, "  \u{b7}")?,
            }
        }

        if self.len() > MAX_PREVIEW {
            writeln!(f, "  ... ({} more)", self.len() - MAX_PREVIEW)?;
        }

        Ok(())
    }
}

impl Shape for BooleanArrayV {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank1(self.len())
    }
}

impl Concatenate for BooleanArrayV {
    /// Concatenates two boolean array views by materialising both to owned boolean arrays,
    /// concatenating them, and wrapping the result back in a view.
    ///
    /// # Notes
    /// - This operation copies data from both views to create owned boolean arrays.
    /// - The resulting view has offset=0 and length equal to the combined length.
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        // Materialise both views to owned boolean arrays
        let self_array =
            Arc::try_unwrap(self.to_boolean_array()).unwrap_or_else(|arc| (*arc).clone());
        let other_array =
            Arc::try_unwrap(other.to_boolean_array()).unwrap_or_else(|arc| (*arc).clone());

        // Concatenate the owned boolean arrays
        let concatenated = self_array.concat(other_array)?;

        // Wrap the result in a new view
        Ok(BooleanArrayV::from(Arc::new(concatenated)))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{Bitmask, BooleanArray};

    #[test]
    fn test_boolean_array_view_basic_indexing_and_slice() {
        let mut arr = BooleanArray::<()>::default();
        arr.push(true);
        arr.push(false);
        arr.push(true);
        arr.push(false);

        let arc = Arc::new(arr);
        let view = BooleanArrayV::new(arc, 1, 2);

        assert_eq!(view.len(), 2);
        assert_eq!(view.offset, 1);
        assert_eq!(view.get_bool(0), Some(false));
        assert_eq!(view.get_bool(1), Some(true));
        assert_eq!(view.get_bool(2), None);

        let sub = view.slice(1, 1);
        assert_eq!(sub.len(), 1);
        assert_eq!(sub.get_bool(0), Some(true));
        assert_eq!(sub.get_bool(1), None);
    }

    #[test]
    fn test_boolean_array_view_null_count_and_cache() {
        let mut arr = BooleanArray::<()>::default();
        arr.push(true);
        arr.push(false);
        arr.push(true);
        arr.push(false);

        let mut mask = Bitmask::new_set_all(4, true);
        mask.set(2, false);
        arr.null_mask = Some(mask);

        let arc = Arc::new(arr);
        let view = BooleanArrayV::new(arc, 0, 4);
        assert_eq!(view.null_count(), 1, "Null count should detect one null");
        assert_eq!(view.null_count(), 1);

        let view2 = view.slice(0, 2);
        assert_eq!(view2.null_count(), 0);
        let view3 = view.slice(2, 2);
        assert_eq!(view3.null_count(), 1);
    }

    #[test]
    fn test_boolean_array_view_with_supplied_null_count() {
        let mut arr = BooleanArray::<()>::default();
        arr.push(true);
        arr.push(false);

        let arc = Arc::new(arr);
        let view = BooleanArrayV::new_nc(arc, 0, 2, 99);
        assert_eq!(view.null_count(), 99);
        // Trying to set again should fail since it's already initialised
        assert!(view.set_null_count(101).is_err());
        // Still returns original value
        assert_eq!(view.null_count(), 99);
    }

    #[test]
    fn test_boolean_array_view_to_boolean_array_and_as_tuple() {
        let mut arr = BooleanArray::<()>::default();
        for v in [
            true, false, true, false, true, true, false, true, false, true,
        ] {
            arr.push(v);
        }
        let arc = Arc::new(arr);
        let view = BooleanArrayV::new(arc.clone(), 4, 3);
        let arr2 = view.to_boolean_array();
        assert_eq!(arr2.len(), 3);
        assert_eq!(arr2.get(0), Some(true));
        assert_eq!(arr2.get(1), Some(true));
        assert_eq!(arr2.get(2), Some(false));

        let tup = view.as_tuple();
        assert!(Arc::ptr_eq(&tup.0, &arc));
        assert_eq!(tup.1, 4);
        assert_eq!(tup.2, 3);
    }

    #[test]
    fn test_boolean_array_view_null_mask_view() {
        let mut arr = BooleanArray::<()>::default();
        arr.push(true);
        arr.push(false);
        arr.push(true);

        let mut mask = Bitmask::new_set_all(3, true);
        mask.set(0, false);
        arr.null_mask = Some(mask);

        let arc = Arc::new(arr);
        let view = BooleanArrayV::new(arc, 1, 2);
        let mask_view = view.null_mask_view().expect("Should have mask");
        assert_eq!(mask_view.len(), 2);
        assert!(mask_view.get(0));
        assert!(mask_view.get(1));
    }

    #[test]
    fn test_boolean_array_view_from_arc_and_array() {
        let mut arr = BooleanArray::<()>::default();
        arr.push(true);
        arr.push(false);

        let arc = Arc::new(arr);
        let view_from_arc = BooleanArrayV::from(arc.clone());
        assert_eq!(view_from_arc.len(), 2);
        assert_eq!(view_from_arc.get_bool(0), Some(true));

        let array = Array::BooleanArray(arc.clone());
        let view_from_array = BooleanArrayV::from(array);
        assert_eq!(view_from_array.len(), 2);
        assert_eq!(view_from_array.get_bool(1), Some(false));
    }

    #[test]
    #[should_panic(expected = "Array is not a BooleanArray")]
    fn test_boolean_array_view_from_array_panics_on_wrong_variant() {
        let array = Array::Null;
        let _view = BooleanArrayV::from(array);
    }

    #[test]
    fn test_boolean_array_view_materialisation_roundtrip() {
        let mut arr = BooleanArray::<()>::default();
        arr.push(true);
        arr.push(false);
        arr.push(true);

        let arc = Arc::new(arr);
        let view = BooleanArrayV::new(arc, 1, 2);
        let materialised = view.to_boolean_array();
        assert_eq!(materialised.len(), 2);
        assert_eq!(materialised.get(0), Some(false));
        assert_eq!(materialised.get(1), Some(true));
    }

    #[test]
    #[should_panic(expected = "BooleanArrayView: window out of bounds")]
    fn test_boolean_array_view_out_of_bounds() {
        let mut arr = BooleanArray::<()>::default();
        arr.push(true);
        arr.push(false);
        let arc = Arc::new(arr);
        let _view = BooleanArrayV::new(arc, 1, 5);
    }

    #[test]
    fn test_boolean_array_view_concat() {
        let mut a = BooleanArray::<()>::default();
        a.push(true);
        a.push(false);
        let mut b = BooleanArray::<()>::default();
        b.push(false);
        b.push(true);
        b.push(true);

        let va = BooleanArrayV::from(Arc::new(a));
        let vb = BooleanArrayV::from(Arc::new(b));
        let combined = va.concat(vb).unwrap();
        assert_eq!(combined.len(), 5);
        assert_eq!(combined.get_bool(0), Some(true));
        assert_eq!(combined.get_bool(1), Some(false));
        assert_eq!(combined.get_bool(2), Some(false));
        assert_eq!(combined.get_bool(3), Some(true));
        assert_eq!(combined.get_bool(4), Some(true));
    }

    #[test]
    fn test_boolean_array_view_from_array_view() {
        let mut arr = BooleanArray::<()>::default();
        arr.push(true);
        arr.push(false);
        arr.push(true);
        arr.push(false);

        let array = Array::BooleanArray(Arc::new(arr));
        let array_view = ArrayV::new(array, 1, 2);
        let bool_view = BooleanArrayV::from(array_view);

        assert_eq!(bool_view.len(), 2);
        assert_eq!(bool_view.offset, 1);
        assert_eq!(bool_view.get_bool(0), Some(false));
        assert_eq!(bool_view.get_bool(1), Some(true));
    }

    #[test]
    fn test_to_boolean_array_full_coverage_shares_arc() {
        let mut arr = BooleanArray::<()>::default();
        arr.push(true);
        arr.push(false);
        arr.push(true);
        let src = Arc::new(arr);

        let view = BooleanArrayV::new(src.clone(), 0, 3);
        let out = view.to_boolean_array();

        assert!(
            Arc::ptr_eq(&src, &out),
            "full-coverage to_boolean_array should share the underlying Arc"
        );
    }

    #[test]
    fn test_to_boolean_array_windowed_copies() {
        let mut arr = BooleanArray::<()>::default();
        arr.push(true);
        arr.push(false);
        arr.push(true);
        let src = Arc::new(arr);

        let view = BooleanArrayV::new(src.clone(), 1, 2);
        let out = view.to_boolean_array();

        assert!(
            !Arc::ptr_eq(&src, &out),
            "windowed to_boolean_array must allocate a fresh buffer"
        );
    }
}
