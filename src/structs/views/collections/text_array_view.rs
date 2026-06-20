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

//! # **TextArrayView Module** - *Windowed View over a TextArray*
//!
//! `TextArrayV` is a **read-only, zero-copy view** into a `[offset .. offset + len)`
//! slice of a [`TextArray`].
//!
//! ## Purpose
//! - Provides windowed access to UTF-8 string data without copying buffers.
//! - Unifies handling for `StringArray` and `CategoricalArray` under the `TextArray` enum.
//! - Allows null counts to be cached per view for faster repeated operations.
//!
//! ## Behaviour
//! - Fully supports both string and dictionary-encoded text variants.
//! - Enables lightweight sub-ranges for use in downstream computations, filtering, or previews.
//! - Creating a view does **not** allocate; slicing is zero-copy.
//!
//! ## Threading
//! - Not thread-safe due to `Cell` for cached null counts.
//! - For multi-threaded code, create per-thread clones via [`slice`](TextArrayV::slice).
//!
//! ## Interop
//! - Convert back to a full `TextArray` via [`to_text_array`](TextArrayV::to_text_array).
//! - Promote to `Array` via [`inner_array`](TextArrayV::inner_array) for unified API calls.
//!
//! ## Invariants
//! - `offset + len <= array.len()`
//! - `len` is the logical number of text elements in the view.

use std::fmt::{self, Debug, Display, Formatter};
use std::sync::OnceLock;

use crate::enums::error::MinarrowError;
use crate::enums::shape_dim::ShapeDim;
use crate::traits::concatenate::Concatenate;
use crate::traits::print::MAX_PREVIEW;
use crate::traits::shape::Shape;
use std::sync::Arc;

use crate::{Array, ArrayV, BitmaskV, StringArray, TextArray};

/// # TextArrayView
///
/// Borrowed, indexable view into a `[offset .. offset + len)` window of a
/// [`TextArray`].
///
/// ## Purpose
/// - Zero-copy access to contiguous UTF-8 values in either `StringArray`
///   or `CategoricalArray` form.
/// - Allows null counts for the view to be cached to speed up scans.
///
/// ## Fields
/// - `array`: the backing [`TextArray`].
/// - `offset`: starting index of the view.
/// - `len`: number of logical elements in the view.
/// - `null_count`: cached `Option<usize>` of nulls for this range.
///
/// ## Notes
/// - `TextArrayV` is not thread-safe due to its use of `Cell` for caching the null count.
/// - Use [`slice`](Self::slice) to derive smaller views from this one.
/// - Use [`to_text_array`](Self::to_text_array) to materialise the data.
#[derive(Clone, PartialEq)]
pub struct TextArrayV {
    /// The **outer array** that this view is derived from - we retain a reference to it. 
    /// Importantly, this is the ***full array*** - not the *view*, and thus should not be 
    /// accessed as though it were the view subset.
    pub array: TextArray,
    /// The index offset from 0 that for where this view starts from the outer array
    pub offset: usize,
    /// The length of the array view
    len: usize,
    /// How many nulls are in the TextArrayView
    /// At construction, this is None, unless constructed via new_nc. When one uses '.null_count()',
    /// the first time it will calculate it (quickly) using Bitmask popcount, and then from that 
    /// point onwards the null count is a cached value. 
    null_count: OnceLock<usize>,
}

impl TextArrayV {
    /// Creates a new `TextArrayView` with the given offset and length.
    pub fn new(array: TextArray, offset: usize, len: usize) -> Self {
        assert!(
            offset + len <= array.len(),
            "TextArrayView: window out of bounds (offset + len = {}, array.len = {})",
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

    /// Creates a new `TextArrayView` with a precomputed null count.
    pub fn new_nc(array: TextArray, offset: usize, len: usize, null_count: usize) -> Self {
        assert!(
            offset + len <= array.len(),
            "TextArrayView: window out of bounds (offset + len = {}, array.len = {})",
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

    /// Returns the value at logical index `i` as an `Option<&str>`.
    #[inline]
    pub fn get_str(&self, i: usize) -> Option<&str> {
        if i >= self.len {
            return None;
        }
        let phys_idx = self.offset + i;
        match &self.array {
            TextArray::String32(arr) => arr.get_str(phys_idx),
            #[cfg(feature = "large_string")]
            TextArray::String64(arr) => arr.get_str(phys_idx),
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            TextArray::Categorical32(arr) => arr.get_str(phys_idx),
            #[cfg(feature = "default_categorical_8")]
            TextArray::Categorical8(arr) => arr.get_str(phys_idx),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical16(arr) => arr.get_str(phys_idx),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical64(arr) => arr.get_str(phys_idx),
            TextArray::Null => None,
        }
    }

    /// Returns a sliced view into a subrange of this view.
    #[inline]
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        assert!(
            offset + len <= self.len,
            "TextArrayView::slice: out of bounds"
        );
        Self {
            array: self.array.clone(),
            offset: self.offset + offset,
            len,
            null_count: OnceLock::new(),
        }
    }

    /// Returns the length of the window
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the full backing array wrapped as an `Array` enum, ignoring the view's offset and length.
    ///
    /// Use this to access inner array methods. The returned array is the unwindowed original.
    #[inline]
    pub fn inner_array(&self) -> Array {
        Array::TextArray(self.array.clone()) // Arc clone for data buffer
    }

    /// Returns an owned `TextArray` for the window.
    ///
    /// If the view covers the entire backing array, returns a cheap Arc clone
    /// of the original variant. Otherwise deep-copies the window via
    /// `slice_clone` through `inner_array`.
    pub fn to_text_array(&self) -> TextArray {
        if self.offset == 0 && self.len == self.array.len() {
            return self.array.clone();
        }
        self.inner_array().slice_clone(self.offset, self.len).str()
    }

    /// Returns the end index of the view.
    #[inline]
    pub fn end(&self) -> usize {
        self.offset + self.len
    }

    /// Returns the view as a tuple `(array, offset, len)`.
    #[inline]
    pub fn as_tuple(&self) -> (TextArray, usize, usize) {
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
    /// This is thread-safe and can only succeed once per TextArrayV instance.
    #[inline]
    pub fn set_null_count(&self, count: usize) -> Result<(), usize> {
        self.null_count.set(count).map_err(|_| count)
    }
}

impl From<TextArray> for TextArrayV {
    fn from(array: TextArray) -> Self {
        let len = array.len();
        TextArrayV {
            array,
            offset: 0,
            len,
            null_count: OnceLock::new(),
        }
    }
}

impl From<Array> for TextArrayV {
    fn from(array: Array) -> Self {
        match array {
            Array::TextArray(arr) => {
                let len = arr.len();
                TextArrayV {
                    array: arr,
                    offset: 0,
                    len,
                    null_count: OnceLock::new(),
                }
            }
            _ => panic!("Array is not a TextArray"),
        }
    }
}

impl From<ArrayV> for TextArrayV {
    fn from(view: ArrayV) -> Self {
        let (array, offset, len) = view.as_tuple();
        match array {
            Array::TextArray(inner) => Self {
                array: inner,
                offset,
                len,
                null_count: OnceLock::new(),
            },
            _ => panic!("From<ArrayView>: expected TextArray variant"),
        }
    }
}

impl From<vec64::Vec64<String>> for TextArrayV {
    fn from(strings: vec64::Vec64<String>) -> Self {
        let array = TextArray::String64(Arc::new(StringArray::<u64>::from_vec64_owned(
            strings, None,
        )));
        TextArrayV::from(array)
    }
}

impl From<Vec<String>> for TextArrayV {
    fn from(strings: Vec<String>) -> Self {
        let v64: vec64::Vec64<String> = strings.into();
        TextArrayV::from(v64)
    }
}

impl<'a> From<&'a [&'a str]> for TextArrayV {
    fn from(strings: &'a [&'a str]) -> Self {
        let array = TextArray::String64(Arc::new(StringArray::<u64>::from_slice(strings)));
        TextArrayV::from(array)
    }
}

impl Debug for TextArrayV {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("TextArrayView")
            .field("offset", &self.offset)
            .field("len", &self.len)
            .field("array", &self.array)
            .field("cached_null_count", &self.null_count.get())
            .finish()
    }
}

impl Display for TextArrayV {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let dtype = match &self.array {
            TextArray::String32(_) => "String32<u32>",
            #[cfg(feature = "large_string")]
            TextArray::String64(_) => "String64<u64>",
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            TextArray::Categorical32(_) => "Categorical32<u32>",
            #[cfg(feature = "default_categorical_8")]
            TextArray::Categorical8(_) => "Categorical8<u8>",
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical16(_) => "Categorical16<u16>",
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical64(_) => "Categorical64<u64>",
            TextArray::Null => "Null",
        };

        writeln!(
            f,
            "TextArrayView<{dtype}> [{} values] (offset: {}, nulls: {})",
            self.len(),
            self.offset,
            self.null_count()
        )?;

        let max = self.len().min(MAX_PREVIEW);
        for i in 0..max {
            match self.get_str(i) {
                Some(s) => writeln!(f, "  \"{s}\"")?,
                None => writeln!(f, "  ·")?,
            }
        }

        if self.len() > MAX_PREVIEW {
            writeln!(f, "  ... ({} more)", self.len() - MAX_PREVIEW)?;
        }

        Ok(())
    }
}

impl Shape for TextArrayV {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank1(self.len())
    }
}

impl Concatenate for TextArrayV {
    /// Concatenates two text array views by materialising both to owned text arrays,
    /// concatenating them, and wrapping the result back in a view.
    ///
    /// # Notes
    /// - This operation copies data from both views to create owned text arrays.
    /// - The resulting view has offset=0 and length equal to the combined length.
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        // Materialise both views to owned text arrays
        let self_array = self.to_text_array();
        let other_array = other.to_text_array();

        // Concatenate the owned text arrays
        let concatenated = self_array.concat(other_array)?;

        // Wrap the result in a new view
        Ok(TextArrayV::from(concatenated))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{Bitmask, StringArray, TextArray};

    #[test]
    fn test_text_array_view_basic_indexing_and_slice() {
        let arr = StringArray::<u32>::from_slice(&["abc", "def", "xyz", "uvw"]);
        let text = TextArray::String32(Arc::new(arr));
        let view = TextArrayV::new(text, 1, 2);

        assert_eq!(view.len(), 2);
        assert_eq!(view.offset, 1);
        assert_eq!(view.get_str(0), Some("def"));
        assert_eq!(view.get_str(1), Some("xyz"));
        assert_eq!(view.get_str(2), None);

        let sub = view.slice(1, 1);
        assert_eq!(sub.len(), 1);
        assert_eq!(sub.get_str(0), Some("xyz"));
        assert_eq!(sub.get_str(1), None);
    }

    #[test]
    fn test_text_array_view_null_count_and_cache() {
        let mut arr = StringArray::<u32>::from_slice(&["a", "b", "c", "d"]);
        let mut mask = Bitmask::new_set_all(4, true);
        mask.set(2, false);
        arr.null_mask = Some(mask);

        let text = TextArray::String32(Arc::new(arr));
        let view = TextArrayV::new(text, 0, 4);
        assert_eq!(view.null_count(), 1, "Null count should detect one null");
        assert_eq!(view.null_count(), 1);

        let view2 = view.slice(0, 2);
        assert_eq!(view2.null_count(), 0);
        let view3 = view.slice(2, 2);
        assert_eq!(view3.null_count(), 1);
    }

    #[test]
    fn test_text_array_view_with_supplied_null_count() {
        let arr = StringArray::<u32>::from_slice(&["g", "h"]);
        let text = TextArray::String32(Arc::new(arr));
        let view = TextArrayV::new_nc(text, 0, 2, 99);
        assert_eq!(view.null_count(), 99);
        // Trying to set again should fail since it's already initialised
        assert!(view.set_null_count(101).is_err());
        // Still returns original value
        assert_eq!(view.null_count(), 99);
    }

    #[test]
    fn test_text_array_view_to_text_array_and_as_tuple() {
        let arr =
            StringArray::<u32>::from_slice(&["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]);
        let text = TextArray::String32(Arc::new(arr));
        let view = TextArrayV::new(text.clone(), 4, 3);
        let arr2 = view.to_text_array();
        if let TextArray::String32(a2) = arr2 {
            assert_eq!(a2.iter_str().collect::<Vec<_>>(), vec!["e", "f", "g"]);
        } else {
            panic!("Unexpected variant");
        }
        let tup = view.as_tuple();
        assert_eq!(&tup.0, &text);
        assert_eq!(tup.1, 4);
        assert_eq!(tup.2, 3);
    }

    #[test]
    fn test_text_array_view_null_mask_view() {
        let mut arr = StringArray::<u32>::from_slice(&["j", "k", "l"]);
        let mut mask = Bitmask::new_set_all(3, true);
        mask.set(0, false);
        arr.null_mask = Some(mask);

        let text = TextArray::String32(Arc::new(arr));
        let view = TextArrayV::new(text, 1, 2);
        let mask_view = view.null_mask_view().expect("Should have mask");
        assert_eq!(mask_view.len(), 2);
        assert!(mask_view.get(0));
        assert!(mask_view.get(1));
    }

    #[test]
    fn test_text_array_view_from_text_array_and_array() {
        let arr = StringArray::<u32>::from_slice(&["x", "y"]);
        let text = TextArray::String32(Arc::new(arr));
        let view_from_text = TextArrayV::from(text.clone());
        assert_eq!(view_from_text.len(), 2);
        assert_eq!(view_from_text.get_str(0), Some("x"));

        let array = Array::TextArray(text.clone());
        let view_from_array = TextArrayV::from(array.clone());
        assert_eq!(view_from_array.len(), 2);
        assert_eq!(view_from_array.get_str(1), Some("y"));
    }

    #[test]
    #[should_panic(expected = "Array is not a TextArray")]
    fn test_text_array_view_from_array_panics_on_wrong_variant() {
        let array = Array::Null;
        let _view = TextArrayV::from(array);
    }

    #[test]
    fn test_to_text_array_full_coverage_shares_arc() {
        let arr = StringArray::<u32>::from_slice(&["a", "b", "c"]);
        let src = Arc::new(arr);
        let text = TextArray::String32(src.clone());

        let view = TextArrayV::new(text, 0, 3);
        let out = view.to_text_array();

        match out {
            TextArray::String32(out_arc) => assert!(
                Arc::ptr_eq(&src, &out_arc),
                "full-coverage to_text_array should share the underlying Arc"
            ),
            _ => panic!("expected String32 variant"),
        }
    }

    #[test]
    fn test_to_text_array_windowed_copies() {
        let arr = StringArray::<u32>::from_slice(&["a", "b", "c"]);
        let src = Arc::new(arr);
        let text = TextArray::String32(src.clone());

        let view = TextArrayV::new(text, 1, 2);
        let out = view.to_text_array();

        match out {
            TextArray::String32(out_arc) => assert!(
                !Arc::ptr_eq(&src, &out_arc),
                "windowed to_text_array must allocate a fresh buffer"
            ),
            _ => panic!("expected String32 variant"),
        }
    }
}
