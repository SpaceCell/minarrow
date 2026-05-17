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

//! # **ArrayView Module** - *Windowed View over an Array*
//!
//! `ArrayV` is a **logical, read-only, zero-copy view** into a contiguous window
//! `[offset .. offset + len)` of any [`Array`] variant.
//!
//! ## Purpose
//! - Provides indexable, bounds-checked access to a subrange of an array without copying buffers.
//! - Caches null counts per view for efficient repeated queries.
//! - Acts as a unifying abstraction for windowed operations across all array types.
//!
//! ## Behaviour
//! - All indices are **relative** to the view's start.
//! - Internally retains an `Arc` reference to the parent array's buffers.
//! - Windowing and slicing are O(1) operations (pointer + metadata updates only).
//! - Cached null counts are stored in an `OnceLock` for thread-safe lazy initialization.
//!
//! ## Threading
//! - Thread-safe for sharing across threads (uses `OnceLock` for null count caching).
//! - Safe to share via `Arc` for parallel processing.
//!
//! ## Interop
//! - Convert back to a full array via [`to_array`](ArrayV::to_array).
//! - Promote to `(Array, offset, len)` tuple with [`as_tuple`](ArrayV::as_tuple).
//! - Access raw data pointer and element size via [`data_ptr_and_byte_len`](ArrayV::data_ptr_and_byte_len).
//!
//! ## Invariants
//! - `offset + len <= array.len()`
//! - `len` reflects the **logical** number of elements in the view.

use std::fmt::{self, Debug, Display, Formatter};
use std::sync::OnceLock;

use crate::enums::error::MinarrowError;
use crate::enums::shape_dim::ShapeDim;
use crate::traits::concatenate::Concatenate;
use crate::traits::print::MAX_PREVIEW;
#[cfg(feature = "select")]
use crate::traits::selection::{DataSelector, RowSelection};
use crate::traits::shape::Shape;
use crate::{Array, BitmaskV, FieldArray, MaskedArray, TextArray};

/// # ArrayView
///
/// Logical, windowed view over an `Array`.
/// 
/// ArrayView handles indexing offsets automatically so that the View behaves
/// like a regular array.
///
/// ## Purpose
/// This is used to return an indexable view over a subset of the array.
/// Additionally, it can be used to cache null counts for those regions,
/// which can be used to speed up calculations.
///
/// ## Behaviour
/// - Indices are always relative to the window.
/// - Holds a reference to the original `Array` and window bounds.
/// - Windowing uses an arc clone
/// - All access (get/index, etc.) is offset-correct and bounds-checked.
/// - Null count is computed once (on demand or at creation) and cached for subsequent use.
///
/// ## Notes
/// - Use [`slice`](Self::slice) to derive smaller views without data copy.
/// - Use [`to_array`](Self::to_array) to materialise as an owned array.
#[derive(Clone, PartialEq)]
pub struct ArrayV {
    /// The **outer array** that this view is derived from - we retain a reference to it. 
    /// Importantly, this is the ***full array*** - not the *view*, and thus should not be 
    /// accessed as though it were the view subset.
    pub array: Array, // contains Arc<inner>
    /// The index offset from 0 that for where this view starts from the outer array
    pub offset: usize,
    /// The length of the array view
    len: usize,
    /// How many nulls are in the ArrayView
    /// At construction, this is None, unless constructed via new_nc. When one uses '.null_count()',
    /// the first time it will calculate it (quickly) using Bitmask popcount, and then from that 
    /// point onwards the null count is a cached value. 
    null_count: OnceLock<usize>,
}

impl ArrayV {
    /// Construct a windowed view of `array[offset..offset+len)`, with optional precomputed null count.
    #[inline]
    pub fn new(array: Array, offset: usize, len: usize) -> Self {
        let array_len = array.len();
        assert!(
            len <= array_len && offset <= array_len - len,
            "ArrayView: window out of bounds (offset = {offset}, len = {len}, array.len = {array_len})"
        );
        Self {
            array,
            offset,
            len,
            null_count: OnceLock::new(),
        }
    }

    /// Construct a windowed view, supplying a precomputed null count.
    #[inline]
    pub fn new_nc(array: Array, offset: usize, len: usize, null_count: usize) -> Self {
        let array_len = array.len();
        assert!(
            len <= array_len && offset <= array_len - len,
            "ArrayView: window out of bounds (offset = {offset}, len = {len}, array.len = {array_len})"
        );
        let lock = OnceLock::new();
        let _ = lock.set(null_count); // Pre-initialize with the provided count
        Self {
            array,
            offset,
            len,
            null_count: lock,
        }
    }

    /// Return the logical length of the view.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the view is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// True when the view spans the entirety of its backing array
    /// (offset == 0 and length matches the underlying array length).
    /// When true, `to_array()` Arc-bumps the backing array directly with
    /// no buffer copy. When false the view is genuinely windowed and
    /// `to_array()` falls through to `slice_clone`, reallocating each
    /// buffer.
    #[inline]
    pub fn spans_backing(&self) -> bool {
        self.offset == 0 && self.len == self.array.len()
    }

    /// Returns the value at logical index `i` within the window, or `None` if out of bounds or null.
    #[inline]
    pub fn get<T: MaskedArray + 'static>(&self, i: usize) -> Option<T::CopyType<'_>> {
        if i >= self.len {
            return None;
        }
        self.array.inner::<T>().get(self.offset + i)
    }

    /// Returns the value at logical index `i` within the window (unchecked).
    ///
    /// # Safety
    /// `i` must be less than the view's logical length. No bounds check is performed.
    #[inline]
    pub unsafe fn get_unchecked<T: MaskedArray + 'static>(
        &self,
        i: usize,
    ) -> Option<T::CopyType<'_>> {
        unsafe { self.array.inner::<T>().get_unchecked(self.offset + i) }
    }

    /// Returns the string value at logical index `i` within the window, or `None` if out of bounds or null.
    #[inline]
    pub fn get_str(&self, i: usize) -> Option<&str> {
        if i >= self.len {
            return None;
        }

        match &self.array {
            Array::TextArray(TextArray::String32(arr)) => arr.get_str(self.offset + i),
            #[cfg(feature = "large_string")]
            Array::TextArray(TextArray::String64(arr)) => arr.get_str(self.offset + i),
            #[cfg(feature = "default_categorical_8")]
            Array::TextArray(TextArray::Categorical8(arr)) => arr.get_str(self.offset + i),
            #[cfg(feature = "extended_categorical")]
            Array::TextArray(TextArray::Categorical16(arr)) => arr.get_str(self.offset + i),
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            Array::TextArray(TextArray::Categorical32(arr)) => arr.get_str(self.offset + i),
            #[cfg(feature = "extended_categorical")]
            Array::TextArray(TextArray::Categorical64(arr)) => arr.get_str(self.offset + i),
            _ => None,
        }
    }

    /// Returns the string value at logical index `i` within the window.
    ///
    /// # Safety
    /// Skips bounds checks, but will still return `None` if null.
    #[inline]
    pub unsafe fn get_str_unchecked(&self, i: usize) -> Option<&str> {
        match &self.array {
            Array::TextArray(TextArray::String32(arr)) => {
                if arr.is_null(self.offset + i) {
                    None
                } else {
                    Some(unsafe { arr.get_str_unchecked(self.offset + i) })
                }
            }
            #[cfg(feature = "large_string")]
            Array::TextArray(TextArray::String64(arr)) => {
                if arr.is_null(self.offset + i) {
                    None
                } else {
                    Some(unsafe { arr.get_str_unchecked(self.offset + i) })
                }
            }
            #[cfg(feature = "default_categorical_8")]
            Array::TextArray(TextArray::Categorical8(arr)) => {
                if arr.is_null(self.offset + i) {
                    None
                } else {
                    Some(unsafe { arr.get_str_unchecked(self.offset + i) })
                }
            }
            #[cfg(feature = "extended_categorical")]
            Array::TextArray(TextArray::Categorical16(arr)) => {
                if arr.is_null(self.offset + i) {
                    None
                } else {
                    Some(unsafe { arr.get_str_unchecked(self.offset + i) })
                }
            }
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            Array::TextArray(TextArray::Categorical32(arr)) => {
                if arr.is_null(self.offset + i) {
                    None
                } else {
                    Some(unsafe { arr.get_str_unchecked(self.offset + i) })
                }
            }
            #[cfg(feature = "extended_categorical")]
            Array::TextArray(TextArray::Categorical64(arr)) => {
                if arr.is_null(self.offset + i) {
                    None
                } else {
                    Some(unsafe { arr.get_str_unchecked(self.offset + i) })
                }
            }
            _ => None,
        }
    }

    /// Returns the value at logical index `i` as a `Scalar`, respecting nulls.
    ///
    /// Delegates to `Array::get_scalar` with the view's offset applied.
    #[cfg(feature = "scalar_type")]
    #[inline]
    pub fn get_scalar(&self, i: usize) -> Option<crate::Scalar> {
        if i >= self.len {
            return None;
        }
        self.array.get_scalar(self.offset + i)
    }

    /// Returns a new window view into a sub-range of this view.
    #[inline]
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        assert!(offset + len <= self.len, "ArrayView::slice: out of bounds");
        Self {
            array: self.array.clone(), // arc clone
            offset: self.offset + offset,
            len,
            null_count: OnceLock::new(),
        }
    }

    /// Materialise the view window as an owned `Array`.
    ///
    /// If the view covers the entire backing array, returns a cheap clone
    /// with no data copy. Otherwise deep-copies the window via slice_clone.
    #[inline]
    pub fn to_array(&self) -> Array {
        if self.offset == 0 && self.len == self.array.len() {
            return self.array.clone();
        }
        self.array.slice_clone(self.offset, self.len)
    }

    /// Extract array data as `Vec64<T>`, casting numeric values if necessary.
    ///
    /// - If array type matches T exactly, copies the slice directly
    /// - If array is a different numeric type, casts each element via `NumCast`
    /// - Returns error if array type is not numeric or nulls are present
    ///
    /// # Example
    /// ```ignore
    /// let av = ArrayV::from(Array::from_float64(...));
    /// let floats: Vec64<f64> = av.to_typed_vec::<f64>()?;
    /// ```
    pub fn to_typed_vec<T: crate::Numeric>(
        &self,
    ) -> Result<crate::Vec64<T>, crate::enums::error::KernelError> {
        use crate::enums::error::KernelError;
        use crate::{NumericArray, Vec64};
        use num_traits::NumCast;

        let offset = self.offset;
        let len = self.len;

        macro_rules! cast_slice {
            ($arr:expr) => {{
                let slice = &$arr.data.as_slice()[offset..offset + len];
                slice
                    .iter()
                    .map(|&v| {
                        NumCast::from(v).ok_or_else(|| {
                            KernelError::UnsupportedType("numeric cast failed".into())
                        })
                    })
                    .collect::<Result<Vec64<T>, _>>()
            }};
        }

        match &self.array {
            Array::NumericArray(num) => match num {
                NumericArray::Int32(a) => cast_slice!(a),
                NumericArray::Int64(a) => cast_slice!(a),
                NumericArray::UInt32(a) => cast_slice!(a),
                NumericArray::UInt64(a) => cast_slice!(a),
                NumericArray::Float32(a) => cast_slice!(a),
                NumericArray::Float64(a) => cast_slice!(a),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(a) => cast_slice!(a),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(a) => cast_slice!(a),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(a) => cast_slice!(a),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(a) => cast_slice!(a),
                NumericArray::Null => {
                    Err(KernelError::UnsupportedType("null numeric array".into()))
                }
            },
            Array::BooleanArray(a) => {
                // Convert bools to numeric: true=1, false=0
                (0..len)
                    .map(|i| {
                        let v = a.get(offset + i).unwrap_or(false);
                        NumCast::from(if v { 1u8 } else { 0u8 }).ok_or_else(|| {
                            KernelError::UnsupportedType("bool to numeric cast failed".into())
                        })
                    })
                    .collect::<Result<Vec64<T>, _>>()
            }
            _ => Err(KernelError::UnsupportedType(
                "to_typed_vec requires a numeric array".into(),
            )),
        }
    }

    /// Gather specific indices from this view into a new materialised Array.
    /// Indices are relative to this view's window.
    #[cfg(feature = "select")]
    pub fn gather_indices(&self, indices: &[usize]) -> Array {
        use crate::{
            BooleanArray, CategoricalArray, FloatArray, IntegerArray, NumericArray, StringArray,
            TextArray,
        };
        #[cfg(feature = "datetime")]
        use crate::{DatetimeArray, TemporalArray};

        match &self.array {
            Array::Null => Array::Null,
            Array::NumericArray(num_arr) => match num_arr {
                NumericArray::Int32(_) => {
                    let mut new_arr = IntegerArray::<i32>::with_capacity(indices.len(), true);
                    for &idx in indices {
                        if let Some(val) = self.get::<IntegerArray<i32>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_int32(new_arr)
                }
                NumericArray::Int64(_) => {
                    let mut new_arr = IntegerArray::<i64>::with_capacity(indices.len(), true);
                    for &idx in indices {
                        if let Some(val) = self.get::<IntegerArray<i64>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_int64(new_arr)
                }
                NumericArray::Float32(_) => {
                    let mut new_arr = FloatArray::<f32>::with_capacity(indices.len(), true);
                    for &idx in indices {
                        if let Some(val) = self.get::<FloatArray<f32>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_float32(new_arr)
                }
                NumericArray::Float64(_) => {
                    let mut new_arr = FloatArray::<f64>::with_capacity(indices.len(), true);
                    for &idx in indices {
                        if let Some(val) = self.get::<FloatArray<f64>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_float64(new_arr)
                }
                NumericArray::UInt32(_) => {
                    let mut new_arr = IntegerArray::<u32>::with_capacity(indices.len(), true);
                    for &idx in indices {
                        if let Some(val) = self.get::<IntegerArray<u32>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_uint32(new_arr)
                }
                NumericArray::UInt64(_) => {
                    let mut new_arr = IntegerArray::<u64>::with_capacity(indices.len(), true);
                    for &idx in indices {
                        if let Some(val) = self.get::<IntegerArray<u64>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_uint64(new_arr)
                }
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(_) => {
                    let mut new_arr = IntegerArray::<i8>::with_capacity(indices.len(), true);
                    for &idx in indices {
                        if let Some(val) = self.get::<IntegerArray<i8>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_int8(new_arr)
                }
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(_) => {
                    let mut new_arr = IntegerArray::<i16>::with_capacity(indices.len(), true);
                    for &idx in indices {
                        if let Some(val) = self.get::<IntegerArray<i16>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_int16(new_arr)
                }
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(_) => {
                    let mut new_arr = IntegerArray::<u8>::with_capacity(indices.len(), true);
                    for &idx in indices {
                        if let Some(val) = self.get::<IntegerArray<u8>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_uint8(new_arr)
                }
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(_) => {
                    let mut new_arr = IntegerArray::<u16>::with_capacity(indices.len(), true);
                    for &idx in indices {
                        if let Some(val) = self.get::<IntegerArray<u16>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_uint16(new_arr)
                }
                NumericArray::Null => Array::Null,
            },
            Array::TextArray(text_arr) => match text_arr {
                TextArray::String32(_) => {
                    let mut values: Vec<&str> = Vec::with_capacity(indices.len());
                    for &idx in indices {
                        if let Some(val) = self.get_str(idx) {
                            values.push(val);
                        } else {
                            values.push("");
                        }
                    }
                    let mut new_arr = StringArray::<u32>::from_vec(values, None);
                    for (i, &idx) in indices.iter().enumerate() {
                        if self.get_str(idx).is_none() {
                            new_arr.set_null(i);
                        }
                    }
                    Array::from_string32(new_arr)
                }
                #[cfg(feature = "large_string")]
                TextArray::String64(_) => {
                    let mut values: Vec<&str> = Vec::with_capacity(indices.len());
                    for &idx in indices {
                        if let Some(val) = self.get_str(idx) {
                            values.push(val);
                        } else {
                            values.push("");
                        }
                    }
                    let mut new_arr = StringArray::<u64>::from_vec(values, None);
                    for (i, &idx) in indices.iter().enumerate() {
                        if self.get_str(idx).is_none() {
                            new_arr.set_null(i);
                        }
                    }
                    Array::from_string64(new_arr)
                }
                #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
                TextArray::Categorical32(_) => {
                    let mut values: Vec<&str> = Vec::with_capacity(indices.len());
                    for &idx in indices {
                        if let Some(val) = self.get_str(idx) {
                            values.push(val);
                        } else {
                            values.push("");
                        }
                    }
                    let mut new_arr = CategoricalArray::<u32>::from_vec(values, None);
                    for (i, &idx) in indices.iter().enumerate() {
                        if self.get_str(idx).is_none() {
                            new_arr.set_null(i);
                        }
                    }
                    Array::from_categorical32(new_arr)
                }
                #[cfg(feature = "default_categorical_8")]
                TextArray::Categorical8(_) => {
                    let mut values: Vec<&str> = Vec::with_capacity(indices.len());
                    for &idx in indices {
                        if let Some(val) = self.get_str(idx) {
                            values.push(val);
                        } else {
                            values.push("");
                        }
                    }
                    let mut new_arr = CategoricalArray::<u8>::from_vec(values, None);
                    for (i, &idx) in indices.iter().enumerate() {
                        if self.get_str(idx).is_none() {
                            new_arr.set_null(i);
                        }
                    }
                    Array::from_categorical8(new_arr)
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(_) => {
                    let mut values: Vec<&str> = Vec::with_capacity(indices.len());
                    for &idx in indices {
                        if let Some(val) = self.get_str(idx) {
                            values.push(val);
                        } else {
                            values.push("");
                        }
                    }
                    let mut new_arr = CategoricalArray::<u16>::from_vec(values, None);
                    for (i, &idx) in indices.iter().enumerate() {
                        if self.get_str(idx).is_none() {
                            new_arr.set_null(i);
                        }
                    }
                    Array::from_categorical16(new_arr)
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(_) => {
                    let mut values: Vec<&str> = Vec::with_capacity(indices.len());
                    for &idx in indices {
                        if let Some(val) = self.get_str(idx) {
                            values.push(val);
                        } else {
                            values.push("");
                        }
                    }
                    let mut new_arr = CategoricalArray::<u64>::from_vec(values, None);
                    for (i, &idx) in indices.iter().enumerate() {
                        if self.get_str(idx).is_none() {
                            new_arr.set_null(i);
                        }
                    }
                    Array::from_categorical64(new_arr)
                }
                TextArray::Null => Array::Null,
            },
            Array::BooleanArray(_) => {
                let mut new_arr = BooleanArray::with_capacity(indices.len(), true);
                for &idx in indices {
                    if let Some(val) = self.get::<BooleanArray<()>>(idx) {
                        new_arr.push(val);
                    } else {
                        new_arr.push_null();
                    }
                }
                Array::from_bool(new_arr)
            }
            #[cfg(feature = "datetime")]
            Array::TemporalArray(temp_arr) => match temp_arr {
                TemporalArray::Datetime32(arr) => {
                    let mut new_arr = DatetimeArray::<i32>::with_capacity(
                        indices.len(),
                        true,
                        Some(arr.time_unit),
                    );
                    for &idx in indices {
                        if let Some(val) = self.get::<DatetimeArray<i32>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_datetime_i32(new_arr)
                }
                TemporalArray::Datetime64(arr) => {
                    let mut new_arr = DatetimeArray::<i64>::with_capacity(
                        indices.len(),
                        true,
                        Some(arr.time_unit),
                    );
                    for &idx in indices {
                        if let Some(val) = self.get::<DatetimeArray<i64>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_datetime_i64(new_arr)
                }
                TemporalArray::Null => Array::Null,
            },
        }
    }

    /// Returns a pointer and metadata for raw access
    ///
    /// This is not logical length - it is total raw bytes in the buffer,
    /// so for non-fixed width types such as bit-packed booleans
    /// or strings, please factor this in accordingly.
    #[inline]
    pub fn data_ptr_and_byte_len(&self) -> (*const u8, usize, usize) {
        let (ptr, _total_len, elem_size) = self.array.data_ptr_and_byte_len();
        let windowed_ptr = unsafe { ptr.add(self.offset * elem_size) };
        (windowed_ptr, self.len, elem_size)
    }

    /// Returns the exclusive end index of the window (relative to parent array).
    #[inline]
    pub fn end(&self) -> usize {
        self.offset + self.len
    }

    /// Returns the underlying window as a tuple: (Array, offset, len).
    ///
    /// Note: This clones the Arc-wrapped Array.
    #[inline]
    pub fn as_tuple(&self) -> (Array, usize, usize) {
        (self.array.clone(), self.offset, self.len) // arc clone
    }

    /// Returns a reference tuple: (&Array, offset, len).
    ///
    /// This avoids cloning the Arc and returns a reference with a lifetime
    /// tied to this ArrayV.
    #[inline]
    pub fn as_tuple_ref(&self) -> (&Array, usize, usize) {
        (&self.array, self.offset, self.len)
    }

    /// Returns the null count in the window, caching the result after first calculation.
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

    /// Returns a windowed view over the underlying null mask, if any.
    #[inline]
    pub fn null_mask_view(&self) -> Option<BitmaskV> {
        self.array
            .null_mask()
            .map(|mask| mask.view(self.offset, self.len))
    }

    /// Set the cached null count (advanced use only).
    ///
    /// Returns Ok(()) if the value was set, or Err(count) if it was already initialized.
    /// This is thread-safe and can only succeed once per ArrayV instance.
    #[inline]
    pub fn set_null_count(&self, count: usize) -> Result<(), usize> {
        self.null_count.set(count).map_err(|_| count)
    }
}

/// Array -> ArrayView
///
/// Uses Offset 0 and length self.len()
impl From<Array> for ArrayV {
    fn from(array: Array) -> Self {
        let len = array.len();
        let null_count = array.null_count();
        ArrayV {
            array,
            offset: 0,
            len,
            null_count: null_count.into(),
        }
    }
}

/// ArrayView -> Array
///
/// Delegates to `to_array`, which Arc-bumps the underlying allocation when the
/// view spans its full backing array (offset = 0, len = array.len()) and only
/// reallocates via `slice_clone` for genuinely windowed views.
impl From<ArrayV> for Array {
    fn from(view: ArrayV) -> Self {
        view.to_array()
    }
}

/// FieldArray -> ArrayView
///
/// Takes self.array then offset 0, length self.len())
impl From<FieldArray> for ArrayV {
    fn from(field_array: FieldArray) -> Self {
        let len = field_array.len();
        let null_count = field_array.null_count();
        ArrayV {
            array: field_array.array,
            offset: 0,
            len,
            null_count: null_count.into(),
        }
    }
}

/// &FieldArray -> ArrayView
///
/// Arc bumps inner array with offset 0, length self.len().
impl From<&FieldArray> for ArrayV {
    fn from(field_array: &FieldArray) -> Self {
        let len = field_array.len();
        let null_count = field_array.null_count();
        ArrayV {
            array: field_array.array.clone(),
            offset: 0,
            len,
            null_count: null_count.into(),
        }
    }
}

/// NumericArrayView -> ArrayView
///
/// Converts by wrapping the inner NumericArray as Array::NumericArray.
#[cfg(feature = "views")]
impl From<crate::NumericArrayV> for ArrayV {
    fn from(view: crate::NumericArrayV) -> Self {
        let len = view.len();
        ArrayV::new(Array::NumericArray(view.array), view.offset, len)
    }
}

/// TextArrayView -> ArrayView
///
/// Converts by wrapping the inner TextArray as Array::TextArray.
#[cfg(feature = "views")]
impl From<crate::TextArrayV> for ArrayV {
    fn from(view: crate::TextArrayV) -> Self {
        let len = view.len();
        ArrayV::new(Array::TextArray(view.array), view.offset, len)
    }
}

/// TemporalArrayView -> ArrayView
///
/// Converts by wrapping the inner TemporalArray as Array::TemporalArray.
#[cfg(all(feature = "views", feature = "datetime"))]
impl From<crate::TemporalArrayV> for ArrayV {
    fn from(view: crate::TemporalArrayV) -> Self {
        let len = view.len();
        ArrayV::new(Array::TemporalArray(view.array), view.offset, len)
    }
}

/// BooleanArrayView -> ArrayView
///
/// Converts by wrapping the inner Arc<BooleanArray> as Array::BooleanArray.
#[cfg(feature = "views")]
impl From<crate::BooleanArrayV> for ArrayV {
    fn from(view: crate::BooleanArrayV) -> Self {
        let len = view.len();
        ArrayV::new(Array::BooleanArray(view.array), view.offset, len)
    }
}

/// Scalar -> ArrayView
///
/// Converts a Scalar to a length-1 ArrayV, enabling scalar broadcasting
/// in functions that accept `impl Into<ArrayV>`.
#[cfg(feature = "scalar_type")]
impl From<crate::Scalar> for ArrayV {
    fn from(scalar: crate::Scalar) -> Self {
        let array = scalar.array_from_value(1);
        ArrayV::new(array, 0, 1)
    }
}

// We do not implement `Index` as `ArrayView` cannot safely return
// a reference to an element.

impl Debug for ArrayV {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ArrayView")
            .field("offset", &self.offset)
            .field("len", &self.len)
            .field("array", &self.array)
            .field("cached_null_count", &self.null_count.get())
            .finish()
    }
}

impl Display for ArrayV {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let nulls = self.null_count();
        let head_len = self.len.min(MAX_PREVIEW);

        writeln!(
            f,
            "ArrayView [{} values] (offset: {}, nulls: {})",
            self.len, self.offset, nulls
        )?;

        // Delegate to the inner array's Display by formatting a slice of it
        let sliced_array = self.array.slice_clone(self.offset, head_len);

        for line in format!("{}", sliced_array).lines() {
            writeln!(f, "  {line}")?;
        }

        if self.len > MAX_PREVIEW {
            writeln!(f, "  ... ({} more rows)", self.len - MAX_PREVIEW)?;
        }

        Ok(())
    }
}

impl Shape for ArrayV {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank1(self.len())
    }
}

impl Concatenate for ArrayV {
    /// Concatenates two array views by materialising both to owned arrays,
    /// concatenating them, and wrapping the result back in a view.
    ///
    /// # Notes
    /// - This operation copies data from both views to create owned arrays.
    /// - The resulting view has offset=0 and length equal to the combined length.
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        // Materialise both views to owned arrays
        let self_array = self.to_array();
        let other_array = other.to_array();

        // Concatenate the owned arrays
        let concatenated = self_array.concat(other_array)?;

        // Wrap the result in a new view
        Ok(ArrayV::from(concatenated))
    }
}

// ===== Selection Trait Implementation =====

#[cfg(feature = "select")]
impl RowSelection for ArrayV {
    type View = ArrayV;

    fn r<S: DataSelector>(&self, selection: S) -> ArrayV {
        if selection.is_contiguous() {
            // Contiguous selection (ranges): adjust offset and len
            let indices = selection.resolve_indices(self.len());
            if indices.is_empty() {
                return ArrayV::new(self.array.clone(), self.offset, 0);
            }
            let new_offset = self.offset + indices[0];
            let new_len = indices.len();
            ArrayV::new(self.array.clone(), new_offset, new_len)
        } else {
            // Non-contiguous selection (index arrays): gather into new array
            let indices = selection.resolve_indices(self.len());
            let gathered_array = self.gather_indices(&indices);
            ArrayV::new(gathered_array, 0, indices.len())
        }
    }

    fn get_row_count(&self) -> usize {
        self.len
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{Array, Bitmask, IntegerArray, NumericArray, vec64};

    #[test]
    fn test_array_view_basic_indexing_and_slice() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(11);
        arr.push(22);
        arr.push(33);
        arr.push(44);

        let array = Array::NumericArray(NumericArray::Int32(Arc::new(arr)));
        let view = ArrayV::new(array, 1, 2);

        // Basic indexing within window
        assert_eq!(view.len(), 2);
        assert_eq!(view.offset, 1);
        assert_eq!(view.get::<IntegerArray<i32>>(0), Some(22));
        assert_eq!(view.get::<IntegerArray<i32>>(1), Some(33));
        assert_eq!(view.get::<IntegerArray<i32>>(2), None);

        // Slicing the view produces the correct sub-window
        let sub = view.slice(1, 1);
        assert_eq!(sub.len(), 1);
        assert_eq!(sub.get::<IntegerArray<i32>>(0), Some(33));
        assert_eq!(sub.get::<IntegerArray<i32>>(1), None);
    }

    #[test]
    fn test_array_view_null_count_and_cache() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(1);
        arr.push(2);
        arr.push(3);
        arr.push(4);

        // Null mask: only index 2 is null
        let mut mask = Bitmask::new_set_all(4, true);
        mask.set(2, false);
        arr.null_mask = Some(mask);

        let array = Array::NumericArray(NumericArray::Int32(Arc::new(arr)));

        let view = ArrayV::new(array, 0, 4);
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
    fn test_array_view_with_supplied_null_count() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(5);
        arr.push(6);

        let array = Array::NumericArray(NumericArray::Int32(Arc::new(arr)));
        let view = ArrayV::new_nc(array, 0, 2, 99);
        // Should always report the supplied cached value
        assert_eq!(view.null_count(), 99);
        // Trying to set again should fail since it's already initialized
        assert!(view.set_null_count(101).is_err());
        // Still returns original value
        assert_eq!(view.null_count(), 99);
    }

    #[test]
    fn test_array_view_to_array_and_as_tuple() {
        let mut arr = IntegerArray::<i32>::default();
        for v in 10..20 {
            arr.push(v);
        }
        let array = Array::NumericArray(NumericArray::Int32(Arc::new(arr)));
        let view = ArrayV::new(array.clone(), 4, 3);
        let arr2 = view.to_array();
        // Copy should be [14, 15, 16]
        if let Array::NumericArray(NumericArray::Int32(a2)) = arr2 {
            assert_eq!(a2.data, vec64![14, 15, 16]);
        } else {
            panic!("Unexpected variant");
        }

        // as_tuple returns correct metadata
        let tup = view.as_tuple();
        assert_eq!(&tup.0, &array);
        assert_eq!(tup.1, 4);
        assert_eq!(tup.2, 3);
    }

    #[test]
    fn test_array_view_null_mask_view() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(2);
        arr.push(4);
        arr.push(6);

        let mut mask = Bitmask::new_set_all(3, true);
        mask.set(0, false);
        arr.null_mask = Some(mask);

        let array = Array::NumericArray(NumericArray::Int32(Arc::new(arr)));
        let view = ArrayV::new(array.clone(), 1, 2);
        let mask_view = view.null_mask_view().expect("Should have mask");
        assert_eq!(mask_view.len(), 2);
        // Should map to bits 1 and 2 of original mask
        assert!(mask_view.get(0));
        assert!(mask_view.get(1));
    }
}
