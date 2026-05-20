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

//! # **MaskedArray Module** - *Standardises all Inner Array types and null handling in Minarrow*
//!
//! Defines the `MaskedArray` trait - the common interface for all nullable array types in Minarrow.
//!
//! This module standardises how arrays store and manage optional null bitmasks,
//! ensuring consistent null-handling behaviour across fixed-width and variable-length arrays.
//! It also provides default implementations for common mask operations to reduce duplication.

use crate::{Bitmask, Length, Offset, enums::error::MinarrowError};

/// # MaskedArray
///
/// MaskedArray is implemented by all inner, nullable arrays.
///
/// ### Purpose
/// - MaskedArray ensures interface consistency across `BooleanArray`,
/// `CategoricalArray`, `DatetimeArray`, `FloatArray`, `IntegerArray`
/// and `StringArray`.
/// - It avoids repeition through default boilerplate implementations,
/// focusing on null value handling.
/// - This serves to enforce the base pattern contract, and is either overriden
/// on non-fixed width types (e.g., `BooleanArray`, `StringArray`), or, for fixed
/// width types (e.g., `FloatArray`, `IntegerArray`), is supported by macros.
pub trait MaskedArray {
    /// The element type (e.g. `f32`, `bool`, etc.)
    /// Or, utility type e.g., `Offsets` for cases
    /// like `String`
    type T: Default + PartialEq + Clone + Copy;

    /// The backing store (e.g. `Vec64<Self::Elem>` or `Bitmask`)
    type Container;

    /// The logical type that the data carries
    type LogicalType: Default;

    /// The element type returned by `get`/`iter` etc.
    ///
    /// Modelled as a generic associated type so that string-like arrays
    /// can return `&'a str` borrowed from `&self`, instead of widening the
    /// borrow to `'static` (which would be unsound). For fixed-width types
    /// like numerics the lifetime is unused and the alias resolves to the
    /// element type itself, e.g. `type CopyType<'a> = T`.
    type CopyType<'a>: Default
    where
        Self: 'a;

    /// **************************************************
    /// The below methods differ for the Boolean (bit-packed),
    /// and String (variable-length) variants and thus are
    /// implemented via macros for the standard variants,
    /// and then implemented on those types directly
    /// *************************************************

    /// Returns the number of elements in the array.
    fn len(&self) -> usize;

    /// Returns true if the array is empty.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a reference to the underlying data.
    fn data(&self) -> &Self::Container;

    /// Returns a mutable reference to the underlying data.
    fn data_mut(&mut self) -> &mut Self::Container;

    /// Retrieves the value at the given index, or None if null or beyond length.
    fn get(&self, idx: usize) -> Option<Self::CopyType<'_>>;

    /// Sets the value at the given index, updating the null‐mask.
    fn set(&mut self, idx: usize, value: Self::LogicalType);

    /// Like `get`, but skips the `idx >= len()` check.
    unsafe fn get_unchecked(&self, idx: usize) -> Option<Self::CopyType<'_>>;

    /// Like `set`, but skips bounds checks.
    unsafe fn set_unchecked(&mut self, idx: usize, value: Self::LogicalType);

    /// Low-level accessor for when working directly with
    /// mutable array variants.
    ///
    /// Borrows with window parameters as a tuple,
    /// for 'DIY' window access, retaining access to the whole original array.
    ///
    /// `Offset` and `Length` are `usize` aliases.
    ///
    /// For the standard zero-copy accessors, see the `View` trait.
    fn tuple_ref(&self, offset: usize, len: usize) -> (&Self, Offset, Length) {
        (&self, offset, len)
    }

    /// Returns an iterator over the T values in this array.
    fn iter(&self) -> impl Iterator<Item = Self::CopyType<'_>> + '_;

    /// Returns an iterator over the T values, as `Option<Self::T>`.
    fn iter_opt(&self) -> impl Iterator<Item = Option<Self::CopyType<'_>>> + '_;

    /// Returns an iterator over a range of T values in this array.
    fn iter_range(
        &self,
        offset: usize,
        len: usize,
    ) -> impl Iterator<Item = Self::CopyType<'_>> + '_;

    /// Returns an iterator over a range of T values, as `Option<T>`.
    fn iter_opt_range(
        &self,
        offset: usize,
        len: usize,
    ) -> impl Iterator<Item = Option<Self::CopyType<'_>>> + '_;

    /// Appends a value to the array, updating masks if present.
    fn push(&mut self, value: Self::LogicalType);

    /// Appends a value to the array, updating masks if present,
    /// without bounds checks.
    ///
    /// # Safety
    /// The caller must make sure there is enough pre-allocated
    /// size in the array, and no thread contention.
    unsafe fn push_unchecked(&mut self, value: Self::LogicalType);

    /// Returns a logical slice of the MaskedArray<Self::T> [offset, offset+len)
    /// as a new MaskedArray<Self::T> object via clone.
    ///
    /// Prefer `View` trait slicers for zero-copy.
    fn slice_clone(&self, offset: usize, len: usize) -> Self;

    /// Resizes the array to contain `n` elements, via a call into self.data.resize().
    ///
    /// If `n` is greater than the current length, new elements are added using `T::default()`.
    /// If `n` is smaller, the array is truncated. This only affects the data buffer,
    /// not the null mask.
    fn resize(&mut self, n: usize, value: Self::LogicalType);

    /// **************************************************
    ///  We handle null masks consistently across all variants
    /// and thus their implementation sits on the trait, other
    /// than trait methods that need to access data state.
    /// **************************************************

    /// Returns a reference to the optional null mask.
    fn null_mask(&self) -> Option<&Bitmask>;

    /// Returns true if the value at the given index is null.
    #[inline]
    fn is_null(&self, idx: usize) -> bool {
        match &self.null_mask() {
            Some(mask) => !mask.get(idx),
            None => false,
        }
    }

    /// Checks if the array has a null bitmask.
    fn is_nullable(&self) -> bool {
        self.null_mask().is_some()
    }

    /// Returns the total number of nulls.
    fn null_count(&self) -> usize {
        match self.null_mask().as_ref() {
            Some(mask) => mask.count_zeros(),
            None => 0,
        }
    }

    /// Returns true when the array has at least one null.
    ///
    /// Fast path: absent mask short-circuits to `false` with no popcount.
    /// A present mask defers to `Bitmask::has_nulls`, which only traverses the
    /// bits needed to identify a cleared entry.
    fn has_nulls(&self) -> bool {
        self.null_mask().is_some_and(|m| m.has_nulls())
    }

    /// Append a null value to the array, creating mask if needed
    #[inline]
    fn push_null(&mut self) {
        self.push(Self::LogicalType::default());
        let i = self.len() - 1;
        match self.null_mask_mut() {
            Some(m) => m.set(i, false),
            None => {
                let mut m = Bitmask::new_set_all(self.len(), true);
                m.set(i, false);
                self.set_null_mask(Some(m));
            }
        }
    }

    /// Returns a mutable reference to the optional null mask.
    fn null_mask_mut(&mut self) -> Option<&mut Bitmask>;

    /// Sets the null mask.
    fn set_null_mask(&mut self, mask: Option<Bitmask>);

    /// Appends a null value _without_ any bounds‐checks on the mask.
    ///
    /// # Safety
    /// You must ensure that after `push`, the data and mask (if present)
    /// have capacity for the new index, or you risk OOB on either.
    #[inline(always)]
    unsafe fn push_null_unchecked(&mut self) {
        // first, append a default element
        let idx = self.len();
        unsafe { self.set_unchecked(idx, Self::LogicalType::default()) };

        if let Some(mask) = self.null_mask_mut() {
            // mark null
            unsafe { mask.set_unchecked(idx, false) };
        } else {
            // initialise a new mask and mark this slot null
            let mut m = Bitmask::new_set_all(idx, true);
            unsafe { m.set_unchecked(idx, false) };
            self.set_null_mask(Some(m));
        }
    }

    /// Marks the value at the given index as null.
    #[inline]
    fn set_null(&mut self, idx: usize) {
        if let Some(nmask) = &mut self.null_mask_mut() {
            if nmask.len() <= idx {
                nmask.resize(idx + 1, true);
            }
            nmask.set(idx, false);
        } else {
            let mut m = Bitmask::new_set_all(self.len(), true);
            m.set(idx, false);
            self.set_null_mask(Some(m));
        }
    }

    /// Like `set_null`, but skips bounds checks.
    #[inline(always)]
    unsafe fn set_null_unchecked(&mut self, idx: usize) {
        if let Some(mask) = self.null_mask_mut() {
            mask.set(idx, false);
        } else {
            let mut m = Bitmask::new_set_all(self.len(), true);
            m.set(idx, false);
            self.set_null_mask(Some(m));
        }
    }

    /// Bulk-extend this array with `n` null entries
    #[inline]
    fn push_nulls(&mut self, n: usize) {
        let start = self.len();
        let end = start + n;

        self.resize(end, Self::LogicalType::default());

        if let Some(mask) = self.null_mask_mut() {
            mask.set_range(start, end, false);
        } else {
            let mut m = Bitmask::new_set_all(end, true);
            m.set_range(start, end, false);
            self.set_null_mask(Some(m));
        }
    }

    /// Bulk-extend this array with `n` null entries, using unchecked mask writes.
    ///
    /// # Safety
    /// Caller must ensure there are no data races across threads on this mask.
    #[inline(always)]
    unsafe fn push_nulls_unchecked(&mut self, n: usize) {
        let start = self.len();
        let end = start + n;

        self.resize(end, Self::LogicalType::default());

        if let Some(mask) = self.null_mask_mut() {
            mask.resize(end, true);
            for i in 0..n {
                unsafe { mask.set_unchecked(start + i, false) };
            }
        } else {
            let mut m = Bitmask::new_set_all(end, true);
            for i in start..end {
                unsafe { m.set_unchecked(i, false) };
            }
            self.set_null_mask(Some(m));
        }
    }

    /// Appends all values (and null mask if present) from `other` to `self`.
    ///
    /// The appended array must be of the same concrete type and element type.
    ///
    /// If this array is wrapped in a `FieldArray`, it will not be possible to
    /// mutate the array without reconstructing first, and a `ChunkedArray`
    /// is an alternative option.
    fn append_array(&mut self, other: &Self);

    /// Appends rows `[offset..offset+len)` from another array into self.
    ///
    /// Like `append_array` but for a sub-range. Data and null masks are
    /// extended from the source range. The destination grows via its
    /// backing allocator.
    fn append_range(
        &mut self,
        other: &Self,
        offset: usize,
        len: usize,
    ) -> Result<(), MinarrowError>;

    /// Inserts all values (and null mask if present) from `other` into `self` at the specified index.
    ///
    /// The inserted array must be of the same concrete type and element type.
    /// Elements at and after `index` are shifted to make room for the inserted values.
    ///
    /// # Performance
    /// This is an **O(n)** operation, where n is the number of elements that need to be shifted.
    /// For appending to the end, prefer `append_array` which is more efficient.
    ///
    /// # Arguments
    /// * `index` - Position before which to insert (0 = prepend, self.len() = append)
    /// * `other` - Array to insert
    ///
    /// # Errors
    /// Returns an error if:
    /// * `index > self.len()` (out of bounds)
    /// * Type mismatch between self and other (for enum variants)
    ///
    /// # Example
    /// ```ignore
    /// let mut arr = IntegerArray::from(vec![1, 2, 5, 6]);
    /// let insert = IntegerArray::from(vec![3, 4]);
    /// arr.insert_rows(2, &insert)?; // Now: [1, 2, 3, 4, 5, 6]
    /// ```
    fn insert_rows(&mut self, index: usize, other: &Self) -> Result<(), MinarrowError>;

    /// Splits this array at the specified index, consuming self and returning two arrays.
    ///
    /// The original array's memory is split between the two resulting arrays without copying.
    /// The first array contains elements [0..index), the second contains [index..len).
    ///
    /// # Arguments
    /// * `index` - Position to split at (0 < index < len)
    ///
    /// # Errors
    /// - Returns an error if index == 0 or index >= len()
    ///
    /// # Returns
    /// A tuple of (before, after) where before contains [0..index) and after contains [index..len)
    fn split(self, index: usize) -> Result<(Self, Self), MinarrowError>
    where
        Self: Sized;

    /// Extends the array from an iterator with pre-allocated capacity.
    ///
    /// Pre-allocates the specified additional capacity to avoid reallocations during bulk insertion,
    /// providing optimal performance for large datasets where the final size is known in advance.
    fn extend_from_iter_with_capacity<I>(&mut self, iter: I, additional_capacity: usize)
    where
        I: Iterator<Item = Self::LogicalType>;

    /// Extends the array from a slice of values.
    ///
    /// More efficient than individual `push` operations as it pre-allocates capacity
    /// and can use bulk copy operations for compatible data types. For variable-length
    /// types like strings, calculates total byte requirements upfront.
    fn extend_from_slice(&mut self, slice: &[Self::LogicalType]);

    /// Creates a new array filled with the specified value repeated `count` times.
    ///
    /// Pre-allocates exact capacity to avoid reallocations and uses
    /// efficient bulk operations where possible. For string types,
    /// calculates total byte requirements to minimise memory overhead.
    fn fill(value: Self::LogicalType, count: usize) -> Self;
}
