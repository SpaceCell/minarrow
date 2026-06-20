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

//! # **CategoricalArray Module** - *Mid-Level, Inner Typed Categorical Array*
//!
//! CategoricalArray uses dictionary-encoded strings where each row stores a
//! small integer “code” that references a per-column dictionary of unique strings.
//! This saves memory and accelerates comparisons/joins when many values repeat.
//!
//! ## Interop
//! - Arrow-compatible dictionary layout (`indices` + string `dictionary`), and
//!   round-trips over the Arrow C Data Interface to/from the `Dictionary` array type.
//! - Index width is the generic `T` (e.g., `u8/u16/u32/u64`) and corresponds to
//!   Arrow’s `CategoricalIndexType`.
//!
//! ## Features
//! - Optional `null_mask`: bit-packed, where `1 = valid`, `0 = null`
//! - Builders from raw values (`from_values`, `from_vec64`) and from raw parts.
//! - Iterators over indices and over resolved strings (nullable and non-nullable).
//! - Convert to a dense `StringArray` via `to_string_array()` when needed.
//! - Parallel helpers behind `parallel_proc` feature.
//!
//! ## When to use
//! Use for arrays with repeated strings to reduce memory and speed up operations.

use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::slice::{Iter, IterMut};

#[cfg(feature = "parallel_proc")]
use rayon::iter::ParallelIterator;

use crate::aliases::CategoricalAVT;
use crate::enums::error::MinarrowError;
use crate::enums::shape_dim::ShapeDim;
#[cfg(feature = "shared_dict")]
use crate::structs::dictionary::Dictionary;
use crate::traits::concatenate::Concatenate;
use crate::traits::shape::Shape;
use crate::traits::type_unions::Integer;
use crate::utils::validate_null_mask_len;
use crate::{
    Bitmask, Buffer, Length, MaskedArray, Offset, StringArray, impl_arc_masked_array,
    impl_array_ref_deref,
};
use ::vec64::{Vec64, Vec64Alloc};

/// Without `shared_dict`, returns the existing code for `value` if it
/// is already in `unique_values`, else pushes it and returns the new
/// code. Linear scan. Panics if the new cardinality would exceed the
/// capacity of `T`.
#[cfg(not(feature = "shared_dict"))]
#[inline]
fn add_category<T: Integer>(unique_values: &mut Vec64<String>, value: &str) -> T {
    if let Some(pos) = unique_values.iter().position(|s| s.as_str() == value) {
        return T::from_usize(pos);
    }
    let i = unique_values.len();
    let c = T::try_from(i).ok().unwrap_or_else(|| {
        panic!(
            "Categorical cardinality exceeded the capacity of the index \
             type {}. Consider a wider index width.",
            std::any::type_name::<T>()
        )
    });
    unique_values.push(value.to_owned());
    c
}

/// # CategoricalArray
///
/// Categorical array with unique string instances mapped to indices.
///
/// ## Role
/// - Many will prefer the higher level `Array` type, which dispatches to this when
/// necessary.
/// - Can be used as a standalone text array or as the text arm of `TextArray` / `Array`.
///
/// ## Description
/// Compatible with the `Arrow Dictionary` memory layout, where each value is
/// represented as an index into a dictionary of unique strings, and materialises
/// into the format over FFI.
///
/// ### Fields:
/// - `data`: indices buffer referencing entries in `unique_values`.
/// - `unique_values`: dictionary of unique string values.
/// - `null_mask`: optional bit-packed validity bitmap (1=valid, 0=null).
///
/// ## Purpose
/// Consider this when you have a common set of unique string values, and want to
/// save space and increase speed by storing the string values only once
/// *(in the `unique_values` Vec)*, and then only the integers that map to them
/// in the `data` field.
///
/// ## Example
/// ```rust
/// use minarrow::{CategoricalArray, MaskedArray};
///
/// let arr = CategoricalArray::<u8>::from_values(vec!["apple", "banana", "apple", "cherry"]);
/// assert_eq!(arr.len(), 4);
///
/// // Indices into the unique_values dictionary
/// assert_eq!(arr.indices(), &[0u8, 1, 0, 2]);
///
/// // Dictionary of unique values
/// assert_eq!(arr.unique_values(), &["apple".to_string(), "banana".to_string(), "cherry".to_string()]);
///
/// // Resolved value lookups
/// assert_eq!(arr.get_str(0), Some("apple"));
/// assert_eq!(arr.get_str(1), Some("banana"));
/// assert_eq!(arr.get_str(2), Some("apple"));
/// assert_eq!(arr.get_str(3), Some("cherry"));
/// ```
#[repr(C, align(64))]
#[derive(PartialEq, Clone, Debug, Default)]
pub struct CategoricalArray<T: Integer> {
    /// Indices buffer (references into the dictionary).
    pub data: Buffer<T>,
    /// Dictionary values, i.e., the unique strings indexed by `data`.
    #[cfg(not(feature = "shared_dict"))]
    pub unique_values: Vec64<String>,
    /// When the `shared_dict` feature is on, a shared dictionary
    /// reference is used to ensure that categories remain aligned across
    /// related categorical arrays.
    #[cfg(feature = "shared_dict")]
    pub dictionary: Dictionary<T>,
    /// Optional null mask (bit-packed; 1=valid, 0=null).
    pub null_mask: Option<Bitmask>,
}

impl<T: Integer> CategoricalArray<T> {
    /// Constructs a new CategoricalArray
    #[inline]
    pub fn new(
        data: impl Into<Buffer<T>>,
        unique_values: Vec64<String>,
        null_mask: Option<Bitmask>,
    ) -> Self {
        let data: Buffer<T> = data.into();

        validate_null_mask_len(data.len(), &null_mask);
        // Per the Arrow spec, values at null positions are unspecified, so we
        // skip them here. Pandas, for instance, writes a sentinel (-1, which
        // wraps to 255 for u8 indices) into null slots when exporting a
        // Categorical over the C Data Interface.
        for (i, code) in data.iter().enumerate() {
            let is_valid = null_mask.as_ref().map_or(true, |m| m.get(i));
            if !is_valid {
                continue;
            }
            let idx = code
                .to_usize()
                .unwrap_or_else(|| panic!("Failed to convert code to usize at position {}", i));
            assert!(
                idx < unique_values.len(),
                "Index {} out of bounds for unique_values (len = {}) at position {}",
                idx,
                unique_values.len(),
                i
            );
        }

        Self {
            data,
            #[cfg(not(feature = "shared_dict"))]
            unique_values,
            #[cfg(feature = "shared_dict")]
            dictionary: Dictionary::from(unique_values),
            null_mask,
        }
    }

    /// Constructs a `CategoricalArray` that joins an existing dictionary's
    /// sharing group. The provided `Dictionary` is cloned (Arc bump), so
    /// the resulting array's codes are mutually meaningful with every
    /// other array sharing that dictionary. Used by streaming batch
    /// consolidation and by FFI imports that have already deduplicated
    /// dictionaries upstream.
    #[cfg(feature = "shared_dict")]
    #[inline]
    pub fn new_existing_dict(
        data: impl Into<Buffer<T>>,
        dictionary: Dictionary<T>,
        null_mask: Option<Bitmask>,
    ) -> Self {
        let data: Buffer<T> = data.into();
        validate_null_mask_len(data.len(), &null_mask);
        Self {
            data,
            dictionary,
            null_mask,
        }
    }

    /// Construct an empty categorical with reserved capacity for `cap` indices.
    /// Pass `unique_values` to pre-populate the dictionary, or `None` for an
    /// empty one.
    #[inline]
    pub fn with_capacity(
        cap: usize,
        unique_values: Option<Vec64<String>>,
        null_mask: bool,
    ) -> Self {
        Self {
            data: Vec64::with_capacity(cap).into(),
            #[cfg(not(feature = "shared_dict"))]
            unique_values: unique_values.unwrap_or_default(),
            #[cfg(feature = "shared_dict")]
            dictionary: unique_values.map(Dictionary::from).unwrap_or_default(),
            null_mask: if null_mask {
                // All-valid (1) default - reserved validity slots default to
                // valid under Arrow's 1=valid, 0=null convention.
                Some(Bitmask::new_set_all(cap, true))
            } else {
                None
            },
        }
    }

    /// Build a categorical column from raw string values, auto-deriving the dictionary.
    #[inline]
    pub fn from_vec64(values: Vec64<&str>, null_mask: Option<Bitmask>) -> Self {
        validate_null_mask_len(values.len(), &null_mask);

        let len = values.len();
        let mut codes = Vec64::with_capacity(len);
        let mut unique_values: Vec64<String> = Vec64::new();
        let mut dict = HashMap::new();

        for (i, s) in values.into_iter().enumerate() {
            // nulls get the default code, but do not participate in the dictionary
            let is_valid = null_mask.as_ref().map_or(true, |m| m.get(i));
            if !is_valid {
                codes.push(T::default());
                continue;
            }

            if let Some(&code) = dict.get(&s) {
                codes.push(code);
            } else {
                let idx = unique_values.len();
                let code = T::try_from(idx).ok().unwrap_or_else(|| {
                    panic!(
                        "Unique category count ({}) exceeds capacity of index type {}",
                        idx + 1,
                        std::any::type_name::<T>()
                    )
                });
                unique_values.push(s.to_string());
                dict.insert(s, code);
                codes.push(code);
            }
        }

        Self {
            data: codes.into(),
            #[cfg(not(feature = "shared_dict"))]
            unique_values,
            #[cfg(feature = "shared_dict")]
            dictionary: Dictionary::from(unique_values),
            null_mask,
        }
    }

    /// Vec wrapper
    #[inline]
    pub fn from_vec(values: Vec<&str>, null_mask: Option<Bitmask>) -> Self {
        Self::from_vec64(values.into(), null_mask)
    }

    /// Constructs a new CategoricalArray without validation. The caller must ensure consistency.
    #[inline]
    pub fn new_unchecked(
        data: Vec64<T>,
        unique_values: Vec64<String>,
        null_mask: Option<Bitmask>,
    ) -> Self {
        Self {
            data: data.into(),
            #[cfg(not(feature = "shared_dict"))]
            unique_values,
            #[cfg(feature = "shared_dict")]
            dictionary: Dictionary::from(unique_values),
            null_mask,
        }
    }

    /// Constructs a dense DictionaryArray from index and value slices (no nulls).
    #[inline]
    pub fn from_slices(indices: &[T], unique_values: &[String]) -> Self {
        assert!(
            indices.iter().all(|&idx| {
                let i = idx.to_usize();
                i < unique_values.len()
            }),
            "All indices must be valid for unique_values"
        );
        let dict_values: Vec64<String> = Vec64(unique_values.to_vec_in(Vec64Alloc::default()));
        Self {
            data: Vec64(indices.to_vec_in(Vec64Alloc::default())).into(),
            #[cfg(not(feature = "shared_dict"))]
            unique_values: dict_values,
            #[cfg(feature = "shared_dict")]
            dictionary: Dictionary::from(dict_values),
            null_mask: None,
        }
    }

    /// Returns the current dictionary values as a slice.
    ///
    /// Under `shared_dict` the dictionary may be updated concurrently
    /// by other clones in the sharing group; this method returns the
    /// published prefix at the moment of the call.
    #[inline]
    pub fn unique_values(&self) -> &[String] {
        #[cfg(not(feature = "shared_dict"))]
        {
            &self.unique_values
        }
        #[cfg(feature = "shared_dict")]
        {
            self.dictionary.values()
        }
    }

    /// Returns the dictionary indices as a slice.
    ///
    /// Remember, the indices are the data,
    /// because the values are the unique Strings,
    /// in contrast to what a dictionary usually refers to.
    #[inline]
    pub fn indices(&self) -> &[T] {
        &self.data
    }

    /// Returns an iterator of dictionary indices (backing buffer).
    pub fn indices_iter(&self) -> Iter<'_, T> {
        self.data.iter()
    }

    /// Returns an iterator of dictionary values (unique strings).
    pub fn values_iter(&self) -> Iter<'_, String> {
        self.unique_values().iter()
    }

    /// Returns a mutable iterator over indices buffer.
    pub fn indices_iter_mut(&mut self) -> IterMut<'_, T> {
        self.data.iter_mut()
    }

    /// Returns a mutable iterator over dictionary values.
    ///
    /// Mutating an existing entry replaces the string that every code
    /// assigned at that position decodes to; codes against the old value
    /// no longer mean what they previously meant. For adding new values,
    /// `push_str` is the append-only path.
    ///
    /// Under `shared_dict` this categorical's dictionary is detached
    /// from its sharing group first, so sibling chunks and any parent
    /// `SuperArray` / `SuperTable` manager keep pointing at the
    /// original dictionary.
    pub fn values_iter_mut(&mut self) -> IterMut<'_, String> {
        #[cfg(not(feature = "shared_dict"))]
        {
            self.unique_values.iter_mut()
        }
        #[cfg(feature = "shared_dict")]
        {
            self.dictionary.detach_to_owned();
            self.dictionary
                .try_values_iter_mut()
                .expect("detach_to_owned just left this Arc unique")
        }
    }

    /// Extend with an iterator of &str.
    pub fn extend<'a, I: Iterator<Item = &'a str>>(&mut self, iter: I) {
        for s in iter {
            self.push(s.to_owned());
        }
    }

    /// Append string, adding to dictionary if new. Returns dictionary index used.
    #[inline]
    pub fn push_str(&mut self, value: &str) -> T {
        #[cfg(not(feature = "shared_dict"))]
        let code: T = add_category(&mut self.unique_values, value);
        #[cfg(feature = "shared_dict")]
        let code: T = self.dictionary.add_cat(value).expect(
            "Dictionary category interning failed: cardinality exceeded capacity \
             of the categorical integer. Consider a CategoricalArray<T> with a \
             greater `T` capacity.",
        );
        self.data.push(code);
        let row = self.len() - 1;
        if let Some(mask) = &mut self.null_mask {
            mask.set(row, true);
        }
        code
    }

    /// Appends a string without bounds checks, adding to the dictionary if new.
    ///
    /// # Safety
    /// - The caller must ensure `self.data` has sufficient capacity (i.e., already resized).
    /// - `self.null_mask`, if present, must also have space for this index.
    /// - This method assumes exclusive mutable access and no concurrent modification.
    #[inline(always)]
    pub unsafe fn push_str_unchecked(&mut self, value: &str) {
        let idx = self.data.len();
        unsafe { self.set_str_unchecked(idx, value) };
    }

    /// Retrieves the value at the given index, or None if null.
    #[inline]
    pub fn get_str(&self, idx: usize) -> Option<&str> {
        if self.is_null(idx) {
            return None;
        }
        let dict_idx = self.data[idx].to_usize();
        Some(&self.unique_values()[dict_idx])
    }

    /// Like `get`, but skips bounds checks.
    #[inline(always)]
    pub unsafe fn get_str_unchecked(&self, idx: usize) -> &str {
        if let Some(mask) = &self.null_mask {
            if !unsafe { mask.get_unchecked(idx) } {
                return "";
            }
        }
        let dict_idx = unsafe { self.data.get_unchecked(idx).to_usize().unwrap() };
        unsafe { self.unique_values().get_unchecked(dict_idx) }
    }

    /// Sets the value at `idx`. Marks as valid.
    #[inline]
    pub fn set_str(&mut self, idx: usize, value: &str) {
        assert!(idx < self.data.len(), "index out of bounds");

        #[cfg(not(feature = "shared_dict"))]
        let code: T = add_category(&mut self.unique_values, value);
        #[cfg(feature = "shared_dict")]
        let code: T = self.dictionary.add_cat(value).expect(
            "Dictionary category interning failed: cardinality exceeded capacity \
             of the categorical integer. Consider a CategoricalArray<T> with a \
             greater `T` capacity.",
        );

        self.data[idx] = code;

        if let Some(mask) = &mut self.null_mask {
            mask.set(idx, true);
        } else {
            let mut m = Bitmask::new_set_all(self.data.len(), false);
            m.set(idx, true);
            self.null_mask = Some(m);
        }
    }

    /// Like `set`, but skips all bounds checks.
    #[inline(always)]
    pub unsafe fn set_str_unchecked(&mut self, idx: usize, value: &str) {
        #[cfg(not(feature = "shared_dict"))]
        let code: T = add_category(&mut self.unique_values, value);
        #[cfg(feature = "shared_dict")]
        let code: T = self.dictionary.add_cat(value).expect(
            "Dictionary category interning failed: cardinality exceeded capacity \
             of the categorical integer. Consider a CategoricalArray<T> with a \
             greater `T` capacity.",
        );
        let data = self.data.as_mut_slice();
        data[idx] = code;
        if let Some(mask) = &mut self.null_mask {
            mask.set(idx, true);
        } else {
            let mut m = Bitmask::new_set_all(self.len(), false);
            m.set(idx, true);
            self.null_mask = Some(m);
        }
    }

    /// Returns an iterator of &str (nulls yielded as empty string).
    #[inline]
    pub fn iter_str(&self) -> impl Iterator<Item = &str> + '_ {
        self.data.iter().enumerate().map(move |(idx, &dict_idx)| {
            if self.is_null(idx) {
                ""
            } else {
                &self.unique_values()[dict_idx.to_usize()]
            }
        })
    }

    /// Returns an iterator of Option<&str>, None if value is null.
    #[inline]
    pub fn iter_str_opt(&self) -> impl Iterator<Item = Option<&str>> + '_ {
        self.data.iter().enumerate().map(move |(idx, &dict_idx)| {
            if self.is_null(idx) {
                None
            } else {
                Some(self.unique_values()[dict_idx.to_usize()].as_str())
            }
        })
    }

    /// Returns an iterator of `&str` values (nulls yield `""`) for a specified range.
    #[inline]
    pub fn iter_str_range(&self, offset: usize, len: usize) -> impl Iterator<Item = &str> + '_ {
        self.data[offset..offset + len]
            .iter()
            .enumerate()
            .map(move |(i, &dict_idx)| {
                let idx = offset + i;
                if self.is_null(idx) {
                    ""
                } else {
                    &self.unique_values()[dict_idx.to_usize()]
                }
            })
    }

    /// Returns an iterator of `Option<&str>` values for a specified range.
    #[inline]
    pub fn iter_str_opt_range(
        &self,
        offset: usize,
        len: usize,
    ) -> impl Iterator<Item = Option<&str>> + '_ {
        self.data[offset..offset + len]
            .iter()
            .enumerate()
            .map(move |(i, &dict_idx)| {
                let idx = offset + i;
                if self.is_null(idx) {
                    None
                } else {
                    Some(self.unique_values()[dict_idx.to_usize()].as_str())
                }
            })
    }

    /// Build from an iterator of &str in one pass.
    pub fn from_values<'a, I: IntoIterator<Item = &'a str>>(iter: I) -> Self {
        use std::collections::HashMap;
        let mut dict = Vec64::<String>::new();
        let mut map = HashMap::<&str, usize>::new();
        let mut idx_buf = Vec64::<T>::new();

        for s in iter {
            let pos = *map.entry(s).or_insert_with(|| {
                let i = dict.len();
                dict.push(s.to_owned());
                i
            });
            idx_buf.push(<T>::from_usize(pos));
        }

        Self {
            data: idx_buf.into(),
            #[cfg(not(feature = "shared_dict"))]
            unique_values: dict,
            #[cfg(feature = "shared_dict")]
            dictionary: Dictionary::from(dict),
            null_mask: None,
        }
    }

    /// Create from raw buffers (indices & dictionary) without copying.
    #[inline]
    pub fn from_parts(
        indices: Vec64<T>,
        unique_values: Vec64<String>,
        null_mask: Option<Bitmask>,
    ) -> Self {
        Self {
            data: indices.into(),
            #[cfg(not(feature = "shared_dict"))]
            unique_values,
            #[cfg(feature = "shared_dict")]
            dictionary: Dictionary::from(unique_values),
            null_mask,
        }
    }

    /// Materialise the categorical as a dense StringArray<T>.
    #[inline]
    pub fn to_string_array(&self) -> StringArray<T> {
        let len = self.data.len();
        let mut offsets = Vec64::with_capacity(len + 1);
        let mut data = Vec64::<u8>::new();
        offsets.push(T::zero());

        for i in 0..len {
            if self.is_null(i) {
                offsets.push(T::from(data.len()).unwrap());
            } else {
                let dict_idx = self.data[i].to_usize();
                let s = &self.unique_values()[dict_idx];
                data.extend_from_slice(s.as_bytes());
                offsets.push(T::from(data.len()).unwrap());
            }
        }

        StringArray {
            offsets: offsets.into(),
            data: data.into(),
            null_mask: self.null_mask.clone(),
        }
    }
}

impl<T: Integer> MaskedArray for CategoricalArray<T> {
    type T = T;

    type Container = Buffer<T>;

    type LogicalType = String;

    type CopyType<'a> = &'a str where Self: 'a;

    /// Removes the rows in `[start, end)`, shifting later rows left.
    /// The dictionary is unchanged: entries left unreferenced remain valid.
    ///
    /// # Panics
    /// Panics if `start > end` or `end > len`.
    fn delete_range(&mut self, start: usize, end: usize) {
        self.data.delete_range(start, end);
        if let Some(mask) = &mut self.null_mask {
            mask.delete_range(start, end);
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }

    fn data(&self) -> &Self::Container {
        &self.data
    }

    fn data_mut(&mut self) -> &mut Self::Container {
        &mut self.data
    }

    /// Retrieves the value at the given index, or `None` if null.
    ///
    /// The returned `&str` borrows from `self`, tied to the lifetime of `&self`
    /// via the trait's GAT `CopyType<'a>`.
    ///
    /// # Panics
    /// Panics if `idx >= self.len()` or if `data[idx]` is an invalid index into `unique_values`.
    #[inline]
    fn get(&self, idx: usize) -> Option<&str> {
        if self.is_null(idx) {
            return None;
        }

        let dict_idx = self.data[idx].to_usize();
        Some(&self.unique_values()[dict_idx])
    }

    /// Sets the value at `idx`. Marks as valid.
    ///
    /// Prefer `set_str` when you have a `&str` to avoid the `String` allocation.
    #[inline]
    fn set(&mut self, idx: usize, value: Self::LogicalType) {
        self.set_str(idx, &value)
    }

    /// Like `get`, but skips bounds checks on both the data and dictionary index.
    ///
    /// # Safety
    /// Caller must ensure:
    /// - `idx` is within bounds of `self.data`
    /// - `self.data[idx]` yields a valid index into `self.unique_values`
    #[inline]
    unsafe fn get_unchecked(&self, idx: usize) -> Option<&str> {
        if let Some(mask) = &self.null_mask {
            if !mask.get(idx) {
                return None;
            }
        }

        let dict_idx = unsafe { self.data.get_unchecked(idx).to_usize().unwrap() };
        Some(unsafe { self.unique_values().get_unchecked(dict_idx).as_str() })
    }

    /// Like `set`, but skips all bounds checks.
    ///
    /// Prefer `set_str_unchecked` when you have a `&str` to avoid the `String` allocation.
    #[inline]
    unsafe fn set_unchecked(&mut self, idx: usize, value: Self::LogicalType) {
        #[cfg(not(feature = "shared_dict"))]
        let code: T = add_category(&mut self.unique_values, &value);
        #[cfg(feature = "shared_dict")]
        let code: T = self.dictionary.add_cat(&value).expect(
            "Dictionary category interning failed: cardinality exceeded capacity \
             of the categorical integer. Consider a CategoricalArray<T> with a \
             greater `T` capacity.",
        );
        let data = self.data.as_mut_slice();
        data[idx] = code;
        if let Some(mask) = &mut self.null_mask {
            mask.set(idx, true);
        } else {
            let mut m = Bitmask::new_set_all(self.len(), false);
            m.set(idx, true);
            self.null_mask = Some(m);
        }
    }

    /// Returns an iterator of `&str` values borrowed from `self`.
    ///
    /// Nulls are represented as an empty string `""`.
    #[inline]
    fn iter(&self) -> impl Iterator<Item = &str> + '_ {
        self.data.iter().enumerate().map(move |(idx, &dict_idx)| {
            if self.is_null(idx) {
                ""
            } else {
                self.unique_values()[dict_idx.to_usize()].as_str()
            }
        })
    }

    /// Returns an iterator over `Option<&str>`, yielding `None` for nulls.
    ///
    /// The returned references borrow from `self`.
    #[inline]
    fn iter_opt(&self) -> impl Iterator<Item = Option<&str>> + '_ {
        self.data.iter().enumerate().map(move |(idx, &dict_idx)| {
            if self.is_null(idx) {
                None
            } else {
                Some(self.unique_values()[dict_idx.to_usize()].as_str())
            }
        })
    }

    /// Returns an iterator of `&str` values for a specified range.
    /// Nulls yield `""`.
    #[inline]
    fn iter_range(&self, offset: usize, len: usize) -> impl Iterator<Item = &str> + '_ {
        self.data[offset..offset + len]
            .iter()
            .enumerate()
            .map(move |(i, &dict_idx)| {
                let idx = offset + i;
                if self.is_null(idx) {
                    ""
                } else {
                    self.unique_values()[dict_idx.to_usize()].as_str()
                }
            })
    }

    /// Returns an iterator over `Option<&str>` values for a specified range.
    #[inline]
    fn iter_opt_range(
        &self,
        offset: usize,
        len: usize,
    ) -> impl Iterator<Item = Option<&str>> + '_ {
        self.data[offset..offset + len]
            .iter()
            .enumerate()
            .map(move |(i, &dict_idx)| {
                let idx = offset + i;
                if self.is_null(idx) {
                    None
                } else {
                    Some(self.unique_values()[dict_idx.to_usize()].as_str())
                }
            })
    }

    /// Append string, adding to dictionary if new.
    ///
    /// Prefer `push_str` when you have a `&str` to avoid the `String` allocation;
    /// it also returns the assigned dictionary code.
    #[inline]
    fn push(&mut self, value: Self::LogicalType) {
        self.push_str(&value);
    }

    /// Append string, adding to dictionary if new, without bounds checking.
    ///
    /// Prefer `push_str_unchecked` when you have a `&str` to avoid the `String` allocation.
    ///
    /// # Safety
    /// - The caller must ensure `self.data` has sufficient capacity (i.e., already resized).
    /// - `self.null_mask`, if present, must also have space for this index.
    /// - This method assumes exclusive mutable access and no concurrent modification.
    #[inline]
    unsafe fn push_unchecked(&mut self, value: Self::LogicalType) {
        self.push_str(&value);
    }

    /// Returns a logical slice of the categorical array [offset, offset+len)
    /// as a new `CategoricalArray` object.
    ///
    /// For a non-copy slice view, use `slice` from the parent Array object
    fn slice_clone(&self, offset: usize, len: usize) -> Self {
        assert!(
            offset + len <= self.data.len(),
            "slice window out of bounds"
        );

        let data = self.data[offset..offset + len].to_vec_in(Vec64Alloc::default());
        let null_mask = self
            .null_mask
            .as_ref()
            .map(|nm| nm.slice_clone(offset, len));
        Self {
            data: Vec64(data).into(),
            #[cfg(not(feature = "shared_dict"))]
            unique_values: self.unique_values.clone(),
            #[cfg(feature = "shared_dict")]
            dictionary: self.dictionary.clone(),
            null_mask,
        }
    }

    /// Borrows a `CategoricalArray` with its window parameters
    /// to a `CategoricalArrayView<'a>` alias. Like a slice, but
    /// retains access to the `&CategoricalArray`.
    ///
    /// `Offset` and `Length` are `usize` aliases.
    #[inline(always)]
    fn tuple_ref<'a>(&'a self, offset: Offset, len: Length) -> CategoricalAVT<'a, T> {
        (&self, offset, len)
    }

    /// Returns the total number of nulls.
    fn null_count(&self) -> usize {
        self.null_mask
            .as_ref()
            .map(|m| m.count_zeros())
            .unwrap_or(0)
    }

    /// Resizes the data in-place so that `len` is equal to `new_len`.
    fn resize(&mut self, n: usize, value: Self::LogicalType) {
        let current_len = self.len();

        #[cfg(not(feature = "shared_dict"))]
        let encoded: T = add_category(&mut self.unique_values, &value);
        #[cfg(feature = "shared_dict")]
        let encoded: T = self.dictionary.add_cat(&value).expect(
            "Dictionary category interning failed: cardinality exceeded capacity \
             of the categorical integer. Consider a CategoricalArray<T> with a \
             greater `T` capacity.",
        );

        if n > current_len {
            self.data.reserve(n - current_len);
            for _ in current_len..n {
                self.data.push(encoded);
            }
        } else if n < current_len {
            self.data.truncate(n);
        }
    }

    /// Returns a reference to the null bitmask
    fn null_mask(&self) -> Option<&Bitmask> {
        self.null_mask.as_ref()
    }

    /// Returns a mutable reference to the null bitmask
    fn null_mask_mut(&mut self) -> Option<&mut Bitmask> {
        self.null_mask.as_mut()
    }

    /// Sets the bitmask from a supplied one or `None`
    fn set_null_mask(&mut self, mask: Option<Bitmask>) {
        self.null_mask = mask
    }

    /// Appends all values (and null mask if present) from `other` to `self`.
    fn append_array(&mut self, other: &Self) {
        let orig_len = self.len();
        let other_len = other.len();
        if other_len == 0 { return; }

        self.data_mut().extend_from_slice(other.data());

        match (self.null_mask_mut(), other.null_mask()) {
            (Some(self_mask), Some(other_mask)) => {
                self_mask.extend_from_bitmask(other_mask);
            }
            (Some(self_mask), None) => {
                self_mask.resize(orig_len + other_len, true);
            }
            (None, Some(other_mask)) => {
                let mut mask = Bitmask::new_set_all(orig_len, true);
                mask.extend_from_bitmask(other_mask);
                self.set_null_mask(Some(mask));
            }
            (None, None) => {}
        }
    }

    fn append_range(&mut self, other: &Self, offset: usize, len: usize) -> Result<(), MinarrowError> {
        if len == 0 { return Ok(()); }
        if offset + len > other.len() {
            return Err(MinarrowError::IndexError(
                format!("append_range: offset {} + len {} exceeds source length {}", offset, len, other.len())
            ));
        }
        let orig_len = self.len();

        self.data_mut().extend_from_slice(&other.data()[offset..offset + len]);

        match (self.null_mask_mut(), other.null_mask()) {
            (Some(self_mask), Some(other_mask)) => {
                self_mask.extend_from_bitmask_range(other_mask, offset, len);
            }
            (Some(self_mask), None) => {
                self_mask.resize(orig_len + len, true);
            }
            (None, Some(other_mask)) => {
                let mut mask = Bitmask::new_set_all(orig_len, true);
                mask.extend_from_bitmask_range(other_mask, offset, len);
                self.set_null_mask(Some(mask));
            }
            (None, None) => {}
        }
        Ok(())
    }

    /// Inserts all values from `other` into `self` at the specified index.
    ///
    /// This is an O(n) operation for CategoricalArray.
    fn insert_rows(&mut self, index: usize, other: &Self) -> Result<(), MinarrowError> {
        use crate::enums::error::MinarrowError;

        let orig_len = self.len();
        let other_len = other.len();

        if index > orig_len {
            return Err(MinarrowError::IndexError(format!(
                "Index {} out of bounds for array of length {}",
                index, orig_len
            )));
        }

        if other_len == 0 {
            return Ok(());
        }

        // Map each of `other`'s dictionary codes to the code that the
        // same string will have in `self`. Existing strings are looked
        // up; novel strings are added.
        #[cfg(not(feature = "shared_dict"))]
        let index_map: Vec<T> = {
            let mut m = Vec::with_capacity(other.unique_values.len());
            for other_value in other.unique_values.iter() {
                m.push(add_category(&mut self.unique_values, other_value));
            }
            m
        };
        #[cfg(feature = "shared_dict")]
        let index_map: Vec<T> = {
            let mut m = Vec::with_capacity(other.dictionary.len());
            for other_value in other.dictionary.values().iter() {
                let code = match self.dictionary.lookup(other_value) {
                    Some(code) => code,
                    None => self.dictionary.add_cat(other_value)?,
                };
                m.push(code);
            }
            m
        };

        // Insert and remap other's data
        let new_len = orig_len + other_len;
        self.data.resize(new_len, T::from_usize(0));

        // Shift existing elements using unchecked operations
        for i in (index..orig_len).rev() {
            unsafe {
                let val = *self.data.as_ref().get_unchecked(i);
                *self.data.as_mut().get_unchecked_mut(i + other_len) = val;
            }
        }

        // Copy and remap other's data
        for i in 0..other_len {
            unsafe {
                let other_idx = *other.data.as_ref().get_unchecked(i);
                let remapped_idx = *index_map.get_unchecked(other_idx.to_usize());
                *self.data.as_mut().get_unchecked_mut(index + i) = remapped_idx;
            }
        }

        // Handle null masks with unchecked operations
        match (self.null_mask.as_mut(), other.null_mask.as_ref()) {
            (Some(self_mask), Some(other_mask)) => {
                let mut new_mask = Bitmask::new_set_all(new_len, true);
                for i in 0..index {
                    unsafe {
                        new_mask.set_unchecked(i, self_mask.get_unchecked(i));
                    }
                }
                for i in 0..other_len {
                    unsafe {
                        new_mask.set_unchecked(index + i, other_mask.get_unchecked(i));
                    }
                }
                for i in index..orig_len {
                    unsafe {
                        new_mask.set_unchecked(other_len + i, self_mask.get_unchecked(i));
                    }
                }
                *self_mask = new_mask;
            }
            (Some(self_mask), None) => {
                let mut new_mask = Bitmask::new_set_all(new_len, true);
                for i in 0..index {
                    unsafe {
                        new_mask.set_unchecked(i, self_mask.get_unchecked(i));
                    }
                }
                for i in index..orig_len {
                    unsafe {
                        new_mask.set_unchecked(other_len + i, self_mask.get_unchecked(i));
                    }
                }
                *self_mask = new_mask;
            }
            (None, Some(other_mask)) => {
                let mut new_mask = Bitmask::new_set_all(new_len, true);
                for i in 0..other_len {
                    unsafe {
                        new_mask.set_unchecked(index + i, other_mask.get_unchecked(i));
                    }
                }
                self.null_mask = Some(new_mask);
            }
            (None, None) => {}
        }

        Ok(())
    }

    /// Splits the CategoricalArray at the specified index, consuming self and returning two arrays.
    fn split(mut self, index: usize) -> Result<(Self, Self), MinarrowError> {
        use crate::enums::error::MinarrowError;

        if index == 0 || index >= self.len() {
            return Err(MinarrowError::IndexError(format!(
                "Split index {} out of valid range (0, {})",
                index,
                self.len()
            )));
        }

        // Split the data buffer
        let after_data = self.data.split_off(index);

        // Split null mask
        let after_mask = self.null_mask.as_mut().map(|mask| mask.split_off(index));

        // Both arrays share the same dictionary handle (cheap clone:
        // a `Vec64` clone under no `shared_dict`; an Arc bump under it).
        let after = CategoricalArray {
            data: after_data,
            #[cfg(not(feature = "shared_dict"))]
            unique_values: self.unique_values.clone(),
            #[cfg(feature = "shared_dict")]
            dictionary: self.dictionary.clone(),
            null_mask: after_mask,
        };

        Ok((self, after))
    }

    /// Extends the categorical array from an iterator with pre-allocated capacity.
    /// Reserves capacity in the underlying index buffer to avoid reallocations
    /// during bulk insertion. Dictionary is expanded as new unique values are encountered.
    fn extend_from_iter_with_capacity<I>(&mut self, iter: I, additional_capacity: usize)
    where
        I: Iterator<Item = Self::LogicalType>,
    {
        self.data.reserve(additional_capacity);
        let values: Vec<Self::LogicalType> = iter.collect();
        let start_len = self.data.len();
        // Extend the length to accommodate new elements
        self.data.resize(start_len + values.len(), T::from_usize(0));
        // Extend null mask if it exists
        if let Some(mask) = &mut self.null_mask {
            mask.resize(start_len + values.len(), true);
        }
        for (i, value) in values.iter().enumerate() {
            let owned = value.to_string();
            #[cfg(not(feature = "shared_dict"))]
            let code: T = add_category(&mut self.unique_values, &owned);
            #[cfg(feature = "shared_dict")]
            let code: T = self.dictionary.add_cat(&owned).expect(
                "Dictionary category interning failed: cardinality exceeded capacity \
                 of the categorical integer. Consider a CategoricalArray<T> with a \
                 greater `T` capacity.",
            );
            {
                let data = self.data.as_mut_slice();
                data[start_len + i] = code;
            }
            if let Some(mask) = &mut self.null_mask {
                unsafe { mask.set_unchecked(start_len + i, true) };
            }
        }
    }

    /// Extends the categorical array from a slice of string values.
    /// Pre-allocates capacity for the index buffer and efficiently processes
    /// each string through the internal dictionary for optimal categorical encoding.
    fn extend_from_slice(&mut self, slice: &[Self::LogicalType]) {
        let start_len = self.data.len();
        self.data.reserve(slice.len());
        // Extend the length to accommodate new elements
        self.data.resize(start_len + slice.len(), T::from_usize(0));
        // Extend null mask if it exists
        if let Some(mask) = &mut self.null_mask {
            mask.resize(start_len + slice.len(), true);
        }
        for (i, value) in slice.iter().enumerate() {
            let owned = value.to_string();
            #[cfg(not(feature = "shared_dict"))]
            let code: T = add_category(&mut self.unique_values, &owned);
            #[cfg(feature = "shared_dict")]
            let code: T = self.dictionary.add_cat(&owned).expect(
                "Dictionary category interning failed: cardinality exceeded capacity \
                 of the categorical integer. Consider a CategoricalArray<T> with a \
                 greater `T` capacity.",
            );
            {
                let data = self.data.as_mut_slice();
                data[start_len + i] = code;
            }
            if let Some(mask) = &mut self.null_mask {
                unsafe { mask.set_unchecked(start_len + i, true) };
            }
        }
    }

    /// Creates a new categorical array filled with the specified string repeated `count` times.
    /// The dictionary will contain only one unique value, making this highly memory-efficient
    /// for repeated categorical values.
    fn fill(value: Self::LogicalType, count: usize) -> Self {
        let mut array = CategoricalArray::<T>::from_vec64(crate::Vec64::with_capacity(count), None);
        // Extend the length to accommodate new elements
        array.data.resize(count, T::from_usize(0));
        // Fresh array; dictionary holds one entry once we intern.
        let owned_value = value.to_string();
        #[cfg(not(feature = "shared_dict"))]
        let dict_index: T = add_category(&mut array.unique_values, &owned_value);
        #[cfg(feature = "shared_dict")]
        let dict_index: T = array.dictionary.add_cat(&owned_value).expect(
            "Dictionary category interning failed: cardinality exceeded capacity \
             of the categorical integer. Consider a CategoricalArray<T> with a \
             greater `T` capacity.",
        );
        // Now use unchecked operations since we have proper length
        for i in 0..count {
            {
                let data = array.data.as_mut_slice();
                data[i] = dict_index;
            }
        }
        array
    }
}

#[cfg(feature = "parallel_proc")]
impl<T: Integer + Send + Sync> CategoricalArray<T> {
    /// Parallel iterator over &str (null yields "").
    #[inline]
    pub fn par_iter(&self) -> rayon::slice::Iter<'_, T> {
        self.data.par_iter()
    }

    /// Parallel mut iterator over &str (null yields "").
    #[inline]
    pub fn par_iter_mut(&mut self) -> rayon::slice::IterMut<'_, T> {
        self.data.par_iter_mut()
    }

    /// Parallel iterator over Option<&str> (None if null).
    #[inline]
    pub fn par_iter_opt(&self) -> impl ParallelIterator<Item = Option<&str>> + '_ {
        self.par_iter_range_opt(0, self.len())
    }

    /// `[start,end)` -> `&str` (null ⇒ `""`)
    #[inline]
    pub fn par_iter_range(
        &self,
        start: usize,
        end: usize,
    ) -> impl ParallelIterator<Item = &str> + '_ {
        use rayon::prelude::*;
        let null_mask = self.null_mask.as_ref();
        let dict = self.unique_values();
        let idx_buf = &self.data;
        debug_assert!(start <= end && end <= idx_buf.len());
        (start..end).into_par_iter().map(move |i| {
            if null_mask.map(|m| !m.get(i)).unwrap_or(false) {
                ""
            } else {
                &dict[idx_buf[i].to_usize()]
            }
        })
    }

    // `[start,end)` -> `Option<&str>`
    #[inline]
    pub fn par_iter_range_opt(
        &self,
        start: usize,
        end: usize,
    ) -> impl ParallelIterator<Item = Option<&str>> + '_ {
        use rayon::prelude::*;
        let null_mask = self.null_mask.as_ref();
        let dict = self.unique_values();
        let idx_buf = &self.data;
        debug_assert!(start <= end && end <= idx_buf.len());
        (start..end).into_par_iter().map(move |i| {
            if null_mask.map(|m| !m.get(i)).unwrap_or(false) {
                None
            } else {
                Some(dict[idx_buf[i].to_usize()].as_str())
            }
        })
    }

    /// `[start,end)` -> `&str` (null ⇒ `""`) - no bounds checks
    #[inline]
    pub fn par_iter_range_unchecked(
        &self,
        start: usize,
        end: usize,
    ) -> impl rayon::prelude::ParallelIterator<Item = &str> + '_ {
        use rayon::prelude::*;
        let null_mask = self.null_mask.as_ref();
        let dict = self.unique_values();
        let idx_buf = &self.data;
        (start..end).into_par_iter().map(move |i| {
            if let Some(mask) = null_mask {
                if !unsafe { mask.get_unchecked(i) } {
                    return "";
                }
            }
            let idx = unsafe { *idx_buf.get_unchecked(i) }.to_usize();
            unsafe { dict.get_unchecked(idx).as_str() }
        })
    }

    /// `[start,end)` -> `Option<&str>` -  no bounds checks
    #[inline]
    pub fn par_iter_range_opt_unchecked(
        &self,
        start: usize,
        end: usize,
    ) -> impl rayon::prelude::ParallelIterator<Item = Option<&str>> + '_ {
        use rayon::prelude::*;
        let null_mask = self.null_mask.as_ref();
        let dict = self.unique_values();
        let idx_buf = &self.data;
        (start..end).into_par_iter().map(move |i| {
            if let Some(mask) = null_mask {
                if !unsafe { mask.get_unchecked(i) } {
                    return None;
                }
            }
            let idx = unsafe { *idx_buf.get_unchecked(i) }.to_usize();
            Some(unsafe { dict.get_unchecked(idx).as_str() })
        })
    }
}

#[cfg(feature = "chunked")]
impl<'a, T: Integer> crate::traits::consolidate::Consolidate
    for Vec<crate::aliases::CategoricalAVT<'a, T>>
{
    type Output = CategoricalArray<T>;

    /// Consolidate a vector of `(CategoricalArray<T>, offset, len)` view
    /// tuples into one contiguous `CategoricalArray<T>`.
    ///
    /// When every chunk shares the same `Shared` dictionary
    /// (`Arc::ptr_eq` via `shares_with`), the indices buffers are
    /// concatenated directly and the result binds to the same dictionary
    /// snapshot - one copy per chunk, no dictionary work. Otherwise
    /// each view is slice-cloned and folded via `Concatenate::concat`,
    /// which handles the prefix and divergent-intern paths internally.
    fn consolidate(self) -> CategoricalArray<T> {
        use crate::traits::masked_array::MaskedArray;

        assert!(!self.is_empty(), "consolidate() called on empty Vec<CategoricalAVT>");

        // Fast path: all chunks point at the same Shared dictionary Arc.
        // `shares_with` is always `false` without the `shared_dict`
        // feature, so this branch is only ever taken under it.
        #[cfg(feature = "shared_dict")]
        {
            use crate::structs::bitmask::Bitmask;
            use crate::traits::consolidate::extend_null_mask;

            let first_dict = &self[0].0.dictionary;
            let all_same_dict = self
                .iter()
                .all(|(arr, _, _)| arr.dictionary.shares_with(first_dict));

            if all_same_dict {
                let total_len: usize = self.iter().map(|(_, _, len)| *len).sum();
                let has_nulls = self.iter().any(|(arr, _, _)| arr.null_mask.is_some());

                let mut result_data: Vec64<T> = Vec64::with_capacity(total_len);
                let mut result_mask: Option<Bitmask> = if has_nulls {
                    Some(Bitmask::default())
                } else {
                    None
                };
                let mut current_len = 0;

                for (arr, offset, len) in &self {
                    let data: &[T] = &arr.data[*offset..*offset + *len];
                    result_data.extend_from_slice(data);
                    extend_null_mask(
                        &mut result_mask,
                        current_len,
                        arr.null_mask(),
                        *offset,
                        *len,
                    );
                    current_len += *len;
                }

                // `Dictionary` is always Arc-backed under `shared_dict`;
                // clone bumps the Arc to join the same sharing group.
                let dict_handle = first_dict.clone();
                return CategoricalArray::<T>::new_existing_dict(
                    result_data,
                    dict_handle,
                    result_mask,
                );
            }
        }

        // Fallback: divergent dictionaries. Slice-clone each view and
        // fold through `Concatenate::concat`, which already handles the
        // prefix and divergent-intern paths.
        let mut iter = self.into_iter();
        let (first_arr, first_off, first_len) = iter.next().expect("non-empty");
        let mut result = first_arr.slice_clone(first_off, first_len);
        for (arr, off, len) in iter {
            let chunk = arr.slice_clone(off, len);
            result = result
                .concat(chunk)
                .expect("Failed to concatenate CategoricalArray");
        }
        result
    }
}

impl<T: Integer> Shape for CategoricalArray<T> {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank1(self.len())
    }
}

impl<T: Integer> Concatenate for CategoricalArray<T> {
    /// Concatenates `other` onto `self` with three dictionary-handling paths:
    ///
    /// 1. **Shared Arc** (`Arc::ptr_eq`): both batches already point at the
    ///    same dictionary. Codes are mutually meaningful, so this is a pure
    ///    buffer concat with no dictionary work.
    /// 2. **Prefix**: one dictionary is a prefix of the other. Codes from
    ///    the shorter side decode identically against the longer side, so
    ///    the result adopts the longer Arc and the data buffer is appended
    ///    without remapping.
    /// 3. **Divergent**: both dictionaries grew independently. Append the
    ///    missing entries into `self`'s dictionary via `intern` (O(1) per
    ///    string) and remap `other`'s codes into the combined space.
    fn concat(
        mut self,
        other: Self,
    ) -> core::result::Result<Self, crate::enums::error::MinarrowError> {
        let orig_len = self.len();
        let other_len = other.len();

        if other_len == 0 {
            return Ok(self);
        }

        #[cfg(feature = "shared_dict")]
        {
            let share = self.dictionary.shares_with(&other.dictionary);
            if share {
                // Same dictionary instance: pure buffer concat.
                self.data.extend_from_slice(other.data.as_ref());
            } else if other.dictionary.values().len() <= self.dictionary.values().len()
                && other.dictionary.is_prefix_of(&self.dictionary)
            {
                // `other`'s codes are already valid against the longer `self` dictionary.
                self.data.extend_from_slice(other.data.as_ref());
            } else if self.dictionary.is_prefix_of(&other.dictionary) {
                // `self`'s codes are valid against the longer `other` dictionary.
                // Adopt `other`'s dictionary and append `other`'s data verbatim.
                self.dictionary = other.dictionary.clone();
                self.data.extend_from_slice(other.data.as_ref());
            } else {
                // Divergent: bring missing entries from other into self's
                // dictionary, then remap other's codes through the union.
                let n_other_codes = other.dictionary.values().len();
                let mut remap: Vec<T> = Vec::with_capacity(n_other_codes);
                for other_value in other.dictionary.values().iter() {
                    let code = self.dictionary.add_cat(other_value)?;
                    remap.push(code);
                }
                for &other_code in other.data.iter() {
                    let mapped = remap[other_code.to_usize()];
                    self.data.push(mapped);
                }
            }
        }
        #[cfg(not(feature = "shared_dict"))]
        {
            // Without `shared_dict` each categorical owns its dictionary
            // outright; merge by interning every entry of `other`'s
            // dictionary into `self`'s, then remap `other`'s codes.
            let mut remap: Vec<T> = Vec::with_capacity(other.unique_values.len());
            for other_value in other.unique_values.iter() {
                remap.push(add_category(&mut self.unique_values, other_value));
            }
            for &other_code in other.data.iter() {
                let mapped = remap[other_code.to_usize()];
                self.data.push(mapped);
            }
        }

        // Merge null masks
        match (self.null_mask_mut(), other.null_mask()) {
            (Some(self_mask), Some(other_mask)) => {
                self_mask.extend_from_bitmask(other_mask);
            }
            (Some(self_mask), None) => {
                self_mask.resize(orig_len + other_len, true);
            }
            (None, Some(other_mask)) => {
                let mut mask = Bitmask::new_set_all(orig_len + other_len, true);
                for i in 0..other_len {
                    mask.set(orig_len + i, other_mask.get(i));
                }
                self.set_null_mask(Some(mask));
            }
            (None, None) => {
                // No mask in either: nothing to do.
            }
        }

        Ok(self)
    }
}

impl_arc_masked_array!(
    Inner = CategoricalArray<T>,
    T = T,
    Container = Buffer<T>,
    LogicalType = String,
    CopyType = &'a str,
    BufferT = T,
    Variant = TextArray,
    Bound = Integer,
);

impl_array_ref_deref!(CategoricalArray<T>: Integer);

impl<T> Display for CategoricalArray<T>
where
    T: Integer + std::fmt::Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let len = self.len();
        let null_count = self.null_count();
        let dict_size = self.unique_values().len();

        writeln!(
            f,
            "CategoricalArray [{} values]s] (dtype: categorical[str], nulls: {}, dictionary size: {})",
            len, null_count, dict_size
        )?;

        const MAX_PREVIEW: usize = 25;
        write!(f, "[")?;
        for i in 0..usize::min(len, MAX_PREVIEW) {
            if i > 0 {
                write!(f, ", ")?;
            }
            match self.get(i) {
                Some(s) => write!(f, "\"{}\"", s)?,
                None => write!(f, "null")?,
            }
        }
        if len > MAX_PREVIEW {
            write!(f, ", … ({} total)", len)?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::traits::masked_array::MaskedArray;
    use crate::vec64;

    fn bm(bits: &[bool]) -> Bitmask {
        let mut m = Bitmask::new_set_all(bits.len(), false);
        for (i, &b) in bits.iter().enumerate() {
            m.set(i, b);
        }
        m
    }

    #[test]
    fn empty_new() {
        let arr = CategoricalArray::<u8>::default();
        assert!(arr.is_empty());
        assert!(arr.unique_values().is_empty());
    }

    #[test]
    fn test_new_and_with_capacity() {
        let mut arr = CategoricalArray::<u32>::with_capacity(8, None, true);
        assert_eq!(arr.len(), 0);
        assert!(arr.data.capacity() >= 8);
        assert!(arr.null_mask.is_some());

        // Reserved null-mask slots must default to valid (1).
        assert_eq!(arr.null_count(), 0);

        arr.push_str("alpha");
        arr.push_str("beta");
        assert_eq!(arr.null_count(), 0);

        arr.push_null();
        assert_eq!(arr.null_count(), 1);
    }

    #[test]
    fn push_and_get() {
        let mut arr = CategoricalArray::<u8>::default();
        let i1 = arr.push_str("hello");
        let i2 = arr.push_str("world");
        let i3 = arr.push_str("hello");
        assert_eq!(i1, 0);
        assert_eq!(i2, 1);
        assert_eq!(i3, 0);
        assert_eq!(arr.indices(), &[0u8, 1, 0]);
        assert_eq!(arr.unique_values(), &["hello", "world".into()]);
        assert_eq!(arr.get(1), Some("world"));
    }

    #[test]
    fn null_handling() {
        let mut arr = CategoricalArray::<u16>::default();
        arr.push_str("a");
        arr.push_null();
        arr.push_str("b");
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.get(0), Some("a"));
        assert_eq!(arr.get(1), None);
        assert!(arr.is_null(1));
        assert_eq!(arr.get(2), Some("b"));
    }

    #[test]
    fn new_tolerates_out_of_range_indices_at_null_positions() {
        // Mirrors pandas' Arrow export of a Categorical with NA: the indices
        // buffer holds -1 at null slots, which becomes 255 when the index
        // type is u8. The null mask correctly marks the slot invalid.
        let data: Vec64<u8> = vec64![0, 1, 255, 0];
        let unique_values: Vec64<String> =
            vec64!["Yes".to_string(), "No".to_string()];
        let mask = bm(&[true, true, false, true]);

        let arr = CategoricalArray::<u8>::new(data, unique_values, Some(mask));

        assert_eq!(arr.len(), 4);
        assert_eq!(arr.get_str(0), Some("Yes"));
        assert_eq!(arr.get_str(1), Some("No"));
        assert_eq!(arr.get_str(2), None);
        assert_eq!(arr.get_str(3), Some("Yes"));
    }

    #[test]
    #[should_panic(expected = "Index 255 out of bounds")]
    fn new_still_rejects_out_of_range_indices_at_valid_positions() {
        // Same shape as above, but the offending slot is marked valid -
        // construction must still fail loudly.
        let data: Vec64<u8> = vec64![0, 1, 255, 0];
        let unique_values: Vec64<String> =
            vec64!["Yes".to_string(), "No".to_string()];
        let mask = bm(&[true, true, true, true]);

        let _ = CategoricalArray::<u8>::new(data, unique_values, Some(mask));
    }

    #[test]
    fn set_overwrite_and_new() {
        let mut arr = CategoricalArray::<u32>::default();
        arr.push_str("x");
        arr.push_str("y");
        arr.set_str(1, "x");
        assert_eq!(arr.get(1), Some("x"));
        arr.set_str(0, "zebra");
        assert!(arr.unique_values().contains(&"zebra".to_string()));
        assert_eq!(arr.get(0), Some("zebra"));
    }

    #[test]
    fn extend_and_builder() {
        let mut arr = CategoricalArray::<u8>::default();
        arr.extend(["a", "b", "a", "c"].iter().copied());
        assert_eq!(arr.len(), 4);
        assert_eq!(arr.get(2), Some("a"));

        let built = CategoricalArray::<u8>::from_values(vec!["k", "l", "k"]);
        assert_eq!(built.indices(), &[0u8, 1, 0]);
        assert_eq!(built.get(1), Some("l"));
    }

    #[test]
    fn set_null_after_push() {
        let mut arr = CategoricalArray::<u8>::default();
        arr.push_str("one");
        arr.push_str("two");
        arr.set_null(1);
        assert!(arr.is_null(1));
        assert_eq!(arr.get(1), None);
    }

    #[test]
    fn test_categorical_iter() {
        let arr =
            CategoricalArray::from_slices(&[0u32, 1, 2], &["a".into(), "b".into(), "c".into()]);
        let vals: Vec<_> = arr.iter().collect();
        assert_eq!(vals, vec!["a", "b", "c"]);
        let opt: Vec<_> = arr.iter_str_opt().collect();
        assert_eq!(opt, vec![Some("a"), Some("b"), Some("c")]);
    }

    #[test]
    fn test_categorical_array_slice() {
        let arr = CategoricalArray::<u8>::new(
            vec64![2u8, 1, 0],
            vec64!["green".to_string(), "blue".to_string(), "red".to_string()],
            Some(Bitmask::from_bools(&[false, true, true])),
        );
        let sliced = arr.slice_clone(0, 3);
        assert_eq!(
            sliced.iter_str_opt().collect::<Vec<_>>(),
            vec![None, Some("blue"), Some("green")]
        );
    }

    #[test]
    fn test_categorical_set_and_get() {
        let mut arr = CategoricalArray::<u32>::from_values(["a", "b", "c"].iter().cloned());
        // initial null mask none => all valid
        assert!(arr.null_mask.is_none());

        // set index 1 to "d" (new entry)
        arr.set_str(1, "d");
        assert_eq!(arr.get(1), Some("d"));
        // dictionary should have "d" appended
        assert_eq!(arr.unique_values().len(), 4);
        assert!(arr.unique_values().contains(&"d".to_string()));

        // set index 2 to existing "a"
        arr.set_str(2, "a");
        assert_eq!(arr.get(2), Some("a"));
        // dictionary length unchanged
        assert_eq!(arr.unique_values().len(), 4);
    }

    #[test]
    fn test_categorical_set_unchecked_and_null_mask() {
        let mut arr = CategoricalArray::<u32>::from_values(["x", "y", "z"].iter().cloned());
        arr.null_mask = Some(bm(&[true, false, true]));

        // unsafe unchecked set index 1 to "w"
        unsafe { arr.set_str_unchecked(1, "w") };
        // now index 1 should be "w"
        assert_eq!(arr.get(1), Some("w"));
        // null mask at 1 now true
        let mask = arr.null_mask.as_ref().unwrap();
        assert!(mask.get(1));
        // dictionary should contain "w"
        assert!(arr.unique_values().contains(&"w".to_string()));
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_categorical_set_oob() {
        let mut arr = CategoricalArray::<u32>::from_values(["foo"].iter().cloned());
        // this should panic
        arr.set_str(5, "bar");
    }

    #[test]
    fn test_to_string_array() {
        let unique = vec64!["foo".to_string(), "bar".to_string()];
        let data = vec64![0u32, 0u32, 1u32];
        let mut mask = Bitmask::new_set_all(3, true);
        mask.set(1, false); // second entry is null

        let cat = CategoricalArray {
            data: data.into(),
            #[cfg(not(feature = "shared_dict"))]
            unique_values: unique,
            #[cfg(feature = "shared_dict")]
            dictionary: Dictionary::from(unique),
            null_mask: Some(mask),
        };

        let str_arr = cat.to_string_array();

        assert_eq!(str_arr.get(0), Some("foo"));
        assert_eq!(str_arr.get(1), None);
        assert_eq!(str_arr.get(2), Some("bar"));

        assert_eq!(str_arr.offsets, vec64![0u32, 3, 3, 6]);
        assert_eq!(str_arr.data, Vec64::from_slice(b"foobar"));
        assert_eq!(str_arr.null_mask.unwrap().count_zeros(), 1);
    }

    #[test]
    fn test_iterators_yield_correct_values() {
        let mut arr = CategoricalArray::<u8>::default();
        arr.push_str("cat");
        arr.push_str("dog");
        arr.push_str("bird");

        let mut it = arr.indices_iter();
        assert_eq!(it.next(), Some(&0u8));
        assert_eq!(it.next(), Some(&1u8));

        let mut it = arr.values_iter();
        assert!(it.any(|s| s == "cat"));
        assert!(it.any(|s| s == "dog"));

        let mut it_mut = arr.indices_iter_mut();
        if let Some(v) = it_mut.next() {
            *v = 2;
        }
        assert_eq!(arr.get(0), Some("bird"));
    }

    #[test]
    fn test_resize_expands_and_truncates() {
        let mut arr = CategoricalArray::<u8>::default();
        arr.push_str("one");
        arr.push_str("two");

        arr.resize(5, "two".to_string());
        assert_eq!(arr.len(), 5);
        assert_eq!(arr.get(4), Some("two"));

        arr.resize(2, "ignored".to_string());
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn test_from_parts_exact_match() {
        let data = vec64![0u8, 1u8];
        let dict = vec64!["alpha".to_string(), "beta".to_string()];
        let mask = Some(Bitmask::from_bools(&[true, false]));
        let arr = CategoricalArray::from_parts(data, dict, mask.clone());

        assert_eq!(arr.get(0), Some("alpha"));
        assert_eq!(arr.get(1), None);
        assert_eq!(arr.null_mask(), mask.as_ref());
    }

    #[test]
    fn test_batch_extend_from_iter_with_capacity() {
        let mut arr = CategoricalArray::<u32>::default();
        let data = vec![
            "cat".to_string(),
            "dog".to_string(),
            "cat".to_string(),
            "bird".to_string(),
        ];

        arr.extend_from_iter_with_capacity(data.into_iter(), 4);

        assert_eq!(arr.len(), 4);
        assert_eq!(arr.get(0), Some("cat"));
        assert_eq!(arr.get(1), Some("dog"));
        assert_eq!(arr.get(2), Some("cat"));
        assert_eq!(arr.get(3), Some("bird"));

        // Dictionary should have 3 unique values
        assert_eq!(arr.unique_values().len(), 3);
    }

    #[test]
    fn test_batch_extend_from_slice_dictionary_growth() {
        let mut arr = CategoricalArray::<u32>::default();
        arr.push("initial".to_string());

        let data = &[
            "apple".to_string(),
            "banana".to_string(),
            "apple".to_string(),
        ];
        arr.extend_from_slice(data);

        assert_eq!(arr.len(), 4);
        assert_eq!(arr.get(0), Some("initial"));
        assert_eq!(arr.get(1), Some("apple"));
        assert_eq!(arr.get(2), Some("banana"));
        assert_eq!(arr.get(3), Some("apple"));

        // Dictionary: initial, apple, banana
        assert_eq!(arr.unique_values().len(), 3);
    }

    #[test]
    fn test_batch_fill_single_category() {
        let arr = CategoricalArray::<u32>::fill("repeated".to_string(), 100);

        assert_eq!(arr.len(), 100);
        assert_eq!(arr.null_count(), 0);

        // All values should be the same category
        for i in 0..100 {
            assert_eq!(arr.get(i), Some("repeated"));
        }

        // Dictionary should contain only one unique value
        assert_eq!(arr.unique_values().len(), 1);
        assert_eq!(arr.unique_values()[0], "repeated");

        // All indices should point to the same dictionary entry (0)
        for i in 0..100 {
            assert_eq!(arr.data[i], 0u32);
        }
    }

    #[test]
    fn test_batch_operations_with_nulls() {
        let mut arr = CategoricalArray::<u32>::default();
        arr.push("first".to_string());
        arr.push_null();

        let data = &["second".to_string(), "first".to_string()];
        arr.extend_from_slice(data);

        assert_eq!(arr.len(), 4);
        assert_eq!(arr.get(0), Some("first"));
        assert_eq!(arr.get(1), None);
        assert_eq!(arr.get(2), Some("second"));
        assert_eq!(arr.get(3), Some("first"));
        assert!(arr.null_count() >= 1); // At least the initial null

        // Dictionary: first, second
        assert!(arr.unique_values().len() >= 2); // At least first and second
    }

    #[test]
    fn test_batch_operations_preserve_categorical_efficiency() {
        let mut arr = CategoricalArray::<u32>::default();

        // Create data with many repeated categories
        let categories = ["A", "B", "C"];
        let mut data = Vec::new();
        for _ in 0..100 {
            for cat in &categories {
                data.push(cat.to_string());
            }
        }

        arr.extend_from_slice(&data);

        assert_eq!(arr.len(), 300);
        assert_eq!(arr.unique_values().len(), 3); // Only 3 unique despite 300 entries

        // Verify all categories are represented correctly
        for i in 0..300 {
            let expected = categories[i % 3];
            assert_eq!(arr.get(i), Some(expected));
        }
    }

    #[test]
    fn test_categorical_array_concat() {
        let arr1 = CategoricalArray::<u32>::from_values(["apple", "banana", "apple"]);
        let arr2 = CategoricalArray::<u32>::from_values(["cherry", "apple"]);

        let result = arr1.concat(arr2).unwrap();

        assert_eq!(result.len(), 5);
        assert_eq!(result.get_str(0), Some("apple"));
        assert_eq!(result.get_str(1), Some("banana"));
        assert_eq!(result.get_str(2), Some("apple"));
        assert_eq!(result.get_str(3), Some("cherry"));
        assert_eq!(result.get_str(4), Some("apple"));

        // Dictionary should be merged: apple, banana, cherry
        assert_eq!(result.unique_values().len(), 3);
        assert!(result.unique_values().contains(&"apple".to_string()));
        assert!(result.unique_values().contains(&"banana".to_string()));
        assert!(result.unique_values().contains(&"cherry".to_string()));
    }

    #[test]
    fn test_categorical_array_concat_with_nulls() {
        let mut arr1 = CategoricalArray::<u32>::default();
        arr1.push_str("red");
        arr1.push_null();
        arr1.push_str("blue");

        let mut arr2 = CategoricalArray::<u32>::default();
        arr2.push_str("green");
        arr2.push_null();

        let result = arr1.concat(arr2).unwrap();

        assert_eq!(result.len(), 5);
        assert_eq!(result.get_str(0), Some("red"));
        assert_eq!(result.get_str(1), None);
        assert_eq!(result.get_str(2), Some("blue"));
        assert_eq!(result.get_str(3), Some("green"));
        assert_eq!(result.get_str(4), None);
        assert_eq!(result.null_count(), 2);
    }

    #[test]
    fn test_categorical_array_concat_disjoint_dictionaries() {
        // First array with dictionary: [red, blue, green]
        let arr1 = CategoricalArray::<u32>::from_values(["red", "blue", "green", "red", "blue"]);

        // Second array with completely different dictionary: [alpha, beta, gamma]
        let arr2 = CategoricalArray::<u32>::from_values(["alpha", "beta", "gamma", "alpha"]);

        // Verify initial state
        assert_eq!(arr1.unique_values().len(), 3); // red, blue, green
        assert_eq!(arr2.unique_values().len(), 3); // alpha, beta, gamma

        // Verify arr1 indices point to correct values
        assert_eq!(arr1.get_str(0), Some("red"));
        assert_eq!(arr1.get_str(1), Some("blue"));
        assert_eq!(arr1.get_str(2), Some("green"));
        assert_eq!(arr1.get_str(3), Some("red"));
        assert_eq!(arr1.get_str(4), Some("blue"));

        // Verify arr2 indices point to correct values
        assert_eq!(arr2.get_str(0), Some("alpha"));
        assert_eq!(arr2.get_str(1), Some("beta"));
        assert_eq!(arr2.get_str(2), Some("gamma"));
        assert_eq!(arr2.get_str(3), Some("alpha"));

        let result = arr1.concat(arr2).unwrap();

        // After concatenation, dictionary should have all 6 unique values
        assert_eq!(result.unique_values().len(), 6);
        assert!(result.unique_values().contains(&"red".to_string()));
        assert!(result.unique_values().contains(&"blue".to_string()));
        assert!(result.unique_values().contains(&"green".to_string()));
        assert!(result.unique_values().contains(&"alpha".to_string()));
        assert!(result.unique_values().contains(&"beta".to_string()));
        assert!(result.unique_values().contains(&"gamma".to_string()));

        // Verify all values are correctly accessible after remapping
        assert_eq!(result.len(), 9);

        // Original arr1 values should be unchanged
        assert_eq!(result.get_str(0), Some("red"));
        assert_eq!(result.get_str(1), Some("blue"));
        assert_eq!(result.get_str(2), Some("green"));
        assert_eq!(result.get_str(3), Some("red"));
        assert_eq!(result.get_str(4), Some("blue"));

        // arr2 values should be correctly remapped
        assert_eq!(result.get_str(5), Some("alpha"));
        assert_eq!(result.get_str(6), Some("beta"));
        assert_eq!(result.get_str(7), Some("gamma"));
        assert_eq!(result.get_str(8), Some("alpha"));
    }
}

#[cfg(test)]
#[cfg(feature = "parallel_proc")]
mod parallel_tests {
    use super::*;
    use crate::vec64;
    #[test]
    fn test_categorical_par_iter() {
        let arr =
            CategoricalArray::from_slices(&[0u32, 1, 2], &["a".into(), "b".into(), "c".into()]);
        let vals: Vec<_> = arr.par_iter().collect();
        assert_eq!(vals.len(), 3);
        let opt: Vec<_> = arr.par_iter_opt().collect();
        assert!(opt.iter().all(|v| v.is_some()));
    }

    #[test]
    fn test_categoricalarray_par_iter_opt() {
        let mut arr = CategoricalArray::<u32>::default();
        arr.push_str("alpha");
        arr.push_str("beta");
        arr.push_null();
        arr.push_str("gamma");

        let par: Vec<_> = arr.par_iter_opt().collect();
        let expected = vec![Some("alpha"), Some("beta"), None, Some("gamma")];
        assert_eq!(par, expected);
    }

    #[test]
    fn test_categoricalarray_par_iter_range_unchecked() {
        let dict = vec64!["one".to_string(), "two".to_string(), "three".to_string()];
        let arr = CategoricalArray::<u32>::from_parts(vec64![0, 2, 1, 0, 2], dict, None);
        let out: Vec<&str> = arr.par_iter_range_unchecked(1, 4).collect();
        assert_eq!(out, vec!["three", "two", "one"]);
    }

    #[test]
    fn test_categoricalarray_par_iter_range_opt_unchecked() {
        let dict = vec64!["x".to_string(), "y".to_string(), "z".to_string()];
        let mut arr = CategoricalArray::<u32>::from_parts(vec64![1, 0, 2, 1, 0], dict, None);
        arr.null_mask = Some(Bitmask::from_bools(&[true, false, true, false, true]));
        let out: Vec<Option<&str>> = arr.par_iter_range_opt_unchecked(0, 5).collect();
        assert_eq!(
            out,
            vec![
                Some("y"), // 0 (valid)
                None,      // 1 (null)
                Some("z"), // 2 (valid)
                None,      // 3 (null)
                Some("x")  // 4 (valid)
            ]
        );
    }
}
