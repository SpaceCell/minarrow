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

//! # **Dictionary Module** - *Append-only shared string dictionary for categorical arrays*
//!
//! This module is available behind the `shared_dict` feature. Without
//! that feature, `CategoricalArray<T>` stores its unique values directly
//! as `unique_values: Vec64<String>` and no shared dictionary is used.
//!
//! ## Structure
//! `Dictionary<T>` is a lightweight handle around `Arc<DictionaryInner<T>>`.
//! Cloning a dictionary is therefore just an `Arc` bump: every clone
//! points at the same underlying storage and observes the same updates.
//!
//! The inner dictionary contains two data structures:
//!
//! - `values: AppendOnlyVec<String>` - the code-indexed value array.
//!   Values are stored in a fixed contiguous allocation that is never
//!   reallocated, so published entries have stable addresses for the
//!   lifetime of the dictionary. Readers access the published prefix
//!   without taking a lock. Appenders reserve slots concurrently, then
//!   publish in order so the readable prefix always remains contiguous.
//!
//! - `index: ShardedIndex<T>` - the reverse string-to-code lookup.
//!   The index is split across 64 `Mutex<HashMap<_, _>>` shards, using
//!   the candidate string's hash to select a shard and spread contention
//!   across independent locks.
//!
//! ## Intern flow
//! 1. Hash the candidate string using the shared `ahash::RandomState`
//!    when `fast_dict` is enabled, otherwise use `DefaultHasher`.
//! 2. Use the low bits of the hash to select an index shard, then take
//!    that shard's mutex.
//! 3. Probe the shard's `HashMap<String, T>` for the candidate string.
//!    With `fast_dict`, this uses `raw_entry_mut().from_hash(...)` and
//!    the precomputed hash to avoid hashing the string a second time.
//!    Without `fast_dict`, this uses the standard `HashMap::get` path.
//! 4. If the string is already present, return its existing code.
//! 5. Otherwise, append the string to `values`. The append path reserves
//!    a slot with a cap-bounded CAS loop, so the dictionary's configured
//!    capacity is honoured exactly even under concurrent writers.
//! 6. Publish the new value, insert the string-to-code mapping into the
//!    shard's `HashMap`, release the mutex, and return the new code.
//!
//! ## Append-only invariant
//! Once a string has been assigned a code, that mapping is permanent.
//! Entries are never reordered, replaced, or removed.

#[cfg(not(feature = "fast_dict"))]
use std::collections::hash_map::DefaultHasher;
use std::fmt;
#[cfg(not(feature = "fast_dict"))]
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use ::vec64::Vec64;

use ::vec64::AppendOnlyVec;
use crate::traits::type_unions::Integer;

// Shard mutex. `fast_dict` swaps to `parking_lot::Mutex` for a
// faster uncontended fast path (~5 ns vs std's ~15 ns).
#[cfg(feature = "fast_dict")]
type ShardMutex<T> = parking_lot::Mutex<T>;
#[cfg(not(feature = "fast_dict"))]
type ShardMutex<T> = std::sync::Mutex<T>;

// Inner per-shard hashmap. Stores `(String, T)` per entry. Both
// configurations share this shape: the duplicate string in the
// hashmap key keeps the eq check on the hot probe path to one
// contiguous cache-line load, which matters at large dictionary
// cardinality where the `AppendOnlyVec` storage spills past L3.
//
// `fast_dict` uses `hashbrown::HashMap` so the `add_cat` / `lookup`
// paths can call `raw_entry_mut().from_hash(...)` with the
// precomputed `ahash` hash and skip the inner map's hashing pass.
// Plain `shared_dict` falls back to `std::HashMap` (or `ahash::AHashMap`
// under `fast_hash`), which re-hashes the input string on lookup.
#[cfg(feature = "fast_dict")]
type IndexMap<T> = hashbrown::HashMap<String, T, ahash::RandomState>;
#[cfg(all(not(feature = "fast_dict"), feature = "fast_hash"))]
type IndexMap<T> = ahash::AHashMap<String, T>;
#[cfg(all(not(feature = "fast_dict"), not(feature = "fast_hash")))]
type IndexMap<T> = std::collections::HashMap<String, T>;

/// Number of shards in the reverse string-to-code index. 64 means
/// distinct novel strings spread cheaply across the index without
/// serialising on a single mutex.
const N_INDEX_SHARDS: usize = 64;

/// Errors arising from mutating a `Dictionary<T>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DictionaryError {
    /// The new cardinality would exceed the capacity of the index type `T`
    /// (e.g. 256 entries for `u8`). The dictionary is left unchanged and
    /// no slot is reserved.
    Overflow,
}

impl fmt::Display for DictionaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overflow => write!(
                f,
                "dictionary cardinality would exceed the capacity of the index type"
            ),
        }
    }
}

impl std::error::Error for DictionaryError {}

/// Sharded reverse index. Each shard holds its slice of strings under a
/// `Mutex`, selected via the low bits of a `BuildHasher` hash so
/// contention is spread across shards rather than serialising on a
/// single mutex.
///
/// Under `fast_dict` the shard-selection hasher is an
/// `ahash::RandomState` shared across every shard's inner `HashMap`,
/// so the hash bits used for shard selection are bit-identical to the
/// hash bits the map would compute internally - letting `add_cat` /
/// `lookup` reach the inner via `raw_entry_mut().from_hash(...)` with
/// one hash pass instead of two.
struct ShardedIndex<T: Integer> {
    #[cfg(feature = "fast_dict")]
    hasher: ahash::RandomState,
    shards: Box<[ShardMutex<IndexMap<T>>; N_INDEX_SHARDS]>,
}

impl<T: Integer> Default for ShardedIndex<T> {
    fn default() -> Self {
        #[cfg(feature = "fast_dict")]
        {
            // One `ahash::RandomState` per dictionary. It seeds both
            // the outer shard-selection hash and every shard's inner
            // HashMap, so the precomputed hash we pass to
            // `raw_entry_mut().from_hash(...)` matches the bits the
            // inner map would have computed on its own. The inner
            // map's HashMap<String, T> rebuckets on resize via
            // `hash(&String)` against this same seed - bit-identical
            // to the external hash, so entries stay findable.
            let hasher = ahash::RandomState::new();
            let shards: [ShardMutex<IndexMap<T>>; N_INDEX_SHARDS] =
                std::array::from_fn(|_| ShardMutex::new(IndexMap::with_hasher(hasher.clone())));
            Self {
                hasher,
                shards: Box::new(shards),
            }
        }
        #[cfg(not(feature = "fast_dict"))]
        {
            let shards: [ShardMutex<IndexMap<T>>; N_INDEX_SHARDS] =
                std::array::from_fn(|_| ShardMutex::new(IndexMap::default()));
            Self {
                shards: Box::new(shards),
            }
        }
    }
}

impl<T: Integer> ShardedIndex<T> {
    /// Shard index for `s`. Used by the paths that don't have the
    /// `fast_dict` feature; with `fast_dict`, `add_cat` and `lookup`
    /// compute the hash inline so the same `u64` serves both shard
    /// selection and the `raw_entry_mut().from_hash(...)` probe.
    #[cfg(not(feature = "fast_dict"))]
    #[inline]
    #[allow(clippy::unused_self)]
    fn shard_for(&self, s: &str) -> usize {
        let mut h = DefaultHasher::new();
        s.hash(&mut h);
        (h.finish() as usize) % N_INDEX_SHARDS
    }

}

/// Backing storage for a `Dictionary<T>`. Held behind the dictionary's
/// `Arc`. Reads of the value array via `values()` are lock-free;
/// `add_cat` briefly takes a per-shard mutex for the check-and-insert
/// step.
pub struct DictionaryInner<T: Integer> {
    /// Code-indexed string array. Lock-free reads; lock-free multi-writer
    /// `push_bounded` under the categorical's width cap.
    pub values: AppendOnlyVec<String>,
    /// Reverse lookup, sharded.
    index: ShardedIndex<T>,
}

impl<T: Integer> Default for DictionaryInner<T> {
    fn default() -> Self {
        // Pre-allocate the value array to the type's natural cap.
        // The cap is fixed for the dictionary's lifetime (the
        // `AppendOnlyVec` never reallocates); `push` returns `None`
        // when the cap is reached, which `add_cat` surfaces as
        // `DictionaryError::Overflow`.
        Self {
            values: AppendOnlyVec::with_capacity(max_cap::<T>()),
            index: ShardedIndex::default(),
        }
    }
}

/// Append-only string dictionary backing `CategoricalArray<T>` under
/// `shared_dict`. Cloning a `Dictionary` is an Arc bump on the underlying
/// inner; both clones observe the same updates immediately. `add_cat`
/// takes `&self` and is concurrent-safe across many writers without
/// global serialisation.
#[derive(Clone)]
pub struct Dictionary<T: Integer> {
    inner: Arc<DictionaryInner<T>>,
}

impl<T: Integer> Default for Dictionary<T> {
    fn default() -> Self {
        Self {
            inner: Arc::new(DictionaryInner::default()),
        }
    }
}

impl<T: Integer> Dictionary<T> {
    /// Empty dictionary in a fresh sharing group.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a dictionary from an ordered list of values. Deduplicates
    /// while preserving first-occurrence order, and builds the reverse
    /// index alongside. Panics if the input contains more unique values
    /// than the capacity of `T`.
    pub fn from_values(values: impl Into<Vec64<String>>) -> Self {
        let values: Vec64<String> = values.into();
        let d = Self::default();
        let cap = d.inner.values.capacity();
        for s in values {
            #[cfg(feature = "fast_dict")]
            {
                use hashbrown::hash_map::RawEntryMut;
                let values_ref = &d.inner.values;
                let hash = d.inner.index.hasher.hash_one(&s);
                let shard_idx = (hash as usize) & (N_INDEX_SHARDS - 1);
                let shard = &d.inner.index.shards[shard_idx];
                let mut g = shard.lock();
                match g.raw_entry_mut().from_hash(hash, |k: &String| k.as_str() == s.as_str()) {
                    RawEntryMut::Occupied(_) => continue,
                    RawEntryMut::Vacant(vac) => {
                        assert!(
                            values_ref.count() < cap,
                            "Dictionary input has more unique values than the capacity of {} ({})",
                            std::any::type_name::<T>(),
                            cap
                        );
                        // SAFETY: assert above guarantees `reserved < cap`
                        // and we hold the shard mutex, so push is the
                        // sole pending writer on this slot.
                        let idx = unsafe { values_ref.push(s.clone()).unwrap_unchecked() };
                        vac.insert_hashed_nocheck(hash, s, T::from_usize(idx));
                    }
                }
            }
            #[cfg(not(feature = "fast_dict"))]
            {
                let shard = &d.inner.index.shards[d.inner.index.shard_for(&s)];
                // Suppress poison and clear the flag so later lock calls
                // take the Ok path. Each critical section is a single
                // hashmap+vec op, atomic at the data-structure level, so
                // a previous panic cannot leave observable inconsistency.
                let mut g = shard.lock().unwrap_or_else(|p| {
                    shard.clear_poison();
                    p.into_inner()
                });
                if g.get(&s).is_some() {
                    continue;
                }
                assert!(
                    d.inner.values.count() < cap,
                    "Dictionary input has more unique values than the capacity of {} ({})",
                    std::any::type_name::<T>(),
                    cap
                );
                // SAFETY: assert above guarantees `reserved < cap` and we
                // hold the shard mutex, so push is the sole pending
                // writer on this slot.
                let idx = unsafe { d.inner.values.push(s.clone()).unwrap_unchecked() };
                g.insert(s, T::from_usize(idx));
            }
        }
        d
    }

    /// Borrow the published prefix of the value array as a slice.
    /// Lock-free; indexing returns `&String`, which derefs to `&str`
    /// for the lifetime of `&self`. Concurrent pushes may publish
    /// additional slots after this call returns; those are visible to
    /// subsequent invocations.
    #[inline]
    pub fn values(&self) -> &[String] {
        self.inner.values.as_slice()
    }

    /// Number of entries currently visible to readers.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.values.count()
    }

    /// True if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Code for `s` if present at the moment of the call, otherwise `None`.
    pub fn lookup(&self, s: &str) -> Option<T> {
        #[cfg(feature = "fast_dict")]
        {
            let hash = self.inner.index.hasher.hash_one(s);
            let shard_idx = (hash as usize) & (N_INDEX_SHARDS - 1);
            let shard = &self.inner.index.shards[shard_idx];
            let g = shard.lock();
            // Probe with the precomputed hash so the inner map does
            // not re-hash on the lookup. The eq callback compares the
            // stored `String` against the input directly - one
            // contiguous cache line on the hash-bucket entry.
            g.raw_entry()
                .from_hash(hash, |k: &String| k.as_str() == s)
                .map(|(_, v)| *v)
        }
        #[cfg(not(feature = "fast_dict"))]
        {
            let shard = &self.inner.index.shards[self.inner.index.shard_for(s)];
            // Suppress poison and clear the flag so later lock calls
            // take the Ok path. Each critical section operates on the
            // HashMap and the AppendOnlyVec atomically with respect to
            // the data structures, so a panic in a previous holder
            // cannot leave the inner state observably inconsistent.
            shard
                .lock()
                .unwrap_or_else(|p| {
                    shard.clear_poison();
                    p.into_inner()
                })
                .get(s)
                .copied()
        }
    }

    /// Adds `value` to the dictionary atomically and returns its code.
    /// Concurrent-safe via `&self`. Returns
    /// `Err(DictionaryError::Overflow)` if the new cardinality would
    /// exceed the capacity of `T`, leaving the dictionary unchanged and
    /// no slot reserved.
    pub fn add_cat(&self, value: &str) -> Result<T, DictionaryError> {
        #[cfg(feature = "fast_dict")]
        {
            use hashbrown::hash_map::RawEntryMut;
            let hash = self.inner.index.hasher.hash_one(value);
            let shard_idx = (hash as usize) & (N_INDEX_SHARDS - 1);
            let shard = &self.inner.index.shards[shard_idx];
            let mut g = shard.lock();
            // Probe with the precomputed hash so the inner map does
            // not re-hash. The eq callback compares the stored `String`
            // key directly against the input - one contiguous read on
            // the hash-bucket entry.
            match g.raw_entry_mut().from_hash(hash, |k: &String| k.as_str() == value) {
                RawEntryMut::Occupied(e) => Ok(*e.get()),
                RawEntryMut::Vacant(vac) => {
                    // Allocate the owned string once and share between
                    // the value-array push and the hashmap insert.
                    let owned = value.to_owned();
                    let idx = self
                        .inner
                        .values
                        .push(owned.clone())
                        .ok_or(DictionaryError::Overflow)?;
                    let code = T::from_usize(idx);
                    vac.insert_hashed_nocheck(hash, owned, code);
                    Ok(code)
                }
            }
        }
        #[cfg(not(feature = "fast_dict"))]
        {
            let shard = &self.inner.index.shards[self.inner.index.shard_for(value)];
            let mut g = shard.lock().unwrap_or_else(|p| {
                shard.clear_poison();
                p.into_inner()
            });
            if let Some(&code) = g.get(value) {
                return Ok(code);
            }
            // Without `fast_dict` the string is stored twice, in the
            // value array and as the hashmap key - the price of the
            // simpler stable-std design without `raw_entry`.
            let idx = self
                .inner
                .values
                .push(value.to_owned())
                .ok_or(DictionaryError::Overflow)?;
            let code = T::from_usize(idx);
            g.insert(value.to_owned(), code);
            Ok(code)
        }
    }

    /// Absorb `cat` into this sharing group. Interns every entry of
    /// `cat`'s current dictionary into `self`, remaps `cat`'s data buffer
    /// to the resulting codes if any code shifted, and rebinds `cat`'s
    /// dictionary to a clone of `self` so the chunk joins the sharing
    /// group.
    pub fn add_remap_cat(&self, cat: &mut crate::CategoricalArray<T>) {
        let incoming = &cat.dictionary.inner.values;
        let mut shifted = false;
        let mut remap: Vec<T> = Vec::with_capacity(incoming.count());
        for (incoming_code, s) in incoming.iter() {
            let Ok(new_code) = self.add_cat(s) else { return };
            if new_code.to_usize() != incoming_code {
                shifted = true;
            }
            remap.push(new_code);
        }
        if shifted {
            for code in cat.data.iter_mut() {
                *code = remap[code.to_usize()];
            }
        }
        cat.dictionary = self.clone();
    }

    /// True if `self`'s values at the moment of the call are a prefix of
    /// `other`'s. Every code valid against `self` decodes to the same
    /// string in `other`.
    pub fn is_prefix_of(&self, other: &Self) -> bool {
        let a = &self.inner.values;
        let b = &other.inner.values;
        if a.count() > b.count() {
            return false;
        }
        for (i, s) in a.iter() {
            match b.get(i) {
                Some(t) if t.as_str() == s.as_str() => {}
                _ => return false,
            }
        }
        true
    }

    /// True if `self` and `other` are clones of the same dictionary, so
    /// updates through one are visible through the other.
    #[inline]
    pub fn shares_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Detach this dictionary from its sharing group. Future mutations are
    /// independent of the original group; the data is preserved.
    ///
    /// User-only; no internal call sites. Use when you want to keep the
    /// current entries but stop receiving updates from the group.
    pub fn detach_to_owned(&mut self) {
        let fresh = Dictionary::<T>::default();
        for (_, s) in self.inner.values.iter() {
            #[cfg(feature = "fast_dict")]
            {
                use hashbrown::hash_map::RawEntryMut;
                let values_ref = &fresh.inner.values;
                let hash = fresh.inner.index.hasher.hash_one(s);
                let shard_idx = (hash as usize) & (N_INDEX_SHARDS - 1);
                let shard = &fresh.inner.index.shards[shard_idx];
                let mut g = shard.lock();
                // The source dictionary is deduplicated by construction,
                // so every entry is novel against the fresh dict.
                let vac = match g.raw_entry_mut().from_hash(hash, |k: &String| k.as_str() == s.as_str()) {
                    RawEntryMut::Vacant(v) => v,
                    RawEntryMut::Occupied(_) => continue,
                };
                // SAFETY: fresh dict's cap matches the source dict's cap
                // and source is dedup'd, so push always succeeds.
                let idx = unsafe { values_ref.push(s.clone()).unwrap_unchecked() };
                vac.insert_hashed_nocheck(hash, s.clone(), T::from_usize(idx));
            }
            #[cfg(not(feature = "fast_dict"))]
            {
                let shard = &fresh.inner.index.shards[fresh.inner.index.shard_for(s)];
                let mut g = shard.lock().unwrap_or_else(|p| {
                    shard.clear_poison();
                    p.into_inner()
                });
                // SAFETY: fresh dict's cap matches the source dict's cap
                // and source is dedup'd, so push always succeeds.
                let idx = unsafe { fresh.inner.values.push(s.clone()).unwrap_unchecked() };
                g.insert(s.clone(), T::from_usize(idx));
            }
        }
        self.inner = fresh.inner;
    }

    /// Mutable iteration over the dictionary's values when this handle
    /// has exclusive ownership of its inner allocation (Arc refcount 1).
    /// Returns `None` when other Arc clones still exist; callers in
    /// that situation must explicitly `detach_to_owned` first, accepting
    /// that the resulting handle is decoupled from the sharing group.
    ///
    /// Mutating an existing entry replaces the string that every code
    /// assigned at that position decodes to; codes against the old value
    /// no longer mean what they previously meant. For adding new values,
    /// `add_cat` is the append-only path.
    pub fn try_values_iter_mut(
        &mut self,
    ) -> Option<std::slice::IterMut<'_, String>> {
        Arc::get_mut(&mut self.inner).map(|inner| inner.values.iter_mut())
    }
}

impl<T: Integer> PartialEq for Dictionary<T> {
    fn eq(&self, other: &Self) -> bool {
        if Arc::ptr_eq(&self.inner, &other.inner) {
            return true;
        }
        let a = &self.inner.values;
        let b = &other.inner.values;
        if a.count() != b.count() {
            return false;
        }
        for (i, s) in a.iter() {
            match b.get(i) {
                Some(t) if t.as_str() == s.as_str() => {}
                _ => return false,
            }
        }
        true
    }
}

impl<T: Integer> std::fmt::Debug for Dictionary<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dictionary")
            .field("len", &self.inner.values.count())
            .finish()
    }
}

impl<T: Integer> From<Vec64<String>> for Dictionary<T> {
    fn from(values: Vec64<String>) -> Self {
        Self::from_values(values)
    }
}

impl<T: Integer> From<Vec<String>> for Dictionary<T> {
    fn from(values: Vec<String>) -> Self {
        Self::from_values(Vec64::from(values))
    }
}

impl<T: Integer, S: Into<String>> FromIterator<S> for Dictionary<T> {
    fn from_iter<I: IntoIterator<Item = S>>(iter: I) -> Self {
        let owned: Vec64<String> = Vec64::from(iter.into_iter().map(Into::into).collect::<Vec<_>>());
        Self::from_values(owned)
    }
}

/// Practical entry cap used by `Dictionary::default()`. Narrow widths
/// receive their natural cap; `u32`/`u64` receive a soft cap of
/// `1 << 20` (1 048 576) entries because preallocating their natural
/// cap would reserve ~100 GB+ of virtual address space per dictionary
/// (allocators reject the request even with overcommit). Users who
/// genuinely need a larger cap can pass it via `with_capacity`.
///
/// `u8 -> 256`, `u16 -> 65 536`, `u32/u64 -> 1 048 576`.
const DEFAULT_WIDE_CAP: usize = 1 << 20;

#[inline]
fn max_cap<T: Integer>() -> usize {
    let type_max = T::max_value().to_usize().saturating_add(1);
    if type_max > DEFAULT_WIDE_CAP {
        DEFAULT_WIDE_CAP
    } else {
        type_max
    }
}

// =============================================================================
// CategoryManagerT - width-erased holder used by parent containers.
// =============================================================================

/// Width-erased `Dictionary` so a parent container (`SuperTable`,
/// `SuperArray`) can hold one entry per categorical column without being
/// generic over each column's width. Each variant carries the column's
/// typed `Dictionary`; cloning a `CategoryManagerT` is an Arc bump on the
/// underlying inner.
#[derive(Debug, Clone)]
pub enum CategoryManagerT {
    #[cfg(feature = "default_categorical_8")]
    U8(Dictionary<u8>),
    #[cfg(feature = "extended_categorical")]
    U16(Dictionary<u16>),
    #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
    U32(Dictionary<u32>),
    #[cfg(feature = "extended_categorical")]
    U64(Dictionary<u64>),
}

impl CategoryManagerT {
    /// Install a fresh dispatch from a batch's categorical column by
    /// cloning the chunk's dictionary (Arc bump). Subsequent chunks
    /// added through `add_remap_cat` merge their values into this
    /// dictionary and rebind themselves to share the same Arc.
    ///
    /// Returns `None` if the array is not categorical at any enabled width.
    pub fn install_from(array: &mut crate::Array) -> Option<Self> {
        use crate::{Array, TextArray};
        match array {
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            Array::TextArray(TextArray::Categorical32(arc)) => {
                let cat = Arc::make_mut(arc);
                Some(CategoryManagerT::U32(cat.dictionary.clone()))
            }
            #[cfg(feature = "default_categorical_8")]
            Array::TextArray(TextArray::Categorical8(arc)) => {
                let cat = Arc::make_mut(arc);
                Some(CategoryManagerT::U8(cat.dictionary.clone()))
            }
            #[cfg(feature = "extended_categorical")]
            Array::TextArray(TextArray::Categorical16(arc)) => {
                let cat = Arc::make_mut(arc);
                Some(CategoryManagerT::U16(cat.dictionary.clone()))
            }
            #[cfg(feature = "extended_categorical")]
            Array::TextArray(TextArray::Categorical64(arc)) => {
                let cat = Arc::make_mut(arc);
                Some(CategoryManagerT::U64(cat.dictionary.clone()))
            }
            _ => None,
        }
    }

    /// Dispatches on the dispatch variant and the array's categorical
    /// width, calling `Dictionary::add_remap_cat` on the matching pair.
    /// Width mismatch is a schema error upstream and is treated as a
    /// no-op here.
    pub fn add_remap_cat(&self, array: &mut crate::Array) {
        use crate::{Array, TextArray};
        match (self, array) {
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            (CategoryManagerT::U32(d), Array::TextArray(TextArray::Categorical32(arc))) => {
                d.add_remap_cat(Arc::make_mut(arc));
            }
            #[cfg(feature = "default_categorical_8")]
            (CategoryManagerT::U8(d), Array::TextArray(TextArray::Categorical8(arc))) => {
                d.add_remap_cat(Arc::make_mut(arc));
            }
            #[cfg(feature = "extended_categorical")]
            (CategoryManagerT::U16(d), Array::TextArray(TextArray::Categorical16(arc))) => {
                d.add_remap_cat(Arc::make_mut(arc));
            }
            #[cfg(feature = "extended_categorical")]
            (CategoryManagerT::U64(d), Array::TextArray(TextArray::Categorical64(arc))) => {
                d.add_remap_cat(Arc::make_mut(arc));
            }
            _ => {}
        }
    }

    /// Install or merge a sequence of categorical chunks into `slot`.
    ///
    /// If `slot` is `None`, the first categorical chunk seeds it via
    /// `install_from`. Subsequent chunks are merged through
    /// `add_remap_cat`, which adds each chunk's dictionary values into
    /// the shared store and remaps codes if the union shifted them.
    /// Non-categorical chunks are skipped.
    ///
    /// This is the entry point for parent containers (`SuperTable`,
    /// `SuperArray`) that hold a per-column manager slot. The per-chunk
    /// install / merge dichotomy lives here so the parent does not
    /// need to express it.
    pub fn add_remap_cats<'a, I>(slot: &mut Option<Self>, chunks: I)
    where
        I: IntoIterator<Item = &'a mut crate::Array>,
    {
        for chunk in chunks {
            match slot {
                Some(m) => m.add_remap_cat(chunk),
                None => *slot = Self::install_from(chunk),
            }
        }
    }

    /// Number of entries currently in the dispatch's dictionary.
    pub fn len(&self) -> usize {
        match self {
            #[cfg(feature = "default_categorical_8")]
            CategoryManagerT::U8(d) => d.len(),
            #[cfg(feature = "extended_categorical")]
            CategoryManagerT::U16(d) => d.len(),
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            CategoryManagerT::U32(d) => d.len(),
            #[cfg(feature = "extended_categorical")]
            CategoryManagerT::U64(d) => d.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_dictionary_starts_empty() {
        let d: Dictionary<u32> = Dictionary::new();
        assert_eq!(d.len(), 0);
        assert!(d.is_empty());
        assert_eq!(d.lookup("anything"), None);
    }

    #[test]
    fn intern_assigns_dense_sequential_codes() {
        let d: Dictionary<u32> = Dictionary::new();
        assert_eq!(d.add_cat("a"), Ok(0));
        assert_eq!(d.add_cat("b"), Ok(1));
        assert_eq!(d.add_cat("c"), Ok(2));
        assert_eq!(d.add_cat("a"), Ok(0));
        assert_eq!(d.len(), 3);
        let values: Vec<&str> = d.values().iter().map(|s| s.as_str()).collect();
        assert_eq!(values, vec!["a", "b", "c"]);
    }

    #[test]
    fn clones_share_state() {
        let d: Dictionary<u32> = Dictionary::new();
        let cloned = d.clone();
        assert!(d.shares_with(&cloned));
        // Update through one is visible through the other.
        assert_eq!(d.add_cat("a"), Ok(0));
        let values: Vec<&str> = cloned.values().iter().map(|s| s.as_str()).collect();
        assert_eq!(values, vec!["a"]);
    }

    #[test]
    fn detach_breaks_sharing() {
        let a: Dictionary<u32> = Dictionary::new();
        let _ = a.add_cat("x").unwrap();
        let mut b = a.clone();
        b.detach_to_owned();
        assert_eq!(a.values().get(0).map(|s| s.as_str()), Some("x"));
        assert_eq!(b.values().get(0).map(|s| s.as_str()), Some("x"));
        assert!(!a.shares_with(&b));
        let _ = b.add_cat("y").unwrap();
        assert_eq!(a.len(), 1);
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn is_prefix_of_recognises_prefix() {
        let a: Dictionary<u32> = Dictionary::from_iter(["x", "y"]);
        let b: Dictionary<u32> = Dictionary::from_iter(["x", "y", "z"]);
        assert!(a.is_prefix_of(&b));
        assert!(!b.is_prefix_of(&a));
        let c: Dictionary<u32> = Dictionary::from_iter(["x", "z"]);
        assert!(!a.is_prefix_of(&c));
    }

    /// Narrow-width cap is honoured exactly: 256 successful interns,
    /// the 257th returns Overflow with no leaked slot.
    #[test]
    fn intern_returns_overflow_at_u8_cap() {
        let d: Dictionary<u8> = Dictionary::new();
        for i in 0..256u32 {
            d.add_cat(&format!("v{i}")).unwrap();
        }
        assert_eq!(d.add_cat("overflow"), Err(DictionaryError::Overflow));
        assert_eq!(d.len(), 256);
    }

    /// Many threads concurrently interning into the same u8 dictionary
    /// must collectively succeed 256 times and not exceed the
    /// cap. No leaked slots or double-assigned codes.
    #[test]
    fn concurrent_intern_under_u8_cap_no_leaks() {
        use std::sync::Arc;
        use std::thread;

        let d: Arc<Dictionary<u8>> = Arc::new(Dictionary::new());
        let mut handles = Vec::new();
        for t in 0..16 {
            let d = Arc::clone(&d);
            handles.push(thread::spawn(move || {
                let mut successes = 0u32;
                let mut overflows = 0u32;
                for i in 0..100 {
                    let s = format!("t{t}_v{i}");
                    match d.add_cat(&s) {
                        Ok(_) => successes += 1,
                        Err(DictionaryError::Overflow) => overflows += 1,
                    }
                }
                (successes, overflows)
            }));
        }
        let (mut total_succ, mut total_ovf) = (0u32, 0u32);
        for h in handles {
            let (s, o) = h.join().unwrap();
            total_succ += s;
            total_ovf += o;
        }
        assert_eq!(d.len(), 256);
        assert_eq!(total_succ, 256);
        assert_eq!(total_ovf, 16 * 100 - 256);
    }

    /// Many threads interning distinct novel strings into a wide dict;
    /// every string ends up represented once.
    #[test]
    fn concurrent_intern_distinct_strings_no_duplicates() {
        use std::sync::Arc;
        use std::thread;

        let d: Arc<Dictionary<u32>> = Arc::new(Dictionary::new());
        let mut handles = Vec::new();
        for t in 0..8 {
            let d = Arc::clone(&d);
            handles.push(thread::spawn(move || {
                for i in 0..500 {
                    let _ = d.add_cat(&format!("t{t}_v{i}")).unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(d.len(), 8 * 500);
    }
}
