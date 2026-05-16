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

//! # **Dictionary Module** - *Append-only String Dictionary for Categorical Arrays*
//!
//! Backs `CategoricalArray<T>`. Pairs the ordered list of unique strings with
//! an internal hashmap so interning runs in O(1) rather than O(n) linear scan.
//!
//! ## Two ownership modes
//! `Dictionary<T>` is an enum mirroring how `Buffer<T>` distinguishes
//! standalone from shared storage:
//!
//! - `Owned(DictionaryInner<T>)`: the categorical owns its dictionary outright
//!   and is free to mutate it. This is the standalone path.
//! - `Shared(Arc<DictionaryInner<T>>)`: the categorical holds an immutable view
//!   of a dictionary that is canonically held by a parent (`SuperTable` /
//!   `SuperArray`). All sibling batches in that parent point at the same Arc,
//!   so codes are mutually meaningful across the entire structure. Mutating a
//!   `Shared` dictionary directly panics; growth must go through the parent's
//!   mediated API so the parent can rebind every sibling.
//!
//! ## Append-only invariant
//! Once a string is interned and assigned a code, that mapping is permanent
//! in both modes: entries are never reordered, replaced, or removed. A
//! dictionary that is a prefix of another agrees on every code they share,
//! which is what makes `Shared` rebinding cheap when the parent grows the
//! canonical: the old codes remain valid against the new Arc.
//!
//! ## Generic over T
//! `T` is the index width of the owning `CategoricalArray<T>`. Codes are
//! minted as `T` directly, capping cardinality at the width's limit
//! (256 for `u8`, etc.) and removing the cast ceremony at call-sites.

use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

use ::vec64::Vec64;

use crate::traits::type_unions::Integer;

/// Errors that may arise from mutating a `Dictionary<T>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DictionaryError {
    /// The dictionary is in `Shared` mode, meaning it is bound to a canonical
    /// dictionary held by a parent (`SuperTable` / `SuperArray`). Growth must
    /// go through the parent's mediated API so every sibling batch stays
    /// coherent with the canonical. Attempting to grow it in isolation would
    /// silently desynchronise the parent and other batches.
    ///
    /// On receiving this error, route the operation through the parent that
    /// owns the categorical column (typically a `SuperTable` method that
    /// updates the canonical and rebinds sibling batches).
    Shared,
    /// The new cardinality would exceed the capacity of the index type `T`
    /// (e.g. 256 entries for `u8`). The dictionary is left unchanged.
    Overflow,
}

impl fmt::Display for DictionaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shared => write!(
                f,
                "dictionary is Shared; growth must be requested through the parent SuperTable / SuperArray"
            ),
            Self::Overflow => write!(
                f,
                "dictionary cardinality would exceed the capacity of the index type"
            ),
        }
    }
}

impl std::error::Error for DictionaryError {}

#[cfg(feature = "fast_hash")]
type DictIndex<T> = ahash::AHashMap<String, T>;
#[cfg(not(feature = "fast_hash"))]
type DictIndex<T> = std::collections::HashMap<String, T>;

/// # DictionaryInner
///
/// The actual storage backing a `Dictionary<T>`: an ordered list of unique
/// strings paired with a reverse-lookup hashmap. Lives inside `Dictionary`'s
/// `Owned` variant (inline) or behind an `Arc` in the `Shared` variant.
#[derive(Clone, Debug, Default)]
pub struct DictionaryInner<T: Integer> {
    /// Ordered list of unique values. Position is the code.
    pub values: Vec64<String>,
    /// Reverse index from string to code. Populated in lockstep with `values`.
    index: DictIndex<T>,
}

impl<T: Integer> DictionaryInner<T> {
    /// Constructs an empty inner dictionary.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Constructs an empty inner dictionary with capacity reserved for `cap` entries.
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            values: Vec64::with_capacity(cap),
            index: DictIndex::<T>::with_capacity(cap),
        }
    }

    /// Builds from an ordered list of values, preserving the input verbatim
    /// and rebuilding the index. Codes already minted against `values[i]`
    /// remain valid because positions are not changed. Panics if the input
    /// length exceeds the capacity of `T`.
    pub fn from_values(values: impl Into<Vec64<String>>) -> Self {
        let values = values.into();
        let mut index = DictIndex::<T>::with_capacity(values.len());
        for (i, s) in values.iter().enumerate() {
            let code = T::try_from(i).ok().unwrap_or_else(|| {
                panic!(
                    "Dictionary input has {} entries, exceeds capacity of index type {}",
                    values.len(),
                    std::any::type_name::<T>()
                )
            });
            index.entry(s.clone()).or_insert(code);
        }
        Self { values, index }
    }

    /// Number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// True if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns the code for `s` if already interned, otherwise `None`.
    #[inline]
    pub fn lookup(&self, s: &str) -> Option<T> {
        self.index.get(s).copied()
    }

    /// Interns `s`, returning its code. Existing entries keep their code;
    /// new entries receive the next sequential code. Returns
    /// `Err(DictionaryError::Overflow)` if the new cardinality would exceed
    /// the capacity of `T`, leaving the dictionary unchanged.
    #[inline]
    pub fn intern(&mut self, s: &str) -> Result<T, DictionaryError> {
        if let Some(&code) = self.index.get(s) {
            return Ok(code);
        }
        let idx = self.values.len();
        let code = T::try_from(idx).map_err(|_| DictionaryError::Overflow)?;
        self.values.push(s.to_owned());
        self.index.insert(s.to_owned(), code);
        Ok(code)
    }

    /// Reserves space for at least `additional` more entries.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
        self.index.reserve(additional);
    }
}

impl<T: Integer> PartialEq for DictionaryInner<T> {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
    }
}

impl<T: Integer> Deref for DictionaryInner<T> {
    type Target = [String];
    #[inline]
    fn deref(&self) -> &[String] {
        &self.values
    }
}

/// # Dictionary
///
/// Append-only string dictionary in one of two ownership modes.
///
/// - `Owned(DictionaryInner<T>)`: standalone. The owning categorical can mutate
///   freely via [`Dictionary::intern`].
/// - `Shared(Arc<DictionaryInner<T>>)`: linked to a canonical dictionary owned
///   by a parent (`SuperTable` / `SuperArray`). Mutation panics; growth must
///   be requested through the parent's mediated API so every sibling batch
///   stays bound to the same canonical Arc.
///
/// Reads and lookups are uniform across both variants. The `Shared` fast-path
/// for cross-batch identity is exposed via [`Dictionary::shares_with`], which
/// is `Arc::ptr_eq` under the hood.
#[derive(Clone, Debug)]
pub enum Dictionary<T: Integer> {
    /// Standalone ownership. Mutable.
    Owned(DictionaryInner<T>),
    /// Linked view of a canonical dictionary owned by a parent. Immutable
    /// from the categorical's side; growth happens via the parent's API.
    Shared(Arc<DictionaryInner<T>>),
}

impl<T: Integer> Default for Dictionary<T> {
    fn default() -> Self {
        Self::Owned(DictionaryInner::default())
    }
}

impl<T: Integer> Dictionary<T> {
    /// Constructs an empty owned dictionary.
    #[inline]
    pub fn new() -> Self {
        Self::Owned(DictionaryInner::new())
    }

    /// Constructs an empty owned dictionary with reserved capacity.
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self::Owned(DictionaryInner::with_capacity(cap))
    }

    /// Builds an owned dictionary from a list of values.
    pub fn from_values(values: impl Into<Vec64<String>>) -> Self {
        Self::Owned(DictionaryInner::from_values(values))
    }

    /// Returns the underlying values as a slice.
    #[inline]
    pub fn values(&self) -> &[String] {
        match self {
            Self::Owned(d) => &d.values,
            Self::Shared(a) => &a.values,
        }
    }

    /// Number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Owned(d) => d.len(),
            Self::Shared(a) => a.len(),
        }
    }

    /// True if the dictionary is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the code for `s` if already interned, otherwise `None`.
    #[inline]
    pub fn lookup(&self, s: &str) -> Option<T> {
        match self {
            Self::Owned(d) => d.lookup(s),
            Self::Shared(a) => a.lookup(s),
        }
    }

    /// Interns `s`, returning its code.
    ///
    /// Returns `Err(DictionaryError::Shared)` if this is a `Shared`
    /// dictionary: growth on a shared dictionary must go through the parent
    /// (`SuperTable` / `SuperArray`) that owns the canonical, so every
    /// sibling batch stays coherent. Callers in that situation should route
    /// the operation through the parent's mediated growth API.
    ///
    /// Returns `Err(DictionaryError::Overflow)` if the new cardinality would
    /// exceed the capacity of `T`.
    #[inline]
    pub fn intern(&mut self, s: &str) -> Result<T, DictionaryError> {
        match self {
            Self::Owned(d) => d.intern(s),
            Self::Shared(_) => Err(DictionaryError::Shared),
        }
    }

    /// True if this dictionary is `Shared`.
    #[inline]
    pub fn is_shared(&self) -> bool {
        matches!(self, Self::Shared(_))
    }

    /// True if this dictionary is `Owned`.
    #[inline]
    pub fn is_owned(&self) -> bool {
        matches!(self, Self::Owned(_))
    }

    /// True if both dictionaries are `Shared` and point at the same canonical
    /// (`Arc::ptr_eq` under the hood). Two `Owned` dictionaries are never
    /// considered shared even if their contents match.
    #[inline]
    pub fn shares_with(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Shared(a), Self::Shared(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }

    /// Returns the inner `Arc` if this is `Shared`, otherwise `None`.
    #[inline]
    pub fn as_shared(&self) -> Option<&Arc<DictionaryInner<T>>> {
        match self {
            Self::Shared(a) => Some(a),
            Self::Owned(_) => None,
        }
    }

    /// Converts a `Shared` dictionary back to `Owned` by cloning the inner
    /// state. No-op if already `Owned`. Used by extraction paths that release
    /// a batch from its parent and need a self-contained dictionary again.
    pub fn into_owned(self) -> DictionaryInner<T> {
        match self {
            Self::Owned(d) => d,
            Self::Shared(a) => (*a).clone(),
        }
    }

    /// If this dictionary is `Shared`, copy the snapshot into a private
    /// `Owned` dictionary and replace `self` with it. No-op if already
    /// `Owned`.
    ///
    /// Mutating call-sites use this so they never have to return a
    /// `Shared` error. The categorical's existing codes still mean the same
    /// strings (the new dictionary is an exact copy), but anything added
    /// from here on lives only in this categorical and other categoricals
    /// that held the same shared dictionary will not see those new values.
    /// A `log::warn` is emitted so the caller can find where this happened
    /// and silence or filter it if expected.
    pub fn demote_to_owned(&mut self) {
        if let Self::Shared(arc) = self {
            log::warn!(
                target: "minarrow::dictionary",
                "Categorical dictionary was Shared and is now Owned. \
                 Any new values added from here will not be seen by other \
                 categoricals that held the same shared dictionary."
            );
            let owned = (**arc).clone();
            *self = Self::Owned(owned);
        }
    }

    /// True if `self.values()` is a prefix of `other.values()`, meaning every
    /// code valid against `self` decodes to the same string in `other`.
    pub fn is_prefix_of(&self, other: &Dictionary<T>) -> bool {
        let a = self.values();
        let b = other.values();
        a.len() <= b.len() && a.iter().zip(b.iter()).all(|(x, y)| x == y)
    }
}

impl<T: Integer> PartialEq for Dictionary<T> {
    fn eq(&self, other: &Self) -> bool {
        self.values() == other.values()
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

impl<T: Integer> From<DictionaryInner<T>> for Dictionary<T> {
    fn from(inner: DictionaryInner<T>) -> Self {
        Self::Owned(inner)
    }
}

impl<T: Integer, S: Into<String>> FromIterator<S> for Dictionary<T> {
    fn from_iter<I: IntoIterator<Item = S>>(iter: I) -> Self {
        let owned: Vec64<String> = Vec64::from(iter.into_iter().map(Into::into).collect::<Vec<_>>());
        Self::from_values(owned)
    }
}

/// # CategoryManager
///
/// Owns the canonical dictionary for a single categorical column on behalf
/// of a parent container (`SuperTable` / `SuperArray`), or for a set of
/// peer-linked standalone categoricals (`sync_dict`).
///
/// ## Concurrency model
///
/// `intern(&self, value)` may be called concurrently from multiple threads.
/// The canonical is stored as `Arc<DictionaryInner<T>>` so children's
/// `Shared` snapshots are immutable Arc clones. Growth replaces the Arc;
/// existing snapshots remain valid against any new superset Arc by the
/// append-only invariant.
///
/// ## Storage backends
///
/// - Default: `Mutex<Arc<DictionaryInner<T>>>`. Growth takes the mutex
///   briefly to clone-extend-swap. Reads on children are lock-free because
///   each child holds its own snapshot Arc and never consults the manager
///   on the hot path.
/// - Feature `contended_dict`: `arc_swap::ArcSwap<DictionaryInner<T>>`.
///   Growth uses a CAS-retry loop; no lock acquired. Use when concurrent
///   novel-value insertion is a hot path (heavy multi-thread ingestion of
///   genuinely new strings).
pub struct CategoryManager<T: Integer> {
    #[cfg(not(feature = "contended_dict"))]
    canonical: std::sync::Mutex<Arc<DictionaryInner<T>>>,
    #[cfg(feature = "contended_dict")]
    canonical: arc_swap::ArcSwap<DictionaryInner<T>>,
}

impl<T: Integer> CategoryManager<T> {
    /// Empty manager.
    pub fn new() -> Self {
        Self::from_inner(DictionaryInner::default())
    }

    /// Construct from an existing dictionary. Used by parents when absorbing
    /// a batch's `Owned` dictionary into the canonical for the first time.
    pub fn from_inner(inner: DictionaryInner<T>) -> Self {
        let arc = Arc::new(inner);
        #[cfg(not(feature = "contended_dict"))]
        {
            Self {
                canonical: std::sync::Mutex::new(arc),
            }
        }
        #[cfg(feature = "contended_dict")]
        {
            Self {
                canonical: arc_swap::ArcSwap::from(arc),
            }
        }
    }

    /// Cheap clone of the current canonical Arc. Hand to a freshly bound
    /// `Shared` categorical as its snapshot.
    pub fn snapshot(&self) -> Arc<DictionaryInner<T>> {
        #[cfg(not(feature = "contended_dict"))]
        {
            Arc::clone(&*self.canonical.lock().expect("canonical mutex poisoned"))
        }
        #[cfg(feature = "contended_dict")]
        {
            self.canonical.load_full()
        }
    }

    /// Intern `value` into the canonical and return its code.
    ///
    /// Concurrent-safe via `&self`. The default backend takes the mutex,
    /// looks up, and either returns the existing code or clones-extends-swaps
    /// the canonical Arc — all under the lock so observers are consistent.
    /// The `contended_dict` backend does the same logic via a lock-free
    /// CAS-retry loop on the canonical Arc.
    pub fn intern(&self, value: &str) -> Result<T, DictionaryError> {
        #[cfg(not(feature = "contended_dict"))]
        {
            let mut guard = self.canonical.lock().expect("canonical mutex poisoned");
            if let Some(code) = guard.lookup(value) {
                return Ok(code);
            }
            // Need to extend. Clone the inner, intern locally, swap the Arc.
            let mut new_inner: DictionaryInner<T> = (**guard).clone();
            let code = new_inner.intern(value)?;
            *guard = Arc::new(new_inner);
            Ok(code)
        }
        #[cfg(feature = "contended_dict")]
        {
            loop {
                let current = self.canonical.load();
                if let Some(code) = current.lookup(value) {
                    return Ok(code);
                }
                let mut new_inner: DictionaryInner<T> = (**current).clone();
                let code = new_inner.intern(value)?;
                let new_arc = Arc::new(new_inner);
                let prev = self.canonical.compare_and_swap(&*current, new_arc);
                if Arc::ptr_eq(&prev, &*current) {
                    return Ok(code);
                }
                // CAS lost — another writer interned concurrently. Retry:
                // on the next pass we will either find `value` already
                // present (returns existing code) or extend off the new
                // canonical.
            }
        }
    }
}

impl<T: Integer> Default for CategoryManager<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Integer> Clone for CategoryManager<T> {
    /// Clones the manager by taking a snapshot of the canonical and wrapping
    /// it in a fresh backing store. The clone is independent of the original:
    /// future growth on either side is not visible to the other.
    fn clone(&self) -> Self {
        Self::from_inner((*self.snapshot()).clone())
    }
}

impl<T: Integer> std::fmt::Debug for CategoryManager<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let snap = self.snapshot();
        f.debug_struct("CategoryManager")
            .field("len", &snap.len())
            .finish()
    }
}

/// # CategoryDispatch
///
/// `CategoryManager` width-erased across index types so a parent container
/// (`SuperTable`) can hold one entry per categorical column without being
/// generic over each column's width. Parents pattern-match on this to reach
/// the typed manager and call its `intern` / `snapshot` methods.
#[derive(Debug, Clone)]
pub enum CategoryDispatch {
    #[cfg(feature = "default_categorical_8")]
    U8(CategoryManager<u8>),
    #[cfg(feature = "extended_categorical")]
    U16(CategoryManager<u16>),
    #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
    U32(CategoryManager<u32>),
    #[cfg(feature = "extended_categorical")]
    U64(CategoryManager<u64>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_dictionary_starts_empty() {
        let d: Dictionary<u32> = Dictionary::new();
        assert_eq!(d.len(), 0);
        assert!(d.is_empty());
        assert!(d.is_owned());
        assert!(!d.is_shared());
        assert_eq!(d.lookup("anything"), None);
    }

    #[test]
    fn intern_assigns_dense_sequential_codes() {
        let mut d: Dictionary<u32> = Dictionary::new();
        assert_eq!(d.intern("a"), Ok(0));
        assert_eq!(d.intern("b"), Ok(1));
        assert_eq!(d.intern("c"), Ok(2));
        assert_eq!(d.intern("a"), Ok(0));
        assert_eq!(d.len(), 3);
        assert_eq!(d.values(), &["a", "b", "c"]);
    }

    #[test]
    fn from_values_preserves_input_verbatim() {
        let d: Dictionary<u32> = Dictionary::from(Vec64::from(vec![
            "a".to_string(),
            "b".to_string(),
            "a".to_string(),
            "c".to_string(),
        ]));
        assert_eq!(d.values(), &["a", "b", "a", "c"]);
        assert_eq!(d.lookup("a"), Some(0));
        assert_eq!(d.lookup("b"), Some(1));
        assert_eq!(d.lookup("c"), Some(3));
    }

    #[test]
    fn shared_dictionaries_compare_by_arc_pointer() {
        let inner = DictionaryInner::<u32>::from_values(Vec64::from(vec![
            "a".to_string(),
            "b".to_string(),
        ]));
        let arc = Arc::new(inner);
        let a: Dictionary<u32> = Dictionary::Shared(Arc::clone(&arc));
        let b: Dictionary<u32> = Dictionary::Shared(Arc::clone(&arc));
        assert!(a.shares_with(&b));

        // A separately constructed Owned with the same contents is equal but
        // not shared.
        let c: Dictionary<u32> = Dictionary::from_iter(["a", "b"]);
        assert_eq!(a, c);
        assert!(!a.shares_with(&c));
    }

    #[test]
    fn intern_on_shared_returns_shared_error() {
        let inner = DictionaryInner::<u32>::from_values(Vec64::from(vec!["a".to_string()]));
        let mut d: Dictionary<u32> = Dictionary::Shared(Arc::new(inner));
        assert_eq!(d.intern("b"), Err(DictionaryError::Shared));
    }

    #[test]
    fn into_owned_clones_shared_state() {
        let inner = DictionaryInner::<u32>::from_values(Vec64::from(vec![
            "a".to_string(),
            "b".to_string(),
        ]));
        let d: Dictionary<u32> = Dictionary::Shared(Arc::new(inner));
        let inner = d.into_owned();
        assert_eq!(inner.values.as_slice(), &["a", "b"]);
    }

    #[test]
    fn is_prefix_of_recognises_shared_prefix() {
        let a: Dictionary<u32> = Dictionary::from_iter(["x", "y"]);
        let b: Dictionary<u32> = Dictionary::from_iter(["x", "y", "z"]);
        assert!(a.is_prefix_of(&b));
        assert!(!b.is_prefix_of(&a));
        let c: Dictionary<u32> = Dictionary::from_iter(["x", "z"]);
        assert!(!a.is_prefix_of(&c));
    }

    #[test]
    fn intern_returns_overflow_on_narrow_width() {
        let mut d: Dictionary<u8> = Dictionary::new();
        for i in 0..256u32 {
            d.intern(&format!("v{i}")).unwrap();
        }
        // u8 cardinality cap; next intern of a NEW value yields Overflow.
        assert_eq!(d.intern("overflow"), Err(DictionaryError::Overflow));
        // Length unchanged after the failed intern.
        assert_eq!(d.len(), 256);
    }

    #[test]
    fn category_manager_serial_intern() {
        let m: CategoryManager<u32> = CategoryManager::new();
        assert_eq!(m.intern("a"), Ok(0));
        assert_eq!(m.intern("b"), Ok(1));
        // Re-intern returns existing code.
        assert_eq!(m.intern("a"), Ok(0));
        let snap = m.snapshot();
        assert_eq!(snap.values.as_slice(), &["a", "b"]);
    }

    #[test]
    fn category_manager_concurrent_intern() {
        use std::sync::Arc as StdArc;
        use std::thread;

        // Eight threads inserting strings drawn from four distinct prefixes.
        // Even if multiple threads simultaneously try to intern the same
        // novel value, the manager must end up with exactly the union set.
        let m: StdArc<CategoryManager<u32>> = StdArc::new(CategoryManager::new());
        let mut handles = Vec::new();
        for t in 0..8 {
            let m = StdArc::clone(&m);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    let _ = m.intern(&format!("v{}_{i}", t % 4)).unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let snap = m.snapshot();
        // 4 distinct prefixes × 100 indices = 400 unique strings expected.
        assert_eq!(snap.values.len(), 400);
    }
}
