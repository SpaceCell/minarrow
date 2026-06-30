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

//! # **SharedBuffer Internal Module** - Backs *Buffer* for ZC MMAP and foreign buffer sharing
//!
//! Zero-copy, reference-counted byte buffer with 64-byte SIMD alignment.
//!
//! This is an internal module that backs the `Buffer` type supporting
//! the typed Arrays in *Minarrow*.

use std::sync::Arc;

use crate::Vec64;
use crate::structs::shared_buffer::internal::owned::{OWNED_VT, Owned};
use crate::structs::shared_buffer::internal::pvec::PromotableVec;
use crate::structs::shared_buffer::internal::vtable::{
    PROMO_EVEN_VT, PROMO_ODD_VT, PROMO64_EVEN_VT, PROMO64_ODD_VT, STATIC_VT, Vtable,
};
use core::ops::RangeBounds;
use core::{ptr, slice};
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Deref;
use std::sync::atomic::{AtomicPtr, AtomicUsize};

mod internal {
    pub(crate) mod owned;
    pub(crate) mod pvec;
    pub(crate) mod vtable;
}

/// Memfd-backed buffer for zero-copy cross-process sharing
///
/// Works on Linux only
#[cfg(all(target_os = "linux", feature = "memfd"))]
mod memfd;
#[cfg(all(target_os = "linux", feature = "memfd"))]
pub use memfd::MemfdBuffer;

/// # SharedBuffer
///
/// Zero-copy, reference-counted byte buffer with SIMD alignment support.
///
/// ## Purpose
/// This is an internal type that usually should not to be used directly.
/// It's primary purpose is to support the `Buffer` type in zero-copy IO cases,
/// enabling efficient reuse of bytes from network, memory-mapped files, and IPC
/// without copying data, whilst maintaining 64-byte SIMD alignment during operations.
///
/// ## Features
/// - O(1) pointer-based cloning and slicing
/// - Multiple backend sources: `Vec<u8>`, `Vec64<u8>`, MMAP, `Arc<[u8]>`, static slices
/// - Zero-copy extraction to owned types when unique
/// - Thread-safe reference counting via compact vtables
///
/// ## Usage
/// ```rust
/// use minarrow::SharedBuffer;
/// let sb = SharedBuffer::from_vec(vec![1,2,3,4,5]);
/// let slice = sb.slice(0..2);        // Zero-copy slice
/// let clone = sb.clone();            // O(1) reference increment
/// let owned = clone.into_vec();      // Extract to Vec<u8>
/// ```
///
/// Supports `from_vec64()` for SIMD-aligned buffers, `from_owner()` for arbitrary
/// containers, and `from_static()` for constant data.
#[repr(C)]
pub struct SharedBuffer {
    ptr: *const u8,
    len: usize,
    data: AtomicPtr<()>, // header or null
    vtable: &'static Vtable,
}

impl SharedBuffer {
    /// Constructs a new, empty `SharedBuffer`
    pub const fn new() -> Self {
        const EMPTY: &[u8] = &[];
        Self::from_static(EMPTY)
    }

    /// Constructs a `SharedBuffer` from a static slice
    pub const fn from_static(s: &'static [u8]) -> Self {
        Self {
            ptr: s.as_ptr(),
            len: s.len(),
            data: AtomicPtr::new(ptr::null_mut()),
            vtable: &STATIC_VT,
        }
    }

    pub fn from_vec(mut v: Vec<u8>) -> Self {
        let ptr = v.as_mut_ptr();
        let len = v.len();
        let cap = v.capacity();
        let raw = Box::into_raw(Box::new(PromotableVec::<Vec<u8>> {
            ref_cnt: AtomicUsize::new(1),
            inner: v,
        }));
        Self {
            ptr,
            len,
            data: AtomicPtr::new(raw.cast()),
            vtable: if cap & 1 == 0 {
                &PROMO_EVEN_VT
            } else {
                &PROMO_ODD_VT
            },
        }
    }

    /// Constructs a `SharedBuffer` from a SIMD-aligned Vec64<u8>.
    pub fn from_vec64(mut v: Vec64<u8>) -> Self {
        let ptr = v.as_mut_ptr();
        let len = v.len();
        let cap = v.capacity();
        let raw = Box::into_raw(Box::new(PromotableVec::<Vec64<u8>> {
            ref_cnt: AtomicUsize::new(1),
            inner: v,
        }));
        Self {
            ptr,
            len,
            data: AtomicPtr::new(raw.cast()),
            vtable: if cap & 1 == 0 {
                &PROMO64_EVEN_VT
            } else {
                &PROMO64_ODD_VT
            },
        }
    }
    /// Constructs a `SharedBuffer` from a SIMD-aligned Vec64<T>, reinterpreting
    /// the allocation as raw bytes. This is a zero-copy operation that transfers
    /// ownership of the allocation to the SharedBuffer.
    ///
    /// Used by Matrix::to_table to freeze the matrix data buffer so that
    /// per-column Buffer<f64> instances can share it via zero-copy windows.
    ///
    /// # Safety
    /// T must be a plain data type with no drop logic.
    pub unsafe fn from_vec64_typed<T>(v: Vec64<T>) -> Self {
        let size_of_t = std::mem::size_of::<T>();
        let byte_len = v
            .len()
            .checked_mul(size_of_t)
            .expect("from_vec64_typed: len * size_of::<T> overflow");
        let byte_cap =
            v.0.capacity()
                .checked_mul(size_of_t)
                .expect("from_vec64_typed: capacity * size_of::<T> overflow");
        let ptr = v.0.as_ptr() as *mut u8;
        std::mem::forget(v);
        let raw_vec = unsafe {
            Vec::from_raw_parts_in(ptr, byte_len, byte_cap, vec64::Vec64Alloc::default())
        };
        Self::from_vec64(Vec64(raw_vec))
    }

    /// Constructs a `SharedBuffer` from an `Arc<M>` where `M: AsRef<[u8]>`.
    ///
    /// Handles the double deref internally so callers don't need a wrapper
    /// type. Use `.slice()` for sub-region views.
    pub fn from_arc<M: ?Sized + AsRef<[u8]> + Send + Sync + 'static>(arc: Arc<M>) -> Self {
        // ArcOwner adapts Arc<M> to AsRef<[u8]> for from_owner.
        // Always Sized since Arc is a pointer regardless of M.
        struct ArcOwner<M: ?Sized>(Arc<M>);
        impl<M: ?Sized + AsRef<[u8]>> AsRef<[u8]> for ArcOwner<M> {
            #[inline]
            fn as_ref(&self) -> &[u8] {
                (*self.0).as_ref()
            }
        }
        unsafe impl<M: ?Sized + Send + Sync> Send for ArcOwner<M> {}
        unsafe impl<M: ?Sized + Send + Sync> Sync for ArcOwner<M> {}

        Self::from_owner(ArcOwner(arc))
    }

    /// Constructs a `SharedBuffer` from an arbitrary owner (e.g. Arc<[u8]>, mmap, etc).
    ///
    /// The owner must implement `AsRef<[u8]> + Send + Sync + 'static`.
    pub fn from_owner<T>(owner: T) -> Self
    where
        T: AsRef<[u8]> + Send + Sync + 'static,
    {
        unsafe fn drop_typed<T: AsRef<[u8]> + Send + Sync + 'static>(ptr: *mut ()) {
            unsafe {
                drop(Box::from_raw(ptr as *mut Owned<T>));
            }
        }
        let raw: *mut Owned<T> = Box::into_raw(Box::new(Owned {
            ref_cnt: AtomicUsize::new(1),
            drop_fn: drop_typed::<T>,
            owner,
        }));
        let buf = unsafe { (*raw).owner.as_ref() };
        Self {
            ptr: buf.as_ptr(),
            len: buf.len(),
            data: AtomicPtr::new(raw.cast()),
            vtable: &OWNED_VT,
        }
    }

    /// Returns the number of bytes in this buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if this buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a read-only view of the data as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Returns a zero-copy slice of this buffer's data.
    ///
    /// Panics if range is out of bounds.
    pub fn slice(&self, range: impl RangeBounds<usize>) -> Self {
        use core::ops::Bound::*;
        let start = match range.start_bound() {
            Unbounded => 0,
            Included(&n) => n,
            Excluded(&n) => n + 1,
        };
        let end = match range.end_bound() {
            Unbounded => self.len,
            Included(&n) => n + 1,
            Excluded(&n) => n,
        };
        assert!(start <= end && end <= self.len);
        if start == end {
            return SharedBuffer::new();
        }

        let mut s = self.clone();
        s.ptr = unsafe { s.ptr.add(start) };
        s.len = end - start;
        s
    }

    /// Attempts to convert into an owned `Vec<u8>`.
    ///
    /// If this is the unique owner, and it was originally allocated with a Vec<u8>,
    /// this is zero-copy. Otherwise, the data is cloned.
    #[inline]
    pub fn into_vec(self) -> Vec<u8> {
        // move‑out without running Drop first:
        let me = core::mem::ManuallyDrop::new(self);
        unsafe { (me.vtable.to_vec)(&me.data, me.ptr, me.len) }
    }

    /// Attempts to convert into an owned, SIMD-aligned `Vec64<u8>`.
    ///
    /// If this is the unique owner, and it was originally allocated with a Vec64<u8>
    /// this is zero-copy. Otherwise, the data is cloned.
    #[inline]
    pub fn into_vec64(self) -> Vec64<u8> {
        let me = core::mem::ManuallyDrop::new(self);
        unsafe { (me.vtable.to_vec64)(&me.data, me.ptr, me.len) }
    }

    /// Returns `true` when `a` and `b` view the same underlying allocation.
    ///
    /// This is an identity test, distinct from `PartialEq` which compares
    /// byte content. Two `SharedBuffer`s produced by cloning a common origin
    /// share both the base pointer and the byte length; any sub-slice created via
    /// `.slice()` changes `ptr`, so it will no longer identity-match.
    ///
    /// Callers may use this to verify that multiple columns of a `TableV` refer
    /// back to the same `SharedBuffer` before treating them as a contiguous
    /// column-major allocation.
    #[inline]
    pub fn ptr_eq(a: &Self, b: &Self) -> bool {
        std::ptr::eq(a.ptr, b.ptr) && a.len == b.len
    }

    /// Returns `true` if this buffer is the unique owner of its underlying storage.
    ///
    /// ## Behaviour by backend:
    /// - **Vec / Vec64** (`PROMO*`, `OWNED_VT`): Returns `true` only if the internal
    ///   reference count is `1`.
    /// - **Static buffers** (`STATIC_VT`): Always returns `true`, as the memory is
    ///   immutable, globally shared, and never deallocated.  
    ///   This does **not** imply transfer of ownership - only that no additional
    ///   runtime references are tracked.
    /// - **Foreign owners** (e.g., `Arc<[u8]>`): Returns `true` only when there are
    ///   no other references to the underlying allocation.
    ///
    /// This method is primarily for determining whether zero-copy
    /// conversion to an owned type is possible without cloning the underlying data.
    /// For static buffers, this will always report `true` because the data is
    /// permanently resident and immutable.
    #[inline]
    pub fn is_unique(&self) -> bool {
        unsafe { (self.vtable.is_unique)(&self.data) }
    }
}

impl Clone for SharedBuffer {
    /// Clones this buffer. Always O(1), increases refcount if needed.
    fn clone(&self) -> Self {
        unsafe { (self.vtable.clone)(&self.data, self.ptr, self.len) }
    }
}
impl Drop for SharedBuffer {
    /// Drops this buffer, decrementing the reference count and releasing memory if unique.
    fn drop(&mut self) {
        unsafe { (self.vtable.drop)(&mut self.data, self.ptr, self.len) }
    }
}

/// Default for an empty buffer (same as new()).
impl Default for SharedBuffer {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Compare for equality (byte-wise).
impl PartialEq for SharedBuffer {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}
impl Eq for SharedBuffer {}

impl PartialEq<[u8]> for SharedBuffer {
    #[inline]
    fn eq(&self, other: &[u8]) -> bool {
        self.as_slice() == other
    }
}
impl PartialEq<SharedBuffer> for [u8] {
    #[inline]
    fn eq(&self, other: &SharedBuffer) -> bool {
        self == other.as_slice()
    }
}
impl PartialEq<Vec<u8>> for SharedBuffer {
    #[inline]
    fn eq(&self, other: &Vec<u8>) -> bool {
        self.as_slice() == other.as_slice()
    }
}
impl PartialEq<SharedBuffer> for Vec<u8> {
    #[inline]
    fn eq(&self, other: &SharedBuffer) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl PartialOrd for SharedBuffer {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}
impl Ord for SharedBuffer {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl Hash for SharedBuffer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state)
    }
}

impl fmt::Debug for SharedBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("SharedBuffer")
            .field(&self.as_slice())
            .finish()
    }
}

/// Deref to [u8] for zero-copy APIs.
impl Deref for SharedBuffer {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsRef<[u8]> for SharedBuffer {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}
impl Borrow<[u8]> for SharedBuffer {
    #[inline]
    fn borrow(&self) -> &[u8] {
        self.as_slice()
    }
}

/// From/Into for Vec and Vec64.
impl From<Vec<u8>> for SharedBuffer {
    #[inline]
    fn from(v: Vec<u8>) -> Self {
        Self::from_vec(v)
    }
}
impl From<Vec64<u8>> for SharedBuffer {
    #[inline]
    fn from(v: Vec64<u8>) -> Self {
        Self::from_vec64(v)
    }
}
impl From<&'static [u8]> for SharedBuffer {
    #[inline]
    fn from(s: &'static [u8]) -> Self {
        Self::from_static(s)
    }
}

/// IntoIterator over bytes (by value).
impl IntoIterator for SharedBuffer {
    type Item = u8;
    type IntoIter = std::vec::IntoIter<u8>;
    fn into_iter(self) -> Self::IntoIter {
        self.into_vec().into_iter()
    }
}

/// By-ref iterator over bytes.
impl<'a> IntoIterator for &'a SharedBuffer {
    type Item = &'a u8;
    type IntoIter = std::slice::Iter<'a, u8>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

/// Construction from iterator.
impl FromIterator<u8> for SharedBuffer {
    #[inline]
    fn from_iter<I: IntoIterator<Item = u8>>(iter: I) -> Self {
        let v: Vec<u8> = iter.into_iter().collect();
        Self::from_vec(v)
    }
}

impl fmt::Display for SharedBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match std::str::from_utf8(self.as_slice()) {
            Ok(s) => f.write_str(s),
            Err(_) => {
                // fallback to hex
                for byte in self.as_slice() {
                    write!(f, "{:02x}", byte)?;
                }
                Ok(())
            }
        }
    }
}

// SAFETY: SharedBuffer is always safe to send and share between threads.
unsafe impl Send for SharedBuffer {}
unsafe impl Sync for SharedBuffer {}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn roundtrip_vec() {
        let v = vec![1, 2, 3, 4, 5];
        let sb = SharedBuffer::from_vec(v);
        assert_eq!(sb.as_slice(), &[1, 2, 3, 4, 5]);
        let v2 = sb.clone().into_vec();
        assert_eq!(v2, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn roundtrip_vec64() {
        let mut v64 = Vec64::with_capacity(5);
        v64.extend_from_slice(&[9, 8, 7, 6, 5]);
        let sb = SharedBuffer::from_vec64(v64);
        assert_eq!(sb.as_slice(), &[9, 8, 7, 6, 5]);
        let v64_out = sb.clone().into_vec64();
        assert_eq!(v64_out.as_slice(), &[9, 8, 7, 6, 5]);
    }

    #[test]
    fn owned_unique_check() {
        let mmap = Arc::new([10u8, 11, 12, 13]) as Arc<[u8]>;
        let sb = SharedBuffer::from_owner(mmap);
        assert!(sb.is_unique());
        let sb2 = sb.clone();
        assert!(!sb.is_unique());
        drop(sb2);
        assert!(sb.is_unique());
    }
}
