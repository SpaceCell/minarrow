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

//! # **LBuffer Module** - *Single-writer, multi-reader, append-only typed buffer*
//!
//! `LBuffer<T>` is the writer-side handle on a fixed-capacity typed buffer
//! with a stable base address. [`LBufferV`] is the shared read view: each
//! call reads the published length with one `Acquire` load, and indexed
//! access carries no atomics at all.
//!
//! ## Design
//!
//! - One contiguous allocation, fixed capacity, stable base address for the
//!   buffer's lifetime. `LBuffer` never reallocates; `push` returns
//!   `Err(value)` once at capacity.
//! - `push` is `&mut self`, so the single-writer invariant is
//!   compile-enforced.
//! - Any number of [`LBufferV`] handles can be created via
//!   [`LBuffer::view`]. They share the allocation through an `Arc` and may
//!   outlive the `LBuffer`.
//!
//! ## Memory model
//!
//! The writer fills `[len, capacity)` through the base pointer, then
//! publishes each element with a `Release` store on the atomic length.
//! Readers `Acquire`-load the length and access `[0, len)` through the same
//! stable base. The two regions are disjoint by construction, and the
//! atomic length carries happens-before from every write to any reader
//! that observes it.
//!
//! The published length is monotonically non-decreasing: a view created at
//! any point observes elements appended afterwards, and any length a reader
//! has observed remains a valid bound for the buffer's lifetime.
//!
//! ## Sealing
//!
//! [`LBuffer::seal`] flips an atomic flag marking the contents final. After
//! seal, `push` returns `Err(value)` and the published length no longer
//! changes. Views observe the flag via [`LBufferV::is_sealed`]. Dropping
//! the `LBuffer` also seals, so views see the flag even when the writer
//! drops without calling `seal`. The allocation is released through the
//! shared `Arc` once the `LBuffer` and all views drop.
//!
//! ## Validity masks
//!
//! [`LBuffer::with_capacity_masked`] pairs the value buffer with a
//! bit-packed validity buffer. [`LBuffer::push`] and
//! [`LBuffer::push_null`] keep the two in step, and
//! [`LBuffer::as_bitmask`] shares the validity as a [`crate::Bitmask`].
//!
//! ## Allocation
//!
//! Backing memory is obtained through `Vec64Alloc` (the alias for
//! `MAllocPg64` under the `mmap` feature). Allocations >= 2 MiB go through
//! the mmap path and pick up `MADV_HUGEPAGE` automatically; smaller
//! allocations use the 64-byte aligned heap path.
//!
//! ## Access pattern
//!
//! For tight loops, capture the length once via [`LBufferV::len`] (or via
//! [`LBufferV::as_slice`], which carries it in the slice header) and index
//! without further atomics:
//!
//! ```
//! # use minarrow::LBuffer;
//! let mut buf = LBuffer::<u64>::with_capacity(1024);
//! for i in 0..100u64 { buf.push(i).unwrap(); }
//! let v = buf.view();
//!
//! // Capture the bound once, then index with no atomics.
//! let n = v.len();                            // one Acquire load
//! let mut sum = 0u64;
//! for i in 0..n {
//!     // SAFETY: i < n, n was observed via Acquire, and the published
//!     // length is monotonic so i remains a valid index.
//!     sum += unsafe { *v.get_unchecked(i) };
//! }
//! assert_eq!(sum, (0..100u64).sum::<u64>());
//! ```
//!
//! [`as_slice`](LBufferV::as_slice) is the equivalent safe form for
//! slice-shaped iteration.

use core::alloc::{Allocator, Layout};
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};

use crate::{Bitmask, Buffer, Vec64Alloc};

/// # LBuffer
///
/// Writer-side handle on a single-writer, multi-reader, append-only
/// typed buffer.
///
/// One contiguous allocation, fixed capacity, stable base address for the
/// buffer's lifetime. `push` is `&mut self`, so there is one writer; views
/// created via [`view`](Self::view) read its published state.
///
/// Dropping the `LBuffer` seals it, so views report sealed via
/// [`LBufferV::is_sealed`] even when [`seal`](Self::seal) was never called.
pub struct LBuffer<T, const MASK: bool = false> {
    inner: Arc<LBufferInner<T>>,
    /// Validity buffer for a masked column. `Some` only when `MASK` is true.
    /// Holds the bit-packed validity bytes plus the [`LMaskTail`] fill
    /// cursor, and is shared with any [`Bitmask`] created through
    /// [`as_bitmask`](LBuffer::as_bitmask). `None` for an unmasked buffer.
    mask: Option<Arc<LBufferInner<u8>>>,
}

/// Shared cell between the writer and any outstanding views.
struct LBufferInner<T> {
    /// Stable base of the backing allocation. Set at construction, never
    /// changes for the cell's lifetime.
    base: NonNull<T>,
    /// Fixed element capacity.
    capacity: usize,
    /// Published element count. The writer `Release`-stores after each
    /// element write; readers `Acquire`-load to see new elements.
    len: AtomicUsize,
    /// Sealed flag. The writer `Release`-stores `true` once no further
    /// writes will occur; readers `Acquire`-load it. After seal the
    /// published length is fixed and `push` returns `Err`.
    sealed: AtomicBool,
    /// Fill position of a mask buffer. `Some` only on the cell of a validity
    /// buffer; `None` on a value buffer. All the validity bytes - settled and
    /// the one being filled - live in this cell's allocation; this only
    /// records how far the fill has reached.
    mask_tail: Option<LMaskTail>,
}

/// Tracks how far a mask buffer has been filled, and marks it as a bit-packed
/// mask.
///
/// A validity buffer is bit-packed: 8 rows share one byte. The bytes before
/// the one being filled are settled - never written again - and read as plain
/// `u8`. The byte being filled is the last used byte of the allocation, held
/// as an `AtomicU8`. This records its index and how many of its bits are set,
/// so a reader trusts the settled bytes plus those filled bits and nothing
/// beyond.
///
/// One `AtomicU64` packs `(byte_index, filled)`: bits `[0, 8)` are how many
/// bits of the last byte are set (`0..=8` while a byte is completing), the
/// rest are the byte index. The logical bit-length is `byte_index * 8 +
/// filled`. The byte data itself lives in the allocation, not here.
pub(crate) struct LMaskTail {
    cell: AtomicU64,
}

impl LMaskTail {
    const COUNT_MASK: u64 = 0xFF;
    const INDEX_SHIFT: u64 = 8;

    /// Byte 0, nothing filled.
    #[inline]
    fn new() -> Self {
        Self {
            cell: AtomicU64::new(0),
        }
    }

    #[inline]
    fn pack(index: usize, count: usize) -> u64 {
        ((index as u64) << Self::INDEX_SHIFT) | count as u64
    }

    /// Load `(byte_index, filled)`. The `Acquire` pairs with the writer's
    /// `Release` in [`publish`](Self::publish), carrying the last byte's
    /// writes to the reader.
    #[inline]
    pub(crate) fn load(&self) -> (usize, usize) {
        let v = self.cell.load(Ordering::Acquire);
        (
            (v >> Self::INDEX_SHIFT) as usize,
            (v & Self::COUNT_MASK) as usize,
        )
    }

    /// Writer-side load of its own cursor. `Relaxed` is sound: the writer
    /// is the sole mutator (enforced by `&mut self` on push), so
    /// single-location coherence makes it observe its own most recent store.
    #[inline]
    fn load_relaxed(&self) -> (usize, usize) {
        let v = self.cell.load(Ordering::Relaxed);
        (
            (v >> Self::INDEX_SHIFT) as usize,
            (v & Self::COUNT_MASK) as usize,
        )
    }

    /// Publish the cursor with `Release`, after the last byte's writes.
    #[inline]
    fn publish(&self, index: usize, count: usize) {
        self.cell.store(Self::pack(index, count), Ordering::Release);
    }
}

// Safety: the writer writes to `[len, capacity)` through `base`;
// readers access `[0, len)` through `base`. The two regions are
// disjoint by construction because `len` advances only after a write.
// `base` is stable for the cell's lifetime. The atomic `len` and
// `sealed` carry happens-before from the writer to any reader
// that observes them via `Acquire`.
unsafe impl<T: Send + Sync> Send for LBufferInner<T> {}
unsafe impl<T: Send + Sync> Sync for LBufferInner<T> {}

impl<T> LBuffer<T, false> {
    /// Allocate a fixed-capacity buffer through `vec64`'s allocator.
    ///
    /// Allocations >= 2 MiB go through the mmap path with `MADV_HUGEPAGE`
    /// applied automatically; smaller allocations use the 64-byte aligned
    /// heap path.
    ///
    /// # Panics
    /// Panics if `capacity * size_of::<T>()` overflows or the allocator
    /// returns an error.
    pub fn with_capacity(capacity: usize) -> Self {
        let layout = Layout::array::<T>(capacity).expect("LBuffer layout overflow");
        let alloc = Vec64Alloc::default();
        let raw = alloc.allocate(layout).expect("LBuffer allocation failed");
        let base = raw.cast::<T>();
        Self {
            inner: Arc::new(LBufferInner {
                base,
                capacity,
                len: AtomicUsize::new(0),
                sealed: AtomicBool::new(false),
                mask_tail: None,
            }),
            mask: None,
        }
    }

    /// Allocate a fixed-capacity *masked* buffer: a value region of
    /// `capacity` elements plus a bit-packed validity buffer of
    /// `ceil(capacity / 8)` bytes, every bit initialised valid. The handle
    /// writes both halves on each push; [`as_bitmask`](LBuffer::as_bitmask)
    /// hands out the validity as a [`Bitmask`].
    ///
    /// # Panics
    /// Panics if either layout overflows or the allocator returns an error.
    pub fn with_capacity_masked(capacity: usize) -> LBuffer<T, true> {
        let alloc = Vec64Alloc::default();
        let vlayout = Layout::array::<T>(capacity).expect("LBuffer layout overflow");
        let base = alloc
            .allocate(vlayout)
            .expect("LBuffer allocation failed")
            .cast::<T>();
        let inner = Arc::new(LBufferInner {
            base,
            capacity,
            len: AtomicUsize::new(0),
            sealed: AtomicBool::new(false),
            mask_tail: None,
        });
        let n_bytes = (capacity + 7) / 8;
        let mlayout = Layout::array::<u8>(n_bytes).expect("LBuffer mask layout overflow");
        let mbase = alloc
            .allocate(mlayout)
            .expect("LBuffer mask allocation failed")
            .cast::<u8>();
        let mask = Arc::new(LBufferInner {
            base: mbase,
            capacity: n_bytes,
            len: AtomicUsize::new(0),
            sealed: AtomicBool::new(false),
            mask_tail: Some(LMaskTail::new()),
        });
        LBuffer {
            inner,
            mask: Some(mask),
        }
    }
}

impl<T, const MASK: bool> LBuffer<T, MASK> {
    /// Append a value. Returns `Err(value)` when the buffer is sealed or
    /// at capacity.
    ///
    /// Writes the element into the slot at the current length, then
    /// publishes it with a `Release` store on the atomic length. The
    /// single-writer invariant is enforced by `&mut self`.
    pub fn push(&mut self, value: T) -> Result<(), T> {
        // Reject pushes after seal. The same writer thread does seal and
        // push so Relaxed would suffice for self-observation, but Acquire
        // keeps the ordering uniform with reader-side loads of `sealed`.
        if self.inner.sealed.load(Ordering::Acquire) {
            return Err(value);
        }
        // The sole writer reads its own length with Relaxed.
        let n = self.inner.len.load(Ordering::Relaxed);
        if n == self.inner.capacity {
            return Err(value);
        }
        // Write the element first, then publish via Release. Any reader
        // that subsequently observes the new length via Acquire also
        // observes this write.
        unsafe {
            self.inner.base.as_ptr().add(n).write(value);
        }
        // Advance the validity tail with a valid bit before publishing the
        // value length, so the mask always covers any index a reader can
        // ask for. Const-folded out for an unmasked buffer.
        if MASK {
            self.advance_mask(true);
        }
        self.inner.len.store(n + 1, Ordering::Release);
        Ok(())
    }

    /// Append a null. Writes `T::default()` into the value slot - the value
    /// slice still spans it - and clears the corresponding validity bit.
    ///
    /// Returns `Err(())` when the buffer is sealed, at capacity, or not a
    /// masked buffer.
    pub fn push_null(&mut self) -> Result<(), ()>
    where
        T: Default,
    {
        if !MASK || self.inner.sealed.load(Ordering::Acquire) {
            return Err(());
        }
        let n = self.inner.len.load(Ordering::Relaxed);
        if n == self.inner.capacity {
            return Err(());
        }
        unsafe {
            self.inner.base.as_ptr().add(n).write(T::default());
        }
        self.advance_mask(false);
        self.inner.len.store(n + 1, Ordering::Release);
        Ok(())
    }

    /// Append `count` nulls in one batch, publishing the value length with a
    /// single `Release` store. Equivalent to calling [`push_null`](Self::push_null)
    /// `count` times.
    ///
    /// Returns `Err(())` when the buffer is sealed, not masked, or the batch
    /// would exceed capacity. All-or-nothing: a failing call writes nothing.
    pub fn push_nulls(&mut self, count: usize) -> Result<(), ()>
    where
        T: Default,
    {
        if !MASK || self.inner.sealed.load(Ordering::Acquire) {
            return Err(());
        }
        if count == 0 {
            return Ok(());
        }
        let n = self.inner.len.load(Ordering::Relaxed);
        match n.checked_add(count) {
            Some(end) if end <= self.inner.capacity => {}
            _ => return Err(()),
        }
        for i in 0..count {
            unsafe {
                self.inner.base.as_ptr().add(n + i).write(T::default());
            }
            self.advance_mask(false);
        }
        self.inner.len.store(n + count, Ordering::Release);
        Ok(())
    }

    /// Set one validity bit in the last byte and advance the fill position.
    /// `valid` keeps the bit set (a fresh byte starts all-valid); a null
    /// clears it. When the byte fills, the position simply advances - the
    /// byte is already in the allocation and becomes settled.
    ///
    /// Only reached when `MASK` is true, where `self.mask` is `Some`.
    #[inline]
    fn advance_mask(&self, valid: bool) {
        let mask = self
            .mask
            .as_ref()
            .expect("masked buffer carries a validity cell");
        let tail = mask
            .mask_tail
            .as_ref()
            .expect("validity buffer carries a tail");
        let (index, count) = tail.load_relaxed();
        // The byte being filled is the last byte of the allocation, written
        // atomically while it is the one a reader may still be tailing.
        // SAFETY: the value buffer's capacity check bounds the row count, so
        // `index < ceil(capacity / 8)` and `base + index` is within the
        // validity allocation. The writer (`&mut self`) is the sole mutator,
        // and readers touch this byte only atomically while it is being
        // filled, so atomic access here races nothing.
        let byte = unsafe { AtomicU8::from_ptr(mask.base.as_ptr().add(index)) };
        if count == 0 {
            // Fresh byte: all-valid until a null clears a bit.
            byte.store(0xFF, Ordering::Relaxed);
        }
        if !valid {
            byte.fetch_and(!(1u8 << count), Ordering::Relaxed);
        }
        // Publish the new fill position. Release pairs with the reader's
        // Acquire, carrying the byte write above. A full byte advances the
        // index; it is already in place, so no copy and nothing settles
        // separately.
        if count + 1 == 8 {
            tail.publish(index + 1, 0);
        } else {
            tail.publish(index, count + 1);
        }
    }

    /// Current published length.
    pub fn len(&self) -> usize {
        self.inner.len.load(Ordering::Acquire)
    }

    /// Maximum number of elements this buffer can hold.
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    /// `true` once the buffer is at capacity. Further pushes return `Err`.
    pub fn is_full(&self) -> bool {
        self.len() == self.inner.capacity
    }

    /// `true` when no elements have been published yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Create a read view of this buffer. One `Arc` clone, no atomic.
    pub fn view(&self) -> LBufferV<T> {
        LBufferV {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Wrap a fresh read view as a [`Buffer<T>`].
    /// Shortcut to achieve `Buffer::from_lbuffer(self.view())`.
    pub fn as_buffer(&self) -> Buffer<T> {
        Buffer::from_lbuffer(self.view())
    }

    /// Mark this buffer as sealed. After this call, [`push`](Self::push)
    /// returns `Err(value)` and the published length is final. Views
    /// observe the flag via [`LBufferV::is_sealed`].
    ///
    /// Sealing is idempotent and does not consume the `LBuffer`; dropping
    /// it later seals again with the same result.
    pub fn seal(&mut self) {
        self.inner.sealed.store(true, Ordering::Release);
        if MASK {
            if let Some(mask) = self.mask.as_ref() {
                self.finalise_mask_byte(mask);
                mask.sealed.store(true, Ordering::Release);
            }
        }
    }

    /// Clear the padding bits above the fill position in the last byte, so
    /// the sealed mask is contiguous, Arrow-compliant, and the last byte can
    /// join [`Bitmask::as_slice`]. A no-op when the last byte is empty or
    /// full.
    #[inline]
    fn finalise_mask_byte(&self, mask: &Arc<LBufferInner<u8>>) {
        let tail = mask
            .mask_tail
            .as_ref()
            .expect("validity buffer carries a tail");
        let (index, count) = tail.load_relaxed();
        if count > 0 && count < 8 {
            // SAFETY: `index < ceil(capacity / 8)`, so `base + index` is within
            // the validity allocation. Called from seal/drop with `&mut self`,
            // so this is the final write to the byte.
            let byte = unsafe { AtomicU8::from_ptr(mask.base.as_ptr().add(index)) };
            let keep = ((1u16 << count) - 1) as u8;
            byte.fetch_and(keep, Ordering::Relaxed);
        }
    }

    /// `true` once [`seal`](Self::seal) has been called.
    pub fn is_sealed(&self) -> bool {
        self.inner.sealed.load(Ordering::Acquire)
    }

    /// Append every element of `values` in one batch, then publish via a
    /// single `Release` store on the atomic length.
    ///
    /// Equivalent to calling [`push`](Self::push) in a loop, but with one
    /// atomic store at the end instead of one per element. Suits any bulk
    /// write that doesn't need per-element publication, e.g. a block of
    /// string bytes published by a single offset.
    ///
    /// Returns `Err(())` when the buffer is sealed or when the batch
    /// would exceed capacity. All-or-nothing: a failing call writes no
    /// elements and leaves the published length unchanged. The caller
    /// still owns the input slice.
    pub fn push_slice(&mut self, values: &[T]) -> Result<(), ()>
    where
        T: Copy,
    {
        if self.inner.sealed.load(Ordering::Acquire) {
            return Err(());
        }
        let take = values.len();
        if take == 0 {
            return Ok(());
        }
        let n = self.inner.len.load(Ordering::Relaxed);
        if n.saturating_add(take) > self.inner.capacity {
            return Err(());
        }
        // Bulk copy into the spare-capacity region. `copy_nonoverlapping`
        // is sound: `values` is borrowed (so disjoint from our heap), and
        // dst points into our owned allocation past the published region.
        unsafe {
            let dst = self.inner.base.as_ptr().add(n);
            std::ptr::copy_nonoverlapping(values.as_ptr(), dst, take);
        }
        // A bulk push carries no nulls: advance the validity tail by `take`
        // valid bits so a masked buffer's mask keeps pace with the values.
        if MASK {
            for _ in 0..take {
                self.advance_mask(true);
            }
        }
        // One Release after all writes. A reader that observes the new
        // length via Acquire also observes every element written above.
        self.inner.len.store(n + take, Ordering::Release);
        Ok(())
    }

    /// Writer-only mutation of an element that has already been pushed.
    ///
    /// Serves the bit-packed Boolean push path: a fresh zero byte is
    /// pushed when crossing an 8-bit boundary, then the relevant bit is
    /// set in that byte via this method. The sibling publication on the
    /// wrapping array layer (e.g. `BooleanArray`'s bit-count atomic) is
    /// what makes the change visible to readers; mutations through this
    /// method are not themselves a publication.
    ///
    /// # Safety
    /// The caller must ensure all of:
    /// - `idx` is less than the published length, so the slot at `idx` is
    ///   initialised.
    /// - No reader has yet been told to observe `idx` via the sibling
    ///   publication mechanism the caller controls. The simplest way to
    ///   guarantee this is to mutate only the slot the sibling atomic has
    ///   not yet been `Release`d past.
    /// - The single-writer invariant on this `LBuffer` holds (already
    ///   enforced by `&mut self`).
    pub unsafe fn modify_at_unchecked<F: FnOnce(&mut T)>(&mut self, idx: usize, f: F) {
        // SAFETY: by caller invariant, idx is below the published length,
        // so base+idx points at an initialised slot written earlier.
        // `&mut self` precludes concurrent writer mutation. Reader
        // visibility is governed by the caller's sibling publication
        // mechanism.
        unsafe { f(&mut *self.inner.base.as_ptr().add(idx)) }
    }
}

impl<T> LBuffer<T, true> {
    /// The validity as a [`Bitmask`]. The returned mask shares the validity
    /// bytes and the [`LMaskTail`] cursor through an `Arc`, the way
    /// [`as_buffer`](LBuffer::as_buffer) shares the values, so it covers
    /// every published element. Suits an array's `null_mask` field
    /// directly.
    ///
    /// Call it once and store the result; the mask spans elements pushed
    /// afterwards and is final once sealed.
    pub fn as_bitmask(&self) -> Bitmask {
        let mask = self
            .mask
            .as_ref()
            .expect("masked buffer carries a validity cell");
        Bitmask::from_lbuffer(LBufferV {
            inner: Arc::clone(mask),
        })
    }
}

impl<T, const MASK: bool> Drop for LBuffer<T, MASK> {
    fn drop(&mut self) {
        // Seal on drop so views observe the flag even when the writer
        // never called seal(). Idempotent: an already sealed buffer ends
        // up with the same value.
        //
        // Allocation cleanup happens through Arc on `inner` (and `mask`
        // for a masked buffer): when the last reference - writer or any
        // outstanding view - drops, LBufferInner::drop returns the
        // allocation through vec64's deallocate path.
        self.inner.sealed.store(true, Ordering::Release);
        if MASK {
            if let Some(mask) = self.mask.as_ref() {
                self.finalise_mask_byte(mask);
                mask.sealed.store(true, Ordering::Release);
            }
        }
    }
}

impl<T> Drop for LBufferInner<T> {
    fn drop(&mut self) {
        // Drop the elements we wrote. `get_mut` is sound here: Drop only
        // runs when the Arc refcount hits zero, so we have unique access.
        let n = *self.len.get_mut();
        for i in 0..n {
            unsafe { self.base.as_ptr().add(i).drop_in_place() };
        }
        // Return the allocation through vec64's deallocate path.
        let layout = Layout::array::<T>(self.capacity).expect("LBufferInner drop layout");
        let alloc = Vec64Alloc::default();
        unsafe {
            alloc.deallocate(self.base.cast::<u8>(), layout);
        }
    }
}

/// # LBufferV
///
/// Shared read view of an [`LBuffer`].
///
/// Holds the inner cell alive via `Arc` for the view's lifetime; the
/// underlying allocation stays valid even after the `LBuffer` has
/// dropped. Each access reads the latest published length via `Acquire`;
/// indexed access against the cell's stable base is plain pointer
/// arithmetic with no atomics on the access path itself.
///
/// For tight loops, capture the length once with [`len`](Self::len) (or
/// via [`as_slice`](Self::as_slice), which carries it in the slice
/// header) and access the buffer without further atomics. The published
/// length is monotonic, so any observed bound stays valid for the
/// buffer's lifetime.
pub struct LBufferV<T> {
    inner: Arc<LBufferInner<T>>,
}

impl<T> LBufferV<T> {
    /// Currently published length. One `Acquire` load, pairing with the
    /// `Release` store in [`push`](LBuffer::push) and
    /// [`push_slice`](LBuffer::push_slice).
    pub fn len(&self) -> usize {
        self.inner.len.load(Ordering::Acquire)
    }

    /// Published bit count of a mask buffer, or `None` for a value buffer.
    /// One `Acquire` load of the fill cursor.
    pub(crate) fn mask_bits(&self) -> Option<usize> {
        let (settled, filled) = self.inner.mask_tail.as_ref()?.load();
        Some(settled * 8 + filled)
    }

    /// A consistent read of a mask buffer's state for a [`Bitmask`], or `None`
    /// for a value buffer. Returns `(base, settled_bytes, filled_bits, sealed)`:
    /// the allocation base, the count of settled bytes, the bits set in the
    /// byte being filled, and whether the buffer is sealed. The logical
    /// bit-length is `settled_bytes * 8 + filled_bits`.
    pub(crate) fn mask_state(&self) -> Option<(*const u8, usize, usize, bool)> {
        let tail = self.inner.mask_tail.as_ref()?;
        let (settled, filled) = tail.load();
        let base = self.inner.base.as_ptr() as *const u8;
        let sealed = self.inner.sealed.load(Ordering::Acquire);
        Some((base, settled, filled, sealed))
    }

    /// `true` when no elements are currently published.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Capacity of the underlying buffer.
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    /// `true` once the buffer is sealed, whether through [`LBuffer::seal`]
    /// or by the `LBuffer` dropping. Once sealed, no further writes occur
    /// and the published length is final.
    pub fn is_sealed(&self) -> bool {
        self.inner.sealed.load(Ordering::Acquire)
    }

    /// Safe indexed access. One `Acquire` load for the bounds check,
    /// then a plain element access. For tight loops, prefer
    /// [`as_slice`](Self::as_slice) or the cached-bound
    /// [`get_unchecked`](Self::get_unchecked) pattern.
    pub fn get(&self, i: usize) -> Option<&T> {
        let n = self.inner.len.load(Ordering::Acquire);
        if i >= n {
            return None;
        }
        // SAFETY: i < n, where n was observed via Acquire and the
        // writer Released past index i, so the write at base+i is
        // visible. base is stable; [0, len) is never overwritten.
        unsafe { Some(&*self.inner.base.as_ptr().add(i)) }
    }

    /// Unchecked indexed access. Zero atomics, zero bounds checks.
    ///
    /// # Safety
    /// `i` must be strictly less than a length previously observed via
    /// [`len`](Self::len) or another `Acquire`-loaded source on the same
    /// buffer. The published length is monotonic, so any such observed
    /// bound remains valid for the buffer's lifetime.
    pub unsafe fn get_unchecked(&self, i: usize) -> &T {
        // SAFETY: by caller invariant, i < some previously-observed
        // length n, and len is monotonic so i remains valid.
        unsafe { &*self.inner.base.as_ptr().add(i) }
    }

    /// Slice covering the currently published range. One `Acquire`
    /// load captures the length; iteration of the resulting slice is
    /// plain pointer math with no atomics.
    ///
    /// The slice is valid for the view's lifetime: the writer never
    /// writes into `[0, n)`, and the `Arc` keeps the cell's allocation
    /// alive.
    pub fn as_slice(&self) -> &[T] {
        let n = self.inner.len.load(Ordering::Acquire);
        // SAFETY: same reasoning as `get`. The returned slice references
        // initialised, immutable data.
        unsafe { std::slice::from_raw_parts(self.inner.base.as_ptr(), n) }
    }
}

impl<T> Clone for LBufferV<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T> AsRef<[T]> for LBufferV<T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for LBufferV<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LBufferV")
            .field("len", &self.len())
            .field("capacity", &self.capacity())
            .field("sealed", &self.is_sealed())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize as DropCount, Ordering};
    use std::thread;

    #[test]
    fn push_and_view_basic() {
        let mut buf = LBuffer::<u64>::with_capacity(1024);
        for i in 0..100u64 {
            buf.push(i).unwrap();
        }
        let v = buf.view();
        assert_eq!(v.len(), 100);
        for i in 0..100 {
            assert_eq!(v.get(i).copied(), Some(i as u64));
        }
        assert_eq!(v.as_slice(), &(0u64..100).collect::<Vec<_>>()[..]);
    }

    #[test]
    fn view_observes_the_latest_length() {
        let mut buf = LBuffer::<u64>::with_capacity(1024);
        let v = buf.view();
        assert_eq!(v.len(), 0);
        buf.push(1).unwrap();
        assert_eq!(v.len(), 1);
        buf.push(2).unwrap();
        assert_eq!(v.len(), 2);
        // Same view, taken at len 0, sees subsequent writes.
        assert_eq!(v.as_slice(), &[1, 2]);
    }

    #[test]
    fn views_outlive_writer() {
        let v = {
            let mut buf = LBuffer::<u64>::with_capacity(64);
            for i in 0..10u64 {
                buf.push(i).unwrap();
            }
            buf.view()
        };
        // Writer dropped; Arc keeps the cell alive for the view.
        // Drop also sealed it.
        assert_eq!(v.len(), 10);
        assert!(v.is_sealed());
        assert_eq!(v.get(9).copied(), Some(9));
    }

    #[test]
    fn get_unchecked_with_cached_bound() {
        let mut buf = LBuffer::<u64>::with_capacity(1024);
        for i in 0..50u64 {
            buf.push(i).unwrap();
        }
        let v = buf.view();
        // Capture n once, then iterate with no atomic in the loop.
        let n = v.len();
        let sum: u64 = (0..n).map(|i| unsafe { *v.get_unchecked(i) }).sum();
        assert_eq!(sum, (0..50u64).sum::<u64>());
    }

    #[test]
    fn push_returns_err_at_capacity() {
        let mut buf = LBuffer::<u8>::with_capacity(4);
        for i in 0u8..4 {
            buf.push(i).unwrap();
        }
        assert!(matches!(buf.push(99), Err(99)));
        assert_eq!(buf.len(), 4);
    }

    #[test]
    fn many_readers_see_growing_length() {
        let mut buf = LBuffer::<u64>::with_capacity(1 << 16);
        let view = buf.view();
        let stop = Arc::new(AtomicBool::new(false));

        let stop_r = Arc::clone(&stop);
        let reader = thread::spawn(move || {
            let mut last_len = 0usize;
            loop {
                if stop_r.load(Ordering::Acquire) {
                    return last_len;
                }
                let n = view.len();
                assert!(n >= last_len);
                if n > 0 {
                    let want = (n - 1) as u64;
                    assert_eq!(view.get(n - 1).copied(), Some(want));
                }
                last_len = n;
            }
        });

        for i in 0..50_000u64 {
            buf.push(i).unwrap();
        }
        stop.store(true, Ordering::Release);
        let observed = reader.join().unwrap();
        assert!(observed <= 50_000);
    }

    #[test]
    fn element_drops_run_when_inner_drops() {
        #[derive(Debug)]
        struct DropTracker(Arc<DropCount>);
        impl Drop for DropTracker {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let counter = Arc::new(DropCount::new(0));
        {
            let mut buf = LBuffer::<DropTracker>::with_capacity(8);
            for _ in 0..5 {
                buf.push(DropTracker(Arc::clone(&counter))).unwrap();
            }
            // Buffer drops here; no views outstanding, so the inner drops
            // and each pushed element's destructor runs.
        }
        assert_eq!(counter.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn unwritten_capacity_is_not_dropped() {
        // Capacity > written length; only the written elements should
        // have Drop run. The remaining slots are uninitialised and must
        // not be touched at drop time.
        #[derive(Debug)]
        struct DropTracker(Arc<DropCount>);
        impl Drop for DropTracker {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let counter = Arc::new(DropCount::new(0));
        {
            let mut buf = LBuffer::<DropTracker>::with_capacity(64);
            for _ in 0..3 {
                buf.push(DropTracker(Arc::clone(&counter))).unwrap();
            }
        }
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn new_buffer_is_not_sealed() {
        let buf = LBuffer::<u8>::with_capacity(8);
        assert!(!buf.is_sealed());
        let v = buf.view();
        assert!(!v.is_sealed());
    }

    #[test]
    fn seal_blocks_subsequent_pushes() {
        let mut buf = LBuffer::<u8>::with_capacity(8);
        buf.push(1).unwrap();
        buf.push(2).unwrap();
        buf.seal();
        assert!(buf.is_sealed());
        assert!(matches!(buf.push(3), Err(3)));
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn view_observes_seal() {
        let mut buf = LBuffer::<u8>::with_capacity(8);
        let v = buf.view();
        assert!(!v.is_sealed());
        buf.seal();
        assert!(v.is_sealed());
    }

    #[test]
    fn sealed_view_still_reads_data_and_wraps_as_buffer() {
        use crate::Buffer;
        let mut buf = LBuffer::<i32>::with_capacity(8);
        for i in 0..4i32 {
            buf.push(i * 10).unwrap();
        }
        buf.seal();
        let view = buf.view();
        assert!(view.is_sealed());
        assert_eq!(view.len(), 4);
        assert_eq!(view.as_slice(), &[0, 10, 20, 30]);
        let buffer: Buffer<i32> = Buffer::from_lbuffer(view);
        assert_eq!(buffer.len(), 4);
        assert_eq!(buffer.as_slice(), &[0, 10, 20, 30]);
    }

    #[test]
    fn dropping_writer_seals_for_views() {
        let v = {
            let mut buf = LBuffer::<u32>::with_capacity(16);
            for i in 0..5u32 {
                buf.push(i).unwrap();
            }
            let v = buf.view();
            assert!(!v.is_sealed());
            v
        };
        // seal() was never called; the Drop on LBuffer flipped the flag
        // so the view reports sealed.
        assert!(v.is_sealed());
        assert_eq!(v.len(), 5);
        assert_eq!(v.as_slice(), &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn push_slice_appends_in_one_release() {
        let mut buf = LBuffer::<u64>::with_capacity(16);
        let v = buf.view();
        buf.push_slice(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(buf.len(), 5);
        assert_eq!(v.as_slice(), &[1u64, 2, 3, 4, 5]);
        buf.push_slice(&[6, 7]).unwrap();
        assert_eq!(v.as_slice(), &[1u64, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn push_slice_empty_is_a_noop() {
        let mut buf = LBuffer::<u32>::with_capacity(8);
        buf.push(1).unwrap();
        let len_before = buf.len();
        buf.push_slice(&[]).unwrap();
        assert_eq!(buf.len(), len_before);
    }

    #[test]
    fn push_slice_at_capacity_returns_err_and_writes_nothing() {
        let mut buf = LBuffer::<u8>::with_capacity(4);
        buf.push_slice(&[1, 2, 3]).unwrap();
        // Would overflow; must not write any prefix.
        assert!(matches!(buf.push_slice(&[4, 5]), Err(())));
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.view().as_slice(), &[1u8, 2, 3]);
        // A fitting batch still works.
        buf.push_slice(&[4]).unwrap();
        assert_eq!(buf.view().as_slice(), &[1u8, 2, 3, 4]);
    }

    #[test]
    fn push_slice_after_seal_returns_err() {
        let mut buf = LBuffer::<u16>::with_capacity(8);
        buf.push(7).unwrap();
        buf.seal();
        assert!(matches!(buf.push_slice(&[1, 2]), Err(())));
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn modify_at_unchecked_mutates_in_place() {
        let mut buf = LBuffer::<u8>::with_capacity(8);
        buf.push_slice(&[0u8; 4]).unwrap();
        // Caller invariant: the sibling publication (here, just our own
        // local logic) has not yet revealed bit-level state; idx is below
        // the published length.
        unsafe {
            buf.modify_at_unchecked(0, |b| *b |= 0b0000_0001);
            buf.modify_at_unchecked(1, |b| *b |= 0b1000_0000);
        }
        let v = buf.view();
        assert_eq!(v.as_slice(), &[0b0000_0001, 0b1000_0000, 0, 0]);
    }

    #[test]
    fn allocation_freed_when_last_handle_drops() {
        // The allocation is owned by LBufferInner via vec64's allocator.
        // Through Arc, it stays alive while any handle (writer or view)
        // exists. Once all drop, LBufferInner::drop runs and returns the
        // allocation. We verify this via element destructors firing.
        #[derive(Debug)]
        struct DropTracker(Arc<DropCount>);
        impl Drop for DropTracker {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let counter = Arc::new(DropCount::new(0));
        let view = {
            let mut buf = LBuffer::<DropTracker>::with_capacity(8);
            for _ in 0..4 {
                buf.push(DropTracker(Arc::clone(&counter))).unwrap();
            }
            buf.view()
            // Writer drops here; view still holds Arc, inner stays alive.
        };
        assert_eq!(counter.load(Ordering::SeqCst), 0, "view keeps cell alive");
        drop(view);
        // Last handle gone; inner drops, element destructors run.
        assert_eq!(counter.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn masked_push_and_null_read_via_bitmask() {
        let mut buf = LBuffer::<f64>::with_capacity_masked(64);
        buf.push(1.0).unwrap();
        buf.push_null().unwrap();
        buf.push(2.5).unwrap();

        let mask = buf.as_bitmask();
        assert_eq!(mask.len(), 3);
        assert!(mask.get(0));
        assert!(!mask.get(1));
        assert!(mask.get(2));
        assert_eq!(mask.count_zeros(), 1);
        assert!(mask.has_nulls());

        // The null slot holds T::default() in the value buffer.
        assert_eq!(buf.as_buffer().as_slice(), &[1.0, 0.0, 2.5]);
    }

    #[test]
    fn masked_bitmask_tracks_writer() {
        // Bitmask taken before any push still sees subsequent writes.
        let mut buf = LBuffer::<i64>::with_capacity_masked(64);
        let mask = buf.as_bitmask();
        assert_eq!(mask.len(), 0);
        buf.push(1).unwrap();
        assert_eq!(mask.len(), 1);
        assert!(mask.get(0));
        buf.push_null().unwrap();
        assert_eq!(mask.len(), 2);
        assert!(!mask.get(1));
        assert_eq!(mask.count_zeros(), 1);
    }

    #[test]
    fn masked_crossover_settled_and_trailing() {
        // Window 16, nulls at 3, 7, 10. Rows 0-7 fill byte 0 (settled once
        // the 8th bit lands); rows 8-11 stay in the trailing byte.
        let mut buf = LBuffer::<i64>::with_capacity_masked(16);
        for i in 0..12i64 {
            if i == 3 || i == 7 || i == 10 {
                buf.push_null().unwrap();
            } else {
                buf.push(i * 10).unwrap();
            }
        }
        let mask = buf.as_bitmask();
        assert_eq!(mask.len(), 12);
        for i in 0..12 {
            let is_null = i == 3 || i == 7 || i == 10;
            assert_eq!(mask.get(i), !is_null, "bit {i}");
        }
        assert_eq!(mask.count_zeros(), 3);
    }

    #[test]
    fn masked_push_nulls_bulk() {
        let mut buf = LBuffer::<u32>::with_capacity_masked(32);
        buf.push(5).unwrap();
        buf.push_nulls(3).unwrap();
        buf.push(9).unwrap();
        let mask = buf.as_bitmask();
        assert_eq!(mask.len(), 5);
        assert!(mask.get(0));
        assert!(!mask.get(1) && !mask.get(2) && !mask.get(3));
        assert!(mask.get(4));
        assert_eq!(mask.count_zeros(), 3);
    }

    #[test]
    fn masked_all_valid_has_no_nulls() {
        let mut buf = LBuffer::<i64>::with_capacity_masked(16);
        for i in 0..10i64 {
            buf.push(i).unwrap();
        }
        let mask = buf.as_bitmask();
        assert_eq!(mask.len(), 10);
        assert!(!mask.has_nulls());
        assert_eq!(mask.count_zeros(), 0);
        assert!((0..10).all(|i| mask.get(i)));
    }

    #[test]
    fn push_null_on_unmasked_is_err() {
        let mut buf = LBuffer::<i64>::with_capacity(8);
        assert!(buf.push_null().is_err());
        assert!(buf.push_nulls(2).is_err());
    }

    #[test]
    fn masked_integration_with_float_array() {
        use crate::{FloatArray, MaskedArray};

        let mut price = LBuffer::<f64>::with_capacity_masked(64);
        price.push(100.0).unwrap();
        price.push_null().unwrap();
        price.push(101.5).unwrap();

        let arr = FloatArray::<f64> {
            data: price.as_buffer(),
            null_mask: Some(price.as_bitmask()),
        };
        assert_eq!(arr.len(), 3);
        assert!(!arr.is_null(0));
        assert!(arr.is_null(1));
        assert!(!arr.is_null(2));
        assert_eq!(arr.null_count(), 1);
    }

    #[test]
    fn masked_as_slice_is_settled_while_unsealed_then_complete_on_seal() {
        // 10 rows, nulls at 3 and 9. Byte 0 (rows 0-7) settles; byte 1 holds
        // rows 8-9 and is the one still being filled.
        let mut buf = LBuffer::<i64>::with_capacity_masked(64);
        for i in 0..10i64 {
            if i == 3 || i == 9 {
                buf.push_null().unwrap();
            } else {
                buf.push(i).unwrap();
            }
        }

        // Before seal, as_slice is the settled bytes only - byte 0. The byte
        // still being filled (byte 1) is excluded; the bit API still sees it.
        let mask = buf.as_bitmask();
        assert_eq!(mask.len(), 10);
        assert_eq!(mask.as_slice(), &[0b1111_0111]); // byte 0, null at bit 3
        assert!(mask.get(8) && !mask.get(9)); // byte 1 read via the bit path
        assert_eq!(mask.count_zeros(), 2);

        // Sealing finalises the last byte in place (padding cleared) so it
        // joins as_slice - now the complete contiguous mask.
        buf.seal();
        let mask = buf.as_bitmask();
        assert_eq!(mask.len(), 10);
        // byte 1: bit 0 set (row 8), bit 1 clear (row 9), padding zeroed.
        assert_eq!(mask.as_slice(), &[0b1111_0111, 0b0000_0001]);
        assert_eq!(mask.count_zeros(), 2);
    }

    #[test]
    fn masked_byte_and_unchecked_accessors_route_while_unsealed() {
        // The byte accessors and the unchecked/iter/Display paths must reflect
        // an unsealed mask, not the empty stored buffer.
        let mut buf = LBuffer::<i64>::with_capacity_masked(64);
        for i in 0..10i64 {
            if i == 3 || i == 9 {
                buf.push_null().unwrap();
            } else {
                buf.push(i).unwrap();
            }
        }
        let mask = buf.as_bitmask();

        // as_ref / Deref / buffer reflect the settled bytes, not empty.
        assert_eq!(mask.as_ref(), &[0b1111_0111]);
        assert_eq!(&*mask, &[0b1111_0111]);
        assert_eq!(mask.buffer(), &[0b1111_0111]);

        // get_unchecked routes settled bytes and the byte being filled.
        for i in 0..10usize {
            let want = i != 3 && i != 9;
            assert_eq!(unsafe { mask.get_unchecked(i) }, want, "bit {i}");
        }
        assert_eq!(unsafe { mask.get_unchecked_byte(0) }, 0b1111_0111);

        // iter_cleared sees the nulls in both the settled and the filling byte.
        assert_eq!(mask.iter_cleared().collect::<Vec<_>>(), vec![3, 9]);

        // Display reflects the published length, not zero.
        assert!(format!("{mask}").contains("10 bits"));
    }

    #[test]
    fn kernel_validity_merge_over_unsealed_masks() {
        use crate::kernels::bitmask::merge_bitmasks_to_new;

        // Two unsealed masked columns, nulls at different rows.
        let mut a = LBuffer::<f64>::with_capacity_masked(64);
        let mut b = LBuffer::<f64>::with_capacity_masked(64);
        for i in 0..20i64 {
            if i % 5 == 0 {
                a.push_null().unwrap();
            } else {
                a.push(i as f64).unwrap();
            }
            if i % 7 == 0 {
                b.push_null().unwrap();
            } else {
                b.push(i as f64 * 2.0).unwrap();
            }
        }
        let a_mask = a.as_bitmask();
        let b_mask = b.as_bitmask();

        // The exact validity merge the arithmetic/comparison kernels run,
        // reading the unsealed masks via get(), bounded by the published
        // length.
        let merged = merge_bitmasks_to_new(Some(&a_mask), Some(&b_mask), 20).unwrap();
        for i in 0..20usize {
            let want = !(i % 5 == 0 || i % 7 == 0);
            assert_eq!(merged.get(i), want, "row {i}");
        }

        // The value path the SIMD body reads: the published slice, length-bounded.
        let a_vals = a.as_buffer();
        assert_eq!(a_vals.len(), 20);
        assert_eq!(a_vals.as_slice()[6], 6.0);
    }
}
