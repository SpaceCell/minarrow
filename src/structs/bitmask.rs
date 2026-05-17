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

//! # **Bitmask Module** - *Fast Bitpacked Byte Bitmask*
//!
//! Arrow-compatible, packed validity/boolean bitmask with 64-byte alignment.
//!
//! ## Purpose
//! - Validity (null) masks for all array types (1 = valid, 0 = null).
//! - Backing storage for `BooleanArray`.
//!
//! ## Behaviour
//! - LSB corresponds to the first logical element.
//! - Zero-copy windowing via [`BitmaskV`] (`view`, `slice`).
//! - Trailing padding bits are always masked off for Arrow spec compliance.
//!
//! ## Interop
//! - Memory layout matches Arrow, and is safe to pass over the Arrow C Data Interface.

use std::fmt::{Debug, Display, Formatter, Result as FmtResult};
use std::ops::{BitAnd, BitOr, Deref, DerefMut, Index, Not};

use crate::enums::shape_dim::ShapeDim;
use crate::traits::concatenate::Concatenate;
use crate::traits::shape::Shape;
use crate::{BitmaskV, Buffer, Length, Offset};
use vec64::Vec64;

/// TODO: Move bitmask kernels here

/// # Bitmask
///
/// 64-byte–aligned packed bitmask.
///
/// ### Description
/// - Used for `BooleanArray` data and as the validity/null mask for all datatypes.
/// - Arrow-compatible: LSB = first element, 1 = set/valid, 0 = cleared/null.
/// - Automatically enforced alignment enables efficient bitwise filtering on SIMD targets.
///
/// # Example
/// ```rust
/// use minarrow::Bitmask;
///
/// // Start with 10 cleared bits, flip 2 on
/// let mut m = Bitmask::new_set_all(10, false);
/// m.set(3, true);
/// m.set(7, true);
/// assert!(m.get(3) && m.get(7));
///
/// // Create a zero-copy window over [2..8)
/// let v = m.view(2, 6);
/// assert_eq!(v.len(), 6);
/// assert_eq!(v.get(1), true); // corresponds to original bit 3
/// ```
#[repr(C, align(64))]
#[derive(Clone, PartialEq, Default)]
pub struct Bitmask {
    pub bits: Buffer<u8>,
    pub len: usize,
}

impl Bitmask {
    /// Constructs a new, empty array.
    #[inline]
    pub fn new(data: impl Into<Buffer<u8>>, len: usize) -> Self {
        let data: Buffer<u8> = data.into();
        Self { bits: data, len }
    }

    /// Ensures all unused bits above self.len are zeroed, per Arrow spec.
    #[inline]
    pub fn mask_trailing_bits(&mut self) {
        if self.len == 0 || (self.len & 7) == 0 {
            return;
        }
        // Index the last byte that holds logical bits, not the last physical
        // byte of the backing buffer. The two coincide for tightly-sized
        // buffers but diverge when the bitmask was built with `Bitmask::new`
        // over a larger Buffer<u8>.
        let last = (self.len + 7) / 8 - 1;
        let mask = (1u8 << (self.len & 7)) - 1;
        self.bits[last] &= mask;
    }

    /// Create new mask, length = `len`, all bits set if `set` else cleared.
    #[inline]
    pub fn new_set_all(len: usize, set: bool) -> Self {
        let n_bytes = (len + 7) / 8;
        let mut data = Vec64::with_capacity(n_bytes);
        let fill = if set { 0xFF } else { 0 };
        data.resize(n_bytes, fill);
        let mut mask = Self {
            bits: data.into(),
            len,
        };
        mask.mask_trailing_bits();
        mask
    }

    /// Create with reserved capacity (bits), all bits cleared.
    #[inline]
    pub fn with_capacity(bits: usize) -> Self {
        let n_bytes = (bits + 7) / 8;
        let mut data = Vec64::with_capacity(n_bytes);
        data.resize(n_bytes, 0);
        let mut mask = Self {
            bits: data.into(),
            len: bits,
        };
        mask.mask_trailing_bits();
        mask
    }

    /// Create a Bitmask from a raw pointer to a bit-packed buffer.
    ///
    /// - `ptr`: Pointer to a packed `[u8]` (as per Arrow and C FFI).
    /// - `len`: Number of logical bits.
    ///
    /// # Safety
    /// - Caller must ensure `ptr` points to at least `(len + 7) / 8` bytes.
    /// - The contents must be valid for the entire bitmask.
    pub unsafe fn from_raw_slice(ptr: *const u8, len: usize) -> Self {
        if ptr.is_null() || len == 0 {
            return Bitmask::default();
        }
        let n_bytes = (len + 7) / 8;
        let slice = unsafe { std::slice::from_raw_parts(ptr, n_bytes) };
        let mut buf = Vec64::with_capacity(n_bytes);
        buf.extend_from_slice(slice);
        let mut out = Bitmask {
            bits: buf.into(),
            len,
        };
        out.mask_trailing_bits();
        out
    }

    /// Returns a ref slice to the raw u8 bytes
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8] {
        self.as_ref()
    }

    /// Creates a bitmask from an existing byte buffer
    pub fn from_bytes(bytes: impl AsRef<[u8]>, len: usize) -> Self {
        let mut mask = Bitmask::with_capacity(len);
        let bytes = bytes.as_ref();
        for i in 0..len {
            let valid = (bytes[i >> 3] >> (i & 7)) & 1 != 0;
            mask.set(i, valid);
        }
        mask
    }

    /// Returns the logical length of the bitmask
    ///
    /// *Excludes padding*
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return logical number of bits (slots).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns true if all bits set (i.e. valid for null-mask).
    #[inline]
    pub fn all_set(&self) -> bool {
        self.count_ones() == self.len
    }

    /// Returns true if all bits cleared.
    #[inline]
    pub fn all_unset(&self) -> bool {
        self.count_ones() == 0
    }

    /// Returns true if any bit is cleared.
    #[inline]
    pub fn has_cleared(&self) -> bool {
        !self.all_set()
    }

    /// Creates an owned copy of the bitmask.
    /// Always creates a fresh owned copy, even if already owned.
    #[inline]
    pub fn to_owned_copy(&self) -> Self {
        let owned_bits = self.bits.to_owned_copy();
        Bitmask {
            bits: owned_bits,
            len: self.len,
        }
    }

    /// Returns bit *idx*.  
    /// - If `idx ≥ self.len` but still inside the physical buffer, returns `false`.  
    /// Panics only when `idx` exceeds the physical capacity.
    #[inline]
    pub fn get(&self, idx: usize) -> bool {
        let cap_bits = self.bits.len() * 8;
        assert!(
            idx < cap_bits,
            "Bitmask::get out of physical bounds (idx={idx}, cap={cap_bits})"
        );
        if idx >= self.len {
            return false;
        }
        // SAFETY: idx / 8 is within the slice.
        let byte = unsafe { self.bits.get_unchecked(idx >> 3) };
        (byte >> (idx & 7)) & 1 != 0
    }

    /// Set or clear bit at index `i`.
    #[inline]
    pub fn set(&mut self, i: usize, value: bool) {
        self.ensure_capacity(i + 1);
        let byte = &mut self.bits[i >> 3];
        let bit = 1u8 << (i & 7);
        if value {
            *byte |= bit;
        } else {
            *byte &= !bit;
        }
        self.mask_trailing_bits();
    }

    /// Set or clear the bit at index `i` without any bounds or trailing‐bit checks.
    ///
    /// # Safety
    /// - The caller must ensure that `i` is within the existing capacity (i.e. `i < self.data.len() * 8`).
    /// - The caller is responsible for maintaining any invariants around trailing bits.
    #[inline(always)]
    pub unsafe fn set_unchecked(&mut self, i: usize, value: bool) {
        // locate the byte
        let byte = unsafe { self.bits.get_unchecked_mut(i >> 3) };
        // compute the mask for this bit
        let bit = 1u8 << (i & 7);
        if value {
            *byte |= bit;
        } else {
            *byte &= !bit;
        }
    }

    /// Returns the `w`-th 64-bit word without bounds checks.
    ///
    /// # Safety
    /// Caller guarantees `w < self.bits.len() / 8` *and* that the word is inside
    /// the logical range (`w * 64 < self.len()`).
    #[inline(always)]
    pub unsafe fn word_unchecked(&self, w: usize) -> u64 {
        unsafe { *self.bits.as_ptr().cast::<u64>().add(w) }
    }

    /// Writes `word` into the `w`-th 64-bit slot without bounds checks.
    ///
    /// # Safety
    /// Same pre-conditions as `word_unchecked`.
    #[inline(always)]
    pub unsafe fn set_word_unchecked(&mut self, w: usize, word: u64) {
        unsafe { *self.bits.as_mut_ptr().cast::<u64>().add(w) = word };
    }

    /// Ensure at least `bits` bits are allocated.
    #[inline]
    pub fn ensure_capacity(&mut self, bits: usize) {
        let needed = (bits + 7) / 8;
        if self.bits.len() < needed {
            self.bits.resize(needed, 0);
        }
        if bits > self.len {
            self.len = bits;
            self.mask_trailing_bits();
        }
    }

    /// Set a chunk of bits from a u64 value at offset `start`, `n_bits` bits.
    #[inline]
    pub fn set_bits_chunk(&mut self, start: usize, value: u64, n_bits: usize) {
        assert!(n_bits <= 64, "set_bits_chunk: n_bits > 64");
        for i in 0..n_bits {
            let bit_val = ((value >> i) & 1) != 0;
            self.set(start + i, bit_val);
        }
        self.mask_trailing_bits();
    }

    /// Bulk-append `n` bits, all set or cleared.
    #[inline]
    pub fn push_bits(&mut self, value: bool, n: usize) {
        self.resize(self.len + n, value);
    }

    /// Returns true if all bits are set (all valid).
    #[inline]
    pub fn all_true(&self) -> bool {
        if self.len == 0 {
            return true;
        }
        let full_bytes = self.len / 8;
        let last_bits = self.len & 7;
        if !self.bits[..full_bytes].iter().all(|&b| b == 0xFF) {
            return false;
        }
        if last_bits != 0 {
            let mask = (1u8 << last_bits) - 1;
            self.bits[full_bytes] & mask == mask
        } else {
            true
        }
    }

    /// Returns true if all bits are cleared (all null).
    #[inline]
    pub fn all_false(&self) -> bool {
        if self.len == 0 {
            return true;
        }
        let full_bytes = self.len / 8;
        let last_bits = self.len & 7;
        if !self.bits[..full_bytes].iter().all(|&b| b == 0) {
            return false;
        }
        if last_bits != 0 {
            let mask = (1u8 << last_bits) - 1;
            self.bits[full_bytes] & mask == 0
        } else {
            true
        }
    }

    /// Construct from a slice of bools (true = set).
    #[inline]
    pub fn from_bools(bits: &[bool]) -> Self {
        let len = bits.len();
        let n_bytes = (len + 7) / 8;
        let mut data = Vec64::with_capacity(n_bytes);
        data.resize(n_bytes, 0);
        for (i, &b) in bits.iter().enumerate() {
            if b {
                data[i >> 3] |= 1u8 << (i & 7);
            }
        }
        let mut mask = Self {
            bits: data.into(),
            len,
        };
        mask.mask_trailing_bits();
        mask
    }

    /// Returns true if there are any cleared bits (any nulls).
    #[inline]
    pub fn has_nulls(&self) -> bool {
        !self.all_true()
    }

    /// Returns the pointer to the start of the mask.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.bits.as_ptr()
    }

    /// Set bit at index to true
    #[inline]
    pub fn set_true(&mut self, idx: usize) {
        self.set(idx, true)
    }

    /// Set bit at index to false
    #[inline]
    pub fn set_false(&mut self, idx: usize) {
        self.set(idx, false)
    }

    /// Count number of set (1) bits.
    #[inline]
    pub fn count_ones(&self) -> usize {
        let full_bytes = self.len / 8;
        let mut count = self.bits[..full_bytes]
            .iter()
            .map(|&b| b.count_ones() as usize)
            .sum::<usize>();
        let rem = self.len & 7;
        if rem != 0 {
            let mask = (1u8 << rem) - 1;
            count += (self.bits[full_bytes] & mask).count_ones() as usize;
        }
        count
    }

    /// Count number of cleared (0) bits.
    #[inline]
    pub fn count_zeros(&self) -> usize {
        self.len - self.count_ones()
    }

    /// Returns the number of bits set to false.
    #[inline]
    pub fn null_count(&self) -> usize {
        self.count_zeros()
    }

    /// Resizes mask to new_len. New bits set or cleared per `set`.
    pub fn resize(&mut self, new_len: usize, set: bool) {
        let new_bytes = (new_len + 7) / 8;
        let fill = if set { 0xFF } else { 0 };
        self.bits.resize(new_bytes, fill);
        self.len = new_len;
        self.mask_trailing_bits();
    }

    /// Splits the bitmask at the given bit position, returning a new Bitmask
    /// containing bits [at..len) and leaving self with bits [0..at).
    ///
    /// For byte-aligned splits (at % 8 == 0), this uses an efficient buffer split.
    /// For non-byte-aligned splits, this creates a new buffer and repositions bits.
    ///
    /// # Panics
    /// Panics if called on a Shared buffer or if `at > self.len`.
    pub fn split_off(&mut self, at: usize) -> Self {
        assert!(at <= self.len, "split_off index out of bounds");

        if at == self.len {
            // Splitting at the end - return empty mask
            return Bitmask::new_set_all(0, false);
        }

        let start_byte = at / 8;
        let bit_offset = at % 8;
        let new_len = self.len - at;

        if bit_offset == 0 {
            // Byte-aligned - clean split using buffer split_off
            let after_bits = self.bits.split_off(start_byte);
            self.len = at;
            self.mask_trailing_bits();

            let mut after = Bitmask {
                bits: after_bits,
                len: new_len,
            };
            after.mask_trailing_bits();
            return after;
        }

        // Non-byte-aligned - need to shift bits into a new buffer
        let after_bytes_needed = (new_len + 7) / 8;
        let mut after_buf = Vec64::with_capacity(after_bytes_needed);
        after_buf.resize(after_bytes_needed, 0);

        // Copy and reposition bits from [at..len) to [0..new_len) in new buffer
        let original_bytes = self.bits.as_slice();
        for i in 0..new_len {
            let src_bit = at + i;
            let src_byte = src_bit / 8;
            let src_offset = src_bit % 8;

            if src_byte < original_bytes.len() {
                let bit_value = (original_bytes[src_byte] >> src_offset) & 1;

                let dst_byte = i / 8;
                let dst_offset = i % 8;
                after_buf[dst_byte] |= bit_value << dst_offset;
            }
        }

        // Truncate self to `at` bits
        let self_bytes_needed = (at + 7) / 8;
        self.bits.resize(self_bytes_needed, 0);
        self.len = at;
        self.mask_trailing_bits();

        let mut after = Bitmask {
            bits: after_buf.into(),
            len: new_len,
        };
        after.mask_trailing_bits();
        after
    }

    /// Extends the Bitmask with bits from an iterator of bools.
    /// The new bits are appended after the current length.
    #[inline]
    pub fn extend<I: IntoIterator<Item = bool>>(&mut self, iter: I) {
        for bit in iter {
            self.set(self.len, bit);
            self.len += 1;
        }
        self.mask_trailing_bits();
    }

    /// Appends all bits from another Bitmask.
    pub fn extend_from_bitmask(&mut self, other: &Bitmask) {
        let old_len = self.len();
        self.resize(old_len + other.len(), true);
        for i in 0..other.len() {
            // Safety: falls within established lengths.
            // Provided another thread isn't mutating this at the same
            // time it's ok.
            unsafe { self.set_unchecked(old_len + i, other.get_unchecked(i)) };
        }
    }

    /// Appends bits `[offset..offset+len)` from another bitmask into self.
    /// Byte-aligned sources copy whole bytes directly. Unaligned sources
    /// shift bytes to align before copying.
    pub fn extend_from_bitmask_range(&mut self, other: &Bitmask, offset: usize, len: usize) {
        if len == 0 { return; }
        let src_bytes = other.bits.as_slice();
        if offset & 7 == 0 {
            // Source is byte-aligned - pass the bytes starting at the offset
            self.extend_from_slice(&src_bytes[offset >> 3..], len);
        } else {
            // Unaligned source - shift bytes to produce an aligned slice
            let src_byte_start = offset >> 3;
            let bit_shift = (offset & 7) as u32;
            let n_src_bytes = ((len + 7) >> 3) + 1; // +1 for the shifted tail
            let end = (src_byte_start + n_src_bytes).min(src_bytes.len());
            let mut shifted = Vec::with_capacity(n_src_bytes);
            for i in src_byte_start..end {
                let lo = src_bytes[i] >> bit_shift;
                let hi = if i + 1 < src_bytes.len() {
                    src_bytes[i + 1] << (8 - bit_shift)
                } else {
                    0
                };
                shifted.push(lo | hi);
            }
            self.extend_from_slice(&shifted, len);
        }
    }

    /// Extends the bitmask by appending `len` bits from a bit-packed `[u8]` slice.
    ///
    /// - `src`: The source byte slice (bit-packed; LSB = first bit).
    /// - `len`: Number of bits to append from `src`.
    ///
    /// The bit-ordering and null semantics match Arrow conventions.
    pub fn extend_from_slice(&mut self, src: &[u8], len: usize) {
        let start = self.len;
        let total = start + len;
        self.resize(total, false);

        let dst = self.bits.as_mut_slice();

        // Fast path - both self and src are byte-aligned
        if (start & 7) == 0 {
            // dst is byte-aligned; copy whole bytes first, then tail
            let dst_byte = start >> 3;
            let n_full_bytes = len >> 3;
            for i in 0..n_full_bytes {
                dst[dst_byte + i] = src[i];
            }
            let tail = len & 7;
            if tail != 0 {
                let mask = (1u8 << tail) - 1;
                dst[dst_byte + n_full_bytes] &= !mask;
                dst[dst_byte + n_full_bytes] |= src[n_full_bytes] & mask;
            }
            self.mask_trailing_bits();
            return;
        }

        // General case: bit-level append
        for i in 0..len {
            let bit = (src[i >> 3] >> (i & 7)) & 1;
            if bit != 0 {
                let j = start + i;
                dst[j >> 3] |= 1 << (j & 7);
            } else {
                let j = start + i;
                dst[j >> 3] &= !(1 << (j & 7));
            }
        }
        self.mask_trailing_bits();
    }

    /// Returns the entire underlying byte slice representing the packed bitmask.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        self.bits.as_slice()
    }

    // TODO: Optimise with word version

    /// Slices by copying the data
    #[inline]
    pub fn slice_clone(&self, offset: usize, len: usize) -> Self {
        assert!(
            offset + len <= self.len,
            "Bitmask::slice_clone out of bounds"
        );
        let mut out = Bitmask::new_set_all(len, false);
        let src = self.bits.as_slice();
        let dst = out.bits.as_mut_slice();

        for i in 0..len {
            let src_idx = offset + i;
            let src_byte = src_idx / 8;
            let src_bit = src_idx % 8;

            if (src[src_byte] & (1 << src_bit)) != 0 {
                let dst_byte = i / 8;
                let dst_bit = i % 8;
                dst[dst_byte] |= 1 << dst_bit;
            }
        }
        out.mask_trailing_bits();
        out
    }

    /// Slice view (no copy): returns (&[u8], bit_offset, len).
    #[inline]
    pub fn slice(&self, offset: usize, len: usize) -> (&[u8], usize, usize) {
        assert!(offset + len <= self.len, "Bitmask::slice out of bounds");
        let start_byte = offset / 8;
        let end_bit = offset + len;
        let end_byte = (end_bit + 7) / 8;
        (&self.bits[start_byte..end_byte], offset % 8, len)
    }

    /// BCreates a `Bitmask` as a `BitmaskView` over `[offset, offset + len)`.
    /// Provides a zero-copy logical window over the parent bitmask.
    ///
    /// `Offset` and `Length` are semantic `usize` aliases.
    #[inline(always)]
    pub fn view(&self, offset: Offset, len: Length) -> BitmaskV {
        BitmaskV::new(self.clone(), offset, len)
    }

    /// Combine two optional null masks. None means "no nulls".
    ///
    /// Returns None when both inputs are None, otherwise merges them
    /// using bitwise OR via `union()`.
    pub fn union_opt(a: Option<&Bitmask>, b: Option<&Bitmask>) -> Option<Bitmask> {
        match (a, b) {
            (None, None) => None,
            (Some(m), None) | (None, Some(m)) => Some(m.clone()),
            (Some(a), Some(b)) => Some(a.union(b)),
        }
    }

    /// Logical 'or' (elementwise) with another mask.
    #[inline]
    pub fn union(&self, other: &Self) -> Self {
        assert_eq!(self.len, other.len, "Bitmask::union length mismatch");
        let mut out = self.clone();
        for (a, b) in out.bits.iter_mut().zip(other.bits.iter()) {
            *a |= *b;
        }
        out.mask_trailing_bits();
        out
    }

    /// Logical 'and' (elementwise) with another mask.
    #[inline]
    pub fn intersect(&self, other: &Self) -> Self {
        assert_eq!(self.len, other.len, "Bitmask::intersect length mismatch");
        let mut out = self.clone();
        for (a, b) in out.bits.iter_mut().zip(other.bits.iter()) {
            *a &= *b;
        }
        out.mask_trailing_bits();
        out
    }

    /// Invert all bits (set <-> clear).
    #[inline]
    pub fn invert(&self) -> Self {
        let mut out = self.clone();
        for b in out.bits.iter_mut() {
            *b = !*b;
        }
        out.mask_trailing_bits();
        out
    }

    /// Iterator over all indices with set bits (valid).
    pub fn iter_set(&self) -> impl Iterator<Item = usize> + '_ {
        let n = self.len;
        self.bits.iter().enumerate().flat_map(move |(byte_i, &b)| {
            let base = byte_i * 8;
            (0..8).filter_map(move |bit| {
                let idx = base + bit;
                if idx < n && ((b >> bit) & 1) != 0 {
                    Some(idx)
                } else {
                    None
                }
            })
        })
    }

    /// Iterator over all indices with cleared bits (nulls).
    pub fn iter_cleared(&self) -> impl Iterator<Item = usize> + '_ {
        let n = self.len;
        self.bits.iter().enumerate().flat_map(move |(byte_i, &b)| {
            let base = byte_i * 8;
            (0..8).filter_map(move |bit| {
                let idx = base + bit;
                if idx < n && ((b >> bit) & 1) == 0 {
                    Some(idx)
                } else {
                    None
                }
            })
        })
    }

    /// Set all bits to set/cleared.
    #[inline]
    pub fn fill(&mut self, value: bool) {
        let fill = if value { 0xFF } else { 0 };
        for b in &mut self.bits {
            *b = fill;
        }
        self.mask_trailing_bits();
    }

    /// Returns raw (bitpacked) buffer slice
    #[inline]
    pub fn buffer(&self) -> &[u8] {
        &self.bits
    }

    /// Fast bit access with no bounds checking. Caller guarantees idx < self.len.
    //#[cfg(feature = "unchecked")]
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, idx: usize) -> bool {
        let byte = unsafe { self.bits.get_unchecked(idx >> 3) };
        ((*byte) >> (idx & 7)) & 1 != 0
    }

    /// Returns the byte at `byte_idx` with no bounds checking.
    #[inline(always)]
    pub unsafe fn get_unchecked_byte(&self, byte_idx: usize) -> u8 {
        *unsafe { self.bits.get_unchecked(byte_idx) }
    }
}

#[cfg(feature = "parallel_proc")]
mod parallel {
    use rayon::prelude::*;

    use super::Bitmask;

    impl Bitmask {
        /// Parallel iterator over every bit in `[0, len)`.
        #[inline]
        pub fn par_iter(&self) -> impl ParallelIterator<Item = bool> + '_ {
            (0..self.len)
                .into_par_iter()
                .map(move |i| unsafe { self.get_unchecked(i) })
        }

        /// Parallel iterator over the half-open window `[start, end)`.
        #[inline]
        pub fn par_iter_range(
            &self,
            start: usize,
            end: usize,
        ) -> impl ParallelIterator<Item = bool> + '_ {
            debug_assert!(start <= end && end <= self.len);
            (start..end)
                .into_par_iter()
                .map(move |i| unsafe { self.get_unchecked(i) })
        }
    }
}

impl Index<usize> for Bitmask {
    type Output = bool;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        if self.get(index) {
            &true
        } else {
            &false
        }
    }
}

impl Debug for Bitmask {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("Bitmask")
            .field("len", &self.len)
            .field("ones", &self.count_ones())
            .field("zeros", &self.count_zeros())
            .field("buffer", &self.bits)
            .finish()
    }
}

impl BitAnd for &Bitmask {
    type Output = Bitmask;
    #[inline]
    fn bitand(self, rhs: Self) -> Bitmask {
        self.intersect(rhs)
    }
}
impl BitOr for &Bitmask {
    type Output = Bitmask;
    #[inline]
    fn bitor(self, rhs: Self) -> Bitmask {
        self.union(rhs)
    }
}

impl Not for &Bitmask {
    type Output = Bitmask;
    #[inline]
    fn not(self) -> Bitmask {
        self.invert()
    }
}

impl Not for Bitmask {
    type Output = Bitmask;
    #[inline]
    fn not(self) -> Bitmask {
        self.invert()
    }
}

impl AsRef<[u8]> for Bitmask {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.bits.as_ref()
    }
}

impl AsMut<[u8]> for Bitmask {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        self.bits.as_mut()
    }
}

impl Deref for Bitmask {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.bits.as_ref()
    }
}

impl DerefMut for Bitmask {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.bits.as_mut()
    }
}

impl Display for Bitmask {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let ones = self.count_ones();
        let zeros = self.count_zeros();
        writeln!(
            f,
            "Bitmask [{} bits] (ones: {}, zeros: {})",
            self.len, ones, zeros
        )?;

        const MAX_PREVIEW: usize = 64;
        write!(f, "[")?;

        for i in 0..usize::min(self.len, MAX_PREVIEW) {
            if i > 0 {
                write!(f, " ")?;
            }
            write!(
                f,
                "{}",
                if unsafe { self.get_unchecked(i) } {
                    '1'
                } else {
                    '0'
                }
            )?;
        }

        if self.len > MAX_PREVIEW {
            write!(f, " … ({} total)", self.len)?;
        }

        write!(f, "]")
    }
}

impl Shape for Bitmask {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank1(self.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitmask_new_set_get() {
        let mut m = Bitmask::new_set_all(10, false);
        for i in 0..10 {
            assert!(!m.get(i));
        }
        m.set(3, true);
        assert!(m.get(3));
        m.set(3, false);
        assert!(!m.get(3));
    }

    #[test]
    fn test_ensure_capacity_and_resize() {
        let mut m = Bitmask::new_set_all(1, false);
        m.ensure_capacity(20);
        assert!(m.len >= 20);
        m.set(15, true);
        assert!(m.get(15));
        m.resize(100, false);
        assert!(m.len == 100);
    }

    #[test]
    fn test_count_and_all() {
        let mut m = Bitmask::new_set_all(16, true);
        assert_eq!(m.count_ones(), 16);
        assert!(m.all_set());
        m.set(0, false);
        assert_eq!(m.count_zeros(), 1);
        assert!(!m.all_set());
        assert!(!m.all_unset());
    }

    #[test]
    fn test_invert_union_and_intersect() {
        let mut a = Bitmask::new_set_all(8, false);
        let mut b = Bitmask::new_set_all(8, false);
        a.set(1, true);
        a.set(3, true);
        b.set(3, true);
        b.set(4, true);
        let u = &a | &b;
        assert!(u.get(1) && u.get(3) && u.get(4));
        let i = &a & &b;
        assert!(!i.get(1) && i.get(3));
        let inv = !&a;
        assert!(!inv.get(3) && inv.get(2));
    }

    #[test]
    fn test_set_bits_chunk_and_push_bits() {
        let mut m = Bitmask::new_set_all(16, false);
        m.set_bits_chunk(0, 0b10101, 5);
        assert!(m.get(0));
        assert!(!m.get(1));
        assert!(m.get(2));
        assert!(!m.get(3));
        assert!(m.get(4));
        m.push_bits(true, 3);
        for i in 16..19 {
            assert!(m.get(i));
        }
    }

    #[test]
    fn test_slice_clone_and_view() {
        let mut m = Bitmask::new_set_all(10, false);
        m.set(2, true);
        m.set(5, true);
        let sub = m.slice_clone(2, 4);
        assert_eq!(sub.capacity(), 4);
        assert!(sub.get(0) && sub.get(3));
        let (buf, offset, len) = m.slice(2, 4);
        let bit = (buf[0] >> offset) & 1 != 0;
        assert_eq!(bit, true);
        assert_eq!(len, 4);
    }

    #[test]
    fn test_iter_set_and_iter_cleared() {
        let mut m = Bitmask::new_set_all(12, false);
        m.set(2, true);
        m.set(5, true);
        m.set(10, true);
        let set: Vec<_> = m.iter_set().collect();
        assert_eq!(set, vec![2, 5, 10]);
        let cleared: Vec<_> = m.iter_cleared().collect();
        assert!(cleared.contains(&0) && cleared.contains(&11) && !cleared.contains(&2));
    }

    #[test]
    fn test_extend_from_slice() {
        // Bitmask starting with 5 bits: 10101 (LSB-first in one byte)
        let mut mask = Bitmask::new_set_all(5, false);
        mask.set(0, true);
        mask.set(2, true);
        mask.set(4, true);

        // Next 7 bits to append: 1100110 (packed in one byte, 0b01100110)
        let src_bytes = [0b01100110u8];
        mask.extend_from_slice(&src_bytes, 7);

        // Combined bits should be: 1 0 1 0 1 | 0 1 1 0 0 1 1 0 (LSB first)
        // i:    0 1 2 3 4 | 5 6 7 8 9 10 11
        let expected = [
            true, false, true, false, true, // original 5
            false, true, true, false, false, true, true, // appended 7
        ];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(mask.get(i), exp, "Mismatch at bit {}", i);
        }

        // Appending a byte-aligned chunk (8 bits)
        let mut m2 = Bitmask::new_set_all(8, true);
        let add_bytes = [0b10101100u8]; // bits: 0 0 1 1 0 1 0 1
        m2.extend_from_slice(&add_bytes, 8);

        let expected2 = [
            true, true, true, true, true, true, true, true, // original 8
            false, false, true, true, false, true, false, true, // appended 8
        ];
        for (i, &exp) in expected2.iter().enumerate() {
            assert_eq!(m2.get(i), exp, "Mismatch at bit {}", i);
        }

        // Appending empty
        let mut m3 = Bitmask::new_set_all(3, false);
        let empty_bytes = [0u8];
        m3.extend_from_slice(&empty_bytes, 0);
        assert_eq!(m3.len(), 3);
    }

    #[test]
    fn test_union_opt_none_none() {
        assert!(Bitmask::union_opt(None, None).is_none());
    }

    #[test]
    fn test_union_opt_some_none() {
        let m = Bitmask::from_bools(&[true, false, true]);
        let result = Bitmask::union_opt(Some(&m), None).unwrap();
        assert_eq!(result, m);
    }

    #[test]
    fn test_union_opt_none_some() {
        let m = Bitmask::from_bools(&[false, true, false]);
        let result = Bitmask::union_opt(None, Some(&m)).unwrap();
        assert_eq!(result, m);
    }

    #[test]
    fn test_union_opt_some_some() {
        let a = Bitmask::from_bools(&[true, false, false, true]);
        let b = Bitmask::from_bools(&[false, true, false, true]);
        let result = Bitmask::union_opt(Some(&a), Some(&b)).unwrap();
        assert!(result.get(0)); // true | false
        assert!(result.get(1)); // false | true
        assert!(!result.get(2)); // false | false
        assert!(result.get(3)); // true | true
    }

    #[test]
    fn test_concatenate() {
        let mut m1 = Bitmask::new_set_all(5, false);
        m1.set(0, true);
        m1.set(2, true);
        m1.set(4, true);

        let mut m2 = Bitmask::new_set_all(4, false);
        m2.set(1, true);
        m2.set(3, true);

        let result = m1.concat(m2).unwrap();
        assert_eq!(result.len(), 9);
        // First 5 bits from m1
        assert!(result.get(0));
        assert!(!result.get(1));
        assert!(result.get(2));
        assert!(!result.get(3));
        assert!(result.get(4));
        // Next 4 bits from m2
        assert!(!result.get(5));
        assert!(result.get(6));
        assert!(!result.get(7));
        assert!(result.get(8));
    }
}

// Concatenate Trait Implementation

impl Concatenate for Bitmask {
    fn concat(
        mut self,
        other: Self,
    ) -> core::result::Result<Self, crate::enums::error::MinarrowError> {
        // Consume other and extend self with its bits
        self.extend_from_bitmask(&other);
        Ok(self)
    }
}
