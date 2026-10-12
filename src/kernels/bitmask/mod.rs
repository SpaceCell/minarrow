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

//! # **Bitmask Kernels Module** - *High-Performance Null-Aware Bitmask Operations*
//!
//! SIMD-optimised bitmask operations for Arrow-compatible nullable array processing with efficient null handling.
//!
//! ## Overview
//!
//! This module provides the foundational bitmask operations that enable null-aware and bit-packed boolean computing
//! throughout the minarrow ecosystem, but can be applied to any bitmasking contenxt.
//! These kernels handle bitwise logical operations, set membership tests, equality comparisons,
//! and population counts on Arrow-format bitmasks with optimal performance characteristics.
//!
//! ## Architecture
//!
//! The bitmask module follows a three-tier architecture:
//! - **Dispatch layer**: Smart runtime selection between SIMD and scalar implementations
//! - **SIMD kernels**: Vectorised implementations using `std::simd` with portable lane counts
//! - **Scalar kernels**: High-performance but non-SIMD fallback implementations for compatibility
//!
//! ## Modules
//! - **`dispatch`**: Runtime dispatch layer selecting SIMD vs scalar implementations based on feature flags
//! - **`simd`**: SIMD-accelerated implementations using vectorised bitwise operations with configurable lane counts
//! - **`std`**: Scalar fallback implementations for word-level operations on 64-bit boundaries
//!
//! ## Core Operations
//!
//! ### **Logical Operations**
//! - **`and_masks`**: Bitwise AND across two bitmasks for intersection operations
//! - **`or_masks`**: Bitwise OR across two bitmasks for union operations  
//! - **`xor_masks`**: Bitwise XOR across two bitmasks for symmetric difference
//! - **`not_mask`**: Bitwise NOT for complement operations
//!
//! ### **Set Membership**
//! - **`in_mask`**: Set inclusion tests - output bits indicate membership of LHS values in RHS set
//! - **`not_in_mask`**: Set exclusion tests - complement of inclusion operations
//!
//! ### **Equality Testing**
//! - **`eq_mask`**: Element-wise equality comparison producing result bitmask
//! - **`ne_mask`**: Element-wise inequality comparison producing result bitmask
//! - **`all_eq`**: Bulk equality test across entire bitmask windows
//! - **`all_ne`**: Bulk inequality test across entire bitmask windows
//!
//! ### **Population Analysis**
//! - **`popcount_mask`**: Fast population count (number of set bits) using SIMD reduction
//! - **`all_true_mask`**: Test if all bits in bitmask are set to 1
//! - **`all_false_mask`**: Test if all bits in bitmask are set to 0
//!
//! ### **Bit Position Iteration**
//! - **`iter_window_bits`**: Enumerate window-relative indices of set or cleared bits
//!
//! ## Arrow Compatibility
//!
//! All operations maintain full compatibility with Apache Arrow's bitmask format:
//! - **LSB bit ordering**: Bit 0 is the least significant bit in each byte
//! - **Byte-packed storage**: 8 bits per byte with proper alignment handling
//! - **Trailing bit management**: Automatic masking of unused bits in final bytes
//! - **64-bit word alignment**: Optimised for modern CPU architectures

pub mod dispatch;
#[cfg(feature = "simd")]
pub mod simd;
#[cfg(not(feature = "simd"))]
pub mod std;

use crate::{Bitmask, BitmaskVT};
use core::mem;

/// Fundamental word type for bitmask operations on 64-bit architectures.
///
/// Defines the basic unit of bitmask storage and manipulation. All bitmask
/// operations are optimised around 64-bit word boundaries for maximum
/// performance on modern CPU architectures.
///
/// # Architecture Alignment
/// This type is chosen to align with:
/// - 64-bit CPU registers: Native register width on x86-64 and AArch64
/// - Cache line efficiency: Optimal memory access patterns
/// - SIMD compatibility: Natural alignment for vectorised operations
pub type Word = u64;

/// Number of bits in a `Word` for bit-level bitmask calculations.
///
/// This constant determines the fundamental granularity of bitmask operations,
/// enabling efficient bit manipulation algorithms that operate on word
/// boundaries.
pub const WORD_BITS: usize = mem::size_of::<Word>() * 8;

/// Helper to compute number of u64 words required for bitmask of `len` bits.
#[inline(always)]
pub fn words_for(len: usize) -> usize {
    (len + WORD_BITS - 1) / WORD_BITS
}

/// Cast &[u8] to &[u64] for word-wise access.
#[inline(always)]
pub unsafe fn mask_bits_as_words(bits: &[u8]) -> *const Word {
    bits.as_ptr() as *const Word
}

/// Cast &mut [u8] to &mut [u64] for word-wise mutation.
#[inline(always)]
pub unsafe fn mask_bits_as_words_mut(bits: &mut [u8]) -> *mut Word {
    bits.as_mut_ptr() as *mut Word
}

/// Create zeroed bitmask of length `len` bits.
#[inline(always)]
pub fn new_mask(len: usize) -> Bitmask {
    Bitmask::new_set_all(len, false)
}

/// Return window into bitmask's bits slice covering offset..offset+len bits.
#[inline(always)]
pub fn bitmask_window_bytes(mask: &Bitmask, offset: usize, len: usize) -> &[u8] {
    let start = offset / 8;
    let end = (offset + len + 7) / 8;
    &mask.bits[start..end]
}

/// Return mutable window into bitmask's bits slice covering offset..offset+len bits.
/// Enables efficient in-place modification of bitmask regions.
#[inline(always)]
pub fn bitmask_window_bytes_mut(mask: &mut Bitmask, offset: usize, len: usize) -> &mut [u8] {
    let start = offset / 8;
    let end = (offset + len + 7) / 8;
    &mut mask.bits.as_mut_slice()[start..end]
}

/// Assemble one 64-bit word of the mask starting at `bit_start`, LSB
/// first, reading past the buffer's end as zeros. An unaligned start
/// combines the two overlapping loads with a shift, so the caller walks
/// any bit window in whole words.
#[inline(always)]
pub fn load_word(mask: &Bitmask, bit_start: usize) -> u64 {
    let byte_start = bit_start / 8;
    let shift = bit_start % 8;
    let mut buf = [0u8; 9];
    let end = (byte_start + 9).min(mask.bits.len());
    if byte_start < end {
        buf[..end - byte_start].copy_from_slice(&mask.bits[byte_start..end]);
    }
    let lo = u64::from_le_bytes(buf[0..8].try_into().unwrap());
    if shift == 0 {
        lo
    } else {
        (lo >> shift) | ((buf[8] as u64) << (64 - shift))
    }
}

/// Zero all slack bits ≥ `bm.len()`.
#[inline(always)]
pub fn clear_trailing_bits(bm: &mut Bitmask) {
    let len = bm.len();
    if len == 0 {
        return;
    }
    let used = len & 7;
    if used != 0 {
        let last = bm.bits.last_mut().unwrap();
        *last &= (1u8 << used) - 1;
    }
}

/// Quick population count of true bits
#[inline(always)]
pub fn popcount_bits(m: BitmaskVT<'_>) -> usize {
    #[cfg(feature = "simd")]
    {
        // Use a default SIMD width if not otherwise specified
        use crate::kernels::bitmask::simd::popcount_mask_simd;
        popcount_mask_simd::<8>(m)
    }
    #[cfg(not(feature = "simd"))]
    {
        use crate::kernels::bitmask::std::popcount_mask;
        popcount_mask(m)
    }
}

/// Merge two optional Bitmasks into a new output mask, computing per-row AND.
/// Returns None if both inputs are None (output is dense).
#[inline]
pub fn merge_bitmasks_to_new(
    lhs: Option<&Bitmask>,
    rhs: Option<&Bitmask>,
    len: usize,
) -> Option<Bitmask> {
    match (lhs, rhs) {
        (None, None) => None,
        (Some(l), None) | (None, Some(l)) => {
            debug_assert!(l.len() >= len, "Bitmask too short in merge");
            let mut out = Bitmask::new_set_all(len, true);
            for i in 0..len {
                out.set(i, l.get(i));
            }
            Some(out)
        }
        (Some(l), Some(r)) => {
            debug_assert!(l.len() >= len, "Left Bitmask too short in merge");
            debug_assert!(r.len() >= len, "Right Bitmask too short in merge");
            let mut out = Bitmask::new_set_all(len, true);
            for i in 0..len {
                out.set(i, l.get(i) && r.get(i));
            }
            Some(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enums::operators::{LogicalOperator, UnaryOperator};
    use crate::{Bitmask, Vec64};

    #[test]
    fn test_words_for() {
        assert_eq!(words_for(0), 0);
        assert_eq!(words_for(1), 1);
        assert_eq!(words_for(63), 1);
        assert_eq!(words_for(64), 1);
        assert_eq!(words_for(65), 2);
        assert_eq!(words_for(128), 2);
        assert_eq!(words_for(129), 3);
    }

    #[test]
    fn test_mask_bits_as_words_and_mut() {
        // Test reading words from bits
        let mut mask = Bitmask::new_set_all(128, false);
        // Write known pattern into mask
        mask.set(0, true);
        mask.set(63, true);
        mask.set(64, true);
        mask.set(127, true);
        let bits = &mask.bits;
        unsafe {
            let words = mask_bits_as_words(bits);
            assert_eq!(*words, 1u64 | (1u64 << 63));
            assert_eq!(*words.add(1), 1u64 | (1u64 << 63));
        }

        // Test writing via mask_bits_as_words_mut
        let mut mask = Bitmask::new_set_all(128, false);
        let bits = &mut mask.bits;
        unsafe {
            let words_mut = mask_bits_as_words_mut(bits);
            *words_mut = 0xDEADBEEFDEADBEEF;
            *words_mut.add(1) = 0xCAFEBABECAFEBABE;
        }
        assert_eq!(mask.bits[0], 0xEF);
        assert_eq!(mask.bits[7], 0xDE);
        assert_eq!(mask.bits[8], 0xBE);
        assert_eq!(mask.bits[15], 0xCA);
    }

    #[test]
    fn test_bitmask_window_bytes() {
        let mut mask = Bitmask::new_set_all(24, false);
        mask.set(0, true);
        mask.set(7, true);
        mask.set(8, true);
        mask.set(15, true);
        mask.set(16, true);
        mask.set(23, true);
        // Window: bytes 1..3 should cover bits 8..24
        let bytes = bitmask_window_bytes(&mask, 8, 16);
        assert_eq!(bytes.len(), 2); // 16 bits = 2 bytes
        assert_eq!(bytes[0], 0b10000001); // bits 8 and 15 set
        assert_eq!(bytes[1], 0b10000001); // bits 16 and 23 set
    }

    #[test]
    fn test_bitmask_window_bytes_mut() {
        let mut mask = Bitmask::new_set_all(16, false);
        {
            let window = bitmask_window_bytes_mut(&mut mask, 0, 16);
            window[0] = 0xAA;
            window[1] = 0x55;
        }
        assert_eq!(mask.bits[0], 0xAA);
        assert_eq!(mask.bits[1], 0x55);
    }

    #[test]
    fn test_clear_trailing_bits() {
        let mut mask = Bitmask::new_set_all(10, true);
        // The last byte should have only the low 2 bits set after clearing trailing bits
        clear_trailing_bits(&mut mask);
        // 10 bits => 2 bytes, last byte should have only bits 0 and 1 set (0b00000011 == 0x03)
        assert_eq!(mask.bits[1], 0x03);
        // The remaining bits should still be set
        for i in 0..10 {
            assert!(mask.get(i));
        }
        // Any bits beyond 10 should be cleared
        for i in 10..16 {
            assert!(!mask.get(i));
        }
    }

    /// The `_into` output-window path matches the allocating dispatch bit-for-bit
    /// across lengths that exercise full SIMD strides, scalar word tails and the
    /// final bit tail.
    #[test]
    fn binop_into_matches_allocating() {
        use crate::kernels::bitmask::dispatch::{
            and_masks, and_masks_into, or_masks, or_masks_into, xor_masks, xor_masks_into,
        };

        for &len in &[1usize, 7, 8, 63, 64, 65, 127, 128, 200, 1000] {
            let mut a = Bitmask::new_set_all(len, false);
            let mut b = Bitmask::new_set_all(len, false);
            for i in 0..len {
                if i % 3 == 0 {
                    a.set(i, true);
                }
                if i % 5 == 0 {
                    b.set(i, true);
                }
            }

            let cases: [(Bitmask, fn(&mut Bitmask, usize, BitmaskVT, BitmaskVT)); 3] = [
                (and_masks((&a, 0, len), (&b, 0, len)), and_masks_into),
                (or_masks((&a, 0, len), (&b, 0, len)), or_masks_into),
                (xor_masks((&a, 0, len), (&b, 0, len)), xor_masks_into),
            ];
            for (reference, into_fn) in cases {
                let mut out = Bitmask::new_set_all(len, false);
                into_fn(&mut out, 0, (&a, 0, len), (&b, 0, len));
                for i in 0..len {
                    assert_eq!(out.get(i), reference.get(i), "len {len} bit {i}");
                }
            }
        }
    }

    /// `not_mask_into` matches the allocating `not_mask` across the same lengths.
    #[test]
    fn unop_into_matches_allocating() {
        use crate::kernels::bitmask::dispatch::{not_mask, not_mask_into};

        for &len in &[1usize, 7, 64, 65, 127, 200] {
            let mut src = Bitmask::new_set_all(len, false);
            for i in 0..len {
                if i % 2 == 0 {
                    src.set(i, true);
                }
            }
            let reference = not_mask((&src, 0, len));
            let mut out = Bitmask::new_set_all(len, false);
            not_mask_into(&mut out, 0, (&src, 0, len));
            for i in 0..len {
                assert_eq!(out.get(i), reference.get(i), "len {len} bit {i}");
            }
        }
    }

    /// A windowed `_into` write touches only its own bit range. Bits before the
    /// window and beyond its end are left untouched, so adjacent windows of a
    /// shared output buffer stay independent.
    #[test]
    fn binop_into_writes_only_its_window() {
        use crate::kernels::bitmask::dispatch::and_masks_into;

        // Window length is not a multiple of 64, so the write ends in the bit tail.
        let win_len = 70usize;
        let out_off = 128usize; // byte-aligned
        let win_end = out_off + win_len; // 198

        let mut a = Bitmask::new_set_all(win_len, false);
        let mut b = Bitmask::new_set_all(win_len, false);
        for i in 0..win_len {
            if i % 3 == 0 {
                a.set(i, true);
            }
            if i % 4 == 0 {
                b.set(i, true);
            }
        }

        // Pre-set the whole output buffer so any stray write flips a bit we check.
        let mut out = Bitmask::new_set_all(256, true);
        and_masks_into(&mut out, out_off, (&a, 0, win_len), (&b, 0, win_len));

        for i in 0..out_off {
            assert!(out.get(i), "bit {i} before the window was disturbed");
        }
        for i in 0..win_len {
            let expect = a.get(i) & b.get(i);
            assert_eq!(out.get(out_off + i), expect, "window bit {i}");
        }
        for i in win_end..256 {
            assert!(out.get(i), "bit {i} past the window was disturbed");
        }
    }

    /// Window offsets for the offset sweeps. They cover bit positions within a
    /// byte and within a word, whole-word offsets, and offsets that start
    /// several words into the mask.
    pub(super) const SWEEP_OFFSETS: [usize; 13] = [0, 1, 2, 3, 7, 8, 13, 63, 64, 65, 67, 512, 515];

    /// Window lengths for the offset sweeps at a SIMD width of `lanes` words.
    /// The two longest lengths span one and three whole vectors plus a partial
    /// word.
    pub(super) fn sweep_lengths(lanes: usize) -> [usize; 10] {
        // Bits in one vector of `lanes` words.
        let v = 64 * lanes;
        [1, 57, 61, 63, 64, 65, 130, 200, v + 37, 3 * v + 37]
    }

    /// Sweep offsets for windows of `len` bits, each paired with the length of
    /// the mask that contains the window. Every offset appears twice, once in a
    /// mask that ends with the window and once in a mask that extends one whole
    /// vector and a partial word past it.
    pub(super) fn sweep_windows(len: usize, lanes: usize) -> impl Iterator<Item = (usize, usize)> {
        SWEEP_OFFSETS
            .into_iter()
            .flat_map(move |off| [(off, off + len), (off, off + len + 64 * lanes + 13)])
    }

    /// Mask of `n` bits with bit `i` set when `i % a == 0 || i % b == 1`. The
    /// sweeps use values of `a` and `b` for which the pattern does not repeat
    /// every 8 bits.
    pub(super) fn sweep_mask(n: usize, a: usize, b: usize) -> Bitmask {
        let mut m = Bitmask::new_set_all(n, false);
        for i in 0..n {
            if i % a == 0 || i % b == 1 {
                m.set(i, true);
            }
        }
        m
    }

    /// Checks a binary `_into` kernel against a bit-by-bit `Bitmask::get`
    /// reference, over every pair of sweep offsets, the sweep lengths and
    /// output offsets 0, 8 and 64. Output bits outside the window keep their
    /// prior values.
    pub(super) fn sweep_binop_into(
        lanes: usize,
        kernel: impl Fn(&mut Bitmask, usize, BitmaskVT<'_>, BitmaskVT<'_>, LogicalOperator),
    ) {
        let ops = [
            LogicalOperator::And,
            LogicalOperator::Or,
            LogicalOperator::Xor,
        ];
        for len in sweep_lengths(lanes) {
            for out_off in [0usize, 8, 64] {
                let out_n = out_off + len + 70;
                let before = sweep_mask(out_n, 4, 9);
                for (lhs_off, lhs_n) in sweep_windows(len, lanes) {
                    let lhs = sweep_mask(lhs_n, 3, 7);
                    for (rhs_off, rhs_n) in sweep_windows(len, lanes) {
                        let rhs = sweep_mask(rhs_n, 5, 11);
                        for op in ops {
                            let mut out = before.clone();
                            let lhs_window = (&lhs, lhs_off, len);
                            let rhs_window = (&rhs, rhs_off, len);
                            kernel(&mut out, out_off, lhs_window, rhs_window, op);
                            for i in 0..out_n {
                                let expected = if i < out_off || i >= out_off + len {
                                    before.get(i)
                                } else {
                                    let a = lhs.get(lhs_off + i - out_off);
                                    let b = rhs.get(rhs_off + i - out_off);
                                    match op {
                                        LogicalOperator::And => a & b,
                                        LogicalOperator::Or => a | b,
                                        LogicalOperator::Xor => a ^ b,
                                    }
                                };
                                assert_eq!(
                                    out.get(i),
                                    expected,
                                    "{op:?}: lhs ({lhs_off}, {len}) of {lhs_n} bits, \
                                     rhs ({rhs_off}, {len}) of {rhs_n} bits, \
                                     out_off {out_off}, out bit {i}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Checks a unary `_into` kernel against a bit-by-bit `Bitmask::get`
    /// reference, over the sweep offsets, the sweep lengths and output offsets
    /// 0, 8 and 64. Output bits outside the window keep their prior values.
    pub(super) fn sweep_unop_into(
        lanes: usize,
        kernel: impl Fn(&mut Bitmask, usize, BitmaskVT<'_>, UnaryOperator),
    ) {
        for len in sweep_lengths(lanes) {
            for out_off in [0usize, 8, 64] {
                let out_n = out_off + len + 70;
                let before = sweep_mask(out_n, 4, 9);
                for (off, n) in sweep_windows(len, lanes) {
                    let src = sweep_mask(n, 3, 7);
                    let mut out = before.clone();
                    kernel(&mut out, out_off, (&src, off, len), UnaryOperator::Not);
                    for i in 0..out_n {
                        let expected = if i < out_off || i >= out_off + len {
                            before.get(i)
                        } else {
                            !src.get(off + i - out_off)
                        };
                        assert_eq!(
                            out.get(i),
                            expected,
                            "Not: src ({off}, {len}) of {n} bits, out_off {out_off}, out bit {i}"
                        );
                    }
                }
            }
        }
    }

    /// Checks a popcount kernel against a bit-by-bit `Bitmask::get` count over
    /// the sweep offsets and lengths. Rows past the end of the window are
    /// excluded from the count.
    pub(super) fn sweep_popcount(lanes: usize, kernel: impl Fn(BitmaskVT<'_>) -> usize) {
        for len in sweep_lengths(lanes) {
            for (off, n) in sweep_windows(len, lanes) {
                let m = sweep_mask(n, 3, 7);
                let expected = (off..off + len).filter(|&i| m.get(i)).count();
                assert_eq!(
                    kernel((&m, off, len)),
                    expected,
                    "window ({off}, {len}) of {n} bits"
                );
            }
        }
    }

    /// Mask lengths for the all-bits checks at a SIMD width of `lanes` words.
    /// Every length up to 256 is covered, together with the lengths around the
    /// end of one and two whole vectors.
    pub(super) fn all_bits_lengths(lanes: usize) -> impl Iterator<Item = usize> {
        // Bits in one vector of `lanes` words.
        let v = 64 * lanes;
        (0..=256)
            .chain(v - 8..=v + 64)
            .chain(2 * v + 56..=2 * v + 64)
    }

    /// Bit positions flipped in the all-bits checks of a mask of `len` bits.
    /// Every position is flipped in masks of up to 256 bits. Longer masks have
    /// their first word, final two words, and the first and last bit of every
    /// other word flipped.
    pub(super) fn all_bits_flips(len: usize) -> impl Iterator<Item = usize> {
        (0..len).filter(move |&p| {
            len <= 256 || p < 64 || p + 128 >= len || p % 64 == 0 || p % 64 == 63
        })
    }

    /// Checks `all_true_mask` and `all_false_mask` kernels against a
    /// bit-by-bit `Bitmask::get` reference over the all-bits lengths.
    ///
    /// Each length is checked with masks that hold one value throughout, the
    /// same masks with one bit flipped, masks whose unused final-byte bits hold
    /// the opposite value, and windows copied from longer masks at offsets
    /// inside a byte.
    pub(super) fn sweep_all_bits(
        lanes: usize,
        all_true: impl Fn(&Bitmask) -> bool,
        all_false: impl Fn(&Bitmask) -> bool,
    ) {
        // Reference over the `len` bits of `src` that start at bit `off`.
        let check = |m: &Bitmask, src: &Bitmask, off: usize, case: &str| {
            let len = m.len();
            let expected_true = (off..off + len).all(|i| src.get(i));
            let expected_false = (off..off + len).all(|i| !src.get(i));
            assert_eq!(all_true(m), expected_true, "all_true: {case}, len {len}");
            assert_eq!(all_false(m), expected_false, "all_false: {case}, len {len}");
        };
        for len in all_bits_lengths(lanes) {
            for value in [true, false] {
                let uniform = Bitmask::new_set_all(len, value);
                check(&uniform, &uniform, 0, &format!("all {value}"));

                for p in all_bits_flips(len) {
                    let mut m = uniform.clone();
                    m.set(p, !value);
                    check(&m, &m, 0, &format!("all {value} except bit {p}"));
                }

                // Unused bits of the final byte hold the opposite value.
                // `Bitmask::new` keeps them as written.
                let n_bytes = len.div_ceil(8);
                let mut bytes = Vec64::<u8>::with_capacity(n_bytes);
                bytes.resize(n_bytes, if value { 0xFF } else { 0x00 });
                if len % 8 != 0 {
                    bytes[n_bytes - 1] ^= !((1u8 << (len % 8)) - 1);
                }
                let slack = Bitmask::new(bytes, len);
                check(&slack, &slack, 0, &format!("all {value}, unused bits {}", !value));

                // Windows at offsets inside a byte, copied from masks that hold
                // the opposite value outside the window.
                for off in [1usize, 3, 5, 11, 19, 63, 67] {
                    let mut src = Bitmask::new_set_all(off + len + 13, !value);
                    for i in off..off + len {
                        src.set(i, value);
                    }
                    let window = src.slice_clone(off, len);
                    check(&window, &src, off, &format!("all {value}, window at {off}"));
                }
            }
        }
    }

    /// Checks an `all_ne` kernel against a bit-by-bit `Bitmask::get`
    /// reference. The kernel returns true when at least one corresponding bit
    /// pair of the two windows differs. The windows start at offset 0 of masks
    /// that end with the window.
    pub(super) fn check_all_ne(
        lanes: usize,
        all_ne: impl Fn(BitmaskVT<'_>, BitmaskVT<'_>) -> bool,
    ) {
        // Only the second bit pair is equal.
        let a = Bitmask::from_bools(&[true, false, true, true]);
        let b = Bitmask::from_bools(&[false, false, false, false]);
        assert!(all_ne((&a, 0, 4), (&b, 0, 4)), "one equal bit pair");

        for len in all_bits_lengths(lanes) {
            let a = sweep_mask(len, 3, 7);
            let mut complement = a.clone();
            for i in 0..len {
                complement.set(i, !a.get(i));
            }
            let mut cases = vec![(a.clone(), "equal windows".to_string())];
            cases.push((complement, "every bit pair differs".to_string()));
            for p in all_bits_flips(len) {
                let mut b = a.clone();
                b.set(p, !a.get(p));
                cases.push((b, format!("bit {p} differs")));
            }
            for (b, case) in cases {
                let expected = (0..len).any(|i| a.get(i) != b.get(i));
                assert_eq!(all_ne((&a, 0, len), (&b, 0, len)), expected, "{case}, len {len}");
            }
        }
    }

    /// Window offsets for the `in_mask` sweeps. They cover bit positions within
    /// a byte, byte offsets that are not whole words, and whole-word offsets.
    const IN_MASK_OFFSETS: [usize; 10] = [0, 1, 3, 5, 8, 11, 19, 63, 64, 67];

    /// Rhs masks for the `in_mask` sweeps at window length `len`, each paired
    /// with its window offset and a case description.
    ///
    /// Each window holds one value throughout, or that value with its first or
    /// last bit flipped, and the bits outside the window hold the opposite
    /// value. Every offset appears twice, once in a mask that ends with the
    /// window and once in a mask that extends one whole vector and a partial
    /// word past it.
    fn in_mask_rhs_windows(len: usize, lanes: usize) -> Vec<(Bitmask, usize, String)> {
        let mut windows = Vec::new();
        for value in [true, false] {
            for flip in [None, Some(0), Some(len - 1)] {
                for off in IN_MASK_OFFSETS {
                    for n in [off + len, off + len + 64 * lanes + 13] {
                        let mut rhs = Bitmask::new_set_all(n, !value);
                        for i in off..off + len {
                            rhs.set(i, value);
                        }
                        if let Some(p) = flip {
                            rhs.set(off + p, !value);
                        }
                        let case =
                            format!("rhs ({off}, {len}) of {n} bits, {value} with flip {flip:?}");
                        windows.push((rhs, off, case));
                    }
                }
            }
        }
        windows
    }

    /// Checks `in_mask` and `not_in_mask` kernels against a bit-by-bit
    /// `Bitmask::get` reference, over every pairing of lhs window offsets with
    /// the rhs windows of `in_mask_rhs_windows`, at the sweep lengths.
    pub(super) fn sweep_in_mask(
        lanes: usize,
        in_mask: impl Fn(BitmaskVT<'_>, BitmaskVT<'_>) -> Bitmask,
        not_in_mask: impl Fn(BitmaskVT<'_>, BitmaskVT<'_>) -> Bitmask,
    ) {
        for len in sweep_lengths(lanes).into_iter().chain([7, 8, 9, 40]) {
            let lhs_masks = IN_MASK_OFFSETS.map(|off| sweep_mask(off + len + 70, 3, 7));
            for (rhs, rhs_off, case) in in_mask_rhs_windows(len, lanes) {
                let has_true = (rhs_off..rhs_off + len).any(|i| rhs.get(i));
                let has_false = (rhs_off..rhs_off + len).any(|i| !rhs.get(i));
                for (lhs_off, lhs) in IN_MASK_OFFSETS.into_iter().zip(&lhs_masks) {
                    let lhs_window = (lhs, lhs_off, len);
                    let rhs_window = (&rhs, rhs_off, len);
                    let ins = in_mask(lhs_window, rhs_window);
                    let not_ins = not_in_mask(lhs_window, rhs_window);
                    assert_eq!(ins.len(), len, "in_mask output length, {case}");
                    assert_eq!(not_ins.len(), len, "not_in_mask output length, {case}");
                    // Lhs rows are members when the rhs window holds their value.
                    let member = |i: usize| {
                        if lhs.get(lhs_off + i) {
                            has_true
                        } else {
                            has_false
                        }
                    };
                    let wrong_in = (0..len).filter(|&i| ins.get(i) != member(i)).count();
                    let wrong_not_in = (0..len).filter(|&i| not_ins.get(i) == member(i)).count();
                    assert_eq!(
                        wrong_in,
                        0,
                        "in_mask: lhs ({lhs_off}, {len}), {case}: {wrong_in} of {len} rows wrong"
                    );
                    assert_eq!(
                        wrong_not_in,
                        0,
                        "not_in_mask: lhs ({lhs_off}, {len}), {case}: \
                         {wrong_not_in} of {len} rows wrong"
                    );
                }
            }
        }
    }

    /// The dispatched binary `_into` kernel reads windows at any input offset.
    #[test]
    fn binop_into_window_offsets() {
        use crate::kernels::bitmask::dispatch::{W8, bitmask_binop_into};
        sweep_binop_into(W8, bitmask_binop_into);
    }

    /// The dispatched unary `_into` kernel reads windows at any input offset.
    #[test]
    fn unop_into_window_offsets() {
        use crate::kernels::bitmask::dispatch::{W8, bitmask_unop_into};
        sweep_unop_into(W8, bitmask_unop_into);
    }

    /// The dispatched popcount counts the rows of windows at any offset.
    #[test]
    fn popcount_window_offsets() {
        use crate::kernels::bitmask::dispatch::{W8, popcount_mask};
        sweep_popcount(W8, popcount_mask);
    }

    /// The dispatched all-bits checks read every bit of masks of any length.
    #[test]
    fn all_true_false_mask_lengths() {
        use crate::kernels::bitmask::dispatch::{W8, all_false_mask, all_true_mask};
        sweep_all_bits(W8, all_true_mask, all_false_mask);
    }

    /// The dispatched `all_ne` returns true when any bit pair differs.
    #[test]
    fn all_ne_any_bit_differs() {
        use crate::kernels::bitmask::dispatch::{W8, all_ne};
        check_all_ne(W8, all_ne);
    }

    /// The dispatched `in_mask` and `not_in_mask` read only the rhs window, at
    /// any offset.
    #[test]
    fn in_mask_window_offsets() {
        use crate::kernels::bitmask::dispatch::{W8, in_mask, not_in_mask};
        sweep_in_mask(W8, in_mask, not_in_mask);
    }
}
