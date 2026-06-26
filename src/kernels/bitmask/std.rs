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

//! # **Bitmask Scalar Kernels** - *Word-Level Bitmask Operations*
//!
//! Scalar implementations of bitmask operations optimised for word-level processing without SIMD dependencies.
//! These kernels provide universal compatibility with great performance through careful
//! bit manipulation and efficient memory access patterns.
//!
//! ## Overview
//!
//! This module contains the scalar fallback implementations for all bitmask operations, for
//! high performance on any target architecture. The implementations focus on 64-bit word operations
//! to maximise throughput whilst maintaining simplicity and debuggability.
//!
//! ## Architecture Principles
//!
//! - **Word-level operations**: Process 64 bits simultaneously using native CPU instructions  
//! - **Minimal branching**: Reduce pipeline stalls through branchless bit manipulation
//! - **Cache-friendly access**: Sequential memory access patterns for optimal cache utilisation
//! - **Trailing bit handling**: Proper masking of unused bits in partial words
//!
//! ## Arrow Compatibility
//!
//! All implementations maintain Arrow format compatibility:
//! - **LSB bit ordering**: Bit 0 is least significant in each byte
//! - **Proper alignment**: Operations respect byte and word boundaries
//! - **Trailing bit masking**: Unused bits in final bytes are properly cleared
//! - **Window support**: Efficient processing of bitmask slices at arbitrary offsets
//!
//! ## Error Handling
//!
//! The scalar implementations include safety checks:
//! - Debug assertions for length mismatches and invalid offsets
//! - Panic conditions for alignment requirements (eq_mask, all_eq_mask)
//! - Proper bounds checking for window operations
//! - Graceful handling of zero-length inputs
//!
use crate::{Bitmask, BitmaskVT};

use crate::{
    enums::operators::{LogicalOperator, UnaryOperator},
    kernels::bitmask::{
        bitmask_window_bytes, bitmask_window_bytes_mut, clear_trailing_bits,
    },
};

/// Performs bitwise binary operations (AND/OR/XOR) over two bitmask slices using word-level processing.
///
/// Core scalar implementation for logical operations between bitmask windows. Processes data in 64-bit
/// words for optimal performance, with automatic trailing bit masking to ensure Arrow compatibility.
///
/// # Parameters
/// - `lhs`: Left-hand side bitmask window as `(mask, offset, length)` tuple
/// - `rhs`: Right-hand side bitmask window as `(mask, offset, length)` tuple
/// - `op`: Logical operation to perform (AND, OR, XOR)
///
/// # Returns
/// A new `Bitmask` containing the element-wise results with proper trailing bit handling.
#[inline(always)]
pub fn bitmask_binop_std(lhs: BitmaskVT<'_>, rhs: BitmaskVT<'_>, op: LogicalOperator) -> Bitmask {
    let len = lhs.2;
    let mut out = Bitmask::new_set_all(len, false);
    bitmask_binop_std_into(&mut out, 0, lhs, rhs, op);
    out
}

/// Scalar binary logical AND, OR or XOR, writing the result into `out` at bit
/// offset `out_off` rather than allocating a fresh mask.
///
/// Full 64-bit words lying entirely within `[0, len)` run word-wise. The final
/// `len % 64` bits run one bit at a time, so the write touches only valid
/// positions and never the trailing bits past `len`. With byte-aligned offsets
/// the words never straddle a window edge, so adjacent windows of a shared
/// output buffer stay independent.
///
/// All offsets are byte-aligned i.e. a multiple of 8 bits.
#[inline(always)]
pub fn bitmask_binop_std_into(
    out: &mut Bitmask,
    out_off: usize,
    lhs: BitmaskVT<'_>,
    rhs: BitmaskVT<'_>,
    op: LogicalOperator,
) {
    let (lhs_mask, lhs_off, len) = lhs;
    let (rhs_mask, rhs_off, _) = rhs;
    if len == 0 {
        return;
    }
    debug_assert_eq!(out_off % 8, 0, "bitmask_binop_std_into: out_off must be byte-aligned");
    debug_assert_eq!(lhs_off % 8, 0, "bitmask_binop_std_into: lhs offset must be byte-aligned");
    debug_assert_eq!(rhs_off % 8, 0, "bitmask_binop_std_into: rhs offset must be byte-aligned");
    let full_words = len / 64;
    let tail_bits = len % 64;
    {
        let lhs_bytes = bitmask_window_bytes(lhs_mask, lhs_off, len);
        let rhs_bytes = bitmask_window_bytes(rhs_mask, rhs_off, len);
        let out_bytes = bitmask_window_bytes_mut(out, out_off, len);
        unsafe {
            let lp = lhs_bytes.as_ptr().cast::<u64>();
            let rp = rhs_bytes.as_ptr().cast::<u64>();
            let dp = out_bytes.as_mut_ptr().cast::<u64>();
            for k in 0..full_words {
                *dp.add(k) = match op {
                    LogicalOperator::And => *lp.add(k) & *rp.add(k),
                    LogicalOperator::Or => *lp.add(k) | *rp.add(k),
                    LogicalOperator::Xor => *lp.add(k) ^ *rp.add(k),
                };
            }
        }
    }
    // Final partial word written one bit at a time.
    let base = full_words * 64;
    unsafe {
        for i in 0..tail_bits {
            let a = lhs_mask.get_unchecked(lhs_off + base + i);
            let b = rhs_mask.get_unchecked(rhs_off + base + i);
            let v = match op {
                LogicalOperator::And => a & b,
                LogicalOperator::Or => a | b,
                LogicalOperator::Xor => a ^ b,
            };
            out.set_unchecked(out_off + base + i, v);
        }
    }
}

/// Bitwise unary operation (`NOT`) over a bitmask slice.
#[inline(always)]
pub fn bitmask_unop_std(src: BitmaskVT<'_>, op: UnaryOperator) -> Bitmask {
    let len = src.2;
    let mut out = Bitmask::new_set_all(len, false);
    bitmask_unop_std_into(&mut out, 0, src, op);
    out
}

/// Scalar unary `NOT`, writing the result into `out` at bit offset `out_off`
/// rather than allocating a fresh mask.
///
/// Full 64-bit words run word-wise. The final `len % 64` bits run one bit at a
/// time, so the write never touches the trailing bits past `len`.
///
/// All offsets are byte-aligned i.e. a multiple of 8 bits.
#[inline(always)]
pub fn bitmask_unop_std_into(
    out: &mut Bitmask,
    out_off: usize,
    src: BitmaskVT<'_>,
    op: UnaryOperator,
) {
    let (mask, offset, len) = src;
    if len == 0 {
        return;
    }
    debug_assert_eq!(out_off % 8, 0, "bitmask_unop_std_into: out_off must be byte-aligned");
    debug_assert_eq!(offset % 8, 0, "bitmask_unop_std_into: src offset must be byte-aligned");
    let full_words = len / 64;
    let tail_bits = len % 64;
    {
        let src_bytes = bitmask_window_bytes(mask, offset, len);
        let out_bytes = bitmask_window_bytes_mut(out, out_off, len);
        unsafe {
            let sp = src_bytes.as_ptr().cast::<u64>();
            let dp = out_bytes.as_mut_ptr().cast::<u64>();
            for k in 0..full_words {
                *dp.add(k) = match op {
                    UnaryOperator::Not => !*sp.add(k),
                    _ => unreachable!(), // Positive Negative invalid for bools
                };
            }
        }
    }
    // Final partial word written one bit at a time.
    let base = full_words * 64;
    unsafe {
        for i in 0..tail_bits {
            let v = match op {
                UnaryOperator::Not => !mask.get_unchecked(offset + base + i),
                _ => unreachable!(),
            };
            out.set_unchecked(out_off + base + i, v);
        }
    }
}

// Entry points

/// Element-wise bitwise `AND` on bitmask slices.
#[inline]
pub fn and_masks(lhs: BitmaskVT<'_>, rhs: BitmaskVT<'_>) -> Bitmask {
    bitmask_binop_std(lhs, rhs, LogicalOperator::And)
}

/// Element-wise bitwise `OR` on bitmask slices.
#[inline]
pub fn or_masks(lhs: BitmaskVT<'_>, rhs: BitmaskVT<'_>) -> Bitmask {
    bitmask_binop_std(lhs, rhs, LogicalOperator::Or)
}

/// Element-wise bitwise `XOR` on bitmask slices.
#[inline]
pub fn xor_masks(lhs: BitmaskVT<'_>, rhs: BitmaskVT<'_>) -> Bitmask {
    bitmask_binop_std(lhs, rhs, LogicalOperator::Xor)
}

/// Bitwise `NOT` over a bitmask slice.
#[inline]
pub fn not_mask(src: BitmaskVT<'_>) -> Bitmask {
    bitmask_unop_std(src, UnaryOperator::Not)
}

/// Logical inclusion: output bit is 1 if the corresponding LHS bit value is present in the RHS bit-set.
///
/// Implements set membership semantics for boolean bitmasks. The algorithm first scans the RHS bitmask
/// to determine which values (true/false) are present, then selects an optimal strategy based on the
/// composition of the RHS set.
///
/// # Parameters
/// - `lhs`: Source bitmask window to test for membership
/// - `rhs`: Reference bitmask window representing the set of allowed values
///
/// # Returns
/// A new `Bitmask` where each bit indicates whether the corresponding LHS value is present in RHS.
#[inline]
pub fn in_mask(lhs: BitmaskVT<'_>, rhs: BitmaskVT<'_>) -> Bitmask {
    let (lhs_mask, lhs_off, len) = lhs;
    let (rhs_mask, rhs_off, _) = rhs;
    let mut has_true = false;
    let mut has_false = false;
    for i in 0..len {
        let v = unsafe { rhs_mask.get_unchecked(rhs_off + i) };
        if v {
            has_true = true;
        } else {
            has_false = true;
        }
        if has_true && has_false {
            break;
        }
    }
    match (has_true, has_false) {
        // mixed -> every lhs bit (true/false) is present in rhs -> all true
        (true, true) => Bitmask::new_set_all(len, true),
        // only true in rhs -> pass through lhs true bits
        (true, false) => lhs_mask.slice_clone(lhs_off, len),
        // only false in rhs -> pass through lhs false bits (invert lhs)
        (false, true) => not_mask((lhs_mask, lhs_off, len)),
        // rhs empty -> nothing matches -> all false
        (false, false) => Bitmask::new_set_all(len, false),
    }
}
/// Logical exclusion: output bit is 1 if lhs ∉ rhs bit-set.
#[inline]
pub fn not_in_mask(lhs: BitmaskVT<'_>, rhs: BitmaskVT<'_>) -> Bitmask {
    let mask = in_mask(lhs, rhs);
    not_mask((&mask, 0, mask.len()))
}

/// Element-wise equality: output bit is 1 if bits at position are equal.
#[inline]
pub fn eq_mask(a: BitmaskVT<'_>, b: BitmaskVT<'_>) -> Bitmask {
    let (am, ao, len) = a;
    let (bm, bo, blen) = b;
    debug_assert_eq!(len, blen, "BitWindow length mismatch in eq_mask");
    if len == 0 {
        return Bitmask::new_set_all(0, true);
    }
    if ao % 64 != 0 || bo % 64 != 0 {
        panic!(
            "eq_mask: offsets must be 64-bit aligned (got a: {}, b: {})",
            ao, bo
        );
    }
    let mut out = Bitmask::new_set_all(len, false);
    {
        let a_bytes = bitmask_window_bytes(am, ao, len);
        let b_bytes = bitmask_window_bytes(bm, bo, len);
        let dp_bytes = bitmask_window_bytes_mut(&mut out, 0, len);
        let total_bytes = a_bytes.len();
        let full_words = total_bytes / 8;
        let tail_bytes = total_bytes % 8;
        unsafe {
            let ap = a_bytes.as_ptr().cast::<u64>();
            let bp = b_bytes.as_ptr().cast::<u64>();
            let dp = dp_bytes.as_mut_ptr().cast::<u64>();
            for k in 0..full_words {
                *dp.add(k) = !(*ap.add(k) ^ *bp.add(k));
            }
        }
        let base = full_words * 8;
        for k in 0..tail_bytes {
            dp_bytes[base + k] = !(a_bytes[base + k] ^ b_bytes[base + k]);
        }
    }
    out.mask_trailing_bits();
    out
}

/// Element-wise inequality: output bit is 1 if bits at position differ.
#[inline]
pub fn ne_mask(a: BitmaskVT<'_>, b: BitmaskVT<'_>) -> Bitmask {
    let eq = eq_mask(a, b);
    not_mask((&eq, 0, eq.len()))
}

/// Returns true if all bits are equal across two slices.
/// Logical equality on the **valid** bits only.
#[inline]
pub fn all_eq_mask(a: BitmaskVT<'_>, b: BitmaskVT<'_>) -> bool {
    let (am, ao, len) = a;
    let (bm, bo, blen) = b;
    debug_assert_eq!(len, blen);

    if len == 0 {
        return true;
    }
    if ao % 64 != 0 || bo % 64 != 0 {
        panic!(
            "all_eq_mask: offsets must be 64-bit aligned (got a: {}, b: {})",
            ao, bo
        );
    }

    let a_bytes = bitmask_window_bytes(am, ao, len);
    let b_bytes = bitmask_window_bytes(bm, bo, len);
    let total_bytes = a_bytes.len();
    let full_words = total_bytes / 8;
    let tail_bytes = total_bytes % 8;
    let full_logical_bytes = len / 8;
    let last_bits = len & 7;

    unsafe {
        let ap = a_bytes.as_ptr().cast::<u64>();
        let bp = b_bytes.as_ptr().cast::<u64>();
        for k in 0..full_words {
            if *ap.add(k) != *bp.add(k) {
                return false;
            }
        }
    }
    let base = full_words * 8;
    for k in 0..tail_bytes {
        let byte_index = base + k;
        let av = a_bytes[byte_index];
        let bv = b_bytes[byte_index];
        if byte_index < full_logical_bytes {
            if av != bv {
                return false;
            }
        } else if last_bits != 0 {
            let m = (1u8 << last_bits) - 1;
            if (av & m) != (bv & m) {
                return false;
            }
        }
    }
    true
}

/// Returns true if all bits differ across two slices.
#[inline]
pub fn all_ne_mask(a: BitmaskVT<'_>, b: BitmaskVT<'_>) -> bool {
    !all_eq_mask(a, b)
}

/// Count of set `1` bits in the bitmask using native hardware popcount instructions.
///
/// Efficiently computes the population count (number of set bits) across the specified bitmask window.
/// The implementation processes data in 64-bit words and uses native CPU popcount instructions for
/// optimal performance.
///
/// # Parameters
/// - `m`: Bitmask window as `(mask, offset, length)` tuple
///
/// # Returns
/// The total number of set bits in the specified window.
#[inline]
pub fn popcount_mask(m: BitmaskVT<'_>) -> usize {
    let (mask, offset, len) = m;
    if len == 0 {
        return 0;
    }
    let bytes = bitmask_window_bytes(mask, offset, len);
    let total_bytes = bytes.len();
    let full_words = total_bytes / 8;
    let tail_bytes = total_bytes % 8;
    let full_logical_bytes = len / 8;
    let last_bits = len & 7;
    let mut acc = 0usize;

    unsafe {
        let words_ptr = bytes.as_ptr().cast::<u64>();
        for k in 0..full_words {
            acc += (*words_ptr.add(k)).count_ones() as usize;
        }
    }
    let base = full_words * 8;
    for k in 0..tail_bytes {
        let byte_index = base + k;
        let b = bytes[byte_index];
        if byte_index < full_logical_bytes {
            acc += b.count_ones() as usize;
        } else if last_bits != 0 {
            let m = (1u8 << last_bits) - 1;
            acc += (b & m).count_ones() as usize;
        }
    }
    acc
}

/// Are *all* logical bits `1`?
#[inline]
pub fn all_true_mask(mask: &Bitmask) -> bool {
    let n_bits = mask.len();
    if n_bits == 0 {
        return true;
    }
    let bytes = mask.bits.as_ref();
    let total_bytes = (n_bits + 7) / 8;
    let full_words = total_bytes / 8;
    let tail_bytes = total_bytes % 8;
    let full_logical_bytes = n_bits / 8;
    let last_bits = n_bits & 7;

    unsafe {
        let words_ptr = bytes.as_ptr().cast::<u64>();
        for k in 0..full_words {
            if *words_ptr.add(k) != !0u64 {
                return false;
            }
        }
    }
    let base = full_words * 8;
    for k in 0..tail_bytes {
        let byte_index = base + k;
        let b = bytes[byte_index];
        if byte_index < full_logical_bytes {
            if b != 0xFF {
                return false;
            }
        } else if last_bits != 0 {
            let m = (1u8 << last_bits) - 1;
            if (b & m) != m {
                return false;
            }
        }
    }
    true
}

/// Are *all* logical bits `0`?
#[inline]
pub fn all_false_mask(mask: &Bitmask) -> bool {
    let n_bits = mask.len();
    if n_bits == 0 {
        return true;
    }
    let bytes = mask.bits.as_ref();
    let total_bytes = (n_bits + 7) / 8;
    let full_words = total_bytes / 8;
    let tail_bytes = total_bytes % 8;
    let full_logical_bytes = n_bits / 8;
    let last_bits = n_bits & 7;

    unsafe {
        let words_ptr = bytes.as_ptr().cast::<u64>();
        for k in 0..full_words {
            if *words_ptr.add(k) != 0u64 {
                return false;
            }
        }
    }
    let base = full_words * 8;
    for k in 0..tail_bytes {
        let byte_index = base + k;
        let b = bytes[byte_index];
        if byte_index < full_logical_bytes {
            if b != 0 {
                return false;
            }
        } else if last_bits != 0 {
            let m = (1u8 << last_bits) - 1;
            if (b & m) != 0 {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Bitmask;

    // Helper: Create a Bitmask from a bool slice.
    fn bm(bits: &[bool]) -> Bitmask {
        let mut bm = Bitmask::new_set_all(bits.len(), false);
        for (i, &b) in bits.iter().enumerate() {
            unsafe { bm.set_unchecked(i, b) };
        }
        bm
    }

    // AND, OR, XOR, NOT, windowing, popcount, equality etc.
    #[test]
    fn test_and_masks() {
        let a = bm(&[true, false, true, true, false, false, true, true]);
        let b = bm(&[false, false, true, false, true, false, true, false]);
        let out = and_masks((&a, 0, a.len()), (&b, 0, b.len()));
        let expected = bm(&[false, false, true, false, false, false, true, false]);
        for i in 0..8 {
            assert_eq!(out.get(i), expected.get(i), "Mismatch at bit {}", i);
        }
    }

    #[test]
    fn test_or_masks() {
        let a = bm(&[true, false, true, true]);
        let b = bm(&[false, false, true, false]);
        let out = or_masks((&a, 0, a.len()), (&b, 0, b.len()));
        let expected = bm(&[true, false, true, true]);
        for i in 0..4 {
            assert_eq!(out.get(i), expected.get(i));
        }
    }

    #[test]
    fn test_xor_masks() {
        let a = bm(&[true, false, true, false]);
        let b = bm(&[false, true, true, false]);
        let out = xor_masks((&a, 0, a.len()), (&b, 0, b.len()));
        let expected = bm(&[true, true, false, false]);
        for i in 0..4 {
            assert_eq!(out.get(i), expected.get(i));
        }
    }

    #[test]
    fn test_not_mask() {
        let a = bm(&[true, false, true, false]);
        let out = not_mask((&a, 0, a.len()));
        let expected = bm(&[false, true, false, true]);
        for i in 0..4 {
            assert_eq!(out.get(i), expected.get(i));
        }
    }

    #[test]
    fn test_in_mask_all() {
        let a = bm(&[true, false, true]);
        let b = bm(&[true, false, true]); // has both true/false
        let out = in_mask((&a, 0, a.len()), (&b, 0, b.len()));
        for i in 0..a.len() {
            assert!(out.get(i), "in_mask (all true/false in rhs) bit {}", i);
        }
    }

    #[test]
    fn test_in_mask_true_only() {
        let a = bm(&[true, false, true]);
        let b = bm(&[true, true, true]);
        let out = in_mask((&a, 0, a.len()), (&b, 0, b.len()));
        // Only output bits set where a is true
        assert!(out.get(0));
        assert!(!out.get(1));
        assert!(out.get(2));
    }

    #[test]
    fn test_in_mask_false_only() {
        let a = bm(&[true, false, true]);
        let b = bm(&[false, false, false]);
        let out = in_mask((&a, 0, a.len()), (&b, 0, b.len()));
        // Only output bits set where a is false
        assert!(!out.get(0));
        assert!(out.get(1));
        assert!(!out.get(2));
    }

    #[test]
    fn test_not_in_mask() {
        let a = bm(&[true, false]);
        let b = bm(&[true, false]);
        let out = not_in_mask((&a, 0, a.len()), (&b, 0, b.len()));
        // Both 'in', so not_in_mask should be all false.
        for i in 0..a.len() {
            assert!(!out.get(i));
        }
    }

    #[test]
    fn test_eq_mask() {
        let a = bm(&[true, false, true]);
        let b = bm(&[true, false, false]);
        let out = eq_mask((&a, 0, a.len()), (&b, 0, b.len()));
        let expected = bm(&[true, true, false]);
        for i in 0..a.len() {
            assert_eq!(out.get(i), expected.get(i));
        }
    }

    #[test]
    fn test_ne_mask() {
        let a = bm(&[true, false, true]);
        let b = bm(&[true, true, false]);
        let out = ne_mask((&a, 0, a.len()), (&b, 0, b.len()));
        let expected = bm(&[false, true, true]);
        for i in 0..a.len() {
            assert_eq!(out.get(i), expected.get(i));
        }
    }

    #[test]
    fn test_all_eq_mask_true() {
        let a = bm(&[true, false, true, false]);
        let b = bm(&[true, false, true, false]);
        assert!(all_eq_mask((&a, 0, a.len()), (&b, 0, b.len())));
    }

    #[test]
    fn test_all_eq_mask_false() {
        let a = bm(&[true, false, true, false]);
        let b = bm(&[false, true, false, true]);
        assert!(!all_eq_mask((&a, 0, a.len()), (&b, 0, b.len())));
    }

    #[test]
    fn test_all_ne_mask_true() {
        let a = bm(&[true, false]);
        let b = bm(&[false, true]);
        assert!(all_ne_mask((&a, 0, a.len()), (&b, 0, b.len())));
    }

    #[test]
    fn test_all_ne_mask_false() {
        let a = bm(&[true, false]);
        let b = bm(&[true, false]);
        assert!(!all_ne_mask((&a, 0, a.len()), (&b, 0, b.len())));
    }

    #[test]
    fn test_popcount_mask() {
        let a = bm(&[true, false, true, false, true, true]);
        assert_eq!(popcount_mask((&a, 0, a.len())), 4);
    }

    #[test]
    fn test_all_true_mask() {
        let a = bm(&[true, true, true, true]);
        assert!(all_true_mask(&a));
        let b = bm(&[true, true, false, true]);
        assert!(!all_true_mask(&b));
    }

    #[test]
    fn test_all_false_mask() {
        let a = bm(&[false, false, false, false]);
        assert!(all_false_mask(&a));
        let b = bm(&[false, true, false, false]);
        assert!(!all_false_mask(&b));
    }

    #[test]
    fn test_clear_trailing_bits_and_window() {
        // Bitmask of len=9, underlying bytes=2, last 7 bits are slack
        let mut a = Bitmask::new_set_all(9, true);
        a.bits[1] = 0xFF; // force all bits set
        clear_trailing_bits(&mut a);
        // Only 9 bits should remain set (not 16)
        assert!(a.get(8));
        if a.bits[1] >> 1 != 0 {
            // Only the lowest bit (bit 8) is in use in byte 1
            panic!("Trailing slack bits not cleared");
        }

        // bitmask_window_bytes correctness
        let a = bm(&[true, false, true, true, false, false, true, false]);
        let window = bitmask_window_bytes(&a, 2, 4);
        assert_eq!(window.len(), 1); // 4 bits is within one byte
        let mut b = a.clone();
        let window_mut = bitmask_window_bytes_mut(&mut b, 2, 4);
        assert_eq!(window_mut.len(), 1);
    }
}
