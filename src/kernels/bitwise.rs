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

//! # **Bitwise Kernels Module** - *Element-Wise Integer Bitwise Operations*
//!
//! Element-wise bitwise AND, OR, XOR over two integer arrays, plus unary bitwise
//! complement, with null-aware semantics. Bitwise operations never fail, so a
//! result carries the same validity as its input.
//!
//! ## Overview
//! - **SIMD path**: `std::simd` vectorisation with a scalar tail, selected for
//!   64-byte aligned inputs when the `simd` feature is enabled
//! - **Scalar path**: portable fallback for unaligned data or non-SIMD builds
//! - **Null-aware**: no-null and masked variants, mirroring the
//!   arithmetic kernels
//!
//! ## Scope
//! These do not leverage parallel-thread processing, as this is expected to be
//! applied in the engine layer, which is app-specific.

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use num_traits::PrimInt;

use crate::enums::error::KernelError;
use crate::enums::operators::BitwiseOperator;
use crate::structs::variants::integer::IntegerArray;
use crate::utils::confirm_equal_len;
use crate::{Bitmask, Vec64};

#[cfg(feature = "simd")]
use core::simd::{Mask, Simd, SimdElement};
#[cfg(feature = "simd")]
use std::ops::{BitAnd, BitOr, BitXor, Not};
#[cfg(feature = "simd")]
use std::simd::Select;
#[cfg(feature = "simd")]
use num_traits::Zero;
#[cfg(feature = "simd")]
use crate::kernels::bitmask::simd::all_true_mask_simd;
#[cfg(feature = "simd")]
use crate::utils::{is_simd_aligned, simd_mask, write_simd_mask_bits};

// ---------------------------------------------------------------------------
// Scalar bodies
// ---------------------------------------------------------------------------

/// Scalar integer bitwise kernel for arrays without nulls.
#[inline(always)]
pub fn int_bitwise_body_std<T: PrimInt>(
    op: BitwiseOperator,
    lhs: &[T],
    rhs: &[T],
    out: &mut [T],
) {
    let n = lhs.len();
    macro_rules! run {
        (|$x:ident, $y:ident| $scalar:expr) => {{
            for i in 0..n {
                let $x = lhs[i];
                let $y = rhs[i];
                out[i] = $scalar;
            }
        }};
    }
    match op {
        BitwiseOperator::And => run!(|x, y| x & y),
        BitwiseOperator::Or => run!(|x, y| x | y),
        BitwiseOperator::Xor => run!(|x, y| x ^ y),
    }
}

/// Scalar integer bitwise kernel with null mask support.
/// Bitwise operations never nullify, so the output validity equals the input
/// validity. Invalid lanes carry a zero value.
#[inline(always)]
pub fn int_bitwise_masked_body_std<T: PrimInt>(
    op: BitwiseOperator,
    lhs: &[T],
    rhs: &[T],
    mask: &Bitmask,
    out: &mut [T],
    out_mask: &mut Bitmask,
) {
    let n = lhs.len();
    macro_rules! run {
        (|$x:ident, $y:ident| $scalar:expr) => {{
            for i in 0..n {
                let valid = unsafe { mask.get_unchecked(i) };
                if valid {
                    let $x = lhs[i];
                    let $y = rhs[i];
                    out[i] = $scalar;
                    unsafe { out_mask.set_unchecked(i, true) };
                } else {
                    out[i] = T::zero();
                    unsafe { out_mask.set_unchecked(i, false) };
                }
            }
        }};
    }
    match op {
        BitwiseOperator::And => run!(|x, y| x & y),
        BitwiseOperator::Or => run!(|x, y| x | y),
        BitwiseOperator::Xor => run!(|x, y| x ^ y),
    }
}

/// Scalar bitwise complement (`!x`) for arrays without nulls.
#[inline(always)]
pub fn int_not_body_std<T: PrimInt>(input: &[T], out: &mut [T]) {
    for i in 0..input.len() {
        out[i] = !input[i];
    }
}

/// Scalar bitwise complement (`!x`) with null mask support.
#[inline(always)]
pub fn int_not_masked_body_std<T: PrimInt>(
    input: &[T],
    mask: &Bitmask,
    out: &mut [T],
    out_mask: &mut Bitmask,
) {
    for i in 0..input.len() {
        let valid = unsafe { mask.get_unchecked(i) };
        if valid {
            out[i] = !input[i];
            unsafe { out_mask.set_unchecked(i, true) };
        } else {
            out[i] = T::zero();
            unsafe { out_mask.set_unchecked(i, false) };
        }
    }
}

// ---------------------------------------------------------------------------
// SIMD bodies
// ---------------------------------------------------------------------------

/// SIMD integer bitwise kernel for arrays without nulls, with a scalar tail.
#[cfg(feature = "simd")]
#[inline(always)]
pub fn int_bitwise_body_simd<T, const LANES: usize>(
    op: BitwiseOperator,
    lhs: &[T],
    rhs: &[T],
    out: &mut [T],
) where
    T: Copy + PrimInt + SimdElement,
    Simd<T, LANES>: BitAnd<Output = Simd<T, LANES>>
        + BitOr<Output = Simd<T, LANES>>
        + BitXor<Output = Simd<T, LANES>>,
{
    let n = lhs.len();
    macro_rules! run {
        ($vec_op:tt) => {{
            let vectorisable = n / LANES * LANES;
            let mut i = 0;
            while i < vectorisable {
                let a = Simd::<T, LANES>::from_slice(&lhs[i..i + LANES]);
                let b = Simd::<T, LANES>::from_slice(&rhs[i..i + LANES]);
                (a $vec_op b).copy_to_slice(&mut out[i..i + LANES]);
                i += LANES;
            }
            // Scalar tail
            for idx in vectorisable..n {
                out[idx] = lhs[idx] $vec_op rhs[idx];
            }
        }};
    }
    match op {
        BitwiseOperator::And => run!(&),
        BitwiseOperator::Or => run!(|),
        BitwiseOperator::Xor => run!(^),
    }
}

/// SIMD integer bitwise kernel with null mask support, with a scalar tail.
/// Validity is preserved exactly: the output mask equals the input mask.
#[cfg(feature = "simd")]
#[inline(always)]
pub fn int_bitwise_masked_body_simd<T, const LANES: usize>(
    op: BitwiseOperator,
    lhs: &[T],
    rhs: &[T],
    mask: &Bitmask,
    out: &mut [T],
    out_mask: &mut Bitmask,
) where
    T: Copy + PrimInt + Zero + SimdElement,
    Simd<T, LANES>: BitAnd<Output = Simd<T, LANES>>
        + BitOr<Output = Simd<T, LANES>>
        + BitXor<Output = Simd<T, LANES>>,
{
    let n = lhs.len();
    let no_nulls = all_true_mask_simd::<LANES>(mask);

    if no_nulls {
        let vectorisable = n / LANES * LANES;
        macro_rules! run {
            ($vec_op:tt) => {{
                let mut i = 0;
                while i < vectorisable {
                    let a = Simd::<T, LANES>::from_slice(&lhs[i..i + LANES]);
                    let b = Simd::<T, LANES>::from_slice(&rhs[i..i + LANES]);
                    (a $vec_op b).copy_to_slice(&mut out[i..i + LANES]);
                    write_simd_mask_bits(
                        out_mask,
                        i,
                        Mask::<<T as SimdElement>::Mask, LANES>::splat(true),
                    );
                    i += LANES;
                }
                for idx in vectorisable..n {
                    out[idx] = lhs[idx] $vec_op rhs[idx];
                    unsafe { out_mask.set_unchecked(idx, true) };
                }
            }};
        }
        match op {
            BitwiseOperator::And => run!(&),
            BitwiseOperator::Or => run!(|),
            BitwiseOperator::Xor => run!(^),
        }
        return;
    }

    let mut i = 0;
    macro_rules! run {
        ($vec_op:tt) => {{
            while i + LANES <= n {
                let a = Simd::<T, LANES>::from_slice(&lhs[i..i + LANES]);
                let b = Simd::<T, LANES>::from_slice(&rhs[i..i + LANES]);
                let m_src: Mask<<T as SimdElement>::Mask, LANES> = simd_mask(mask, i, n);
                let selected = m_src.select(a $vec_op b, Simd::splat(T::zero()));
                selected.copy_to_slice(&mut out[i..i + LANES]);
                write_simd_mask_bits(out_mask, i, m_src);
                i += LANES;
            }
        }};
    }
    match op {
        BitwiseOperator::And => run!(&),
        BitwiseOperator::Or => run!(|),
        BitwiseOperator::Xor => run!(^),
    }
    // Scalar tail
    for idx in i..n {
        let valid = unsafe { mask.get_unchecked(idx) };
        if valid {
            out[idx] = match op {
                BitwiseOperator::And => lhs[idx] & rhs[idx],
                BitwiseOperator::Or => lhs[idx] | rhs[idx],
                BitwiseOperator::Xor => lhs[idx] ^ rhs[idx],
            };
            unsafe { out_mask.set_unchecked(idx, true) };
        } else {
            out[idx] = T::zero();
            unsafe { out_mask.set_unchecked(idx, false) };
        }
    }
}

/// SIMD bitwise complement (`!x`) for arrays without nulls, with a scalar tail.
#[cfg(feature = "simd")]
#[inline(always)]
pub fn int_not_body_simd<T, const LANES: usize>(input: &[T], out: &mut [T])
where
    T: Copy + PrimInt + SimdElement,
    Simd<T, LANES>: Not<Output = Simd<T, LANES>>,
{
    let n = input.len();
    let vectorisable = n / LANES * LANES;
    let mut i = 0;
    while i < vectorisable {
        let a = Simd::<T, LANES>::from_slice(&input[i..i + LANES]);
        (!a).copy_to_slice(&mut out[i..i + LANES]);
        i += LANES;
    }
    for idx in vectorisable..n {
        out[idx] = !input[idx];
    }
}

/// SIMD bitwise complement (`!x`) with null mask support, with a scalar tail.
#[cfg(feature = "simd")]
#[inline(always)]
pub fn int_not_masked_body_simd<T, const LANES: usize>(
    input: &[T],
    mask: &Bitmask,
    out: &mut [T],
    out_mask: &mut Bitmask,
) where
    T: Copy + PrimInt + Zero + SimdElement,
    Simd<T, LANES>: Not<Output = Simd<T, LANES>>,
{
    let n = input.len();
    let mut i = 0;
    while i + LANES <= n {
        let a = Simd::<T, LANES>::from_slice(&input[i..i + LANES]);
        let m_src: Mask<<T as SimdElement>::Mask, LANES> = simd_mask(mask, i, n);
        let selected = m_src.select(!a, Simd::splat(T::zero()));
        selected.copy_to_slice(&mut out[i..i + LANES]);
        write_simd_mask_bits(out_mask, i, m_src);
        i += LANES;
    }
    for idx in i..n {
        let valid = unsafe { mask.get_unchecked(idx) };
        if valid {
            out[idx] = !input[idx];
            unsafe { out_mask.set_unchecked(idx, true) };
        } else {
            out[idx] = T::zero();
            unsafe { out_mask.set_unchecked(idx, false) };
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Generates element-wise integer bitwise functions with SIMD/scalar dispatch.
/// The SIMD path is selected for 64-byte aligned inputs when the `simd` feature
/// is enabled, otherwise the scalar path runs.
macro_rules! impl_apply_int_bitwise {
    ($fn_name:ident, $ty:ty, $lanes:expr) => {
        #[doc = concat!(
            "Element-wise integer `BitwiseOperator` over two `&[", stringify!($ty),
            "]`, SIMD-accelerated using ", stringify!($lanes), " lanes when available, \
            otherwise scalar. Returns `IntegerArray<", stringify!($ty), ">`."
        )]
        #[inline(always)]
        pub fn $fn_name(
            lhs: &[$ty],
            rhs: &[$ty],
            op: BitwiseOperator,
            mask: Option<&Bitmask>,
        ) -> Result<IntegerArray<$ty>, KernelError> {
            let len = lhs.len();
            confirm_equal_len("apply bitwise: length mismatch", len, rhs.len())?;

            #[cfg(feature = "simd")]
            {
                if is_simd_aligned(lhs) && is_simd_aligned(rhs) {
                    let mut out = Vec64::with_capacity(len);
                    unsafe { out.set_len(len) };
                    match mask {
                        Some(mask) => {
                            let mut out_mask = Bitmask::new_set_all(len, true);
                            int_bitwise_masked_body_simd::<$ty, $lanes>(
                                op, lhs, rhs, mask, &mut out, &mut out_mask,
                            );
                            return Ok(IntegerArray { data: out.into(), null_mask: Some(out_mask) });
                        }
                        None => {
                            int_bitwise_body_simd::<$ty, $lanes>(op, lhs, rhs, &mut out);
                            return Ok(IntegerArray { data: out.into(), null_mask: None });
                        }
                    }
                }
            }

            let mut out = Vec64::with_capacity(len);
            unsafe { out.set_len(len) };
            match mask {
                Some(mask) => {
                    let mut out_mask = Bitmask::new_set_all(len, true);
                    int_bitwise_masked_body_std::<$ty>(op, lhs, rhs, mask, &mut out, &mut out_mask);
                    Ok(IntegerArray { data: out.into(), null_mask: Some(out_mask) })
                }
                None => {
                    int_bitwise_body_std::<$ty>(op, lhs, rhs, &mut out);
                    Ok(IntegerArray { data: out.into(), null_mask: None })
                }
            }
        }
    };
}

/// Generates element-wise integer bitwise complement functions with SIMD/scalar
/// dispatch, mirroring [`impl_apply_int_bitwise`].
macro_rules! impl_apply_int_not {
    ($fn_name:ident, $ty:ty, $lanes:expr) => {
        #[doc = concat!(
            "Element-wise bitwise complement (`!x`) over `&[", stringify!($ty),
            "]`, SIMD-accelerated using ", stringify!($lanes), " lanes when available, \
            otherwise scalar. Returns `IntegerArray<", stringify!($ty), ">`."
        )]
        #[inline(always)]
        pub fn $fn_name(
            input: &[$ty],
            mask: Option<&Bitmask>,
        ) -> Result<IntegerArray<$ty>, KernelError> {
            let len = input.len();

            #[cfg(feature = "simd")]
            {
                if is_simd_aligned(input) {
                    let mut out = Vec64::with_capacity(len);
                    unsafe { out.set_len(len) };
                    match mask {
                        Some(mask) => {
                            let mut out_mask = Bitmask::new_set_all(len, true);
                            int_not_masked_body_simd::<$ty, $lanes>(input, mask, &mut out, &mut out_mask);
                            return Ok(IntegerArray { data: out.into(), null_mask: Some(out_mask) });
                        }
                        None => {
                            int_not_body_simd::<$ty, $lanes>(input, &mut out);
                            return Ok(IntegerArray { data: out.into(), null_mask: None });
                        }
                    }
                }
            }

            let mut out = Vec64::with_capacity(len);
            unsafe { out.set_len(len) };
            match mask {
                Some(mask) => {
                    let mut out_mask = Bitmask::new_set_all(len, true);
                    int_not_masked_body_std::<$ty>(input, mask, &mut out, &mut out_mask);
                    Ok(IntegerArray { data: out.into(), null_mask: Some(out_mask) })
                }
                None => {
                    int_not_body_std::<$ty>(input, &mut out);
                    Ok(IntegerArray { data: out.into(), null_mask: None })
                }
            }
        }
    };
}

impl_apply_int_bitwise!(apply_int_bitwise_i32, i32, W32);
impl_apply_int_bitwise!(apply_int_bitwise_u32, u32, W32);
impl_apply_int_bitwise!(apply_int_bitwise_i64, i64, W64);
impl_apply_int_bitwise!(apply_int_bitwise_u64, u64, W64);
#[cfg(feature = "extended_numeric_types")]
impl_apply_int_bitwise!(apply_int_bitwise_i16, i16, W16);
#[cfg(feature = "extended_numeric_types")]
impl_apply_int_bitwise!(apply_int_bitwise_u16, u16, W16);
#[cfg(feature = "extended_numeric_types")]
impl_apply_int_bitwise!(apply_int_bitwise_i8, i8, W8);
#[cfg(feature = "extended_numeric_types")]
impl_apply_int_bitwise!(apply_int_bitwise_u8, u8, W8);

impl_apply_int_not!(apply_int_not_i32, i32, W32);
impl_apply_int_not!(apply_int_not_u32, u32, W32);
impl_apply_int_not!(apply_int_not_i64, i64, W64);
impl_apply_int_not!(apply_int_not_u64, u64, W64);
#[cfg(feature = "extended_numeric_types")]
impl_apply_int_not!(apply_int_not_i16, i16, W16);
#[cfg(feature = "extended_numeric_types")]
impl_apply_int_not!(apply_int_not_u16, u16, W16);
#[cfg(feature = "extended_numeric_types")]
impl_apply_int_not!(apply_int_not_i8, i8, W8);
#[cfg(feature = "extended_numeric_types")]
impl_apply_int_not!(apply_int_not_u8, u8, W8);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MaskedArray;

    #[test]
    fn bitwise_and_or_xor_no_nulls() {
        let lhs: Vec<i64> = (0..40).collect();
        let rhs: Vec<i64> = (0..40).map(|x| x * 3 + 1).collect();

        let and = apply_int_bitwise_i64(&lhs, &rhs, BitwiseOperator::And, None).unwrap();
        let or = apply_int_bitwise_i64(&lhs, &rhs, BitwiseOperator::Or, None).unwrap();
        let xor = apply_int_bitwise_i64(&lhs, &rhs, BitwiseOperator::Xor, None).unwrap();

        for i in 0..40 {
            assert_eq!(and.data[i], lhs[i] & rhs[i]);
            assert_eq!(or.data[i], lhs[i] | rhs[i]);
            assert_eq!(xor.data[i], lhs[i] ^ rhs[i]);
        }
        assert!(and.null_mask.is_none());
    }

    #[test]
    fn bitwise_not_no_nulls() {
        let input: Vec<i32> = (-20..20).collect();
        let out = apply_int_not_i32(&input, None).unwrap();
        for i in 0..input.len() {
            assert_eq!(out.data[i], !input[i]);
        }
    }

    #[test]
    fn bitwise_and_equals_predicate() {
        // The filter's BITWISE_AND_EQUALS use case: (col & value) == value.
        let col: Vec<i64> = vec![0b0000, 0b0110, 0b0100, 0b0111, 0b1100];
        let flag = 0b0100i64;
        let rhs = vec![flag; col.len()];
        let masked = apply_int_bitwise_i64(&col, &rhs, BitwiseOperator::And, None).unwrap();
        let has_flag: Vec<bool> = (0..col.len()).map(|i| masked.data[i] == flag).collect();
        assert_eq!(has_flag, vec![false, true, true, true, true]);
    }

    #[test]
    fn bitwise_masked_preserves_validity() {
        let lhs: Vec<i64> = (0..20).collect();
        let rhs: Vec<i64> = (0..20).map(|x| x + 5).collect();
        let mut mask = Bitmask::new_set_all(20, true);
        mask.set(3, false);
        mask.set(17, false);

        let out = apply_int_bitwise_i64(&lhs, &rhs, BitwiseOperator::Or, Some(&mask)).unwrap();
        let out_mask = out.null_mask.as_ref().unwrap();
        for i in 0..20 {
            assert_eq!(out_mask.get(i), i != 3 && i != 17);
            if out_mask.get(i) {
                assert_eq!(out.data[i], lhs[i] | rhs[i]);
            }
        }
    }
}
