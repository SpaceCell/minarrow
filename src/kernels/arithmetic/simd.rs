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

//! # **SIMD Arithmetic Kernels Module** - *High-Performance Arithmetic*
//!
//! Inner SIMD-accelerated implementations using `std::simd` for maximum performance on modern hardware.
//! Prefer dispatch.rs for easily handling the general case, otherwise you can use these inner functions
//! directly (e.g., "dense_simd") vs. "maybe masked, maybe simd".
//!
//! ## Overview
//! - **Portable SIMD**: Uses `std::simd` for cross-platform vectorisation with compile-time lane optimisation
//! - **Null masks**: Dense (no nulls) and masked variants for Arrow-compatible null handling.
//!   These are uniified in dispatch.rs, and opting out of masking yields no performance penalty.
//! - **Type support**: Integer and floating-point arithmetic with specialised FMA operations
//! - **Safety**: All unsafe operations are bounds-checked or guaranteed by caller invariants
//!
//! ## Architecture Notes
//! - Each kernel dispatches on the `ArithmeticOperator` once and runs a dedicated
//!   branch-free loop per operation
//! - Building blocks for higher-level dispatch layers, or for low-level hot loops
//! where one wants to fully avoid abstraction overhead.
//! - Parallelisation intentionally excluded to allow flexible chunking strategies
//! - Power operations fall back to scalar for integers, use logarithmic computation for floats

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use core::simd::{Mask, Simd, SimdElement};
use std::ops::{Add, Div, Mul, Rem, Sub};
use std::simd::cmp::SimdPartialEq;
use std::simd::{Select, StdFloat};

use crate::Bitmask;
use num_traits::{One, PrimInt, ToPrimitive, WrappingAdd, WrappingMul, WrappingSub, Zero};

use crate::enums::operators::ArithmeticOperator;
use crate::kernels::bitmask::simd::all_true_mask_simd;
use crate::utils::{simd_mask, write_simd_mask_bits};

/// SIMD integer arithmetic kernel for dense arrays (no nulls).
/// Vectorised operations with scalar fallback for power operations and array tails.
/// Panics on division/remainder by zero (consistent with scalar behaviour).
#[inline(always)]
pub fn int_dense_body_simd<T, const LANES: usize>(
    op: ArithmeticOperator,
    lhs: &[T],
    rhs: &[T],
    out: &mut [T],
) where
    T: Copy + One + PrimInt + ToPrimitive + Zero + SimdElement + WrappingMul,
    Simd<T, LANES>: Add<Output = Simd<T, LANES>>
        + Sub<Output = Simd<T, LANES>>
        + Mul<Output = Simd<T, LANES>>
        + Div<Output = Simd<T, LANES>>
        + Rem<Output = Simd<T, LANES>>,
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
        ArithmeticOperator::Add => run!(+),
        ArithmeticOperator::Subtract => run!(-),
        ArithmeticOperator::Multiply => run!(*),
        ArithmeticOperator::Divide => run!(/),    // Panics if divisor is zero
        ArithmeticOperator::Remainder => run!(%), // Panics if divisor is zero
        // Power and floor division run per element on the whole input.
        ArithmeticOperator::Power => {
            for idx in 0..n {
                let mut acc = T::one();
                let exp = rhs[idx].to_u32().unwrap_or(0);
                for _ in 0..exp {
                    acc = acc.wrapping_mul(&lhs[idx]);
                }
                out[idx] = acc;
            }
        }
        ArithmeticOperator::FloorDiv => {
            for idx in 0..n {
                out[idx] = if rhs[idx] == T::zero() {
                    panic!("Floor division by zero")
                } else {
                    let d = lhs[idx] / rhs[idx];
                    let r = lhs[idx] % rhs[idx];
                    if r != T::zero() && (lhs[idx] ^ rhs[idx]) < T::zero() {
                        d - T::one()
                    } else {
                        d
                    }
                };
            }
        }
    }
}

/// SIMD integer arithmetic kernel with null mask support.
/// Division/remainder by zero produces null results (mask=false) rather than panicking.
#[inline(always)]
pub fn int_masked_body_simd<T, const LANES: usize>(
    op: ArithmeticOperator,
    lhs: &[T],
    rhs: &[T],
    mask: &Bitmask,
    out: &mut [T],
    out_mask: &mut Bitmask,
) where
    T: Copy
        + PrimInt
        + ToPrimitive
        + Zero
        + One
        + SimdElement
        + PartialEq
        + WrappingAdd
        + WrappingMul
        + WrappingSub,
    Simd<T, LANES>: Add<Output = Simd<T, LANES>>
        + SimdPartialEq<Mask = Mask<<T as SimdElement>::Mask, LANES>>
        + Sub<Output = Simd<T, LANES>>
        + Mul<Output = Simd<T, LANES>>
        + Div<Output = Simd<T, LANES>>
        + Rem<Output = Simd<T, LANES>>,
{
    let n = lhs.len();
    let dense = all_true_mask_simd::<LANES>(mask);

    /* If dense, we unfortunately need to near-replicate the dense implementation
    as that dedicated function panics on `div/0` as it needs to stay mask-free,
    to support varied workloads. This one works on the same dense principles,
    but substitutes the null mask when any div/0 issues occur. */

    if dense {
        // This block replaces the int_dense_body_simd call and handles masking for div/rem.
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
            }};
        }
        // Division-family arms substitute a divisor of one where the true
        // divisor is zero, zero the affected lanes, and mark them null.
        macro_rules! run_div {
            ($vec_op:tt) => {{
                let mut i = 0;
                while i < vectorisable {
                    let a = Simd::<T, LANES>::from_slice(&lhs[i..i + LANES]);
                    let b = Simd::<T, LANES>::from_slice(&rhs[i..i + LANES]);
                    let div_zero = b.simd_eq(Simd::splat(T::zero()));
                    let safe_b = div_zero.select(Simd::splat(T::one()), b);
                    let r = div_zero.select(Simd::splat(T::zero()), a $vec_op safe_b);
                    r.copy_to_slice(&mut out[i..i + LANES]);
                    write_simd_mask_bits(out_mask, i, !div_zero);
                    i += LANES;
                }
            }};
        }
        match op {
            ArithmeticOperator::Add => run!(+),
            ArithmeticOperator::Subtract => run!(-),
            ArithmeticOperator::Multiply => run!(*),
            ArithmeticOperator::Divide => run_div!(/),
            ArithmeticOperator::Remainder => run_div!(%),
            ArithmeticOperator::Power => {
                let mut i = 0;
                while i < vectorisable {
                    let a = Simd::<T, LANES>::from_slice(&lhs[i..i + LANES]);
                    let b = Simd::<T, LANES>::from_slice(&rhs[i..i + LANES]);
                    let mut tmp = [T::zero(); LANES];
                    for l in 0..LANES {
                        tmp[l] = a[l].pow(b[l].to_u32().unwrap_or(0));
                    }
                    Simd::<T, LANES>::from_array(tmp).copy_to_slice(&mut out[i..i + LANES]);
                    write_simd_mask_bits(
                        out_mask,
                        i,
                        Mask::<<T as SimdElement>::Mask, LANES>::splat(true),
                    );
                    i += LANES;
                }
            }
            ArithmeticOperator::FloorDiv => {
                let mut i = 0;
                while i < vectorisable {
                    let a = Simd::<T, LANES>::from_slice(&lhs[i..i + LANES]);
                    let b = Simd::<T, LANES>::from_slice(&rhs[i..i + LANES]);
                    let div_zero = b.simd_eq(Simd::splat(T::zero()));
                    // Per-lane floor division with sign correction
                    let mut tmp = [T::zero(); LANES];
                    for l in 0..LANES {
                        if b[l] != T::zero() {
                            let d = a[l] / b[l];
                            let r = a[l] % b[l];
                            tmp[l] = if r != T::zero() && (a[l] ^ b[l]) < T::zero() {
                                d - T::one()
                            } else {
                                d
                            };
                        }
                    }
                    Simd::<T, LANES>::from_array(tmp).copy_to_slice(&mut out[i..i + LANES]);
                    write_simd_mask_bits(out_mask, i, !div_zero);
                    i += LANES;
                }
            }
        }
        // Scalar tail
        for idx in vectorisable..n {
            match op {
                ArithmeticOperator::Add => {
                    out[idx] = lhs[idx].wrapping_add(&rhs[idx]);
                    unsafe {
                        out_mask.set_unchecked(idx, true);
                    }
                }
                ArithmeticOperator::Subtract => {
                    out[idx] = lhs[idx].wrapping_sub(&rhs[idx]);
                    unsafe {
                        out_mask.set_unchecked(idx, true);
                    }
                }
                ArithmeticOperator::Multiply => {
                    out[idx] = lhs[idx].wrapping_mul(&rhs[idx]);
                    unsafe {
                        out_mask.set_unchecked(idx, true);
                    }
                }
                ArithmeticOperator::Power => {
                    out[idx] = lhs[idx].pow(rhs[idx].to_u32().unwrap_or(0));
                    unsafe {
                        out_mask.set_unchecked(idx, true);
                    }
                }
                ArithmeticOperator::Divide | ArithmeticOperator::Remainder => {
                    if rhs[idx] == T::zero() {
                        out[idx] = T::zero();
                        unsafe {
                            out_mask.set_unchecked(idx, false);
                        }
                    } else {
                        out[idx] = match op {
                            ArithmeticOperator::Divide => lhs[idx] / rhs[idx],
                            ArithmeticOperator::Remainder => lhs[idx] % rhs[idx],
                            _ => unreachable!(),
                        };
                        unsafe {
                            out_mask.set_unchecked(idx, true);
                        }
                    }
                }
                ArithmeticOperator::FloorDiv => {
                    if rhs[idx] == T::zero() {
                        out[idx] = T::zero();
                        unsafe {
                            out_mask.set_unchecked(idx, false);
                        }
                    } else {
                        let d = lhs[idx] / rhs[idx];
                        let r = lhs[idx] % rhs[idx];
                        out[idx] = if r != T::zero() && (lhs[idx] ^ rhs[idx]) < T::zero() {
                            d - T::one()
                        } else {
                            d
                        };
                        unsafe {
                            out_mask.set_unchecked(idx, true);
                        }
                    }
                }
            }
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
    macro_rules! run_div {
        ($vec_op:tt) => {{
            while i + LANES <= n {
                let a = Simd::<T, LANES>::from_slice(&lhs[i..i + LANES]);
                let b = Simd::<T, LANES>::from_slice(&rhs[i..i + LANES]);
                let m_src: Mask<<T as SimdElement>::Mask, LANES> = simd_mask(mask, i, n);
                let div_zero: Mask<_, LANES> = b.simd_eq(Simd::splat(T::zero()));
                let safe_b = div_zero.select(Simd::splat(T::one()), b); // 0 -> 1
                let res = div_zero.select(Simd::splat(T::zero()), a $vec_op safe_b);
                let selected = m_src.select(res, Simd::splat(T::zero()));
                selected.copy_to_slice(&mut out[i..i + LANES]);
                write_simd_mask_bits(out_mask, i, m_src & !div_zero);
                i += LANES;
            }
        }};
    }
    match op {
        ArithmeticOperator::Add => run!(+),
        ArithmeticOperator::Subtract => run!(-),
        ArithmeticOperator::Multiply => run!(*),
        ArithmeticOperator::Divide => run_div!(/),
        ArithmeticOperator::Remainder => run_div!(%),
        ArithmeticOperator::Power => {
            while i + LANES <= n {
                let a = Simd::<T, LANES>::from_slice(&lhs[i..i + LANES]);
                let b = Simd::<T, LANES>::from_slice(&rhs[i..i + LANES]);
                let m_src: Mask<<T as SimdElement>::Mask, LANES> = simd_mask(mask, i, n);
                // scalar per-lane power
                let mut tmp = [T::zero(); LANES];
                for l in 0..LANES {
                    tmp[l] = a[l].pow(b[l].to_u32().unwrap_or(0));
                }
                let selected =
                    m_src.select(Simd::<T, LANES>::from_array(tmp), Simd::splat(T::zero()));
                selected.copy_to_slice(&mut out[i..i + LANES]);
                write_simd_mask_bits(out_mask, i, m_src);
                i += LANES;
            }
        }
        ArithmeticOperator::FloorDiv => {
            while i + LANES <= n {
                let a = Simd::<T, LANES>::from_slice(&lhs[i..i + LANES]);
                let b = Simd::<T, LANES>::from_slice(&rhs[i..i + LANES]);
                let m_src: Mask<<T as SimdElement>::Mask, LANES> = simd_mask(mask, i, n);
                let div_zero: Mask<_, LANES> = b.simd_eq(Simd::splat(T::zero()));
                // Per-lane floor division with sign correction
                let mut tmp = [T::zero(); LANES];
                for l in 0..LANES {
                    if b[l] != T::zero() {
                        let d = a[l] / b[l];
                        let r = a[l] % b[l];
                        tmp[l] = if r != T::zero() && (a[l] ^ b[l]) < T::zero() {
                            d - T::one()
                        } else {
                            d
                        };
                    }
                }
                let selected =
                    m_src.select(Simd::<T, LANES>::from_array(tmp), Simd::splat(T::zero()));
                selected.copy_to_slice(&mut out[i..i + LANES]);
                write_simd_mask_bits(out_mask, i, m_src & !div_zero);
                i += LANES;
            }
        }
    }

    // scalar tail
    for j in i..n {
        let valid = unsafe { mask.get_unchecked(j) };
        if valid {
            let (result, final_valid) = match op {
                ArithmeticOperator::Add => (lhs[j].wrapping_add(&rhs[j]), true),
                ArithmeticOperator::Subtract => (lhs[j].wrapping_sub(&rhs[j]), true),
                ArithmeticOperator::Multiply => (lhs[j].wrapping_mul(&rhs[j]), true),
                ArithmeticOperator::Divide => {
                    if rhs[j] == T::zero() {
                        (T::zero(), false) // division by zero -> invalid
                    } else {
                        (lhs[j] / rhs[j], true)
                    }
                }
                ArithmeticOperator::Remainder => {
                    if rhs[j] == T::zero() {
                        (T::zero(), false) // remainder by zero -> invalid
                    } else {
                        (lhs[j] % rhs[j], true)
                    }
                }
                ArithmeticOperator::Power => (lhs[j].pow(rhs[j].to_u32().unwrap_or(0)), true),
                ArithmeticOperator::FloorDiv => {
                    if rhs[j] == T::zero() {
                        (T::zero(), false)
                    } else {
                        let d = lhs[j] / rhs[j];
                        let r = lhs[j] % rhs[j];
                        if r != T::zero() && (lhs[j] ^ rhs[j]) < T::zero() {
                            (d - T::one(), true)
                        } else {
                            (d, true)
                        }
                    }
                }
            };
            out[j] = result;
            unsafe { out_mask.set_unchecked(j, final_valid) };
        } else {
            out[j] = T::zero();
            unsafe { out_mask.set_unchecked(j, false) };
        }
    }
}

/// SIMD f32 arithmetic kernel with null mask support.
/// Preserves IEEE 754 semantics: division by zero produces Inf/NaN, no exceptions.
/// Power operations use scalar fallback with logarithmic computation.
#[inline(always)]
pub fn float_masked_body_f32_simd<const LANES: usize>(
    op: ArithmeticOperator,
    lhs: &[f32],
    rhs: &[f32],
    mask: &Bitmask,
    out: &mut [f32],
    out_mask: &mut Bitmask,
) {
    type M = <f32 as SimdElement>::Mask;

    let n = lhs.len();
    let dense = all_true_mask_simd::<LANES>(mask);

    if dense {
        float_dense_body_f32_simd::<LANES>(op, lhs, rhs, out);
        out_mask.fill(true);
        return;
    }

    macro_rules! run {
        (|$a:ident, $b:ident| $vec:expr, |$x:ident, $y:ident| $scalar:expr) => {{
            let mut i = 0;
            while i + LANES <= n {
                let $a = Simd::<f32, LANES>::from_slice(&lhs[i..i + LANES]);
                let $b = Simd::<f32, LANES>::from_slice(&rhs[i..i + LANES]);
                let m: Mask<M, LANES> = simd_mask::<M, LANES>(mask, i, n);

                let res = $vec;
                let selected = m.select(res, Simd::<f32, LANES>::splat(0.0));
                selected.copy_to_slice(&mut out[i..i + LANES]);

                write_simd_mask_bits(out_mask, i, m);
                i += LANES;
            }

            // The tail covers the final `n % LANES` rows with the scalar form
            for j in i..n {
                let valid = unsafe { mask.get_unchecked(j) };
                if valid {
                    let $x = lhs[j];
                    let $y = rhs[j];
                    out[j] = $scalar;
                    unsafe { out_mask.set_unchecked(j, true) };
                } else {
                    out[j] = 0.0;
                    unsafe { out_mask.set_unchecked(j, false) };
                }
            }
        }};
    }
    match op {
        ArithmeticOperator::Add => run!(|a, b| a + b, |x, y| x + y),
        ArithmeticOperator::Subtract => run!(|a, b| a - b, |x, y| x - y),
        ArithmeticOperator::Multiply => run!(|a, b| a * b, |x, y| x * y),
        ArithmeticOperator::Divide => run!(|a, b| a / b, |x, y| x / y),
        ArithmeticOperator::Remainder => run!(|a, b| a % b, |x, y| x % y),
        ArithmeticOperator::Power => run!(|a, b| (b * a.ln()).exp(), |x, y| (y * x.ln()).exp()),
        ArithmeticOperator::FloorDiv => run!(|a, b| (a / b).floor(), |x, y| (x / y).floor()),
    }
}

/// SIMD f64 arithmetic kernel with null mask support.
/// Preserves IEEE 754 semantics: division by zero produces Inf/NaN, no exceptions.
/// Power operations use scalar fallback with logarithmic computation.
#[inline(always)]
pub fn float_masked_body_f64_simd<const LANES: usize>(
    op: ArithmeticOperator,
    lhs: &[f64],
    rhs: &[f64],
    mask: &Bitmask,
    out: &mut [f64],
    out_mask: &mut Bitmask,
) {
    type M = <f64 as SimdElement>::Mask;

    let n = lhs.len();
    let dense = all_true_mask_simd::<LANES>(mask);

    if dense {
        // hot
        float_dense_body_f64_simd::<LANES>(op, lhs, rhs, out);
        out_mask.fill(true);
        return;
    }

    macro_rules! run {
        (|$a:ident, $b:ident| $vec:expr, |$x:ident, $y:ident| $scalar:expr) => {{
            let mut i = 0;
            while i + LANES <= n {
                let $a = Simd::<f64, LANES>::from_slice(&lhs[i..i + LANES]);
                let $b = Simd::<f64, LANES>::from_slice(&rhs[i..i + LANES]);
                let m: Mask<M, LANES> = simd_mask::<M, LANES>(mask, i, n);

                let res = $vec;
                let selected = m.select(res, Simd::<f64, LANES>::splat(0.0));
                selected.copy_to_slice(&mut out[i..i + LANES]);

                write_simd_mask_bits(out_mask, i, m);
                i += LANES;
            }

            // The tail covers the final `n % LANES` rows with the scalar form
            for j in i..n {
                let valid = unsafe { mask.get_unchecked(j) };
                if valid {
                    let $x = lhs[j];
                    let $y = rhs[j];
                    out[j] = $scalar;
                    unsafe { out_mask.set_unchecked(j, true) };
                } else {
                    out[j] = 0.0;
                    unsafe { out_mask.set_unchecked(j, false) };
                }
            }
        }};
    }
    match op {
        ArithmeticOperator::Add => run!(|a, b| a + b, |x, y| x + y),
        ArithmeticOperator::Subtract => run!(|a, b| a - b, |x, y| x - y),
        ArithmeticOperator::Multiply => run!(|a, b| a * b, |x, y| x * y),
        ArithmeticOperator::Divide => run!(|a, b| a / b, |x, y| x / y),
        ArithmeticOperator::Remainder => run!(|a, b| a % b, |x, y| x % y),
        ArithmeticOperator::Power => run!(|a, b| (b * a.ln()).exp(), |x, y| (y * x.ln()).exp()),
        ArithmeticOperator::FloorDiv => run!(|a, b| (a / b).floor(), |x, y| (x / y).floor()),
    }
}

/// SIMD f32 arithmetic kernel for dense arrays (no nulls).
/// Vectorised operations with scalar fallback for power operations and array tails.
/// Division by zero produces Inf/NaN following IEEE 754 semantics.
#[inline(always)]
pub fn float_dense_body_f32_simd<const LANES: usize>(
    op: ArithmeticOperator,
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
) {
    let n = lhs.len();
    macro_rules! run {
        (|$a:ident, $b:ident| $vec:expr, |$x:ident, $y:ident| $scalar:expr) => {{
            let mut i = 0;
            while i + LANES <= n {
                let $a = Simd::<f32, LANES>::from_slice(&lhs[i..i + LANES]);
                let $b = Simd::<f32, LANES>::from_slice(&rhs[i..i + LANES]);
                let res = $vec;
                res.copy_to_slice(&mut out[i..i + LANES]);
                i += LANES;
            }
            // The tail covers the final `n % LANES` rows with the scalar form
            for j in i..n {
                let $x = lhs[j];
                let $y = rhs[j];
                out[j] = $scalar;
            }
        }};
    }
    match op {
        ArithmeticOperator::Add => run!(|a, b| a + b, |x, y| x + y),
        ArithmeticOperator::Subtract => run!(|a, b| a - b, |x, y| x - y),
        ArithmeticOperator::Multiply => run!(|a, b| a * b, |x, y| x * y),
        ArithmeticOperator::Divide => run!(|a, b| a / b, |x, y| x / y),
        ArithmeticOperator::Remainder => run!(|a, b| a % b, |x, y| x % y),
        ArithmeticOperator::Power => run!(|a, b| (b * a.ln()).exp(), |x, y| (y * x.ln()).exp()),
        ArithmeticOperator::FloorDiv => run!(|a, b| (a / b).floor(), |x, y| (x / y).floor()),
    }
}

/// SIMD f64 arithmetic kernel for dense arrays (no nulls).
/// Vectorised operations with scalar fallback for power operations and array tails.
/// Division by zero produces Inf/NaN following IEEE 754 semantics.
#[inline(always)]
pub fn float_dense_body_f64_simd<const LANES: usize>(
    op: ArithmeticOperator,
    lhs: &[f64],
    rhs: &[f64],
    out: &mut [f64],
) {
    let n = lhs.len();
    macro_rules! run {
        (|$a:ident, $b:ident| $vec:expr, |$x:ident, $y:ident| $scalar:expr) => {{
            let mut i = 0;
            while i + LANES <= n {
                let $a = Simd::<f64, LANES>::from_slice(&lhs[i..i + LANES]);
                let $b = Simd::<f64, LANES>::from_slice(&rhs[i..i + LANES]);
                let res = $vec;
                res.copy_to_slice(&mut out[i..i + LANES]);
                i += LANES;
            }
            // The tail covers the final `n % LANES` rows with the scalar form
            for j in i..n {
                let $x = lhs[j];
                let $y = rhs[j];
                out[j] = $scalar;
            }
        }};
    }
    match op {
        ArithmeticOperator::Add => run!(|a, b| a + b, |x, y| x + y),
        ArithmeticOperator::Subtract => run!(|a, b| a - b, |x, y| x - y),
        ArithmeticOperator::Multiply => run!(|a, b| a * b, |x, y| x * y),
        ArithmeticOperator::Divide => run!(|a, b| a / b, |x, y| x / y),
        ArithmeticOperator::Remainder => run!(|a, b| a % b, |x, y| x % y),
        ArithmeticOperator::Power => run!(|a, b| (b * a.ln()).exp(), |x, y| (y * x.ln()).exp()),
        ArithmeticOperator::FloorDiv => run!(|a, b| (a / b).floor(), |x, y| (x / y).floor()),
    }
}

/// SIMD f32 fused multiply-add kernel with null mask support.
/// Hardware-accelerated `a.mul_add(b, c)` with proper null propagation.
#[inline(always)]
pub fn fma_masked_body_f32_simd<const LANES: usize>(
    lhs: &[f32],
    rhs: &[f32],
    acc: &[f32],
    mask: &Bitmask,
    out: &mut [f32],
    out_mask: &mut Bitmask,
) {
    use core::simd::{Mask, Simd};

    let n = lhs.len();
    let mut i = 0;
    let dense = all_true_mask_simd::<LANES>(mask);

    if dense {
        fma_dense_body_f32_simd::<LANES>(lhs, rhs, acc, out);
        out_mask.fill(true);
        return;
    }

    while i + LANES <= n {
        let a = Simd::<f32, LANES>::from_slice(&lhs[i..i + LANES]);
        let b = Simd::<f32, LANES>::from_slice(&rhs[i..i + LANES]);
        let c = Simd::<f32, LANES>::from_slice(&acc[i..i + LANES]);
        let m: Mask<i32, LANES> = simd_mask::<i32, LANES>(mask, i, n);

        let res = a.mul_add(b, c);

        let selected = m.select(res, Simd::<f32, LANES>::splat(0.0));
        selected.copy_to_slice(&mut out[i..i + LANES]);

        write_simd_mask_bits(out_mask, i, m);
        i += LANES;
    }

    // Scalar tail

    for j in i..n {
        let valid = unsafe { mask.get_unchecked(j) };
        if valid {
            out[j] = lhs[j].mul_add(rhs[j], acc[j]);
            unsafe { out_mask.set_unchecked(j, true) };
        } else {
            out[j] = 0.0;
            unsafe { out_mask.set_unchecked(j, false) };
        }
    }
}

/// SIMD f64 fused multiply-add kernel with null mask support.
/// Hardware-accelerated `a.mul_add(b, c)` with proper null propagation.
#[inline(always)]
pub fn fma_masked_body_f64_simd<const LANES: usize>(
    lhs: &[f64],
    rhs: &[f64],
    acc: &[f64],
    mask: &Bitmask,
    out: &mut [f64],
    out_mask: &mut Bitmask,
) {
    use core::simd::{Mask, Simd};

    let n = lhs.len();
    let mut i = 0;
    let dense = all_true_mask_simd::<LANES>(mask);

    if dense {
        // Hot
        fma_dense_body_f64_simd::<LANES>(lhs, rhs, acc, out);
        out_mask.fill(true);
        return;
    }

    while i + LANES <= n {
        let a = Simd::<f64, LANES>::from_slice(&lhs[i..i + LANES]);
        let b = Simd::<f64, LANES>::from_slice(&rhs[i..i + LANES]);
        let c = Simd::<f64, LANES>::from_slice(&acc[i..i + LANES]);
        let m: Mask<i64, LANES> = simd_mask::<i64, LANES>(mask, i, n);

        let res = a.mul_add(b, c);

        let selected = m.select(res, Simd::<f64, LANES>::splat(0.0));
        selected.copy_to_slice(&mut out[i..i + LANES]);

        write_simd_mask_bits(out_mask, i, m);
        i += LANES;
    }

    // Scalar tail

    for j in i..n {
        let valid = unsafe { mask.get_unchecked(j) };
        if valid {
            out[j] = lhs[j].mul_add(rhs[j], acc[j]);
            unsafe { out_mask.set_unchecked(j, true) };
        } else {
            out[j] = 0.0;
            unsafe { out_mask.set_unchecked(j, false) };
        }
    }
}

/// SIMD f32 fused multiply-add kernel for dense arrays (no nulls).
/// Hardware-accelerated `a.mul_add(b, c)` with vectorised and scalar tail processing.
#[inline(always)]
pub fn fma_dense_body_f32_simd<const LANES: usize>(
    lhs: &[f32],
    rhs: &[f32],
    acc: &[f32],
    out: &mut [f32],
) {
    use core::simd::Simd;

    let n = lhs.len();
    let mut i = 0;

    while i + LANES <= n {
        let a = Simd::<f32, LANES>::from_slice(&lhs[i..i + LANES]);
        let b = Simd::<f32, LANES>::from_slice(&rhs[i..i + LANES]);
        let c = Simd::<f32, LANES>::from_slice(&acc[i..i + LANES]);
        let res = a.mul_add(b, c);
        res.copy_to_slice(&mut out[i..i + LANES]);
        i += LANES;
    }

    for j in i..n {
        out[j] = lhs[j].mul_add(rhs[j], acc[j]);
    }
}

/// SIMD f64 fused multiply-add kernel for dense arrays (no nulls).
/// Hardware-accelerated `a.mul_add(b, c)` with vectorised and scalar tail processing.
#[inline(always)]
pub fn fma_dense_body_f64_simd<const LANES: usize>(
    lhs: &[f64],
    rhs: &[f64],
    acc: &[f64],
    out: &mut [f64],
) {
    use core::simd::Simd;

    let n = lhs.len();
    let mut i = 0;

    while i + LANES <= n {
        let a = Simd::<f64, LANES>::from_slice(&lhs[i..i + LANES]);
        let b = Simd::<f64, LANES>::from_slice(&rhs[i..i + LANES]);
        let c = Simd::<f64, LANES>::from_slice(&acc[i..i + LANES]);
        let res = a.mul_add(b, c);
        res.copy_to_slice(&mut out[i..i + LANES]);
        i += LANES;
    }

    // Tail uses scalar fallback
    for j in i..n {
        out[j] = lhs[j].mul_add(rhs[j], acc[j]);
    }
}
