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

//! # **Standard Arithmetic Kernels Module** - *Scalar Fallback / Non-SIMD Implementations*
//!
//! Portable scalar implementations of arithmetic operations for compatibility and unaligned data.
//!
//! Prefer dispatch.rs for easily handling the general case, otherwise you can use these inner functions
//! directly (e.g., "dense_std") vs. "maybe masked, maybe std".
//!
//! ## Overview
//! - **Scalar loops**: Standard element-wise operations without vectorisation
//! - **Fallback role**: Used when SIMD alignment requirements aren't met or SIMD is disabled
//! - **Full compatibility**: Works on any architecture regardless of SIMD support
//! - **Null-aware**: Supports Arrow-compatible null mask propagation
//!
//! ## Design Notes
//! - Each kernel dispatches on the `ArithmeticOperator` once and runs a dedicated
//!   branch-free loop per operation
//! - Intentionally avoids parallelisation to allow higher-level chunking strategies
//! - Wrapping arithmetic for integers to prevent overflow panics
//! - Division by zero handling: panics for integers, produces Inf/NaN for floats

use crate::Bitmask;
use crate::enums::operators::ArithmeticOperator;
use num_traits::{Float, PrimInt, ToPrimitive, WrappingAdd, WrappingMul, WrappingSub};

/// Scalar integer arithmetic kernel for dense arrays (no nulls).
/// Performs element-wise operations using wrapping arithmetic to prevent overflow panics.
/// Panics on division/remainder by zero.
#[inline(always)]
pub fn int_dense_body_std<T: PrimInt + ToPrimitive + WrappingAdd + WrappingSub + WrappingMul>(
    op: ArithmeticOperator,
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
        ArithmeticOperator::Add => run!(|x, y| x.wrapping_add(&y)),
        ArithmeticOperator::Subtract => run!(|x, y| x.wrapping_sub(&y)),
        ArithmeticOperator::Multiply => run!(|x, y| x.wrapping_mul(&y)),
        ArithmeticOperator::Divide => run!(|x, y| {
            if y == T::zero() {
                panic!("Division by zero")
            } else {
                x / y
            }
        }),
        ArithmeticOperator::Remainder => run!(|x, y| {
            if y == T::zero() {
                panic!("Remainder by zero")
            } else {
                x % y
            }
        }),
        ArithmeticOperator::Power => run!(|x, y| x.pow(y.to_u32().unwrap_or(0))),
        ArithmeticOperator::FloorDiv => run!(|x, y| {
            if y == T::zero() {
                panic!("Floor division by zero")
            } else {
                let d = x / y;
                let r = x % y;
                // If remainder is non-zero and signs differ, floor toward -inf
                if r != T::zero() && (x ^ y) < T::zero() {
                    d - T::one()
                } else {
                    d
                }
            }
        }),
    }
}

/// Scalar integer arithmetic kernel with null mask support.
/// Handles division by zero gracefully by marking results as null instead of panicking.
/// Invalid inputs (mask=false) and zero division produce null outputs.
#[inline(always)]
pub fn int_masked_body_std<T: PrimInt + ToPrimitive + WrappingAdd + WrappingSub + WrappingMul>(
    op: ArithmeticOperator,
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
                    let (result, final_valid) = $scalar;
                    out[i] = result;
                    unsafe {
                        out_mask.set_unchecked(i, final_valid);
                    }
                } else {
                    out[i] = T::zero();
                    unsafe {
                        out_mask.set_unchecked(i, false);
                    }
                }
            }
        }};
    }
    match op {
        ArithmeticOperator::Add => run!(|x, y| (x.wrapping_add(&y), true)),
        ArithmeticOperator::Subtract => run!(|x, y| (x.wrapping_sub(&y), true)),
        ArithmeticOperator::Multiply => run!(|x, y| (x.wrapping_mul(&y), true)),
        ArithmeticOperator::Divide => run!(|x, y| {
            if y == T::zero() {
                (T::zero(), false) // division by zero -> invalid
            } else {
                (x / y, true)
            }
        }),
        ArithmeticOperator::Remainder => run!(|x, y| {
            if y == T::zero() {
                (T::zero(), false) // remainder by zero -> invalid
            } else {
                (x % y, true)
            }
        }),
        ArithmeticOperator::Power => run!(|x, y| (x.pow(y.to_u32().unwrap_or(0)), true)),
        ArithmeticOperator::FloorDiv => run!(|x, y| {
            if y == T::zero() {
                (T::zero(), false)
            } else {
                let d = x / y;
                let r = x % y;
                if r != T::zero() && (x ^ y) < T::zero() {
                    (d - T::one(), true)
                } else {
                    (d, true)
                }
            }
        }),
    }
}

/// Scalar floating-point arithmetic kernel for dense arrays (no nulls).
/// Division by zero produces Inf/NaN rather than panicking.
/// Power operations use logarithmic exponentiation: `exp(b * ln(a))`.
#[inline(always)]
pub fn float_dense_body_std<T: Float>(op: ArithmeticOperator, lhs: &[T], rhs: &[T], out: &mut [T]) {
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
        ArithmeticOperator::Add => run!(|x, y| x + y),
        ArithmeticOperator::Subtract => run!(|x, y| x - y),
        ArithmeticOperator::Multiply => run!(|x, y| x * y),
        ArithmeticOperator::Divide => run!(|x, y| x / y),
        ArithmeticOperator::Remainder => run!(|x, y| x % y),
        ArithmeticOperator::Power => run!(|x, y| (y * x.ln()).exp()),
        ArithmeticOperator::FloorDiv => run!(|x, y| (x / y).floor()),
    }
}

/// Scalar floating-point arithmetic kernel with null mask support.
/// Preserves IEEE 754 semantics: division by zero produces Inf/NaN, no panicking.
/// Invalid inputs (mask=false) produce null outputs with zero values.
#[inline(always)]
pub fn float_masked_body_std<T: Float>(
    op: ArithmeticOperator,
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
                    unsafe {
                        out_mask.set_unchecked(i, true);
                    }
                } else {
                    out[i] = T::zero();
                    unsafe {
                        out_mask.set_unchecked(i, false);
                    }
                }
            }
        }};
    }
    match op {
        ArithmeticOperator::Add => run!(|x, y| x + y),
        ArithmeticOperator::Subtract => run!(|x, y| x - y),
        ArithmeticOperator::Multiply => run!(|x, y| x * y),
        ArithmeticOperator::Divide => run!(|x, y| x / y),
        ArithmeticOperator::Remainder => run!(|x, y| x % y),
        ArithmeticOperator::Power => run!(|x, y| (y * x.ln()).exp()),
        ArithmeticOperator::FloorDiv => run!(|x, y| (x / y).floor()),
    }
}

/// Fused multiply add (a * b + acc) with null mask
#[inline(always)]
pub fn fma_masked_body_std<T: Float>(
    lhs: &[T],
    rhs: &[T],
    acc: &[T],
    mask: &Bitmask,
    out: &mut [T],
    out_mask: &mut Bitmask,
) {
    let n = lhs.len();
    for i in 0..n {
        let valid = unsafe { mask.get_unchecked(i) };
        if valid {
            out[i] = lhs[i].mul_add(rhs[i], acc[i]);
            unsafe {
                out_mask.set_unchecked(i, true);
            }
        } else {
            out[i] = T::zero();
            unsafe {
                out_mask.set_unchecked(i, false);
            }
        }
    }
}

/// Dense fused multiply add (a * b + acc)
#[inline(always)]
pub fn fma_dense_body_std<T: Float>(lhs: &[T], rhs: &[T], acc: &[T], out: &mut [T]) {
    let n = lhs.len();
    for i in 0..n {
        out[i] = lhs[i].mul_add(rhs[i], acc[i]);
    }
}
