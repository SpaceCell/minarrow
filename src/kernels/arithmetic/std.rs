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
    for i in 0..n {
        out[i] = match op {
            ArithmeticOperator::Add => lhs[i].wrapping_add(&rhs[i]),
            ArithmeticOperator::Subtract => lhs[i].wrapping_sub(&rhs[i]),
            ArithmeticOperator::Multiply => lhs[i].wrapping_mul(&rhs[i]),
            ArithmeticOperator::Divide => {
                if rhs[i] == T::zero() {
                    panic!("Division by zero")
                } else {
                    lhs[i] / rhs[i]
                }
            }
            ArithmeticOperator::Remainder => {
                if rhs[i] == T::zero() {
                    panic!("Remainder by zero")
                } else {
                    lhs[i] % rhs[i]
                }
            }
            ArithmeticOperator::Power => lhs[i].pow(rhs[i].to_u32().unwrap_or(0)),
            ArithmeticOperator::FloorDiv => {
                if rhs[i] == T::zero() {
                    panic!("Floor division by zero")
                } else {
                    let d = lhs[i] / rhs[i];
                    let r = lhs[i] % rhs[i];
                    // If remainder is non-zero and signs differ, floor toward -inf
                    if r != T::zero() && (lhs[i] ^ rhs[i]) < T::zero() {
                        d - T::one()
                    } else {
                        d
                    }
                }
            }
        };
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
    for i in 0..n {
        let valid = unsafe { mask.get_unchecked(i) };
        if valid {
            let (result, final_valid) = match op {
                ArithmeticOperator::Add => (lhs[i].wrapping_add(&rhs[i]), true),
                ArithmeticOperator::Subtract => (lhs[i].wrapping_sub(&rhs[i]), true),
                ArithmeticOperator::Multiply => (lhs[i].wrapping_mul(&rhs[i]), true),
                ArithmeticOperator::Divide => {
                    if rhs[i] == T::zero() {
                        (T::zero(), false) // division by zero -> invalid
                    } else {
                        (lhs[i] / rhs[i], true)
                    }
                }
                ArithmeticOperator::Remainder => {
                    if rhs[i] == T::zero() {
                        (T::zero(), false) // remainder by zero -> invalid
                    } else {
                        (lhs[i] % rhs[i], true)
                    }
                }
                ArithmeticOperator::Power => (lhs[i].pow(rhs[i].to_u32().unwrap_or(0)), true),
                ArithmeticOperator::FloorDiv => {
                    if rhs[i] == T::zero() {
                        (T::zero(), false)
                    } else {
                        let d = lhs[i] / rhs[i];
                        let r = lhs[i] % rhs[i];
                        if r != T::zero() && (lhs[i] ^ rhs[i]) < T::zero() {
                            (d - T::one(), true)
                        } else {
                            (d, true)
                        }
                    }
                }
            };
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
}

/// Scalar floating-point arithmetic kernel for dense arrays (no nulls).
/// Division by zero produces Inf/NaN rather than panicking.
/// Power operations use logarithmic exponentiation: `exp(b * ln(a))`.
#[inline(always)]
pub fn float_dense_body_std<T: Float>(op: ArithmeticOperator, lhs: &[T], rhs: &[T], out: &mut [T]) {
    let n = lhs.len();
    for i in 0..n {
        out[i] = match op {
            ArithmeticOperator::Add => lhs[i] + rhs[i],
            ArithmeticOperator::Subtract => lhs[i] - rhs[i],
            ArithmeticOperator::Multiply => lhs[i] * rhs[i],
            ArithmeticOperator::Divide => lhs[i] / rhs[i],
            ArithmeticOperator::Remainder => lhs[i] % rhs[i],
            ArithmeticOperator::Power => (rhs[i] * lhs[i].ln()).exp(),
            ArithmeticOperator::FloorDiv => (lhs[i] / rhs[i]).floor(),
        };
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
    for i in 0..n {
        let valid = unsafe { mask.get_unchecked(i) };
        if valid {
            out[i] = match op {
                ArithmeticOperator::Add => lhs[i] + rhs[i],
                ArithmeticOperator::Subtract => lhs[i] - rhs[i],
                ArithmeticOperator::Multiply => lhs[i] * rhs[i],
                ArithmeticOperator::Divide => lhs[i] / rhs[i],
                ArithmeticOperator::Remainder => lhs[i] % rhs[i],
                ArithmeticOperator::Power => (rhs[i] * lhs[i].ln()).exp(),
                ArithmeticOperator::FloorDiv => (lhs[i] / rhs[i]).floor(),
            };
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
