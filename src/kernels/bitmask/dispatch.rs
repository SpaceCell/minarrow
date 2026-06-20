// Copyright 2025-2026 Peter Garfield Bower. All Rights Reserved.
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

//! # **Bitmask Dispatch Module** - *Compile-Time SIMD/Scalar Selection for Bitmask Operations*
//!
//! Dispatcher that selects between SIMD and scalar implementations
//! at compile time based on feature flags and target architecture capabilities.
//!
//! Prefer this unless you want to access the underlying kernel functions directly.

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use crate::enums::operators::{LogicalOperator, UnaryOperator};
use crate::{Bitmask, BitmaskVT};

// --- Binary/Unary Bitmask Operations ---

/// Performs a binary logical operation (AND, OR, XOR) on two bitmask windows with automatic SIMD/scalar dispatch.
///
/// Executes the specified logical operation element-wise across two bitmask windows, producing a new bitmask
/// containing the results. The implementation is automatically selected based on compile-time feature flags.
///
/// # Parameters
/// - `lhs`: Left-hand side bitmask window as `(mask, offset, length)` tuple
/// - `rhs`: Right-hand side bitmask window as `(mask, offset, length)` tuple  
/// - `op`: Logical operation to perform (AND, OR, XOR)
///
/// # Returns
/// A new `Bitmask` containing the element-wise results of the logical operation.
///
/// # Performance Notes
/// - SIMD path processes multiple u64 words simultaneously for improved throughput
/// - Scalar path provides universal compatibility with word-level optimisations
/// - Output bitmask automatically handles trailing bit masking for correctness
#[inline(always)]
pub fn bitmask_binop(lhs: BitmaskVT<'_>, rhs: BitmaskVT<'_>, op: LogicalOperator) -> Bitmask {
    #[cfg(feature = "simd")]
    {
        crate::kernels::bitmask::simd::bitmask_binop_simd::<W8>(lhs, rhs, op)
    }
    #[cfg(not(feature = "simd"))]
    {
        crate::kernels::bitmask::std::bitmask_binop_std(lhs, rhs, op)
    }
}

/// Performs a unary operation (`NOT`) on a bitmask window.
#[inline(always)]
pub fn bitmask_unop(src: BitmaskVT<'_>, op: UnaryOperator) -> Bitmask {
    #[cfg(feature = "simd")]
    {
        crate::kernels::bitmask::simd::bitmask_unop_simd::<W8>(src, op)
    }
    #[cfg(not(feature = "simd"))]
    {
        crate::kernels::bitmask::std::bitmask_unop_std(src, op)
    }
}

// --- Entry Points (Standard Logical Operations) ---

/// Computes the element-wise bitwise AND of two bitmask windows for intersection operations.
///
/// Performs logical AND across corresponding bits in two bitmask windows, commonly used for
/// combining null masks in nullable array operations. The result bit is set only when both
/// input bits are set.
///
/// # Parameters
/// - `lhs`: Left-hand side bitmask window as `(mask, offset, length)` tuple
/// - `rhs`: Right-hand side bitmask window as `(mask, offset, length)` tuple
///
/// # Returns
/// A new `Bitmask` where each bit represents `lhs[i] AND rhs[i]`.
///
/// # Usage
/// ```rust,ignore
/// // Combine validity masks from two nullable arrays
/// let combined_validity = and_masks(
///     (&array_a.null_mask, 0, array_a.len()),
///     (&array_b.null_mask, 0, array_b.len())
/// );
/// // Result contains valid elements only where both arrays are non-null
/// ```
#[inline(always)]
pub fn and_masks(lhs: BitmaskVT<'_>, rhs: BitmaskVT<'_>) -> Bitmask {
    #[cfg(feature = "simd")]
    {
        crate::kernels::bitmask::simd::and_masks_simd::<W8>(lhs, rhs)
    }
    #[cfg(not(feature = "simd"))]
    {
        crate::kernels::bitmask::std::and_masks(lhs, rhs)
    }
}

/// Computes the element-wise bitwise OR of two bitmask windows.
#[inline(always)]
pub fn or_masks(lhs: BitmaskVT<'_>, rhs: BitmaskVT<'_>) -> Bitmask {
    #[cfg(feature = "simd")]
    {
        crate::kernels::bitmask::simd::or_masks_simd::<W8>(lhs, rhs)
    }
    #[cfg(not(feature = "simd"))]
    {
        crate::kernels::bitmask::std::or_masks(lhs, rhs)
    }
}

/// Computes the element-wise bitwise XOR of two bitmask windows.
#[inline(always)]
pub fn xor_masks(lhs: BitmaskVT<'_>, rhs: BitmaskVT<'_>) -> Bitmask {
    #[cfg(feature = "simd")]
    {
        crate::kernels::bitmask::simd::xor_masks_simd::<W8>(lhs, rhs)
    }
    #[cfg(not(feature = "simd"))]
    {
        crate::kernels::bitmask::std::xor_masks(lhs, rhs)
    }
}

/// Computes the element-wise bitwise NOT of a bitmask window.
#[inline(always)]
pub fn not_mask(src: BitmaskVT<'_>) -> Bitmask {
    #[cfg(feature = "simd")]
    {
        crate::kernels::bitmask::simd::not_mask_simd::<W8>(src)
    }
    #[cfg(not(feature = "simd"))]
    {
        crate::kernels::bitmask::std::not_mask(src)
    }
}

// --- Set Logic ---

/// Performs bitwise inclusion: output bit is true if lhs bit is present in rhs bit-set.
#[inline(always)]
pub fn in_mask(lhs: BitmaskVT<'_>, rhs: BitmaskVT<'_>) -> Bitmask {
    #[cfg(feature = "simd")]
    {
        crate::kernels::bitmask::simd::in_mask_simd::<W8>(lhs, rhs)
    }
    #[cfg(not(feature = "simd"))]
    {
        crate::kernels::bitmask::std::in_mask(lhs, rhs)
    }
}

/// Performs bitwise exclusion: output bit is true if lhs bit is not present in rhs bit-set.
#[inline(always)]
pub fn not_in_mask(lhs: BitmaskVT<'_>, rhs: BitmaskVT<'_>) -> Bitmask {
    #[cfg(feature = "simd")]
    {
        crate::kernels::bitmask::simd::not_in_mask_simd::<W8>(lhs, rhs)
    }
    #[cfg(not(feature = "simd"))]
    {
        crate::kernels::bitmask::std::not_in_mask(lhs, rhs)
    }
}

// --- Equality/Inequality Masks ---

/// Computes an equality mask: output bit is true if bits are equal at each position.
#[inline(always)]
pub fn eq_mask(a: BitmaskVT<'_>, b: BitmaskVT<'_>) -> Bitmask {
    #[cfg(feature = "simd")]
    {
        crate::kernels::bitmask::simd::eq_mask_simd::<W8>(a, b)
    }
    #[cfg(not(feature = "simd"))]
    {
        crate::kernels::bitmask::std::eq_mask(a, b)
    }
}

/// Computes an inequality mask: output bit is true if bits differ at each position.
#[inline(always)]
pub fn ne_mask(a: BitmaskVT<'_>, b: BitmaskVT<'_>) -> Bitmask {
    #[cfg(feature = "simd")]
    {
        crate::kernels::bitmask::simd::ne_mask_simd::<W8>(a, b)
    }
    #[cfg(not(feature = "simd"))]
    {
        crate::kernels::bitmask::std::ne_mask(a, b)
    }
}

/// Returns true if all bits are equal between two bitmask windows.
#[inline(always)]
pub fn all_eq(a: BitmaskVT<'_>, b: BitmaskVT<'_>) -> bool {
    #[cfg(feature = "simd")]
    {
        crate::kernels::bitmask::simd::all_eq_mask_simd::<W8>(a, b)
    }
    #[cfg(not(feature = "simd"))]
    {
        crate::kernels::bitmask::std::all_eq_mask(a, b)
    }
}

/// Returns true if all bits differ between two bitmask windows.
#[inline(always)]
pub fn all_ne(a: BitmaskVT<'_>, b: BitmaskVT<'_>) -> bool {
    #[cfg(feature = "simd")]
    {
        crate::kernels::bitmask::simd::all_ne_mask_simd::<W8>(a, b)
    }
    #[cfg(not(feature = "simd"))]
    {
        crate::kernels::bitmask::std::all_ne_mask(a, b)
    }
}

// --- Popcount ---

/// Returns the number of set bits (population count) in the bitmask window using fast SIMD reduction.
///
/// Counts the number of `1` bits in the specified bitmask window, commonly used to determine
/// the number of valid (non-null) elements in nullable arrays. The implementation uses
/// vectorised population count instructions for optimal performance.
///
/// # Parameters
/// - `m`: Bitmask window as `(mask, offset, length)` tuple
///
/// # Returns
/// The count of set bits in the specified window as a `usize`.
///
/// # Performance Characteristics
/// - SIMD path uses vectorised `popcount` instructions with SIMD reduction
/// - Automatically handles partial words and trailing bit masking
/// - O(n/64) complexity for word-aligned operations
///
/// # Usage
/// ```rust,ignore
/// // Determine if computation is worthwhile
/// let valid_count = popcount_mask((&validity_mask, 0, array.len()));
/// if valid_count == 0 {
///     // Skip expensive operations on all-null data
///     return Array::new_null(array.len());
/// }
/// // Proceed with computation on `valid_count` elements
/// ```
#[inline(always)]
pub fn popcount_mask(m: BitmaskVT<'_>) -> usize {
    #[cfg(feature = "simd")]
    {
        crate::kernels::bitmask::simd::popcount_mask_simd::<W8>(m)
    }
    #[cfg(not(feature = "simd"))]
    {
        crate::kernels::bitmask::std::popcount_mask(m)
    }
}

// --- All-True / All-False Checks ---

/// Returns true if all bits in the mask are set (1).
#[inline(always)]
pub fn all_true_mask(mask: &Bitmask) -> bool {
    #[cfg(feature = "simd")]
    {
        crate::kernels::bitmask::simd::all_true_mask_simd::<W8>(mask)
    }
    #[cfg(not(feature = "simd"))]
    {
        crate::kernels::bitmask::std::all_true_mask(mask)
    }
}

/// Returns true if all bits in the mask are unset (0).
#[inline(always)]
pub fn all_false_mask(mask: &Bitmask) -> bool {
    #[cfg(feature = "simd")]
    {
        crate::kernels::bitmask::simd::all_false_mask_simd::<W8>(mask)
    }
    #[cfg(not(feature = "simd"))]
    {
        crate::kernels::bitmask::std::all_false_mask(mask)
    }
}
