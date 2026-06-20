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

//! # **Internal module**
//!
//! Vtable implementations for SharedBuffer memory management backends.
//!
//! Defines static function tables for different buffer ownership models:
//! - `STATIC_VT`: Read-only static data (no reference counting)
//! - `PROMO_*_VT`: Lazy heap promotion variants for Vec<u8> and Vec64<u8>

use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::{ptr, slice};

use crate::Vec64;
use crate::structs::shared_buffer::SharedBuffer;
use crate::structs::shared_buffer::internal::pvec::{
    PromotableVec, promo_clone, promo_drop, promo_is_unique, promo64_clone, promo64_drop,
};

/// Function table for `SharedBuffer` backend-specific memory operations.
///
/// Static vtable enabling `SharedBuffer` to handle heterogeneous buffer sources
/// (Vec, Vec64, MMAP, static) through dynamic dispatch whilst remaining a small value type.
///
/// Enables zero-copy reference counting, safe deallocation, and optimised extraction
/// to owned types with backend-specific paths (zero-copy when unique, copy otherwise).
///
/// ### Fields
/// - `clone`: Increments reference count, returns new `SharedBuffer` sharing data
/// - `drop`: Decrements reference count, destroys storage if unique
/// - `is_unique`: Returns `true` if exclusively owned (refcount=1)
/// - `to_vec`: Extracts `Vec<u8>` (zero-copy if unique, otherwise copies)
/// - `to_vec64`: Like `to_vec` but produces SIMD-aligned `Vec64<u8>`
/// - `memfd_fd`: Returns the memfd file descriptor if this is a memfd-backed buffer
///
/// Each backend defines static `Vtable` instances (e.g., `OWNED_VT`, `STATIC_VT`)
/// assigned at buffer creation.
pub(crate) struct Vtable {
    pub(crate) clone: unsafe fn(&AtomicPtr<()>, *const u8, usize) -> SharedBuffer,
    pub(crate) drop: unsafe fn(&mut AtomicPtr<()>, *const u8, usize),
    pub(crate) is_unique: unsafe fn(&AtomicPtr<()>) -> bool,
    pub(crate) to_vec: unsafe fn(&AtomicPtr<()>, *const u8, usize) -> Vec<u8>,
    pub(crate) to_vec64: unsafe fn(&AtomicPtr<()>, *const u8, usize) -> Vec64<u8>,
    /// Returns the memfd file descriptor if this buffer is backed by a memfd.
    /// Returns None for non-memfd backends.
    #[cfg(all(target_os = "linux", feature = "memfd"))]
    pub(crate) memfd_fd: unsafe fn(&AtomicPtr<()>) -> Option<i32>,
}

/// Vtable for static/const data requiring no reference counting.
pub(crate) static STATIC_VT: Vtable = Vtable {
    clone: |_, p, l| SharedBuffer {
        ptr: p,
        len: l,
        data: AtomicPtr::new(ptr::null_mut()),
        vtable: &STATIC_VT,
    },
    drop: |_, _, _| {},
    is_unique: |_| true,
    to_vec: |_, p, l| unsafe { slice::from_raw_parts(p, l) }.to_vec(),
    to_vec64: |_, p, l| {
        let mut v = Vec64::with_capacity(l);
        unsafe {
            v.extend_from_slice(slice::from_raw_parts(p, l));
        }
        v
    },
    #[cfg(all(target_os = "linux", feature = "memfd"))]
    memfd_fd: |_| None,
};

/// Vtable for Vec<u8> with lazy heap promotion (even variant).
pub(crate) static PROMO_EVEN_VT: Vtable = Vtable {
    clone: promo_clone,
    drop: promo_drop,
    is_unique: |h| promo_is_unique::<Vec<u8>>(h),
    to_vec: |h, p, l| {
        if promo_is_unique::<Vec<u8>>(h) {
            let raw = h.swap(ptr::null_mut(), Ordering::AcqRel);
            if !raw.is_null() {
                return unsafe { Box::from_raw(raw as *mut PromotableVec<Vec<u8>>).inner };
            }
        }
        unsafe { slice::from_raw_parts(p, l) }.to_vec()
    },
    to_vec64: |_, p, l| {
        let mut v = Vec64::with_capacity(l);
        unsafe {
            v.extend_from_slice(slice::from_raw_parts(p, l));
        }
        v
    },
    #[cfg(all(target_os = "linux", feature = "memfd"))]
    memfd_fd: |_| None,
};

/// Vtable for Vec<u8> with lazy heap promotion (odd variant).
pub(crate) static PROMO_ODD_VT: Vtable = Vtable {
    #[cfg(all(target_os = "linux", feature = "memfd"))]
    memfd_fd: |_| None,
    ..PROMO_EVEN_VT
};

/// Vtable for Vec64<u8> with lazy heap promotion (even variant).
pub(crate) static PROMO64_EVEN_VT: Vtable = Vtable {
    clone: promo64_clone,
    drop: promo64_drop,
    is_unique: |h| promo_is_unique::<Vec64<u8>>(h),
    to_vec: |h, p, l| {
        if promo_is_unique::<Vec64<u8>>(h) {
            let raw = h.swap(ptr::null_mut(), Ordering::AcqRel);
            if !raw.is_null() {
                return ManuallyDrop::new(unsafe {
                    Box::from_raw(raw as *mut PromotableVec<Vec64<u8>>).inner
                })
                .to_vec();
            }
        }
        unsafe { slice::from_raw_parts(p, l) }.to_vec()
    },
    to_vec64: |h, p, l| {
        if promo_is_unique::<Vec64<u8>>(h) {
            let raw = h.swap(ptr::null_mut(), Ordering::AcqRel);
            if !raw.is_null() {
                return unsafe { Box::from_raw(raw as *mut PromotableVec<Vec64<u8>>).inner };
            }
        }
        let mut v = Vec64::with_capacity(l);
        unsafe {
            v.extend_from_slice(slice::from_raw_parts(p, l));
        }
        v
    },
    #[cfg(all(target_os = "linux", feature = "memfd"))]
    memfd_fd: |_| None,
};

/// Vtable for Vec64<u8> with lazy heap promotion (odd variant).
pub(crate) static PROMO64_ODD_VT: Vtable = Vtable {
    #[cfg(all(target_os = "linux", feature = "memfd"))]
    memfd_fd: |_| None,
    ..PROMO64_EVEN_VT
};
