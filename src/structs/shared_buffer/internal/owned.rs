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

//! # **Internal module**
//!
//! Static vtable implementations for SharedBuffer backends.
//!
//! Provides function tables for different buffer ownership models:
//! - `OWNED_VT`: Reference-counted arbitrary backing types (Vec, Vec64, custom containers)
//! - `STATIC_VT`: Read-only static/const data (zero refcounting)
//! - `PROMO_*_VT`: Lazy promotion from stack to heap allocation
//!
//! Comparable to Tokio's `Bytes` implementation but avoids external dependencies
//! and ensures 64-byte SIMD alignment support.

use core::slice;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use crate::Vec64;
use crate::structs::shared_buffer::SharedBuffer;
use crate::structs::shared_buffer::internal::vtable::Vtable;

/// Reference-counted wrapper for arbitrary backing storage types.
///
/// Enables SharedBuffer to manage any container implementing AsRef<[u8]>
/// with atomic reference counting for safe sharing.
///
/// The `drop_fn` field stores a type-erased destructor so that
/// `owned_drop` can properly clean up the concrete `Owned<T>` without
/// knowing T at the vtable level.
#[repr(C)]
pub(crate) struct Owned<T: AsRef<[u8]> + Send + Sync + 'static> {
    pub(crate) ref_cnt: AtomicUsize,
    pub(crate) drop_fn: unsafe fn(*mut ()),
    pub(crate) owner: T,
}

/// Clones owned buffer by incrementing reference count.
unsafe fn owned_clone(h: &AtomicPtr<()>, p: *const u8, l: usize) -> SharedBuffer {
    let raw = h.load(Ordering::Acquire);
    assert!(!raw.is_null());
    // SAFETY: ref_cnt is first field, layout #[repr(C)]
    let ref_cnt = unsafe { &*(raw as *const AtomicUsize) };
    ref_cnt.fetch_add(1, Ordering::Relaxed);
    SharedBuffer {
        ptr: p,
        len: l,
        data: AtomicPtr::new(raw),
        vtable: &OWNED_VT,
    }
}

/// Decrements reference count, deallocating if last reference.
///
/// Reads the type-erased destructor stored in the `Owned` header
/// to properly drop the concrete `Owned<T>` and run T's destructor.
unsafe fn owned_drop(h: &mut AtomicPtr<()>, _p: *const u8, _l: usize) {
    let raw = h.load(Ordering::Acquire);
    if raw.is_null() {
        return;
    }
    let ref_cnt = unsafe { &*(raw as *const AtomicUsize) };
    if ref_cnt.fetch_sub(1, Ordering::AcqRel) == 1 {
        // Read the drop function stored after ref_cnt in the Owned header.
        // Owned is #[repr(C)] with ref_cnt first, drop_fn second, so the
        // layout is the same for all Owned<T>.
        unsafe {
            let drop_fn_ptr = (raw as *const u8).add(std::mem::size_of::<AtomicUsize>())
                as *const unsafe fn(*mut ());
            let drop_fn = *drop_fn_ptr;
            drop_fn(raw);
        }
    }
}

/// Vtable for reference-counted owned buffers (Vec, Vec64, custom containers).
pub(crate) static OWNED_VT: Vtable = Vtable {
    clone: owned_clone,
    drop: owned_drop,
    is_unique: |h| {
        let raw = h.load(Ordering::Acquire);
        if raw.is_null() {
            return false;
        }
        let ref_cnt = unsafe { &*(raw as *const AtomicUsize) };
        ref_cnt.load(Ordering::Acquire) == 1
    },
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
