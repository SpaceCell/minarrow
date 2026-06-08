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

//! # **OutputBuffer**
//!
//! `OutputBuffer<'a, T>` is a typed mutable view into caller-owned storage.
//! Kernels that accept a mutable buffer write their result into the
//! slice directly, returning `Ok(())` instead of an owned buffer.
//!
//! This is an optional pre-Arrow array step that can be useful
//! in performance critical scenarios when minimising unnecessary
//! allocations across multiple array chunks.
//!
//! ## Element type
//!
//! `T: Primitive` covers `f32`, `f64`, signed and unsigned integers, and
//! `bool`.

use crate::structs::bitmask::Bitmask;
use crate::traits::type_unions::Primitive;

/// Typed mutable output buffer.
///
/// `data` is the typed output slice. `mask` is the optional output null
/// bitmask; populated when the kernel needs to write per-row validity.
pub struct OutputBuffer<'a, T: Primitive> {
    pub data: &'a mut [T],
    pub mask: Option<&'a mut Bitmask>,
}

impl<'a, T: Primitive> OutputBuffer<'a, T> {
    /// Construct an OutputBuffer from a typed slice and an optional mask.
    #[inline]
    pub fn new(data: &'a mut [T], mask: Option<&'a mut Bitmask>) -> Self {
        Self { data, mask }
    }

    /// Length in rows of the data slice.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True if the data slice has zero rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
