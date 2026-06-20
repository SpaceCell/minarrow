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

//! # **View Trait Module** - *Standardises Slicing and View Moves in Minarrow*
//!
//! Zero-copy array view abstractions for `MinArrow`.
//!
//! This module defines the [`View`] trait, which provides a unified interface
//! for creating lightweight, zero-copy “windows” into arrays without duplicating
//! their underlying buffers.  
//!
//! It supports three main access patterns:
//! - **Native slices** – direct `&[T]` or `&[u8]` access for fixed- and variable-width types.
//! - **ArrayView** – a typed, windowed view backed by `Arc`-cloned array data.
//! - **TupleView** – a minimal `(&Array, offset, length)` form for maximum performance.
//!
//! These views allow efficient subsetting, iteration, and type-specific access
//! (`.num()`, `.text()`, `.dt()`, `.bool()`), while preserving the semantics of
//! the original array type.
//!
//! Unlike Apache Arrow’s trait-based array references, `MinArrow` stores its arrays
//! in a concrete [`Array`] type already holding an `Arc` to its inner buffers.
//! This means `ArrayView` only needs to enforce logical offset/length constraints,
//! avoiding additional indirection or ref-counting overhead.
//!
//! Use these abstractions in pipelines, joins, and analytic operations where you
//! need read-only views over subsets of arrays without copying or reallocation.

use crate::{Array, ArrayV, Length, MaskedArray, Offset};

/// # View trait
///
/// Zero-copy, windowed access to array data with multiple abstraction levels.
///
/// ## Description
/// The [`View`] trait provides a unified interface for creating logical subviews
/// into arrays without duplicating their underlying buffers. It is implemented by
/// all [`MaskedArray`] types and supports three main access patterns:
///
/// - **Native slice access** – direct `&[T]` or `&[u8]` for fixed- and variable-width data.
/// - **ArrayView** – an `Arc`-cloned, type-aware view with safe windowing and typed accessors.
/// - **TupleView** – a minimal `(&Array, offset, length)` form for maximum performance.
///
/// ## Purpose
/// This trait indirectly supports pipelines, joins, and analytics that need read-only
/// subsets of arrays without the cost of copying or reallocating.
///
/// ### Ownership Semantics
/// - When called on an `Arc`-wrapped array (e.g., [`Array`]), `.view()` consumes the `Arc`.
///   Clone the `Arc` first if you need to retain the original.
/// - When called on a direct array variant, `.view()` consumes ownership.
///   Wrap in `Array` first if you need continued access.
///
/// ### Behaviour
/// - Views enforce logical offset/length constraints.
/// - Access methods such as `.num()`, `.text()`, `.dt()`, `.bool()` return typed view variants.
/// - Always zero-copy: only offset and length metadata change, not the backing buffers.
///
/// ### Compared to Apache Arrow
/// Arrow arrays are lightweight views over reference-counted buffers
/// *(the view + buffers are separate types)*. In **MinArrow**, an [`Array`]
/// already owns an `Arc` to its inner buffers; a `View` (e.g., [`ArrayV`])
/// simply adds window (offset/length) metadata on top of that same Arc’d data.
/// In both designs there’s effectively one layer of ref counting; the key
/// difference is that **Apache Arrow** bakes “view-ness” into the array type itself,
/// whereas **MinArrow** keeps arrays concrete and layers windowing as a separate view.
pub trait View: MaskedArray + Into<Array> + Clone
where
    <Self as MaskedArray>::Container: AsRef<[Self::BufferT]>,
{
    /// The fixed-width buffer type (e.g. `u8`, `f32`, `bool`, etc.)
    type BufferT: Default + PartialEq + Clone + Copy + Sized;

    /// Returns the whole array buffer as a &[T] slice.
    fn as_slice(&self) -> &[Self::BufferT] {
        self.data().as_ref()
    }

    /// Slices the data values from offset to offset + length.
    ///
    /// # Parameters
    /// - `offset`: Starting index.
    /// - `len`: Number of elements.
    ///
    /// # Returns
    /// A native slice of the data values.
    fn slice(&self, offset: usize, len: usize) -> (&[Self::BufferT], Offset, Length) {
        (&self.data().as_ref()[offset..offset + len], offset, len)
    }

    /// Returns a zero-copy, windowed view (`ArrayView`) into this array.
    ///
    /// ## Ownership Semantics
    /// - For `Arc`-wrapped arrays (e.g., `Array`), this method consumes the `Arc`.
    ///   If you need to retain access to the original array after calling `view`,
    ///   clone the `Arc` at the call site (cheap pointer clone).
    /// - For direct array variants (e.g., `IntegerArray<u64>`), calling `view` consumes ownership.
    ///   If continued access to the original is required, promote the variant into an `Arc` (or `Array`) first.
    ///
    /// ## Behaviour
    /// - The returned `ArrayView` enforces the window (`offset`, `length`) and provides safe, zero-copy access.
    /// - The backing array remains accessible via `&Array` within the view.
    /// - Type-specific accessors (`.num()`, `.str()`, `.bool()`, `.dt()`) are available on `ArrayView` for ergonomic, typed access.
    ///
    /// # Implementation Notes
    /// - Unlike `Apache Arrow`, where arrays are trait-based views over reference-counted
    ///   buffers, `MinArrow` uses a concrete `Array` struct which already holds a *zero-copy*
    ///   `Arc` reference to its inner array data.
    /// - `ArrayView` therefore provides a *windowed zero-copy view* into a specific region of
    ///   this array.
    ///
    /// # Parameters
    /// - `offset`: The logical start index for the window.
    /// - `len`:    The logical length of the window.
    ///
    /// # Returns
    /// An `ArrayView` representing the specified logical window.
    fn view(self, offset: usize, len: usize) -> ArrayV {
        ArrayV::new(self.into(), offset, len)
    }

    /// Returns a tuple view of (&Array, offset, length).
    ///
    /// # Parameters
    /// - `offset`: Starting index.
    /// - `length`: Number of elements.
    ///
    /// # Returns
    /// Tuple containing array reference, offset, and length.
    fn view_tuple(self, offset: usize, len: usize) -> (Array, usize, usize) {
        self.view(offset, len).as_tuple()
    }
}
