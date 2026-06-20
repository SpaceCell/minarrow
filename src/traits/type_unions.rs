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

//! **Numeric Traits Module** - *Contains *num-trait* wrappers that simplify Type Signature Ergonomics*
//!
//! Numeric trait bounds used throughout Minarrow.
//!
//! This module defines small, crate-specific trait aliases over `num_traits`
//! to keep generic signatures concise and consistent across arrays/kernels:
//! - `Float`: `f32`, `f64` (wrapper over `num_traits::Float`).
//! - `Integer`: integer types that also support lossless casts to/from `usize`
//!   (implemented via `impl_usize_conversions!`), useful for offsets/indices.
//! - `Numeric`: all supported number types (`ints` + `floats`) for math kernels.
//! - `Primitive`: scalars (`ints`, `floats`, `bool`) used in low-level paths.
//!
//! Use these in APIs like `fn foo<T: Numeric>()` or `fn bar<I: Integer>()` to
//! express intent and rely on a single source of truth for allowed scalar types.

use std::fmt::Debug;

use num_traits::{Float as NumFloat, Num, NumCast, PrimInt, ToPrimitive};

use crate::impl_usize_conversions;

/// Trait for types valid as float elements in columnar arrays.
///
/// Useful when specifying `my_fn::<T: Float>() {}`.
///
/// Extends and constrains the *num-traits* `Float` implementation to fit the crate's type universe.
pub trait Float: NumFloat + Copy + Default + ToPrimitive + PartialEq + 'static {}
impl Float for f32 {}
impl Float for f64 {}

/// Trait for types valid as integer elements in columnar arrays.
pub trait Integer:
    PrimInt + TryFrom<usize> + Default + Debug + ToPrimitive + std::hash::Hash + 'static
{
    /// Lossless cast to `usize`
    fn to_usize(self) -> usize;

    /// Lossless cast from `usize`
    fn from_usize(v: usize) -> Self;
}

impl_usize_conversions!(u8, u16, u32, u64, i8, i16, i32, i64);

/// Trait for types valid as numerical.
///
/// Useful when specifying `my_fn::<T: Numeric>() {}`.
///
/// Extends and constrains the *num-traits* `Num` implementation to fit the crate's type universe.
pub trait Numeric: Num + NumCast + Copy + Default + ToPrimitive + PartialEq + 'static {}
impl Numeric for f32 {}
impl Numeric for f64 {}
impl Numeric for i8 {}
impl Numeric for i16 {}
impl Numeric for i32 {}
impl Numeric for i64 {}
impl Numeric for u8 {}
impl Numeric for u16 {}
impl Numeric for u32 {}
impl Numeric for u64 {}

/// Trait for types valid as primitive, i.e.., floats, integers, and booleans.
///
/// Useful when specifying `my_fn::<T: Primitive>() {}`.
pub trait Primitive: Copy + Default + PartialEq + 'static {}
impl Primitive for f32 {}
impl Primitive for f64 {}
impl Primitive for i8 {}
impl Primitive for i16 {}
impl Primitive for i32 {}
impl Primitive for i64 {}
impl Primitive for u8 {}
impl Primitive for u16 {}
impl Primitive for u32 {}
impl Primitive for u64 {}
impl Primitive for bool {}
