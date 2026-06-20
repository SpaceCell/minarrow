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

//! # **IntegerArray Module**- *Mid-Level, Inner Typed Integer Array*
//!
//! Arrow-compatible, SIMD-aligned integer array optimised for analytical workloads.
//!
//! ## Overview
//! - Logical type: fixed-width signed/unsigned integers (`T: Integer`).
//! - Physical storage: `Buffer<T>` (backed by `Vec64<T>` for 64-byte alignment) plus
//!   optional bit-packed validity mask (`Bitmask`).
//! - Usable standalone or as the numeric arm of higher-level enums (`NumericArray`, `Array`).
//! - Zero-copy friendly and integrates with Arrow FFI.
//!
//! ## Features
//! - **Construction** from slices, `Vec`, or `Vec64`, with optional null mask.
//! - **Mutation**: push/set, bulk null insertion, resize.
//! - **Iteration**: safe and (optionally) parallel, via the `MaskedArray` trait surface.
//! - **Null handling**: Arrow-style bitmask (`1 = valid`, `0 = null`) with length validation.
//!
//! ## Usage Tips
//! Prefer function signatures with `&[T]` or generic `T: Integer` where possible to keep
//! callsites simple while remaining zero-copy compatible with `IntegerArray`.
//!
//! ## Performance Benchmarks
//! Benchmarks under `examples/hotloop_benchmark_std` compare construction + sum over 1000 elems,
//! averaged across 1000 runs on a 2024 Intel(R) Core(TM) Ultra 7 155H (22 logical CPUs, 32 GB RAM).
//!
//! **Averaged Results (1000 runs, size = 1000)**
//! - `Vec<i64>` (raw):                        avg = 0.131 µs
//! - `IntegerArray` (minarrow, direct):       avg = 0.130 µs
//! - `Int64Array` (arrow-rs, struct):         avg = 0.327 µs
//! - `IntegerArray` (minarrow, enum):         avg = 2.494 µs
//! - `Int64Array` (arrow-rs, dynamic trait):  avg = 3.283 µs
//!
//! **Build note:** use `--release` for representative numbers; debug builds distort results.
//!
//! **Takeaways**
//! - Direct `IntegerArray` matches raw `Vec<i64>` while adding SIMD alignment + Arrow interop.
//! - Enum dispatch adds modest overhead but remains faster than dynamic dispatch (arrow-rs).
//! - For (extreme) latency-sensitive paths, prefer direct `IntegerArray`; otherwise differences are minor
//!   vs. real compute/system overhead.

use std::fmt::{Display, Formatter};

use crate::enums::shape_dim::ShapeDim;
use crate::traits::concatenate::Concatenate;
use crate::traits::print::MAX_PREVIEW;
use crate::traits::shape::Shape;
use crate::traits::type_unions::Integer;
use crate::{
    Bitmask, Buffer, Length, MaskedArray, Offset, impl_arc_masked_array, impl_array_ref_deref,
    impl_from_vec_primitive, impl_masked_array, impl_numeric_array_constructors,
};
use vec64::Vec64;

/// # IntegerArray
///
/// Arrow-compatible, 64-byte aligned integer array with optional null mask.
///
/// ## Role
/// - Many will prefer the higher level `Array` type, which dispatches to this when
/// necessary.
/// - Can be used as a standalone array or as the numeric arm of `NumericArray` / `Array`.
///
/// ## Description
/// - Stores fixed-width integer values in a contiguous `Buffer<T>` (`Vec64<T>` under the hood).
/// - Optional Arrow-style validity bitmap (`1 = valid`, `0 = null`) via `Bitmask`.
/// - Implements [`MaskedArray`] for consistent nullable array behavior and interop with
///   higher-level containers/enums.
/// - Can be used as a standalone numeric buffer or as the numeric arm of `NumericArray` / `Array`.
///
/// ### Fields
/// - `data`: backing buffer of integer values (`Buffer<T>`).
/// - `null_mask`: optional bit-packed validity bitmap.
///
/// ## Example
/// ```rust
/// use minarrow::{IntegerArray, MaskedArray};
///
/// // Dense, no nulls
/// let arr = IntegerArray::<i64>::from_slice(&[1, 2, 3, 4]);
/// assert_eq!(arr.len(), 4);
/// assert_eq!(arr.get(2), Some(3));
///
/// // With nulls
/// let mut arr = IntegerArray::<i32>::with_capacity(3, true);
/// arr.push(10);
/// arr.push_null();
/// arr.push(30);
/// assert_eq!(arr.get(1), None);
/// assert_eq!(arr.null_count(), 1);
/// ```
#[derive(PartialEq, Clone, Debug, Default)]
pub struct IntegerArray<T> {
    /// Backing buffer for values (Arrow-compatible).
    pub data: Buffer<T>,
    /// Optional null mask (bit-packed; 1=valid, 0=null).
    pub null_mask: Option<Bitmask>,
}

impl_numeric_array_constructors!(IntegerArray, Integer);
impl_masked_array!(IntegerArray, Integer, Buffer<T>, T);
impl_from_vec_primitive!(IntegerArray);
impl_array_ref_deref!(IntegerArray<T>);
impl_arc_masked_array!(
    Inner = IntegerArray<T>,
    T = T,
    Container = Buffer<T>,
    LogicalType = T,
    CopyType = T,
    BufferT = T,
    Variant = NumericArray,
    Bound = Integer,
);

impl<T> Display for IntegerArray<T>
where
    T: Integer + Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let len = self.len();
        let nulls = self.null_count();

        writeln!(
            f,
            "IntegerArray [{} values] (dtype: int, nulls: {})",
            len, nulls
        )?;

        write!(f, "[")?;

        for i in 0..usize::min(len, MAX_PREVIEW) {
            if i > 0 {
                write!(f, ", ")?;
            }

            match self.get(i) {
                Some(val) => write!(f, "{}", val)?,
                None => write!(f, "null")?,
            }
        }

        if len > MAX_PREVIEW {
            write!(f, ", … ({} total)", len)?;
        }

        write!(f, "]")
    }
}

impl<T: Integer> Shape for IntegerArray<T> {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank1(self.len())
    }
}

#[cfg(feature = "chunked")]
impl<'a, T: Integer> crate::traits::consolidate::Consolidate
    for Vec<crate::aliases::IntegerAVT<'a, T>>
{
    type Output = IntegerArray<T>;

    /// Consolidate a vector of `(IntegerArray<T>, offset, len)` view
    /// tuples into one contiguous `IntegerArray<T>`. Reads each window
    /// directly from the source buffer - no intermediate copy.
    fn consolidate(self) -> IntegerArray<T> {
        use crate::structs::bitmask::Bitmask;
        use crate::traits::consolidate::extend_null_mask;
        use crate::traits::masked_array::MaskedArray;

        assert!(!self.is_empty(), "consolidate() called on empty Vec<IntegerAVT>");

        let total_len: usize = self.iter().map(|(_, _, len)| *len).sum();
        let has_nulls = self.iter().any(|(arr, _, _)| arr.null_mask.is_some());

        let mut result = IntegerArray::<T>::with_capacity(total_len, has_nulls);
        let mut result_mask: Option<Bitmask> = if has_nulls {
            Some(Bitmask::default())
        } else {
            None
        };
        let mut current_len = 0;

        for (arr, offset, len) in &self {
            let data: &[T] = &arr.data[*offset..*offset + *len];
            result.data.extend_from_slice(data);
            extend_null_mask(
                &mut result_mask,
                current_len,
                arr.null_mask(),
                *offset,
                *len,
            );
            current_len += *len;
        }

        if let Some(mask) = result_mask {
            result.set_null_mask(Some(mask));
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structs::bitmask::Bitmask;
    use crate::traits::masked_array::MaskedArray;
    use crate::vec64;

    #[test]
    fn test_new_and_with_capacity() {
        let arr = IntegerArray::<i32>::default();
        assert_eq!(arr.data.len(), 0);
        assert!(arr.null_mask.is_none());

        let mut arr = IntegerArray::<u16>::with_capacity(8, true);
        assert_eq!(arr.data.len(), 0);
        assert!(arr.data.capacity() >= 8);
        assert!(arr.null_mask.is_some());

        // Reserved null-mask slots must default to valid (1).
        // null_count() == 0 on a fresh, empty array.
        assert_eq!(arr.null_count(), 0);

        // After non-null pushes, null_count remains 0 - the pre-allocated
        // tail bits beyond len() must not poison the count.
        arr.push(10);
        arr.push(20);
        assert_eq!(arr.null_count(), 0);

        // A pushed null is the only thing that should bump the count.
        arr.push_null();
        assert_eq!(arr.null_count(), 1);
    }

    #[test]
    fn test_push_and_get_no_null_mask() {
        let mut arr = IntegerArray::<i64>::with_capacity(4, false);
        arr.push(123);
        arr.push(-456);
        assert_eq!(arr.data, vec64![123, -456]);
        assert_eq!(arr.get(0), Some(123));
        assert_eq!(arr.get(1), Some(-456));
        assert!(!arr.is_null(0));
        assert!(!arr.is_null(1));
    }

    #[test]
    fn test_push_and_get_with_null_mask() {
        let mut arr = IntegerArray::<u8>::with_capacity(3, true);
        arr.push(42);
        arr.push_null();
        arr.push(7);
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.get(0), Some(42));
        assert_eq!(arr.get(1), None);
        assert_eq!(arr.get(2), Some(7));
        assert!(!arr.is_null(0));
        assert!(arr.is_null(1));
        assert!(!arr.is_null(2));
    }

    #[test]
    fn test_push_null_auto_mask() {
        let mut arr = IntegerArray::<i16>::default();
        arr.push_null();
        assert_eq!(arr.data, vec64![0]);
        assert!(arr.is_null(0));
        assert!(arr.null_mask.is_some());
    }

    #[test]
    fn test_set_and_set_null() {
        let mut arr = IntegerArray::<u32>::with_capacity(3, true);
        arr.push(100);
        arr.push(200);
        arr.push(300);
        arr.set(1, 222);
        assert_eq!(arr.get(1), Some(222));
        arr.set_null(2);
        assert_eq!(arr.get(2), None);
        assert!(arr.is_null(2));
    }

    #[test]
    fn test_trait_masked_array() {
        let mut arr = IntegerArray::<u64>::with_capacity(2, true);
        arr.push(111);
        arr.push_null();
        assert_eq!(arr.data, vec64![111, 0]);
        assert_eq!(arr.get(0), Some(111));
        assert_eq!(arr.get(1), None);
        assert!(arr.is_null(1));
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn test_trait_mutable_array() {
        let mut arr = IntegerArray::<i8>::with_capacity(2, true);
        arr.push(7);
        arr.push(8);
        arr.set(0, 77);
        arr.set_null(1);
        assert_eq!(arr.get(0), Some(77));
        assert_eq!(arr.get(1), None);
        let data = arr.data_mut();
        data[1] = -9;
        assert_eq!(arr.data[1], -9);
    }

    #[test]
    fn test_bulk_push_nulls() {
        let mut arr = IntegerArray::<u16>::with_capacity(8, true);
        arr.push(19);
        arr.push_nulls(3);
        assert_eq!(arr.len(), 4);
        assert_eq!(arr.get(0), Some(19));
        assert_eq!(arr.get(1), None);
        assert_eq!(arr.get(3), None);
        assert!(arr.is_null(2));
    }

    #[test]
    fn test_is_empty_and_len() {
        let mut arr = IntegerArray::<i32>::default();
        assert!(arr.is_empty());
        arr.push(1);
        assert!(!arr.is_empty());
        assert_eq!(arr.len(), 1);
    }

    #[test]
    fn test_out_of_bounds() {
        let arr = IntegerArray::<i64>::default();
        assert_eq!(arr.get(0), None);
        assert_eq!(arr.get(100), None);
    }

    #[test]
    fn test_null_mask_replace() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(9);
        let mut mask = Bitmask::new_set_all(1, false);
        mask.set(0, true);
        arr.set_null_mask(Some(mask));
        assert!(!arr.is_null(0));
    }

    #[test]
    fn test_integer_array_slice() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(10);
        arr.push(20);
        arr.push(30);
        arr.push_null();
        arr.push(50);

        let sliced = arr.slice_clone(1, 3);
        assert_eq!(*sliced.data(), vec64![20, 30, 0]); // default for null = 0
        assert_eq!(sliced.len(), 3);
        assert_eq!(sliced.get(0), Some(20));
        assert_eq!(sliced.get(1), Some(30));
        assert_eq!(sliced.get(2), None); // was null
        assert_eq!(sliced.null_count(), 1);
    }

    #[test]
    fn test_batch_extend_from_slice() {
        let mut arr = IntegerArray::<i32>::default();
        let data = &[1, 2, 3, 4, 5];

        arr.extend_from_slice(data);

        assert_eq!(arr.len(), 5);
        assert_eq!(arr.get(0), Some(1));
        assert_eq!(arr.get(4), Some(5));
    }

    #[test]
    fn test_batch_extend_from_iter_with_capacity() {
        let mut arr = IntegerArray::<i64>::default();
        let data = vec![10i64, 20, 30];

        arr.extend_from_iter_with_capacity(data.into_iter(), 3);

        assert_eq!(arr.len(), 3);
        assert_eq!(arr.get(0), Some(10));
        assert_eq!(arr.get(2), Some(30));
    }

    #[test]
    fn test_batch_fill() {
        let arr = IntegerArray::<i32>::fill(42, 100);

        assert_eq!(arr.len(), 100);
        for i in 0..100 {
            assert_eq!(arr.get(i), Some(42));
        }
    }

    #[test]
    fn test_batch_extend_from_iter_with_capacity_performance() {
        let mut arr = IntegerArray::<u64>::default();
        let data: Vec<u64> = (0..1000).collect();

        arr.extend_from_iter_with_capacity(data.into_iter(), 1000);

        assert_eq!(arr.len(), 1000);
        for i in 0..1000 {
            assert_eq!(arr.get(i), Some(i as u64));
        }
        assert!(!arr.is_nullable()); // No nulls added
    }

    #[test]
    fn test_batch_extend_from_slice_with_nulls() {
        let mut arr = IntegerArray::<i16>::with_capacity(10, true);
        arr.push(100);
        arr.push_null();

        let data = &[200i16, 300, 400];
        arr.extend_from_slice(data);

        assert_eq!(arr.len(), 5);
        assert_eq!(arr.get(0), Some(100));
        assert_eq!(arr.get(1), None);
        assert_eq!(arr.get(2), Some(200));
        assert_eq!(arr.get(3), Some(300));
        assert_eq!(arr.get(4), Some(400));
        assert!(arr.null_count() >= 1); // At least the initial null
    }

    #[test]
    fn test_batch_fill_large() {
        let arr = IntegerArray::<i8>::fill(-127, 500);

        assert_eq!(arr.len(), 500);
        assert_eq!(arr.null_count(), 0);
        for i in 0..500 {
            assert_eq!(arr.get(i), Some(-127));
        }
    }

    #[test]
    fn test_batch_operations_preserve_capacity() {
        let mut arr = IntegerArray::<u32>::with_capacity(100, false);
        let initial_capacity = arr.data.capacity();

        // Should not reallocate since we pre-allocated enough
        let data = vec![1u32, 2, 3, 4, 5];
        arr.extend_from_slice(&data);

        assert!(arr.data.capacity() >= initial_capacity);
        assert_eq!(arr.len(), 5);
    }
}

#[cfg(test)]
#[cfg(feature = "parallel_proc")]
mod parallel_tests {
    use rayon::prelude::*;

    use super::*;
    use crate::structs::bitmask::Bitmask;
    use crate::vec64;

    #[test]
    fn test_integerarray_par_iter() {
        let mut arr = IntegerArray::<i32>::from_slice(&[1, 2, 3, 4]);
        // Only idx 1 valid
        let mut mask = Bitmask::new_set_all(4, false);
        mask.set(1, true);
        arr.null_mask = Some(mask);
        let vals: Vec<i32> = arr.par_iter().map(|v| *v).collect();
        assert_eq!(vals, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_integerarray_par_iter_opt() {
        let mut arr = IntegerArray::<i32>::from_slice(&[1, 2, 3, 4]);
        let mut mask = Bitmask::new_set_all(4, false);
        mask.set(1, true);
        arr.null_mask = Some(mask);
        let opt: Vec<Option<i32>> = arr.par_iter_opt().map(|opt| opt.copied()).collect();
        assert_eq!(opt, vec![None, Some(2), None, None]);
    }

    #[test]
    fn test_integerarray_par_iter_mut() {
        let mut arr = IntegerArray::<i32>::from_slice(&[10, 20, 30, 40]);
        arr.par_iter_mut().for_each(|v| *v += 1);
        let expected = vec![11, 21, 31, 41];
        let actual: Vec<i32> = arr.par_iter().map(|v| *v).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_integerarray_par_iter_range_unchecked() {
        let arr = IntegerArray::<i32>::from_slice(&[1, 2, 3, 4, 5, 6]);
        let out: Vec<&i32> = unsafe { arr.par_iter_range_unchecked(2, 5).collect() };
        assert_eq!(*out[0], 3);
        assert_eq!(*out[1], 4);
        assert_eq!(*out[2], 5);
    }

    #[test]
    fn test_integerarray_par_iter_range_opt_unchecked() {
        let mut arr = IntegerArray::<i32>::from_slice(&[1, 2, 3, 4, 5]);
        let mut mask = Bitmask::new_set_all(5, false);
        mask.set(0, true);
        mask.set(2, true);
        mask.set(4, true);
        arr.null_mask = Some(mask);
        let out: Vec<Option<&i32>> = unsafe { arr.par_iter_range_opt_unchecked(0, 5).collect() };
        assert_eq!(
            out.iter().map(|x| x.copied()).collect::<Vec<_>>(),
            vec![Some(1), None, Some(3), None, Some(5)]
        );
    }

    #[test]
    fn test_integer_array_append_array() {
        use crate::traits::masked_array::MaskedArray;

        // Base: [10, 20, 30] (no nulls)
        let mut arr1 = IntegerArray::<i16>::from_slice(&[10, 20, 30]);
        assert_eq!(arr1.null_mask(), None);

        // Append: [40, 50] (no nulls)
        let arr2 = IntegerArray::<i16>::from_slice(&[40, 50]);
        arr1.append_array(&arr2);
        assert_eq!(*arr1.data(), vec64![10, 20, 30, 40, 50]);
        assert_eq!(arr1.null_mask(), None);
        assert_eq!(arr1.len(), 5);
        assert_eq!(arr1.get(0), Some(10));
        assert_eq!(arr1.get(4), Some(50));

        // Now, add nulls to both and append
        let mut arr3 = IntegerArray::<i16>::with_capacity(3, true);
        arr3.push(60); // valid
        arr3.push_null(); // null
        arr3.push(70); // valid

        let mut arr4 = IntegerArray::<i16>::with_capacity(2, true);
        arr4.push_null(); // null
        arr4.push(80); // valid

        arr3.append_array(&arr4);
        // arr3: [60, None, 70, None, 80]
        assert_eq!(arr3.len(), 5);
        let vals: Vec<Option<i16>> = (0..arr3.len()).map(|i| arr3.get(i)).collect();
        assert_eq!(vals, vec![Some(60), None, Some(70), None, Some(80)]);
        // Null mask correct
        let mask = arr3.null_mask().unwrap();
        assert!(mask.get(0));
        assert!(!mask.get(1));
        assert!(mask.get(2));
        assert!(!mask.get(3));
        assert!(mask.get(4));
        assert_eq!(arr3.null_count(), 2);

        // Append from maskless into masked
        let mut arr5 = IntegerArray::<i16>::with_capacity(2, true);
        arr5.push(90);
        arr5.push(91);
        let arr6 = IntegerArray::<i16>::from_slice(&[92, 93]);
        arr5.append_array(&arr6);
        assert_eq!(*arr5.data(), vec64![90, 91, 92, 93]);
        assert!(arr5.null_mask().unwrap().all_set());

        // Append from masked into maskless
        let mut arr7 = IntegerArray::<i16>::from_slice(&[100, 101]);
        let mut arr8 = IntegerArray::<i16>::with_capacity(2, true);
        arr8.push_null();
        arr8.push(103);
        arr7.append_array(&arr8);
        // arr7: [100, 101, None, 103]
        assert_eq!(arr7.get(2), None);
        assert_eq!(arr7.get(3), Some(103));
        assert_eq!(arr7.null_count(), 1);
    }
}

// Concatenate Trait Implementation

impl<T: Integer> Concatenate for IntegerArray<T> {
    fn concat(
        mut self,
        other: Self,
    ) -> core::result::Result<Self, crate::enums::error::MinarrowError> {
        // Consume other and extend self with its data
        self.append_array(&other);
        Ok(self)
    }
}

#[cfg(test)]
mod concat_tests {
    use super::*;

    #[test]
    fn test_integer_array_concat() {
        let arr1 = IntegerArray::<i32>::from_slice(&[1, 2, 3]);
        let arr2 = IntegerArray::<i32>::from_slice(&[4, 5, 6]);
        let result = arr1.concat(arr2).unwrap();
        assert_eq!(result.len(), 6);
        assert_eq!(result.get(0), Some(1));
        assert_eq!(result.get(5), Some(6));
    }

    #[test]
    fn test_integer_array_concat_with_nulls() {
        let mut arr1 = IntegerArray::<i32>::with_capacity(2, true);
        arr1.push(10);
        arr1.push_null();

        let mut arr2 = IntegerArray::<i32>::with_capacity(2, true);
        arr2.push_null();
        arr2.push(40);

        let result = arr1.concat(arr2).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result.get(0), Some(10));
        assert_eq!(result.get(1), None);
        assert_eq!(result.get(2), None);
        assert_eq!(result.get(3), Some(40));
        assert_eq!(result.null_count(), 2);
    }
}
