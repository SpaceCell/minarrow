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

//! # **FloatArray Module** - *Mid-Level, Inner Typed Float Array*
//!
//! Arrow-compatible, SIMD-aligned floating-point array optimised for analytical workloads.
//!
//! ## Overview
//! - Logical type: fixed-width floats (`T: Float`).
//! - Physical storage: `Buffer<T>` (backed by `Vec64<T>` for 64-byte alignment) plus optional
//!   Arrow-style validity mask (`Bitmask`).
//! - Usable standalone or as the floating-point arm of higher-level enums (`NumericArray`, `Array`).
//! - Supports both explicit null masks and NaN-as-null semantics (at the caller’s discretion).
//! - Zero-copy compatible with standard `Vec` and slice types.
//!
//! ## Features
//! - **Construction** from slices, `Vec`, or `Vec64`, with or without a null mask.
//! - **Mutation**: push/set, bulk null insertion, resize.
//! - **Iteration**: safe and (optionally) parallel via the `MaskedArray` trait surface.
//! - **Display**: formatted preview with numeric formatting helpers.
//!
//! ## Null Semantics
//! - **Explicit mask**: Arrow-compatible null representation (`1 = valid`, `0 = null`).
//! - **NaN-based**: one also has the option to skip the mask entirely and use `NaN` as a sentinel for missing values
//!   (similar to Pandas/NumPy conventions), but this is not recommended as it is not a supported pattern in this
//! library suite and therefore may yield unexpected results.
//!
//! ## Performance Benchmarks
//! Benchmarks under `examples/hotloop_benchmark_std` compare construction + sum over 1000 elems,
//! averaged across 1000 runs on a 2024 Intel(R) Core(TM) Ultra 7 155H (22 logical CPUs, 32 GB RAM).
//!
//! **Build note:** use `--release` for representative numbers; debug builds distort results.
//!
//! **Averaged Results (1000 runs, size = 1000)**
//! - `Vec<f64>` (raw):                        avg = 0.632 µs
//! - `FloatArray` (minarrow, direct):         avg = 0.636 µs
//! - `Float64Array` (arrow-rs, struct):       avg = 0.798 µs
//! - `FloatArray` (minarrow, enum):           avg = 1.047 µs
//! - `Float64Array` (arrow-rs, dynamic trait): avg = 1.255 µs
//!
//! **Takeaways**
//! - Direct `FloatArray` matches raw `Vec<f64>` performance while adding SIMD alignment + Arrow interop.
//! - NaN-based null handling eliminates mask overhead in NaN-tolerant workflows.
//! - Enum wrapping adds small overhead but remains faster than arrow-rs dynamic dispatch.
//! - For latency-critical workloads, use direct `FloatArray`; otherwise differences are negligible.
//!
use std::fmt::{Display, Formatter};

use crate::enums::shape_dim::ShapeDim;
use crate::traits::concatenate::Concatenate;
use crate::traits::print::{MAX_PREVIEW, format_float};
use crate::traits::shape::Shape;
use crate::traits::type_unions::Float;
use crate::{
    Bitmask, Buffer, Length, MaskedArray, Offset, impl_arc_masked_array, impl_array_ref_deref,
    impl_from_vec_primitive, impl_masked_array, impl_numeric_array_constructors,
};
use vec64::Vec64;

/// # FloatArray
///
/// Arrow-compatible, 64-byte aligned floating-point array with optional null mask.
///
/// ## Role
/// - Many will prefer the higher level `Array` type, which dispatches to this when
/// necessary.
/// - Can be used as a standalone array or as the numeric arm of `NumericArray` / `Array`.
///
/// ## Description
/// - Stores floating-point values in a contiguous `Buffer<T>` (`Vec64<T>` under the hood).
/// - Optional Arrow-style validity bitmap (`1 = valid`, `0 = null`) via `Bitmask`.
/// - Can omit null mask entirely and represent missing values with `NaN` where acceptable.
/// - Implements [`MaskedArray`] for consistent nullable array behaviour and interop with
///   higher-level containers/enums.
/// - Can be used as a standalone numeric buffer or as the float arm of `NumericArray` / `Array`.
///
/// ### Fields
/// - `data`: backing buffer of float values (`Buffer<T>`).
/// - `null_mask`: optional bit-packed validity bitmap.
///
/// ## Example
/// ```rust
/// use minarrow::{FloatArray, MaskedArray};
///
/// // Dense, no nulls
/// let arr = FloatArray::<f64>::from_slice(&[1.1, 2.2, 3.3]);
/// assert_eq!(arr.len(), 3);
/// assert_eq!(arr.get(1), Some(2.2));
///
/// // With nulls
/// let mut arr = FloatArray::<f32>::with_capacity(3, true);
/// arr.push(10.0);
/// arr.push_null();
/// arr.push(30.5);
/// assert_eq!(arr.get(1), None);
/// assert_eq!(arr.null_count(), 1);
/// ```
#[repr(C, align(64))]
#[derive(PartialEq, Clone, Debug, Default)]
pub struct FloatArray<T> {
    /// Backing buffer for values.
    pub data: Buffer<T>,
    /// Optional null mask (bit-packed; 1=valid, 0=null).
    pub null_mask: Option<Bitmask>,
}

impl_numeric_array_constructors!(FloatArray, Float);
impl_masked_array!(FloatArray, Float, Buffer<T>, T);
impl_from_vec_primitive!(FloatArray);
impl_array_ref_deref!(FloatArray<T>);
impl_arc_masked_array!(
    Inner = FloatArray<T>,
    T = T,
    Container = Buffer<T>,
    LogicalType = T,
    CopyType = T,
    BufferT = T,
    Variant = NumericArray,
    Bound = Float,
);

impl<T> Display for FloatArray<T>
where
    T: Float + Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let len = self.len();
        let nulls = self.null_count();

        writeln!(
            f,
            "FloatArray [{} values] (dtype: float, nulls: {})",
            len, nulls
        )?;

        write!(f, "[")?;

        for i in 0..usize::min(len, MAX_PREVIEW) {
            if i > 0 {
                write!(f, ", ")?;
            }

            match self.get(i) {
                Some(v) => write!(f, "{}", format_float(v))?,
                None => write!(f, "null")?,
            }
        }

        if len > MAX_PREVIEW {
            write!(f, ", … ({} total)", len)?;
        }

        write!(f, "]")
    }
}

impl<T: Float> Shape for FloatArray<T> {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank1(self.len())
    }
}

impl<T: Float> Concatenate for FloatArray<T> {
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
mod tests {
    use super::*;
    use crate::traits::masked_array::MaskedArray;
    use crate::vec64;

    #[test]
    fn test_new_and_with_capacity() {
        let arr = FloatArray::<f64>::default();
        assert_eq!(arr.data.len(), 0);
        assert!(arr.null_mask.is_none());

        let mut arr = FloatArray::<f32>::with_capacity(16, true);
        assert_eq!(arr.data.len(), 0);
        assert!(arr.data.capacity() >= 16);
        assert!(arr.null_mask.is_some());

        // Reserved null-mask slots must default to valid (1).
        assert_eq!(arr.null_count(), 0);

        // After non-null pushes, null_count remains 0 - the pre-allocated
        // tail bits beyond len() must not poison the count.
        arr.push(1.5);
        arr.push(2.5);
        assert_eq!(arr.null_count(), 0);

        arr.push_null();
        assert_eq!(arr.null_count(), 1);
    }

    #[test]
    fn test_push_and_get_no_null_mask() {
        let mut arr = FloatArray::<f64>::with_capacity(2, false);
        arr.push(3.14);
        arr.push(2.71);
        assert_eq!(arr.data, vec64![3.14, 2.71],);
        assert_eq!(arr.get(0), Some(3.14));
        assert_eq!(arr.get(1), Some(2.71));
        assert!(!arr.is_null(0));
        assert!(!arr.is_null(1));
    }

    #[test]
    fn test_push_and_get_with_null_mask() {
        let mut arr = FloatArray::<f32>::with_capacity(3, true);
        arr.push(1.23);
        arr.push_null();
        arr.push(9.87);
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.get(0), Some(1.23));
        assert_eq!(arr.get(1), None);
        assert_eq!(arr.get(2), Some(9.87));
        assert!(!arr.is_null(0));
        assert!(arr.is_null(1));
        assert!(!arr.is_null(2));
    }

    #[test]
    fn test_push_null_auto_mask() {
        let mut arr = FloatArray::<f32>::default();
        arr.push_null();
        assert_eq!(arr.data, vec64![0.0]);
        assert!(arr.is_null(0));
        assert!(arr.null_mask.is_some());
    }

    #[test]
    fn test_set_and_set_null() {
        let mut arr = FloatArray::<f32>::with_capacity(4, true);
        arr.push(0.1);
        arr.push(0.2);
        arr.push(0.3);
        arr.set(1, 7.7);
        assert_eq!(arr.get(1), Some(7.7));
        arr.set_null(2);
        assert_eq!(arr.get(2), None);
        assert!(arr.is_null(2));
    }

    #[test]
    fn test_trait_masked_array() {
        let mut arr = FloatArray::<f64>::with_capacity(2, true);
        arr.push(11.1);
        arr.push_null();
        assert_eq!(arr.data, vec64![11.1, 0.0]);
        assert_eq!(arr.get(0), Some(11.1));
        assert_eq!(arr.get(1), None);
        assert!(arr.is_null(1));
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn test_trait_mutable_array() {
        let mut arr = FloatArray::<f32>::with_capacity(2, true);
        arr.push(1.0);
        arr.push(2.0);
        arr.set(0, 10.0);
        arr.set_null(1);
        assert_eq!(arr.get(0), Some(10.0));
        assert_eq!(arr.get(1), None);
        let data = arr.data_mut();
        data[1] = 123.0;
        assert_eq!(arr.data[1], 123.0);
    }

    #[test]
    fn test_bulk_push_nulls() {
        let mut arr = FloatArray::<f64>::with_capacity(8, true);
        arr.push(9.9);
        arr.push_nulls(3);
        assert_eq!(arr.len(), 4);
        assert_eq!(arr.get(0), Some(9.9));
        assert_eq!(arr.get(1), None);
        assert_eq!(arr.get(3), None);
    }

    #[test]
    fn test_is_empty_and_len() {
        let mut arr = FloatArray::<f64>::default();
        assert!(arr.is_empty());
        arr.push(7.0);
        assert!(!arr.is_empty());
        assert_eq!(arr.len(), 1);
    }

    #[test]
    fn test_out_of_bounds() {
        let arr = FloatArray::<f32>::default();
        assert_eq!(arr.get(0), None);
        assert_eq!(arr.get(10), None);
    }

    #[test]
    fn test_null_mask_replace() {
        let mut arr = FloatArray::<f64>::default();
        arr.push(1.0);
        arr.set_null_mask(Some(Bitmask::from_bytes([0b0000_0001], 1)));
        assert!(!arr.is_null(0));
    }

    #[test]
    fn test_float_array_slice() {
        let mut arr = FloatArray::<f64>::default();
        arr.push(1.5);
        arr.push(2.5);
        arr.push_null();
        arr.push(4.5);
        arr.push(5.5);

        let sliced = arr.slice_clone(1, 3);
        assert_eq!(sliced.len(), 3);
        assert_eq!(sliced.get(0), Some(2.5));
        assert_eq!(sliced.get(1), None); // was null
        assert_eq!(sliced.get(2), Some(4.5));
        assert_eq!(sliced.null_count(), 1);
    }

    #[test]
    fn test_batch_extend_from_iter_with_capacity() {
        let mut arr = FloatArray::<f64>::default();
        let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.5).collect();

        arr.extend_from_iter_with_capacity(data.into_iter(), 100);

        assert_eq!(arr.len(), 100);
        for i in 0..100 {
            assert_eq!(arr.get(i), Some(i as f64 * 0.5));
        }
    }

    #[test]
    fn test_batch_extend_from_slice() {
        let mut arr = FloatArray::<f32>::with_capacity(5, true);
        arr.push(1.1);
        arr.push_null();

        let data = &[2.2f32, 3.3, 4.4];
        arr.extend_from_slice(data);

        assert_eq!(arr.len(), 5);
        assert_eq!(arr.get(0), Some(1.1));
        assert_eq!(arr.get(1), None);
        assert_eq!(arr.get(2), Some(2.2));
        assert_eq!(arr.get(3), Some(3.3));
        assert_eq!(arr.get(4), Some(4.4));
    }

    #[test]
    fn test_batch_fill_with_special_values() {
        let arr = FloatArray::<f64>::fill(f64::NAN, 10);

        assert_eq!(arr.len(), 10);
        for i in 0..10 {
            assert!(arr.get(i).unwrap().is_nan());
        }
    }

    #[test]
    fn test_batch_fill_infinity() {
        let arr = FloatArray::<f32>::fill(f32::INFINITY, 5);

        assert_eq!(arr.len(), 5);
        for i in 0..5 {
            assert_eq!(arr.get(i), Some(f32::INFINITY));
        }
    }

    #[test]
    fn test_float_array_concat() {
        let arr1 = FloatArray::<f64>::from_slice(&[1.1, 2.2, 3.3]);
        let arr2 = FloatArray::<f64>::from_slice(&[4.4, 5.5]);

        let result = arr1.concat(arr2).unwrap();

        assert_eq!(result.len(), 5);
        assert_eq!(result.get(0), Some(1.1));
        assert_eq!(result.get(1), Some(2.2));
        assert_eq!(result.get(2), Some(3.3));
        assert_eq!(result.get(3), Some(4.4));
        assert_eq!(result.get(4), Some(5.5));
    }

    #[test]
    fn test_float_array_concat_with_nulls() {
        let mut arr1 = FloatArray::<f32>::with_capacity(3, true);
        arr1.push(1.0);
        arr1.push_null();
        arr1.push(3.0);

        let mut arr2 = FloatArray::<f32>::with_capacity(2, true);
        arr2.push(4.0);
        arr2.push_null();

        let result = arr1.concat(arr2).unwrap();

        assert_eq!(result.len(), 5);
        assert_eq!(result.get(0), Some(1.0));
        assert_eq!(result.get(1), None);
        assert_eq!(result.get(2), Some(3.0));
        assert_eq!(result.get(3), Some(4.0));
        assert_eq!(result.get(4), None);
        assert_eq!(result.null_count(), 2);
    }
}

#[cfg(test)]
#[cfg(feature = "parallel_proc")]
mod parallel_tests {
    use rayon::prelude::*;

    use crate::{Bitmask, FloatArray};

    #[test]
    fn test_floatarray_par_iter() {
        let mut arr = FloatArray::<f32>::from_slice(&[1.0, 2.5, 3.5]);
        let mut mask = Bitmask::new_set_all(3, false);
        mask.set_bits_chunk(0, 0b0000_0001, 3);
        arr.null_mask = Some(mask);
        let vals: Vec<f32> = arr.par_iter().map(|v| *v).collect();
        assert_eq!(vals, vec![1.0, 2.5, 3.5]);
    }

    #[test]
    fn test_floatarray_par_iter_opt() {
        let mut arr = FloatArray::<f32>::from_slice(&[1.0, 2.5, 3.5]);
        let mut mask = Bitmask::new_set_all(3, false);
        mask.set_bits_chunk(0, 0b0000_0001, 3);
        arr.null_mask = Some(mask);
        let opt: Vec<Option<f32>> = arr.par_iter_opt().map(|v| v.copied()).collect();
        assert_eq!(opt, vec![Some(1.0), None, None]);
    }

    #[test]
    fn test_floatarray_par_iter_mut() {
        let mut arr = FloatArray::<f32>::from_slice(&[1.0, 2.0, 3.0]);
        arr.par_iter_mut().for_each(|v| *v *= 10.0);
        let expected = vec![10.0, 20.0, 30.0];
        let actual: Vec<f32> = arr.par_iter().map(|v| *v).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_floatarray_par_iter_range_unchecked() {
        let arr = FloatArray::<f32>::from_slice(&[1.0, 2.5, 3.5, 4.5]);
        let out: Vec<&f32> = unsafe { arr.par_iter_range_unchecked(1, 4).collect() };
        assert_eq!(*out[0], 2.5);
        assert_eq!(*out[1], 3.5);
        assert_eq!(*out[2], 4.5);
    }

    #[test]
    fn test_floatarray_par_iter_range_opt_unchecked() {
        let mut arr = FloatArray::<f32>::from_slice(&[1.1, 2.2, 3.3, 4.4]);
        let mut mask = Bitmask::new_set_all(4, false);
        mask.set_bits_chunk(0, 0b0000_0110, 4); // indices 1,2 valid
        arr.null_mask = Some(mask);
        let out: Vec<Option<&f32>> = unsafe { arr.par_iter_range_opt_unchecked(0, 4).collect() };
        assert_eq!(
            out.iter().map(|x| x.copied()).collect::<Vec<_>>(),
            vec![None, Some(2.2), Some(3.3), None]
        );
    }
}
