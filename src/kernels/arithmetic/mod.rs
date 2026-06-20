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

//! # **Arithmetic Kernels Module** - *High-Performance Arithmetic*
//!
//! SIMD-optimised arithmetic operations for numeric arrays with null-aware semantics.
//!
//! ## Modules
//! - **`dispatch`**: Smart dispatch layer selecting SIMD vs scalar implementations based on alignment
//! - **`simd`**: SIMD-accelerated implementations using `std::simd` with portable vectorisation  
//! - **`std`**: Scalar fallback implementations for compatibility and unaligned data
//! - **`string`**: Specialised arithmetic operations for string concatenation and manipulation
//!
//! ## Operations
//! Supports standard arithmetic operations (add, subtract, multiply, divide, remainder, power)
//! plus fused multiply-add (FMA) for floating-point types with hardware acceleration.
//!
//! ## Scope
//! **These do not leverage parallel-thread processing, as this is expected to be applied in the engine layer,
//! which is app-specific.**.

pub mod dispatch;
#[cfg(feature = "simd")]
pub mod simd;
pub mod std;
pub mod string;
pub mod string_ops;
#[cfg(feature = "broadcast")]
pub mod types;

// Shared tests for SIMD and Std

#[cfg(test)]
mod tests {
    use crate::kernels::bitmask::merge_bitmasks_to_new;
    use crate::structs::variants::float::FloatArray;
    use crate::structs::variants::integer::IntegerArray;
    use crate::{Bitmask, MaskedArray, vec64};

    use crate::enums::operators::ArithmeticOperator;
    use crate::kernels::arithmetic::dispatch::{
        apply_float_f32, apply_float_f64, apply_fma_f32, apply_fma_f64, apply_int_i32,
        apply_int_i64, apply_int_u32, apply_int_u64,
    };
    #[cfg(feature = "extended_numeric_types")]
    use crate::kernels::arithmetic::dispatch::{
        apply_int_i8, apply_int_i16, apply_int_u8, apply_int_u16,
    };
    #[cfg(feature = "simd")]
    use crate::kernels::arithmetic::simd::int_dense_body_simd;

    fn assert_int<T>(arr: &IntegerArray<T>, values: &[T], valid: Option<&[bool]>)
    where
        T: num_traits::PrimInt + std::fmt::Debug,
    {
        assert_eq!(arr.data.as_slice(), values);
        match (valid, &arr.null_mask) {
            (None, None) => {}
            (Some(expected), Some(mask)) => {
                for (i, bit) in expected.iter().enumerate() {
                    assert_eq!(
                        unsafe { mask.get_unchecked(i) },
                        *bit,
                        "mask mismatch at {i}"
                    );
                }
            }
            (None, Some(mask)) => {
                assert!(mask.all_true(), "mask unexpectedly present");
            }
            (Some(_), None) => panic!("expected mask missing"),
        }
    }

    fn assert_float<T>(arr: &FloatArray<T>, values: &[T], valid: Option<&[bool]>)
    where
        T: num_traits::Float + std::fmt::Debug,
    {
        assert_eq!(arr.data.as_slice(), values);
        match (valid, &arr.null_mask) {
            (None, None) => {}
            (Some(expected), Some(mask)) => {
                for (i, bit) in expected.iter().enumerate() {
                    assert_eq!(
                        unsafe { mask.get_unchecked(i) },
                        *bit,
                        "mask mismatch at {i}"
                    );
                }
            }
            (None, Some(mask)) => {
                assert!(mask.all_true(), "mask unexpectedly present");
            }
            (Some(_), None) => panic!("expected mask missing"),
        }
    }

    fn bitmask(bits: &[bool]) -> Bitmask {
        let mut m = Bitmask::new_set_all(bits.len(), false);
        for (i, b) in bits.iter().enumerate() {
            unsafe { m.set_unchecked(i, *b) };
        }
        m
    }

    macro_rules! int_kernel_suite {
        ($fn_dense:ident, $fn_masked:ident, $fn_empty:ident, $ty:ty, $apply_fn:ident) => {
            #[test]
            fn $fn_dense() {
                let lhs = vec64![1, 4, 9, 16];
                let rhs = vec64![1, 2, 3, 4];

                let out = $apply_fn(&lhs, &rhs, ArithmeticOperator::Add, None).unwrap();
                assert_int(
                    &out,
                    &IntegerArray::<$ty>::from_slice(&[2, 6, 12, 20]),
                    None,
                );

                let out = $apply_fn(&lhs, &rhs, ArithmeticOperator::Subtract, None).unwrap();
                assert_int(&out, &IntegerArray::<$ty>::from_slice(&[0, 2, 6, 12]), None);

                let out = $apply_fn(&lhs, &rhs, ArithmeticOperator::Multiply, None).unwrap();
                assert_int(
                    &out,
                    &IntegerArray::<$ty>::from_slice(&[1, 8, 27, 64]),
                    None,
                );

                let out = $apply_fn(&lhs, &rhs, ArithmeticOperator::Divide, None).unwrap();
                assert_int(&out, &IntegerArray::<$ty>::from_slice(&[1, 2, 3, 4]), None);

                let out = $apply_fn(&lhs, &rhs, ArithmeticOperator::Remainder, None).unwrap();
                assert_int(&out, &IntegerArray::<$ty>::from_slice(&[0, 0, 0, 0]), None);

                let expected: Vec<$ty> = lhs
                    .iter()
                    .zip(rhs.iter())
                    .map(|(&a, &b)| {
                        let mut acc = <$ty as num_traits::One>::one();
                        for _ in 0..(b as u32) {
                            acc = acc.wrapping_mul(a);
                        }
                        acc
                    })
                    .collect();
                let out = $apply_fn(&lhs, &rhs, ArithmeticOperator::Power, None).unwrap();
                assert_int(&out, &IntegerArray::<$ty>::from_slice(&expected), None);

                // Division by zero should panic
                let rhs_divzero: &[$ty] = &[0, 0, 0, 0];
                let result = std::panic::catch_unwind(|| {
                    $apply_fn(&lhs, rhs_divzero, ArithmeticOperator::Divide, None).unwrap()
                });
                assert!(
                    result.is_err(),
                    "Dense integer kernel division by zero must panic"
                );

                let result = std::panic::catch_unwind(|| {
                    $apply_fn(&lhs, rhs_divzero, ArithmeticOperator::Remainder, None).unwrap()
                });
                assert!(
                    result.is_err(),
                    "Dense integer kernel remainder by zero must panic"
                );
            }

            #[test]
            fn $fn_masked() {
                let lhs = vec64![10, 20, 30, 40];
                let rhs = vec64![2, 0, 3, 5];
                let mask = bitmask(&[true, false, true, false]);

                // Division: mask==true and rhs!=0 are valid, otherwise null
                let out = $apply_fn(&lhs, &rhs, ArithmeticOperator::Divide, Some(&mask)).unwrap();
                assert_int(
                    &out,
                    &IntegerArray::<$ty>::from_slice(&[5, 0, 10, 0]),
                    Some(&[true, false, true, false]),
                );

                // Remainder with mask, matching above
                let out =
                    $apply_fn(&lhs, &rhs, ArithmeticOperator::Remainder, Some(&mask)).unwrap();
                assert_int(
                    &out,
                    &IntegerArray::<$ty>::from_slice(&[0, 0, 0, 0]),
                    Some(&[true, false, true, false]),
                );

                // Division by zero where mask is true but rhs is zero must yield null in mask (false) and output 0
                let mask_divzero = bitmask(&[true, true, true, true]);
                let rhs_divzero: &[$ty] = &[1, 0, 2, 0];
                let lhs2: &[$ty] = &[100, 100, 100, 100];

                let out = $apply_fn(
                    lhs2,
                    rhs_divzero,
                    ArithmeticOperator::Divide,
                    Some(&mask_divzero),
                )
                .unwrap();
                assert_int(
                    &out,
                    &IntegerArray::<$ty>::from_slice(&[100, 0, 50, 0]),
                    Some(&[true, false, true, false]),
                );
            }

            #[test]
            fn $fn_empty() {
                let lhs = vec64![];
                let rhs = vec64![];
                let out = $apply_fn(&lhs, &rhs, ArithmeticOperator::Add, None).unwrap();
                assert!(out.is_empty());
            }
        };
    }

    #[cfg(feature = "extended_numeric_types")]
    int_kernel_suite!(
        apply_int_i8_dense,
        apply_int_i8_masked,
        apply_int_i8_empty,
        i8,
        apply_int_i8
    );
    #[cfg(feature = "extended_numeric_types")]
    int_kernel_suite!(
        apply_int_u8_dense,
        apply_int_u8_masked,
        apply_int_u8_empty,
        u8,
        apply_int_u8
    );
    #[cfg(feature = "extended_numeric_types")]
    int_kernel_suite!(
        apply_int_i16_dense,
        apply_int_i16_masked,
        apply_int_i16_empty,
        i16,
        apply_int_i16
    );
    #[cfg(feature = "extended_numeric_types")]
    int_kernel_suite!(
        apply_int_u16_dense,
        apply_int_u16_masked,
        apply_int_u16_empty,
        u16,
        apply_int_u16
    );
    int_kernel_suite!(
        apply_int_i32_dense,
        apply_int_i32_masked,
        apply_int_i32_empty,
        i32,
        apply_int_i32
    );
    int_kernel_suite!(
        apply_int_u32_dense,
        apply_int_u32_masked,
        apply_int_u32_empty,
        u32,
        apply_int_u32
    );
    int_kernel_suite!(
        apply_int_i64_dense,
        apply_int_i64_masked,
        apply_int_i64_empty,
        i64,
        apply_int_i64
    );
    int_kernel_suite!(
        apply_int_u64_dense,
        apply_int_u64_masked,
        apply_int_u64_empty,
        u64,
        apply_int_u64
    );

    macro_rules! float_kernel_suite {
        ($test_fn:ident, $ty:ty, $apply_fn:ident, $eps:expr) => {
            #[test]
            fn $test_fn() {
                let lhs = vec64![1.0, 4.0, 9.0, 16.0];
                let rhs = vec64![0.5, 2.0, 3.0, 4.0];

                let lhs: &[$ty] = lhs.as_slice();
                let rhs: &[$ty] = rhs.as_slice();

                let arr = $apply_fn(lhs, rhs, ArithmeticOperator::Add, None).unwrap();
                assert_eq!(arr.data.as_slice(), &[1.5 as $ty, 6.0, 12.0, 20.0]);

                let arr = $apply_fn(lhs, rhs, ArithmeticOperator::Subtract, None).unwrap();
                assert_eq!(arr.data.as_slice(), &[0.5 as $ty, 2.0, 6.0, 12.0]);

                let arr = $apply_fn(lhs, rhs, ArithmeticOperator::Multiply, None).unwrap();
                assert_eq!(arr.data.as_slice(), &[0.5 as $ty, 8.0, 27.0, 64.0]);

                let arr = $apply_fn(lhs, rhs, ArithmeticOperator::Divide, None).unwrap();
                assert_eq!(arr.data.as_slice(), &[2.0 as $ty, 2.0, 3.0, 4.0]);

                let arr = $apply_fn(lhs, rhs, ArithmeticOperator::Remainder, None).unwrap();
                assert!(
                    arr.data
                        .as_slice()
                        .iter()
                        .zip(
                            [1.0 % 0.5, 4.0 % 2.0, 9.0 % 3.0, 16.0 % 4.0]
                                .iter()
                                .map(|&x| x as $ty)
                        )
                        .all(|(a, b)| (*a - b).abs() < $eps)
                );

                let arr = $apply_fn(lhs, rhs, ArithmeticOperator::Power, None).unwrap();
                let expected: Vec<$ty> = lhs
                    .iter()
                    .zip(rhs.iter())
                    .map(|(&a, &b)| (b * a.ln()).exp())
                    .collect();
                assert!(
                    arr.data
                        .as_slice()
                        .iter()
                        .zip(expected.iter())
                        .all(|(a, b)| (*a - *b).abs() < $eps)
                );

                // Division by zero for floats yields Inf/NaN, never panics
                let rhs_divzero: &[$ty] = &[0.0, 0.0, 0.0, 0.0];
                let arr = $apply_fn(lhs, rhs_divzero, ArithmeticOperator::Divide, None).unwrap();
                assert!(
                    arr.data.iter().all(|&x| x.is_infinite()),
                    "Float division by zero should yield Inf"
                );

                let arr = $apply_fn(lhs, rhs_divzero, ArithmeticOperator::Remainder, None).unwrap();
                assert!(
                    arr.data.iter().all(|&x| x.is_nan()),
                    "Float remainder by zero should yield NaN"
                );

                // Masked test
                let mask = bitmask(&[true, false, true, false]);
                let arr = $apply_fn(lhs, rhs, ArithmeticOperator::Multiply, Some(&mask)).unwrap();
                assert_eq!(arr.data.as_slice(), &[0.5 as $ty, 0.0, 27.0, 0.0]);
                assert_eq!(arr.null_mask.as_ref().unwrap().len(), 4);

                // Empty
                let arr = $apply_fn(&[], &[], ArithmeticOperator::Add, None).unwrap();
                assert!(arr.is_empty());
            }
        };
    }

    float_kernel_suite!(apply_float_f32_dense, f32, apply_float_f32, 1e-6f32);
    float_kernel_suite!(apply_float_f64_dense, f64, apply_float_f64, 1e-12f64);

    #[test]
    fn fma_f32() {
        let lhs = vec64![1.0f32, 2.0, 3.0];
        let rhs = vec64![4.0f32, 5.0, 6.0];
        let acc = vec64![0.5f32, 0.5, 0.5];
        let out = apply_fma_f32(&lhs, &rhs, &acc, None).unwrap();
        assert_float(&out, &[4.5, 10.5, 18.5], None);

        let mask = bitmask(&[true, false, true]);
        let out = apply_fma_f32(&lhs, &rhs, &acc, Some(&mask)).unwrap();
        assert_float(&out, &[4.5, 0.0, 18.5], Some(&[true, false, true]));

        let out = apply_fma_f32(&[], &[], &[], None).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn fma_f64() {
        let lhs = vec64![1.0f64, 2.0, 3.0];
        let rhs = vec64![4.0f64, 5.0, 6.0];
        let acc = vec64![0.5f64, 0.5, 0.5];
        let out = apply_fma_f64(&lhs, &rhs, &acc, None).unwrap();
        assert_float(&out, &[4.5, 10.5, 18.5], None);

        let mask = bitmask(&[true, false, true]);
        let out = apply_fma_f64(&lhs, &rhs, &acc, Some(&mask)).unwrap();
        assert_float(&out, &[4.5, 0.0, 18.5], Some(&[true, false, true]));
    }

    #[test]
    fn merge_masks_correctness() {
        let a = bitmask(&[true, false, true, true]);
        let b = bitmask(&[true, true, false, true]);
        let merged = merge_bitmasks_to_new(Some(&a), Some(&b), 4).unwrap();
        let expected = vec![true, false, false, true];
        let merged_vec: Vec<bool> = (0..4).map(|i| merged.get(i)).collect();
        assert_eq!(merged_vec, expected);
    }

    // Datetime Kernels
    #[cfg(feature = "datetime")]
    use crate::structs::variants::datetime::DatetimeArray;

    #[cfg(feature = "datetime")]
    use crate::kernels::arithmetic::dispatch::apply_datetime_i64;

    #[cfg(feature = "datetime")]
    #[test]
    fn datetime_add() {
        let lhs = DatetimeArray::<i64>::from_slice(&[1_000i64, 2_000, 3_000], None);
        let rhs = DatetimeArray::<i64>::from_slice(&[10, 20, 30], None);
        let lhs_slice = (&lhs, 0, lhs.len());
        let rhs_slice = (&rhs, 0, rhs.len());
        let out = apply_datetime_i64(lhs_slice, rhs_slice, ArithmeticOperator::Add).unwrap();
        assert_eq!(out.data.as_slice(), &[1_010, 2_020, 3_030]);
        assert!(out.null_mask.is_none());
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn datetime_all_ops() {
        let lhs = DatetimeArray::<i64>::from_slice(&[10, 20, 30, 40], None);
        let rhs = DatetimeArray::<i64>::from_slice(&[1, 2, 3, 4], None);
        let lhs_slice = (&lhs, 0, lhs.len());
        let rhs_slice = (&rhs, 0, rhs.len());

        let out = apply_datetime_i64(lhs_slice, rhs_slice, ArithmeticOperator::Add).unwrap();
        assert_eq!(out.data.as_slice(), &[11, 22, 33, 44]);

        let out = apply_datetime_i64(lhs_slice, rhs_slice, ArithmeticOperator::Subtract).unwrap();
        assert_eq!(out.data.as_slice(), &[9, 18, 27, 36]);

        let out = apply_datetime_i64(lhs_slice, rhs_slice, ArithmeticOperator::Multiply).unwrap();
        assert_eq!(out.data.as_slice(), &[10, 40, 90, 160]);

        let out = apply_datetime_i64(lhs_slice, rhs_slice, ArithmeticOperator::Divide).unwrap();
        assert_eq!(out.data.as_slice(), &[10, 10, 10, 10]);

        let out = apply_datetime_i64(lhs_slice, rhs_slice, ArithmeticOperator::Remainder).unwrap();
        assert_eq!(out.data.as_slice(), &[0, 0, 0, 0]);

        let out = apply_datetime_i64(lhs_slice, rhs_slice, ArithmeticOperator::Power).unwrap();
        assert_eq!(
            out.data.as_slice(),
            &[10_i64.pow(1), 20_i64.pow(2), 30_i64.pow(3), 40_i64.pow(4)]
        );
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn datetime_masked_and_empty() {
        let lhs = DatetimeArray::<i64>::from_slice(&[10, 20, 30, 40], None);
        let rhs = DatetimeArray::<i64>::from_slice(&[1, 2, 3, 4], None);
        let mask = bitmask(&[true, false, true, true]);
        let lhs_slice = (&lhs, 0, lhs.len());
        let rhs_slice = (&rhs, 0, rhs.len());

        let out = apply_datetime_i64(lhs_slice, rhs_slice, ArithmeticOperator::Add).unwrap();
        assert_eq!(out.data.as_slice(), &[11, 22, 33, 44]);

        // Masked
        let mut lhs_masked = lhs.clone();
        lhs_masked.null_mask = Some(mask.clone());
        let lhs_slice_masked = (&lhs_masked, 0, lhs_masked.len());
        let out = apply_datetime_i64(lhs_slice_masked, rhs_slice, ArithmeticOperator::Add).unwrap();
        let expected = vec![11, 0, 33, 44];
        let mask_vec: Vec<bool> = (0..4).map(|i| mask.get(i)).collect();
        assert_eq!(out.data.as_slice(), &expected);
        assert_eq!(
            out.null_mask
                .as_ref()
                .map(|m| (0..4).map(|i| m.get(i)).collect::<Vec<_>>()),
            Some(mask_vec)
        );

        // Empty
        let lhs_empty = DatetimeArray::<i64>::from_slice(&[], None);
        let rhs_empty = DatetimeArray::<i64>::from_slice(&[], None);
        let lhs_slice = (&lhs_empty, 0, lhs_empty.len());
        let rhs_slice = (&rhs_empty, 0, rhs_empty.len());
        let out = apply_datetime_i64(lhs_slice, rhs_slice, ArithmeticOperator::Add).unwrap();
        assert!(out.is_empty());
    }

    #[cfg(feature = "datetime")]
    #[test]
    #[should_panic(expected = "apply_datetime: length mismatch")]
    fn datetime_len_mismatch_panics() {
        let lhs = DatetimeArray::<i64>::from_slice(&[1_000i64, 2_000], None);
        let rhs = DatetimeArray::<i64>::from_slice(&[10], None);
        let lhs_slice = (&lhs, 0, lhs.len());
        let rhs_slice = (&rhs, 0, rhs.len());
        let _ = apply_datetime_i64(lhs_slice, rhs_slice, ArithmeticOperator::Add).unwrap();
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_int_dense_power_short_vs_long_input_simd() {
        let lhs_short = vec64![2u32; 16];
        let rhs_short = vec64![10u32; 16];
        let mut out_short = vec64![0u32; 16];

        let lhs_long = vec64![2u32; 128];
        let rhs_long = vec64![10u32; 128];
        let mut out_long = vec64![0u32; 128];

        int_dense_body_simd::<u32, 4>(
            ArithmeticOperator::Power,
            &lhs_short,
            &rhs_short,
            &mut out_short,
        );
        int_dense_body_simd::<u32, 4>(
            ArithmeticOperator::Power,
            &lhs_long,
            &rhs_long,
            &mut out_long,
        );

        for &v in out_short.iter() {
            assert_eq!(v, 1024);
        }
        for &v in out_long.iter() {
            assert_eq!(v, 1024);
        }
    }
}
