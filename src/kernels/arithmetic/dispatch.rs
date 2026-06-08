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

//! # **Arithmetic Dispatch Module** - *SIMD/Scalar Dispatch Layer for Arithmetic Operations*
//!
//! High-performance arithmetic kernel dispatcher that automatically selects between SIMD and scalar
//! implementations based on data alignment and feature flags.
//!
//! ## Overview
//! - **Dual-path execution**: SIMD-accelerated path with scalar fallback for unaligned data
//! - **Type-specific dispatch**: Optimised kernels for integers (i32/i64/u32/u64), floats (f32/f64), and datetime types
//! - **Null-aware operations**: Arrow-compatible null mask propagation and handling
//! - **Build-time SIMD lanes**: Lane counts determined at build time based on target architecture
//!
//! ## Supported Operations
//! - **Basic arithmetic**: Add, subtract, multiply, divide, remainder, power
//! - **Fused multiply-add (FMA)**: Hardware-accelerated `a * b + c` operations for floats
//! - **Datetime arithmetic**: Temporal operations with integer kernel delegation
//!
//! ## Performance Strategy
//! - SIMD requires 64-byte aligned input data. This is automatic with Vec64.
//! - Scalar fallback ensures correctness regardless of input alignment

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

#[cfg(feature = "datetime")]
use crate::DatetimeAVT;
#[cfg(feature = "datetime")]
use crate::DatetimeArray;
use crate::enums::error::KernelError;
use crate::enums::operators::ArithmeticOperator::{self};
#[cfg(feature = "simd")]
use crate::kernels::arithmetic::simd::{
    float_dense_body_f32_simd, float_dense_body_f64_simd, float_masked_body_f32_simd,
    float_masked_body_f64_simd, fma_dense_body_f32_simd, fma_dense_body_f64_simd,
    fma_masked_body_f32_simd, fma_masked_body_f64_simd, int_dense_body_simd, int_masked_body_simd,
};
use crate::kernels::arithmetic::std::{
    float_dense_body_std, float_masked_body_std, int_dense_body_std, int_masked_body_std,
};
#[cfg(feature = "datetime")]
use crate::kernels::bitmask::merge_bitmasks_to_new;
use crate::structs::variants::float::FloatArray;
use crate::structs::variants::integer::IntegerArray;
use crate::utils::confirm_equal_len;
#[cfg(feature = "simd")]
use crate::utils::is_simd_aligned;
use crate::{Bitmask, Vec64};
// Kernels

/// Generates element-wise integer arithmetic functions with SIMD/scalar dispatch.
/// Creates functions that operate on `&[T]` slices, returning `IntegerArray<T>` with proper null handling.
/// Automatically selects SIMD path for 64-byte aligned inputs, falls back to scalar otherwise.
macro_rules! impl_apply_int {
    ($fn_name:ident, $ty:ty, $lanes:expr) => {
        #[doc = concat!(
            "Performs element-wise integer `ArithmeticOperator` over two `&[", stringify!($ty),
            "]`, SIMD-accelerated using ", stringify!($lanes), " lanes if available, \
            otherwise falls back to scalar. \
            Returns `IntegerArray<", stringify!($ty), ">` with appropriate null-mask handling."
        )]
        #[inline(always)]
        pub fn $fn_name(
            lhs: &[$ty],
            rhs: &[$ty],
            op: ArithmeticOperator,
            mask: Option<&Bitmask>
        ) -> Result<IntegerArray<$ty>, KernelError> {
            let len = lhs.len();
            confirm_equal_len("apply numeric: length mismatch", len, rhs.len())?;

            #[cfg(feature = "simd")]
            {
                // Check if both arrays are 64-byte aligned for SIMD
                if is_simd_aligned(lhs) && is_simd_aligned(rhs) {
                    // SIMD path - safe because we verified alignment
                    let mut out = Vec64::with_capacity(len);
                    unsafe { out.set_len(len) };
                    match mask {
                        Some(mask) => {
                            let mut out_mask = crate::Bitmask::new_set_all(len, true);
                            int_masked_body_simd::<$ty, $lanes>(op, lhs, rhs, mask, &mut out, &mut out_mask);
                            return Ok(IntegerArray {
                                data: out.into(),
                                null_mask: Some(out_mask),
                            });
                        }
                        None => {
                            int_dense_body_simd::<$ty, $lanes>(op, lhs, rhs, &mut out);
                            return Ok(IntegerArray {
                                data: out.into(),
                                null_mask: None,
                            });
                        }
                    }
                }
                // Fall through to scalar path if alignment check failed
            }

            // Scalar fallback - alignment check failed
            let mut out = Vec64::with_capacity(len);
            unsafe { out.set_len(len) };
            match mask {
                Some(mask) => {
                    let mut out_mask = crate::Bitmask::new_set_all(len, true);
                    int_masked_body_std::<$ty>(op, lhs, rhs, mask, &mut out, &mut out_mask);
                    Ok(IntegerArray {
                        data: out.into(),
                        null_mask: Some(out_mask),
                    })
                }
                None => {
                    int_dense_body_std::<$ty>(op, lhs, rhs, &mut out);
                    Ok(IntegerArray {
                        data: out.into(),
                        null_mask: None,
                    })
                }
            }
        }
    };
}

/// Destination-passing sibling of `impl_apply_int`. Writes the result into
/// the caller-provided `out` slice; the masked path also writes into a
/// caller-provided `out_mask`.
///
/// Uses the same SIMD/scalar dispatch and alignment rules as `impl_apply_int`. The
/// difference is buffer ownership, as here the caller owns it.
macro_rules! impl_apply_int_into {
    ($fn_name:ident, $ty:ty, $lanes:expr) => {
        #[doc = concat!(
            "Destination-passing variant of `apply_int_", stringify!($ty), "`. Writes the \
            element-wise integer `ArithmeticOperator` result directly into `out: &mut [", stringify!($ty),
            "]`. SIMD-accelerated using ", stringify!($lanes), " lanes when both inputs are \
            64-byte aligned; falls back to scalar otherwise. Returns `Ok(())` on success."
        )]
        #[inline(always)]
        pub fn $fn_name(
            lhs: &[$ty],
            rhs: &[$ty],
            op: ArithmeticOperator,
            mask: Option<&Bitmask>,
            out: &mut [$ty],
            out_mask: Option<&mut Bitmask>,
        ) -> Result<(), KernelError> {
            let len = lhs.len();
            confirm_equal_len("apply numeric: length mismatch", len, rhs.len())?;
            confirm_equal_len("apply numeric: out length mismatch", len, out.len())?;

            #[cfg(feature = "simd")]
            {
                if is_simd_aligned(lhs) && is_simd_aligned(rhs) {
                    match (mask, out_mask) {
                        (Some(mask), Some(out_mask)) => {
                            int_masked_body_simd::<$ty, $lanes>(op, lhs, rhs, mask, out, out_mask);
                            return Ok(());
                        }
                        (None, _) => {
                            int_dense_body_simd::<$ty, $lanes>(op, lhs, rhs, out);
                            return Ok(());
                        }
                        (Some(_), None) => {
                            return Err(KernelError::InvalidArguments(
                                "apply_int_into: input mask supplied without an output mask".to_string()
                            ));
                        }
                    }
                }
            }

            match (mask, out_mask) {
                (Some(mask), Some(out_mask)) => {
                    int_masked_body_std::<$ty>(op, lhs, rhs, mask, out, out_mask);
                    Ok(())
                }
                (None, _) => {
                    int_dense_body_std::<$ty>(op, lhs, rhs, out);
                    Ok(())
                }
                (Some(_), None) => Err(KernelError::InvalidArguments(
                    "apply_int_into: input mask supplied without an output mask".to_string()
                )),
            }
        }
    };
}

/// Generates element-wise floating-point arithmetic functions with SIMD/scalar dispatch.
/// Creates functions that operate on `&[T]` slices, returning `FloatArray<T>` with proper null handling.
/// Supports hardware-accelerated operations including FMA when available.
macro_rules! impl_apply_float {
    ($fn_name:ident, $ty:ty, $lanes:expr, $dense_body_simd:ident, $masked_body_simd:ident) => {
        #[doc = concat!(
                    "Performs element-wise float `ArithmeticOperator` on `&[", stringify!($ty),
                    "]` using SIMD (", stringify!($lanes), " lanes) for dense/masked cases,  \
                    Falls back to standard scalar ops when the `simd` feature is not enabled. \
            Returns `FloatArray<", stringify!($ty), ">` and handles optional null-mask."
                )]
        #[inline(always)]
        pub fn $fn_name(
            lhs: &[$ty],
            rhs: &[$ty],
            op: ArithmeticOperator,
            mask: Option<&Bitmask>
        ) -> Result<FloatArray<$ty>, KernelError> {
            let len = lhs.len();
            confirm_equal_len("apply numeric: length mismatch", len, rhs.len())?;

            #[cfg(feature = "simd")]
            {
                // Check if both arrays are 64-byte aligned for SIMD
                if is_simd_aligned(lhs) && is_simd_aligned(rhs) {
                    // SIMD path - safe because we verified alignment
                    let mut out = Vec64::with_capacity(len);
                    unsafe { out.set_len(len) };
                    match mask {
                        Some(mask) => {
                            let mut out_mask = crate::Bitmask::new_set_all(len, true);
                            $masked_body_simd::<$lanes>(op, lhs, rhs, mask, &mut out, &mut out_mask);
                            return Ok(FloatArray {
                                data: out.into(),
                                null_mask: Some(out_mask),
                            });
                        }
                        None => {
                            $dense_body_simd::<$lanes>(op, lhs, rhs, &mut out);
                            return Ok(FloatArray {
                                data: out.into(),
                                null_mask: None,
                            });
                        }
                    }
                }
                // Fall through to scalar path if alignment check failed
            }

            // Scalar fallback - alignment check failed
            let mut out = Vec64::with_capacity(len);
            unsafe { out.set_len(len) };
            match mask {
                Some(mask) => {
                    let mut out_mask = crate::Bitmask::new_set_all(len, true);
                    float_masked_body_std::<$ty>(op, lhs, rhs, mask, &mut out, &mut out_mask);
                    Ok(FloatArray {
                        data: out.into(),
                        null_mask: Some(out_mask),
                    })
                }
                None => {
                    float_dense_body_std::<$ty>(op, lhs, rhs, &mut out);
                    Ok(FloatArray {
                        data: out.into(),
                        null_mask: None,
                    })
                }
            }
        }
    };
}

/// Destination-passing sibling of `impl_apply_float`. Writes the result into
/// the caller-provided `out` slice; the masked path also writes into a
/// caller-provided `out_mask`.
macro_rules! impl_apply_float_into {
    ($fn_name:ident, $ty:ty, $lanes:expr, $dense_body_simd:ident, $masked_body_simd:ident) => {
        #[doc = concat!(
            "Destination-passing variant of `apply_float_", stringify!($ty), "`. Writes the \
            element-wise float `ArithmeticOperator` result directly into `out: &mut [", stringify!($ty),
            "]`. SIMD-accelerated using ", stringify!($lanes), " lanes when both inputs are \
            64-byte aligned; falls back to scalar otherwise. Returns `Ok(())` on success."
        )]
        #[inline(always)]
        pub fn $fn_name(
            lhs: &[$ty],
            rhs: &[$ty],
            op: ArithmeticOperator,
            mask: Option<&Bitmask>,
            out: &mut [$ty],
            out_mask: Option<&mut Bitmask>,
        ) -> Result<(), KernelError> {
            let len = lhs.len();
            confirm_equal_len("apply numeric: length mismatch", len, rhs.len())?;
            confirm_equal_len("apply numeric: out length mismatch", len, out.len())?;

            #[cfg(feature = "simd")]
            {
                if is_simd_aligned(lhs) && is_simd_aligned(rhs) {
                    match (mask, out_mask) {
                        (Some(mask), Some(out_mask)) => {
                            $masked_body_simd::<$lanes>(op, lhs, rhs, mask, out, out_mask);
                            return Ok(());
                        }
                        (None, _) => {
                            $dense_body_simd::<$lanes>(op, lhs, rhs, out);
                            return Ok(());
                        }
                        (Some(_), None) => {
                            return Err(KernelError::InvalidArguments(
                                "apply_float_into: input mask supplied without an output mask".to_string()
                            ));
                        }
                    }
                }
            }

            match (mask, out_mask) {
                (Some(mask), Some(out_mask)) => {
                    float_masked_body_std::<$ty>(op, lhs, rhs, mask, out, out_mask);
                    Ok(())
                }
                (None, _) => {
                    float_dense_body_std::<$ty>(op, lhs, rhs, out);
                    Ok(())
                }
                (Some(_), None) => Err(KernelError::InvalidArguments(
                    "apply_float_into: input mask supplied without an output mask".to_string()
                )),
            }
        }
    };
}

/// Generates fused multiply-add (FMA) functions with SIMD/scalar dispatch.
/// Creates `a * b + c` operations on `&[T]` slices, returning `FloatArray<T>`.
/// Uses `mul_add()`, which leverages hardware FMA when available.
macro_rules! impl_apply_fma_float {
    ($fn_name:ident, $ty:ty, $lanes:expr, $dense_simd:ident, $masked_simd:ident) => {
        #[doc = concat!(
            "Performs element-wise fused multiply-add (`a * b + acc`) on `&[", stringify!($ty),
            "]` using SIMD (", stringify!($lanes), " lanes; dense or masked, via `",
            stringify!($dense), "`/`", stringify!($masked), "` as needed. \
            Falls back to standard scalar ops when the `simd` feature is not enabled. \
            Results in a `FloatArray<", stringify!($ty), ">`."
        )]
        #[inline(always)]
        pub fn $fn_name(
            lhs: &[$ty],
            rhs: &[$ty],
            acc: &[$ty],
            mask: Option<&Bitmask>
        ) -> Result<FloatArray<$ty>, KernelError> {
            let len = lhs.len();
            confirm_equal_len("apply numeric: length mismatch", len, rhs.len())?;
            confirm_equal_len("acc length mismatch", len, acc.len())?;

            let mut out = Vec64::with_capacity(len);
            unsafe { out.set_len(len) };
            let mut out_mask = crate::Bitmask::new_set_all(len, true);

            #[cfg(feature = "simd")]
            {
                // Check if all arrays are properly aligned for SIMD (cheap runtime check)
                if is_simd_aligned(lhs) && is_simd_aligned(rhs) && is_simd_aligned(acc) {
                    // SIMD path - safe because we verified alignment
                    match mask {
                        Some(mask) => {
                            $masked_simd::<$lanes>(lhs, rhs, acc, mask, &mut out, &mut out_mask);
                            return Ok(FloatArray {
                                data: out.into(),
                                null_mask: Some(out_mask),
                            });
                        }
                        None => {
                            $dense_simd::<$lanes>(lhs, rhs, acc, &mut out);
                            return Ok(FloatArray {
                                data: out.into(),
                                null_mask: None,
                            });
                        }
                    }
                }
                // Fall through to scalar path if alignment check failed
            }

            // Scalar fallback - alignment check failed
            match mask {
                Some(mask) => {
                    // Masked FMA: a * b + acc with null handling
                    for i in 0..len {
                        if unsafe { mask.get_unchecked(i) } {
                            out[i] = lhs[i] * rhs[i] + acc[i];
                        } else {
                            out[i] = 0 as $ty;  // Initialize masked values to zero
                            out_mask.set(i, false);
                        }
                    }
                    Ok(FloatArray {
                        data: out.into(),
                        null_mask: Some(out_mask),
                    })
                }
                None => {
                    // Dense FMA: a * b + acc
                    for i in 0..len {
                        out[i] = lhs[i] * rhs[i] + acc[i];
                    }
                    Ok(FloatArray {
                        data: out.into(),
                        null_mask: None,
                    })
                }
            }
        }
    };
}

/// Performs element-wise arithmetic between two `DatetimeArray<T>`s with SIMD/scalar fallback,
/// using the standard integer SIMD/scalar kernels for the underlying data.
///
/// Returns `DatetimeArray<T>` with correct null propagation.
///
/// # Supported operations
/// - Add, Subtract, Multiply, Divide, Remainder
/// - Power: defined as left-value (lhs) preserved (see notes)
///
/// # Notes
/// - **"Power" for dates/times is undefined: returns lhs unchanged**.
/// - All other ops delegate directly to the integer kernels for correctness/performance.
/// - Any future date-specific ops should be implemented in the stub below.
/// Generates datetime arithmetic functions with SIMD/scalar dispatch.
/// Creates functions that operate on `DatetimeArray<T>` types, delegating to integer kernels
/// for the underlying temporal data while preserving datetime semantics.
#[cfg(feature = "datetime")]
macro_rules! impl_apply_datetime {
    ($fn_name:ident, $ty:ty, $lanes:expr) => {
        #[inline(always)]
        pub fn $fn_name(
            lhs: DatetimeAVT<$ty>,
            rhs: DatetimeAVT<$ty>,
            op: ArithmeticOperator,
        ) -> Result<DatetimeArray<$ty>, KernelError> {
            let (larr, loff, llen) = lhs;
            let (rarr, roff, rlen) = rhs;
            confirm_equal_len("apply_datetime: length mismatch", llen, rlen)?;

            let out_mask =
                merge_bitmasks_to_new(larr.null_mask.as_ref(), rarr.null_mask.as_ref(), llen);
            let ldata = &larr.data[loff..loff + llen];
            let rdata = &rarr.data[roff..roff + rlen];

            let mut out = Vec64::<$ty>::with_capacity(llen);
            unsafe {
                out.set_len(llen);
            }

            match out_mask.as_ref() {
                Some(mask) => {
                    let mut result_mask = crate::Bitmask::new_set_all(llen, true);
                    #[cfg(feature = "simd")]
                    {
                        int_masked_body_simd::<$ty, $lanes>(
                            op,
                            ldata,
                            rdata,
                            mask,
                            &mut out,
                            &mut result_mask,
                        );
                    }
                    #[cfg(not(feature = "simd"))]
                    {
                        int_masked_body_std::<$ty>(
                            op,
                            ldata,
                            rdata,
                            mask,
                            &mut out,
                            &mut result_mask,
                        );
                    }
                    Ok(DatetimeArray::from_vec64(out, Some(result_mask), None))
                }
                None => {
                    #[cfg(feature = "simd")]
                    {
                        int_dense_body_simd::<$ty, $lanes>(op, ldata, rdata, &mut out);
                    }
                    #[cfg(not(feature = "simd"))]
                    {
                        int_dense_body_std::<$ty>(op, ldata, rdata, &mut out);
                    }
                    Ok(DatetimeArray::from_vec64(out, None, None))
                }
            }
        }
    };
}

// Generates i32, u32, i64, u64, f32, f64 variants using lane counts via simd_lanes.rs

impl_apply_int!(apply_int_i32, i32, W32);
impl_apply_int!(apply_int_u32, u32, W32);
impl_apply_int!(apply_int_i64, i64, W64);
impl_apply_int!(apply_int_u64, u64, W64);
#[cfg(feature = "extended_numeric_types")]
impl_apply_int!(apply_int_i16, i16, W16);
#[cfg(feature = "extended_numeric_types")]
impl_apply_int!(apply_int_u16, u16, W16);
#[cfg(feature = "extended_numeric_types")]
impl_apply_int!(apply_int_i8, i8, W8);
#[cfg(feature = "extended_numeric_types")]
impl_apply_int!(apply_int_u8, u8, W8);

impl_apply_int_into!(apply_int_i32_into, i32, W32);
impl_apply_int_into!(apply_int_u32_into, u32, W32);
impl_apply_int_into!(apply_int_i64_into, i64, W64);
impl_apply_int_into!(apply_int_u64_into, u64, W64);
#[cfg(feature = "extended_numeric_types")]
impl_apply_int_into!(apply_int_i16_into, i16, W16);
#[cfg(feature = "extended_numeric_types")]
impl_apply_int_into!(apply_int_u16_into, u16, W16);
#[cfg(feature = "extended_numeric_types")]
impl_apply_int_into!(apply_int_i8_into, i8, W8);
#[cfg(feature = "extended_numeric_types")]
impl_apply_int_into!(apply_int_u8_into, u8, W8);

impl_apply_float!(
    apply_float_f32,
    f32,
    W32,
    float_dense_body_f32_simd,
    float_masked_body_f32_simd
);
impl_apply_float!(
    apply_float_f64,
    f64,
    W64,
    float_dense_body_f64_simd,
    float_masked_body_f64_simd
);

impl_apply_float_into!(
    apply_float_f32_into,
    f32,
    W32,
    float_dense_body_f32_simd,
    float_masked_body_f32_simd
);
impl_apply_float_into!(
    apply_float_f64_into,
    f64,
    W64,
    float_dense_body_f64_simd,
    float_masked_body_f64_simd
);

impl_apply_fma_float!(
    apply_fma_f32,
    f32,
    W32,
    fma_dense_body_f32_simd,
    fma_masked_body_f32_simd
);

impl_apply_fma_float!(
    apply_fma_f64,
    f64,
    W64,
    fma_dense_body_f64_simd,
    fma_masked_body_f64_simd
);

#[cfg(feature = "datetime")]
impl_apply_datetime!(apply_datetime_i32, i32, W32);
#[cfg(feature = "datetime")]
impl_apply_datetime!(apply_datetime_u32, u32, W32);
#[cfg(feature = "datetime")]
impl_apply_datetime!(apply_datetime_i64, i64, W64);
#[cfg(feature = "datetime")]
impl_apply_datetime!(apply_datetime_u64, u64, W64);
