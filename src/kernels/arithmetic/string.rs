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

//! # **String Arithmetic Module** - *String Operations with Numeric Interactions*
//!
//! String-specific arithmetic operations including string multiplication, concatenation, and manipulation.
//! This unifies strings into a typical numeric-compatible workloads. E.g., "hello" + "there" = "hellothere".
//! These are opt-in via the "str_arithmetic" feature.
//!
//! ## Overview
//! - **String multiplication**: Repeat strings by numeric factors with configurable limits
//! - **String-numeric conversions**: Format numbers into string representations  
//! - **Categorical operations**: Efficient string deduplication and categorical array generation
//! - **Null-aware processing**: Full Arrow-compatible null propagation
//!
//! ## Features
//! - **Memory efficiency**: Uses string interning and categorical encoding to reduce allocation overhead
//! - **Safety limits**: Configurable multiplication limits prevent excessive memory usage
//! - **Optional dependencies**: String-numeric arithmetic gated behind `str_arithmetic` feature

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

#[cfg(feature = "fast_hash")]
use ahash::AHashMap;
#[cfg(feature = "str_arithmetic")]
use core::ptr::copy_nonoverlapping;
#[cfg(not(feature = "fast_hash"))]
use std::collections::HashMap;

use crate::enums::error::{KernelError, log_length_mismatch};
use crate::kernels::bitmask::merge_bitmasks_to_new;
#[cfg(feature = "shared_dict")]
use crate::structs::dictionary::Dictionary;
use crate::structs::variants::categorical::CategoricalArray;
#[cfg(feature = "str_arithmetic")]
use memchr::memmem::Finder;

use crate::structs::variants::string::StringArray;
use crate::traits::type_unions::Integer;
use crate::{Bitmask, Vec64};
#[cfg(feature = "str_arithmetic")]
use num_traits::ToPrimitive;

use crate::enums::operators::ArithmeticOperator::{self};
#[cfg(feature = "str_arithmetic")]
use crate::utils::{
    confirm_mask_capacity, estimate_categorical_cardinality, estimate_string_cardinality,
};
use crate::{CategoricalAVT, StringAVTExt};
#[cfg(feature = "str_arithmetic")]
use crate::{MaskedArray, StringAVT};

#[cfg(feature = "str_arithmetic")]
use ryu::Float;
#[cfg(feature = "str_arithmetic")]
use std::mem::MaybeUninit;

/// Maximum allowed repetitions for string multiplication operations.
///
/// Prevents excessive memory allocation when repeating strings through multiplication.
/// Operations exceeding this limit will return an error rather than allocating unbounded memory.
pub const STRING_MULTIPLICATION_LIMIT: usize = 1_000_000;

/// Threshold for dictionary size checks in categorical array operations.
///
/// Controls when the `cmp_dict_in` function in `kernels/logical.rs` switches from dictionary
/// lookups to direct string comparisons. Arrays with fewer unique values than this threshold
/// use optimised dictionary-based comparisons.
pub const MAX_DICT_CHECK: usize = 256;

/// String-numeric arithmetic operation dispatcher.
/// Supports string multiplication (repeat string N times) with safety limits.
/// Other operations pass through unchanged. Handles null propagation correctly.
///
/// # Type Parameters
/// - `T`: Offset type for the input array (e.g., `u32`, `u64`)
/// - `N`: Numeric type convertible to `usize`
/// - `O`: Offset type for the output array
pub fn apply_str_num<T, N, O>(
    lhs: StringAVTExt<T>,
    rhs: &[N],
    op: ArithmeticOperator,
) -> Result<StringArray<O>, KernelError>
where
    T: Integer,
    N: num_traits::ToPrimitive + Copy,
    O: Integer + num_traits::NumCast,
{
    let (array, offset, logical_len, physical_bytes_len) = lhs;

    if logical_len != rhs.len() {
        return Err(KernelError::LengthMismatch(log_length_mismatch(
            "apply_str_num".to_string(),
            logical_len,
            rhs.len(),
        )));
    }

    // Preallocate offsets and estimate capacity
    let lhs_mask = array.null_mask.as_ref();
    let mut out_mask = lhs_mask.map(|_| crate::Bitmask::new_set_all(logical_len, true));

    let mut offsets = Vec64::<O>::with_capacity(logical_len + 1);
    offsets.push(O::zero()); // initial offset 0

    let estimated_bytes = physical_bytes_len.min(STRING_MULTIPLICATION_LIMIT * logical_len);
    let mut data = Vec64::with_capacity(estimated_bytes);

    for (out_idx, i) in (offset..offset + logical_len).enumerate() {
        let valid = lhs_mask.map_or(true, |mask| unsafe { mask.get_unchecked(i) });

        if let Some(mask) = &mut out_mask {
            unsafe { mask.set_unchecked(out_idx, valid) };
        }

        if valid {
            let s = unsafe { array.get_str_unchecked(i) };
            let n = rhs[out_idx].to_usize().unwrap_or(0);

            match op {
                ArithmeticOperator::Multiply => {
                    let count = n.min(STRING_MULTIPLICATION_LIMIT);
                    for _ in 0..count {
                        data.extend_from_slice(s.as_bytes());
                    }
                }
                _ => {
                    data.extend_from_slice(s.as_bytes());
                }
            }
        }

        // Push offset regardless of validity to keep offsets aligned
        // This ensures we can still slice [a..b] intuitively.
        let new_offset = O::from(data.len()).expect("offset conversion overflow");
        offsets.push(new_offset);
    }

    Ok(StringArray {
        offsets: offsets.into(),
        data: data.into(),
        null_mask: out_mask,
    })
}

/// Applies an element-wise binary operation between a `StringArray<T>` and a slice of floating-point values,
/// producing a new `StringArray<T>`. Each operation is performed by interpreting the float as a finite
/// decimal string representation (`f64` formatted with `ryu`), and applying string transformations accordingly.
///
/// Supported operations:
///
/// - `Add`: Appends the stringified float to the string from `lhs`.
/// - `Subtract`: Removes the first occurrence of the stringified float from `lhs`, if present.
///               If not found, `lhs` is returned unchanged.
/// - `Multiply`: Repeats the `lhs` string `N` times, where `N = abs(round(rhs)) % (STRING_MULTIPLICATION_LIMIT + 1)`.
/// - `Divide`: Splits the `lhs` string by the stringified float, and joins the segments using `'|'`.
///             If the float pattern is not found, the original string is returned unchanged.
///
/// Null handling:
/// - If the `lhs` value is null at an index, the result is null at that index.
/// - The float operand cannot be null; the caller must guarantee its presence.
///
/// Output:
/// - A new `StringArray<T>` with the same length as `lhs`.
/// - The underlying byte storage is preallocated based on a prepass analysis of required capacity.
///
/// Errors:
/// - Returns `KernelError::LengthMismatch` if `lhs` and `rhs` lengths differ.
/// - Returns `KernelError::UnsupportedType` if the operator is not one of Add, Subtract, Multiply, Divide.
///
/// # Features
/// This function is only available when the `str_arithmetic` feature is enabled.
///
/// This kernel is optional as it pulls in external dependencies, and fits more niche use cases. It's a
/// good fit for flex-typing scenarios where users are concatenating strings and numbers, or working
/// with semi-structured web content, string formatting pipelines etc.
///
/// # Safety
/// - Uses unchecked access and raw pointer copies for performance. Invariants around memory safety must hold.
/// - Assumes `rhs` contains only finite floating-point values.
#[cfg(feature = "str_arithmetic")]
pub fn apply_str_float<T, F>(
    lhs: StringAVT<T>,
    rhs: &[F],
    op: ArithmeticOperator,
) -> Result<StringArray<T>, KernelError>
where
    T: Integer,
    F: Into<f64> + Copy + ryu::Float,
{
    // Destructure the string slice: array, offset, and logical length
    let (array, offset, logical_len) = lhs;

    // Validate inputs

    use std::mem::MaybeUninit;
    if rhs.len() != logical_len {
        return Err(KernelError::LengthMismatch(log_length_mismatch(
            "apply_str_float".into(),
            logical_len,
            rhs.len(),
        )));
    }
    let lhs_mask = &array.null_mask;
    let _ = confirm_mask_capacity(array.len(), lhs_mask.as_ref())?;

    // 1st pass: size accounting
    let mut total_bytes = 0usize;
    let mut fmt_buf: [MaybeUninit<u8>; 24] = unsafe { MaybeUninit::uninit().assume_init() };

    for (out_idx, i) in (offset..offset + logical_len).enumerate() {
        if !lhs_mask
            .as_ref()
            .map_or(true, |m| unsafe { m.get_unchecked(i) })
        {
            continue;
        }

        // src_len = string length at physical index
        let src_len = {
            let a = array.offsets[i].to_usize();
            let b = array.offsets[i + 1].to_usize();
            b - a
        };
        let n_s = format_finite(&mut fmt_buf, rhs[out_idx]);
        total_bytes += match op {
            ArithmeticOperator::Add => src_len + n_s.len(),
            ArithmeticOperator::Subtract => src_len,
            ArithmeticOperator::Multiply => {
                let times =
                    rhs[out_idx].into().round().abs() as usize % (STRING_MULTIPLICATION_LIMIT + 1);
                src_len * times
            }
            ArithmeticOperator::Divide => {
                let pat_len = n_s.len();
                let splits = (src_len + pat_len).saturating_sub(1) / pat_len;
                src_len + splits
            }
            _ => {
                return Err(KernelError::UnsupportedType(format!(
                    "Unsupported {:?}",
                    op
                )));
            }
        };
    }

    // allocate outputs once
    let mut offsets = Vec64::<T>::with_capacity(logical_len + 1);

    // 2nd pass: copy / build strings
    let mut data = Vec64::<u8>::with_capacity(total_bytes);
    unsafe {
        offsets.set_len(logical_len + 1);
        data.set_len(total_bytes);
    }

    let mut out_mask = lhs_mask
        .as_ref()
        .map(|_| Bitmask::new_set_all(logical_len, false));

    let mut cursor = 0usize;
    offsets[0] = T::zero();

    for (out_idx, i) in (offset..offset + logical_len).enumerate() {
        let valid = lhs_mask
            .as_ref()
            .map_or(true, |m| unsafe { m.get_unchecked(i) });
        if let Some(mask) = &mut out_mask {
            unsafe { mask.set_unchecked(out_idx, valid) };
        }

        if !valid {
            offsets[out_idx + 1] = T::from(cursor).unwrap();
            continue;
        }

        let start = array.offsets[i].to_usize();
        let end = array.offsets[i + 1].to_usize();
        let src = &array.data[start..end];
        let n_s = format_finite(&mut fmt_buf, rhs[out_idx]);
        let pat = n_s.as_bytes();

        let mut write = |bytes: &[u8]| unsafe {
            copy_nonoverlapping(bytes.as_ptr(), data.as_mut_ptr().add(cursor), bytes.len());
            cursor += bytes.len();
        };

        match op {
            ArithmeticOperator::Add => {
                write(src);
                write(pat);
            }
            ArithmeticOperator::Subtract => {
                if let Some(idx) = Finder::new(pat).find(src) {
                    write(&src[..idx]);
                    write(&src[(idx + pat.len())..]);
                } else {
                    write(src);
                }
            }
            ArithmeticOperator::Multiply => {
                let times =
                    rhs[out_idx].into().round().abs() as usize % (STRING_MULTIPLICATION_LIMIT + 1);
                for _ in 0..times {
                    write(src);
                }
            }
            ArithmeticOperator::Divide => {
                let finder = Finder::new(pat);
                let mut start_pos = 0;
                let mut first = true;
                while let Some(idx) = finder.find(&src[start_pos..]) {
                    if !first {
                        data[cursor] = b'|';
                        cursor += 1;
                    }
                    let rel_idx = idx;
                    let segment = &src[start_pos..start_pos + rel_idx];
                    unsafe {
                        copy_nonoverlapping(
                            segment.as_ptr(),
                            data.as_mut_ptr().add(cursor),
                            segment.len(),
                        );
                        cursor += segment.len();
                    }
                    start_pos += rel_idx + pat.len();
                    first = false;
                }
                if !first {
                    data[cursor] = b'|';
                    cursor += 1;
                }
                let tail = &src[start_pos..];
                unsafe {
                    copy_nonoverlapping(tail.as_ptr(), data.as_mut_ptr().add(cursor), tail.len());
                    cursor += tail.len();
                }
            }
            _ => unreachable!(),
        }
        offsets[out_idx + 1] = T::from(cursor).unwrap();
    }

    // build & return

    Ok(StringArray {
        offsets: offsets.into(),
        data: data.into(),
        null_mask: out_mask,
    })
}

/// String interning helper for categorical array generation with fast hashing.
/// Deduplicates strings and assigns numeric codes for memory efficiency.
#[cfg(feature = "fast_hash")]
#[inline(always)]
fn intern<T: Integer>(s: &str, dict: &mut AHashMap<String, T>, uniq: &mut Vec64<String>) -> T {
    if let Some(&code) = dict.get(s) {
        code
    } else {
        let idx = T::from_usize(uniq.len());
        uniq.push(s.to_owned());
        dict.insert(s.to_owned(), idx);
        idx
    }
}

/// String interning for categorical array generation with standard hashing.
/// Deduplicates strings and assigns numeric codes for memory efficiency.
#[cfg(not(feature = "fast_hash"))]
#[inline(always)]
fn intern<T: Integer>(s: &str, dict: &mut HashMap<String, T>, uniq: &mut Vec64<String>) -> T {
    if let Some(&code) = dict.get(s) {
        code
    } else {
        let idx = T::from_usize(uniq.len());
        uniq.push(s.to_owned());
        dict.insert(s.to_owned(), idx);
        idx
    }
}

/// Applies an element-wise binary operation between two `CategoricalArray<T>` arrays,
/// producing a new `CategoricalArray<T>`. The result reuses or extends the unified dictionary
/// from both input arrays and ensures deterministic interned value codes.
///
/// Supported operations:
///
/// - `Add`: Concatenates strings from `lhs` and `rhs`. Result is interned into the output dictionary.
/// - `Subtract`: Removes the first occurrence of `rhs` from `lhs`. If `rhs` is empty or not found,
///               returns `lhs` unchanged.
/// - `Multiply`: Returns `lhs` unchanged. No actual repetition occurs - identity operation.
/// - `Divide`: Splits `lhs` by occurrences of `rhs`, and each resulting segment is interned separately.
///             If `rhs` is empty, `lhs` is returned unchanged.
///
/// Null handling:
/// - If either side is null at a given index, the result is marked null and the empty string code is emitted.
/// - Null mask is propagated accordingly.
///
/// Output:
/// - The resulting `CategoricalArray<T>` may have a different length than the input
///   if `Divide` produces multiple segments per row.
/// - The dictionary (`unique_values`) is the union of all unique values observed in inputs and results,
///   with stable interned codes.
///
/// Errors:
/// - Returns `KernelError::LengthMismatch` if `lhs` and `rhs` differ in length.
/// - Returns `KernelError::UnsupportedType` for any operator other than Add, Subtract, Multiply, Divide.
///
/// # Panics
/// - Panics if internal memory allocation fails or if invariants are violated in unsafe regions.
/// - Panics if the dictionary grows beyond `T::MAX` entries (relevant for small index types like `u8`).
fn apply_dict_dict_impl<T: Integer>(
    lhs: CategoricalAVT<T>,
    rhs: CategoricalAVT<T>,
    op: ArithmeticOperator,
) -> Result<CategoricalArray<T>, KernelError> {
    // Destructure slice tuples for offset/length-local processing
    let (lhs_array, lhs_offset, lhs_logical_len) = lhs;
    let (rhs_array, rhs_offset, rhs_logical_len) = rhs;

    if lhs_logical_len != rhs_logical_len {
        return Err(KernelError::LengthMismatch(log_length_mismatch(
            "apply_dict_dict_impl".into(),
            lhs_logical_len,
            rhs_logical_len,
        )));
    }

    // Input mask: merge only over local window
    let in_mask = merge_bitmasks_to_new(
        lhs_array.null_mask.as_ref(),
        rhs_array.null_mask.as_ref(),
        lhs_logical_len,
    );

    // Build unique dictionary for the output, initially union of both inputs
    let mut uniq: Vec64<String> = Vec64::with_capacity(
        lhs_array.unique_values().len() + rhs_array.unique_values().len() + lhs_logical_len,
    );

    #[cfg(feature = "fast_hash")]
    let mut dict: AHashMap<String, T> = AHashMap::with_capacity(uniq.capacity());

    #[cfg(not(feature = "fast_hash"))]
    let mut dict: HashMap<String, T> = HashMap::with_capacity(uniq.capacity());

    for v in lhs_array
        .unique_values()
        .iter()
        .chain(rhs_array.unique_values().iter())
    {
        if !dict.contains_key(v) {
            let idx = T::from_usize(uniq.len());
            uniq.push(v.clone());
            dict.insert(uniq.last().unwrap().clone(), idx);
        }
    }

    // Ensure "" is present and get its code
    let empty_code = *dict.entry("".to_owned()).or_insert_with(|| {
        let idx = T::from_usize(uniq.len());
        uniq.push("".to_owned());
        idx
    });

    // 1st pass: Count output rows for precise allocation
    let mut total_out = 0usize;
    for local_idx in 0..lhs_logical_len {
        let i = lhs_offset + local_idx;
        let j = rhs_offset + local_idx;
        let valid = in_mask
            .as_ref()
            .map_or(true, |m| unsafe { m.get_unchecked(local_idx) });
        if !valid {
            total_out += 1;
        } else if let ArithmeticOperator::Divide = op {
            let l = unsafe { lhs_array.get_str_unchecked(i) };
            let r = unsafe { rhs_array.get_str_unchecked(j) };
            if r.is_empty() {
                total_out += 1;
            } else {
                let mut parts = 0;
                let mut start = 0;
                while let Some(pos) = l[start..].find(r) {
                    parts += 1;
                    start += pos + r.len();
                }
                total_out += parts + 1;
            }
        } else {
            total_out += 1;
        }
    }

    // Preallocate output buffer for window only
    let mut out_data = Vec64::with_capacity(total_out);
    unsafe {
        out_data.set_len(total_out);
    }
    let mut out_mask = Bitmask::new_set_all(total_out, false);

    // 2nd pass: Populate output for this slice only
    let mut write_ptr = 0;
    for local_idx in 0..lhs_logical_len {
        let i = lhs_offset + local_idx;
        let j = rhs_offset + local_idx;
        let valid = in_mask
            .as_ref()
            .map_or(true, |m| unsafe { m.get_unchecked(local_idx) });

        if !valid {
            out_data.push(empty_code);
            unsafe { out_mask.set_unchecked(write_ptr, false) };
            write_ptr += 1;
            continue;
        }

        let l = unsafe { lhs_array.get_str_unchecked(i) };
        let r = unsafe { rhs_array.get_str_unchecked(j) };

        match op {
            ArithmeticOperator::Add => {
                let mut tmp = String::with_capacity(l.len() + r.len());
                tmp.push_str(l);
                tmp.push_str(r);
                let code = intern(&tmp, &mut dict, &mut uniq);
                unsafe {
                    *out_data.get_unchecked_mut(write_ptr) = code;
                }
                out_mask.set(write_ptr, true);
                write_ptr += 1;
            }
            ArithmeticOperator::Subtract => {
                let result = if r.is_empty() {
                    l.to_owned()
                } else if let Some(pos) = l.find(r) {
                    let mut tmp = String::with_capacity(l.len() - r.len());
                    tmp.push_str(&l[..pos]);
                    tmp.push_str(&l[pos + r.len()..]);
                    tmp
                } else {
                    l.to_owned()
                };
                let code = intern(&result, &mut dict, &mut uniq);
                unsafe {
                    *out_data.get_unchecked_mut(write_ptr) = code;
                }
                out_mask.set(write_ptr, true);
                write_ptr += 1;
            }
            ArithmeticOperator::Multiply => {
                let code = intern(l, &mut dict, &mut uniq);
                unsafe {
                    *out_data.get_unchecked_mut(write_ptr) = code;
                }
                out_mask.set(write_ptr, true);
                write_ptr += 1;
            }
            ArithmeticOperator::Divide => {
                if r.is_empty() {
                    let code = intern(l, &mut dict, &mut uniq);
                    unsafe {
                        *out_data.get_unchecked_mut(write_ptr) = code;
                    }
                    out_mask.set(write_ptr, true);
                    write_ptr += 1;
                } else {
                    let mut start = 0;
                    while let Some(pos) = l[start..].find(r) {
                        let part = &l[start..start + pos];
                        let code = intern(part, &mut dict, &mut uniq);
                        unsafe {
                            *out_data.get_unchecked_mut(write_ptr) = code;
                        }
                        out_mask.set(write_ptr, true);
                        write_ptr += 1;
                        start += pos + r.len();
                    }
                    let tail = &l[start..];
                    let code = intern(tail, &mut dict, &mut uniq);
                    unsafe {
                        *out_data.get_unchecked_mut(write_ptr) = code;
                    }
                    out_mask.set(write_ptr, true);
                    write_ptr += 1;
                }
            }
            _ => {
                return Err(KernelError::UnsupportedType(format!(
                    "Unsupported apply_dict_dict_impl op={:?}",
                    op
                )));
            }
        }
    }

    debug_assert_eq!(write_ptr, total_out);

    Ok(CategoricalArray {
        data: out_data.into(),
        #[cfg(not(feature = "shared_dict"))]
        unique_values: uniq,
        #[cfg(feature = "shared_dict")]
        dictionary: Dictionary::from(uniq),
        null_mask: Some(out_mask),
    })
}

/// Applies an element-wise binary operation between two `StringArray`s,
/// producing a new `StringArray`. Requires both arrays to have the same length.
///
/// Supported operations:
///
/// - `Add`: Concatenates each pair of strings (`a + b`).
/// - `Subtract`: Removes the first occurrence of `b` from `a`, if present.
///               If `b` is empty or not found, `a` is returned unchanged.
/// - `Multiply`: Repeats string `a` N times, where `N = min(b.len(), STRING_MULTIPLICATION_LIMIT)`.
/// - `Divide`: Splits string `a` by occurrences of `b` and rejoins the segments using a `'|'` separator.
///             If `b` is empty, returns `a` unchanged.
///
/// Null handling:
/// - If either side is null at an index, the output will be null at that index.
///
/// Returns:
/// - A new `StringArray<T>` containing the result of applying the binary operation to each pair.
///
/// Errors:
/// - Returns `KernelError::LengthMismatch` if `lhs` and `rhs` lengths differ.
/// - Returns `KernelError::UnsupportedType` if an unsupported binary operator is passed.
///
/// # Features
/// This function is available only when the `str_arithmetic` feature is enabled.
#[cfg(feature = "str_arithmetic")]
pub fn apply_str_str<T, U>(
    lhs: StringAVT<T>,
    rhs: StringAVT<U>,
    op: ArithmeticOperator,
) -> Result<StringArray<T>, KernelError>
where
    T: Integer,
    U: Integer,
{
    let (larr, loff, llen) = lhs;
    let (rarr, roff, rlen) = rhs;

    if llen != rlen {
        return Err(KernelError::LengthMismatch(log_length_mismatch(
            "apply_str_str".to_string(),
            llen,
            rlen,
        )));
    }

    // slice the incoming masks down to [offset .. offset+llen)
    let lmask_slice = larr.null_mask.as_ref().map(|m| {
        let mut m2 = Bitmask::new_set_all(llen, true);
        for i in 0..llen {
            unsafe {
                m2.set_unchecked(i, m.get_unchecked(loff + i));
            }
        }
        m2
    });
    let rmask_slice = rarr.null_mask.as_ref().map(|m| {
        let mut m2 = Bitmask::new_set_all(llen, true);
        for i in 0..llen {
            unsafe {
                m2.set_unchecked(i, m.get_unchecked(roff + i));
            }
        }
        m2
    });
    let lmask_ref = lmask_slice.as_ref();
    let rmask_ref = rmask_slice.as_ref();

    // build per-position validity
    let lmask = lmask_ref;
    let rmask = rmask_ref;
    let mut out_mask = Bitmask::new_set_all(llen, false);
    let _ = confirm_mask_capacity(llen, lmask)?;
    let _ = confirm_mask_capacity(llen, rmask)?;

    // 1) size pass
    let mut total_bytes = 0;
    for idx in 0..llen {
        let valid = lmask.map_or(true, |m| unsafe { m.get_unchecked(idx) })
            && rmask.map_or(true, |m| unsafe { m.get_unchecked(idx) });
        if !valid {
            continue;
        }
        let a = unsafe { larr.get_str_unchecked(loff + idx) };
        let b = unsafe { rarr.get_str_unchecked(roff + idx) };
        total_bytes += match op {
            ArithmeticOperator::Add => a.len() + b.len(),
            ArithmeticOperator::Subtract => a.len(),
            ArithmeticOperator::Multiply => a.len() * b.len().min(STRING_MULTIPLICATION_LIMIT),
            ArithmeticOperator::Divide => {
                if b.is_empty() {
                    a.len()
                } else {
                    a.len() + a.matches(b).count().saturating_sub(1)
                }
            }
            _ => {
                return Err(KernelError::UnsupportedType(format!(
                    "Unsupported {:?}",
                    op
                )));
            }
        };
    }

    // 2) allocate
    let mut offsets = Vec64::<T>::with_capacity(llen + 1);
    let mut data = Vec64::<u8>::with_capacity(total_bytes);
    offsets.push(T::zero());

    // 3) build pass
    for idx in 0..llen {
        let valid = lmask.map_or(true, |m| unsafe { m.get_unchecked(idx) })
            && rmask.map_or(true, |m| unsafe { m.get_unchecked(idx) });
        if valid {
            let a = unsafe { larr.get_str_unchecked(loff + idx) };
            let b = unsafe { rarr.get_str_unchecked(roff + idx) };
            match op {
                ArithmeticOperator::Add => {
                    data.extend_from_slice(a.as_bytes());
                    data.extend_from_slice(b.as_bytes());
                }
                ArithmeticOperator::Subtract => {
                    if b.is_empty() {
                        data.extend_from_slice(a.as_bytes());
                    } else if let Some(p) =
                        memchr::memmem::Finder::new(b.as_bytes()).find(a.as_bytes())
                    {
                        data.extend_from_slice(&a.as_bytes()[..p]);
                        data.extend_from_slice(&a.as_bytes()[p + b.len()..]);
                    } else {
                        data.extend_from_slice(a.as_bytes());
                    }
                }
                ArithmeticOperator::Multiply => {
                    let times = b.len().min(STRING_MULTIPLICATION_LIMIT);
                    for _ in 0..times {
                        data.extend_from_slice(a.as_bytes());
                    }
                }
                ArithmeticOperator::Divide => {
                    if b.is_empty() {
                        data.extend_from_slice(a.as_bytes());
                    } else {
                        let finder = memchr::memmem::Finder::new(b.as_bytes());
                        let mut start = 0;
                        let mut first = true;
                        while let Some(p) = finder.find(&a.as_bytes()[start..]) {
                            if !first {
                                data.push(b'|');
                            }
                            let abs = start + p;
                            data.extend_from_slice(&a.as_bytes()[start..abs]);
                            start = abs + b.len();
                            first = false;
                        }
                        if !first {
                            data.push(b'|');
                        }
                        data.extend_from_slice(&a.as_bytes()[start..]);
                    }
                }
                _ => unreachable!(),
            }
            unsafe { out_mask.set_unchecked(idx, true) };
        }
        offsets.push(T::from_usize(data.len()));
    }

    Ok(StringArray {
        offsets: offsets.into(),
        data: data.into(),
        null_mask: Some(out_mask),
    })
}

/// Generates a public dict-dict arithmetic function for a specific index type.
macro_rules! impl_apply_dict_dict {
    ($fn_name:ident, $idx:ty) => {
        pub fn $fn_name(
            lhs: CategoricalAVT<$idx>,
            rhs: CategoricalAVT<$idx>,
            op: ArithmeticOperator,
        ) -> Result<CategoricalArray<$idx>, KernelError> {
            apply_dict_dict_impl(lhs, rhs, op)
        }
    };
}

#[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
impl_apply_dict_dict!(apply_dict32_dict32, u32);

#[cfg(feature = "default_categorical_8")]
impl_apply_dict_dict!(apply_dict8_dict8, u8);

/// Applies element-wise binary arithmetic ops between a `CategoricalArray<u32>` and a `StringArray<T>`.
#[cfg(feature = "str_arithmetic")]
pub fn apply_dict32_str<T>(
    lhs: CategoricalAVT<u32>,
    rhs: StringAVT<T>,
    op: ArithmeticOperator,
) -> Result<CategoricalArray<u32>, KernelError>
where
    T: Integer,
{
    const SAMPLE_SIZE: usize = 256;
    const CARDINALITY_THRESHOLD: f64 = 0.75;

    // Destructure slice for local scope
    let (larr, loff, llen) = lhs;
    let (rarr, roff, rlen) = rhs;

    if llen != rlen {
        return Err(KernelError::LengthMismatch(log_length_mismatch(
            "apply_dict32_str".to_string(),
            llen,
            rlen,
        )));
    }

    // --- Estimate string diversity, pick path ---
    let cat_ratio = estimate_categorical_cardinality(larr, SAMPLE_SIZE);
    let str_ratio = estimate_string_cardinality(rarr, SAMPLE_SIZE);
    let max_ratio = cat_ratio.max(str_ratio);

    if max_ratio > CARDINALITY_THRESHOLD {
        // High cardinality: materialise, do flat string ops, then re-categorise.
        let lhs_str = larr.to_string_array();
        let str_result = apply_str_str((&lhs_str, loff, llen), (rarr, roff, rlen), op)?;
        return Ok(str_result.to_categorical_array());
    }

    // --- Low-cardinality: interned path ---
    let out_mask = merge_bitmasks_to_new(larr.null_mask.as_ref(), rarr.null_mask.as_ref(), llen);

    // First pass: Pre-count total number of output rows
    let mut total_out = 0usize;
    for local_idx in 0..llen {
        let valid = out_mask
            .as_ref()
            .map_or(true, |m| unsafe { m.get_unchecked(local_idx) });
        if !valid {
            total_out += 1;
        } else if let ArithmeticOperator::Divide = op {
            let i = loff + local_idx;
            let j = roff + local_idx;
            let l_val = unsafe { larr.get_str_unchecked(i) };
            let r_val = unsafe { rarr.get_str_unchecked(j) };
            if r_val.is_empty() {
                total_out += 1;
            } else {
                let mut start = 0;
                while let Some(pos) = l_val[start..].find(r_val) {
                    total_out += 1;
                    start += pos + r_val.len();
                }
                total_out += 1; // final segment
            }
        } else {
            total_out += 1;
        }
    }

    // Pre-allocate output buffers
    let mut out_data = Vec64::<u32>::with_capacity(total_out);
    unsafe {
        out_data.set_len(total_out);
    }
    let mut out_null = Bitmask::new_set_all(total_out, false);

    // Prepare dictionary and unique values (for this slice)
    let mut uniq: Vec64<String> = Vec64::with_capacity(larr.unique_values().len() + llen);
    uniq.extend(larr.unique_values().iter().cloned());

    #[cfg(feature = "fast_hash")]
    let mut dict: AHashMap<String, u32> = AHashMap::with_capacity(uniq.len());

    #[cfg(not(feature = "fast_hash"))]
    let mut dict: HashMap<String, u32> = HashMap::with_capacity(uniq.len());

    for (i, s) in uniq.iter().enumerate() {
        dict.insert(s.clone(), i as u32);
    }
    // Ensure "" is interned
    let empty_code = *dict.entry("".to_string()).or_insert_with(|| {
        let idx = uniq.len() as u32;
        uniq.push(String::new());
        idx
    });

    // Second pass: Fill output buffers for the slice
    let mut write_ptr = 0usize;
    for local_idx in 0..llen {
        let valid = out_mask
            .as_ref()
            .map_or(true, |m| unsafe { m.get_unchecked(local_idx) });
        if !valid {
            out_data.push(empty_code);
            out_null.set(write_ptr, false);
            write_ptr += 1;
            continue;
        }
        let i = loff + local_idx;
        let j = roff + local_idx;
        let l_val = unsafe { larr.get_str_unchecked(i) };
        let r_val = unsafe { rarr.get_str_unchecked(j) };
        match op {
            ArithmeticOperator::Add => {
                let mut s = String::with_capacity(l_val.len() + r_val.len());
                s.push_str(l_val);
                s.push_str(r_val);
                let code = intern(&s, &mut dict, &mut uniq);
                *unsafe { out_data.get_unchecked_mut(write_ptr) } = code;
                out_null.set(write_ptr, true);
                write_ptr += 1;
            }
            ArithmeticOperator::Subtract => {
                let result = if r_val.is_empty() {
                    l_val.to_string()
                } else if let Some(pos) = l_val.find(r_val) {
                    let mut s = l_val[..pos].to_owned();
                    s.push_str(&l_val[pos + r_val.len()..]);
                    s
                } else {
                    l_val.to_string()
                };
                let code = intern(&result, &mut dict, &mut uniq);
                *unsafe { out_data.get_unchecked_mut(write_ptr) } = code;
                out_null.set(write_ptr, true);
                write_ptr += 1;
            }
            ArithmeticOperator::Multiply => {
                let code = intern(l_val, &mut dict, &mut uniq);
                *unsafe { out_data.get_unchecked_mut(write_ptr) } = code;
                out_null.set(write_ptr, true);
                write_ptr += 1;
            }
            ArithmeticOperator::Divide => {
                if r_val.is_empty() {
                    let code = intern(l_val, &mut dict, &mut uniq);
                    *unsafe { out_data.get_unchecked_mut(write_ptr) } = code;
                    out_null.set(write_ptr, true);
                    write_ptr += 1;
                } else {
                    let mut start = 0;
                    loop {
                        match l_val[start..].find(r_val) {
                            Some(pos) => {
                                let part = &l_val[start..start + pos];
                                let code = intern(part, &mut dict, &mut uniq);
                                *unsafe { out_data.get_unchecked_mut(write_ptr) } = code;
                                out_null.set(write_ptr, true);
                                write_ptr += 1;
                                start += pos + r_val.len();
                            }
                            None => {
                                let tail = &l_val[start..];
                                let code = intern(tail, &mut dict, &mut uniq);
                                *unsafe { out_data.get_unchecked_mut(write_ptr) } = code;
                                out_null.set(write_ptr, true);
                                write_ptr += 1;
                                break;
                            }
                        }
                    }
                }
            }
            _ => {
                return Err(KernelError::UnsupportedType(
                    "Unsupported Type Error.".to_string(),
                ));
            }
        }
    }

    debug_assert_eq!(write_ptr, total_out);

    Ok(CategoricalArray {
        data: out_data.into(),
        #[cfg(not(feature = "shared_dict"))]
        unique_values: uniq,
        #[cfg(feature = "shared_dict")]
        dictionary: Dictionary::from(uniq),
        null_mask: Some(out_null),
    })
}

/// Applies element-wise binary arithmetic ops between `StringArray<T>` and `CategoricalArray<u32>`
#[cfg(feature = "str_arithmetic")]
pub fn apply_str_dict32<T>(
    lhs: StringAVT<T>,
    rhs: CategoricalAVT<u32>,
    op: ArithmeticOperator,
) -> Result<StringArray<T>, KernelError>
where
    T: Integer,
{
    // Destructure input slices
    let (larr, loff, llen) = lhs;
    let (rarr, roff, rlen) = rhs;

    if llen != rlen {
        return Err(KernelError::LengthMismatch(log_length_mismatch(
            "apply_str_dict32".to_string(),
            llen,
            rlen,
        )));
    }

    let out_mask = merge_bitmasks_to_new(larr.null_mask.as_ref(), rarr.null_mask.as_ref(), llen);

    // --- Pre-count output rows and byte size ---
    let mut total_rows = 0usize;
    let mut total_bytes = 0usize;

    for local_idx in 0..llen {
        let valid = out_mask
            .as_ref()
            .map_or(true, |m| unsafe { m.get_unchecked(local_idx) });
        if !valid {
            total_rows += 1;
            continue;
        }

        let i = loff + local_idx;
        let j = roff + local_idx;

        let l = unsafe { larr.get_str_unchecked(i) };
        let r = unsafe { rarr.get_str_unchecked(j) };

        match op {
            ArithmeticOperator::Divide => {
                total_rows += l.split(r).count();
                total_bytes += l.len(); // splitting doesn't remove data
            }
            ArithmeticOperator::Add => {
                total_rows += 1;
                total_bytes += l.len() + r.len();
            }
            ArithmeticOperator::Subtract => {
                total_rows += 1;
                total_bytes += l.len();
            }
            ArithmeticOperator::Multiply => {
                total_rows += 1;
                total_bytes += l.len();
            }
            _ => {
                return Err(KernelError::UnsupportedType(
                    "Unsupported Type Error.".to_string(),
                ));
            }
        }
    }

    // Allocate output buffers for local window
    let mut offsets = Vec64::<T>::with_capacity(total_rows + 1);
    let mut data = Vec64::<u8>::with_capacity(total_bytes);

    unsafe {
        offsets.set_len(total_rows + 1);
    }
    offsets[0] = T::zero();

    let mut cursor = 0;
    let mut offset_idx = 1;

    for local_idx in 0..llen {
        let valid = out_mask
            .as_ref()
            .map_or(true, |m| unsafe { m.get_unchecked(local_idx) });
        if !valid {
            offsets[offset_idx] = T::from_usize(cursor);
            offset_idx += 1;
            continue;
        }

        let i = loff + local_idx;
        let j = roff + local_idx;

        let l = unsafe { larr.get_str_unchecked(i) };
        let r = unsafe { rarr.get_str_unchecked(j) };

        match op {
            ArithmeticOperator::Divide => {
                for part in l.split(r) {
                    data.extend_from_slice(part.as_bytes());
                    cursor += part.len();
                    offsets[offset_idx] = T::from_usize(cursor);
                    offset_idx += 1;
                }
            }
            ArithmeticOperator::Add => {
                data.extend_from_slice(l.as_bytes());
                data.extend_from_slice(r.as_bytes());
                cursor += l.len() + r.len();
                offsets[offset_idx] = T::from_usize(cursor);
                offset_idx += 1;
            }
            ArithmeticOperator::Subtract => {
                if r.is_empty() {
                    data.extend_from_slice(l.as_bytes());
                    cursor += l.len();
                } else if let Some(pos) = l.find(r) {
                    data.extend_from_slice(&l.as_bytes()[..pos]);
                    data.extend_from_slice(&l.as_bytes()[pos + r.len()..]);
                    cursor += l.len() - r.len();
                } else {
                    data.extend_from_slice(l.as_bytes());
                    cursor += l.len();
                }
                offsets[offset_idx] = T::from_usize(cursor);
                offset_idx += 1;
            }
            ArithmeticOperator::Multiply => {
                data.extend_from_slice(l.as_bytes());
                cursor += l.len();
                offsets[offset_idx] = T::from_usize(cursor);
                offset_idx += 1;
            }
            _ => unreachable!(),
        }
    }

    debug_assert_eq!(offset_idx, total_rows + 1);

    Ok(StringArray {
        offsets: offsets.into(),
        data: data.into(),
        null_mask: out_mask,
    })
}

/// Applies element-wise binary arithmetic op between `CategoricalArray<u32>`s
/// a numeric slice.
#[cfg(feature = "str_arithmetic")]
pub fn apply_dict32_num<T>(
    lhs: CategoricalAVT<u32>,
    rhs: &[T],
    op: ArithmeticOperator,
) -> Result<CategoricalArray<u32>, KernelError>
where
    T: ToPrimitive + Copy,
{
    #[cfg(feature = "fast_hash")]
    use ahash::{HashMap, HashMapExt};

    #[cfg(not(feature = "fast_hash"))]
    use std::collections::HashMap;

    let (larr, loff, llen) = lhs;

    if llen != rhs.len() {
        return Err(KernelError::LengthMismatch(log_length_mismatch(
            "apply_dict32_num".to_string(),
            llen,
            rhs.len(),
        )));
    }

    let has_mask = larr.null_mask.is_some();
    let mut out_mask = if has_mask {
        Some(Bitmask::new_set_all(llen, true))
    } else {
        None
    };

    let mut data = Vec64::<u32>::with_capacity(llen);
    unsafe {
        data.set_len(llen);
    }

    let mut unique_values = Vec64::<String>::with_capacity(llen);
    let mut seen: HashMap<String, u32> = HashMap::with_capacity(llen);
    let mut unique_idx = 0;

    for local_idx in 0..llen {
        let valid = !has_mask
            || unsafe {
                larr.null_mask
                    .as_ref()
                    .unwrap()
                    .get_unchecked(loff + local_idx)
            };

        if valid {
            let i = loff + local_idx;
            let l_val = unsafe { larr.get_str_unchecked(i) };
            let n = rhs[local_idx].to_usize().unwrap_or(0);

            let cat = match op {
                ArithmeticOperator::Multiply => {
                    let count = n.min(1_000_000);
                    l_val.repeat(count)
                }
                _ => l_val.to_owned(),
            };

            let idx = if let Some(&ix) = seen.get(&cat) {
                ix
            } else {
                let ix = unique_idx as u32;
                seen.insert(cat.clone(), ix);
                unique_values.push(cat);
                unique_idx += 1;
                ix
            };

            unsafe {
                *data.get_unchecked_mut(local_idx) = idx;
                if let Some(mask) = &mut out_mask {
                    mask.set_unchecked(local_idx, true);
                }
            }
        } else {
            unsafe {
                *data.get_unchecked_mut(local_idx) = 0;
                if let Some(mask) = &mut out_mask {
                    mask.set_unchecked(local_idx, false);
                }
            }
        }
    }

    Ok(CategoricalArray {
        data: data.into(),
        #[cfg(not(feature = "shared_dict"))]
        unique_values,
        #[cfg(feature = "shared_dict")]
        dictionary: Dictionary::from(unique_values),
        null_mask: out_mask,
    })
}

/// Strips '.0' from concatenated decimal values so 'Hello1.0' becomes 'Hello1'.
#[inline]
#[cfg(feature = "str_arithmetic")]
pub fn format_finite<F: Float>(buf: &mut [MaybeUninit<u8>; 24], f: F) -> &str {
    unsafe {
        let ptr = buf.as_mut_ptr() as *mut u8;
        let n = f.write_to_ryu_buffer(ptr);
        debug_assert!(n <= buf.len());

        let slice = core::slice::from_raw_parts(ptr, n);
        let s = core::str::from_utf8_unchecked(slice);

        // Strip trailing ".0" if present
        if s.ends_with(".0") {
            let trimmed_len = s.len() - 2;
            core::str::from_utf8_unchecked(&slice[..trimmed_len])
        } else {
            s
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::MaskedArray;
    use crate::structs::variants::string::StringArray;
    #[cfg(feature = "str_arithmetic")]
    use crate::{Bitmask, CategoricalArray};

    use super::*;
    use crate::enums::operators::ArithmeticOperator;
    #[cfg(feature = "str_arithmetic")]
    use crate::vec64;

    // Helpers

    /// Assert that a `StringArray<T>` matches the supplied `Vec<&str>` and nullity.
    fn assert_str<T>(arr: &StringArray<T>, expect: &[&str], valid: Option<&[bool]>)
    where
        T: Integer + std::fmt::Debug,
    {
        assert_eq!(arr.len(), expect.len());
        for (i, exp) in expect.iter().enumerate() {
            assert_eq!(unsafe { arr.get_str_unchecked(i) }, *exp);
        }
        match (valid, &arr.null_mask) {
            (None, None) => {}
            (Some(expected), Some(mask)) => {
                for (i, bit) in expected.iter().enumerate() {
                    assert_eq!(unsafe { mask.get_unchecked(i) }, *bit);
                }
            }
            (None, Some(mask)) => {
                assert!(mask.all_true());
            }
            (Some(_), None) => panic!("expected mask missing"),
        }
    }

    // String - Numeric Kernels

    #[test]
    fn str_num_multiply() {
        let input = StringArray::<u32>::from_slice(&["hi", "bye", "x"]);
        let nums: &[i32] = &[3, 2, 0];
        let input_slice = (&input, 0, input.len(), input.data.len());
        let out =
            super::apply_str_num::<u32, i32, u32>(input_slice, nums, ArithmeticOperator::Multiply)
                .unwrap();
        assert_str(&out, &["hihihi", "byebye", ""], None);
    }

    #[test]
    fn str_num_multiply_chunk() {
        let base = StringArray::<u32>::from_slice(&["pad", "hi", "bye", "x", "pad2"]);
        let nums: &[i32] = &[3, 2, 0];
        // Window: ["hi", "bye", "x"]
        let input_slice = (&base, 1, 3, base.data.len());
        let out =
            super::apply_str_num::<u32, i32, u32>(input_slice, nums, ArithmeticOperator::Multiply)
                .unwrap();
        assert_str(&out, &["hihihi", "byebye", ""], None);
    }

    #[test]
    fn str_num_len_mismatch() {
        let input = StringArray::<u32>::from_slice(&["a"]);
        let nums: &[i32] = &[1, 2];
        let input_slice = (&input, 0, input.len(), input.data.len());
        let err = super::apply_str_num::<u32, i32, u32>(input_slice, nums, ArithmeticOperator::Add)
            .unwrap_err();
        match err {
            KernelError::LengthMismatch(_) => {}
            _ => panic!("wrong error variant"),
        }
    }

    #[test]
    fn str_num_len_mismatch_chunk() {
        let base = StringArray::<u32>::from_slice(&["pad", "a", "pad2"]);
        let nums: &[i32] = &[1, 2];
        // Window: ["a"]
        let input_slice = (&base, 1, 1, base.data.len());
        let err = super::apply_str_num::<u32, i32, u32>(input_slice, nums, ArithmeticOperator::Add)
            .unwrap_err();
        match err {
            KernelError::LengthMismatch(_) => {}
            _ => panic!("wrong error variant"),
        }
    }

    #[cfg(feature = "str_arithmetic")]
    #[test]
    fn str_float_all_ops() {
        let input = StringArray::<u32>::from_slice(&["foo", "bar1", "baz"]);
        let nums: &[f64] = &[1.0, 1.0, 2.0];
        let input_slice = (&input, 0, input.len());
        // Add
        let add = super::apply_str_float(input_slice, nums, ArithmeticOperator::Add).unwrap();
        assert_str(&add, &["foo1", "bar11", "baz2"], None);
        // Subtract
        let sub = super::apply_str_float(input_slice, nums, ArithmeticOperator::Subtract).unwrap();
        assert_str(&sub, &["foo", "bar", "baz"], None);
        // Multiply
        let mul = super::apply_str_float(input_slice, nums, ArithmeticOperator::Multiply).unwrap();
        assert_str(&mul, &["foo", "bar1", "bazbaz"], None);
        // Divide
        let div = super::apply_str_float(input_slice, nums, ArithmeticOperator::Divide).unwrap();
        assert_str(&div, &["foo", "bar|", "baz"], None);
    }

    #[cfg(feature = "str_arithmetic")]
    #[test]
    fn str_float_all_ops_chunk() {
        let base = StringArray::<u32>::from_slice(&["pad", "foo", "bar1", "baz", "pad2"]);
        let nums: &[f64] = &[1.0, 1.0, 2.0];
        // Window: ["foo", "bar1", "baz"]
        let input_slice = (&base, 1, 3);
        // Add
        let add = super::apply_str_float(input_slice, nums, ArithmeticOperator::Add).unwrap();
        assert_str(&add, &["foo1", "bar11", "baz2"], None);
        // Subtract
        let sub = super::apply_str_float(input_slice, nums, ArithmeticOperator::Subtract).unwrap();
        assert_str(&sub, &["foo", "bar", "baz"], None);
        // Multiply
        let mul = super::apply_str_float(input_slice, nums, ArithmeticOperator::Multiply).unwrap();
        assert_str(&mul, &["foo", "bar1", "bazbaz"], None);
        // Divide
        let div = super::apply_str_float(input_slice, nums, ArithmeticOperator::Divide).unwrap();
        assert_str(&div, &["foo", "bar|", "baz"], None);
    }

    // Dictionary Kernels

    #[cfg(feature = "str_arithmetic")]
    fn cat(values: &[&str]) -> CategoricalArray<u32> {
        CategoricalArray::<u32>::from_values(values.iter().copied())
    }

    #[cfg(feature = "str_arithmetic")]
    #[test]
    fn dict32_dict32_add() {
        let lhs = cat(&["A", "B", ""]);
        let rhs = cat(&["1", "2", "3"]);
        let lhs_slice = (&lhs, 0, lhs.data.len());
        let rhs_slice = (&rhs, 0, rhs.data.len());
        let out =
            super::apply_dict32_dict32(lhs_slice, rhs_slice, ArithmeticOperator::Add).unwrap();
        let expected = vec64!["A1", "B2", "3"];
        for (i, exp) in expected.iter().enumerate() {
            assert_eq!(out.get(i).unwrap_or(""), *exp);
        }
    }

    #[cfg(feature = "str_arithmetic")]
    #[test]
    fn dict32_dict32_add_chunk() {
        let lhs = cat(&["pad", "A", "B", "", "pad2"]);
        let rhs = cat(&["padx", "1", "2", "3", "pady"]);
        let lhs_slice = (&lhs, 1, 3); // "A", "B", ""
        let rhs_slice = (&rhs, 1, 3); // "1", "2", "3"
        let out =
            super::apply_dict32_dict32(lhs_slice, rhs_slice, ArithmeticOperator::Add).unwrap();
        let expected = vec64!["A1", "B2", "3"];
        for (i, exp) in expected.iter().enumerate() {
            assert_eq!(out.get(i).unwrap_or(""), *exp);
        }
    }

    #[cfg(feature = "str_arithmetic")]
    #[test]
    fn dict32_str_subtract() {
        let lhs = cat(&["hello", "yellow"]);
        let rhs = StringArray::<u32>::from_slice(&["l", "el"]);
        let lhs_slice = (&lhs, 0, lhs.data.len());
        let rhs_slice = (&rhs, 0, rhs.len());
        let out =
            super::apply_dict32_str(lhs_slice, rhs_slice, ArithmeticOperator::Subtract).unwrap();
        assert_eq!(out.get(0).unwrap(), "helo");
        assert_eq!(out.get(1).unwrap(), "ylow");
    }

    #[cfg(feature = "str_arithmetic")]
    #[test]
    fn dict32_str_subtract_chunk() {
        let lhs = cat(&["pad", "hello", "yellow", "pad2"]);
        let rhs = StringArray::<u32>::from_slice(&["pad", "l", "el", "pad2"]);
        let lhs_slice = (&lhs, 1, 2); // "hello", "yellow"
        let rhs_slice = (&rhs, 1, 2); // "l", "el"
        let out =
            super::apply_dict32_str(lhs_slice, rhs_slice, ArithmeticOperator::Subtract).unwrap();
        assert_eq!(out.get(0).unwrap(), "helo");
        assert_eq!(out.get(1).unwrap(), "ylow");
    }

    #[cfg(feature = "str_arithmetic")]
    #[test]
    fn str_dict32_divide() {
        let lhs = StringArray::<u32>::from_slice(&["a:b:c"]);
        let rhs = cat(&[":"]);
        let lhs_slice = (&lhs, 0, lhs.len());
        let rhs_slice = (&rhs, 0, rhs.data.len());
        let out =
            super::apply_str_dict32(lhs_slice, rhs_slice, ArithmeticOperator::Divide).unwrap();
        assert_str(&out, &["a", "b", "c"], None);
    }

    #[cfg(feature = "str_arithmetic")]
    #[test]
    fn str_dict32_divide_chunk() {
        // Extended arrays for windowing
        let lhs = StringArray::<u32>::from_slice(&["pad", "a:b:c", "pad2"]);
        let rhs = cat(&["pad", ":", "pad2"]);
        let lhs_slice = (&lhs, 1, 1); // only "a:b:c"
        let rhs_slice = (&rhs, 1, 1); // only ":"
        let out =
            super::apply_str_dict32(lhs_slice, rhs_slice, ArithmeticOperator::Divide).unwrap();
        assert_str(&out, &["a", "b", "c"], None);
    }

    #[cfg(feature = "str_arithmetic")]
    #[test]
    fn dict32_num_multiply() {
        let lhs = cat(&["x", "y"]);
        let nums: &[u32] = &[3, 1];
        let lhs_slice = (&lhs, 0, lhs.data.len());
        let nums_window = &nums[0..lhs.data.len()];
        let out =
            super::apply_dict32_num(lhs_slice, nums_window, ArithmeticOperator::Multiply).unwrap();
        assert_eq!(out.get(0).unwrap(), "xxx");
        assert_eq!(out.get(1).unwrap(), "y");
    }

    #[cfg(feature = "str_arithmetic")]
    #[test]
    fn dict32_num_multiply_chunk() {
        let lhs = cat(&["pad", "x", "y", "pad2"]);
        let nums: &[u32] = &[0, 3, 1, 0];
        let lhs_slice = (&lhs, 1, 2); // only "x", "y"
        let nums_window = &nums[1..3];
        let out =
            super::apply_dict32_num(lhs_slice, nums_window, ArithmeticOperator::Multiply).unwrap();
        assert_eq!(out.get(0).unwrap(), "xxx");
        assert_eq!(out.get(1).unwrap(), "y");
    }

    #[cfg(feature = "str_arithmetic")]
    fn cat32_str_arr(strings: &[&str]) -> (CategoricalArray<u32>, StringArray<u32>) {
        let str_arr = StringArray::from_vec(strings.to_vec(), None);
        let cat_arr = str_arr.to_categorical_array();
        (cat_arr, str_arr)
    }

    #[cfg(feature = "str_arithmetic")]
    #[test]
    fn test_apply_dict32_str_add_and_divide() {
        let (lhs_cat, rhs_str) = cat32_str_arr(&["foo", "bar|baz", ""]);
        // Add: Use slices
        let lhs_cat_slice = (&lhs_cat, 0, lhs_cat.data.len());
        let rhs_str_slice = (&rhs_str, 0, rhs_str.len());
        let added =
            apply_dict32_str(lhs_cat_slice, rhs_str_slice, ArithmeticOperator::Add).unwrap();
        let expected_cat = apply_str_str(
            (&lhs_cat.to_string_array(), 0, lhs_cat.len()),
            rhs_str_slice,
            ArithmeticOperator::Add,
        )
        .unwrap()
        .to_categorical_array();
        assert_eq!(added.unique_values(), expected_cat.unique_values());
        assert_eq!(added.data, expected_cat.data);

        // Divide: Use slices
        let divided =
            apply_dict32_str(lhs_cat_slice, rhs_str_slice, ArithmeticOperator::Divide).unwrap();
        let expected_div = apply_str_str(
            (&lhs_cat.to_string_array(), 0, lhs_cat.len()),
            rhs_str_slice,
            ArithmeticOperator::Divide,
        )
        .unwrap()
        .to_categorical_array();
        assert_eq!(divided.unique_values(), expected_div.unique_values());
        assert_eq!(divided.data, expected_div.data);
    }

    #[cfg(feature = "str_arithmetic")]
    #[test]
    fn test_apply_dict32_str_add_and_divide_chunk() {
        let (lhs_cat, rhs_str) = cat32_str_arr(&["pad", "foo", "bar|baz", "", "pad2"]);
        let lhs_cat_slice = (&lhs_cat, 1, 3); // only "foo", "bar|baz", ""
        let rhs_str_slice = (&rhs_str, 1, 3);

        // Add
        let added =
            apply_dict32_str(lhs_cat_slice, rhs_str_slice, ArithmeticOperator::Add).unwrap();
        let expected_cat = apply_str_str(
            (&lhs_cat.to_string_array(), 1, 3),
            rhs_str_slice,
            ArithmeticOperator::Add,
        )
        .unwrap()
        .to_categorical_array();
        assert_eq!(added.unique_values(), expected_cat.unique_values());
        assert_eq!(added.data, expected_cat.data);

        // Divide
        let divided =
            apply_dict32_str(lhs_cat_slice, rhs_str_slice, ArithmeticOperator::Divide).unwrap();
        let expected_div = apply_str_str(
            (&lhs_cat.to_string_array(), 1, 3),
            rhs_str_slice,
            ArithmeticOperator::Divide,
        )
        .unwrap()
        .to_categorical_array();
        assert_eq!(divided.unique_values(), expected_div.unique_values());
        assert_eq!(divided.data, expected_div.data);
    }

    // String arithmetic

    #[cfg(feature = "str_arithmetic")]
    fn string_array<T: Integer>(data: &[&str], nulls: Option<&[bool]>) -> StringArray<T> {
        let array = StringArray::from_vec(data.to_vec(), nulls.map(Bitmask::from_bools));
        assert_eq!(array.len(), data.len());
        array
    }

    #[cfg(feature = "str_arithmetic")]
    #[test]
    fn test_add_str() {
        let lhs = string_array::<u32>(&["a", "b", "c"], None);
        let rhs = string_array::<u32>(&["x", "y", "z"], None);
        let lhs_slice = (&lhs, 0, lhs.len());
        let rhs_slice = (&rhs, 0, rhs.len());
        let result = apply_str_str(lhs_slice, rhs_slice, ArithmeticOperator::Add).unwrap();

        assert_eq!(result.get(0), Some("ax"));
        assert_eq!(result.get(1), Some("by"));
        assert_eq!(result.get(2), Some("cz"));
    }

    #[cfg(feature = "str_arithmetic")]
    #[test]
    fn test_add_str_chunk() {
        let lhs = string_array::<u32>(&["pad", "a", "b", "c", "pad2"], None);
        let rhs = string_array::<u32>(&["pad", "x", "y", "z", "pad2"], None);
        // window: indices 1..4
        let lhs_slice = (&lhs, 1, 3);
        let rhs_slice = (&rhs, 1, 3);
        let result = apply_str_str(lhs_slice, rhs_slice, ArithmeticOperator::Add).unwrap();

        assert_eq!(result.get(0), Some("ax"));
        assert_eq!(result.get(1), Some("by"));
        assert_eq!(result.get(2), Some("cz"));
    }

    #[cfg(feature = "str_arithmetic")]
    #[test]
    fn test_subtract_str() {
        let lhs = string_array::<u32>(&["hello", "goodbye", "test"], None);
        let rhs = string_array::<u32>(&["l", "bye", "xyz"], None);
        let lhs_slice = (&lhs, 0, lhs.len());
        let rhs_slice = (&rhs, 0, rhs.len());
        let result = apply_str_str(lhs_slice, rhs_slice, ArithmeticOperator::Subtract).unwrap();

        assert_eq!(result.get(0), Some("helo"));
        assert_eq!(result.get(1), Some("good"));
        assert_eq!(result.get(2), Some("test")); // no match
    }

    #[cfg(feature = "str_arithmetic")]
    #[test]
    fn test_subtract_str_chunk() {
        let lhs = string_array::<u32>(&["pad", "hello", "goodbye", "test", "pad2"], None);
        let rhs = string_array::<u32>(&["pad", "l", "bye", "xyz", "pad2"], None);
        // window: indices 1..4
        let lhs_slice = (&lhs, 1, 3);
        let rhs_slice = (&rhs, 1, 3);
        let result = apply_str_str(lhs_slice, rhs_slice, ArithmeticOperator::Subtract).unwrap();

        assert_eq!(result.get(0), Some("helo"));
        assert_eq!(result.get(1), Some("good"));
        assert_eq!(result.get(2), Some("test")); // no match
    }

    #[cfg(feature = "str_arithmetic")]
    #[test]
    fn test_multiply_str() {
        let lhs = string_array::<u32>(&["x", "ab", "c"], None);
        let rhs = string_array::<u32>(&["123", "12", "long_string"], None);
        let lhs_slice = (&lhs, 0, lhs.len());
        let rhs_slice = (&rhs, 0, rhs.len());
        let result = apply_str_str(lhs_slice, rhs_slice, ArithmeticOperator::Multiply).unwrap();

        assert_eq!(result.get(0), Some("xxx"));
        assert_eq!(result.get(1), Some("abab"));
        assert_eq!(
            result.get(2),
            Some("c".repeat("long_string".len()).as_str())
        );
    }

    #[cfg(feature = "str_arithmetic")]
    #[test]
    fn test_multiply_str_chunk() {
        let lhs = string_array::<u32>(&["pad", "x", "ab", "c", "pad2"], None);
        let rhs = string_array::<u32>(&["pad", "123", "12", "long_string", "pad2"], None);
        // window: indices 1..4
        let lhs_slice = (&lhs, 1, 3);
        let rhs_slice = (&rhs, 1, 3);
        let result = apply_str_str(lhs_slice, rhs_slice, ArithmeticOperator::Multiply).unwrap();

        assert_eq!(result.get(0), Some("xxx"));
        assert_eq!(result.get(1), Some("abab"));
        assert_eq!(
            result.get(2),
            Some("c".repeat("long_string".len()).as_str())
        );
    }

    #[cfg(feature = "str_arithmetic")]
    #[test]
    fn test_divide_str() {
        let lhs = string_array::<u32>(&["a,b,c", "a--b--c", "abc"], None);
        let rhs = string_array::<u32>(&[",", "--", ""], None);
        let lhs_slice = (&lhs, 0, lhs.len());
        let rhs_slice = (&rhs, 0, rhs.len());
        let result = apply_str_str(lhs_slice, rhs_slice, ArithmeticOperator::Divide).unwrap();

        assert_eq!(result.get(0), Some("a|b|c"));
        assert_eq!(result.get(1), Some("a|b|c"));
        assert_eq!(result.get(2), Some("abc"));
    }

    #[cfg(feature = "str_arithmetic")]
    #[test]
    fn test_divide_str_chunk() {
        let lhs = string_array::<u32>(&["xxx", "a,b,c", "a--b--c", "abc", "yyy"], None);
        let rhs = string_array::<u32>(&["", ",", "--", "", ""], None);
        // operate only on the window: indices 1,2,3
        let lhs_slice = (&lhs, 1, 3);
        let rhs_slice = (&rhs, 1, 3);
        let result = apply_str_str(lhs_slice, rhs_slice, ArithmeticOperator::Divide).unwrap();

        assert_eq!(result.get(0), Some("a|b|c"));
        assert_eq!(result.get(1), Some("a|b|c"));
        assert_eq!(result.get(2), Some("abc"));
    }

    #[cfg(feature = "str_arithmetic")]
    #[test]
    fn test_nulls_str() {
        let lhs = string_array::<u32>(&["a", "b", "c"], Some(&[true, false, true]));
        let rhs = string_array::<u32>(&["x", "y", "z"], Some(&[true, true, false]));
        let lhs_slice = (&lhs, 0, lhs.len());
        let rhs_slice = (&rhs, 0, rhs.len());
        let result = apply_str_str(lhs_slice, rhs_slice, ArithmeticOperator::Add).unwrap();

        assert_eq!(result.get(0), Some("ax"));
        assert_eq!(result.get(1), None);
        assert_eq!(result.get(2), None);
    }

    #[cfg(feature = "str_arithmetic")]
    #[test]
    fn test_nulls_str_chunk() {
        let lhs = string_array::<u32>(
            &["0", "a", "b", "c", "9"],
            Some(&[false, true, false, true, false]),
        );
        let rhs = string_array::<u32>(
            &["y", "x", "y", "z", "w"],
            Some(&[true, true, true, false, false]),
        );
        // window covering indices 1..4
        let lhs_slice = (&lhs, 1, 3);
        let rhs_slice = (&rhs, 1, 3);
        let result = apply_str_str(lhs_slice, rhs_slice, ArithmeticOperator::Add).unwrap();

        assert_eq!(result.get(0), Some("ax"));
        assert_eq!(result.get(1), None);
        assert_eq!(result.get(2), None);
    }

    #[cfg(feature = "str_arithmetic")]
    #[test]
    fn test_mismatched_length_str() {
        let lhs = string_array::<u32>(&["a", "b"], None);
        let rhs = string_array::<u32>(&["x"], None);
        let lhs_slice = (&lhs, 0, lhs.len());
        let rhs_slice = (&rhs, 0, rhs.len());
        let result = apply_str_str(lhs_slice, rhs_slice, ArithmeticOperator::Add);
        assert!(matches!(result, Err(KernelError::LengthMismatch(_))));
    }

    #[cfg(feature = "str_arithmetic")]
    #[test]
    fn test_mismatched_length_str_chunk() {
        let lhs = string_array::<u32>(&["a", "b", "c"], None);
        let rhs = string_array::<u32>(&["x"], None);
        // windowed call: indices 1..3, lhs has 2 elements, rhs has 1, so mismatch
        let lhs_slice = (&lhs, 1, 2);
        let rhs_slice = (&rhs, 0, 1);
        let result = apply_str_str(lhs_slice, rhs_slice, ArithmeticOperator::Add);
        assert!(matches!(result, Err(KernelError::LengthMismatch(_))));
    }
}
