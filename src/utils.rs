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

//! # **Utilities** - *Internal Helper Utilities*
//!
//! A small collection of internal utilities that support validation, parsing, and text conversion
//! elsewhere within the crate.

#[cfg(feature = "fast_hash")]
use ahash::AHashSet as HashSet;
#[cfg(not(feature = "fast_hash"))]
use std::collections::HashSet;
use std::simd::{Mask, MaskElement};
use std::{fmt::Display, sync::Arc};

use crate::enums::error::KernelError;
#[cfg(feature = "chunked")]
use crate::enums::error::MinarrowError;
#[cfg(feature = "chunked")]
use crate::structs::field_array::create_field_for_array;
use crate::traits::masked_array::MaskedArray;
#[cfg(feature = "chunked")]
use crate::{Array, FieldArray, SuperArray};
use crate::{
    Bitmask, CategoricalArray, Float, FloatArray, Integer, IntegerArray, StringArray, TextArray,
};

#[inline(always)]
pub fn validate_null_mask_len(data_len: usize, null_mask: &Option<Bitmask>) {
    if let Some(mask) = null_mask {
        assert_eq!(
            mask.len(),
            data_len,
            "Validation Error: Null mask length ({}) does not match data length ({})",
            mask.len(),
            data_len
        );
    }
}

/// Parses a string into a timestamp in milliseconds since the Unix epoch.
/// Returns `Some(i64)` on success, or `None` if the string could not be parsed.
///
/// Attempts common ISO8601/RFC3339 and custom date/time formats if the `time`
/// feature is enabled.
pub fn parse_datetime_str(s: &str) -> Option<i64> {
    // Empty string is always None/null
    if s.is_empty() {
        return None;
    }

    #[cfg(feature = "datetime_ops")]
    {
        use time::{
            Date, OffsetDateTime, PrimitiveDateTime, Time,
            format_description::well_known::{Iso8601, Rfc3339},
            macros::format_description,
        };

        // Try to parse as RFC3339/ISO8601 string (with timezone)
        if let Ok(dt) = OffsetDateTime::parse(s, &Rfc3339) {
            return Some(dt.unix_timestamp() * 1_000 + (dt.nanosecond() / 1_000_000) as i64);
        }

        // Try ISO8601 format
        if let Ok(dt) = OffsetDateTime::parse(s, &Iso8601::DEFAULT) {
            return Some(dt.unix_timestamp() * 1_000 + (dt.nanosecond() / 1_000_000) as i64);
        }

        // Try parsing as full date-time (no timezone) "%Y-%m-%d %H:%M:%S"
        let format = format_description!("[year]-[month]-[day] [hour]:[minute]:[second]");
        if let Ok(dt) = PrimitiveDateTime::parse(s, format) {
            let dt_utc = dt.assume_utc();
            return Some(
                dt_utc.unix_timestamp() * 1_000 + (dt_utc.nanosecond() / 1_000_000) as i64,
            );
        }

        // Try parsing as date only "%Y-%m-%d"
        let date_format = format_description!("[year]-[month]-[day]");
        if let Ok(date) = Date::parse(s, date_format) {
            if let Ok(dt) = date.with_hms(0, 0, 0) {
                let dt_utc = dt.assume_utc();
                return Some(dt_utc.unix_timestamp() * 1_000);
            }
        }

        // Try parsing as time only "%H:%M:%S" (use today's date)
        let time_format = format_description!("[hour]:[minute]:[second]");
        if let Ok(time) = Time::parse(s, time_format) {
            let today = OffsetDateTime::now_utc().date();
            let dt_primitive = today.with_time(time);
            let dt_utc = dt_primitive.assume_utc();
            return Some(
                dt_utc.unix_timestamp() * 1_000 + (dt_utc.nanosecond() / 1_000_000) as i64,
            );
        }
    }

    // Fallback: parse as i64 integer (milliseconds since epoch)
    if let Ok(ms) = s.parse::<i64>() {
        return Some(ms);
    }

    None
}

/// Converts an integer array to a String32 TextArray, preserving nulls.
pub fn int_to_text_array<T: Display + Integer>(arr: &Arc<IntegerArray<T>>) -> TextArray {
    let mut strings: Vec<String> = Vec::with_capacity(arr.len());
    for i in 0..arr.len() {
        if arr.is_null(i) {
            strings.push(String::new()); // This "" keeps the correct length
        } else {
            strings.push(format!("{}", arr.data[i]));
        }
    }
    let refs: Vec<&str> = strings.iter().map(String::as_str).collect();
    let string_array = StringArray::<u32>::from_vec(refs, arr.null_mask.clone());
    TextArray::String32(Arc::new(string_array))
}

/// Converts a float array to a String32 TextArray, preserving nulls.
pub fn float_to_text_array<T: Display + Float>(arr: &Arc<FloatArray<T>>) -> TextArray {
    let mut strings: Vec<String> = Vec::with_capacity(arr.len());
    for i in 0..arr.len() {
        if arr.is_null(i) {
            strings.push(String::new()); // This "" keeps the correct length
        } else {
            strings.push(format!("{}", arr.data[i]));
        }
    }
    let refs: Vec<&str> = strings.iter().map(String::as_str).collect();
    let string_array = StringArray::<u32>::from_vec(refs, arr.null_mask.clone());
    TextArray::String32(Arc::new(string_array))
}

/// Validates that two lengths are equal for binary kernel operations.
///
/// Critical validation function ensuring input arrays have matching lengths before performing
/// binary operations like comparisons, arithmetic, or logical operations. Prevents undefined
/// behaviour and provides clear error diagnostics when length mismatches occur.
///
/// # Parameters
/// - `label`: Descriptive context label for error reporting (e.g., "compare numeric")
/// - `a`: Length of the first input array or data structure
/// - `b`: Length of the second input array or data structure
///
/// # Returns
/// `Ok(())` if lengths are equal, otherwise `KernelError::LengthMismatch` with diagnostic details.
#[inline(always)]
pub fn confirm_equal_len(label: &str, a: usize, b: usize) -> Result<(), KernelError> {
    if a != b {
        return Err(KernelError::LengthMismatch(format!(
            "{}: length mismatch (lhs: {}, rhs: {})",
            label, a, b
        )));
    }
    Ok(())
}

/// Round `byte_count` up to the next 64-byte boundary.
///
/// Useful for pre-calculating buffer sizes that must honour SIMD or
/// Arena alignment requirements.
#[inline(always)]
pub fn align64(byte_count: usize) -> usize {
    (byte_count + 63) & !63
}

/// SIMD Alignment check. Returns true if the slice is properly
/// 64-byte aligned for SIMD operations, false otherwise.
#[inline(always)]
pub fn is_simd_aligned<T>(slice: &[T]) -> bool {
    if slice.is_empty() {
        true
    } else {
        (slice.as_ptr() as usize) % 64 == 0
    }
}

/// Creates a SIMD mask from a bitmask window for vectorised conditional operations.
///
/// Converts a contiguous section of a bitmask into a SIMD mask.
/// The resulting mask can be used to selectively enable/disable SIMD lanes during
/// computation, providing efficient support for sparse or conditional operations.
///
/// # Type Parameters
/// - `T`: Mask element type implementing `MaskElement` (typically i8, i16, i32, or i64)
/// - `N`: Number of SIMD lanes, must match the SIMD vector width for the target operation
///
/// # Parameters
/// - `mask`: Source bitmask containing validity information
/// - `offset`: Starting bit offset within the bitmask
/// - `len`: Maximum number of bits to consider (bounds checking)
///
/// # Returns
/// A `Mask<T, N>` where each lane corresponds to the validity of the corresponding input element.
/// Lanes beyond `len` are set to false for safety.
///
/// # Usage Example
/// ```rust,ignore
/// use simd_kernels::utils::simd_mask;
///
/// // Create 8-lane mask for conditional SIMD operations  
/// let mask: Mask<i32, 8> = simd_mask(&bitmask, 0, 64);
/// let result = simd_vector.select(mask, default_vector);
/// ```
#[inline(always)]
pub fn simd_mask<T: MaskElement, const N: usize>(
    mask: &Bitmask,
    offset: usize,
    len: usize,
) -> Mask<T, N>
where
{
    // Extract the packed bits covering this SIMD chunk from the bitmask.
    // The bitmask is LSB-ordered which matches Mask::from_bitmask convention.
    let word_idx = offset / 64;
    let bit_shift = offset % 64;
    let raw = unsafe { mask.word_unchecked(word_idx) } >> bit_shift;

    // If the chunk straddles a word boundary, pull in bits from the next word
    let raw = if bit_shift > 0 && word_idx + 1 < (mask.len() + 63) / 64 {
        raw | (unsafe { mask.word_unchecked(word_idx + 1) } << (64 - bit_shift))
    } else {
        raw
    };

    // Zero out lanes that are beyond the array length
    let remaining = if offset < len { len - offset } else { 0 };
    let raw = if remaining < N && remaining < 64 {
        raw & ((1u64 << remaining) - 1)
    } else {
        raw
    };

    Mask::from_bitmask(raw)
}

/// Writes a SIMD mask's packed bits directly into the output bitmask at the given offset.
/// This is the write-side complement to `simd_mask`, avoiding per-lane `set_unchecked` calls.
#[inline(always)]
pub fn write_simd_mask_bits<T: MaskElement, const N: usize>(
    out_mask: &mut Bitmask,
    offset: usize,
    m: Mask<T, N>,
)
where
{
    let mbits = m.to_bitmask();
    let word_idx = offset / 64;
    let bit_shift = offset % 64;

    unsafe {
        // Read-modify-write the target word
        let existing = out_mask.word_unchecked(word_idx);
        // Clear the N bits at bit_shift, then OR in the new bits
        let lane_mask = if N >= 64 { !0u64 } else { (1u64 << N) - 1 };
        let cleared = existing & !(lane_mask << bit_shift);
        out_mask.set_word_unchecked(word_idx, cleared | (mbits << bit_shift));

        // If the chunk straddles a word boundary, write the overflow to the next word
        if bit_shift > 0 && bit_shift + N > 64 {
            let overflow_bits = N - (64 - bit_shift);
            let next_existing = out_mask.word_unchecked(word_idx + 1);
            let overflow_mask = (1u64 << overflow_bits) - 1;
            let cleared_next = next_existing & !overflow_mask;
            out_mask.set_word_unchecked(word_idx + 1, cleared_next | (mbits >> (64 - bit_shift)));
        }
    }
}

/// Checks the mask capacity is large enough
/// Used so we can avoid bounds checks in the hot loop
#[inline(always)]
pub fn confirm_mask_capacity(cmp_len: usize, mask: Option<&Bitmask>) -> Result<(), KernelError> {
    if let Some(m) = mask {
        confirm_capacity("mask (Bitmask)", m.capacity(), cmp_len)?;
    }
    Ok(())
}

/// Validates that actual capacity matches expected capacity for kernel operations.
///
/// Validation function used throughout the kernel library to ensure data structure
/// capacities are correct before performing operations. Prevents buffer overruns and ensures
/// memory safety by catching capacity mismatches early with descriptive error messages.
///
/// # Parameters
/// - `label`: Descriptive label for the validation context (used in error messages)
/// - `actual`: The actual capacity of the data structure being validated
/// - `expected`: The expected capacity required for the operation
///
/// # Returns
/// `Ok(())` if capacities match, otherwise `KernelError::InvalidArguments` with detailed message.
///
/// # Error Conditions
/// Returns `KernelError::InvalidArguments` when `actual != expected`, providing a clear
/// error message indicating the mismatch and context.
#[inline(always)]
pub fn confirm_capacity(label: &str, actual: usize, expected: usize) -> Result<(), KernelError> {
    if actual != expected {
        return Err(KernelError::InvalidArguments(format!(
            "{}: capacity mismatch (expected {}, got {})",
            label, expected, actual
        )));
    }
    Ok(())
}

/// Estimate cardinality ratio on a sample from a CategoricalArray.
/// Used to quickly figure out the optimal strategy when comparing
/// StringArray and CategoricalArrays.
#[inline(always)]
pub fn estimate_categorical_cardinality(cat: &CategoricalArray<u32>, sample_size: usize) -> f64 {
    let len = cat.data.len();
    if len == 0 {
        return 0.0;
    }
    let mut seen = HashSet::with_capacity(sample_size.min(len));
    let step = (len / sample_size.max(1)).max(1);
    for i in (0..len).step_by(step) {
        let s = unsafe { cat.get_str_unchecked(i) };
        seen.insert(s);
        if seen.len() >= sample_size {
            break;
        }
    }
    (seen.len() as f64) / (sample_size.min(len) as f64)
}

/// Estimate cardinality ratio on a sample from a StringArray.
/// Used to quickly figure out the optimal strategy when comparing
/// StringArray and CategoricalArrays.
#[inline(always)]
pub fn estimate_string_cardinality<T: Integer>(arr: &StringArray<T>, sample_size: usize) -> f64 {
    let len = arr.len();
    if len == 0 {
        return 0.0;
    }
    let mut seen = HashSet::with_capacity(sample_size.min(len));
    let step = (len / sample_size.max(1)).max(1);
    for i in (0..len).step_by(step) {
        let s = unsafe { arr.get_str_unchecked(i) };
        seen.insert(s);
        if seen.len() >= sample_size {
            break;
        }
    }
    (seen.len() as f64) / (sample_size.min(len) as f64)
}

#[cfg(feature = "chunked")]
/// Helper function to handle mask union between Array and SuperArray
fn union_array_superarray_masks(
    array: &Array,
    super_array: &SuperArray,
) -> Result<Option<Bitmask>, MinarrowError> {
    let array_mask = array.null_mask();
    let super_array_masks: Vec<Option<&Bitmask>> = super_array
        .chunks()
        .iter()
        .map(|chunk| chunk.null_mask())
        .collect();

    let super_array_concatenated_mask = if super_array_masks
        .iter()
        .any(|m: &Option<&Bitmask>| m.is_some())
    {
        let mut concatenated_bits = Vec::new();
        for (chunk, mask_opt) in super_array.chunks().iter().zip(super_array_masks.iter()) {
            if let Some(mask) = mask_opt {
                concatenated_bits.extend((0..mask.len()).map(|i| mask.get(i)));
            } else {
                concatenated_bits.extend(std::iter::repeat(true).take(chunk.len()));
            }
        }
        Some(Bitmask::from_bools(&concatenated_bits))
    } else {
        None
    };

    match (array_mask, super_array_concatenated_mask) {
        (Some(arr_mask), Some(super_mask)) => {
            if arr_mask.len() == super_mask.len() {
                Ok(Some(arr_mask.union(&super_mask)))
            } else {
                Err(MinarrowError::ShapeError {
                    message: format!(
                        "Mask lengths must match for union: {} vs {}",
                        arr_mask.len(),
                        super_mask.len()
                    ),
                })
            }
        }
        (Some(mask), None) => Ok(Some(mask.clone())),
        (None, Some(mask)) => Ok(Some(mask)),
        (None, None) => Ok(None),
    }
}

#[cfg(feature = "chunked")]
/// Helper function to create aligned chunks from Array to match SuperArray chunk structure
pub fn create_aligned_chunks_from_array(
    array: Array,
    super_array: &SuperArray,
    field_name: &str,
) -> Result<SuperArray, MinarrowError> {
    // Check total lengths match
    if array.len() != super_array.len() {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "Array and SuperArray must have same total length for broadcasting: {} vs {}",
                array.len(),
                super_array.len()
            ),
        });
    }

    // Union the masks
    let full_mask = union_array_superarray_masks(&array, super_array)?;

    // Extract chunk lengths from SuperArray
    let chunk_lengths: Vec<usize> = super_array
        .chunks()
        .iter()
        .map(|chunk| chunk.len())
        .collect();

    // Create aligned chunks from Array using view function
    let mut start = 0;
    let mut mask_start = 0;
    let chunks: Result<Vec<_>, _> = chunk_lengths
        .iter()
        .map(|&chunk_len| {
            let end = start + chunk_len;
            if end > array.len() {
                return Err(MinarrowError::ShapeError {
                    message: format!(
                        "Chunk alignment failed: index {} out of bounds for length {}",
                        end,
                        array.len()
                    ),
                });
            }
            let view = array.view(start, chunk_len);
            let mut array_chunk = view.array.slice_clone(view.offset, view.len());

            // Apply portion of full_mask to this chunk
            if let Some(ref mask) = full_mask {
                let mask_end = mask_start + chunk_len;
                let chunk_mask_bits: Vec<bool> =
                    (mask_start..mask_end).map(|i| mask.get(i)).collect();
                let chunk_mask = Bitmask::from_bools(&chunk_mask_bits);
                array_chunk.set_null_mask(chunk_mask);
                mask_start = mask_end;
            }

            start = end;
            let first_super_chunk = &super_array.chunks()[0];
            let field =
                create_field_for_array(field_name, &array_chunk, Some(first_super_chunk), None);
            Ok(FieldArray::new(field, array_chunk))
        })
        .collect();

    Ok(SuperArray::from_chunks(chunks?))
}
