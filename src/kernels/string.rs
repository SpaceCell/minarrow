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

//! # **String Operations Kernels Module** - *High-Performance String Processing and Text Analysis*
//!
//! String processing kernels for text manipulation, pattern matching,
//! and string analysis operations with UTF-8 awareness and null-safe semantics. Essential infrastructure
//! for text analytics, data cleansing, and string-heavy analytical workloads.
//!
//! ## Core Operations
//! - **String transformations**: Case conversion, trimming, padding, and substring operations
//! - **Pattern matching**: Regular expression support with compiled pattern caching  
//! - **String comparison**: Lexicographic ordering with UTF-8 aware collation
//! - **Text analysis**: Length calculation, character counting, and encoding detection
//! - **String aggregation**: Concatenation with configurable delimiters and null handling
//! - **Search operations**: Contains, starts with, ends with predicates with optimised implementations

#[cfg(feature = "fast_hash")]
use ahash::{AHashMap, AHashSet};
#[cfg(not(feature = "fast_hash"))]
use std::collections::{HashMap, HashSet};

use crate::{
    Bitmask, BooleanArray, CategoricalArray, Integer, IntegerArray, MaskedArray,
    StringArray, Vec64,
    aliases::{CategoricalAVT, StringAVT},
};
#[cfg(feature = "shared_dict")]
use crate::Dictionary;
#[cfg(feature = "views")]
use crate::TextArrayV;
#[cfg(feature = "regex")]
use regex::Regex;

use crate::enums::error::KernelError;
use crate::utils::confirm_mask_capacity;

/// Side for string padding operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadSide {
    Left,
    Right,
    Both,
}


// Concatenation

/// Concatenates corresponding string pairs from two string arrays element-wise.
///
/// Performs element-wise concatenation of strings from two `StringArray` sources,
/// producing a new string array where each result string is the concatenation of
/// the corresponding left and right input strings.
///
/// # Parameters
/// - `lhs`: Left-hand string array view tuple `(StringArray, offset, length)`
/// - `rhs`: Right-hand string array view tuple `(StringArray, offset, length)`
///
/// # Returns
/// A new `StringArray<T>` containing concatenated strings with proper null handling.
///
/// # Null Handling
/// - If either input string is null, the result is null
/// - Output null mask reflects the union of input null positions
///
/// # Performance
/// - Pre-computes total memory requirements to minimise allocations
/// - Uses unsafe unchecked access for validated indices
/// - Optimised for bulk string concatenation operations
pub fn concat_str_str<T: Integer>(lhs: StringAVT<T>, rhs: StringAVT<T>) -> StringArray<T> {
    let (larr, loff, llen) = lhs;
    let (rarr, roff, rlen) = rhs;
    let len = llen.min(rlen);

    let lmask = larr.null_mask.as_ref();
    let rmask = rarr.null_mask.as_ref();
    let mut out_mask = Bitmask::new_set_all(len, false);
    let _ = confirm_mask_capacity(larr.len(), lmask);
    let _ = confirm_mask_capacity(rarr.len(), rmask);

    // Compute total byte size required
    let mut total_bytes = 0;
    for i in 0..len {
        let valid = lmask.map_or(true, |m| unsafe { m.get_unchecked(loff + i) })
            && rmask.map_or(true, |m| unsafe { m.get_unchecked(roff + i) });
        if valid {
            let l = unsafe { larr.get_str_unchecked(loff + i) };
            let r = unsafe { rarr.get_str_unchecked(roff + i) };
            total_bytes += l.len() + r.len();
        }
    }

    // Allocate offsets and data buffers
    let mut offsets = Vec64::<T>::with_capacity(len + 1);
    unsafe {
        offsets.set_len(len + 1);
    }
    let mut values = Vec64::<u8>::with_capacity(total_bytes);

    // Fill values and offsets
    offsets[0] = T::zero();
    let mut cur = 0;

    for i in 0..len {
        let valid = lmask.map_or(true, |m| unsafe { m.get_unchecked(loff + i) })
            && rmask.map_or(true, |m| unsafe { m.get_unchecked(roff + i) });

        if valid {
            let l = unsafe { larr.get_str_unchecked(loff + i).as_bytes() };
            let r = unsafe { rarr.get_str_unchecked(roff + i).as_bytes() };

            values.extend_from_slice(l);
            values.extend_from_slice(r);
            cur += l.len() + r.len();

            unsafe {
                out_mask.set_unchecked(i, true);
            }
        } else {
            unsafe {
                out_mask.set_unchecked(i, false);
            }
        }

        offsets[i + 1] = T::from_usize(cur);
    }

    StringArray {
        offsets: offsets.into(),
        data: values.into(),
        null_mask: Some(out_mask),
    }
}

/// Concatenates corresponding string pairs from two categorical arrays element-wise.
///
/// Performs element-wise concatenation by looking up dictionary values from both
/// categorical arrays and concatenating the resolved strings. Creates a new string
/// array with the concatenated results.
///
/// # Parameters
/// - `lhs`: Left-hand categorical array view tuple `(CategoricalArray, offset, length)`
/// - `rhs`: Right-hand categorical array view tuple `(CategoricalArray, offset, length)`
///
/// # Returns
/// A new `StringArray<T>` containing concatenated dictionary strings.
///
/// # Null Handling
/// - If either categorical value is null, the result is null
/// - Output null mask reflects the union of input null positions
///
/// # Implementation
/// - Dictionary lookups resolve categorical codes to actual strings
/// - Memory allocation optimised based on total concatenated length
pub fn concat_dict_dict<T: Integer>(
    lhs: CategoricalAVT<T>,
    rhs: CategoricalAVT<T>,
) -> Result<CategoricalArray<T>, KernelError> {
    let (larr, loff, llen) = lhs;
    let (rarr, roff, rlen) = rhs;
    let len = llen.min(rlen);

    let lmask = larr.null_mask.as_ref();
    let rmask = rarr.null_mask.as_ref();
    let mut out_mask = Bitmask::new_set_all(len, false);
    let _ = confirm_mask_capacity(larr.data.len(), lmask)?;
    let _ = confirm_mask_capacity(rarr.data.len(), rmask)?;

    // Use max possible unique count for preallocation. Worst case is all unique.
    let mut data = Vec64::<T>::with_capacity(len);
    unsafe {
        data.set_len(len);
    }

    let mut unique_values = Vec64::<String>::with_capacity(len);
    #[cfg(feature = "fast_hash")]
    let mut seen: AHashMap<String, T> = AHashMap::with_capacity(len);
    #[cfg(not(feature = "fast_hash"))]
    let mut seen: HashMap<String, T> = HashMap::with_capacity(len);
    let mut unique_idx = 0;

    for i in 0..len {
        let valid = lmask.map_or(true, |m| unsafe { m.get_unchecked(loff + i) })
            && rmask.map_or(true, |m| unsafe { m.get_unchecked(roff + i) });

        if valid {
            let l = unsafe { larr.get_str_unchecked(loff + i) };
            let r = unsafe { rarr.get_str_unchecked(roff + i) };
            let cat = format!("{l}{r}");

            let idx = match seen.get(&cat) {
                Some(ix) => *ix,
                None => {
                    let ix = T::from_usize(unique_idx);
                    unique_values.push(cat.clone());
                    seen.insert(cat, ix);
                    unique_idx += 1;
                    ix
                }
            };

            unsafe {
                *data.get_unchecked_mut(i) = idx;
                out_mask.set_unchecked(i, true);
            }
        } else {
            unsafe {
                *data.get_unchecked_mut(i) = T::zero();
                out_mask.set_unchecked(i, false);
            }
        }
    }

    unsafe {
        unique_values.set_len(unique_idx);
    }

    Ok(CategoricalArray {
        data: data.into(),
        #[cfg(not(feature = "shared_dict"))]
        unique_values,
        #[cfg(feature = "shared_dict")]
        dictionary: Dictionary::from(unique_values),
        null_mask: Some(out_mask),
    })
}

/// Concatenates strings from a string array with dictionary values from a categorical array.
///
/// Performs element-wise concatenation where left operands come from a string array
/// and right operands are resolved from a categorical array's dictionary.
///
/// # Parameters
/// - `lhs`: String array view tuple `(StringArray, offset, length)`
/// - `rhs`: Categorical array view tuple `(CategoricalArray, offset, length)`
///
/// # Type Parameters
/// - `T`: Integer type for string array offsets
/// - `U`: Integer type for categorical array indices
///
/// # Returns
/// A new `StringArray<T>` containing concatenated string-dictionary pairs.
///
/// # Null Handling
/// Results are null if either input value is null at the corresponding position.
pub fn concat_str_dict<T: Integer, U: Integer>(
    lhs: StringAVT<T>,
    rhs: CategoricalAVT<U>,
) -> Result<StringArray<T>, KernelError> {
    let (larr, loff, llen) = lhs;
    let (rarr, roff, rlen) = rhs;
    let len = llen.min(rlen);

    let lmask = larr.null_mask.as_ref();
    let rmask = rarr.null_mask.as_ref();
    let mut out_mask = Bitmask::new_set_all(len, false);
    let _ = confirm_mask_capacity(larr.len(), lmask)?;
    let _ = confirm_mask_capacity(rarr.data.len(), rmask)?;

    // Compute total byte size required
    let mut total_bytes = 0;
    for i in 0..len {
        let valid = lmask.map_or(true, |m| unsafe { m.get_unchecked(loff + i) })
            && rmask.map_or(true, |m| unsafe { m.get_unchecked(roff + i) });
        if valid {
            let a = unsafe { larr.get_str_unchecked(loff + i) };
            let b = unsafe { rarr.get_str_unchecked(roff + i) };
            total_bytes += a.len() + b.len();
        }
    }

    // Preallocate offsets and values
    let mut offsets = Vec64::<T>::with_capacity(len + 1);
    unsafe {
        offsets.set_len(len + 1);
    }
    let mut values = Vec64::<u8>::with_capacity(total_bytes);

    // Fill values and offsets
    offsets[0] = T::zero();
    let mut cur = 0;

    for i in 0..len {
        let valid = lmask.map_or(true, |m| unsafe { m.get_unchecked(loff + i) })
            && rmask.map_or(true, |m| unsafe { m.get_unchecked(roff + i) });

        if valid {
            let a = unsafe { larr.get_str_unchecked(loff + i).as_bytes() };
            let b = unsafe { rarr.get_str_unchecked(roff + i).as_bytes() };

            values.extend_from_slice(a);
            values.extend_from_slice(b);
            cur += a.len() + b.len();

            unsafe {
                out_mask.set_unchecked(i, true);
            }
        } else {
            unsafe {
                out_mask.set_unchecked(i, false);
            }
        }

        offsets[i + 1] = T::from_usize(cur);
    }

    Ok(StringArray {
        offsets: offsets.into(),
        data: values.into(),
        null_mask: Some(out_mask),
    })
}

/// Concatenates dictionary values from a categorical array with strings from a string array.
///
/// Performs element-wise concatenation where left operands are resolved from a
/// categorical array's dictionary and right operands come from a string array.
///
/// # Parameters
/// - `lhs`: Categorical array view tuple `(CategoricalArray, offset, length)`
/// - `rhs`: String array view tuple `(StringArray, offset, length)`
///
/// # Type Parameters
/// - `T`: Integer type for string array offsets
/// - `U`: Integer type for categorical array indices
///
/// # Returns
/// A new `StringArray<T>` containing concatenated dictionary-string pairs.
///
/// # Null Handling
/// Results are null if either input value is null at the corresponding position.
pub fn concat_dict_str<T: Integer, U: Integer>(
    lhs: CategoricalAVT<U>,
    rhs: StringAVT<T>,
) -> Result<StringArray<T>, KernelError> {
    concat_str_dict(rhs, lhs)
}

macro_rules! binary_str_pred_loop {
    ($len:expr, $lmask:expr, $rmask:expr, $out_mask:expr, $lhs:expr, $rhs:expr, $method:ident) => {{
        let mut data = Bitmask::new_set_all($len, false);
        // ensure masks cover offset + len
        let lhs_off = $lhs.1;
        let rhs_off = $rhs.1;
        let _ = confirm_mask_capacity(lhs_off + $len, $lmask)?;
        let _ = confirm_mask_capacity(rhs_off + $len, $rmask)?;
        for i in 0..$len {
            let li = lhs_off + i;
            let ri = rhs_off + i;
            let valid = $lmask.map_or(true, |m| unsafe { m.get_unchecked(li) })
                && $rmask.map_or(true, |m| unsafe { m.get_unchecked(ri) });
            let result = valid && {
                let s = unsafe { $lhs.0.get_str_unchecked(li) };
                let pat = unsafe { $rhs.0.get_str_unchecked(ri) };
                !pat.is_empty() && s.$method(pat)
            };
            unsafe {
                data.set_unchecked(i, result);
                $out_mask.set_unchecked(i, valid);
            }
        }
        data
    }};
}

// STRING PREDICATES - contains/starts_with/ends_with

/// Generates string predicate functions that compare string arrays.
macro_rules! str_predicate {
    ($fn_name:ident, $method:ident) => {
        /// Performs string predicate operations between two string arrays.
        ///
        /// Applies the specified string method (contains, starts_with, ends_with)
        /// to compare corresponding elements of two string arrays.
        ///
        /// # Type Parameters
        ///
        /// * `T` - Integer type for left string array offsets
        /// * `U` - Integer type for right string array offsets
        ///
        /// # Arguments
        ///
        /// * `lhs` - Left string array (data, offset, length)
        /// * `rhs` - Right string array (data, offset, length)
        ///
        /// # Returns
        ///
        /// Boolean array containing comparison results
        pub fn $fn_name<T: Integer, U: Integer>(
            lhs: StringAVT<T>,
            rhs: StringAVT<U>,
        ) -> BooleanArray<()> {
            let (larr, loff, llen) = lhs;
            let (rarr, roff, rlen) = rhs;
            let len = llen.min(rlen);
            // Grab raw pointers & slices once
            let lmask = larr.null_mask.as_ref();
            let rmask = rarr.null_mask.as_ref();
            let mut out = Bitmask::new_set_all(len, false);

            for i in 0..len {
                unsafe {
                    // Check null‐mask validity without bounds checks
                    let lv = lmask.map_or(true, |m| m.get_unchecked(loff + i));
                    let rv = rmask.map_or(true, |m| m.get_unchecked(roff + i));
                    if !lv || !rv {
                        // leave out[i]=false
                        continue;
                    }
                    // Slice out the raw bytes
                    let ls = larr.offsets[loff + i].to_usize();
                    let le = larr.offsets[loff + i + 1].to_usize();
                    let rs = rarr.offsets[roff + i].to_usize();
                    let re = rarr.offsets[roff + i + 1].to_usize();
                    let s = std::str::from_utf8_unchecked(&larr.data[ls..le]);
                    let p = std::str::from_utf8_unchecked(&rarr.data[rs..re]);
                    // Only non-empty pattern can match
                    if !p.is_empty() && s.$method(p) {
                        out.set_unchecked(i, true);
                    }
                }
            }
            // Tight bitmask with no nulls - nulls became 'false'
            BooleanArray::new(out.into(), None)
        }
    };
}

/// Generates string-to-categorical predicate functions.
macro_rules! str_cat_predicate {
    ($fn_name:ident, $method:ident) => {
        /// Performs string predicate operations between string and categorical arrays.
        ///
        /// Applies the specified string method (contains, starts_with, ends_with)
        /// to compare string array elements with categorical array elements.
        ///
        /// # Type Parameters
        ///
        /// * `T` - Integer type for string array offsets
        /// * `U` - Integer type for categorical array offsets
        ///
        /// # Arguments
        ///
        /// * `lhs` - String array (data, offset, length)
        /// * `rhs` - Categorical array (data, offset, length)
        ///
        /// # Returns
        ///
        /// Result containing boolean array with comparison results, or error
        pub fn $fn_name<T: Integer, U: Integer>(
            lhs: StringAVT<T>,
            rhs: CategoricalAVT<U>,
        ) -> Result<BooleanArray<()>, KernelError> {
            let (larr, loff, llen) = lhs;
            let (rarr, roff, rlen) = rhs;
            let len = llen.min(rlen);

            let lmask = larr.null_mask.as_ref();
            let rmask = rarr.null_mask.as_ref();
            let mut out_mask = Bitmask::new_set_all(len, false);

            let data = binary_str_pred_loop!(
                len,
                lmask,
                rmask,
                out_mask,
                (larr, loff),
                (rarr, roff),
                $method
            );

            Ok(BooleanArray::new(data.into(), Some(out_mask)))
        }
    };
}

/// Generates categorical-to-categorical predicate functions.
macro_rules! cat_cat_predicate {
    ($fn_name:ident, $method:ident) => {
        /// Performs string predicate operations between two categorical arrays.
        ///
        /// Applies the specified string method (contains, starts_with, ends_with)
        /// to compare corresponding elements of two categorical arrays.
        ///
        /// # Type Parameters
        ///
        /// * `T` - Integer type for categorical array offsets
        ///
        /// # Arguments
        ///
        /// * `lhs` - Left categorical array (data, offset, length)
        /// * `rhs` - Right categorical array (data, offset, length)
        ///
        /// # Returns
        ///
        /// Result containing boolean array with comparison results, or error
        pub fn $fn_name<T: Integer>(
            lhs: CategoricalAVT<T>,
            rhs: CategoricalAVT<T>,
        ) -> Result<BooleanArray<()>, KernelError> {
            let (larr, loff, llen) = lhs;
            let (rarr, roff, rlen) = rhs;
            let len = llen.min(rlen);

            let lmask = larr.null_mask.as_ref();
            let rmask = rarr.null_mask.as_ref();
            let mut out_mask = Bitmask::new_set_all(len, false);

            let data = binary_str_pred_loop!(
                len,
                lmask,
                rmask,
                out_mask,
                (larr, loff),
                (rarr, roff),
                $method
            );

            Ok(BooleanArray::new(data.into(), Some(out_mask)))
        }
    };
}

/// Generates categorical-to-string predicate functions.
macro_rules! dict_str_predicate {
    ($fn_name:ident, $method:ident) => {
        /// Performs string predicate operations between categorical and string arrays.
        ///
        /// Applies the specified string method (contains, starts_with, ends_with)
        /// to compare categorical array elements with string array elements.
        ///
        /// # Type Parameters
        ///
        /// * `T` - Integer type for categorical array offsets
        /// * `U` - Integer type for string array offsets
        ///
        /// # Arguments
        ///
        /// * `lhs` - Categorical array (data, offset, length)
        /// * `rhs` - String array (data, offset, length)
        ///
        /// # Returns
        ///
        /// Result containing boolean array with comparison results, or error
        pub fn $fn_name<T: Integer, U: Integer>(
            lhs: CategoricalAVT<T>,
            rhs: StringAVT<U>,
        ) -> Result<BooleanArray<()>, KernelError> {
            let (larr, loff, llen) = lhs;
            let (rarr, roff, rlen) = rhs;
            let len = llen.min(rlen);

            let lmask = larr.null_mask.as_ref();
            let rmask = rarr.null_mask.as_ref();
            let mut out_mask = Bitmask::new_set_all(len, false);
            let _ = confirm_mask_capacity(larr.data.len(), lmask)?;
            let _ = confirm_mask_capacity(rarr.len(), rmask)?;

            let mut data = Bitmask::new_set_all(len, false);
            for i in 0..len {
                let valid = lmask.map_or(true, |m| unsafe { m.get_unchecked(loff + i) })
                    && rmask.map_or(true, |m| unsafe { m.get_unchecked(roff + i) });
                let match_i = valid && {
                    let hay = unsafe { larr.get_str_unchecked(loff + i) };
                    let needle = unsafe { rarr.get_str_unchecked(roff + i) };
                    !needle.is_empty() && hay.$method(needle)
                };
                unsafe { data.set_unchecked(i, match_i) };
                unsafe { out_mask.set_unchecked(i, valid) };
            }

            Ok(BooleanArray::new(data.into(), Some(out_mask)))
        }
    };
}

str_predicate!(contains_str_str, contains);
str_predicate!(starts_with_str_str, starts_with);
str_predicate!(ends_with_str_str, ends_with);
str_cat_predicate!(contains_str_dict, contains);
cat_cat_predicate!(contains_dict_dict, contains);
str_cat_predicate!(starts_with_str_dict, starts_with);
cat_cat_predicate!(starts_with_dict_dict, starts_with);
str_cat_predicate!(ends_with_str_dict, ends_with);
cat_cat_predicate!(ends_with_dict_dict, ends_with);
dict_str_predicate!(contains_dict_str, contains);
dict_str_predicate!(starts_with_dict_str, starts_with);
dict_str_predicate!(ends_with_dict_str, ends_with);

// Regex match

#[cfg(feature = "regex")]
macro_rules! regex_match_loop {
    ($len:expr, $lmask:expr, $rmask:expr, $out_mask:expr, $lhs_arr:expr, $lhs_off:expr, $rhs_arr:expr, $rhs_off:expr) => {{
        let mut data = Bitmask::new_set_all($len, false);
        let _ = confirm_mask_capacity($len + $lhs_off, $lmask)?;
        let _ = confirm_mask_capacity($len + $rhs_off, $rmask)?;
        for i in 0..$len {
            let valid = $lmask.map_or(true, |m| unsafe { m.get_unchecked($lhs_off + i) })
                && $rmask.map_or(true, |m| unsafe { m.get_unchecked($rhs_off + i) });
            let matched = if valid {
                let s = unsafe { $lhs_arr.get_str_unchecked($lhs_off + i) };
                let pat = unsafe { $rhs_arr.get_str_unchecked($rhs_off + i) };
                if pat.is_empty() {
                    false
                } else {
                    match Regex::new(pat) {
                        Ok(re) => re.is_match(s),
                        Err(_) => {
                            return Err(KernelError::InvalidArguments(
                                "Invalid regex string".to_string(),
                            ));
                        }
                    }
                }
            } else {
                false
            };
            unsafe { data.set_unchecked(i, matched) };
            unsafe { $out_mask.set_unchecked(i, valid) };
        }
        data
    }};
}

/// Applies regular expression pattern matching between two string arrays.
///
/// Evaluates regex patterns from the right-hand string array against corresponding
/// strings in the left-hand array, producing a boolean array indicating matches.
///
/// # Parameters
/// - `lhs`: Source string array view tuple `(StringArray, offset, length)`
/// - `rhs`: Pattern string array view tuple `(StringArray, offset, length)`
///
/// # Type Parameters
/// - `T`: Integer type for left array offsets
/// - `U`: Integer type for right array offsets
///
/// # Returns
/// `Result<BooleanArray<()>, KernelError>` where true indicates pattern match.
///
/// # Errors
/// Returns `KernelError` for invalid regular expression patterns.
///
/// # Feature Gate
/// Requires the `regex` feature to be enabled.
///
/// # Performance
/// - Regex compilation overhead amortised across bulk operations
/// - Pattern caching opportunities for repeated patterns
#[cfg(feature = "regex")]
pub fn regex_str_str<'a, T: Integer, U: Integer>(
    lhs: StringAVT<'a, T>,
    rhs: StringAVT<'a, U>,
) -> Result<BooleanArray<()>, KernelError> {
    let (larr, loff, llen) = lhs;
    let (rarr, roff, rlen) = rhs;
    let len = llen.min(rlen);
    let lmask = larr.null_mask.as_ref();
    let rmask = rarr.null_mask.as_ref();
    let mut out_mask = Bitmask::new_set_all(len, false);

    let data = regex_match_loop!(len, lmask, rmask, out_mask, larr, loff, rarr, roff);
    Ok(BooleanArray::new(data.into(), Some(out_mask)))
}

/// Applies regular expression patterns to categorical array values against string patterns.
///
/// Evaluates regex patterns from the string array against dictionary-resolved strings
/// from the categorical array, producing a boolean array indicating matches.
///
/// # Parameters
/// - `lhs`: Categorical array view tuple `(CategoricalArray, offset, length)`
/// - `rhs`: Pattern string array view tuple `(StringArray, offset, length)`
///
/// # Type Parameters
/// - `U`: Integer type for categorical array indices
/// - `T`: Integer type for pattern array offsets
///
/// # Returns
/// `Result<BooleanArray<()>, KernelError>` where true indicates pattern match.
///
/// # Errors
/// Returns `KernelError` for invalid regular expression patterns.
///
/// # Feature Gate
/// Requires the `regex` feature to be enabled.
#[cfg(feature = "regex")]
pub fn regex_dict_str<'a, U: Integer, T: Integer>(
    lhs: CategoricalAVT<'a, U>,
    rhs: StringAVT<'a, T>,
) -> Result<BooleanArray<()>, KernelError> {
    let (larr, loff, llen) = lhs;
    let (rarr, roff, rlen) = rhs;
    let len = llen.min(rlen);
    let lmask = larr.null_mask.as_ref();
    let rmask = rarr.null_mask.as_ref();
    let mut out_mask = Bitmask::new_set_all(len, false);

    let data = regex_match_loop!(len, lmask, rmask, out_mask, larr, loff, rarr, roff);
    Ok(BooleanArray::new(data.into(), Some(out_mask)))
}

/// Applies regular expression patterns from categorical dictionary against string values.
///
/// Evaluates regex patterns resolved from the categorical array's dictionary against
/// corresponding strings in the string array, producing a boolean array of matches.
///
/// # Parameters
/// - `lhs`: Source string array view tuple `(StringArray, offset, length)`
/// - `rhs`: Pattern categorical array view tuple `(CategoricalArray, offset, length)`
///
/// # Type Parameters
/// - `T`: Integer type for string array offsets
/// - `U`: Integer type for categorical array indices
///
/// # Returns
/// `Result<BooleanArray<()>, KernelError>` where true indicates pattern match.
///
/// # Errors
/// Returns `KernelError` for invalid regular expression patterns in dictionary.
///
/// # Feature Gate
/// Requires the `regex` feature to be enabled.
#[cfg(feature = "regex")]
pub fn regex_str_dict<'a, T: Integer, U: Integer>(
    lhs: StringAVT<'a, T>,
    rhs: CategoricalAVT<'a, U>,
) -> Result<BooleanArray<()>, KernelError> {
    let (larr, loff, llen) = lhs;
    let (rarr, roff, rlen) = rhs;
    let len = llen.min(rlen);
    let lmask = larr.null_mask.as_ref();
    let rmask = rarr.null_mask.as_ref();
    let mut out_mask = Bitmask::new_set_all(len, false);

    let data = regex_match_loop!(len, lmask, rmask, out_mask, larr, loff, rarr, roff);
    Ok(BooleanArray::new(data.into(), Some(out_mask)))
}

/// Applies regular expression patterns between two categorical arrays via dictionary lookup.
///
/// Evaluates regex patterns by resolving both pattern and target strings from their
/// respective categorical dictionaries, producing a boolean array indicating matches.
///
/// # Parameters
/// - `lhs`: Source categorical array view tuple `(CategoricalArray, offset, length)`
/// - `rhs`: Pattern categorical array view tuple `(CategoricalArray, offset, length)`
///
/// # Type Parameters
/// - `T`: Integer type for categorical array indices
///
/// # Returns
/// `Result<BooleanArray<()>, KernelError>` where true indicates pattern match.
///
/// # Errors
/// Returns `KernelError` for invalid regular expression patterns in dictionaries.
///
/// # Feature Gate
/// Requires the `regex` feature to be enabled.
///
/// # Performance
/// - Dictionary lookups amortised across categorical operations
/// - Pattern compilation cached for repeated dictionary patterns
#[cfg(feature = "regex")]
pub fn regex_dict_dict<'a, T: Integer>(
    lhs: CategoricalAVT<'a, T>,
    rhs: CategoricalAVT<'a, T>,
) -> Result<BooleanArray<()>, KernelError> {
    let (larr, loff, llen) = lhs;
    let (rarr, roff, rlen) = rhs;
    let len = llen.min(rlen);
    let lmask = larr.null_mask.as_ref();
    let rmask = rarr.null_mask.as_ref();
    let mut out_mask = Bitmask::new_set_all(len, false);

    let data = regex_match_loop!(len, lmask, rmask, out_mask, larr, loff, rarr, roff);
    Ok(BooleanArray::new(data.into(), Some(out_mask)))
}

/// Computes the character length of each string in a `StringArray<T>` slice,
/// returning an `IntegerArray<T>` with the same length and null semantics.
///
/// This applies to a windowed slice (`offset`, `len`) of the input array.
/// The output null mask mirrors the sliced portion of the input mask.
///
/// # Parameters
/// - `input`: A `(StringArray<T>, offset, len)` tuple defining the slice to operate on.
///
/// # Returns
/// An `IntegerArray<T>` of the same length, with each element representing the character count
/// of the corresponding (non-null) string value.
pub fn len_str<'a, T: Integer + Copy>(
    input: StringAVT<'a, T>,
) -> Result<IntegerArray<T>, KernelError> {
    let (array, offset, len) = input;
    debug_assert!(offset + len <= array.offsets.len() - 1);

    let mask_opt = array.null_mask.as_ref().map(|orig| {
        let mut m = Bitmask::new_set_all(len, true);
        for i in 0..len {
            unsafe {
                m.set_unchecked(i, orig.get_unchecked(offset + i));
            }
        }
        m
    });

    let mut data = Vec64::<T>::with_capacity(len);
    unsafe { data.set_len(len) };
    for i in 0..len {
        let valid = mask_opt
            .as_ref()
            .map_or(true, |m| unsafe { m.get_unchecked(i) });
        if valid {
            let start = array.offsets[offset + i].to_usize();
            let end = array.offsets[offset + i + 1].to_usize();
            let s = unsafe { std::str::from_utf8_unchecked(&array.data[start..end]) };
            data[i] = T::from(s.chars().count()).unwrap();
        } else {
            data[i] = T::zero();
        }
    }

    Ok(IntegerArray {
        data: data.into(),
        null_mask: mask_opt,
    })
}

/// Computes the character length of each string in a `CategoricalArray<T>` slice,
/// returning an `IntegerArray<T>` with the same length and null semantics.
///
/// This applies to a windowed slice (`offset`, `len`) of the input categorical array,
/// using dictionary lookup to resolve each string.
///
/// # Parameters
/// - `input`: A `(CategoricalArray<T>, offset, len)` tuple defining the slice to operate on.
///
/// # Returns
/// An `IntegerArray<T>` of the same length, with each element representing the character count
/// of the corresponding (non-null) resolved string.
pub fn len_dict<'a, T: Integer>(
    input: CategoricalAVT<'a, T>,
) -> Result<IntegerArray<T>, KernelError> {
    let (array, offset, len) = input;
    debug_assert!(offset + len <= array.data.len());

    let mask_opt = array.null_mask.as_ref().map(|orig| {
        let mut m = Bitmask::new_set_all(len, true);
        for i in 0..len {
            unsafe {
                m.set_unchecked(i, orig.get_unchecked(offset + i));
            }
        }
        m
    });

    let mut data = Vec64::<T>::with_capacity(len);
    unsafe { data.set_len(len) };
    for i in 0..len {
        let valid = mask_opt
            .as_ref()
            .map_or(true, |m| unsafe { m.get_unchecked(i) });
        data[i] = if valid {
            T::from(
                unsafe { array.get_str_unchecked(offset + i) }
                    .chars()
                    .count(),
            )
            .unwrap()
        } else {
            T::zero()
        };
    }

    Ok(IntegerArray {
        data: data.into(),
        null_mask: mask_opt,
    })
}

/// Finds the lexicographically minimum string in a string array window.
///
/// Scans through a windowed portion of a string array to determine the minimum
/// string value according to lexicographic ordering, ignoring null values.
///
/// # Parameters
/// - `window`: String array view tuple `(StringArray, offset, length)` defining the scan window
///
/// # Returns
/// `Option<String>` containing the minimum string, or `None` if all values are null.
#[inline]
pub fn min_string_array<T: Integer>(window: StringAVT<T>) -> Option<String> {
    let (arr, offset, len) = window;
    let mut min_str: Option<&str> = None;
    for i in offset..offset + len {
        if arr
            .null_mask
            .as_ref()
            .map_or(true, |b| unsafe { b.get_unchecked(i) })
        {
            let s = unsafe { arr.get_str_unchecked(i) };
            if min_str.map_or(true, |min| s < min) {
                min_str = Some(s);
            }
        }
    }
    min_str.map(str::to_owned)
}

/// Finds the lexicographically maximum string in a string array window.
///
/// Scans through a windowed portion of a string array to determine the maximum
/// string value according to lexicographic ordering, ignoring null values.
///
/// # Parameters
/// - `window`: String array view tuple `(StringArray, offset, length)` defining the scan window
///
/// # Returns
/// `Option<String>` containing the maximum string, or `None` if all values are null.
#[inline]
pub fn max_string_array<T: Integer>(window: StringAVT<T>) -> Option<String> {
    let (arr, offset, len) = window;
    let mut max_str: Option<&str> = None;
    for i in offset..offset + len {
        if arr
            .null_mask
            .as_ref()
            .map_or(true, |b| unsafe { b.get_unchecked(i) })
        {
            let s = unsafe { arr.get_str_unchecked(i) };
            if max_str.map_or(true, |max| s > max) {
                max_str = Some(s);
            }
        }
    }
    max_str.map(str::to_owned)
}

/// Finds the lexicographically minimum dictionary string in a categorical array window.
///
/// Scans through a windowed portion of a categorical array, resolves dictionary values,
/// and determines the minimum string according to lexicographic ordering.
///
/// # Parameters
/// - `window`: Categorical array view tuple `(CategoricalArray, offset, length)` defining the scan window
///
/// # Returns
/// `Option<String>` containing the minimum dictionary string, or `None` if all values are null.
#[inline]
pub fn min_categorical_array<T: Integer>(window: CategoricalAVT<T>) -> Option<String> {
    let (arr, offset, len) = window;
    let mut min_code: Option<T> = None;
    for i in offset..offset + len {
        if arr
            .null_mask
            .as_ref()
            .map_or(true, |b| unsafe { b.get_unchecked(i) })
        {
            let code = arr.data[i];
            if min_code.map_or(true, |min| {
                let sc = &arr.unique_values()[code.to_usize()];
                let sm = &arr.unique_values()[min.to_usize()];
                sc < sm
            }) {
                min_code = Some(code);
            }
        }
    }
    min_code.map(|code| arr.unique_values()[code.to_usize()].clone())
}

/// Finds the lexicographically maximum dictionary string in a categorical array window.
///
/// Scans through a windowed portion of a categorical array, resolves dictionary values,
/// and determines the maximum string according to lexicographic ordering.
///
/// # Parameters
/// - `window`: Categorical array view tuple `(CategoricalArray, offset, length)` defining the scan window
///
/// # Returns
/// `Option<String>` containing the maximum dictionary string, or `None` if all values are null.
#[inline]
pub fn max_categorical_array<T: Integer>(window: CategoricalAVT<T>) -> Option<String> {
    let (arr, offset, len) = window;
    let mut max_code: Option<T> = None;
    for i in offset..offset + len {
        if arr
            .null_mask
            .as_ref()
            .map_or(true, |b| unsafe { b.get_unchecked(i) })
        {
            let code = arr.data[i];
            if max_code.map_or(true, |max| {
                let sc = &arr.unique_values()[code.to_usize()];
                let sm = &arr.unique_values()[max.to_usize()];
                sc > sm
            }) {
                max_code = Some(code);
            }
        }
    }
    max_code.map(|code| arr.unique_values()[code.to_usize()].clone())
}

/// Counts the number of distinct string values in a string array window.
///
/// Computes the cardinality of unique strings within a windowed portion of a
/// string array, using efficient hash-based deduplication.
///
/// # Parameters
/// - `window`: String array view tuple `(StringArray, offset, length)` defining the count window
///
/// # Returns
/// `usize` representing the count of distinct non-null string values.
///
/// # Hash Algorithm
/// Uses either AHash (if `fast_hash` feature enabled) or standard HashMap for deduplication.
#[inline(always)]
pub fn count_distinct_string<T: Integer>(window: StringAVT<T>) -> usize {
    let (arr, offset, len) = window;
    #[cfg(feature = "fast_hash")]
    let mut set = AHashSet::with_capacity(len);
    #[cfg(not(feature = "fast_hash"))]
    let mut set = HashSet::with_capacity(len);
    let null_mask = arr.null_mask.as_ref();

    for i in offset..offset + len {
        let valid = null_mask.map_or(true, |b| unsafe { b.get_unchecked(i) });
        if valid {
            let s = unsafe { arr.get_str_unchecked(i) };
            set.insert(s);
            if set.len() == len {
                break;
            }
        }
    }
    set.len()
}

// Unary string transformations

/// Generates a unary string transform kernel for StringArray.
/// The transform closure receives `&str` and returns `String`.
macro_rules! unary_str_transform {
    ($fn_name:ident, $doc:expr, $transform:expr) => {
        #[doc = $doc]
        pub fn $fn_name<T: Integer>(input: StringAVT<T>) -> Result<StringArray<T>, KernelError> {
            let (arr, offset, len) = input;

            let mask_opt = arr.null_mask.as_ref().map(|orig| {
                let mut m = Bitmask::new_set_all(len, true);
                for i in 0..len {
                    unsafe { m.set_unchecked(i, orig.get_unchecked(offset + i)); }
                }
                m
            });

            let input_bytes = arr.offsets[offset + len].to_usize()
                - arr.offsets[offset].to_usize();
            let mut offsets = Vec64::<T>::with_capacity(len + 1);
            unsafe { offsets.set_len(len + 1); }
            let mut data = Vec64::<u8>::with_capacity(input_bytes);

            offsets[0] = T::zero();
            let mut cur = 0usize;

            for i in 0..len {
                let valid = mask_opt.as_ref()
                    .map_or(true, |m| unsafe { m.get_unchecked(i) });

                if valid {
                    let s = unsafe { arr.get_str_unchecked(offset + i) };
                    let t: String = ($transform)(s);
                    data.extend_from_slice(t.as_bytes());
                    cur += t.len();
                }
                offsets[i + 1] = T::from_usize(cur);
            }

            Ok(StringArray {
                offsets: offsets.into(),
                data: data.into(),
                null_mask: mask_opt,
            })
        }
    };
}

/// Generates a unary string transform kernel for CategoricalArray.
/// Transforms dictionary values once and remaps indices.
macro_rules! unary_dict_transform {
    ($fn_name:ident, $doc:expr, $transform:expr) => {
        #[doc = $doc]
        pub fn $fn_name<T: Integer>(input: CategoricalAVT<T>) -> Result<CategoricalArray<T>, KernelError> {
            let (arr, offset, len) = input;

            let mask_opt = arr.null_mask.as_ref().map(|orig| {
                let mut m = Bitmask::new_set_all(len, true);
                for i in 0..len {
                    unsafe { m.set_unchecked(i, orig.get_unchecked(offset + i)); }
                }
                m
            });

            // Transform each dictionary value once, build old->new index mapping
            #[cfg(feature = "fast_hash")]
            let mut seen: AHashMap<String, T> = AHashMap::with_capacity(arr.unique_values().len());
            #[cfg(not(feature = "fast_hash"))]
            let mut seen: std::collections::HashMap<String, T> =
                std::collections::HashMap::with_capacity(arr.unique_values().len());

            let mut new_unique = Vec64::<String>::new();
            let mut idx_map = Vec64::<T>::with_capacity(arr.unique_values().len());

            for old_val in arr.unique_values().iter() {
                let t: String = ($transform)(old_val.as_str());
                let new_idx = match seen.get(&t) {
                    Some(&ix) => ix,
                    None => {
                        let ix = T::from_usize(new_unique.len());
                        new_unique.push(t.clone());
                        seen.insert(t, ix);
                        ix
                    }
                };
                idx_map.push(new_idx);
            }

            // Remap indices for the window
            let mut data = Vec64::<T>::with_capacity(len);
            unsafe { data.set_len(len); }
            for i in 0..len {
                let valid = mask_opt.as_ref().map_or(true, |m| unsafe { m.get_unchecked(i) });
                data[i] = if valid {
                    idx_map[arr.data[offset + i].to_usize()]
                } else {
                    T::zero()
                };
            }

            Ok(CategoricalArray {
                data: data.into(),
                #[cfg(not(feature = "shared_dict"))]
        unique_values: new_unique,
        #[cfg(feature = "shared_dict")]
        dictionary: Dictionary::from(new_unique),
                null_mask: mask_opt,
            })
        }
    };
}

unary_str_transform!(to_uppercase_str,
    "Converts each string element to uppercase.",
    |s: &str| s.to_uppercase()
);
unary_dict_transform!(to_uppercase_dict,
    "Converts each categorical string element to uppercase.",
    |s: &str| s.to_uppercase()
);

unary_str_transform!(to_lowercase_str,
    "Converts each string element to lowercase.",
    |s: &str| s.to_lowercase()
);
unary_dict_transform!(to_lowercase_dict,
    "Converts each categorical string element to lowercase.",
    |s: &str| s.to_lowercase()
);

unary_str_transform!(trim_str,
    "Trims leading and trailing whitespace from each string element.",
    |s: &str| s.trim().to_owned()
);
unary_dict_transform!(trim_dict,
    "Trims leading and trailing whitespace from each categorical string element.",
    |s: &str| s.trim().to_owned()
);

unary_str_transform!(ltrim_str,
    "Trims leading whitespace from each string element.",
    |s: &str| s.trim_start().to_owned()
);
unary_dict_transform!(ltrim_dict,
    "Trims leading whitespace from each categorical string element.",
    |s: &str| s.trim_start().to_owned()
);

unary_str_transform!(rtrim_str,
    "Trims trailing whitespace from each string element.",
    |s: &str| s.trim_end().to_owned()
);
unary_dict_transform!(rtrim_dict,
    "Trims trailing whitespace from each categorical string element.",
    |s: &str| s.trim_end().to_owned()
);

unary_str_transform!(reverse_str,
    "Reverses each string element by Unicode characters.",
    |s: &str| s.chars().rev().collect::<String>()
);
unary_dict_transform!(reverse_dict,
    "Reverses each categorical string element by Unicode characters.",
    |s: &str| s.chars().rev().collect::<String>()
);

// Byte length

/// Computes the byte length of each string in a StringArray slice.
pub fn byte_length_str<T: Integer + Copy>(
    input: StringAVT<T>,
) -> Result<IntegerArray<T>, KernelError> {
    let (array, offset, len) = input;

    let mask_opt = array.null_mask.as_ref().map(|orig| {
        let mut m = Bitmask::new_set_all(len, true);
        for i in 0..len {
            unsafe { m.set_unchecked(i, orig.get_unchecked(offset + i)); }
        }
        m
    });

    let mut data = Vec64::<T>::with_capacity(len);
    unsafe { data.set_len(len); }
    for i in 0..len {
        let valid = mask_opt.as_ref().map_or(true, |m| unsafe { m.get_unchecked(i) });
        if valid {
            let start = array.offsets[offset + i].to_usize();
            let end = array.offsets[offset + i + 1].to_usize();
            data[i] = T::from_usize(end - start);
        } else {
            data[i] = T::zero();
        }
    }

    Ok(IntegerArray {
        data: data.into(),
        null_mask: mask_opt,
    })
}

/// Computes the byte length of each string in a CategoricalArray slice.
pub fn byte_length_dict<T: Integer>(
    input: CategoricalAVT<T>,
) -> Result<IntegerArray<T>, KernelError> {
    let (array, offset, len) = input;

    let mask_opt = array.null_mask.as_ref().map(|orig| {
        let mut m = Bitmask::new_set_all(len, true);
        for i in 0..len {
            unsafe { m.set_unchecked(i, orig.get_unchecked(offset + i)); }
        }
        m
    });

    let mut data = Vec64::<T>::with_capacity(len);
    unsafe { data.set_len(len); }
    for i in 0..len {
        let valid = mask_opt.as_ref().map_or(true, |m| unsafe { m.get_unchecked(i) });
        data[i] = if valid {
            T::from_usize(unsafe { array.get_str_unchecked(offset + i) }.len())
        } else {
            T::zero()
        };
    }

    Ok(IntegerArray {
        data: data.into(),
        null_mask: mask_opt,
    })
}

// Find and count

/// Finds the first byte index of `needle` in each string element. Returns -1 if not found.
pub fn find_str<T: Integer>(
    input: StringAVT<T>,
    needle: &str,
) -> Result<IntegerArray<i32>, KernelError> {
    let (arr, offset, len) = input;

    let mask_opt = arr.null_mask.as_ref().map(|orig| {
        let mut m = Bitmask::new_set_all(len, true);
        for i in 0..len {
            unsafe { m.set_unchecked(i, orig.get_unchecked(offset + i)); }
        }
        m
    });

    let mut data = Vec64::<i32>::with_capacity(len);
    unsafe { data.set_len(len); }
    for i in 0..len {
        let valid = mask_opt.as_ref().map_or(true, |m| unsafe { m.get_unchecked(i) });
        data[i] = if valid {
            let s = unsafe { arr.get_str_unchecked(offset + i) };
            s.find(needle).map_or(-1, |pos| pos as i32)
        } else {
            0
        };
    }

    Ok(IntegerArray {
        data: data.into(),
        null_mask: mask_opt,
    })
}

/// Finds the first byte index of `needle` in each categorical string element. Returns -1 if not found.
pub fn find_dict<T: Integer>(
    input: CategoricalAVT<T>,
    needle: &str,
) -> Result<IntegerArray<i32>, KernelError> {
    let (arr, offset, len) = input;

    let mask_opt = arr.null_mask.as_ref().map(|orig| {
        let mut m = Bitmask::new_set_all(len, true);
        for i in 0..len {
            unsafe { m.set_unchecked(i, orig.get_unchecked(offset + i)); }
        }
        m
    });

    let mut data = Vec64::<i32>::with_capacity(len);
    unsafe { data.set_len(len); }
    for i in 0..len {
        let valid = mask_opt.as_ref().map_or(true, |m| unsafe { m.get_unchecked(i) });
        data[i] = if valid {
            let s = unsafe { arr.get_str_unchecked(offset + i) };
            s.find(needle).map_or(-1, |pos| pos as i32)
        } else {
            0
        };
    }

    Ok(IntegerArray {
        data: data.into(),
        null_mask: mask_opt,
    })
}

/// Counts non-overlapping occurrences of `needle` in each string element.
pub fn count_match_str<T: Integer>(
    input: StringAVT<T>,
    needle: &str,
) -> Result<IntegerArray<i32>, KernelError> {
    let (arr, offset, len) = input;

    let mask_opt = arr.null_mask.as_ref().map(|orig| {
        let mut m = Bitmask::new_set_all(len, true);
        for i in 0..len {
            unsafe { m.set_unchecked(i, orig.get_unchecked(offset + i)); }
        }
        m
    });

    let mut data = Vec64::<i32>::with_capacity(len);
    unsafe { data.set_len(len); }
    for i in 0..len {
        let valid = mask_opt.as_ref().map_or(true, |m| unsafe { m.get_unchecked(i) });
        data[i] = if valid {
            let s = unsafe { arr.get_str_unchecked(offset + i) };
            s.matches(needle).count() as i32
        } else {
            0
        };
    }

    Ok(IntegerArray {
        data: data.into(),
        null_mask: mask_opt,
    })
}

/// Counts non-overlapping occurrences of `needle` in each categorical string element.
pub fn count_match_dict<T: Integer>(
    input: CategoricalAVT<T>,
    needle: &str,
) -> Result<IntegerArray<i32>, KernelError> {
    let (arr, offset, len) = input;

    let mask_opt = arr.null_mask.as_ref().map(|orig| {
        let mut m = Bitmask::new_set_all(len, true);
        for i in 0..len {
            unsafe { m.set_unchecked(i, orig.get_unchecked(offset + i)); }
        }
        m
    });

    let mut data = Vec64::<i32>::with_capacity(len);
    unsafe { data.set_len(len); }
    for i in 0..len {
        let valid = mask_opt.as_ref().map_or(true, |m| unsafe { m.get_unchecked(i) });
        data[i] = if valid {
            let s = unsafe { arr.get_str_unchecked(offset + i) };
            s.matches(needle).count() as i32
        } else {
            0
        };
    }

    Ok(IntegerArray {
        data: data.into(),
        null_mask: mask_opt,
    })
}

// Substring

/// Extracts a character-based substring from each string element.
/// `start` is the 0-based character offset. `opt_len` limits the number of characters taken.
pub fn substring_str<T: Integer>(
    input: StringAVT<T>,
    start: usize,
    opt_len: Option<usize>,
) -> Result<StringArray<T>, KernelError> {
    let (arr, offset, len) = input;
    let mask_opt = arr.null_mask.as_ref().map(|orig| {
        let mut m = Bitmask::new_set_all(len, true);
        for i in 0..len {
            unsafe { m.set_unchecked(i, orig.get_unchecked(offset + i)); }
        }
        m
    });

    let input_bytes = arr.offsets[offset + len].to_usize()
        - arr.offsets[offset].to_usize();
    let mut offsets = Vec64::<T>::with_capacity(len + 1);
    unsafe { offsets.set_len(len + 1); }
    let mut data = Vec64::<u8>::with_capacity(input_bytes);

    offsets[0] = T::zero();
    let mut cur = 0usize;

    for i in 0..len {
        let valid = mask_opt.as_ref().map_or(true, |m| unsafe { m.get_unchecked(i) });
        if valid {
            let s = unsafe { arr.get_str_unchecked(offset + i) };
            let chars = s.chars().skip(start);
            let t: String = match opt_len {
                Some(n) => chars.take(n).collect(),
                None => chars.collect(),
            };
            data.extend_from_slice(t.as_bytes());
            cur += t.len();
        }
        offsets[i + 1] = T::from_usize(cur);
    }

    Ok(StringArray {
        offsets: offsets.into(),
        data: data.into(),
        null_mask: mask_opt,
    })
}

/// Extracts a character-based substring from each categorical string element.
pub fn substring_dict<T: Integer>(
    input: CategoricalAVT<T>,
    start: usize,
    opt_len: Option<usize>,
) -> Result<CategoricalArray<T>, KernelError> {
    let (arr, offset, len) = input;
    let mask_opt = arr.null_mask.as_ref().map(|orig| {
        let mut m = Bitmask::new_set_all(len, true);
        for i in 0..len {
            unsafe { m.set_unchecked(i, orig.get_unchecked(offset + i)); }
        }
        m
    });

    #[cfg(feature = "fast_hash")]
    let mut seen: AHashMap<String, T> = AHashMap::with_capacity(arr.unique_values().len());
    #[cfg(not(feature = "fast_hash"))]
    let mut seen: std::collections::HashMap<String, T> =
        std::collections::HashMap::with_capacity(arr.unique_values().len());

    let mut new_unique = Vec64::<String>::new();
    let mut idx_map = Vec64::<T>::with_capacity(arr.unique_values().len());

    for old_val in arr.unique_values().iter() {
        let chars = old_val.chars().skip(start);
        let t: String = match opt_len {
            Some(n) => chars.take(n).collect(),
            None => chars.collect(),
        };
        let new_idx = match seen.get(&t) {
            Some(&ix) => ix,
            None => {
                let ix = T::from_usize(new_unique.len());
                new_unique.push(t.clone());
                seen.insert(t, ix);
                ix
            }
        };
        idx_map.push(new_idx);
    }

    let mut data = Vec64::<T>::with_capacity(len);
    unsafe { data.set_len(len); }
    for i in 0..len {
        let valid = mask_opt.as_ref().map_or(true, |m| unsafe { m.get_unchecked(i) });
        data[i] = if valid {
            idx_map[arr.data[offset + i].to_usize()]
        } else {
            T::zero()
        };
    }

    Ok(CategoricalArray {
        data: data.into(),
        #[cfg(not(feature = "shared_dict"))]
        unique_values: new_unique,
        #[cfg(feature = "shared_dict")]
        dictionary: Dictionary::from(new_unique),
        null_mask: mask_opt,
    })
}

// Replace

/// Replaces all occurrences of `from` with `to` in each string element.
pub fn replace_str<T: Integer>(
    input: StringAVT<T>,
    from: &str,
    to: &str,
) -> Result<StringArray<T>, KernelError> {
    let (arr, offset, len) = input;
    let mask_opt = arr.null_mask.as_ref().map(|orig| {
        let mut m = Bitmask::new_set_all(len, true);
        for i in 0..len {
            unsafe { m.set_unchecked(i, orig.get_unchecked(offset + i)); }
        }
        m
    });

    let input_bytes = arr.offsets[offset + len].to_usize()
        - arr.offsets[offset].to_usize();
    let mut offsets = Vec64::<T>::with_capacity(len + 1);
    unsafe { offsets.set_len(len + 1); }
    let mut data = Vec64::<u8>::with_capacity(input_bytes);

    offsets[0] = T::zero();
    let mut cur = 0usize;

    for i in 0..len {
        let valid = mask_opt.as_ref().map_or(true, |m| unsafe { m.get_unchecked(i) });
        if valid {
            let s = unsafe { arr.get_str_unchecked(offset + i) };
            let t = s.replace(from, to);
            data.extend_from_slice(t.as_bytes());
            cur += t.len();
        }
        offsets[i + 1] = T::from_usize(cur);
    }

    Ok(StringArray {
        offsets: offsets.into(),
        data: data.into(),
        null_mask: mask_opt,
    })
}

/// Replaces all occurrences of `from` with `to` in each categorical string element.
pub fn replace_dict<T: Integer>(
    input: CategoricalAVT<T>,
    from: &str,
    to: &str,
) -> Result<CategoricalArray<T>, KernelError> {
    let (arr, offset, len) = input;
    let mask_opt = arr.null_mask.as_ref().map(|orig| {
        let mut m = Bitmask::new_set_all(len, true);
        for i in 0..len {
            unsafe { m.set_unchecked(i, orig.get_unchecked(offset + i)); }
        }
        m
    });

    #[cfg(feature = "fast_hash")]
    let mut seen: AHashMap<String, T> = AHashMap::with_capacity(arr.unique_values().len());
    #[cfg(not(feature = "fast_hash"))]
    let mut seen: std::collections::HashMap<String, T> =
        std::collections::HashMap::with_capacity(arr.unique_values().len());

    let mut new_unique = Vec64::<String>::new();
    let mut idx_map = Vec64::<T>::with_capacity(arr.unique_values().len());

    for old_val in arr.unique_values().iter() {
        let t = old_val.replace(from, to);
        let new_idx = match seen.get(&t) {
            Some(&ix) => ix,
            None => {
                let ix = T::from_usize(new_unique.len());
                new_unique.push(t.clone());
                seen.insert(t, ix);
                ix
            }
        };
        idx_map.push(new_idx);
    }

    let mut data = Vec64::<T>::with_capacity(len);
    unsafe { data.set_len(len); }
    for i in 0..len {
        let valid = mask_opt.as_ref().map_or(true, |m| unsafe { m.get_unchecked(i) });
        data[i] = if valid {
            idx_map[arr.data[offset + i].to_usize()]
        } else {
            T::zero()
        };
    }

    Ok(CategoricalArray {
        data: data.into(),
        #[cfg(not(feature = "shared_dict"))]
        unique_values: new_unique,
        #[cfg(feature = "shared_dict")]
        dictionary: Dictionary::from(new_unique),
        null_mask: mask_opt,
    })
}

// Repeat

/// Repeats each string element `n` times.
pub fn repeat_str<T: Integer>(
    input: StringAVT<T>,
    n: usize,
) -> Result<StringArray<T>, KernelError> {
    let (arr, offset, len) = input;
    let mask_opt = arr.null_mask.as_ref().map(|orig| {
        let mut m = Bitmask::new_set_all(len, true);
        for i in 0..len {
            unsafe { m.set_unchecked(i, orig.get_unchecked(offset + i)); }
        }
        m
    });

    let input_bytes = arr.offsets[offset + len].to_usize()
        - arr.offsets[offset].to_usize();
    let mut offsets = Vec64::<T>::with_capacity(len + 1);
    unsafe { offsets.set_len(len + 1); }
    let mut data = Vec64::<u8>::with_capacity(input_bytes * n);

    offsets[0] = T::zero();
    let mut cur = 0usize;

    for i in 0..len {
        let valid = mask_opt.as_ref().map_or(true, |m| unsafe { m.get_unchecked(i) });
        if valid {
            let s = unsafe { arr.get_str_unchecked(offset + i) };
            let bytes = s.as_bytes();
            for _ in 0..n {
                data.extend_from_slice(bytes);
            }
            cur += bytes.len() * n;
        }
        offsets[i + 1] = T::from_usize(cur);
    }

    Ok(StringArray {
        offsets: offsets.into(),
        data: data.into(),
        null_mask: mask_opt,
    })
}

/// Repeats each categorical string element `n` times.
pub fn repeat_dict<T: Integer>(
    input: CategoricalAVT<T>,
    n: usize,
) -> Result<CategoricalArray<T>, KernelError> {
    let (arr, offset, len) = input;
    let mask_opt = arr.null_mask.as_ref().map(|orig| {
        let mut m = Bitmask::new_set_all(len, true);
        for i in 0..len {
            unsafe { m.set_unchecked(i, orig.get_unchecked(offset + i)); }
        }
        m
    });

    // Repeat is a 1:1 mapping on dictionary values, no merging possible
    let mut new_unique = Vec64::<String>::new();
    for old_val in arr.unique_values().iter() {
        new_unique.push(old_val.repeat(n));
    }

    let mut data = Vec64::<T>::with_capacity(len);
    unsafe { data.set_len(len); }
    for i in 0..len {
        let valid = mask_opt.as_ref().map_or(true, |m| unsafe { m.get_unchecked(i) });
        data[i] = if valid {
            arr.data[offset + i]
        } else {
            T::zero()
        };
    }

    Ok(CategoricalArray {
        data: data.into(),
        #[cfg(not(feature = "shared_dict"))]
        unique_values: new_unique,
        #[cfg(feature = "shared_dict")]
        dictionary: Dictionary::from(new_unique),
        null_mask: mask_opt,
    })
}

// Pad

/// Pads each string element to `width` characters using `fill_char`.
pub fn pad_str<T: Integer>(
    input: StringAVT<T>,
    width: usize,
    fill_char: char,
    side: PadSide,
) -> Result<StringArray<T>, KernelError> {
    let (arr, offset, len) = input;
    let mask_opt = arr.null_mask.as_ref().map(|orig| {
        let mut m = Bitmask::new_set_all(len, true);
        for i in 0..len {
            unsafe { m.set_unchecked(i, orig.get_unchecked(offset + i)); }
        }
        m
    });

    let max_padded = width * fill_char.len_utf8();
    let mut offsets = Vec64::<T>::with_capacity(len + 1);
    unsafe { offsets.set_len(len + 1); }
    let mut data = Vec64::<u8>::with_capacity(len * max_padded);

    offsets[0] = T::zero();
    let mut cur = 0usize;

    for i in 0..len {
        let valid = mask_opt.as_ref().map_or(true, |m| unsafe { m.get_unchecked(i) });
        if valid {
            let s = unsafe { arr.get_str_unchecked(offset + i) };
            let char_len = s.chars().count();
            if char_len >= width {
                data.extend_from_slice(s.as_bytes());
                cur += s.len();
            } else {
                let padding = width - char_len;
                let pad_bytes = fill_char.len_utf8();
                let mut pad_buf = [0u8; 4];
                let pad_slice = fill_char.encode_utf8(&mut pad_buf);
                match side {
                    PadSide::Left => {
                        for _ in 0..padding { data.extend_from_slice(pad_slice.as_bytes()); }
                        data.extend_from_slice(s.as_bytes());
                    }
                    PadSide::Right => {
                        data.extend_from_slice(s.as_bytes());
                        for _ in 0..padding { data.extend_from_slice(pad_slice.as_bytes()); }
                    }
                    PadSide::Both => {
                        let left = padding / 2;
                        let right = padding - left;
                        for _ in 0..left { data.extend_from_slice(pad_slice.as_bytes()); }
                        data.extend_from_slice(s.as_bytes());
                        for _ in 0..right { data.extend_from_slice(pad_slice.as_bytes()); }
                    }
                }
                cur += s.len() + padding * pad_bytes;
            }
        }
        offsets[i + 1] = T::from_usize(cur);
    }

    Ok(StringArray {
        offsets: offsets.into(),
        data: data.into(),
        null_mask: mask_opt,
    })
}

/// Pads each categorical string element to `width` characters using `fill_char`.
pub fn pad_dict<T: Integer>(
    input: CategoricalAVT<T>,
    width: usize,
    fill_char: char,
    side: PadSide,
) -> Result<CategoricalArray<T>, KernelError> {
    let (arr, offset, len) = input;
    let mask_opt = arr.null_mask.as_ref().map(|orig| {
        let mut m = Bitmask::new_set_all(len, true);
        for i in 0..len {
            unsafe { m.set_unchecked(i, orig.get_unchecked(offset + i)); }
        }
        m
    });

    let pad_one = |s: &str| -> String {
        let char_len = s.chars().count();
        if char_len >= width {
            return s.to_owned();
        }
        let padding = width - char_len;
        match side {
            PadSide::Left => {
                let mut out = String::with_capacity(s.len() + padding * fill_char.len_utf8());
                for _ in 0..padding { out.push(fill_char); }
                out.push_str(s);
                out
            }
            PadSide::Right => {
                let mut out = String::with_capacity(s.len() + padding * fill_char.len_utf8());
                out.push_str(s);
                for _ in 0..padding { out.push(fill_char); }
                out
            }
            PadSide::Both => {
                let left = padding / 2;
                let right = padding - left;
                let mut out = String::with_capacity(s.len() + padding * fill_char.len_utf8());
                for _ in 0..left { out.push(fill_char); }
                out.push_str(s);
                for _ in 0..right { out.push(fill_char); }
                out
            }
        }
    };

    // Pad is 1:1 on dictionary values since padding a unique string always produces a unique result
    let mut new_unique = Vec64::<String>::new();
    for old_val in arr.unique_values().iter() {
        new_unique.push(pad_one(old_val));
    }

    let mut data = Vec64::<T>::with_capacity(len);
    unsafe { data.set_len(len); }
    for i in 0..len {
        let valid = mask_opt.as_ref().map_or(true, |m| unsafe { m.get_unchecked(i) });
        data[i] = if valid {
            arr.data[offset + i]
        } else {
            T::zero()
        };
    }

    Ok(CategoricalArray {
        data: data.into(),
        #[cfg(not(feature = "shared_dict"))]
        unique_values: new_unique,
        #[cfg(feature = "shared_dict")]
        dictionary: Dictionary::from(new_unique),
        null_mask: mask_opt,
    })
}

// Join (aggregation)

/// Joins all non-null string elements with `delimiter`, returning a single string.
/// Returns `None` if all elements are null.
pub fn join_str<T: Integer>(input: StringAVT<T>, delimiter: &str) -> Option<String> {
    let (arr, offset, len) = input;
    let mut parts: Vec<&str> = Vec::with_capacity(len);
    for i in offset..offset + len {
        let valid = arr.null_mask.as_ref().map_or(true, |b| unsafe { b.get_unchecked(i) });
        if valid {
            parts.push(unsafe { arr.get_str_unchecked(i) });
        }
    }
    if parts.is_empty() { None } else { Some(parts.join(delimiter)) }
}

/// Joins all non-null categorical string elements with `delimiter`, returning a single string.
/// Returns `None` if all elements are null.
pub fn join_dict<T: Integer>(input: CategoricalAVT<T>, delimiter: &str) -> Option<String> {
    let (arr, offset, len) = input;
    let mut parts: Vec<&str> = Vec::with_capacity(len);
    for i in offset..offset + len {
        let valid = arr.null_mask.as_ref().map_or(true, |b| unsafe { b.get_unchecked(i) });
        if valid {
            parts.push(unsafe { arr.get_str_unchecked(i) });
        }
    }
    if parts.is_empty() { None } else { Some(parts.join(delimiter)) }
}

// Regex replace

/// Replaces all regex matches in each string element with `replacement`.
#[cfg(feature = "regex")]
pub fn regex_replace_str<T: Integer>(
    input: StringAVT<T>,
    pattern: &str,
    replacement: &str,
) -> Result<StringArray<T>, KernelError> {
    let re = Regex::new(pattern).map_err(|_| {
        KernelError::InvalidArguments("Invalid regex pattern".to_string())
    })?;
    let (arr, offset, len) = input;
    let mask_opt = arr.null_mask.as_ref().map(|orig| {
        let mut m = Bitmask::new_set_all(len, true);
        for i in 0..len {
            unsafe { m.set_unchecked(i, orig.get_unchecked(offset + i)); }
        }
        m
    });

    let input_bytes = arr.offsets[offset + len].to_usize()
        - arr.offsets[offset].to_usize();
    let mut offsets = Vec64::<T>::with_capacity(len + 1);
    unsafe { offsets.set_len(len + 1); }
    let mut data = Vec64::<u8>::with_capacity(input_bytes);

    offsets[0] = T::zero();
    let mut cur = 0usize;

    for i in 0..len {
        let valid = mask_opt.as_ref().map_or(true, |m| unsafe { m.get_unchecked(i) });
        if valid {
            let s = unsafe { arr.get_str_unchecked(offset + i) };
            let t = re.replace_all(s, replacement).into_owned();
            data.extend_from_slice(t.as_bytes());
            cur += t.len();
        }
        offsets[i + 1] = T::from_usize(cur);
    }

    Ok(StringArray {
        offsets: offsets.into(),
        data: data.into(),
        null_mask: mask_opt,
    })
}

/// Replaces all regex matches in each categorical string element with `replacement`.
#[cfg(feature = "regex")]
pub fn regex_replace_dict<T: Integer>(
    input: CategoricalAVT<T>,
    pattern: &str,
    replacement: &str,
) -> Result<CategoricalArray<T>, KernelError> {
    let re = Regex::new(pattern).map_err(|_| {
        KernelError::InvalidArguments("Invalid regex pattern".to_string())
    })?;
    let (arr, offset, len) = input;
    let mask_opt = arr.null_mask.as_ref().map(|orig| {
        let mut m = Bitmask::new_set_all(len, true);
        for i in 0..len {
            unsafe { m.set_unchecked(i, orig.get_unchecked(offset + i)); }
        }
        m
    });

    #[cfg(feature = "fast_hash")]
    let mut seen: AHashMap<String, T> = AHashMap::with_capacity(arr.unique_values().len());
    #[cfg(not(feature = "fast_hash"))]
    let mut seen: std::collections::HashMap<String, T> =
        std::collections::HashMap::with_capacity(arr.unique_values().len());

    let mut new_unique = Vec64::<String>::new();
    let mut idx_map = Vec64::<T>::with_capacity(arr.unique_values().len());

    for old_val in arr.unique_values().iter() {
        let t = re.replace_all(old_val, replacement).into_owned();
        let new_idx = match seen.get(&t) {
            Some(&ix) => ix,
            None => {
                let ix = T::from_usize(new_unique.len());
                new_unique.push(t.clone());
                seen.insert(t, ix);
                ix
            }
        };
        idx_map.push(new_idx);
    }

    let mut data = Vec64::<T>::with_capacity(len);
    unsafe { data.set_len(len); }
    for i in 0..len {
        let valid = mask_opt.as_ref().map_or(true, |m| unsafe { m.get_unchecked(i) });
        data[i] = if valid {
            idx_map[arr.data[offset + i].to_usize()]
        } else {
            T::zero()
        };
    }

    Ok(CategoricalArray {
        data: data.into(),
        #[cfg(not(feature = "shared_dict"))]
        unique_values: new_unique,
        #[cfg(feature = "shared_dict")]
        dictionary: Dictionary::from(new_unique),
        null_mask: mask_opt,
    })
}

// Cross-tabulation

/// Cross-tabulate two text array views into a contingency table.
///
/// Iterates both views in lockstep, discovering categories on first appearance and
/// incrementing the corresponding cell for each jointly non-null pair. Null positions
/// in either array cause the row to be skipped.
///
/// Returns the count matrix, row labels from `x`, column labels from `y`,
/// and total number of valid pairs counted.
#[cfg(feature = "views")]
pub fn cross_tabulate(
    x: &TextArrayV,
    y: &TextArrayV,
) -> Result<(Vec64<Vec64<u64>>, Vec64<String>, Vec64<String>, usize), KernelError> {
    let n = x.len().min(y.len());

    if x.len() != y.len() {
        return Err(KernelError::LengthMismatch(format!(
            "cross_tabulate: x has length {} but y has length {}",
            x.len(),
            y.len()
        )));
    }

    #[cfg(feature = "fast_hash")]
    let mut row_map: AHashMap<String, usize> = AHashMap::with_capacity(n.min(256));
    #[cfg(not(feature = "fast_hash"))]
    let mut row_map: HashMap<String, usize> = HashMap::with_capacity(n.min(256));

    #[cfg(feature = "fast_hash")]
    let mut col_map: AHashMap<String, usize> = AHashMap::with_capacity(n.min(256));
    #[cfg(not(feature = "fast_hash"))]
    let mut col_map: HashMap<String, usize> = HashMap::with_capacity(n.min(256));

    let mut row_labels = Vec64::<String>::with_capacity(32);
    let mut col_labels = Vec64::<String>::with_capacity(32);

    // Single pass: discover categories and count co-occurrences.
    // We build a flat count vec keyed by (ri * max_cols + ci) and resize as
    // new categories are discovered. For typical cardinalities this stays small.
    let mut counts = Vec64::<u64>::new();
    let mut n_cols_seen = 0_usize;
    let mut total = 0_usize;

    for i in 0..n {
        let (Some(xv), Some(yv)) = (x.get_str(i), y.get_str(i)) else {
            continue;
        };

        let row_len = row_labels.len();
        let ri = *row_map.entry(xv.to_string()).or_insert_with(|| {
            row_labels.push(xv.to_string());
            row_len
        });

        let col_len = col_labels.len();
        let ci = *col_map.entry(yv.to_string()).or_insert_with(|| {
            col_labels.push(yv.to_string());
            col_len
        });

        // If new columns were added, widen every existing row in the flat buffer
        if col_labels.len() > n_cols_seen {
            let old_cols = n_cols_seen;
            n_cols_seen = col_labels.len();
            let nrows = row_labels.len();
            // Rebuild flat buffer with wider stride
            let mut new_counts = Vec64::<u64>::with_capacity(nrows * n_cols_seen);
            new_counts.resize(nrows * n_cols_seen, 0);
            for r in 0..nrows.min(ri + 1) {
                for c in 0..old_cols {
                    if r * old_cols + c < counts.len() {
                        new_counts[r * n_cols_seen + c] = counts[r * old_cols + c];
                    }
                }
            }
            counts = new_counts;
        }

        // New row may need the buffer extended
        let needed = (ri + 1) * n_cols_seen;
        if needed > counts.len() {
            counts.resize(needed, 0);
        }

        counts[ri * n_cols_seen + ci] += 1;
        total += 1;
    }

    if total == 0 {
        return Err(KernelError::InvalidArguments(
            "cross_tabulate: no valid non-null pairs found".to_string(),
        ));
    }

    // Convert flat counts to Vec64<Vec64<u64>> matrix
    let nrows = row_labels.len();
    let ncols = col_labels.len();
    let mut matrix = Vec64::<Vec64<u64>>::with_capacity(nrows);
    for r in 0..nrows {
        let start = r * ncols;
        let end = start + ncols;
        let row = Vec64::from_slice(&counts[start..end]);
        matrix.push(row);
    }

    Ok((matrix, row_labels, col_labels, total))
}

#[cfg(test)]
mod tests {
    use crate::{CategoricalArray, StringArray, vec64};

    use super::*;

    // --- Helper constructors

    fn str_array<T: Integer>(vals: &[&str]) -> StringArray<T> {
        StringArray::<T>::from_slice(vals)
    }

    fn dict_array<T: Integer>(vals: &[&str]) -> CategoricalArray<T> {
        let owned: Vec<&str> = vals.to_vec();
        CategoricalArray::<T>::from_values(owned)
    }

    fn bm(bools: &[bool]) -> Bitmask {
        Bitmask::from_bools(bools)
    }

    // --- Concat

    #[test]
    fn test_concat_str_str() {
        let a = str_array::<u32>(&["foo", "bar", ""]);
        let b = str_array::<u32>(&["baz", "qux", "quux"]);
        let out = concat_str_str((&a, 0, a.len()), (&b, 0, b.len()));
        assert_eq!(out.get(0), Some("foobaz"));
        assert_eq!(out.get(1), Some("barqux"));
        assert_eq!(out.get(2), Some("quux"));
        assert!(out.null_mask.as_ref().unwrap().all_set());
    }

    #[test]
    fn test_concat_str_str_chunk() {
        let a = str_array::<u32>(&["XXX", "foo", "bar", ""]);
        let b = str_array::<u32>(&["YYY", "baz", "qux", "quux"]);
        // Window is [1..4) for both, i.e., ["foo", "bar", ""]
        let out = concat_str_str((&a, 1, 3), (&b, 1, 3));
        assert_eq!(out.get(0), Some("foobaz"));
        assert_eq!(out.get(1), Some("barqux"));
        assert_eq!(out.get(2), Some("quux"));
        assert!(out.null_mask.as_ref().unwrap().all_set());
    }

    #[test]
    fn test_concat_dict_dict() {
        let a = dict_array::<u32>(&["x", "y"]);
        let b = dict_array::<u32>(&["1", "2"]);
        let out = concat_dict_dict((&a, 0, a.len()), (&b, 0, b.len())).unwrap();
        let s0 = out.get(0).unwrap();
        let s1 = out.get(1).unwrap();
        assert!(["x1", "y2"].contains(&s0));
        assert!(["x1", "y2"].contains(&s1));
        assert!(out.null_mask.as_ref().unwrap().all_set());
    }

    #[test]
    fn test_concat_dict_dict_chunk() {
        let a = dict_array::<u32>(&["foo", "x", "y", "bar"]);
        let b = dict_array::<u32>(&["A", "1", "2", "B"]);
        let out = concat_dict_dict((&a, 1, 2), (&b, 1, 2)).unwrap();
        let s0 = out.get(0).unwrap();
        let s1 = out.get(1).unwrap();
        assert!(["x1", "y2"].contains(&s0));
        assert!(["x1", "y2"].contains(&s1));
        assert!(out.null_mask.as_ref().unwrap().all_set());
    }

    #[test]
    fn test_concat_str_dict() {
        let a = str_array::<u32>(&["ab", "cd", ""]);
        let b = dict_array::<u32>(&["xy", "zq", ""]);
        let out = concat_str_dict((&a, 0, a.len()), (&b, 0, b.len())).unwrap();
        assert_eq!(out.get(0), Some("abxy"));
        assert_eq!(out.get(1), Some("cdzq"));
        assert_eq!(out.get(2), Some(""));
        assert!(out.null_mask.as_ref().unwrap().all_set());
    }

    #[test]
    fn test_concat_str_dict_chunk() {
        let a = str_array::<u32>(&["dummy", "ab", "cd", ""]);
        let b = dict_array::<u32>(&["dummy", "xy", "zq", ""]);
        let out = concat_str_dict((&a, 1, 3), (&b, 1, 3)).unwrap();
        assert_eq!(out.get(0), Some("abxy"));
        assert_eq!(out.get(1), Some("cdzq"));
        assert_eq!(out.get(2), Some(""));
        assert!(out.null_mask.as_ref().unwrap().all_set());
    }

    #[test]
    fn test_concat_dict_str() {
        let a = dict_array::<u32>(&["hi", "ho"]);
        let b = str_array::<u32>(&["yo", "no"]);
        let out = concat_dict_str((&a, 0, a.len()), (&b, 0, b.len())).unwrap();
        assert_eq!(out.get(0), Some("yohi"));
        assert_eq!(out.get(1), Some("noho"));
        assert!(out.null_mask.as_ref().unwrap().all_set());
    }

    #[test]
    fn test_concat_dict_str_chunk() {
        let a = dict_array::<u32>(&["dummy", "hi", "ho", "zzz"]);
        let b = str_array::<u32>(&["dummy", "yo", "no", "xxx"]);
        let out = concat_dict_str((&a, 1, 2), (&b, 1, 2)).unwrap();
        assert_eq!(out.get(0), Some("yohi"));
        assert_eq!(out.get(1), Some("noho"));
        assert!(out.null_mask.as_ref().unwrap().all_set());
    }

    // --- String predicates

    #[test]
    fn test_contains_str_str() {
        let s = str_array::<u32>(&["abc", "def", "ghijk"]);
        let p = str_array::<u32>(&["b", "x", "jk"]);
        let out = contains_str_str((&s, 0, s.len()), (&p, 0, p.len()));
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(false));
        assert_eq!(out.get(2), Some(true));
    }

    #[test]
    fn test_contains_str_str_chunk() {
        let s = str_array::<u32>(&["dummy", "abc", "def", "ghijk"]);
        let p = str_array::<u32>(&["dummy", "b", "x", "jk"]);
        let out = contains_str_str((&s, 1, 3), (&p, 1, 3));
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(false));
        assert_eq!(out.get(2), Some(true));
    }

    #[test]
    fn test_starts_with_str_str() {
        let s = str_array::<u32>(&["apricot", "banana", "apple"]);
        let p = str_array::<u32>(&["ap", "ba", "a"]);
        let out = starts_with_str_str((&s, 0, s.len()), (&p, 0, p.len()));
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
        assert_eq!(out.get(2), Some(true));
    }

    #[test]
    fn test_starts_with_str_str_chunk() {
        let s = str_array::<u32>(&["dummy", "apricot", "banana", "apple"]);
        let p = str_array::<u32>(&["dummy", "ap", "ba", "a"]);
        let out = starts_with_str_str((&s, 1, 3), (&p, 1, 3));
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
        assert_eq!(out.get(2), Some(true));
    }

    #[test]
    fn test_ends_with_str_str() {
        let s = str_array::<u32>(&["robot", "fast", "last"]);
        let p = str_array::<u32>(&["ot", "st", "ast"]);
        let out = ends_with_str_str((&s, 0, s.len()), (&p, 0, p.len()));
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
        assert_eq!(out.get(2), Some(true));
    }

    #[test]
    fn test_ends_with_str_str_chunk() {
        let s = str_array::<u32>(&["dummy", "robot", "fast", "last"]);
        let p = str_array::<u32>(&["dummy", "ot", "st", "ast"]);
        let out = ends_with_str_str((&s, 1, 3), (&p, 1, 3));
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
        assert_eq!(out.get(2), Some(true));
    }

    #[test]
    fn test_contains_str_dict() {
        let s = str_array::<u32>(&["abcde", "xyz", "qrstuv"]);
        let p = dict_array::<u32>(&["c", "z", "tu"]);
        let out = contains_str_dict((&s, 0, s.len()), (&p, 0, p.len())).unwrap();
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
        assert_eq!(out.get(2), Some(true));
    }

    #[test]
    fn test_contains_str_dict_chunk() {
        let s = str_array::<u32>(&["dummy", "abcde", "xyz", "qrstuv"]);
        let p = dict_array::<u32>(&["dummy", "c", "z", "tu"]);
        let out = contains_str_dict((&s, 1, 3), (&p, 1, 3)).unwrap();
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
        assert_eq!(out.get(2), Some(true));
    }

    #[test]
    fn test_contains_dict_dict() {
        let s = dict_array::<u32>(&["cdef", "foo", "bar"]);
        let p = dict_array::<u32>(&["cd", "oo", "baz"]);
        let out = contains_dict_dict((&s, 0, s.len()), (&p, 0, p.len())).unwrap();
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
        assert_eq!(out.get(2), Some(false));
    }

    #[test]
    fn test_contains_dict_dict_chunk() {
        let s = dict_array::<u32>(&["dummy", "cdef", "foo", "bar"]);
        let p = dict_array::<u32>(&["dummy", "cd", "oo", "baz"]);
        let out = contains_dict_dict((&s, 1, 3), (&p, 1, 3)).unwrap();
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
        assert_eq!(out.get(2), Some(false));
    }

    #[test]
    fn test_contains_dict_str() {
        let s = dict_array::<u32>(&["hello", "world"]);
        let p = str_array::<u32>(&["he", "o"]);
        let out = contains_dict_str((&s, 0, s.len()), (&p, 0, p.len())).unwrap();
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
    }

    #[test]
    fn test_contains_dict_str_chunk() {
        let s = dict_array::<u32>(&["dummy", "hello", "world"]);
        let p = str_array::<u32>(&["dummy", "he", "o"]);
        let out = contains_dict_str((&s, 1, 2), (&p, 1, 2)).unwrap();
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
    }

    #[test]
    fn test_starts_with_str_dict() {
        let s = str_array::<u32>(&["abcdef", "foobar", "quux"]);
        let p = dict_array::<u32>(&["ab", "foo", "qu"]);
        let out = starts_with_str_dict((&s, 0, s.len()), (&p, 0, p.len())).unwrap();
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
        assert_eq!(out.get(2), Some(true));
    }

    #[test]
    fn test_starts_with_str_dict_chunk() {
        let s = str_array::<u32>(&["dummy", "abcdef", "foobar", "quux"]);
        let p = dict_array::<u32>(&["dummy", "ab", "foo", "qu"]);
        let out = starts_with_str_dict((&s, 1, 3), (&p, 1, 3)).unwrap();
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
        assert_eq!(out.get(2), Some(true));
    }

    #[test]
    fn test_starts_with_dict_dict() {
        let s = dict_array::<u32>(&["qwerty", "banana"]);
        let p = dict_array::<u32>(&["qw", "ban"]);
        let out = starts_with_dict_dict((&s, 0, s.len()), (&p, 0, p.len())).unwrap();
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
    }

    #[test]
    fn test_starts_with_dict_dict_chunk() {
        let s = dict_array::<u32>(&["dummy", "qwerty", "banana"]);
        let p = dict_array::<u32>(&["dummy", "qw", "ban"]);
        let out = starts_with_dict_dict((&s, 1, 2), (&p, 1, 2)).unwrap();
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
    }

    #[test]
    fn test_ends_with_str_dict() {
        let s = str_array::<u32>(&["poem", "dome", "gnome"]);
        let p = dict_array::<u32>(&["em", "me", "ome"]);
        let out = ends_with_str_dict((&s, 0, s.len()), (&p, 0, p.len())).unwrap();
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
        assert_eq!(out.get(2), Some(true));
    }

    #[test]
    fn test_ends_with_str_dict_chunk() {
        let s = str_array::<u32>(&["dummy", "poem", "dome", "gnome"]);
        let p = dict_array::<u32>(&["dummy", "em", "me", "ome"]);
        let out = ends_with_str_dict((&s, 1, 3), (&p, 1, 3)).unwrap();
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
        assert_eq!(out.get(2), Some(true));
    }

    #[test]
    fn test_ends_with_dict_dict() {
        let s = dict_array::<u32>(&["tablet", "let", "bet"]);
        let p = dict_array::<u32>(&["let", "et", "xyz"]);
        let out = ends_with_dict_dict((&s, 0, s.len()), (&p, 0, p.len())).unwrap();
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
        assert_eq!(out.get(2), Some(false));
    }

    #[test]
    fn test_ends_with_dict_dict_chunk() {
        let s = dict_array::<u32>(&["dummy", "tablet", "let", "bet"]);
        let p = dict_array::<u32>(&["dummy", "let", "et", "xyz"]);
        let out = ends_with_dict_dict((&s, 1, 3), (&p, 1, 3)).unwrap();
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
        assert_eq!(out.get(2), Some(false));
    }

    // --- len_str, len_dict

    #[test]
    fn test_len_str() {
        let arr = str_array::<u32>(&["", "a", "abc", "bar"]);
        let out = len_str((&arr, 0, arr.len())).unwrap();
        assert_eq!(&out.data[..], &[0, 1, 3, 3]);
    }

    #[test]
    fn test_len_str_chunk() {
        let arr = str_array::<u32>(&["zzz", "", "a", "abc", "bar"]);
        let out = len_str((&arr, 1, 4)).unwrap(); // ["", "a", "abc", "bar"]
        assert_eq!(&out.data[..], &[0, 1, 3, 3]);
    }

    #[test]
    fn test_len_dict() {
        let arr = dict_array::<u32>(&["", "one", "seven"]);
        let out = len_dict((&arr, 0, arr.len())).unwrap();
        assert_eq!(&out.data[..], &[0, 3, 5]);
    }

    #[test]
    fn test_len_dict_chunk() {
        let arr = dict_array::<u32>(&["q", "", "one", "seven"]);
        let out = len_dict((&arr, 1, 3)).unwrap(); // ["", "one", "seven"]
        assert_eq!(&out.data[..], &[0, 3, 5]);
    }

    #[test]
    fn test_contains_empty_pattern() {
        let s = str_array::<u32>(&["foo", "bar"]);
        let p = str_array::<u32>(&["", ""]);
        let out = contains_str_str((&s, 0, s.len()), (&p, 0, p.len()));
        // always false
        assert_eq!(out.get(0), Some(false));
        assert_eq!(out.get(1), Some(false));
        assert!(out.null_mask.as_ref().is_none());
    }

    #[test]
    fn test_contains_empty_pattern_chunk() {
        let s = str_array::<u32>(&["z", "foo", "bar"]);
        let p = str_array::<u32>(&["z", "", ""]);
        let out = contains_str_str((&s, 1, 2), (&p, 1, 2));
        assert_eq!(out.get(0), Some(false));
        assert_eq!(out.get(1), Some(false));
        assert!(out.null_mask.as_ref().is_none());
    }

    #[test]
    fn test_contains_str_str_nulls_on_pattern() {
        let mut s = str_array::<u32>(&["abc", "def"]);
        s.null_mask = Some(bm(&[true, true]));
        let mut p = str_array::<u32>(&["b", "e"]);
        p.null_mask = Some(bm(&[true, false])); // second pattern is null
        let out = contains_str_str((&s, 0, s.len()), (&p, 0, p.len()));
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(false));
    }

    #[test]
    fn test_contains_str_str_nulls_on_pattern_chunk() {
        let mut s = str_array::<u32>(&["X", "abc", "def"]);
        s.null_mask = Some(bm(&[true, true, true]));
        let mut p = str_array::<u32>(&["X", "b", "e"]);
        p.null_mask = Some(bm(&[true, true, false])); // last pattern is null
        let out = contains_str_str((&s, 1, 2), (&p, 1, 2));
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(false));
    }

    #[cfg(feature = "regex")]
    #[test]
    fn test_regex_invalid_pattern_returns_err() {
        let s = str_array::<u32>(&["abc"]);
        let p = str_array::<u32>(&["["]);
        let err = regex_str_str((&s, 0, s.len()), (&p, 0, p.len())).unwrap_err();
        match err {
            KernelError::InvalidArguments(_) => {}
            _ => panic!("expected InvalidArguments"),
        }
    }

    #[cfg(feature = "regex")]
    #[test]
    fn test_regex_invalid_pattern_returns_err_chunk() {
        let s = str_array::<u32>(&["foo", "abc"]);
        let p = str_array::<u32>(&["bar", "["]);
        let err = regex_str_str((&s, 1, 1), (&p, 1, 1)).unwrap_err();
        match err {
            KernelError::InvalidArguments(_) => {}
            _ => panic!("expected InvalidArguments"),
        }
    }

    #[cfg(feature = "regex")]
    #[test]
    fn test_regex_empty_pattern_always_false() {
        let s = str_array::<u32>(&["abc", "def"]);
        let p = str_array::<u32>(&["", ""]);
        let out = regex_str_str((&s, 0, s.len()), (&p, 0, p.len())).unwrap();
        assert_eq!(out.get(0), Some(false));
        assert_eq!(out.get(1), Some(false));
        assert!(out.null_mask.unwrap().all_set());
    }

    #[cfg(feature = "regex")]
    #[test]
    fn test_regex_empty_pattern_always_false_chunk() {
        let s = str_array::<u32>(&["z", "abc", "def"]);
        let p = str_array::<u32>(&["z", "", ""]);
        let out = regex_str_str((&s, 1, 2), (&p, 1, 2)).unwrap();
        assert_eq!(out.get(0), Some(false));
        assert_eq!(out.get(1), Some(false));
        assert!(out.null_mask.unwrap().all_set());
    }

    #[test]
    fn test_len_str_with_nulls() {
        let mut arr = str_array::<u32>(&["foo", "", "bar"]);
        arr.null_mask = Some(bm(&[true, false, true]));
        let len_arr = len_str((&arr, 0, arr.len())).unwrap();
        assert_eq!(len_arr.data.as_slice(), &[3, 0, 3]);
        assert_eq!(
            len_arr.null_mask.unwrap().as_slice(),
            bm(&[true, false, true]).as_slice()
        );
    }

    #[test]
    fn test_len_str_with_nulls_chunk() {
        let mut arr = str_array::<u32>(&["x", "foo", "", "bar"]);
        arr.null_mask = Some(bm(&[true, true, false, true]));
        let len_arr = len_str((&arr, 1, 3)).unwrap();
        assert_eq!(len_arr.data.as_slice(), &[3, 0, 3]);
        assert_eq!(
            len_arr.null_mask.unwrap().as_slice(),
            bm(&[true, false, true]).as_slice()
        );
    }

    #[test]
    fn test_len_dict_with_nulls() {
        let mut arr = dict_array::<u32>(&["x", "yy", "zzz"]);
        arr.null_mask = Some(bm(&[false, true, true]));
        let len_arr = len_dict((&arr, 0, arr.len())).unwrap();
        assert_eq!(len_arr.data.as_slice(), &[0, 2, 3]);
        assert_eq!(
            len_arr.null_mask.unwrap().as_slice(),
            bm(&[false, true, true]).as_slice()
        );
    }

    #[test]
    fn test_len_dict_with_nulls_chunk() {
        let mut arr = dict_array::<u32>(&["z", "x", "yy", "zzz"]);
        arr.null_mask = Some(bm(&[true, false, true, true]));
        let len_arr = len_dict((&arr, 1, 3)).unwrap();
        assert_eq!(len_arr.data.as_slice(), &[0, 2, 3]);
        assert_eq!(
            len_arr.null_mask.unwrap().as_slice(),
            bm(&[false, true, true]).as_slice()
        );
    }

    fn bitmask_from_vec(v: &[bool]) -> Bitmask {
        let mut bm = Bitmask::with_capacity(v.len());
        for (i, &b) in v.iter().enumerate() {
            bm.set(i, b);
        }
        bm
    }

    #[test]
    fn test_min_string_array_all_valid() {
        let arr = StringArray::<u32>::from_slice(&["zulu", "alpha", "echo", "bravo"]);
        let view = (&arr, 0, arr.len());
        let result = min_string_array::<u32>(view);
        assert_eq!(result, Some("alpha".to_string()));
    }

    #[test]
    fn test_min_string_array_with_nulls() {
        let mut arr = StringArray::<u32>::from_slice(&["zulu", "alpha", "echo", "bravo"]);
        arr.null_mask = Some(bitmask_from_vec(&[false, true, true, true]));
        let view = (&arr, 0, arr.len());
        let result = min_string_array::<u32>(view);
        assert_eq!(result, Some("alpha".to_string()));
    }

    #[test]
    fn test_min_string_array_all_null() {
        let arr = StringArray::<u32>::from_slice(&["zulu", "alpha", "echo", "bravo"]);
        let mut null_mask = Bitmask::with_capacity(arr.len());
        for i in 0..arr.len() {
            null_mask.set(i, false);
        }
        let arr = StringArray::<u32> {
            null_mask: Some(null_mask),
            ..arr
        };
        let view = (&arr, 0, arr.len());
        let result = min_string_array::<u32>(view);
        assert_eq!(result, None);
    }

    #[test]
    fn test_max_string_array_all_valid() {
        let arr = StringArray::<u32>::from_slice(&["zulu", "alpha", "echo", "bravo"]);
        let view = (&arr, 0, arr.len());
        let result = max_string_array::<u32>(view);
        assert_eq!(result, Some("zulu".to_string()));
    }

    #[test]
    fn test_max_string_array_with_nulls() {
        let mut arr = StringArray::<u32>::from_slice(&["zulu", "alpha", "echo", "bravo"]);
        arr.null_mask = Some(bitmask_from_vec(&[true, false, true, false]));
        let view = (&arr, 0, arr.len());
        let result = max_string_array::<u32>(view);
        assert_eq!(result, Some("zulu".to_string()));
    }

    #[test]
    fn test_max_string_array_all_null() {
        let arr = StringArray::<u32>::from_slice(&["zulu", "alpha", "echo", "bravo"]);
        let mut null_mask = Bitmask::with_capacity(arr.len());
        for i in 0..arr.len() {
            null_mask.set(i, false);
        }
        let arr = StringArray::<u32> {
            null_mask: Some(null_mask),
            ..arr
        };
        let view = (&arr, 0, arr.len());
        let result = max_string_array::<u32>(view);
        assert_eq!(result, None);
    }

    #[test]
    fn test_min_categorical_array() {
        let uniques = vec64![
            "dog".to_string(),
            "zebra".to_string(),
            "ant".to_string(),
            "bee".to_string()
        ];
        let indices = vec64![1u32, 0, 3, 2]; // "zebra", "dog", "bee", "ant"
        let cat = CategoricalArray {
            data: indices.clone().into(),
            #[cfg(not(feature = "shared_dict"))]
            unique_values: Vec64::from(uniques.clone()),
            #[cfg(feature = "shared_dict")]
            dictionary: Dictionary::from(Vec64::from(uniques.clone())),
            null_mask: None,
        };
        let result = min_categorical_array((&cat, 0, indices.len()));
        assert_eq!(result, Some("ant".to_string()));
    }

    #[test]
    fn test_max_categorical_array() {
        let uniques = vec64![
            "dog".to_string(),
            "zebra".to_string(),
            "ant".to_string(),
            "bee".to_string()
        ];
        let indices = vec64![2u32, 0, 1, 3]; // "ant", "dog", "zebra", "bee"
        let cat = CategoricalArray {
            data: indices.clone().into(),
            #[cfg(not(feature = "shared_dict"))]
            unique_values: Vec64::from(uniques.clone()),
            #[cfg(feature = "shared_dict")]
            dictionary: Dictionary::from(Vec64::from(uniques.clone())),
            null_mask: None,
        };
        let result = max_categorical_array((&cat, 0, indices.len()));
        assert_eq!(result, Some("zebra".to_string()));
    }

    #[test]
    fn test_min_categorical_array_with_nulls() {
        let uniques = vec64!["dog".to_string(), "zebra".to_string(), "ant".to_string()];
        let indices = vec64![1u32, 2, 0];
        let mut null_mask = Bitmask::with_capacity(indices.len());
        null_mask.set(0, true);
        null_mask.set(1, false);
        null_mask.set(2, true);
        let cat = CategoricalArray {
            data: indices.clone().into(),
            #[cfg(not(feature = "shared_dict"))]
            unique_values: Vec64::from(uniques.clone()),
            #[cfg(feature = "shared_dict")]
            dictionary: Dictionary::from(Vec64::from(uniques.clone())),
            null_mask: Some(null_mask),
        };
        let result = min_categorical_array((&cat, 0, indices.len()));
        assert_eq!(result, Some("dog".to_string())); // Only positions 0 and 2 valid: "zebra", "dog" -> "dog" is smaller
    }

    #[test]
    fn test_max_categorical_array_with_nulls() {
        let uniques = vec64!["dog".to_string(), "zebra".to_string(), "ant".to_string()];
        let indices = vec64![1u32, 2, 0];
        let mut null_mask = Bitmask::with_capacity(indices.len());
        null_mask.set(0, true);
        null_mask.set(1, false);
        null_mask.set(2, true);
        let cat = CategoricalArray {
            data: indices.clone().into(),
            #[cfg(not(feature = "shared_dict"))]
            unique_values: Vec64::from(uniques.clone()),
            #[cfg(feature = "shared_dict")]
            dictionary: Dictionary::from(Vec64::from(uniques.clone())),
            null_mask: Some(null_mask),
        };
        let result = max_categorical_array((&cat, 0, indices.len()));
        assert_eq!(result, Some("zebra".to_string())); // Only positions 0 and 2 valid: "zebra", "dog" -> "zebra" is larger
    }

    // --- Unary transforms

    #[test]
    fn test_to_uppercase_str() {
        let a = str_array::<u32>(&["hello", "World", "FOO"]);
        let out = to_uppercase_str((&a, 0, a.len())).unwrap();
        assert_eq!(out.get(0), Some("HELLO"));
        assert_eq!(out.get(1), Some("WORLD"));
        assert_eq!(out.get(2), Some("FOO"));
    }

    #[test]
    fn test_to_uppercase_str_chunk() {
        let a = str_array::<u32>(&["skip", "hello", "world", "skip"]);
        let out = to_uppercase_str((&a, 1, 2)).unwrap();
        assert_eq!(out.get(0), Some("HELLO"));
        assert_eq!(out.get(1), Some("WORLD"));
    }

    #[test]
    fn test_to_uppercase_dict() {
        let a = dict_array::<u32>(&["hello", "world"]);
        let out = to_uppercase_dict((&a, 0, a.data.len())).unwrap();
        assert_eq!(out.get_str(0), Some("HELLO"));
        assert_eq!(out.get_str(1), Some("WORLD"));
    }

    #[test]
    fn test_to_lowercase_str() {
        let a = str_array::<u32>(&["HELLO", "World"]);
        let out = to_lowercase_str((&a, 0, a.len())).unwrap();
        assert_eq!(out.get(0), Some("hello"));
        assert_eq!(out.get(1), Some("world"));
    }

    #[test]
    fn test_trim_str() {
        let a = str_array::<u32>(&["  hello  ", "world ", " foo"]);
        let out = trim_str((&a, 0, a.len())).unwrap();
        assert_eq!(out.get(0), Some("hello"));
        assert_eq!(out.get(1), Some("world"));
        assert_eq!(out.get(2), Some("foo"));
    }

    #[test]
    fn test_ltrim_str() {
        let a = str_array::<u32>(&["  hello", "  world "]);
        let out = ltrim_str((&a, 0, a.len())).unwrap();
        assert_eq!(out.get(0), Some("hello"));
        assert_eq!(out.get(1), Some("world "));
    }

    #[test]
    fn test_rtrim_str() {
        let a = str_array::<u32>(&["hello  ", " world "]);
        let out = rtrim_str((&a, 0, a.len())).unwrap();
        assert_eq!(out.get(0), Some("hello"));
        assert_eq!(out.get(1), Some(" world"));
    }

    #[test]
    fn test_reverse_str() {
        let a = str_array::<u32>(&["abc", "hello"]);
        let out = reverse_str((&a, 0, a.len())).unwrap();
        assert_eq!(out.get(0), Some("cba"));
        assert_eq!(out.get(1), Some("olleh"));
    }

    #[test]
    fn test_reverse_dict() {
        let a = dict_array::<u32>(&["abc", "xy"]);
        let out = reverse_dict((&a, 0, a.data.len())).unwrap();
        assert_eq!(out.get_str(0), Some("cba"));
        assert_eq!(out.get_str(1), Some("yx"));
    }

    // --- Byte length

    #[test]
    fn test_byte_length_str() {
        let a = str_array::<u32>(&["hi", "café"]);
        let out = byte_length_str((&a, 0, a.len())).unwrap();
        assert_eq!(out.data[0], 2u32); // "hi" = 2 bytes
        assert_eq!(out.data[1], 5u32); // "café" = 5 bytes (é is 2 bytes in UTF-8)
    }

    #[test]
    fn test_byte_length_dict() {
        let a = dict_array::<u32>(&["hi", "café"]);
        let out = byte_length_dict((&a, 0, a.data.len())).unwrap();
        assert_eq!(out.data[0], 2u32);
        assert_eq!(out.data[1], 5u32);
    }

    // --- Find and count

    #[test]
    fn test_find_str() {
        let a = str_array::<u32>(&["hello world", "foo bar", "baz"]);
        let out = find_str((&a, 0, a.len()), "world").unwrap();
        assert_eq!(out.data[0], 6);
        assert_eq!(out.data[1], -1);
        assert_eq!(out.data[2], -1);
    }

    #[test]
    fn test_find_dict() {
        let a = dict_array::<u32>(&["hello", "world"]);
        let out = find_dict((&a, 0, a.data.len()), "llo").unwrap();
        assert_eq!(out.data[0], 2);
        assert_eq!(out.data[1], -1);
    }

    #[test]
    fn test_count_match_str() {
        let a = str_array::<u32>(&["abcabc", "abc", "xyz"]);
        let out = count_match_str((&a, 0, a.len()), "abc").unwrap();
        assert_eq!(out.data[0], 2);
        assert_eq!(out.data[1], 1);
        assert_eq!(out.data[2], 0);
    }

    #[test]
    fn test_count_match_dict() {
        let a = dict_array::<u32>(&["aaa", "a"]);
        let out = count_match_dict((&a, 0, a.data.len()), "a").unwrap();
        assert_eq!(out.data[0], 3);
        assert_eq!(out.data[1], 1);
    }

    // --- Substring

    #[test]
    fn test_substring_str() {
        let a = str_array::<u32>(&["hello world", "foo"]);
        let out = substring_str((&a, 0, a.len()), 6, None).unwrap();
        assert_eq!(out.get(0), Some("world"));
        assert_eq!(out.get(1), Some(""));
    }

    #[test]
    fn test_substring_str_with_len() {
        let a = str_array::<u32>(&["hello world"]);
        let out = substring_str((&a, 0, a.len()), 0, Some(5)).unwrap();
        assert_eq!(out.get(0), Some("hello"));
    }

    #[test]
    fn test_substring_dict() {
        let a = dict_array::<u32>(&["hello", "world"]);
        let out = substring_dict((&a, 0, a.data.len()), 1, Some(3)).unwrap();
        assert_eq!(out.get_str(0), Some("ell"));
        assert_eq!(out.get_str(1), Some("orl"));
    }

    // --- Replace

    #[test]
    fn test_replace_str() {
        let a = str_array::<u32>(&["hello world", "foo bar foo"]);
        let out = replace_str((&a, 0, a.len()), "foo", "baz").unwrap();
        assert_eq!(out.get(0), Some("hello world"));
        assert_eq!(out.get(1), Some("baz bar baz"));
    }

    #[test]
    fn test_replace_dict() {
        let a = dict_array::<u32>(&["aXb", "cXd"]);
        let out = replace_dict((&a, 0, a.data.len()), "X", "Y").unwrap();
        assert_eq!(out.get_str(0), Some("aYb"));
        assert_eq!(out.get_str(1), Some("cYd"));
    }

    // --- Repeat

    #[test]
    fn test_repeat_str() {
        let a = str_array::<u32>(&["ab", "x"]);
        let out = repeat_str((&a, 0, a.len()), 3).unwrap();
        assert_eq!(out.get(0), Some("ababab"));
        assert_eq!(out.get(1), Some("xxx"));
    }

    #[test]
    fn test_repeat_dict() {
        let a = dict_array::<u32>(&["ha"]);
        let out = repeat_dict((&a, 0, a.data.len()), 2).unwrap();
        assert_eq!(out.get_str(0), Some("haha"));
    }

    // --- Pad

    #[test]
    fn test_pad_str_left() {
        let a = str_array::<u32>(&["hi", "hello"]);
        let out = pad_str((&a, 0, a.len()), 5, ' ', PadSide::Left).unwrap();
        assert_eq!(out.get(0), Some("   hi"));
        assert_eq!(out.get(1), Some("hello")); // already 5 chars
    }

    #[test]
    fn test_pad_str_right() {
        let a = str_array::<u32>(&["hi"]);
        let out = pad_str((&a, 0, a.len()), 5, '-', PadSide::Right).unwrap();
        assert_eq!(out.get(0), Some("hi---"));
    }

    #[test]
    fn test_pad_str_both() {
        let a = str_array::<u32>(&["hi"]);
        let out = pad_str((&a, 0, a.len()), 6, '*', PadSide::Both).unwrap();
        assert_eq!(out.get(0), Some("**hi**"));
    }

    #[test]
    fn test_pad_dict() {
        let a = dict_array::<u32>(&["x"]);
        let out = pad_dict((&a, 0, a.data.len()), 3, '0', PadSide::Left).unwrap();
        assert_eq!(out.get_str(0), Some("00x"));
    }

    // --- Join

    #[test]
    fn test_join_str() {
        let a = str_array::<u32>(&["a", "b", "c"]);
        assert_eq!(join_str((&a, 0, a.len()), ", "), Some("a, b, c".to_owned()));
    }

    #[test]
    fn test_join_str_chunk() {
        let a = str_array::<u32>(&["skip", "a", "b", "skip"]);
        assert_eq!(join_str((&a, 1, 2), "-"), Some("a-b".to_owned()));
    }

    #[test]
    fn test_join_dict() {
        let a = dict_array::<u32>(&["x", "y"]);
        assert_eq!(join_dict((&a, 0, a.data.len()), "+"), Some("x+y".to_owned()));
    }

    // --- Regex replace (feature-gated)

    #[cfg(feature = "regex")]
    #[test]
    fn test_regex_replace_str() {
        let a = str_array::<u32>(&["abc123def456", "no digits"]);
        let out = regex_replace_str((&a, 0, a.len()), r"\d+", "N").unwrap();
        assert_eq!(out.get(0), Some("abcNdefN"));
        assert_eq!(out.get(1), Some("no digits"));
    }

    #[cfg(feature = "regex")]
    #[test]
    fn test_regex_replace_dict() {
        let a = dict_array::<u32>(&["foo123"]);
        let out = regex_replace_dict((&a, 0, a.data.len()), r"\d+", "").unwrap();
        assert_eq!(out.get_str(0), Some("foo"));
    }

    // --- Null handling tests

    #[test]
    fn test_to_uppercase_str_with_nulls() {
        let mut a = str_array::<u32>(&["hello", "skip", "world"]);
        a.null_mask = Some(bm(&[true, false, true]));
        let out = to_uppercase_str((&a, 0, a.len())).unwrap();
        assert_eq!(out.get(0), Some("HELLO"));
        assert!(!out.null_mask.as_ref().unwrap().get(1));
        assert_eq!(out.get(2), Some("WORLD"));
    }

    #[test]
    fn test_find_str_with_nulls() {
        let mut a = str_array::<u32>(&["hello", "skip"]);
        a.null_mask = Some(bm(&[true, false]));
        let out = find_str((&a, 0, a.len()), "ello").unwrap();
        assert_eq!(out.data[0], 1);
        assert_eq!(out.data[1], 0);
        assert!(!out.null_mask.as_ref().unwrap().get(1));
    }

    #[test]
    fn test_join_str_with_nulls() {
        let mut a = str_array::<u32>(&["a", "SKIP", "b"]);
        a.null_mask = Some(bm(&[true, false, true]));
        assert_eq!(join_str((&a, 0, a.len()), ","), Some("a,b".to_owned()));
    }
}
