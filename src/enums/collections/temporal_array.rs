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

//! # **TemporalArray Module** - *High-Level DateTimes Array Type for Unified Signature Dispatch*
//!
//! TemporalArray unifies all datetime-based arrays into a single enum for
//! standardised temporal operations.
//!   
//! ## Features:
//! - direct variant access
//! - zero-cost casts when the type is known
//! - lossless conversions between 32-bit and 64-bit datetime types.  
//! - simplifies function signatures by accepting `impl Into<TemporalArray>`
//! - centralises dispatch
//! - preserves SIMD-aligned buffers across all temporal variants.

use std::{
    fmt::{Display, Formatter},
    sync::Arc,
};

use crate::{Bitmask, DatetimeArray, MaskedArray};
use crate::{
    enums::{error::MinarrowError, shape_dim::ShapeDim},
    traits::{concatenate::Concatenate, shape::Shape},
};

/// Temporal Array
///
/// Unified datetime array container
///
/// ## Purpose
/// Exists to unify datetime operations,
/// simplify API's and streamline user ergonomics.
///
/// ## Usage:
/// - It is accessible from `Array` using `.dt()`,
/// and provides typed variant access via for e.g.,
/// `.dt32()`, so one can drill down to the required
/// granularity via `myarr.dt().dt32()`
/// - This streamlines function implementations *(at least for the `NumericArray`
/// case where this pattern is the most useful)*,
/// and, despite the additional `enum` layer,
/// matching lanes in many real-world scenarios.
/// This is because one can for e.g., unify a
/// function signature with `impl Into<TemporalArray>`,
/// and all of the subtypes, plus `Array` and `TemporalArray`,
/// all qualify.
/// - Additionally, you can then use one `Temporal` implementation
/// on the enum dispatch arm for all `Temporal` variants, or,
/// in many cases, for the entire datetime arm when they are the same.
///
/// ### Handling Times, Durations, etc.
/// We use one Physical type to hold all datetime variants,
/// i.e., the *Apache Arrow* types `DATE32`, `TIME32`, `DURATION` etc.,
/// and the Logical type is stored on the `Field` as metadata, given they
/// otherwise have the same underlying data representation. To treat
/// them differently in API usage, you can use the `TimeUnit` and `IntervalUnit`,
/// along with the `ArrowType` that is stored on the `Field` in `Minarrow`,
/// and match on these for any desired behaviour. The `Field` is packaged together
/// with `Array` *(which then drill-down accesses `TemporalArray` on the fly, or
/// in dispatch routing scenarios)*.
///
/// ### Typecasting behaviour
/// - If the enum already holds the given type *(which should be known at compile-time)*,
/// then using accessors like `.dt32()` is zero-cost, as it transfers ownership.
/// - If you want to keep the original, of course use `.clone()` beforehand.
/// - If you use an accessor to a different base type, e.g., `.dt64()` when it's a
/// `.dt32()` already in the enum, it will convert it. Therefore, be mindful
/// of performance when this occurs.
#[repr(C, align(64))]
#[derive(PartialEq, Clone, Debug, Default)]
pub enum TemporalArray {
    // The datetimes are chunked by their common memory layout rather than logical type
    // These can be casted to the relevant Arrow type at the FFI layer as needed
    Datetime32(Arc<DatetimeArray<i32>>), // DATE32, TIME32, DURATION(s), DURATION(ms) (32-bit)
    // DATE64, TIMESTAMP (ms/us/ns), DURATION (ms/us/ns), TIME64, DURATION(us), DURATION(ns)
    Datetime64(Arc<DatetimeArray<i64>>),
    #[default]
    Null, // Default Marker for mem::take
}

impl TemporalArray {
    /// Returns the logical length of the temporal array.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            TemporalArray::Datetime32(arr) => arr.len(),
            TemporalArray::Datetime64(arr) => arr.len(),
            TemporalArray::Null => 0,
        }
    }

    /// Returns the underlying null mask, if any.
    #[inline]
    pub fn null_mask(&self) -> Option<&Bitmask> {
        match self {
            TemporalArray::Datetime32(arr) => arr.null_mask.as_ref(),
            TemporalArray::Datetime64(arr) => arr.null_mask.as_ref(),
            TemporalArray::Null => None,
        }
    }

    /// Returns true when the variant holds at least one null.
    ///
    /// Delegates to each inner array's `MaskedArray::has_nulls`; `Null` is
    /// treated as empty (no elements means no nulls).
    #[inline]
    pub fn has_nulls(&self) -> bool {
        match self {
            TemporalArray::Datetime32(arr) => arr.has_nulls(),
            TemporalArray::Datetime64(arr) => arr.has_nulls(),
            TemporalArray::Null => false,
        }
    }

    /// Appends all values (and null mask if present) from `other` into `self`.
    ///
    /// Panics if the two arrays are of different variants or incompatible types.
    ///
    /// This function uses copy-on-write semantics for arrays wrapped in `Arc`.
    /// If `self` is the only owner of its data, appends are performed in place without copying.
    /// If the array data is shared (`Arc` reference count > 1), the data is first cloned
    /// (so the mutation does not affect other owners), and the append is then performed on the unique copy.
    ///
    /// This ensures that calling `append_array` never mutates data referenced elsewhere,
    /// but also avoids unnecessary cloning when the data is uniquely owned.
    pub fn append_array(&mut self, other: &Self) {
        match (self, other) {
            (TemporalArray::Datetime32(a), TemporalArray::Datetime32(b)) => {
                Arc::make_mut(a).append_array(b)
            }
            (TemporalArray::Datetime64(a), TemporalArray::Datetime64(b)) => {
                Arc::make_mut(a).append_array(b)
            }
            (TemporalArray::Null, TemporalArray::Null) => (),
            (lhs, rhs) => panic!("Cannot append {:?} into {:?}", rhs, lhs),
        }
    }

    pub fn append_range(&mut self, other: &Self, offset: usize, len: usize) -> Result<(), MinarrowError> {
        match (self, other) {
            (TemporalArray::Datetime32(a), TemporalArray::Datetime32(b)) => Arc::make_mut(a).append_range(b, offset, len),
            (TemporalArray::Datetime64(a), TemporalArray::Datetime64(b)) => Arc::make_mut(a).append_range(b, offset, len),
            (TemporalArray::Null, TemporalArray::Null) => Ok(()),
            (lhs, rhs) => Err(MinarrowError::TypeError {
                from: "TemporalArray",
                to: "TemporalArray",
                message: Some(format!("Cannot append_range {:?} into {:?}", rhs, lhs)),
            }),
        }
    }

    /// Inserts all values (and null mask if present) from `other` into `self` at the specified index.
    ///
    /// This is an **O(n)** operation.
    ///
    /// Returns an error if the two arrays are of different variants or incompatible types,
    /// or if the index is out of bounds.
    ///
    /// This function uses copy-on-write semantics for arrays wrapped in `Arc`.
    pub fn insert_rows(&mut self, index: usize, other: &Self) -> Result<(), MinarrowError> {
        match (self, other) {
            (TemporalArray::Datetime32(a), TemporalArray::Datetime32(b)) => {
                Arc::make_mut(a).insert_rows(index, b)
            }
            (TemporalArray::Datetime64(a), TemporalArray::Datetime64(b)) => {
                Arc::make_mut(a).insert_rows(index, b)
            }
            (TemporalArray::Null, TemporalArray::Null) => Ok(()),
            (lhs, rhs) => Err(MinarrowError::TypeError {
                from: "TemporalArray",
                to: "TemporalArray",
                message: Some(format!(
                    "Cannot insert {} into {}: incompatible types",
                    temporal_variant_name(rhs),
                    temporal_variant_name(lhs)
                )),
            }),
        }
    }

    /// Splits the TemporalArray at the specified index, consuming self and returning two arrays.
    pub fn split(self, index: usize) -> Result<(Self, Self), MinarrowError> {
        use std::sync::Arc;

        match self {
            TemporalArray::Datetime32(a) => {
                let (left, right) = Arc::try_unwrap(a)
                    .unwrap_or_else(|arc| (*arc).clone())
                    .split(index)?;
                Ok((
                    TemporalArray::Datetime32(Arc::new(left)),
                    TemporalArray::Datetime32(Arc::new(right)),
                ))
            }
            TemporalArray::Datetime64(a) => {
                let (left, right) = Arc::try_unwrap(a)
                    .unwrap_or_else(|arc| (*arc).clone())
                    .split(index)?;
                Ok((
                    TemporalArray::Datetime64(Arc::new(left)),
                    TemporalArray::Datetime64(Arc::new(right)),
                ))
            }
            TemporalArray::Null => Err(MinarrowError::IndexError(
                "Cannot split Null array".to_string(),
            )),
        }
    }

    /// Returns a reference to the inner `DatetimeArray<i32>` if the variant matches.
    /// No conversion or cloning is performed.
    pub fn dt32_ref(&self) -> Result<&DatetimeArray<i32>, MinarrowError> {
        match self {
            TemporalArray::Datetime32(arr) => Ok(arr),
            TemporalArray::Datetime64(_) => Err(MinarrowError::TypeError {
                from: "Datetime64",
                to: "DatetimeArray<i32>",
                message: None,
            }),
            TemporalArray::Null => Err(MinarrowError::NullError { message: None }),
        }
    }

    /// Returns a reference to the inner `DatetimeArray<i64>` if the variant matches.
    /// No conversion or cloning is performed.
    pub fn dt64_ref(&self) -> Result<&DatetimeArray<i64>, MinarrowError> {
        match self {
            TemporalArray::Datetime64(arr) => Ok(arr),
            TemporalArray::Datetime32(_) => Err(MinarrowError::TypeError {
                from: "Datetime32",
                to: "DatetimeArray<i64>",
                message: None,
            }),
            TemporalArray::Null => Err(MinarrowError::NullError { message: None }),
        }
    }

    /// Returns an Arc<DatetimeArray<i32>> (casting if needed).
    pub fn dt32(self) -> Result<DatetimeArray<i32>, MinarrowError> {
        match self {
            TemporalArray::Datetime32(arr) => match Arc::try_unwrap(arr) {
                Ok(inner) => Ok(inner),
                Err(shared) => Ok((*shared).clone()),
            },
            TemporalArray::Datetime64(arr) => Ok(DatetimeArray::<i32>::try_from(&*arr)?),
            TemporalArray::Null => Err(MinarrowError::NullError { message: None }),
        }
    }

    /// Returns an Arc<DatetimeArray<i64>> (casting if needed).
    pub fn dt64(self) -> Result<DatetimeArray<i64>, MinarrowError> {
        match self {
            TemporalArray::Datetime64(arr) => match Arc::try_unwrap(arr) {
                Ok(inner) => Ok(inner),
                Err(shared) => Ok((*shared).clone()),
            },
            TemporalArray::Datetime32(arr) => Ok(DatetimeArray::<i64>::from(&*arr)),
            TemporalArray::Null => Err(MinarrowError::NullError { message: None }),
        }
    }
}

impl Shape for TemporalArray {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank1(self.len())
    }
}

impl Concatenate for TemporalArray {
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        match (self, other) {
            (TemporalArray::Datetime32(a), TemporalArray::Datetime32(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(TemporalArray::Datetime32(Arc::new(a.concat(b)?)))
            }
            (TemporalArray::Datetime64(a), TemporalArray::Datetime64(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(TemporalArray::Datetime64(Arc::new(a.concat(b)?)))
            }
            (TemporalArray::Null, TemporalArray::Null) => Ok(TemporalArray::Null),
            (lhs, rhs) => Err(MinarrowError::IncompatibleTypeError {
                from: "TemporalArray",
                to: "TemporalArray",
                message: Some(format!(
                    "Cannot concatenate mismatched TemporalArray variants: {:?} and {:?}",
                    temporal_variant_name(&lhs),
                    temporal_variant_name(&rhs)
                )),
            }),
        }
    }
}

#[cfg(feature = "datetime_ops")]
use crate::DatetimeOps;

#[cfg(feature = "datetime_ops")]
use crate::enums::time_units::TimeUnit;

#[cfg(feature = "datetime_ops")]
use time::Duration;

#[cfg(feature = "datetime_ops")]
use crate::structs::variants::{boolean::BooleanArray, integer::IntegerArray};

#[cfg(feature = "datetime_ops")]
impl DatetimeOps for TemporalArray {
    // Component Extraction - delegate to inner variant, return directly

    fn year(&self) -> IntegerArray<i32> {
        match self {
            TemporalArray::Datetime32(arr) => arr.year(),
            TemporalArray::Datetime64(arr) => arr.year(),
            TemporalArray::Null => IntegerArray::default(),
        }
    }

    fn month(&self) -> IntegerArray<i32> {
        match self {
            TemporalArray::Datetime32(arr) => arr.month(),
            TemporalArray::Datetime64(arr) => arr.month(),
            TemporalArray::Null => IntegerArray::default(),
        }
    }

    fn day(&self) -> IntegerArray<i32> {
        match self {
            TemporalArray::Datetime32(arr) => arr.day(),
            TemporalArray::Datetime64(arr) => arr.day(),
            TemporalArray::Null => IntegerArray::default(),
        }
    }

    fn hour(&self) -> IntegerArray<i32> {
        match self {
            TemporalArray::Datetime32(arr) => arr.hour(),
            TemporalArray::Datetime64(arr) => arr.hour(),
            TemporalArray::Null => IntegerArray::default(),
        }
    }

    fn minute(&self) -> IntegerArray<i32> {
        match self {
            TemporalArray::Datetime32(arr) => arr.minute(),
            TemporalArray::Datetime64(arr) => arr.minute(),
            TemporalArray::Null => IntegerArray::default(),
        }
    }

    fn second(&self) -> IntegerArray<i32> {
        match self {
            TemporalArray::Datetime32(arr) => arr.second(),
            TemporalArray::Datetime64(arr) => arr.second(),
            TemporalArray::Null => IntegerArray::default(),
        }
    }

    fn weekday(&self) -> IntegerArray<i32> {
        match self {
            TemporalArray::Datetime32(arr) => arr.weekday(),
            TemporalArray::Datetime64(arr) => arr.weekday(),
            TemporalArray::Null => IntegerArray::default(),
        }
    }

    fn day_of_year(&self) -> IntegerArray<i32> {
        match self {
            TemporalArray::Datetime32(arr) => arr.day_of_year(),
            TemporalArray::Datetime64(arr) => arr.day_of_year(),
            TemporalArray::Null => IntegerArray::default(),
        }
    }

    fn iso_week(&self) -> IntegerArray<i32> {
        match self {
            TemporalArray::Datetime32(arr) => arr.iso_week(),
            TemporalArray::Datetime64(arr) => arr.iso_week(),
            TemporalArray::Null => IntegerArray::default(),
        }
    }

    fn quarter(&self) -> IntegerArray<i32> {
        match self {
            TemporalArray::Datetime32(arr) => arr.quarter(),
            TemporalArray::Datetime64(arr) => arr.quarter(),
            TemporalArray::Null => IntegerArray::default(),
        }
    }

    fn week_of_year(&self) -> IntegerArray<i32> {
        match self {
            TemporalArray::Datetime32(arr) => arr.week_of_year(),
            TemporalArray::Datetime64(arr) => arr.week_of_year(),
            TemporalArray::Null => IntegerArray::default(),
        }
    }

    fn is_leap_year(&self) -> BooleanArray<()> {
        match self {
            TemporalArray::Datetime32(arr) => arr.is_leap_year(),
            TemporalArray::Datetime64(arr) => arr.is_leap_year(),
            TemporalArray::Null => BooleanArray::default(),
        }
    }

    // Arithmetic - delegate, wrap result back into enum variant

    fn add_duration(&self, duration: Duration) -> Result<Self, MinarrowError> {
        match self {
            TemporalArray::Datetime32(arr) => Ok(TemporalArray::Datetime32(Arc::new(
                arr.add_duration(duration)?,
            ))),
            TemporalArray::Datetime64(arr) => Ok(TemporalArray::Datetime64(Arc::new(
                arr.add_duration(duration)?,
            ))),
            TemporalArray::Null => Err(MinarrowError::NullError { message: None }),
        }
    }

    fn sub_duration(&self, duration: Duration) -> Result<Self, MinarrowError> {
        match self {
            TemporalArray::Datetime32(arr) => Ok(TemporalArray::Datetime32(Arc::new(
                arr.sub_duration(duration)?,
            ))),
            TemporalArray::Datetime64(arr) => Ok(TemporalArray::Datetime64(Arc::new(
                arr.sub_duration(duration)?,
            ))),
            TemporalArray::Null => Err(MinarrowError::NullError { message: None }),
        }
    }

    fn add_days(&self, days: i64) -> Result<Self, MinarrowError> {
        match self {
            TemporalArray::Datetime32(arr) => {
                Ok(TemporalArray::Datetime32(Arc::new(arr.add_days(days)?)))
            }
            TemporalArray::Datetime64(arr) => {
                Ok(TemporalArray::Datetime64(Arc::new(arr.add_days(days)?)))
            }
            TemporalArray::Null => Err(MinarrowError::NullError { message: None }),
        }
    }

    fn add_months(&self, months: i32) -> Result<Self, MinarrowError> {
        match self {
            TemporalArray::Datetime32(arr) => {
                Ok(TemporalArray::Datetime32(Arc::new(arr.add_months(months)?)))
            }
            TemporalArray::Datetime64(arr) => {
                Ok(TemporalArray::Datetime64(Arc::new(arr.add_months(months)?)))
            }
            TemporalArray::Null => Err(MinarrowError::NullError { message: None }),
        }
    }

    fn add_years(&self, years: i32) -> Result<Self, MinarrowError> {
        match self {
            TemporalArray::Datetime32(arr) => {
                Ok(TemporalArray::Datetime32(Arc::new(arr.add_years(years)?)))
            }
            TemporalArray::Datetime64(arr) => {
                Ok(TemporalArray::Datetime64(Arc::new(arr.add_years(years)?)))
            }
            TemporalArray::Null => Err(MinarrowError::NullError { message: None }),
        }
    }

    // Comparison - match on (self, other) tuple, verify same variant

    fn diff(&self, other: &Self, unit: TimeUnit) -> Result<IntegerArray<i64>, MinarrowError> {
        match (self, other) {
            (TemporalArray::Datetime32(a), TemporalArray::Datetime32(b)) => a.diff(b, unit),
            (TemporalArray::Datetime64(a), TemporalArray::Datetime64(b)) => a.diff(b, unit),
            (TemporalArray::Null, _) | (_, TemporalArray::Null) => {
                Err(MinarrowError::NullError { message: None })
            }
            _ => Err(MinarrowError::TypeError {
                from: "TemporalArray",
                to: "TemporalArray",
                message: Some("Mismatched temporal variants".to_string()),
            }),
        }
    }

    fn abs_diff(&self, other: &Self, unit: TimeUnit) -> Result<IntegerArray<i64>, MinarrowError> {
        match (self, other) {
            (TemporalArray::Datetime32(a), TemporalArray::Datetime32(b)) => a.abs_diff(b, unit),
            (TemporalArray::Datetime64(a), TemporalArray::Datetime64(b)) => a.abs_diff(b, unit),
            (TemporalArray::Null, _) | (_, TemporalArray::Null) => {
                Err(MinarrowError::NullError { message: None })
            }
            _ => Err(MinarrowError::TypeError {
                from: "TemporalArray",
                to: "TemporalArray",
                message: Some("Mismatched temporal variants".to_string()),
            }),
        }
    }

    fn is_before(&self, other: &Self) -> Result<BooleanArray<()>, MinarrowError> {
        match (self, other) {
            (TemporalArray::Datetime32(a), TemporalArray::Datetime32(b)) => a.is_before(b),
            (TemporalArray::Datetime64(a), TemporalArray::Datetime64(b)) => a.is_before(b),
            (TemporalArray::Null, _) | (_, TemporalArray::Null) => {
                Err(MinarrowError::NullError { message: None })
            }
            _ => Err(MinarrowError::TypeError {
                from: "TemporalArray",
                to: "TemporalArray",
                message: Some("Mismatched temporal variants".to_string()),
            }),
        }
    }

    fn is_after(&self, other: &Self) -> Result<BooleanArray<()>, MinarrowError> {
        match (self, other) {
            (TemporalArray::Datetime32(a), TemporalArray::Datetime32(b)) => a.is_after(b),
            (TemporalArray::Datetime64(a), TemporalArray::Datetime64(b)) => a.is_after(b),
            (TemporalArray::Null, _) | (_, TemporalArray::Null) => {
                Err(MinarrowError::NullError { message: None })
            }
            _ => Err(MinarrowError::TypeError {
                from: "TemporalArray",
                to: "TemporalArray",
                message: Some("Mismatched temporal variants".to_string()),
            }),
        }
    }

    fn between(&self, start: &Self, end: &Self) -> Result<BooleanArray<()>, MinarrowError> {
        match (self, start, end) {
            (
                TemporalArray::Datetime32(a),
                TemporalArray::Datetime32(s),
                TemporalArray::Datetime32(e),
            ) => a.between(s, e),
            (
                TemporalArray::Datetime64(a),
                TemporalArray::Datetime64(s),
                TemporalArray::Datetime64(e),
            ) => a.between(s, e),
            (TemporalArray::Null, _, _)
            | (_, TemporalArray::Null, _)
            | (_, _, TemporalArray::Null) => Err(MinarrowError::NullError { message: None }),
            _ => Err(MinarrowError::TypeError {
                from: "TemporalArray",
                to: "TemporalArray",
                message: Some("Mismatched temporal variants".to_string()),
            }),
        }
    }

    // Truncation - delegate, wrap result back into enum variant

    fn truncate(&self, unit: &str) -> Result<Self, MinarrowError> {
        match self {
            TemporalArray::Datetime32(arr) => {
                Ok(TemporalArray::Datetime32(Arc::new(arr.truncate(unit)?)))
            }
            TemporalArray::Datetime64(arr) => {
                Ok(TemporalArray::Datetime64(Arc::new(arr.truncate(unit)?)))
            }
            TemporalArray::Null => Err(MinarrowError::NullError { message: None }),
        }
    }

    fn us(&self) -> Self {
        match self {
            TemporalArray::Datetime32(arr) => TemporalArray::Datetime32(Arc::new(arr.us())),
            TemporalArray::Datetime64(arr) => TemporalArray::Datetime64(Arc::new(arr.us())),
            TemporalArray::Null => TemporalArray::Null,
        }
    }

    fn ms(&self) -> Self {
        match self {
            TemporalArray::Datetime32(arr) => TemporalArray::Datetime32(Arc::new(arr.ms())),
            TemporalArray::Datetime64(arr) => TemporalArray::Datetime64(Arc::new(arr.ms())),
            TemporalArray::Null => TemporalArray::Null,
        }
    }

    fn sec(&self) -> Self {
        match self {
            TemporalArray::Datetime32(arr) => TemporalArray::Datetime32(Arc::new(arr.sec())),
            TemporalArray::Datetime64(arr) => TemporalArray::Datetime64(Arc::new(arr.sec())),
            TemporalArray::Null => TemporalArray::Null,
        }
    }

    fn min(&self) -> Self {
        match self {
            TemporalArray::Datetime32(arr) => TemporalArray::Datetime32(Arc::new(arr.min())),
            TemporalArray::Datetime64(arr) => TemporalArray::Datetime64(Arc::new(arr.min())),
            TemporalArray::Null => TemporalArray::Null,
        }
    }

    fn hr(&self) -> Self {
        match self {
            TemporalArray::Datetime32(arr) => TemporalArray::Datetime32(Arc::new(arr.hr())),
            TemporalArray::Datetime64(arr) => TemporalArray::Datetime64(Arc::new(arr.hr())),
            TemporalArray::Null => TemporalArray::Null,
        }
    }

    fn week(&self) -> Self {
        match self {
            TemporalArray::Datetime32(arr) => TemporalArray::Datetime32(Arc::new(arr.week())),
            TemporalArray::Datetime64(arr) => TemporalArray::Datetime64(Arc::new(arr.week())),
            TemporalArray::Null => TemporalArray::Null,
        }
    }

    // Type Casting

    fn cast_time_unit(&self, new_unit: TimeUnit) -> Result<Self, MinarrowError> {
        match self {
            TemporalArray::Datetime32(arr) => Ok(TemporalArray::Datetime32(Arc::new(
                arr.cast_time_unit(new_unit)?,
            ))),
            TemporalArray::Datetime64(arr) => Ok(TemporalArray::Datetime64(Arc::new(
                arr.cast_time_unit(new_unit)?,
            ))),
            TemporalArray::Null => Err(MinarrowError::NullError { message: None }),
        }
    }
}

/// Helper function to get the variant name for error messages
fn temporal_variant_name(arr: &TemporalArray) -> &'static str {
    match arr {
        TemporalArray::Datetime32(_) => "Datetime32",
        TemporalArray::Datetime64(_) => "Datetime64",
        TemporalArray::Null => "Null",
    }
}

impl Display for TemporalArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TemporalArray::Datetime32(arr) => {
                write_temporal_array_with_header(f, "Datetime32", arr.as_ref())
            }
            TemporalArray::Datetime64(arr) => {
                write_temporal_array_with_header(f, "Datetime64", arr.as_ref())
            }
            TemporalArray::Null => writeln!(f, "TemporalArray::Null [0 values]"),
        }
    }
}

/// Writes the standard header, then delegates to the contained array's Display.
fn write_temporal_array_with_header(
    f: &mut Formatter<'_>,
    dtype: &str,
    arr: &(impl MaskedArray + Display + ?Sized),
) -> std::fmt::Result {
    writeln!(
        f,
        "TemporalArray [{dtype}] [{} values] (null count: {})",
        arr.len(),
        arr.null_count()
    )?;
    Display::fmt(arr, f)
}
