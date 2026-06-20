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

//! # **DatetimeOps Trait** - *Datetime Operations for Temporal Arrays*
//!
//! Defines the `DatetimeOps` trait for datetime component extraction, arithmetic,
//! comparison, truncation, and type casting operations.
//!
//! Implemented by both `DatetimeArray<T>` and `TemporalArray`, enabling datetime
//! operations on enum-wrapped arrays without extracting the inner variant first.

use time::Duration;

use crate::{
    enums::{
        error::MinarrowError,
        time_units::{TimePeriod, TimeUnit},
    },
    structs::variants::{boolean::BooleanArray, integer::IntegerArray},
};

/// Datetime operations for arrays containing temporal data.
///
/// Provides component extraction, arithmetic, comparison, truncation,
/// and type casting methods. Implemented by `DatetimeArray<T>` for direct
/// access, and by `TemporalArray` for enum-level dispatch.
pub trait DatetimeOps: Sized {
    // Component Extraction

    /// Extracts the year component from all datetime values.
    fn year(&self) -> IntegerArray<i32>;

    /// Extracts the month component (1-12) from all datetime values.
    fn month(&self) -> IntegerArray<i32>;

    /// Extracts the day of month (1-31) from all datetime values.
    fn day(&self) -> IntegerArray<i32>;

    /// Extracts the hour component (0-23) from all datetime values.
    fn hour(&self) -> IntegerArray<i32>;

    /// Extracts the minute component (0-59) from all datetime values.
    fn minute(&self) -> IntegerArray<i32>;

    /// Extracts the second component (0-59) from all datetime values.
    fn second(&self) -> IntegerArray<i32>;

    /// Extracts the weekday (1=Sunday, 2=Monday, ..., 7=Saturday) from all datetime values.
    fn weekday(&self) -> IntegerArray<i32>;

    /// Extracts the day of year (1-366) from all datetime values.
    fn day_of_year(&self) -> IntegerArray<i32>;

    /// Extracts the ISO week number (1-53) from all datetime values.
    fn iso_week(&self) -> IntegerArray<i32>;

    /// Extracts the quarter (1-4) from all datetime values.
    fn quarter(&self) -> IntegerArray<i32>;

    /// Extracts the week of year (0-53) from all datetime values.
    /// Week 0 contains days before the first Sunday.
    fn week_of_year(&self) -> IntegerArray<i32>;

    /// Returns boolean array indicating whether each datetime's year is a leap year.
    fn is_leap_year(&self) -> BooleanArray<()>;

    // Arithmetic

    /// Adds a duration to all datetime values in the array.
    fn add_duration(&self, duration: Duration) -> Result<Self, MinarrowError>;

    /// Subtracts a duration from all datetime values in the array.
    fn sub_duration(&self, duration: Duration) -> Result<Self, MinarrowError>;

    /// Adds a number of days to all datetime values.
    fn add_days(&self, days: i64) -> Result<Self, MinarrowError>;

    /// Adds a number of months to all datetime values.
    fn add_months(&self, months: i32) -> Result<Self, MinarrowError>;

    /// Adds a number of years to all datetime values.
    fn add_years(&self, years: i32) -> Result<Self, MinarrowError>;

    // Comparison

    /// Calculate the duration between this datetime array and another.
    /// Returns an IntegerArray<i64> representing the difference in the specified unit.
    fn diff(&self, other: &Self, unit: TimeUnit) -> Result<IntegerArray<i64>, MinarrowError>;

    /// Calculate the absolute duration between elements (always positive).
    fn abs_diff(&self, other: &Self, unit: TimeUnit) -> Result<IntegerArray<i64>, MinarrowError>;

    /// Compares this array with another, returning a boolean array indicating
    /// where values in `self` are before values in `other`.
    fn is_before(&self, other: &Self) -> Result<BooleanArray<()>, MinarrowError>;

    /// Returns a boolean array indicating where values are after `other`.
    fn is_after(&self, other: &Self) -> Result<BooleanArray<()>, MinarrowError>;

    /// Returns a boolean array indicating where values fall within the range [start, end].
    fn between(&self, start: &Self, end: &Self) -> Result<BooleanArray<()>, MinarrowError>;

    // Truncation

    /// Floor datetime values to the start of `period`.
    ///
    /// `period` accepts a [`TimePeriod`] or its string label (e.g. `"day"`).
    fn truncate<P: Into<TimePeriod>>(&self, period: P) -> Self;

    /// Truncate to microsecond boundaries.
    fn us(&self) -> Self;

    /// Truncate to millisecond boundaries.
    fn ms(&self) -> Self;

    /// Truncate to second boundaries.
    fn sec(&self) -> Self;

    /// Truncate to minute boundaries.
    fn min(&self) -> Self;

    /// Truncate to hour boundaries.
    fn hr(&self) -> Self;

    /// Truncate to week boundaries (Sunday 00:00:00).
    fn week(&self) -> Self;

    // Type Casting

    /// Cast this array to a different TimeUnit.
    fn cast_time_unit(&self, new_unit: TimeUnit) -> Result<Self, MinarrowError>;
}
