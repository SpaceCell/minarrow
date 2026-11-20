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

//! # **DatetimeOps Module** - *Datetime Operations for DatetimeArray*
//!
//! Datetime operations for `DatetimeArray` built on the `time` crate.
//!
//! ## Features
//! - **Arithmetic**: add/subtract durations, days, months, years with overflow checking
//! - **Duration calculations**: diff and absolute diff between datetime arrays
//! - **Comparisons**: temporal ordering (before/after) and range checks (between)
//! - **Component extraction**: year, month, day, hour, minute, second, weekday, quarter, ISO week
//! - **Truncation**: round down to year, month, week, day, hour, minute, second, or sub-second boundaries
//! - **Type casting**: convert between time units (seconds <-> milliseconds <-> microseconds <-> nanoseconds)
//!
//! ## Output buffer variants
//! These methods allocate their result. The output-buffer (`_into`) forms that compute
//! straight into a caller-provided buffer for parallel execution live in
//! [`crate::kernels::datetime`]. Reach for those only when writing into engine-owned memory.
//!
//! ## Timezone Handling
//! `DatetimeArray` stores raw UTC timestamps as integers. Timezone information is metadata-only
//! and held in `Dt::Timestamp(TimeUnit, Option<String>)`, on the array and its `Field`.
//! For timezone operations, use `FieldArray::with_timezone()` and `FieldArray::to_utc()` which
//! update metadata without modifying the underlying UTC data.
//!
//! With the `time_zones` feature, operations on timestamps with a time zone read the
//! wall time of that time zone. The stored values stay UTC integers. Each operation
//! adds the UTC offset of the time zone at that instant to the stored value, runs the
//! same calendar arithmetic as for UTC values, and writes results that are timestamps
//! back as UTC values.
//! - Component extraction, such as `hour` and `day`, returns the components of the wall time.
//! - Truncation floors the wall time and writes the UTC value of the start of the period.
//!   The offset in force at the input is used where it is in force at the start of the
//!   period. Otherwise the result is the first instant at which the wall clock reads the
//!   start of the period or later.
//! - `add_months` and `add_years` keep the wall time. Wall times that occur twice use the
//!   earlier instant, and skipped wall times use the offset in force before the transition.
//! - `add_duration`, `sub_duration`, `add_days`, `diff`, the comparisons and
//!   `cast_time_unit` operate on instants, and return the same values with or without a
//!   time zone. `add_days` adds days of 24 hours.
//!
//! Offsets come from the IANA Time Zone Database, including daylight saving and historical
//! transitions. Without `time_zones`, every operation reads the UTC value, and timestamps
//! with a time zone print in UTC.
//!
//! ## Result types
//! The arithmetic and truncation operations return arrays of the input's `Dt`.
//! `cast_time_unit` keeps the type's variant and timezone, and changes its unit.
//!
//! ## Requirements
//! Requires the `time` feature to be enabled.

#[cfg(feature = "datetime_ops")]
use time::Duration;

#[cfg(feature = "datetime_ops")]
use crate::kernels::datetime::{
    add_duration_into, add_months_into, diff_into, is_leap_year_into, truncate_into,
};
#[cfg(feature = "datetime_ops")]
use crate::{
    Bitmask, DatetimeArray, DatetimeOps,
    enums::{
        error::MinarrowError,
        time_units::{TimePeriod, TimeUnit},
    },
    ffi::arrow_dtype::Dt,
    structs::variants::{boolean::BooleanArray, integer::IntegerArray},
    traits::{masked_array::MaskedArray, type_unions::Integer},
};
#[cfg(feature = "time_zones")]
use crate::structs::variants::datetime::tz::{utc_offset_seconds, wall_time_instants};
#[cfg(feature = "datetime_ops")]
use ::vec64::Vec64;
#[cfg(feature = "datetime_ops")]
use num_traits::FromPrimitive;

#[cfg(feature = "datetime_ops")]
impl<T: Integer + FromPrimitive> DatetimeArray<T> {
    /// Convert i64 value to OffsetDateTime based on time unit.
    #[inline(always)]
    pub(crate) fn i64_to_datetime(
        val_i64: i64,
        time_unit: TimeUnit,
    ) -> Option<time::OffsetDateTime> {
        use time::OffsetDateTime;
        match time_unit {
            TimeUnit::Seconds => OffsetDateTime::from_unix_timestamp(val_i64).ok(),
            TimeUnit::Milliseconds => {
                OffsetDateTime::from_unix_timestamp_nanos((val_i64 as i128) * 1_000_000).ok()
            }
            TimeUnit::Microseconds => {
                OffsetDateTime::from_unix_timestamp_nanos((val_i64 as i128) * 1_000).ok()
            }
            TimeUnit::Nanoseconds => {
                OffsetDateTime::from_unix_timestamp_nanos(val_i64 as i128).ok()
            }
            TimeUnit::Days => time::Date::from_julian_day((val_i64 + 2440588) as i32)
                .ok()
                .and_then(|date| date.with_hms(0, 0, 0).ok())
                .map(|dt| dt.assume_utc()),
        }
    }

    /// Returns the UTC offset lookup for the array's time zone.
    ///
    /// The lookup receives a stored UTC value and returns the offset of the time
    /// zone at that instant, in the array's time unit. Adding the offset to the
    /// stored value produces the wall time in the same unit. The kernels run
    /// their calendar arithmetic on that wall time, and stored values stay in UTC.
    ///
    /// The offset is 0 when the array has no time zone, when `utc_offset_seconds`
    /// cannot resolve the time zone text, and when the `time_zones` feature is off.
    pub(crate) fn wall_time_offset(&self) -> impl Fn(i64) -> i64 {
        #[cfg(feature = "time_zones")]
        {
            let zone = match &self.dtype {
                Dt::Timestamp(unit, Some(tz)) => {
                    utc_offset_seconds(tz).zip(Self::values_per_second(*unit))
                }
                _ => None,
            };
            move |value: i64| match &zone {
                Some((offset_seconds, per_second)) => {
                    offset_seconds(value.div_euclid(*per_second)).saturating_mul(*per_second)
                }
                None => 0,
            }
        }
        #[cfg(not(feature = "time_zones"))]
        {
            |_value: i64| 0
        }
    }

    /// Returns the conversion from wall times of the array's time zone to stored
    /// UTC values.
    ///
    /// The conversion receives a wall time in the array's time unit and returns
    /// two UTC values, as `wall_time_instants` defines them:
    /// - The first value is the earliest UTC value where the wall clock shows the
    ///   wall time or later. Truncation uses it.
    /// - The second value resolves skipped wall times with the offset from before
    ///   the transition. Month arithmetic uses it.
    ///
    /// Both values equal the wall time when the array has no time zone, when
    /// `utc_offset_seconds` cannot resolve the time zone text, and when the
    /// `time_zones` feature is off.
    pub(crate) fn wall_time_values(&self) -> impl Fn(i64) -> (i64, i64) {
        #[cfg(feature = "time_zones")]
        {
            let zone = match &self.dtype {
                Dt::Timestamp(unit, Some(tz)) => {
                    utc_offset_seconds(tz).zip(Self::values_per_second(*unit))
                }
                _ => None,
            };
            move |wall: i64| match &zone {
                Some((offset_seconds, per_second)) => {
                    let fraction = wall.rem_euclid(*per_second);
                    let (first, second) =
                        wall_time_instants(offset_seconds, wall.div_euclid(*per_second));
                    let second_value = second.saturating_mul(*per_second).saturating_add(fraction);
                    // The instant of a transition is a whole second.
                    if first == second {
                        (second_value, second_value)
                    } else {
                        (first.saturating_mul(*per_second), second_value)
                    }
                }
                None => (wall, wall),
            }
        }
        #[cfg(not(feature = "time_zones"))]
        {
            |wall: i64| (wall, wall)
        }
    }

    /// Returns the number of stored values per second for `unit`.
    ///
    /// Returns `None` for `Days`, because timestamps with a time zone are not
    /// stored in days.
    #[cfg(feature = "time_zones")]
    fn values_per_second(unit: TimeUnit) -> Option<i64> {
        match unit {
            TimeUnit::Seconds => Some(1),
            TimeUnit::Milliseconds => Some(1_000),
            TimeUnit::Microseconds => Some(1_000_000),
            TimeUnit::Nanoseconds => Some(1_000_000_000),
            TimeUnit::Days => None,
        }
    }

    /// Convert OffsetDateTime back to i64 value based on time unit.
    #[inline(always)]
    pub(crate) fn datetime_to_i64(dt: time::OffsetDateTime, time_unit: TimeUnit) -> i64 {
        match time_unit {
            TimeUnit::Seconds => dt.unix_timestamp(),
            TimeUnit::Milliseconds => {
                dt.unix_timestamp() * 1_000i64 + (dt.nanosecond() / 1_000_000) as i64
            }
            TimeUnit::Microseconds => {
                dt.unix_timestamp() * 1_000_000i64 + (dt.nanosecond() / 1_000) as i64
            }
            TimeUnit::Nanoseconds => {
                dt.unix_timestamp() * 1_000_000_000i64 + dt.nanosecond() as i64
            }
            TimeUnit::Days => dt.unix_timestamp() / 86400i64,
        }
    }

    /// Helper to convert a value at index to OffsetDateTime.
    ///
    /// Performs bounds and null checks. For hot loops, prefer direct i64 conversion
    /// with `i64_to_datetime` after manual validation.
    fn value_to_datetime(&self, i: usize) -> Option<time::OffsetDateTime> {
        if self.is_null(i) || i >= self.len() {
            return None;
        }
        let val_i64 = self.data[i].to_i64()?;
        Self::i64_to_datetime(val_i64, self.time_unit())
    }
}

#[cfg(feature = "datetime_ops")]
impl<T: Integer + FromPrimitive> DatetimeOps for DatetimeArray<T> {
    fn add_duration(&self, duration: Duration) -> Result<Self, MinarrowError> {
        let len = self.len();
        let mut data = Vec64::from_slice(&self.data[..]);
        let mut mask = match &self.null_mask {
            Some(m) => m.clone(),
            None => Bitmask::new_set_all(len, true),
        };
        add_duration_into(self, 0, duration, data.as_mut_slice(), Some(&mut mask))?;
        let null_mask = if self.null_mask.is_some() || mask.has_cleared() {
            Some(mask)
        } else {
            None
        };
        Ok(Self::from_vec64(data, null_mask, self.dtype.clone()))
    }

    /// Subtracts a duration from all datetime values in the array.
    fn sub_duration(&self, duration: Duration) -> Result<Self, MinarrowError> {
        self.add_duration(-duration)
    }

    /// Adds a number of days to all datetime values.
    fn add_days(&self, days: i64) -> Result<Self, MinarrowError> {
        self.add_duration(Duration::days(days))
    }

    /// Adds a number of months to all datetime values.
    ///
    /// Variable month length handling - if the day would be invalid in the target month
    /// (e.g., Jan 31 + 1 month), the day is clamped to the last valid day (Feb 28/29).
    /// Time-of-day is preserved.
    fn add_months(&self, months: i32) -> Result<Self, MinarrowError> {
        let len = self.len();
        let mut data = Vec64::from_slice(&self.data[..]);
        let mut mask = match &self.null_mask {
            Some(m) => m.clone(),
            None => Bitmask::new_set_all(len, true),
        };
        add_months_into(self, 0, months, data.as_mut_slice(), Some(&mut mask));
        let null_mask = if self.null_mask.is_some() || mask.has_cleared() {
            Some(mask)
        } else {
            None
        };
        Ok(Self::from_vec64(data, null_mask, self.dtype.clone()))
    }

    /// Adds a number of years to all datetime values.
    fn add_years(&self, years: i32) -> Result<Self, MinarrowError> {
        self.add_months(years * 12)
    }

    // Duration Operations

    /// Calculate the duration between this datetime array and another.
    /// Returns an IntegerArray<i64> representing the difference in the specified unit.
    ///
    /// # Arguments
    /// * `other` - The datetime array to subtract from self
    /// * `unit` - The TimeUnit for the result (Seconds, Milliseconds, Microseconds, Nanoseconds)
    fn diff(&self, other: &Self, unit: TimeUnit) -> Result<IntegerArray<i64>, MinarrowError> {
        if self.len() != other.len() {
            return Err(MinarrowError::TypeError {
                from: "DatetimeArray",
                to: "IntegerArray",
                message: Some(format!(
                    "Array lengths do not match: {} vs {}",
                    self.len(),
                    other.len()
                )),
            });
        }

        let len = self.len();
        let mut data = Vec64::<i64>::with_capacity(len);
        unsafe { data.set_len(len) };
        let mut mask = match (&self.null_mask, &other.null_mask) {
            (None, None) => Bitmask::new_set_all(len, true),
            (Some(a), None) => a.clone(),
            (None, Some(b)) => b.clone(),
            (Some(a), Some(b)) => a.intersect(b),
        };
        diff_into(self, 0, other, 0, unit, &mut data, Some(&mut mask));
        let null_mask =
            if self.null_mask.is_some() || other.null_mask.is_some() || mask.has_cleared() {
                Some(mask)
            } else {
                None
            };
        Ok(IntegerArray::from_vec64(data, null_mask))
    }

    /// Calculate the absolute duration between elements (always positive).
    fn abs_diff(&self, other: &Self, unit: TimeUnit) -> Result<IntegerArray<i64>, MinarrowError> {
        let diff = self.diff(other, unit)?;
        let mut result = IntegerArray::with_capacity(diff.len(), diff.is_nullable());

        for i in 0..diff.len() {
            if diff.is_null(i) {
                result.push_null();
            } else {
                result.push(diff.data[i].abs());
            }
        }

        Ok(result)
    }

    // Comparison Operations

    /// Compares this array with another, returning a boolean array indicating
    /// where values in `self` are before values in `other`.
    fn is_before(&self, other: &Self) -> Result<BooleanArray<()>, MinarrowError> {
        if self.len() != other.len() {
            return Err(MinarrowError::TypeError {
                from: "DatetimeArray",
                to: "BooleanArray",
                message: Some(format!(
                    "Array lengths do not match: {} vs {}",
                    self.len(),
                    other.len()
                )),
            });
        }

        let mut result = BooleanArray::with_capacity(self.len(), true);

        for i in 0..self.len() {
            if self.is_null(i) || other.is_null(i) {
                result.push_null();
            } else {
                let self_dt = self.value_to_datetime(i);
                let other_dt = other.value_to_datetime(i);

                if let (Some(a), Some(b)) = (self_dt, other_dt) {
                    result.push(if a < b { true } else { false });
                } else {
                    result.push_null();
                }
            }
        }

        Ok(result)
    }

    /// Returns a boolean array indicating where values are after `other`.
    fn is_after(&self, other: &Self) -> Result<BooleanArray<()>, MinarrowError> {
        if self.len() != other.len() {
            return Err(MinarrowError::TypeError {
                from: "DatetimeArray",
                to: "BooleanArray",
                message: Some(format!(
                    "Array lengths do not match: {} vs {}",
                    self.len(),
                    other.len()
                )),
            });
        }

        let mut result = BooleanArray::with_capacity(self.len(), true);

        for i in 0..self.len() {
            if self.is_null(i) || other.is_null(i) {
                result.push_null();
            } else {
                let self_dt = self.value_to_datetime(i);
                let other_dt = other.value_to_datetime(i);

                if let (Some(a), Some(b)) = (self_dt, other_dt) {
                    result.push(if a > b { true } else { false });
                } else {
                    result.push_null();
                }
            }
        }

        Ok(result)
    }

    /// Returns a boolean array indicating where values fall within the range [start, end].
    fn between(&self, start: &Self, end: &Self) -> Result<BooleanArray<()>, MinarrowError> {
        if self.len() != start.len() || self.len() != end.len() {
            return Err(MinarrowError::TypeError {
                from: "DatetimeArray",
                to: "BooleanArray",
                message: Some("Array lengths do not match".to_string()),
            });
        }

        let mut result = BooleanArray::with_capacity(self.len(), true);

        for i in 0..self.len() {
            if self.is_null(i) || start.is_null(i) || end.is_null(i) {
                result.push_null();
            } else {
                let self_dt = self.value_to_datetime(i);
                let start_dt = start.value_to_datetime(i);
                let end_dt = end.value_to_datetime(i);

                if let (Some(val), Some(s), Some(e)) = (self_dt, start_dt, end_dt) {
                    result.push(if val >= s && val <= e { true } else { false });
                } else {
                    result.push_null();
                }
            }
        }

        Ok(result)
    }

    // Component Extraction Operations

    /// Extracts the year component from all datetime values.
    fn year(&self) -> IntegerArray<i32> {
        let mut result = IntegerArray::with_capacity(self.len(), self.is_nullable());
        let wall_time_offset = self.wall_time_offset();

        for i in 0..self.len() {
            if self.is_null(i) {
                result.push_null();
            } else if let Some(dt) = self.data[i]
                .to_i64()
                .and_then(|v| v.checked_add(wall_time_offset(v)))
                .and_then(|v| Self::i64_to_datetime(v, self.time_unit()))
            {
                result.push(dt.year());
            } else {
                result.push_null();
            }
        }

        result
    }

    /// Extracts the month component (1-12) from all datetime values.
    fn month(&self) -> IntegerArray<i32> {
        let mut result = IntegerArray::with_capacity(self.len(), self.is_nullable());
        let wall_time_offset = self.wall_time_offset();

        for i in 0..self.len() {
            if self.is_null(i) {
                result.push_null();
            } else if let Some(dt) = self.data[i]
                .to_i64()
                .and_then(|v| v.checked_add(wall_time_offset(v)))
                .and_then(|v| Self::i64_to_datetime(v, self.time_unit()))
            {
                result.push(dt.month() as i32);
            } else {
                result.push_null();
            }
        }

        result
    }

    /// Extracts the day of month (1-31) from all datetime values.
    fn day(&self) -> IntegerArray<i32> {
        let mut result = IntegerArray::with_capacity(self.len(), self.is_nullable());
        let wall_time_offset = self.wall_time_offset();

        for i in 0..self.len() {
            if self.is_null(i) {
                result.push_null();
            } else if let Some(dt) = self.data[i]
                .to_i64()
                .and_then(|v| v.checked_add(wall_time_offset(v)))
                .and_then(|v| Self::i64_to_datetime(v, self.time_unit()))
            {
                result.push(dt.day() as i32);
            } else {
                result.push_null();
            }
        }

        result
    }

    /// Extracts the hour component (0-23) from all datetime values.
    fn hour(&self) -> IntegerArray<i32> {
        let mut result = IntegerArray::with_capacity(self.len(), self.is_nullable());
        let wall_time_offset = self.wall_time_offset();

        for i in 0..self.len() {
            if self.is_null(i) {
                result.push_null();
            } else if let Some(dt) = self.data[i]
                .to_i64()
                .and_then(|v| v.checked_add(wall_time_offset(v)))
                .and_then(|v| Self::i64_to_datetime(v, self.time_unit()))
            {
                result.push(dt.hour() as i32);
            } else {
                result.push_null();
            }
        }

        result
    }

    /// Extracts the minute component (0-59) from all datetime values.
    fn minute(&self) -> IntegerArray<i32> {
        let mut result = IntegerArray::with_capacity(self.len(), self.is_nullable());
        let wall_time_offset = self.wall_time_offset();

        for i in 0..self.len() {
            if self.is_null(i) {
                result.push_null();
            } else if let Some(dt) = self.data[i]
                .to_i64()
                .and_then(|v| v.checked_add(wall_time_offset(v)))
                .and_then(|v| Self::i64_to_datetime(v, self.time_unit()))
            {
                result.push(dt.minute() as i32);
            } else {
                result.push_null();
            }
        }

        result
    }

    /// Extracts the second component (0-59) from all datetime values.
    fn second(&self) -> IntegerArray<i32> {
        let mut result = IntegerArray::with_capacity(self.len(), self.is_nullable());
        let wall_time_offset = self.wall_time_offset();

        for i in 0..self.len() {
            if self.is_null(i) {
                result.push_null();
            } else if let Some(dt) = self.data[i]
                .to_i64()
                .and_then(|v| v.checked_add(wall_time_offset(v)))
                .and_then(|v| Self::i64_to_datetime(v, self.time_unit()))
            {
                result.push(dt.second() as i32);
            } else {
                result.push_null();
            }
        }

        result
    }

    /// Extracts the weekday (1=Sunday, 2=Monday, ..., 7=Saturday) from all datetime values.
    fn weekday(&self) -> IntegerArray<i32> {
        let mut result = IntegerArray::with_capacity(self.len(), self.is_nullable());
        let wall_time_offset = self.wall_time_offset();

        for i in 0..self.len() {
            if self.is_null(i) {
                result.push_null();
            } else if let Some(dt) = self.data[i]
                .to_i64()
                .and_then(|v| v.checked_add(wall_time_offset(v)))
                .and_then(|v| Self::i64_to_datetime(v, self.time_unit()))
            {
                result.push(dt.weekday().number_from_sunday() as i32);
            } else {
                result.push_null();
            }
        }

        result
    }

    /// Extracts the day of year (1-366) from all datetime values.
    fn day_of_year(&self) -> IntegerArray<i32> {
        let mut result = IntegerArray::with_capacity(self.len(), self.is_nullable());
        let wall_time_offset = self.wall_time_offset();

        for i in 0..self.len() {
            if self.is_null(i) {
                result.push_null();
            } else if let Some(dt) = self.data[i]
                .to_i64()
                .and_then(|v| v.checked_add(wall_time_offset(v)))
                .and_then(|v| Self::i64_to_datetime(v, self.time_unit()))
            {
                result.push(dt.ordinal() as i32);
            } else {
                result.push_null();
            }
        }

        result
    }

    /// Extracts the ISO week number (1-53) from all datetime values.
    fn iso_week(&self) -> IntegerArray<i32> {
        let mut result = IntegerArray::with_capacity(self.len(), self.is_nullable());
        let wall_time_offset = self.wall_time_offset();

        for i in 0..self.len() {
            if self.is_null(i) {
                result.push_null();
            } else if let Some(dt) = self.data[i]
                .to_i64()
                .and_then(|v| v.checked_add(wall_time_offset(v)))
                .and_then(|v| Self::i64_to_datetime(v, self.time_unit()))
            {
                result.push(dt.iso_week() as i32);
            } else {
                result.push_null();
            }
        }

        result
    }

    /// Extracts the quarter (1-4) from all datetime values.
    fn quarter(&self) -> IntegerArray<i32> {
        let mut result = IntegerArray::with_capacity(self.len(), self.is_nullable());
        let wall_time_offset = self.wall_time_offset();

        for i in 0..self.len() {
            if self.is_null(i) {
                result.push_null();
            } else if let Some(dt) = self.data[i]
                .to_i64()
                .and_then(|v| v.checked_add(wall_time_offset(v)))
                .and_then(|v| Self::i64_to_datetime(v, self.time_unit()))
            {
                let month = dt.month() as i32;
                let quarter = ((month - 1) / 3) + 1;
                result.push(quarter);
            } else {
                result.push_null();
            }
        }

        result
    }

    /// Extracts the week of year (0-53) from all datetime values.
    /// Week 0 contains days before the first Sunday.
    fn week_of_year(&self) -> IntegerArray<i32> {
        let mut result = IntegerArray::with_capacity(self.len(), self.is_nullable());
        let wall_time_offset = self.wall_time_offset();

        for i in 0..self.len() {
            if self.is_null(i) {
                result.push_null();
            } else if let Some(dt) = self.data[i]
                .to_i64()
                .and_then(|v| v.checked_add(wall_time_offset(v)))
                .and_then(|v| Self::i64_to_datetime(v, self.time_unit()))
            {
                let day_of_year = dt.ordinal() as i32;
                let weekday = dt.weekday().number_from_sunday() as i32;
                // Calculate week based on day of year and weekday (Sunday = 1)
                let week = (day_of_year + 7 - weekday) / 7;
                result.push(week);
            } else {
                result.push_null();
            }
        }

        result
    }

    /// Returns boolean array indicating whether each datetime's year is a leap year.
    fn is_leap_year(&self) -> BooleanArray<()> {
        let len = self.len();
        let mut bits = Bitmask::new_set_all(len, false);
        let mut mask = match &self.null_mask {
            Some(m) => m.clone(),
            None => Bitmask::new_set_all(len, true),
        };
        is_leap_year_into(self, 0, &mut bits, Some(&mut mask));
        let null_mask = if self.null_mask.is_some() || mask.has_cleared() {
            Some(mask)
        } else {
            None
        };
        BooleanArray::new(bits, null_mask)
    }

    // Rounding/Truncating Operations

    /// Floor datetime values to the start of `period`.
    ///
    /// `period` accepts a [`TimePeriod`] or its string label (e.g. `"day"`).
    fn truncate<P: Into<TimePeriod>>(&self, period: P) -> Self {
        let period = period.into();
        let len = self.len();
        let mut data = Vec64::from_slice(&self.data[..]);
        let mut mask = match &self.null_mask {
            Some(m) => m.clone(),
            None => Bitmask::new_set_all(len, true),
        };
        truncate_into(self, 0, period, data.as_mut_slice(), Some(&mut mask));
        let null_mask = if self.null_mask.is_some() || mask.has_cleared() {
            Some(mask)
        } else {
            None
        };
        Self::from_vec64(data, null_mask, self.dtype.clone())
    }

    // Truncation Shorthand Methods

    /// Truncate to microsecond boundaries.
    ///
    /// Only meaningful for nanosecond precision arrays.
    fn us(&self) -> Self {
        self.truncate(TimePeriod::Microsecond)
    }

    /// Truncate to millisecond boundaries.
    ///
    /// Meaningful for nanosecond and microsecond precision arrays.
    fn ms(&self) -> Self {
        self.truncate(TimePeriod::Millisecond)
    }

    /// Truncate to second boundaries.
    fn sec(&self) -> Self {
        self.truncate(TimePeriod::Second)
    }

    /// Truncate to minute boundaries.
    fn min(&self) -> Self {
        self.truncate(TimePeriod::Minute)
    }

    /// Truncate to hour boundaries.
    fn hr(&self) -> Self {
        self.truncate(TimePeriod::Hour)
    }

    /// Truncate to week boundaries (Sunday 00:00:00).
    fn week(&self) -> Self {
        self.truncate(TimePeriod::Week)
    }

    // Type Casting Operations

    /// Cast this DatetimeArray to a different TimeUnit.
    ///
    /// This converts the internal representation while preserving the logical datetime values.
    ///
    /// The result keeps the `Dt` variant and timezone, with the new unit.
    ///
    /// ## Dates
    /// - Dates have a fixed unit, and casting `Date32` or `Date64` to its own unit
    ///   keeps the type.
    /// - Casting `Date64` to another unit returns a `Timestamp` without a timezone.
    /// - Casting `Date32` to another unit returns an error, as no 32-bit temporal
    ///   type represents instants at a sub-day unit.
    fn cast_time_unit(&self, new_unit: TimeUnit) -> Result<Self, MinarrowError> {
        let dtype = match &self.dtype {
            Dt::Date32 if new_unit == TimeUnit::Days => Dt::Date32,
            Dt::Date32 => {
                return Err(MinarrowError::IncompatibleTypeError {
                    from: "Date32",
                    to: "DatetimeArray<i32>",
                    message: Some(format!(
                        "Date32 values cannot be cast to {new_unit}: no 32-bit temporal type holds instants at that unit"
                    )),
                });
            }
            Dt::Date64 if new_unit == TimeUnit::Milliseconds => Dt::Date64,
            Dt::Date64 => Dt::Timestamp(new_unit, None),
            Dt::Time32(_) => Dt::Time32(new_unit),
            Dt::Time64(_) => Dt::Time64(new_unit),
            Dt::Duration32(_) => Dt::Duration32(new_unit),
            Dt::Duration64(_) => Dt::Duration64(new_unit),
            Dt::Timestamp(_, tz) => Dt::Timestamp(new_unit, tz.clone()),
            Dt::Interval(_) => {
                return Err(MinarrowError::IncompatibleTypeError {
                    from: "Interval",
                    to: "DatetimeArray",
                    message: Some("DatetimeArray does not hold interval data".to_string()),
                });
            }
        };
        let mut result = Self::with_capacity(self.len(), self.is_nullable(), dtype);

        for i in 0..self.len() {
            if self.is_null(i) {
                result.push_null();
            } else if let Some(dt) = self.value_to_datetime(i) {
                // Convert to new unit
                let new_val: i64 = match new_unit {
                    TimeUnit::Seconds => dt.unix_timestamp(),
                    TimeUnit::Milliseconds => {
                        dt.unix_timestamp() * 1_000i64 + (dt.nanosecond() / 1_000_000) as i64
                    }
                    TimeUnit::Microseconds => {
                        dt.unix_timestamp() * 1_000_000i64 + (dt.nanosecond() / 1_000) as i64
                    }
                    TimeUnit::Nanoseconds => {
                        dt.unix_timestamp() * 1_000_000_000i64 + dt.nanosecond() as i64
                    }
                    TimeUnit::Days => dt.unix_timestamp() / 86400i64,
                };

                if let Some(new_val_t) = T::from_i64(new_val) {
                    result.push(new_val_t);
                } else {
                    result.push_null();
                }
            } else {
                result.push_null();
            }
        }

        Ok(result)
    }
}

#[cfg(all(test, feature = "datetime_ops"))]
mod tests {
    use super::*;
    use crate::DatetimeOps;
    use time::Duration;
    use vec64::vec64;

    #[test]
    fn test_add_duration() {
        let arr = DatetimeArray::<i64>::from_slice(
            &[1000, 2000, 3000],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        let result = arr.add_duration(Duration::seconds(10)).unwrap();

        assert_eq!(result.value(0), Some(1010));
        assert_eq!(result.value(1), Some(2010));
        assert_eq!(result.value(2), Some(3010));
    }

    #[test]
    fn test_sub_duration() {
        let arr = DatetimeArray::<i64>::from_slice(
            &[1000, 2000, 3000],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        let result = arr.sub_duration(Duration::seconds(10)).unwrap();

        assert_eq!(result.value(0), Some(990));
        assert_eq!(result.value(1), Some(1990));
        assert_eq!(result.value(2), Some(2990));
    }

    #[test]
    fn test_add_days() {
        let arr = DatetimeArray::<i64>::from_slice(
            &[0, 86400, 172800],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        let result = arr.add_days(1).unwrap();

        assert_eq!(result.value(0), Some(86400));
        assert_eq!(result.value(1), Some(172800));
        assert_eq!(result.value(2), Some(259200));
    }

    #[test]
    fn test_is_before() {
        let arr1 = DatetimeArray::<i64>::from_slice(
            &[1000, 2000, 3000],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        let arr2 = DatetimeArray::<i64>::from_slice(
            &[1500, 2500, 2500],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );

        let result = arr1.is_before(&arr2).unwrap();

        assert_eq!(result.get(0), Some(true));
        assert_eq!(result.get(1), Some(true));
        assert_eq!(result.get(2), Some(false));
    }

    #[test]
    fn test_is_after() {
        let arr1 = DatetimeArray::<i64>::from_slice(
            &[1000, 2000, 3000],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        let arr2 = DatetimeArray::<i64>::from_slice(
            &[1500, 2500, 2500],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );

        let result = arr1.is_after(&arr2).unwrap();

        assert_eq!(result.get(0), Some(false));
        assert_eq!(result.get(1), Some(false));
        assert_eq!(result.get(2), Some(true));
    }

    #[test]
    fn test_component_extraction() {
        // 2023-11-14 22:13:20 UTC
        let timestamp = 1_700_000_000i64;
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );

        assert_eq!(arr.year().get(0), Some(2023));
        assert_eq!(arr.month().get(0), Some(11));
        assert_eq!(arr.day().get(0), Some(14));
        assert_eq!(arr.hour().get(0), Some(22));
        assert_eq!(arr.minute().get(0), Some(13));
        assert_eq!(arr.second().get(0), Some(20));
    }

    // Arithmetic Operations Tests

    #[test]
    fn test_add_months() {
        // 2023-01-15 -> add 2 months -> 2023-03-15
        let timestamp = 1_673_740_800i64; // 2023-01-15 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        let result = arr.add_months(2).unwrap();

        // Verify we got March 15th, 2023
        assert_eq!(result.year().get(0), Some(2023));
        assert_eq!(result.month().get(0), Some(3));
        assert_eq!(result.day().get(0), Some(15));
    }

    #[test]
    fn test_add_months_overflow_year() {
        // 2023-11-15 -> add 3 months -> 2024-02-15
        let timestamp = 1_700_006_400i64; // 2023-11-15 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        let result = arr.add_months(3).unwrap();

        assert_eq!(result.year().get(0), Some(2024));
        assert_eq!(result.month().get(0), Some(2));
        assert_eq!(result.day().get(0), Some(15));
    }

    #[test]
    fn test_add_months_end_of_month() {
        // 2023-01-31 -> add 1 month -> 2023-02-28 (not 02-31)
        let timestamp = 1_675_123_200i64; // 2023-01-31 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        let result = arr.add_months(1).unwrap();

        assert_eq!(result.year().get(0), Some(2023));
        assert_eq!(result.month().get(0), Some(2));
        assert_eq!(result.day().get(0), Some(28)); // Clamped to Feb 28
    }

    #[test]
    fn test_add_years() {
        let timestamp = 1_700_000_000i64; // 2023-11-14 22:13:20 UTC
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        let result = arr.add_years(2).unwrap();

        assert_eq!(result.year().get(0), Some(2025));
        assert_eq!(result.month().get(0), Some(11));
        assert_eq!(result.day().get(0), Some(14));
    }

    // Duration Operations Tests

    #[test]
    fn test_diff() {
        let arr1 = DatetimeArray::<i64>::from_slice(
            &[1000, 2000, 3000],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        let arr2 = DatetimeArray::<i64>::from_slice(
            &[1500, 2500, 2500],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );

        let result = arr1.diff(&arr2, TimeUnit::Seconds).unwrap();

        assert_eq!(result.get(0), Some(-500));
        assert_eq!(result.get(1), Some(-500));
        assert_eq!(result.get(2), Some(500));
    }

    #[test]
    fn test_abs_diff() {
        let arr1 = DatetimeArray::<i64>::from_slice(
            &[1000, 2000, 3000],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        let arr2 = DatetimeArray::<i64>::from_slice(
            &[1500, 2500, 2500],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );

        let result = arr1.abs_diff(&arr2, TimeUnit::Seconds).unwrap();

        assert_eq!(result.get(0), Some(500));
        assert_eq!(result.get(1), Some(500));
        assert_eq!(result.get(2), Some(500));
    }

    #[test]
    fn test_diff_different_units() {
        let arr1 = DatetimeArray::<i64>::from_slice(
            &[1_000_000, 2_000_000],
            Dt::Timestamp(TimeUnit::Milliseconds, None),
        );
        let arr2 = DatetimeArray::<i64>::from_slice(
            &[1_500_000, 2_500_000],
            Dt::Timestamp(TimeUnit::Milliseconds, None),
        );

        let result = arr1.diff(&arr2, TimeUnit::Milliseconds).unwrap();

        assert_eq!(result.get(0), Some(-500_000));
        assert_eq!(result.get(1), Some(-500_000));
    }

    // Comparison Operations Tests

    #[test]
    fn test_between() {
        let arr = DatetimeArray::<i64>::from_slice(
            &[1000, 2000, 3000, 4000],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        let start = DatetimeArray::<i64>::from_slice(
            &[1500, 1500, 1500, 1500],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        let end = DatetimeArray::<i64>::from_slice(
            &[3500, 3500, 3500, 3500],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );

        let result = arr.between(&start, &end).unwrap();

        assert_eq!(result.get(0), Some(false)); // 1000 < 1500
        assert_eq!(result.get(1), Some(true)); // 1500 <= 2000 <= 3500
        assert_eq!(result.get(2), Some(true)); // 1500 <= 3000 <= 3500
        assert_eq!(result.get(3), Some(false)); // 4000 > 3500
    }

    // Component Extraction Tests

    #[test]
    fn test_weekday() {
        // 2023-11-14 is a Tuesday (1=Sunday, 2=Monday, 3=Tuesday, ...)
        let timestamp = 1_700_000_000i64;
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );

        assert_eq!(arr.weekday().get(0), Some(3)); // Tuesday = 3
    }

    #[test]
    fn test_weekday_all_days() {
        // Test all days of the week
        let timestamps = vec![
            1_699_747_200i64, // 2023-11-12 Sunday
            1_699_833_600i64, // 2023-11-13 Monday
            1_699_920_000i64, // 2023-11-14 Tuesday
            1_700_006_400i64, // 2023-11-15 Wednesday
            1_700_092_800i64, // 2023-11-16 Thursday
            1_700_179_200i64, // 2023-11-17 Friday
            1_700_265_600i64, // 2023-11-18 Saturday
        ];
        let arr = DatetimeArray::<i64>::from_slice(
            &timestamps,
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        let weekdays = arr.weekday();

        assert_eq!(weekdays.get(0), Some(1)); // Sunday
        assert_eq!(weekdays.get(1), Some(2)); // Monday
        assert_eq!(weekdays.get(2), Some(3)); // Tuesday
        assert_eq!(weekdays.get(3), Some(4)); // Wednesday
        assert_eq!(weekdays.get(4), Some(5)); // Thursday
        assert_eq!(weekdays.get(5), Some(6)); // Friday
        assert_eq!(weekdays.get(6), Some(7)); // Saturday
    }

    #[test]
    fn test_day_of_year() {
        // 2023-11-14 is the 318th day of the year
        let timestamp = 1_700_000_000i64;
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );

        assert_eq!(arr.day_of_year().get(0), Some(318));
    }

    #[test]
    fn test_iso_week() {
        // 2023-11-14 is in ISO week 46
        let timestamp = 1_700_000_000i64;
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );

        assert_eq!(arr.iso_week().get(0), Some(46));
    }

    #[test]
    fn test_quarter() {
        let timestamps = vec![
            1_672_531_200i64, // 2023-01-01 -> Q1
            1_680_307_200i64, // 2023-04-01 -> Q2
            1_688_169_600i64, // 2023-07-01 -> Q3
            1_696_118_400i64, // 2023-10-01 -> Q4
        ];
        let arr = DatetimeArray::<i64>::from_slice(
            &timestamps,
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        let quarters = arr.quarter();

        assert_eq!(quarters.get(0), Some(1));
        assert_eq!(quarters.get(1), Some(2));
        assert_eq!(quarters.get(2), Some(3));
        assert_eq!(quarters.get(3), Some(4));
    }

    #[test]
    fn test_week_of_year() {
        // 2023-11-14 is in week 46
        let timestamp = 1_700_000_000i64;
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );

        let week = arr.week_of_year().get(0).unwrap();
        assert!(week >= 45 && week <= 47); // Allow some variation in week calculation
    }

    #[test]
    fn test_is_leap_year() {
        let timestamps = vec![
            1_672_531_200i64, // 2023-01-01 -> not leap
            1_704_067_200i64, // 2024-01-01 -> leap year
        ];
        let arr = DatetimeArray::<i64>::from_slice(
            &timestamps,
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        let leap_years = arr.is_leap_year();

        assert_eq!(leap_years.get(0), Some(false));
        assert_eq!(leap_years.get(1), Some(true));
    }

    // Truncation Operations Tests

    #[test]
    fn test_truncate_to_year() {
        let timestamp = 1_700_000_000i64; // 2023-11-14 22:13:20 UTC
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );

        let result = arr.truncate(TimePeriod::Year);

        assert_eq!(result.year().get(0), Some(2023));
        assert_eq!(result.month().get(0), Some(1));
        assert_eq!(result.day().get(0), Some(1));
        assert_eq!(result.hour().get(0), Some(0));
        assert_eq!(result.minute().get(0), Some(0));
        assert_eq!(result.second().get(0), Some(0));
    }

    #[test]
    fn test_truncate_to_month() {
        let timestamp = 1_700_000_000i64; // 2023-11-14 22:13:20 UTC
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );

        let result = arr.truncate(TimePeriod::Month);

        assert_eq!(result.year().get(0), Some(2023));
        assert_eq!(result.month().get(0), Some(11));
        assert_eq!(result.day().get(0), Some(1));
        assert_eq!(result.hour().get(0), Some(0));
    }

    #[test]
    fn test_truncate_to_day() {
        let timestamp = 1_700_000_000i64; // 2023-11-14 22:13:20 UTC
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );

        let result = arr.truncate(TimePeriod::Day);

        assert_eq!(result.hour().get(0), Some(0));
        assert_eq!(result.minute().get(0), Some(0));
        assert_eq!(result.second().get(0), Some(0));
    }

    #[test]
    fn test_truncate_to_hour() {
        let timestamp = 1_700_000_000i64; // 2023-11-14 22:13:20 UTC
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );

        let result = arr.truncate(TimePeriod::Hour);

        assert_eq!(result.hour().get(0), Some(22));
        assert_eq!(result.minute().get(0), Some(0));
        assert_eq!(result.second().get(0), Some(0));
    }

    #[test]
    fn test_truncate_shorthand_methods() {
        let timestamp = 1_700_000_000i64;
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );

        // Test sec(), min(), hr()
        let sec_result = arr.sec();
        let min_result = arr.min();
        let hr_result = arr.hr();

        assert_eq!(sec_result.second().get(0), Some(20));
        assert_eq!(min_result.minute().get(0), Some(13));
        assert_eq!(min_result.second().get(0), Some(0));
        assert_eq!(hr_result.hour().get(0), Some(22));
        assert_eq!(hr_result.minute().get(0), Some(0));
    }

    #[test]
    fn test_truncate_week() {
        // 2023-11-14 is a Tuesday; week should start on Sunday 2023-11-12
        let timestamp = 1_700_000_000i64; // 2023-11-14 22:13:20 UTC
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );

        let result = arr.week();

        assert_eq!(result.day().get(0), Some(12)); // Sunday
        assert_eq!(result.hour().get(0), Some(0));
        assert_eq!(result.minute().get(0), Some(0));
    }

    #[test]
    fn test_truncate_subsecond_us() {
        let arr = DatetimeArray::<i64>::from_slice(
            &[1_234_567],
            Dt::Timestamp(TimeUnit::Nanoseconds, None),
        );
        let result = arr.us();

        // Should truncate to nearest microsecond (1000 ns)
        assert_eq!(result.value(0), Some(1_234_000));
    }

    #[test]
    fn test_truncate_subsecond_ms() {
        let arr = DatetimeArray::<i64>::from_slice(
            &[1_234_567_890],
            Dt::Timestamp(TimeUnit::Nanoseconds, None),
        );
        let result = arr.ms();

        // Should truncate to nearest millisecond (1_000_000 ns)
        assert_eq!(result.value(0), Some(1_234_000_000));
    }

    // Type Casting Tests

    #[test]
    fn test_cast_time_unit_seconds_to_milliseconds() {
        let arr = DatetimeArray::<i64>::from_slice(
            &[1, 2, 3],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        let result = arr.cast_time_unit(TimeUnit::Milliseconds).unwrap();

        assert_eq!(result.value(0), Some(1_000));
        assert_eq!(result.value(1), Some(2_000));
        assert_eq!(result.value(2), Some(3_000));
        assert_eq!(result.time_unit(), TimeUnit::Milliseconds);
    }

    #[test]
    fn test_cast_time_unit_milliseconds_to_seconds() {
        let arr = DatetimeArray::<i64>::from_slice(
            &[1_000, 2_000, 3_000],
            Dt::Timestamp(TimeUnit::Milliseconds, None),
        );
        let result = arr.cast_time_unit(TimeUnit::Seconds).unwrap();

        assert_eq!(result.value(0), Some(1));
        assert_eq!(result.value(1), Some(2));
        assert_eq!(result.value(2), Some(3));
        assert_eq!(result.time_unit(), TimeUnit::Seconds);
    }

    #[test]
    fn test_cast_time_unit_preserves_datetime() {
        // Cast should preserve the actual datetime, just change representation
        let timestamp = 1_700_000_000i64;
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );

        let as_ms = arr.cast_time_unit(TimeUnit::Milliseconds).unwrap();
        let back_to_sec = as_ms.cast_time_unit(TimeUnit::Seconds).unwrap();

        assert_eq!(back_to_sec.value(0), Some(timestamp));
    }

    // Null Handling and Edge Cases

    #[test]
    fn test_operations_with_nulls() {
        use crate::Bitmask;

        let mask = Bitmask::from_bools(&[true, false, true]);
        let arr = DatetimeArray::new(
            vec64![1000i64, 2000, 3000],
            Some(mask),
            Dt::Timestamp(TimeUnit::Seconds, None),
        );

        let result = arr.add_duration(Duration::seconds(10)).unwrap();

        assert_eq!(result.value(0), Some(1010));
        assert_eq!(result.value(1), None); // null preserved
        assert_eq!(result.value(2), Some(3010));
    }

    #[test]
    fn test_comparison_with_nulls() {
        use crate::Bitmask;

        let mask1 = Bitmask::from_bools(&[true, false, true]);
        let arr1 = DatetimeArray::new(
            vec64![1000i64, 2000, 3000],
            Some(mask1),
            Dt::Timestamp(TimeUnit::Seconds, None),
        );

        let arr2 = DatetimeArray::<i64>::from_slice(
            &[1500, 2500, 2500],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );

        let result = arr1.is_before(&arr2).unwrap();

        assert_eq!(result.get(0), Some(true));
        assert_eq!(result.get(1), None); // null input -> null output
        assert_eq!(result.get(2), Some(false));
    }

    #[test]
    fn test_diff_length_mismatch() {
        let arr1 = DatetimeArray::<i64>::from_slice(
            &[1000, 2000],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        let arr2 = DatetimeArray::<i64>::from_slice(
            &[1500],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );

        let result = arr1.diff(&arr2, TimeUnit::Seconds);
        assert!(result.is_err());
    }

    #[test]
    fn test_component_extraction_with_nulls() {
        use crate::Bitmask;

        let mask = Bitmask::from_bools(&[true, false]);
        let arr = DatetimeArray::new(
            vec64![1_700_000_000i64, 1_700_086_400],
            Some(mask),
            Dt::Timestamp(TimeUnit::Seconds, None),
        );

        let years = arr.year();
        assert_eq!(years.get(0), Some(2023));
        assert_eq!(years.get(1), None);
    }

    // Edge Case Tests

    #[test]
    fn test_leap_year_feb_29() {
        // 2024-02-29 00:00:00 UTC (leap year)
        let timestamp = 1_709_164_800i64;
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );

        assert_eq!(arr.year().get(0), Some(2024));
        assert_eq!(arr.month().get(0), Some(2));
        assert_eq!(arr.day().get(0), Some(29));
        assert_eq!(arr.is_leap_year().get(0), Some(true));
    }

    #[test]
    fn test_add_months_leap_year_feb_29() {
        // 2024-02-29 -> add 1 month -> 2024-03-29
        let timestamp = 1_709_164_800i64; // 2024-02-29 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        let result = arr.add_months(1).unwrap();

        assert_eq!(result.year().get(0), Some(2024));
        assert_eq!(result.month().get(0), Some(3));
        assert_eq!(result.day().get(0), Some(29));
    }

    #[test]
    fn test_add_months_leap_year_to_non_leap() {
        // 2024-02-29 -> add 12 months -> 2025-02-28 (clamped, not leap year)
        let timestamp = 1_709_164_800i64; // 2024-02-29 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        let result = arr.add_months(12).unwrap();

        assert_eq!(result.year().get(0), Some(2025));
        assert_eq!(result.month().get(0), Some(2));
        assert_eq!(result.day().get(0), Some(28)); // Clamped to Feb 28
        assert_eq!(result.is_leap_year().get(0), Some(false));
    }

    #[test]
    fn test_add_years_leap_year() {
        // 2024-02-29 -> add 4 years -> 2028-02-29 (both leap years)
        let timestamp = 1_709_164_800i64; // 2024-02-29 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        let result = arr.add_years(4).unwrap();

        assert_eq!(result.year().get(0), Some(2028));
        assert_eq!(result.month().get(0), Some(2));
        assert_eq!(result.day().get(0), Some(29));
    }

    #[test]
    fn test_add_years_leap_to_non_leap() {
        // 2024-02-29 -> add 1 year -> 2025-02-28 (clamped)
        let timestamp = 1_709_164_800i64; // 2024-02-29 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        let result = arr.add_years(1).unwrap();

        assert_eq!(result.year().get(0), Some(2025));
        assert_eq!(result.month().get(0), Some(2));
        assert_eq!(result.day().get(0), Some(28)); // Clamped
    }

    #[test]
    fn test_year_boundary_dec_31_to_jan_1() {
        // 2023-12-31 -> add 1 day -> 2024-01-01
        let timestamp = 1_704_067_199i64; // 2023-12-31 23:59:59 UTC
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        let result = arr.add_days(1).unwrap();

        assert_eq!(result.year().get(0), Some(2024));
        assert_eq!(result.month().get(0), Some(1));
        assert_eq!(result.day().get(0), Some(1));
    }

    #[test]
    fn test_month_boundary_jan_31_to_feb_1() {
        // 2024-01-31 -> add 1 day -> 2024-02-01
        let timestamp = 1_706_659_200i64; // 2024-01-31 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        let result = arr.add_days(1).unwrap();

        assert_eq!(result.year().get(0), Some(2024));
        assert_eq!(result.month().get(0), Some(2));
        assert_eq!(result.day().get(0), Some(1));
    }

    #[test]
    fn test_negative_months() {
        // 2023-03-15 -> subtract 3 months -> 2022-12-15
        let timestamp = 1_678_838_400i64; // 2023-03-15 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        let result = arr.add_months(-3).unwrap();

        assert_eq!(result.year().get(0), Some(2022));
        assert_eq!(result.month().get(0), Some(12));
        assert_eq!(result.day().get(0), Some(15));
    }

    #[test]
    fn test_negative_years() {
        // 2023-11-14 -> subtract 5 years -> 2018-11-14
        let timestamp = 1_700_000_000i64;
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        let result = arr.add_years(-5).unwrap();

        assert_eq!(result.year().get(0), Some(2018));
        assert_eq!(result.month().get(0), Some(11));
        assert_eq!(result.day().get(0), Some(14));
    }

    #[test]
    fn test_century_leap_year_2000() {
        // 2000 is a leap year (divisible by 400)
        let timestamp = 951_868_800i64; // 2000-03-01 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        let prev_day = arr.add_days(-1).unwrap();

        assert_eq!(prev_day.year().get(0), Some(2000));
        assert_eq!(prev_day.month().get(0), Some(2));
        assert_eq!(prev_day.day().get(0), Some(29)); // Feb 29 exists
    }

    #[test]
    fn test_century_non_leap_year_1900() {
        // 1900 was not a leap year (divisible by 100 but not 400)
        // March 1, 1900 minus 1 day should be Feb 28, not Feb 29
        let timestamp = -2_203_891_200i64; // 1900-03-01 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        let prev_day = arr.add_days(-1).unwrap();

        assert_eq!(prev_day.year().get(0), Some(1900));
        assert_eq!(prev_day.month().get(0), Some(2));
        assert_eq!(prev_day.day().get(0), Some(28)); // Feb 28, not 29
    }

    #[test]
    fn test_extreme_timestamp_min() {
        // Test with a very old date (year 1)
        let timestamp = -62_135_596_800i64; // 0001-01-01 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );

        assert_eq!(arr.year().get(0), Some(1));
        assert_eq!(arr.month().get(0), Some(1));
        assert_eq!(arr.day().get(0), Some(1));
    }

    #[test]
    fn test_extreme_timestamp_far_future() {
        // Test with a far future date (year 3000)
        let timestamp = 32_503_680_000i64; // 3000-01-01 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );

        assert_eq!(arr.year().get(0), Some(3000));
        assert_eq!(arr.month().get(0), Some(1));
        assert_eq!(arr.day().get(0), Some(1));
    }

    #[test]
    fn test_different_time_units_precision() {
        // Test that different time units maintain precision correctly
        let arr_sec = DatetimeArray::<i64>::from_slice(
            &[1_700_000_000],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        let arr_ms = arr_sec.cast_time_unit(TimeUnit::Milliseconds).unwrap();
        let arr_us = arr_ms.cast_time_unit(TimeUnit::Microseconds).unwrap();
        let arr_ns = arr_us.cast_time_unit(TimeUnit::Nanoseconds).unwrap();

        // Convert back to seconds
        let back_to_sec = arr_ns.cast_time_unit(TimeUnit::Seconds).unwrap();

        assert_eq!(back_to_sec.value(0), Some(1_700_000_000));
    }

    #[test]
    fn test_week_start_on_sunday() {
        // 2023-11-12 is Sunday, week should start on itself
        let timestamp = 1_699_747_200i64; // 2023-11-12 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );

        let result = arr.week();

        assert_eq!(result.day().get(0), Some(12)); // Same day (Sunday)
        assert_eq!(result.hour().get(0), Some(0));
    }

    #[test]
    fn test_week_start_on_saturday() {
        // 2023-11-18 is Saturday, week should start on Sunday 2023-11-12
        let timestamp = 1_700_265_600i64; // 2023-11-18 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(
            &[timestamp],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );

        let result = arr.week();

        assert_eq!(result.day().get(0), Some(12)); // Previous Sunday
    }

    #[test]
    fn test_add_months_varying_lengths() {
        // Test adding months across months of different lengths
        // Jan (31) -> Feb (28/29) -> Mar (31) -> Apr (30)
        let jan_31 = 1_675_123_200i64; // 2023-01-31 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(
            &[jan_31],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );

        // Jan 31 + 1 month = Feb 28 (clamped)
        let feb = arr.add_months(1).unwrap();
        assert_eq!(feb.month().get(0), Some(2));
        assert_eq!(feb.day().get(0), Some(28));

        // Feb 28 + 1 month = Mar 28
        let mar = feb.add_months(1).unwrap();
        assert_eq!(mar.month().get(0), Some(3));
        assert_eq!(mar.day().get(0), Some(28));
    }

    // Result types

    #[test]
    fn arithmetic_and_truncation_keep_the_input_type() {
        let types = [
            Dt::Timestamp(TimeUnit::Seconds, None),
            Dt::Timestamp(TimeUnit::Milliseconds, Some("UTC".to_string())),
            Dt::Date64,
            Dt::Time64(TimeUnit::Nanoseconds),
            Dt::Duration64(TimeUnit::Microseconds),
        ];
        for dtype in types {
            let arr = DatetimeArray::<i64>::from_slice(&[1_700_000_000_000], dtype.clone());
            assert_eq!(arr.add_duration(Duration::seconds(1)).unwrap().dtype, dtype);
            assert_eq!(arr.sub_duration(Duration::seconds(1)).unwrap().dtype, dtype);
            assert_eq!(arr.add_days(1).unwrap().dtype, dtype);
            assert_eq!(arr.add_months(1).unwrap().dtype, dtype);
            assert_eq!(arr.add_years(1).unwrap().dtype, dtype);
            assert_eq!(arr.truncate(TimePeriod::Day).dtype, dtype);
            assert_eq!(arr.sec().dtype, dtype);
        }
        for dtype in [
            Dt::Date32,
            Dt::Time32(TimeUnit::Seconds),
            Dt::Duration32(TimeUnit::Milliseconds),
        ] {
            let arr = DatetimeArray::<i32>::from_slice(&[19_000], dtype.clone());
            assert_eq!(arr.add_days(1).unwrap().dtype, dtype);
            assert_eq!(arr.add_months(1).unwrap().dtype, dtype);
            assert_eq!(arr.truncate(TimePeriod::Day).dtype, dtype);
        }
    }

    #[test]
    fn cast_time_unit_keeps_variant_and_timezone() {
        let cases = [
            (
                Dt::Timestamp(TimeUnit::Seconds, Some("Asia/Tokyo".to_string())),
                Dt::Timestamp(TimeUnit::Nanoseconds, Some("Asia/Tokyo".to_string())),
            ),
            (
                Dt::Timestamp(TimeUnit::Milliseconds, None),
                Dt::Timestamp(TimeUnit::Nanoseconds, None),
            ),
            (
                Dt::Time64(TimeUnit::Microseconds),
                Dt::Time64(TimeUnit::Nanoseconds),
            ),
            (
                Dt::Duration64(TimeUnit::Microseconds),
                Dt::Duration64(TimeUnit::Nanoseconds),
            ),
            (
                Dt::Date64,
                Dt::Timestamp(TimeUnit::Nanoseconds, None),
            ),
        ];
        for (from, to) in cases {
            let arr = DatetimeArray::<i64>::from_slice(&[1_000], from);
            let cast = arr.cast_time_unit(TimeUnit::Nanoseconds).unwrap();
            assert_eq!(cast.dtype, to);
            assert_eq!(cast.time_unit(), TimeUnit::Nanoseconds);
        }

        let date64 = DatetimeArray::<i64>::from_slice(&[1_000], Dt::Date64);
        assert_eq!(
            date64.cast_time_unit(TimeUnit::Milliseconds).unwrap().dtype,
            Dt::Date64
        );

        let time32 =
            DatetimeArray::<i32>::from_slice(&[1], Dt::Time32(TimeUnit::Seconds));
        assert_eq!(
            time32.cast_time_unit(TimeUnit::Milliseconds).unwrap().dtype,
            Dt::Time32(TimeUnit::Milliseconds)
        );
        let duration32 =
            DatetimeArray::<i32>::from_slice(&[1], Dt::Duration32(TimeUnit::Seconds));
        assert_eq!(
            duration32
                .cast_time_unit(TimeUnit::Milliseconds)
                .unwrap()
                .dtype,
            Dt::Duration32(TimeUnit::Milliseconds)
        );

        let date32 = DatetimeArray::<i32>::from_slice(&[19_000], Dt::Date32);
        assert_eq!(
            date32.cast_time_unit(TimeUnit::Days).unwrap().dtype,
            Dt::Date32
        );
        assert!(date32.cast_time_unit(TimeUnit::Seconds).is_err());
    }
}

#[cfg(all(test, feature = "time_zones"))]
mod time_zone_tests {
    use std::sync::Arc;

    use super::*;
    use crate::kernels::datetime::{hour_into, minute_into};
    use crate::{Array, TemporalArray, TemporalArrayV};

    // The tables are from Python's `zoneinfo`, which reads the IANA Time Zone
    // Database. They cover daylight saving in both hemispheres, offsets of 30 and
    // 45 minutes, daylight saving of 30 minutes, transitions at midnight, the
    // seconds either side of each transition of 2026, and historical rules.
    //
    // Truncation floors the wall time and reads it with the offset in force at
    // the input. Where that offset is not in force at the start of the period,
    // the result is the first instant at which the wall clock reads the start of
    // the period or later. Month arithmetic keeps the wall time: wall times that
    // occur twice use the earlier instant, and skipped wall times use the offset
    // in force before the transition.

    /// Time zone, instant in seconds, then year, month, day, hour, minute, second,
    /// weekday from 1 for Sunday, day of year, ISO week, quarter, week of year, and
    /// whether the year is a leap year.
    const COMPONENTS: &[(&str, i64, [i32; 11], bool)] = &[
        ("Australia/Sydney", 1767259800, [2026, 1, 1, 20, 30, 0, 5, 1, 1, 1, 0], false),
        ("Australia/Sydney", 1767277800, [2026, 1, 2, 1, 30, 0, 6, 2, 1, 1, 0], false),
        ("Australia/Sydney", 1775313000, [2026, 4, 5, 1, 30, 0, 1, 95, 14, 2, 14], false),
        ("Australia/Sydney", 1775318399, [2026, 4, 5, 2, 59, 59, 1, 95, 14, 2, 14], false),
        ("Australia/Sydney", 1775318400, [2026, 4, 5, 2, 0, 0, 1, 95, 14, 2, 14], false),
        ("Australia/Sydney", 1775320200, [2026, 4, 5, 2, 30, 0, 1, 95, 14, 2, 14], false),
        ("Australia/Sydney", 1775323800, [2026, 4, 5, 3, 30, 0, 1, 95, 14, 2, 14], false),
        ("Australia/Sydney", 1782898200, [2026, 7, 1, 19, 30, 0, 4, 182, 27, 3, 26], false),
        ("Australia/Sydney", 1791037800, [2026, 10, 4, 0, 30, 0, 1, 277, 40, 4, 40], false),
        ("Australia/Sydney", 1791043199, [2026, 10, 4, 1, 59, 59, 1, 277, 40, 4, 40], false),
        ("Australia/Sydney", 1791043200, [2026, 10, 4, 3, 0, 0, 1, 277, 40, 4, 40], false),
        ("Australia/Sydney", 1791045000, [2026, 10, 4, 3, 30, 0, 1, 277, 40, 4, 40], false),
        ("Australia/Sydney", 1791048600, [2026, 10, 4, 4, 30, 0, 1, 277, 40, 4, 40], false),
        ("America/New_York", 1142856000, [2006, 3, 20, 7, 0, 0, 2, 79, 12, 1, 12], false),
        ("America/New_York", 1767259800, [2026, 1, 1, 4, 30, 0, 5, 1, 1, 1, 0], false),
        ("America/New_York", 1767277800, [2026, 1, 1, 9, 30, 0, 5, 1, 1, 1, 0], false),
        ("America/New_York", 1772947800, [2026, 3, 8, 0, 30, 0, 1, 67, 10, 1, 10], false),
        ("America/New_York", 1772953199, [2026, 3, 8, 1, 59, 59, 1, 67, 10, 1, 10], false),
        ("America/New_York", 1772953200, [2026, 3, 8, 3, 0, 0, 1, 67, 10, 1, 10], false),
        ("America/New_York", 1772955000, [2026, 3, 8, 3, 30, 0, 1, 67, 10, 1, 10], false),
        ("America/New_York", 1772958600, [2026, 3, 8, 4, 30, 0, 1, 67, 10, 1, 10], false),
        ("America/New_York", 1782898200, [2026, 7, 1, 5, 30, 0, 4, 182, 27, 3, 26], false),
        ("America/New_York", 1793507400, [2026, 11, 1, 0, 30, 0, 1, 305, 44, 4, 44], false),
        ("America/New_York", 1793512799, [2026, 11, 1, 1, 59, 59, 1, 305, 44, 4, 44], false),
        ("America/New_York", 1793512800, [2026, 11, 1, 1, 0, 0, 1, 305, 44, 4, 44], false),
        ("America/New_York", 1793514600, [2026, 11, 1, 1, 30, 0, 1, 305, 44, 4, 44], false),
        ("America/New_York", 1793518200, [2026, 11, 1, 2, 30, 0, 1, 305, 44, 4, 44], false),
        ("Europe/London", 0, [1970, 1, 1, 1, 0, 0, 5, 1, 1, 1, 0], false),
        ("Europe/London", 1767259800, [2026, 1, 1, 9, 30, 0, 5, 1, 1, 1, 0], false),
        ("Europe/London", 1767277800, [2026, 1, 1, 14, 30, 0, 5, 1, 1, 1, 0], false),
        ("Europe/London", 1774740600, [2026, 3, 28, 23, 30, 0, 7, 87, 13, 1, 12], false),
        ("Europe/London", 1774745999, [2026, 3, 29, 0, 59, 59, 1, 88, 13, 1, 13], false),
        ("Europe/London", 1774746000, [2026, 3, 29, 2, 0, 0, 1, 88, 13, 1, 13], false),
        ("Europe/London", 1774747800, [2026, 3, 29, 2, 30, 0, 1, 88, 13, 1, 13], false),
        ("Europe/London", 1774751400, [2026, 3, 29, 3, 30, 0, 1, 88, 13, 1, 13], false),
        ("Europe/London", 1782898200, [2026, 7, 1, 10, 30, 0, 4, 182, 27, 3, 26], false),
        ("Europe/London", 1792884600, [2026, 10, 25, 0, 30, 0, 1, 298, 43, 4, 43], false),
        ("Europe/London", 1792889999, [2026, 10, 25, 1, 59, 59, 1, 298, 43, 4, 43], false),
        ("Europe/London", 1792890000, [2026, 10, 25, 1, 0, 0, 1, 298, 43, 4, 43], false),
        ("Europe/London", 1792891800, [2026, 10, 25, 1, 30, 0, 1, 298, 43, 4, 43], false),
        ("Europe/London", 1792895400, [2026, 10, 25, 2, 30, 0, 1, 298, 43, 4, 43], false),
        ("Australia/Lord_Howe", 1767259800, [2026, 1, 1, 20, 30, 0, 5, 1, 1, 1, 0], false),
        ("Australia/Lord_Howe", 1767277800, [2026, 1, 2, 1, 30, 0, 6, 2, 1, 1, 0], false),
        ("Australia/Lord_Howe", 1775309400, [2026, 4, 5, 0, 30, 0, 1, 95, 14, 2, 14], false),
        ("Australia/Lord_Howe", 1775314799, [2026, 4, 5, 1, 59, 59, 1, 95, 14, 2, 14], false),
        ("Australia/Lord_Howe", 1775314800, [2026, 4, 5, 1, 30, 0, 1, 95, 14, 2, 14], false),
        ("Australia/Lord_Howe", 1775316600, [2026, 4, 5, 2, 0, 0, 1, 95, 14, 2, 14], false),
        ("Australia/Lord_Howe", 1775320200, [2026, 4, 5, 3, 0, 0, 1, 95, 14, 2, 14], false),
        ("Australia/Lord_Howe", 1782898200, [2026, 7, 1, 20, 0, 0, 4, 182, 27, 3, 26], false),
        ("Australia/Lord_Howe", 1791036000, [2026, 10, 4, 0, 30, 0, 1, 277, 40, 4, 40], false),
        ("Australia/Lord_Howe", 1791041399, [2026, 10, 4, 1, 59, 59, 1, 277, 40, 4, 40], false),
        ("Australia/Lord_Howe", 1791041400, [2026, 10, 4, 2, 30, 0, 1, 277, 40, 4, 40], false),
        ("Australia/Lord_Howe", 1791043200, [2026, 10, 4, 3, 0, 0, 1, 277, 40, 4, 40], false),
        ("Australia/Lord_Howe", 1791046800, [2026, 10, 4, 4, 0, 0, 1, 277, 40, 4, 40], false),
        ("Asia/Kolkata", 1767259800, [2026, 1, 1, 15, 0, 0, 5, 1, 1, 1, 0], false),
        ("Asia/Kolkata", 1767277800, [2026, 1, 1, 20, 0, 0, 5, 1, 1, 1, 0], false),
        ("Asia/Kolkata", 1782898200, [2026, 7, 1, 15, 0, 0, 4, 182, 27, 3, 26], false),
        ("Asia/Kathmandu", 1767259800, [2026, 1, 1, 15, 15, 0, 5, 1, 1, 1, 0], false),
        ("Asia/Kathmandu", 1767277800, [2026, 1, 1, 20, 15, 0, 5, 1, 1, 1, 0], false),
        ("Asia/Kathmandu", 1782898200, [2026, 7, 1, 15, 15, 0, 4, 182, 27, 3, 26], false),
        ("America/Havana", 1767259800, [2026, 1, 1, 4, 30, 0, 5, 1, 1, 1, 0], false),
        ("America/Havana", 1767277800, [2026, 1, 1, 9, 30, 0, 5, 1, 1, 1, 0], false),
        ("America/Havana", 1772940600, [2026, 3, 7, 22, 30, 0, 7, 66, 10, 1, 9], false),
        ("America/Havana", 1772945999, [2026, 3, 7, 23, 59, 59, 7, 66, 10, 1, 9], false),
        ("America/Havana", 1772946000, [2026, 3, 8, 1, 0, 0, 1, 67, 10, 1, 10], false),
        ("America/Havana", 1772947800, [2026, 3, 8, 1, 30, 0, 1, 67, 10, 1, 10], false),
        ("America/Havana", 1772951400, [2026, 3, 8, 2, 30, 0, 1, 67, 10, 1, 10], false),
        ("America/Havana", 1782898200, [2026, 7, 1, 5, 30, 0, 4, 182, 27, 3, 26], false),
        ("America/Havana", 1793503800, [2026, 10, 31, 23, 30, 0, 7, 304, 44, 4, 43], false),
        ("America/Havana", 1793509199, [2026, 11, 1, 0, 59, 59, 1, 305, 44, 4, 44], false),
        ("America/Havana", 1793509200, [2026, 11, 1, 0, 0, 0, 1, 305, 44, 4, 44], false),
        ("America/Havana", 1793511000, [2026, 11, 1, 0, 30, 0, 1, 305, 44, 4, 44], false),
        ("America/Havana", 1793514600, [2026, 11, 1, 1, 30, 0, 1, 305, 44, 4, 44], false),
        ("Asia/Beirut", 1767259800, [2026, 1, 1, 11, 30, 0, 5, 1, 1, 1, 0], false),
        ("Asia/Beirut", 1767277800, [2026, 1, 1, 16, 30, 0, 5, 1, 1, 1, 0], false),
        ("Asia/Beirut", 1774729800, [2026, 3, 28, 22, 30, 0, 7, 87, 13, 1, 12], false),
        ("Asia/Beirut", 1774735199, [2026, 3, 28, 23, 59, 59, 7, 87, 13, 1, 12], false),
        ("Asia/Beirut", 1774735200, [2026, 3, 29, 1, 0, 0, 1, 88, 13, 1, 13], false),
        ("Asia/Beirut", 1774737000, [2026, 3, 29, 1, 30, 0, 1, 88, 13, 1, 13], false),
        ("Asia/Beirut", 1774740600, [2026, 3, 29, 2, 30, 0, 1, 88, 13, 1, 13], false),
        ("Asia/Beirut", 1782898200, [2026, 7, 1, 12, 30, 0, 4, 182, 27, 3, 26], false),
        ("Asia/Beirut", 1792870200, [2026, 10, 24, 22, 30, 0, 7, 297, 43, 4, 42], false),
        ("Asia/Beirut", 1792875599, [2026, 10, 24, 23, 59, 59, 7, 297, 43, 4, 42], false),
        ("Asia/Beirut", 1792875600, [2026, 10, 24, 23, 0, 0, 7, 297, 43, 4, 42], false),
        ("Asia/Beirut", 1792877400, [2026, 10, 24, 23, 30, 0, 7, 297, 43, 4, 42], false),
        ("Asia/Beirut", 1792881000, [2026, 10, 25, 0, 30, 0, 1, 298, 43, 4, 43], false),
        ("America/Santiago", 1767259800, [2026, 1, 1, 6, 30, 0, 5, 1, 1, 1, 0], false),
        ("America/Santiago", 1767277800, [2026, 1, 1, 11, 30, 0, 5, 1, 1, 1, 0], false),
        ("America/Santiago", 1775352600, [2026, 4, 4, 22, 30, 0, 7, 94, 14, 2, 13], false),
        ("America/Santiago", 1775357999, [2026, 4, 4, 23, 59, 59, 7, 94, 14, 2, 13], false),
        ("America/Santiago", 1775358000, [2026, 4, 4, 23, 0, 0, 7, 94, 14, 2, 13], false),
        ("America/Santiago", 1775359800, [2026, 4, 4, 23, 30, 0, 7, 94, 14, 2, 13], false),
        ("America/Santiago", 1775363400, [2026, 4, 5, 0, 30, 0, 1, 95, 14, 2, 14], false),
        ("America/Santiago", 1782898200, [2026, 7, 1, 5, 30, 0, 4, 182, 27, 3, 26], false),
        ("America/Santiago", 1788661800, [2026, 9, 5, 22, 30, 0, 7, 248, 36, 3, 35], false),
        ("America/Santiago", 1788667199, [2026, 9, 5, 23, 59, 59, 7, 248, 36, 3, 35], false),
        ("America/Santiago", 1788667200, [2026, 9, 6, 1, 0, 0, 1, 249, 36, 3, 36], false),
        ("America/Santiago", 1788669000, [2026, 9, 6, 1, 30, 0, 1, 249, 36, 3, 36], false),
        ("America/Santiago", 1788672600, [2026, 9, 6, 2, 30, 0, 1, 249, 36, 3, 36], false),
    ];

    /// Time zone, period, instant in seconds, and the instant of the start of the period.
    const TRUNCATIONS: &[(&str, &str, i64, i64)] = &[
        ("Australia/Sydney", "year", 1767259800, 1767186000),
        ("Australia/Sydney", "quarter", 1767259800, 1767186000),
        ("Australia/Sydney", "month", 1767259800, 1767186000),
        ("Australia/Sydney", "week", 1767259800, 1766840400),
        ("Australia/Sydney", "day", 1767259800, 1767186000),
        ("Australia/Sydney", "hour", 1767259800, 1767258000),
        ("Australia/Sydney", "minute", 1767259800, 1767259800),
        ("Australia/Sydney", "year", 1767277800, 1767186000),
        ("Australia/Sydney", "quarter", 1767277800, 1767186000),
        ("Australia/Sydney", "month", 1767277800, 1767186000),
        ("Australia/Sydney", "week", 1767277800, 1766840400),
        ("Australia/Sydney", "day", 1767277800, 1767272400),
        ("Australia/Sydney", "hour", 1767277800, 1767276000),
        ("Australia/Sydney", "minute", 1767277800, 1767277800),
        ("Australia/Sydney", "year", 1775313000, 1767186000),
        ("Australia/Sydney", "quarter", 1775313000, 1774962000),
        ("Australia/Sydney", "month", 1775313000, 1774962000),
        ("Australia/Sydney", "week", 1775313000, 1775307600),
        ("Australia/Sydney", "day", 1775313000, 1775307600),
        ("Australia/Sydney", "hour", 1775313000, 1775311200),
        ("Australia/Sydney", "minute", 1775313000, 1775313000),
        ("Australia/Sydney", "year", 1775318399, 1767186000),
        ("Australia/Sydney", "quarter", 1775318399, 1774962000),
        ("Australia/Sydney", "month", 1775318399, 1774962000),
        ("Australia/Sydney", "week", 1775318399, 1775307600),
        ("Australia/Sydney", "day", 1775318399, 1775307600),
        ("Australia/Sydney", "hour", 1775318399, 1775314800),
        ("Australia/Sydney", "minute", 1775318399, 1775318340),
        ("Australia/Sydney", "year", 1775318400, 1767186000),
        ("Australia/Sydney", "quarter", 1775318400, 1774962000),
        ("Australia/Sydney", "month", 1775318400, 1774962000),
        ("Australia/Sydney", "week", 1775318400, 1775307600),
        ("Australia/Sydney", "day", 1775318400, 1775307600),
        ("Australia/Sydney", "hour", 1775318400, 1775318400),
        ("Australia/Sydney", "minute", 1775318400, 1775318400),
        ("Australia/Sydney", "year", 1775320200, 1767186000),
        ("Australia/Sydney", "quarter", 1775320200, 1774962000),
        ("Australia/Sydney", "month", 1775320200, 1774962000),
        ("Australia/Sydney", "week", 1775320200, 1775307600),
        ("Australia/Sydney", "day", 1775320200, 1775307600),
        ("Australia/Sydney", "hour", 1775320200, 1775318400),
        ("Australia/Sydney", "minute", 1775320200, 1775320200),
        ("Australia/Sydney", "year", 1775323800, 1767186000),
        ("Australia/Sydney", "quarter", 1775323800, 1774962000),
        ("Australia/Sydney", "month", 1775323800, 1774962000),
        ("Australia/Sydney", "week", 1775323800, 1775307600),
        ("Australia/Sydney", "day", 1775323800, 1775307600),
        ("Australia/Sydney", "hour", 1775323800, 1775322000),
        ("Australia/Sydney", "minute", 1775323800, 1775323800),
        ("Australia/Sydney", "year", 1782898200, 1767186000),
        ("Australia/Sydney", "quarter", 1782898200, 1782828000),
        ("Australia/Sydney", "month", 1782898200, 1782828000),
        ("Australia/Sydney", "week", 1782898200, 1782568800),
        ("Australia/Sydney", "day", 1782898200, 1782828000),
        ("Australia/Sydney", "hour", 1782898200, 1782896400),
        ("Australia/Sydney", "minute", 1782898200, 1782898200),
        ("Australia/Sydney", "year", 1791037800, 1767186000),
        ("Australia/Sydney", "quarter", 1791037800, 1790776800),
        ("Australia/Sydney", "month", 1791037800, 1790776800),
        ("Australia/Sydney", "week", 1791037800, 1791036000),
        ("Australia/Sydney", "day", 1791037800, 1791036000),
        ("Australia/Sydney", "hour", 1791037800, 1791036000),
        ("Australia/Sydney", "minute", 1791037800, 1791037800),
        ("Australia/Sydney", "year", 1791043199, 1767186000),
        ("Australia/Sydney", "quarter", 1791043199, 1790776800),
        ("Australia/Sydney", "month", 1791043199, 1790776800),
        ("Australia/Sydney", "week", 1791043199, 1791036000),
        ("Australia/Sydney", "day", 1791043199, 1791036000),
        ("Australia/Sydney", "hour", 1791043199, 1791039600),
        ("Australia/Sydney", "minute", 1791043199, 1791043140),
        ("Australia/Sydney", "year", 1791043200, 1767186000),
        ("Australia/Sydney", "quarter", 1791043200, 1790776800),
        ("Australia/Sydney", "month", 1791043200, 1790776800),
        ("Australia/Sydney", "week", 1791043200, 1791036000),
        ("Australia/Sydney", "day", 1791043200, 1791036000),
        ("Australia/Sydney", "hour", 1791043200, 1791043200),
        ("Australia/Sydney", "minute", 1791043200, 1791043200),
        ("Australia/Sydney", "year", 1791045000, 1767186000),
        ("Australia/Sydney", "quarter", 1791045000, 1790776800),
        ("Australia/Sydney", "month", 1791045000, 1790776800),
        ("Australia/Sydney", "week", 1791045000, 1791036000),
        ("Australia/Sydney", "day", 1791045000, 1791036000),
        ("Australia/Sydney", "hour", 1791045000, 1791043200),
        ("Australia/Sydney", "minute", 1791045000, 1791045000),
        ("Australia/Sydney", "year", 1791048600, 1767186000),
        ("Australia/Sydney", "quarter", 1791048600, 1790776800),
        ("Australia/Sydney", "month", 1791048600, 1790776800),
        ("Australia/Sydney", "week", 1791048600, 1791036000),
        ("Australia/Sydney", "day", 1791048600, 1791036000),
        ("Australia/Sydney", "hour", 1791048600, 1791046800),
        ("Australia/Sydney", "minute", 1791048600, 1791048600),
        ("America/New_York", "year", 1142856000, 1136091600),
        ("America/New_York", "quarter", 1142856000, 1136091600),
        ("America/New_York", "month", 1142856000, 1141189200),
        ("America/New_York", "week", 1142856000, 1142744400),
        ("America/New_York", "day", 1142856000, 1142830800),
        ("America/New_York", "hour", 1142856000, 1142856000),
        ("America/New_York", "minute", 1142856000, 1142856000),
        ("America/New_York", "year", 1767259800, 1767243600),
        ("America/New_York", "quarter", 1767259800, 1767243600),
        ("America/New_York", "month", 1767259800, 1767243600),
        ("America/New_York", "week", 1767259800, 1766898000),
        ("America/New_York", "day", 1767259800, 1767243600),
        ("America/New_York", "hour", 1767259800, 1767258000),
        ("America/New_York", "minute", 1767259800, 1767259800),
        ("America/New_York", "year", 1767277800, 1767243600),
        ("America/New_York", "quarter", 1767277800, 1767243600),
        ("America/New_York", "month", 1767277800, 1767243600),
        ("America/New_York", "week", 1767277800, 1766898000),
        ("America/New_York", "day", 1767277800, 1767243600),
        ("America/New_York", "hour", 1767277800, 1767276000),
        ("America/New_York", "minute", 1767277800, 1767277800),
        ("America/New_York", "year", 1772947800, 1767243600),
        ("America/New_York", "quarter", 1772947800, 1767243600),
        ("America/New_York", "month", 1772947800, 1772341200),
        ("America/New_York", "week", 1772947800, 1772946000),
        ("America/New_York", "day", 1772947800, 1772946000),
        ("America/New_York", "hour", 1772947800, 1772946000),
        ("America/New_York", "minute", 1772947800, 1772947800),
        ("America/New_York", "year", 1772953199, 1767243600),
        ("America/New_York", "quarter", 1772953199, 1767243600),
        ("America/New_York", "month", 1772953199, 1772341200),
        ("America/New_York", "week", 1772953199, 1772946000),
        ("America/New_York", "day", 1772953199, 1772946000),
        ("America/New_York", "hour", 1772953199, 1772949600),
        ("America/New_York", "minute", 1772953199, 1772953140),
        ("America/New_York", "year", 1772953200, 1767243600),
        ("America/New_York", "quarter", 1772953200, 1767243600),
        ("America/New_York", "month", 1772953200, 1772341200),
        ("America/New_York", "week", 1772953200, 1772946000),
        ("America/New_York", "day", 1772953200, 1772946000),
        ("America/New_York", "hour", 1772953200, 1772953200),
        ("America/New_York", "minute", 1772953200, 1772953200),
        ("America/New_York", "year", 1772955000, 1767243600),
        ("America/New_York", "quarter", 1772955000, 1767243600),
        ("America/New_York", "month", 1772955000, 1772341200),
        ("America/New_York", "week", 1772955000, 1772946000),
        ("America/New_York", "day", 1772955000, 1772946000),
        ("America/New_York", "hour", 1772955000, 1772953200),
        ("America/New_York", "minute", 1772955000, 1772955000),
        ("America/New_York", "year", 1772958600, 1767243600),
        ("America/New_York", "quarter", 1772958600, 1767243600),
        ("America/New_York", "month", 1772958600, 1772341200),
        ("America/New_York", "week", 1772958600, 1772946000),
        ("America/New_York", "day", 1772958600, 1772946000),
        ("America/New_York", "hour", 1772958600, 1772956800),
        ("America/New_York", "minute", 1772958600, 1772958600),
        ("America/New_York", "year", 1782898200, 1767243600),
        ("America/New_York", "quarter", 1782898200, 1782878400),
        ("America/New_York", "month", 1782898200, 1782878400),
        ("America/New_York", "week", 1782898200, 1782619200),
        ("America/New_York", "day", 1782898200, 1782878400),
        ("America/New_York", "hour", 1782898200, 1782896400),
        ("America/New_York", "minute", 1782898200, 1782898200),
        ("America/New_York", "year", 1793507400, 1767243600),
        ("America/New_York", "quarter", 1793507400, 1790827200),
        ("America/New_York", "month", 1793507400, 1793505600),
        ("America/New_York", "week", 1793507400, 1793505600),
        ("America/New_York", "day", 1793507400, 1793505600),
        ("America/New_York", "hour", 1793507400, 1793505600),
        ("America/New_York", "minute", 1793507400, 1793507400),
        ("America/New_York", "year", 1793512799, 1767243600),
        ("America/New_York", "quarter", 1793512799, 1790827200),
        ("America/New_York", "month", 1793512799, 1793505600),
        ("America/New_York", "week", 1793512799, 1793505600),
        ("America/New_York", "day", 1793512799, 1793505600),
        ("America/New_York", "hour", 1793512799, 1793509200),
        ("America/New_York", "minute", 1793512799, 1793512740),
        ("America/New_York", "year", 1793512800, 1767243600),
        ("America/New_York", "quarter", 1793512800, 1790827200),
        ("America/New_York", "month", 1793512800, 1793505600),
        ("America/New_York", "week", 1793512800, 1793505600),
        ("America/New_York", "day", 1793512800, 1793505600),
        ("America/New_York", "hour", 1793512800, 1793512800),
        ("America/New_York", "minute", 1793512800, 1793512800),
        ("America/New_York", "year", 1793514600, 1767243600),
        ("America/New_York", "quarter", 1793514600, 1790827200),
        ("America/New_York", "month", 1793514600, 1793505600),
        ("America/New_York", "week", 1793514600, 1793505600),
        ("America/New_York", "day", 1793514600, 1793505600),
        ("America/New_York", "hour", 1793514600, 1793512800),
        ("America/New_York", "minute", 1793514600, 1793514600),
        ("America/New_York", "year", 1793518200, 1767243600),
        ("America/New_York", "quarter", 1793518200, 1790827200),
        ("America/New_York", "month", 1793518200, 1793505600),
        ("America/New_York", "week", 1793518200, 1793505600),
        ("America/New_York", "day", 1793518200, 1793505600),
        ("America/New_York", "hour", 1793518200, 1793516400),
        ("America/New_York", "minute", 1793518200, 1793518200),
        ("Europe/London", "year", 0, -3600),
        ("Europe/London", "quarter", 0, -3600),
        ("Europe/London", "month", 0, -3600),
        ("Europe/London", "week", 0, -349200),
        ("Europe/London", "day", 0, -3600),
        ("Europe/London", "hour", 0, 0),
        ("Europe/London", "minute", 0, 0),
        ("Europe/London", "year", 1767259800, 1767225600),
        ("Europe/London", "quarter", 1767259800, 1767225600),
        ("Europe/London", "month", 1767259800, 1767225600),
        ("Europe/London", "week", 1767259800, 1766880000),
        ("Europe/London", "day", 1767259800, 1767225600),
        ("Europe/London", "hour", 1767259800, 1767258000),
        ("Europe/London", "minute", 1767259800, 1767259800),
        ("Europe/London", "year", 1767277800, 1767225600),
        ("Europe/London", "quarter", 1767277800, 1767225600),
        ("Europe/London", "month", 1767277800, 1767225600),
        ("Europe/London", "week", 1767277800, 1766880000),
        ("Europe/London", "day", 1767277800, 1767225600),
        ("Europe/London", "hour", 1767277800, 1767276000),
        ("Europe/London", "minute", 1767277800, 1767277800),
        ("Europe/London", "year", 1774740600, 1767225600),
        ("Europe/London", "quarter", 1774740600, 1767225600),
        ("Europe/London", "month", 1774740600, 1772323200),
        ("Europe/London", "week", 1774740600, 1774137600),
        ("Europe/London", "day", 1774740600, 1774656000),
        ("Europe/London", "hour", 1774740600, 1774738800),
        ("Europe/London", "minute", 1774740600, 1774740600),
        ("Europe/London", "year", 1774745999, 1767225600),
        ("Europe/London", "quarter", 1774745999, 1767225600),
        ("Europe/London", "month", 1774745999, 1772323200),
        ("Europe/London", "week", 1774745999, 1774742400),
        ("Europe/London", "day", 1774745999, 1774742400),
        ("Europe/London", "hour", 1774745999, 1774742400),
        ("Europe/London", "minute", 1774745999, 1774745940),
        ("Europe/London", "year", 1774746000, 1767225600),
        ("Europe/London", "quarter", 1774746000, 1767225600),
        ("Europe/London", "month", 1774746000, 1772323200),
        ("Europe/London", "week", 1774746000, 1774742400),
        ("Europe/London", "day", 1774746000, 1774742400),
        ("Europe/London", "hour", 1774746000, 1774746000),
        ("Europe/London", "minute", 1774746000, 1774746000),
        ("Europe/London", "year", 1774747800, 1767225600),
        ("Europe/London", "quarter", 1774747800, 1767225600),
        ("Europe/London", "month", 1774747800, 1772323200),
        ("Europe/London", "week", 1774747800, 1774742400),
        ("Europe/London", "day", 1774747800, 1774742400),
        ("Europe/London", "hour", 1774747800, 1774746000),
        ("Europe/London", "minute", 1774747800, 1774747800),
        ("Europe/London", "year", 1774751400, 1767225600),
        ("Europe/London", "quarter", 1774751400, 1767225600),
        ("Europe/London", "month", 1774751400, 1772323200),
        ("Europe/London", "week", 1774751400, 1774742400),
        ("Europe/London", "day", 1774751400, 1774742400),
        ("Europe/London", "hour", 1774751400, 1774749600),
        ("Europe/London", "minute", 1774751400, 1774751400),
        ("Europe/London", "year", 1782898200, 1767225600),
        ("Europe/London", "quarter", 1782898200, 1782860400),
        ("Europe/London", "month", 1782898200, 1782860400),
        ("Europe/London", "week", 1782898200, 1782601200),
        ("Europe/London", "day", 1782898200, 1782860400),
        ("Europe/London", "hour", 1782898200, 1782896400),
        ("Europe/London", "minute", 1782898200, 1782898200),
        ("Europe/London", "year", 1792884600, 1767225600),
        ("Europe/London", "quarter", 1792884600, 1790809200),
        ("Europe/London", "month", 1792884600, 1790809200),
        ("Europe/London", "week", 1792884600, 1792882800),
        ("Europe/London", "day", 1792884600, 1792882800),
        ("Europe/London", "hour", 1792884600, 1792882800),
        ("Europe/London", "minute", 1792884600, 1792884600),
        ("Europe/London", "year", 1792889999, 1767225600),
        ("Europe/London", "quarter", 1792889999, 1790809200),
        ("Europe/London", "month", 1792889999, 1790809200),
        ("Europe/London", "week", 1792889999, 1792882800),
        ("Europe/London", "day", 1792889999, 1792882800),
        ("Europe/London", "hour", 1792889999, 1792886400),
        ("Europe/London", "minute", 1792889999, 1792889940),
        ("Europe/London", "year", 1792890000, 1767225600),
        ("Europe/London", "quarter", 1792890000, 1790809200),
        ("Europe/London", "month", 1792890000, 1790809200),
        ("Europe/London", "week", 1792890000, 1792882800),
        ("Europe/London", "day", 1792890000, 1792882800),
        ("Europe/London", "hour", 1792890000, 1792890000),
        ("Europe/London", "minute", 1792890000, 1792890000),
        ("Europe/London", "year", 1792891800, 1767225600),
        ("Europe/London", "quarter", 1792891800, 1790809200),
        ("Europe/London", "month", 1792891800, 1790809200),
        ("Europe/London", "week", 1792891800, 1792882800),
        ("Europe/London", "day", 1792891800, 1792882800),
        ("Europe/London", "hour", 1792891800, 1792890000),
        ("Europe/London", "minute", 1792891800, 1792891800),
        ("Europe/London", "year", 1792895400, 1767225600),
        ("Europe/London", "quarter", 1792895400, 1790809200),
        ("Europe/London", "month", 1792895400, 1790809200),
        ("Europe/London", "week", 1792895400, 1792882800),
        ("Europe/London", "day", 1792895400, 1792882800),
        ("Europe/London", "hour", 1792895400, 1792893600),
        ("Europe/London", "minute", 1792895400, 1792895400),
        ("Australia/Lord_Howe", "year", 1767259800, 1767186000),
        ("Australia/Lord_Howe", "quarter", 1767259800, 1767186000),
        ("Australia/Lord_Howe", "month", 1767259800, 1767186000),
        ("Australia/Lord_Howe", "week", 1767259800, 1766840400),
        ("Australia/Lord_Howe", "day", 1767259800, 1767186000),
        ("Australia/Lord_Howe", "hour", 1767259800, 1767258000),
        ("Australia/Lord_Howe", "minute", 1767259800, 1767259800),
        ("Australia/Lord_Howe", "year", 1767277800, 1767186000),
        ("Australia/Lord_Howe", "quarter", 1767277800, 1767186000),
        ("Australia/Lord_Howe", "month", 1767277800, 1767186000),
        ("Australia/Lord_Howe", "week", 1767277800, 1766840400),
        ("Australia/Lord_Howe", "day", 1767277800, 1767272400),
        ("Australia/Lord_Howe", "hour", 1767277800, 1767276000),
        ("Australia/Lord_Howe", "minute", 1767277800, 1767277800),
        ("Australia/Lord_Howe", "year", 1775309400, 1767186000),
        ("Australia/Lord_Howe", "quarter", 1775309400, 1774962000),
        ("Australia/Lord_Howe", "month", 1775309400, 1774962000),
        ("Australia/Lord_Howe", "week", 1775309400, 1775307600),
        ("Australia/Lord_Howe", "day", 1775309400, 1775307600),
        ("Australia/Lord_Howe", "hour", 1775309400, 1775307600),
        ("Australia/Lord_Howe", "minute", 1775309400, 1775309400),
        ("Australia/Lord_Howe", "year", 1775314799, 1767186000),
        ("Australia/Lord_Howe", "quarter", 1775314799, 1774962000),
        ("Australia/Lord_Howe", "month", 1775314799, 1774962000),
        ("Australia/Lord_Howe", "week", 1775314799, 1775307600),
        ("Australia/Lord_Howe", "day", 1775314799, 1775307600),
        ("Australia/Lord_Howe", "hour", 1775314799, 1775311200),
        ("Australia/Lord_Howe", "minute", 1775314799, 1775314740),
        ("Australia/Lord_Howe", "year", 1775314800, 1767186000),
        ("Australia/Lord_Howe", "quarter", 1775314800, 1774962000),
        ("Australia/Lord_Howe", "month", 1775314800, 1774962000),
        ("Australia/Lord_Howe", "week", 1775314800, 1775307600),
        ("Australia/Lord_Howe", "day", 1775314800, 1775307600),
        ("Australia/Lord_Howe", "hour", 1775314800, 1775311200),
        ("Australia/Lord_Howe", "minute", 1775314800, 1775314800),
        ("Australia/Lord_Howe", "year", 1775316600, 1767186000),
        ("Australia/Lord_Howe", "quarter", 1775316600, 1774962000),
        ("Australia/Lord_Howe", "month", 1775316600, 1774962000),
        ("Australia/Lord_Howe", "week", 1775316600, 1775307600),
        ("Australia/Lord_Howe", "day", 1775316600, 1775307600),
        ("Australia/Lord_Howe", "hour", 1775316600, 1775316600),
        ("Australia/Lord_Howe", "minute", 1775316600, 1775316600),
        ("Australia/Lord_Howe", "year", 1775320200, 1767186000),
        ("Australia/Lord_Howe", "quarter", 1775320200, 1774962000),
        ("Australia/Lord_Howe", "month", 1775320200, 1774962000),
        ("Australia/Lord_Howe", "week", 1775320200, 1775307600),
        ("Australia/Lord_Howe", "day", 1775320200, 1775307600),
        ("Australia/Lord_Howe", "hour", 1775320200, 1775320200),
        ("Australia/Lord_Howe", "minute", 1775320200, 1775320200),
        ("Australia/Lord_Howe", "year", 1782898200, 1767186000),
        ("Australia/Lord_Howe", "quarter", 1782898200, 1782826200),
        ("Australia/Lord_Howe", "month", 1782898200, 1782826200),
        ("Australia/Lord_Howe", "week", 1782898200, 1782567000),
        ("Australia/Lord_Howe", "day", 1782898200, 1782826200),
        ("Australia/Lord_Howe", "hour", 1782898200, 1782898200),
        ("Australia/Lord_Howe", "minute", 1782898200, 1782898200),
        ("Australia/Lord_Howe", "year", 1791036000, 1767186000),
        ("Australia/Lord_Howe", "quarter", 1791036000, 1790775000),
        ("Australia/Lord_Howe", "month", 1791036000, 1790775000),
        ("Australia/Lord_Howe", "week", 1791036000, 1791034200),
        ("Australia/Lord_Howe", "day", 1791036000, 1791034200),
        ("Australia/Lord_Howe", "hour", 1791036000, 1791034200),
        ("Australia/Lord_Howe", "minute", 1791036000, 1791036000),
        ("Australia/Lord_Howe", "year", 1791041399, 1767186000),
        ("Australia/Lord_Howe", "quarter", 1791041399, 1790775000),
        ("Australia/Lord_Howe", "month", 1791041399, 1790775000),
        ("Australia/Lord_Howe", "week", 1791041399, 1791034200),
        ("Australia/Lord_Howe", "day", 1791041399, 1791034200),
        ("Australia/Lord_Howe", "hour", 1791041399, 1791037800),
        ("Australia/Lord_Howe", "minute", 1791041399, 1791041340),
        ("Australia/Lord_Howe", "year", 1791041400, 1767186000),
        ("Australia/Lord_Howe", "quarter", 1791041400, 1790775000),
        ("Australia/Lord_Howe", "month", 1791041400, 1790775000),
        ("Australia/Lord_Howe", "week", 1791041400, 1791034200),
        ("Australia/Lord_Howe", "day", 1791041400, 1791034200),
        ("Australia/Lord_Howe", "hour", 1791041400, 1791041400),
        ("Australia/Lord_Howe", "minute", 1791041400, 1791041400),
        ("Australia/Lord_Howe", "year", 1791043200, 1767186000),
        ("Australia/Lord_Howe", "quarter", 1791043200, 1790775000),
        ("Australia/Lord_Howe", "month", 1791043200, 1790775000),
        ("Australia/Lord_Howe", "week", 1791043200, 1791034200),
        ("Australia/Lord_Howe", "day", 1791043200, 1791034200),
        ("Australia/Lord_Howe", "hour", 1791043200, 1791043200),
        ("Australia/Lord_Howe", "minute", 1791043200, 1791043200),
        ("Australia/Lord_Howe", "year", 1791046800, 1767186000),
        ("Australia/Lord_Howe", "quarter", 1791046800, 1790775000),
        ("Australia/Lord_Howe", "month", 1791046800, 1790775000),
        ("Australia/Lord_Howe", "week", 1791046800, 1791034200),
        ("Australia/Lord_Howe", "day", 1791046800, 1791034200),
        ("Australia/Lord_Howe", "hour", 1791046800, 1791046800),
        ("Australia/Lord_Howe", "minute", 1791046800, 1791046800),
        ("Asia/Kolkata", "year", 1767259800, 1767205800),
        ("Asia/Kolkata", "quarter", 1767259800, 1767205800),
        ("Asia/Kolkata", "month", 1767259800, 1767205800),
        ("Asia/Kolkata", "week", 1767259800, 1766860200),
        ("Asia/Kolkata", "day", 1767259800, 1767205800),
        ("Asia/Kolkata", "hour", 1767259800, 1767259800),
        ("Asia/Kolkata", "minute", 1767259800, 1767259800),
        ("Asia/Kolkata", "year", 1767277800, 1767205800),
        ("Asia/Kolkata", "quarter", 1767277800, 1767205800),
        ("Asia/Kolkata", "month", 1767277800, 1767205800),
        ("Asia/Kolkata", "week", 1767277800, 1766860200),
        ("Asia/Kolkata", "day", 1767277800, 1767205800),
        ("Asia/Kolkata", "hour", 1767277800, 1767277800),
        ("Asia/Kolkata", "minute", 1767277800, 1767277800),
        ("Asia/Kolkata", "year", 1782898200, 1767205800),
        ("Asia/Kolkata", "quarter", 1782898200, 1782844200),
        ("Asia/Kolkata", "month", 1782898200, 1782844200),
        ("Asia/Kolkata", "week", 1782898200, 1782585000),
        ("Asia/Kolkata", "day", 1782898200, 1782844200),
        ("Asia/Kolkata", "hour", 1782898200, 1782898200),
        ("Asia/Kolkata", "minute", 1782898200, 1782898200),
        ("Asia/Kathmandu", "year", 1767259800, 1767204900),
        ("Asia/Kathmandu", "quarter", 1767259800, 1767204900),
        ("Asia/Kathmandu", "month", 1767259800, 1767204900),
        ("Asia/Kathmandu", "week", 1767259800, 1766859300),
        ("Asia/Kathmandu", "day", 1767259800, 1767204900),
        ("Asia/Kathmandu", "hour", 1767259800, 1767258900),
        ("Asia/Kathmandu", "minute", 1767259800, 1767259800),
        ("Asia/Kathmandu", "year", 1767277800, 1767204900),
        ("Asia/Kathmandu", "quarter", 1767277800, 1767204900),
        ("Asia/Kathmandu", "month", 1767277800, 1767204900),
        ("Asia/Kathmandu", "week", 1767277800, 1766859300),
        ("Asia/Kathmandu", "day", 1767277800, 1767204900),
        ("Asia/Kathmandu", "hour", 1767277800, 1767276900),
        ("Asia/Kathmandu", "minute", 1767277800, 1767277800),
        ("Asia/Kathmandu", "year", 1782898200, 1767204900),
        ("Asia/Kathmandu", "quarter", 1782898200, 1782843300),
        ("Asia/Kathmandu", "month", 1782898200, 1782843300),
        ("Asia/Kathmandu", "week", 1782898200, 1782584100),
        ("Asia/Kathmandu", "day", 1782898200, 1782843300),
        ("Asia/Kathmandu", "hour", 1782898200, 1782897300),
        ("Asia/Kathmandu", "minute", 1782898200, 1782898200),
        ("America/Havana", "year", 1767259800, 1767243600),
        ("America/Havana", "quarter", 1767259800, 1767243600),
        ("America/Havana", "month", 1767259800, 1767243600),
        ("America/Havana", "week", 1767259800, 1766898000),
        ("America/Havana", "day", 1767259800, 1767243600),
        ("America/Havana", "hour", 1767259800, 1767258000),
        ("America/Havana", "minute", 1767259800, 1767259800),
        ("America/Havana", "year", 1767277800, 1767243600),
        ("America/Havana", "quarter", 1767277800, 1767243600),
        ("America/Havana", "month", 1767277800, 1767243600),
        ("America/Havana", "week", 1767277800, 1766898000),
        ("America/Havana", "day", 1767277800, 1767243600),
        ("America/Havana", "hour", 1767277800, 1767276000),
        ("America/Havana", "minute", 1767277800, 1767277800),
        ("America/Havana", "year", 1772940600, 1767243600),
        ("America/Havana", "quarter", 1772940600, 1767243600),
        ("America/Havana", "month", 1772940600, 1772341200),
        ("America/Havana", "week", 1772940600, 1772341200),
        ("America/Havana", "day", 1772940600, 1772859600),
        ("America/Havana", "hour", 1772940600, 1772938800),
        ("America/Havana", "minute", 1772940600, 1772940600),
        ("America/Havana", "year", 1772945999, 1767243600),
        ("America/Havana", "quarter", 1772945999, 1767243600),
        ("America/Havana", "month", 1772945999, 1772341200),
        ("America/Havana", "week", 1772945999, 1772341200),
        ("America/Havana", "day", 1772945999, 1772859600),
        ("America/Havana", "hour", 1772945999, 1772942400),
        ("America/Havana", "minute", 1772945999, 1772945940),
        ("America/Havana", "year", 1772946000, 1767243600),
        ("America/Havana", "quarter", 1772946000, 1767243600),
        ("America/Havana", "month", 1772946000, 1772341200),
        ("America/Havana", "week", 1772946000, 1772946000),
        ("America/Havana", "day", 1772946000, 1772946000),
        ("America/Havana", "hour", 1772946000, 1772946000),
        ("America/Havana", "minute", 1772946000, 1772946000),
        ("America/Havana", "year", 1772947800, 1767243600),
        ("America/Havana", "quarter", 1772947800, 1767243600),
        ("America/Havana", "month", 1772947800, 1772341200),
        ("America/Havana", "week", 1772947800, 1772946000),
        ("America/Havana", "day", 1772947800, 1772946000),
        ("America/Havana", "hour", 1772947800, 1772946000),
        ("America/Havana", "minute", 1772947800, 1772947800),
        ("America/Havana", "year", 1772951400, 1767243600),
        ("America/Havana", "quarter", 1772951400, 1767243600),
        ("America/Havana", "month", 1772951400, 1772341200),
        ("America/Havana", "week", 1772951400, 1772946000),
        ("America/Havana", "day", 1772951400, 1772946000),
        ("America/Havana", "hour", 1772951400, 1772949600),
        ("America/Havana", "minute", 1772951400, 1772951400),
        ("America/Havana", "year", 1782898200, 1767243600),
        ("America/Havana", "quarter", 1782898200, 1782878400),
        ("America/Havana", "month", 1782898200, 1782878400),
        ("America/Havana", "week", 1782898200, 1782619200),
        ("America/Havana", "day", 1782898200, 1782878400),
        ("America/Havana", "hour", 1782898200, 1782896400),
        ("America/Havana", "minute", 1782898200, 1782898200),
        ("America/Havana", "year", 1793503800, 1767243600),
        ("America/Havana", "quarter", 1793503800, 1790827200),
        ("America/Havana", "month", 1793503800, 1790827200),
        ("America/Havana", "week", 1793503800, 1792900800),
        ("America/Havana", "day", 1793503800, 1793419200),
        ("America/Havana", "hour", 1793503800, 1793502000),
        ("America/Havana", "minute", 1793503800, 1793503800),
        ("America/Havana", "year", 1793509199, 1767243600),
        ("America/Havana", "quarter", 1793509199, 1790827200),
        ("America/Havana", "month", 1793509199, 1793505600),
        ("America/Havana", "week", 1793509199, 1793505600),
        ("America/Havana", "day", 1793509199, 1793505600),
        ("America/Havana", "hour", 1793509199, 1793505600),
        ("America/Havana", "minute", 1793509199, 1793509140),
        ("America/Havana", "year", 1793509200, 1767243600),
        ("America/Havana", "quarter", 1793509200, 1790827200),
        ("America/Havana", "month", 1793509200, 1793509200),
        ("America/Havana", "week", 1793509200, 1793509200),
        ("America/Havana", "day", 1793509200, 1793509200),
        ("America/Havana", "hour", 1793509200, 1793509200),
        ("America/Havana", "minute", 1793509200, 1793509200),
        ("America/Havana", "year", 1793511000, 1767243600),
        ("America/Havana", "quarter", 1793511000, 1790827200),
        ("America/Havana", "month", 1793511000, 1793509200),
        ("America/Havana", "week", 1793511000, 1793509200),
        ("America/Havana", "day", 1793511000, 1793509200),
        ("America/Havana", "hour", 1793511000, 1793509200),
        ("America/Havana", "minute", 1793511000, 1793511000),
        ("America/Havana", "year", 1793514600, 1767243600),
        ("America/Havana", "quarter", 1793514600, 1790827200),
        ("America/Havana", "month", 1793514600, 1793509200),
        ("America/Havana", "week", 1793514600, 1793509200),
        ("America/Havana", "day", 1793514600, 1793509200),
        ("America/Havana", "hour", 1793514600, 1793512800),
        ("America/Havana", "minute", 1793514600, 1793514600),
        ("Asia/Beirut", "year", 1767259800, 1767218400),
        ("Asia/Beirut", "quarter", 1767259800, 1767218400),
        ("Asia/Beirut", "month", 1767259800, 1767218400),
        ("Asia/Beirut", "week", 1767259800, 1766872800),
        ("Asia/Beirut", "day", 1767259800, 1767218400),
        ("Asia/Beirut", "hour", 1767259800, 1767258000),
        ("Asia/Beirut", "minute", 1767259800, 1767259800),
        ("Asia/Beirut", "year", 1767277800, 1767218400),
        ("Asia/Beirut", "quarter", 1767277800, 1767218400),
        ("Asia/Beirut", "month", 1767277800, 1767218400),
        ("Asia/Beirut", "week", 1767277800, 1766872800),
        ("Asia/Beirut", "day", 1767277800, 1767218400),
        ("Asia/Beirut", "hour", 1767277800, 1767276000),
        ("Asia/Beirut", "minute", 1767277800, 1767277800),
        ("Asia/Beirut", "year", 1774729800, 1767218400),
        ("Asia/Beirut", "quarter", 1774729800, 1767218400),
        ("Asia/Beirut", "month", 1774729800, 1772316000),
        ("Asia/Beirut", "week", 1774729800, 1774130400),
        ("Asia/Beirut", "day", 1774729800, 1774648800),
        ("Asia/Beirut", "hour", 1774729800, 1774728000),
        ("Asia/Beirut", "minute", 1774729800, 1774729800),
        ("Asia/Beirut", "year", 1774735199, 1767218400),
        ("Asia/Beirut", "quarter", 1774735199, 1767218400),
        ("Asia/Beirut", "month", 1774735199, 1772316000),
        ("Asia/Beirut", "week", 1774735199, 1774130400),
        ("Asia/Beirut", "day", 1774735199, 1774648800),
        ("Asia/Beirut", "hour", 1774735199, 1774731600),
        ("Asia/Beirut", "minute", 1774735199, 1774735140),
        ("Asia/Beirut", "year", 1774735200, 1767218400),
        ("Asia/Beirut", "quarter", 1774735200, 1767218400),
        ("Asia/Beirut", "month", 1774735200, 1772316000),
        ("Asia/Beirut", "week", 1774735200, 1774735200),
        ("Asia/Beirut", "day", 1774735200, 1774735200),
        ("Asia/Beirut", "hour", 1774735200, 1774735200),
        ("Asia/Beirut", "minute", 1774735200, 1774735200),
        ("Asia/Beirut", "year", 1774737000, 1767218400),
        ("Asia/Beirut", "quarter", 1774737000, 1767218400),
        ("Asia/Beirut", "month", 1774737000, 1772316000),
        ("Asia/Beirut", "week", 1774737000, 1774735200),
        ("Asia/Beirut", "day", 1774737000, 1774735200),
        ("Asia/Beirut", "hour", 1774737000, 1774735200),
        ("Asia/Beirut", "minute", 1774737000, 1774737000),
        ("Asia/Beirut", "year", 1774740600, 1767218400),
        ("Asia/Beirut", "quarter", 1774740600, 1767218400),
        ("Asia/Beirut", "month", 1774740600, 1772316000),
        ("Asia/Beirut", "week", 1774740600, 1774735200),
        ("Asia/Beirut", "day", 1774740600, 1774735200),
        ("Asia/Beirut", "hour", 1774740600, 1774738800),
        ("Asia/Beirut", "minute", 1774740600, 1774740600),
        ("Asia/Beirut", "year", 1782898200, 1767218400),
        ("Asia/Beirut", "quarter", 1782898200, 1782853200),
        ("Asia/Beirut", "month", 1782898200, 1782853200),
        ("Asia/Beirut", "week", 1782898200, 1782594000),
        ("Asia/Beirut", "day", 1782898200, 1782853200),
        ("Asia/Beirut", "hour", 1782898200, 1782896400),
        ("Asia/Beirut", "minute", 1782898200, 1782898200),
        ("Asia/Beirut", "year", 1792870200, 1767218400),
        ("Asia/Beirut", "quarter", 1792870200, 1790802000),
        ("Asia/Beirut", "month", 1792870200, 1790802000),
        ("Asia/Beirut", "week", 1792870200, 1792270800),
        ("Asia/Beirut", "day", 1792870200, 1792789200),
        ("Asia/Beirut", "hour", 1792870200, 1792868400),
        ("Asia/Beirut", "minute", 1792870200, 1792870200),
        ("Asia/Beirut", "year", 1792875599, 1767218400),
        ("Asia/Beirut", "quarter", 1792875599, 1790802000),
        ("Asia/Beirut", "month", 1792875599, 1790802000),
        ("Asia/Beirut", "week", 1792875599, 1792270800),
        ("Asia/Beirut", "day", 1792875599, 1792789200),
        ("Asia/Beirut", "hour", 1792875599, 1792872000),
        ("Asia/Beirut", "minute", 1792875599, 1792875540),
        ("Asia/Beirut", "year", 1792875600, 1767218400),
        ("Asia/Beirut", "quarter", 1792875600, 1790802000),
        ("Asia/Beirut", "month", 1792875600, 1790802000),
        ("Asia/Beirut", "week", 1792875600, 1792270800),
        ("Asia/Beirut", "day", 1792875600, 1792789200),
        ("Asia/Beirut", "hour", 1792875600, 1792875600),
        ("Asia/Beirut", "minute", 1792875600, 1792875600),
        ("Asia/Beirut", "year", 1792877400, 1767218400),
        ("Asia/Beirut", "quarter", 1792877400, 1790802000),
        ("Asia/Beirut", "month", 1792877400, 1790802000),
        ("Asia/Beirut", "week", 1792877400, 1792270800),
        ("Asia/Beirut", "day", 1792877400, 1792789200),
        ("Asia/Beirut", "hour", 1792877400, 1792875600),
        ("Asia/Beirut", "minute", 1792877400, 1792877400),
        ("Asia/Beirut", "year", 1792881000, 1767218400),
        ("Asia/Beirut", "quarter", 1792881000, 1790802000),
        ("Asia/Beirut", "month", 1792881000, 1790802000),
        ("Asia/Beirut", "week", 1792881000, 1792879200),
        ("Asia/Beirut", "day", 1792881000, 1792879200),
        ("Asia/Beirut", "hour", 1792881000, 1792879200),
        ("Asia/Beirut", "minute", 1792881000, 1792881000),
        ("America/Santiago", "year", 1767259800, 1767236400),
        ("America/Santiago", "quarter", 1767259800, 1767236400),
        ("America/Santiago", "month", 1767259800, 1767236400),
        ("America/Santiago", "week", 1767259800, 1766890800),
        ("America/Santiago", "day", 1767259800, 1767236400),
        ("America/Santiago", "hour", 1767259800, 1767258000),
        ("America/Santiago", "minute", 1767259800, 1767259800),
        ("America/Santiago", "year", 1767277800, 1767236400),
        ("America/Santiago", "quarter", 1767277800, 1767236400),
        ("America/Santiago", "month", 1767277800, 1767236400),
        ("America/Santiago", "week", 1767277800, 1766890800),
        ("America/Santiago", "day", 1767277800, 1767236400),
        ("America/Santiago", "hour", 1767277800, 1767276000),
        ("America/Santiago", "minute", 1767277800, 1767277800),
        ("America/Santiago", "year", 1775352600, 1767236400),
        ("America/Santiago", "quarter", 1775352600, 1775012400),
        ("America/Santiago", "month", 1775352600, 1775012400),
        ("America/Santiago", "week", 1775352600, 1774753200),
        ("America/Santiago", "day", 1775352600, 1775271600),
        ("America/Santiago", "hour", 1775352600, 1775350800),
        ("America/Santiago", "minute", 1775352600, 1775352600),
        ("America/Santiago", "year", 1775357999, 1767236400),
        ("America/Santiago", "quarter", 1775357999, 1775012400),
        ("America/Santiago", "month", 1775357999, 1775012400),
        ("America/Santiago", "week", 1775357999, 1774753200),
        ("America/Santiago", "day", 1775357999, 1775271600),
        ("America/Santiago", "hour", 1775357999, 1775354400),
        ("America/Santiago", "minute", 1775357999, 1775357940),
        ("America/Santiago", "year", 1775358000, 1767236400),
        ("America/Santiago", "quarter", 1775358000, 1775012400),
        ("America/Santiago", "month", 1775358000, 1775012400),
        ("America/Santiago", "week", 1775358000, 1774753200),
        ("America/Santiago", "day", 1775358000, 1775271600),
        ("America/Santiago", "hour", 1775358000, 1775358000),
        ("America/Santiago", "minute", 1775358000, 1775358000),
        ("America/Santiago", "year", 1775359800, 1767236400),
        ("America/Santiago", "quarter", 1775359800, 1775012400),
        ("America/Santiago", "month", 1775359800, 1775012400),
        ("America/Santiago", "week", 1775359800, 1774753200),
        ("America/Santiago", "day", 1775359800, 1775271600),
        ("America/Santiago", "hour", 1775359800, 1775358000),
        ("America/Santiago", "minute", 1775359800, 1775359800),
        ("America/Santiago", "year", 1775363400, 1767236400),
        ("America/Santiago", "quarter", 1775363400, 1775012400),
        ("America/Santiago", "month", 1775363400, 1775012400),
        ("America/Santiago", "week", 1775363400, 1775361600),
        ("America/Santiago", "day", 1775363400, 1775361600),
        ("America/Santiago", "hour", 1775363400, 1775361600),
        ("America/Santiago", "minute", 1775363400, 1775363400),
        ("America/Santiago", "year", 1782898200, 1767236400),
        ("America/Santiago", "quarter", 1782898200, 1782878400),
        ("America/Santiago", "month", 1782898200, 1782878400),
        ("America/Santiago", "week", 1782898200, 1782619200),
        ("America/Santiago", "day", 1782898200, 1782878400),
        ("America/Santiago", "hour", 1782898200, 1782896400),
        ("America/Santiago", "minute", 1782898200, 1782898200),
        ("America/Santiago", "year", 1788661800, 1767236400),
        ("America/Santiago", "quarter", 1788661800, 1782878400),
        ("America/Santiago", "month", 1788661800, 1788235200),
        ("America/Santiago", "week", 1788661800, 1788062400),
        ("America/Santiago", "day", 1788661800, 1788580800),
        ("America/Santiago", "hour", 1788661800, 1788660000),
        ("America/Santiago", "minute", 1788661800, 1788661800),
        ("America/Santiago", "year", 1788667199, 1767236400),
        ("America/Santiago", "quarter", 1788667199, 1782878400),
        ("America/Santiago", "month", 1788667199, 1788235200),
        ("America/Santiago", "week", 1788667199, 1788062400),
        ("America/Santiago", "day", 1788667199, 1788580800),
        ("America/Santiago", "hour", 1788667199, 1788663600),
        ("America/Santiago", "minute", 1788667199, 1788667140),
        ("America/Santiago", "year", 1788667200, 1767236400),
        ("America/Santiago", "quarter", 1788667200, 1782878400),
        ("America/Santiago", "month", 1788667200, 1788235200),
        ("America/Santiago", "week", 1788667200, 1788667200),
        ("America/Santiago", "day", 1788667200, 1788667200),
        ("America/Santiago", "hour", 1788667200, 1788667200),
        ("America/Santiago", "minute", 1788667200, 1788667200),
        ("America/Santiago", "year", 1788669000, 1767236400),
        ("America/Santiago", "quarter", 1788669000, 1782878400),
        ("America/Santiago", "month", 1788669000, 1788235200),
        ("America/Santiago", "week", 1788669000, 1788667200),
        ("America/Santiago", "day", 1788669000, 1788667200),
        ("America/Santiago", "hour", 1788669000, 1788667200),
        ("America/Santiago", "minute", 1788669000, 1788669000),
        ("America/Santiago", "year", 1788672600, 1767236400),
        ("America/Santiago", "quarter", 1788672600, 1782878400),
        ("America/Santiago", "month", 1788672600, 1788235200),
        ("America/Santiago", "week", 1788672600, 1788667200),
        ("America/Santiago", "day", 1788672600, 1788667200),
        ("America/Santiago", "hour", 1788672600, 1788670800),
        ("America/Santiago", "minute", 1788672600, 1788672600),
    ];

    /// Time zone, instant in seconds, months added, and the resulting instant.
    const MONTHS: &[(&str, i64, i32, i64)] = &[
        ("Australia/Sydney", 1788453000, 1, 1791045000),
        ("Australia/Sydney", 1772638200, 1, 1775316600),
        ("Australia/Sydney", 1773529200, 1, 1776211200),
        ("Australia/Sydney", 1769807700, 1, 1772226900),
        ("Australia/Sydney", 1782871200, 6, 1798765200),
        ("Australia/Sydney", 1782871200, -6, 1767229200),
        ("America/New_York", 1770535800, 1, 1772955000),
        ("America/New_York", 1790832600, 1, 1793511000),
        ("America/New_York", 1768485600, 12, 1800021600),
        ("America/New_York", 1709215200, 12, 1740751200),
        ("Europe/London", 1772242200, 1, 1774661400),
        ("Europe/London", 1772242200, 13, 1806197400),
        ("Australia/Lord_Howe", 1788450300, 1, 1791042300),
        ("America/Havana", 1770528600, 1, 1772947800),
        ("America/Havana", 1790829000, 1, 1793507400),
        ("Asia/Kolkata", 1769884140, 1, 1772303340),
        ("Asia/Kathmandu", 1780189200, -3, 1772240400),
    ];

    const UNITS: [(TimeUnit, i64); 4] = [
        (TimeUnit::Seconds, 1),
        (TimeUnit::Milliseconds, 1_000),
        (TimeUnit::Microseconds, 1_000_000),
        (TimeUnit::Nanoseconds, 1_000_000_000),
    ];

    fn zoned(values: &[i64], tz: &str, unit: TimeUnit) -> DatetimeArray<i64> {
        DatetimeArray::<i64>::from_slice(values, Dt::Timestamp(unit, Some(tz.into())))
    }

    fn components(arr: &DatetimeArray<i64>, row: usize) -> [i32; 11] {
        [
            arr.year().data[row],
            arr.month().data[row],
            arr.day().data[row],
            arr.hour().data[row],
            arr.minute().data[row],
            arr.second().data[row],
            arr.weekday().data[row],
            arr.day_of_year().data[row],
            arr.iso_week().data[row],
            arr.quarter().data[row],
            arr.week_of_year().data[row],
        ]
    }

    #[test]
    fn test_components_read_the_wall_time_of_the_time_zone() {
        for (unit, scale) in UNITS {
            for (tz, instant, fields, leap) in COMPONENTS {
                // A fraction of a second is added for the sub-second units.
                let value = instant * scale + scale / 4;
                let arr = zoned(&[value], tz, unit);
                assert_eq!(components(&arr, 0), *fields, "{tz} at {instant} in {unit:?}");
                assert_eq!(arr.is_leap_year().get(0), Some(*leap), "{tz} at {instant}");
                // The stored value is the UTC value.
                assert_eq!(arr.data[0], value);
            }
        }
    }

    #[test]
    fn test_truncation_writes_the_utc_value_of_the_period_start_on_the_wall_clock() {
        for (unit, scale) in UNITS {
            for (tz, period, instant, start) in TRUNCATIONS {
                let value = instant * scale + scale / 4;
                let arr = zoned(&[value], tz, unit);
                let truncated = arr.truncate(*period);
                assert_eq!(
                    truncated.data[0],
                    start * scale,
                    "{tz} at {instant} to the {period} in {unit:?}"
                );
                assert_eq!(truncated.dtype, arr.dtype);
                assert_eq!(arr.data[0], value);
            }
        }
    }

    #[test]
    fn test_add_months_keeps_the_wall_time_and_writes_its_utc_value() {
        for (unit, scale) in UNITS {
            for (tz, instant, months, result) in MONTHS {
                let value = instant * scale + scale / 4;
                let arr = zoned(&[value], tz, unit);
                let added = arr.add_months(*months).unwrap();
                assert_eq!(
                    added.data[0],
                    result * scale + scale / 4,
                    "{tz} at {instant} plus {months} months in {unit:?}"
                );
                assert_eq!(added.dtype, arr.dtype);
                assert_eq!(arr.data[0], value);
            }
        }
    }

    #[test]
    fn test_add_years_adds_twelve_months_on_the_wall_time() {
        for (tz, instant, months, result) in MONTHS.iter().filter(|case| case.2 == 12) {
            let arr = zoned(&[*instant], tz, TimeUnit::Seconds);
            assert_eq!(arr.add_years(months / 12).unwrap().data[0], *result, "{tz} at {instant}");
        }
    }

    #[test]
    fn test_null_rows_stay_null_and_other_rows_read_their_wall_time() {
        let rows: Vec<&(&str, i64, [i32; 11], bool)> = COMPONENTS
            .iter()
            .filter(|row| row.0 == "Australia/Sydney")
            .collect();
        let values: Vec<i64> = rows.iter().map(|row| row.1).collect();
        let mut arr = zoned(&values, "Australia/Sydney", TimeUnit::Seconds);
        let mut mask = Bitmask::new_set_all(values.len(), true);
        mask.set(1, false);
        arr.null_mask = Some(mask);

        let hours = arr.hour();
        let days = arr.truncate(TimePeriod::Day);
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(hours.is_null(i), i == 1);
            assert_eq!(days.is_null(i), i == 1);
            if i != 1 {
                assert_eq!(hours.data[i], row.2[3], "row {i}");
            }
        }
        assert_eq!(&arr.data[..], &values[..]);
    }

    #[test]
    fn test_views_and_the_temporal_enum_read_the_wall_time_of_the_time_zone() {
        let rows: Vec<&(&str, i64, [i32; 11], bool)> = COMPONENTS
            .iter()
            .filter(|row| row.0 == "America/New_York")
            .collect();
        let values: Vec<i64> = rows.iter().map(|row| row.1).collect();
        let temporal = TemporalArray::Datetime64(Arc::new(zoned(
            &values,
            "America/New_York",
            TimeUnit::Seconds,
        )));

        let hours = temporal.hour();
        let days = temporal.day();
        let leap = temporal.is_leap_year();
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(hours.data[i], row.2[3], "enum row {i}");
            assert_eq!(days.data[i], row.2[2], "enum row {i}");
            assert_eq!(leap.get(i), Some(row.3), "enum row {i}");
        }

        // The view covers the rows after the first one.
        let view = TemporalArrayV::new(temporal, 1, values.len() - 1);
        let hours = view.hour();
        let days = view.day();
        let weekdays = view.weekday();
        let leap = view.is_leap_year();
        for (i, row) in rows.iter().skip(1).enumerate() {
            assert_eq!(hours.data[i], row.2[3], "view row {i}");
            assert_eq!(days.data[i], row.2[2], "view row {i}");
            assert_eq!(weekdays.data[i], row.2[6], "view row {i}");
            assert_eq!(leap.get(i), Some(row.3), "view row {i}");
        }
    }

    // Arrays of many rows reuse the offsets of the days they have read. Each row
    // must equal the result of an array holding that row alone, in time order,
    // in reverse order and in an order that moves between distant days.
    #[test]
    fn test_rows_of_one_array_equal_the_results_of_single_row_arrays() {
        for tz in ["Australia/Sydney", "America/New_York", "Australia/Lord_Howe", "America/Havana"] {
            // Every 20 minutes from 20 days before to 20 days after each transition of 2026.
            let transitions: Vec<i64> = TRUNCATIONS
                .iter()
                .filter(|row| row.0 == tz && row.1 == "day")
                .map(|row| row.2)
                .collect();
            let mut values: Vec<i64> = Vec::new();
            for transition in [transitions[3], transitions[transitions.len() - 3]] {
                values.extend((-1_440..1_440).map(|step| transition + step * 1_200));
            }
            let reversed: Vec<i64> = values.iter().rev().copied().collect();
            let half = values.len() / 2;
            let interleaved: Vec<i64> = (0..half).flat_map(|i| [values[i], values[half + i]]).collect();

            for order in [&values, &reversed, &interleaved] {
                let arr = zoned(order, tz, TimeUnit::Seconds);
                let hours = arr.hour();
                let days = arr.truncate(TimePeriod::Day);
                let months = arr.truncate(TimePeriod::Month);
                let added = arr.add_months(1).unwrap();
                for (i, value) in order.iter().enumerate() {
                    let single = zoned(&[*value], tz, TimeUnit::Seconds);
                    assert_eq!(hours.data[i], single.hour().data[0], "{tz} hour at {value}");
                    assert_eq!(
                        days.data[i],
                        single.truncate(TimePeriod::Day).data[0],
                        "{tz} day at {value}"
                    );
                    assert_eq!(
                        months.data[i],
                        single.truncate(TimePeriod::Month).data[0],
                        "{tz} month at {value}"
                    );
                    assert_eq!(
                        added.data[i],
                        single.add_months(1).unwrap().data[0],
                        "{tz} plus one month at {value}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_output_buffer_kernels_read_the_wall_time_of_the_time_zone() {
        let rows: Vec<&(&str, i64, [i32; 11], bool)> = COMPONENTS
            .iter()
            .filter(|row| row.0 == "Australia/Lord_Howe")
            .collect();
        let values: Vec<i64> = rows.iter().map(|row| row.1).collect();
        let arr = zoned(&values, "Australia/Lord_Howe", TimeUnit::Seconds);

        // The window covers the rows after the first one.
        let len = values.len() - 1;
        let mut hours = vec![0i32; len];
        let mut minutes = vec![0i32; len];
        let mut mask = Bitmask::new_set_all(len, true);
        hour_into(&arr, 1, &mut hours, Some(&mut mask));
        minute_into(&arr, 1, &mut minutes, None);
        let mut leap = Bitmask::new_set_all(len, false);
        is_leap_year_into(&arr, 1, &mut leap, None);
        for (i, row) in rows.iter().skip(1).enumerate() {
            assert_eq!((hours[i], minutes[i]), (row.2[3], row.2[4]), "row {i}");
            assert_eq!(leap.get(i), row.3, "row {i}");
            assert!(mask.get(i));
        }
    }

    #[test]
    fn test_instant_operations_return_the_values_of_arrays_without_a_time_zone() {
        let values: Vec<i64> = COMPONENTS.iter().map(|row| row.1 * 1_000 + 250).collect();
        let later: Vec<i64> = values.iter().map(|value| value + 5_400_000).collect();
        let plain = DatetimeArray::<i64>::from_slice(&values, Dt::Timestamp(TimeUnit::Milliseconds, None));
        let plain_later =
            DatetimeArray::<i64>::from_slice(&later, Dt::Timestamp(TimeUnit::Milliseconds, None));
        let arr = zoned(&values, "Australia/Sydney", TimeUnit::Milliseconds);
        let arr_later = zoned(&later, "Australia/Sydney", TimeUnit::Milliseconds);

        let duration = Duration::hours(36);
        assert_eq!(
            &arr.add_duration(duration).unwrap().data[..],
            &plain.add_duration(duration).unwrap().data[..]
        );
        assert_eq!(
            &arr.sub_duration(duration).unwrap().data[..],
            &plain.sub_duration(duration).unwrap().data[..]
        );
        // `add_days` adds days of 24 hours to the instant.
        assert_eq!(&arr.add_days(1).unwrap().data[..], &plain.add_days(1).unwrap().data[..]);
        assert_eq!(
            &arr_later.diff(&arr, TimeUnit::Seconds).unwrap().data[..],
            &plain_later.diff(&plain, TimeUnit::Seconds).unwrap().data[..]
        );
        assert_eq!(arr.is_before(&arr_later).unwrap(), plain.is_before(&plain_later).unwrap());
        assert_eq!(arr.is_after(&arr_later).unwrap(), plain.is_after(&plain_later).unwrap());
        assert_eq!(
            &arr.cast_time_unit(TimeUnit::Seconds).unwrap().data[..],
            &plain.cast_time_unit(TimeUnit::Seconds).unwrap().data[..]
        );
        // Truncation to seconds is not affected by offsets of whole seconds.
        assert_eq!(&arr.sec().data[..], &plain.sec().data[..]);
    }

    #[test]
    fn test_unresolved_time_zone_text_is_read_as_utc() {
        let values: Vec<i64> = COMPONENTS.iter().map(|row| row.1).collect();
        let plain = DatetimeArray::<i64>::from_slice(&values, Dt::Timestamp(TimeUnit::Seconds, None));
        let arr = zoned(&values, "Mars/Olympus_Mons", TimeUnit::Seconds);
        assert_eq!(&arr.hour().data[..], &plain.hour().data[..]);
        assert_eq!(&arr.day().data[..], &plain.day().data[..]);
        assert_eq!(
            &arr.truncate(TimePeriod::Day).data[..],
            &plain.truncate(TimePeriod::Day).data[..]
        );
        assert_eq!(&arr.add_months(1).unwrap().data[..], &plain.add_months(1).unwrap().data[..]);
    }

    #[test]
    fn test_offsets_and_abbreviations_shift_the_wall_time_by_their_fixed_offset() {
        // 2026-01-01 09:30:00 UTC.
        let arr = zoned(&[1_767_259_800], "+05:30", TimeUnit::Seconds);
        assert_eq!((arr.hour().data[0], arr.minute().data[0]), (15, 0));
        let arr = zoned(&[1_767_259_800], "-08:00", TimeUnit::Seconds);
        assert_eq!((arr.day().data[0], arr.hour().data[0]), (1, 1));
        let arr = zoned(&[1_767_259_800], "AEST", TimeUnit::Seconds);
        assert_eq!((arr.hour().data[0], arr.minute().data[0]), (19, 30));
        // The day starts at 14:00 UTC of the previous day at ten hours ahead of UTC.
        assert_eq!(arr.truncate(TimePeriod::Day).data[0], 1_767_189_600);
    }

    #[test]
    fn test_printed_text_and_components_read_the_same_wall_time() {
        for (tz, instant, fields, _) in COMPONENTS {
            let arr = zoned(&[*instant], tz, TimeUnit::Seconds);
            let text = Array::from_datetime_i64(arr.clone()).value_to_string(0);
            // The text is `YYYY-MM-DD HH:MM:SS` followed by the offset and the time zone.
            let printed: Vec<i32> = [0..4, 5..7, 8..10, 11..13, 14..16, 17..19]
                .into_iter()
                .map(|range| text[range].parse().unwrap())
                .collect();
            assert_eq!(printed, fields[..6], "{tz} at {instant} prints {text}");
            assert_eq!(components(&arr, 0)[..6], fields[..6]);
        }
    }
}
