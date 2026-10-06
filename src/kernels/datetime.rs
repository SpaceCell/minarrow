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

//! # **Datetime Kernels Module** - *Output-Buffer Datetime Compute*
//!
//! The `_into` datetime kernels compute a result straight into a caller-provided
//! buffer plus an optional output null mask. They back the allocating methods on
//! [`DatetimeArray`] - `truncate`, `add_duration`, `add_months` and the truncation
//! shorthands - which call these kernels and own the result allocation.
//!
//! These `_into` versions let you supply a single output buffer upfront which can
//! be useful when writing to chunks in parallel, as it minimises re-allocations.
//!
//! For an everyday result, call the methods on [`DatetimeArray`] instead.
//!
//! ## Contract
//! `out` receives one value per input element. When `out_mask` is supplied, every slot
//! is written - valid where the input is valid and the compute succeeds, and null where the
//! value cannot be represented as a datetime or overflows the storage type. A null or
//! overflow slot keeps the original input value in `out`.
//!
//! Requires the `datetime_ops` feature.

use num_traits::FromPrimitive;
use time::{Date, Duration};

use crate::enums::error::MinarrowError;
use crate::enums::time_units::{TimePeriod, TimeUnit};
use crate::traits::masked_array::MaskedArray;
use crate::traits::type_unions::Integer;
use crate::{Bitmask, DatetimeArray};

/// Floor each value of `src` in the window `[src_offset, src_offset + out.len())` to
/// the start of `period`, writing into `out`.
///
/// `out.len()` rows are processed, reading `src` from `src_offset`. The allocating
/// `DatetimeArray::truncate` passes `src_offset = 0` over the whole array.
///
/// Handles the full calendar range (`Year` through `Second`, plus `Week`) and the
/// sub-second steps (`Millisecond`, `Microsecond`). Sub-second steps are no-ops on
/// arrays whose stored resolution is already at or coarser than the target.
pub fn truncate_into<T: Integer + FromPrimitive>(
    src: &DatetimeArray<T>,
    src_offset: usize,
    period: TimePeriod,
    out: &mut [T],
    mut out_mask: Option<&mut Bitmask>,
) {
    let time_unit = src.time_unit;
    let len = out.len();

    // Resolve the unit scaling and sub-second divisor once for the whole window.
    // `None` for `per_second` identifies the `Days` unit. A divisor of 1 leaves
    // values unchanged when the stored resolution is at or coarser than the
    // sub-second target.
    let per_second = match time_unit {
        TimeUnit::Seconds => Some(1),
        TimeUnit::Milliseconds => Some(1_000),
        TimeUnit::Microseconds => Some(1_000_000),
        TimeUnit::Nanoseconds => Some(1_000_000_000),
        TimeUnit::Days => None,
    };
    let sub_second_divisor = match (period, time_unit) {
        (TimePeriod::Microsecond, TimeUnit::Nanoseconds) => Some(1_000),
        (TimePeriod::Millisecond, TimeUnit::Nanoseconds) => Some(1_000_000),
        (TimePeriod::Millisecond, TimeUnit::Microseconds) => Some(1_000),
        (TimePeriod::Millisecond | TimePeriod::Microsecond, _) => Some(1),
        _ => None,
    };

    for i in 0..len {
        let original = src.data[src_offset + i];
        out[i] = original;
        let valid = if src.is_null(src_offset + i) {
            false
        } else {
            match original
                .to_i64()
                .and_then(|v| match sub_second_divisor {
                    Some(divisor) => v.div_euclid(divisor).checked_mul(divisor),
                    None => floor_calendar(v, period, per_second),
                })
                .and_then(T::from_i64)
            {
                Some(t) => {
                    out[i] = t;
                    true
                }
                None => false,
            }
        };
        if let Some(mask) = out_mask.as_deref_mut() {
            mask.set(i, valid);
        }
    }
}

const UNIX_EPOCH_JULIAN_DAY: i64 = 2_440_588;
const SECONDS_PER_DAY: i64 = 86_400;
const MIN_JULIAN_DAY: i64 = Date::MIN.to_julian_day() as i64;
const MAX_JULIAN_DAY: i64 = Date::MAX.to_julian_day() as i64;
const MIN_EPOCH_DAY: i64 = MIN_JULIAN_DAY - UNIX_EPOCH_JULIAN_DAY;
const MAX_EPOCH_DAY: i64 = MAX_JULIAN_DAY - UNIX_EPOCH_JULIAN_DAY;

/// Floors a raw datetime value to the start of a calendar `period`, from `Year`
/// through `Second`, using integer arithmetic.
///
/// The value is split into an epoch day and a second of day, the relevant
/// component is floored, and the result is scaled back by `per_second`, where
/// `None` identifies the `Days` unit. `Week` starts on Sunday 00:00 UTC.
///
/// Returns `None` for values outside the `time` crate's date range, for `Week`
/// results that start before `Date::MIN`, and when scaling the result overflows
/// `i64`. The `Days` unit converts through an `i32` Julian day, matching
/// `DatetimeArray::i64_to_datetime`.
fn floor_calendar(v: i64, period: TimePeriod, per_second: Option<i64>) -> Option<i64> {
    // Split into the days since the epoch and the second of day, applying the
    // same range check as `i64_to_datetime`.
    let (day, second_of_day) = match per_second {
        None => {
            let julian_day = v.wrapping_add(UNIX_EPOCH_JULIAN_DAY) as i32 as i64;
            if !(MIN_JULIAN_DAY..=MAX_JULIAN_DAY).contains(&julian_day) {
                return None;
            }
            (julian_day - UNIX_EPOCH_JULIAN_DAY, 0)
        }
        Some(per_second) => {
            let seconds = v.div_euclid(per_second);
            if seconds < MIN_EPOCH_DAY * SECONDS_PER_DAY
                || seconds > MAX_EPOCH_DAY * SECONDS_PER_DAY + SECONDS_PER_DAY - 1
            {
                return None;
            }
            (
                seconds.div_euclid(SECONDS_PER_DAY),
                seconds.rem_euclid(SECONDS_PER_DAY),
            )
        }
    };

    let floored_seconds = match period {
        TimePeriod::Year => year_and_month_start(day).0 * SECONDS_PER_DAY,
        TimePeriod::Month => year_and_month_start(day).1 * SECONDS_PER_DAY,
        // The epoch day is a Thursday, so `(day + 4) mod 7` counts the days back
        // to the preceding Sunday.
        TimePeriod::Week => {
            let week_start = day - (day + 4).rem_euclid(7);
            if week_start < MIN_EPOCH_DAY {
                return None;
            }
            week_start * SECONDS_PER_DAY
        }
        TimePeriod::Day => day * SECONDS_PER_DAY,
        TimePeriod::Hour => day * SECONDS_PER_DAY + second_of_day - second_of_day % 3_600,
        TimePeriod::Minute => day * SECONDS_PER_DAY + second_of_day - second_of_day % 60,
        TimePeriod::Second => day * SECONDS_PER_DAY + second_of_day,
        TimePeriod::Millisecond | TimePeriod::Microsecond => {
            unreachable!("sub-second periods floor through the raw divisor")
        }
    };

    match per_second {
        None => Some(floored_seconds / SECONDS_PER_DAY),
        Some(per_second) => floored_seconds.checked_mul(per_second),
    }
}

/// Shift in 400-year cycles that makes every in-range epoch day non-negative in
/// the March-based calendar used by `year_and_month_start`.
const ERA_SHIFT_DAYS: i64 = (-(MIN_EPOCH_DAY + 719_468) / 146_097 + 1) * 146_097;

/// Returns the epoch days of the first day of the year and the first day of the
/// month containing `day`, in the proleptic Gregorian calendar.
///
/// Uses unsigned arithmetic on a calendar that starts each year on 1 March, so
/// that the leap day falls at the end of the year. `day` must lie within
/// `MIN_EPOCH_DAY..=MAX_EPOCH_DAY`.
fn year_and_month_start(day: i64) -> (i64, i64) {
    let shifted = (day + 719_468 + ERA_SHIFT_DAYS) as u64;
    let day_of_era = shifted % 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    // Day 0 is 1 March.
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    // Month 0 is March, and months 10 and 11 are January and February.
    let month = (5 * day_of_year + 2) / 153;
    let day_of_month = day_of_year - (153 * month + 2) / 5;

    let month_start = day - day_of_month as i64;
    // 1 January is day 306 of the March-based year. From March onwards, the
    // year starts 59 days before 1 March, plus one in a leap year.
    let year_start = if month >= 10 {
        day - (day_of_year - 306) as i64
    } else {
        let leap = year_of_era % 4 == 0 && (year_of_era % 100 != 0 || year_of_era == 0);
        day - day_of_year as i64 - 59 - leap as i64
    };
    (year_start, month_start)
}

/// Add `duration` to the values of `src` in the window `[src_offset, src_offset +
/// out.len())`, writing into `out`.
///
/// `duration` is first converted to the array's time unit. If it is too large to
/// represent in that unit, the call returns an error and writes nothing. A value whose
/// sum overflows the storage type is marked null in `out_mask`. The allocating
/// `DatetimeArray::add_duration` passes `src_offset = 0` over the whole array.
pub fn add_duration_into<T: Integer + FromPrimitive>(
    src: &DatetimeArray<T>,
    src_offset: usize,
    duration: Duration,
    out: &mut [T],
    mut out_mask: Option<&mut Bitmask>,
) -> Result<(), MinarrowError> {
    let duration_value: i64 = match src.time_unit {
        TimeUnit::Seconds => duration.whole_seconds(),
        TimeUnit::Milliseconds => {
            duration
                .whole_milliseconds()
                .try_into()
                .map_err(|_| MinarrowError::Overflow {
                    value: format!("{} ms", duration.whole_milliseconds()),
                    target: "i64",
                })?
        }
        TimeUnit::Microseconds => {
            duration
                .whole_microseconds()
                .try_into()
                .map_err(|_| MinarrowError::Overflow {
                    value: format!("{} μs", duration.whole_microseconds()),
                    target: "i64",
                })?
        }
        TimeUnit::Nanoseconds => {
            duration
                .whole_nanoseconds()
                .try_into()
                .map_err(|_| MinarrowError::Overflow {
                    value: format!("{} ns", duration.whole_nanoseconds()),
                    target: "i64",
                })?
        }
        TimeUnit::Days => duration.whole_days(),
    };

    for i in 0..out.len() {
        let original = src.data[src_offset + i];
        out[i] = original;
        let valid = if src.is_null(src_offset + i) {
            false
        } else {
            match original
                .to_i64()
                .and_then(|v| v.checked_add(duration_value))
                .and_then(T::from_i64)
            {
                Some(t) => {
                    out[i] = t;
                    true
                }
                None => false,
            }
        };
        if let Some(mask) = out_mask.as_deref_mut() {
            mask.set(i, valid);
        }
    }
    Ok(())
}

/// Add `months` to every value of `src`, writing into `out`.
///
/// A day that does not exist in the destination month is clamped to that month's last
/// day, and time-of-day is preserved. A value whose result is not a valid datetime is
/// marked null in `out_mask`.
pub fn add_months_into<T: Integer + FromPrimitive>(
    src: &DatetimeArray<T>,
    src_offset: usize,
    months: i32,
    out: &mut [T],
    mut out_mask: Option<&mut Bitmask>,
) {
    let time_unit = src.time_unit;
    for i in 0..out.len() {
        let original = src.data[src_offset + i];
        out[i] = original;
        let computed = if src.is_null(src_offset + i) {
            None
        } else {
            original
                .to_i64()
                .and_then(|v| DatetimeArray::<T>::i64_to_datetime(v, time_unit))
                .and_then(|dt| {
                    let date = dt.date();
                    let total_months = date.year() * 12 + (date.month() as i32) - 1 + months;
                    let new_year = total_months / 12;
                    let new_month = (total_months % 12 + 1) as u8;
                    let new_month_enum = time::Month::try_from(new_month).ok()?;
                    let days_in_month = new_month_enum.length(new_year);
                    let day = date.day().min(days_in_month);
                    let new_date =
                        time::Date::from_calendar_date(new_year, new_month_enum, day).ok()?;
                    let new_dt = new_date.with_time(dt.time()).assume_utc();
                    T::from_i64(DatetimeArray::<T>::datetime_to_i64(new_dt, time_unit))
                })
        };
        let valid = match computed {
            Some(t) => {
                out[i] = t;
                true
            }
            None => false,
        };
        if let Some(mask) = out_mask.as_deref_mut() {
            mask.set(i, valid);
        }
    }
}

/// Extract an `i32` calendar field from each value of `src` in the window
/// `[src_offset, src_offset + out.len())`, writing into `out`. A null input or a
/// value that is not a representable datetime writes `0` and clears its `out_mask`
/// bit. The allocating extract methods on `DatetimeArray` pass `src_offset = 0`.
fn extract_i32_into<T, F>(
    src: &DatetimeArray<T>,
    src_offset: usize,
    out: &mut [i32],
    mut out_mask: Option<&mut Bitmask>,
    extract: F,
) where
    T: Integer + FromPrimitive,
    F: Fn(time::OffsetDateTime) -> i32,
{
    let time_unit = src.time_unit;
    for i in 0..out.len() {
        let dt = if src.is_null(src_offset + i) {
            None
        } else {
            src.data[src_offset + i]
                .to_i64()
                .and_then(|v| DatetimeArray::<T>::i64_to_datetime(v, time_unit))
        };
        let valid = match dt {
            Some(dt) => {
                out[i] = extract(dt);
                true
            }
            None => {
                out[i] = 0;
                false
            }
        };
        if let Some(mask) = out_mask.as_deref_mut() {
            mask.set(i, valid);
        }
    }
}

macro_rules! dt_component_into {
    ($name:ident, $doc:literal, $extract:expr) => {
        #[doc = $doc]
        pub fn $name<T: Integer + FromPrimitive>(
            src: &DatetimeArray<T>,
            src_offset: usize,
            out: &mut [i32],
            out_mask: Option<&mut Bitmask>,
        ) {
            extract_i32_into(src, src_offset, out, out_mask, $extract)
        }
    };
}

dt_component_into!(
    year_into,
    "Calendar year of each datetime in the window.",
    |dt| dt.year()
);
dt_component_into!(
    month_into,
    "Month (1-12) of each datetime in the window.",
    |dt| dt.month() as i32
);
dt_component_into!(
    day_into,
    "Day of month (1-31) of each datetime in the window.",
    |dt| dt.day() as i32
);
dt_component_into!(
    hour_into,
    "Hour (0-23) of each datetime in the window.",
    |dt| dt.hour() as i32
);
dt_component_into!(
    minute_into,
    "Minute (0-59) of each datetime in the window.",
    |dt| dt.minute() as i32
);
dt_component_into!(
    second_into,
    "Second (0-59) of each datetime in the window.",
    |dt| dt.second() as i32
);
dt_component_into!(
    weekday_into,
    "Weekday (1=Sunday .. 7=Saturday) of each datetime in the window.",
    |dt| dt.weekday().number_from_sunday() as i32
);
dt_component_into!(
    day_of_year_into,
    "Day of year (1-366) of each datetime in the window.",
    |dt| dt.ordinal() as i32
);
dt_component_into!(
    iso_week_into,
    "ISO week number (1-53) of each datetime in the window.",
    |dt| dt.iso_week() as i32
);
dt_component_into!(
    quarter_into,
    "Quarter (1-4) of each datetime in the window.",
    |dt| ((dt.month() as i32 - 1) / 3) + 1
);
dt_component_into!(
    week_of_year_into,
    "Week of year (0-53, week 0 holds days before the first Sunday) of each datetime in the window.",
    |dt| (dt.ordinal() as i32 + 7 - dt.weekday().number_from_sunday() as i32) / 7
);

/// Evaluate a boolean predicate on each datetime of `src` in the window
/// `[src_offset, src_offset + out_bits.len())`, writing the bit-packed result into
/// `out_bits`. A value that is not a representable datetime clears its `out_mask` bit.
/// Input nulls already arrive in `out_mask`, so they are not re-checked here.
///
/// Public so a caller can supply a predicate the kernels do not name, such as a
/// calendar-config weekend or business-day test.
pub fn extract_bool_into<T, F>(
    src: &DatetimeArray<T>,
    src_offset: usize,
    out_bits: &mut Bitmask,
    mut out_mask: Option<&mut Bitmask>,
    predicate: F,
) where
    T: Integer + FromPrimitive,
    F: Fn(time::OffsetDateTime) -> bool,
{
    let time_unit = src.time_unit;
    for i in 0..out_bits.len() {
        match src.data[src_offset + i]
            .to_i64()
            .and_then(|v| DatetimeArray::<T>::i64_to_datetime(v, time_unit))
        {
            // SAFETY: `i < out_bits.len()`, so the bit index is within the bitmask's
            // capacity, satisfying `set_unchecked`'s precondition.
            Some(dt) => unsafe { out_bits.set_unchecked(i, predicate(dt)) },
            None => {
                if let Some(mask) = out_mask.as_deref_mut() {
                    mask.set(i, false);
                }
            }
        }
    }
}

macro_rules! bool_predicate_into {
    ($name:ident, $doc:literal, $predicate:expr) => {
        #[doc = $doc]
        pub fn $name<T: Integer + FromPrimitive>(
            src: &DatetimeArray<T>,
            src_offset: usize,
            out_bits: &mut Bitmask,
            out_mask: Option<&mut Bitmask>,
        ) {
            extract_bool_into(src, src_offset, out_bits, out_mask, $predicate)
        }
    };
}

bool_predicate_into!(
    is_leap_year_into,
    "Whether each datetime in the window falls in a leap year.",
    |dt| time::util::is_leap_year(dt.year())
);
bool_predicate_into!(
    is_month_start_into,
    "Whether each datetime in the window is the first day of its month.",
    |dt| dt.day() == 1
);
bool_predicate_into!(
    is_month_end_into,
    "Whether each datetime in the window is the last day of its month.",
    |dt| dt.day() == dt.month().length(dt.year())
);
bool_predicate_into!(
    is_year_start_into,
    "Whether each datetime in the window is the first day of its year.",
    |dt| dt.month() == time::Month::January && dt.day() == 1
);
bool_predicate_into!(
    is_year_end_into,
    "Whether each datetime in the window is the last day of its year.",
    |dt| dt.month() == time::Month::December && dt.day() == 31
);

/// Evaluate a boolean predicate over each pair of datetimes from the `lhs`/`rhs`
/// windows, converting each side through its own time unit so mixed units and widths
/// compare correctly. An unrepresentable value on either side clears the `out_mask`
/// bit; input nulls already arrive in `out_mask`.
fn binary_bool_into<T, U, F>(
    lhs: &DatetimeArray<T>,
    lhs_offset: usize,
    rhs: &DatetimeArray<U>,
    rhs_offset: usize,
    out_bits: &mut Bitmask,
    mut out_mask: Option<&mut Bitmask>,
    predicate: F,
) where
    T: Integer + FromPrimitive,
    U: Integer + FromPrimitive,
    F: Fn(time::OffsetDateTime, time::OffsetDateTime) -> bool,
{
    let lhs_unit = lhs.time_unit;
    let rhs_unit = rhs.time_unit;
    for i in 0..out_bits.len() {
        let a = lhs.data[lhs_offset + i]
            .to_i64()
            .and_then(|v| DatetimeArray::<T>::i64_to_datetime(v, lhs_unit));
        let b = rhs.data[rhs_offset + i]
            .to_i64()
            .and_then(|v| DatetimeArray::<U>::i64_to_datetime(v, rhs_unit));
        match (a, b) {
            // SAFETY: `i < out_bits.len()`, within the bitmask's capacity.
            (Some(a), Some(b)) => unsafe { out_bits.set_unchecked(i, predicate(a, b)) },
            _ => {
                if let Some(mask) = out_mask.as_deref_mut() {
                    mask.set(i, false);
                }
            }
        }
    }
}

/// Whether each `lhs` datetime is strictly before the matching `rhs`. See [`binary_bool_into`].
pub fn is_before_into<T: Integer + FromPrimitive, U: Integer + FromPrimitive>(
    lhs: &DatetimeArray<T>,
    lhs_offset: usize,
    rhs: &DatetimeArray<U>,
    rhs_offset: usize,
    out_bits: &mut Bitmask,
    out_mask: Option<&mut Bitmask>,
) {
    binary_bool_into(
        lhs,
        lhs_offset,
        rhs,
        rhs_offset,
        out_bits,
        out_mask,
        |a, b| a < b,
    )
}

/// Whether each `lhs` datetime is strictly after the matching `rhs`. See [`binary_bool_into`].
pub fn is_after_into<T: Integer + FromPrimitive, U: Integer + FromPrimitive>(
    lhs: &DatetimeArray<T>,
    lhs_offset: usize,
    rhs: &DatetimeArray<U>,
    rhs_offset: usize,
    out_bits: &mut Bitmask,
    out_mask: Option<&mut Bitmask>,
) {
    binary_bool_into(
        lhs,
        lhs_offset,
        rhs,
        rhs_offset,
        out_bits,
        out_mask,
        |a, b| a > b,
    )
}

/// Whether each `value` datetime falls within `[start, end]` of the matching windows,
/// each side converted through its own time unit.
pub fn between_into<T, U, V>(
    value: &DatetimeArray<T>,
    value_offset: usize,
    start: &DatetimeArray<U>,
    start_offset: usize,
    end: &DatetimeArray<V>,
    end_offset: usize,
    out_bits: &mut Bitmask,
    mut out_mask: Option<&mut Bitmask>,
) where
    T: Integer + FromPrimitive,
    U: Integer + FromPrimitive,
    V: Integer + FromPrimitive,
{
    let value_unit = value.time_unit;
    let start_unit = start.time_unit;
    let end_unit = end.time_unit;
    for i in 0..out_bits.len() {
        let v = value.data[value_offset + i]
            .to_i64()
            .and_then(|x| DatetimeArray::<T>::i64_to_datetime(x, value_unit));
        let s = start.data[start_offset + i]
            .to_i64()
            .and_then(|x| DatetimeArray::<U>::i64_to_datetime(x, start_unit));
        let e = end.data[end_offset + i]
            .to_i64()
            .and_then(|x| DatetimeArray::<V>::i64_to_datetime(x, end_unit));
        match (v, s, e) {
            // SAFETY: `i < out_bits.len()`, within the bitmask's capacity.
            (Some(v), Some(s), Some(e)) => unsafe { out_bits.set_unchecked(i, v >= s && v <= e) },
            _ => {
                if let Some(mask) = out_mask.as_deref_mut() {
                    mask.set(i, false);
                }
            }
        }
    }
}

/// Difference `lhs - rhs` for each pair in the windows, expressed in `unit`, writing
/// `i64` into `out`. Each side converts through its own time unit. An unrepresentable
/// value on either side writes `0` and clears the `out_mask` bit.
pub fn diff_into<T, U>(
    lhs: &DatetimeArray<T>,
    lhs_offset: usize,
    rhs: &DatetimeArray<U>,
    rhs_offset: usize,
    unit: TimeUnit,
    out: &mut [i64],
    mut out_mask: Option<&mut Bitmask>,
) where
    T: Integer + FromPrimitive,
    U: Integer + FromPrimitive,
{
    let lhs_unit = lhs.time_unit;
    let rhs_unit = rhs.time_unit;
    for i in 0..out.len() {
        let a = lhs.data[lhs_offset + i]
            .to_i64()
            .and_then(|v| DatetimeArray::<T>::i64_to_datetime(v, lhs_unit));
        let b = rhs.data[rhs_offset + i]
            .to_i64()
            .and_then(|v| DatetimeArray::<U>::i64_to_datetime(v, rhs_unit));
        match (a, b) {
            (Some(a), Some(b)) => {
                let d = a - b;
                out[i] = match unit {
                    TimeUnit::Seconds => d.whole_seconds(),
                    TimeUnit::Milliseconds => d.whole_milliseconds() as i64,
                    TimeUnit::Microseconds => d.whole_microseconds() as i64,
                    TimeUnit::Nanoseconds => d.whole_nanoseconds() as i64,
                    TimeUnit::Days => d.whole_days(),
                };
            }
            _ => {
                out[i] = 0;
                if let Some(mask) = out_mask.as_deref_mut() {
                    mask.set(i, false);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `truncate_into` reads the source from `src_offset` for the output slice's
    /// length, leaving rows outside the window untouched. Under the old whole-array
    /// form this either panicked (writing src.len() rows into a shorter slice) or
    /// read the wrong rows.
    #[test]
    fn truncate_into_respects_source_offset() {
        const MICROS_PER_DAY: i64 = 86_400_000_000;
        let vals: [i64; 5] = [
            1_710_000_000_000_000,
            1_710_000_007_000_000,
            1_710_050_000_000_000,
            1_710_086_400_000_000,
            1_710_100_000_000_000,
        ];
        let src = DatetimeArray::<i64>::from_slice(&vals, Some(TimeUnit::Microseconds));
        let mut out = [0i64; 3];
        let mut mask = Bitmask::new_set_all(3, true);
        truncate_into(&src, 2, TimePeriod::Day, &mut out, Some(&mut mask));
        for i in 0..3 {
            assert_eq!(
                out[i],
                (vals[2 + i] / MICROS_PER_DAY) * MICROS_PER_DAY,
                "window row {i} reads src[2 + {i}] and floors to day start"
            );
            assert!(mask.get(i));
        }
    }
}

#[cfg(test)]
mod floor_tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::time::{Duration as StdDuration, Instant};

    use super::*;
    use crate::Vec64;

    /// Calendar periods tested against the `time::OffsetDateTime` reference.
    const PERIODS: [TimePeriod; 7] = [
        TimePeriod::Year,
        TimePeriod::Month,
        TimePeriod::Week,
        TimePeriod::Day,
        TimePeriod::Hour,
        TimePeriod::Minute,
        TimePeriod::Second,
    ];

    /// Truncation through `time::OffsetDateTime`, as `truncate_into` ran the
    /// calendar periods before the integer path, kept as the reference for the
    /// exact comparison.
    fn truncate_reference<T: Integer + FromPrimitive>(
        src: &DatetimeArray<T>,
        src_offset: usize,
        period: TimePeriod,
        out: &mut [T],
        mut out_mask: Option<&mut Bitmask>,
    ) {
        let time_unit = src.time_unit;
        let len = out.len();
        let trunc: fn(time::OffsetDateTime) -> Option<time::OffsetDateTime> = match period {
            TimePeriod::Year => |dt| {
                Date::from_calendar_date(dt.year(), time::Month::January, 1)
                    .ok()
                    .and_then(|d| d.with_hms(0, 0, 0).ok())
                    .map(|pdt| pdt.assume_utc())
            },
            TimePeriod::Month => |dt| {
                Date::from_calendar_date(dt.year(), dt.month(), 1)
                    .ok()
                    .and_then(|d| d.with_hms(0, 0, 0).ok())
                    .map(|pdt| pdt.assume_utc())
            },
            TimePeriod::Week => |dt| {
                let days_to_sunday = (dt.weekday().number_from_sunday() - 1) as i64;
                dt.checked_sub(time::Duration::days(days_to_sunday))
                    .and_then(|week_start| week_start.date().with_hms(0, 0, 0).ok())
                    .map(|pdt| pdt.assume_utc())
            },
            TimePeriod::Day => {
                |dt| dt.date().with_hms(0, 0, 0).ok().map(|pdt| pdt.assume_utc())
            }
            TimePeriod::Hour => |dt| {
                dt.date()
                    .with_hms(dt.hour(), 0, 0)
                    .ok()
                    .map(|pdt| pdt.assume_utc())
            },
            TimePeriod::Minute => |dt| {
                dt.date()
                    .with_hms(dt.hour(), dt.minute(), 0)
                    .ok()
                    .map(|pdt| pdt.assume_utc())
            },
            TimePeriod::Second => |dt| {
                dt.date()
                    .with_hms(dt.hour(), dt.minute(), dt.second())
                    .ok()
                    .map(|pdt| pdt.assume_utc())
            },
            TimePeriod::Millisecond | TimePeriod::Microsecond => {
                unreachable!("sub-second periods have no calendar reference")
            }
        };
        for i in 0..len {
            let original = src.data[src_offset + i];
            out[i] = original;
            let valid = if src.is_null(src_offset + i) {
                false
            } else {
                match original
                    .to_i64()
                    .and_then(|v| DatetimeArray::<T>::i64_to_datetime(v, time_unit))
                    .and_then(trunc)
                    .map(|dt| DatetimeArray::<T>::datetime_to_i64(dt, time_unit))
                    .and_then(T::from_i64)
                {
                    Some(t) => {
                        out[i] = t;
                        true
                    }
                    None => false,
                }
            };
            if let Some(mask) = out_mask.as_deref_mut() {
                mask.set(i, valid);
            }
        }
    }

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    const UNITS: [TimeUnit; 5] = [
        TimeUnit::Seconds,
        TimeUnit::Milliseconds,
        TimeUnit::Microseconds,
        TimeUnit::Nanoseconds,
        TimeUnit::Days,
    ];

    fn units_per_day(unit: TimeUnit) -> i64 {
        match unit {
            TimeUnit::Seconds => 86_400,
            TimeUnit::Milliseconds => 86_400_000,
            TimeUnit::Microseconds => 86_400_000_000,
            TimeUnit::Nanoseconds => 86_400_000_000_000,
            TimeUnit::Days => 1,
        }
    }

    /// Values at and around the epoch, Sunday week starts, the `time` crate's
    /// range limits, the `i32` Julian day wrap of the `Days` unit and the ends
    /// of the `i64` and `i32` ranges.
    fn boundary_values(unit: TimeUnit) -> Vec<i64> {
        let per_day = units_per_day(unit) as i128;
        let min_day = Date::MIN.to_julian_day() as i128 - 2_440_588;
        let max_day = Date::MAX.to_julian_day() as i128 - 2_440_588;
        let mut anchors: Vec<i128> = vec![
            0,
            // 1969-12-28 and 1970-01-04, the Sundays around the epoch.
            -4 * per_day,
            3 * per_day,
            min_day * per_day,
            (max_day + 1) * per_day,
            i64::MIN as i128,
            i64::MAX as i128,
            i32::MIN as i128,
            i32::MAX as i128,
        ];
        for k in -15..15 {
            anchors.push(k * per_day);
        }
        for k in 0..8 {
            anchors.push((min_day + k) * per_day);
            anchors.push((max_day - k) * per_day);
        }
        if matches!(unit, TimeUnit::Days) {
            // The Julian day conversion wraps through `i32`.
            for base in [1i128 << 31, 1i128 << 32, -(1i128 << 31), -(1i128 << 32), 5i128 << 32] {
                anchors.push(base - 2_440_588);
                anchors.push(base - 2_440_588 + min_day + 2_440_588);
                anchors.push(base + max_day);
            }
            anchors.push(i64::MAX as i128 - 2_440_588);
        }
        let mut out = Vec::new();
        for a in anchors {
            for d in [-per_day - 1, -per_day, -2, -1, 0, 1, 2, per_day - 1, per_day] {
                let v = a + d;
                if v >= i64::MIN as i128 && v <= i64::MAX as i128 {
                    out.push(v as i64);
                }
            }
        }
        out
    }

    /// Runs both forms over the window, returning `None` when the form panics.
    fn run_both<T: Integer + FromPrimitive + std::fmt::Debug>(
        src: &DatetimeArray<T>,
        offset: usize,
        len: usize,
        period: TimePeriod,
        with_mask: bool,
    ) -> (Option<(Vec<T>, Option<Bitmask>)>, Option<(Vec<T>, Option<Bitmask>)>) {
        let run = |reference: bool| {
            catch_unwind(AssertUnwindSafe(|| {
                let mut out = vec![T::zero(); len];
                let mut mask = with_mask.then(|| Bitmask::new_set_all(len, true));
                if reference {
                    truncate_reference(src, offset, period, &mut out, mask.as_mut());
                } else {
                    truncate_into(src, offset, period, &mut out, mask.as_mut());
                }
                (out, mask)
            }))
            .ok()
        };
        (run(false), run(true))
    }

    fn assert_same<T: Integer + FromPrimitive + std::fmt::Debug>(
        src: &DatetimeArray<T>,
        offset: usize,
        len: usize,
        context: &str,
    ) {
        for period in PERIODS {
            for with_mask in [true, false] {
                let (new, old) = run_both(src, offset, len, period, with_mask);
                let context = format!("{period:?} {context}");
                let (new_out, new_mask) =
                    new.unwrap_or_else(|| panic!("integer path panics: {context}"));
                // The reference panics where scaling the result overflows `i64`,
                // while the integer path marks that slot null. See
                // `overflowing_floor_marks_null`.
                if let Some((old_out, old_mask)) = old {
                    assert_eq!(new_out, old_out, "values differ: {context}");
                    assert_eq!(new_mask, old_mask, "mask differs: {context}");
                }
            }
        }
    }

    #[test]
    fn floor_matches_reference_at_boundaries() {
        for unit in UNITS {
            for v in boundary_values(unit) {
                let src = DatetimeArray::<i64>::from_slice(&[v], Some(unit));
                assert_same(&src, 0, 1, &format!("i64 unit={unit:?} v={v}"));
                if let Ok(v32) = i32::try_from(v) {
                    let src = DatetimeArray::<i32>::from_slice(&[v32], Some(unit));
                    assert_same(&src, 0, 1, &format!("i32 unit={unit:?} v={v32}"));
                }
            }
        }
    }

    #[test]
    fn floor_matches_reference_across_i64_range() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for unit in UNITS {
            let per_day = units_per_day(unit);
            for scenario in 0..3 {
                let values: Vec<i64> = (0..4000)
                    .map(|_| match scenario {
                        // The full i64 range.
                        0 => rng.next() as i64,
                        // Within roughly 270 years of the epoch.
                        1 => (rng.next() as i64 % 100_000) * per_day
                            + (rng.next() % per_day as u64) as i64,
                        // Within a few weeks of the epoch.
                        _ => rng.next() as i64 % (40 * per_day),
                    })
                    .collect();
                for density in [0u64, 5, 50, 100] {
                    let mask = (density < 100).then(|| {
                        let mut m = Bitmask::new_set_all(values.len(), true);
                        for i in 0..values.len() {
                            if rng.next() % 100 < density {
                                m.set(i, false);
                            }
                        }
                        m
                    });
                    let src = DatetimeArray::<i64>::new(Vec64::from_slice(&values), mask, Some(unit));
                    let n = values.len();
                    for (offset, len) in [(0, n), (3, n - 3), (65, 1000), (n, 0)] {
                        assert_same(
                            &src,
                            offset,
                            len,
                            &format!("unit={unit:?} scenario={scenario} density={density} offset={offset}"),
                        );
                    }
                    let values32: Vec<i32> = values.iter().map(|v| *v as i32).collect();
                    let mask32 = src.null_mask.clone();
                    let src32 =
                        DatetimeArray::<i32>::new(Vec64::from_slice(&values32), mask32, Some(unit));
                    assert_same(&src32, 0, n, &format!("i32 unit={unit:?} scenario={scenario}"));
                }
            }
        }
    }

    #[test]
    fn floor_large_completes_within_bound() {
        let len = 1usize << 24;
        let values: Vec<i64> = (0..len as i64).map(|i| i * 61_000).collect();
        let src = DatetimeArray::<i64>::from_slice(&values, Some(TimeUnit::Milliseconds));
        let mut out = vec![0i64; len];
        let mut mask = Bitmask::new_set_all(len, true);
        let start = Instant::now();
        truncate_into(&src, 0, TimePeriod::Week, &mut out, Some(&mut mask));
        assert!(start.elapsed() < StdDuration::from_secs(60));
        assert_eq!(out[len - 1] % (7 * 86_400_000), 3 * 86_400_000);
    }

    #[test]
    fn sub_second_floors_toward_negative_infinity() {
        let src = DatetimeArray::<i64>::from_slice(&[-1_500, 1_500], Some(TimeUnit::Microseconds));
        let mut out = vec![0i64; 2];
        truncate_into(&src, 0, TimePeriod::Millisecond, &mut out, None);
        assert_eq!(out, vec![-2_000, 1_000]);
    }

    #[test]
    fn overflowing_floor_marks_null() {
        // Every period floors `i64::MIN` and `i64::MIN + 1` nanoseconds below the
        // `i64` range.
        let values = [i64::MIN, i64::MIN + 1, 0];
        let src = DatetimeArray::<i64>::from_slice(&values, Some(TimeUnit::Nanoseconds));
        let periods = PERIODS
            .into_iter()
            .chain([TimePeriod::Millisecond, TimePeriod::Microsecond]);
        for period in periods {
            let mut out = vec![0i64; values.len()];
            let mut mask = Bitmask::new_set_all(values.len(), true);
            truncate_into(&src, 0, period, &mut out, Some(&mut mask));
            assert!(!mask.get(0), "{period:?} i64::MIN is null");
            assert!(!mask.get(1), "{period:?} i64::MIN + 1 is null");
            assert_eq!(out[..2], values[..2], "{period:?} null slots keep the input");
            assert!(mask.get(2), "{period:?} the epoch is valid");
        }
    }
}
