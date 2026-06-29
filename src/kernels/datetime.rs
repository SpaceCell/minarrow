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
use time::Duration;

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

    // Calendar periods floor through a datetime. Sub-second periods divide the raw
    // value instead, since the datetime path cannot express them directly.
    let calendar: Option<fn(time::OffsetDateTime) -> Option<time::OffsetDateTime>> = match period {
        TimePeriod::Year => Some(|dt| {
            time::Date::from_calendar_date(dt.year(), time::Month::January, 1)
                .ok()
                .and_then(|d| d.with_hms(0, 0, 0).ok())
                .map(|pdt| pdt.assume_utc())
        }),
        TimePeriod::Month => Some(|dt| {
            time::Date::from_calendar_date(dt.year(), dt.month(), 1)
                .ok()
                .and_then(|d| d.with_hms(0, 0, 0).ok())
                .map(|pdt| pdt.assume_utc())
        }),
        TimePeriod::Week => Some(|dt| {
            // Weekday is 1=Sunday .. 7=Saturday, so step back to Sunday.
            let days_to_sunday = (dt.weekday().number_from_sunday() - 1) as i64;
            dt.checked_sub(time::Duration::days(days_to_sunday))
                .and_then(|week_start| week_start.date().with_hms(0, 0, 0).ok())
                .map(|pdt| pdt.assume_utc())
        }),
        TimePeriod::Day => Some(|dt| dt.date().with_hms(0, 0, 0).ok().map(|pdt| pdt.assume_utc())),
        TimePeriod::Hour => {
            Some(|dt| dt.date().with_hms(dt.hour(), 0, 0).ok().map(|pdt| pdt.assume_utc()))
        }
        TimePeriod::Minute => Some(|dt| {
            dt.date().with_hms(dt.hour(), dt.minute(), 0).ok().map(|pdt| pdt.assume_utc())
        }),
        TimePeriod::Second => Some(|dt| {
            dt.date()
                .with_hms(dt.hour(), dt.minute(), dt.second())
                .ok()
                .map(|pdt| pdt.assume_utc())
        }),
        TimePeriod::Millisecond | TimePeriod::Microsecond => None,
    };

    if let Some(trunc) = calendar {
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
        return;
    }

    // Sub-second flooring. The divisor exists only when the stored resolution is
    // finer than the target - a coarser array has no sub-target detail to remove
    // and passes through unchanged.
    let divisor: Option<i64> = match (period, time_unit) {
        (TimePeriod::Microsecond, TimeUnit::Nanoseconds) => Some(1_000),
        (TimePeriod::Millisecond, TimeUnit::Nanoseconds) => Some(1_000_000),
        (TimePeriod::Millisecond, TimeUnit::Microseconds) => Some(1_000),
        _ => None,
    };
    for i in 0..len {
        let original = src.data[src_offset + i];
        out[i] = original;
        let valid = if src.is_null(src_offset + i) {
            false
        } else {
            match divisor {
                None => true,
                Some(divisor) => {
                    match original.to_i64().map(|v| (v / divisor) * divisor).and_then(T::from_i64) {
                        Some(t) => {
                            out[i] = t;
                            true
                        }
                        None => false,
                    }
                }
            }
        };
        if let Some(mask) = out_mask.as_deref_mut() {
            mask.set(i, valid);
        }
    }
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
            duration.whole_milliseconds().try_into().map_err(|_| MinarrowError::Overflow {
                value: format!("{} ms", duration.whole_milliseconds()),
                target: "i64",
            })?
        }
        TimeUnit::Microseconds => {
            duration.whole_microseconds().try_into().map_err(|_| MinarrowError::Overflow {
                value: format!("{} μs", duration.whole_microseconds()),
                target: "i64",
            })?
        }
        TimeUnit::Nanoseconds => {
            duration.whole_nanoseconds().try_into().map_err(|_| MinarrowError::Overflow {
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
            match original.to_i64().and_then(|v| v.checked_add(duration_value)).and_then(T::from_i64)
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

dt_component_into!(year_into, "Calendar year of each datetime in the window.", |dt| dt.year());
dt_component_into!(month_into, "Month (1-12) of each datetime in the window.", |dt| dt.month() as i32);
dt_component_into!(day_into, "Day of month (1-31) of each datetime in the window.", |dt| dt.day() as i32);
dt_component_into!(hour_into, "Hour (0-23) of each datetime in the window.", |dt| dt.hour() as i32);
dt_component_into!(minute_into, "Minute (0-59) of each datetime in the window.", |dt| dt.minute() as i32);
dt_component_into!(second_into, "Second (0-59) of each datetime in the window.", |dt| dt.second() as i32);
dt_component_into!(
    weekday_into,
    "Weekday (1=Sunday .. 7=Saturday) of each datetime in the window.",
    |dt| dt.weekday().number_from_sunday() as i32
);
dt_component_into!(day_of_year_into, "Day of year (1-366) of each datetime in the window.", |dt| dt.ordinal() as i32);
dt_component_into!(iso_week_into, "ISO week number (1-53) of each datetime in the window.", |dt| dt.iso_week() as i32);
dt_component_into!(quarter_into, "Quarter (1-4) of each datetime in the window.", |dt| ((dt.month() as i32 - 1) / 3) + 1);
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
    binary_bool_into(lhs, lhs_offset, rhs, rhs_offset, out_bits, out_mask, |a, b| a < b)
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
    binary_bool_into(lhs, lhs_offset, rhs, rhs_offset, out_bits, out_mask, |a, b| a > b)
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
