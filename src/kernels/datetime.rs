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

/// Floor each value of `src` to the start of `period`, writing into `out`.
///
/// Handles the full calendar range (`Year` through `Second`, plus `Week`) and the
/// sub-second steps (`Millisecond`, `Microsecond`). Sub-second steps are no-ops on
/// arrays whose stored resolution is already at or coarser than the target.
pub fn truncate_into<T: Integer + FromPrimitive>(
    src: &DatetimeArray<T>,
    period: TimePeriod,
    out: &mut [T],
    mut out_mask: Option<&mut Bitmask>,
) {
    let time_unit = src.time_unit;

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
        for i in 0..src.len() {
            let original = src.data[i];
            out[i] = original;
            let valid = if src.is_null(i) {
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
    for i in 0..src.len() {
        let original = src.data[i];
        out[i] = original;
        let valid = if src.is_null(i) {
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

/// Add `duration` to every value of `src`, writing into `out`.
///
/// `duration` is first converted to the array's time unit. If it is too large to
/// represent in that unit, the call returns an error and writes nothing. A value whose
/// sum overflows the storage type is marked null in `out_mask`.
pub fn add_duration_into<T: Integer + FromPrimitive>(
    src: &DatetimeArray<T>,
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

    for i in 0..src.len() {
        let original = src.data[i];
        out[i] = original;
        let valid = if src.is_null(i) {
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
    months: i32,
    out: &mut [T],
    mut out_mask: Option<&mut Bitmask>,
) {
    let time_unit = src.time_unit;
    for i in 0..src.len() {
        let original = src.data[i];
        out[i] = original;
        let computed = if src.is_null(i) {
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
