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

//! # **Print Module** - *Pretty Printing with Attitude*
//!
//! Contains implementations of the Display trait
//! and an additional `Print` trait which wraps it to provide
//! `myobj.print()` for any object that implements it.
use std::fmt::{self, Display, Formatter};

use crate::{Array, Buffer, Float, NumericArray, TextArray};
#[cfg(feature = "datetime")]
use crate::{DatetimeArray, Integer, TemporalArray};

pub(crate) const MAX_PREVIEW: usize = 50;

/// # Print
///
/// Loaded print trait for pretty printing tables
///
/// Provides a more convenient way to activate `Display`
/// for other types such as arrays via `myarr.print()`,
/// avoiding the need to write `println!("{}", myarr);`
pub trait Print {
    #[inline]
    fn print(&self)
    where
        Self: Display,
    {
        println!("{}", self);
    }
}

impl<T: Display> Print for T where T: Display {}

// Helper functions

pub(crate) fn value_to_string(arr: &Array, idx: usize) -> String {
    // Null checks (handles absent mask too)
    if let Some(mask) = arr.null_mask() {
        if !mask.get(idx) {
            return "null".into();
        }
    }
    match arr {
        // ------------------------- numeric ------------------------------
        Array::NumericArray(inner) => match inner {
            NumericArray::Int32(a) => a.data[idx].to_string(),
            NumericArray::Int64(a) => a.data[idx].to_string(),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(a) => a.data[idx].to_string(),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(a) => a.data[idx].to_string(),
            NumericArray::UInt32(a) => a.data[idx].to_string(),
            NumericArray::UInt64(a) => a.data[idx].to_string(),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(a) => a.data[idx].to_string(),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(a) => a.data[idx].to_string(),
            NumericArray::Float32(a) => format_float(a.data[idx] as f64),
            NumericArray::Float64(a) => format_float(a.data[idx]),
            NumericArray::Null => "null".into(),
        },
        // ------------------------- boolean ------------------------------
        Array::BooleanArray(b) => {
            let bit = b.data.get(idx);
            bit.to_string()
        }
        // ------------------------- string / categorical -----------------
        Array::TextArray(inner) => match inner {
            TextArray::String32(s) => string_value(&s.offsets, &s.data, idx),
            #[cfg(feature = "large_string")]
            TextArray::String64(s) => string_value(&s.offsets, &s.data, idx),
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            TextArray::Categorical32(cat) => {
                let key = cat.data[idx] as usize;
                cat.unique_values()[key].clone()
            }
            #[cfg(feature = "default_categorical_8")]
            TextArray::Categorical8(cat) => {
                let key = cat.data[idx] as usize;
                cat.unique_values()[key].clone()
            }
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical16(cat) => {
                let key = cat.data[idx] as usize;
                cat.unique_values()[key].clone()
            }
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical64(cat) => {
                let key = cat.data[idx] as usize;
                cat.unique_values()[key].clone()
            }
            TextArray::Null => "null".into(),
        },
        // ------------------------- datetime -----------------------------
        #[cfg(feature = "datetime")]
        Array::TemporalArray(inner) => match inner {
            TemporalArray::Datetime32(dt) => format_datetime_value(dt, idx, None),
            TemporalArray::Datetime64(dt) => format_datetime_value(dt, idx, None),
            TemporalArray::Null => "null".into(),
        },
        // ------------------------- fallback -----------------------------
        Array::Null => "null".into(),
    }
}

fn string_value<T: Copy>(offsets: &Buffer<T>, data: &Buffer<u8>, idx: usize) -> String
where
    T: Copy + Into<u64>,
{
    // Convert to u64, then to usize (explicitly)
    let start = offsets[idx].into() as usize;
    let end = offsets[idx + 1].into() as usize;
    let slice = &data[start..end];

    // Safety: Arrow guarantees valid UTF-8 encoding
    let s = unsafe { std::str::from_utf8_unchecked(slice) };
    s.to_string()
}

pub(crate) fn print_rule(
    f: &mut Formatter<'_>,
    idx_width: usize,
    col_widths: &[usize],
) -> fmt::Result {
    write!(f, "+{:-<w$}+", "", w = idx_width + 2)?; // idx column (+2 for spaces)
    for &w in col_widths {
        write!(f, "{:-<w$}+", "", w = w + 2)?; // +2 for spaces
    }
    writeln!(f)
}

pub(crate) fn print_header_row(
    f: &mut Formatter<'_>,
    idx_width: usize,
    headers: &[String],
    col_widths: &[usize],
) -> fmt::Result {
    write!(f, "| {hdr:^w$} |", hdr = "idx", w = idx_width)?;
    for (hdr, &w) in headers.iter().zip(col_widths) {
        write!(f, " {hdr:^w$} |", hdr = hdr, w = w)?;
    }
    writeln!(f)
}

pub(crate) fn print_ellipsis_row(
    f: &mut Formatter<'_>,
    idx_width: usize,
    col_widths: &[usize],
) -> fmt::Result {
    write!(f, "| {dots:^w$} |", dots = "…", w = idx_width)?;
    for &w in col_widths {
        write!(f, " {dots:^w$} |", dots = "…", w = w)?;
    }
    writeln!(f)
}

/// Formats floating point numbers:
/// - Keeps up to 6 decimal digits
/// - Trims trailing zeroes and unnecessary decimal point
#[inline]
pub(crate) fn format_float<T: Float + Display>(v: T) -> String {
    let s = format!("{:.6}", v);
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

#[cfg(feature = "datetime")]
pub(crate) fn format_datetime_value<T>(
    arr: &DatetimeArray<T>,
    idx: usize,
    timezone: Option<&str>,
) -> String
where
    T: Integer + std::fmt::Display,
{
    use crate::MaskedArray;
    if arr.is_null(idx) {
        return "null".into();
    }

    #[cfg(feature = "datetime_ops")]
    {
        use crate::TimeUnit;
        use time::OffsetDateTime;

        let utc_dt = match arr.time_unit {
            TimeUnit::Seconds => {
                let secs = arr.data[idx].to_i64().unwrap();
                OffsetDateTime::from_unix_timestamp(secs).ok()
            }
            TimeUnit::Milliseconds => {
                let v = arr.data[idx].to_i64().unwrap();
                OffsetDateTime::from_unix_timestamp_nanos((v as i128) * 1_000_000).ok()
            }
            TimeUnit::Microseconds => {
                let v = arr.data[idx].to_i64().unwrap();
                OffsetDateTime::from_unix_timestamp_nanos((v as i128) * 1_000).ok()
            }
            TimeUnit::Nanoseconds => {
                let v = arr.data[idx].to_i64().unwrap();
                OffsetDateTime::from_unix_timestamp_nanos(v as i128).ok()
            }
            TimeUnit::Days => {
                use crate::structs::variants::datetime::UNIX_EPOCH_JULIAN_DAY;
                let days = arr.data[idx].to_i64().unwrap();
                time::Date::from_julian_day((days + UNIX_EPOCH_JULIAN_DAY) as i32)
                    .ok()
                    .and_then(|d| d.with_hms(0, 0, 0).ok())
                    .map(|dt| dt.assume_utc())
            }
        };

        if let Some(dt) = utc_dt {
            if let Some(tz) = timezone {
                format_with_timezone(dt, tz)
            } else {
                dt.to_string()
            }
        } else {
            let v = arr.data[idx];
            let suffix = match arr.time_unit {
                TimeUnit::Seconds => "s",
                TimeUnit::Milliseconds => "ms",
                TimeUnit::Microseconds => "µs",
                TimeUnit::Nanoseconds => "ns",
                TimeUnit::Days => "d",
            };
            format!("{v}{suffix}")
        }
    }
    #[cfg(not(feature = "datetime_ops"))]
    {
        use crate::TimeUnit;

        if timezone.is_some() {
            panic!(
                "Timezone functionality requires the 'datetime_ops' feature. \
                Enable it in Cargo.toml with: features = [\"datetime_ops\"]"
            );
        }

        let v = arr.data[idx];
        let suffix = match arr.time_unit {
            TimeUnit::Seconds => "s",
            TimeUnit::Milliseconds => "ms",
            TimeUnit::Microseconds => "µs",
            TimeUnit::Nanoseconds => "ns",
            TimeUnit::Days => "d",
        };
        format!("{v}{suffix}")
    }
}

#[cfg(all(feature = "datetime", feature = "datetime_ops"))]
fn format_with_timezone(utc_dt: time::OffsetDateTime, tz: &str) -> String {
    // Try to parse as offset string first (e.g., "+05:00", "-08:00")
    if let Some(offset) = parse_timezone_offset(tz) {
        let local_dt = utc_dt.to_offset(offset);
        format!("{} {}", local_dt, tz)
    } else {
        // For IANA timezones (e.g., "America/New_York"), we can't do full conversion
        // without a timezone database. Just append the timezone name.
        format!("{} {}", utc_dt, tz)
    }
}

#[cfg(all(feature = "datetime", feature = "datetime_ops"))]
fn parse_timezone_offset(tz: &str) -> Option<time::UtcOffset> {
    use crate::structs::variants::datetime::tz::lookup_timezone;
    use time::UtcOffset;

    // First try timezone database lookup (handles IANA IDs, abbreviations, and direct offsets)
    let tz_offset = lookup_timezone(tz)?;

    // Now parse the resolved offset string
    let tz = tz_offset.trim();

    // Handle UTC specially
    if tz.eq_ignore_ascii_case("UTC") || tz.eq_ignore_ascii_case("Z") {
        return Some(UtcOffset::UTC);
    }

    // Parse offset strings like "+05:00", "-08:00", "+0530"
    if !tz.starts_with('+') && !tz.starts_with('-') {
        return None;
    }

    let (sign, rest) = tz.split_at(1);
    let sign = if sign == "+" { 1 } else { -1 };

    // Try parsing HH:MM format
    if let Some((hours_str, mins_str)) = rest.split_once(':') {
        let hours: i8 = hours_str.parse().ok()?;
        let mins: i8 = mins_str.parse().ok()?;
        let seconds = sign * (hours as i32 * 3600 + mins as i32 * 60);
        return UtcOffset::from_whole_seconds(seconds).ok();
    }

    // Try parsing HHMM format
    if rest.len() == 4 {
        let hours: i8 = rest[0..2].parse().ok()?;
        let mins: i8 = rest[2..4].parse().ok()?;
        let seconds = sign * (hours as i32 * 3600 + mins as i32 * 60);
        return UtcOffset::from_whole_seconds(seconds).ok();
    }

    None
}
