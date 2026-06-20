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

//! # **Datetime Parse Module** - *Wire-Format Timestamp Parsing for DatetimeArray*
//!
//! Parses ISO 8601 / RFC 3339 timestamp strings into the integer values a
//! [`DatetimeArray`](crate::DatetimeArray) stores, scaled to the array's
//! [`TimeUnit`]. Built for live ingest paths: one fixed-position pass over
//! the bytes, allocation-free, independent of the `time` crate.

use crate::enums::time_units::TimeUnit;

/// Reads `width` ASCII decimal digits at `at` as an integer, returning `None`
/// if any byte in the range is not a digit.
#[inline(always)]
fn ascii_decimal(bytes: &[u8], at: usize, width: usize) -> Option<i64> {
    let mut value = 0;
    for &byte in &bytes[at..at + width] {
        let digit = byte.wrapping_sub(b'0');
        if digit > 9 {
            return None;
        }
        value = value * 10 + digit as i64;
    }
    Some(value)
}

/// Days from the Unix epoch to the given civil date, via Howard Hinnant's
/// `days_from_civil` algorithm: pure integer arithmetic over 400-year eras,
/// exact for all proleptic Gregorian dates.
#[inline(always)]
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = if month > 2 { month - 3 } else { month + 9 };
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Days in the given month, leap-aware.
#[inline(always)]
fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
            if leap { 29 } else { 28 }
        }
    }
}

/// Parses an ISO 8601 / RFC 3339 timestamp string into nanoseconds since
/// the Unix epoch.
///
/// Accepts `YYYY-MM-DDTHH:MM:SS` with a `T`, `t` or space separator, an
/// optional fractional-second part of up to nanosecond precision (finer
/// digits truncate), and an optional `Z`, `z` or `±HH:MM` offset. A
/// missing offset reads as UTC. Returns `None` on any malformed input.
///
/// The `i64` nanosecond range covers 1677-09-21 through 2262-04-11;
/// dates outside it return `None`. For values destined for a
/// [`DatetimeArray`](crate::DatetimeArray), [`parse_iso8601_utc`] scales
/// to the array's [`TimeUnit`] directly.
pub fn parse_iso8601_utc_ns(s: &str) -> Option<i64> {
    let bytes = s.as_bytes();
    // "YYYY-MM-DDTHH:MM:SS" is 19 bytes.
    if bytes.len() < 19 {
        return None;
    }

    let year = ascii_decimal(bytes, 0, 4)?;
    if bytes[4] != b'-' {
        return None;
    }
    let month = ascii_decimal(bytes, 5, 2)?;
    if bytes[7] != b'-' {
        return None;
    }
    let day = ascii_decimal(bytes, 8, 2)?;
    if !matches!(bytes[10], b'T' | b't' | b' ') {
        return None;
    }
    let hour = ascii_decimal(bytes, 11, 2)?;
    if bytes[13] != b':' {
        return None;
    }
    let minute = ascii_decimal(bytes, 14, 2)?;
    if bytes[16] != b':' {
        return None;
    }
    let second = ascii_decimal(bytes, 17, 2)?;

    // Leap seconds arrive as :60 on the wire and roll into the next
    // minute through the epoch arithmetic below.
    if !(1..=12).contains(&month)
        || !(1..=days_in_month(year, month)).contains(&day)
        || hour > 23
        || minute > 59
        || second > 60
    {
        return None;
    }

    // Fractional seconds: read up to nanosecond precision, truncate the rest.
    let mut cursor = 19;
    let mut frac_ns: i64 = 0;
    if cursor < bytes.len() && bytes[cursor] == b'.' {
        cursor += 1;
        let frac_start = cursor;
        let mut value: i64 = 0;
        while cursor < bytes.len() && bytes[cursor].is_ascii_digit() && cursor - frac_start < 9 {
            value = value * 10 + (bytes[cursor] - b'0') as i64;
            cursor += 1;
        }
        if cursor == frac_start {
            return None;
        }
        let mut read = cursor - frac_start;
        while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
            cursor += 1;
        }
        while read < 9 {
            value *= 10;
            read += 1;
        }
        frac_ns = value;
    }

    // Offset: Z, ±HH:MM, or absent meaning UTC.
    let offset_seconds: i64 = match bytes.get(cursor) {
        None => 0,
        Some(b'Z' | b'z') if cursor + 1 == bytes.len() => 0,
        Some(sign @ (b'+' | b'-')) => {
            if cursor + 6 != bytes.len() || bytes[cursor + 3] != b':' {
                return None;
            }
            let offset_hour = ascii_decimal(bytes, cursor + 1, 2)?;
            let offset_minute = ascii_decimal(bytes, cursor + 4, 2)?;
            if offset_hour > 23 || offset_minute > 59 {
                return None;
            }
            let seconds = offset_hour * 3_600 + offset_minute * 60;
            if *sign == b'-' { -seconds } else { seconds }
        }
        _ => return None,
    };

    let days = days_from_civil(year, month, day);
    let epoch_seconds = days * 86_400 + hour * 3_600 + minute * 60 + second - offset_seconds;
    epoch_seconds.checked_mul(1_000_000_000)?.checked_add(frac_ns)
}

/// Parses an ISO 8601 / RFC 3339 timestamp string into the value a
/// [`DatetimeArray`](crate::DatetimeArray) with the given [`TimeUnit`]
/// stores.
///
/// Scales the parsed nanoseconds down to `unit`, truncating digits finer
/// than the unit. See [`parse_iso8601_utc_ns`] for the accepted formats.
pub fn parse_iso8601_utc(s: &str, unit: TimeUnit) -> Option<i64> {
    let ns = parse_iso8601_utc_ns(s)?;
    Some(match unit {
        TimeUnit::Seconds => ns / 1_000_000_000,
        TimeUnit::Milliseconds => ns / 1_000_000,
        TimeUnit::Microseconds => ns / 1_000,
        TimeUnit::Nanoseconds => ns,
        TimeUnit::Days => ns / 86_400_000_000_000,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DatetimeArray;

    // 2026-06-09T07:48:36Z is 1_780_991_316 seconds since the Unix epoch.
    const EPOCH_SECONDS: i64 = 1_780_991_316;

    #[test]
    fn parses_kraken_microsecond_timestamp() {
        let ns = parse_iso8601_utc_ns("2026-06-09T07:48:36.123456Z").unwrap();
        assert_eq!(ns, EPOCH_SECONDS * 1_000_000_000 + 123_456_000);
    }

    #[test]
    fn parses_whole_seconds_and_naive_as_utc() {
        assert_eq!(
            parse_iso8601_utc_ns("2026-06-09T07:48:36Z").unwrap(),
            EPOCH_SECONDS * 1_000_000_000,
        );
        // Missing offset reads as UTC; space separator accepted.
        assert_eq!(
            parse_iso8601_utc_ns("2026-06-09 07:48:36").unwrap(),
            EPOCH_SECONDS * 1_000_000_000,
        );
    }

    #[test]
    fn applies_numeric_offsets() {
        // 17:48:36 at +10:00 is 07:48:36 UTC.
        assert_eq!(
            parse_iso8601_utc_ns("2026-06-09T17:48:36+10:00").unwrap(),
            EPOCH_SECONDS * 1_000_000_000,
        );
        // 02:18:36 at -05:30 is 07:48:36 UTC.
        assert_eq!(
            parse_iso8601_utc_ns("2026-06-09T02:18:36-05:30").unwrap(),
            EPOCH_SECONDS * 1_000_000_000,
        );
    }

    #[test]
    fn truncates_fractional_digits_beyond_nanoseconds() {
        let ns = parse_iso8601_utc_ns("2026-06-09T07:48:36.1234567899Z").unwrap();
        assert_eq!(ns, EPOCH_SECONDS * 1_000_000_000 + 123_456_789);
    }

    #[test]
    fn rejects_malformed_input() {
        for s in [
            "",
            "not a datetime",
            "2026-06-09",
            "2026-06-09T07:48",
            "2026-13-09T07:48:36Z",
            "2026-02-30T07:48:36Z",
            "2026-06-09T24:48:36Z",
            "2026-06-09T07:48:36.Z",
            "2026-06-09T07:48:36+10",
            "2026-06-09X07:48:36Z",
        ] {
            assert_eq!(parse_iso8601_utc_ns(s), None, "accepted: {s}");
        }
    }

    #[test]
    fn scales_to_time_unit() {
        let s = "2026-06-09T07:48:36.123456Z";
        assert_eq!(
            parse_iso8601_utc(s, TimeUnit::Seconds).unwrap(),
            EPOCH_SECONDS,
        );
        assert_eq!(
            parse_iso8601_utc(s, TimeUnit::Milliseconds).unwrap(),
            EPOCH_SECONDS * 1_000 + 123,
        );
        assert_eq!(
            parse_iso8601_utc(s, TimeUnit::Microseconds).unwrap(),
            EPOCH_SECONDS * 1_000_000 + 123_456,
        );
    }

    #[test]
    fn parsed_values_land_in_a_datetime_array() {
        let wire = ["2026-06-09T07:48:36.123456Z", "2026-06-09T07:48:36.123457Z"];
        let values: Vec<i64> = wire
            .iter()
            .map(|s| parse_iso8601_utc(s, TimeUnit::Microseconds).unwrap())
            .collect();
        let array = DatetimeArray::<i64>::from_slice(&values, Some(TimeUnit::Microseconds));
        assert_eq!(array.time_unit, TimeUnit::Microseconds);
        assert_eq!(array.data.as_slice()[0], EPOCH_SECONDS * 1_000_000 + 123_456);
        assert_eq!(array.data.as_slice()[1], EPOCH_SECONDS * 1_000_000 + 123_457);
    }
}
