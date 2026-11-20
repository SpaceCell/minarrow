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

//! # **Print Module** - *Pretty Printing with Attitude*
//!
//! Contains implementations of the Display trait
//! and an additional `Print` trait which wraps it to provide
//! `myobj.print()` for any object that implements it.
use std::fmt::{self, Display, Formatter};

use crate::{Array, Buffer, Float, NumericArray, TextArray};
#[cfg(feature = "datetime")]
use crate::{DatetimeArray, Integer, TemporalArray};
#[cfg(feature = "decimal")]
use crate::structs::variants::decimal::format_decimal_value;

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
            #[cfg(feature = "decimal")]
            NumericArray::Decimal32(a) => format_decimal_value(a.data[idx], a.scale),
            #[cfg(feature = "decimal")]
            NumericArray::Decimal64(a) => format_decimal_value(a.data[idx], a.scale),
            #[cfg(feature = "decimal")]
            NumericArray::Decimal128(a) => format_decimal_value(a.data[idx], a.scale),
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
            TemporalArray::Datetime32(dt) => format_datetime_value(dt, idx),
            TemporalArray::Datetime64(dt) => format_datetime_value(dt, idx),
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
    write!(f, "| {hdr:^w$} |", hdr = "#", w = idx_width)?;
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

/// Render a dense numeric grid in the bordered table layout, eliding
/// rows beyond [`MAX_PREVIEW`].
#[cfg(any(feature = "matrix", feature = "ndarray"))]
pub(crate) fn print_float_grid<T: Float + Display>(
    f: &mut Formatter<'_>,
    headers: &[String],
    n_rows: usize,
    cell: impl Fn(usize, usize) -> T,
) -> fmt::Result {
    let n_cols = headers.len();

    // Show every row for a short grid, otherwise the first and last ten.
    let row_indices: Vec<usize> = if n_rows <= MAX_PREVIEW {
        (0..n_rows).collect()
    } else {
        let mut idx = (0..10).collect::<Vec<_>>();
        idx.extend((n_rows - 10)..n_rows);
        idx
    };

    // Each column widens to fit its header and the values shown beneath it.
    let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
    let mut rows: Vec<Vec<String>> = Vec::with_capacity(row_indices.len());
    for &r in &row_indices {
        let mut row = Vec::with_capacity(n_cols);
        for c in 0..n_cols {
            let text = format_float(cell(r, c));
            widths[c] = widths[c].max(text.len());
            row.push(text);
        }
        rows.push(row);
    }

    let idx_width = usize::max(
        3,
        ((n_rows.saturating_sub(1)) as f64).log10().floor() as usize + 1,
    );

    print_rule(f, idx_width, &widths)?;
    print_header_row(f, idx_width, headers, &widths)?;
    print_rule(f, idx_width, &widths)?;

    for (logical_row, cells) in rows.iter().enumerate() {
        let physical_row = row_indices[logical_row];
        write!(f, "| {idx:^w$} |", idx = physical_row, w = idx_width)?;
        for (c, text) in cells.iter().enumerate() {
            write!(f, " {val:^w$} |", val = text, w = widths[c])?;
        }
        writeln!(f)?;
        if logical_row == 9 && n_rows > MAX_PREVIEW {
            print_ellipsis_row(f, idx_width, &widths)?;
        }
    }
    print_rule(f, idx_width, &widths)
}

/// Render the body of an N-dimensional float array beneath a caller-written
/// title, with the leading axis as rows and trailing axes flattened into
/// columns.
#[cfg(feature = "ndarray")]
pub(crate) fn print_ndarray_body<T: Float + Display>(
    f: &mut Formatter<'_>,
    shape: &[usize],
    cell: impl Fn(&[usize]) -> T,
) -> fmt::Result {
    match shape.len() {
        0 => writeln!(f, "  {}", format_float(cell(&[]))),
        1 => {
            let headers = [String::from("value")];
            print_float_grid(f, &headers, shape[0], |r, _| cell(&[r]))
        }
        2 => {
            let headers: Vec<String> = (0..shape[1]).map(|c| format!("col_{c}")).collect();
            print_float_grid(f, &headers, shape[0], |r, c| cell(&[r, c]))
        }
        _ => {
            // The trailing axes flatten into columns, so each column header
            // names the coordinate its values carry on those axes.
            let outer = &shape[1..];
            let n_outer: usize = outer.iter().product();
            let headers: Vec<String> = (0..n_outer)
                .map(|j| {
                    let mut remaining = j;
                    let mut coords = Vec::with_capacity(outer.len());
                    for &size in outer {
                        coords.push((remaining % size).to_string());
                        remaining /= size;
                    }
                    format!("({})", coords.join(","))
                })
                .collect();
            print_float_grid(f, &headers, shape[0], |r, j| {
                let mut index = Vec::with_capacity(shape.len());
                index.push(r);
                let mut remaining = j;
                for &size in &shape[1..] {
                    index.push(remaining % size);
                    remaining /= size;
                }
                cell(&index)
            })
        }
    }
}

/// Formats the datetime value at `idx` according to the array's `Dt`.
///
/// ## Behaviour
/// With `datetime_ops`:
/// - `Date32` and `Date64` print as `YYYY-MM-DD`.
/// - `Timestamp` prints as `YYYY-MM-DD HH:MM:SS`, followed by the fraction
///   digits of its time unit: 3 for milliseconds, 6 for microseconds and 9 for
///   nanoseconds. The values of an array therefore print at one width.
/// - Timestamps with a time zone print the UTC value with its offset, as in
///   `2026-01-01 09:30:00+00:00`.
/// - `Time32` and `Time64` print the time of day as `HH:MM:SS`, with the
///   fraction digits of their time unit.
/// - `Duration32` and `Duration64` print the stored count with its time unit
///   suffix, as in `90s`.
///
/// With `time_zones`:
/// - Timestamps with a time zone print the wall time of that time zone and
///   its UTC offset at that instant, as in `2026-01-01 14:30:00+05:00`. Time
///   zone text other than an offset follows the value, as in
///   `2026-01-01 20:30:00+11:00 Australia/Sydney`.
/// - IANA identifiers are resolved through the IANA Time Zone Database, which
///   includes daylight saving and historical transitions. The same time zone
///   therefore prints `+11:00` in January and `+10:00` in July for Sydney.
/// - Offsets with seconds, which occur before standard time was adopted,
///   print as `+HH:MM:SS`.
/// - Time zone text that is not an offset, an abbreviation or an IANA
///   identifier prints the UTC value with `+00:00` and without that text.
///
/// Without `datetime_ops`, and for dates and timestamps outside the `time`
/// crate's date range, the stored integer prints with its time unit suffix.
/// Nulls print as `null`.
#[cfg(feature = "datetime")]
pub(crate) fn format_datetime_value<T>(arr: &DatetimeArray<T>, idx: usize) -> String
where
    T: Integer + std::fmt::Display,
{
    use crate::{MaskedArray, TimeUnit};
    if arr.is_null(idx) {
        return "null".into();
    }

    #[cfg(feature = "datetime_ops")]
    {
        use crate::Dt;
        use time::{OffsetDateTime, UtcOffset};

        let unit = arr.time_unit();
        let value = arr.data[idx].to_i64().unwrap();

        // The time unit fixes the number of fraction digits.
        let fraction = |nanosecond: u32| match unit {
            TimeUnit::Milliseconds => format!(".{:03}", nanosecond / 1_000_000),
            TimeUnit::Microseconds => format!(".{:06}", nanosecond / 1_000),
            TimeUnit::Nanoseconds => format!(".{:09}", nanosecond),
            TimeUnit::Seconds | TimeUnit::Days => String::new(),
        };
        let format_date = |dt: OffsetDateTime| {
            format!("{:04}-{:02}-{:02}", dt.year(), u8::from(dt.month()), dt.day())
        };
        let format_timestamp = |dt: OffsetDateTime| {
            format!(
                "{} {:02}:{:02}:{:02}{}",
                format_date(dt),
                dt.hour(),
                dt.minute(),
                dt.second(),
                fraction(dt.nanosecond())
            )
        };

        match &arr.dtype {
            Dt::Date32 | Dt::Date64 => {
                // The conversion depends only on the time unit, and is shared with `DatetimeOps`.
                if let Some(dt) = DatetimeArray::<i64>::i64_to_datetime(value, unit) {
                    return format_date(dt);
                }
            }
            Dt::Timestamp(_, None) => {
                if let Some(dt) = DatetimeArray::<i64>::i64_to_datetime(value, unit) {
                    return format_timestamp(dt);
                }
            }
            Dt::Timestamp(_, Some(tz)) => {
                let utc_dt = DatetimeArray::<i64>::i64_to_datetime(value, unit);
                #[cfg(feature = "time_zones")]
                let offset = utc_dt.and_then(|dt| timezone_offset(tz, dt.unix_timestamp()));
                // Time zone conversion is part of the `time_zones` feature.
                #[cfg(not(feature = "time_zones"))]
                let offset: Option<UtcOffset> = None;
                let local_dt =
                    utc_dt.and_then(|dt| dt.checked_to_offset(offset.unwrap_or(UtcOffset::UTC)));
                if let Some(local_dt) = local_dt {
                    let seconds = local_dt.offset().whole_seconds();
                    let sign = if seconds < 0 { '-' } else { '+' };
                    let mut text = format!(
                        "{}{}{:02}:{:02}",
                        format_timestamp(local_dt),
                        sign,
                        seconds.abs() / 3_600,
                        seconds.abs() / 60 % 60
                    );
                    if seconds % 60 != 0 {
                        text.push_str(&format!(":{:02}", seconds.abs() % 60));
                    }
                    // Time zone text that is an offset is already printed with the value.
                    let is_offset_text = tz.starts_with(['+', '-'])
                        || tz.eq_ignore_ascii_case("UTC")
                        || tz.eq_ignore_ascii_case("Z");
                    // The time zone text follows wall times of that zone only. Values
                    // printed in UTC end with their `+00:00` offset.
                    return if offset.is_some() && !is_offset_text {
                        format!("{text} {tz}")
                    } else {
                        text
                    };
                }
            }
            Dt::Time32(_) | Dt::Time64(_) if value >= 0 && unit != TimeUnit::Days => {
                let per_second: i64 = match unit {
                    TimeUnit::Milliseconds => 1_000,
                    TimeUnit::Microseconds => 1_000_000,
                    TimeUnit::Nanoseconds => 1_000_000_000,
                    TimeUnit::Seconds | TimeUnit::Days => 1,
                };
                let seconds = value / per_second;
                let nanosecond = (value % per_second * (1_000_000_000 / per_second)) as u32;
                return format!(
                    "{:02}:{:02}:{:02}{}",
                    seconds / 3_600,
                    seconds / 60 % 60,
                    seconds % 60,
                    fraction(nanosecond)
                );
            }
            _ => {}
        }
    }

    let v = arr.data[idx];
    let suffix = match arr.time_unit() {
        TimeUnit::Seconds => "s",
        TimeUnit::Milliseconds => "ms",
        TimeUnit::Microseconds => "µs",
        TimeUnit::Nanoseconds => "ns",
        TimeUnit::Days => "d",
    };
    format!("{v}{suffix}")
}

/// Returns the UTC offset in effect at `unix_seconds` for the time zone text
/// `tz`, or `None` for text that `utc_offset_seconds` does not resolve.
#[cfg(feature = "time_zones")]
fn timezone_offset(tz: &str, unix_seconds: i64) -> Option<time::UtcOffset> {
    use crate::structs::variants::datetime::tz::utc_offset_seconds;
    use time::UtcOffset;

    let seconds = utc_offset_seconds(tz)?(unix_seconds);
    UtcOffset::from_whole_seconds(i32::try_from(seconds).ok()?).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "decimal")]
    #[test]
    fn test_value_to_string_decimal_positive_scale() {
        use crate::DecimalArray;
        let arr = Array::from_decimal64(DecimalArray::from_slice(&[12345i64, -67890], 10, 2));
        assert_eq!(value_to_string(&arr, 0), "123.45");
        assert_eq!(value_to_string(&arr, 1), "-678.90");
    }

    #[cfg(feature = "decimal")]
    #[test]
    fn test_value_to_string_decimal_zero_scale() {
        use crate::DecimalArray;
        let arr = Array::from_decimal32(DecimalArray::from_slice(&[42i32, 100], 5, 0));
        assert_eq!(value_to_string(&arr, 0), "42");
        assert_eq!(value_to_string(&arr, 1), "100");
    }

    #[cfg(feature = "decimal")]
    #[test]
    fn test_value_to_string_decimal_negative_scale() {
        use crate::DecimalArray;
        let arr = Array::from_decimal32(DecimalArray::from_slice(&[123i32], 10, -2));
        assert_eq!(value_to_string(&arr, 0), "12300");
    }

    #[cfg(feature = "decimal")]
    #[test]
    fn test_value_to_string_decimal_null() {
        use crate::{Bitmask, DecimalArray};
        let mut da = DecimalArray::from_slice(&[0i64, 100], 10, 2);
        da.null_mask = Some(Bitmask::from_bools(&[false, true]));
        let arr = Array::from_decimal64(da);
        assert_eq!(value_to_string(&arr, 0), "null");
        assert_eq!(value_to_string(&arr, 1), "1.00");
    }

    #[cfg(feature = "decimal")]
    #[test]
    fn test_value_to_string_decimal128() {
        use crate::DecimalArray;
        let arr = Array::from_decimal128(DecimalArray::from_slice(&[999_999i128], 38, 4));
        assert_eq!(value_to_string(&arr, 0), "99.9999");
    }

    #[cfg(feature = "decimal")]
    #[test]
    fn test_table_display_with_decimal_column() {
        use crate::{DecimalArray, FieldArray, Table};
        use crate::ffi::arrow_dtype::ArrowType;
        use crate::Field;

        let decimal_arr = DecimalArray::<i64>::from_slice(&[12345, 67890, 100], 10, 2);
        let arr = Array::from_decimal64(decimal_arr);
        let field = Field::new(
            "amount".to_string(),
            ArrowType::Decimal128(10, 2),
            false,
            None,
        );
        let table = Table::build(
            vec![FieldArray::new(field, arr)],
            3,
            "test".to_string(),
        );
        let display = format!("{}", table);
        assert!(display.contains("123.45"), "display should contain 123.45, got: {}", display);
        assert!(display.contains("678.90"), "display should contain 678.90, got: {}", display);
        assert!(display.contains("1.00"), "display should contain 1.00, got: {}", display);
    }

    #[cfg(feature = "datetime_ops")]
    fn timestamp_array(values: &[Option<i64>], dtype: crate::Dt) -> crate::DatetimeArray<i64> {
        let data: Vec<i64> = values.iter().map(|v| v.unwrap_or(0)).collect();
        let mut arr = crate::DatetimeArray::<i64>::from_slice(&data, dtype);
        if values.iter().any(|v| v.is_none()) {
            arr.null_mask = Some(crate::Bitmask::from_bools(
                &values.iter().map(|v| v.is_some()).collect::<Vec<_>>(),
            ));
        }
        arr
    }

    // Asserts the text through `Array::value_to_string`, `DatetimeArray` Display and
    // `FieldArray` Display for each value of a 64-bit datetime array.
    #[cfg(feature = "datetime_ops")]
    fn assert_datetime_text(arr: crate::DatetimeArray<i64>, expected: &[&str]) {
        let list = format!("[{}]", expected.join(", "));
        assert_eq!(arr.to_string().lines().last(), Some(list.as_str()));

        let array = Array::from_datetime_i64(arr);
        for (i, text) in expected.iter().enumerate() {
            assert_eq!(array.value_to_string(i), *text);
        }

        let field_array = crate::FieldArray::from_arr("ts", array);
        let display = field_array.to_string();
        assert_eq!(display.lines().last(), Some(list.as_str()));
    }

    #[cfg(feature = "datetime_ops")]
    #[test]
    fn test_date32_prints_iso_date() {
        use crate::{DatetimeArray, Dt};
        let mut arr = DatetimeArray::<i32>::from_slice(&[20461, 0, -1], Dt::Date32);
        arr.null_mask = Some(crate::Bitmask::from_bools(&[true, true, false]));
        assert_eq!(arr.to_string().lines().last(), Some("[2026-01-08, 1970-01-01, null]"));

        let array = Array::from_datetime_i32(arr);
        assert_eq!(array.value_to_string(0), "2026-01-08");
        assert_eq!(array.value_to_string(1), "1970-01-01");
        assert_eq!(array.value_to_string(2), "null");

        let display = crate::FieldArray::from_arr("d", array).to_string();
        assert_eq!(display.lines().last(), Some("[2026-01-08, 1970-01-01, null]"));
    }

    #[cfg(feature = "datetime_ops")]
    #[test]
    fn test_date64_prints_iso_date() {
        use crate::Dt;
        let arr = timestamp_array(&[Some(1_767_830_400_000), None], Dt::Date64);
        assert_datetime_text(arr, &["2026-01-08", "null"]);
    }

    #[cfg(feature = "datetime_ops")]
    #[test]
    fn test_timestamp_seconds_prints_iso_datetime() {
        use crate::{Dt, TimeUnit};
        let arr = timestamp_array(
            &[Some(1_767_259_800), None, Some(0)],
            Dt::Timestamp(TimeUnit::Seconds, None),
        );
        assert_datetime_text(arr, &["2026-01-01 09:30:00", "null", "1970-01-01 00:00:00"]);
    }

    #[cfg(feature = "datetime_ops")]
    #[test]
    fn test_timestamp_milliseconds_prints_three_fraction_digits() {
        use crate::{Dt, TimeUnit};
        let arr = timestamp_array(
            &[Some(1_767_259_800_123), Some(1_767_259_800_000), None],
            Dt::Timestamp(TimeUnit::Milliseconds, None),
        );
        assert_datetime_text(
            arr,
            &["2026-01-01 09:30:00.123", "2026-01-01 09:30:00.000", "null"],
        );
    }

    #[cfg(feature = "datetime_ops")]
    #[test]
    fn test_timestamp_microseconds_prints_six_fraction_digits() {
        use crate::{Dt, TimeUnit};
        let arr = timestamp_array(
            &[Some(1_767_259_800_500_000), Some(1_767_259_800_000_000), Some(1_767_259_800_000_250)],
            Dt::Timestamp(TimeUnit::Microseconds, None),
        );
        assert_datetime_text(
            arr,
            &["2026-01-01 09:30:00.500000", "2026-01-01 09:30:00.000000", "2026-01-01 09:30:00.000250"],
        );
    }

    #[cfg(feature = "datetime_ops")]
    #[test]
    fn test_timestamp_nanoseconds_prints_nine_fraction_digits() {
        use crate::{Dt, TimeUnit};
        let arr = timestamp_array(
            &[Some(1_767_259_800_000_000_001), Some(1_767_259_800_000_000_000), None],
            Dt::Timestamp(TimeUnit::Nanoseconds, None),
        );
        assert_datetime_text(
            arr,
            &["2026-01-01 09:30:00.000000001", "2026-01-01 09:30:00.000000000", "null"],
        );
    }

    #[cfg(feature = "time_zones")]
    #[test]
    fn test_timestamp_with_offset_prints_local_time_and_offset() {
        use crate::{Dt, TimeUnit};
        let arr = timestamp_array(
            &[Some(1_767_259_800_000_000), Some(1_767_259_800_250_000), None],
            Dt::Timestamp(TimeUnit::Microseconds, Some("+05:00".into())),
        );
        assert_datetime_text(
            arr,
            &["2026-01-01 14:30:00.000000+05:00", "2026-01-01 14:30:00.250000+05:00", "null"],
        );

        let arr = timestamp_array(
            &[Some(1_767_259_800)],
            Dt::Timestamp(TimeUnit::Seconds, Some("-08:00".into())),
        );
        assert_datetime_text(arr, &["2026-01-01 01:30:00-08:00"]);
    }

    #[cfg(feature = "time_zones")]
    #[test]
    fn test_timestamp_with_named_zone_prints_local_time_offset_and_zone() {
        use crate::{Dt, TimeUnit};
        let arr = timestamp_array(
            &[Some(1_767_259_800_000), None],
            Dt::Timestamp(TimeUnit::Milliseconds, Some("Asia/Tokyo".into())),
        );
        assert_datetime_text(arr, &["2026-01-01 18:30:00.000+09:00 Asia/Tokyo", "null"]);

        let arr = timestamp_array(
            &[Some(1_767_259_800)],
            Dt::Timestamp(TimeUnit::Seconds, Some("AEST".into())),
        );
        assert_datetime_text(arr, &["2026-01-01 19:30:00+10:00 AEST"]);

        let arr = timestamp_array(
            &[Some(1_767_259_800)],
            Dt::Timestamp(TimeUnit::Seconds, Some("UTC".into())),
        );
        assert_datetime_text(arr, &["2026-01-01 09:30:00+00:00"]);
    }

    #[cfg(all(feature = "datetime_ops", not(feature = "time_zones")))]
    #[test]
    fn test_timestamp_with_time_zone_prints_utc_value_without_time_zones() {
        use crate::{Dt, TimeUnit};
        for tz in ["+05:00", "AEST", "Asia/Tokyo", "Australia/Sydney", "UTC"] {
            let arr = timestamp_array(
                &[Some(1_767_259_800), None],
                Dt::Timestamp(TimeUnit::Seconds, Some(tz.into())),
            );
            assert_datetime_text(arr, &["2026-01-01 09:30:00+00:00", "null"]);
        }
        let arr = timestamp_array(
            &[Some(1_767_259_800_250_000)],
            Dt::Timestamp(TimeUnit::Microseconds, Some("Australia/Sydney".into())),
        );
        assert_datetime_text(arr, &["2026-01-01 09:30:00.250000+00:00"]);
    }

    #[cfg(feature = "datetime_ops")]
    #[test]
    fn test_timestamp_with_unresolved_zone_prints_utc_value_with_utc_offset() {
        use crate::{Dt, TimeUnit};
        let arr = timestamp_array(
            &[Some(1_767_259_800), None],
            Dt::Timestamp(TimeUnit::Seconds, Some("Mars/Olympus_Mons".into())),
        );
        assert_datetime_text(
            arr,
            &["2026-01-01 09:30:00+00:00", "null"],
        );
    }

    // The expected text is from Python's `zoneinfo`, which reads the IANA Time
    // Zone Database. It covers daylight saving in both hemispheres, the seconds
    // either side of a transition, historical rules, offsets of 30 and 45
    // minutes, and local mean time offsets with seconds.
    #[cfg(feature = "time_zones")]
    #[test]
    fn test_timestamp_with_iana_zone_prints_wall_time_and_offset_of_the_instant() {
        use crate::{Dt, TimeUnit};
        let cases: [(&str, i64, &str); 17] = [
            ("Australia/Sydney", 1767259800, "2026-01-01 20:30:00+11:00 Australia/Sydney"),
            ("Australia/Sydney", 1782898200, "2026-07-01 19:30:00+10:00 Australia/Sydney"),
            ("America/New_York", 1767259800, "2026-01-01 04:30:00-05:00 America/New_York"),
            ("America/New_York", 1782898200, "2026-07-01 05:30:00-04:00 America/New_York"),
            ("Europe/London", 1782898200, "2026-07-01 10:30:00+01:00 Europe/London"),
            ("Australia/Sydney", 1775318399, "2026-04-05 02:59:59+11:00 Australia/Sydney"),
            ("Australia/Sydney", 1775318400, "2026-04-05 02:00:00+10:00 Australia/Sydney"),
            ("America/New_York", 1772953199, "2026-03-08 01:59:59-05:00 America/New_York"),
            ("America/New_York", 1772953200, "2026-03-08 03:00:00-04:00 America/New_York"),
            ("America/New_York", 1142856000, "2006-03-20 07:00:00-05:00 America/New_York"),
            ("Europe/London", 0, "1970-01-01 01:00:00+01:00 Europe/London"),
            ("Asia/Kolkata", 1767259800, "2026-01-01 15:00:00+05:30 Asia/Kolkata"),
            ("Asia/Kathmandu", 1767259800, "2026-01-01 15:15:00+05:45 Asia/Kathmandu"),
            ("Australia/Lord_Howe", 1767259800, "2026-01-01 20:30:00+11:00 Australia/Lord_Howe"),
            ("Australia/Lord_Howe", 1782898200, "2026-07-01 20:00:00+10:30 Australia/Lord_Howe"),
            ("Australia/Sydney", -2524521600, "1890-01-01 10:04:52+10:04:52 Australia/Sydney"),
            ("America/New_York", -2840140800, "1879-12-31 19:03:58-04:56:02 America/New_York"),
        ];
        for (zone, unix_seconds, expected) in cases {
            let arr = timestamp_array(
                &[Some(unix_seconds), None],
                Dt::Timestamp(TimeUnit::Seconds, Some(zone.into())),
            );
            assert_datetime_text(arr, &[expected, "null"]);
        }
    }

    #[cfg(feature = "time_zones")]
    #[test]
    fn test_timestamp_with_iana_zone_prints_fraction_digits_of_its_unit() {
        use crate::{Dt, TimeUnit};
        let arr = timestamp_array(
            &[Some(1_767_259_800_123), Some(1_782_898_200_000)],
            Dt::Timestamp(TimeUnit::Milliseconds, Some("Australia/Sydney".into())),
        );
        assert_datetime_text(
            arr,
            &[
                "2026-01-01 20:30:00.123+11:00 Australia/Sydney",
                "2026-07-01 19:30:00.000+10:00 Australia/Sydney",
            ],
        );
    }

    // 1 January 2026 and 1 July 2026 at 09:30:00 UTC.
    #[cfg(feature = "time_zones")]
    const JANUARY: i64 = 1_767_259_800;
    #[cfg(feature = "time_zones")]
    const JULY: i64 = 1_782_898_200;

    #[cfg(feature = "time_zones")]
    fn offset_text(tz: &str, unix_seconds: i64) -> Option<String> {
        timezone_offset(tz, unix_seconds).map(|offset| {
            let seconds = offset.whole_seconds();
            let sign = if seconds < 0 { '-' } else { '+' };
            format!("{sign}{:02}:{:02}", seconds.abs() / 3_600, seconds.abs() / 60 % 60)
        })
    }

    #[cfg(feature = "time_zones")]
    #[test]
    fn test_timezone_offset_of_iana_identifiers() {
        assert_eq!(offset_text("Australia/Sydney", JULY).as_deref(), Some("+10:00"));
        assert_eq!(offset_text("Australia/Sydney", JANUARY).as_deref(), Some("+11:00"));
        assert_eq!(offset_text("America/New_York", JANUARY).as_deref(), Some("-05:00"));
        assert_eq!(offset_text("America/New_York", JULY).as_deref(), Some("-04:00"));
        assert_eq!(offset_text("Europe/London", JANUARY).as_deref(), Some("+00:00"));
        assert_eq!(offset_text("Europe/London", JULY).as_deref(), Some("+01:00"));
        assert_eq!(offset_text("Asia/Tokyo", JANUARY).as_deref(), Some("+09:00"));
        assert_eq!(offset_text("Pacific/Auckland", JULY).as_deref(), Some("+12:00"));
        assert_eq!(offset_text("Pacific/Auckland", JANUARY).as_deref(), Some("+13:00"));
    }

    #[cfg(feature = "time_zones")]
    #[test]
    fn test_timezone_offset_of_abbreviations() {
        for (abbreviation, offset) in [
            ("AEST", "+10:00"),
            ("EST", "-05:00"),
            ("GMT", "+00:00"),
            ("UTC", "+00:00"),
            ("PST", "-08:00"),
            ("JST", "+09:00"),
        ] {
            // Abbreviations have one offset throughout the year.
            assert_eq!(offset_text(abbreviation, JANUARY).as_deref(), Some(offset));
            assert_eq!(offset_text(abbreviation, JULY).as_deref(), Some(offset));
        }
    }

    #[cfg(feature = "time_zones")]
    #[test]
    fn test_timezone_offset_of_direct_offsets() {
        for offset in ["+10:00", "-05:00", "+00:00", "+05:30", "-03:30"] {
            assert_eq!(offset_text(offset, JANUARY).as_deref(), Some(offset));
        }
        assert_eq!(offset_text("+0530", JANUARY).as_deref(), Some("+05:30"));
        assert_eq!(offset_text("Z", JANUARY).as_deref(), Some("+00:00"));
    }

    #[cfg(feature = "time_zones")]
    #[test]
    fn test_timezone_offset_of_invalid_text() {
        assert_eq!(offset_text("Invalid/Timezone", JANUARY), None);
        assert_eq!(offset_text("INVALID", JANUARY), None);
        assert_eq!(offset_text("25:00", JANUARY), None);
    }

    #[cfg(feature = "time_zones")]
    #[test]
    fn test_timezone_offset_of_unusual_offsets() {
        assert_eq!(offset_text("Australia/Eucla", JANUARY).as_deref(), Some("+08:45"));
        assert_eq!(offset_text("Asia/Kathmandu", JANUARY).as_deref(), Some("+05:45"));
        assert_eq!(offset_text("Asia/Colombo", JANUARY).as_deref(), Some("+05:30"));
        assert_eq!(offset_text("America/St_Johns", JANUARY).as_deref(), Some("-03:30"));
        assert_eq!(offset_text("America/St_Johns", JULY).as_deref(), Some("-02:30"));
        assert_eq!(offset_text("Pacific/Chatham", JULY).as_deref(), Some("+12:45"));
        assert_eq!(offset_text("Pacific/Chatham", JANUARY).as_deref(), Some("+13:45"));
    }

    #[cfg(feature = "time_zones")]
    #[test]
    fn test_every_iana_identifier_has_an_offset() {
        for name in tzdb_data::TZ_NAMES {
            assert!(timezone_offset(name, JANUARY).is_some(), "no offset for {name}");
            assert!(timezone_offset(name, JULY).is_some(), "no offset for {name}");
        }
    }

    #[cfg(feature = "datetime_ops")]
    #[test]
    fn test_time_of_day_prints_hours_minutes_seconds() {
        use crate::{DatetimeArray, Dt, TimeUnit};
        let mut arr = DatetimeArray::<i32>::from_slice(&[34_200, 0, 86_399, 0], Dt::Time32(TimeUnit::Seconds));
        arr.null_mask = Some(crate::Bitmask::from_bools(&[true, true, true, false]));
        assert_eq!(arr.to_string().lines().last(), Some("[09:30:00, 00:00:00, 23:59:59, null]"));

        let arr = timestamp_array(
            &[Some(34_200_250_000), Some(0), None],
            Dt::Time64(TimeUnit::Microseconds),
        );
        assert_datetime_text(arr, &["09:30:00.250000", "00:00:00.000000", "null"]);

        let arr = timestamp_array(&[Some(34_200_000_000_001)], Dt::Time64(TimeUnit::Nanoseconds));
        assert_datetime_text(arr, &["09:30:00.000000001"]);
    }

    #[cfg(feature = "datetime_ops")]
    #[test]
    fn test_durations_print_count_with_unit_suffix() {
        use crate::{DatetimeArray, Dt, TimeUnit};
        let arr = timestamp_array(&[Some(90), Some(-5), None], Dt::Duration64(TimeUnit::Seconds));
        assert_datetime_text(arr, &["90s", "-5s", "null"]);

        let arr = DatetimeArray::<i32>::from_slice(&[1_500], Dt::Duration32(TimeUnit::Milliseconds));
        assert_eq!(arr.to_string().lines().last(), Some("[1500ms]"));
    }

    #[cfg(all(feature = "datetime", not(feature = "datetime_ops")))]
    #[test]
    fn test_datetime_prints_raw_value_without_datetime_ops() {
        use crate::{DatetimeArray, Dt, TimeUnit};
        let date = Array::from_datetime_i32(DatetimeArray::from_slice(&[20461], Dt::Date32));
        assert_eq!(date.value_to_string(0), "20461d");
        let ts = DatetimeArray::<i64>::from_slice(
            &[1_767_259_800_000_000],
            Dt::Timestamp(TimeUnit::Microseconds, None),
        );
        assert_eq!(ts.to_string().lines().last(), Some("[1767259800000000µs]"));
        let ts = Array::from_datetime_i64(ts);
        assert_eq!(ts.value_to_string(0), "1767259800000000µs");
    }
}
