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

//! Example of datetime operations with the `time` crate.
//!
//! Demonstrates arithmetic, duration calculations, component extraction,
//! comparisons, truncation, and type casting operations on datetime arrays.
//!
//! Run with:
//!     cargo run --example datetime_operations --features datetime_ops

#[cfg(feature = "datetime_ops")]
fn main() {
    use minarrow::ffi::arrow_dtype::ArrowType;
    use minarrow::{DatetimeArray, DatetimeOps, FieldArray, MaskedArray, Print, TimeUnit};
    use time::Duration;

    println!("  Minarrow Datetime Operations Example");

    // Create datetime array with timestamps
    // - seconds since Unix epoch
    let timestamps = vec![
        1_700_000_000, // 2023-11-14 22:13:20 UTC
        1_700_086_400, // 2023-11-15 22:13:20 UTC
        1_700_172_800, // 2023-11-16 22:13:20 UTC
    ];
    let arr = DatetimeArray::<i64>::from_slice(&timestamps, Some(TimeUnit::Seconds));

    println!("Original datetime array:");
    for i in 0..arr.len() {
        if let Some(dt) = arr.as_datetime(i) {
            println!("  [{}] {}", i, dt);
        }
    }

    // Arithmetic Operations
    println!("\n--- Arithmetic Operations ---");

    let plus_one_hour = arr.add_duration(Duration::hours(1)).unwrap();
    println!("\nAfter adding 1 hour:");
    for i in 0..plus_one_hour.len() {
        if let Some(dt) = plus_one_hour.as_datetime(i) {
            println!("  [{}] {}", i, dt);
        }
    }

    let plus_7_days = arr.add_days(7).unwrap();
    println!("\nAfter adding 7 days:");
    for i in 0..plus_7_days.len() {
        if let Some(dt) = plus_7_days.as_datetime(i) {
            println!("  [{}] {}", i, dt);
        }
    }

    let plus_2_months = arr.add_months(2).unwrap();
    println!("\nAfter adding 2 months:");
    for i in 0..plus_2_months.len() {
        if let Some(dt) = plus_2_months.as_datetime(i) {
            println!("  [{}] {}", i, dt);
        }
    }

    // Duration/Diff Operations
    println!("\n--- Duration Operations ---");

    // Create array with some times before and some after arr
    let arr2 = DatetimeArray::<i64>::from_slice(
        &[
            1_699_999_000, // 1000 seconds BEFORE arr[0]
            1_700_086_400, // Same as arr[1]
            1_700_173_900, // 100 seconds AFTER arr[2]
        ],
        Some(TimeUnit::Seconds),
    );

    // diff() returns signed difference i.e. arr2 - arr
    let diff_seconds = arr2.diff(&arr, TimeUnit::Seconds).unwrap();
    println!("\nSigned difference (arr2 - arr) in seconds:");
    for i in 0..diff_seconds.len() {
        if let Some(val) = diff_seconds.get(i) {
            println!("  [{}] {} seconds", i, val);
        }
    }

    // abs_diff() returns absolute (unsigned) difference
    let abs_diff = arr2.abs_diff(&arr, TimeUnit::Seconds).unwrap();
    println!("\nAbsolute difference in seconds:");
    for i in 0..abs_diff.len() {
        if let Some(val) = abs_diff.get(i) {
            println!("  [{}] {} seconds", i, val);
        }
    }

    // Component Extraction
    println!("\n--- Component Extraction ---");

    // Create a high-precision datetime array to show all granularities
    // 2023-11-14 22:13:20.123456789 UTC
    let precise_dt =
        DatetimeArray::<i64>::from_slice(&[1_700_000_000_123_456_789], Some(TimeUnit::Nanoseconds));

    let years = precise_dt.year();
    let months = precise_dt.month();
    let days = precise_dt.day();
    let hours = precise_dt.hour();
    let minutes = precise_dt.minute();
    let seconds = precise_dt.second();
    let weekdays = precise_dt.weekday();
    let quarters = precise_dt.quarter();
    let iso_weeks = precise_dt.iso_week();

    println!("\nExtracted components from datetime:");
    if let Some(dt) = precise_dt.as_datetime(0) {
        println!("  Full datetime: {}", dt);
    }
    println!("  Year:         {}", years.get(0).unwrap());
    println!("  Month:        {}", months.get(0).unwrap());
    println!("  Day:          {}", days.get(0).unwrap());
    println!("  Hour:         {}", hours.get(0).unwrap());
    println!("  Minute:       {}", minutes.get(0).unwrap());
    println!("  Second:       {}", seconds.get(0).unwrap());

    // Show sub-second components by converting to different time units
    if let Ok(as_millis) = precise_dt.cast_time_unit(TimeUnit::Milliseconds) {
        let millis_remainder = as_millis.data[0] % 1000;
        println!("  Millisecond:  {}", millis_remainder);
    }
    if let Ok(as_micros) = precise_dt.cast_time_unit(TimeUnit::Microseconds) {
        let micros_remainder = as_micros.data[0] % 1_000_000;
        println!("  Microsecond:  {}", micros_remainder % 1000);
    }
    let nanos_remainder = precise_dt.data[0] % 1_000_000_000;
    println!("  Nanosecond:   {}", nanos_remainder % 1000);

    println!(
        "  Weekday:      {} (1=Sunday, 2=Monday, ..., 7=Saturday)",
        weekdays.get(0).unwrap()
    );
    println!("  Quarter:      {}", quarters.get(0).unwrap());
    println!("  ISO Week:     {}", iso_weeks.get(0).unwrap());

    let leap_years = precise_dt.is_leap_year();
    println!("  Leap year?:   {}", leap_years.get(0).unwrap());

    // Comparison Operations
    println!("\n--- Comparison Operations ---");

    let is_before = arr.is_before(&arr2).unwrap();
    println!("\narr is_before arr2:");
    for i in 0..is_before.len() {
        if let Some(val) = is_before.get(i) {
            println!("  [{}] {}", i, val);
        }
    }

    let is_after = arr.is_after(&arr2).unwrap();
    println!("\narr is_after arr2:");
    for i in 0..is_after.len() {
        if let Some(val) = is_after.get(i) {
            println!("  [{}] {}", i, val);
        }
    }

    // Truncation Operations
    println!("\n--- Truncation Operations ---");

    let truncated_day = arr.truncate("day");
    println!("\nTruncated to start of day:");
    for i in 0..truncated_day.len() {
        if let Some(dt) = truncated_day.as_datetime(i) {
            println!("  [{}] {}", i, dt);
        }
    }

    let truncated_month = arr.truncate("month");
    println!("\nTruncated to start of month:");
    for i in 0..truncated_month.len() {
        if let Some(dt) = truncated_month.as_datetime(i) {
            println!("  [{}] {}", i, dt);
        }
    }

    // FieldArray with Timezone Metadata
    println!("\n--- FieldArray with Timezone Metadata ---");
    println!("\nFieldArray allows timezone to be encoded in the ArrowType metadata,");
    println!("creating a self-describing datetime column with permanent timezone info.");
    println!("This is the standard pattern Tables with tz-aware columns leverage.");
    println!("After these examples, we will look at inline Tz cases.");

    // Example 1: Single FieldArray with timezone
    println!("\n1. FieldArray with Sydney timezone:");
    let sydney_events = DatetimeArray::<i64>::from_slice(
        &[
            1_700_000_000, // 2023-11-14 22:13:20 UTC = 2023-11-15 09:13:20 AEDT
            1_700_086_400, // 2023-11-15 22:13:20 UTC = 2023-11-16 09:13:20 AEDT
            1_700_172_800, // 2023-11-16 22:13:20 UTC = 2023-11-17 09:13:20 AEDT
        ],
        Some(TimeUnit::Seconds),
    );
    let sydney_field_array = FieldArray::from_parts(
        "event_time",
        ArrowType::Timestamp(TimeUnit::Seconds, Some("Australia/Sydney".to_string())),
        Some(false), // not nullable
        None,        // no additional metadata
        sydney_events.into(),
    );

    println!("   Field name: {}", sydney_field_array.field.name);
    println!("   Field type: {:?}", sydney_field_array.field.dtype);
    println!("   Data:");
    sydney_field_array.print();

    // Example 2: Multiple FieldArrays with different timezones
    println!("\n2. Multiple FieldArrays with different timezones:");

    // New York events (EST/EDT)
    let ny_events =
        DatetimeArray::<i64>::from_slice(&[1_700_000_000, 1_700_086_400], Some(TimeUnit::Seconds));
    let ny_field_array = FieldArray::from_parts(
        "ny_time",
        ArrowType::Timestamp(TimeUnit::Seconds, Some("America/New_York".to_string())),
        Some(false),
        None,
        ny_events.into(),
    );

    // Tokyo events (JST)
    let tokyo_events =
        DatetimeArray::<i64>::from_slice(&[1_700_000_000, 1_700_086_400], Some(TimeUnit::Seconds));
    let tokyo_field_array = FieldArray::from_parts(
        "tokyo_time",
        ArrowType::Timestamp(TimeUnit::Seconds, Some("Asia/Tokyo".to_string())),
        Some(false),
        None,
        tokyo_events.into(),
    );

    // Extract timezone from ArrowType for display
    let ny_tz = if let ArrowType::Timestamp(_, Some(tz)) = &ny_field_array.field.dtype {
        tz.as_str()
    } else {
        "UTC"
    };
    let tokyo_tz = if let ArrowType::Timestamp(_, Some(tz)) = &tokyo_field_array.field.dtype {
        tz.as_str()
    } else {
        "UTC"
    };

    println!("\n   New York events ({}):", ny_tz);
    ny_field_array.print();

    println!("\n   Tokyo events ({}):", tokyo_tz);
    tokyo_field_array.print();

    println!("\n   Note: All FieldArrays contain the same UTC timestamps,");
    println!("   but display in their respective local timezones based on ArrowType metadata.");

    // Example 3: FieldArray with unusual offset
    println!("\n3. FieldArray with unusual timezone offset:");
    let nepal_events = DatetimeArray::<i64>::from_slice(&[1_700_000_000], Some(TimeUnit::Seconds));
    let nepal_field_array = FieldArray::from_parts(
        "nepal_event",
        ArrowType::Timestamp(TimeUnit::Seconds, Some("Asia/Kathmandu".to_string())),
        Some(false),
        None,
        nepal_events.into(),
    );

    println!("   Kathmandu uses +05:45 offset (Nepal Time):");
    nepal_field_array.print();

    // Type Casting
    println!("\n--- Type Casting ---");

    let as_millis = arr.cast_time_unit(TimeUnit::Milliseconds).unwrap();
    println!("\nCast to milliseconds:");
    for i in 0..as_millis.len() {
        if let Some(dt) = as_millis.as_datetime(i) {
            println!("  [{}] {}", i, dt);
        }
    }

    let as_micros = arr.cast_time_unit(TimeUnit::Microseconds).unwrap();
    println!("\nCast to microseconds:");
    for i in 0..as_micros.len() {
        if let Some(dt) = as_micros.as_datetime(i) {
            println!("  [{}] {}", i, dt);
        }
    }

    // Inline Timezone Operations
    println!("\n--- Inline Timezone Operations ---");
    println!("\nTimezone support includes:");
    println!("  1. IANA timezone identifiers (e.g., 'Australia/Sydney')");
    println!("  2. Timezone abbreviations (e.g., 'AEST', 'EST', 'JST')");
    println!("  3. Direct offset strings (e.g., '+10:00', '-05:00')");

    // Use the first timestamp for timezone examples
    let utc_dt = DatetimeArray::<i64>::from_slice(&[timestamps[0]], Some(TimeUnit::Seconds));

    println!("\nOriginal UTC time: 2023-11-14 22:13:20 UTC");
    println!("\n1. IANA Timezone Identifiers:");
    println!("   Australia/Sydney:");
    utc_dt.tz("Australia/Sydney").print();
    println!();

    println!("   America/New_York:");
    utc_dt.tz("America/New_York").print();
    println!();

    println!("   Europe/London:");
    utc_dt.tz("Europe/London").print();
    println!();

    println!("   Asia/Tokyo:");
    utc_dt.tz("Asia/Tokyo").print();
    println!();

    println!("   America/Los_Angeles:");
    utc_dt.tz("America/Los_Angeles").print();
    println!();

    println!("\n2. Timezone Abbreviations:");
    println!("   AEST (Australian Eastern Standard Time):");
    utc_dt.tz("AEST").print();
    println!();

    println!("   EST (Eastern Standard Time):");
    utc_dt.tz("EST").print();
    println!();

    println!("   JST (Japan Standard Time):");
    utc_dt.tz("JST").print();
    println!();

    println!("   GMT (Greenwich Mean Time):");
    utc_dt.tz("GMT").print();
    println!();

    println!("   PST (Pacific Standard Time):");
    utc_dt.tz("PST").print();
    println!();

    println!("\n3. Direct Offset Strings:");
    println!("   +10:00 (Eastern Australia):");
    utc_dt.tz("+10:00").print();
    println!();

    println!("   -05:00 (US Eastern):");
    utc_dt.tz("-05:00").print();
    println!();

    println!("   +05:30 (India Standard Time):");
    utc_dt.tz("+05:30").print();
    println!();

    println!("   -03:30 (Newfoundland):");
    utc_dt.tz("-03:30").print();
    println!();

    println!("   +09:00 (Japan):");
    utc_dt.tz("+09:00").print();
    println!();

    // Demonstrate unusual offsets
    println!("\n4. Unusual Timezone Offsets:");
    println!("   Australia/Eucla (+08:45):");
    utc_dt.tz("Australia/Eucla").print();
    println!();

    println!("   Asia/Kathmandu (+05:45):");
    utc_dt.tz("Asia/Kathmandu").print();
    println!();

    println!("   Pacific/Chatham (+12:45):");
    utc_dt.tz("Pacific/Chatham").print();
    println!();

    println!("\n  Example End.");
}

#[cfg(not(feature = "datetime_ops"))]
fn main() {
    eprintln!("This example requires the 'datetime_ops' feature.");
    eprintln!("Run with: cargo run --example datetime_ops --features datetime_ops");
}
