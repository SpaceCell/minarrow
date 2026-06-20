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

//! ---------------------------------------------------------
//! Builds a demo table and prints it.
//!
//! Run with:
//!     cargo run --example print_arrays
//! ---------------------------------------------------------

use std::sync::Arc;

use minarrow::aliases::{BoolArr, FltArr, IntArr, StrArr};
use minarrow::enums::array::Array;
use minarrow::{Bitmask, MaskedArray, NumericArray, Print, TextArray};

#[cfg(feature = "views")]
use minarrow::{ArrayV, BitmaskV, NumericArrayV, TextArrayV};

#[cfg(feature = "datetime")]
use minarrow::DatetimeArray;

fn main() {
    // Numeric (Integer, Float, all sizes)
    let col_i32 = IntArr::<i32>::from_slice(&[1, 2, 3, 4, 5]);
    let col_u32 = IntArr::<u32>::from_slice(&[100, 200, 300, 400, 500]);
    let col_f32 = FltArr::<f32>::from_slice(&[1.1, 2.2, 3.3, 4.4, 5.5]);

    // Boolean with nulls
    let mut col_bool = BoolArr::from_slice(&[true, false, true, false, true]);
    col_bool.set_null_mask(Some(Bitmask::from_bools(&[true, true, true, false, true])));

    // String and Dictionary/Categorical
    let col_str32 = StrArr::from_slice(&["red", "blue", "green", "yellow", "purple"]);

    // --- Print NumericArray, TextArray, TemporalArray enums
    println!("\n--- Enums: NumericArray, TextArray, TemporalArray ---");
    NumericArray::Int32(Arc::new(col_i32.clone())).print();
    println!("\n");
    NumericArray::UInt32(Arc::new(col_u32.clone())).print();
    println!("\n");
    NumericArray::Float32(Arc::new(col_f32.clone())).print();
    println!("\n");
    TextArray::String32(Arc::new(col_str32.clone())).print();
    println!("\n");

    println!("\n--- Array (top-level) ---");
    Array::from_int32(col_i32.clone()).print();
    println!("\n");
    Array::from_uint32(col_u32.clone()).print();
    println!("\n");
    Array::from_float32(col_f32.clone()).print();
    println!("\n");
    Array::from_string32(col_str32.clone()).print();
    println!("\n");
    // --- Print Array Views (ArrayV, NumericArrayV, TextArrayV, TemporalArrayV)
    #[cfg(feature = "views")]
    println!("\n--- Array Views ---");
    #[cfg(feature = "views")]
    ArrayV::new(Array::from_int32(col_i32.clone()), 1, 3).print();

    let num_arr = NumericArray::Int32(Arc::new(col_i32.clone()));
    num_arr.print();

    #[cfg(feature = "views")]
    let num_view = NumericArrayV::new(num_arr, 1, 3);
    #[cfg(feature = "views")]
    num_view.print();

    let txt_arr = TextArray::String32(Arc::new(col_str32.clone()));
    txt_arr.print();

    #[cfg(feature = "views")]
    let txt_view = TextArrayV::new(txt_arr, 1, 3);
    #[cfg(feature = "views")]
    txt_view.print();

    // --- Print Bitmask and BitmaskV
    println!("\n--- Bitmask & BitmaskV ---");
    let bm = Bitmask::from_bools(&[true, false, true, true, false]);
    bm.print();
    #[cfg(feature = "views")]
    BitmaskV::new(&bm, 1, 3).print();

    // Datetime - various time units
    #[cfg(feature = "datetime")]
    {
        use minarrow::enums::time_units::TimeUnit;

        println!("\n--- Datetime Arrays (various time units) ---");

        // Seconds since Unix epoch (1970-01-01 00:00:00 UTC)
        let dt_seconds = DatetimeArray::<i64>::from_slice(
            &[
                1_700_000_000, // 2023-11-14 22:13:20 UTC
                1_700_086_400, // 2023-11-15 22:13:20 UTC
                1_700_172_800, // 2023-11-16 22:13:20 UTC
            ],
            Some(TimeUnit::Seconds),
        );
        println!("Seconds:");
        dt_seconds.print();
        println!();

        // Milliseconds
        let dt_millis = DatetimeArray::<i64>::from_slice(
            &[
                1_700_000_000_000, // 2023-11-14 22:13:20.000 UTC
                1_700_086_400_000, // 2023-11-15 22:13:20.000 UTC
                1_700_172_800_000, // 2023-11-16 22:13:20.000 UTC
            ],
            Some(TimeUnit::Milliseconds),
        );
        println!("Milliseconds:");
        dt_millis.print();
        println!();

        // Microseconds
        let dt_micros = DatetimeArray::<i64>::from_slice(
            &[
                1_700_000_000_000_000, // 2023-11-14 22:13:20.000000 UTC
                1_700_086_400_000_000, // 2023-11-15 22:13:20.000000 UTC
                1_700_172_800_000_000, // 2023-11-16 22:13:20.000000 UTC
            ],
            Some(TimeUnit::Microseconds),
        );
        println!("Microseconds:");
        dt_micros.print();
        println!();

        // Nanoseconds
        let dt_nanos = DatetimeArray::<i64>::from_slice(
            &[
                1_700_000_000_000_000_000, // 2023-11-14 22:13:20.000000000 UTC
                1_700_086_400_000_000_000, // 2023-11-15 22:13:20.000000000 UTC
                1_700_172_800_000_000_000, // 2023-11-16 22:13:20.000000000 UTC
            ],
            Some(TimeUnit::Nanoseconds),
        );
        println!("Nanoseconds:");
        dt_nanos.print();
        println!();

        // Days since Unix epoch
        let dt_days = DatetimeArray::<i32>::from_slice(
            &[
                19_670, // 2023-11-14
                19_671, // 2023-11-15
                19_672, // 2023-11-16
            ],
            Some(TimeUnit::Days),
        );
        println!("Days:");
        dt_days.print();
        println!();

        // With timezone operations (requires datetime_ops feature)
        #[cfg(feature = "datetime_ops")]
        {
            println!(
                "--- Datetime with Timezone Conversions (requires 'datetime_ops' feature) ---"
            );

            // UTC datetime
            let utc_dt =
                DatetimeArray::<i64>::from_slice(&[1_700_000_000], Some(TimeUnit::Seconds));

            // Test IANA timezone identifiers
            println!("IANA Timezone Identifiers:");
            println!("America/New_York:");
            utc_dt.tz("America/New_York").print();
            println!();

            println!("Australia/Sydney:");
            utc_dt.tz("Australia/Sydney").print();
            println!();

            println!("Europe/London:");
            utc_dt.tz("Europe/London").print();
            println!();

            println!("Asia/Tokyo:");
            utc_dt.tz("Asia/Tokyo").print();
            println!();

            // Test timezone abbreviations
            println!("\nTimezone Abbreviations:");
            println!("EST:");
            utc_dt.tz("EST").print();
            println!();

            println!("AEST:");
            utc_dt.tz("AEST").print();
            println!();

            println!("JST:");
            utc_dt.tz("JST").print();
            println!();

            // Test direct offset strings
            println!("\nDirect Offset Strings:");
            println!("UTC:");
            utc_dt.tz("UTC").print();
            println!();

            println!("+05:30 (India):");
            utc_dt.tz("+05:30").print();
            println!();

            println!("-03:30 (Newfoundland):");
            utc_dt.tz("-03:30").print();
            println!();
        }

        #[cfg(not(feature = "datetime_ops"))]
        {
            println!(
                "Note: Enable 'datetime_ops' feature for timezone conversions and datetime operations."
            );
            println!();
        }
    }
}
