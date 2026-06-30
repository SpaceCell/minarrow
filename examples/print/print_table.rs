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
//!     cargo run --example print_table
//! ---------------------------------------------------------

#[cfg(any(
    not(feature = "default_categorical_8"),
    feature = "extended_categorical"
))]
use minarrow::aliases::CatArr;
use minarrow::aliases::{BoolArr, FltArr, IntArr, StrArr};
use minarrow::{Bitmask, FieldArray, MaskedArray, Print, Table};
#[cfg(feature = "datetime")]
use minarrow::{DatetimeArray, enums::time_units::TimeUnit};

fn main() {
    // Inner arrays

    // Numeric
    let col_i32 = IntArr::<i32>::from_slice(&[1, 2, 3, 4, 5]);
    let col_u32 = IntArr::<u32>::from_slice(&[100, 200, 300, 400, 500]);
    let col_i64 = IntArr::<i64>::from_slice(&[10, 20, 30, 40, 50]);
    let col_u64 = IntArr::<u64>::from_slice(&[101, 201, 301, 401, 501]);
    let col_f32 = FltArr::<f32>::from_slice(&[1.1, 2.2, 3.3, 4.4, 5.5]);
    let col_f64 = FltArr::<f64>::from_slice(&[2.2, 3.3, 4.4, 5.5, 6.6]);

    // Boolean with nulls
    let mut col_bool = BoolArr::from_slice(&[true, false, true, false, true]);
    col_bool.set_null_mask(Some(Bitmask::from_bools(&[true, true, true, false, true])));

    // String and Dictionary/Categorical
    let col_str32 = StrArr::<u32>::from_slice(&["red", "blue", "green", "yellow", "purple"]);
    #[cfg(any(
        not(feature = "default_categorical_8"),
        feature = "extended_categorical"
    ))]
    let col_cat32 = CatArr::<u32>::from_values(
        ["apple", "banana", "cherry", "banana", "apple"]
            .iter()
            .copied(),
    );

    // Datetime
    #[cfg(feature = "datetime")]
    let col_dt32 = DatetimeArray::<i32>::from_slice(
        &[1000, 2000, 3000, 4000, 5000],
        Some(TimeUnit::Milliseconds),
    );
    #[cfg(feature = "datetime")]
    let col_dt64 = DatetimeArray::<i64>::from_slice(
        &[
            1_000_000_000,
            2_000_000_000,
            3_000_000_000,
            4_000_000_000,
            5_000_000_000,
        ],
        Some(TimeUnit::Nanoseconds),
    );

    // FieldArray (column) construction
    let fa_i32 = FieldArray::from_arr("int32_col", col_i32);
    let fa_u32 = FieldArray::from_arr("uint32_col", col_u32);
    let fa_i64 = FieldArray::from_arr("int64_col", col_i64);
    let fa_u64 = FieldArray::from_arr("uint64_col", col_u64);
    let fa_f32 = FieldArray::from_arr("float32_col", col_f32);
    let fa_f64 = FieldArray::from_arr("float64_col", col_f64);
    let fa_bool = FieldArray::from_arr("bool_col", col_bool);
    let fa_str32 = FieldArray::from_arr("utf8_col", col_str32);
    #[cfg(any(
        not(feature = "default_categorical_8"),
        feature = "extended_categorical"
    ))]
    let fa_cat32 = FieldArray::from_arr("dict32_col", col_cat32);
    #[cfg(feature = "datetime")]
    let fa_dt32 = FieldArray::from_arr("datetime32_col", col_dt32);
    #[cfg(feature = "datetime")]
    let fa_dt64 = FieldArray::from_arr("datetime64_col", col_dt64);

    // Build Table
    let mut tbl = Table::new("MyTable".to_string(), None);
    tbl.add_col(fa_i32);
    tbl.add_col(fa_u32);
    tbl.add_col(fa_i64);
    tbl.add_col(fa_u64);
    tbl.add_col(fa_f32);
    tbl.add_col(fa_f64);
    tbl.add_col(fa_bool);
    tbl.add_col(fa_str32);
    #[cfg(any(
        not(feature = "default_categorical_8"),
        feature = "extended_categorical"
    ))]
    tbl.add_col(fa_cat32);
    #[cfg(feature = "datetime")]
    tbl.add_col(fa_dt32);
    #[cfg(feature = "datetime")]
    tbl.add_col(fa_dt64);

    // Print the table
    tbl.print();
}
