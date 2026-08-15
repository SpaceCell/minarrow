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
//! Display output for every printable Minarrow type.
//!
//! Run with:
//!     cargo run --example print \
//!         --features value_type,views,chunked,scalar_type,matrix,ndarray,xarray,cube,select
//! ---------------------------------------------------------

fn main() {
    arrays_section();
    table_section();

    #[cfg(feature = "matrix")]
    matrix_section();
    #[cfg(feature = "ndarray")]
    ndarray_section();
    #[cfg(all(feature = "ndarray", feature = "views", feature = "select"))]
    ndarray_view_section();
    #[cfg(all(feature = "ndarray", feature = "chunked"))]
    super_ndarray_section();
    #[cfg(feature = "chunked")]
    super_array_section();
    #[cfg(feature = "chunked")]
    super_table_section();
    #[cfg(feature = "xarray")]
    xarray_section();
    #[cfg(feature = "cube")]
    cube_section();
    #[cfg(feature = "value_type")]
    value_section();
}

fn arrays_section() {
    use std::sync::Arc;

    use minarrow::aliases::{BoolArr, FltArr, IntArr, StrArr};
    use minarrow::enums::array::Array;
    use minarrow::{Bitmask, MaskedArray, NumericArray, Print, TextArray};

    let col_i32 = IntArr::<i32>::from_slice(&[1, 2, 3, 4, 5]);
    let col_u32 = IntArr::<u32>::from_slice(&[100, 200, 300, 400, 500]);
    let col_f32 = FltArr::<f32>::from_slice(&[1.1, 2.2, 3.3, 4.4, 5.5]);

    // Boolean with nulls
    let mut col_bool = BoolArr::from_slice(&[true, false, true, false, true]);
    col_bool.set_null_mask(Some(Bitmask::from_bools(&[true, true, true, false, true])));

    let col_str32 = StrArr::from_slice(&["red", "blue", "green", "yellow", "purple"]);

    println!("===== Enums: NumericArray, TextArray =====\n");
    NumericArray::Int32(Arc::new(col_i32.clone())).print();
    println!();
    NumericArray::UInt32(Arc::new(col_u32.clone())).print();
    println!();
    NumericArray::Float32(Arc::new(col_f32.clone())).print();
    println!();
    TextArray::String32(Arc::new(col_str32.clone())).print();

    println!("\n===== Array (top-level) =====\n");
    Array::from_int32(col_i32.clone()).print();
    println!();
    Array::from_uint32(col_u32.clone()).print();
    println!();
    Array::from_float32(col_f32.clone()).print();
    println!();
    Array::from_string32(col_str32.clone()).print();

    #[cfg(feature = "views")]
    {
        use minarrow::{ArrayV, BitmaskV, NumericArrayV, TextArrayV};

        println!("\n===== Array views =====\n");
        ArrayV::new(Array::from_int32(col_i32.clone()), 1, 3).print();

        let num_arr = NumericArray::Int32(Arc::new(col_i32.clone()));
        NumericArrayV::new(num_arr, 1, 3).print();

        let txt_arr = TextArray::String32(Arc::new(col_str32.clone()));
        TextArrayV::new(txt_arr, 1, 3).print();

        println!("\n===== Bitmask & BitmaskV =====\n");
        let bm = Bitmask::from_bools(&[true, false, true, true, false]);
        bm.print();
        BitmaskV::new(&bm, 1, 3).print();
    }

    // Datetime - various time units
    #[cfg(feature = "datetime")]
    {
        use minarrow::DatetimeArray;
        use minarrow::enums::time_units::TimeUnit;

        println!("\n===== Datetime arrays (various time units) =====\n");

        // Seconds since the Unix epoch (1970-01-01 00:00:00 UTC)
        let dt_seconds = DatetimeArray::<i64>::from_slice(
            &[1_700_000_000, 1_700_086_400, 1_700_172_800],
            Some(TimeUnit::Seconds),
        );
        println!("Seconds:");
        dt_seconds.print();
        println!();

        let dt_millis = DatetimeArray::<i64>::from_slice(
            &[1_700_000_000_000, 1_700_086_400_000, 1_700_172_800_000],
            Some(TimeUnit::Milliseconds),
        );
        println!("Milliseconds:");
        dt_millis.print();
        println!();

        let dt_days =
            DatetimeArray::<i32>::from_slice(&[19_670, 19_671, 19_672], Some(TimeUnit::Days));
        println!("Days:");
        dt_days.print();

        // Timezone conversions (requires the datetime_ops feature)
        #[cfg(feature = "datetime_ops")]
        {
            println!("\n===== Datetime with timezone conversions =====\n");
            let utc_dt = DatetimeArray::<i64>::from_slice(&[1_700_000_000], Some(TimeUnit::Seconds));
            println!("America/New_York:");
            utc_dt.tz("America/New_York").print();
            println!("\nAustralia/Sydney:");
            utc_dt.tz("Australia/Sydney").print();
            println!("\n+05:30 (India):");
            utc_dt.tz("+05:30").print();
        }
    }
}

fn table_section() {
    use minarrow::aliases::{BoolArr, FltArr, IntArr, StrArr};
    use minarrow::{Bitmask, FieldArray, MaskedArray, Print, Table};

    println!("\n===== Table =====\n");

    let col_i32 = IntArr::<i32>::from_slice(&[1, 2, 3, 4, 5]);
    let col_u64 = IntArr::<u64>::from_slice(&[101, 201, 301, 401, 501]);
    let col_f64 = FltArr::<f64>::from_slice(&[2.2, 3.3, 4.4, 5.5, 6.6]);

    let mut col_bool = BoolArr::from_slice(&[true, false, true, false, true]);
    col_bool.set_null_mask(Some(Bitmask::from_bools(&[true, true, true, false, true])));

    let col_str32 = StrArr::<u32>::from_slice(&["red", "blue", "green", "yellow", "purple"]);

    let mut table = Table::new("MyTable".to_string(), None);
    table.add_col(FieldArray::from_arr("int32_col", col_i32));
    table.add_col(FieldArray::from_arr("uint64_col", col_u64));
    table.add_col(FieldArray::from_arr("float64_col", col_f64));
    table.add_col(FieldArray::from_arr("bool_col", col_bool));
    table.add_col(FieldArray::from_arr("utf8_col", col_str32));

    #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
    {
        use minarrow::aliases::CatArr;
        let col_cat32 = CatArr::<u32>::from_values(
            ["apple", "banana", "cherry", "banana", "apple"].iter().copied(),
        );
        table.add_col(FieldArray::from_arr("dict32_col", col_cat32));
    }

    #[cfg(feature = "datetime")]
    {
        use minarrow::DatetimeArray;
        use minarrow::enums::time_units::TimeUnit;
        let col_dt64 = DatetimeArray::<i64>::from_slice(
            &[1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000, 5_000_000_000],
            Some(TimeUnit::Nanoseconds),
        );
        table.add_col(FieldArray::from_arr("datetime64_col", col_dt64));
    }

    table.print();
}

#[cfg(feature = "matrix")]
fn matrix_section() {
    use minarrow::{Matrix, Print, mat};

    println!("\n===== Matrix =====\n");
    mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].print();

    // A matrix taller than the preview shows its first and last ten rows.
    println!("\n----- Matrix (preview elision) -----\n");
    let tall: Vec<f64> = (0..120).map(|x| x as f64).collect();
    Matrix::from_f64_unaligned(&tall, 60, 2, Some("tall".to_string())).print();
}

#[cfg(feature = "ndarray")]
fn ndarray_section() {
    use minarrow::{NdArray, Print};

    println!("\n===== NdArray =====\n");
    println!("----- rank 1 -----\n");
    NdArray::from_slice(&[10.0, 20.0, 30.0, 40.0], &[4]).print();

    println!("\n----- rank 2 -----\n");
    NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).print();

    println!("\n----- rank 3 (trailing axes flatten into columns) -----\n");
    NdArray::from_slice(&(0..24).map(|x| x as f64).collect::<Vec<_>>(), &[2, 3, 4]).print();
}

#[cfg(all(feature = "ndarray", feature = "views", feature = "select"))]
fn ndarray_view_section() {
    use minarrow::{NdArray, Print, nd};

    println!("\n===== NdArrayView =====\n");
    let nd2 = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    nd2.slice(nd![1..3, 0..2]).print();
}

#[cfg(all(feature = "ndarray", feature = "chunked"))]
fn super_ndarray_section() {
    use minarrow::{NdArray, Print, SuperNdArray};

    println!("\n===== SuperNdArray =====\n");
    let batch_a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let batch_b = NdArray::from_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[3, 2]);
    let super_nd = SuperNdArray::from_batches(vec![batch_a, batch_b], "sensor");
    super_nd.print();

    #[cfg(feature = "views")]
    {
        println!("\n===== SuperNdArrayView =====\n");
        super_nd.slice(1, 3).print();
    }
}

#[cfg(feature = "chunked")]
fn super_array_section() {
    use minarrow::{Print, SuperArray, fa_i64};

    println!("\n===== SuperArray =====\n");
    let super_arr =
        SuperArray::from_chunks(vec![fa_i64!("price", 100, 200, 300), fa_i64!("price", 400, 500)]);
    super_arr.print();

    #[cfg(feature = "views")]
    {
        println!("\n===== SuperArrayView =====\n");
        super_arr.slice(1, 3).print();
    }
}

#[cfg(feature = "chunked")]
fn super_table_section() {
    use std::sync::Arc;

    use minarrow::{Print, SuperTable, fa_f64, fa_i64, tbl};

    println!("\n===== SuperTable =====\n");
    let batch_1 = tbl!("b1", fa_i64!("qty", 1, 2), fa_f64!("px", 9.5, 10.5));
    let batch_2 = tbl!("b2", fa_i64!("qty", 3, 4, 5), fa_f64!("px", 11.5, 12.5, 13.5));
    let super_tbl = SuperTable::from_batches(
        vec![Arc::new(batch_1), Arc::new(batch_2)],
        Some("orders".to_string()),
    );
    super_tbl.print();

    #[cfg(feature = "views")]
    {
        println!("\n===== SuperTableView =====\n");
        super_tbl.view(1, 3).print();
    }
}

#[cfg(feature = "xarray")]
fn xarray_section() {
    use minarrow::{NdArray, Print, XArray, arr_f64};

    println!("\n===== XArray =====\n");
    let mut xa = XArray::new(
        NdArray::from_slice(&[20.1, 20.4, 20.9, 1.01, 1.02, 1.00], &[3, 2]),
        &["hour", "measurement"],
    );
    xa.assign_coords("hour", arr_f64![0.0, 1.0, 2.0]);
    xa.print();
}

#[cfg(feature = "cube")]
fn cube_section() {
    use minarrow::{Cube, Print, fa_bool, fa_i64, tbl};

    println!("\n===== Cube =====\n");
    let source = tbl!(
        "trades",
        fa_i64!("id", 1, 2, 1, 3),
        fa_bool!("flag", true, false, true, false)
    );
    Cube::from_table(&source, "id", "grouped").unwrap().print();
}

#[cfg(feature = "value_type")]
fn value_section() {
    use std::sync::Arc;

    use minarrow::{Print, Value, fa_i64, tbl};

    println!("\n===== Value =====\n");
    let table = tbl!("px", fa_i64!("qty", 1, 2, 3));
    Value::Table(Arc::new(table.clone())).print();

    println!("\n----- Value::Tuple2 -----\n");
    Value::Tuple2(Arc::new((
        Value::Table(Arc::new(table.clone())),
        Value::Table(Arc::new(table.clone())),
    )))
    .print();

    println!("\n----- Value::VecValue -----\n");
    Value::VecValue(Arc::new(vec![
        Value::Table(Arc::new(table.clone())),
        Value::Table(Arc::new(table)),
    ]))
    .print();

    #[cfg(feature = "matrix")]
    {
        use minarrow::{Matrix, mat};

        println!("\n----- Value::Matrix -----\n");
        let matrix: Matrix = mat![[1.0, 2.0], [3.0, 4.0]];
        Value::Matrix(Arc::new(matrix)).print();
    }
}
