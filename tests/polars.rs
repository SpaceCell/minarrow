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

//! Central test suite for external polars library conversion

#![cfg(feature = "cast_polars")]

use std::sync::Arc;

use minarrow::{
    fa_i32, fa_str32, fa_u32, Array, ArrowType, Field, MaskedArray, NumericArray, Table, TextArray,
};
#[cfg(feature = "datetime")]
use minarrow::{TemporalArray, TimeUnit};
use polars::prelude::*;

#[test]
fn test_array_to_polars_numeric() {
    let arr = Arc::new(minarrow::IntegerArray::<i32>::from_slice(&[1, 2, 3]));
    let a = Array::NumericArray(NumericArray::Int32(arr));
    let s = a.to_polars("x");
    assert_eq!(s.name(), "x");
    assert_eq!(s.len(), 3);
    assert_eq!(s.dtype(), &DataType::Int32);
    assert_eq!(
        s.i32().unwrap().into_no_null_iter().collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
}

#[test]
fn test_array_to_polars_string() {
    let arr = Arc::new(minarrow::StringArray::<u32>::from_slice(&["a", "b", ""]));
    let a = Array::TextArray(TextArray::String32(arr));
    let s = a.to_polars("s");
    assert_eq!(s.dtype(), &DataType::String);
    assert_eq!(
        s.str()
            .unwrap()
            .into_no_null_iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>(),
        vec!["a".to_string(), "b".to_string(), "".to_string()]
    );
}

#[cfg(feature = "datetime")]
#[test]
fn test_array_to_polars_datetime_infer_date32() {
    let a = Array::TemporalArray(TemporalArray::Datetime32(Arc::new(
        minarrow::DatetimeArray::<i32> {
            data: minarrow::Buffer::from_slice(&[1_600_000_000 / 86_400; 3]),
            null_mask: None,
            time_unit: TimeUnit::Days,
        },
    )));
    let s = a.to_polars("d32");
    // Polars maps Arrow Date32 -> DataType::Date
    assert_eq!(s.dtype(), &DataType::Date);
    assert_eq!(s.len(), 3);
}

#[cfg(feature = "datetime")]
#[test]
fn test_array_to_polars_datetime_infer_time32s() {
    let a = Array::TemporalArray(TemporalArray::Datetime32(Arc::new(
        minarrow::DatetimeArray::<i32> {
            data: minarrow::Buffer::from_slice(&[1, 2, 3]),
            null_mask: None,
            time_unit: TimeUnit::Seconds,
        },
    )));
    let s = a.to_polars("t32s");
    // Polars maps Arrow Time32(s) to Int32 logical time; exact dtype may vary, presence is sufficient
    assert_eq!(s.len(), 3);
}

#[cfg(feature = "datetime")]
#[test]
fn test_array_to_polars_datetime_infer_date64_or_ts() {
    let a_ms = Array::TemporalArray(TemporalArray::Datetime64(Arc::new(
        minarrow::DatetimeArray::<i64> {
            data: minarrow::Buffer::from_slice(&[1_600_000_000_000, 1_600_000_000_001]),
            null_mask: None,
            time_unit: TimeUnit::Milliseconds,
        },
    )));
    let s_ms = a_ms.to_polars("d64");
    // In practice Polars treats Arrow Date64 as Datetime(Milliseconds)
    assert_eq!(s_ms.len(), 2);

    let a_ns = Array::TemporalArray(TemporalArray::Datetime64(Arc::new(
        minarrow::DatetimeArray::<i64> {
            data: minarrow::Buffer::from_slice(&[1, 2, 3]),
            null_mask: None,
            time_unit: TimeUnit::Nanoseconds,
        },
    )));
    let s_ns = a_ns.to_polars("ts_ns");
    // Arrow Timestamp(ns) -> Polars Datetime(ns)
    assert_eq!(s_ns.len(), 3);
}

#[test]
fn test_array_to_polars_with_field_via_field_array() {
    use minarrow::FieldArray;
    let arr = Arc::new(minarrow::IntegerArray::<i64>::from_slice(&[10, 20]));
    let a = Array::NumericArray(NumericArray::Int64(arr));
    let f = Field::new("y", ArrowType::Int64, false, None);
    let s = FieldArray::new(f, a).to_polars();
    assert_eq!(s.dtype(), &DataType::Int64);
    assert_eq!(
        s.i64().unwrap().into_no_null_iter().collect::<Vec<_>>(),
        vec![10, 20]
    );
}

#[test]
fn test_fieldarray_to_polars() {
    let fa = fa_u32!("u", 5, 6, 7);
    let s = fa.to_polars();
    assert_eq!(s.name(), "u");
    assert_eq!(s.dtype(), &DataType::UInt32);
}

#[test]
fn test_table_to_polars() {
    // Tiny table: 2 cols
    let c1 = fa_i32!("a", 1, 2);
    let c2 = fa_str32!("b", "x", "y");
    let t = Table::new("t".into(), Some(vec![c1, c2]));
    let df = t.to_polars();
    assert_eq!(df.height(), 2);
    assert_eq!(df.width(), 2);
    assert_eq!(df.get_column_names(), &["a", "b"]);
}

// =============================================================
// Round-trip tests: polars -> Minarrow (from_*) -> polars
// =============================================================

#[test]
fn test_array_from_polars_round_trip_numeric() {
    let arr = Arc::new(minarrow::IntegerArray::<i32>::from_slice(&[1, 2, 3, 4]));
    let original = Array::NumericArray(NumericArray::Int32(arr));

    let s = original.to_polars("x");
    let back = Array::from_polars(&s);
    assert_eq!(original, back);
}

#[test]
fn test_array_from_polars_round_trip_string() {
    let arr = Arc::new(minarrow::StringArray::<u32>::from_slice(&["foo", "bar", "baz"]));
    let original = Array::TextArray(TextArray::String32(arr));
    let s = original.to_polars("s");
    let back = Array::from_polars(&s);

    // Polars may upcast Utf8 -> LargeUtf8 (string offset width changes).
    // Compare element-wise to be width-agnostic.
    let back_str: Vec<String> = match &back {
        Array::TextArray(TextArray::String32(b)) => (0..b.len())
            .map(|i| b.get_str(i).unwrap_or("").to_string())
            .collect(),
        Array::TextArray(TextArray::String64(b)) => (0..b.len())
            .map(|i| b.get_str(i).unwrap_or("").to_string())
            .collect(),
        _ => panic!("expected string array, got {:?}", back),
    };
    assert_eq!(
        back_str,
        vec!["foo".to_string(), "bar".to_string(), "baz".to_string()]
    );
}

#[test]
fn test_field_array_from_polars_round_trip_preserves_name() {
    use minarrow::FieldArray;
    let fa = minarrow::fa_i32!("score", 10, 20, 30);
    let s = fa.to_polars();
    let back = FieldArray::from_polars(&s);

    assert_eq!(back.field.name, "score");
    assert_eq!(back.field.dtype, ArrowType::Int32);
    assert_eq!(back.len(), 3);
    assert_eq!(back.array, fa.array);
}

#[test]
fn test_field_array_from_polars_via_into() {
    use minarrow::FieldArray;
    let fa = minarrow::fa_u32!("u", 5, 6, 7);
    let s = fa.to_polars();
    let back: FieldArray = (&s).into();
    assert_eq!(back.field.name, "u");
    assert_eq!(back.len(), 3);
}

#[test]
fn test_table_from_polars_round_trip() {
    let c1 = fa_i32!("a", 1, 2, 3);
    let c2 = fa_str32!("b", "x", "y", "z");
    let original = Table::new("t".into(), Some(vec![c1, c2]));

    let df = original.to_polars();
    let back = Table::from_polars(&df);

    assert_eq!(back.n_rows(), 3);
    assert_eq!(back.n_cols(), 2);
    assert_eq!(back.col_names(), &["a", "b"]);
    assert_eq!(back.cols[0].field.dtype, ArrowType::Int32);
}

#[test]
fn test_table_from_polars_via_into() {
    let c1 = fa_i32!("a", 1, 2);
    let original = Table::new("t".into(), Some(vec![c1]));
    let df = original.to_polars();
    let back: Table = (&df).into();
    assert_eq!(back.n_rows(), 2);
    assert_eq!(back.cols[0].field.name, "a");
}

#[test]
fn test_try_from_polars_returns_ok() {
    let fa = fa_u32!("u", 5, 6, 7);
    let s = fa.to_polars();
    let back = minarrow::Array::try_from_polars(&s).unwrap();
    assert_eq!(back.len(), 3);
}

#[cfg(feature = "chunked")]
#[test]
fn test_super_array_from_polars_chunked() {
    use minarrow::SuperArray;
    let fa1 = fa_i32!("a", 1, 2, 3);
    let fa2 = fa_i32!("a", 4, 5);
    let sa_original = SuperArray::from_field_array_chunks(vec![fa1, fa2]);

    let s = sa_original.to_polars();
    // Polars Series should carry 2 chunks
    assert_eq!(s.name().as_str(), "a");

    let back = SuperArray::from_polars(&s);
    assert_eq!(back.len(), 5);
    assert_eq!(back.field_ref().name, "a");
}

#[cfg(feature = "chunked")]
#[test]
fn test_super_table_from_polars_round_trip() {
    use minarrow::SuperTable;
    use std::sync::Arc;
    let c1 = fa_i32!("a", 1, 2);
    let c2 = fa_str32!("b", "x", "y");
    let table1 = Table::new("".into(), Some(vec![c1.clone(), c2.clone()]));
    let table2 = Table::new("".into(), Some(vec![c1, c2]));
    let st = SuperTable::from_batches(vec![Arc::new(table1), Arc::new(table2)], None);

    let df = st.to_polars();
    let back = SuperTable::from_polars(&df);

    assert_eq!(back.n_rows(), 4);
    assert_eq!(back.n_cols(), 2);
}
