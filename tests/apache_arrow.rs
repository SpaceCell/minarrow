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

//! Central test suite for Apache Arrow conversion

#![cfg(feature = "cast_arrow")]

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, Date32Array, Date64Array, Int32Array, Int64Array, StringArray,
    Time32SecondArray, TimestampNanosecondArray, UInt32Array,
};
use arrow::datatypes::{DataType as ADataType, TimeUnit as ATimeUnit};
use arrow::record_batch::RecordBatch;

use minarrow::{
    fa_i32, fa_str32, fa_u32, Array as MArray, ArrowType, Field, FieldArray, NumericArray, Table,
    TextArray,
};

#[cfg(feature = "datetime")]
use minarrow::{TemporalArray, TimeUnit};

// -------------------------------
// Array -> Arrow (numeric)
// -------------------------------
#[test]
fn test_array_to_arrow_numeric() {
    let arr = Arc::new(minarrow::IntegerArray::<i32>::from_slice(&[1, 2, 3]));
    let a = MArray::NumericArray(NumericArray::Int32(arr));
    let ar: ArrayRef = a.to_apache_arrow("x");

    assert_eq!(ar.data_type(), &ADataType::Int32);

    let col = ar.as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(col.len(), 3);
    assert_eq!(col.value(0), 1);
    assert_eq!(col.value(1), 2);
    assert_eq!(col.value(2), 3);
}

// -------------------------------
// Array -> Arrow (utf8)
// -------------------------------
#[test]
fn test_array_to_arrow_string() {
    let arr = Arc::new(minarrow::StringArray::<u32>::from_slice(&["a", "b", ""]));
    let a = MArray::TextArray(TextArray::String32(arr));
    let ar = a.to_apache_arrow("s");

    assert_eq!(ar.data_type(), &ADataType::Utf8);

    let col = ar.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(col.len(), 3);
    assert_eq!(col.value(0), "a");
    assert_eq!(col.value(1), "b");
    assert_eq!(col.value(2), "");
}

#[cfg(feature = "datetime")]
#[test]
fn test_array_to_arrow_datetime_infer_date32() {
    // Date32 = days since epoch
    let a = MArray::TemporalArray(TemporalArray::Datetime32(Arc::new(
        minarrow::DatetimeArray::<i32> {
            data: minarrow::Buffer::from_slice(&[1_600_000_000 / 86_400; 3]),
            null_mask: None,
            time_unit: TimeUnit::Days,
        },
    )));
    let ar = a.to_apache_arrow("d32");

    assert_eq!(ar.data_type(), &ADataType::Date32);
    let col = ar.as_any().downcast_ref::<Date32Array>().unwrap();
    assert_eq!(col.len(), 3);
}

#[cfg(feature = "datetime")]
#[test]
fn test_array_to_arrow_datetime_infer_time32s() {
    // Time32(Second) - use explicit Field so logical type matches Seconds
    let a = MArray::TemporalArray(TemporalArray::Datetime32(Arc::new(
        minarrow::DatetimeArray::<i32> {
            data: minarrow::Buffer::from_slice(&[1, 2, 3]),
            null_mask: None,
            time_unit: TimeUnit::Seconds,
        },
    )));
    let f = Field::new("t32s", ArrowType::Time32(TimeUnit::Seconds), false, None);
    let ar = FieldArray::new(f, a).to_apache_arrow();

    assert_eq!(ar.data_type(), &ADataType::Time32(ATimeUnit::Second));
    let col = ar.as_any().downcast_ref::<Time32SecondArray>().unwrap();
    assert_eq!(col.len(), 3);
    assert_eq!(col.value(0), 1);
    assert_eq!(col.value(1), 2);
    assert_eq!(col.value(2), 3);
}

#[cfg(feature = "datetime")]
#[test]
fn test_array_to_arrow_datetime_infer_date64_and_ts_ns() {
    // Date64 = ms since epoch
    let a_ms = MArray::TemporalArray(TemporalArray::Datetime64(Arc::new(
        minarrow::DatetimeArray::<i64> {
            data: minarrow::Buffer::from_slice(&[1_600_000_000_000, 1_600_000_000_001]),
            null_mask: None,
            time_unit: TimeUnit::Milliseconds,
        },
    )));
    let ar_ms = a_ms.to_apache_arrow("d64");
    assert_eq!(ar_ms.data_type(), &ADataType::Date64);
    let c_ms = ar_ms.as_any().downcast_ref::<Date64Array>().unwrap();
    assert_eq!(c_ms.len(), 2);
    assert_eq!(c_ms.value(0), 1_600_000_000_000);
    assert_eq!(c_ms.value(1), 1_600_000_000_001);

    // Timestamp(ns) - explicit Field
    let a_ns = MArray::TemporalArray(TemporalArray::Datetime64(Arc::new(
        minarrow::DatetimeArray::<i64> {
            data: minarrow::Buffer::from_slice(&[1, 2, 3]),
            null_mask: None,
            time_unit: TimeUnit::Nanoseconds,
        },
    )));
    let f_tsns = Field::new(
        "ts_ns",
        ArrowType::Timestamp(TimeUnit::Nanoseconds, None),
        false,
        None,
    );
    let ar_ns = FieldArray::new(f_tsns, a_ns).to_apache_arrow();
    assert_eq!(
        ar_ns.data_type(),
        &ADataType::Timestamp(ATimeUnit::Nanosecond, None)
    );
    let c_ns = ar_ns
        .as_any()
        .downcast_ref::<TimestampNanosecondArray>()
        .unwrap();
    assert_eq!(c_ns.len(), 3);
    assert_eq!(c_ns.value(0), 1);
    assert_eq!(c_ns.value(1), 2);
    assert_eq!(c_ns.value(2), 3);
}

#[test]
fn test_array_to_arrow_with_field_via_field_array() {
    let arr = Arc::new(minarrow::IntegerArray::<i64>::from_slice(&[10, 20]));
    let a = MArray::NumericArray(NumericArray::Int64(arr));
    let f = Field::new("y", ArrowType::Int64, false, None);
    let ar = FieldArray::new(f, a).to_apache_arrow();

    assert_eq!(ar.data_type(), &ADataType::Int64);
    let col = ar.as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(col.len(), 2);
    assert_eq!(col.value(0), 10);
    assert_eq!(col.value(1), 20);
}

#[test]
fn test_fieldarray_to_arrow() {
    let fa = fa_u32!("u", 5, 6, 7);

    let ar = fa.to_apache_arrow();
    assert_eq!(ar.data_type(), &ADataType::UInt32);

    let col = ar.as_any().downcast_ref::<UInt32Array>().unwrap();
    assert_eq!(col.len(), 3);
    assert_eq!(col.value(0), 5);
    assert_eq!(col.value(1), 6);
    assert_eq!(col.value(2), 7);
}

// =============================================================
// Round-trip tests: arrow-rs -> Minarrow (from_*) -> arrow-rs
// =============================================================

#[test]
fn test_array_from_arrow_round_trip_numeric() {
    let arr = Arc::new(minarrow::IntegerArray::<i32>::from_slice(&[1, 2, 3, 4]));
    let original = MArray::NumericArray(NumericArray::Int32(arr));

    // Export to arrow-rs, then re-import as bare Array (field metadata dropped)
    let arrow_ref = original.to_apache_arrow("x");
    let back = MArray::from_apache_arrow(&arrow_ref);

    assert_eq!(original, back);
}

#[test]
fn test_array_from_arrow_round_trip_string() {
    let arr = Arc::new(minarrow::StringArray::<u32>::from_slice(&["foo", "bar", ""]));
    let original = MArray::TextArray(TextArray::String32(arr));
    let arrow_ref = original.to_apache_arrow("s");
    let back = MArray::from_apache_arrow(&arrow_ref);
    assert_eq!(original, back);
}

#[test]
fn test_field_array_from_arrow_round_trip_preserves_field() {
    let fa = fa_i32!("x", 10, 20, 30);
    let arrow_ref = fa.to_apache_arrow();
    let back = FieldArray::from_apache_arrow("x", &arrow_ref);

    assert_eq!(back.field.name, "x");
    assert_eq!(back.field.dtype, ArrowType::Int32);
    assert_eq!(back.len(), 3);
    assert_eq!(back.array, fa.array);
}

#[cfg(feature = "datetime")]
#[test]
fn test_field_array_from_arrow_round_trip_timestamp() {
    let dt = MArray::TemporalArray(TemporalArray::Datetime64(Arc::new(
        minarrow::DatetimeArray::<i64> {
            data: minarrow::Buffer::from_slice(&[1, 2, 3]),
            null_mask: None,
            time_unit: TimeUnit::Nanoseconds,
        },
    )));
    let field = Field::new(
        "ts_ns",
        ArrowType::Timestamp(TimeUnit::Nanoseconds, None),
        false,
        None,
    );
    let fa = FieldArray::new(field.clone(), dt);

    let arrow_ref = fa.to_apache_arrow();
    let back = FieldArray::from_apache_arrow("ts_ns", &arrow_ref);

    assert_eq!(back.field.name, "ts_ns");
    assert_eq!(
        back.field.dtype,
        ArrowType::Timestamp(TimeUnit::Nanoseconds, None)
    );
}

#[test]
fn test_table_from_arrow_round_trip() {
    let c1 = fa_i32!("a", 1, 2, 3);
    let c2 = fa_str32!("b", "x", "y", "z");
    let original = Table::new("t".into(), Some(vec![c1, c2]));

    let rb: RecordBatch = original.to_apache_arrow();
    let back = Table::from_apache_arrow(&rb);

    assert_eq!(back.n_rows(), 3);
    assert_eq!(back.n_cols(), 2);
    assert_eq!(back.col_names(), &["a", "b"]);
    for (lhs, rhs) in original.cols.iter().zip(back.cols.iter()) {
        assert_eq!(lhs.field.dtype, rhs.field.dtype);
        assert_eq!(lhs.array, rhs.array);
    }
}

#[test]
fn test_table_from_arrow_via_into() {
    let c1 = fa_i32!("a", 1, 2);
    let original = Table::new("t".into(), Some(vec![c1]));
    let rb: RecordBatch = original.to_apache_arrow();
    let back: Table = (&rb).into();
    assert_eq!(back.n_rows(), 2);
    assert_eq!(back.cols[0].field.name, "a");
}

#[test]
fn test_try_from_arrow_returns_ok() {
    let fa = fa_u32!("u", 5, 6, 7);
    let arrow_ref = fa.to_apache_arrow();
    let back = MArray::try_from_apache_arrow(&arrow_ref).unwrap();
    assert_eq!(back.len(), 3);
}

// =============================================================
// SuperArray / SuperTable round-trip tests
// =============================================================

#[cfg(feature = "chunked")]
#[test]
fn test_super_array_apache_arrow_round_trip() {
    use minarrow::SuperArray;
    let fa1 = fa_i32!("a", 1, 2, 3);
    let fa2 = fa_i32!("a", 4, 5);
    let sa = SuperArray::from_field_array_chunks(vec![fa1, fa2]);

    let chunks: Vec<ArrayRef> = sa.to_apache_arrow();
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].len(), 3);
    assert_eq!(chunks[1].len(), 2);

    let back = SuperArray::from_apache_arrow("a", &chunks);
    assert_eq!(back.n_chunks(), 2);
    assert_eq!(back.len(), 5);
    assert_eq!(back.field_ref().name, "a");
    assert_eq!(back.field_ref().dtype, ArrowType::Int32);
}

#[cfg(feature = "chunked")]
#[test]
fn test_super_table_apache_arrow_round_trip() {
    use minarrow::SuperTable;
    use std::sync::Arc;

    let c1a = fa_i32!("a", 1, 2);
    let c2a = fa_str32!("b", "x", "y");
    let batch1 = Table::new("".into(), Some(vec![c1a, c2a]));

    let c1b = fa_i32!("a", 3, 4, 5);
    let c2b = fa_str32!("b", "z", "w", "v");
    let batch2 = Table::new("".into(), Some(vec![c1b, c2b]));

    let st = SuperTable::from_batches(vec![Arc::new(batch1), Arc::new(batch2)], None);

    let rbs: Vec<RecordBatch> = st.to_apache_arrow();
    assert_eq!(rbs.len(), 2);
    assert_eq!(rbs[0].num_rows(), 2);
    assert_eq!(rbs[1].num_rows(), 3);

    let back = SuperTable::from_apache_arrow(&rbs);
    assert_eq!(back.n_batches(), 2);
    assert_eq!(back.n_rows(), 5);
    assert_eq!(back.n_cols(), 2);
    assert_eq!(back.batches()[0].n_rows(), 2);
    assert_eq!(back.batches()[1].n_rows(), 3);
}

#[cfg(feature = "chunked")]
#[test]
fn test_super_table_apache_arrow_via_into() {
    use minarrow::SuperTable;
    use std::sync::Arc;
    let c1 = fa_i32!("a", 1, 2);
    let batch = Table::new("".into(), Some(vec![c1]));
    let st = SuperTable::from_batches(vec![Arc::new(batch)], None);
    let rbs: Vec<RecordBatch> = st.to_apache_arrow();
    let back: SuperTable = rbs.as_slice().into();
    assert_eq!(back.n_batches(), 1);
    assert_eq!(back.n_rows(), 2);
}

#[test]
fn test_table_to_arrow_record_batch() {
    // 2 columns
    let c1 = fa_i32!("a", 1, 2);
    let c2 = fa_str32!("b", "x", "y");
    let t = Table::new("t".into(), Some(vec![c1, c2]));

    let rb: RecordBatch = t.to_apache_arrow();
    assert_eq!(rb.num_rows(), 2);
    assert_eq!(rb.num_columns(), 2);

    let a = rb.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let b = rb.column(1).as_any().downcast_ref::<StringArray>().unwrap();

    assert_eq!(a.value(0), 1);
    assert_eq!(a.value(1), 2);
    assert_eq!(b.value(0), "x");
    assert_eq!(b.value(1), "y");
}
