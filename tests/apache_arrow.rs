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
    Array as MArray, ArrowType, Field, FieldArray, MaskedArray, NumericArray, Table, TextArray,
    fa_i32, fa_str32, fa_u32,
};

#[cfg(feature = "datetime")]
use minarrow::{TemporalArray, TimeUnit};

// ----- helpers -----

/// Round-trip an `Array` through arrow-rs at the bare-Array level and assert
/// data equality. Discards Field metadata.
#[track_caller]
fn round_trip_array(a: MArray, name: &str) {
    let ar = a.to_apache_arrow(name);
    let back = MArray::from_apache_arrow(&ar);
    assert_eq!(a, back, "Array round-trip mismatch for '{}'", name);
}

/// Round-trip a `FieldArray` through arrow-rs preserving Field metadata.
#[track_caller]
fn round_trip_field_array(fa: FieldArray) {
    let name = fa.field.name.clone();
    let dtype = fa.field.dtype.clone();
    let ar = fa.to_apache_arrow();
    let back = FieldArray::from_apache_arrow(&name, &ar);
    assert_eq!(back.field.name, name, "name lost");
    assert_eq!(back.field.dtype, dtype, "dtype lost");
    assert_eq!(back.array, fa.array, "data mismatch for '{}'", name);
}

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
    let arr = Arc::new(minarrow::StringArray::<u32>::from_slice(&[
        "foo", "bar", "",
    ]));
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
    let original = Table::new("t", Some(vec![c1, c2]));

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
    let original = Table::new("t", Some(vec![c1]));
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
    let batch1 = Table::new("", Some(vec![c1a, c2a]));

    let c1b = fa_i32!("a", 3, 4, 5);
    let c2b = fa_str32!("b", "z", "w", "v");
    let batch2 = Table::new("", Some(vec![c1b, c2b]));

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
    let batch = Table::new("", Some(vec![c1]));
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
    let t = Table::new("t", Some(vec![c1, c2]));

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

// =============================================================
// Exhaustive type coverage round-trip
// =============================================================

#[test]
fn rt_arrow_i32() {
    round_trip_array(arr_i32_like(&[1, -2, 3, i32::MAX, i32::MIN]), "i32");
}

#[test]
fn rt_arrow_i64() {
    let mut a = minarrow::IntegerArray::<i64>::default();
    for v in &[1i64, -2, 3, i64::MAX, i64::MIN] {
        a.push(*v);
    }
    round_trip_array(MArray::from_int64(a), "i64");
}

#[test]
fn rt_arrow_u32() {
    let mut a = minarrow::IntegerArray::<u32>::default();
    for v in &[0u32, 1, u32::MAX] {
        a.push(*v);
    }
    round_trip_array(MArray::from_uint32(a), "u32");
}

#[test]
fn rt_arrow_u64() {
    let mut a = minarrow::IntegerArray::<u64>::default();
    for v in &[0u64, 1, u64::MAX] {
        a.push(*v);
    }
    round_trip_array(MArray::from_uint64(a), "u64");
}

#[test]
fn rt_arrow_f32() {
    let mut a = minarrow::FloatArray::<f32>::default();
    for v in &[0.0_f32, -1.5, 3.14, f32::INFINITY, f32::MIN, f32::MAX] {
        a.push(*v);
    }
    round_trip_array(MArray::from_float32(a), "f32");
}

#[test]
fn rt_arrow_f64() {
    let mut a = minarrow::FloatArray::<f64>::default();
    for v in &[0.0_f64, -1.5, 3.14, f64::INFINITY, f64::MIN, f64::MAX] {
        a.push(*v);
    }
    round_trip_array(MArray::from_float64(a), "f64");
}

#[test]
fn rt_arrow_bool() {
    let mut a = minarrow::BooleanArray::<()>::default();
    for v in &[true, false, true, true, false] {
        a.push(*v);
    }
    round_trip_array(MArray::BooleanArray(std::sync::Arc::new(a)), "bool");
}

#[test]
fn rt_arrow_string32() {
    let arr = std::sync::Arc::new(minarrow::StringArray::<u32>::from_slice(&[
        "alpha", "beta", "gamma", "",
    ]));
    round_trip_array(MArray::TextArray(TextArray::String32(arr)), "string32");
}

#[cfg(feature = "large_string")]
#[test]
fn rt_arrow_string64() {
    let arr = std::sync::Arc::new(minarrow::StringArray::<u64>::from_slice(&[
        "one", "two", "three",
    ]));
    round_trip_array(MArray::TextArray(TextArray::String64(arr)), "string64");
}

#[cfg(any(
    not(feature = "default_categorical_8"),
    feature = "extended_categorical"
))]
#[test]
fn rt_arrow_categorical32() {
    let arr = std::sync::Arc::new(minarrow::CategoricalArray::<u32>::from_slices(
        &[0u32, 1, 2, 0, 1],
        &["red".to_string(), "green".to_string(), "blue".to_string()],
    ));
    round_trip_array(MArray::TextArray(TextArray::Categorical32(arr)), "cat32");
}

// ----- Datetime logical types (via FieldArray) -----

#[cfg(feature = "datetime")]
#[test]
fn rt_arrow_date32() {
    let a = MArray::TemporalArray(TemporalArray::Datetime32(std::sync::Arc::new(
        minarrow::DatetimeArray::<i32> {
            data: minarrow::Buffer::from_slice(&[18000, 18500, 19000]),
            null_mask: None,
            time_unit: TimeUnit::Days,
        },
    )));
    let fa = FieldArray::new(Field::new("d32", ArrowType::Date32, false, None), a);
    round_trip_field_array(fa);
}

#[cfg(feature = "datetime")]
#[test]
fn rt_arrow_date64() {
    let a = MArray::TemporalArray(TemporalArray::Datetime64(std::sync::Arc::new(
        minarrow::DatetimeArray::<i64> {
            data: minarrow::Buffer::from_slice(&[1_600_000_000_000_i64, 1_700_000_000_000]),
            null_mask: None,
            time_unit: TimeUnit::Milliseconds,
        },
    )));
    let fa = FieldArray::new(Field::new("d64", ArrowType::Date64, false, None), a);
    round_trip_field_array(fa);
}

#[cfg(feature = "datetime")]
#[test]
fn rt_arrow_time32_sec() {
    let a = MArray::TemporalArray(TemporalArray::Datetime32(std::sync::Arc::new(
        minarrow::DatetimeArray::<i32> {
            data: minarrow::Buffer::from_slice(&[0_i32, 3600, 86399]),
            null_mask: None,
            time_unit: TimeUnit::Seconds,
        },
    )));
    let fa = FieldArray::new(
        Field::new("t32s", ArrowType::Time32(TimeUnit::Seconds), false, None),
        a,
    );
    round_trip_field_array(fa);
}

#[cfg(feature = "datetime")]
#[test]
fn rt_arrow_time32_ms() {
    let a = MArray::TemporalArray(TemporalArray::Datetime32(std::sync::Arc::new(
        minarrow::DatetimeArray::<i32> {
            data: minarrow::Buffer::from_slice(&[0_i32, 1000, 86_399_000]),
            null_mask: None,
            time_unit: TimeUnit::Milliseconds,
        },
    )));
    let fa = FieldArray::new(
        Field::new(
            "t32m",
            ArrowType::Time32(TimeUnit::Milliseconds),
            false,
            None,
        ),
        a,
    );
    round_trip_field_array(fa);
}

#[cfg(feature = "datetime")]
#[test]
fn rt_arrow_time64_us() {
    let a = MArray::TemporalArray(TemporalArray::Datetime64(std::sync::Arc::new(
        minarrow::DatetimeArray::<i64> {
            data: minarrow::Buffer::from_slice(&[0_i64, 1_000_000, 86_399_000_000]),
            null_mask: None,
            time_unit: TimeUnit::Microseconds,
        },
    )));
    let fa = FieldArray::new(
        Field::new(
            "t64u",
            ArrowType::Time64(TimeUnit::Microseconds),
            false,
            None,
        ),
        a,
    );
    round_trip_field_array(fa);
}

#[cfg(feature = "datetime")]
#[test]
fn rt_arrow_time64_ns() {
    let a = MArray::TemporalArray(TemporalArray::Datetime64(std::sync::Arc::new(
        minarrow::DatetimeArray::<i64> {
            data: minarrow::Buffer::from_slice(&[0_i64, 1_000_000_000, 86_399_000_000_000]),
            null_mask: None,
            time_unit: TimeUnit::Nanoseconds,
        },
    )));
    let fa = FieldArray::new(
        Field::new(
            "t64n",
            ArrowType::Time64(TimeUnit::Nanoseconds),
            false,
            None,
        ),
        a,
    );
    round_trip_field_array(fa);
}

#[cfg(feature = "datetime")]
#[test]
fn rt_arrow_timestamp_s() {
    let a = MArray::TemporalArray(TemporalArray::Datetime64(std::sync::Arc::new(
        minarrow::DatetimeArray::<i64> {
            data: minarrow::Buffer::from_slice(&[1_600_000_000_i64, 1_700_000_000]),
            null_mask: None,
            time_unit: TimeUnit::Seconds,
        },
    )));
    let fa = FieldArray::new(
        Field::new(
            "ts_s",
            ArrowType::Timestamp(TimeUnit::Seconds, None),
            false,
            None,
        ),
        a,
    );
    round_trip_field_array(fa);
}

#[cfg(feature = "datetime")]
#[test]
fn rt_arrow_timestamp_ms() {
    let a = MArray::TemporalArray(TemporalArray::Datetime64(std::sync::Arc::new(
        minarrow::DatetimeArray::<i64> {
            data: minarrow::Buffer::from_slice(&[1_600_000_000_000_i64]),
            null_mask: None,
            time_unit: TimeUnit::Milliseconds,
        },
    )));
    let fa = FieldArray::new(
        Field::new(
            "ts_ms",
            ArrowType::Timestamp(TimeUnit::Milliseconds, None),
            false,
            None,
        ),
        a,
    );
    round_trip_field_array(fa);
}

#[cfg(feature = "datetime")]
#[test]
fn rt_arrow_timestamp_us() {
    let a = MArray::TemporalArray(TemporalArray::Datetime64(std::sync::Arc::new(
        minarrow::DatetimeArray::<i64> {
            data: minarrow::Buffer::from_slice(&[1_600_000_000_000_000_i64]),
            null_mask: None,
            time_unit: TimeUnit::Microseconds,
        },
    )));
    let fa = FieldArray::new(
        Field::new(
            "ts_us",
            ArrowType::Timestamp(TimeUnit::Microseconds, None),
            false,
            None,
        ),
        a,
    );
    round_trip_field_array(fa);
}

#[cfg(feature = "datetime")]
#[test]
fn rt_arrow_timestamp_ns_with_tz() {
    let a = MArray::TemporalArray(TemporalArray::Datetime64(std::sync::Arc::new(
        minarrow::DatetimeArray::<i64> {
            data: minarrow::Buffer::from_slice(&[1_600_000_000_000_000_000_i64]),
            null_mask: None,
            time_unit: TimeUnit::Nanoseconds,
        },
    )));
    let fa = FieldArray::new(
        Field::new(
            "ts_ns_utc",
            ArrowType::Timestamp(TimeUnit::Nanoseconds, Some("UTC".to_string())),
            false,
            None,
        ),
        a,
    );
    round_trip_field_array(fa);
}

#[cfg(feature = "datetime")]
#[test]
fn rt_arrow_duration32_sec() {
    let a = MArray::TemporalArray(TemporalArray::Datetime32(std::sync::Arc::new(
        minarrow::DatetimeArray::<i32> {
            data: minarrow::Buffer::from_slice(&[10_i32, 20, 30]),
            null_mask: None,
            time_unit: TimeUnit::Seconds,
        },
    )));
    let fa = FieldArray::new(
        Field::new(
            "dur32s",
            ArrowType::Duration32(TimeUnit::Seconds),
            false,
            None,
        ),
        a,
    );
    round_trip_field_array(fa);
}

#[cfg(feature = "datetime")]
#[test]
fn rt_arrow_duration64_ns() {
    let a = MArray::TemporalArray(TemporalArray::Datetime64(std::sync::Arc::new(
        minarrow::DatetimeArray::<i64> {
            data: minarrow::Buffer::from_slice(&[1_000_000_i64, 2_000_000]),
            null_mask: None,
            time_unit: TimeUnit::Nanoseconds,
        },
    )));
    let fa = FieldArray::new(
        Field::new(
            "dur64n",
            ArrowType::Duration64(TimeUnit::Nanoseconds),
            false,
            None,
        ),
        a,
    );
    round_trip_field_array(fa);
}

// ----- Nullability -----

#[test]
fn rt_arrow_i32_with_nulls() {
    let mut a = minarrow::IntegerArray::<i32>::default();
    a.push(1);
    a.push_null();
    a.push(3);
    a.push_null();
    a.push(5);
    round_trip_array(MArray::from_int32(a), "i32_nulls");
}

#[test]
fn rt_arrow_f64_with_nulls() {
    let mut a = minarrow::FloatArray::<f64>::default();
    a.push(1.5);
    a.push_null();
    a.push(2.5);
    round_trip_array(MArray::from_float64(a), "f64_nulls");
}

#[test]
fn rt_arrow_bool_with_nulls() {
    let mut a = minarrow::BooleanArray::<()>::default();
    a.push(true);
    a.push_null();
    a.push(false);
    a.push_null();
    a.push(true);
    round_trip_array(MArray::BooleanArray(std::sync::Arc::new(a)), "bool_nulls");
}

#[test]
fn rt_arrow_string_with_nulls() {
    let mut a = minarrow::StringArray::<u32>::default();
    a.push_str("foo");
    a.push_null();
    a.push_str("bar");
    a.push_null();
    a.push_str("");
    round_trip_array(MArray::from_string32(a), "string_nulls");
}

#[test]
fn rt_arrow_i32_all_null() {
    let mut a = minarrow::IntegerArray::<i32>::default();
    for _ in 0..5 {
        a.push_null();
    }
    round_trip_array(MArray::from_int32(a), "i32_all_null");
}

#[test]
fn rt_arrow_string_all_null() {
    let mut a = minarrow::StringArray::<u32>::default();
    for _ in 0..3 {
        a.push_null();
    }
    round_trip_array(MArray::from_string32(a), "string_all_null");
}

// ----- Empty arrays -----

#[test]
fn rt_arrow_empty_i32() {
    round_trip_array(
        MArray::from_int32(minarrow::IntegerArray::<i32>::default()),
        "empty_i32",
    );
}

#[test]
fn rt_arrow_empty_f64() {
    round_trip_array(
        MArray::from_float64(minarrow::FloatArray::<f64>::default()),
        "empty_f64",
    );
}

#[test]
fn rt_arrow_empty_bool() {
    round_trip_array(
        MArray::BooleanArray(std::sync::Arc::new(minarrow::BooleanArray::<()>::default())),
        "empty_bool",
    );
}

#[test]
fn rt_arrow_empty_string() {
    round_trip_array(
        MArray::from_string32(minarrow::StringArray::<u32>::default()),
        "empty_string",
    );
}

// ----- Single-element -----

#[test]
fn rt_arrow_single_element_i32() {
    let mut a = minarrow::IntegerArray::<i32>::default();
    a.push(42);
    round_trip_array(MArray::from_int32(a), "single_i32");
}

#[test]
fn rt_arrow_single_element_string() {
    let arr = std::sync::Arc::new(minarrow::StringArray::<u32>::from_slice(&["solo"]));
    round_trip_array(MArray::TextArray(TextArray::String32(arr)), "single_str");
}

// ----- Empty chunked containers -----

#[cfg(feature = "chunked")]
#[test]
fn rt_arrow_empty_super_array() {
    let sa = minarrow::SuperArray::new();
    let chunks = sa.to_apache_arrow();
    assert_eq!(chunks.len(), 0);
}

#[cfg(feature = "chunked")]
#[test]
fn rt_arrow_empty_super_table() {
    let st = minarrow::SuperTable::new("".into());
    let rbs = st.to_apache_arrow();
    assert_eq!(rbs.len(), 0);
    let back = minarrow::SuperTable::from_apache_arrow(&rbs);
    assert_eq!(back.n_batches(), 0);
}

// ----- helpers -----

fn arr_i32_like(vals: &[i32]) -> MArray {
    let mut a = minarrow::IntegerArray::<i32>::default();
    for v in vals {
        a.push(*v);
    }
    MArray::from_int32(a)
}

// ----- Cross-batch shared categorical dictionary roundtrip -----

/// Build a SuperTable with two batches that share a categorical column.
/// Round-trip through arrow-rs `RecordBatch` and verify decoded strings are
/// identical pre/post, that the rebuilt SuperTable's chunks point at a
/// coherent shared dictionary (later snapshot is a superset of earlier),
/// and that codes assigned in the first batch still decode the same in
/// the rebuilt structure.
#[cfg(all(
    feature = "shared_dict",
    any(
        not(feature = "default_categorical_8"),
        feature = "extended_categorical"
    )
))]
#[test]
fn rt_arrow_super_table_shared_categorical32() {
    use minarrow::structs::dictionary::Dictionary;
    use minarrow::{CategoricalArray, SuperTable};

    // Two batches, each Categorical32. The second batch introduces a value
    // ("c") not present in the first, exercising the manager's grow path.
    let cat_a = Arc::new(CategoricalArray::<u32>::from_slices(
        &[0u32, 1, 0],
        &["a".to_string(), "b".to_string()],
    ));
    let cat_b = Arc::new(CategoricalArray::<u32>::from_slices(
        &[1u32, 2, 0],
        &["a".to_string(), "b".to_string(), "c".to_string()],
    ));

    let fa_a = FieldArray::new(
        Field::new(
            "cat",
            ArrowType::Dictionary(minarrow::ffi::arrow_dtype::CategoricalIndexType::UInt32),
            false,
            None,
        ),
        MArray::TextArray(TextArray::Categorical32(cat_a)),
    );
    let fa_b = FieldArray::new(
        Field::new(
            "cat",
            ArrowType::Dictionary(minarrow::ffi::arrow_dtype::CategoricalIndexType::UInt32),
            false,
            None,
        ),
        MArray::TextArray(TextArray::Categorical32(cat_b)),
    );

    let tbl_a = Table::new("t", Some(vec![fa_a]));
    let tbl_b = Table::new("t", Some(vec![fa_b]));

    let mut st = SuperTable::new("st".into());
    st.push(Arc::new(tbl_a));
    st.push(Arc::new(tbl_b));

    // Capture the decoded strings pre-export for later comparison.
    let strings_before: Vec<Vec<String>> = st
        .batches
        .iter()
        .map(|b| match &b.cols[0].array {
            MArray::TextArray(TextArray::Categorical32(c)) => (0..c.data.len())
                .map(|i| c.dictionary.values()[c.data[i] as usize].clone())
                .collect(),
            _ => panic!("expected Categorical32"),
        })
        .collect();
    assert_eq!(strings_before[0], vec!["a", "b", "a"]);
    assert_eq!(strings_before[1], vec!["b", "c", "a"]);

    // Round-trip.
    let rbs = st.to_apache_arrow();
    assert_eq!(rbs.len(), 2);
    let back = SuperTable::from_apache_arrow(&rbs);
    assert_eq!(back.n_batches(), 2);

    // Verify decoded strings match per row, per batch.
    let strings_after: Vec<Vec<String>> = back
        .batches
        .iter()
        .map(|b| match &b.cols[0].array {
            MArray::TextArray(TextArray::Categorical32(c)) => (0..c.data.len())
                .map(|i| c.dictionary.values()[c.data[i] as usize].clone())
                .collect(),
            _ => panic!("expected Categorical32"),
        })
        .collect();
    assert_eq!(strings_before, strings_after);

    // Both rebuilt chunks are Shared, and the first batch's dictionary is a
    // Rebuilt batches share the same dictionary handle (Arc bumped at
    // push time); both observe the union of all added strings.
    let d0 = match &back.batches[0].cols[0].array {
        MArray::TextArray(TextArray::Categorical32(c)) => c.dictionary.clone(),
        _ => unreachable!(),
    };
    let d1 = match &back.batches[1].cols[0].array {
        MArray::TextArray(TextArray::Categorical32(c)) => c.dictionary.clone(),
        _ => unreachable!(),
    };
    assert!(d0.shares_with(&d1));
    assert_eq!(d0.values(), &["a", "b", "c"]);
    assert_eq!(d1.values(), &["a", "b", "c"]);
}
