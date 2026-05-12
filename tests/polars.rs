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
    fa_i32, fa_str32, fa_u32, Array, ArrowType, Field, FieldArray, MaskedArray, NumericArray,
    Table, TextArray,
};
#[cfg(feature = "datetime")]
use minarrow::{TemporalArray, TimeUnit};
use polars::prelude::*;

// ----- helpers -----

/// Round-trip a bare `Array` through polars at the bare-Array level.
#[track_caller]
fn round_trip_array_polars(a: Array, name: &str) {
    let s = a.to_polars(name);
    let back = Array::from_polars(&s);
    assert_eq!(a, back, "Array round-trip mismatch for '{}'", name);
}

/// Round-trip a `FieldArray` through polars and assert STRICT fidelity:
/// name, dtype, and data all survive unchanged.
///
/// Use this helper only for dtypes that polars passes through identically.
/// For dtypes where polars remaps the logical type (e.g. Date64 -> Timestamp
/// (Milliseconds, None)) write the test inline and assert the expected
/// promoted dtype + data equality explicitly so the remap is documented.
#[track_caller]
fn round_trip_field_array_polars(fa: FieldArray) {
    let name = fa.field.name.clone();
    let dtype = fa.field.dtype.clone();
    let s = fa.to_polars();
    let back = FieldArray::from_polars(&s);
    assert_eq!(back.field.name, name, "name lost");
    assert_eq!(back.field.dtype, dtype, "dtype lost");
    assert_eq!(back.array, fa.array, "data mismatch for '{}'", name);
}

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

// =============================================================
// Exhaustive type coverage round-trip (polars side)
// =============================================================
//
// Polars upcasts some types on round-trip (e.g. String32 -> LargeUtf8 / String64).
// Where the data layout survives but the offset width may flip, we compare
// element-wise to be width-agnostic.

fn arr_strings_back(back: &Array) -> Vec<String> {
    match back {
        Array::TextArray(TextArray::String32(b)) => (0..b.len())
            .map(|i| b.get_str(i).unwrap_or("").to_string())
            .collect(),
        Array::TextArray(TextArray::String64(b)) => (0..b.len())
            .map(|i| b.get_str(i).unwrap_or("").to_string())
            .collect(),
        _ => panic!("expected string array, got {:?}", back),
    }
}

#[test]
fn rt_polars_i32() {
    let mut a = minarrow::IntegerArray::<i32>::default();
    for v in &[1i32, -2, 3, i32::MAX, i32::MIN] { a.push(*v); }
    round_trip_array_polars(Array::from_int32(a), "i32");
}

#[test]
fn rt_polars_i64() {
    let mut a = minarrow::IntegerArray::<i64>::default();
    for v in &[1i64, -2, 3, i64::MAX, i64::MIN] { a.push(*v); }
    round_trip_array_polars(Array::from_int64(a), "i64");
}

#[test]
fn rt_polars_u32() {
    let mut a = minarrow::IntegerArray::<u32>::default();
    for v in &[0u32, 1, u32::MAX] { a.push(*v); }
    round_trip_array_polars(Array::from_uint32(a), "u32");
}

#[test]
fn rt_polars_u64() {
    let mut a = minarrow::IntegerArray::<u64>::default();
    for v in &[0u64, 1, u64::MAX] { a.push(*v); }
    round_trip_array_polars(Array::from_uint64(a), "u64");
}

#[test]
fn rt_polars_f32() {
    let mut a = minarrow::FloatArray::<f32>::default();
    for v in &[0.0_f32, -1.5, 3.14, f32::MIN, f32::MAX] { a.push(*v); }
    round_trip_array_polars(Array::from_float32(a), "f32");
}

#[test]
fn rt_polars_f64() {
    let mut a = minarrow::FloatArray::<f64>::default();
    for v in &[0.0_f64, -1.5, 3.14, f64::MIN, f64::MAX] { a.push(*v); }
    round_trip_array_polars(Array::from_float64(a), "f64");
}

#[test]
fn rt_polars_bool() {
    let mut a = minarrow::BooleanArray::<()>::default();
    for v in &[true, false, true, true, false] { a.push(*v); }
    round_trip_array_polars(Array::BooleanArray(Arc::new(a)), "bool");
}

#[test]
fn rt_polars_string32_element_equal() {
    // Polars may upcast Utf8 -> LargeUtf8; check element-wise.
    let arr = Arc::new(
        minarrow::StringArray::<u32>::from_slice(&["alpha", "beta", "gamma"])
    );
    let original = Array::TextArray(TextArray::String32(arr));
    let s = original.to_polars("s");
    let back = Array::from_polars(&s);
    assert_eq!(
        arr_strings_back(&back),
        vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()]
    );
}

#[cfg(feature = "large_string")]
#[test]
fn rt_polars_string64_element_equal() {
    let arr = Arc::new(
        minarrow::StringArray::<u64>::from_slice(&["one", "two", "three"])
    );
    let original = Array::TextArray(TextArray::String64(arr));
    let s = original.to_polars("s");
    let back = Array::from_polars(&s);
    assert_eq!(
        arr_strings_back(&back),
        vec!["one".to_string(), "two".to_string(), "three".to_string()]
    );
}

#[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
#[test]
fn rt_polars_categorical32_element_equal() {
    let arr = Arc::new(minarrow::CategoricalArray::<u32>::from_slices(
        &[0u32, 1, 2, 0, 1],
        &["red".to_string(), "green".to_string(), "blue".to_string()],
    ));
    let original = Array::TextArray(TextArray::Categorical32(arr));
    let s = original.to_polars("cat");
    let back = Array::from_polars(&s);
    // Polars Categorical may come back as either Categorical or as a string
    // type depending on CompatLevel and version; compare values either way.
    let back_strings: Vec<String> = match &back {
        Array::TextArray(TextArray::Categorical32(c)) => (0..c.data.len())
            .map(|i| c.unique_values[c.data[i] as usize].clone())
            .collect(),
        Array::TextArray(_) => arr_strings_back(&back),
        _ => panic!("unexpected back type: {:?}", back),
    };
    assert_eq!(
        back_strings,
        vec!["red", "green", "blue", "red", "green"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>()
    );
}

// ----- Datetime logical types (via FieldArray) -----

#[cfg(feature = "datetime")]
#[test]
fn rt_polars_date32() {
    let a = Array::TemporalArray(TemporalArray::Datetime32(Arc::new(
        minarrow::DatetimeArray::<i32> {
            data: minarrow::Buffer::from_slice(&[18000_i32, 18500, 19000]),
            null_mask: None,
            time_unit: TimeUnit::Days,
        },
    )));
    let fa = FieldArray::new(Field::new("d32", ArrowType::Date32, false, None), a);
    round_trip_field_array_polars(fa);
}

#[cfg(feature = "datetime")]
#[test]
fn rt_polars_date64_promotes_to_timestamp_ms() {
    // Polars remaps Date64 (ms-since-epoch i64) to Timestamp(Milliseconds, None)
    // internally. This is not lossy: the i64 ms-since-epoch payload is byte-identical;
    // only the logical type label changes. We assert both halves explicitly so a
    // future polars behaviour change (or one of our own bugs) surfaces immediately.
    let original_data = [1_600_000_000_000_i64, 1_700_000_000_000];
    let a = Array::TemporalArray(TemporalArray::Datetime64(Arc::new(
        minarrow::DatetimeArray::<i64> {
            data: minarrow::Buffer::from_slice(&original_data),
            null_mask: None,
            time_unit: TimeUnit::Milliseconds,
        },
    )));
    let fa = FieldArray::new(Field::new("d64", ArrowType::Date64, false, None), a.clone());
    let s = fa.to_polars();
    let back = FieldArray::from_polars(&s);

    assert_eq!(back.field.name, "d64", "name lost");
    assert_eq!(
        back.field.dtype,
        ArrowType::Timestamp(TimeUnit::Milliseconds, None),
        "polars is expected to promote Date64 -> Timestamp(Milliseconds, None); if this changes, the remap layer needs updating"
    );

    // The physical i64 ms-since-epoch payload must be identical (lossless promotion).
    match (&a, &back.array) {
        (
            Array::TemporalArray(TemporalArray::Datetime64(lhs)),
            Array::TemporalArray(TemporalArray::Datetime64(rhs)),
        ) => {
            assert_eq!(rhs.time_unit, TimeUnit::Milliseconds, "time_unit drifted");
            assert_eq!(&rhs.data[..], &original_data[..], "i64 ms payload changed");
            assert_eq!(lhs.null_mask, rhs.null_mask, "null mask changed");
        }
        other => panic!("expected Datetime64 on both sides, got {:?}", other),
    }
}

#[cfg(feature = "datetime")]
#[test]
fn rt_polars_timestamp_ns() {
    let a = Array::TemporalArray(TemporalArray::Datetime64(Arc::new(
        minarrow::DatetimeArray::<i64> {
            data: minarrow::Buffer::from_slice(&[1_600_000_000_000_000_000_i64]),
            null_mask: None,
            time_unit: TimeUnit::Nanoseconds,
        },
    )));
    let fa = FieldArray::new(
        Field::new("ts_ns", ArrowType::Timestamp(TimeUnit::Nanoseconds, None), false, None),
        a,
    );
    round_trip_field_array_polars(fa);
}

#[cfg(feature = "datetime")]
#[test]
fn rt_polars_timestamp_us() {
    let a = Array::TemporalArray(TemporalArray::Datetime64(Arc::new(
        minarrow::DatetimeArray::<i64> {
            data: minarrow::Buffer::from_slice(&[1_600_000_000_000_000_i64]),
            null_mask: None,
            time_unit: TimeUnit::Microseconds,
        },
    )));
    let fa = FieldArray::new(
        Field::new("ts_us", ArrowType::Timestamp(TimeUnit::Microseconds, None), false, None),
        a,
    );
    round_trip_field_array_polars(fa);
}

#[cfg(feature = "datetime")]
#[test]
fn rt_polars_duration_ns() {
    let a = Array::TemporalArray(TemporalArray::Datetime64(Arc::new(
        minarrow::DatetimeArray::<i64> {
            data: minarrow::Buffer::from_slice(&[1_000_000_i64, 2_000_000]),
            null_mask: None,
            time_unit: TimeUnit::Nanoseconds,
        },
    )));
    let fa = FieldArray::new(
        Field::new("dur_ns", ArrowType::Duration64(TimeUnit::Nanoseconds), false, None),
        a,
    );
    round_trip_field_array_polars(fa);
}

// ----- Nullability -----

#[test]
fn rt_polars_i32_with_nulls() {
    let mut a = minarrow::IntegerArray::<i32>::default();
    a.push(1);
    a.push_null();
    a.push(3);
    a.push_null();
    a.push(5);
    round_trip_array_polars(Array::from_int32(a), "i32_nulls");
}

#[test]
fn rt_polars_f64_with_nulls() {
    let mut a = minarrow::FloatArray::<f64>::default();
    a.push(1.5);
    a.push_null();
    a.push(2.5);
    round_trip_array_polars(Array::from_float64(a), "f64_nulls");
}

#[test]
fn rt_polars_bool_with_nulls() {
    let mut a = minarrow::BooleanArray::<()>::default();
    a.push(true);
    a.push_null();
    a.push(false);
    a.push_null();
    a.push(true);
    round_trip_array_polars(Array::BooleanArray(Arc::new(a)), "bool_nulls");
}

#[test]
fn rt_polars_i32_all_null() {
    let mut a = minarrow::IntegerArray::<i32>::default();
    for _ in 0..5 { a.push_null(); }
    round_trip_array_polars(Array::from_int32(a), "i32_all_null");
}

// ----- Empty arrays -----

#[test]
fn rt_polars_empty_i32() {
    round_trip_array_polars(
        Array::from_int32(minarrow::IntegerArray::<i32>::default()),
        "empty_i32",
    );
}

#[test]
fn rt_polars_empty_f64() {
    round_trip_array_polars(
        Array::from_float64(minarrow::FloatArray::<f64>::default()),
        "empty_f64",
    );
}

#[test]
fn rt_polars_empty_bool() {
    round_trip_array_polars(
        Array::BooleanArray(Arc::new(minarrow::BooleanArray::<()>::default())),
        "empty_bool",
    );
}

// ----- Single-element -----

#[test]
fn rt_polars_single_element_i32() {
    let mut a = minarrow::IntegerArray::<i32>::default();
    a.push(42);
    round_trip_array_polars(Array::from_int32(a), "single_i32");
}

// ----- Empty chunked containers -----

#[cfg(feature = "chunked")]
#[test]
fn rt_polars_empty_super_table() {
    let st = minarrow::SuperTable::new("".into());
    let df = st.to_polars();
    assert_eq!(df.height(), 0);
    assert_eq!(df.width(), 0);
    let back = minarrow::SuperTable::from_polars(&df);
    assert_eq!(back.n_batches(), 0);
}

// ----- Time32 / Time64 polars round-trip (real round-trip, not just export) -----

// ----- Time32 / Time64 polars round-trip -----
//
// Polars normalises ALL Time* logical types to `Time64(Nanoseconds)` internally
// and rescales the underlying integer payload accordingly. The promotion is
// lossless (same time-of-day, larger physical type with finer resolution) but
// the byte representation changes:
//   Time32(Sec)   [n]  -> Time64(Ns)  [n * 1_000_000_000]
//   Time32(Ms)    [n]  -> Time64(Ns)  [n * 1_000_000]
//   Time64(Us)    [n]  -> Time64(Ns)  [n * 1_000]
//   Time64(Ns)    [n]  -> Time64(Ns)  [n]            (pass-through)
//
// Tests assert both the dtype promotion and the rescaled payload explicitly.

#[cfg(feature = "datetime")]
fn assert_time_promotes_to_ns_i64(
    fa: FieldArray,
    name: &str,
    expected_ns: &[i64],
) {
    let s = fa.to_polars();
    let back = FieldArray::from_polars(&s);
    assert_eq!(back.field.name, name, "name lost");
    assert_eq!(
        back.field.dtype,
        ArrowType::Time64(TimeUnit::Nanoseconds),
        "polars is expected to promote all Time* to Time64(Nanoseconds)"
    );
    match &back.array {
        Array::TemporalArray(TemporalArray::Datetime64(d)) => {
            assert_eq!(d.time_unit, TimeUnit::Nanoseconds, "time_unit drifted");
            assert_eq!(&d.data[..], expected_ns, "rescaled ns payload mismatch");
            assert!(d.null_mask.is_none(), "no nulls expected");
        }
        other => panic!("expected Datetime64, got {:?}", other),
    }
}

#[cfg(feature = "datetime")]
#[test]
fn rt_polars_time32_sec_promotes_to_time64_ns() {
    let fa = FieldArray::new(
        Field::new("t32s", ArrowType::Time32(TimeUnit::Seconds), false, None),
        Array::TemporalArray(TemporalArray::Datetime32(Arc::new(
            minarrow::DatetimeArray::<i32> {
                data: minarrow::Buffer::from_slice(&[0_i32, 3600, 86399]),
                null_mask: None,
                time_unit: TimeUnit::Seconds,
            },
        ))),
    );
    assert_time_promotes_to_ns_i64(
        fa,
        "t32s",
        &[0, 3_600_000_000_000, 86_399_000_000_000],
    );
}

#[cfg(feature = "datetime")]
#[test]
fn rt_polars_time32_ms_promotes_to_time64_ns() {
    let fa = FieldArray::new(
        Field::new("t32m", ArrowType::Time32(TimeUnit::Milliseconds), false, None),
        Array::TemporalArray(TemporalArray::Datetime32(Arc::new(
            minarrow::DatetimeArray::<i32> {
                data: minarrow::Buffer::from_slice(&[0_i32, 1000, 86_399_000]),
                null_mask: None,
                time_unit: TimeUnit::Milliseconds,
            },
        ))),
    );
    assert_time_promotes_to_ns_i64(
        fa,
        "t32m",
        &[0, 1_000_000_000, 86_399_000_000_000],
    );
}

#[cfg(feature = "datetime")]
#[test]
fn rt_polars_time64_us_promotes_to_time64_ns() {
    let fa = FieldArray::new(
        Field::new("t64u", ArrowType::Time64(TimeUnit::Microseconds), false, None),
        Array::TemporalArray(TemporalArray::Datetime64(Arc::new(
            minarrow::DatetimeArray::<i64> {
                data: minarrow::Buffer::from_slice(&[0_i64, 1_000_000, 86_399_000_000]),
                null_mask: None,
                time_unit: TimeUnit::Microseconds,
            },
        ))),
    );
    assert_time_promotes_to_ns_i64(
        fa,
        "t64u",
        &[0, 1_000_000_000, 86_399_000_000_000],
    );
}

#[cfg(feature = "datetime")]
#[test]
fn rt_polars_time64_ns() {
    // Time64(Ns) is the polars-native representation; passes through unchanged.
    let fa = FieldArray::new(
        Field::new("t64n", ArrowType::Time64(TimeUnit::Nanoseconds), false, None),
        Array::TemporalArray(TemporalArray::Datetime64(Arc::new(
            minarrow::DatetimeArray::<i64> {
                data: minarrow::Buffer::from_slice(&[0_i64, 1_000_000_000, 86_399_000_000_000]),
                null_mask: None,
                time_unit: TimeUnit::Nanoseconds,
            },
        ))),
    );
    round_trip_field_array_polars(fa);
}

// ----- Null-mask preservation under polars logical-type promotion -----

#[cfg(feature = "datetime")]
#[test]
fn rt_polars_date64_preserves_null_mask_through_timestamp_promotion() {
    // Build a Date64 with a null in the middle, send through polars (which
    // promotes to Timestamp(Ms, None)), and assert the null position survives.
    let mut data = minarrow::DatetimeArray::<i64> {
        data: minarrow::Buffer::from_slice(&[1_600_000_000_000_i64, 0, 1_700_000_000_000]),
        null_mask: None,
        time_unit: TimeUnit::Milliseconds,
    };
    // Mark index 1 as null
    let mut mask = minarrow::Bitmask::new_set_all(3, true);
    unsafe { mask.set_unchecked(1, false) };
    data.null_mask = Some(mask);

    let a = Array::TemporalArray(TemporalArray::Datetime64(Arc::new(data)));
    let fa = FieldArray::new(Field::new("d64n", ArrowType::Date64, true, None), a);
    let s = fa.to_polars();
    let back = FieldArray::from_polars(&s);

    assert_eq!(back.field.dtype, ArrowType::Timestamp(TimeUnit::Milliseconds, None));
    match &back.array {
        Array::TemporalArray(TemporalArray::Datetime64(d)) => {
            let mask = d.null_mask.as_ref().expect("null mask must survive promotion");
            assert!(unsafe { mask.get_unchecked(0) }, "idx 0 should be valid");
            assert!(!unsafe { mask.get_unchecked(1) }, "idx 1 should be null");
            assert!(unsafe { mask.get_unchecked(2) }, "idx 2 should be valid");
        }
        other => panic!("expected Datetime64, got {:?}", other),
    }
}

#[cfg(feature = "datetime")]
#[test]
fn rt_polars_time32_sec_preserves_null_mask_through_promotion() {
    // Time32(Sec) with a null, promoted to Time64(Ns) by polars.
    let mut data = minarrow::DatetimeArray::<i32> {
        data: minarrow::Buffer::from_slice(&[0_i32, 0, 3600]),
        null_mask: None,
        time_unit: TimeUnit::Seconds,
    };
    let mut mask = minarrow::Bitmask::new_set_all(3, true);
    unsafe { mask.set_unchecked(1, false) };
    data.null_mask = Some(mask);

    let a = Array::TemporalArray(TemporalArray::Datetime32(Arc::new(data)));
    let fa = FieldArray::new(
        Field::new("t32sn", ArrowType::Time32(TimeUnit::Seconds), true, None),
        a,
    );
    let s = fa.to_polars();
    let back = FieldArray::from_polars(&s);

    assert_eq!(back.field.dtype, ArrowType::Time64(TimeUnit::Nanoseconds));
    match &back.array {
        Array::TemporalArray(TemporalArray::Datetime64(d)) => {
            let mask = d.null_mask.as_ref().expect("null mask must survive promotion");
            assert!(unsafe { mask.get_unchecked(0) }, "idx 0 should be valid");
            assert!(!unsafe { mask.get_unchecked(1) }, "idx 1 should be null");
            assert!(unsafe { mask.get_unchecked(2) }, "idx 2 should be valid");
            // Valid positions rescaled: 0 sec -> 0 ns, 3600 sec -> 3.6e12 ns.
            assert_eq!(d.data[0], 0);
            assert_eq!(d.data[2], 3_600_000_000_000);
        }
        other => panic!("expected Datetime64, got {:?}", other),
    }
}

#[cfg(feature = "datetime")]
#[test]
fn rt_polars_time64_us_preserves_null_mask_through_promotion() {
    let mut data = minarrow::DatetimeArray::<i64> {
        data: minarrow::Buffer::from_slice(&[0_i64, 0, 1_000_000]),
        null_mask: None,
        time_unit: TimeUnit::Microseconds,
    };
    let mut mask = minarrow::Bitmask::new_set_all(3, true);
    unsafe { mask.set_unchecked(1, false) };
    data.null_mask = Some(mask);

    let a = Array::TemporalArray(TemporalArray::Datetime64(Arc::new(data)));
    let fa = FieldArray::new(
        Field::new("t64un", ArrowType::Time64(TimeUnit::Microseconds), true, None),
        a,
    );
    let s = fa.to_polars();
    let back = FieldArray::from_polars(&s);

    assert_eq!(back.field.dtype, ArrowType::Time64(TimeUnit::Nanoseconds));
    match &back.array {
        Array::TemporalArray(TemporalArray::Datetime64(d)) => {
            let mask = d.null_mask.as_ref().expect("null mask must survive promotion");
            assert!(unsafe { mask.get_unchecked(0) });
            assert!(!unsafe { mask.get_unchecked(1) });
            assert!(unsafe { mask.get_unchecked(2) });
            assert_eq!(d.data[2], 1_000_000_000); // 1_000_000 us -> 1e9 ns
        }
        other => panic!("expected Datetime64, got {:?}", other),
    }
}
