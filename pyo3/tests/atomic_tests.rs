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

//! # Atomic Integration Test Runner
//!
//! Runs all roundtrip tests for MinArrow <-> PyArrow conversions.
//! Tests arrays, tables, nullability, and edge cases such as empty arrays.
//!
//! ## Prerequisites
//!
//! Python with PyArrow installed:
//! ```bash
//! pip install pyarrow
//! # or use the project's venv:
//! cd pyo3 && source .venv/bin/activate
//! ```
//!
//! ## Running
//!
//! From the `pyo3/` directory:
//!
//! ```bash
//! PYO3_PYTHON=$(pwd)/.venv/bin/python \
//!   PYTHONHOME=/usr \
//!   PYTHONPATH=$(pwd)/.venv/lib/python3.12/site-packages \
//!   LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
//!   cargo test --test atomic_tests \
//!     --no-default-features \
//!     --features "datetime,extended_numeric_types,extended_categorical"
//! ```
//!
//! ### Why --no-default-features?
//!
//! The default features include `extension-module` which tells PyO3 not to link
//! against libpython, since Python loads the extension at runtime. For standalone
//! binaries like this test runner, we need to link against libpython, so we disable
//! that feature.
//!
//! ### Why PYTHONHOME?
//!
//! When embedding Python in a Rust binary, Python needs to know where its standard
//! library is located. Set PYTHONHOME to the Python prefix, e.g. `/usr/local` or
//! `/usr`. You can find this via: `python3 -c "import sys; print(sys.prefix)"`

use minarrow::ffi::arrow_dtype::ArrowType;
use minarrow::{
    Array, BooleanArray, Field, FieldArray, FloatArray, IntegerArray, MaskedArray, NumericArray,
    StringArray, SuperArray, SuperTable, Table, TextArray,
};
use minarrow_pyo3::ffi::{to_py, to_rust};
use minarrow_pyo3::{PyArray, PyRecordBatch, PyTable};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use std::sync::Arc;

fn main() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| {
        println!("=== MinArrow <-> PyArrow Comprehensive Tests ===\n");

        let mut passed = 0;
        let mut failed = 0;

        // Integer tests
        run_test("i32 roundtrip", || test_int32_roundtrip(py), &mut passed, &mut failed);
        run_test("i64 roundtrip", || test_int64_roundtrip(py), &mut passed, &mut failed);
        run_test("u32 roundtrip", || test_uint32_roundtrip(py), &mut passed, &mut failed);
        run_test("u64 roundtrip", || test_uint64_roundtrip(py), &mut passed, &mut failed);

        // Float tests
        run_test("f32 roundtrip", || test_float32_roundtrip(py), &mut passed, &mut failed);
        run_test("f64 roundtrip", || test_float64_roundtrip(py), &mut passed, &mut failed);

        // Boolean test
        run_test("boolean roundtrip", || test_boolean_roundtrip(py), &mut passed, &mut failed);

        // String tests
        run_test("string32 roundtrip", || test_string32_roundtrip(py), &mut passed, &mut failed);

        // Nullable tests
        run_test("nullable i32 roundtrip", || test_nullable_int32_roundtrip(py), &mut passed, &mut failed);
        run_test("nullable f64 roundtrip", || test_nullable_float64_roundtrip(py), &mut passed, &mut failed);
        run_test("nullable string roundtrip", || test_nullable_string_roundtrip(py), &mut passed, &mut failed);
        run_test("nullable boolean roundtrip", || test_nullable_boolean_roundtrip(py), &mut passed, &mut failed);

        // Table tests
        run_test("table roundtrip", || test_table_roundtrip(py), &mut passed, &mut failed);
        run_test("table with nulls roundtrip", || test_table_with_nulls_roundtrip(py), &mut passed, &mut failed);

        // Edge case tests
        run_test("empty array roundtrip", || test_empty_array_roundtrip(py), &mut passed, &mut failed);
        run_test("single element roundtrip", || test_single_element_array_roundtrip(py), &mut passed, &mut failed);
        run_test("all nulls roundtrip", || test_all_nulls_array_roundtrip(py), &mut passed, &mut failed);
        run_test("empty strings roundtrip", || test_empty_string_array_roundtrip(py), &mut passed, &mut failed);
        run_test("large array roundtrip", || test_large_array_roundtrip(py), &mut passed, &mut failed);

        // PyCapsule tests
        println!("\n--- PyCapsule Tests ---");
        run_test("capsule array export", || test_capsule_array_export(py), &mut passed, &mut failed);
        run_test("capsule stream table export", || test_capsule_stream_table_export(py), &mut passed, &mut failed);
        run_test("capsule import from pyarrow array", || test_capsule_import_pyarrow_array(py), &mut passed, &mut failed);
        run_test("capsule import from pyarrow table", || test_capsule_import_pyarrow_table(py), &mut passed, &mut failed);
        run_test("capsule nullable array export", || test_capsule_nullable_export(py), &mut passed, &mut failed);
        run_test("capsule super array stream", || test_capsule_super_array_stream(py), &mut passed, &mut failed);
        run_test("capsule super table stream", || test_capsule_super_table_stream(py), &mut passed, &mut failed);

        // Extended numeric types: Rust → Python → Rust
        println!("\n--- Extended Numeric Types ---");
        run_test("i8 roundtrip", || test_int8_roundtrip(py), &mut passed, &mut failed);
        run_test("i16 roundtrip", || test_int16_roundtrip(py), &mut passed, &mut failed);
        run_test("u8 roundtrip", || test_uint8_roundtrip(py), &mut passed, &mut failed);
        run_test("u16 roundtrip", || test_uint16_roundtrip(py), &mut passed, &mut failed);

        // Temporal types: Rust → Python → Rust
        println!("\n--- Temporal Types ---");
        run_test("date32 roundtrip", || test_date32_roundtrip(py), &mut passed, &mut failed);
        run_test("date64 roundtrip", || test_date64_roundtrip(py), &mut passed, &mut failed);
        run_test("timestamp roundtrip", || test_timestamp_roundtrip(py), &mut passed, &mut failed);
        run_test("timestamp with tz roundtrip", || test_timestamp_tz_roundtrip(py), &mut passed, &mut failed);
        run_test("duration roundtrip", || test_duration_roundtrip(py), &mut passed, &mut failed);

        // Categorical: Rust → Python → Rust
        println!("\n--- Categorical ---");
        run_test("categorical32 roundtrip", || test_categorical32_roundtrip(py), &mut passed, &mut failed);

        // Python → Rust import tests
        println!("\n--- Python → Rust Import Tests ---");
        run_test("import i8 from pyarrow", || test_import_int8(py), &mut passed, &mut failed);
        run_test("import u16 from pyarrow", || test_import_uint16(py), &mut passed, &mut failed);
        run_test("import date32 from pyarrow", || test_import_date32(py), &mut passed, &mut failed);
        run_test("import timestamp from pyarrow", || test_import_timestamp(py), &mut passed, &mut failed);
        run_test("import timestamp+tz from pyarrow", || test_import_timestamp_tz(py), &mut passed, &mut failed);
        run_test("import duration from pyarrow", || test_import_duration(py), &mut passed, &mut failed);
        run_test("import dictionary from pyarrow", || test_import_dictionary(py), &mut passed, &mut failed);
        run_test("import float32 from pyarrow", || test_import_float32(py), &mut passed, &mut failed);
        run_test("import boolean from pyarrow", || test_import_boolean(py), &mut passed, &mut failed);
        run_test("import string from pyarrow", || test_import_string(py), &mut passed, &mut failed);
        run_test("import nullable from pyarrow", || test_import_nullable(py), &mut passed, &mut failed);
        run_test("import mixed table from pyarrow", || test_import_mixed_table(py), &mut passed, &mut failed);

        // Table name preservation tests
        println!("\n--- Table Name Preservation ---");
        run_test("record batch name roundtrip", || test_record_batch_name_roundtrip(py), &mut passed, &mut failed);
        run_test("super table name roundtrip", || test_super_table_name_roundtrip(py), &mut passed, &mut failed);

        println!("\n=== Test Results ===");
        println!("Passed: {}", passed);
        println!("Failed: {}", failed);

        if failed > 0 {
            std::process::exit(1);
        }

        Ok(())
    })
}

fn run_test<F>(name: &str, test_fn: F, passed: &mut usize, failed: &mut usize)
where
    F: FnOnce() -> Result<(), String>,
{
    match test_fn() {
        Ok(()) => {
            println!("✓ {}", name);
            *passed += 1;
        }
        Err(e) => {
            println!("✗ {} - {}", name, e);
            *failed += 1;
        }
    }
}

/// Helper to convert MinArrow Array to PyArrow and back
fn roundtrip_array(py: Python<'_>, array: Array) -> Result<Array, String> {
    let py_array = PyArray::from(array);
    let py_obj = py_array
        .into_pyobject(py)
        .map_err(|e| format!("Failed to convert to PyArrow: {}", e))?;
    let back: PyArray = py_obj
        .extract()
        .map_err(|e| format!("Failed to convert from PyArrow: {}", e))?;
    Ok(back.into_inner().array)
}

// 
// Integer Array Tests
// 

fn test_int32_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = IntegerArray::<i32>::default();
    arr.push(1);
    arr.push(2);
    arr.push(3);
    arr.push(i32::MAX);
    arr.push(i32::MIN);

    let original = Array::from_int32(arr);
    let result = roundtrip_array(py, original.clone())?;

    if original.len() != result.len() {
        return Err(format!("Length mismatch: {} vs {}", original.len(), result.len()));
    }

    match (&original, &result) {
        (
            Array::NumericArray(NumericArray::Int32(a)),
            Array::NumericArray(NumericArray::Int32(b)),
        ) => {
            for i in 0..a.len() {
                if a.get(i) != b.get(i) {
                    return Err(format!("Mismatch at index {}", i));
                }
            }
        }
        _ => return Err("Type mismatch after roundtrip".to_string()),
    }
    Ok(())
}

fn test_int64_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = IntegerArray::<i64>::default();
    arr.push(100);
    arr.push(200);
    arr.push(300);
    arr.push(i64::MAX);
    arr.push(i64::MIN);

    let original = Array::from_int64(arr);
    let result = roundtrip_array(py, original.clone())?;

    if original.len() != result.len() {
        return Err(format!("Length mismatch: {} vs {}", original.len(), result.len()));
    }

    match (&original, &result) {
        (
            Array::NumericArray(NumericArray::Int64(a)),
            Array::NumericArray(NumericArray::Int64(b)),
        ) => {
            for i in 0..a.len() {
                if a.get(i) != b.get(i) {
                    return Err(format!("Mismatch at index {}", i));
                }
            }
        }
        _ => return Err("Type mismatch after roundtrip".to_string()),
    }
    Ok(())
}

fn test_uint32_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = IntegerArray::<u32>::default();
    arr.push(0);
    arr.push(1);
    arr.push(u32::MAX);

    let original = Array::from_uint32(arr);
    let result = roundtrip_array(py, original.clone())?;

    if original.len() != result.len() {
        return Err(format!("Length mismatch: {} vs {}", original.len(), result.len()));
    }

    match (&original, &result) {
        (
            Array::NumericArray(NumericArray::UInt32(a)),
            Array::NumericArray(NumericArray::UInt32(b)),
        ) => {
            for i in 0..a.len() {
                if a.get(i) != b.get(i) {
                    return Err(format!("Mismatch at index {}", i));
                }
            }
        }
        _ => return Err("Type mismatch after roundtrip".to_string()),
    }
    Ok(())
}

fn test_uint64_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = IntegerArray::<u64>::default();
    arr.push(0);
    arr.push(1);
    arr.push(u64::MAX);

    let original = Array::from_uint64(arr);
    let result = roundtrip_array(py, original.clone())?;

    if original.len() != result.len() {
        return Err(format!("Length mismatch: {} vs {}", original.len(), result.len()));
    }

    match (&original, &result) {
        (
            Array::NumericArray(NumericArray::UInt64(a)),
            Array::NumericArray(NumericArray::UInt64(b)),
        ) => {
            for i in 0..a.len() {
                if a.get(i) != b.get(i) {
                    return Err(format!("Mismatch at index {}", i));
                }
            }
        }
        _ => return Err("Type mismatch after roundtrip".to_string()),
    }
    Ok(())
}

// 
// Float Array Tests
// 

fn test_float32_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = FloatArray::<f32>::default();
    arr.push(1.5);
    arr.push(2.25);
    arr.push(f32::MAX);
    arr.push(f32::MIN);
    arr.push(0.0);
    arr.push(-0.0);

    let original = Array::from_float32(arr);
    let result = roundtrip_array(py, original.clone())?;

    if original.len() != result.len() {
        return Err(format!("Length mismatch: {} vs {}", original.len(), result.len()));
    }

    match (&original, &result) {
        (
            Array::NumericArray(NumericArray::Float32(a)),
            Array::NumericArray(NumericArray::Float32(b)),
        ) => {
            for i in 0..a.len() {
                let av = a.get(i).unwrap();
                let bv = b.get(i).unwrap();
                if !((av - bv).abs() < f32::EPSILON || (av.is_nan() && bv.is_nan())) {
                    return Err(format!("Mismatch at index {}: {} vs {}", i, av, bv));
                }
            }
        }
        _ => return Err("Type mismatch after roundtrip".to_string()),
    }
    Ok(())
}

fn test_float64_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = FloatArray::<f64>::default();
    arr.push(1.5);
    arr.push(2.25);
    arr.push(std::f64::consts::PI);
    arr.push(std::f64::consts::E);
    arr.push(f64::MAX);
    arr.push(f64::MIN);

    let original = Array::from_float64(arr);
    let result = roundtrip_array(py, original.clone())?;

    if original.len() != result.len() {
        return Err(format!("Length mismatch: {} vs {}", original.len(), result.len()));
    }

    match (&original, &result) {
        (
            Array::NumericArray(NumericArray::Float64(a)),
            Array::NumericArray(NumericArray::Float64(b)),
        ) => {
            for i in 0..a.len() {
                let av = a.get(i).unwrap();
                let bv = b.get(i).unwrap();
                if !((av - bv).abs() < f64::EPSILON || (av.is_nan() && bv.is_nan())) {
                    return Err(format!("Mismatch at index {}: {} vs {}", i, av, bv));
                }
            }
        }
        _ => return Err("Type mismatch after roundtrip".to_string()),
    }
    Ok(())
}

// 
// Boolean Array Tests
// 

fn test_boolean_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = BooleanArray::default();
    arr.push(true);
    arr.push(false);
    arr.push(true);
    arr.push(false);
    arr.push(true);

    let original = Array::from_bool(arr);
    let result = roundtrip_array(py, original.clone())?;

    if original.len() != result.len() {
        return Err(format!("Length mismatch: {} vs {}", original.len(), result.len()));
    }

    match (&original, &result) {
        (Array::BooleanArray(a), Array::BooleanArray(b)) => {
            for i in 0..a.len() {
                if a.get(i) != b.get(i) {
                    return Err(format!("Mismatch at index {}", i));
                }
            }
        }
        _ => return Err("Type mismatch after roundtrip".to_string()),
    }
    Ok(())
}

// 
// String Array Tests
// 

fn test_string32_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = StringArray::<u32>::default();
    arr.push_str("hello");
    arr.push_str("world");
    arr.push_str("");
    arr.push_str("with spaces");
    arr.push_str("special: αβγ 日本語 🎉");

    let original = Array::from_string32(arr);
    let result = roundtrip_array(py, original.clone())?;

    if original.len() != result.len() {
        return Err(format!("Length mismatch: {} vs {}", original.len(), result.len()));
    }

    match (&original, &result) {
        (
            Array::TextArray(TextArray::String32(a)),
            Array::TextArray(TextArray::String32(b)),
        ) => {
            for i in 0..a.len() {
                if a.get(i) != b.get(i) {
                    return Err(format!("Mismatch at index {}", i));
                }
            }
        }
        _ => return Err("Type mismatch after roundtrip".to_string()),
    }
    Ok(())
}

// 
// Nullable Array Tests
// 

fn test_nullable_int32_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = IntegerArray::<i32>::default();
    arr.push(1);
    arr.push_null();
    arr.push(3);
    arr.push_null();
    arr.push(5);

    let original = Array::from_int32(arr);
    if original.null_count() != 2 {
        return Err(format!("Expected 2 nulls, got {}", original.null_count()));
    }

    let result = roundtrip_array(py, original.clone())?;

    if original.len() != result.len() {
        return Err(format!("Length mismatch: {} vs {}", original.len(), result.len()));
    }
    if original.null_count() != result.null_count() {
        return Err(format!(
            "Null count mismatch: {} vs {}",
            original.null_count(),
            result.null_count()
        ));
    }

    match (&original, &result) {
        (
            Array::NumericArray(NumericArray::Int32(a)),
            Array::NumericArray(NumericArray::Int32(b)),
        ) => {
            for i in 0..a.len() {
                if a.is_null(i) != b.is_null(i) {
                    return Err(format!("Null mismatch at index {}", i));
                }
                if !a.is_null(i) && a.get(i) != b.get(i) {
                    return Err(format!("Value mismatch at index {}", i));
                }
            }
        }
        _ => return Err("Type mismatch after roundtrip".to_string()),
    }
    Ok(())
}

fn test_nullable_float64_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = FloatArray::<f64>::default();
    arr.push(1.1);
    arr.push_null();
    arr.push(3.3);
    arr.push_null();

    let original = Array::from_float64(arr);
    if original.null_count() != 2 {
        return Err(format!("Expected 2 nulls, got {}", original.null_count()));
    }

    let result = roundtrip_array(py, original.clone())?;

    if original.len() != result.len() {
        return Err(format!("Length mismatch: {} vs {}", original.len(), result.len()));
    }
    if original.null_count() != result.null_count() {
        return Err(format!(
            "Null count mismatch: {} vs {}",
            original.null_count(),
            result.null_count()
        ));
    }
    Ok(())
}

fn test_nullable_string_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = StringArray::<u32>::default();
    arr.push_str("hello");
    arr.push_null();
    arr.push_str("world");
    arr.push_null();

    let original = Array::from_string32(arr);
    if original.null_count() != 2 {
        return Err(format!("Expected 2 nulls, got {}", original.null_count()));
    }

    let result = roundtrip_array(py, original.clone())?;

    if original.len() != result.len() {
        return Err(format!("Length mismatch: {} vs {}", original.len(), result.len()));
    }
    if original.null_count() != result.null_count() {
        return Err(format!(
            "Null count mismatch: {} vs {}",
            original.null_count(),
            result.null_count()
        ));
    }

    match (&original, &result) {
        (
            Array::TextArray(TextArray::String32(a)),
            Array::TextArray(TextArray::String32(b)),
        ) => {
            for i in 0..a.len() {
                if a.is_null(i) != b.is_null(i) {
                    return Err(format!("Null mismatch at index {}", i));
                }
                if !a.is_null(i) && a.get(i) != b.get(i) {
                    return Err(format!("Value mismatch at index {}", i));
                }
            }
        }
        _ => return Err("Type mismatch after roundtrip".to_string()),
    }
    Ok(())
}

fn test_nullable_boolean_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = BooleanArray::default();
    arr.push(true);
    arr.push_null();
    arr.push(false);
    arr.push_null();

    let original = Array::from_bool(arr);
    if original.null_count() != 2 {
        return Err(format!("Expected 2 nulls, got {}", original.null_count()));
    }

    let result = roundtrip_array(py, original.clone())?;

    if original.len() != result.len() {
        return Err(format!("Length mismatch: {} vs {}", original.len(), result.len()));
    }
    if original.null_count() != result.null_count() {
        return Err(format!(
            "Null count mismatch: {} vs {}",
            original.null_count(),
            result.null_count()
        ));
    }
    Ok(())
}

// 
// Table / RecordBatch Tests
// 

fn test_table_roundtrip(py: Python<'_>) -> Result<(), String> {
    // Create arrays
    let mut int_arr = IntegerArray::<i64>::default();
    int_arr.push(100);
    int_arr.push(200);
    int_arr.push(300);

    let mut float_arr = FloatArray::<f64>::default();
    float_arr.push(1.1);
    float_arr.push(2.2);
    float_arr.push(3.3);

    let mut str_arr = StringArray::<u32>::default();
    str_arr.push_str("alpha");
    str_arr.push_str("beta");
    str_arr.push_str("gamma");

    let mut bool_arr = BooleanArray::default();
    bool_arr.push(true);
    bool_arr.push(false);
    bool_arr.push(true);

    // Create FieldArrays
    let fa1 = FieldArray::new(
        Field::new("id", ArrowType::Int64, false, None),
        Array::from_int64(int_arr),
    );
    let fa2 = FieldArray::new(
        Field::new("value", ArrowType::Float64, false, None),
        Array::from_float64(float_arr),
    );
    let fa3 = FieldArray::new(
        Field::new("name", ArrowType::String, false, None),
        Array::from_string32(str_arr),
    );
    let fa4 = FieldArray::new(
        Field::new("active", ArrowType::Boolean, false, None),
        Array::from_bool(bool_arr),
    );

    // Create Table
    let original = Table::new("test".to_string(), Some(vec![fa1, fa2, fa3, fa4]));

    // Roundtrip
    let py_batch = PyRecordBatch::from(original.clone());
    let py_obj = py_batch
        .into_pyobject(py)
        .map_err(|e| format!("Failed to convert to PyArrow: {}", e))?;
    let back: PyRecordBatch = py_obj
        .extract()
        .map_err(|e| format!("Failed to convert from PyArrow: {}", e))?;
    let result = back.into_inner();

    // Verify
    if original.n_rows() != result.n_rows() {
        return Err(format!(
            "Row count mismatch: {} vs {}",
            original.n_rows(),
            result.n_rows()
        ));
    }
    if original.n_cols() != result.n_cols() {
        return Err(format!(
            "Column count mismatch: {} vs {}",
            original.n_cols(),
            result.n_cols()
        ));
    }

    // Verify each column
    for (i, (orig_col, result_col)) in original.cols.iter().zip(result.cols.iter()).enumerate() {
        if orig_col.array.len() != result_col.array.len() {
            return Err(format!("Column {} length mismatch", i));
        }
    }

    Ok(())
}

fn test_table_with_nulls_roundtrip(py: Python<'_>) -> Result<(), String> {
    // Create arrays with nulls
    let mut int_arr = IntegerArray::<i32>::default();
    int_arr.push(1);
    int_arr.push_null();
    int_arr.push(3);

    let mut str_arr = StringArray::<u32>::default();
    str_arr.push_str("a");
    str_arr.push_null();
    str_arr.push_str("c");

    // Create FieldArrays
    let fa1 = FieldArray::new(
        Field::new("id", ArrowType::Int32, true, None),
        Array::from_int32(int_arr),
    );
    let fa2 = FieldArray::new(
        Field::new("name", ArrowType::String, true, None),
        Array::from_string32(str_arr),
    );

    // Create Table
    let original = Table::new("nullable_test".to_string(), Some(vec![fa1, fa2]));

    // Roundtrip
    let py_batch = PyRecordBatch::from(original.clone());
    let py_obj = py_batch
        .into_pyobject(py)
        .map_err(|e| format!("Failed to convert to PyArrow: {}", e))?;
    let back: PyRecordBatch = py_obj
        .extract()
        .map_err(|e| format!("Failed to convert from PyArrow: {}", e))?;
    let result = back.into_inner();

    // Verify
    if original.n_rows() != result.n_rows() {
        return Err(format!(
            "Row count mismatch: {} vs {}",
            original.n_rows(),
            result.n_rows()
        ));
    }
    if original.n_cols() != result.n_cols() {
        return Err(format!(
            "Column count mismatch: {} vs {}",
            original.n_cols(),
            result.n_cols()
        ));
    }

    // Verify null counts preserved
    for (i, (orig_col, result_col)) in original.cols.iter().zip(result.cols.iter()).enumerate() {
        if orig_col.null_count != result_col.null_count {
            return Err(format!("Column {} null count mismatch", i));
        }
    }

    Ok(())
}

// 
// Edge Case Tests
// 

fn test_empty_array_roundtrip(py: Python<'_>) -> Result<(), String> {
    let arr = IntegerArray::<i32>::default();
    let original = Array::from_int32(arr);
    if original.len() != 0 {
        return Err(format!("Expected empty array, got length {}", original.len()));
    }

    let result = roundtrip_array(py, original)?;
    if result.len() != 0 {
        return Err(format!("Expected empty result, got length {}", result.len()));
    }
    Ok(())
}

fn test_single_element_array_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = IntegerArray::<i32>::default();
    arr.push(42);
    let original = Array::from_int32(arr);

    let result = roundtrip_array(py, original.clone())?;
    if result.len() != 1 {
        return Err(format!("Expected 1 element, got {}", result.len()));
    }

    match (&original, &result) {
        (
            Array::NumericArray(NumericArray::Int32(a)),
            Array::NumericArray(NumericArray::Int32(b)),
        ) => {
            if a.get(0) != b.get(0) {
                return Err("Value mismatch".to_string());
            }
        }
        _ => return Err("Type mismatch".to_string()),
    }
    Ok(())
}

fn test_all_nulls_array_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = IntegerArray::<i32>::default();
    arr.push_null();
    arr.push_null();
    arr.push_null();

    let original = Array::from_int32(arr);
    if original.null_count() != 3 {
        return Err(format!("Expected 3 nulls, got {}", original.null_count()));
    }

    let result = roundtrip_array(py, original)?;
    if result.len() != 3 {
        return Err(format!("Expected 3 elements, got {}", result.len()));
    }
    if result.null_count() != 3 {
        return Err(format!("Expected 3 nulls in result, got {}", result.null_count()));
    }
    Ok(())
}

fn test_empty_string_array_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = StringArray::<u32>::default();
    arr.push_str("");
    arr.push_str("");
    arr.push_str("");

    let original = Array::from_string32(arr);
    let result = roundtrip_array(py, original.clone())?;

    if original.len() != result.len() {
        return Err(format!("Length mismatch: {} vs {}", original.len(), result.len()));
    }

    match (&original, &result) {
        (
            Array::TextArray(TextArray::String32(a)),
            Array::TextArray(TextArray::String32(b)),
        ) => {
            for i in 0..a.len() {
                if a.get(i) != b.get(i) {
                    return Err(format!("Mismatch at index {}", i));
                }
                if a.get(i) != Some("") {
                    return Err(format!("Expected empty string at index {}", i));
                }
            }
        }
        _ => return Err("Type mismatch".to_string()),
    }
    Ok(())
}

fn test_large_array_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = IntegerArray::<i64>::default();
    for i in 0..10_000 {
        arr.push(i);
    }

    let original = Array::from_int64(arr);
    let result = roundtrip_array(py, original.clone())?;

    if original.len() != result.len() {
        return Err(format!("Length mismatch: {} vs {}", original.len(), result.len()));
    }
    if result.len() != 10_000 {
        return Err(format!("Expected 10000 elements, got {}", result.len()));
    }

    match (&original, &result) {
        (
            Array::NumericArray(NumericArray::Int64(a)),
            Array::NumericArray(NumericArray::Int64(b)),
        ) => {
            // Spot check
            if a.get(0) != b.get(0) {
                return Err("Mismatch at index 0".to_string());
            }
            if a.get(5000) != b.get(5000) {
                return Err("Mismatch at index 5000".to_string());
            }
            if a.get(9999) != b.get(9999) {
                return Err("Mismatch at index 9999".to_string());
            }
        }
        _ => return Err("Type mismatch".to_string()),
    }
    Ok(())
}

// PyCapsule Tests

fn test_capsule_array_export(py: Python<'_>) -> Result<(), String> {
    let mut arr = IntegerArray::<i64>::default();
    arr.push(10);
    arr.push(20);
    arr.push(30);

    let original = Array::from_int64(arr);
    let field = Field::new("test", ArrowType::Int64, false, None);

    let (schema_cap, array_cap) =
        to_py::array_to_capsules(Arc::new(original), &field, py)
            .map_err(|e| format!("Failed to create capsules: {}", e))?;

    if schema_cap.is_none(py) || array_cap.is_none(py) {
        return Err("Capsule objects are None".to_string());
    }
    Ok(())
}

fn test_capsule_stream_table_export(py: Python<'_>) -> Result<(), String> {
    let mut int_arr = IntegerArray::<i32>::default();
    int_arr.push(1);
    int_arr.push(2);
    int_arr.push(3);

    let mut float_arr = FloatArray::<f64>::default();
    float_arr.push(1.1);
    float_arr.push(2.2);
    float_arr.push(3.3);

    let fa1 = FieldArray::new(
        Field::new("id", ArrowType::Int32, false, None),
        Array::from_int32(int_arr),
    );
    let fa2 = FieldArray::new(
        Field::new("value", ArrowType::Float64, false, None),
        Array::from_float64(float_arr),
    );

    let table = Table::new("test".to_string(), Some(vec![fa1, fa2]));

    let capsule = to_py::table_to_stream_capsule(&table, py)
        .map_err(|e| format!("Failed to create stream capsule: {}", e))?;

    if capsule.is_none(py) {
        return Err("Stream capsule is None".to_string());
    }
    Ok(())
}

fn test_capsule_import_pyarrow_array(py: Python<'_>) -> Result<(), String> {
    let pyarrow = py.import("pyarrow")
        .map_err(|e| format!("Failed to import pyarrow: {}", e))?;

    let py_array = pyarrow
        .call_method1("array", (vec![1i64, 2, 3, 4, 5],))
        .map_err(|e| format!("Failed to create PyArrow array: {}", e))?;

    let has_capsule = py_array.hasattr("__arrow_c_array__")
        .map_err(|e| format!("hasattr check failed: {}", e))?;
    if !has_capsule {
        return Err("PyArrow array does not support __arrow_c_array__".to_string());
    }

    match to_rust::try_capsule_array(&py_array) {
        Some(Ok(fa)) => {
            if fa.array.len() != 5 {
                return Err(format!("Expected 5 elements, got {}", fa.array.len()));
            }
            Ok(())
        }
        Some(Err(e)) => Err(format!("PyCapsule import failed: {}", e)),
        None => Err("PyCapsule protocol not detected".to_string()),
    }
}

fn test_capsule_import_pyarrow_table(py: Python<'_>) -> Result<(), String> {
    let pyarrow = py.import("pyarrow")
        .map_err(|e| format!("Failed to import pyarrow: {}", e))?;

    let py_table = pyarrow
        .call_method1(
            "table",
            (vec![
                ("x", pyarrow.call_method1("array", (vec![1i64, 2, 3],))
                    .map_err(|e| format!("Failed to create array: {}", e))?),
                ("y", pyarrow.call_method1("array", (vec![4.0f64, 5.0, 6.0],))
                    .map_err(|e| format!("Failed to create array: {}", e))?),
            ]
            .into_py_dict(py)
            .map_err(|e| format!("Failed to create dict: {}", e))?,),
        )
        .map_err(|e| format!("Failed to create PyArrow table: {}", e))?;

    let has_stream = py_table.hasattr("__arrow_c_stream__")
        .map_err(|e| format!("hasattr check failed: {}", e))?;
    if !has_stream {
        return Err("PyArrow table does not support __arrow_c_stream__".to_string());
    }

    match to_rust::try_capsule_record_batch_stream(&py_table) {
        Some(Ok((batches, _metadata))) => {
            if batches.is_empty() {
                return Err("No batches returned".to_string());
            }
            let first_batch = &batches[0];
            if first_batch.len() != 2 {
                return Err(format!("Expected 2 columns, got {}", first_batch.len()));
            }
            if first_batch[0].0.len() != 3 {
                return Err(format!("Expected 3 rows, got {}", first_batch[0].0.len()));
            }
            Ok(())
        }
        Some(Err(e)) => Err(format!("PyCapsule stream import failed: {}", e)),
        None => Err("PyCapsule stream protocol not detected".to_string()),
    }
}

fn test_capsule_nullable_export(py: Python<'_>) -> Result<(), String> {
    let mut arr = IntegerArray::<i32>::default();
    arr.push(1);
    arr.push_null();
    arr.push(3);
    arr.push_null();
    arr.push(5);

    let original = Array::from_int32(arr);
    if original.null_count() != 2 {
        return Err(format!("Expected 2 nulls, got {}", original.null_count()));
    }

    let field = Field::new("nullable", ArrowType::Int32, true, None);

    let (schema_cap, array_cap) =
        to_py::array_to_capsules(Arc::new(original), &field, py)
            .map_err(|e| format!("Failed to create capsules: {}", e))?;

    if schema_cap.is_none(py) || array_cap.is_none(py) {
        return Err("Capsule objects are None".to_string());
    }
    Ok(())
}

fn test_capsule_super_array_stream(py: Python<'_>) -> Result<(), String> {
    let mut arr1 = IntegerArray::<i64>::default();
    arr1.push(1);
    arr1.push(2);
    let chunk1 = Array::from_int64(arr1);

    let mut arr2 = IntegerArray::<i64>::default();
    arr2.push(3);
    arr2.push(4);
    let chunk2 = Array::from_int64(arr2);

    let fa1 = FieldArray::new(Field::new("vals", ArrowType::Int64, false, None), chunk1);
    let fa2 = FieldArray::new(Field::new("vals", ArrowType::Int64, false, None), chunk2);

    let super_array = SuperArray::from_chunks(vec![fa1, fa2]);

    let capsule = to_py::super_array_to_stream_capsule(&super_array, py)
        .map_err(|e| format!("Failed to create stream capsule: {}", e))?;

    if capsule.is_none(py) {
        return Err("Stream capsule is None".to_string());
    }
    Ok(())
}

fn test_capsule_super_table_stream(py: Python<'_>) -> Result<(), String> {
    let mut int_arr1 = IntegerArray::<i32>::default();
    int_arr1.push(1);
    int_arr1.push(2);

    let mut int_arr2 = IntegerArray::<i32>::default();
    int_arr2.push(3);
    int_arr2.push(4);

    let batch1 = Table::new("t".to_string(), Some(vec![
        FieldArray::new(
            Field::new("id", ArrowType::Int32, false, None),
            Array::from_int32(int_arr1),
        ),
    ]));
    let batch2 = Table::new("t".to_string(), Some(vec![
        FieldArray::new(
            Field::new("id", ArrowType::Int32, false, None),
            Array::from_int32(int_arr2),
        ),
    ]));

    let super_table = SuperTable {
        name: "test".to_string(),
        batches: vec![Arc::new(batch1), Arc::new(batch2)],
        schema: vec![Arc::new(Field::new("id", ArrowType::Int32, false, None))],
        n_rows: 4,
    };

    let capsule = to_py::super_table_to_stream_capsule(&super_table, py)
        .map_err(|e| format!("Failed to create stream capsule: {}", e))?;

    if capsule.is_none(py) {
        return Err("Stream capsule is None".to_string());
    }
    Ok(())
}

// ── Extended numeric types: Rust → Python → Rust ──────────────────

fn test_int8_roundtrip(py: Python<'_>) -> Result<(), String> {
    let arr = IntegerArray::<i8>::from_slice(&[1, -128, 127, 0]);
    let array = Array::from_int8(arr);
    let result = roundtrip_array(py, array)?;
    match &result {
        Array::NumericArray(NumericArray::Int8(a)) => {
            assert_eq!(a.len(), 4);
            assert_eq!(a.get(0), Some(1i8));
            assert_eq!(a.get(1), Some(-128i8));
            assert_eq!(a.get(2), Some(127i8));
        }
        _ => return Err(format!("Expected Int8, got {:?}", "unexpected")),
    }
    Ok(())
}

fn test_int16_roundtrip(py: Python<'_>) -> Result<(), String> {
    let arr = IntegerArray::<i16>::from_slice(&[1, -32768, 32767]);
    let array = Array::from_int16(arr);
    let result = roundtrip_array(py, array)?;
    match &result {
        Array::NumericArray(NumericArray::Int16(a)) => {
            assert_eq!(a.len(), 3);
            assert_eq!(a.get(1), Some(-32768i16));
            assert_eq!(a.get(2), Some(32767i16));
        }
        _ => return Err(format!("Expected Int16, got {:?}", "unexpected")),
    }
    Ok(())
}

fn test_uint8_roundtrip(py: Python<'_>) -> Result<(), String> {
    let arr = IntegerArray::<u8>::from_slice(&[0, 128, 255]);
    let array = Array::from_uint8(arr);
    let result = roundtrip_array(py, array)?;
    match &result {
        Array::NumericArray(NumericArray::UInt8(a)) => {
            assert_eq!(a.len(), 3);
            assert_eq!(a.get(2), Some(255u8));
        }
        _ => return Err(format!("Expected UInt8, got {:?}", "unexpected")),
    }
    Ok(())
}

fn test_uint16_roundtrip(py: Python<'_>) -> Result<(), String> {
    let arr = IntegerArray::<u16>::from_slice(&[0, 32768, 65535]);
    let array = Array::from_uint16(arr);
    let result = roundtrip_array(py, array)?;
    match &result {
        Array::NumericArray(NumericArray::UInt16(a)) => {
            assert_eq!(a.len(), 3);
            assert_eq!(a.get(2), Some(65535u16));
        }
        _ => return Err(format!("Expected UInt16, got {:?}", "unexpected")),
    }
    Ok(())
}

// ── Temporal types: Rust → Python → Rust ──────────────────────────

fn roundtrip_field_array(py: Python<'_>, fa: FieldArray) -> Result<FieldArray, String> {
    let py_arr = PyArray::from(fa);
    let py_obj = py_arr.into_pyobject(py).map_err(|e| format!("{}", e))?;
    let back: PyArray = py_obj.extract().map_err(|e| format!("{}", e))?;
    Ok(back.into_inner())
}

fn test_date32_roundtrip(py: Python<'_>) -> Result<(), String> {
    use minarrow::{DatetimeArray, TimeUnit};
    let arr = DatetimeArray::<i32>::from_slice(&[0, 1, 100, 19000], Some(TimeUnit::Days));
    let array = Array::from_datetime_i32(arr);
    let field = Field::new("d", ArrowType::Date32, false, None);
    let result = roundtrip_field_array(py, FieldArray::new(field, array))?;
    match &result.array {
        Array::TemporalArray(minarrow::TemporalArray::Datetime32(a)) => {
            assert_eq!(a.len(), 4);
            assert_eq!(a.get(3), Some(19000));
        }
        _ => return Err(format!("Expected Datetime32, got {:?}", "unexpected")),
    }
    assert_eq!(result.field.dtype, ArrowType::Date32);
    Ok(())
}

fn test_date64_roundtrip(py: Python<'_>) -> Result<(), String> {
    use minarrow::{DatetimeArray, TimeUnit};
    let arr = DatetimeArray::<i64>::from_slice(&[0, 86400000, 172800000], Some(TimeUnit::Milliseconds));
    let array = Array::from_datetime_i64(arr);
    let field = Field::new("d", ArrowType::Date64, false, None);
    let result = roundtrip_field_array(py, FieldArray::new(field, array))?;
    match &result.array {
        Array::TemporalArray(minarrow::TemporalArray::Datetime64(a)) => {
            assert_eq!(a.len(), 3);
            assert_eq!(a.get(1), Some(86400000));
        }
        _ => return Err(format!("Expected Datetime64, got {:?}", "unexpected")),
    }
    assert_eq!(result.field.dtype, ArrowType::Date64);
    Ok(())
}

fn test_timestamp_roundtrip(py: Python<'_>) -> Result<(), String> {
    use minarrow::{DatetimeArray, TimeUnit};
    let arr = DatetimeArray::<i64>::from_slice(&[1_000_000, 2_000_000, 3_000_000], Some(TimeUnit::Microseconds));
    let array = Array::from_datetime_i64(arr);
    let field = Field::new("ts", ArrowType::Timestamp(TimeUnit::Microseconds, None), false, None);
    let result = roundtrip_field_array(py, FieldArray::new(field, array))?;
    match &result.field.dtype {
        ArrowType::Timestamp(TimeUnit::Microseconds, None) => {}
        _ => return Err(format!("Expected Timestamp(us), got unexpected variant")),
    }
    Ok(())
}

fn test_timestamp_tz_roundtrip(py: Python<'_>) -> Result<(), String> {
    use minarrow::{DatetimeArray, TimeUnit};
    let arr = DatetimeArray::<i64>::from_slice(&[1_000_000, 2_000_000], Some(TimeUnit::Microseconds));
    let array = Array::from_datetime_i64(arr);
    let field = Field::new("ts", ArrowType::Timestamp(TimeUnit::Microseconds, Some("UTC".to_string())), false, None);
    let result = roundtrip_field_array(py, FieldArray::new(field, array))?;
    match &result.field.dtype {
        ArrowType::Timestamp(TimeUnit::Microseconds, Some(tz)) if tz == "UTC" => {}
        _ => return Err(format!("Expected Timestamp(us, UTC), got unexpected variant")),
    }
    Ok(())
}

fn test_duration_roundtrip(py: Python<'_>) -> Result<(), String> {
    use minarrow::{DatetimeArray, TimeUnit};
    let arr = DatetimeArray::<i64>::from_slice(&[1_000_000, 2_000_000], Some(TimeUnit::Microseconds));
    let array = Array::from_datetime_i64(arr);
    let field = Field::new("dur", ArrowType::Duration64(TimeUnit::Microseconds), false, None);
    let result = roundtrip_field_array(py, FieldArray::new(field, array))?;
    match &result.field.dtype {
        ArrowType::Duration64(TimeUnit::Microseconds) => {}
        _ => return Err(format!("Expected Duration64(us), got unexpected variant")),
    }
    Ok(())
}

// ── Categorical: Rust → Python → Rust ─────────────────────────────

fn test_categorical32_roundtrip(py: Python<'_>) -> Result<(), String> {
    use minarrow::CategoricalArray;
    let cat = CategoricalArray::<u32>::from_slices(
        &[0, 1, 0, 2, 1],
        &["cat".to_string(), "dog".to_string(), "bird".to_string()],
    );
    let array = Array::from_categorical32(cat);
    let field = Field::new(
        "label",
        ArrowType::Dictionary(minarrow::ffi::arrow_dtype::CategoricalIndexType::UInt32),
        false,
        None,
    );
    let result = roundtrip_field_array(py, FieldArray::new(field, array))?;
    match &result.array {
        Array::TextArray(TextArray::Categorical32(a)) => {
            assert_eq!(a.len(), 5);
        }
        _ => return Err(format!("Expected Categorical32, got {:?}", "unexpected")),
    }
    Ok(())
}

// ── Python → Rust import tests ────────────────────────────────────

fn test_import_int8(py: Python<'_>) -> Result<(), String> {
    let pa = py.import("pyarrow").map_err(|e| format!("{}", e))?;
    let pa_int8 = pa.getattr("int8").map_err(|e| format!("{}", e))?.call0().map_err(|e| format!("{}", e))?;
    let py_arr = pa.call_method1("array", (vec![1i8, -128, 127, 0], pa_int8)).map_err(|e| format!("{}", e))?;
    let result = to_rust::try_capsule_array(&py_arr)
        .ok_or("__arrow_c_array__ not available")?
        .map_err(|e| format!("{}", e))?;
    match &result.array {
        Array::NumericArray(NumericArray::Int8(a)) => {
            assert_eq!(a.len(), 4);
            assert_eq!(a.get(1), Some(-128i8));
        }
        _ => return Err(format!("Expected Int8, got {:?}", "unexpected")),
    }
    Ok(())
}

fn test_import_uint16(py: Python<'_>) -> Result<(), String> {
    let pa = py.import("pyarrow").map_err(|e| format!("{}", e))?;
    let pa_u16 = pa.getattr("uint16").map_err(|e| format!("{}", e))?.call0().map_err(|e| format!("{}", e))?;
    let py_arr = pa.call_method1("array", (vec![0u16, 32768, 65535], pa_u16)).map_err(|e| format!("{}", e))?;
    let result = to_rust::try_capsule_array(&py_arr)
        .ok_or("__arrow_c_array__ not available")?
        .map_err(|e| format!("{}", e))?;
    match &result.array {
        Array::NumericArray(NumericArray::UInt16(a)) => {
            assert_eq!(a.len(), 3);
            assert_eq!(a.get(2), Some(65535u16));
        }
        _ => return Err(format!("Expected UInt16, got {:?}", "unexpected")),
    }
    Ok(())
}

fn test_import_date32(py: Python<'_>) -> Result<(), String> {
    let pa = py.import("pyarrow").map_err(|e| format!("{}", e))?;
    let pa_date32 = pa.getattr("date32").map_err(|e| format!("{}", e))?.call0().map_err(|e| format!("{}", e))?;
    let py_arr = pa.call_method1("array", (vec![0i32, 1, 100, 19000], pa_date32)).map_err(|e| format!("{}", e))?;
    let result = to_rust::try_capsule_array(&py_arr)
        .ok_or("__arrow_c_array__ not available")?
        .map_err(|e| format!("{}", e))?;
    assert_eq!(result.field.dtype, ArrowType::Date32);
    match &result.array {
        Array::TemporalArray(minarrow::TemporalArray::Datetime32(a)) => {
            assert_eq!(a.len(), 4);
            assert_eq!(a.get(3), Some(19000));
        }
        _ => return Err(format!("Expected Datetime32, got {:?}", "unexpected")),
    }
    Ok(())
}

fn test_import_timestamp(py: Python<'_>) -> Result<(), String> {
    let pa = py.import("pyarrow").map_err(|e| format!("{}", e))?;
    let pa_ts = pa.call_method1("timestamp", ("us",)).map_err(|e| format!("{}", e))?;
    let py_arr = pa.call_method1("array", (vec![1_000_000i64, 2_000_000, 3_000_000], pa_ts)).map_err(|e| format!("{}", e))?;
    let result = to_rust::try_capsule_array(&py_arr)
        .ok_or("__arrow_c_array__ not available")?
        .map_err(|e| format!("{}", e))?;
    match &result.field.dtype {
        ArrowType::Timestamp(minarrow::TimeUnit::Microseconds, None) => {}
        _ => return Err(format!("Expected Timestamp(us), got unexpected variant")),
    }
    Ok(())
}

fn test_import_timestamp_tz(py: Python<'_>) -> Result<(), String> {
    let pa = py.import("pyarrow").map_err(|e| format!("{}", e))?;
    let pa_ts = pa.call_method("timestamp", ("us",), Some(&[("tz", "UTC")].into_py_dict(py).map_err(|e| format!("{}", e))?)).map_err(|e| format!("{}", e))?;
    let py_arr = pa.call_method1("array", (vec![1_000_000i64, 2_000_000], pa_ts)).map_err(|e| format!("{}", e))?;
    let result = to_rust::try_capsule_array(&py_arr)
        .ok_or("__arrow_c_array__ not available")?
        .map_err(|e| format!("{}", e))?;
    match &result.field.dtype {
        ArrowType::Timestamp(minarrow::TimeUnit::Microseconds, Some(tz)) if tz == "UTC" => {}
        _ => return Err(format!("Expected Timestamp(us, UTC), got unexpected variant")),
    }
    Ok(())
}

fn test_import_duration(py: Python<'_>) -> Result<(), String> {
    let pa = py.import("pyarrow").map_err(|e| format!("{}", e))?;
    let pa_dur = pa.call_method1("duration", ("us",)).map_err(|e| format!("{}", e))?;
    let py_arr = pa.call_method1("array", (vec![1_000_000i64, 2_000_000], pa_dur)).map_err(|e| format!("{}", e))?;
    let result = to_rust::try_capsule_array(&py_arr)
        .ok_or("__arrow_c_array__ not available")?
        .map_err(|e| format!("{}", e))?;
    match &result.field.dtype {
        ArrowType::Duration64(minarrow::TimeUnit::Microseconds) => {}
        _ => return Err(format!("Expected Duration64(us), got unexpected variant")),
    }
    Ok(())
}

fn test_import_dictionary(py: Python<'_>) -> Result<(), String> {
    let pa = py.import("pyarrow").map_err(|e| format!("{}", e))?;
    let py_arr = pa.call_method1("array", (vec!["cat", "dog", "cat", "bird"],)).map_err(|e| format!("{}", e))?;
    let dict_arr = py_arr.call_method0("dictionary_encode").map_err(|e| format!("{}", e))?;
    let result = to_rust::try_capsule_array(&dict_arr)
        .ok_or("__arrow_c_array__ not available")?
        .map_err(|e| format!("{}", e))?;
    match &result.array {
        Array::TextArray(TextArray::Categorical32(a)) => {
            assert_eq!(a.len(), 4);
        }
        _ => return Err(format!("Expected Categorical32, got {:?}", "unexpected")),
    }
    Ok(())
}

fn test_import_float32(py: Python<'_>) -> Result<(), String> {
    let pa = py.import("pyarrow").map_err(|e| format!("{}", e))?;
    let pa_f32 = pa.getattr("float32").map_err(|e| format!("{}", e))?.call0().map_err(|e| format!("{}", e))?;
    let py_arr = pa.call_method1("array", (vec![1.5f32, 2.5, 3.5], pa_f32)).map_err(|e| format!("{}", e))?;
    let result = to_rust::try_capsule_array(&py_arr)
        .ok_or("__arrow_c_array__ not available")?
        .map_err(|e| format!("{}", e))?;
    match &result.array {
        Array::NumericArray(NumericArray::Float32(a)) => {
            assert_eq!(a.len(), 3);
            assert!((a.get(0).unwrap() - 1.5f32).abs() < f32::EPSILON);
        }
        _ => return Err(format!("Expected Float32, got {:?}", "unexpected")),
    }
    Ok(())
}

fn test_import_boolean(py: Python<'_>) -> Result<(), String> {
    let pa = py.import("pyarrow").map_err(|e| format!("{}", e))?;
    let py_arr = pa.call_method1("array", (vec![true, false, true, false],)).map_err(|e| format!("{}", e))?;
    let result = to_rust::try_capsule_array(&py_arr)
        .ok_or("__arrow_c_array__ not available")?
        .map_err(|e| format!("{}", e))?;
    match &result.array {
        Array::BooleanArray(a) => assert_eq!(a.len(), 4),
        _ => return Err(format!("Expected Boolean, got {:?}", "unexpected")),
    }
    Ok(())
}

fn test_import_string(py: Python<'_>) -> Result<(), String> {
    let pa = py.import("pyarrow").map_err(|e| format!("{}", e))?;
    let py_arr = pa.call_method1("array", (vec!["hello", "world", "minarrow"],)).map_err(|e| format!("{}", e))?;
    let result = to_rust::try_capsule_array(&py_arr)
        .ok_or("__arrow_c_array__ not available")?
        .map_err(|e| format!("{}", e))?;
    match &result.array {
        Array::TextArray(TextArray::String32(a)) => assert_eq!(a.len(), 3),
        _ => return Err(format!("Expected String32, got {:?}", "unexpected")),
    }
    Ok(())
}

fn test_import_nullable(py: Python<'_>) -> Result<(), String> {
    let pa = py.import("pyarrow").map_err(|e| format!("{}", e))?;
    let py_arr = pa.call_method1("array", (vec![Some(1i64), None, Some(3i64), None, Some(5i64)],)).map_err(|e| format!("{}", e))?;
    let result = to_rust::try_capsule_array(&py_arr)
        .ok_or("__arrow_c_array__ not available")?
        .map_err(|e| format!("{}", e))?;
    match &result.array {
        Array::NumericArray(NumericArray::Int64(a)) => {
            assert_eq!(a.len(), 5);
            assert_eq!(a.null_count(), 2);
            assert_eq!(a.get(0), Some(1));
            assert_eq!(a.get(1), None);
            assert_eq!(a.get(4), Some(5));
        }
        _ => return Err(format!("Expected Int64, got {:?}", "unexpected")),
    }
    Ok(())
}

fn test_import_mixed_table(py: Python<'_>) -> Result<(), String> {
    let pa = py.import("pyarrow").map_err(|e| format!("{}", e))?;
    let dict = vec![
        ("id", pa.call_method1("array", (vec![10i64, 20, 30],)).map_err(|e| format!("{}", e))?),
        ("score", pa.call_method1("array", (vec![1.1f64, 2.2, 3.3],)).map_err(|e| format!("{}", e))?),
        ("name", pa.call_method1("array", (vec!["a", "b", "c"],)).map_err(|e| format!("{}", e))?),
    ].into_py_dict(py).map_err(|e| format!("{}", e))?;
    let py_table = pa.call_method1("table", (dict,)).map_err(|e| format!("{}", e))?;
    let (batches, _metadata) = to_rust::try_capsule_record_batch_stream(&py_table)
        .ok_or("__arrow_c_stream__ not available")?
        .map_err(|e| format!("{}", e))?;
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    assert_eq!(batch.len(), 3);
    match &batch[0].0.as_ref() {
        Array::NumericArray(NumericArray::Int64(_)) => {}
        _ => return Err(format!("Col 0: expected Int64, got unexpected variant")),
    }
    match &batch[1].0.as_ref() {
        Array::NumericArray(NumericArray::Float64(_)) => {}
        _ => return Err(format!("Col 1: expected Float64, got unexpected variant")),
    }
    match &batch[2].0.as_ref() {
        Array::TextArray(TextArray::String32(_)) => {}
        _ => return Err(format!("Col 2: expected String32, got unexpected variant")),
    }
    assert_eq!(batch[0].1.name, "id");
    assert_eq!(batch[1].1.name, "score");
    assert_eq!(batch[2].1.name, "name");
    Ok(())
}

fn test_record_batch_name_roundtrip(py: Python) -> Result<(), String> {
    let mut arr = IntegerArray::<i64>::default();
    arr.push(10);
    arr.push(20);
    let array = Array::from_int64(arr);
    let field = Field::new("col", ArrowType::Int64, false, None);
    let fa = FieldArray::new(field, array);
    let table = Table::new("my_batch_name".to_string(), Some(vec![fa]));

    let py_batch = PyRecordBatch::from(table);
    let py_obj = py_batch
        .into_pyobject(py)
        .map_err(|e| format!("export failed: {}", e))?;

    let back: PyRecordBatch = py_obj
        .extract()
        .map_err(|e| format!("import failed: {}", e))?;
    let recovered: Table = back.into();
    if recovered.name != "my_batch_name" {
        return Err(format!(
            "table name not preserved: expected 'my_batch_name', got '{}'",
            recovered.name
        ));
    }
    Ok(())
}

fn test_super_table_name_roundtrip(py: Python) -> Result<(), String> {
    let mut arr1 = IntegerArray::<i32>::default();
    arr1.push(1);
    arr1.push(2);
    let array1 = Array::from_int32(arr1);
    let field = Field::new("x", ArrowType::Int32, false, None);
    let fa1 = FieldArray::new(field.clone(), array1);
    let table1 = Arc::new(Table::new("named_super".to_string(), Some(vec![fa1])));

    let mut arr2 = IntegerArray::<i32>::default();
    arr2.push(3);
    arr2.push(4);
    let array2 = Array::from_int32(arr2);
    let fa2 = FieldArray::new(field, array2);
    let table2 = Arc::new(Table::new("named_super".to_string(), Some(vec![fa2])));

    let super_table = SuperTable::from_batches(vec![table1, table2], None);

    let py_table = PyTable::from(super_table);
    let py_obj = py_table
        .into_pyobject(py)
        .map_err(|e| format!("export failed: {}", e))?;

    let back: PyTable = py_obj
        .extract()
        .map_err(|e| format!("import failed: {}", e))?;
    let recovered: SuperTable = back.into();
    if recovered.name != "named_super" {
        return Err(format!(
            "SuperTable name not preserved: expected 'named_super', got '{}'",
            recovered.name
        ));
    }
    Ok(())
}
