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

//! # Python Roundtrip Integration Test
//!
//! Demonstrates roundtrip conversion between MinArrow and PyArrow.
//!
//! ## Running
//!
//! ```bash
//! cd pyo3
//! PYO3_PYTHON=$(pwd)/.venv/bin/python \
//!   PYTHONHOME=/usr \
//!   PYTHONPATH=$(pwd)/.venv/lib/python3.12/site-packages \
//!   LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
//!   cargo test --test python_roundtrip \
//!     --no-default-features \
//!     --features "datetime,extended_numeric_types,extended_categorical"
//! ```

use minarrow::{Array, Field, FieldArray, IntegerArray, MaskedArray, StringArray, Table};
use minarrow::ffi::arrow_dtype::ArrowType;
use minarrow_pyo3::{PyArray, PyRecordBatch};
use pyo3::prelude::*;

fn main() -> PyResult<()> {
    // Initialise Python
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| {
        println!("=== MinArrow <-> PyArrow Roundtrip Example ===\n");

        // Test 1: Integer Array roundtrip
        test_integer_array_roundtrip(py)?;

        // Test 2: String Array roundtrip
        test_string_array_roundtrip(py)?;

        // Test 3: Table/RecordBatch roundtrip
        test_table_roundtrip(py)?;

        println!("\n=== All tests passed! ===");
        Ok(())
    })
}

fn test_integer_array_roundtrip(py: Python<'_>) -> PyResult<()> {
    println!("Test 1: Integer Array Roundtrip");
    println!("---------------------------------");

    // Create a MinArrow IntegerArray
    let mut arr = IntegerArray::<i32>::default();
    arr.push(1);
    arr.push(2);
    arr.push(3);
    arr.push(4);
    arr.push(5);

    let array = Array::from_int32(arr);
    let original_len = array.len();
    println!("  Created MinArrow i32 array with {} elements: [1, 2, 3, 4, 5]", original_len);

    // Wrap in PyArray
    let py_array = PyArray::from(array);

    // Convert to PyArrow
    let py_arrow_array = py_array.into_pyobject(py)?;
    println!("  Converted to PyArrow array");

    // Print the PyArrow array
    let py_repr: String = py_arrow_array.call_method0("__repr__")?.extract()?;
    println!("  PyArrow repr: {}", py_repr.lines().take(3).collect::<Vec<_>>().join(" "));

    // Convert back to MinArrow
    let roundtrip: PyArray = py_arrow_array.extract()?;
    let roundtrip_len = roundtrip.inner().len();
    println!("  Converted back to MinArrow array with {} elements", roundtrip_len);

    // Verify
    assert_eq!(original_len, roundtrip_len, "Array length mismatch!");
    println!("  ✓ Roundtrip successful!\n");

    Ok(())
}

fn test_string_array_roundtrip(py: Python<'_>) -> PyResult<()> {
    println!("Test 2: String Array Roundtrip");
    println!("-------------------------------");

    // Create a MinArrow StringArray
    let mut arr = StringArray::<u32>::default();
    arr.push_str("hello");
    arr.push_str("world");
    arr.push_str("from");
    arr.push_str("minarrow");

    let array = Array::from_string32(arr);
    let original_len = array.len();
    println!("  Created MinArrow string array with {} elements", original_len);

    // Wrap in PyArray
    let py_array = PyArray::from(array);

    // Convert to PyArrow
    let py_arrow_array = py_array.into_pyobject(py)?;
    println!("  Converted to PyArrow array");

    // Print the PyArrow array
    let py_repr: String = py_arrow_array.call_method0("__repr__")?.extract()?;
    println!("  PyArrow repr: {}", py_repr.lines().take(3).collect::<Vec<_>>().join(" "));

    // Convert back to MinArrow
    let roundtrip: PyArray = py_arrow_array.extract()?;
    let roundtrip_len = roundtrip.inner().len();
    println!("  Converted back to MinArrow array with {} elements", roundtrip_len);

    // Verify
    assert_eq!(original_len, roundtrip_len, "Array length mismatch!");
    println!("  ✓ Roundtrip successful!\n");

    Ok(())
}

fn test_table_roundtrip(py: Python<'_>) -> PyResult<()> {
    println!("Test 3: Table/RecordBatch Roundtrip");
    println!("------------------------------------");

    // Create MinArrow arrays for the table
    let mut int_arr = IntegerArray::<i64>::default();
    int_arr.push(100);
    int_arr.push(200);
    int_arr.push(300);
    let int_array = Array::from_int64(int_arr);

    let mut str_arr = StringArray::<u32>::default();
    str_arr.push_str("alpha");
    str_arr.push_str("beta");
    str_arr.push_str("gamma");
    let str_array = Array::from_string32(str_arr);

    // Create FieldArrays
    let field1 = Field::new("id", ArrowType::Int64, false, None);
    let field2 = Field::new("name", ArrowType::String, false, None);

    let fa1 = FieldArray::new(field1, int_array);
    let fa2 = FieldArray::new(field2, str_array);

    // Create Table
    let table = Table::new("test_table".to_string(), Some(vec![fa1, fa2]));
    let original_rows = table.n_rows();
    let original_cols = table.n_cols();
    println!("  Created MinArrow Table with {} rows x {} columns", original_rows, original_cols);

    // Wrap in PyRecordBatch
    let py_batch = PyRecordBatch::from(table);

    // Convert to PyArrow
    let py_arrow_batch = py_batch.into_pyobject(py)?;
    println!("  Converted to PyArrow RecordBatch");

    // Print info
    let num_rows: usize = py_arrow_batch.getattr("num_rows")?.extract()?;
    let num_cols: usize = py_arrow_batch.getattr("num_columns")?.extract()?;
    println!("  PyArrow RecordBatch: {} rows x {} columns", num_rows, num_cols);

    // Convert back to MinArrow
    let roundtrip: PyRecordBatch = py_arrow_batch.extract()?;
    let roundtrip_rows = roundtrip.inner().n_rows();
    let roundtrip_cols = roundtrip.inner().n_cols();
    println!("  Converted back to MinArrow Table: {} rows x {} columns", roundtrip_rows, roundtrip_cols);

    // Verify
    assert_eq!(original_rows, roundtrip_rows, "Row count mismatch!");
    assert_eq!(original_cols, roundtrip_cols, "Column count mismatch!");
    println!("  ✓ Roundtrip successful!");

    Ok(())
}
