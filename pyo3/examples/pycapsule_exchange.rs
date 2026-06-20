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

//! # PyCapsule Exchange Example
//!
//! Demonstrates how the Arrow PyCapsule Interface simplifies data exchange
//! between Rust (MinArrow) and Python libraries.
//!
//! ## The problem PyCapsules solve
//!
//! **C Data Interface - Old Arrow approach** 
//! ```python
//! # Worked with PyArrow - requires both sides to know about
//! # PyArrow's private _export_to_c / _import_from_c methods and
//! # to pass raw memory addresses as integers.
//! addr = pyarrow_array._export_to_c()
//! # Pass integer address across FFI boundary...
//! ```
//!
//! **PyCapsule approach**:
//! ```python
//! # Works with various Arrow-compatible libraries natively: PyArrow, Polars, DuckDB,
//! # pandas (with ArrowDtype), nanoarrow, etc.
//! # The producer exposes __arrow_c_array__ / __arrow_c_stream__,
//! # the consumer calls it - all sorted.
//!
//! import polars as pl
//! df = pl.DataFrame({"a": [1, 2, 3]})
//! # Polars implements __arrow_c_stream__, minarrow consumes it directly
//! result = minarrow_pyo3.echo_table(df)
//! ```
//!
//! ## What this example shows
//!
//! 1. Exporting a MinArrow array as PyCapsules, consumed by PyArrow via `pa.array()`
//! 2. Exporting a MinArrow table as a stream PyCapsule, consumed by `RecordBatchReader`
//! 3. Importing a PyArrow array into MinArrow via `__arrow_c_array__`
//! 4. Importing a PyArrow table into MinArrow via `__arrow_c_stream__`
//!
//! ## Running this example
//!
//! ```bash
//! cd pyo3
//!
//! PYO3_PYTHON=.venv/bin/python cargo build --example pycapsule_exchange \
//!     --no-default-features \
//!     --features "datetime,extended_numeric_types,extended_categorical"
//!
//! # PYTHONHOME must point to the system prefix (stdlib lives there),
//! # PYTHONPATH adds the venv's site-packages (pyarrow etc.).
//! PYTHONHOME=/usr \
//!   PYTHONPATH=.venv/lib/python3.12/site-packages \
//!   LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
//!   cargo run --example pycapsule_exchange \
//!     --no-default-features \
//!     --features "datetime,extended_numeric_types,extended_categorical"
//! ```

use minarrow::ffi::arrow_dtype::ArrowType;
use minarrow::{Array, Field, FieldArray, FloatArray, IntegerArray, MaskedArray, NumericArray, Table};
use minarrow_pyo3::ffi::{to_py, to_rust};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use std::sync::Arc;

fn main() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| {
        println!("=== Arrow PyCapsule Interface Examples ===\n");

        example_1_export_array(py)?;
        example_2_export_table_stream(py)?;
        example_3_import_from_pyarrow_array(py)?;
        example_4_import_from_pyarrow_table(py)?;

        println!("\n=== All examples completed ===");
        Ok(())
    })
}

/// Export a MinArrow array as PyCapsules, then import into PyArrow.
///
/// MinArrow produces a wrapper with `__arrow_c_array__`. PyArrow's
/// `pa.array()` calls that method automatically to consume the data.
fn example_1_export_array(py: Python<'_>) -> PyResult<()> {
    println!("Example 1: Export MinArrow array -> PyArrow via __arrow_c_array__");
    println!("----------------------------------------------------------------");

    // Build an array in Rust
    let mut arr = IntegerArray::<i64>::default();
    for i in 0..5 {
        arr.push(i * 10);
    }
    let array = Array::from_int64(arr);
    let field = Field::new("values", ArrowType::Int64, false, None);
    println!("  Created MinArrow i64 array: [0, 10, 20, 30, 40]");

    // Export as PyCapsules (i.e., schema + array)
    let (schema_capsule, array_capsule) =
        to_py::array_to_capsules(Arc::new(array), &field, py)?;
    println!("  Exported as PyCapsules");

    // Build a wrapper that implements __arrow_c_array__ so PyArrow can consume it.
    // In a real #[pyfunction] you'd return ArrowArrayWrapper directly;
    // here we simulate by calling __arrow_c_array__ manually.
    let pyarrow = py.import("pyarrow")?;

    // PyArrow.Array._import_from_c expects (array_ptr, schema_ptr) as integers,
    // but we can extract the raw pointers from the capsules properly:
    let array_ptr = unsafe {
        pyo3::ffi::PyCapsule_GetPointer(
            array_capsule.as_ptr(),
            c"arrow_array".as_ptr(),
        )
    } as usize;
    let schema_ptr = unsafe {
        pyo3::ffi::PyCapsule_GetPointer(
            schema_capsule.as_ptr(),
            c"arrow_schema".as_ptr(),
        )
    } as usize;

    let pa_array = pyarrow
        .getattr("Array")?
        .call_method1("_import_from_c", (array_ptr, schema_ptr))?;

    let repr: String = pa_array.call_method0("__repr__")?.extract()?;
    println!("  PyArrow received: {}", repr.lines().next().unwrap_or(""));
    println!("  Done.\n");
    Ok(())
}

/// Export a MinArrow table as a stream PyCapsule, then consume in PyArrow.
///
/// For multi-column data, the stream interface is more natural - the
/// consumer gets a single capsule and pulls batches from it.
fn example_2_export_table_stream(py: Python<'_>) -> PyResult<()> {
    println!("Example 2: Export MinArrow table -> PyArrow via __arrow_c_stream__");
    println!("------------------------------------------------------------------");

    let mut ids = IntegerArray::<i32>::default();
    ids.push(1);
    ids.push(2);
    ids.push(3);

    let mut scores = FloatArray::<f64>::default();
    scores.push(9.5);
    scores.push(8.3);
    scores.push(7.1);

    let table = Table::new(
        "results".to_string(),
        Some(vec![
            FieldArray::new(
                Field::new("id", ArrowType::Int32, false, None),
                Array::from_int32(ids),
            ),
            FieldArray::new(
                Field::new("score", ArrowType::Float64, false, None),
                Array::from_float64(scores),
            ),
        ]),
    );
    println!("  Created MinArrow table: 3 rows x 2 columns (id, score)");

    // Export as a stream capsule and extract the raw pointer for PyArrow
    let stream_capsule = to_py::table_to_stream_capsule(&table, py)?;
    let stream_ptr = unsafe {
        pyo3::ffi::PyCapsule_GetPointer(
            stream_capsule.as_ptr(),
            c"arrow_array_stream".as_ptr(),
        )
    } as usize;

    let pyarrow = py.import("pyarrow")?;
    let reader = pyarrow
        .getattr("RecordBatchReader")?
        .call_method1("_import_from_c", (stream_ptr,))?;
    let pa_table = reader.call_method0("read_all")?;

    let num_rows: usize = pa_table.getattr("num_rows")?.extract()?;
    let schema_repr: String = pa_table.getattr("schema")?.call_method0("__repr__")?.extract()?;
    println!("  PyArrow received: {} rows", num_rows);
    println!("  Schema: {}", schema_repr.lines().next().unwrap_or(""));
    println!("  Done.\n");
    Ok(())
}

/// Import a PyArrow array into MinArrow via __arrow_c_array__.
///
/// This is the Python-to-Rust direction using PyCapsules. We call the
/// standard __arrow_c_array__ protocol method - the same code works with
/// Polars Series, nanoarrow arrays, etc.
fn example_3_import_from_pyarrow_array(py: Python<'_>) -> PyResult<()> {
    println!("Example 3: Import PyArrow array -> MinArrow via __arrow_c_array__");
    println!("-----------------------------------------------------------------");

    let pyarrow = py.import("pyarrow")?;
    let py_array = pyarrow.call_method1("array", (vec![100i64, 200, 300, 400, 500],))?;
    println!("  Created PyArrow array: [100, 200, 300, 400, 500]");

    let result = to_rust::try_capsule_array(&py_array);

    match result {
        Some(Ok(field_array)) => {
            println!("  Imported into MinArrow: {} elements", field_array.array.len());

            match &field_array.array {
                Array::NumericArray(NumericArray::Int64(a)) => {
                    let values: Vec<i64> = (0..a.len())
                        .map(|i| a.get(i).unwrap_or(0))
                        .collect();
                    println!("  Values: {:?}", values);
                }
                other => println!("  Got type: {:?}", std::mem::discriminant(other)),
            }
        }
        Some(Err(e)) => println!("  Import failed: {}", e),
        None => println!("  __arrow_c_array__ not available on this object"),
    }

    println!("  Done.\n");
    Ok(())
}

/// Import a PyArrow table into MinArrow via __arrow_c_stream__.
///
/// The stream protocol is ideal for tabular data - the consumer gets
/// schema and batches through a single interface.
fn example_4_import_from_pyarrow_table(py: Python<'_>) -> PyResult<()> {
    println!("Example 4: Import PyArrow table -> MinArrow via __arrow_c_stream__");
    println!("------------------------------------------------------------------");

    let pyarrow = py.import("pyarrow")?;
    let dict = vec![
        (
            "name",
            pyarrow.call_method1("array", (vec!["Alice", "Bob", "Charlie"],))?,
        ),
        (
            "age",
            pyarrow.call_method1("array", (vec![30i64, 25, 35],))?,
        ),
    ]
    .into_py_dict(py)?;

    let py_table = pyarrow.call_method1("table", (dict,))?;
    println!("  Created PyArrow table: name=['Alice','Bob','Charlie'], age=[30,25,35]");

    let result = to_rust::try_capsule_record_batch_stream(&py_table);

    match result {
        Some(Ok((batches, _metadata))) => {
            println!("  Imported {} batch(es) into MinArrow", batches.len());
            for (batch_idx, batch) in batches.iter().enumerate() {
                println!("  Batch {}: {} columns", batch_idx, batch.len());
                for (col_idx, (array, field)) in batch.iter().enumerate() {
                    println!(
                        "    Column {} '{}': {} rows, type={:?}",
                        col_idx,
                        field.name,
                        array.len(),
                        field.dtype
                    );
                }
            }
        }
        Some(Err(e)) => println!("  Import failed: {}", e),
        None => println!("  __arrow_c_stream__ not available on this object"),
    }

    println!("  Done.\n");
    Ok(())
}
