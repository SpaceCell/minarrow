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

//! # minarrow-pyo3 - PyO3 Bindings for MinArrow
//!
//! Zero-copy Python bindings for MinArrow via the Arrow C Data Interface and PyCapsules.
//!
//! This crate provides transparent wrapper types that enable zero-copy conversion
//! between MinArrow's Rust types and PyArrow's Python types.
//!
//! ## Features
//!
//! - **Zero-copy data transfer** via Arrow C Data Interface
//! - **Transparent wrappers** (`PyArray`, `PyRecordBatch`) implementing PyO3 traits
//! - **Idiomatic Rust API** for building Python extensions
//!
//! ## Copy Semantics
//!
//! ### Zero-copy
//!
//! All primary data buffers are transferred without copying in both directions.
//! This applies to all export paths, single array imports, ChunkedArray chunk
//! imports, and RecordBatch/Table column imports via both the PyCapsule stream
//! and legacy `_import_from_c` paths.
//!
//! ### Copied by design
//!
//! The following are copied during import because they require structural
//! transformation between MinArrow and Arrow representations:
//!
//! - **Null bitmasks** — reconstructed into MinArrow's `Bitmask` type on import.
//!   These are small: ceil(N/8) bytes for N elements.
//! - **String offsets** — reconstructed into MinArrow's offset representation.
//! - **Categorical dictionary strings** — Arrow stores dictionaries as contiguous
//!   offsets+data; MinArrow stores them as `Vec64<String>` with individual heap
//!   allocations. The integer codes buffer is zero-copy.
//! - **Field metadata** — names, types, and flags are lightweight and always copied.
//!
//! ## Type Mappings
//! 
//! Minarrow calls an object with a header, rows and columns a 'Table' favouring broader matter-of-factness.
//! Apache Arrow calls it a 'RecordBatch' in line with the Apache Arrow standard, whereby a 'Table' (at least in PyArrow),
//! is considered a chunked composition of those RecordBatches, for a more highly engineered approach.
//! Below is how they map to one another for the equivalent memory and object layout.
//! 
//! | MinArrow | PyArrow | Wrapper Type |
//! |----------|---------|--------------|
//! | `Array` | `pa.Array` | `PyArray` |
//! | `Table` | `pa.RecordBatch` | `PyRecordBatch` |
//! | `SuperTable` | `pa.Table` | `PyTable` |
//! | `SuperArray` | `pa.ChunkedArray` | `PyChunkedArray` |
//!
//! ## Conversion Protocols
//!
//! Two protocols are supported for data exchange:
//!
//! 1. **Arrow PyCapsule Interface** - the standard `__arrow_c_array__` / `__arrow_c_stream__`
//!    protocol. Works with any Arrow-compatible Python library including PyArrow, Polars,
//!    DuckDB, nanoarrow, and pandas with ArrowDtype.
//!
//! 2. **Legacy `_export_to_c`** - PyArrow-specific fallback using raw pointer integers.
//!
//! Import functions try the PyCapsule protocol first, falling back to the legacy approach
//! for older PyArrow versions.
//!
//! For the complete array data type mapping including numeric, temporal, boolean, text,
//! and categorical types, see the [`ffi`] module documentation.
//!
//! ## Example
//!
//! ```ignore
//! use minarrow_pyo3::{PyArray, PyRecordBatch};
//! use pyo3::prelude::*;
//!
//! #[pyfunction]
//! fn process_batch(input: PyRecordBatch) -> PyResult<PyRecordBatch> {
//!     let table: minarrow::Table = input.into();
//!     // Process the table using MinArrow...
//!     Ok(PyRecordBatch::from(table))
//! }
//!
//! #[pymodule]
//! fn my_extension(m: &Bound<'_, PyModule>) -> PyResult<()> {
//!     m.add_function(wrap_pyfunction!(process_batch, m)?)?;
//!     Ok(())
//! }
//! ```
//!
//! In Python:
//! ```python
//! import pyarrow as pa
//! import my_extension
//!
//! batch = pa.RecordBatch.from_pydict({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
//! result = my_extension.process_batch(batch)
//! ```

use pyo3::prelude::*;
use std::sync::Arc;

pub mod error;
pub mod ffi;
pub mod types;

// Re-export the main types for ease of use
pub use error::{PyMinarrowError, PyMinarrowResult};
pub use types::{
    PyArray, PyArrayView, PyChunkedArray, PyChunkedArrayView, PyField, PyRecordBatch,
    PyRecordBatchView, PyTable, PyTableView,
};

// Re-export minarrow types that users might need
pub use minarrow::{Array, Field, FieldArray, MaskedArray, NumericArray, SuperArray, SuperTable, Table, TextArray};

/// Echo back a PyArrow array after roundtrip through MinArrow.
/// Used to test that conversion works correctly.
#[pyfunction]
fn echo_array(arr: PyArray) -> PyResult<PyArray> {
    // The array is converted to MinArrow on input and back to PyArrow on output
    Ok(arr)
}

/// Echo back a PyArrow RecordBatch after roundtrip through MinArrow.
/// Used to test that conversion works correctly.
#[pyfunction]
fn echo_batch(batch: PyRecordBatch) -> PyResult<PyRecordBatch> {
    // The batch is converted to MinArrow Table on input and back to PyArrow on output
    Ok(batch)
}

/// Get information about a PyArrow array after converting to MinArrow.
#[pyfunction]
fn array_info(arr: PyArray) -> PyResult<String> {
    let inner = arr.inner();
    Ok(format!(
        "MinArrow Array: len={}, null_count={}",
        inner.len(),
        inner.null_count()
    ))
}

/// Get information about a PyArrow RecordBatch after converting to MinArrow.
#[pyfunction]
fn batch_info(batch: PyRecordBatch) -> PyResult<String> {
    let inner = batch.inner();
    Ok(format!(
        "MinArrow Table: rows={}, cols={}",
        inner.n_rows(),
        inner.n_cols()
    ))
}

/// Echo back a PyArrow Table after roundtrip through MinArrow.
/// Used to test that conversion works correctly.
#[pyfunction]
fn echo_table(table: PyTable) -> PyResult<PyTable> {
    // The table is converted to MinArrow SuperTable on input and back to PyArrow on output
    Ok(table)
}

/// Echo back a PyArrow ChunkedArray after roundtrip through MinArrow.
/// Used to test that conversion works correctly.
#[pyfunction]
fn echo_chunked(arr: PyChunkedArray) -> PyResult<PyChunkedArray> {
    // The array is converted to MinArrow SuperArray on input and back to PyArrow on output
    Ok(arr)
}

/// Get information about a PyArrow Table after converting to MinArrow.
#[pyfunction]
fn table_info(table: PyTable) -> PyResult<String> {
    let inner = table.inner();
    Ok(format!(
        "MinArrow SuperTable: batches={}, rows={}, cols={}",
        inner.batches.len(),
        inner.n_rows,
        inner.schema.len()
    ))
}

/// Get information about a PyArrow ChunkedArray after converting to MinArrow.
#[pyfunction]
fn chunked_info(arr: PyChunkedArray) -> PyResult<String> {
    let inner = arr.inner();
    Ok(format!(
        "MinArrow SuperArray: chunks={}, len={}",
        inner.n_chunks(),
        inner.len()
    ))
}

/// Export a MinArrow array as a pair of Arrow PyCapsules (schema, array).
///
/// The returned tuple follows the Arrow PyCapsule Interface and can be
/// consumed by any library supporting the protocol.
#[pyfunction]
fn export_array_capsule(py: Python, arr: PyArray) -> PyResult<ArrowArrayWrapper> {
    let fa = arr.field_array();
    let array = Arc::new(fa.array.clone());
    let (schema_capsule, array_capsule) = ffi::to_py::array_to_capsules(array, &fa.field, py)?;
    Ok(ArrowArrayWrapper {
        schema_capsule: Some(schema_capsule),
        array_capsule: Some(array_capsule),
    })
}

/// Export a MinArrow RecordBatch as an ArrowArrayStream PyCapsule.
///
/// The stream yields one struct array representing the record batch.
#[pyfunction]
fn export_batch_stream_capsule(py: Python, batch: PyRecordBatch) -> PyResult<ArrowStream> {
    let table = batch.inner();
    let capsule = ffi::to_py::table_to_stream_capsule(table, py)?;
    Ok(ArrowStream {
        capsule: Some(capsule),
    })
}

/// Export a MinArrow Table as an ArrowArrayStream PyCapsule.
///
/// The stream yields one struct array per batch in the table.
#[pyfunction]
fn export_table_stream_capsule(py: Python, table: PyTable) -> PyResult<ArrowStream> {
    let super_table = table.inner();
    let capsule = ffi::to_py::super_table_to_stream_capsule(super_table, py)?;
    Ok(ArrowStream {
        capsule: Some(capsule),
    })
}

/// Export a MinArrow ChunkedArray as an ArrowArrayStream PyCapsule.
///
/// The stream yields one plain array per chunk.
#[pyfunction]
fn export_chunked_stream_capsule(py: Python, arr: PyChunkedArray) -> PyResult<ArrowStream> {
    let super_array = arr.inner();
    let capsule = ffi::to_py::super_array_to_stream_capsule(super_array, py)?;
    Ok(ArrowStream {
        capsule: Some(capsule),
    })
}

// PyCapsule protocol wrapper types

/// Python-visible wrapper implementing `__arrow_c_stream__`.
///
/// Any Arrow-compatible Python library can consume this object directly,
/// e.g. `pa.RecordBatchReader.from_stream(obj)` or `nanoarrow.ArrayStream(obj)`.
#[pyclass(name = "ArrowStream")]
struct ArrowStream {
    capsule: Option<PyObject>,
}

#[pymethods]
impl ArrowStream {
    /// Arrow PyCapsule stream protocol.
    ///
    /// Returns the underlying ArrowArrayStream capsule. The capsule can
    /// only be consumed once - subsequent calls raise ValueError.
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__(&mut self, requested_schema: Option<PyObject>) -> PyResult<PyObject> {
        let _ = requested_schema;
        self.capsule.take().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "ArrowStream capsule has already been consumed",
            )
        })
    }
}

/// Python-visible wrapper implementing `__arrow_c_array__`.
///
/// Any Arrow-compatible Python library can consume this object directly,
/// e.g. `pa.array(obj)`.
#[pyclass(name = "ArrowArray")]
struct ArrowArrayWrapper {
    schema_capsule: Option<PyObject>,
    array_capsule: Option<PyObject>,
}

#[pymethods]
impl ArrowArrayWrapper {
    /// Arrow PyCapsule array protocol.
    ///
    /// Returns (schema_capsule, array_capsule). The capsules can only
    /// be consumed once - subsequent calls raise ValueError.
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_array__(
        &mut self,
        requested_schema: Option<PyObject>,
    ) -> PyResult<(PyObject, PyObject)> {
        let _ = requested_schema;
        let schema = self.schema_capsule.take();
        let array = self.array_capsule.take();
        match (schema, array) {
            (Some(s), Some(a)) => Ok((s, a)),
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "ArrowArray capsules have already been consumed",
            )),
        }
    }
}

// Data generators - these produce protocol-conforming objects

/// Generate sample data entirely in Rust and return as an ArrowStream.
///
/// The returned object implements `__arrow_c_stream__` and can be consumed
/// by any Arrow-compatible Python library. The consumer never touches
/// minarrow types directly.
///
/// Contains a batch with columns: id (int64), score (float64), label (utf8).
#[pyfunction]
fn generate_sample_batch(py: Python) -> PyResult<ArrowStream> {
    use minarrow::ffi::arrow_dtype::ArrowType;

    let mut ids = minarrow::IntegerArray::<i64>::default();
    let mut scores = minarrow::FloatArray::<f64>::default();
    let mut labels = minarrow::StringArray::<u32>::default();
    for i in 0..5 {
        ids.push(i + 1);
        scores.push((i as f64 + 1.0) * 1.1);
        labels.push_str(&format!("item_{}", i + 1));
    }

    let table = Table::new(
        "sample".to_string(),
        Some(vec![
            FieldArray::new(
                Field::new("id", ArrowType::Int64, false, None),
                Array::from_int64(ids),
            ),
            FieldArray::new(
                Field::new("score", ArrowType::Float64, false, None),
                Array::from_float64(scores),
            ),
            FieldArray::new(
                Field::new("label", ArrowType::String, false, None),
                Array::from_string32(labels),
            ),
        ]),
    );

    let capsule = ffi::to_py::table_to_stream_capsule(&table, py)?;
    Ok(ArrowStream {
        capsule: Some(capsule),
    })
}

/// Generate a sample array with nulls in Rust and return as an ArrowArray.
///
/// The returned object implements `__arrow_c_array__` and can be consumed
/// by `pa.array(obj)` or any Arrow-compatible library.
///
/// Contains a nullable int64 array: [10, null, 30, null, 50].
#[pyfunction]
fn generate_nullable_array(py: Python) -> PyResult<ArrowArrayWrapper> {
    use minarrow::ffi::arrow_dtype::ArrowType;

    let mut arr = minarrow::IntegerArray::<i64>::default();
    arr.push(10);
    arr.push_null();
    arr.push(30);
    arr.push_null();
    arr.push(50);

    let array = Array::from_int64(arr);
    let field = Field::new("values", ArrowType::Int64, true, None);
    let (schema_capsule, array_capsule) =
        ffi::to_py::array_to_capsules(Arc::new(array), &field, py)?;
    Ok(ArrowArrayWrapper {
        schema_capsule: Some(schema_capsule),
        array_capsule: Some(array_capsule),
    })
}

/// Python module definition for minarrow_pyo3.
///
/// This module primarily provides type conversion capabilities via the
/// `PyArray` and `PyRecordBatch` wrapper types. The actual conversions
/// happen automatically when these types are used as function parameters
/// or return values in PyO3 functions.
#[pymodule]
fn minarrow_pyo3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Module-level docstring
    m.add("__doc__", "PyO3 bindings for MinArrow - zero-copy Arrow interop with Python")?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Add test functions
    m.add_function(wrap_pyfunction!(echo_array, m)?)?;
    m.add_function(wrap_pyfunction!(echo_batch, m)?)?;
    m.add_function(wrap_pyfunction!(echo_table, m)?)?;
    m.add_function(wrap_pyfunction!(echo_chunked, m)?)?;
    m.add_function(wrap_pyfunction!(array_info, m)?)?;
    m.add_function(wrap_pyfunction!(batch_info, m)?)?;
    m.add_function(wrap_pyfunction!(table_info, m)?)?;
    m.add_function(wrap_pyfunction!(chunked_info, m)?)?;

    // PyCapsule export functions
    m.add_function(wrap_pyfunction!(export_array_capsule, m)?)?;
    m.add_function(wrap_pyfunction!(export_batch_stream_capsule, m)?)?;
    m.add_function(wrap_pyfunction!(export_table_stream_capsule, m)?)?;
    m.add_function(wrap_pyfunction!(export_chunked_stream_capsule, m)?)?;

    // PyCapsule protocol wrapper types
    m.add_class::<ArrowStream>()?;
    m.add_class::<ArrowArrayWrapper>()?;

    // Data generators (return protocol-conforming objects)
    m.add_function(wrap_pyfunction!(generate_sample_batch, m)?)?;
    m.add_function(wrap_pyfunction!(generate_nullable_array, m)?)?;

    Ok(())
}
