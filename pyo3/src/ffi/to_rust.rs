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

//! # Python to MinArrow Conversion
//!
//! Converts Python Arrow-compatible objects to MinArrow arrays using the
//! Arrow PyCapsule Interface and Arrow C Data Interface.
//!
//! ## Import Strategy
//! Each import function tries the modern PyCapsule protocol first
//! (`__arrow_c_array__` / `__arrow_c_stream__`), falling back to the legacy
//! `_export_to_c` pointer-integer approach for older PyArrow versions.
//! 
use minarrow::Consolidate;
use minarrow::ffi::arrow_c_ffi::{
    ArrowArray, ArrowArrayStream, ArrowSchema, import_from_c_owned,
    import_record_batch_stream_with_metadata,
};
use minarrow::{Field, FieldArray, SuperArray, SuperTable};
use pyo3::ffi::Py_uintptr_t;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::sync::Arc;

use crate::error::{PyMinarrowError, PyMinarrowResult};
use crate::ffi::to_py::TABLE_NAME_KEY;

/// Tries to extract the `minarrow:table_name` value from a PyArrow schema's metadata.
/// Returns an empty string if the metadata key is absent or extraction fails.
fn extract_table_name_from_pyarrow_schema(schema: &Bound<PyAny>) -> String {
    schema
        .getattr("metadata")
        .ok()
        .and_then(|meta| {
            if meta.is_none() {
                return None;
            }
            // PyArrow schema.metadata is a dict with bytes keys
            let key = TABLE_NAME_KEY.as_bytes();
            meta.call_method1("get", (key,))
                .ok()
                .and_then(|val| {
                    if val.is_none() {
                        return None;
                    }
                    // Value is bytes in PyArrow metadata
                    val.extract::<Vec<u8>>()
                        .ok()
                        .and_then(|bytes| String::from_utf8(bytes).ok())
                })
        })
        .unwrap_or_default()
}

/// Splits imported stream metadata into the table name and remaining entries.
///
/// The `minarrow:table_name` key is extracted as the table name. All other
/// entries are returned as remaining metadata for `Table::new_with_metadata`.
#[cfg(feature = "table_metadata")]
fn split_stream_metadata(
    metadata: Option<std::collections::BTreeMap<String, String>>,
) -> (String, std::collections::BTreeMap<String, String>) {
    match metadata {
        None => (String::new(), std::collections::BTreeMap::new()),
        Some(mut m) => {
            let name = m.remove(TABLE_NAME_KEY).unwrap_or_default();
            (name, m)
        }
    }
}

/// Extracts the table name from imported stream metadata.
#[cfg(not(feature = "table_metadata"))]
fn extract_table_name(
    metadata: &Option<std::collections::BTreeMap<String, String>>,
) -> String {
    metadata
        .as_ref()
        .and_then(|m| m.get(TABLE_NAME_KEY))
        .cloned()
        .unwrap_or_default()
}

// PyCapsule helpers

/// Attempts to import a single array via the `__arrow_c_array__` PyCapsule protocol.
///
/// Returns `None` if the object does not support the protocol, allowing the
/// caller to fall back to the legacy approach.
pub fn try_capsule_array(obj: &Bound<PyAny>) -> Option<PyMinarrowResult<FieldArray>> {
    let has_method = obj.hasattr("__arrow_c_array__").ok()?;
    if !has_method {
        return None;
    }
    Some(import_capsule_array(obj))
}

/// Imports a single array from `__arrow_c_array__` PyCapsule pair.
fn import_capsule_array(obj: &Bound<PyAny>) -> PyMinarrowResult<FieldArray> {
    let py = obj.py();

    // Call __arrow_c_array__(requested_schema=None) -> (schema_capsule, array_capsule)
    let result = obj
        .call_method1("__arrow_c_array__", (py.None(),))
        .map_err(|e| {
            PyMinarrowError::PyArrow(format!("Failed to call __arrow_c_array__: {}", e))
        })?;

    let tuple: &Bound<PyTuple> = result.downcast().map_err(|e| {
        PyMinarrowError::PyArrow(format!(
            "__arrow_c_array__ did not return a tuple: {}",
            e
        ))
    })?;

    let schema_capsule = tuple.get_item(0).map_err(|e| {
        PyMinarrowError::PyArrow(format!("Failed to get schema capsule: {}", e))
    })?;
    let array_capsule = tuple.get_item(1).map_err(|e| {
        PyMinarrowError::PyArrow(format!("Failed to get array capsule: {}", e))
    })?;

    // Extract raw pointers from capsules using ctypes
    let schema_ptr = capsule_to_ptr(&schema_capsule, c"arrow_schema")?;
    let array_ptr = capsule_to_ptr(&array_capsule, c"arrow_array")?;

    // Move/consume the C structs out and replace with empty ones to prevent double-free
    let schema_box = unsafe {
        let moved = Box::new(std::ptr::read(schema_ptr as *const ArrowSchema));
        std::ptr::write(schema_ptr as *mut ArrowSchema, ArrowSchema::empty());
        moved
    };
    let array_box = unsafe {
        let moved = Box::new(std::ptr::read(array_ptr as *const ArrowArray));
        std::ptr::write(array_ptr as *mut ArrowArray, ArrowArray::empty());
        moved
    };

    let (array, field) = unsafe { import_from_c_owned(array_box, schema_box) };
    Ok(FieldArray::new(field, (*array).clone()))
}

/// Result type for record-batch stream import: batches plus optional schema metadata.
type StreamImportResult = (
    Vec<Vec<(Arc<minarrow::Array>, Field)>>,
    Option<std::collections::BTreeMap<String, String>>,
);

/// Attempts to import via `__arrow_c_stream__` for record-batch streams.
///
/// Returns `None` if the object does not support the protocol.
pub fn try_capsule_record_batch_stream(
    obj: &Bound<PyAny>,
) -> Option<PyMinarrowResult<StreamImportResult>> {
    let has_method = obj.hasattr("__arrow_c_stream__").ok()?;
    if !has_method {
        return None;
    }
    Some(import_capsule_record_batch_stream(obj))
}

/// Imports a record-batch stream from `__arrow_c_stream__` PyCapsule.
fn import_capsule_record_batch_stream(
    obj: &Bound<PyAny>,
) -> PyMinarrowResult<StreamImportResult> {
    let py = obj.py();

    let capsule = obj
        .call_method1("__arrow_c_stream__", (py.None(),))
        .map_err(|e| {
            PyMinarrowError::PyArrow(format!("Failed to call __arrow_c_stream__: {}", e))
        })?;

    let stream_ptr = capsule_to_ptr(&capsule, c"arrow_array_stream")? as *mut ArrowArrayStream;

    // Move the stream out and replace with empty to prevent double-free by capsule destructor
    let moved_stream = unsafe {
        let s = std::ptr::read(stream_ptr);
        std::ptr::write(stream_ptr, ArrowArrayStream::empty());
        s
    };

    // Write to a new heap allocation for import_record_batch_stream
    let stream_box = Box::new(moved_stream);
    let raw_ptr = Box::into_raw(stream_box);

    let (batches, metadata) = unsafe { import_record_batch_stream_with_metadata(raw_ptr) };
    Ok((batches, metadata))
}

/// Attempts to import via `__arrow_c_stream__` for plain array streams.
///
/// Returns `None` if the object does not support the protocol.
fn try_capsule_array_stream(
    obj: &Bound<PyAny>,
) -> Option<PyMinarrowResult<(Vec<Arc<minarrow::Array>>, Field)>> {
    let has_method = obj.hasattr("__arrow_c_stream__").ok()?;
    if !has_method {
        return None;
    }
    Some(import_capsule_array_stream(obj))
}

/// Imports a plain array stream from `__arrow_c_stream__` PyCapsule.
fn import_capsule_array_stream(
    obj: &Bound<PyAny>,
) -> PyMinarrowResult<(Vec<Arc<minarrow::Array>>, Field)> {
    let py = obj.py();

    let capsule = obj
        .call_method1("__arrow_c_stream__", (py.None(),))
        .map_err(|e| {
            PyMinarrowError::PyArrow(format!("Failed to call __arrow_c_stream__: {}", e))
        })?;

    let stream_ptr = capsule_to_ptr(&capsule, c"arrow_array_stream")? as *mut ArrowArrayStream;

    let moved_stream = unsafe {
        let s = std::ptr::read(stream_ptr);
        std::ptr::write(stream_ptr, ArrowArrayStream::empty());
        s
    };

    let stream_box = Box::new(moved_stream);
    let raw_ptr = Box::into_raw(stream_box);

    let result = unsafe { minarrow::ffi::arrow_c_ffi::import_array_stream(raw_ptr) };
    Ok(result)
}

/// Extracts the raw pointer from a PyCapsule as a uintptr_t integer.
///
/// Uses the Python PyCapsule C API directly to extract the pointer value.
/// The `name` must match the name the capsule was created with.
fn capsule_to_ptr(capsule: &Bound<PyAny>, name: &std::ffi::CStr) -> PyMinarrowResult<usize> {
    let ptr = unsafe {
        pyo3::ffi::PyCapsule_GetPointer(capsule.as_ptr(), name.as_ptr())
    };

    if ptr.is_null() {
        // Check for Python error
        let py = capsule.py();
        if let Some(err) = PyErr::take(py) {
            return Err(PyMinarrowError::PyArrow(format!(
                "PyCapsule_GetPointer failed: {}",
                err,
            )));
        }
        return Err(PyMinarrowError::PyArrow(
            "PyCapsule pointer is null (capsule may have been consumed already)".to_string(),
        ));
    }

    Ok(ptr as usize)
}

// Public import functions

/// Converts a PyArrow Array, pd.Series, pl.Series, or any Arrow-compatible
/// Python object to a MinArrow FieldArray.
///
/// Tries `__arrow_c_array__` first, then `__arrow_c_stream__` for objects
/// that only expose a stream interface e.g. pandas and Polars Series, then
/// falls back to the legacy `_export_to_c` approach.
///
/// # Arguments
/// * `obj` - A Python object implementing the Arrow array interface
///
/// # Returns
/// * `PyMinarrowResult<FieldArray>` - The converted MinArrow FieldArray
pub fn array_to_rust(obj: &Bound<PyAny>) -> PyMinarrowResult<FieldArray> {
    // Try PyCapsule array protocol first
    if let Some(result) = try_capsule_array(obj) {
        return result;
    }

    // Try PyCapsule stream protocol for objects like pd.Series / pl.Series
    // that expose __arrow_c_stream__ but not __arrow_c_array__
    if let Some(result) = try_capsule_array_stream(obj) {
        let (arrays, field) = result?;
        if arrays.is_empty() {
            return Ok(FieldArray::new(field, minarrow::Array::Null));
        }
        if arrays.len() == 1 {
            let array = Arc::try_unwrap(arrays.into_iter().next().unwrap())
                .unwrap_or_else(|arc| (*arc).clone());
            return Ok(FieldArray::new(field, array));
        }
        // Concatenate multiple chunks into a single array
        use minarrow::Concatenate;
        let mut iter = arrays.into_iter();
        let first = Arc::try_unwrap(iter.next().unwrap())
            .unwrap_or_else(|arc| (*arc).clone());
        let combined = iter.fold(first, |acc, chunk| {
            let arr = Arc::try_unwrap(chunk).unwrap_or_else(|arc| (*arc).clone());
            acc.concat(arr).expect("Failed to concatenate array chunks")
        });
        return Ok(FieldArray::new(field, combined));
    }

    // Fall back to _export_to_c approach
    array_to_rust_c(obj)
}

/// Legacy import path using `_export_to_c` pointer integers.
fn array_to_rust_c(obj: &Bound<PyAny>) -> PyMinarrowResult<FieldArray> {
    let array = Box::new(ArrowArray::empty());
    let schema = Box::new(ArrowSchema::empty());

    let array_ptr = Box::into_raw(array);
    let schema_ptr = Box::into_raw(schema);

    obj.call_method1(
        "_export_to_c",
        (array_ptr as Py_uintptr_t, schema_ptr as Py_uintptr_t),
    )
    .map_err(|e| PyMinarrowError::PyArrow(format!("Failed to export PyArrow array: {}", e)))?;

    let array_box = unsafe { Box::from_raw(array_ptr) };
    let schema_box = unsafe { Box::from_raw(schema_ptr) };

    let (array, field) = unsafe { import_from_c_owned(array_box, schema_box) };
    Ok(FieldArray::new(field, (*array).clone()))
}

/// Converts a PyArrow RecordBatch (or compatible object) to a MinArrow Table.
///
/// Tries `__arrow_c_stream__` first (yields one batch), then falls back
/// to the legacy column-by-column approach.
///
/// # Arguments
/// * `obj` - A Python RecordBatch or compatible object
///
/// # Returns
/// * `PyMinarrowResult<minarrow::Table>` - The converted MinArrow Table
pub fn record_batch_to_rust(obj: &Bound<PyAny>) -> PyMinarrowResult<minarrow::Table> {
    // Try PyCapsule stream (RecordBatch may support __arrow_c_stream__)
    if let Some(result) = try_capsule_record_batch_stream(obj) {
        let (batches, metadata) = result?;

        #[cfg(feature = "table_metadata")]
        let (table_name, remaining_meta) = split_stream_metadata(metadata);
        #[cfg(not(feature = "table_metadata"))]
        let table_name = extract_table_name(&metadata);

        if batches.is_empty() {
            return Ok(minarrow::Table::new(table_name, None));
        }
        // Build a Table from each batch, then consolidate into one.
        // For a single batch this returns it directly without copying.
        let tables: Vec<minarrow::Table> = batches
            .into_iter()
            .map(|columns| {
                let cols: Vec<FieldArray> = columns
                    .into_iter()
                    .map(|(array, field)| FieldArray::new(field, (*array).clone()))
                    .collect();
                #[cfg(feature = "table_metadata")]
                let table = if remaining_meta.is_empty() {
                    minarrow::Table::new(table_name.clone(), Some(cols))
                } else {
                    minarrow::Table::new_with_metadata(
                        table_name.clone(),
                        Some(cols),
                        remaining_meta.clone(),
                    )
                };
                #[cfg(not(feature = "table_metadata"))]
                let table = minarrow::Table::new(table_name.clone(), Some(cols));
                table
            })
            .collect();
        return Ok(tables.consolidate());
    }

    // Fall back to legacy approach
    record_batch_to_rust_legacy(obj)
}

/// Legacy RecordBatch import using column-by-column `_export_to_c`.
fn record_batch_to_rust_legacy(obj: &Bound<PyAny>) -> PyMinarrowResult<minarrow::Table> {
    let num_columns: usize = obj
        .getattr("num_columns")
        .map_err(|e| PyMinarrowError::PyArrow(format!("Failed to get num_columns: {}", e)))?
        .extract()
        .map_err(|e| PyMinarrowError::PyArrow(format!("Failed to extract num_columns: {}", e)))?;

    let schema = obj
        .getattr("schema")
        .map_err(|e| PyMinarrowError::PyArrow(format!("Failed to get schema: {}", e)))?;

    // Try to recover the table name from schema metadata
    let table_name = extract_table_name_from_pyarrow_schema(&schema);

    let mut cols = Vec::with_capacity(num_columns);

    for i in 0..num_columns {
        let column = obj
            .call_method1("column", (i,))
            .map_err(|e| {
                PyMinarrowError::PyArrow(format!("Failed to get column {}: {}", i, e))
            })?;

        let field = schema
            .call_method1("field", (i,))
            .map_err(|e| {
                PyMinarrowError::PyArrow(format!("Failed to get field {}: {}", i, e))
            })?;

        let name: String = field
            .getattr("name")
            .map_err(|e| PyMinarrowError::PyArrow(format!("Failed to get field name: {}", e)))?
            .extract()
            .map_err(|e| {
                PyMinarrowError::PyArrow(format!("Failed to extract field name: {}", e))
            })?;

        let mut field_array = array_to_rust(&column)?;

        field_array.field = Arc::new(Field::new(
            name,
            field_array.field.dtype.clone(),
            field_array.field.nullable,
            None,
        ));
        cols.push(field_array);
    }

    let table = minarrow::Table::new(table_name, Some(cols));
    Ok(table)
}

/// Converts a PyArrow Table (or Polars DataFrame, or any Arrow-compatible
/// object with `__arrow_c_stream__`) to a MinArrow SuperTable.
///
/// Tries `__arrow_c_stream__` first, then falls back to `to_batches()`.
///
/// # Arguments
/// * `obj` - A Python Table, DataFrame, or compatible object
///
/// # Returns
/// * `PyMinarrowResult<SuperTable>` - The converted MinArrow SuperTable
pub fn table_to_rust(obj: &Bound<PyAny>) -> PyMinarrowResult<SuperTable> {
    // Try PyCapsule stream
    if let Some(result) = try_capsule_record_batch_stream(obj) {
        let (batches, metadata) = result?;

        #[cfg(feature = "table_metadata")]
        let (table_name, remaining_meta) = split_stream_metadata(metadata);
        #[cfg(not(feature = "table_metadata"))]
        let table_name = extract_table_name(&metadata);

        if batches.is_empty() {
            return Ok(SuperTable::new(table_name));
        }

        let mut tables = Vec::with_capacity(batches.len());
        for columns in batches {
            let cols: Vec<FieldArray> = columns
                .into_iter()
                .map(|(array, field)| FieldArray::new(field, (*array).clone()))
                .collect();
            #[cfg(feature = "table_metadata")]
            let table = if remaining_meta.is_empty() {
                minarrow::Table::new(table_name.clone(), Some(cols))
            } else {
                minarrow::Table::new_with_metadata(
                    table_name.clone(),
                    Some(cols),
                    remaining_meta.clone(),
                )
            };
            #[cfg(not(feature = "table_metadata"))]
            let table = minarrow::Table::new(table_name.clone(), Some(cols));
            tables.push(Arc::new(table));
        }

        return Ok(SuperTable::from_batches(tables, None));
    }

    // Fall back to legacy approach
    table_to_rust_legacy(obj)
}

/// Legacy Table import using `to_batches()`.
/// 
/// For old PyArrow versions where __arrow_c_stream__ is unsupported
fn table_to_rust_legacy(obj: &Bound<PyAny>) -> PyMinarrowResult<SuperTable> {
    // Try to recover the table name from schema metadata
    let table_name = obj
        .getattr("schema")
        .ok()
        .map(|s| extract_table_name_from_pyarrow_schema(&s))
        .unwrap_or_default();

    let batches = obj
        .call_method0("to_batches")
        .map_err(|e| {
            PyMinarrowError::PyArrow(format!("Failed to get batches from Table: {}", e))
        })?;

    let batches_list: Vec<Bound<PyAny>> = batches
        .extract()
        .map_err(|e| {
            PyMinarrowError::PyArrow(format!("Failed to extract batches list: {}", e))
        })?;

    if batches_list.is_empty() {
        return Ok(SuperTable::new(table_name));
    }

    let mut tables = Vec::with_capacity(batches_list.len());
    for batch in batches_list {
        let table = record_batch_to_rust(&batch)?;
        tables.push(Arc::new(table));
    }

    Ok(SuperTable::from_batches(tables, None))
}

/// Converts a PyArrow ChunkedArray (or compatible object) to a MinArrow SuperArray.
///
/// Tries `__arrow_c_stream__` first (yields plain arrays, one per chunk),
/// then falls back to the legacy `.chunks` approach.
///
/// # Arguments
/// * `obj` - A Python ChunkedArray or compatible object
///
/// # Returns
/// * `PyMinarrowResult<SuperArray>` - The converted MinArrow SuperArray
pub fn chunked_array_to_rust(obj: &Bound<PyAny>) -> PyMinarrowResult<SuperArray> {
    // Try PyCapsule stream
    if let Some(result) = try_capsule_array_stream(obj) {
        let (arrays, field) = result?;
        if arrays.is_empty() {
            return Ok(SuperArray::new());
        }

        let field_arrays: Vec<FieldArray> = arrays
            .into_iter()
            .map(|array| FieldArray::new(field.clone(), (*array).clone()))
            .collect();

        return Ok(SuperArray::from_field_array_chunks(field_arrays));
    }

    // Fall back to legacy approach
    chunked_array_to_rust_legacy(obj)
}

/// Legacy ChunkedArray import using `.chunks`.
fn chunked_array_to_rust_legacy(obj: &Bound<PyAny>) -> PyMinarrowResult<SuperArray> {
    let chunks = obj
        .getattr("chunks")
        .map_err(|e| PyMinarrowError::PyArrow(format!("Failed to get chunks: {}", e)))?;

    let chunks_list: Vec<Bound<PyAny>> = chunks
        .extract()
        .map_err(|e| {
            PyMinarrowError::PyArrow(format!("Failed to extract chunks list: {}", e))
        })?;

    if chunks_list.is_empty() {
        return Ok(SuperArray::new());
    }

    let first_fa = array_to_rust(&chunks_list[0])?;
    let field = first_fa.field.clone();

    let mut field_arrays = Vec::with_capacity(chunks_list.len());
    field_arrays.push(first_fa);

    for chunk in chunks_list.iter().skip(1) {
        let chunk_fa = array_to_rust(chunk)?;
        let field_array = FieldArray::new((*field).clone(), chunk_fa.array);
        field_arrays.push(field_array);
    }

    Ok(SuperArray::from_field_array_chunks(field_arrays))
}
