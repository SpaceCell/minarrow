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

//! # MinArrow to Python Conversion
//!
//! Converts MinArrow arrays to PyArrow arrays and PyCapsules using the
//! Arrow C Data Interface and Arrow PyCapsule Interface.
//!
//! ## Export Strategy
//! Two export paths are provided:
//! 1. **PyArrow objects** - convert to PyArrow arrays/tables via `_import_from_c`
//! 2. **PyCapsules** - export as Arrow PyCapsule objects for direct consumption
//!    by any Python library supporting the Arrow PyCapsule Interface

use minarrow::ffi::arrow_c_ffi::{
    export_array_stream, export_record_batch_stream_with_metadata,
    export_record_batch_view_stream, export_super_array_view_stream,
    export_super_table_view_stream, export_to_c, export_view_to_c, ArrowArray, ArrowArrayStream,
    ArrowSchema,
};
use minarrow::ffi::arrow_dtype::{ArrowType, CategoricalIndexType};
#[cfg(feature = "datetime")]
use minarrow::enums::time_units::TimeUnit;
use minarrow::ffi::schema::Schema;
use minarrow::{
    Array, ArrayV, Field, SuperArray, SuperArrayV, SuperTable, SuperTableV, Table, TableV,
};
use pyo3::ffi::Py_uintptr_t;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyList};
use std::sync::Arc;

use crate::error::PyMinarrowError;

/// Key used to store the MinArrow table name in Arrow schema metadata.
pub(crate) const TABLE_NAME_KEY: &str = "minarrow:table_name";

/// Converts a MinArrow TimeUnit to the PyArrow unit string.
#[cfg(feature = "datetime")]
fn time_unit_to_str(unit: &TimeUnit) -> &'static str {
    match unit {
        TimeUnit::Seconds => "s",
        TimeUnit::Milliseconds => "ms",
        TimeUnit::Microseconds => "us",
        TimeUnit::Nanoseconds => "ns",
        TimeUnit::Days => "s", // Days is not a PyArrow unit; fall back to seconds
    }
}

/// Converts an ArrowType to the corresponding PyArrow DataType object.
///
/// This allows building PyArrow schemas from field metadata without
/// needing to create actual zero-length arrays.
fn arrow_type_to_pyarrow<'py>(
    dtype: &ArrowType,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let pa = py.import("pyarrow")?;
    match dtype {
        ArrowType::Null => pa.call_method0("null"),
        ArrowType::Boolean => pa.call_method0("bool_"),

        #[cfg(feature = "extended_numeric_types")]
        ArrowType::Int8 => pa.call_method0("int8"),
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::Int16 => pa.call_method0("int16"),
        ArrowType::Int32 => pa.call_method0("int32"),
        ArrowType::Int64 => pa.call_method0("int64"),
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::UInt8 => pa.call_method0("uint8"),
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::UInt16 => pa.call_method0("uint16"),
        ArrowType::UInt32 => pa.call_method0("uint32"),
        ArrowType::UInt64 => pa.call_method0("uint64"),

        ArrowType::Float32 => pa.call_method0("float32"),
        ArrowType::Float64 => pa.call_method0("float64"),

        ArrowType::String => pa.call_method0("utf8"),
        ArrowType::LargeString => pa.call_method0("large_utf8"),
        // Utf8View data is stored as regular Utf8 after import
        ArrowType::Utf8View => pa.call_method0("utf8"),

        #[cfg(feature = "datetime")]
        ArrowType::Date32 => pa.call_method0("date32"),
        #[cfg(feature = "datetime")]
        ArrowType::Date64 => pa.call_method0("date64"),

        #[cfg(feature = "datetime")]
        ArrowType::Time32(unit) => {
            let unit_str = time_unit_to_str(unit);
            pa.call_method1("time32", (unit_str,))
        }
        #[cfg(feature = "datetime")]
        ArrowType::Time64(unit) => {
            let unit_str = time_unit_to_str(unit);
            pa.call_method1("time64", (unit_str,))
        }
        #[cfg(feature = "datetime")]
        ArrowType::Duration32(unit) => {
            let unit_str = time_unit_to_str(unit);
            pa.call_method1("duration", (unit_str,))
        }
        #[cfg(feature = "datetime")]
        ArrowType::Duration64(unit) => {
            let unit_str = time_unit_to_str(unit);
            pa.call_method1("duration", (unit_str,))
        }
        #[cfg(feature = "datetime")]
        ArrowType::Timestamp(unit, tz) => {
            let unit_str = time_unit_to_str(unit);
            let tz_str = tz.as_deref().unwrap_or("");
            pa.call_method1("timestamp", (unit_str, tz_str))
        }
        #[cfg(feature = "datetime")]
        ArrowType::Interval(_) => {
            // PyArrow doesn't have a direct interval type constructor — fall back to null
            pa.call_method0("null")
        }

        ArrowType::Dictionary(key_type) => {
            let index_ty = match key_type {
                #[cfg(all(feature = "extended_categorical", feature = "extended_numeric_types"))]
                CategoricalIndexType::UInt8 => pa.call_method0("uint8")?,
                #[cfg(all(feature = "extended_categorical", feature = "extended_numeric_types"))]
                CategoricalIndexType::UInt16 => pa.call_method0("uint16")?,
                CategoricalIndexType::UInt32 => pa.call_method0("uint32")?,
                #[cfg(feature = "extended_categorical")]
                CategoricalIndexType::UInt64 => pa.call_method0("uint64")?,
            };
            let value_ty = pa.call_method0("utf8")?;
            pa.call_method1("dictionary", (index_ty, value_ty))
        }
    }
}

/// Builds the Arrow schema metadata map for a stream export.
///
/// Always includes the table name when non-empty. When the `table_metadata`
/// feature is enabled, the table's metadata entries are included too.
fn build_stream_metadata(table: &Table) -> Option<std::collections::BTreeMap<String, String>> {
    #[cfg(feature = "table_metadata")]
    let mut m = table.metadata.clone();
    #[cfg(not(feature = "table_metadata"))]
    let mut m = std::collections::BTreeMap::new();

    if !table.name.is_empty() {
        m.insert(TABLE_NAME_KEY.to_string(), table.name.clone());
    }
    if m.is_empty() { None } else { Some(m) }
}

// PyArrow conversion - legacy C data interface

/// Converts a MinArrow Array to a PyArrow Array.
///
/// Uses the Arrow C Data Interface for zero-copy conversion where possible.
///
/// # Arguments
/// * `array` - The MinArrow array to convert (wrapped in Arc)
/// * `field` - Field metadata for the array
/// * `py` - Python interpreter handle
///
/// # Returns
/// * `PyResult<Bound<'py, PyAny>>` - The PyArrow Array
pub fn array_to_py<'py>(
    array: Arc<Array>,
    field: &Field,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let pyarrow = py.import("pyarrow")?;

    // Build schema from field
    let schema = Schema::from(vec![field.clone()]);

    // Export to Arrow C format (heap-allocates ArrowArray + ArrowSchema via Box)
    let (array_ptr, schema_ptr) = export_to_c(array, schema);

    // Import into PyArrow via _import_from_c.
    // Arrow C++ moves struct contents and sets release=NULL on the originals.
    let result = pyarrow
        .getattr("Array")?
        .call_method1(
            "_import_from_c",
            (array_ptr as Py_uintptr_t, schema_ptr as Py_uintptr_t),
        )
        .map_err(|e| {
            PyMinarrowError::PyArrow(format!("Failed to import array into PyArrow: {}", e))
        });

    // Free the Box allocations for ArrowSchema and ArrowArray.
    // On success: Arrow C++ moved contents and set release=NULL — just free the Boxes.
    // On failure: release callbacks are still set — call them to clean up, then free.
    unsafe {
        if let Some(release) = (*schema_ptr).release {
            release(schema_ptr);
        }
        let _ = Box::from_raw(schema_ptr);
        if let Some(release) = (*array_ptr).release {
            release(array_ptr);
        }
        let _ = Box::from_raw(array_ptr);
    }

    result.map_err(|e| e.into())
}

/// Converts a MinArrow Table to a PyArrow RecordBatch.
///
/// Converts each column to a PyArrow array and assembles them into a RecordBatch.
/// If the table has a non-empty name, it is stored in the PyArrow schema metadata
/// under the `minarrow:table_name` key so it can be recovered on import.
pub fn table_to_py<'py>(table: &Table, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let pyarrow = py.import("pyarrow")?;

    let mut py_fields = Vec::with_capacity(table.n_cols());
    let mut py_arrays = Vec::with_capacity(table.n_cols());

    for fa in &table.cols {
        let array = Arc::new(fa.array.clone());
        let py_array = array_to_py(array, &fa.field, py)?;
        // Extract the PyArrow field from the array's type to preserve type metadata
        let py_field = pyarrow.call_method1(
            "field",
            (fa.field.name.clone(), py_array.getattr("type")?),
        )?;
        py_fields.push(py_field);
        py_arrays.push(py_array);
    }

    let py_fields_list = PyList::new(py, &py_fields)?;

    // Build schema, attaching table name and metadata if present
    let mut schema = pyarrow.call_method1("schema", (py_fields_list,))?;
    #[cfg(feature = "table_metadata")]
    {
        let mut meta_entries: Vec<(String, String)> = table
            .metadata
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        if !table.name.is_empty() {
            meta_entries.push((TABLE_NAME_KEY.to_string(), table.name.clone()));
        }
        if !meta_entries.is_empty() {
            let metadata = meta_entries.into_py_dict(py)?;
            schema = schema.call_method1("with_metadata", (metadata,))?;
        }
    }
    #[cfg(not(feature = "table_metadata"))]
    if !table.name.is_empty() {
        let metadata = [(TABLE_NAME_KEY, &table.name)].into_py_dict(py)?;
        schema = schema.call_method1("with_metadata", (metadata,))?;
    }

    let py_arrays_list = PyList::new(py, py_arrays)?;

    let kwargs = [("schema", schema)].into_py_dict(py)?;
    pyarrow
        .getattr("RecordBatch")?
        .call_method("from_arrays", (py_arrays_list,), Some(&kwargs))
        .map_err(|e| {
            PyMinarrowError::PyArrow(format!("Failed to create PyArrow RecordBatch: {}", e)).into()
        })
}

/// Converts a MinArrow SuperTable to a PyArrow Table.
pub fn super_table_to_py<'py>(
    super_table: &SuperTable,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let pyarrow = py.import("pyarrow")?;

    if super_table.batches.is_empty() {
        // Build a PyArrow schema from the field definitions and create
        // an empty Table directly. This avoids constructing dummy arrays.
        let mut py_fields = Vec::with_capacity(super_table.schema.len());
        for f in &super_table.schema {
            let pa_type = arrow_type_to_pyarrow(&f.dtype, py)?;
            let pa_field = pyarrow.call_method1("field", (&f.name, pa_type))?;
            py_fields.push(pa_field);
        }
        let py_fields_list = PyList::new(py, &py_fields)?;
        let mut schema = pyarrow.call_method1("schema", (py_fields_list,))?;
        #[cfg(feature = "table_metadata")]
        {
            let mut meta_entries: Vec<(String, String)> = super_table
                .metadata()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            if !super_table.name.is_empty() {
                meta_entries.push((TABLE_NAME_KEY.to_string(), super_table.name.clone()));
            }
            if !meta_entries.is_empty() {
                let metadata = meta_entries.into_py_dict(py)?;
                schema = schema.call_method1("with_metadata", (metadata,))?;
            }
        }
        #[cfg(not(feature = "table_metadata"))]
        if !super_table.name.is_empty() {
            let metadata = [(TABLE_NAME_KEY, &super_table.name)].into_py_dict(py)?;
            schema = schema.call_method1("with_metadata", (metadata,))?;
        }
        let empty_list = PyList::empty(py);
        let kwargs = [("schema", schema)].into_py_dict(py)?;
        return pyarrow
            .getattr("Table")?
            .call_method("from_batches", (empty_list,), Some(&kwargs))
            .map_err(|e| {
                PyMinarrowError::PyArrow(format!(
                    "Failed to create empty PyArrow Table: {}",
                    e
                ))
                .into()
            });
    }

    let mut py_batches = Vec::with_capacity(super_table.batches.len());
    for batch in &super_table.batches {
        let py_batch = table_to_py(batch, py)?;
        py_batches.push(py_batch);
    }

    let py_batches_list = PyList::new(py, py_batches)?;

    let py_table = pyarrow
        .getattr("Table")?
        .call_method1("from_batches", (py_batches_list,))
        .map_err(|e| {
            PyMinarrowError::PyArrow(format!("Failed to create PyArrow Table: {}", e))
        })?;

    // Attach table name and metadata as schema metadata if present
    #[cfg(feature = "table_metadata")]
    {
        let mut meta_entries: Vec<(String, String)> = super_table
            .metadata()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        if !super_table.name.is_empty() {
            meta_entries.push((TABLE_NAME_KEY.to_string(), super_table.name.clone()));
        }
        if !meta_entries.is_empty() {
            let metadata = meta_entries.into_py_dict(py)?;
            return py_table
                .call_method1("replace_schema_metadata", (metadata,))
                .map_err(|e| {
                    PyMinarrowError::PyArrow(format!("Failed to set schema metadata: {}", e))
                        .into()
                });
        }
    }
    #[cfg(not(feature = "table_metadata"))]
    if !super_table.name.is_empty() {
        let metadata = [(TABLE_NAME_KEY, &super_table.name)]
            .into_py_dict(py)?;
        return py_table
            .call_method1("replace_schema_metadata", (metadata,))
            .map_err(|e| {
                PyMinarrowError::PyArrow(format!("Failed to set schema metadata: {}", e)).into()
            });
    }

    Ok(py_table)
}

/// Converts a MinArrow SuperArray to a PyArrow ChunkedArray.
pub fn super_array_to_py<'py>(
    super_array: &SuperArray,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let pyarrow = py.import("pyarrow")?;

    let chunks = super_array.chunks();
    if chunks.is_empty() {
        // Build an empty ChunkedArray with the correct type from field metadata.
        let pa_type = if let Some(field) = super_array.field() {
            arrow_type_to_pyarrow(&field.dtype, py)?
        } else {
            // No field metadata — fall back to null type
            pyarrow.call_method0("null")?
        };
        let empty_list = PyList::empty(py);
        let kwargs = [("type", pa_type)].into_py_dict(py)?;
        return pyarrow
            .call_method("chunked_array", (empty_list,), Some(&kwargs))
            .map_err(|e| {
                PyMinarrowError::PyArrow(format!(
                    "Failed to create empty PyArrow ChunkedArray: {}",
                    e
                ))
                .into()
            });
    }

    let field = super_array.field_ref();
    let mut py_arrays = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let array = Arc::new(chunk.clone());
        let py_array = array_to_py(array, field, py)?;
        py_arrays.push(py_array);
    }

    let py_arrays_list = PyList::new(py, py_arrays)?;

    pyarrow
        .call_method1("chunked_array", (py_arrays_list,))
        .map_err(|e| {
            PyMinarrowError::PyArrow(format!("Failed to create PyArrow ChunkedArray: {}", e))
                .into()
        })
}

// PyCapsule export

/// Capsule destructor for ArrowSchema.
/// Called when the PyCapsule is garbage collected without being consumed.
unsafe extern "C" fn arrow_schema_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    unsafe {
        let name = c"arrow_schema";
        let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, name.as_ptr()) as *mut ArrowSchema;
        if !ptr.is_null() {
            let schema = &mut *ptr;
            if let Some(release) = schema.release {
                release(schema);
            }
            let _ = Box::from_raw(ptr);
        }
    }
}

/// Capsule destructor for ArrowArray.
unsafe extern "C" fn arrow_array_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    unsafe {
        let name = c"arrow_array";
        let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, name.as_ptr()) as *mut ArrowArray;
        if !ptr.is_null() {
            let array = &mut *ptr;
            if let Some(release) = array.release {
                release(array);
            }
            let _ = Box::from_raw(ptr);
        }
    }
}

/// Capsule destructor for ArrowArrayStream.
unsafe extern "C" fn arrow_stream_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    unsafe {
        let name = c"arrow_array_stream";
        let ptr =
            pyo3::ffi::PyCapsule_GetPointer(capsule, name.as_ptr()) as *mut ArrowArrayStream;
        if !ptr.is_null() {
            let stream = &mut *ptr;
            if let Some(release) = stream.release {
                release(stream);
            }
            let _ = Box::from_raw(ptr);
        }
    }
}

/// Exports a MinArrow array as a pair of PyCapsules (schema, array).
///
/// Returns `(schema_capsule, array_capsule)` following the Arrow PyCapsule Interface.
/// The capsules have destructors that call the Arrow release callbacks if the
/// capsules are not consumed by a recipient.
pub fn array_to_capsules<'py>(
    array: Arc<Array>,
    field: &Field,
    py: Python<'py>,
) -> PyResult<(PyObject, PyObject)> {
    let schema = Schema::from(vec![field.clone()]);
    let (arr_ptr, sch_ptr) = export_to_c(array, schema);

    // Create schema capsule
    let schema_name = c"arrow_schema";
    let schema_capsule = unsafe {
        let cap = pyo3::ffi::PyCapsule_New(
            sch_ptr as *mut std::ffi::c_void,
            schema_name.as_ptr(),
            Some(arrow_schema_capsule_destructor),
        );
        if cap.is_null() {
            // Clean up on failure
            let s = &mut *sch_ptr;
            if let Some(release) = s.release {
                release(sch_ptr);
            }
            let _ = Box::from_raw(sch_ptr);
            let a = &mut *arr_ptr;
            if let Some(release) = a.release {
                release(arr_ptr);
            }
            let _ = Box::from_raw(arr_ptr);
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Failed to create schema PyCapsule",
            ));
        }
        Bound::from_owned_ptr(py, cap)
    };

    // Create array capsule
    let array_name = c"arrow_array";
    let array_capsule = unsafe {
        let cap = pyo3::ffi::PyCapsule_New(
            arr_ptr as *mut std::ffi::c_void,
            array_name.as_ptr(),
            Some(arrow_array_capsule_destructor),
        );
        if cap.is_null() {
            let a = &mut *arr_ptr;
            if let Some(release) = a.release {
                release(arr_ptr);
            }
            let _ = Box::from_raw(arr_ptr);
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Failed to create array PyCapsule",
            ));
        }
        Bound::from_owned_ptr(py, cap)
    };

    Ok((schema_capsule.unbind(), array_capsule.unbind()))
}

/// Exports a MinArrow Table as an ArrowArrayStream PyCapsule.
///
/// The stream yields one struct array (record batch) corresponding to the table.
pub fn table_to_stream_capsule<'py>(table: &Table, py: Python<'py>) -> PyResult<PyObject> {
    let fields: Vec<Field> = table.cols.iter().map(|fa| (*fa.field).clone()).collect();
    let columns: Vec<(Arc<Array>, Schema)> = table
        .cols
        .iter()
        .map(|fa| {
            (
                Arc::new(fa.array.clone()),
                Schema::from(vec![(*fa.field).clone()]),
            )
        })
        .collect();

    let metadata = build_stream_metadata(table);
    let stream = export_record_batch_stream_with_metadata(vec![columns], fields, metadata);
    let stream_ptr = Box::into_raw(stream);

    let name = c"arrow_array_stream";
    let capsule = unsafe {
        let cap = pyo3::ffi::PyCapsule_New(
            stream_ptr as *mut std::ffi::c_void,
            name.as_ptr(),
            Some(arrow_stream_capsule_destructor),
        );
        if cap.is_null() {
            // Clean up
            let s = &mut *stream_ptr;
            if let Some(release) = s.release {
                release(stream_ptr);
            }
            let _ = Box::from_raw(stream_ptr);
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Failed to create stream PyCapsule",
            ));
        }
        Bound::from_owned_ptr(py, cap)
    };

    Ok(capsule.unbind())
}

/// Exports a MinArrow SuperTable as an ArrowArrayStream PyCapsule.
///
/// The stream yields one struct array per batch in the SuperTable.
pub fn super_table_to_stream_capsule<'py>(
    super_table: &SuperTable,
    py: Python<'py>,
) -> PyResult<PyObject> {
    if super_table.batches.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot export empty SuperTable as stream capsule",
        ));
    }

    // Extract fields from the first batch
    let fields: Vec<Field> = super_table.batches[0]
        .cols
        .iter()
        .map(|fa| (*fa.field).clone())
        .collect();

    // Convert each batch to column (Arc<Array>, Schema) pairs
    let batches: Vec<Vec<(Arc<Array>, Schema)>> = super_table
        .batches
        .iter()
        .map(|table| {
            table
                .cols
                .iter()
                .map(|fa| {
                    (
                        Arc::new(fa.array.clone()),
                        Schema::from(vec![(*fa.field).clone()]),
                    )
                })
                .collect()
        })
        .collect();

    let metadata = build_stream_metadata(&super_table.batches[0]);
    let stream = export_record_batch_stream_with_metadata(batches, fields, metadata);
    let stream_ptr = Box::into_raw(stream);

    let name = c"arrow_array_stream";
    let capsule = unsafe {
        let cap = pyo3::ffi::PyCapsule_New(
            stream_ptr as *mut std::ffi::c_void,
            name.as_ptr(),
            Some(arrow_stream_capsule_destructor),
        );
        if cap.is_null() {
            let s = &mut *stream_ptr;
            if let Some(release) = s.release {
                release(stream_ptr);
            }
            let _ = Box::from_raw(stream_ptr);
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Failed to create stream PyCapsule",
            ));
        }
        Bound::from_owned_ptr(py, cap)
    };

    Ok(capsule.unbind())
}

// ── Zero-Copy View export ─────────────────────────
//
// Mirror of the owned export helpers above, but each entry point keeps the
// caller's view window as `(offset, len)` and lets Arrow C carry it
// through `ArrowArray.offset`/`ArrowArray.length`.

/// Converts a MinArrow [`ArrayV`] to a PyArrow Array via the Arrow C Data
/// Interface, preserving the window through `ArrowArray.offset`.
///
/// The buffers underlying the view are shared with PyArrow (zero-copy);
/// PyArrow constructs its `pa.Array` metadata over those same pointers.
pub fn array_view_to_py<'py>(
    view: &ArrayV,
    field: &Field,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let pyarrow = py.import("pyarrow")?;

    let schema = Schema::from(vec![field.clone()]);
    let array = Arc::new(view.array.clone());
    let (arr_ptr, sch_ptr) =
        export_view_to_c(array, schema, view.offset as i64, view.len() as i64);

    let result = pyarrow
        .getattr("Array")?
        .call_method1(
            "_import_from_c",
            (arr_ptr as Py_uintptr_t, sch_ptr as Py_uintptr_t),
        )
        .map_err(|e| {
            PyMinarrowError::PyArrow(format!("Failed to import view into PyArrow: {}", e))
        });

    // Same cleanup pattern as `array_to_py`.
    unsafe {
        if let Some(release) = (*sch_ptr).release {
            release(sch_ptr);
        }
        let _ = Box::from_raw(sch_ptr);
        if let Some(release) = (*arr_ptr).release {
            release(arr_ptr);
        }
        let _ = Box::from_raw(arr_ptr);
    }

    result.map_err(|e| e.into())
}

/// Converts a MinArrow [`TableV`] to a PyArrow RecordBatch, with each
/// column transferred zero-copy via [`array_view_to_py`].
pub fn table_view_to_py<'py>(
    view: &TableV,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let pyarrow = py.import("pyarrow")?;

    let mut py_fields = Vec::with_capacity(view.cols.len());
    let mut py_arrays = Vec::with_capacity(view.cols.len());

    for (field, col) in view.fields.iter().zip(view.cols.iter()) {
        let py_array = array_view_to_py(col, field, py)?;
        let py_field = pyarrow.call_method1(
            "field",
            (field.name.clone(), py_array.getattr("type")?),
        )?;
        py_fields.push(py_field);
        py_arrays.push(py_array);
    }

    let py_fields_list = PyList::new(py, &py_fields)?;

    let mut schema = pyarrow.call_method1("schema", (py_fields_list,))?;
    if !view.name.is_empty() {
        let metadata = [(TABLE_NAME_KEY, &view.name)].into_py_dict(py)?;
        schema = schema.call_method1("with_metadata", (metadata,))?;
    }

    let py_arrays_list = PyList::new(py, py_arrays)?;
    let kwargs = [("schema", schema)].into_py_dict(py)?;
    pyarrow
        .getattr("RecordBatch")?
        .call_method("from_arrays", (py_arrays_list,), Some(&kwargs))
        .map_err(|e| {
            PyMinarrowError::PyArrow(format!(
                "Failed to create PyArrow RecordBatch from TableV: {}",
                e
            ))
            .into()
        })
}

/// Converts a MinArrow [`SuperTableV`] to a PyArrow Table by assembling
/// one RecordBatch per [`TableV`] slice. Per-column window offsets are
/// preserved zero-copy.
pub fn super_table_view_to_py<'py>(
    view: &SuperTableV,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let pyarrow = py.import("pyarrow")?;

    if view.slices.is_empty() {
        return Err(PyMinarrowError::PyArrow(
            "Cannot convert empty SuperTableV to PyArrow Table".into(),
        )
        .into());
    }

    let mut py_batches = Vec::with_capacity(view.slices.len());
    for batch in &view.slices {
        py_batches.push(table_view_to_py(batch, py)?);
    }

    let py_batches_list = PyList::new(py, py_batches)?;
    pyarrow
        .getattr("Table")?
        .call_method1("from_batches", (py_batches_list,))
        .map_err(|e| {
            PyMinarrowError::PyArrow(format!(
                "Failed to create PyArrow Table from SuperTableV: {}",
                e
            ))
            .into()
        })
}

/// Converts a MinArrow [`SuperArrayV`] to a PyArrow ChunkedArray by
/// assembling one Array per [`ArrayV`] slice. Per-slice window offsets
/// are preserved zero-copy.
pub fn super_array_view_to_py<'py>(
    view: &SuperArrayV,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let pyarrow = py.import("pyarrow")?;
    let field = &*view.field;

    if view.slices.is_empty() {
        let pa_type = arrow_type_to_pyarrow(&field.dtype, py)?;
        let empty_list = PyList::empty(py);
        let kwargs = [("type", pa_type)].into_py_dict(py)?;
        return pyarrow
            .call_method("chunked_array", (empty_list,), Some(&kwargs))
            .map_err(|e| {
                PyMinarrowError::PyArrow(format!(
                    "Failed to create empty PyArrow ChunkedArray from SuperArrayV: {}",
                    e
                ))
                .into()
            });
    }

    let mut py_arrays = Vec::with_capacity(view.slices.len());
    for slice in &view.slices {
        py_arrays.push(array_view_to_py(slice, field, py)?);
    }

    let py_arrays_list = PyList::new(py, py_arrays)?;
    pyarrow
        .call_method1("chunked_array", (py_arrays_list,))
        .map_err(|e| {
            PyMinarrowError::PyArrow(format!(
                "Failed to create PyArrow ChunkedArray from SuperArrayV: {}",
                e
            ))
            .into()
        })
}

/// Exports a MinArrow [`ArrayV`] as a pair of PyCapsules (schema, array).
///
/// Uses `ArrowArray.offset` to convey the window, so no primary buffer copies
/// occur for primitive, string, datetime, boolean, or categorical-codes data.
pub fn array_view_to_capsules<'py>(
    view: &ArrayV,
    field: &Field,
    py: Python<'py>,
) -> PyResult<(PyObject, PyObject)> {
    let schema = Schema::from(vec![field.clone()]);
    let array = Arc::new(view.array.clone());
    let (arr_ptr, sch_ptr) =
        export_view_to_c(array, schema, view.offset as i64, view.len() as i64);

    let schema_name = c"arrow_schema";
    let schema_capsule = unsafe {
        let cap = pyo3::ffi::PyCapsule_New(
            sch_ptr as *mut std::ffi::c_void,
            schema_name.as_ptr(),
            Some(arrow_schema_capsule_destructor),
        );
        if cap.is_null() {
            let s = &mut *sch_ptr;
            if let Some(release) = s.release {
                release(sch_ptr);
            }
            let _ = Box::from_raw(sch_ptr);
            let a = &mut *arr_ptr;
            if let Some(release) = a.release {
                release(arr_ptr);
            }
            let _ = Box::from_raw(arr_ptr);
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Failed to create schema PyCapsule",
            ));
        }
        Bound::from_owned_ptr(py, cap)
    };

    let array_name = c"arrow_array";
    let array_capsule = unsafe {
        let cap = pyo3::ffi::PyCapsule_New(
            arr_ptr as *mut std::ffi::c_void,
            array_name.as_ptr(),
            Some(arrow_array_capsule_destructor),
        );
        if cap.is_null() {
            let a = &mut *arr_ptr;
            if let Some(release) = a.release {
                release(arr_ptr);
            }
            let _ = Box::from_raw(arr_ptr);
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Failed to create array PyCapsule",
            ));
        }
        Bound::from_owned_ptr(py, cap)
    };

    Ok((schema_capsule.unbind(), array_capsule.unbind()))
}

/// Exports a MinArrow [`TableV`] as an `ArrowArrayStream` PyCapsule.
///
/// The stream yields one struct array, with each column carrying its
/// windowed `(offset, len)` at the Arrow C layer.
pub fn table_view_to_stream_capsule<'py>(
    view: &TableV,
    py: Python<'py>,
) -> PyResult<PyObject> {
    let fields: Vec<Field> = view.fields.iter().map(|f| (**f).clone()).collect();

    let metadata = if view.name.is_empty() {
        None
    } else {
        let mut m = std::collections::BTreeMap::new();
        m.insert(TABLE_NAME_KEY.to_string(), view.name.clone());
        Some(m)
    };

    let stream = export_record_batch_view_stream(vec![view.clone()], fields, metadata);
    let stream_ptr = Box::into_raw(stream);

    let name = c"arrow_array_stream";
    let capsule = unsafe {
        let cap = pyo3::ffi::PyCapsule_New(
            stream_ptr as *mut std::ffi::c_void,
            name.as_ptr(),
            Some(arrow_stream_capsule_destructor),
        );
        if cap.is_null() {
            let s = &mut *stream_ptr;
            if let Some(release) = s.release {
                release(stream_ptr);
            }
            let _ = Box::from_raw(stream_ptr);
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Failed to create stream PyCapsule",
            ));
        }
        Bound::from_owned_ptr(py, cap)
    };

    Ok(capsule.unbind())
}

/// Exports a MinArrow [`SuperTableV`] as an `ArrowArrayStream` PyCapsule.
///
/// The stream yields one struct array per underlying [`TableV`] batch,
/// preserving per-column window offsets.
pub fn super_table_view_to_stream_capsule<'py>(
    view: &SuperTableV,
    py: Python<'py>,
) -> PyResult<PyObject> {
    if view.slices.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot export empty SuperTableV as stream capsule",
        ));
    }

    let first_name = &view.slices[0].name;
    let metadata = if first_name.is_empty() {
        None
    } else {
        let mut m = std::collections::BTreeMap::new();
        m.insert(TABLE_NAME_KEY.to_string(), first_name.clone());
        Some(m)
    };

    let stream = export_super_table_view_stream(view, metadata);
    let stream_ptr = Box::into_raw(stream);

    let name = c"arrow_array_stream";
    let capsule = unsafe {
        let cap = pyo3::ffi::PyCapsule_New(
            stream_ptr as *mut std::ffi::c_void,
            name.as_ptr(),
            Some(arrow_stream_capsule_destructor),
        );
        if cap.is_null() {
            let s = &mut *stream_ptr;
            if let Some(release) = s.release {
                release(stream_ptr);
            }
            let _ = Box::from_raw(stream_ptr);
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Failed to create stream PyCapsule",
            ));
        }
        Bound::from_owned_ptr(py, cap)
    };

    Ok(capsule.unbind())
}

/// Exports a MinArrow [`SuperArrayV`] as an `ArrowArrayStream` PyCapsule.
///
/// The stream yields one plain array per underlying [`ArrayV`] slice. Each
/// slice's window is carried zero-copy via `ArrowArray.offset`/`length`.
pub fn super_array_view_to_stream_capsule<'py>(
    view: &SuperArrayV,
    py: Python<'py>,
) -> PyResult<PyObject> {
    if view.slices.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot export empty SuperArrayV as stream capsule",
        ));
    }

    let stream = export_super_array_view_stream(view);
    let stream_ptr = Box::into_raw(stream);

    let name = c"arrow_array_stream";
    let capsule = unsafe {
        let cap = pyo3::ffi::PyCapsule_New(
            stream_ptr as *mut std::ffi::c_void,
            name.as_ptr(),
            Some(arrow_stream_capsule_destructor),
        );
        if cap.is_null() {
            let s = &mut *stream_ptr;
            if let Some(release) = s.release {
                release(stream_ptr);
            }
            let _ = Box::from_raw(stream_ptr);
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Failed to create stream PyCapsule",
            ));
        }
        Bound::from_owned_ptr(py, cap)
    };

    Ok(capsule.unbind())
}

/// Exports a MinArrow SuperArray as an ArrowArrayStream PyCapsule.
///
/// The stream yields one plain array per chunk.
pub fn super_array_to_stream_capsule<'py>(
    super_array: &SuperArray,
    py: Python<'py>,
) -> PyResult<PyObject> {
    let chunks = super_array.chunks();
    if chunks.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot export empty SuperArray as stream capsule",
        ));
    }

    let field = super_array.field_ref().clone();
    let array_chunks: Vec<Arc<Array>> = chunks.iter().map(|c| Arc::new(c.clone())).collect();

    let stream = export_array_stream(array_chunks, field);
    let stream_ptr = Box::into_raw(stream);

    let name = c"arrow_array_stream";
    let capsule = unsafe {
        let cap = pyo3::ffi::PyCapsule_New(
            stream_ptr as *mut std::ffi::c_void,
            name.as_ptr(),
            Some(arrow_stream_capsule_destructor),
        );
        if cap.is_null() {
            let s = &mut *stream_ptr;
            if let Some(release) = s.release {
                release(stream_ptr);
            }
            let _ = Box::from_raw(stream_ptr);
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Failed to create stream PyCapsule",
            ));
        }
        Bound::from_owned_ptr(py, cap)
    };

    Ok(capsule.unbind())
}
