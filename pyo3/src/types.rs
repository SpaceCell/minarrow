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

//! # Type Wrappers for minarrow-pyo3
//!
//! Provides transparent wrapper types around MinArrow types that implement
//! PyO3 conversion traits for seamless Python interoperability.

use minarrow::{
    Array, ArrayV, Field, FieldArray, SuperArray, SuperArrayV, SuperTable, SuperTableV, Table,
    TableV,
};
use pyo3::prelude::*;
use std::sync::Arc;

use crate::ffi::{to_py, to_rust};

// PyArray - Wrapper around MinArrow's FieldArray

/// Transparent wrapper around MinArrow's FieldArray.
///
/// Enables zero-copy conversion to/from PyArrow arrays via the Arrow C Data Interface.
/// Preserves exact Arrow type metadata (e.g., Timestamp vs Date64) through the conversion.
///
/// # Example (Rust)
/// ```ignore
/// use minarrow_pyo3::PyArray;
/// use minarrow::FieldArray;
///
/// #[pyfunction]
/// fn process_array(arr: PyArray) -> PyResult<PyArray> {
///     let field_array: FieldArray = arr.into();
///     // Process...
///     Ok(PyArray::from(field_array))
/// }
/// ```
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct PyArray(pub FieldArray);

impl PyArray {
    /// Creates a new PyArray from a FieldArray.
    pub fn new(field_array: FieldArray) -> Self {
        Self(field_array)
    }

    /// Returns a reference to the inner MinArrow Array.
    pub fn inner(&self) -> &Array {
        &self.0.array
    }

    /// Returns a reference to the inner MinArrow FieldArray.
    pub fn field_array(&self) -> &FieldArray {
        &self.0
    }

    /// Returns a reference to the Field metadata.
    pub fn field(&self) -> &Field {
        &self.0.field
    }

    /// Consumes self and returns the inner FieldArray.
    pub fn into_inner(self) -> FieldArray {
        self.0
    }
}

impl From<FieldArray> for PyArray {
    fn from(field_array: FieldArray) -> Self {
        Self(field_array)
    }
}

impl From<Arc<Array>> for PyArray {
    fn from(array: Arc<Array>) -> Self {
        let field = Field::from_array("", &array, None);
        Self(FieldArray::new(field, (*array).clone()))
    }
}

impl From<Array> for PyArray {
    fn from(array: Array) -> Self {
        let field = Field::from_array("", &array, None);
        Self(FieldArray::new(field, array))
    }
}

impl From<PyArray> for FieldArray {
    fn from(value: PyArray) -> Self {
        value.0
    }
}

impl From<PyArray> for Arc<Array> {
    fn from(value: PyArray) -> Self {
        Arc::new(value.0.array)
    }
}

impl AsRef<Array> for PyArray {
    fn as_ref(&self) -> &Array {
        &self.0.array
    }
}

impl<'py> FromPyObject<'py> for PyArray {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let field_array = to_rust::array_to_rust(ob)?;
        Ok(PyArray(field_array))
    }
}

impl<'py> IntoPyObject<'py> for PyArray {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        // Use the preserved Field metadata for correct Arrow type export
        to_py::array_to_py(Arc::new(self.0.array), &self.0.field, py)
    }
}

// PyRecordBatch - Wrapper around MinArrow's Table

/// Transparent wrapper around MinArrow's Table.
///
/// Enables conversion to/from PyArrow RecordBatch. Equivalent to an Arrow RecordBatch
/// which is a collection of equal-length arrays with schema metadata.
///
/// # Example (Rust)
/// ```ignore
/// use minarrow_pyo3::PyRecordBatch;
/// use minarrow::Table;
///
/// #[pyfunction]
/// fn process_batch(batch: PyRecordBatch) -> PyResult<PyRecordBatch> {
///     let table: Table = batch.into();
///     // Process...
///     Ok(PyRecordBatch::from(table))
/// }
/// ```
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct PyRecordBatch(pub Table);

impl PyRecordBatch {
    /// Creates a new PyRecordBatch from a Table.
    pub fn new(table: Table) -> Self {
        Self(table)
    }

    /// Returns a reference to the inner MinArrow Table.
    pub fn inner(&self) -> &Table {
        &self.0
    }

    /// Consumes self and returns the inner Table.
    pub fn into_inner(self) -> Table {
        self.0
    }
}

impl From<Table> for PyRecordBatch {
    fn from(table: Table) -> Self {
        Self(table)
    }
}

impl From<PyRecordBatch> for Table {
    fn from(value: PyRecordBatch) -> Self {
        value.0
    }
}

impl AsRef<Table> for PyRecordBatch {
    fn as_ref(&self) -> &Table {
        &self.0
    }
}

impl<'py> FromPyObject<'py> for PyRecordBatch {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let table = to_rust::record_batch_to_rust(ob)?;
        Ok(PyRecordBatch(table))
    }
}

impl<'py> IntoPyObject<'py> for PyRecordBatch {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        to_py::table_to_py(&self.0, py)
    }
}

// PyField - Wrapper around MinArrow's Field

/// Transparent wrapper around MinArrow's Field.
///
/// Represents column-level schema metadata including name, type, and nullability.
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct PyField(pub Field);

impl PyField {
    /// Creates a new PyField from a Field.
    pub fn new(field: Field) -> Self {
        Self(field)
    }

    /// Returns a reference to the inner MinArrow Field.
    pub fn inner(&self) -> &Field {
        &self.0
    }

    /// Consumes self and returns the inner Field.
    pub fn into_inner(self) -> Field {
        self.0
    }
}

impl From<Field> for PyField {
    fn from(field: Field) -> Self {
        Self(field)
    }
}

impl From<PyField> for Field {
    fn from(value: PyField) -> Self {
        value.0
    }
}

impl AsRef<Field> for PyField {
    fn as_ref(&self) -> &Field {
        &self.0
    }
}

// PyTable - Wrapper around MinArrow's SuperTable (PyArrow Table)

/// Transparent wrapper around MinArrow's SuperTable.
///
/// Enables conversion to/from PyArrow Table. A PyArrow Table is a collection
/// of RecordBatches (chunked columns), equivalent to MinArrow's SuperTable.
///
/// # Example (Rust)
/// ```ignore
/// use minarrow_pyo3::PyTable;
/// use minarrow::SuperTable;
///
/// #[pyfunction]
/// fn process_table(table: PyTable) -> PyResult<PyTable> {
///     let super_table: SuperTable = table.into();
///     // Process...
///     Ok(PyTable::from(super_table))
/// }
/// ```
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct PyTable(pub SuperTable);

impl PyTable {
    /// Creates a new PyTable from a SuperTable.
    pub fn new(table: SuperTable) -> Self {
        Self(table)
    }

    /// Returns a reference to the inner MinArrow SuperTable.
    pub fn inner(&self) -> &SuperTable {
        &self.0
    }

    /// Consumes self and returns the inner SuperTable.
    pub fn into_inner(self) -> SuperTable {
        self.0
    }
}

impl From<SuperTable> for PyTable {
    fn from(table: SuperTable) -> Self {
        Self(table)
    }
}

impl From<PyTable> for SuperTable {
    fn from(value: PyTable) -> Self {
        value.0
    }
}

impl AsRef<SuperTable> for PyTable {
    fn as_ref(&self) -> &SuperTable {
        &self.0
    }
}

impl<'py> FromPyObject<'py> for PyTable {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let table = to_rust::table_to_rust(ob)?;
        Ok(PyTable(table))
    }
}

impl<'py> IntoPyObject<'py> for PyTable {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        to_py::super_table_to_py(&self.0, py)
    }
}

// PyChunkedArray - Wrapper around MinArrow's SuperArray

/// Transparent wrapper around MinArrow's SuperArray.
///
/// Enables conversion to/from PyArrow ChunkedArray. A PyArrow ChunkedArray
/// contains multiple array chunks, equivalent to MinArrow's SuperArray.
///
/// # Example (Rust)
/// ```ignore
/// use minarrow_pyo3::PyChunkedArray;
/// use minarrow::SuperArray;
///
/// #[pyfunction]
/// fn process_chunked(arr: PyChunkedArray) -> PyResult<PyChunkedArray> {
///     let super_array: SuperArray = arr.into();
///     // Process...
///     Ok(PyChunkedArray::from(super_array))
/// }
/// ```
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct PyChunkedArray(pub SuperArray);

impl PyChunkedArray {
    /// Creates a new PyChunkedArray from a SuperArray.
    pub fn new(array: SuperArray) -> Self {
        Self(array)
    }

    /// Returns a reference to the inner MinArrow SuperArray.
    pub fn inner(&self) -> &SuperArray {
        &self.0
    }

    /// Consumes self and returns the inner SuperArray.
    pub fn into_inner(self) -> SuperArray {
        self.0
    }
}

impl From<SuperArray> for PyChunkedArray {
    fn from(array: SuperArray) -> Self {
        Self(array)
    }
}

impl From<PyChunkedArray> for SuperArray {
    fn from(value: PyChunkedArray) -> Self {
        value.0
    }
}

impl AsRef<SuperArray> for PyChunkedArray {
    fn as_ref(&self) -> &SuperArray {
        &self.0
    }
}

impl<'py> FromPyObject<'py> for PyChunkedArray {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let array = to_rust::chunked_array_to_rust(ob)?;
        Ok(PyChunkedArray(array))
    }
}

impl<'py> IntoPyObject<'py> for PyChunkedArray {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        to_py::super_array_to_py(&self.0, py)
    }
}

// ── View wrappers (zero-copy windowed export) ──────────────────────────
//
// Mirror of the owned wrappers above, but each exports through Arrow C's
// `offset`/`length` so no buffer is copied when handing data to PyArrow,
// Polars, or any other Arrow PyCapsule consumer.

/// Transparent wrapper around a MinArrow [`ArrayV`] paired with a [`Field`].
///
/// Enables zero-copy export of a windowed view to PyArrow. The window is
/// conveyed at the Arrow C layer via `ArrowArray.offset` and
/// `ArrowArray.length`; no buffer is copied.
///
/// Export-only: there is no inbound `FromPyObject` impl. To bring a
/// PyArrow array into Rust, use [`PyArray`].
#[derive(Debug, Clone)]
pub struct PyArrayView {
    pub field: Field,
    pub view: ArrayV,
}

impl PyArrayView {
    /// Creates a new PyArrayView from a Field and ArrayV.
    pub fn new(field: Field, view: ArrayV) -> Self {
        Self { field, view }
    }

    /// Returns a reference to the inner ArrayV.
    pub fn inner(&self) -> &ArrayV {
        &self.view
    }

    /// Returns a reference to the Field metadata.
    pub fn field(&self) -> &Field {
        &self.field
    }
}

impl<'py> IntoPyObject<'py> for PyArrayView {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        to_py::array_view_to_py(&self.view, &self.field, py)
    }
}

/// Transparent wrapper around a MinArrow [`TableV`].
///
/// Exports as a PyArrow `RecordBatch`. Per-column window offsets are
/// propagated at the Arrow C layer, so no buffer is copied.
#[derive(Debug, Clone)]
pub struct PyRecordBatchView(pub TableV);

impl PyRecordBatchView {
    pub fn new(view: TableV) -> Self {
        Self(view)
    }
    pub fn inner(&self) -> &TableV {
        &self.0
    }
    pub fn into_inner(self) -> TableV {
        self.0
    }
}

impl From<TableV> for PyRecordBatchView {
    fn from(view: TableV) -> Self {
        Self(view)
    }
}

impl From<PyRecordBatchView> for TableV {
    fn from(value: PyRecordBatchView) -> Self {
        value.0
    }
}

impl AsRef<TableV> for PyRecordBatchView {
    fn as_ref(&self) -> &TableV {
        &self.0
    }
}

impl<'py> IntoPyObject<'py> for PyRecordBatchView {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        to_py::table_view_to_py(&self.0, py)
    }
}

/// Transparent wrapper around a MinArrow [`SuperTableV`].
///
/// Exports as a PyArrow `Table`. Per-column window offsets are propagated
/// zero-copy across all batches.
#[derive(Debug, Clone)]
pub struct PyTableView(pub SuperTableV);

impl PyTableView {
    pub fn new(view: SuperTableV) -> Self {
        Self(view)
    }
    pub fn inner(&self) -> &SuperTableV {
        &self.0
    }
    pub fn into_inner(self) -> SuperTableV {
        self.0
    }
}

impl From<SuperTableV> for PyTableView {
    fn from(view: SuperTableV) -> Self {
        Self(view)
    }
}

impl AsRef<SuperTableV> for PyTableView {
    fn as_ref(&self) -> &SuperTableV {
        &self.0
    }
}

impl<'py> IntoPyObject<'py> for PyTableView {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        to_py::super_table_view_to_py(&self.0, py)
    }
}

/// Transparent wrapper around a MinArrow [`SuperArrayV`].
///
/// Exports as a PyArrow `ChunkedArray`. Per-slice window offsets are
/// propagated zero-copy.
#[derive(Debug, Clone)]
pub struct PyChunkedArrayView(pub SuperArrayV);

impl PyChunkedArrayView {
    pub fn new(view: SuperArrayV) -> Self {
        Self(view)
    }
    pub fn inner(&self) -> &SuperArrayV {
        &self.0
    }
    pub fn into_inner(self) -> SuperArrayV {
        self.0
    }
}

impl From<SuperArrayV> for PyChunkedArrayView {
    fn from(view: SuperArrayV) -> Self {
        Self(view)
    }
}

impl AsRef<SuperArrayV> for PyChunkedArrayView {
    fn as_ref(&self) -> &SuperArrayV {
        &self.0
    }
}

impl<'py> IntoPyObject<'py> for PyChunkedArrayView {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        to_py::super_array_view_to_py(&self.0, py)
    }
}
