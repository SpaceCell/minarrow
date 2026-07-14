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

//! `ChunkedTable` - a chunked table over minarrow's `SuperTable`.
//!
//! An ordered set of `Table` batches that share one schema. The batches stay
//! separate, so a table assembled from a stream of record batches needs no
//! copy to one contiguous table. Maps to a PyArrow `Table` over the Arrow C
//! Data Interface.

use std::collections::BTreeMap;
use std::sync::Arc;

use minarrow::ffi::schema::Schema;
use minarrow::{Field, SuperTable, Table};
#[cfg(feature = "arrow_interop")]
use minarrow_pyo3::ffi::{to_py, to_rust};
#[cfg(feature = "arrow_interop")]
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::field::PySchema;
use crate::table::{PyTable, PyTableInner};

/// An ordered set of `Table` batches that share a schema. Wraps minarrow's
/// `SuperTable`.
#[pyclass(name = "ChunkedTable", module = "minarrow")]
pub struct PyChunkedTable(pub Arc<SuperTable>, pub Option<Schema>);

#[pymethods]
impl PyChunkedTable {
    /// Construct from a list of `Table` batches. An optional `name` overrides
    /// the table name. An optional `schema` carries field and table metadata
    /// and is returned by `.schema` in place of the schema derived from the
    /// batches.
    #[new]
    #[pyo3(signature = (batches, name=None, schema=None))]
    fn new(
        batches: Vec<Bound<'_, PyTable>>,
        name: Option<String>,
        schema: Option<PyRef<'_, PySchema>>,
    ) -> PyResult<Self> {
        let tables: Vec<Arc<Table>> = batches
            .iter()
            .map(|batch| Arc::new(batch.borrow().0.as_view().to_table()))
            .collect();
        let inner = SuperTable::from_batches(tables, name);
        Ok(PyChunkedTable(Arc::new(inner), schema.map(|s| s.0.clone())))
    }

    /// The number of batches.
    #[getter]
    fn n_batches(&self) -> usize {
        self.0.n_batches()
    }

    /// The number of batches, named to match `ChunkedArray.n_chunks`.
    #[getter]
    fn n_chunks(&self) -> usize {
        self.0.n_batches()
    }

    /// The total number of rows across all batches.
    #[getter]
    fn n_rows(&self) -> usize {
        self.0.n_rows()
    }

    /// The number of columns.
    #[getter]
    fn n_cols(&self) -> usize {
        self.0.n_cols()
    }

    /// The table name.
    #[getter]
    fn name(&self) -> String {
        self.0.name.clone()
    }

    /// The column names in order.
    #[getter]
    fn columns(&self) -> Vec<String> {
        self.0.schema().iter().map(|field| field.name.clone()).collect()
    }

    /// The schema attached at construction, or the schema derived from the
    /// batches when none was given.
    #[getter]
    fn schema(&self) -> PySchema {
        if let Some(schema) = &self.1 {
            return PySchema(schema.clone());
        }
        let fields: Vec<Field> = self.0.schema().iter().map(|field| (**field).clone()).collect();
        PySchema(Schema::new(fields, BTreeMap::new()))
    }

    /// The batch at `index`, or `None` when out of range.
    fn batch(&self, index: usize) -> Option<PyTable> {
        self.0.batch(index).map(|table| PyTable(PyTableInner::Owned(table.clone())))
    }

    /// The batches in order.
    #[getter]
    fn batches(&self) -> Vec<PyTable> {
        self.0
            .batches()
            .iter()
            .map(|table| PyTable(PyTableInner::Owned(table.clone())))
            .collect()
    }

    /// The total number of rows across all batches.
    fn __len__(&self) -> usize {
        self.0.n_rows()
    }

    fn __repr__(&self) -> String {
        format!(
            "ChunkedTable(name: {}, batches: {}, rows: {}, cols: {})",
            self.0.name,
            self.0.n_batches(),
            self.0.n_rows(),
            self.0.n_cols(),
        )
    }

    /// Export to a PyArrow `Table` through the Arrow C Data Interface.
    #[cfg(feature = "arrow_interop")]
    fn to_arrow<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_py::super_table_to_py(&self.0, py)
    }

    /// Import a chunked Arrow producer, such as a PyArrow `Table`, through the
    /// Arrow C Data Interface.
    #[cfg(feature = "arrow_interop")]
    #[staticmethod]
    fn from_arrow(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            to_rust::table_to_rust(obj)
        }))
        .map_err(|_| {
            PyValueError::new_err("from_arrow: the Arrow type is not supported by this build")
        })?
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyChunkedTable(Arc::new(inner), None))
    }

    /// Export through the Arrow C Data Interface PyCapsule protocol, so any
    /// Arrow-aware library reads this chunked table.
    #[cfg(feature = "arrow_interop")]
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__(
        &self,
        py: Python<'_>,
        requested_schema: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let _ = requested_schema;
        to_py::super_table_to_stream_capsule(&self.0, py)
    }
}
