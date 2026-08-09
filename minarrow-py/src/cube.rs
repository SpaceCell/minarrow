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
//
//! A named set of tables over a common schema.
//!
//! Each table is a group, a period or a partition, and carries its own name.
//! Looking one up by name is O(1) rather than a scan.
//!
//! `group_by` is the leading case: one table per distinct key, named by that
//! key, so `cube["BTC"]` reads the group.

use std::collections::BTreeMap;
use std::sync::Arc;

use minarrow::ffi::schema::Schema;
use minarrow::{Cube, Field, Table};
use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError};
use pyo3::prelude::*;

use crate::field::PySchema;
use crate::table::{PyTable, PyTableInner};

/// A named set of tables over a common schema.
#[pyclass(name = "Cube", module = "minarrow")]
pub struct PyCube(pub Arc<Cube>);

impl From<Cube> for PyCube {
    fn from(cube: Cube) -> Self {
        PyCube(Arc::new(cube))
    }
}

/// The table at `index`, as the Python object carrying it.
fn table_at(cube: &Cube, index: usize) -> Option<PyTable> {
    cube.table(index)
        .map(|t| PyTable(PyTableInner::Owned(t.clone())))
}

#[pymethods]
impl PyCube {
    /// Build a cube from a list of tables, each keeping its own name.
    ///
    /// `index_by` names the columns forming the third dimension, e.g. the time
    /// column when the tables are periods of a series.
    #[new]
    #[pyo3(signature = (tables, name=None, index_by=None))]
    fn new(
        tables: Vec<Bound<'_, PyTable>>,
        name: Option<String>,
        index_by: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let tables: Vec<Table> = tables
            .iter()
            .map(|t| t.borrow().0.as_view().to_table())
            .collect();
        let cube = Cube::from_tables(tables, name.unwrap_or_default(), index_by);
        Ok(PyCube(Arc::new(cube)))
    }

    #[getter]
    fn name(&self) -> &str {
        &self.0.name
    }

    /// The number of tables the cube holds.
    #[getter]
    fn n_tables(&self) -> usize {
        self.0.n_tables()
    }

    /// The number of columns each table carries.
    #[getter]
    fn n_cols(&self) -> usize {
        self.0.n_cols()
    }

    /// The row count of each table, in order.
    #[getter]
    fn n_rows(&self) -> Vec<usize> {
        self.0.n_rows()
    }

    /// The name of each table, in order. For a grouped cube, the group keys.
    #[getter]
    fn names(&self) -> Vec<String> {
        self.0.table_names().into_iter().map(str::to_string).collect()
    }

    #[getter]
    fn columns(&self) -> Vec<String> {
        self.0.col_names().into_iter().map(str::to_string).collect()
    }

    #[getter]
    fn schema(&self) -> PySchema {
        let fields: Vec<Field> = self.0.schema().iter().map(|f| (**f).clone()).collect();
        PySchema(Schema::new(fields, BTreeMap::new()))
    }

    /// Every table, in order.
    #[getter]
    fn tables(&self) -> Vec<PyTable> {
        (0..self.0.n_tables())
            .filter_map(|i| table_at(&self.0, i))
            .collect()
    }

    /// The table at a position, or `None` where there is none.
    fn table(&self, index: usize) -> Option<PyTable> {
        table_at(&self.0, index)
    }

    /// The number of tables.
    fn __len__(&self) -> usize {
        self.0.n_tables()
    }

    /// Read a table by position with `cube[0]`, or by name with `cube["BTC"]`.
    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<PyTable> {
        if let Ok(name) = key.extract::<String>() {
            let name = name.as_str();
            let index = self.0.table_index(name).ok_or_else(|| {
                PyKeyError::new_err(format!("no table named '{name}' in this cube"))
            })?;
            return table_at(&self.0, index).ok_or_else(|| {
                PyKeyError::new_err(format!("no table named '{name}' in this cube"))
            });
        }
        if let Ok(index) = key.extract::<usize>() {
            return table_at(&self.0, index).ok_or_else(|| {
                PyIndexError::new_err(format!(
                    "table {index} is outside a cube of {} tables",
                    self.0.n_tables()
                ))
            });
        }
        Err(PyTypeError::new_err(
            "index a cube with a position or with a table name",
        ))
    }

    /// The table for a key, looked up on the key's own type.
    ///
    /// A cube split on an `Int32` column is reached with an `int`, and on a
    /// `Float64` column with a `float`. A cube split on several columns takes
    /// one value per column, in the order they were grouped on. This is the
    /// lookup a grouped cube is built for; `cube[i]` reads by position and
    /// `cube["name"]` by name.
    #[pyo3(signature = (*key))]
    fn group(&self, key: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<PyTable> {
        let parts: Vec<minarrow::Scalar> = key
            .iter()
            .map(|k| crate::convert::py_to_scalar(&k))
            .collect::<PyResult<_>>()?;
        let named = || {
            let joined: Vec<String> = parts.iter().map(ToString::to_string).collect();
            PyKeyError::new_err(format!("no group keyed {} in this cube", joined.join("|")))
        };
        let index = self.0.resolve(&parts).ok_or_else(named)?;
        table_at(&self.0, index).ok_or_else(named)
    }

    /// Whether a table of this name is present.
    fn __contains__(&self, name: &str) -> bool {
        self.0.has_table(name)
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyAny>> {
        let tables: Vec<PyTable> = slf.tables();
        let py = slf.py();
        let list = pyo3::types::PyList::new(py, tables)?;
        Ok(list.as_any().try_iter()?.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        let rows: usize = self.0.n_rows().iter().sum();
        format!(
            "Cube(name: {}, tables: {}, rows: {}, cols: {})",
            self.0.name,
            self.0.n_tables(),
            rows,
            self.0.n_cols()
        )
    }
}
