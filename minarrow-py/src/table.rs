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

//! `Table` - the native Python table object over minarrow.
//!
//! Holds an owned `Arc<Table>` or a zero-copy `Arc<TableV>` window. The data
//! surface lives on `PyTableInner` so the standalone library here and the
//! `astrobears` compute wheel wrap the same inner. Indexing maps onto minarrow's
//! selection traits: string keys select columns, integer and slice keys select
//! positional rows, and a 2-tuple is `(rows, cols)`.

use std::collections::BTreeMap;
use std::sync::Arc;

use minarrow::enums::error::MinarrowError;
use minarrow::ffi::schema::Schema;
use minarrow::{ArrayV, ColumnSelection, Field, FieldArray, RowSelection, Table, TableV};
#[cfg(feature = "arrow_interop")]
use minarrow_pyo3::ffi::{to_py, to_rust};
use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};

use crate::array::PyArray;
use crate::convert::{build_array, build_array_typed, resolve_index};
use crate::dtype::{dtype_from_arrow, DType};
#[cfg(feature = "ndarray")]
use crate::ndarray::{ndarray_from_table, PyNdArray};
use crate::field::PySchema;

/// The natural minarrow form behind a Python `Table`. Carries the whole data
/// surface so any pyclass wrapping it inherits identical behaviour.
pub enum PyTableInner {
    Owned(Arc<Table>),
    View(Arc<TableV>),
}

impl From<Table> for PyTableInner {
    fn from(table: Table) -> Self {
        PyTableInner::Owned(Arc::new(table))
    }
}

impl From<TableV> for PyTableInner {
    fn from(view: TableV) -> Self {
        PyTableInner::View(Arc::new(view))
    }
}

impl PyTableInner {
    /// A whole-table view, used as the base for selection and aggregation.
    pub fn as_view(&self) -> TableV {
        match self {
            PyTableInner::Owned(table) => TableV::from_arc_table(table.clone(), 0, table.n_rows()),
            PyTableInner::View(view) => (**view).clone(),
        }
    }

    /// Resolve a Python row key - int, slice, or list of ints - to a row-windowed
    /// view.
    fn row_view(&self, key: &Bound<'_, PyAny>) -> PyResult<TableV> {
        let view = self.as_view();
        let n = view.n_rows();

        if let Ok(slice) = key.downcast::<PySlice>() {
            let ind = slice.indices(n as isize)?;
            if ind.step == 1 {
                return Ok(view.r((ind.start as usize)..(ind.stop as usize)));
            }
            let mut idxs = Vec::with_capacity(ind.slicelength);
            for k in 0..ind.slicelength {
                idxs.push((ind.start + k as isize * ind.step) as usize);
            }
            return Ok(view.r(idxs));
        }
        if let Ok(list) = key.extract::<Vec<isize>>() {
            let idxs = list
                .into_iter()
                .map(|i| resolve_index(i, n))
                .collect::<PyResult<Vec<usize>>>()?;
            return Ok(view.r(idxs));
        }
        if let Ok(i) = key.extract::<isize>() {
            return Ok(view.r(resolve_index(i, n)?));
        }
        Err(PyTypeError::new_err(
            "row selector must be an int, a slice, or a list of ints",
        ))
    }

    /// The number of rows.
    pub fn n_rows(&self) -> usize {
        match self {
            PyTableInner::Owned(table) => table.n_rows(),
            PyTableInner::View(view) => view.n_rows(),
        }
    }

    /// The number of columns.
    pub fn n_cols(&self) -> usize {
        match self {
            PyTableInner::Owned(table) => table.n_cols(),
            PyTableInner::View(view) => view.n_cols(),
        }
    }

    /// Appends a column. An owned table is mutated in place. A view is
    /// materialised to an owned table first. The column length must match the
    /// table's row count unless the table is empty.
    pub fn add_column(&mut self, field_array: FieldArray) -> Result<(), MinarrowError> {
        if self.n_cols() > 0 && field_array.len() != self.n_rows() {
            return Err(MinarrowError::ShapeError {
                message: format!(
                    "column length {} does not match table row count {}",
                    field_array.len(),
                    self.n_rows()
                ),
            });
        }
        match self {
            PyTableInner::Owned(table) => Arc::make_mut(table).add_col(field_array),
            PyTableInner::View(view) => {
                let mut table = view.to_table();
                table.add_col(field_array);
                *self = PyTableInner::Owned(Arc::new(table));
            }
        }
        Ok(())
    }

    /// The column names in order.
    pub fn col_names(&self) -> Vec<String> {
        match self {
            PyTableInner::Owned(table) => table.col_names().iter().map(|s| s.to_string()).collect(),
            PyTableInner::View(view) => view.col_names().iter().map(|s| s.to_string()).collect(),
        }
    }

    /// The column name to dtype pairs, in order.
    pub fn schema(&self) -> Vec<(String, DType)> {
        match self {
            PyTableInner::Owned(table) => table
                .cols
                .iter()
                .map(|fa| (fa.field.name.clone(), dtype_from_arrow(&fa.field.dtype)))
                .collect(),
            PyTableInner::View(view) => {
                let active = view
                    .active_col_selection
                    .clone()
                    .unwrap_or_else(|| (0..view.fields.len()).collect());
                active
                    .into_iter()
                    .map(|i| {
                        let field = &view.fields[i];
                        (field.name.clone(), dtype_from_arrow(&field.dtype))
                    })
                    .collect()
            }
        }
    }

    /// The column fields in active order.
    pub fn schema_fields(&self) -> Vec<Field> {
        match self {
            PyTableInner::Owned(table) => {
                table.cols.iter().map(|fa| (*fa.field).clone()).collect()
            }
            PyTableInner::View(view) => {
                let active = view
                    .active_col_selection
                    .clone()
                    .unwrap_or_else(|| (0..view.fields.len()).collect());
                active
                    .into_iter()
                    .map(|i| (*view.fields[i]).clone())
                    .collect()
            }
        }
    }

    /// Whether this table is a view over another table's buffers.
    pub fn is_view(&self) -> bool {
        matches!(self, PyTableInner::View(_))
    }

    /// The table name, or `None` if unnamed.
    pub fn name(&self) -> Option<String> {
        let name = match self {
            PyTableInner::Owned(table) => &table.name,
            PyTableInner::View(view) => &view.name,
        };
        if name.is_empty() || name.starts_with("UnnamedTable") {
            None
        } else {
            Some(name.clone())
        }
    }

    /// The minarrow `Print` rendering.
    pub fn repr(&self) -> String {
        match self {
            PyTableInner::Owned(table) => format!("{}", table),
            PyTableInner::View(view) => format!("{}", view),
        }
    }

    /// Pandas-feel indexing. String keys select columns, integer and slice keys
    /// select positional rows, and a 2-tuple is `(rows, cols)`. A single column
    /// is handed to `wrap_array`, multiple columns or a row window to `wrap_table`,
    /// so each binding returns its own `Array`/`Table`.
    pub fn get_item(
        &self,
        key: &Bound<'_, PyAny>,
        wrap_array: &dyn Fn(FieldArray) -> PyResult<PyObject>,
        wrap_table: &dyn Fn(TableV) -> PyResult<PyObject>,
    ) -> PyResult<PyObject> {
        if let Ok(tuple) = key.downcast::<PyTuple>() {
            if tuple.len() == 2 {
                let rows = self.row_view(&tuple.get_item(0)?)?;
                return select_columns(&rows, &tuple.get_item(1)?, wrap_array, wrap_table);
            }
        }
        if let Ok(name) = key.extract::<String>() {
            return column_by_name(&self.as_view(), &name, wrap_array);
        }
        if let Ok(names) = key.extract::<Vec<String>>() {
            return wrap_table(self.as_view().c(names));
        }
        let rows = self.row_view(key)?;
        wrap_table(rows)
    }
}

/// Build a minarrow `Table` from a dict of column name to sequence. Columns must
/// share a length. Each column's dtype is inferred, with `None` as null.
pub fn build_table(data: &Bound<'_, PyDict>) -> PyResult<Table> {
    let mut cols: Vec<FieldArray> = Vec::with_capacity(data.len());
    let mut n_rows: Option<usize> = None;
    for (key, value) in data.iter() {
        let name: String = key.extract()?;
        let array = build_array(&value)?;
        let len = array.len();
        match n_rows {
            None => n_rows = Some(len),
            Some(expected) if expected != len => {
                return Err(PyValueError::new_err(format!(
                    "Table: column '{}' has length {}, expected {}",
                    name, len, expected
                )));
            }
            _ => {}
        }
        cols.push(FieldArray::from_arr(name, array));
    }
    Ok(Table::new(String::new(), Some(cols)))
}

/// Build a minarrow `Table` from a dict of column name to sequence, coercing
/// each column to the matching `schema` field by name. Every schema field must
/// have a column in the dict, the field carries onto the column, and the columns
/// must share a length.
pub fn build_table_typed(data: &Bound<'_, PyDict>, schema: &Schema) -> PyResult<Table> {
    let mut cols: Vec<FieldArray> = Vec::with_capacity(schema.fields.len());
    let mut n_rows: Option<usize> = None;
    for field in &schema.fields {
        let value = data.get_item(field.name.as_str())?.ok_or_else(|| {
            PyKeyError::new_err(format!("Table: schema column '{}' not in data", field.name))
        })?;
        let array = build_array_typed(&value, &field.dtype)?;
        let len = array.len();
        match n_rows {
            None => n_rows = Some(len),
            Some(expected) if expected != len => {
                return Err(PyValueError::new_err(format!(
                    "Table: column '{}' has length {}, expected {}",
                    field.name, len, expected
                )));
            }
            _ => {}
        }
        cols.push(FieldArray::new(field.clone(), array));
    }
    Ok(Table::new(String::new(), Some(cols)))
}

/// A single column by name, handed to `wrap_array`.
fn column_by_name(
    view: &TableV,
    name: &str,
    wrap_array: &dyn Fn(FieldArray) -> PyResult<PyObject>,
) -> PyResult<PyObject> {
    match view.get(name) {
        Some(field) => wrap_array(field),
        None => Err(PyKeyError::new_err(format!("column '{}' not found", name))),
    }
}

/// Apply a column selector to a row-windowed view. A single column goes to
/// `wrap_array`, multiple columns to `wrap_table`.
fn select_columns(
    view: &TableV,
    column_selection: &Bound<'_, PyAny>,
    wrap_array: &dyn Fn(FieldArray) -> PyResult<PyObject>,
    wrap_table: &dyn Fn(TableV) -> PyResult<PyObject>,
) -> PyResult<PyObject> {
    if let Ok(name) = column_selection.extract::<String>() {
        return column_by_name(view, &name, wrap_array);
    }
    if let Ok(names) = column_selection.extract::<Vec<String>>() {
        return wrap_table(view.c(names));
    }
    if let Ok(slice) = column_selection.downcast::<PySlice>() {
        let ind = slice.indices(view.n_cols() as isize)?;
        let mut idxs = Vec::with_capacity(ind.slicelength);
        for k in 0..ind.slicelength {
            idxs.push((ind.start + k as isize * ind.step) as usize);
        }
        return wrap_table(view.c(idxs));
    }
    if let Ok(list) = column_selection.extract::<Vec<isize>>() {
        let n = view.n_cols();
        let idxs = list
            .into_iter()
            .map(|i| resolve_index(i, n))
            .collect::<PyResult<Vec<usize>>>()?;
        return wrap_table(view.c(idxs));
    }
    if let Ok(i) = column_selection.extract::<isize>() {
        let idx = resolve_index(i, view.n_cols())?;
        return match view.col_name(idx) {
            Some(name) => column_by_name(view, name, wrap_array),
            None => Err(PyIndexError::new_err("column index out of range")),
        };
    }
    Err(PyTypeError::new_err(
        "column selector must be a name, position, list, or slice",
    ))
}

/// A minarrow table exposed to Python.
#[pyclass(name = "Table", module = "minarrow")]
pub struct PyTable(pub PyTableInner);

impl From<Table> for PyTable {
    fn from(table: Table) -> Self {
        PyTable(table.into())
    }
}

impl From<TableV> for PyTable {
    fn from(view: TableV) -> Self {
        PyTable(view.into())
    }
}

#[pymethods]
impl PyTable {
    /// Construct from a dict of column name to sequence. Columns must share a
    /// length. Each column's dtype is inferred, with `None` as null.
    #[new]
    #[pyo3(signature = (data, name=None, schema=None))]
    fn new(
        data: &Bound<'_, PyDict>,
        name: Option<String>,
        schema: Option<PyRef<'_, PySchema>>,
    ) -> PyResult<Self> {
        let mut table = match schema {
            Some(schema) => build_table_typed(data, &schema.0)?,
            None => build_table(data)?,
        };
        if let Some(name) = name {
            table.name = name;
        }
        Ok(PyTable::from(table))
    }

    /// Import a PyArrow RecordBatch via the Arrow C Data Interface.
    #[cfg(feature = "arrow_interop")]
    #[staticmethod]
    fn from_arrow(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let table = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            to_rust::record_batch_to_rust(obj)
        }))
        .map_err(|_| {
            PyValueError::new_err("from_arrow: the Arrow type is not supported by this build")
        })?
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyTable(PyTableInner::Owned(Arc::new(table))))
    }

    /// Export to a PyArrow RecordBatch via the Arrow C Data Interface.
    #[cfg(feature = "arrow_interop")]
    fn to_arrow<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.0 {
            PyTableInner::Owned(table) => to_py::table_to_py(table, py),
            PyTableInner::View(view) => to_py::table_to_py(&view.to_table(), py),
        }
    }

    /// Convert to a Polars `DataFrame` through the Arrow PyCapsule interface.
    /// Requires the `polars` package.
    #[cfg(feature = "arrow_interop")]
    fn to_polars<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        py.import("polars")?.call_method1("from_arrow", (&slf,))
    }

    /// Alias for `from_arrow`, accepting any Polars object through the Arrow
    /// PyCapsule interface.
    #[cfg(feature = "arrow_interop")]
    #[staticmethod]
    fn from_polars(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::from_arrow(obj)
    }

    /// Convert to a DuckDB relation through the Arrow PyCapsule interface.
    /// Requires the `duckdb` package.
    #[cfg(feature = "arrow_interop")]
    fn to_duckdb<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        py.import("duckdb")?.call_method1("from_arrow", (&slf,))
    }

    /// Alias for `from_arrow`, accepting any DuckDB object through the Arrow
    /// PyCapsule interface.
    #[cfg(feature = "arrow_interop")]
    #[staticmethod]
    fn from_duckdb(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::from_arrow(obj)
    }

    /// Convert to a Daft `DataFrame` through the Arrow PyCapsule interface.
    /// Requires the `daft` package.
    #[cfg(feature = "arrow_interop")]
    fn to_daft<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        py.import("daft")?.call_method1("from_arrow", (&slf,))
    }

    /// Alias for `from_arrow`, accepting any Daft object through the Arrow
    /// PyCapsule interface.
    #[cfg(feature = "arrow_interop")]
    #[staticmethod]
    fn from_daft(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::from_arrow(obj)
    }

    /// Convert to a DataFusion `DataFrame` through the Arrow PyCapsule interface.
    /// Requires the `datafusion` package.
    #[cfg(feature = "arrow_interop")]
    fn to_datafusion<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let ctx = py.import("datafusion")?.call_method0("SessionContext")?;
        ctx.call_method1("from_arrow", (&slf,))
    }

    /// Alias for `from_arrow`, accepting any DataFusion object through the Arrow
    /// PyCapsule interface.
    #[cfg(feature = "arrow_interop")]
    #[staticmethod]
    fn from_datafusion(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::from_arrow(obj)
    }

    /// Convert to a nanoarrow `ArrayStream` through the Arrow PyCapsule
    /// interface. Requires the `nanoarrow` package.
    #[cfg(feature = "arrow_interop")]
    fn to_nanoarrow<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        py.import("nanoarrow")?.call_method1("ArrayStream", (&slf,))
    }

    /// Alias for `from_arrow`, accepting any nanoarrow object through the Arrow
    /// PyCapsule interface.
    #[cfg(feature = "arrow_interop")]
    #[staticmethod]
    fn from_nanoarrow(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::from_arrow(obj)
    }

    /// Convert to a pandas `DataFrame` through the Arrow PyCapsule interface.
    /// Requires pandas 3.0+ (`DataFrame.from_arrow`).
    #[cfg(feature = "arrow_interop")]
    fn to_pandas<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        py.import("pandas")?.getattr("DataFrame")?.call_method1("from_arrow", (&slf,))
    }

    /// Alias for `from_arrow`, accepting a pandas object through the Arrow
    /// PyCapsule interface.
    #[cfg(feature = "arrow_interop")]
    #[staticmethod]
    fn from_pandas(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::from_arrow(obj)
    }

    /// Convert numeric columns to a 2D float64 `NdArray`, one column per
    /// axis-1 entry. From there `to_numpy`, `to_pytorch`, and the other
    /// DLPack bridges hand the data to each framework.
    #[cfg(feature = "ndarray")]
    fn to_ndarray(&self) -> PyResult<PyNdArray> {
        match &self.0 {
            PyTableInner::Owned(table) => ndarray_from_table(table),
            PyTableInner::View(view) => ndarray_from_table(&view.to_table()),
        }
    }

    /// Convert to a cuDF `DataFrame` through the Arrow PyCapsule interface. Runs
    /// on GPU and requires the `cudf` package.
    #[cfg(feature = "arrow_interop")]
    fn to_cudf<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        py.import("cudf")?.getattr("DataFrame")?.call_method1("from_arrow", (&slf,))
    }

    /// Alias for `from_arrow`, accepting a cuDF object through the Arrow
    /// PyCapsule interface.
    #[cfg(feature = "arrow_interop")]
    #[staticmethod]
    fn from_cudf(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::from_arrow(obj)
    }

    /// Convert to an Ibis table expression. Ibis wraps a pyarrow `Table`, so
    /// this requires `pyarrow` and the `ibis-framework` package.
    #[cfg(feature = "arrow_interop")]
    fn to_ibis<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let batch = self.to_arrow(py)?;
        let table = py
            .import("pyarrow")?
            .getattr("Table")?
            .call_method1("from_batches", (vec![batch],))?;
        py.import("ibis")?.call_method1("memtable", (table,))
    }

    /// Alias for `from_arrow`, accepting an Ibis object through the Arrow
    /// PyCapsule interface.
    #[cfg(feature = "arrow_interop")]
    #[staticmethod]
    fn from_ibis(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::from_arrow(obj)
    }

    /// Convert to a Narwhals `DataFrame` over a pyarrow `Table`. Narwhals wraps
    /// a native frame, so this requires `pyarrow` and `narwhals`.
    #[cfg(feature = "arrow_interop")]
    fn to_narwhals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let batch = self.to_arrow(py)?;
        let table = py
            .import("pyarrow")?
            .getattr("Table")?
            .call_method1("from_batches", (vec![batch],))?;
        py.import("narwhals")?.call_method1("from_native", (table,))
    }

    /// Alias for `from_arrow`, accepting a Narwhals object through the Arrow
    /// PyCapsule interface.
    #[cfg(feature = "arrow_interop")]
    #[staticmethod]
    fn from_narwhals(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::from_arrow(obj)
    }

    /// Ingest this table into a database through an ADBC cursor, over the Arrow
    /// PyCapsule interface. `cursor` is an ADBC DBAPI cursor from a driver such
    /// as `adbc_driver_sqlite`, `adbc_driver_postgresql`, `adbc_driver_snowflake`,
    /// `adbc_driver_bigquery`, or `adbc_driver_flightsql`. `mode` is one of
    /// `create`, `append`, `replace`, or `create_append`. Returns the row count.
    #[cfg(feature = "arrow_interop")]
    #[pyo3(signature = (cursor, table_name, mode = "create"))]
    fn write_adbc<'py>(
        slf: Bound<'py, Self>,
        cursor: &Bound<'py, PyAny>,
        table_name: &str,
        mode: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        cursor.call_method1("adbc_ingest", (table_name, &slf, mode))
    }

    /// Read an ADBC cursor's current result into a `Table`. Call after
    /// `cursor.execute(...)`. `cursor` is an ADBC DBAPI cursor.
    #[cfg(feature = "arrow_interop")]
    #[staticmethod]
    fn read_adbc(cursor: &Bound<'_, PyAny>) -> PyResult<Self> {
        let table = cursor.call_method0("fetch_arrow_table")?;
        Self::from_arrow(&table)
    }

    /// The number of rows.
    #[getter]
    fn n_rows(&self) -> usize {
        self.0.n_rows()
    }

    /// The number of columns.
    #[getter]
    fn n_cols(&self) -> usize {
        self.0.n_cols()
    }

    /// The column names in order.
    #[getter]
    fn columns(&self) -> Vec<String> {
        self.0.col_names()
    }

    /// A name to dtype mapping over the columns, in order.
    #[getter]
    fn dtypes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (name, dtype) in self.0.schema() {
            dict.set_item(name, dtype)?;
        }
        Ok(dict)
    }

    /// Whether this table is a view over another table's buffers.
    #[getter]
    fn is_view(&self) -> bool {
        self.0.is_view()
    }

    /// The table's schema: the column fields in order.
    #[getter]
    fn schema(&self) -> PySchema {
        PySchema(Schema::new(self.0.schema_fields(), BTreeMap::new()))
    }

    /// The table name, or `None` when unnamed.
    #[getter]
    fn name(&self) -> Option<String> {
        self.0.name()
    }

    fn __len__(&self) -> usize {
        self.0.n_rows()
    }

    fn __repr__(&self) -> String {
        self.0.repr()
    }

    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.0.get_item(
            key,
            &|field| Ok(Py::new(py, PyArray::from(field))?.into_any()),
            &|view| Ok(Py::new(py, PyTable::from(view))?.into_any()),
        )
    }

    /// Appends `column` under `name`. A view-backed table is materialised first.
    /// The column length must match the table's row count unless the table is
    /// empty.
    fn add_column(&mut self, name: String, column: PyRef<'_, PyArray>) -> PyResult<()> {
        let array = ArrayV::from(&*column).to_array();
        self.0
            .add_column(FieldArray::from_arr(name, array))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Export via the Arrow C Data Interface PyCapsule protocol as a stream of
    /// one record batch, so any Arrow-aware library reads this table directly.
    #[cfg(feature = "arrow_interop")]
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__(
        &self,
        py: Python<'_>,
        requested_schema: Option<PyObject>,
    ) -> PyResult<PyObject> {
        let _ = requested_schema;
        match &self.0 {
            PyTableInner::Owned(table) => to_py::table_to_stream_capsule(table, py),
            // A full-column row window exports zero-copy. A column subset is
            // materialised, since the view stream export does not narrow columns.
            PyTableInner::View(view) if view.n_cols() == view.fields.len() => {
                to_py::table_view_to_stream_capsule(view, py)
            }
            PyTableInner::View(view) => to_py::table_to_stream_capsule(&view.to_table(), py),
        }
    }
}
