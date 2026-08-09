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

//! Dense `f64` matrix for the routines that hand memory to LAPACK.
//!
//! A `Table` holds each column in its own allocation, which suits scanning and
//! appending. Regression, decomposition and clustering instead want one
//! contiguous column-major buffer with a known stride, so they can pass
//! `(pointer, leading dimension)` straight to BLAS without repacking. `Matrix`
//! is that layout, and `from_table` is the boundary a caller crosses to reach
//! it.
//!
//! Columns are padded so each begins on a 64-byte boundary, which is why
//! `stride` may exceed `n_rows`. The padding is internal, so `n_rows`,
//! `n_cols`, indexing and iteration all read the logical shape.

use minarrow::{Matrix, Table};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::table::{PyTable, PyTableInner};

/// A dense, column-major `f64` matrix.
#[pyclass(name = "Matrix", module = "minarrow")]
#[derive(Clone)]
pub struct PyMatrix(pub Matrix);

impl From<Matrix> for PyMatrix {
    fn from(matrix: Matrix) -> Self {
        PyMatrix(matrix)
    }
}

#[pymethods]
impl PyMatrix {
    /// Build a matrix from rows, each row a list of the same length.
    ///
    /// Rows are the form a literal is normally written in, matching the reading
    /// `[[1, 2], [3, 4]]` gets elsewhere in Python. The buffer underneath is
    /// column-major regardless, so `from_cols` skips the transposition when the
    /// caller already holds columns.
    #[new]
    #[pyo3(signature = (rows, name=None))]
    fn new(rows: Vec<Vec<f64>>, name: Option<String>) -> PyResult<Self> {
        if rows.is_empty() {
            return Err(PyValueError::new_err("Matrix requires at least one row"));
        }
        let n_cols = rows[0].len();
        if n_cols == 0 {
            return Err(PyValueError::new_err("Matrix requires at least one column"));
        }
        if let Some((i, row)) = rows.iter().enumerate().find(|(_, r)| r.len() != n_cols) {
            return Err(PyValueError::new_err(format!(
                "row {i} has {} values against {n_cols} in the first row",
                row.len()
            )));
        }
        let cols: Vec<Vec<f64>> = (0..n_cols)
            .map(|c| rows.iter().map(|r| r[c]).collect())
            .collect();
        let mut matrix = Matrix::from(cols.as_slice());
        if let Some(name) = name {
            matrix.set_name(name);
        }
        Ok(PyMatrix(matrix))
    }

    /// Build a matrix from columns, which is the layout it already stores.
    #[staticmethod]
    #[pyo3(signature = (cols, name=None))]
    fn from_cols(cols: Vec<Vec<f64>>, name: Option<String>) -> PyResult<Self> {
        if cols.is_empty() {
            return Err(PyValueError::new_err("Matrix requires at least one column"));
        }
        let n_rows = cols[0].len();
        if let Some((i, col)) = cols.iter().enumerate().find(|(_, c)| c.len() != n_rows) {
            return Err(PyValueError::new_err(format!(
                "column {i} has {} values against {n_rows} in the first column",
                col.len()
            )));
        }
        let mut matrix = Matrix::from(cols.as_slice());
        if let Some(name) = name {
            matrix.set_name(name);
        }
        Ok(PyMatrix(matrix))
    }

    /// Build a matrix from a table's columns.
    ///
    /// Every column must be `f64` and free of nulls, since the layout has
    /// nowhere to record a missing value and the routines reading it assume
    /// each entry is a number. A table that does not meet this says which
    /// column disagreed.
    #[staticmethod]
    fn from_table(table: PyRef<'_, PyTable>) -> PyResult<Self> {
        Matrix::try_from(&table.0.as_view())
            .map(PyMatrix)
            .map_err(|e| PyValueError::new_err(format!("{e}")))
    }

    #[getter]
    fn n_rows(&self) -> usize {
        self.0.n_rows
    }

    #[getter]
    fn n_cols(&self) -> usize {
        self.0.n_cols
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, [self.0.n_rows, self.0.n_cols])
    }

    /// The physical distance between the start of one column and the next.
    ///
    /// This is `n_rows` rounded up to a 64-byte boundary, and it is what BLAS
    /// calls the leading dimension.
    #[getter]
    fn stride(&self) -> usize {
        self.0.stride
    }

    #[getter]
    fn name(&self) -> Option<&str> {
        self.0.name.as_deref()
    }

    /// The number of rows, so a matrix reads like a sequence of them.
    fn __len__(&self) -> usize {
        self.0.n_rows
    }

    /// Read a single entry with `m[row, col]`, or a whole row with `m[row]`.
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok((row, col)) = key.extract::<(usize, usize)>() {
            if row >= self.0.n_rows || col >= self.0.n_cols {
                return Err(PyIndexError::new_err(format!(
                    "({row}, {col}) is outside a {} by {} matrix",
                    self.0.n_rows, self.0.n_cols
                )));
            }
            return Ok(self.0.get(row, col).into_pyobject(py)?.into_any().unbind());
        }
        if let Ok(row) = key.extract::<usize>() {
            if row >= self.0.n_rows {
                return Err(PyIndexError::new_err(format!(
                    "row {row} is outside a matrix of {} rows",
                    self.0.n_rows
                )));
            }
            return Ok(self.0.row(row).into_pyobject(py)?.into_any().unbind());
        }
        Err(PyTypeError::new_err(
            "index a matrix with a row, or with a (row, column) pair",
        ))
    }

    /// One column as a list of values.
    fn col(&self, col: usize) -> PyResult<Vec<f64>> {
        if col >= self.0.n_cols {
            return Err(PyIndexError::new_err(format!(
                "column {col} is outside a matrix of {} columns",
                self.0.n_cols
            )));
        }
        Ok(self.0.col(col).to_vec())
    }

    /// One row as a list of values, gathered across the columns.
    fn row(&self, row: usize) -> PyResult<Vec<f64>> {
        if row >= self.0.n_rows {
            return Err(PyIndexError::new_err(format!(
                "row {row} is outside a matrix of {} rows",
                self.0.n_rows
            )));
        }
        Ok(self.0.row(row))
    }

    /// The matrix with rows and columns exchanged, in a fresh buffer.
    fn transpose(&self) -> Self {
        PyMatrix(self.0.transpose())
    }

    #[getter]
    fn T(&self) -> Self {
        PyMatrix(self.0.transpose())
    }

    /// The rows at the given positions, in the order given.
    fn extract_rows(&self, indices: Vec<usize>) -> PyResult<Self> {
        if let Some(&i) = indices.iter().find(|&&i| i >= self.0.n_rows) {
            return Err(PyIndexError::new_err(format!(
                "row {i} is outside a matrix of {} rows",
                self.0.n_rows
            )));
        }
        Ok(PyMatrix(self.0.extract_rows(&indices)))
    }

    /// The same values as a table, each column named by its position.
    fn to_table(&self) -> PyTable {
        let table: Table = self.0.clone().to_table_gen();
        PyTable(PyTableInner::from(table))
    }

    fn __repr__(&self) -> String {
        match &self.0.name {
            Some(name) => format!(
                "Matrix('{name}', {} rows x {} cols, f64)",
                self.0.n_rows, self.0.n_cols
            ),
            None => format!(
                "Matrix({} rows x {} cols, f64)",
                self.0.n_rows, self.0.n_cols
            ),
        }
    }
}
