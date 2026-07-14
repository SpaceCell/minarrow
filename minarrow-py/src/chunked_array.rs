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

//! `ChunkedArray` - a chunked column over minarrow's `SuperArray`.
//!
//! An ordered set of `Array` chunks that share one dtype and field. The chunks
//! stay separate, so a column built from many batches needs no copy to a single
//! contiguous buffer. Maps to a PyArrow `ChunkedArray` over the Arrow C Data
//! Interface.

use std::sync::Arc;

use minarrow::{Array, ArrayV, Field, SuperArray};
#[cfg(feature = "arrow_interop")]
use minarrow_pyo3::ffi::{to_py, to_rust};
#[cfg(feature = "arrow_interop")]
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::array::{PyArray, PyArrayInner};
use crate::arrow_type::PyArrowType;
use crate::dtype::{dtype_from_arrow, DType};

/// An ordered set of `Array` chunks that share a dtype and field. Wraps
/// minarrow's `SuperArray`.
#[pyclass(name = "ChunkedArray", module = "minarrow")]
pub struct PyChunkedArray(pub Arc<SuperArray>);

#[pymethods]
impl PyChunkedArray {
    /// Construct from a list of `Array` chunks. An optional `name` sets the
    /// column field shared by every chunk.
    #[new]
    #[pyo3(signature = (chunks, name=None))]
    fn new(chunks: Vec<Bound<'_, PyArray>>, name: Option<String>) -> PyResult<Self> {
        let arrays: Vec<Array> = chunks
            .iter()
            .map(|chunk| ArrayV::from(&chunk.borrow().0).to_array())
            .collect();
        let mut inner = SuperArray::from_arrays(arrays);
        if inner.n_chunks() > 0 {
            let dtype = inner.arrow_type();
            inner.field = Some(Arc::new(Field::new(name.unwrap_or_default(), dtype, true, None)));
        }
        Ok(PyChunkedArray(Arc::new(inner)))
    }

    /// The number of chunks.
    #[getter]
    fn n_chunks(&self) -> usize {
        self.0.n_chunks()
    }

    /// The column name, or `None` when the column is unnamed.
    #[getter]
    fn name(&self) -> Option<String> {
        self.0
            .field()
            .map(|field| field.name.clone())
            .filter(|name| !name.is_empty())
    }

    /// The mapped Arrow logical type.
    #[getter]
    fn arrow_type(&self) -> PyArrowType {
        PyArrowType::from(self.0.arrow_type())
    }

    /// The concrete dtype.
    #[getter]
    fn dtype(&self) -> DType {
        dtype_from_arrow(&self.0.arrow_type())
    }

    /// The total number of nulls across all chunks.
    #[getter]
    fn null_count(&self) -> usize {
        self.0.chunks().iter().map(|array| array.null_count()).sum()
    }

    /// The chunk at `index`, or `None` when out of range.
    fn chunk(&self, index: usize) -> Option<PyArray> {
        self.0.chunk(index).map(|array| PyArray(PyArrayInner::from(array.clone())))
    }

    /// The chunks in order.
    #[getter]
    fn chunks(&self) -> Vec<PyArray> {
        self.0
            .chunks()
            .iter()
            .map(|array| PyArray(PyArrayInner::from(array.clone())))
            .collect()
    }

    /// The total number of rows across all chunks.
    fn __len__(&self) -> usize {
        self.0.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "ChunkedArray(dtype: {}, chunks: {}, len: {}, nulls: {})",
            self.0.arrow_type(),
            self.0.n_chunks(),
            self.0.len(),
            self.0.chunks().iter().map(|array| array.null_count()).sum::<usize>(),
        )
    }

    /// Export to a PyArrow `ChunkedArray` through the Arrow C Data Interface.
    #[cfg(feature = "arrow_interop")]
    fn to_arrow<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_py::super_array_to_py(&self.0, py)
    }

    /// Import a chunked Arrow producer, such as a PyArrow `ChunkedArray`,
    /// through the Arrow C Data Interface.
    #[cfg(feature = "arrow_interop")]
    #[staticmethod]
    fn from_arrow(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            to_rust::chunked_array_to_rust(obj)
        }))
        .map_err(|_| {
            PyValueError::new_err("from_arrow: the Arrow type is not supported by this build")
        })?
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyChunkedArray(Arc::new(inner)))
    }

    /// Export through the Arrow C Data Interface PyCapsule protocol, so any
    /// Arrow-aware library reads this chunked column.
    #[cfg(feature = "arrow_interop")]
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__(
        &self,
        py: Python<'_>,
        requested_schema: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let _ = requested_schema;
        to_py::super_array_to_stream_capsule(&self.0, py)
    }
}
