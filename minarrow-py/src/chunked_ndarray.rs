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

//! `ChunkedNdArray` - Python's name for minarrow's `SuperNdArray`.

use std::sync::Arc;

use minarrow::{AxisSelection, Consolidate, NdArray, SuperNdArray, SuperNdArrayV};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::ndarray::{AxisKey, PyNdArray, PyNdArrayInner, axis_keys, selectors};

pub enum PyChunkedNdArrayInner {
    F32(Arc<SuperNdArray<f32>>),
    F64(Arc<SuperNdArray<f64>>),
    F32View(Arc<SuperNdArrayV<f32>>),
    F64View(Arc<SuperNdArrayV<f64>>),
}

/// An ordered set of compatible NdArray pieces, backed by SuperNdArray.
#[pyclass(name = "ChunkedNdArray", module = "minarrow")]
pub struct PyChunkedNdArray(pub PyChunkedNdArrayInner);

fn validate_chunks<T: minarrow::Float>(chunks: &[NdArray<T>]) -> PyResult<()> {
    let Some(first) = chunks.first() else {
        return Ok(());
    };
    if first.ndim() == 0 {
        return Err(PyValueError::new_err(
            "ChunkedNdArray pieces require an axis 0",
        ));
    }
    for (index, chunk) in chunks.iter().enumerate().skip(1) {
        if chunk.ndim() != first.ndim() || chunk.shape()[1..] != first.shape()[1..] {
            return Err(PyValueError::new_err(format!(
                "chunk {index} has shape {:?}, expected rank {} and trailing shape {:?}",
                chunk.shape(),
                first.ndim(),
                &first.shape()[1..]
            )));
        }
    }
    Ok(())
}

impl From<SuperNdArray<f32>> for PyChunkedNdArray {
    fn from(array: SuperNdArray<f32>) -> Self {
        Self(PyChunkedNdArrayInner::F32(Arc::new(array)))
    }
}

impl From<SuperNdArray<f64>> for PyChunkedNdArray {
    fn from(array: SuperNdArray<f64>) -> Self {
        Self(PyChunkedNdArrayInner::F64(Arc::new(array)))
    }
}

impl From<SuperNdArrayV<f32>> for PyChunkedNdArray {
    fn from(view: SuperNdArrayV<f32>) -> Self {
        Self(PyChunkedNdArrayInner::F32View(Arc::new(view)))
    }
}

impl From<SuperNdArrayV<f64>> for PyChunkedNdArray {
    fn from(view: SuperNdArrayV<f64>) -> Self {
        Self(PyChunkedNdArrayInner::F64View(Arc::new(view)))
    }
}

#[pymethods]
impl PyChunkedNdArray {
    /// Construct from compatible NdArray pieces. Empty input defaults to
    /// float64 unless `dtype="float32"` is supplied.
    #[new]
    #[pyo3(signature = (chunks, name="", dtype=None))]
    fn new(chunks: Vec<Bound<'_, PyNdArray>>, name: &str, dtype: Option<&str>) -> PyResult<Self> {
        let dtype = dtype.unwrap_or_else(|| {
            chunks
                .first()
                .map_or("float64", |chunk| match &chunk.borrow().0 {
                    PyNdArrayInner::F32(_) | PyNdArrayInner::F32View(_) => "float32",
                    PyNdArrayInner::F64(_) | PyNdArrayInner::F64View(_) => "float64",
                })
        });
        match dtype {
            "float32" | "f32" => {
                let arrays = chunks
                    .iter()
                    .map(|chunk| match &chunk.borrow().0 {
                        PyNdArrayInner::F32(a) => Ok((**a).clone()),
                        PyNdArrayInner::F32View(v) => Ok(v.to_ndarray()),
                        _ => Err(PyTypeError::new_err(
                            "all ChunkedNdArray pieces must have dtype float32",
                        )),
                    })
                    .collect::<PyResult<Vec<NdArray<f32>>>>()?;
                validate_chunks(&arrays)?;
                Ok(SuperNdArray::from_batches(arrays, name).into())
            }
            "float64" | "f64" => {
                let arrays = chunks
                    .iter()
                    .map(|chunk| match &chunk.borrow().0 {
                        PyNdArrayInner::F64(a) => Ok((**a).clone()),
                        PyNdArrayInner::F64View(v) => Ok(v.to_ndarray()),
                        _ => Err(PyTypeError::new_err(
                            "all ChunkedNdArray pieces must have dtype float64",
                        )),
                    })
                    .collect::<PyResult<Vec<NdArray<f64>>>>()?;
                validate_chunks(&arrays)?;
                Ok(SuperNdArray::from_batches(arrays, name).into())
            }
            other => Err(PyValueError::new_err(format!(
                "dtype must be 'float32' or 'float64', got '{other}'"
            ))),
        }
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.shape_vec())
    }

    #[getter]
    fn ndim(&self) -> usize {
        match &self.0 {
            PyChunkedNdArrayInner::F32(a) => a.ndim(),
            PyChunkedNdArrayInner::F64(a) => a.ndim(),
            PyChunkedNdArrayInner::F32View(v) => v.ndim(),
            PyChunkedNdArrayInner::F64View(v) => v.ndim(),
        }
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        match &self.0 {
            PyChunkedNdArrayInner::F32(_) | PyChunkedNdArrayInner::F32View(_) => "float32",
            PyChunkedNdArrayInner::F64(_) | PyChunkedNdArrayInner::F64View(_) => "float64",
        }
    }

    #[getter]
    fn size(&self) -> usize {
        match &self.0 {
            PyChunkedNdArrayInner::F32(a) => a.len(),
            PyChunkedNdArrayInner::F64(a) => a.len(),
            PyChunkedNdArrayInner::F32View(v) => v.len(),
            PyChunkedNdArrayInner::F64View(v) => v.len(),
        }
    }

    #[getter]
    fn n_chunks(&self) -> usize {
        match &self.0 {
            PyChunkedNdArrayInner::F32(a) => a.n_batches(),
            PyChunkedNdArrayInner::F64(a) => a.n_batches(),
            PyChunkedNdArrayInner::F32View(v) => v.n_slices(),
            PyChunkedNdArrayInner::F64View(v) => v.n_slices(),
        }
    }

    #[getter]
    fn is_view(&self) -> bool {
        matches!(
            &self.0,
            PyChunkedNdArrayInner::F32View(_) | PyChunkedNdArrayInner::F64View(_)
        )
    }

    #[getter]
    fn name(&self) -> Option<String> {
        let name = match &self.0 {
            PyChunkedNdArrayInner::F32(a) => a.name.as_str(),
            PyChunkedNdArrayInner::F64(a) => a.name.as_str(),
            PyChunkedNdArrayInner::F32View(v) => v.name(),
            PyChunkedNdArrayInner::F64View(v) => v.name(),
        };
        (!name.is_empty()).then(|| name.to_string())
    }

    /// The constituent pieces in order. A window returns its zero-copy
    /// NdArray windows rather than materialising them.
    #[getter]
    fn chunks(&self) -> Vec<PyNdArray> {
        match &self.0 {
            PyChunkedNdArrayInner::F32(a) => {
                a.batches().iter().cloned().map(PyNdArray::from).collect()
            }
            PyChunkedNdArrayInner::F64(a) => {
                a.batches().iter().cloned().map(PyNdArray::from).collect()
            }
            PyChunkedNdArrayInner::F32View(v) => {
                v.slices.iter().cloned().map(PyNdArray::from).collect()
            }
            PyChunkedNdArrayInner::F64View(v) => {
                v.slices.iter().cloned().map(PyNdArray::from).collect()
            }
        }
    }

    /// One constituent piece by position, or `None` when out of range.
    fn chunk(&self, index: usize) -> Option<PyNdArray> {
        match &self.0 {
            PyChunkedNdArrayInner::F32(a) => a.batch(index).cloned().map(PyNdArray::from),
            PyChunkedNdArrayInner::F64(a) => a.batch(index).cloned().map(PyNdArray::from),
            PyChunkedNdArrayInner::F32View(v) => v.slices.get(index).cloned().map(PyNdArray::from),
            PyChunkedNdArrayInner::F64View(v) => v.slices.get(index).cloned().map(PyNdArray::from),
        }
    }

    fn __len__(&self) -> usize {
        self.shape_vec()[0]
    }

    fn __repr__(&self) -> String {
        format!(
            "ChunkedNdArray(shape={:?}, dtype={}, chunks={})",
            self.shape_vec(),
            self.dtype(),
            self.n_chunks()
        )
    }

    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let shape = self.shape_vec();
        let keys = axis_keys(key, &shape)?;
        if keys.iter().all(|key| matches!(key, AxisKey::Index(_))) {
            let indices: Vec<usize> = keys
                .iter()
                .map(|key| match key {
                    AxisKey::Index(index) => *index,
                    AxisKey::Range(_) => unreachable!(),
                })
                .collect();
            let value = match &self.0 {
                PyChunkedNdArrayInner::F32(a) => a.get(&indices) as f64,
                PyChunkedNdArrayInner::F64(a) => a.get(&indices),
                PyChunkedNdArrayInner::F32View(v) => v.get(&indices) as f64,
                PyChunkedNdArrayInner::F64View(v) => v.get(&indices),
            };
            return value.into_py_any(py);
        }

        if let AxisKey::Index(index) = keys[0] {
            let trailing = selectors(&keys[1..]);
            let array = match &self.0 {
                PyChunkedNdArrayInner::F32(a) => PyNdArray::from(a.obs(index).slice(&trailing)),
                PyChunkedNdArrayInner::F64(a) => PyNdArray::from(a.obs(index).slice(&trailing)),
                PyChunkedNdArrayInner::F32View(v) => PyNdArray::from(v.obs(index).slice(&trailing)),
                PyChunkedNdArrayInner::F64View(v) => PyNdArray::from(v.obs(index).slice(&trailing)),
            };
            return Ok(Py::new(py, array)?.into_any());
        }

        let selection = selectors(&keys);
        let view = match &self.0 {
            PyChunkedNdArrayInner::F32(a) => PyChunkedNdArray::from(a.s(&selection)),
            PyChunkedNdArrayInner::F64(a) => PyChunkedNdArray::from(a.s(&selection)),
            PyChunkedNdArrayInner::F32View(v) => PyChunkedNdArray::from(v.s(&selection)),
            PyChunkedNdArrayInner::F64View(v) => PyChunkedNdArray::from(v.s(&selection)),
        };
        Ok(Py::new(py, view)?.into_any())
    }

    /// Materialise to one compact column-major NdArray.
    fn to_ndarray(&self) -> PyNdArray {
        match &self.0 {
            PyChunkedNdArrayInner::F32(a) => PyNdArray::from((**a).clone().consolidate()),
            PyChunkedNdArrayInner::F64(a) => PyNdArray::from((**a).clone().consolidate()),
            PyChunkedNdArrayInner::F32View(v) => PyNdArray::from((**v).clone().consolidate()),
            PyChunkedNdArrayInner::F64View(v) => PyNdArray::from((**v).clone().consolidate()),
        }
    }

    /// Hand each chunk to NumPy through its own DLPack producer. The returned
    /// list preserves one tensor and one data pointer per chunk.
    fn to_numpy(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let numpy = py.import("numpy")?;
        self.chunks()
            .into_iter()
            .map(|chunk| {
                let chunk = Py::new(py, chunk)?;
                Ok(numpy.call_method1("from_dlpack", (chunk,))?.unbind())
            })
            .collect()
    }
}

impl PyChunkedNdArray {
    fn shape_vec(&self) -> Vec<usize> {
        match &self.0 {
            PyChunkedNdArrayInner::F32(a) => a.shape(),
            PyChunkedNdArrayInner::F64(a) => a.shape(),
            PyChunkedNdArrayInner::F32View(v) => v.shape(),
            PyChunkedNdArrayInner::F64View(v) => v.shape(),
        }
    }
}
