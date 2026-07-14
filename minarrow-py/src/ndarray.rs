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

//! `NdArray` - the native Python tensor object over minarrow.
//!
//! One Python type holds an f32 or f64 `NdArray`. The DLPack capsule
//! protocol (`__dlpack__` and `__dlpack_device__`) hands the tensor to
//! NumPy, PyTorch, JAX, TensorFlow, and CuPy without copying where the
//! target supports host memory, and `from_dlpack` accepts any DLPack
//! producer in return. Named bridge methods wrap the same capsule path
//! with the target library imported at call time. The capsule glue
//! itself lives in minarrow-pyo3's `ffi::dlpack`, alongside the Arrow
//! capsule glue this package already shares.

use minarrow::{NdArray, Table};
use minarrow_pyo3::ffi::dlpack::{export_dlpack, import_dlpack};
pub use minarrow_pyo3::ffi::dlpack::PyNdArrayInner;

use crate::table::{PyTable, PyTableInner};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// Dispatch to the inner f32 or f64 NdArray.
macro_rules! dispatch {
    ($inner:expr, |$a:ident| $body:expr) => {
        match $inner {
            PyNdArrayInner::F32($a) => $body,
            PyNdArrayInner::F64($a) => $body,
        }
    };
}

/// N-dimensional f32/f64 tensor with zero-copy DLPack interchange.
#[pyclass(name = "NdArray", module = "minarrow")]
pub struct PyNdArray(pub PyNdArrayInner);

impl From<NdArray<f32>> for PyNdArray {
    fn from(ndarray: NdArray<f32>) -> Self {
        PyNdArray(PyNdArrayInner::from(ndarray))
    }
}

impl From<NdArray<f64>> for PyNdArray {
    fn from(ndarray: NdArray<f64>) -> Self {
        PyNdArray(PyNdArrayInner::from(ndarray))
    }
}

#[pymethods]
impl PyNdArray {
    /// Build from a flat sequence of numbers, shaped column-major.
    /// `shape` defaults to one dimension covering the whole sequence.
    #[new]
    #[pyo3(signature = (data, shape=None, dtype="float64"))]
    fn new(data: Vec<f64>, shape: Option<Vec<usize>>, dtype: &str) -> PyResult<Self> {
        let shape = shape.unwrap_or_else(|| vec![data.len()]);
        if shape.is_empty() {
            return Err(PyValueError::new_err(
                "shape must have at least one dimension",
            ));
        }
        let expected: usize = shape.iter().product();
        if expected != data.len() {
            return Err(PyValueError::new_err(format!(
                "shape {:?} needs {} elements, data has {}",
                shape,
                expected,
                data.len()
            )));
        }
        match dtype {
            "float64" | "f64" => Ok(PyNdArray::from(NdArray::from_slice(&data, &shape))),
            "float32" | "f32" => {
                let data32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
                Ok(PyNdArray::from(NdArray::from_slice(&data32, &shape)))
            }
            other => Err(PyValueError::new_err(format!(
                "dtype must be 'float32' or 'float64', got '{}'",
                other
            ))),
        }
    }

    /// Dimension sizes.
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        dispatch!(&self.0, |a| PyTuple::new(py, a.shape()))
    }

    /// Element strides per dimension.
    #[getter]
    fn strides<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        dispatch!(&self.0, |a| PyTuple::new(py, a.strides()))
    }

    /// Number of dimensions.
    #[getter]
    fn ndim(&self) -> usize {
        dispatch!(&self.0, |a| a.ndim())
    }

    /// Element type name, `float32` or `float64`.
    #[getter]
    fn dtype(&self) -> &'static str {
        match &self.0 {
            PyNdArrayInner::F32(_) => "float32",
            PyNdArrayInner::F64(_) => "float64",
        }
    }

    /// Total element count.
    #[getter]
    fn size(&self) -> usize {
        dispatch!(&self.0, |a| a.len())
    }

    fn __len__(&self) -> usize {
        dispatch!(&self.0, |a| a.shape()[0])
    }

    fn __repr__(&self) -> String {
        let shape = dispatch!(&self.0, |a| a.shape().to_vec());
        format!("NdArray(shape={:?}, dtype={})", shape, self.dtype())
    }

    /// Element access by full index, e.g. `arr[1, 2]`.
    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<f64> {
        let indices: Vec<usize> = if let Ok(idx) = key.extract::<usize>() {
            vec![idx]
        } else {
            key.extract::<Vec<usize>>().map_err(|_| {
                PyTypeError::new_err("index must be an int or a tuple of ints")
            })?
        };
        let shape = dispatch!(&self.0, |a| a.shape().to_vec());
        if indices.len() != shape.len() {
            return Err(PyIndexError::new_err(format!(
                "expected {} indices for shape {:?}, got {}",
                shape.len(),
                shape,
                indices.len()
            )));
        }
        for (d, (&i, &n)) in indices.iter().zip(shape.iter()).enumerate() {
            if i >= n {
                return Err(PyIndexError::new_err(format!(
                    "index {} out of bounds for axis {} with size {}",
                    i, d, n
                )));
            }
        }
        Ok(dispatch!(&self.0, |a| a.get(&indices) as f64))
    }

    /// DLPack producer entry point. Returns a capsule the consumer owns.
    ///
    /// A `max_version` of major 1 or above yields the versioned capsule
    /// with the read-only flag carried, and consumers that support it
    /// should use it. Without it, the unversioned capsule ships for
    /// consumers on the pre-1.0 protocol - that capsule has no read-only
    /// flag, so writes through it are visible to this object and any
    /// clones, per the standard DLPack sharing convention. `copy=True`
    /// exports a fresh compact copy, which is always writable and is
    /// flagged `IS_COPIED` on the versioned capsule.
    #[pyo3(signature = (*, stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__(
        &self,
        py: Python<'_>,
        stream: Option<&Bound<'_, PyAny>>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        export_dlpack(py, &self.0, stream, max_version, dl_device, copy)
    }

    /// DLPack device query. Minarrow data lives on the CPU.
    fn __dlpack_device__(&self) -> (i32, i32) {
        (1, 0)
    }

    /// Import from any DLPack producer, e.g. a NumPy or PyTorch tensor,
    /// or a raw DLPack capsule. Zero-copy when the producer's buffer is
    /// 64-byte aligned, otherwise the data copies into an aligned buffer.
    #[staticmethod]
    fn from_dlpack(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        import_dlpack(py, obj).map(PyNdArray)
    }

    /// Hand to NumPy as an `ndarray` via the capsule protocol.
    fn to_numpy(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let numpy = py.import("numpy")?;
        Ok(numpy.call_method1("from_dlpack", (slf,))?.unbind())
    }

    /// Hand to PyTorch as a `torch.Tensor` via the capsule protocol.
    fn to_pytorch(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let torch = py.import("torch")?;
        Ok(torch.call_method1("from_dlpack", (slf,))?.unbind())
    }

    /// Hand to JAX as a `jax.Array` via the capsule protocol.
    fn to_jax(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let jax_numpy = py.import("jax.numpy")?;
        Ok(jax_numpy.call_method1("from_dlpack", (slf,))?.unbind())
    }

    /// Hand to TensorFlow as a `tf.Tensor`. TensorFlow's DLPack entry
    /// takes the capsule itself rather than the producer object.
    fn to_tensorflow(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let capsule = slf.borrow().__dlpack__(py, None, None, None, None)?;
        let dlpack = py.import("tensorflow.experimental.dlpack")?;
        Ok(dlpack.call_method1("from_dlpack", (capsule,))?.unbind())
    }

    /// Hand to CuPy via the capsule protocol. CuPy holds device memory,
    /// so this copies host data to the GPU on import.
    fn to_cupy(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let cupy = py.import("cupy")?;
        Ok(cupy.call_method1("from_dlpack", (slf,))?.unbind())
    }

    /// Convert a 2D f64 tensor to a `Table` with generated column names.
    fn to_table(&self) -> PyResult<PyTable> {
        match &self.0 {
            PyNdArrayInner::F64(a) => (**a)
                .clone()
                .to_table(None)
                .map(|t| PyTable(PyTableInner::from(t)))
                .map_err(|e| PyValueError::new_err(e.to_string())),
            PyNdArrayInner::F32(_) => Err(PyValueError::new_err(
                "to_table supports float64 tensors, rebuild the tensor as float64 first",
            )),
        }
    }
}

/// Build a PyNdArray from a table of numeric columns.
pub fn ndarray_from_table(table: &Table) -> PyResult<PyNdArray> {
    NdArray::<f64>::try_from(table)
        .map(PyNdArray::from)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}
