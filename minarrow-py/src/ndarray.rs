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
//! One Python type holds an f32 or f64 `NdArray` or `NdArrayV`. Slicing
//! keeps the same public type and stores a zero-copy view internally. The DLPack capsule
//! protocol (`__dlpack__` and `__dlpack_device__`) hands the tensor to
//! NumPy, PyTorch, JAX, TensorFlow, and CuPy without copying where the
//! target supports host memory, and `from_dlpack` accepts any DLPack
//! producer in return. Named bridge methods wrap the same capsule path
//! with the target library imported at call time. The capsule glue
//! itself lives in minarrow-pyo3's `ffi::dlpack`, alongside the Arrow
//! capsule glue this package already shares.

use minarrow::traits::selection::DataSelector;
use minarrow::{NdArray, NdArrayV, Table};
use minarrow_pyo3::ffi::dlpack::{export_dlpack, import_dlpack};
pub use minarrow_pyo3::ffi::dlpack::PyNdArrayInner;

use crate::table::{PyTable, PyTableInner};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PySlice, PyTuple};
use pyo3::IntoPyObjectExt;

use crate::convert::resolve_index;

/// Dispatch across dtype and owned/view storage.
macro_rules! dispatch {
    ($inner:expr, |$a:ident| $body:expr) => {
        match $inner {
            PyNdArrayInner::F32($a) => $body,
            PyNdArrayInner::F64($a) => $body,
            PyNdArrayInner::F32View($a) => $body,
            PyNdArrayInner::F64View($a) => $body,
        }
    };
}

pub(crate) enum AxisKey {
    Index(usize),
    Range(std::ops::Range<usize>),
}

pub(crate) fn axis_keys(key: &Bound<'_, PyAny>, shape: &[usize]) -> PyResult<Vec<AxisKey>> {
    let items: Vec<Bound<'_, PyAny>> = if let Ok(tuple) = key.cast::<PyTuple>() {
        tuple.iter().collect()
    } else {
        vec![key.clone()]
    };
    if items.len() > shape.len() {
        return Err(PyIndexError::new_err(format!(
            "too many indices for {}-dimensional array",
            shape.len()
        )));
    }

    let mut keys = Vec::with_capacity(shape.len());
    for (axis, item) in items.iter().enumerate() {
        if let Ok(slice) = item.cast::<PySlice>() {
            let indices = slice.indices(shape[axis] as isize)?;
            if indices.step != 1 {
                return Err(PyValueError::new_err(
                    "NdArray slices currently require a step of 1",
                ));
            }
            keys.push(AxisKey::Range(
                (indices.start as usize)..(indices.stop as usize),
            ));
        } else if let Ok(index) = item.extract::<isize>() {
            keys.push(AxisKey::Index(resolve_index(index, shape[axis])?));
        } else {
            return Err(PyTypeError::new_err(
                "each index must be an int or a slice",
            ));
        }
    }
    for &size in &shape[keys.len()..] {
        keys.push(AxisKey::Range(0..size));
    }
    Ok(keys)
}

pub(crate) fn selectors(keys: &[AxisKey]) -> Vec<&dyn DataSelector> {
    keys.iter()
        .map(|key| match key {
            AxisKey::Index(index) => index as &dyn DataSelector,
            AxisKey::Range(range) => range as &dyn DataSelector,
        })
        .collect()
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

impl From<NdArrayV<f32>> for PyNdArray {
    fn from(view: NdArrayV<f32>) -> Self {
        PyNdArray(PyNdArrayInner::from(view))
    }
}

impl From<NdArrayV<f64>> for PyNdArray {
    fn from(view: NdArrayV<f64>) -> Self {
        PyNdArray(PyNdArrayInner::from(view))
    }
}

#[pymethods]
impl PyNdArray {
    /// Build from a flat sequence of numbers, shaped column-major.
    /// `shape` defaults to one dimension covering the whole sequence.
    /// `shape=[]` creates a rank-zero array from exactly one value and is
    /// distinct from `shape=[0]`, which creates an empty rank-one array.
    #[new]
    #[pyo3(signature = (data, shape=None, dtype="float64"))]
    fn new(data: Vec<f64>, shape: Option<Vec<usize>>, dtype: &str) -> PyResult<Self> {
        let shape = shape.unwrap_or_else(|| vec![data.len()]);
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
            PyNdArrayInner::F32(_) | PyNdArrayInner::F32View(_) => "float32",
            PyNdArrayInner::F64(_) | PyNdArrayInner::F64View(_) => "float64",
        }
    }

    /// Whether this object is a zero-copy window over another NdArray.
    #[getter]
    fn is_view(&self) -> bool {
        matches!(
            &self.0,
            PyNdArrayInner::F32View(_) | PyNdArrayInner::F64View(_)
        )
    }

    /// Total element count.
    #[getter]
    fn size(&self) -> usize {
        dispatch!(&self.0, |a| a.len())
    }

    fn __len__(&self) -> PyResult<usize> {
        let shape = dispatch!(&self.0, |a| a.shape());
        shape
            .first()
            .copied()
            .ok_or_else(|| PyTypeError::new_err("len() of a rank-zero NdArray"))
    }

    fn __repr__(&self) -> String {
        let shape = dispatch!(&self.0, |a| a.shape().to_vec());
        format!("NdArray(shape={:?}, dtype={})", shape, self.dtype())
    }

    /// NumPy-style positional indexing. A full integer index returns a
    /// scalar; any slice or omitted trailing axis returns another NdArray
    /// backed by a zero-copy view.
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let shape = dispatch!(&self.0, |a| a.shape().to_vec());
        let keys = axis_keys(key, &shape)?;
        if keys.iter().all(|key| matches!(key, AxisKey::Index(_))) {
            let indices: Vec<usize> = keys
                .iter()
                .map(|key| match key {
                    AxisKey::Index(index) => *index,
                    AxisKey::Range(_) => unreachable!(),
                })
                .collect();
            let value = dispatch!(&self.0, |a| a.get(&indices) as f64);
            return value.into_py_any(py);
        }

        let selection = selectors(&keys);
        let view = match &self.0 {
            PyNdArrayInner::F32(a) => PyNdArray::from(a.slice(&selection)),
            PyNdArrayInner::F64(a) => PyNdArray::from(a.slice(&selection)),
            PyNdArrayInner::F32View(v) => PyNdArray::from(v.slice(&selection)),
            PyNdArrayInner::F64View(v) => PyNdArray::from(v.slice(&selection)),
        };
        Ok(Py::new(py, view)?.into_any())
    }

    /// Return a zero-copy view with axes permuted. With no argument the
    /// axis order is reversed.
    #[pyo3(signature = (axes=None))]
    fn transpose(&self, axes: Option<Vec<usize>>) -> PyResult<Self> {
        let ndim = self.ndim();
        let axes = axes.unwrap_or_else(|| (0..ndim).rev().collect());
        if axes.len() != ndim {
            return Err(PyValueError::new_err(format!(
                "transpose expected {} axes, got {}",
                ndim,
                axes.len()
            )));
        }
        let valid = {
            let mut sorted = axes.clone();
            sorted.sort_unstable();
            sorted == (0..ndim).collect::<Vec<_>>()
        };
        if !valid {
            return Err(PyValueError::new_err(format!(
                "axes must be a permutation of 0..{}",
                ndim
            )));
        }
        Ok(match &self.0 {
            PyNdArrayInner::F32(a) => PyNdArray::from(a.as_view().permute_axes(&axes)),
            PyNdArrayInner::F64(a) => PyNdArray::from(a.as_view().permute_axes(&axes)),
            PyNdArrayInner::F32View(v) => PyNdArray::from(v.permute_axes(&axes)),
            PyNdArrayInner::F64View(v) => PyNdArray::from(v.permute_axes(&axes)),
        })
    }

    /// Axis-reversed zero-copy view.
    #[getter(T)]
    fn t(&self) -> PyResult<Self> {
        self.transpose(None)
    }

    /// Materialise this array into its own compact column-major allocation.
    fn copy(&self) -> Self {
        match &self.0 {
            PyNdArrayInner::F32(a) => PyNdArray::from(a.apply(|value| value)),
            PyNdArrayInner::F64(a) => PyNdArray::from(a.apply(|value| value)),
            PyNdArrayInner::F32View(v) => PyNdArray::from(v.to_ndarray()),
            PyNdArrayInner::F64View(v) => PyNdArray::from(v.to_ndarray()),
        }
    }

    /// DLPack producer entry point. Returns a capsule the consumer owns.
    ///
    /// A `max_version` of major 1 or above yields the versioned capsule
    /// with the read-only flag carried, and consumers that support it
    /// should use it. Without it, the unversioned capsule ships for
    /// consumers on the pre-1.0 protocol - that capsule has no read-only
    /// flag, so shared storage is copied before export. `copy=True`
    /// exports a fresh compact copy in either protocol; it is always
    /// writable and is flagged `IS_COPIED` on the versioned capsule.
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
            PyNdArrayInner::F64View(v) => v
                .to_ndarray()
                .to_table(None)
                .map(|t| PyTable(PyTableInner::from(t)))
                .map_err(|e| PyValueError::new_err(e.to_string())),
            PyNdArrayInner::F32(_) | PyNdArrayInner::F32View(_) => Err(PyValueError::new_err(
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
