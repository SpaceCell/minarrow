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

//! `XArray` - labelled NdArray data for Python.

use std::sync::Arc;

use minarrow::structs::xarray::{Axis, XArray};
use minarrow::{ArrayV, NdArray, NdArrayV};
use minarrow_pyo3::ffi::dlpack::{PyNdArrayInner, export_dlpack};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::array::{PyArray, PyArrayInner};
use crate::convert::py_to_scalar;
use crate::ndarray::{AxisKey, PyNdArray, axis_keys, selectors};

pub enum PyXArrayInner {
    F32(Arc<XArray<f32>>),
    F64(Arc<XArray<f64>>),
}

#[pyclass(name = "XArray", module = "minarrow")]
pub struct PyXArray(pub PyXArrayInner);

impl From<XArray<f32>> for PyXArray {
    fn from(array: XArray<f32>) -> Self {
        Self(PyXArrayInner::F32(Arc::new(array)))
    }
}

impl From<XArray<f64>> for PyXArray {
    fn from(array: XArray<f64>) -> Self {
        Self(PyXArrayInner::F64(Arc::new(array)))
    }
}

fn xarray_f32(data: &PyNdArrayInner, dims: &[String]) -> Option<XArray<f32>> {
    let axes = || dims.iter().map(Axis::named).collect();
    match data {
        PyNdArrayInner::F32(a) => {
            let names: Vec<&str> = dims.iter().map(String::as_str).collect();
            Some(XArray::new((**a).clone(), &names))
        }
        PyNdArrayInner::F32View(v) => Some(XArray::from_view((**v).clone(), axes())),
        _ => None,
    }
}

fn xarray_f64(data: &PyNdArrayInner, dims: &[String]) -> Option<XArray<f64>> {
    let axes = || dims.iter().map(Axis::named).collect();
    match data {
        PyNdArrayInner::F64(a) => {
            let names: Vec<&str> = dims.iter().map(String::as_str).collect();
            Some(XArray::new((**a).clone(), &names))
        }
        PyNdArrayInner::F64View(v) => Some(XArray::from_view((**v).clone(), axes())),
        _ => None,
    }
}

fn assign_coords<T: minarrow::Float>(
    array: &mut XArray<T>,
    coords: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    let Some(coords) = coords else {
        return Ok(());
    };
    for (name, values) in coords.iter() {
        let name: String = name.extract()?;
        let values: PyRef<'_, PyArray> = values.extract()?;
        let dim = array.try_dim(&name).ok_or_else(|| {
            PyValueError::new_err(format!("coordinate name '{name}' is not a dimension"))
        })?;
        let expected = array.shape()[dim];
        let actual = values.0.len();
        if actual != expected {
            return Err(PyValueError::new_err(format!(
                "coordinate '{name}' has length {actual}, expected {expected}"
            )));
        }
        array.assign_coords(&name, ArrayV::from(&values.0).to_array());
    }
    Ok(())
}

#[pymethods]
impl PyXArray {
    /// Label an NdArray with dimension names and optional coordinate arrays.
    #[new]
    #[pyo3(signature = (data, dims, coords=None))]
    fn new(
        data: PyRef<'_, PyNdArray>,
        dims: Vec<String>,
        coords: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let ndim = match &data.0 {
            PyNdArrayInner::F32(a) => a.ndim(),
            PyNdArrayInner::F64(a) => a.ndim(),
            PyNdArrayInner::F32View(v) => v.ndim(),
            PyNdArrayInner::F64View(v) => v.ndim(),
        };
        if dims.len() != ndim {
            return Err(PyValueError::new_err(format!(
                "XArray needs {ndim} dimension names, got {}",
                dims.len()
            )));
        }
        let mut unique = dims.clone();
        unique.sort();
        unique.dedup();
        if unique.len() != dims.len() {
            return Err(PyValueError::new_err(
                "XArray dimension names must be unique",
            ));
        }
        if let Some(mut array) = xarray_f32(&data.0, &dims) {
            assign_coords(&mut array, coords)?;
            return Ok(array.into());
        }
        let mut array = xarray_f64(&data.0, &dims).ok_or_else(|| {
            PyValueError::new_err("XArray data must have dtype float32 or float64")
        })?;
        assign_coords(&mut array, coords)?;
        Ok(array.into())
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        match &self.0 {
            PyXArrayInner::F32(a) => PyTuple::new(py, a.shape()),
            PyXArrayInner::F64(a) => PyTuple::new(py, a.shape()),
        }
    }

    #[getter]
    fn ndim(&self) -> usize {
        match &self.0 {
            PyXArrayInner::F32(a) => a.ndim(),
            PyXArrayInner::F64(a) => a.ndim(),
        }
    }

    #[getter]
    fn size(&self) -> usize {
        match &self.0 {
            PyXArrayInner::F32(a) => a.len(),
            PyXArrayInner::F64(a) => a.len(),
        }
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        match &self.0 {
            PyXArrayInner::F32(_) => "float32",
            PyXArrayInner::F64(_) => "float64",
        }
    }

    #[getter]
    fn dims(&self) -> Vec<String> {
        match &self.0 {
            PyXArrayInner::F32(a) => a.dim_names().into_iter().map(str::to_string).collect(),
            PyXArrayInner::F64(a) => a.dim_names().into_iter().map(str::to_string).collect(),
        }
    }

    #[getter]
    fn coords<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        match &self.0 {
            PyXArrayInner::F32(a) => {
                for axis in a.axes() {
                    if let Some(coords) = &axis.coords {
                        result.set_item(
                            &axis.name,
                            Py::new(py, PyArray(PyArrayInner::from(coords.clone())))?,
                        )?;
                    }
                }
            }
            PyXArrayInner::F64(a) => {
                for axis in a.axes() {
                    if let Some(coords) = &axis.coords {
                        result.set_item(
                            &axis.name,
                            Py::new(py, PyArray(PyArrayInner::from(coords.clone())))?,
                        )?;
                    }
                }
            }
        }
        Ok(result)
    }

    /// The underlying data as an NdArray. If this XArray is itself a
    /// selection, the returned NdArray remains a zero-copy view.
    #[getter]
    fn data(&self) -> PyNdArray {
        match &self.0 {
            PyXArrayInner::F32(a) if a.is_owned() => PyNdArray::from((**a).clone().into_ndarray()),
            PyXArrayInner::F64(a) if a.is_owned() => PyNdArray::from((**a).clone().into_ndarray()),
            PyXArrayInner::F32(a) => PyNdArray::from(a.as_view()),
            PyXArrayInner::F64(a) => PyNdArray::from(a.as_view()),
        }
    }

    fn __len__(&self) -> PyResult<usize> {
        let shape = match &self.0 {
            PyXArrayInner::F32(a) => a.shape(),
            PyXArrayInner::F64(a) => a.shape(),
        };
        shape
            .first()
            .copied()
            .ok_or_else(|| PyTypeError::new_err("len() of a rank-zero XArray"))
    }

    fn __repr__(&self) -> String {
        format!(
            "XArray(shape={:?}, dims={:?}, dtype={})",
            match &self.0 {
                PyXArrayInner::F32(a) => a.shape(),
                PyXArrayInner::F64(a) => a.shape(),
            },
            self.dims(),
            self.dtype()
        )
    }

    /// Positional selection with NdArray indexing semantics, preserving the
    /// names and coordinates of axes that remain.
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let shape = match &self.0 {
            PyXArrayInner::F32(a) => a.shape(),
            PyXArrayInner::F64(a) => a.shape(),
        };
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
                PyXArrayInner::F32(a) => a.get(&indices) as f64,
                PyXArrayInner::F64(a) => a.get(&indices),
            };
            return value.into_py_any(py);
        }

        let selection = selectors(&keys);
        let result = match &self.0 {
            PyXArrayInner::F32(a) => {
                let names = a.dim_names();
                let named: Vec<(&str, &dyn minarrow::traits::selection::DataSelector)> =
                    names.into_iter().zip(selection).collect();
                PyXArray::from(a.select(&named))
            }
            PyXArrayInner::F64(a) => {
                let names = a.dim_names();
                let named: Vec<(&str, &dyn minarrow::traits::selection::DataSelector)> =
                    names.into_iter().zip(selection).collect();
                PyXArray::from(a.select(&named))
            }
        };
        Ok(Py::new(py, result)?.into_any())
    }

    /// Exact coordinate selection.
    ///
    /// Behaviour:
    /// - Each keyword is a dimension name
    /// - Selected dimensions collapse in the result.
    #[pyo3(signature = (**indexers))]
    fn sel(&self, indexers: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let indexers = indexers.ok_or_else(|| PyValueError::new_err("sel needs an indexer"))?;
        match &self.0 {
            PyXArrayInner::F32(a) => {
                let mut result = (**a).clone();
                for (name, value) in indexers.iter() {
                    result = result
                        .try_at(&name.extract::<String>()?, py_to_scalar(&value)?)
                        .map_err(|e| PyIndexError::new_err(e.to_string()))?;
                }
                Ok(result.into())
            }
            PyXArrayInner::F64(a) => {
                let mut result = (**a).clone();
                for (name, value) in indexers.iter() {
                    result = result
                        .try_at(&name.extract::<String>()?, py_to_scalar(&value)?)
                        .map_err(|e| PyIndexError::new_err(e.to_string()))?;
                }
                Ok(result.into())
            }
        }
    }

    /// Inclusive coordinate range selection on one dimension.
    fn between(
        &self,
        dim: &str,
        low: &Bound<'_, PyAny>,
        high: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let low = py_to_scalar(low)?;
        let high = py_to_scalar(high)?;
        match &self.0 {
            PyXArrayInner::F32(a) => a
                .try_between(dim, low, high)
                .map(PyXArray::from)
                .map_err(|e| PyIndexError::new_err(e.to_string())),
            PyXArrayInner::F64(a) => a
                .try_between(dim, low, high)
                .map(PyXArray::from)
                .map_err(|e| PyIndexError::new_err(e.to_string())),
        }
    }

    /// Closest coordinate selection on one numeric or datetime dimension.
    fn nearest(&self, dim: &str, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let value = py_to_scalar(value)?;
        match &self.0 {
            PyXArrayInner::F32(a) => a
                .try_nearest(dim, value)
                .map(PyXArray::from)
                .map_err(|e| PyIndexError::new_err(e.to_string())),
            PyXArrayInner::F64(a) => a
                .try_nearest(dim, value)
                .map(PyXArray::from)
                .map_err(|e| PyIndexError::new_err(e.to_string())),
        }
    }

    #[pyo3(signature = (*, stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__(
        &self,
        py: Python<'_>,
        stream: Option<&Bound<'_, PyAny>>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        match &self.0 {
            PyXArrayInner::F32(a) => export_dlpack(
                py,
                &PyNdArrayInner::from(a.as_view()),
                stream,
                max_version,
                dl_device,
                copy,
            ),
            PyXArrayInner::F64(a) => export_dlpack(
                py,
                &PyNdArrayInner::from(a.as_view()),
                stream,
                max_version,
                dl_device,
                copy,
            ),
        }
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (1, 0)
    }

    fn to_numpy(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let numpy = py.import("numpy")?;
        Ok(numpy.call_method1("from_dlpack", (slf,))?.unbind())
    }
}

impl From<NdArray<f32>> for PyXArray {
    fn from(array: NdArray<f32>) -> Self {
        XArray::from_ndarray(array).into()
    }
}

impl From<NdArray<f64>> for PyXArray {
    fn from(array: NdArray<f64>) -> Self {
        XArray::from_ndarray(array).into()
    }
}

impl From<NdArrayV<f32>> for PyXArray {
    fn from(view: NdArrayV<f32>) -> Self {
        let axes = (0..view.ndim())
            .map(|i| Axis::named(format!("dim_{i}")))
            .collect();
        XArray::from_view(view, axes).into()
    }
}

impl From<NdArrayV<f64>> for PyXArray {
    fn from(view: NdArrayV<f64>) -> Self {
        let axes = (0..view.ndim())
            .map(|i| Axis::named(format!("dim_{i}")))
            .collect();
        XArray::from_view(view, axes).into()
    }
}
