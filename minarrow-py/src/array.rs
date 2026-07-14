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

//! `Array` - the native Python array object over minarrow.
//!
//! One Python type holds an unnamed `Array`, a named `FieldArray`, or a
//! zero-copy `ArrayV` window. The data surface lives on `PyArrayInner` so the
//! standalone library here and the `astrobears` compute wheel wrap the same
//! inner and stay in lockstep. Conversions are `From` impls over minarrow's own
//! built-in conversions.
//!
//! The data is held behind `Arc`. An `Arc` is an 8-aligned pointer to the
//! 64-byte-aligned minarrow handle, so the pyclass embeds it with no extra
//! indirection, and a compute result reuses the incoming `Arc` rather than
//! reallocating.

use std::sync::Arc;

use minarrow::enums::error::MinarrowError;
use minarrow::ffi::arrow_dtype::{ArrowType, CategoricalIndexType};
use minarrow::{Array, ArrayV, Bitmask, FieldArray, RowSelection, Scalar};
#[cfg(feature = "arrow_interop")]
use minarrow_pyo3::ffi::{to_py, to_rust};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PySlice;

use crate::arrow_type::PyArrowType;
use crate::convert::{
    build_array, build_array_typed, categorical_from_codes, py_to_scalar, resolve_dtype,
    resolve_index, scalar_repr, scalar_to_py,
};
use crate::dtype::{dtype_from_arrow, width_from_arrow, DType};

/// The natural minarrow form behind a Python `Array`. Carries the whole data
/// surface so any pyclass wrapping it inherits identical behaviour.
pub enum PyArrayInner {
    Array(Arc<Array>),
    Field(Arc<FieldArray>),
    View(Arc<ArrayV>),
}

impl From<Array> for PyArrayInner {
    fn from(array: Array) -> Self {
        PyArrayInner::Array(Arc::new(array))
    }
}

impl From<FieldArray> for PyArrayInner {
    fn from(field: FieldArray) -> Self {
        PyArrayInner::Field(Arc::new(field))
    }
}

impl From<ArrayV> for PyArrayInner {
    fn from(view: ArrayV) -> Self {
        PyArrayInner::View(Arc::new(view))
    }
}

impl From<&PyArrayInner> for ArrayV {
    fn from(inner: &PyArrayInner) -> Self {
        match inner {
            PyArrayInner::Array(array) => ArrayV::from((**array).clone()),
            PyArrayInner::Field(field) => ArrayV::from(&**field),
            PyArrayInner::View(view) => (**view).clone(),
        }
    }
}

impl PyArrayInner {
    /// The Arrow logical type. A `Field`-backed array reports its `Field` dtype,
    /// which holds the full temporal logical type and timezone. An unnamed array
    /// or window reports the physical type.
    pub fn arrow_dtype(&self) -> ArrowType {
        match self {
            PyArrayInner::Array(array) => array.arrow_type(),
            PyArrayInner::Field(field) => field.field.dtype.clone(),
            PyArrayInner::View(view) => view.array.arrow_type(),
        }
    }

    /// The concrete dtype.
    pub fn dtype(&self) -> DType {
        dtype_from_arrow(&self.arrow_dtype())
    }

    /// The physical integer width in bits.
    pub fn bit_width(&self) -> u32 {
        width_from_arrow(&self.arrow_dtype())
    }

    /// The mapped Arrow logical type.
    pub fn arrow_type(&self) -> PyArrowType {
        PyArrowType::from(self.arrow_dtype())
    }

    /// The column name, or `None` for an unnamed array.
    pub fn name(&self) -> Option<String> {
        match self {
            PyArrayInner::Field(field) => Some(field.field.name.clone()),
            _ => None,
        }
    }

    /// Whether this array is a windowed view of a larger buffer.
    pub fn is_view(&self) -> bool {
        match self {
            PyArrayInner::View(view) => !view.spans_backing(),
            _ => false,
        }
    }

    /// The number of nulls.
    pub fn null_count(&self) -> usize {
        match self {
            PyArrayInner::Array(array) => array.null_count(),
            PyArrayInner::Field(field) => field.null_count,
            PyArrayInner::View(view) => view.null_count(),
        }
    }

    /// The number of elements.
    pub fn len(&self) -> usize {
        match self {
            PyArrayInner::Array(array) => array.len(),
            PyArrayInner::Field(field) => field.len(),
            PyArrayInner::View(view) => view.len(),
        }
    }

    /// Whether the array has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A consistent summary - dtype, bit_width, dtype_group, length, and null
    /// count - followed by a values preview capped at the first ten elements.
    pub fn repr(&self) -> String {
        let view: ArrayV = self.into();
        let len = view.len();
        let cap = 10;
        let mut items = Vec::with_capacity(len.min(cap));
        for i in 0..len.min(cap) {
            match view.get_scalar(i) {
                Some(scalar) => items.push(scalar_repr(&scalar)),
                None => items.push("null".to_string()),
            }
        }
        let values = if len > cap {
            format!("[{}, ...]", items.join(", "))
        } else {
            format!("[{}]", items.join(", "))
        };
        let dtype = self.dtype();
        let name = match self.name() {
            Some(field_name) => format!("name: {}, ", field_name),
            None => String::new(),
        };
        format!(
            "Array({}dtype: {}, bit_width: {}, dtype_group: {}, len: {}, nulls: {})\n{}",
            name,
            dtype.name(),
            self.bit_width(),
            dtype.group().name(),
            len,
            self.null_count(),
            values,
        )
    }

    /// Positional access. An integer reads one element, returning a Python scalar
    /// or `None` for a null. A slice or list of integers windows the array; the
    /// window is handed to `wrap_window` so each binding returns its own `Array`.
    /// Negative integers count back from the end.
    pub fn get_item(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        wrap_window: impl Fn(ArrayV) -> PyResult<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let view: ArrayV = self.into();
        let len = view.len();

        if let Ok(slice) = key.cast::<PySlice>() {
            let ind = slice.indices(len as isize)?;
            let windowed = if ind.step == 1 {
                view.r((ind.start as usize)..(ind.stop as usize))
            } else {
                let mut idxs = Vec::with_capacity(ind.slicelength);
                for k in 0..ind.slicelength {
                    idxs.push((ind.start + k as isize * ind.step) as usize);
                }
                view.r(idxs)
            };
            return wrap_window(windowed);
        }
        if let Ok(list) = key.extract::<Vec<isize>>() {
            let idxs = list
                .into_iter()
                .map(|i| resolve_index(i, len))
                .collect::<PyResult<Vec<usize>>>()?;
            return wrap_window(view.r(idxs));
        }
        if let Ok(i) = key.extract::<isize>() {
            let idx = resolve_index(i, len)?;
            return match view.get_scalar(idx) {
                Some(scalar) => scalar_to_py(py, scalar),
                None => Ok(py.None()),
            };
        }
        Err(PyTypeError::new_err(
            "index must be an int, a slice, or a list of ints",
        ))
    }

    /// Appends `value` to the array. A uniquely held owned or named array is
    /// mutated in place; a shared one is cloned once first (copy-on-write). A
    /// view is materialised to an owned array first, since a window cannot grow.
    pub fn push(&mut self, value: Scalar) -> Result<(), MinarrowError> {
        match self {
            PyArrayInner::Array(array) => Arc::make_mut(array).push(value),
            PyArrayInner::Field(field) => Arc::make_mut(field).with_array_mut(|array| array.push(value)),
            PyArrayInner::View(view) => {
                let mut owned = view.to_array();
                owned.push(value)?;
                *self = PyArrayInner::Array(Arc::new(owned));
                Ok(())
            }
        }
    }

    /// Appends a null to the array, materialising a view to an owned array first.
    pub fn push_null(&mut self) -> Result<(), MinarrowError> {
        match self {
            PyArrayInner::Array(array) => Arc::make_mut(array).push_null(),
            PyArrayInner::Field(field) => Arc::make_mut(field).with_array_mut(|array| array.push_null()),
            PyArrayInner::View(view) => {
                let mut owned = view.to_array();
                owned.push_null()?;
                *self = PyArrayInner::Array(Arc::new(owned));
                Ok(())
            }
        }
    }

    /// Sets the element at `idx` to `value`, materialising a view first.
    pub fn set(&mut self, idx: usize, value: Scalar) -> Result<(), MinarrowError> {
        match self {
            PyArrayInner::Array(array) => Arc::make_mut(array).set(idx, value),
            PyArrayInner::Field(field) => Arc::make_mut(field).with_array_mut(|array| array.set(idx, value)),
            PyArrayInner::View(view) => {
                let mut owned = view.to_array();
                owned.set(idx, value)?;
                *self = PyArrayInner::Array(Arc::new(owned));
                Ok(())
            }
        }
    }

    /// Marks the element at `idx` as null/
    /// 
    /// If the Array is backed by a view or there are
    /// multiple objects pointing to the underlying array it materialises first.
    pub fn set_null_at(&mut self, idx: usize) {
        match self {
            PyArrayInner::Array(array) => set_null_bit(Arc::make_mut(array), idx, true),
            PyArrayInner::Field(field) => {
                Arc::make_mut(field).with_array_mut(|array| set_null_bit(array, idx, true))
            }
            PyArrayInner::View(view) => {
                let mut owned = view.to_array();
                set_null_bit(&mut owned, idx, true);
                *self = PyArrayInner::Array(Arc::new(owned));
            }
        }
    }

    /// Clears any null at `idx`, marking the element present.
    pub fn clear_null_at(&mut self, idx: usize) {
        match self {
            PyArrayInner::Array(array) => set_null_bit(Arc::make_mut(array), idx, false),
            PyArrayInner::Field(field) => {
                Arc::make_mut(field).with_array_mut(|array| set_null_bit(array, idx, false))
            }
            PyArrayInner::View(_) => {}
        }
    }

    /// One boolean per element, `true` where the element is null.
    pub fn is_null(&self) -> Vec<bool> {
        match self {
            PyArrayInner::Array(array) => null_bools(array),
            PyArrayInner::Field(field) => null_bools(&field.array),
            PyArrayInner::View(view) => null_bools(&view.to_array()),
        }
    }
}

/// Sets the null mask bit at `idx`. A new all-present null mask is created when the
/// array has none. In the null mask a set bit is present and a cleared bit is null.
fn set_null_bit(array: &mut Array, idx: usize, is_null: bool) {
    let len = array.len();
    let mut null_mask = array
        .null_mask()
        .cloned()
        .unwrap_or_else(|| Bitmask::new_set_all(len, true));
    null_mask.set(idx, !is_null);
    array.set_null_mask(null_mask);
}

/// Reads the null mask as one `true`-is-null boolean per element.
fn null_bools(array: &Array) -> Vec<bool> {
    let len = array.len();
    match array.null_mask() {
        Some(null_mask) => (0..len).map(|i| !null_mask.get(i)).collect(),
        None => vec![false; len],
    }
}

/// A minarrow array exposed to Python.
#[pyclass(name = "Array", module = "minarrow")]
pub struct PyArray(pub PyArrayInner);

impl From<Array> for PyArray {
    fn from(array: Array) -> Self {
        PyArray(array.into())
    }
}

impl From<FieldArray> for PyArray {
    fn from(field: FieldArray) -> Self {
        PyArray(field.into())
    }
}

impl From<ArrayV> for PyArray {
    fn from(view: ArrayV) -> Self {
        PyArray(view.into())
    }
}

impl From<&PyArray> for ArrayV {
    fn from(py_array: &PyArray) -> Self {
        (&py_array.0).into()
    }
}

#[pymethods]
impl PyArray {
    /// Construct from a Python sequence, inferring the dtype. `None` elements
    /// become nulls. An optional `name` makes it a named column.
    #[new]
    #[pyo3(signature = (data, name=None, dtype=None, categories=None))]
    fn new(
        data: &Bound<'_, PyAny>,
        name: Option<String>,
        dtype: Option<&Bound<'_, PyAny>>,
        categories: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let array = if let Some(categories) = categories {
            let index = match dtype {
                Some(dtype) => match resolve_dtype(dtype)? {
                    ArrowType::Dictionary(index) => index,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "categories= requires a categorical dtype, got {other}"
                        )));
                    }
                },
                #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
                None => CategoricalIndexType::UInt32,
                #[cfg(all(feature = "default_categorical_8", not(feature = "extended_categorical")))]
                None => CategoricalIndexType::UInt8,
            };
            categorical_from_codes(data, categories, &index)?
        } else if let Some(dtype) = dtype {
            build_array_typed(data, &resolve_dtype(dtype)?)?
        } else {
            build_array(data)?
        };
        let inner = match name {
            Some(name) => PyArrayInner::Field(Arc::new(FieldArray::from_arr(name, array))),
            None => PyArrayInner::Array(Arc::new(array)),
        };
        Ok(PyArray(inner))
    }

    /// Import a PyArrow or Polars array via the Arrow C Data Interface, zero-copy
    /// for the primary buffers. An Arrow array carries no column name, so the
    /// result is unnamed.
    #[cfg(feature = "arrow_interop")]
    #[staticmethod]
    fn from_arrow(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let field = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            to_rust::array_to_rust(obj)
        }))
        .map_err(|_| {
            PyValueError::new_err("from_arrow: the Arrow type is not supported by this build")
        })?
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyArray(PyArrayInner::Array(Arc::new(field.array))))
    }

    /// Export to a PyArrow array via the Arrow C Data Interface.
    #[cfg(feature = "arrow_interop")]
    fn to_arrow<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let field_array = match &self.0 {
            PyArrayInner::Array(array) => FieldArray::from_arr("", (**array).clone()),
            PyArrayInner::Field(field) => (**field).clone(),
            PyArrayInner::View(view) => FieldArray::from_arr("", view.to_array()),
        };
        to_py::array_to_py(Arc::new(field_array.array), &field_array.field, py)
    }

    /// Convert to a Polars `Series` through the Arrow PyCapsule interface. A
    /// named array carries its name onto the `Series`. Requires the `polars`
    /// package.
    #[cfg(feature = "arrow_interop")]
    fn to_polars<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let name = slf.borrow().0.name();
        let polars = py.import("polars")?;
        match name {
            Some(name) => polars.call_method1("Series", (name, &slf)),
            None => polars.call_method1("Series", ("", &slf)),
        }
    }

    /// Alias for `from_arrow`, accepting any Polars object through the Arrow
    /// PyCapsule interface.
    #[cfg(feature = "arrow_interop")]
    #[staticmethod]
    fn from_polars(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::from_arrow(obj)
    }

    /// Convert to a nanoarrow `Array` through the Arrow PyCapsule interface.
    /// Requires the `nanoarrow` package.
    #[cfg(feature = "arrow_interop")]
    fn to_nanoarrow<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        py.import("nanoarrow")?.call_method1("Array", (&slf,))
    }

    /// Alias for `from_arrow`, accepting any nanoarrow object through the Arrow
    /// PyCapsule interface.
    #[cfg(feature = "arrow_interop")]
    #[staticmethod]
    fn from_nanoarrow(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::from_arrow(obj)
    }

    /// Convert to a pandas `Series` through the Arrow PyCapsule interface.
    /// Requires pandas 3.0+ (`Series.from_arrow`).
    #[cfg(feature = "arrow_interop")]
    fn to_pandas<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        py.import("pandas")?.getattr("Series")?.call_method1("from_arrow", (&slf,))
    }

    /// Alias for `from_arrow`, accepting a pandas object through the Arrow
    /// PyCapsule interface.
    #[cfg(feature = "arrow_interop")]
    #[staticmethod]
    fn from_pandas(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::from_arrow(obj)
    }

    /// Convert to a cuDF `Series` through the Arrow PyCapsule interface. Runs on
    /// GPU and requires the `cudf` package.
    #[cfg(feature = "arrow_interop")]
    fn to_cudf<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        py.import("cudf")?.getattr("Series")?.call_method1("from_arrow", (&slf,))
    }

    /// Alias for `from_arrow`, accepting a cuDF object through the Arrow
    /// PyCapsule interface.
    #[cfg(feature = "arrow_interop")]
    #[staticmethod]
    fn from_cudf(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::from_arrow(obj)
    }

    /// The column name, or `None` for an unnamed array.
    #[getter]
    fn name(&self) -> Option<String> {
        self.0.name()
    }

    /// Whether this array is a windowed view of a larger buffer.
    #[getter]
    fn is_view(&self) -> bool {
        self.0.is_view()
    }

    /// The number of nulls.
    #[getter]
    fn null_count(&self) -> usize {
        self.0.null_count()
    }

    /// The concrete dtype.
    #[getter]
    fn dtype(&self) -> DType {
        self.0.dtype()
    }

    /// The physical integer width in bits.
    #[getter]
    fn bit_width(&self) -> u32 {
        self.0.bit_width()
    }

    /// The mapped Arrow logical type.
    #[getter]
    fn arrow_type(&self) -> PyArrowType {
        self.0.arrow_type()
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    fn __repr__(&self) -> String {
        self.0.repr()
    }

    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.0.get_item(py, key, |window| {
            Ok(Py::new(py, PyArray::from(window))?.into_any())
        })
    }

    /// Appends a value to the array. When the Array is in a view state, or another
    /// object is referencing the underlying memory, it is materialised to an owned
    /// array before appending.
    ///
    /// Suited to appending a few values. To build a large array, consider
    /// constructing it from a Python sequence in one call, `Array([...])`, rather
    /// than many `push` calls, since each call crosses the Python boundary.
    fn push(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let scalar = py_to_scalar(value)?;
        self.0
            .push(scalar)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Appends a null to the array. When the Array is in a view state, or another
    /// object is referencing the underlying memory, it is materialised to an owned
    /// array before appending.
    fn push_null(&mut self) -> PyResult<()> {
        self.0
            .push_null()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Sets the element at `index` to `value`. Negative indices count from the end,
    /// so `-1` is the last element. An out-of-range index raises `IndexError`. When
    /// the Array is in a view state, or another object is referencing the underlying
    /// memory, it is materialised to an owned array before the write, leaving the
    /// source unchanged.
    fn set(&mut self, index: isize, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let idx = resolve_index(index, self.0.len())?;
        if value.is_none() {
            self.0.set_null_at(idx);
            return Ok(());
        }
        let scalar = py_to_scalar(value)?;
        self.0
            .set(idx, scalar)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.0.clear_null_at(idx);
        Ok(())
    }

    /// A list of booleans, one per element, where `True` marks a null. Mirrors the
    /// null mask.
    fn is_null(&self) -> Vec<bool> {
        self.0.is_null()
    }

    /// Export via the Arrow C Data Interface PyCapsule protocol, so any
    /// Arrow-aware library reads this array directly.
    #[cfg(feature = "arrow_interop")]
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_array__(
        &self,
        py: Python<'_>,
        requested_schema: Option<Py<PyAny>>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let _ = requested_schema;
        match &self.0 {
            PyArrayInner::Array(array) => {
                let field = FieldArray::from_arr("", (**array).clone()).field;
                to_py::array_to_capsules(array.clone(), &field, py)
            }
            PyArrayInner::Field(field) => {
                to_py::array_to_capsules(Arc::new(field.array.clone()), &field.field, py)
            }
            PyArrayInner::View(view) => {
                let field = FieldArray::from_arr("", view.array.clone()).field;
                to_py::array_view_to_capsules(view, &field, py)
            }
        }
    }
}
