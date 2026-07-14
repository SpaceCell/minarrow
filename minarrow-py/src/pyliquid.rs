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

//! Embedded Python runtime for Minarrow data.
//!
//! [`PyLiquid`] initialises CPython once and keeps it available for calls from a
//! Rust application. A Minarrow [`Table`] or [`Array`] is exposed to Python as
//! the corresponding native `minarrow` object, and the returned Python value is
//! converted to a Minarrow [`Value`].
//!
//! Columnar values cross the boundary through the Arrow C Data Interface.
//! Aligned numeric buffers with no null mask are taken over by pointer with no
//! copy. String offsets, null masks, and buffers that are not 64-byte aligned
//! are copied into Minarrow-owned memory during import.
//!
//! ## Memory and lifetime
//!
//! A returned value is consumed while it is still alive inside the closure. The
//! exported Arrow array records an owning reference to its buffers in the Arrow
//! `private_data` field, and the imported Minarrow value holds that array, so
//! the Minarrow value owns the buffers. They sit on the producing library's C
//! or Rust heap, not the Python object heap, so Python garbage collection and
//! destruction of the source object do not free them. They are freed once, when
//! the Minarrow value drops and the Arrow release callback runs.
//!
//! PyArrow and Polars back these buffers with C++ or Rust memory and release
//! them without the interpreter, so a Minarrow value can be read and dropped
//! from any thread regardless of Python activity.
//!
//! ## Supported return values
//!
//! Python values are mapped as follows:
//!
//! - `None`, `bool`, `int`, `float` and `str` become [`Value::Scalar`].
//! - A single Arrow array becomes [`Value::Array`]. This covers `minarrow.Array`,
//!   a bare `__arrow_c_array__` object, and an Arrow C stream whose schema is not
//!   a struct, such as a Series or chunked column.
//! - A table becomes [`Value::Table`]. This covers `minarrow.Table` and an Arrow C
//!   stream whose schema is a struct, such as a RecordBatch or DataFrame.
//! - `minarrow.ChunkedTable` becomes [`Value::SuperTable`].
//! - `minarrow.ChunkedArray` becomes [`Value::SuperArray`].
//! - Lists become [`Value::VecValue`] and are classified recursively.
//! - Tuples of two to six elements become the corresponding fixed-arity tuple
//!   variant and are classified recursively.
//!
//! An empty tuple becomes `Scalar::Null`, and a one-element tuple is unwrapped.
//! Tuples longer than six elements become [`Value::VecValue`].
//!
//! NumPy `ndarray` values are not currently supported directly.
//!
//! ## Linking
//!
//! Enable the `embed` feature when embedding Python in a Rust executable. This
//! mode links against `libpython` and must not be combined with
//! `extension-module`, which builds a module intended to be loaded by an
//! existing Python interpreter.

use std::ffi::CStr;
use std::sync::Arc;
use std::sync::Once;

use minarrow::ffi::arrow_c_ffi::{ArrowArrayStream, ArrowSchema};
use minarrow::{Array, Scalar, Table, Value};
use minarrow_pyo3::ffi::to_rust;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyList, PyTuple};

use crate::array::{PyArray, PyArrayInner};
use crate::chunked_array::PyChunkedArray;
use crate::chunked_table::PyChunkedTable;
use crate::table::{PyTable, PyTableInner};

/// Resident embedded-Python runtime.
///
/// Call [`Self::start`] once during application initialisation. Modules required
/// by repeated operations can be loaded with [`Self::preimport`], and Python
/// operations are executed through [`Self::with_python`].
pub struct PyLiquid {
    /// Imported modules retained for the lifetime of the runtime.
    modules: Vec<Py<PyModule>>,
}

impl PyLiquid {
    /// Initialises CPython and registers the built-in `minarrow` module.
    ///
    /// Interpreter initialisation occurs once per process. Subsequent calls
    /// return another handle to the existing interpreter.
    pub fn start() -> Self {
        static INIT: Once = Once::new();

        INIT.call_once(|| {
            crate::append_minarrow_to_inittab();
            pyo3::prepare_freethreaded_python();
        });

        PyLiquid {
            modules: Vec::new(),
        }
    }

    /// Imports modules and retains them for subsequent Python calls.
    ///
    /// Import failures are returned immediately. The runtime is returned to
    /// allow chained initialisation.
    pub fn preimport<I, S>(mut self, names: I) -> PyResult<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Python::with_gil(|py| -> PyResult<()> {
            for name in names {
                let module = py.import(name.as_ref())?;
                self.modules.push(module.unbind());
            }
            Ok(())
        })?;

        Ok(self)
    }

    /// Passes `data` to Python, executes `f` under the GIL and converts the
    /// returned object to a Minarrow [`Value`].
    ///
    /// The closure receives the Python token and a native `minarrow` object.
    /// Returned columnar data is imported through the Arrow C Data Interface.
    /// Buffers are copied into aligned Minarrow storage where required.
    pub fn with_python<D, F>(&self, data: &D, f: F) -> PyResult<Value>
    where
        D: PyInput,
        F: for<'py> FnOnce(Python<'py>, Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>>,
    {
        Python::with_gil(|py| {
            let obj = data.to_python(py)?;
            let out = f(py, obj)?;
            classify(py, &out)
        })
    }
}

/// Converts a Minarrow value into its native Python representation.
pub trait PyInput {
    /// Creates the Python object passed to [`PyLiquid::with_python`].
    fn to_python<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>>;
}

impl PyInput for Table {
    fn to_python<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let obj = PyTable(PyTableInner::from(self.clone()));
        Ok(Bound::new(py, obj)?.into_any())
    }
}

impl PyInput for Array {
    fn to_python<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let obj = PyArray(PyArrayInner::from(self.clone()));
        Ok(Bound::new(py, obj)?.into_any())
    }
}

/// Converts a supported Python object to a Minarrow [`Value`].
///
/// Lists and tuples are processed recursively. Arrow-compatible objects are
/// imported through their C Data Interface methods.
fn classify(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Scalar(Scalar::Null));
    }

    // Python bool is a subclass of int, so it must be checked first.
    if obj.is_instance_of::<PyBool>() {
        return Ok(Value::Scalar(Scalar::Boolean(obj.extract()?)));
    }

    if let Ok(v) = obj.extract::<i64>() {
        return Ok(Value::Scalar(Scalar::Int64(v)));
    }

    if let Ok(v) = obj.extract::<f64>() {
        return Ok(Value::Scalar(Scalar::Float64(v)));
    }

    if let Ok(s) = obj.extract::<String>() {
        return Ok(Value::Scalar(Scalar::String32(s)));
    }

    // Check Minarrow classes before testing the generic Arrow interfaces so the
    // exact container variant is retained.
    if obj.cast::<PyArray>().is_ok() {
        return Ok(Value::Array(Arc::new(to_rust::array_to_rust(obj)?.array)));
    }

    if obj.cast::<PyChunkedTable>().is_ok() {
        return Ok(Value::SuperTable(Arc::new(to_rust::table_to_rust(obj)?)));
    }

    if obj.cast::<PyChunkedArray>().is_ok() {
        return Ok(Value::SuperArray(Arc::new(to_rust::chunked_array_to_rust(
            obj,
        )?)));
    }

    if obj.cast::<PyTable>().is_ok() {
        return Ok(Value::Table(Arc::new(to_rust::record_batch_to_rust(obj)?)));
    }

    // Objects from other Arrow-compatible libraries (polars, pandas, pyarrow,
    // duckdb). A stream with a struct schema is a table, a non-struct stream is a
    // column, and a bare array is a single column. A RecordBatch exposes itself
    // as a struct array through `__arrow_c_array__`, so it is routed by its stream
    // rather than that struct, which Minarrow does not import as a flat array.
    if obj.hasattr("__arrow_c_stream__")? {
        if stream_schema_is_struct(obj)? {
            return Ok(Value::Table(Arc::new(to_rust::record_batch_to_rust(obj)?)));
        }
        return Ok(Value::Array(Arc::new(to_rust::array_to_rust(obj)?.array)));
    }

    if obj.hasattr("__arrow_c_array__")? {
        return Ok(Value::Array(Arc::new(to_rust::array_to_rust(obj)?.array)));
    }

    if let Ok(list) = obj.cast::<PyList>() {
        let mut items = Vec::with_capacity(list.len());

        for item in list.iter() {
            items.push(classify(py, &item)?);
        }

        return Ok(Value::VecValue(Arc::new(items)));
    }

    if let Ok(tuple) = obj.cast::<PyTuple>() {
        return classify_tuple(py, tuple);
    }

    Err(PyTypeError::new_err(
        "with_python: unsupported Python return type",
    ))
}

/// Reports whether an object's Arrow C stream carries a struct schema, which
/// marks a table rather than a single column. The stream is peeked for its
/// schema only and released without reading any data.
fn stream_schema_is_struct(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let capsule = obj.call_method1("__arrow_c_stream__", (obj.py().None(),))?;
    let stream_ptr = unsafe {
        pyo3::ffi::PyCapsule_GetPointer(capsule.as_ptr(), c"arrow_array_stream".as_ptr())
    } as *mut ArrowArrayStream;
    if stream_ptr.is_null() {
        let _ = PyErr::take(obj.py());
        return Err(PyTypeError::new_err("invalid arrow_array_stream capsule"));
    }

    let mut schema = ArrowSchema::empty();
    let is_struct = unsafe {
        let get_schema = (*stream_ptr)
            .get_schema
            .ok_or_else(|| PyTypeError::new_err("arrow stream has no get_schema"))?;
        if get_schema(stream_ptr, &mut schema) != 0 {
            return Err(PyTypeError::new_err("arrow stream get_schema failed"));
        }
        let is_struct =
            !schema.format.is_null() && CStr::from_ptr(schema.format).to_bytes().starts_with(b"+s");
        if let Some(release) = schema.release {
            release(&mut schema);
        }
        is_struct
    };

    // `capsule` drops here, and its destructor releases the peeked stream.
    Ok(is_struct)
}

/// Converts a Python tuple to the corresponding Minarrow tuple representation.
///
/// Empty tuples become `Scalar::Null`, one-element tuples are unwrapped, and
/// tuples longer than six elements become [`Value::VecValue`].
fn classify_tuple(py: Python<'_>, tuple: &Bound<'_, PyTuple>) -> PyResult<Value> {
    let n = tuple.len();
    let mut values = Vec::with_capacity(n);

    for item in tuple.iter() {
        values.push(classify(py, &item)?);
    }

    let mut it = values.into_iter();

    Ok(match n {
        0 => Value::Scalar(Scalar::Null),
        1 => it.next().unwrap(),
        2 => Value::Tuple2(Arc::new((it.next().unwrap(), it.next().unwrap()))),
        3 => Value::Tuple3(Arc::new((
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
        ))),
        4 => Value::Tuple4(Arc::new((
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
        ))),
        5 => Value::Tuple5(Arc::new((
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
        ))),
        6 => Value::Tuple6(Arc::new((
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
        ))),
        _ => Value::VecValue(Arc::new(it.collect())),
    })
}
