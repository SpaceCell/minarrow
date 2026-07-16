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

//! # DLPack Capsule Interchange
//!
//! The Python-boundary half of minarrow's DLPack support: capsule
//! construction and consumption for the `__dlpack__` protocol, shared by
//! this crate's `PyNdArray` and by minarrow-py's `NdArray`, mirroring how
//! [`to_py`](crate::ffi::to_py) and [`to_rust`](crate::ffi::to_rust) host
//! the Arrow capsule glue for both crates.
//!
//! [`export_dlpack`] wraps an f32/f64 [`PyNdArrayInner`] in a legacy or
//! versioned DLPack capsule, and [`import_dlpack`] consumes any DLPack
//! producer object or raw capsule back into one. Consumed capsules rename
//! with the `used_` prefix so their destructors stand down, and
//! unconsumed capsules release the tensor through the capsule destructor.

use std::ffi::{CStr, c_void};
use std::sync::Arc;

use minarrow::{NdArray, NdArrayV};
use minarrow::ffi::dlpack::{
    DLManagedTensor, DLManagedTensorVersioned, DLPACK_FLAG_BITMASK_IS_COPIED,
    DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION, export_to_dlpack, export_to_dlpack_versioned,
    export_view_to_dlpack, export_view_to_dlpack_versioned, import_from_dlpack,
    import_from_dlpack_versioned,
};
use minarrow::traits::type_unions::Float;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

// Capsule names from the DLPack Python protocol. A consumed capsule
// renames with the `used_` prefix so its destructor does not free a
// tensor the consumer now owns.
pub static DLTENSOR: &CStr = c"dltensor";
pub static DLTENSOR_USED: &CStr = c"used_dltensor";
pub static DLTENSOR_VERSIONED: &CStr = c"dltensor_versioned";
pub static DLTENSOR_VERSIONED_USED: &CStr = c"used_dltensor_versioned";

/// The natural minarrow form behind a Python `NdArray`, covering both
/// supported element types and owned or windowed storage.
pub enum PyNdArrayInner {
    F32(Arc<NdArray<f32>>),
    F64(Arc<NdArray<f64>>),
    F32View(Arc<NdArrayV<f32>>),
    F64View(Arc<NdArrayV<f64>>),
}

impl From<NdArray<f32>> for PyNdArrayInner {
    fn from(ndarray: NdArray<f32>) -> Self {
        PyNdArrayInner::F32(Arc::new(ndarray))
    }
}

impl From<NdArray<f64>> for PyNdArrayInner {
    fn from(ndarray: NdArray<f64>) -> Self {
        PyNdArrayInner::F64(Arc::new(ndarray))
    }
}

impl From<NdArrayV<f32>> for PyNdArrayInner {
    fn from(view: NdArrayV<f32>) -> Self {
        PyNdArrayInner::F32View(Arc::new(view))
    }
}

impl From<NdArrayV<f64>> for PyNdArrayInner {
    fn from(view: NdArrayV<f64>) -> Self {
        PyNdArrayInner::F64View(Arc::new(view))
    }
}

/// Destructor for an unconsumed legacy capsule. A consumed capsule is
/// renamed `used_dltensor` and the consumer owns the tensor.
unsafe extern "C" fn dltensor_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    if unsafe { pyo3::ffi::PyCapsule_IsValid(capsule, DLTENSOR.as_ptr()) } == 1 {
        let raw = unsafe { pyo3::ffi::PyCapsule_GetPointer(capsule, DLTENSOR.as_ptr()) }
            as *mut DLManagedTensor;
        if !raw.is_null() {
            unsafe {
                if let Some(deleter) = (*raw).deleter {
                    deleter(raw);
                }
            }
        }
    }
}

/// Destructor for an unconsumed versioned capsule.
unsafe extern "C" fn dltensor_versioned_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    if unsafe { pyo3::ffi::PyCapsule_IsValid(capsule, DLTENSOR_VERSIONED.as_ptr()) } == 1 {
        let raw = unsafe { pyo3::ffi::PyCapsule_GetPointer(capsule, DLTENSOR_VERSIONED.as_ptr()) }
            as *mut DLManagedTensorVersioned;
        if !raw.is_null() {
            unsafe {
                if let Some(deleter) = (*raw).deleter {
                    deleter(raw);
                }
            }
        }
    }
}

/// The `__dlpack__` protocol body. Validates the standard keyword
/// arguments, resolves the requested ABI, and returns a capsule the
/// consumer owns.
///
/// A `max_version` of major 1 or above yields the versioned capsule with
/// the read-only flag carried. Without it, the unversioned capsule ships
/// for consumers on the pre-1.0 protocol - that capsule has no read-only
/// flag, so shared storage is copied before export. `copy=True` exports a
/// fresh compact copy in either protocol; it is always writable and is
/// flagged `IS_COPIED` on the versioned capsule.
pub fn export_dlpack(
    py: Python<'_>,
    inner: &PyNdArrayInner,
    stream: Option<&Bound<'_, PyAny>>,
    max_version: Option<(u32, u32)>,
    dl_device: Option<(i32, i32)>,
    copy: Option<bool>,
) -> PyResult<Py<PyAny>> {
    if let Some(stream) = stream {
        if !stream.is_none() {
            return Err(PyValueError::new_err("stream must be None for CPU tensors"));
        }
    }
    if let Some(device) = dl_device {
        if device != (1, 0) {
            return Err(PyValueError::new_err(format!(
                "cannot export to device {:?}, data lives on CPU (1, 0)",
                device
            )));
        }
    }
    let versioned = matches!(max_version, Some((major, _)) if major >= 1);
    let copy = copy.unwrap_or(false);

    match inner {
        PyNdArrayInner::F32(a) => {
            let source = if copy { a.apply(|v| v) } else { (**a).clone() };
            dlpack_capsule(py, source, versioned, copy)
        }
        PyNdArrayInner::F64(a) => {
            let source = if copy { a.apply(|v| v) } else { (**a).clone() };
            dlpack_capsule(py, source, versioned, copy)
        }
        PyNdArrayInner::F32View(v) => {
            if copy {
                dlpack_capsule(py, v.to_ndarray(), versioned, true)
            } else {
                dlpack_view_capsule(py, (**v).clone(), versioned)
            }
        }
        PyNdArrayInner::F64View(v) => {
            if copy {
                dlpack_capsule(py, v.to_ndarray(), versioned, true)
            } else {
                dlpack_view_capsule(py, (**v).clone(), versioned)
            }
        }
    }
}

/// Wrap an owned NdArray in a DLPack capsule of the requested ABI.
/// `copied` marks a versioned capsule with `DLPACK_FLAG_BITMASK_IS_COPIED`
/// so the consumer knows the data does not alias the exporter's memory.
pub fn dlpack_capsule<T: Float>(
    py: Python<'_>,
    source: NdArray<T>,
    versioned: bool,
    copied: bool,
) -> PyResult<Py<PyAny>> {
    if versioned {
        let raw = export_to_dlpack_versioned(source).into_raw();
        if copied {
            unsafe {
                (*raw).flags |= DLPACK_FLAG_BITMASK_IS_COPIED;
            }
        }
        versioned_capsule(py, raw)
    } else {
        let raw = export_to_dlpack(source).into_raw();
        legacy_capsule(py, raw)
    }
}

/// Wrap a windowed NdArray in a DLPack capsule of the requested ABI.
pub fn dlpack_view_capsule<T: Float>(
    py: Python<'_>,
    view: NdArrayV<T>,
    versioned: bool,
) -> PyResult<Py<PyAny>> {
    if versioned {
        versioned_capsule(py, export_view_to_dlpack_versioned(view).into_raw())
    } else {
        legacy_capsule(py, export_view_to_dlpack(view).into_raw())
    }
}

fn legacy_capsule(py: Python<'_>, raw: *mut DLManagedTensor) -> PyResult<Py<PyAny>> {
    let capsule = unsafe {
        pyo3::ffi::PyCapsule_New(
            raw as *mut c_void,
            DLTENSOR.as_ptr(),
            Some(dltensor_capsule_destructor),
        )
    };
    if capsule.is_null() {
        unsafe {
            if let Some(deleter) = (*raw).deleter {
                deleter(raw);
            }
        }
        return Err(PyErr::fetch(py));
    }
    Ok(unsafe { Bound::from_owned_ptr(py, capsule) }.unbind())
}

fn versioned_capsule(
    py: Python<'_>,
    raw: *mut DLManagedTensorVersioned,
) -> PyResult<Py<PyAny>> {
    let capsule = unsafe {
        pyo3::ffi::PyCapsule_New(
            raw as *mut c_void,
            DLTENSOR_VERSIONED.as_ptr(),
            Some(dltensor_versioned_capsule_destructor),
        )
    };
    if capsule.is_null() {
        unsafe {
            if let Some(deleter) = (*raw).deleter {
                deleter(raw);
            }
        }
        return Err(PyErr::fetch(py));
    }
    Ok(unsafe { Bound::from_owned_ptr(py, capsule) }.unbind())
}

/// Import from any DLPack producer, e.g. a NumPy or PyTorch tensor, or a
/// raw DLPack capsule. Zero-copy when the producer's buffer is 64-byte
/// aligned, otherwise the data copies into an aligned buffer.
pub fn import_dlpack(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<PyNdArrayInner> {
    let capsule = if unsafe { pyo3::ffi::PyCapsule_CheckExact(obj.as_ptr()) } == 1 {
        obj.clone()
    } else {
        let kwargs = PyDict::new(py);
        kwargs.set_item("max_version", (DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION))?;
        match obj.call_method("__dlpack__", (), Some(&kwargs)) {
            Ok(capsule) => capsule,
            // A pre-1.0 producer rejects the max_version keyword.
            Err(e) if e.is_instance_of::<PyTypeError>(py) => obj.call_method0("__dlpack__")?,
            Err(e) => return Err(e),
        }
    };

    let cap_ptr = capsule.as_ptr();
    unsafe {
        if pyo3::ffi::PyCapsule_IsValid(cap_ptr, DLTENSOR_VERSIONED.as_ptr()) == 1 {
            let raw = pyo3::ffi::PyCapsule_GetPointer(cap_ptr, DLTENSOR_VERSIONED.as_ptr())
                as *mut DLManagedTensorVersioned;
            // Ownership transfers to the import, so the capsule renames
            // first and its destructor stands down.
            pyo3::ffi::PyCapsule_SetName(cap_ptr, DLTENSOR_VERSIONED_USED.as_ptr());
            match (*raw).dl_tensor.dtype.bits {
                32 => import_from_dlpack_versioned::<f32>(raw)
                    .map(PyNdArrayInner::from)
                    .map_err(|e| PyValueError::new_err(e.to_string())),
                64 => import_from_dlpack_versioned::<f64>(raw)
                    .map(PyNdArrayInner::from)
                    .map_err(|e| PyValueError::new_err(e.to_string())),
                bits => {
                    if let Some(deleter) = (*raw).deleter {
                        deleter(raw);
                    }
                    Err(PyValueError::new_err(format!(
                        "unsupported DLPack element width {} bits, expected 32 or 64",
                        bits
                    )))
                }
            }
        } else if pyo3::ffi::PyCapsule_IsValid(cap_ptr, DLTENSOR.as_ptr()) == 1 {
            let raw =
                pyo3::ffi::PyCapsule_GetPointer(cap_ptr, DLTENSOR.as_ptr()) as *mut DLManagedTensor;
            pyo3::ffi::PyCapsule_SetName(cap_ptr, DLTENSOR_USED.as_ptr());
            match (*raw).dl_tensor.dtype.bits {
                32 => import_from_dlpack::<f32>(raw)
                    .map(PyNdArrayInner::from)
                    .map_err(|e| PyValueError::new_err(e.to_string())),
                64 => import_from_dlpack::<f64>(raw)
                    .map(PyNdArrayInner::from)
                    .map_err(|e| PyValueError::new_err(e.to_string())),
                bits => {
                    if let Some(deleter) = (*raw).deleter {
                        deleter(raw);
                    }
                    Err(PyValueError::new_err(format!(
                        "unsupported DLPack element width {} bits, expected 32 or 64",
                        bits
                    )))
                }
            }
        } else {
            Err(PyValueError::new_err(
                "expected a DLPack capsule or an object with __dlpack__",
            ))
        }
    }
}
