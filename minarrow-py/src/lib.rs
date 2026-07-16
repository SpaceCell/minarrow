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

//! # minarrow
//!
//! Minimal, focused, fast Apache Arrow for Python. Typed `Array` and `Table`
//! with a faithful dtype surface, pandas-feel indexing, and zero-copy Arrow
//! interop. Users construct, index, and inspect columnar data directly.

mod array;
mod arrow_type;
mod chunked_array;
#[cfg(feature = "ndarray")]
mod chunked_ndarray;
mod chunked_table;
mod convert;
mod dtype;
mod field;
#[cfg(feature = "ndarray")]
mod ndarray;
#[cfg(feature = "embed")]
mod pyliquid;
mod table;
#[cfg(feature = "ndarray")]
mod xarray;

use pyo3::prelude::*;

use array::PyArray;
use table::PyTable;

pub use array::PyArrayInner;
pub use arrow_type::{PyArrowType, PyCategoricalIndexType};
#[cfg(feature = "datetime")]
pub use arrow_type::{PyIntervalUnit, PyTimeUnit};
pub use chunked_array::PyChunkedArray;
#[cfg(feature = "ndarray")]
pub use chunked_ndarray::{PyChunkedNdArray, PyChunkedNdArrayInner};
pub use chunked_table::PyChunkedTable;
pub use convert::{build_array, resolve_index, scalar_to_py};
pub use dtype::{dtype_from_arrow, width_from_arrow, DType, TypeClass};
pub use field::{PyField, PySchema};
#[cfg(feature = "ndarray")]
pub use ndarray::{PyNdArray, PyNdArrayInner};
#[cfg(feature = "embed")]
pub use pyliquid::{PyInput, PyLiquid};
pub use table::{build_table, PyTableInner};
#[cfg(feature = "ndarray")]
pub use xarray::{PyXArray, PyXArrayInner};

/// Registers the `minarrow` module in the embedded interpreter's inittab so
/// `import minarrow` resolves. Kept here in the crate root, where the
/// `#[pymodule]` function and its generated definition are in scope.
#[cfg(feature = "embed")]
pub(crate) fn append_minarrow_to_inittab() {
    pyo3::append_to_inittab!(minarrow_py);
}

/// Native minarrow data objects for Python.
#[pymodule]
#[pyo3(name = "minarrow")]
fn minarrow_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "__doc__",
        "Minimal, focused, fast Apache Arrow for Python - typed Array and Table with zero-copy Arrow interop.",
    )?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_class::<PyArray>()?;
    m.add_class::<PyTable>()?;
    #[cfg(feature = "ndarray")]
    {
        m.add_class::<PyNdArray>()?;
        m.add_class::<PyChunkedNdArray>()?;
        m.add_class::<PyXArray>()?;
    }
    m.add_class::<PyChunkedArray>()?;
    m.add_class::<PyChunkedTable>()?;
    m.add_class::<PyField>()?;
    m.add_class::<PySchema>()?;
    m.add_class::<DType>()?;
    m.add_class::<TypeClass>()?;
    m.add_class::<PyArrowType>()?;
    m.add_class::<PyCategoricalIndexType>()?;
    #[cfg(feature = "datetime")]
    {
        m.add_class::<PyTimeUnit>()?;
        m.add_class::<PyIntervalUnit>()?;
    }

    Ok(())
}
