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

//! `Field` and `Schema` - the named-column and table-shape descriptors, bound
//! straight onto `minarrow::Field` and `minarrow::Schema`. A `Field` carries a
//! name, an `ArrowType`, a nullability flag, and string metadata; a `Schema` is
//! an ordered list of fields plus its own metadata.

use std::collections::BTreeMap;

use minarrow::ffi::schema::Schema;
use minarrow::Field;
use pyo3::exceptions::{PyKeyError, PyTypeError};
use pyo3::prelude::*;

use crate::arrow_type::PyArrowType;
use crate::convert::resolve_index;
use crate::dtype::{dtype_from_arrow, DType};

/// A named column descriptor holding name, Arrow type, nullability, and metadata.
#[pyclass(from_py_object, name = "Field", module = "minarrow")]
#[derive(Clone)]
pub struct PyField(pub Field);

#[pymethods]
impl PyField {
    /// Construct from a name and an `ArrowType`. `metadata` is a string mapping.
    #[new]
    #[pyo3(signature = (name, dtype, nullable=true, metadata=None))]
    fn new(
        name: String,
        dtype: PyArrowType,
        nullable: bool,
        metadata: Option<BTreeMap<String, String>>,
    ) -> Self {
        PyField(Field::new(name, dtype.into(), nullable, metadata))
    }

    /// The field name.
    #[getter]
    fn name(&self) -> String {
        self.0.name.clone()
    }

    /// Whether the field admits nulls.
    #[getter]
    fn nullable(&self) -> bool {
        self.0.nullable
    }

    /// The Arrow logical type.
    #[getter]
    fn arrow_type(&self) -> PyArrowType {
        PyArrowType::from(self.0.dtype.clone())
    }

    /// The concrete dtype.
    #[getter]
    fn dtype(&self) -> DType {
        dtype_from_arrow(&self.0.dtype)
    }

    /// The string metadata mapping.
    #[getter]
    fn metadata(&self) -> BTreeMap<String, String> {
        self.0.metadata.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Field(name: {}, arrow_type: {}, nullable: {})",
            self.0.name, self.0.dtype, self.0.nullable
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0.name == other.0.name
            && self.0.dtype == other.0.dtype
            && self.0.nullable == other.0.nullable
            && self.0.metadata == other.0.metadata
    }
}

/// An ordered list of fields plus table-level metadata.
#[pyclass(from_py_object, name = "Schema", module = "minarrow")]
#[derive(Clone)]
pub struct PySchema(pub Schema);

#[pymethods]
impl PySchema {
    /// Construct from a list of `Field`s and an optional string metadata mapping.
    #[new]
    #[pyo3(signature = (fields, metadata=None))]
    fn new(fields: Vec<PyField>, metadata: Option<BTreeMap<String, String>>) -> Self {
        let fields = fields.into_iter().map(|field| field.0).collect();
        PySchema(Schema::new(fields, metadata.unwrap_or_default()))
    }

    /// The fields in order.
    #[getter]
    fn fields(&self) -> Vec<PyField> {
        self.0.fields.iter().map(|field| PyField(field.clone())).collect()
    }

    /// The field names in order.
    #[getter]
    fn names(&self) -> Vec<String> {
        self.0.fields.iter().map(|field| field.name.clone()).collect()
    }

    /// The string metadata mapping.
    #[getter]
    fn metadata(&self) -> BTreeMap<String, String> {
        self.0.metadata.clone()
    }

    fn __len__(&self) -> usize {
        self.0.fields.len()
    }

    /// A field by position or by name.
    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<PyField> {
        if let Ok(name) = key.extract::<String>() {
            return self
                .0
                .fields
                .iter()
                .find(|field| field.name == name)
                .map(|field| PyField(field.clone()))
                .ok_or_else(|| PyKeyError::new_err(format!("field '{}' not found", name)));
        }
        if let Ok(i) = key.extract::<isize>() {
            let idx = resolve_index(i, self.0.fields.len())?;
            return Ok(PyField(self.0.fields[idx].clone()));
        }
        Err(PyTypeError::new_err(
            "schema index must be a field name or position",
        ))
    }

    fn __repr__(&self) -> String {
        let fields: Vec<String> = self
            .0
            .fields
            .iter()
            .map(|field| format!("{}: {}", field.name, field.dtype))
            .collect();
        format!("Schema([{}])", fields.join(", "))
    }
}
