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

//! dtype surface: a 1:1 mirror of minarrow's type taxonomy.
//!
//! `DType` is the concrete array type. `TypeClass` is minarrow's `Array`
//! enum grouping. `ArrowType` is the mapped Arrow logical type. Physical
//! width rides on `Array.bit_width`. All three map straight off `minarrow`'s
//! own `ArrowType`, so the Python surface mirrors the Rust without invention.

use minarrow::ffi::arrow_dtype::{ArrowType, CategoricalIndexType};
use pyo3::prelude::*;

/// minarrow's `Array` enum grouping.
#[pyclass(from_py_object, eq, eq_int, name = "TypeClass", module = "minarrow")]
#[derive(Clone, Copy, PartialEq)]
pub enum TypeClass {
    Numeric,
    Text,
    Temporal,
    Boolean,
    Null,
}

impl TypeClass {
    /// The grouping's display name.
    pub fn name(&self) -> &'static str {
        match self {
            TypeClass::Numeric => "Numeric",
            TypeClass::Text => "Text",
            TypeClass::Temporal => "Temporal",
            TypeClass::Boolean => "Boolean",
            TypeClass::Null => "Null",
        }
    }
}

#[pymethods]
impl TypeClass {
    fn __repr__(&self) -> &'static str {
        self.name()
    }

    fn __str__(&self) -> &'static str {
        self.name()
    }
}

/// The concrete minarrow array type.
#[pyclass(from_py_object, eq, eq_int, name = "DType", module = "minarrow")]
#[derive(Clone, Copy, PartialEq)]
pub enum DType {
    Integer,
    Float,
    String,
    Categorical,
    Datetime,
    Boolean,
    Null,
}

impl DType {
    /// The dtype's display name.
    pub fn name(&self) -> &'static str {
        match self {
            DType::Integer => "Integer",
            DType::Float => "Float",
            DType::String => "String",
            DType::Categorical => "Categorical",
            DType::Datetime => "Datetime",
            DType::Boolean => "Boolean",
            DType::Null => "Null",
        }
    }
}

#[pymethods]
impl DType {
    fn __repr__(&self) -> &'static str {
        self.name()
    }

    fn __str__(&self) -> &'static str {
        self.name()
    }

    /// The grouping this dtype belongs to.
    #[getter]
    pub fn group(&self) -> TypeClass {
        match self {
            DType::Integer | DType::Float => TypeClass::Numeric,
            DType::String | DType::Categorical => TypeClass::Text,
            DType::Datetime => TypeClass::Temporal,
            DType::Boolean => TypeClass::Boolean,
            DType::Null => TypeClass::Null,
        }
    }

    #[getter]
    fn is_numeric(&self) -> bool {
        matches!(self, DType::Integer | DType::Float)
    }

    #[getter]
    fn is_temporal(&self) -> bool {
        matches!(self, DType::Datetime)
    }

    #[getter]
    fn is_text(&self) -> bool {
        matches!(self, DType::String | DType::Categorical)
    }
}

/// Map a minarrow `ArrowType` to its concrete `DType`.
pub fn dtype_from_arrow(at: &ArrowType) -> DType {
    use ArrowType::*;
    match at {
        Null => DType::Null,
        Boolean => DType::Boolean,
        Int32 | Int64 | UInt32 | UInt64 => DType::Integer,
        #[cfg(feature = "extended_numeric_types")]
        Int8 | Int16 | UInt8 | UInt16 => DType::Integer,
        Float32 | Float64 => DType::Float,
        String | Utf8View => DType::String,
        #[cfg(feature = "large_string")]
        LargeString => DType::String,
        Dictionary(_) => DType::Categorical,
        #[cfg(feature = "datetime")]
        Date32 | Date64 | Time32(_) | Time64(_) | Duration32(_) | Duration64(_)
        | Timestamp(_, _) | Interval(_) => DType::Datetime,
    }
}

/// Physical integer width in bits for a minarrow `ArrowType`: value width for
/// numerics, offset width for strings, index width for categoricals, 1 for
/// boolean, 0 where dimensionless.
pub fn width_from_arrow(at: &ArrowType) -> u32 {
    use ArrowType::*;
    match at {
        Null => 0,
        Boolean => 1,
        #[cfg(feature = "extended_numeric_types")]
        Int8 | UInt8 => 8,
        #[cfg(feature = "extended_numeric_types")]
        Int16 | UInt16 => 16,
        Int32 | UInt32 | Float32 => 32,
        Int64 | UInt64 | Float64 => 64,
        String | Utf8View => 32,
        #[cfg(feature = "large_string")]
        LargeString => 64,
        Dictionary(idx) => cat_index_width(idx),
        #[cfg(feature = "datetime")]
        Date32 | Time32(_) | Duration32(_) => 32,
        #[cfg(feature = "datetime")]
        Date64 | Time64(_) | Duration64(_) | Timestamp(_, _) => 64,
        #[cfg(feature = "datetime")]
        Interval(_) => 0,
    }
}

/// The dictionary index width in bits for a categorical type.
fn cat_index_width(idx: &CategoricalIndexType) -> u32 {
    use CategoricalIndexType::*;
    match idx {
        #[cfg(feature = "default_categorical_8")]
        UInt8 => 8,
        #[cfg(feature = "extended_categorical")]
        UInt16 => 16,
        #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
        UInt32 => 32,
        #[cfg(feature = "extended_categorical")]
        UInt64 => 64,
    }
}
