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

//! A 1:1 mirror of minarrow's Arrow type system for Python: `ArrowType` and its
//! parameter enums `TimeUnit`, `IntervalUnit`, `CategoricalIndexType`. The
//! variants and feature gates match `minarrow::ArrowType`, and conversions run
//! both ways so `.arrow_type` reads it and `Field` construction accepts it.
//!
//! pyo3 represents a data-carrying enum with callable variants, so a variant is
//! built with a call: `ArrowType.Int64()`, `ArrowType.Timestamp(unit, tz)`.

use minarrow::ffi::arrow_dtype::{ArrowType, CategoricalIndexType};
#[cfg(feature = "datetime")]
use minarrow::enums::time_units::{IntervalUnit, TimeUnit};
use pyo3::prelude::*;

/// The unit of a temporal type. Mirrors `minarrow::TimeUnit`.
#[cfg(feature = "datetime")]
#[pyclass(from_py_object, eq, eq_int, name = "TimeUnit", module = "minarrow")]
#[derive(Clone, Copy, PartialEq)]
pub enum PyTimeUnit {
    Seconds,
    Milliseconds,
    Microseconds,
    Nanoseconds,
    Days,
}

#[cfg(feature = "datetime")]
impl From<TimeUnit> for PyTimeUnit {
    fn from(unit: TimeUnit) -> Self {
        match unit {
            TimeUnit::Seconds => PyTimeUnit::Seconds,
            TimeUnit::Milliseconds => PyTimeUnit::Milliseconds,
            TimeUnit::Microseconds => PyTimeUnit::Microseconds,
            TimeUnit::Nanoseconds => PyTimeUnit::Nanoseconds,
            TimeUnit::Days => PyTimeUnit::Days,
        }
    }
}

#[cfg(feature = "datetime")]
impl From<PyTimeUnit> for TimeUnit {
    fn from(unit: PyTimeUnit) -> Self {
        match unit {
            PyTimeUnit::Seconds => TimeUnit::Seconds,
            PyTimeUnit::Milliseconds => TimeUnit::Milliseconds,
            PyTimeUnit::Microseconds => TimeUnit::Microseconds,
            PyTimeUnit::Nanoseconds => TimeUnit::Nanoseconds,
            PyTimeUnit::Days => TimeUnit::Days,
        }
    }
}

/// The unit of an interval type. Mirrors `minarrow::IntervalUnit`.
#[cfg(feature = "datetime")]
#[pyclass(from_py_object, eq, eq_int, name = "IntervalUnit", module = "minarrow")]
#[derive(Clone, Copy, PartialEq)]
pub enum PyIntervalUnit {
    YearMonth,
    DaysTime,
    MonthDaysNs,
}

#[cfg(feature = "datetime")]
impl From<IntervalUnit> for PyIntervalUnit {
    fn from(unit: IntervalUnit) -> Self {
        match unit {
            IntervalUnit::YearMonth => PyIntervalUnit::YearMonth,
            IntervalUnit::DaysTime => PyIntervalUnit::DaysTime,
            IntervalUnit::MonthDaysNs => PyIntervalUnit::MonthDaysNs,
        }
    }
}

#[cfg(feature = "datetime")]
impl From<PyIntervalUnit> for IntervalUnit {
    fn from(unit: PyIntervalUnit) -> Self {
        match unit {
            PyIntervalUnit::YearMonth => IntervalUnit::YearMonth,
            PyIntervalUnit::DaysTime => IntervalUnit::DaysTime,
            PyIntervalUnit::MonthDaysNs => IntervalUnit::MonthDaysNs,
        }
    }
}

/// The dictionary key width of a categorical type. Mirrors
/// `minarrow::CategoricalIndexType` under its feature gates.
#[pyclass(from_py_object, eq, eq_int, name = "CategoricalIndexType", module = "minarrow")]
#[derive(Clone, Copy, PartialEq)]
pub enum PyCategoricalIndexType {
    #[cfg(feature = "default_categorical_8")]
    UInt8,
    #[cfg(feature = "extended_categorical")]
    UInt16,
    #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
    UInt32,
    #[cfg(feature = "extended_categorical")]
    UInt64,
}

impl From<CategoricalIndexType> for PyCategoricalIndexType {
    fn from(index: CategoricalIndexType) -> Self {
        match index {
            #[cfg(feature = "default_categorical_8")]
            CategoricalIndexType::UInt8 => PyCategoricalIndexType::UInt8,
            #[cfg(feature = "extended_categorical")]
            CategoricalIndexType::UInt16 => PyCategoricalIndexType::UInt16,
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            CategoricalIndexType::UInt32 => PyCategoricalIndexType::UInt32,
            #[cfg(feature = "extended_categorical")]
            CategoricalIndexType::UInt64 => PyCategoricalIndexType::UInt64,
        }
    }
}

impl From<PyCategoricalIndexType> for CategoricalIndexType {
    fn from(index: PyCategoricalIndexType) -> Self {
        match index {
            #[cfg(feature = "default_categorical_8")]
            PyCategoricalIndexType::UInt8 => CategoricalIndexType::UInt8,
            #[cfg(feature = "extended_categorical")]
            PyCategoricalIndexType::UInt16 => CategoricalIndexType::UInt16,
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            PyCategoricalIndexType::UInt32 => CategoricalIndexType::UInt32,
            #[cfg(feature = "extended_categorical")]
            PyCategoricalIndexType::UInt64 => CategoricalIndexType::UInt64,
        }
    }
}

/// The Arrow logical type. A 1:1 mirror of `minarrow::ArrowType`, including its
/// feature gates. Construct it for a `Field`, or read it from `Array.arrow_type`.
/// pyo3 makes each variant callable, so a non-parametric type is built with a
/// call: `ArrowType.Int64()`.
#[pyclass(from_py_object, eq, name = "ArrowType", module = "minarrow")]
#[derive(Clone, PartialEq)]
pub enum PyArrowType {
    Null(),
    Boolean(),
    // The extended-width numeric variants are present in every build so the Python
    // `ArrowType` surface stays stable. A build without `extended_numeric_types`
    // upcasts them to their 32-bit form when converting into the core `ArrowType`.
    // These cannot be `cfg`-gated because pyo3's enum codegen references every
    // variant unconditionally.
    Int8(),
    Int16(),
    Int32(),
    Int64(),
    UInt8(),
    UInt16(),
    UInt32(),
    UInt64(),
    Float32(),
    Float64(),
    #[cfg(feature = "datetime")]
    Date32(),
    #[cfg(feature = "datetime")]
    Date64(),
    #[cfg(feature = "datetime")]
    Time32 { unit: PyTimeUnit },
    #[cfg(feature = "datetime")]
    Time64 { unit: PyTimeUnit },
    #[cfg(feature = "datetime")]
    Duration32 { unit: PyTimeUnit },
    #[cfg(feature = "datetime")]
    Duration64 { unit: PyTimeUnit },
    #[cfg(feature = "datetime")]
    Timestamp { unit: PyTimeUnit, tz: Option<String> },
    #[cfg(feature = "datetime")]
    Interval { unit: PyIntervalUnit },
    String(),
    #[cfg(feature = "large_string")]
    LargeString(),
    Utf8View(),
    Dictionary { index: PyCategoricalIndexType },
}

#[pymethods]
impl PyArrowType {
    fn __repr__(&self) -> String {
        format!("{}", ArrowType::from(self.clone()))
    }

    fn __str__(&self) -> String {
        format!("{}", ArrowType::from(self.clone()))
    }
}

impl From<ArrowType> for PyArrowType {
    fn from(dtype: ArrowType) -> Self {
        match dtype {
            ArrowType::Null => PyArrowType::Null(),
            ArrowType::Boolean => PyArrowType::Boolean(),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::Int8 => PyArrowType::Int8(),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::Int16 => PyArrowType::Int16(),
            ArrowType::Int32 => PyArrowType::Int32(),
            ArrowType::Int64 => PyArrowType::Int64(),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::UInt8 => PyArrowType::UInt8(),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::UInt16 => PyArrowType::UInt16(),
            ArrowType::UInt32 => PyArrowType::UInt32(),
            ArrowType::UInt64 => PyArrowType::UInt64(),
            ArrowType::Float32 => PyArrowType::Float32(),
            ArrowType::Float64 => PyArrowType::Float64(),
            #[cfg(feature = "datetime")]
            ArrowType::Date32 => PyArrowType::Date32(),
            #[cfg(feature = "datetime")]
            ArrowType::Date64 => PyArrowType::Date64(),
            #[cfg(feature = "datetime")]
            ArrowType::Time32(unit) => PyArrowType::Time32 { unit: unit.into() },
            #[cfg(feature = "datetime")]
            ArrowType::Time64(unit) => PyArrowType::Time64 { unit: unit.into() },
            #[cfg(feature = "datetime")]
            ArrowType::Duration32(unit) => PyArrowType::Duration32 { unit: unit.into() },
            #[cfg(feature = "datetime")]
            ArrowType::Duration64(unit) => PyArrowType::Duration64 { unit: unit.into() },
            #[cfg(feature = "datetime")]
            ArrowType::Timestamp(unit, tz) => PyArrowType::Timestamp { unit: unit.into(), tz },
            #[cfg(feature = "datetime")]
            ArrowType::Interval(unit) => PyArrowType::Interval { unit: unit.into() },
            ArrowType::String => PyArrowType::String(),
            #[cfg(feature = "large_string")]
            ArrowType::LargeString => PyArrowType::LargeString(),
            ArrowType::Utf8View => PyArrowType::Utf8View(),
            ArrowType::Dictionary(index) => PyArrowType::Dictionary { index: index.into() },
        }
    }
}

impl From<PyArrowType> for ArrowType {
    fn from(dtype: PyArrowType) -> Self {
        match dtype {
            PyArrowType::Null() => ArrowType::Null,
            PyArrowType::Boolean() => ArrowType::Boolean,
            #[cfg(feature = "extended_numeric_types")]
            PyArrowType::Int8() => ArrowType::Int8,
            #[cfg(not(feature = "extended_numeric_types"))]
            PyArrowType::Int8() => ArrowType::Int32,
            #[cfg(feature = "extended_numeric_types")]
            PyArrowType::Int16() => ArrowType::Int16,
            #[cfg(not(feature = "extended_numeric_types"))]
            PyArrowType::Int16() => ArrowType::Int32,
            PyArrowType::Int32() => ArrowType::Int32,
            PyArrowType::Int64() => ArrowType::Int64,
            #[cfg(feature = "extended_numeric_types")]
            PyArrowType::UInt8() => ArrowType::UInt8,
            #[cfg(not(feature = "extended_numeric_types"))]
            PyArrowType::UInt8() => ArrowType::UInt32,
            #[cfg(feature = "extended_numeric_types")]
            PyArrowType::UInt16() => ArrowType::UInt16,
            #[cfg(not(feature = "extended_numeric_types"))]
            PyArrowType::UInt16() => ArrowType::UInt32,
            PyArrowType::UInt32() => ArrowType::UInt32,
            PyArrowType::UInt64() => ArrowType::UInt64,
            PyArrowType::Float32() => ArrowType::Float32,
            PyArrowType::Float64() => ArrowType::Float64,
            #[cfg(feature = "datetime")]
            PyArrowType::Date32() => ArrowType::Date32,
            #[cfg(feature = "datetime")]
            PyArrowType::Date64() => ArrowType::Date64,
            #[cfg(feature = "datetime")]
            PyArrowType::Time32 { unit } => ArrowType::Time32(unit.into()),
            #[cfg(feature = "datetime")]
            PyArrowType::Time64 { unit } => ArrowType::Time64(unit.into()),
            #[cfg(feature = "datetime")]
            PyArrowType::Duration32 { unit } => ArrowType::Duration32(unit.into()),
            #[cfg(feature = "datetime")]
            PyArrowType::Duration64 { unit } => ArrowType::Duration64(unit.into()),
            #[cfg(feature = "datetime")]
            PyArrowType::Timestamp { unit, tz } => ArrowType::Timestamp(unit.into(), tz),
            #[cfg(feature = "datetime")]
            PyArrowType::Interval { unit } => ArrowType::Interval(unit.into()),
            PyArrowType::String() => ArrowType::String,
            #[cfg(feature = "large_string")]
            PyArrowType::LargeString() => ArrowType::LargeString,
            PyArrowType::Utf8View() => ArrowType::Utf8View,
            PyArrowType::Dictionary { index } => ArrowType::Dictionary(index.into()),
        }
    }
}
