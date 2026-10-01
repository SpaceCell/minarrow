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

#[cfg(feature = "datetime")]
impl From<&PyTimeUnit> for TimeUnit {
    fn from(unit: &PyTimeUnit) -> Self {
        (*unit).into()
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

/// Declares `PyArrowType` and its conversions to and from `minarrow::ArrowType` from the
/// variant list. `Name()` mirrors a unit `ArrowType` variant. `Name { field: Type, .. }`
/// mirrors a tuple `ArrowType` variant with members in field order.
macro_rules! define_py_arrow_type {
    ([$($variants:tt)*]) => {
        define_py_arrow_type!(@build [$($variants)*] [] [] []);
    };

    (@build [] [$($variant:tt)*] [$($into_py:tt)*] [$($from_py:tt)*]) => {
        /// The Arrow logical type. A 1:1 mirror of `minarrow::ArrowType`, including its
        /// feature gates. Construct it for a `Field`, or read it from `Array.arrow_type`.
        /// pyo3 makes each variant callable, so a non-parametric type is built with a
        /// call: `ArrowType.Int64()`.
        #[pyclass(from_py_object, eq, name = "ArrowType", module = "minarrow")]
        #[derive(Clone, PartialEq)]
        pub enum PyArrowType {
            $($variant)*
        }

        // Every field is converted through `Into`, including the ones whose Python and
        // core types are the same.
        #[allow(clippy::useless_conversion)]
        impl From<ArrowType> for PyArrowType {
            fn from(dtype: ArrowType) -> Self {
                match dtype {
                    $($into_py)*
                }
            }
        }

        #[allow(clippy::useless_conversion)]
        impl From<PyArrowType> for ArrowType {
            fn from(dtype: PyArrowType) -> Self {
                match dtype {
                    $($from_py)*
                }
            }
        }
    };

    (@build [$name:ident (), $($rest:tt)*] [$($variant:tt)*] [$($into_py:tt)*] [$($from_py:tt)*]) => {
        define_py_arrow_type!(
            @build [$($rest)*]
            [$($variant)* $name(),]
            [$($into_py)* ArrowType::$name => PyArrowType::$name(),]
            [$($from_py)* PyArrowType::$name() => ArrowType::$name,]
        );
    };

    (@build
        [$name:ident { $($field:ident : $ty:ty),+ }, $($rest:tt)*]
        [$($variant:tt)*] [$($into_py:tt)*] [$($from_py:tt)*]
    ) => {
        define_py_arrow_type!(
            @build [$($rest)*]
            [$($variant)* $name { $($field: $ty),+ },]
            [$($into_py)* ArrowType::$name($($field),+) => PyArrowType::$name { $($field: $field.into()),+ },]
            [$($from_py)* PyArrowType::$name { $($field),+ } => ArrowType::$name($($field.into()),+),]
        );
    };
}

/// Appends the 8 and 16-bit numeric variants when `extended_numeric_types` is enabled.
#[cfg(feature = "extended_numeric_types")]
macro_rules! py_arrow_type_numeric {
    ([$($acc:tt)*]) => {
        py_arrow_type_datetime!([
            $($acc)*
            Int8(), Int16(), UInt8(), UInt16(),
        ]);
    };
}

#[cfg(not(feature = "extended_numeric_types"))]
macro_rules! py_arrow_type_numeric {
    ([$($acc:tt)*]) => {
        py_arrow_type_datetime!([$($acc)*]);
    };
}

/// Appends the temporal variants when `datetime` is enabled.
#[cfg(feature = "datetime")]
macro_rules! py_arrow_type_datetime {
    ([$($acc:tt)*]) => {
        py_arrow_type_large_string!([
            $($acc)*
            Date32(), Date64(),
            Time32 { unit: PyTimeUnit },
            Time64 { unit: PyTimeUnit },
            Duration32 { unit: PyTimeUnit },
            Duration64 { unit: PyTimeUnit },
            Timestamp { unit: PyTimeUnit, tz: Option<String> },
            Interval { unit: PyIntervalUnit },
        ]);
    };
}

#[cfg(not(feature = "datetime"))]
macro_rules! py_arrow_type_datetime {
    ([$($acc:tt)*]) => {
        py_arrow_type_large_string!([$($acc)*]);
    };
}

/// Appends `LargeString` when `large_string` is enabled.
#[cfg(feature = "large_string")]
macro_rules! py_arrow_type_large_string {
    ([$($acc:tt)*]) => {
        py_arrow_type_decimal!([
            $($acc)*
            LargeString(),
        ]);
    };
}

#[cfg(not(feature = "large_string"))]
macro_rules! py_arrow_type_large_string {
    ([$($acc:tt)*]) => {
        py_arrow_type_decimal!([$($acc)*]);
    };
}

/// Appends the decimal variants when `decimal` is enabled.
#[cfg(feature = "decimal")]
macro_rules! py_arrow_type_decimal {
    ([$($acc:tt)*]) => {
        define_py_arrow_type!([
            $($acc)*
            Decimal32 { precision: u8, scale: i8 },
            Decimal64 { precision: u8, scale: i8 },
            Decimal128 { precision: u8, scale: i8 },
        ]);
    };
}

#[cfg(not(feature = "decimal"))]
macro_rules! py_arrow_type_decimal {
    ([$($acc:tt)*]) => {
        define_py_arrow_type!([$($acc)*]);
    };
}

py_arrow_type_numeric!([
    Null(),
    Boolean(),
    Int32(),
    Int64(),
    UInt32(),
    UInt64(),
    Float32(),
    Float64(),
    String(),
    Utf8View(),
    Dictionary { index: PyCategoricalIndexType },
]);

impl From<&PyArrowType> for ArrowType {
    fn from(dtype: &PyArrowType) -> Self {
        dtype.clone().into()
    }
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
