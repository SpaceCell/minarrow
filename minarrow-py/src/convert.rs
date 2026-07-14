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

//! Conversions at the Python binding boundary.

use minarrow::ffi::arrow_dtype::{ArrowType, CategoricalIndexType};
#[cfg(feature = "large_string")]
use minarrow::arr_str64_opt;
#[cfg(feature = "extended_numeric_types")]
use minarrow::{arr_i8_opt, arr_i16_opt, arr_u8_opt, arr_u16_opt};
use minarrow::{
    arr_bool_opt, arr_f32_opt, arr_f64_opt, arr_i32_opt, arr_i64_opt, arr_str32_opt, arr_u32_opt,
    arr_u64_opt, Array, Bitmask, CategoricalArray, Scalar, Vec64,
};

use crate::arrow_type::PyArrowType;
use pyo3::exceptions::{PyIndexError, PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBool;
use pyo3::IntoPyObjectExt;

/// Build a minarrow `Array` from a Python sequence, inferring the element type
/// from its values. `None` becomes null. Integers promote to float when a
/// sequence also contains floats. An empty or all-null sequence builds a float
/// array.
///
/// Each branch extracts the whole sequence in one typed pass through pyo3, so
/// there is no per-element type check. `bool` is tried before `int` because a
/// Python `bool` is an `int` subclass.
pub fn build_array(data: &Bound<'_, PyAny>) -> PyResult<Array> {
    if let Ok(values) = data.extract::<Vec<Option<bool>>>() {
        if !values.iter().all(Option::is_none) {
            return Ok(arr_bool_opt!(values.into_iter().collect::<Vec64<_>>()));
        }
    }
    if let Ok(values) = data.extract::<Vec<Option<i64>>>() {
        if !values.iter().all(Option::is_none) {
            return Ok(arr_i64_opt!(values.into_iter().collect::<Vec64<_>>()));
        }
    }
    if let Ok(values) = data.extract::<Vec<Option<f64>>>() {
        return Ok(arr_f64_opt!(values.into_iter().collect::<Vec64<_>>()));
    }
    if let Ok(values) = data.extract::<Vec<Option<String>>>() {
        return Ok(arr_str32_opt!(values.into_iter().collect::<Vec64<_>>()));
    }
    Err(PyTypeError::new_err(
        "Array elements must be bool, int, float, str, or None",
    ))
}

/// Build a minarrow `Array` from a Python sequence coerced to `dtype`. `None`
/// becomes null. A value that does not fit the target type raises the
/// extraction's own `TypeError` or `OverflowError`. Types that need more than a
/// flat sequence, such as categorical and temporal, are not built here.
pub fn build_array_typed(data: &Bound<'_, PyAny>, dtype: &ArrowType) -> PyResult<Array> {
    macro_rules! build {
        ($t:ty, $make:ident) => {{
            let values: Vec64<Option<$t>> =
                data.extract::<Vec<Option<$t>>>()?.into_iter().collect();
            Ok($make!(values))
        }};
    }
    match dtype {
        ArrowType::Boolean => build!(bool, arr_bool_opt),
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::Int8 => build!(i8, arr_i8_opt),
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::Int16 => build!(i16, arr_i16_opt),
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::UInt8 => build!(u8, arr_u8_opt),
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::UInt16 => build!(u16, arr_u16_opt),
        ArrowType::Int32 => build!(i32, arr_i32_opt),
        ArrowType::Int64 => build!(i64, arr_i64_opt),
        ArrowType::UInt32 => build!(u32, arr_u32_opt),
        ArrowType::UInt64 => build!(u64, arr_u64_opt),
        ArrowType::Float32 => build!(f32, arr_f32_opt),
        ArrowType::Float64 => build!(f64, arr_f64_opt),
        ArrowType::String => build!(String, arr_str32_opt),
        #[cfg(feature = "large_string")]
        ArrowType::LargeString => build!(String, arr_str64_opt),
        ArrowType::Dictionary(index) => categorical_from_values(data, index),
        other => Err(PyValueError::new_err(format!(
            "dtype {} cannot be built from a Python sequence; use from_arrow instead",
            other
        ))),
    }
}

/// Parse a dtype string such as `"int32"`, `"f64"`, `"string"`, or `"categorical"`.
/// Categorical granularities beyond `UInt32` are accepted only when the matching
/// feature is compiled into the build.
pub fn parse_dtype(name: &str) -> PyResult<ArrowType> {
    Ok(match name.trim().to_ascii_lowercase().as_str() {
        #[cfg(feature = "extended_numeric_types")]
        "int8" | "i8" => ArrowType::Int8,
        #[cfg(not(feature = "extended_numeric_types"))]
        "int8" | "i8" => ArrowType::Int32,
        #[cfg(feature = "extended_numeric_types")]
        "int16" | "i16" => ArrowType::Int16,
        #[cfg(not(feature = "extended_numeric_types"))]
        "int16" | "i16" => ArrowType::Int32,
        #[cfg(feature = "extended_numeric_types")]
        "uint8" | "u8" => ArrowType::UInt8,
        #[cfg(not(feature = "extended_numeric_types"))]
        "uint8" | "u8" => ArrowType::UInt32,
        #[cfg(feature = "extended_numeric_types")]
        "uint16" | "u16" => ArrowType::UInt16,
        #[cfg(not(feature = "extended_numeric_types"))]
        "uint16" | "u16" => ArrowType::UInt32,
        "int32" | "i32" => ArrowType::Int32,
        "int64" | "i64" => ArrowType::Int64,
        "uint32" | "u32" => ArrowType::UInt32,
        "uint64" | "u64" => ArrowType::UInt64,
        "float32" | "f32" => ArrowType::Float32,
        "float64" | "f64" => ArrowType::Float64,
        "string" | "str" | "utf8" | "str32" => ArrowType::String,
        "large_string" | "largestring" | "str64" => ArrowType::LargeString,
        "bool" | "boolean" => ArrowType::Boolean,
        #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
        "categorical" | "category" | "cat" | "cat32" => {
            ArrowType::Dictionary(CategoricalIndexType::UInt32)
        }
        "cat8" => {
            #[cfg(feature = "default_categorical_8")]
            {
                ArrowType::Dictionary(CategoricalIndexType::UInt8)
            }
            #[cfg(not(feature = "default_categorical_8"))]
            {
                return Err(PyValueError::new_err(
                    "dtype 'cat8' is not available in this build (needs the extended categorical feature)",
                ));
            }
        }
        "cat16" => {
            #[cfg(feature = "extended_categorical")]
            {
                ArrowType::Dictionary(CategoricalIndexType::UInt16)
            }
            #[cfg(not(feature = "extended_categorical"))]
            {
                return Err(PyValueError::new_err(
                    "dtype 'cat16' is not available in this build (needs the extended categorical feature)",
                ));
            }
        }
        "cat64" => {
            #[cfg(feature = "extended_categorical")]
            {
                ArrowType::Dictionary(CategoricalIndexType::UInt64)
            }
            #[cfg(not(feature = "extended_categorical"))]
            {
                return Err(PyValueError::new_err(
                    "dtype 'cat64' is not available in this build (needs the extended categorical feature)",
                ));
            }
        }
        other => return Err(PyValueError::new_err(format!("unknown dtype '{other}'"))),
    })
}

/// Resolve a `dtype` argument that may be a string or an [`ArrowType`].
pub fn resolve_dtype(dtype: &Bound<'_, PyAny>) -> PyResult<ArrowType> {
    if let Ok(name) = dtype.extract::<String>() {
        parse_dtype(&name)
    } else if let Ok(arrow_type) = dtype.extract::<PyRef<'_, PyArrowType>>() {
        Ok(ArrowType::from((*arrow_type).clone()))
    } else {
        Err(PyTypeError::new_err("dtype must be a string or an ArrowType"))
    }
}

/// Build a categorical array at the requested index width by interning a list of
/// strings into a dictionary. `None` becomes a null and the dictionary order follows
/// first appearance. A category count beyond the index width's capacity fails.
fn categorical_from_values(
    data: &Bound<'_, PyAny>,
    index: &CategoricalIndexType,
) -> PyResult<Array> {
    let values: Vec64<Option<String>> = data
        .extract::<Vec<Option<String>>>()
        .map_err(|_| PyTypeError::new_err("categorical values must be a list of strings or None"))?
        .into_iter()
        .collect();
    let mut mask = Bitmask::new_set_all(values.len(), true);
    let mut strings: Vec64<&str> = Vec64::with_capacity(values.len());
    for (row, value) in values.iter().enumerate() {
        match value {
            Some(text) => strings.push(text.as_str()),
            None => {
                strings.push("");
                mask.set(row, false);
            }
        }
    }
    let array = match index {
        #[cfg(feature = "default_categorical_8")]
        CategoricalIndexType::UInt8 => {
            CategoricalArray::<u8>::try_from_vec64(strings, Some(mask)).map(Array::from_categorical8)
        }
        #[cfg(feature = "extended_categorical")]
        CategoricalIndexType::UInt16 => CategoricalArray::<u16>::try_from_vec64(strings, Some(mask))
            .map(Array::from_categorical16),
        #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
        CategoricalIndexType::UInt32 => CategoricalArray::<u32>::try_from_vec64(strings, Some(mask))
            .map(Array::from_categorical32),
        #[cfg(feature = "extended_categorical")]
        CategoricalIndexType::UInt64 => CategoricalArray::<u64>::try_from_vec64(strings, Some(mask))
            .map(Array::from_categorical64),
        #[allow(unreachable_patterns)]
        _ => {
            return Err(PyValueError::new_err(
                "categorical index width is not available in this build",
            ));
        }
    };
    array.map_err(|_| {
        PyValueError::new_err(
            "too many distinct categories for the chosen categorical index width; \
             use a wider width such as cat32 or cat64",
        )
    })
}

/// Build a categorical array at the requested index width from integer codes that
/// index into `categories`. `None` becomes a null. A code outside the dictionary, or
/// a category count beyond the index width's capacity, raises `ValueError`.
pub fn categorical_from_codes(
    data: &Bound<'_, PyAny>,
    categories: Vec<String>,
    index: &CategoricalIndexType,
) -> PyResult<Array> {
    let codes: Vec<Option<i64>> = data
        .extract()
        .map_err(|_| PyTypeError::new_err("categorical codes must be a list of integers or None"))?;
    let category_count = categories.len();
    for code in &codes {
        if let Some(code) = code {
            if *code < 0 || *code as usize >= category_count {
                return Err(PyValueError::new_err(format!(
                    "code {code} is out of range for {category_count} categories"
                )));
            }
        }
    }
    let dictionary: Vec64<String> = categories.into_iter().collect();

    macro_rules! build_codes {
        ($t:ty, $make:ident) => {{
            if category_count > 0 {
                <$t>::try_from(category_count - 1).map_err(|_| {
                    PyValueError::new_err(
                        "too many categories for the chosen categorical index width; \
                         use a wider width such as cat32 or cat64",
                    )
                })?;
            }
            let mut indices: Vec64<$t> = Vec64::with_capacity(codes.len());
            let mut mask = Bitmask::new_set_all(codes.len(), true);
            for (row, code) in codes.iter().enumerate() {
                match code {
                    Some(code) => indices.push(*code as $t),
                    None => {
                        indices.push(0);
                        mask.set(row, false);
                    }
                }
            }
            Ok(Array::$make(CategoricalArray::<$t>::new(
                indices,
                dictionary,
                Some(mask),
            )))
        }};
    }

    match index {
        #[cfg(feature = "default_categorical_8")]
        CategoricalIndexType::UInt8 => build_codes!(u8, from_categorical8),
        #[cfg(feature = "extended_categorical")]
        CategoricalIndexType::UInt16 => build_codes!(u16, from_categorical16),
        #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
        CategoricalIndexType::UInt32 => build_codes!(u32, from_categorical32),
        #[cfg(feature = "extended_categorical")]
        CategoricalIndexType::UInt64 => build_codes!(u64, from_categorical64),
        #[allow(unreachable_patterns)]
        _ => Err(PyValueError::new_err(
            "categorical index width is not available in this build",
        )),
    }
}

/// Resolve a Python index against a length. A negative index counts back from
/// the end. An index that lands outside the range raises `IndexError`.
/// Converts a single Python value to a Minarrow `Scalar`. `None` becomes
/// `Scalar::Null`. `bool` is checked before `int` because a Python `bool` is an
/// `int` subclass. The scalar is later converted to the target array's element
/// type when it is pushed or set.
pub fn py_to_scalar(value: &Bound<'_, PyAny>) -> PyResult<Scalar> {
    if value.is_none() {
        return Ok(Scalar::Null);
    }
    if value.is_instance_of::<PyBool>() {
        return Ok(Scalar::Boolean(value.extract()?));
    }
    if let Ok(number) = value.extract::<i64>() {
        return Ok(Scalar::Int64(number));
    }
    if let Ok(number) = value.extract::<f64>() {
        return Ok(Scalar::Float64(number));
    }
    if let Ok(text) = value.extract::<String>() {
        return Ok(Scalar::String32(text));
    }
    Err(PyTypeError::new_err(
        "value must be None, bool, int, float, or str",
    ))
}

pub fn resolve_index(i: isize, len: usize) -> PyResult<usize> {
    let resolved = if i < 0 { i + len as isize } else { i };
    if resolved < 0 || resolved as usize >= len {
        return Err(PyIndexError::new_err(format!(
            "index {} is out of range for length {}",
            i, len
        )));
    }
    Ok(resolved as usize)
}

/// Coerce a minarrow `Scalar` to its Python-native value. `Null` becomes `None`.
///
/// Temporal values surface as their raw integer. Faithful
/// `datetime.date`/`time`/`datetime` coercion needs the logical unit and
/// timezone carried on the `Field` and lands with the temporal family.
pub fn scalar_to_py(py: Python<'_>, scalar: Scalar) -> PyResult<Py<PyAny>> {
    match scalar {
        Scalar::Null => Ok(py.None()),
        Scalar::Boolean(v) => v.into_py_any(py),
        #[cfg(feature = "extended_numeric_types")]
        Scalar::Int8(v) => v.into_py_any(py),
        #[cfg(feature = "extended_numeric_types")]
        Scalar::Int16(v) => v.into_py_any(py),
        Scalar::Int32(v) => v.into_py_any(py),
        Scalar::Int64(v) => v.into_py_any(py),
        #[cfg(feature = "extended_numeric_types")]
        Scalar::UInt8(v) => v.into_py_any(py),
        #[cfg(feature = "extended_numeric_types")]
        Scalar::UInt16(v) => v.into_py_any(py),
        Scalar::UInt32(v) => v.into_py_any(py),
        Scalar::UInt64(v) => v.into_py_any(py),
        Scalar::Float32(v) => v.into_py_any(py),
        Scalar::Float64(v) => v.into_py_any(py),
        Scalar::String32(v) => v.into_py_any(py),
        #[cfg(feature = "large_string")]
        Scalar::String64(v) => v.into_py_any(py),
        #[cfg(feature = "datetime")]
        Scalar::Datetime32(v) => v.into_py_any(py),
        #[cfg(feature = "datetime")]
        Scalar::Datetime64(v) => v.into_py_any(py),
        #[cfg(feature = "datetime")]
        Scalar::Interval => Err(PyNotImplementedError::new_err(
            "interval scalar access is not supported",
        )),
    }
}

/// Format a minarrow `Scalar` for an array preview. `Null` becomes `null`,
/// strings are quoted. Temporal values surface as their raw integer.
pub fn scalar_repr(scalar: &Scalar) -> String {
    match scalar {
        Scalar::Null => "null".to_string(),
        Scalar::Boolean(v) => v.to_string(),
        #[cfg(feature = "extended_numeric_types")]
        Scalar::Int8(v) => v.to_string(),
        #[cfg(feature = "extended_numeric_types")]
        Scalar::Int16(v) => v.to_string(),
        Scalar::Int32(v) => v.to_string(),
        Scalar::Int64(v) => v.to_string(),
        #[cfg(feature = "extended_numeric_types")]
        Scalar::UInt8(v) => v.to_string(),
        #[cfg(feature = "extended_numeric_types")]
        Scalar::UInt16(v) => v.to_string(),
        Scalar::UInt32(v) => v.to_string(),
        Scalar::UInt64(v) => v.to_string(),
        Scalar::Float32(v) => v.to_string(),
        Scalar::Float64(v) => v.to_string(),
        Scalar::String32(v) => format!("\"{}\"", v),
        #[cfg(feature = "large_string")]
        Scalar::String64(v) => format!("\"{}\"", v),
        #[cfg(feature = "datetime")]
        Scalar::Datetime32(v) => v.to_string(),
        #[cfg(feature = "datetime")]
        Scalar::Datetime64(v) => v.to_string(),
        #[cfg(feature = "datetime")]
        Scalar::Interval => "interval".to_string(),
    }
}
