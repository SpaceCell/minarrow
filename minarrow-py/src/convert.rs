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
use minarrow::enums::array::extract_option_values64;
use minarrow::enums::time_units::TimeUnit;
#[cfg(feature = "decimal")]
use minarrow::DecimalArray;
use minarrow::{
    arr_bool_opt, arr_f32_opt, arr_f64_opt, arr_i32_opt, arr_i64_opt, arr_str32_opt, arr_u32_opt,
    arr_u64_opt, Array, ArrayV, Bitmask, CategoricalArray, DatetimeArray, Scalar, Vec64,
};

use crate::array::PyArray;
use crate::arrow_type::PyArrowType;
use pyo3::exceptions::{PyIndexError, PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyString};
use pyo3::IntoPyObjectExt;

/// Reads a Python sequence into a single 64-byte aligned `Vec64` buffer.
///
/// This is the extraction step for the constructors that accept a Python
/// sequence. Extracting through pyo3's `Vec` would allocate through the
/// global allocator and then need a second allocation and copy to reach
/// minarrow's 64-byte buffer alignment, so each element is instead extracted
/// into the aligned buffer as it is read.
///
/// Acceptance matches pyo3's `Vec` extraction. The input must satisfy the
/// Python sequence protocol and must not be a `str`, otherwise the function
/// raises `TypeError`.
fn read_sequence<'py, T>(data: &Bound<'py, PyAny>) -> PyResult<Vec64<T>>
where
    T: for<'a> FromPyObject<'a, 'py>,
    for<'a> PyErr: From<<T as FromPyObject<'a, 'py>>::Error>,
{
    if data.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err(
            "expected a sequence of values, not a str",
        ));
    }
    // pyo3 gates `extract::<Vec<T>>` on the sequence protocol, so acceptance
    // here matches it and an object passing the gate supports the length and
    // iteration below.
    if unsafe { ffi::PySequence_Check(data.as_ptr()) } == 0 {
        return Err(PyTypeError::new_err(format!(
            "expected a sequence of values, got {}",
            data.get_type().name()?
        )));
    }
    let mut values = Vec64::with_capacity(data.len().unwrap_or(0));
    for item in data.try_iter()? {
        values.push(item?.extract()?);
    }
    Ok(values)
}

/// Build a minarrow `Array` from a Python sequence, inferring the element type
/// from its values. `None` becomes null. Integers promote to float when a
/// sequence also contains floats. An empty or all-null sequence builds a float
/// array.
///
/// Each branch extracts the whole sequence in one typed pass through pyo3, so
/// there is no per-element type check. `bool` is tried before `int` because a
/// Python `bool` is an `int` subclass.
pub fn build_array(data: &Bound<'_, PyAny>) -> PyResult<Array> {
    // Reuse an existing Array as-is rather than re-inferring the dtype from
    // its values, which mistypes temporal and categorical columns. The ArrayV
    // conversion materialises a windowed array into a standalone copy.
    if let Ok(array) = data.extract::<PyRef<'_, PyArray>>() {
        return Ok(ArrayV::from(&array.0).to_array());
    }
    if let Ok(values) = read_sequence::<Option<bool>>(data) {
        if !values.iter().all(Option::is_none) {
            return Ok(arr_bool_opt!(values));
        }
    }
    if let Ok(values) = read_sequence::<Option<i64>>(data) {
        if !values.iter().all(Option::is_none) {
            return Ok(arr_i64_opt!(values));
        }
    }
    if let Ok(values) = read_sequence::<Option<f64>>(data) {
        return Ok(arr_f64_opt!(values));
    }
    if let Ok(values) = read_sequence::<Option<String>>(data) {
        return Ok(arr_str32_opt!(values));
    }
    Err(PyTypeError::new_err(
        "Array elements must be bool, int, float, str, or None",
    ))
}

/// Build a minarrow `Array` from a Python sequence coerced to `dtype`. `None`
/// becomes null. A value that does not fit the target type raises the
/// extraction's own `TypeError` or `OverflowError`.
///
/// Temporal dtypes accept integer values in the unit the dtype declares, so
/// `ArrowType::Timestamp(TimeUnit::Milliseconds, None)` over
/// `[1700000000000, ...]` builds a millisecond-precision timestamp column.
pub fn build_array_typed(data: &Bound<'_, PyAny>, dtype: &ArrowType) -> PyResult<Array> {
    macro_rules! build {
        ($t:ty, $make:ident) => {{
            Ok($make!(read_sequence::<Option<$t>>(data)?))
        }};
    }
    // Temporal dtypes build through DatetimeArray rather than the arr_*_opt
    // macros because the array stores the dtype's TimeUnit alongside its
    // values. Null handling matches the other types, with None entries
    // recorded in the null mask by extract_option_values64.
    macro_rules! build_temporal {
        ($t:ty, $make:ident, $unit:expr) => {{
            let (values, null_mask) =
                extract_option_values64(read_sequence::<Option<$t>>(data)?);
            Ok(Array::$make(DatetimeArray::<$t>::from_vec64(
                values,
                null_mask,
                Some($unit),
            )))
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
        ArrowType::Date32 => build_temporal!(i32, from_datetime_i32, TimeUnit::Days),
        ArrowType::Date64 => build_temporal!(i64, from_datetime_i64, TimeUnit::Milliseconds),
        ArrowType::Time32(unit) | ArrowType::Duration32(unit) => {
            build_temporal!(i32, from_datetime_i32, *unit)
        }
        ArrowType::Time64(unit) | ArrowType::Duration64(unit) => {
            build_temporal!(i64, from_datetime_i64, *unit)
        }
        ArrowType::Timestamp(unit, _) => build_temporal!(i64, from_datetime_i64, *unit),
        #[cfg(feature = "decimal")]
        ArrowType::Decimal32(p, s) => {
            let (values, null_mask) =
                extract_option_values64(read_sequence::<Option<i32>>(data)?);
            Ok(Array::from_decimal32(DecimalArray::<i32>::from_vec64(
                values, null_mask, *p, *s,
            )))
        }
        #[cfg(feature = "decimal")]
        ArrowType::Decimal64(p, s) => {
            let (values, null_mask) =
                extract_option_values64(read_sequence::<Option<i64>>(data)?);
            Ok(Array::from_decimal64(DecimalArray::<i64>::from_vec64(
                values, null_mask, *p, *s,
            )))
        }
        #[cfg(feature = "decimal")]
        ArrowType::Decimal128(p, s) => {
            let (values, null_mask) =
                extract_option_values64(read_sequence::<Option<i128>>(data)?);
            Ok(Array::from_decimal128(DecimalArray::<i128>::from_vec64(
                values, null_mask, *p, *s,
            )))
        }
        other => Err(PyValueError::new_err(format!(
            "dtype {} cannot be built from a Python sequence; use from_arrow instead",
            other
        ))),
    }
}

/// Parse a dtype string such as `"int32"`, `"f64"`, `"string"`, or `"categorical"`.
/// Categorical granularities beyond `UInt32` are accepted only when the matching
/// feature is compiled into the build.
///
/// Temporal dtypes include their unit in the string, so `"timestamp[ms]"`
/// parses to `Timestamp(TimeUnit::Milliseconds, None)` and `"timestamp"`
/// without a unit raises `ValueError`. A timezone cannot be written in a
/// string, so timezone-aware timestamps are constructed as an `ArrowType`
/// and passed to `dtype=` in place of a string.
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
        "date32" => ArrowType::Date32,
        "date64" => ArrowType::Date64,
        "timestamp[s]" | "datetime[s]" => ArrowType::Timestamp(TimeUnit::Seconds, None),
        "timestamp[ms]" | "datetime[ms]" => ArrowType::Timestamp(TimeUnit::Milliseconds, None),
        "timestamp[us]" | "datetime[us]" => ArrowType::Timestamp(TimeUnit::Microseconds, None),
        "timestamp[ns]" | "datetime[ns]" => ArrowType::Timestamp(TimeUnit::Nanoseconds, None),
        "timestamp" | "datetime" => {
            return Err(PyValueError::new_err(
                "a timestamp dtype names its unit, as 'timestamp[ms]' or 'datetime[ms]'. \
                 Pass an ArrowType for a timezone-aware timestamp",
            ));
        }
        "date" => {
            return Err(PyValueError::new_err(
                "a date dtype names its width, as 'date32' (days) or 'date64' (milliseconds)",
            ));
        }
        "categorical" | "category" | "cat" => {
            #[cfg(feature = "default_categorical_8")]
            {
                ArrowType::Dictionary(CategoricalIndexType::UInt8)
            }
            #[cfg(not(feature = "default_categorical_8"))]
            {
                ArrowType::Dictionary(CategoricalIndexType::UInt32)
            }
        }
        #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
        "cat32" => ArrowType::Dictionary(CategoricalIndexType::UInt32),
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
        #[cfg(feature = "decimal")]
        s if s.starts_with("decimal128") || s.starts_with("decimal64") || s.starts_with("decimal32") || s.starts_with("decimal") => {
            return parse_decimal_dtype(s);
        }
        other => return Err(PyValueError::new_err(format!("unknown dtype '{other}'"))),
    })
}

/// Parse a decimal dtype string such as `"decimal128(38,10)"` or `"decimal(10,2)"`.
///
/// Accepted forms:
/// - `decimal128(P,S)` - Decimal128 with precision P and scale S.
/// - `decimal64(P,S)` - Decimal64.
/// - `decimal32(P,S)` - Decimal32.
/// - `decimal(P,S)` - alias for Decimal128.
#[cfg(feature = "decimal")]
fn parse_decimal_dtype(s: &str) -> PyResult<ArrowType> {
    // Determine the width prefix and the remainder after it
    let (width, rest) = if s.starts_with("decimal128") {
        (128u32, &s["decimal128".len()..])
    } else if s.starts_with("decimal64") {
        (64, &s["decimal64".len()..])
    } else if s.starts_with("decimal32") {
        (32, &s["decimal32".len()..])
    } else if s.starts_with("decimal") {
        (128, &s["decimal".len()..])
    } else {
        return Err(PyValueError::new_err(format!("unknown dtype '{s}'")));
    };

    // Expect (P,S) after the prefix
    let rest = rest.trim();
    if !rest.starts_with('(') || !rest.ends_with(')') {
        return Err(PyValueError::new_err(format!(
            "decimal dtype must include precision and scale as 'decimal128(P,S)', got '{s}'"
        )));
    }
    let inner = &rest[1..rest.len() - 1];
    let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
    if parts.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "decimal dtype must have two parameters (precision, scale), got '{s}'"
        )));
    }
    let precision: u8 = parts[0].parse().map_err(|_| {
        PyValueError::new_err(format!("invalid decimal precision in '{s}'"))
    })?;
    let scale: i8 = parts[1].parse().map_err(|_| {
        PyValueError::new_err(format!("invalid decimal scale in '{s}'"))
    })?;

    Ok(match width {
        32 => ArrowType::Decimal32(precision, scale),
        64 => ArrowType::Decimal64(precision, scale),
        _ => ArrowType::Decimal128(precision, scale),
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
    let values: Vec64<Option<String>> = read_sequence::<Option<String>>(data)
        .map_err(|_| PyTypeError::new_err("categorical values must be a list of strings or None"))?;
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

/// Convert a decimal scalar to a Python `decimal.Decimal` value.
///
/// Reconstructs the human-readable decimal string from the raw unscaled integer
/// and scale, then passes it to `decimal.Decimal()` for exact conversion.
#[cfg(feature = "decimal")]
fn decimal_scalar_to_py(py: Python<'_>, raw: i128, scale: i8) -> PyResult<Py<PyAny>> {
    let formatted = format_decimal_string(raw, scale);
    let decimal_mod = py.import("decimal")?;
    let result = decimal_mod.call_method1("Decimal", (formatted,))?;
    Ok(result.unbind())
}

/// Format a decimal value as a string for `decimal.Decimal` construction.
#[cfg(feature = "decimal")]
fn format_decimal_string(raw: i128, scale: i8) -> String {
    let raw_str = format!("{}", raw);
    if scale == 0 {
        return raw_str;
    }
    if scale < 0 {
        let zeros = (-scale) as usize;
        return format!("{}{}", raw_str, "0".repeat(zeros));
    }
    let scale_usize = scale as usize;
    let (is_negative, digits) = if raw_str.starts_with('-') {
        (true, &raw_str[1..])
    } else {
        (false, raw_str.as_str())
    };
    let padded = if digits.len() <= scale_usize {
        format!("{:0>width$}", digits, width = scale_usize + 1)
    } else {
        digits.to_string()
    };
    let split_pos = padded.len() - scale_usize;
    let (int_part, frac_part) = padded.split_at(split_pos);
    if is_negative {
        format!("-{}.{}", int_part, frac_part)
    } else {
        format!("{}.{}", int_part, frac_part)
    }
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
        #[cfg(feature = "decimal")]
        Scalar::Decimal32(v, scale) => decimal_scalar_to_py(py, v as i128, scale),
        #[cfg(feature = "decimal")]
        Scalar::Decimal64(v, scale) => decimal_scalar_to_py(py, v as i128, scale),
        #[cfg(feature = "decimal")]
        Scalar::Decimal128(v, scale) => decimal_scalar_to_py(py, v, scale),
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
        #[cfg(feature = "decimal")]
        Scalar::Decimal32(v, s) => format_decimal_string(*v as i128, *s),
        #[cfg(feature = "decimal")]
        Scalar::Decimal64(v, s) => format_decimal_string(*v as i128, *s),
        #[cfg(feature = "decimal")]
        Scalar::Decimal128(v, s) => format_decimal_string(*v, *s),
        #[cfg(feature = "datetime")]
        Scalar::Datetime32(v) => v.to_string(),
        #[cfg(feature = "datetime")]
        Scalar::Datetime64(v) => v.to_string(),
        #[cfg(feature = "datetime")]
        Scalar::Interval => "interval".to_string(),
    }
}
