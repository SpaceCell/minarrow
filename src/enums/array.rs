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

//! # **Array Module** - *Main High-Level Array Type*
//!
//! `Array` is the primary unified container for all array types in Minarrow.
//!   
//! ## Features:
//! - direct variant access to numeric, temporal, text, and other array categories
//! - zero-cost casts when the contained type is known
//! - lossless conversions between compatible array types
//! - simplifies function signatures by allowing `impl Into<Array>`
//! - centralises dispatch for all array operations
//! - preserves SIMD-aligned buffers and metadata across variants.

use std::any::TypeId;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

#[cfg(any(feature = "cast_arrow", feature = "cast_polars"))]
use crate::ffi::schema::Schema;
#[cfg(feature = "cast_arrow")]
use arrow::array::ArrayRef;

#[cfg(feature = "views")]
use crate::ArrayV;
#[cfg(feature = "views")]
use crate::ArrayVT;
#[cfg(feature = "datetime")]
use crate::DatetimeArray;
#[cfg(feature = "chunked")]
use crate::SuperArray;
#[cfg(feature = "datetime")]
use crate::TemporalArray;
use crate::enums::error::MinarrowError;
use crate::enums::shape_dim::ShapeDim;
use crate::ffi::arrow_dtype::{ArrowType, CategoricalIndexType};
use crate::traits::{concatenate::Concatenate, shape::Shape};
use crate::utils::{float_to_text_array, int_to_text_array};
use crate::{
    Bitmask, BooleanArray, CategoricalArray, Field, FieldArray, FloatArray, IntegerArray,
    MaskedArray, NumericArray, StringArray, TextArray, Vec64, match_array,
};

/// # Array
///
/// Standard `Array` type. Wrap in a `FieldArray` when using inside a `Table`
/// or as a standalone value requiring tagged metadata.
///
/// ## Overview
/// The dual-enum approach may look verbose but works well in practice:
///
/// - Enables clean function signatures with direct access to concrete types
///   (e.g. `&NumericArray`), supporting trait-aligned dispatch without
///   exhaustive matches at every call site.
/// - Supports ergonomic categorisation: functions typically match on the
///   outer enum for broad category handling *(numeric, text, temporal, boolean)*,
///   while allowing inner variant matching for precise type handling.
/// - The focused typeset (no nested types) helps keeps enum size efficient
///   as memory is allocated for the largest variant.
///
/// ## Usage
/// Functions can accept references tailored to the intended match granularity:
///
/// - `&IntegerArray`: direct reference to the inner type e.g., `arr.num().i64()`.
/// - `&NumericArray`: any numeric type via `arr.num()`.
/// - `&Array`: match on categories or individual types.
///
/// ## Benefits
/// - No heap allocation or runtime indirection - all enum variants are inline
///   with minimal discriminant cost.
/// - Unified call sites with compiler-enforced type safety.
/// - Easy casting to inner types (e.g., `.str()` for strings).
/// - Supports aggressive compiler inlining, unlike approaches relying on
///   dynamic dispatch and downcasting.
///
/// ## Trade-offs
/// - Adds ~30–100 ns latency compared to direct inner type calls - only
///   noticeable in extreme low-latency contexts such as HFT.
/// - Requires enum matching at dispatch sites compared to direct inner type usage.

/// ## Examples
/// ```rust
/// use minarrow::{
///     Array, IntegerArray, NumericArray, arr_bool, arr_f64, arr_i32, arr_i64,
///     arr_str32, vec64
/// };
///
/// // Fast macro construction
/// let int_arr = arr_i32![1, 2, 3, 4];
/// let float_arr = arr_f64![0.5, 1.5, 2.5];
/// let bool_arr = arr_bool![true, false, true];
/// let str_arr = arr_str32!["a", "b", "c"];
///
/// assert_eq!(int_arr.len(), 4);
/// assert_eq!(str_arr.len(), 3);
///
/// // Manual construction
/// let int = IntegerArray::<i64>::from_slice(&[100, 200]);
/// let wrapped: NumericArray = NumericArray::Int64(std::sync::Arc::new(int));
/// let array = Array::NumericArray(wrapped);
/// ```
#[repr(C, align(64))]
#[derive(PartialEq, Clone, Debug, Default)]
pub enum Array {
    NumericArray(NumericArray),
    TextArray(TextArray),
    #[cfg(feature = "datetime")]
    TemporalArray(TemporalArray),
    BooleanArray(Arc<BooleanArray<()>>),
    #[default]
    Null, // Default Marker for mem::take
}

impl Array {
    /// Creates an Array enum with an Int8 array.
    #[cfg(feature = "extended_numeric_types")]
    pub fn from_int8(arr: IntegerArray<i8>) -> Self {
        Array::NumericArray(NumericArray::Int8(Arc::new(arr)))
    }

    /// Creates an Array enum with an UInt8 array.
    #[cfg(feature = "extended_numeric_types")]
    pub fn from_uint8(arr: IntegerArray<u8>) -> Self {
        Array::NumericArray(NumericArray::UInt8(Arc::new(arr)))
    }

    /// Creates an Array enum with an Int16 array.
    #[cfg(feature = "extended_numeric_types")]
    pub fn from_int16(arr: IntegerArray<i16>) -> Self {
        Array::NumericArray(NumericArray::Int16(Arc::new(arr)))
    }

    /// Creates an Array enum with an UInt16 array.
    #[cfg(feature = "extended_numeric_types")]
    pub fn from_uint16(arr: IntegerArray<u16>) -> Self {
        Array::NumericArray(NumericArray::UInt16(Arc::new(arr)))
    }

    /// Creates an Array enum with an Int32 array.
    pub fn from_int32(arr: IntegerArray<i32>) -> Self {
        Array::NumericArray(NumericArray::Int32(Arc::new(arr)))
    }

    /// Creates an Array enum with an Int64 array.
    pub fn from_int64(arr: IntegerArray<i64>) -> Self {
        Array::NumericArray(NumericArray::Int64(Arc::new(arr)))
    }

    /// Creates an Array enum with a UInt32 array.
    pub fn from_uint32(arr: IntegerArray<u32>) -> Self {
        Array::NumericArray(NumericArray::UInt32(Arc::new(arr)))
    }

    /// Creates an Array enum with an UInt64 array.
    pub fn from_uint64(arr: IntegerArray<u64>) -> Self {
        Array::NumericArray(NumericArray::UInt64(Arc::new(arr)))
    }

    /// Creates an Array enum with a Float32 array.
    pub fn from_float32(arr: FloatArray<f32>) -> Self {
        Array::NumericArray(NumericArray::Float32(Arc::new(arr)))
    }

    /// Creates an Array enum with a Float64 array.
    pub fn from_float64(arr: FloatArray<f64>) -> Self {
        Array::NumericArray(NumericArray::Float64(Arc::new(arr)))
    }

    /// Creates an Array enum with a String32 array.
    pub fn from_string32(arr: StringArray<u32>) -> Self {
        Array::TextArray(TextArray::String32(Arc::new(arr)))
    }

    /// Creates an Array enum with a String64 array.
    #[cfg(feature = "large_string")]
    pub fn from_string64(arr: StringArray<u64>) -> Self {
        Array::TextArray(TextArray::String64(Arc::new(arr)))
    }

    /// Creates an Array enum with a Categorical32 array.
    #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
    pub fn from_categorical32(arr: CategoricalArray<u32>) -> Self {
        Array::TextArray(TextArray::Categorical32(Arc::new(arr)))
    }

    /// Creates an Array enum with a Categorical8 array.
    #[cfg(feature = "default_categorical_8")]
    pub fn from_categorical8(arr: CategoricalArray<u8>) -> Self {
        Array::TextArray(TextArray::Categorical8(Arc::new(arr)))
    }

    /// Creates an Array enum with a Categorical16 array.
    #[cfg(feature = "extended_categorical")]
    pub fn from_categorical16(arr: CategoricalArray<u16>) -> Self {
        Array::TextArray(TextArray::Categorical16(Arc::new(arr)))
    }

    /// Creates an Array enum with a Categorical64 array.
    #[cfg(feature = "extended_categorical")]
    pub fn from_categorical64(arr: CategoricalArray<u64>) -> Self {
        Array::TextArray(TextArray::Categorical64(Arc::new(arr)))
    }

    /// Creates an Array enum with a DatetimeI32 array.
    #[cfg(feature = "datetime")]
    pub fn from_datetime_i32(arr: DatetimeArray<i32>) -> Self {
        Array::TemporalArray(TemporalArray::Datetime32(Arc::new(arr)))
    }

    /// Creates an Array enum with a DatetimeI64 array.
    #[cfg(feature = "datetime")]
    pub fn from_datetime_i64(arr: DatetimeArray<i64>) -> Self {
        Array::TemporalArray(TemporalArray::Datetime64(Arc::new(arr)))
    }

    /// Creates an Array enum with a Boolean array.
    pub fn from_bool(arr: BooleanArray<()>) -> Self {
        Array::BooleanArray(Arc::new(arr))
    }

    /// Wraps this Array in a FieldArray with the given name.
    ///
    /// Infers the Arrow type and nullability from the array itself.
    ///
    /// # Example
    /// ```rust
    /// use minarrow::{Array, IntegerArray, MaskedArray};
    ///
    /// let mut arr = IntegerArray::<i32>::default();
    /// arr.push(1);
    /// arr.push(2);
    /// let array = Array::from_int32(arr);
    /// let field_array = array.fa("my_column");
    /// assert_eq!(field_array.field.name, "my_column");
    /// ```
    pub fn fa(self, name: impl Into<String>) -> FieldArray {
        let dtype = self.arrow_type();
        let nullable = self.is_nullable();
        let field = Field::new(name, dtype, nullable, None);
        FieldArray::new(field, self)
    }

    // The below provides common accessors that reformat the data into the given type.
    // Because this library leans on enums, it makes for essential ergonomics once operating
    // in the top layer and one needs to match for e.g., to `T: Numeric` etc., as one can
    // can then go `.num()` to get access to all the numerical methods. This avoids polluting the top-level
    // `Array` API with method signatures that would otherwise panic for unsupported variants and flood IDE intellisense.
    // Additionally, when binding to Python, it follows common semantics.
    // I.e., '.dt` for datetime methods to appear, `.str` for strings, etc.
    //
    // Each accessor provides zero-copy for the already native type(s), conversion paths
    // for non-native (e.g., *bool -> integer* ), whilst propagating nulls for rarer nonsensical casts.

    /// Returns an inner `NumericArray`.
    /// - If already a `NumericArray`, returns the inner value as a shared handle with no data copy.
    /// - Other types: casts and copies.
    /// - Panics on `Null`. Consider the try variant for a safe alternative.
    pub fn num(&self) -> NumericArray {
        match self {
            Array::NumericArray(arr) => arr.clone(),

            Array::BooleanArray(arr) => {
                let n = arr.len();
                let mut out = Vec64::with_capacity(n);
                for i in 0..n {
                    let v = match arr.get(i) {
                        Some(true) => 1,
                        Some(false) => 0,
                        None => 0,
                    };
                    out.push(v);
                }
                let null_mask = arr.null_mask.clone();
                NumericArray::Int32(Arc::new(IntegerArray::<i32>::from_vec64(out, null_mask)))
            }

            #[cfg(feature = "datetime")]
            Array::TemporalArray(arr) => match arr {
                TemporalArray::Datetime32(dt) => {
                    let data = Vec64::from_slice(&dt.data);
                    let null_mask = dt.null_mask.clone();
                    NumericArray::Int32(Arc::new(IntegerArray::<i32>::from_vec64(data, null_mask)))
                }
                TemporalArray::Datetime64(dt) => {
                    let data = Vec64::from_slice(&dt.data);
                    let null_mask = dt.null_mask.clone();
                    NumericArray::Int64(Arc::new(IntegerArray::<i64>::from_vec64(data, null_mask)))
                }
                TemporalArray::Null => NumericArray::Null,
            },

            Array::TextArray(arr) => match arr {
                TextArray::String32(s) => {
                    let len = s.len();
                    let mut out = Vec64::with_capacity(len);
                    let mut null_mask = Bitmask::with_capacity(len);
                    for i in 0..len {
                        if s.is_null(i) {
                            out.push(0);
                            null_mask.set(i, false);
                            continue;
                        }
                        let raw = match s.get_str(i) {
                            Some(val) => val,
                            None => {
                                out.push(0);
                                null_mask.set(i, false);
                                continue;
                            }
                        };
                        match raw.parse::<i32>() {
                            Ok(val) => {
                                out.push(val);
                                null_mask.set(i, true);
                            }
                            Err(_) => {
                                out.push(0);
                                null_mask.set(i, false);
                            }
                        }
                    }
                    NumericArray::Int32(Arc::new(IntegerArray::<i32>::from_vec64(
                        out,
                        Some(null_mask),
                    )))
                }

                #[cfg(feature = "large_string")]
                TextArray::String64(s) => {
                    let len = s.len();
                    let mut out = Vec64::with_capacity(len);
                    let mut null_mask = Bitmask::with_capacity(len);
                    for i in 0..len {
                        if s.is_null(i) {
                            out.push(0);
                            null_mask.set(i, false);
                            continue;
                        }
                        let raw = match s.get_str(i) {
                            Some(val) => val,
                            None => {
                                out.push(0);
                                null_mask.set(i, false);
                                continue;
                            }
                        };
                        match raw.parse::<i64>() {
                            Ok(val) => {
                                out.push(val);
                                null_mask.set(i, true);
                            }
                            Err(_) => {
                                out.push(0);
                                null_mask.set(i, false);
                            }
                        }
                    }
                    NumericArray::Int64(Arc::new(IntegerArray::<i64>::from_vec64(
                        out,
                        Some(null_mask),
                    )))
                }

                #[cfg(feature = "default_categorical_8")]
                TextArray::Categorical8(cat) => {
                    let mut out = Vec64::with_capacity(cat.len());
                    let mut mask = Bitmask::with_capacity(cat.len());
                    for i in 0..cat.len() {
                        if cat.is_null(i) {
                            out.push(0);
                            mask.set(i, false);
                        } else {
                            let idx = cat.data[i] as usize;
                            let raw = &cat.unique_values()[idx];
                            match raw.parse::<i32>() {
                                Ok(val) => {
                                    out.push(val);
                                    mask.set(i, true);
                                }
                                Err(_) => {
                                    out.push(0);
                                    mask.set(i, false);
                                }
                            }
                        }
                    }
                    NumericArray::Int32(Arc::new(IntegerArray::<i32>::from_vec64(out, Some(mask))))
                }

                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(cat) => {
                    let mut out = Vec64::with_capacity(cat.len());
                    let mut mask = Bitmask::with_capacity(cat.len());
                    for i in 0..cat.len() {
                        if cat.is_null(i) {
                            out.push(0);
                            mask.set(i, false);
                        } else {
                            let idx = cat.data[i] as usize;
                            let raw = &cat.unique_values()[idx];
                            match raw.parse::<i32>() {
                                Ok(val) => {
                                    out.push(val);
                                    mask.set(i, true);
                                }
                                Err(_) => {
                                    out.push(0);
                                    mask.set(i, false);
                                }
                            }
                        }
                    }
                    NumericArray::Int32(Arc::new(IntegerArray::<i32>::from_vec64(out, Some(mask))))
                }

                #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
                TextArray::Categorical32(cat) => {
                    let mut out = Vec64::with_capacity(cat.len());
                    let mut mask = Bitmask::with_capacity(cat.len());
                    for i in 0..cat.len() {
                        if cat.is_null(i) {
                            out.push(0);
                            mask.set(i, false);
                        } else {
                            let idx = cat.data[i] as usize;
                            let raw = &cat.unique_values()[idx];
                            match raw.parse::<i32>() {
                                Ok(val) => {
                                    out.push(val);
                                    mask.set(i, true);
                                }
                                Err(_) => {
                                    out.push(0);
                                    mask.set(i, false);
                                }
                            }
                        }
                    }
                    NumericArray::Int32(Arc::new(IntegerArray::<i32>::from_vec64(out, Some(mask))))
                }

                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(cat) => {
                    let mut out = Vec64::with_capacity(cat.len());
                    let mut mask = Bitmask::with_capacity(cat.len());
                    for i in 0..cat.len() {
                        if cat.is_null(i) {
                            out.push(0);
                            mask.set(i, false);
                        } else {
                            let idx = cat.data[i] as usize;
                            let raw = &cat.unique_values()[idx];
                            match raw.parse::<i64>() {
                                Ok(val) => {
                                    out.push(val);
                                    mask.set(i, true);
                                }
                                Err(_) => {
                                    out.push(0);
                                    mask.set(i, false);
                                }
                            }
                        }
                    }
                    NumericArray::Int64(Arc::new(IntegerArray::<i64>::from_vec64(out, Some(mask))))
                }

                TextArray::Null => NumericArray::Null,
            },

            Array::Null => panic!("Array::num: array is Null"),
        }
    }

    /// Returns an inner `NumericArray`, with `Err` on `Null`.
    /// - If already a `NumericArray`, returns the inner value as a shared handle with no data copy.
    /// - Other types: casts and copies.
    pub fn try_num(&self) -> Result<NumericArray, MinarrowError> {
        match self {
            Array::Null => Err(MinarrowError::NullError { message: None }),
            _ => Ok(self.num()),
        }
    }

    /// Returns an inner `TextArray`.
    /// - If already a `TextArray`, returns the inner value as a shared handle with no data copy.
    /// - Other types: casts *(to string)* and copies.
    /// - Panics on `Null`. Consider the try variant for a safe alternative.
    pub fn str(&self) -> TextArray {
        match self {
            Array::TextArray(arr) => arr.clone(),

            Array::BooleanArray(arr) => {
                let n = arr.len();
                let mut strings: Vec<String> = Vec::with_capacity(n);
                for i in 0..n {
                    match arr.get(i) {
                        Some(true) => strings.push("true".to_string()),
                        Some(false) => strings.push("false".to_string()),
                        None => strings.push(String::new()),
                    }
                }
                TextArray::String32(Arc::new(StringArray::<u32>::from_slice(
                    &strings.iter().map(String::as_str).collect::<Vec<_>>(),
                )))
            }

            Array::NumericArray(arr) => match arr {
                NumericArray::Int32(a) => int_to_text_array::<i32>(a),
                NumericArray::Int64(a) => int_to_text_array::<i64>(a),
                NumericArray::UInt32(a) => int_to_text_array::<u32>(a),
                NumericArray::UInt64(a) => int_to_text_array::<u64>(a),
                NumericArray::Float32(a) => float_to_text_array::<f32>(a),
                NumericArray::Float64(a) => float_to_text_array::<f64>(a),
                _ => TextArray::Null,
            },

            #[cfg(feature = "datetime")]
            Array::TemporalArray(arr) => match arr {
                TemporalArray::Datetime32(dt) => {
                    let mut strings = Vec::with_capacity(dt.len());
                    for i in 0..dt.len() {
                        if dt.is_null(i) {
                            strings.push(String::new());
                        } else {
                            strings.push(format!("{}", dt.data[i]));
                        }
                    }
                    TextArray::String32(Arc::new(StringArray::<u32>::from_slice(
                        &strings.iter().map(String::as_str).collect::<Vec<_>>(),
                    )))
                }
                TemporalArray::Datetime64(dt) => {
                    let mut strings = Vec::with_capacity(dt.len());
                    for i in 0..dt.len() {
                        if dt.is_null(i) {
                            strings.push(String::new());
                        } else {
                            strings.push(format!("{}", dt.data[i]));
                        }
                    }
                    TextArray::String32(Arc::new(StringArray::<u32>::from_slice(
                        &strings.iter().map(String::as_str).collect::<Vec<_>>(),
                    )))
                }
                _ => TextArray::Null,
            },

            Array::Null => panic!("Array::str: array is Null"),
        }
    }

    /// Returns an inner `TextArray`, with `Err` on `Null`.
    /// - If already a `TextArray`, returns the inner value as a shared handle with no data copy.
    /// - Other types: casts *(to string)* and copies.
    pub fn try_str(&self) -> Result<TextArray, MinarrowError> {
        match self {
            Array::Null => Err(MinarrowError::NullError { message: None }),
            _ => Ok(self.str()),
        }
    }

    /// Returns the inner `BooleanArray`.
    /// - If already a `BooleanArray`, returns the inner value as a shared handle with no data copy.
    /// - Other types: calculates the boolean mask based on whether the value is present, and non-zero,
    ///  and copies. In these cases, any null mask is preserved, rather than becoming `false`.
    /// - Panics on `Null`. Consider the try variant for a safe alternative.
    pub fn bool(&self) -> Arc<BooleanArray<()>> {
        match self {
            Array::BooleanArray(arr) => arr.clone(),
            Array::NumericArray(arr) => {
                macro_rules! to_bool {
                    ($a:expr, $t:ty) => {{
                        let mut bm = Bitmask::with_capacity($a.len());
                        let mut out = Bitmask::with_capacity($a.len());
                        for i in 0..$a.len() {
                            let valid = !$a.is_null(i);
                            bm.set(i, valid);
                            let v = if valid && $a.data[i] != <$t>::default() {
                                true
                            } else {
                                false
                            };
                            out.set(i, v);
                        }
                        BooleanArray::new(out, Some(bm)).into()
                    }};
                }
                match arr {
                    NumericArray::Int32(a) => to_bool!(a, i32),
                    NumericArray::Int64(a) => to_bool!(a, i64),
                    NumericArray::UInt32(a) => to_bool!(a, u32),
                    NumericArray::UInt64(a) => to_bool!(a, u64),
                    NumericArray::Float32(a) => to_bool!(a, f32),
                    NumericArray::Float64(a) => to_bool!(a, f64),
                    _ => BooleanArray::default().into(),
                }
            }
            #[cfg(feature = "datetime")]
            Array::TemporalArray(arr) => match arr {
                TemporalArray::Datetime32(a) => {
                    let mut bm = Bitmask::with_capacity(a.len());
                    let mut out = Bitmask::with_capacity(a.len());
                    for i in 0..a.len() {
                        let valid = !a.is_null(i);
                        bm.set(i, valid);
                        out.set(i, valid);
                    }
                    BooleanArray::new(out, Some(bm)).into()
                }
                TemporalArray::Datetime64(a) => {
                    let mut bm = Bitmask::with_capacity(a.len());
                    let mut out = Bitmask::with_capacity(a.len());
                    for i in 0..a.len() {
                        let valid = !a.is_null(i);
                        bm.set(i, valid);
                        out.set(i, valid);
                    }
                    BooleanArray::new(out, Some(bm)).into()
                }
                _ => BooleanArray::default().into(),
            },
            Array::TextArray(arr) => match arr {
                TextArray::String32(s) => {
                    let mut bm = Bitmask::with_capacity(s.len());
                    let mut out = Bitmask::with_capacity(s.len());
                    for i in 0..s.len() {
                        let valid = !s.is_null(i);
                        bm.set(i, valid);

                        let str_val = if valid { s.get_str(i).unwrap() } else { "" };
                        let true_val = !str_val.eq_ignore_ascii_case("0")
                            && !str_val.eq_ignore_ascii_case("false")
                            && !str_val.eq_ignore_ascii_case("f")
                            && !str_val.is_empty();
                        out.set(i, if str_val.is_empty() { false } else { true_val });
                    }
                    BooleanArray::new(out, Some(bm)).into()
                }
                _ => BooleanArray::default().into(),
            },
            Array::Null => panic!("Array::bool: array is Null"),
        }
    }

    /// Returns the inner `BooleanArray`, with `Err` on `Null`.
    /// - If already a `BooleanArray`, returns the inner value as a shared handle with no data copy.
    /// - Other types: calculates the boolean mask based on whether the value is present, and non-zero,
    ///  and copies. In these cases, any null mask is preserved, rather than becoming `false`.
    pub fn try_bool(&self) -> Result<Arc<BooleanArray<()>>, MinarrowError> {
        match self {
            Array::Null => Err(MinarrowError::NullError { message: None }),
            _ => Ok(self.bool()),
        }
    }

    /// Returns the inner `TemporalArray`.
    /// - If already a `TemporalArray`, returns the inner value as a shared handle with no data copy.
    /// - Other types: casts and (often) copies.
    ///
    /// ### Datetime conversions
    /// - **String** parses a timestamp in milliseconds since the Unix epoch.
    /// If the `datetime_ops` feature is on, it also attempts common ISO8601/RFC3339 and `%Y-%m-%d` formats.
    /// Keep this in mind, because your API will break if you toggle the `datetime_ops` feature on/off but
    /// keep the previous code.
    /// - **Integer** becomes *milliseconds since epoch*.
    /// - **Floats** round as integers to *milliseconds since epoch*.
    /// - **Boolean** returns `TemporalArray::Null`.
    ///
    /// Panics on `Null`. Consider the try variant for a safe alternative.
    #[cfg(feature = "datetime")]
    pub fn dt(&self) -> TemporalArray {
        use crate::enums::time_units::TimeUnit;
        match self {
            Array::TemporalArray(arr) => arr.clone(),
            Array::NumericArray(arr) => match arr {
                NumericArray::Int32(a) => {
                    TemporalArray::Datetime64(Arc::new(DatetimeArray::<i64>::from_vec64(
                        a.data.iter().map(|v| *v as i64).collect(),
                        a.null_mask.clone(),
                        Some(TimeUnit::Milliseconds),
                    )))
                }
                NumericArray::Int64(a) => {
                    TemporalArray::Datetime64(Arc::new(DatetimeArray::<i64>::from_vec64(
                        a.data.iter().copied().collect(),
                        a.null_mask.clone(),
                        Some(TimeUnit::Milliseconds),
                    )))
                }
                NumericArray::UInt32(a) => {
                    TemporalArray::Datetime64(Arc::new(DatetimeArray::<i64>::from_vec64(
                        a.data.iter().map(|v| *v as i64).collect(),
                        a.null_mask.clone(),
                        Some(TimeUnit::Milliseconds),
                    )))
                }
                NumericArray::UInt64(a) => {
                    TemporalArray::Datetime64(Arc::new(DatetimeArray::<i64>::from_vec64(
                        a.data.iter().map(|v| *v as i64).collect(),
                        a.null_mask.clone(),
                        Some(TimeUnit::Milliseconds),
                    )))
                }
                NumericArray::Float32(a) => {
                    TemporalArray::Datetime64(Arc::new(DatetimeArray::<i64>::from_vec64(
                        a.data.iter().map(|v| *v as i64).collect(),
                        a.null_mask.clone(),
                        Some(TimeUnit::Milliseconds),
                    )))
                }
                NumericArray::Float64(a) => {
                    TemporalArray::Datetime64(Arc::new(DatetimeArray::<i64>::from_vec64(
                        a.data.iter().map(|v| *v as i64).collect(),
                        a.null_mask.clone(),
                        Some(TimeUnit::Milliseconds),
                    )))
                }
                _ => TemporalArray::Null,
            },
            Array::BooleanArray(_) => TemporalArray::Null,
            Array::TextArray(arr) => match arr {
                TextArray::String32(s) => {
                    let mut out = Vec64::with_capacity(s.len());
                    let mut null_mask = Bitmask::with_capacity(s.len());
                    for i in 0..s.len() {
                        let valid = !s.is_null(i);
                        let val = if valid {
                            use crate::utils::parse_datetime_str;
                            let str_val = unsafe { s.get_str_unchecked(i) };
                            parse_datetime_str(str_val)
                        } else {
                            None
                        };
                        match val {
                            Some(dt) => {
                                out.push(dt);
                                null_mask.set(i, true);
                            }
                            None => {
                                out.push(0);
                                null_mask.set(i, false);
                            }
                        }
                    }
                    TemporalArray::Datetime64(Arc::new(DatetimeArray::<i64>::from_vec64(
                        out,
                        Some(null_mask),
                        Some(TimeUnit::Milliseconds),
                    )))
                }
                _ => TemporalArray::Null,
            },
            Array::Null => panic!("Array::dt: array is Null"),
        }
    }

    /// Returns the inner `TemporalArray`, with `Err` on `Null`.
    /// - If already a `TemporalArray`, returns the inner value as a shared handle with no data copy.
    /// - Other types: casts and (often) copies. See `dt` for the conversion rules.
    #[cfg(feature = "datetime")]
    pub fn try_dt(&self) -> Result<TemporalArray, MinarrowError> {
        match self {
            Array::Null => Err(MinarrowError::NullError { message: None }),
            _ => Ok(self.dt()),
        }
    }

    /// Returns the length of the array.
    pub fn len(&self) -> usize {
        match self {
            Self::Null => 0,
            _ => match_array!(self, len),
        }
    }

    /// Removes the rows in `[start, end)`, shifting later rows left.
    /// A shared inner array is cloned first i.e. copy-on-write.
    ///
    /// # Panics
    /// Panics if `start > end` or `end > len`.
    pub fn delete_range(&mut self, start: usize, end: usize) {
        match self {
            Array::NumericArray(arr) => arr.delete_range(start, end),
            Array::TextArray(arr) => arr.delete_range(start, end),
            #[cfg(feature = "datetime")]
            Array::TemporalArray(arr) => arr.delete_range(start, end),
            Array::BooleanArray(arr) => arr.delete_range(start, end),
            Array::Null => {
                assert!(
                    start == 0 && end == 0,
                    "Array::Null: delete_range out of bounds"
                );
            }
        }
    }

    /// Returns a metadata view and reference over the specified window of this array.
    ///
    /// Does not slice the object (yet).
    ///
    /// Panics if out of bounds.
    #[cfg(feature = "views")]
    pub fn view(&self, offset: usize, len: usize) -> ArrayV {
        assert!(offset <= self.len(), "offset out of bounds");
        assert!(offset + len <= self.len(), "slice window out of bounds");
        ArrayV::new(self.clone(), offset, len)
    }

    /// Returns a metadata view and reference over the specified window of this array.
    ///
    /// Does not slice the object (yet).
    ///
    /// Panics if out of bounds.
    #[cfg(feature = "views")]
    pub fn view_tuple(&self, offset: usize, len: usize) -> ArrayVT<'_> {
        assert!(offset <= self.len(), "offset out of bounds");
        assert!(offset + len <= self.len(), "slice window out of bounds");
        (self, offset, len)
    }

    /// Returns a reference to the inner array as type `Arc<T>`.
    ///
    /// This is compile-time safe if `T` matches the actual payload, but will panic otherwise.
    /// Prefer `.inner_check()` for Option-based pattern.
    #[inline]
    pub fn inner<T: 'static>(&self) -> &Arc<T> {
        macro_rules! match_arm {
            ($inner_enum:ident, $variant:ident, $ty:ty) => {
                if let Array::$inner_enum(inner) = self {
                    if let $inner_enum::$variant(inner2) = inner {
                        // Arc<T> always lives here, so we compare against Arc<T>
                        if TypeId::of::<T>() == TypeId::of::<$ty>() {
                            // inner2: Arc<Ty>, T == Ty
                            // safe to cast: Arc<Ty> -> Arc<T>
                            return unsafe { &*(inner2 as *const Arc<$ty> as *const Arc<T>) };
                        }
                    }
                }
            };
        }

        // NumericArray
        #[cfg(feature = "extended_numeric_types")]
        match_arm!(NumericArray, Int8, IntegerArray<i8>);
        #[cfg(feature = "extended_numeric_types")]
        match_arm!(NumericArray, Int16, IntegerArray<i16>);
        match_arm!(NumericArray, Int32, IntegerArray<i32>);
        match_arm!(NumericArray, Int64, IntegerArray<i64>);
        #[cfg(feature = "extended_numeric_types")]
        match_arm!(NumericArray, UInt8, IntegerArray<u8>);
        #[cfg(feature = "extended_numeric_types")]
        match_arm!(NumericArray, UInt16, IntegerArray<u16>);
        match_arm!(NumericArray, UInt32, IntegerArray<u32>);
        match_arm!(NumericArray, UInt64, IntegerArray<u64>);
        match_arm!(NumericArray, Float32, FloatArray<f32>);
        match_arm!(NumericArray, Float64, FloatArray<f64>);

        // TextArray
        match_arm!(TextArray, String32, StringArray<u32>);
        #[cfg(feature = "large_string")]
        match_arm!(TextArray, String64, StringArray<u64>);
        #[cfg(feature = "default_categorical_8")]
        match_arm!(TextArray, Categorical8, CategoricalArray<u8>);
        #[cfg(feature = "extended_categorical")]
        match_arm!(TextArray, Categorical16, CategoricalArray<u16>);
        #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
        match_arm!(TextArray, Categorical32, CategoricalArray<u32>);
        #[cfg(feature = "extended_categorical")]
        match_arm!(TextArray, Categorical64, CategoricalArray<u64>);

        // TemporalArray
        #[cfg(feature = "datetime")]
        match_arm!(TemporalArray, Datetime32, DatetimeArray<i32>);
        #[cfg(feature = "datetime")]
        match_arm!(TemporalArray, Datetime64, DatetimeArray<i64>);

        // Boolean
        if let Array::BooleanArray(inner) = self {
            if TypeId::of::<T>() == TypeId::of::<BooleanArray<()>>() {
                return unsafe { &*(inner as *const Arc<BooleanArray<()>> as *const Arc<T>) };
            }
        }

        panic!(
            "Type mismatch: attempted to access Array::{:?} as incompatible type",
            self.arrow_type()
        );
    }

    /// Returns a mutable reference to the inner array as type `T`.
    ///
    /// This method is compile-time safe when the type `T` matches the actual inner type,
    /// but relies on `TypeId` checks and unsafe casting. If an incorrect type is specified,
    /// this will panic at runtime.
    ///
    /// Prefer `inner_check_mut` if you want an `Option`-based version that avoids panics.
    #[inline]
    pub fn inner_mut<T: 'static>(&mut self) -> &mut Arc<T> {
        use std::any::TypeId;

        macro_rules! match_arm {
            ($inner_enum:ident, $variant:ident, $ty:ty) => {
                if let Array::$inner_enum(inner) = self {
                    if let $inner_enum::$variant(inner2) = inner {
                        if TypeId::of::<T>() == TypeId::of::<$ty>() {
                            return unsafe { &mut *(inner2 as *mut Arc<$ty> as *mut Arc<T>) };
                        }
                    }
                }
            };
        }

        // NumericArray
        #[cfg(feature = "extended_numeric_types")]
        match_arm!(NumericArray, Int8, IntegerArray<i8>);
        #[cfg(feature = "extended_numeric_types")]
        match_arm!(NumericArray, Int16, IntegerArray<i16>);
        match_arm!(NumericArray, Int32, IntegerArray<i32>);
        match_arm!(NumericArray, Int64, IntegerArray<i64>);
        #[cfg(feature = "extended_numeric_types")]
        match_arm!(NumericArray, UInt8, IntegerArray<u8>);
        #[cfg(feature = "extended_numeric_types")]
        match_arm!(NumericArray, UInt16, IntegerArray<u16>);
        match_arm!(NumericArray, UInt32, IntegerArray<u32>);
        match_arm!(NumericArray, UInt64, IntegerArray<u64>);
        match_arm!(NumericArray, Float32, FloatArray<f32>);
        match_arm!(NumericArray, Float64, FloatArray<f64>);

        // TextArray
        match_arm!(TextArray, String32, StringArray<u32>);
        #[cfg(feature = "large_string")]
        match_arm!(TextArray, String64, StringArray<u64>);
        #[cfg(feature = "default_categorical_8")]
        match_arm!(TextArray, Categorical8, CategoricalArray<u8>);
        #[cfg(feature = "extended_categorical")]
        match_arm!(TextArray, Categorical16, CategoricalArray<u16>);
        #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
        match_arm!(TextArray, Categorical32, CategoricalArray<u32>);
        #[cfg(feature = "extended_categorical")]
        match_arm!(TextArray, Categorical64, CategoricalArray<u64>);

        // TemporalArray
        #[cfg(feature = "datetime")]
        match_arm!(TemporalArray, Datetime32, DatetimeArray<i32>);
        #[cfg(feature = "datetime")]
        match_arm!(TemporalArray, Datetime64, DatetimeArray<i64>);

        // Boolean
        if let Array::BooleanArray(inner) = self {
            if TypeId::of::<T>() == TypeId::of::<BooleanArray<()>>() {
                return unsafe { &mut *(inner as *mut Arc<BooleanArray<()>> as *mut Arc<T>) };
            }
        }

        panic!(
            "Type mismatch: attempted to mutably access Array::{:?} as incompatible type",
            self.arrow_type()
        )
    }

    /// Returns a reference to the inner array as type `T`, if the type matches.
    ///
    /// This method performs a runtime `TypeId` check to verify that the provided type `T`
    /// corresponds to the actual inner variant. If the types match, returns `Some(&T)`;
    /// otherwise, returns `None` without panicking.
    ///
    /// Use when the type of the variant is uncertain at compile time.
    #[inline]
    pub fn inner_check<T: 'static>(&self) -> Option<&Arc<T>> {
        use std::any::TypeId;

        macro_rules! match_inner_type {
            ($outer:ident, $variant:ident, $ty:ty) => {
                if TypeId::of::<T>() == TypeId::of::<$ty>() {
                    if let Array::$outer(inner) = self {
                        if let $outer::$variant(inner2) = inner {
                            return Some(unsafe { &*(inner2 as *const Arc<$ty> as *const Arc<T>) });
                        }
                    }
                }
            };
        }

        #[cfg(feature = "extended_numeric_types")]
        match_inner_type!(NumericArray, Int8, IntegerArray<i8>);
        #[cfg(feature = "extended_numeric_types")]
        match_inner_type!(NumericArray, Int16, IntegerArray<i16>);
        match_inner_type!(NumericArray, Int32, IntegerArray<i32>);
        match_inner_type!(NumericArray, Int64, IntegerArray<i64>);
        #[cfg(feature = "extended_numeric_types")]
        match_inner_type!(NumericArray, UInt8, IntegerArray<u8>);
        #[cfg(feature = "extended_numeric_types")]
        match_inner_type!(NumericArray, UInt16, IntegerArray<u16>);
        match_inner_type!(NumericArray, UInt32, IntegerArray<u32>);
        match_inner_type!(NumericArray, UInt64, IntegerArray<u64>);
        match_inner_type!(NumericArray, Float32, FloatArray<f32>);
        match_inner_type!(NumericArray, Float64, FloatArray<f64>);

        match_inner_type!(TextArray, String32, StringArray<u32>);
        #[cfg(feature = "large_string")]
        match_inner_type!(TextArray, String64, StringArray<u64>);
        #[cfg(feature = "default_categorical_8")]
        match_inner_type!(TextArray, Categorical8, CategoricalArray<u8>);
        #[cfg(feature = "extended_categorical")]
        match_inner_type!(TextArray, Categorical16, CategoricalArray<u16>);
        #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
        match_inner_type!(TextArray, Categorical32, CategoricalArray<u32>);
        #[cfg(feature = "extended_categorical")]
        match_inner_type!(TextArray, Categorical64, CategoricalArray<u64>);

        #[cfg(feature = "datetime")]
        match_inner_type!(TemporalArray, Datetime32, DatetimeArray<i32>);
        #[cfg(feature = "datetime")]
        match_inner_type!(TemporalArray, Datetime64, DatetimeArray<i64>);

        if TypeId::of::<T>() == TypeId::of::<BooleanArray<()>>() {
            if let Array::BooleanArray(inner) = self {
                return Some(unsafe { &*(inner as *const Arc<BooleanArray<()>> as *const Arc<T>) });
            }
        }

        None
    }

    /// Returns a mutable reference to the inner array as type `T`, if the type matches.
    ///
    /// This method performs a runtime `TypeId` check to verify that the provided type `T`
    /// corresponds to the actual inner variant. If the types match, returns `Some(&mut T)`;
    /// otherwise, returns `None` without panicking.
    ///
    /// Use when the type of the variant is uncertain at compile time.
    #[inline]
    pub fn inner_check_mut<T: 'static>(&mut self) -> Option<&mut Arc<T>> {
        use std::any::TypeId;

        macro_rules! match_inner_type_mut {
            ($outer:ident, $variant:ident, $ty:ty) => {
                if TypeId::of::<T>() == TypeId::of::<$ty>() {
                    if let Array::$outer(inner) = self {
                        if let $outer::$variant(inner2) = inner {
                            return Some(unsafe { &mut *(inner2 as *mut Arc<$ty> as *mut Arc<T>) });
                        }
                    }
                }
            };
        }

        #[cfg(feature = "extended_numeric_types")]
        match_inner_type_mut!(NumericArray, Int8, IntegerArray<i8>);
        #[cfg(feature = "extended_numeric_types")]
        match_inner_type_mut!(NumericArray, Int16, IntegerArray<i16>);
        match_inner_type_mut!(NumericArray, Int32, IntegerArray<i32>);
        match_inner_type_mut!(NumericArray, Int64, IntegerArray<i64>);
        #[cfg(feature = "extended_numeric_types")]
        match_inner_type_mut!(NumericArray, UInt8, IntegerArray<u8>);
        #[cfg(feature = "extended_numeric_types")]
        match_inner_type_mut!(NumericArray, UInt16, IntegerArray<u16>);
        match_inner_type_mut!(NumericArray, UInt32, IntegerArray<u32>);
        match_inner_type_mut!(NumericArray, UInt64, IntegerArray<u64>);
        match_inner_type_mut!(NumericArray, Float32, FloatArray<f32>);
        match_inner_type_mut!(NumericArray, Float64, FloatArray<f64>);

        match_inner_type_mut!(TextArray, String32, StringArray<u32>);
        #[cfg(feature = "large_string")]
        match_inner_type_mut!(TextArray, String64, StringArray<u64>);
        #[cfg(feature = "default_categorical_8")]
        match_inner_type_mut!(TextArray, Categorical8, CategoricalArray<u8>);
        #[cfg(feature = "extended_categorical")]
        match_inner_type_mut!(TextArray, Categorical16, CategoricalArray<u16>);
        #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
        match_inner_type_mut!(TextArray, Categorical32, CategoricalArray<u32>);
        #[cfg(feature = "extended_categorical")]
        match_inner_type_mut!(TextArray, Categorical64, CategoricalArray<u64>);

        #[cfg(feature = "datetime")]
        match_inner_type_mut!(TemporalArray, Datetime32, DatetimeArray<i32>);
        #[cfg(feature = "datetime")]
        match_inner_type_mut!(TemporalArray, Datetime64, DatetimeArray<i64>);

        if TypeId::of::<T>() == TypeId::of::<BooleanArray<()>>() {
            if let Array::BooleanArray(inner) = self {
                return Some(unsafe { &mut *(inner as *mut Arc<BooleanArray<()>> as *mut Arc<T>) });
            }
        }

        None
    }

    #[inline]
    pub fn as_slice<T>(&self, offset: usize, len: usize) -> &[T] {
        match self {
            Array::NumericArray(inner) => match inner {
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(arr) => {
                    cast_slice::<i8, T>(arr.data(), offset, len).expect("cast failed")
                }

                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(arr) => {
                    cast_slice::<i16, T>(arr.data(), offset, len).expect("cast failed")
                }

                NumericArray::Int32(arr) => {
                    cast_slice::<i32, T>(arr.data(), offset, len).expect("cast failed")
                }

                NumericArray::Int64(arr) => {
                    cast_slice::<i64, T>(arr.data(), offset, len).expect("cast failed")
                }

                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(arr) => {
                    cast_slice::<u8, T>(arr.data(), offset, len).expect("cast failed")
                }

                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(arr) => {
                    cast_slice::<u16, T>(arr.data(), offset, len).expect("cast failed")
                }

                NumericArray::UInt32(arr) => {
                    cast_slice::<u32, T>(arr.data(), offset, len).expect("cast failed")
                }

                NumericArray::UInt64(arr) => {
                    cast_slice::<u64, T>(arr.data(), offset, len).expect("cast failed")
                }

                NumericArray::Float32(arr) => {
                    cast_slice::<f32, T>(arr.data(), offset, len).expect("cast failed")
                }

                NumericArray::Float64(arr) => {
                    cast_slice::<f64, T>(arr.data(), offset, len).expect("cast failed")
                }

                NumericArray::Null => panic!("Null array has no data payload"),
            },

            Array::TextArray(inner) => match inner {
                TextArray::String32(_) | TextArray::Null => {
                    panic!(
                        "Strings use UTF-8 + offsets. Use logical accessor instead, or `slice_raw` if you do want byte access."
                    )
                }
                #[cfg(feature = "large_string")]
                TextArray::String64(_) => {
                    panic!(
                        "Strings use UTF-8 + offsets. Use logical accessor instead, or `slice_raw` if you do want byte access."
                    )
                }
                #[cfg(feature = "default_categorical_8")]
                TextArray::Categorical8(arr) => {
                    cast_slice::<u8, T>(arr.data(), offset, len).expect("cast failed")
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(arr) => {
                    cast_slice::<u16, T>(arr.data(), offset, len).expect("cast failed")
                }
                #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
                TextArray::Categorical32(arr) => {
                    cast_slice::<u32, T>(arr.data(), offset, len).expect("cast failed")
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(arr) => {
                    cast_slice::<u64, T>(arr.data(), offset, len).expect("cast failed")
                }
            },
            #[cfg(feature = "datetime")]
            Array::TemporalArray(inner) => match inner {
                TemporalArray::Datetime32(arr) => {
                    cast_slice::<i32, T>(arr.data(), offset, len).expect("cast failed")
                }
                TemporalArray::Datetime64(arr) => {
                    cast_slice::<i64, T>(arr.data(), offset, len).expect("cast failed")
                }
                TemporalArray::Null => panic!("Null array has no data payload"),
            },

            Array::BooleanArray(_) => {
                panic!(
                    "Bool arrays are bit-packed; use logical accessor instead, or `slice_raw` if you do want byte access."
                )
            }

            Array::Null => panic!("Null array has no data payload"),
        }
    }

    #[inline]
    pub fn slice_raw<T: 'static>(&self, offset: usize, len: usize) -> Option<&[T]> {
        use std::any::TypeId;

        match self {
            Array::NumericArray(inner) => match inner {
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(a) if TypeId::of::<T>() == TypeId::of::<i8>() => {
                    cast_slice::<i8, T>(&a.data, offset, len)
                }
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(a) if TypeId::of::<T>() == TypeId::of::<i16>() => {
                    cast_slice::<i16, T>(&a.data, offset, len)
                }
                NumericArray::Int32(a) if TypeId::of::<T>() == TypeId::of::<i32>() => {
                    cast_slice::<i32, T>(&a.data, offset, len)
                }
                NumericArray::Int64(a) if TypeId::of::<T>() == TypeId::of::<i64>() => {
                    cast_slice::<i64, T>(&a.data, offset, len)
                }
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(a) if TypeId::of::<T>() == TypeId::of::<u8>() => {
                    cast_slice::<u8, T>(&a.data, offset, len)
                }
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(a) if TypeId::of::<T>() == TypeId::of::<u16>() => {
                    cast_slice::<u16, T>(&a.data, offset, len)
                }
                NumericArray::UInt32(a) if TypeId::of::<T>() == TypeId::of::<u32>() => {
                    cast_slice::<u32, T>(&a.data, offset, len)
                }
                NumericArray::UInt64(a) if TypeId::of::<T>() == TypeId::of::<u64>() => {
                    cast_slice::<u64, T>(&a.data, offset, len)
                }
                NumericArray::Float32(a) if TypeId::of::<T>() == TypeId::of::<f32>() => {
                    cast_slice::<f32, T>(&a.data, offset, len)
                }
                NumericArray::Float64(a) if TypeId::of::<T>() == TypeId::of::<f64>() => {
                    cast_slice::<f64, T>(&a.data, offset, len)
                }
                _ => None,
            },

            Array::BooleanArray(a) if TypeId::of::<T>() == TypeId::of::<u8>() => {
                let start = offset / 8;
                let end = (offset + len + 7) / 8;
                let slice = &a[start..end];
                Some(unsafe { &*(slice as *const [u8] as *const [T]) })
            }

            Array::TextArray(inner) => match inner {
                TextArray::String32(a) if TypeId::of::<T>() == TypeId::of::<u8>() => {
                    cast_slice::<u8, T>(&a.data, offset, len)
                }
                #[cfg(feature = "large_string")]
                TextArray::String64(a) if TypeId::of::<T>() == TypeId::of::<u8>() => {
                    cast_slice::<u8, T>(&a.data, offset, len)
                }
                #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
                TextArray::Categorical32(a) if TypeId::of::<T>() == TypeId::of::<u32>() => {
                    cast_slice::<u32, T>(&a.data, offset, len)
                }
                #[cfg(feature = "default_categorical_8")]
                TextArray::Categorical8(a) if TypeId::of::<T>() == TypeId::of::<u8>() => {
                    cast_slice::<u8, T>(&a.data, offset, len)
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(a) if TypeId::of::<T>() == TypeId::of::<u16>() => {
                    cast_slice::<u16, T>(&a.data, offset, len)
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(a) if TypeId::of::<T>() == TypeId::of::<u64>() => {
                    cast_slice::<u64, T>(&a.data, offset, len)
                }
                _ => None,
            },
            #[cfg(feature = "datetime")]
            Array::TemporalArray(inner) => match inner {
                TemporalArray::Datetime32(a) if TypeId::of::<T>() == TypeId::of::<i32>() => {
                    cast_slice::<i32, T>(&a.data, offset, len)
                }
                TemporalArray::Datetime64(a) if TypeId::of::<T>() == TypeId::of::<i64>() => {
                    cast_slice::<i64, T>(&a.data, offset, len)
                }
                _ => None,
            },

            _ => None,
        }
    }

    /// Returns a new `Array` of the same variant sliced to the given offset and length
    /// .
    /// Copies the data of the scoped range that's selected.
    ///  
    /// If out-of-bounds, returns Self::Null.
    /// All null mask, offsets, etc. are trimmed.
    #[inline]
    pub fn slice_clone(&self, offset: usize, len: usize) -> Self {
        match self {
            Array::NumericArray(inner) => Self::NumericArray(match inner {
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(arr) => NumericArray::Int8(arr.slice_clone(offset, len)),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(arr) => NumericArray::Int16(arr.slice_clone(offset, len)),
                NumericArray::Int32(arr) => NumericArray::Int32(arr.slice_clone(offset, len)),
                NumericArray::Int64(arr) => NumericArray::Int64(arr.slice_clone(offset, len)),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(arr) => NumericArray::UInt8(arr.slice_clone(offset, len)),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(arr) => NumericArray::UInt16(arr.slice_clone(offset, len)),
                NumericArray::UInt32(arr) => NumericArray::UInt32(arr.slice_clone(offset, len)),
                NumericArray::UInt64(arr) => NumericArray::UInt64(arr.slice_clone(offset, len)),
                NumericArray::Float32(arr) => NumericArray::Float32(arr.slice_clone(offset, len)),
                NumericArray::Float64(arr) => NumericArray::Float64(arr.slice_clone(offset, len)),
                NumericArray::Null => NumericArray::Null,
            }),
            Array::TextArray(inner) => Self::TextArray(match inner {
                TextArray::String32(arr) => TextArray::String32(arr.slice_clone(offset, len)),
                #[cfg(feature = "large_string")]
                TextArray::String64(arr) => TextArray::String64(arr.slice_clone(offset, len)),
                #[cfg(feature = "default_categorical_8")]
                TextArray::Categorical8(arr) => {
                    TextArray::Categorical8(arr.slice_clone(offset, len))
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(arr) => {
                    TextArray::Categorical16(arr.slice_clone(offset, len))
                }
                #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
                TextArray::Categorical32(arr) => {
                    TextArray::Categorical32(arr.slice_clone(offset, len))
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(arr) => {
                    TextArray::Categorical64(arr.slice_clone(offset, len))
                }
                TextArray::Null => TextArray::Null,
            }),
            #[cfg(feature = "datetime")]
            Array::TemporalArray(inner) => Self::TemporalArray(match inner {
                TemporalArray::Datetime32(arr) => {
                    TemporalArray::Datetime32(arr.slice_clone(offset, len))
                }
                TemporalArray::Datetime64(arr) => {
                    TemporalArray::Datetime64(arr.slice_clone(offset, len))
                }
                TemporalArray::Null => TemporalArray::Null,
            }),
            Array::BooleanArray(arr) => Self::BooleanArray(arr.slice_clone(offset, len)),
            Array::Null => Self::Null,
        }
    }

    /// Arrow physical type for this array.
    pub fn arrow_type(&self) -> ArrowType {
        match self {
            Array::NumericArray(inner) => match inner {
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(_) => ArrowType::Int8,
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(_) => ArrowType::Int16,
                NumericArray::Int32(_) => ArrowType::Int32,
                NumericArray::Int64(_) => ArrowType::Int64,
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(_) => ArrowType::UInt8,
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(_) => ArrowType::UInt16,
                NumericArray::UInt32(_) => ArrowType::UInt32,
                NumericArray::UInt64(_) => ArrowType::UInt64,
                NumericArray::Float32(_) => ArrowType::Float32,
                NumericArray::Float64(_) => ArrowType::Float64,
                NumericArray::Null => ArrowType::Null,
            },
            Array::TextArray(inner) => match inner {
                TextArray::String32(_) => ArrowType::String,
                #[cfg(feature = "large_string")]
                TextArray::String64(_) => ArrowType::LargeString,
                #[cfg(feature = "default_categorical_8")]
                TextArray::Categorical8(_) => ArrowType::Dictionary(CategoricalIndexType::UInt8),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(_) => ArrowType::Dictionary(CategoricalIndexType::UInt16),
                #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
                TextArray::Categorical32(_) => ArrowType::Dictionary(CategoricalIndexType::UInt32),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(_) => ArrowType::Dictionary(CategoricalIndexType::UInt64),
                TextArray::Null => ArrowType::Null,
            },
            #[cfg(feature = "datetime")]
            Array::TemporalArray(inner) => match inner {
                TemporalArray::Datetime32(_) => ArrowType::Date32,
                TemporalArray::Datetime64(_) => ArrowType::Date64,
                TemporalArray::Null => ArrowType::Null,
            },
            Array::BooleanArray(_) => ArrowType::Boolean,
            Array::Null => ArrowType::Null,
        }
    }

    /// Column nullability
    pub fn is_nullable(&self) -> bool {
        match self {
            Self::Null => true,
            _ => match_array!(self, is_nullable),
        }
    }

    // ───────────── Type Predicates ─────────────

    /// Returns true if this is a categorical array.
    #[inline]
    pub fn is_categorical_array(&self) -> bool {
        match self {
            Array::TextArray(text) => match text {
                #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
                TextArray::Categorical32(_) => true,
                #[cfg(feature = "default_categorical_8")]
                TextArray::Categorical8(_) => true,
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(_) => true,
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(_) => true,
                TextArray::String32(_) => false,
                #[cfg(feature = "large_string")]
                TextArray::String64(_) => false,
                TextArray::Null => false,
            },
            Array::NumericArray(_) => false,
            #[cfg(feature = "datetime")]
            Array::TemporalArray(_) => false,
            Array::BooleanArray(_) => false,
            Array::Null => false,
        }
    }

    /// Returns true if this is a string array i.e. non-categorical text.
    #[inline]
    pub fn is_string_array(&self) -> bool {
        match self {
            Array::TextArray(text) => match text {
                TextArray::String32(_) => true,
                #[cfg(feature = "large_string")]
                TextArray::String64(_) => true,
                #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
                TextArray::Categorical32(_) => false,
                #[cfg(feature = "default_categorical_8")]
                TextArray::Categorical8(_) => false,
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(_) => false,
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(_) => false,
                TextArray::Null => false,
            },
            Array::NumericArray(_) => false,
            #[cfg(feature = "datetime")]
            Array::TemporalArray(_) => false,
            Array::BooleanArray(_) => false,
            Array::Null => false,
        }
    }

    /// Returns true if this is any text array, string or categorical.
    #[inline]
    pub fn is_text_array(&self) -> bool {
        match self {
            Array::TextArray(text) => match text {
                TextArray::String32(_) => true,
                #[cfg(feature = "large_string")]
                TextArray::String64(_) => true,
                #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
                TextArray::Categorical32(_) => true,
                #[cfg(feature = "default_categorical_8")]
                TextArray::Categorical8(_) => true,
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(_) => true,
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(_) => true,
                TextArray::Null => false,
            },
            Array::NumericArray(_) => false,
            #[cfg(feature = "datetime")]
            Array::TemporalArray(_) => false,
            Array::BooleanArray(_) => false,
            Array::Null => false,
        }
    }

    /// Returns true if this is a boolean array.
    #[inline]
    pub fn is_boolean_array(&self) -> bool {
        match self {
            Array::BooleanArray(_) => true,
            Array::NumericArray(_) => false,
            Array::TextArray(_) => false,
            #[cfg(feature = "datetime")]
            Array::TemporalArray(_) => false,
            Array::Null => false,
        }
    }

    /// Returns true if this is an integer array.
    #[inline]
    pub fn is_integer_array(&self) -> bool {
        match self {
            Array::NumericArray(num) => match num {
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(_) => true,
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(_) => true,
                NumericArray::Int32(_) => true,
                NumericArray::Int64(_) => true,
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(_) => true,
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(_) => true,
                NumericArray::UInt32(_) => true,
                NumericArray::UInt64(_) => true,
                NumericArray::Float32(_) => false,
                NumericArray::Float64(_) => false,
                NumericArray::Null => false,
            },
            Array::TextArray(_) => false,
            #[cfg(feature = "datetime")]
            Array::TemporalArray(_) => false,
            Array::BooleanArray(_) => false,
            Array::Null => false,
        }
    }

    /// Returns true if this is a floating-point array.
    #[inline]
    pub fn is_float_array(&self) -> bool {
        match self {
            Array::NumericArray(num) => match num {
                NumericArray::Float32(_) => true,
                NumericArray::Float64(_) => true,
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(_) => false,
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(_) => false,
                NumericArray::Int32(_) => false,
                NumericArray::Int64(_) => false,
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(_) => false,
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(_) => false,
                NumericArray::UInt32(_) => false,
                NumericArray::UInt64(_) => false,
                NumericArray::Null => false,
            },
            Array::TextArray(_) => false,
            #[cfg(feature = "datetime")]
            Array::TemporalArray(_) => false,
            Array::BooleanArray(_) => false,
            Array::Null => false,
        }
    }

    /// Returns true if this is any numeric array, integer or float.
    #[inline]
    pub fn is_numerical_array(&self) -> bool {
        match self {
            Array::NumericArray(num) => match num {
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(_) => true,
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(_) => true,
                NumericArray::Int32(_) => true,
                NumericArray::Int64(_) => true,
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(_) => true,
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(_) => true,
                NumericArray::UInt32(_) => true,
                NumericArray::UInt64(_) => true,
                NumericArray::Float32(_) => true,
                NumericArray::Float64(_) => true,
                NumericArray::Null => false,
            },
            Array::TextArray(_) => false,
            #[cfg(feature = "datetime")]
            Array::TemporalArray(_) => false,
            Array::BooleanArray(_) => false,
            Array::Null => false,
        }
    }

    /// Returns true if this is a datetime/temporal array.
    #[inline]
    #[cfg(feature = "datetime")]
    pub fn is_datetime_array(&self) -> bool {
        match self {
            Array::TemporalArray(temp) => match temp {
                TemporalArray::Datetime32(_) => true,
                TemporalArray::Datetime64(_) => true,
                TemporalArray::Null => false,
            },
            Array::NumericArray(_) => false,
            Array::TextArray(_) => false,
            Array::BooleanArray(_) => false,
            Array::Null => false,
        }
    }

    /// Returns the underlying null mask of the array
    pub fn null_mask(&self) -> Option<&Bitmask> {
        match self {
            Array::NumericArray(inner) => match inner {
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(arr) => arr.null_mask.as_ref(),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(arr) => arr.null_mask.as_ref(),
                NumericArray::Int32(arr) => arr.null_mask.as_ref(),
                NumericArray::Int64(arr) => arr.null_mask.as_ref(),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(arr) => arr.null_mask.as_ref(),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(arr) => arr.null_mask.as_ref(),
                NumericArray::UInt32(arr) => arr.null_mask.as_ref(),
                NumericArray::UInt64(arr) => arr.null_mask.as_ref(),
                NumericArray::Float32(arr) => arr.null_mask.as_ref(),
                NumericArray::Float64(arr) => arr.null_mask.as_ref(),
                NumericArray::Null => None,
            },
            Array::BooleanArray(arr) => arr.null_mask.as_ref(),
            Array::TextArray(inner) => match inner {
                TextArray::String32(arr) => arr.null_mask.as_ref(),
                #[cfg(feature = "large_string")]
                TextArray::String64(arr) => arr.null_mask.as_ref(),
                #[cfg(feature = "default_categorical_8")]
                TextArray::Categorical8(arr) => arr.null_mask.as_ref(),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(arr) => arr.null_mask.as_ref(),
                #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
                TextArray::Categorical32(arr) => arr.null_mask.as_ref(),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(arr) => arr.null_mask.as_ref(),
                TextArray::Null => None,
            },
            #[cfg(feature = "datetime")]
            Array::TemporalArray(inner) => match inner {
                TemporalArray::Datetime32(arr) => arr.null_mask.as_ref(),
                TemporalArray::Datetime64(arr) => arr.null_mask.as_ref(),
                TemporalArray::Null => None,
            },
            Array::Null => None,
        }
    }

    /// Returns true when the array holds at least one null.
    ///
    /// Delegates straight to the variant's `has_nulls`, which itself resolves
    /// to the inner array's `MaskedArray::has_nulls`. `Null` is treated as
    /// empty and reports no nulls.
    #[inline]
    pub fn has_nulls(&self) -> bool {
        match self {
            Array::NumericArray(inner) => inner.has_nulls(),
            Array::BooleanArray(arr) => arr.has_nulls(),
            Array::TextArray(inner) => inner.has_nulls(),
            #[cfg(feature = "datetime")]
            Array::TemporalArray(inner) => inner.has_nulls(),
            Array::Null => false,
        }
    }

    // ── Element-level operations ────────────────────────────────────

    /// Format the element at `idx` as a human-readable string.
    ///
    /// Returns `"null"` for null elements. Uses the same formatting as
    /// the array's Display implementation.
    #[inline]
    pub fn value_to_string(&self, idx: usize) -> String {
        crate::traits::print::value_to_string(self, idx)
    }

    /// Extract the element at `idx` as a `Scalar`, or `None` if out of bounds.
    ///
    /// Returns `Scalar::Null` for null elements.
    #[cfg(feature = "scalar_type")]
    pub fn get_scalar(&self, idx: usize) -> Option<crate::Scalar> {
        use crate::Scalar;
        if idx >= self.len() {
            return None;
        }
        let is_null = self.null_mask().is_some_and(|m| !m.get(idx));
        if is_null {
            return Some(Scalar::Null);
        }
        match self {
            Array::NumericArray(num) => match num {
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(a) => Some(Scalar::Int8(a.data[idx])),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(a) => Some(Scalar::Int16(a.data[idx])),
                NumericArray::Int32(a) => Some(Scalar::Int32(a.data[idx])),
                NumericArray::Int64(a) => Some(Scalar::Int64(a.data[idx])),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(a) => Some(Scalar::UInt8(a.data[idx])),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(a) => Some(Scalar::UInt16(a.data[idx])),
                NumericArray::UInt32(a) => Some(Scalar::UInt32(a.data[idx])),
                NumericArray::UInt64(a) => Some(Scalar::UInt64(a.data[idx])),
                NumericArray::Float32(a) => Some(Scalar::Float32(a.data[idx])),
                NumericArray::Float64(a) => Some(Scalar::Float64(a.data[idx])),
                NumericArray::Null => Some(Scalar::Null),
            },
            Array::TextArray(text) => match text {
                TextArray::String32(a) => Some(Scalar::String32(a.get_str(idx)?.to_owned())),
                #[cfg(feature = "large_string")]
                TextArray::String64(a) => Some(Scalar::String64(a.get_str(idx)?.to_owned())),
                #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
                TextArray::Categorical32(a) => Some(Scalar::String32(a.get_str(idx)?.to_owned())),
                #[cfg(feature = "default_categorical_8")]
                TextArray::Categorical8(a) => Some(Scalar::String32(a.get_str(idx)?.to_owned())),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(a) => Some(Scalar::String32(a.get_str(idx)?.to_owned())),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(a) => Some(Scalar::String32(a.get_str(idx)?.to_owned())),
                TextArray::Null => Some(Scalar::Null),
            },
            Array::BooleanArray(a) => Some(Scalar::Boolean(a.get(idx)?)),
            #[cfg(feature = "datetime")]
            Array::TemporalArray(temp) => match temp {
                crate::TemporalArray::Datetime32(a) => Some(Scalar::Datetime32(a.data[idx])),
                crate::TemporalArray::Datetime64(a) => Some(Scalar::Datetime64(a.data[idx])),
                crate::TemporalArray::Null => Some(Scalar::Null),
            },
            Array::Null => Some(Scalar::Null),
        }
    }

    /// Create an all-null array of the given ArrowType with `n_rows` elements.
    ///
    /// The data buffer is zero-filled and every element is masked as null.
    /// For datetime types, set the time_unit on the returned array afterwards.
    pub fn null_array(arrow_type: &ArrowType, n_rows: usize) -> Array {
        let mask = Bitmask::with_capacity(n_rows);
        match arrow_type {
            ArrowType::Null => Array::Null,
            ArrowType::Boolean => {
                Array::BooleanArray(Arc::new(BooleanArray::from_vec(vec![false; n_rows], Some(mask))))
            }
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::Int8 => {
                Array::from_int8(IntegerArray::new(Vec64::from_slice(&vec![0i8; n_rows]), Some(mask)))
            }
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::Int16 => {
                Array::from_int16(IntegerArray::new(Vec64::from_slice(&vec![0i16; n_rows]), Some(mask)))
            }
            ArrowType::Int32 => {
                Array::from_int32(IntegerArray::new(Vec64::from_slice(&vec![0i32; n_rows]), Some(mask)))
            }
            ArrowType::Int64 => {
                Array::from_int64(IntegerArray::new(Vec64::from_slice(&vec![0i64; n_rows]), Some(mask)))
            }
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::UInt8 => {
                Array::NumericArray(NumericArray::UInt8(Arc::new(IntegerArray::new(Vec64::from_slice(&vec![0u8; n_rows]), Some(mask)))))
            }
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::UInt16 => {
                Array::NumericArray(NumericArray::UInt16(Arc::new(IntegerArray::new(Vec64::from_slice(&vec![0u16; n_rows]), Some(mask)))))
            }
            ArrowType::UInt32 => {
                Array::NumericArray(NumericArray::UInt32(Arc::new(IntegerArray::new(Vec64::from_slice(&vec![0u32; n_rows]), Some(mask)))))
            }
            ArrowType::UInt64 => {
                Array::NumericArray(NumericArray::UInt64(Arc::new(IntegerArray::new(Vec64::from_slice(&vec![0u64; n_rows]), Some(mask)))))
            }
            ArrowType::Float32 => {
                Array::NumericArray(NumericArray::Float32(Arc::new(FloatArray::new(Vec64::from_slice(&vec![0.0f32; n_rows]), Some(mask)))))
            }
            ArrowType::Float64 => {
                Array::from_float64(FloatArray::new(Vec64::from_slice(&vec![0.0f64; n_rows]), Some(mask)))
            }
            ArrowType::String => {
                let strs: Vec<&str> = vec![""; n_rows];
                let mut arr = StringArray::<u32>::from_slice(&strs);
                arr.null_mask = Some(mask);
                Array::from_string32(arr)
            }
            #[cfg(feature = "large_string")]
            ArrowType::LargeString => {
                let strs: Vec<&str> = vec![""; n_rows];
                let mut arr = StringArray::<u64>::from_slice(&strs);
                arr.null_mask = Some(mask);
                Array::TextArray(TextArray::String64(Arc::new(arr)))
            }
            ArrowType::Dictionary(cat_idx) => {
                let strs: Vec<&str> = vec![""; n_rows];
                match cat_idx {
                    #[cfg(feature = "default_categorical_8")]
                    CategoricalIndexType::UInt8 => {
                        let mut arr = CategoricalArray::<u8>::from_vec(strs, None);
                        arr.null_mask = Some(mask);
                        Array::TextArray(TextArray::Categorical8(Arc::new(arr)))
                    }
                    #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
                    CategoricalIndexType::UInt32 => {
                        let mut arr = CategoricalArray::<u32>::from_vec(strs, None);
                        arr.null_mask = Some(mask);
                        Array::TextArray(TextArray::Categorical32(Arc::new(arr)))
                    }
                    #[cfg(feature = "extended_categorical")]
                    _ => Array::Null,
                }
            }
            #[cfg(feature = "datetime")]
            ArrowType::Date32 | ArrowType::Time32(_) | ArrowType::Duration32(_) => {
                Array::TemporalArray(crate::TemporalArray::Datetime32(Arc::new(
                    crate::DatetimeArray::new(Vec64::from_slice(&vec![0i32; n_rows]), Some(mask), None),
                )))
            }
            #[cfg(feature = "datetime")]
            ArrowType::Date64
            | ArrowType::Time64(_)
            | ArrowType::Duration64(_)
            | ArrowType::Timestamp(_, _) => {
                Array::TemporalArray(crate::TemporalArray::Datetime64(Arc::new(
                    crate::DatetimeArray::new(Vec64::from_slice(&vec![0i64; n_rows]), Some(mask), None),
                )))
            }
            #[cfg(feature = "datetime")]
            ArrowType::Interval(_) => Array::Null,
            ArrowType::Utf8View => {
                let strs: Vec<&str> = vec![""; n_rows];
                let mut arr = StringArray::<u32>::from_slice(&strs);
                arr.null_mask = Some(mask);
                Array::from_string32(arr)
            }
        }
    }

    /// Zero-row Array of the given ArrowType, built from each variant's Default.
    pub fn from_arrow_dtype(dtype: &ArrowType) -> Array {
        match dtype {
            ArrowType::Null => Array::Null,
            ArrowType::Boolean => Array::BooleanArray(Arc::new(BooleanArray::<()>::default())),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::Int8 => Array::from_int8(IntegerArray::<i8>::default()),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::Int16 => Array::from_int16(IntegerArray::<i16>::default()),
            ArrowType::Int32 => Array::from_int32(IntegerArray::<i32>::default()),
            ArrowType::Int64 => Array::from_int64(IntegerArray::<i64>::default()),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::UInt8 => Array::NumericArray(NumericArray::UInt8(Arc::new(
                IntegerArray::<u8>::default(),
            ))),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::UInt16 => Array::NumericArray(NumericArray::UInt16(Arc::new(
                IntegerArray::<u16>::default(),
            ))),
            ArrowType::UInt32 => Array::NumericArray(NumericArray::UInt32(Arc::new(
                IntegerArray::<u32>::default(),
            ))),
            ArrowType::UInt64 => Array::NumericArray(NumericArray::UInt64(Arc::new(
                IntegerArray::<u64>::default(),
            ))),
            ArrowType::Float32 => Array::NumericArray(NumericArray::Float32(Arc::new(
                FloatArray::<f32>::default(),
            ))),
            ArrowType::Float64 => Array::from_float64(FloatArray::<f64>::default()),
            ArrowType::String | ArrowType::Utf8View => {
                Array::from_string32(StringArray::<u32>::default())
            }
            #[cfg(feature = "large_string")]
            ArrowType::LargeString => Array::TextArray(TextArray::String64(Arc::new(
                StringArray::<u64>::default(),
            ))),
            ArrowType::Dictionary(cat_idx) => match cat_idx {
                #[cfg(feature = "default_categorical_8")]
                CategoricalIndexType::UInt8 => Array::TextArray(TextArray::Categorical8(
                    Arc::new(CategoricalArray::<u8>::default()),
                )),
                #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
                CategoricalIndexType::UInt32 => Array::TextArray(TextArray::Categorical32(
                    Arc::new(CategoricalArray::<u32>::default()),
                )),
                #[cfg(feature = "extended_categorical")]
                CategoricalIndexType::UInt16 => Array::TextArray(TextArray::Categorical16(
                    Arc::new(CategoricalArray::<u16>::default()),
                )),
                #[cfg(feature = "extended_categorical")]
                CategoricalIndexType::UInt64 => Array::TextArray(TextArray::Categorical64(
                    Arc::new(CategoricalArray::<u64>::default()),
                )),
                #[allow(unreachable_patterns)]
                _ => Array::Null,
            },
            #[cfg(feature = "datetime")]
            ArrowType::Date32 | ArrowType::Time32(_) | ArrowType::Duration32(_) => {
                Array::TemporalArray(crate::TemporalArray::Datetime32(Arc::new(
                    crate::DatetimeArray::<i32>::default(),
                )))
            }
            #[cfg(feature = "datetime")]
            ArrowType::Date64
            | ArrowType::Time64(_)
            | ArrowType::Duration64(_)
            | ArrowType::Timestamp(_, _) => Array::TemporalArray(crate::TemporalArray::Datetime64(
                Arc::new(crate::DatetimeArray::<i64>::default()),
            )),
            #[cfg(feature = "datetime")]
            ArrowType::Interval(crate::IntervalUnit::YearMonth) => {
                Array::TemporalArray(crate::TemporalArray::Datetime32(Arc::new(
                    crate::DatetimeArray::<i32>::default(),
                )))
            }
            #[cfg(feature = "datetime")]
            ArrowType::Interval(_) => Array::TemporalArray(crate::TemporalArray::Datetime64(
                Arc::new(crate::DatetimeArray::<i64>::default()),
            )),
        }
    }

    /// Build an array from a slice of Scalars.
    ///
    /// All scalars must be the same type. The type is inferred from the first
    /// non-Null element. If all elements are Null, returns `Array::Null`.
    #[cfg(feature = "scalar_type")]
    pub fn from_scalars(scalars: &[crate::Scalar]) -> Array {
        use crate::Scalar;
        if scalars.is_empty() {
            return Array::default();
        }

        // Find the first non-null to determine type
        let template = scalars.iter().find(|s| !matches!(s, Scalar::Null));
        let Some(template) = template else {
            return Array::Null;
        };

        match template {
            Scalar::Float64(_) => {
                let mut data = Vec64::<f64>::with_capacity(scalars.len());
                let mut mask = Bitmask::new_set_all(scalars.len(), true);
                for (i, s) in scalars.iter().enumerate() {
                    match s {
                        Scalar::Float64(v) => data.push(*v),
                        Scalar::Null => { data.push(0.0); mask.set(i, false); }
                        _ => data.push(s.f64()),
                    }
                }
                let has_nulls = mask.count_zeros() > 0;
                Array::from_float64(FloatArray::new(crate::Buffer::from_vec64(data), if has_nulls { Some(mask) } else { None }))
            }
            Scalar::Float32(_) => {
                let mut data = Vec64::<f32>::with_capacity(scalars.len());
                let mut mask = Bitmask::new_set_all(scalars.len(), true);
                for (i, s) in scalars.iter().enumerate() {
                    match s {
                        Scalar::Float32(v) => data.push(*v),
                        Scalar::Null => { data.push(0.0); mask.set(i, false); }
                        _ => data.push(s.f64() as f32),
                    }
                }
                let has_nulls = mask.count_zeros() > 0;
                Array::NumericArray(NumericArray::Float32(Arc::new(FloatArray::new(crate::Buffer::from_vec64(data), if has_nulls { Some(mask) } else { None }))))
            }
            Scalar::Int32(_) => {
                let mut data = Vec64::<i32>::with_capacity(scalars.len());
                let mut mask = Bitmask::new_set_all(scalars.len(), true);
                for (i, s) in scalars.iter().enumerate() {
                    match s {
                        Scalar::Int32(v) => data.push(*v),
                        Scalar::Null => { data.push(0); mask.set(i, false); }
                        _ => data.push(s.f64() as i32),
                    }
                }
                let has_nulls = mask.count_zeros() > 0;
                Array::from_int32(IntegerArray::new(crate::Buffer::from_vec64(data), if has_nulls { Some(mask) } else { None }))
            }
            Scalar::Int64(_) => {
                let mut data = Vec64::<i64>::with_capacity(scalars.len());
                let mut mask = Bitmask::new_set_all(scalars.len(), true);
                for (i, s) in scalars.iter().enumerate() {
                    match s {
                        Scalar::Int64(v) => data.push(*v),
                        Scalar::Null => { data.push(0); mask.set(i, false); }
                        _ => data.push(s.f64() as i64),
                    }
                }
                let has_nulls = mask.count_zeros() > 0;
                Array::from_int64(IntegerArray::new(crate::Buffer::from_vec64(data), if has_nulls { Some(mask) } else { None }))
            }
            Scalar::UInt32(_) => {
                let mut data = Vec64::<u32>::with_capacity(scalars.len());
                let mut mask = Bitmask::new_set_all(scalars.len(), true);
                for (i, s) in scalars.iter().enumerate() {
                    match s {
                        Scalar::UInt32(v) => data.push(*v),
                        Scalar::Null => { data.push(0); mask.set(i, false); }
                        _ => data.push(s.f64() as u32),
                    }
                }
                let has_nulls = mask.count_zeros() > 0;
                Array::NumericArray(NumericArray::UInt32(Arc::new(IntegerArray::new(crate::Buffer::from_vec64(data), if has_nulls { Some(mask) } else { None }))))
            }
            Scalar::UInt64(_) => {
                let mut data = Vec64::<u64>::with_capacity(scalars.len());
                let mut mask = Bitmask::new_set_all(scalars.len(), true);
                for (i, s) in scalars.iter().enumerate() {
                    match s {
                        Scalar::UInt64(v) => data.push(*v),
                        Scalar::Null => { data.push(0); mask.set(i, false); }
                        _ => data.push(s.f64() as u64),
                    }
                }
                let has_nulls = mask.count_zeros() > 0;
                Array::NumericArray(NumericArray::UInt64(Arc::new(IntegerArray::new(crate::Buffer::from_vec64(data), if has_nulls { Some(mask) } else { None }))))
            }
            Scalar::Boolean(_) => {
                let mut data = Vec::with_capacity(scalars.len());
                let mut mask = Bitmask::new_set_all(scalars.len(), true);
                for (i, s) in scalars.iter().enumerate() {
                    match s {
                        Scalar::Boolean(v) => data.push(*v),
                        Scalar::Null => { data.push(false); mask.set(i, false); }
                        _ => data.push(false),
                    }
                }
                let has_nulls = mask.count_zeros() > 0;
                Array::BooleanArray(Arc::new(BooleanArray::from_vec(data, if has_nulls { Some(mask) } else { None })))
            }
            Scalar::String32(_) => {
                let strs: Vec<String> = scalars.iter().map(|s| match s {
                    Scalar::String32(v) => v.clone(),
                    #[cfg(feature = "large_string")]
                    Scalar::String64(v) => v.clone(),
                    Scalar::Null => String::new(),
                    _ => String::new(),
                }).collect();
                let mut mask = Bitmask::new_set_all(scalars.len(), true);
                for (i, s) in scalars.iter().enumerate() {
                    if matches!(s, Scalar::Null) { mask.set(i, false); }
                }
                let refs: Vec<&str> = strs.iter().map(|s| s.as_str()).collect();
                let mut arr = StringArray::<u32>::from_slice(&refs);
                let has_nulls = mask.count_zeros() > 0;
                if has_nulls { arr.null_mask = Some(mask); }
                Array::from_string32(arr)
            }
            #[cfg(feature = "large_string")]
            Scalar::String64(_) => {
                let strs: Vec<String> = scalars.iter().map(|s| match s {
                    Scalar::String64(v) | Scalar::String32(v) => v.clone(),
                    Scalar::Null => String::new(),
                    _ => String::new(),
                }).collect();
                let mut mask = Bitmask::new_set_all(scalars.len(), true);
                for (i, s) in scalars.iter().enumerate() {
                    if matches!(s, Scalar::Null) { mask.set(i, false); }
                }
                let refs: Vec<&str> = strs.iter().map(|s| s.as_str()).collect();
                let mut arr = StringArray::<u64>::from_slice(&refs);
                let has_nulls = mask.count_zeros() > 0;
                if has_nulls { arr.null_mask = Some(mask); }
                Array::TextArray(TextArray::String64(Arc::new(arr)))
            }
            #[cfg(feature = "datetime")]
            Scalar::Datetime32(_) => {
                let mut data = Vec64::<i32>::with_capacity(scalars.len());
                let mut mask = Bitmask::new_set_all(scalars.len(), true);
                for (i, s) in scalars.iter().enumerate() {
                    match s {
                        Scalar::Datetime32(v) => data.push(*v),
                        Scalar::Null => { data.push(0); mask.set(i, false); }
                        _ => data.push(0),
                    }
                }
                let has_nulls = mask.count_zeros() > 0;
                Array::TemporalArray(crate::TemporalArray::Datetime32(Arc::new(
                    crate::DatetimeArray::new(crate::Buffer::from_vec64(data), if has_nulls { Some(mask) } else { None }, None),
                )))
            }
            #[cfg(feature = "datetime")]
            Scalar::Datetime64(_) => {
                let mut data = Vec64::<i64>::with_capacity(scalars.len());
                let mut mask = Bitmask::new_set_all(scalars.len(), true);
                for (i, s) in scalars.iter().enumerate() {
                    match s {
                        Scalar::Datetime64(v) => data.push(*v),
                        Scalar::Null => { data.push(0); mask.set(i, false); }
                        _ => data.push(0),
                    }
                }
                let has_nulls = mask.count_zeros() > 0;
                Array::TemporalArray(crate::TemporalArray::Datetime64(Arc::new(
                    crate::DatetimeArray::new(crate::Buffer::from_vec64(data), if has_nulls { Some(mask) } else { None }, None),
                )))
            }
            #[cfg(feature = "datetime")]
            Scalar::Interval => Array::Null,
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int8(_) => {
                let mut data = Vec64::<i8>::with_capacity(scalars.len());
                let mut mask = Bitmask::new_set_all(scalars.len(), true);
                for (i, s) in scalars.iter().enumerate() {
                    match s {
                        Scalar::Int8(v) => data.push(*v),
                        Scalar::Null => { data.push(0); mask.set(i, false); }
                        _ => data.push(s.f64() as i8),
                    }
                }
                let has_nulls = mask.count_zeros() > 0;
                Array::from_int8(IntegerArray::new(crate::Buffer::from_vec64(data), if has_nulls { Some(mask) } else { None }))
            }
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int16(_) => {
                let mut data = Vec64::<i16>::with_capacity(scalars.len());
                let mut mask = Bitmask::new_set_all(scalars.len(), true);
                for (i, s) in scalars.iter().enumerate() {
                    match s {
                        Scalar::Int16(v) => data.push(*v),
                        Scalar::Null => { data.push(0); mask.set(i, false); }
                        _ => data.push(s.f64() as i16),
                    }
                }
                let has_nulls = mask.count_zeros() > 0;
                Array::from_int16(IntegerArray::new(crate::Buffer::from_vec64(data), if has_nulls { Some(mask) } else { None }))
            }
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt8(_) => {
                let mut data = Vec64::<u8>::with_capacity(scalars.len());
                let mut mask = Bitmask::new_set_all(scalars.len(), true);
                for (i, s) in scalars.iter().enumerate() {
                    match s {
                        Scalar::UInt8(v) => data.push(*v),
                        Scalar::Null => { data.push(0); mask.set(i, false); }
                        _ => data.push(s.f64() as u8),
                    }
                }
                let has_nulls = mask.count_zeros() > 0;
                Array::from_uint8(IntegerArray::new(crate::Buffer::from_vec64(data), if has_nulls { Some(mask) } else { None }))
            }
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt16(_) => {
                let mut data = Vec64::<u16>::with_capacity(scalars.len());
                let mut mask = Bitmask::new_set_all(scalars.len(), true);
                for (i, s) in scalars.iter().enumerate() {
                    match s {
                        Scalar::UInt16(v) => data.push(*v),
                        Scalar::Null => { data.push(0); mask.set(i, false); }
                        _ => data.push(s.f64() as u16),
                    }
                }
                let has_nulls = mask.count_zeros() > 0;
                Array::from_uint16(IntegerArray::new(crate::Buffer::from_vec64(data), if has_nulls { Some(mask) } else { None }))
            }
            Scalar::Null => Array::Null,
        }
    }

    /// Compare two elements within the same array by index.
    ///
    /// Uses total ordering for floats via `total_cmp()`. Nulls sort last:
    /// null > any value, null == null.
    pub fn compare_at(&self, i: usize, j: usize) -> std::cmp::Ordering {
        use std::cmp::Ordering;

        // Handle nulls first: null sorts last
        let null_mask = self.null_mask();
        let i_null = null_mask.is_some_and(|m| !m.get(i));
        let j_null = null_mask.is_some_and(|m| !m.get(j));
        match (i_null, j_null) {
            (true, true) => return Ordering::Equal,
            (true, false) => return Ordering::Greater,
            (false, true) => return Ordering::Less,
            (false, false) => {}
        }

        match self {
            Array::NumericArray(inner) => match inner {
                NumericArray::Int32(a) => a.data[i].cmp(&a.data[j]),
                NumericArray::Int64(a) => a.data[i].cmp(&a.data[j]),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(a) => a.data[i].cmp(&a.data[j]),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(a) => a.data[i].cmp(&a.data[j]),
                NumericArray::UInt32(a) => a.data[i].cmp(&a.data[j]),
                NumericArray::UInt64(a) => a.data[i].cmp(&a.data[j]),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(a) => a.data[i].cmp(&a.data[j]),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(a) => a.data[i].cmp(&a.data[j]),
                NumericArray::Float32(a) => a.data[i].total_cmp(&a.data[j]),
                NumericArray::Float64(a) => a.data[i].total_cmp(&a.data[j]),
                NumericArray::Null => Ordering::Equal,
            },
            Array::BooleanArray(b) => b.data.get(i).cmp(&b.data.get(j)),
            Array::TextArray(inner) => match inner {
                TextArray::String32(s) => s.get_str(i).cmp(&s.get_str(j)),
                #[cfg(feature = "large_string")]
                TextArray::String64(s) => s.get_str(i).cmp(&s.get_str(j)),
                #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
                TextArray::Categorical32(c) => c.get_str(i).cmp(&c.get_str(j)),
                #[cfg(feature = "default_categorical_8")]
                TextArray::Categorical8(c) => c.get_str(i).cmp(&c.get_str(j)),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(c) => c.get_str(i).cmp(&c.get_str(j)),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(c) => c.get_str(i).cmp(&c.get_str(j)),
                TextArray::Null => Ordering::Equal,
            },
            #[cfg(feature = "datetime")]
            Array::TemporalArray(inner) => match inner {
                TemporalArray::Datetime32(a) => a.data[i].cmp(&a.data[j]),
                TemporalArray::Datetime64(a) => a.data[i].cmp(&a.data[j]),
                TemporalArray::Null => Ordering::Equal,
            },
            Array::Null => Ordering::Equal,
        }
    }

    /// Hash the element at `idx` into the provided hasher.
    ///
    /// Null elements hash a fixed sentinel. Floats use `to_bits()` so
    /// the hash is consistent with the `to_bits()` equality convention.
    #[cfg(feature = "hash")]
    pub fn hash_element_at<H: std::hash::Hasher>(&self, idx: usize, state: &mut H) {
        use std::hash::Hash;

        // Hash null status
        if let Some(mask) = self.null_mask() {
            if !mask.get(idx) {
                // Sentinel for null
                0xDEAD_BEEF_u64.hash(state);
                return;
            }
        }

        match self {
            Array::NumericArray(inner) => match inner {
                NumericArray::Int32(a) => a.data[idx].hash(state),
                NumericArray::Int64(a) => a.data[idx].hash(state),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(a) => a.data[idx].hash(state),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(a) => a.data[idx].hash(state),
                NumericArray::UInt32(a) => a.data[idx].hash(state),
                NumericArray::UInt64(a) => a.data[idx].hash(state),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(a) => a.data[idx].hash(state),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(a) => a.data[idx].hash(state),
                NumericArray::Float32(a) => a.data[idx].to_bits().hash(state),
                NumericArray::Float64(a) => a.data[idx].to_bits().hash(state),
                NumericArray::Null => 0xDEAD_BEEF_u64.hash(state),
            },
            Array::BooleanArray(b) => b.data.get(idx).hash(state),
            Array::TextArray(inner) => match inner {
                TextArray::String32(s) => s.get_str(idx).hash(state),
                #[cfg(feature = "large_string")]
                TextArray::String64(s) => s.get_str(idx).hash(state),
                #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
                TextArray::Categorical32(c) => c.get_str(idx).hash(state),
                #[cfg(feature = "default_categorical_8")]
                TextArray::Categorical8(c) => c.get_str(idx).hash(state),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(c) => c.get_str(idx).hash(state),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(c) => c.get_str(idx).hash(state),
                TextArray::Null => 0xDEAD_BEEF_u64.hash(state),
            },
            #[cfg(feature = "datetime")]
            Array::TemporalArray(inner) => match inner {
                TemporalArray::Datetime32(a) => a.data[idx].hash(state),
                TemporalArray::Datetime64(a) => a.data[idx].hash(state),
                TemporalArray::Null => 0xDEAD_BEEF_u64.hash(state),
            },
            Array::Null => 0xDEAD_BEEF_u64.hash(state),
        }
    }

    /// Set null mask on Array by matching on variants
    pub fn set_null_mask(&mut self, mask: Bitmask) {
        match self {
            Array::NumericArray(num_arr) => {
                match num_arr {
                    NumericArray::Int32(arr) => {
                        Arc::make_mut(arr).set_null_mask(Some(mask));
                    }
                    NumericArray::Int64(arr) => {
                        Arc::make_mut(arr).set_null_mask(Some(mask));
                    }
                    NumericArray::Float32(arr) => {
                        Arc::make_mut(arr).set_null_mask(Some(mask));
                    }
                    NumericArray::Float64(arr) => {
                        Arc::make_mut(arr).set_null_mask(Some(mask));
                    }
                    #[cfg(feature = "extended_numeric_types")]
                    NumericArray::Int8(arr) => {
                        Arc::make_mut(arr).set_null_mask(Some(mask));
                    }
                    #[cfg(feature = "extended_numeric_types")]
                    NumericArray::Int16(arr) => {
                        Arc::make_mut(arr).set_null_mask(Some(mask));
                    }
                    #[cfg(feature = "extended_numeric_types")]
                    NumericArray::UInt8(arr) => {
                        Arc::make_mut(arr).set_null_mask(Some(mask));
                    }
                    #[cfg(feature = "extended_numeric_types")]
                    NumericArray::UInt16(arr) => {
                        Arc::make_mut(arr).set_null_mask(Some(mask));
                    }
                    NumericArray::UInt32(arr) => {
                        Arc::make_mut(arr).set_null_mask(Some(mask));
                    }
                    NumericArray::UInt64(arr) => {
                        Arc::make_mut(arr).set_null_mask(Some(mask));
                    }
                    NumericArray::Null => {} // No-op for null arrays
                }
            }
            Array::TextArray(text_arr) => match text_arr {
                TextArray::String32(arr) => {
                    Arc::make_mut(arr).set_null_mask(Some(mask));
                }
                #[cfg(feature = "large_string")]
                TextArray::String64(arr) => {
                    Arc::make_mut(arr).set_null_mask(Some(mask));
                }
                #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
                TextArray::Categorical32(arr) => {
                    Arc::make_mut(arr).set_null_mask(Some(mask));
                }
                #[cfg(feature = "default_categorical_8")]
                TextArray::Categorical8(arr) => {
                    Arc::make_mut(arr).set_null_mask(Some(mask));
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(arr) => {
                    Arc::make_mut(arr).set_null_mask(Some(mask));
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(arr) => {
                    Arc::make_mut(arr).set_null_mask(Some(mask));
                }
                TextArray::Null => {}
            },
            #[cfg(feature = "datetime")]
            Array::TemporalArray(temp_arr) => match temp_arr {
                TemporalArray::Datetime32(arr) => {
                    Arc::make_mut(arr).set_null_mask(Some(mask));
                }
                TemporalArray::Datetime64(arr) => {
                    Arc::make_mut(arr).set_null_mask(Some(mask));
                }
                TemporalArray::Null => {}
            },
            Array::BooleanArray(arr) => {
                Arc::make_mut(arr).set_null_mask(Some(mask));
            }
            Array::Null => {}
        }
    }

    /// Returns a pointer to the backing data (contiguous bytes), length in elements, and element size.
    ///
    /// This is not logical length - it is total raw bytes in the buffer, so for non-fixed width
    /// types such as bit-packed booleans or strings, please factor this in accordingly.
    pub fn data_ptr_and_byte_len(&self) -> (*const u8, usize, usize) {
        match self {
            Array::NumericArray(inner) => match inner {
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<i8>(),
                ),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<i16>(),
                ),
                NumericArray::Int32(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<i32>(),
                ),
                NumericArray::Int64(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<i64>(),
                ),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<u8>(),
                ),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<u16>(),
                ),
                NumericArray::UInt32(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<u32>(),
                ),
                NumericArray::UInt64(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<u64>(),
                ),
                NumericArray::Float32(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<f32>(),
                ),
                NumericArray::Float64(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<f64>(),
                ),
                NumericArray::Null => (std::ptr::null(), 0, 0),
            },
            Array::BooleanArray(a) => (a.data.as_ptr() as *const u8, a.data.len(), 1),
            Array::TextArray(inner) => match inner {
                TextArray::String32(a) => (a.data.as_ptr(), a.data.len(), 1),
                #[cfg(feature = "large_string")]
                TextArray::String64(a) => (a.data.as_ptr(), a.data.len(), 1),
                #[cfg(feature = "default_categorical_8")]
                TextArray::Categorical8(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<u8>(),
                ),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<u16>(),
                ),
                #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
                TextArray::Categorical32(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<u32>(),
                ),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<u64>(),
                ),
                TextArray::Null => (std::ptr::null(), 0, 0),
            },
            #[cfg(feature = "datetime")]
            Array::TemporalArray(inner) => match inner {
                TemporalArray::Datetime32(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<i32>(),
                ),
                TemporalArray::Datetime64(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<i64>(),
                ),
                TemporalArray::Null => (std::ptr::null(), 0, 0),
            },
            Array::Null => (std::ptr::null(), 0, 0),
        }
    }

    /// Returns a pointer to the null mask and its length in bytes, if present.
    pub fn null_mask_ptr_and_byte_len(&self) -> Option<(*const u8, usize)> {
        match self {
            Array::NumericArray(inner) => match inner {
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(a) => a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity())),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(a) => a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity())),
                NumericArray::Int32(a) => a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity())),
                NumericArray::Int64(a) => a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity())),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(a) => a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity())),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(a) => a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity())),
                NumericArray::UInt32(a) => a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity())),
                NumericArray::UInt64(a) => a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity())),
                NumericArray::Float32(a) => {
                    a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity()))
                }
                NumericArray::Float64(a) => {
                    a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity()))
                }
                NumericArray::Null => None,
            },
            Array::BooleanArray(a) => a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity())),
            Array::TextArray(inner) => match inner {
                TextArray::String32(a) => a.null_mask.as_ref().map(|m| (m.as_ptr(), m.len())),
                #[cfg(feature = "large_string")]
                TextArray::String64(a) => a.null_mask.as_ref().map(|m| (m.as_ptr(), m.len())),
                #[cfg(feature = "default_categorical_8")]
                TextArray::Categorical8(a) => {
                    a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity()))
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(a) => {
                    a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity()))
                }
                #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
                TextArray::Categorical32(a) => {
                    a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity()))
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(a) => {
                    a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity()))
                }
                TextArray::Null => None,
            },
            #[cfg(feature = "datetime")]
            Array::TemporalArray(inner) => match inner {
                TemporalArray::Datetime32(a) => {
                    a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity()))
                }
                TemporalArray::Datetime64(a) => {
                    a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity()))
                }
                TemporalArray::Null => None,
            },
            Array::Null => None,
        }
    }

    /// Offsets pointer/len for variable-length types
    pub fn offsets_ptr_and_len(&self) -> Option<(*const u8, usize)> {
        match self {
            Array::TextArray(inner) => match inner {
                TextArray::String32(a) => Some((
                    a.offsets.as_ptr() as *const u8,
                    a.offsets.len() * std::mem::size_of::<u32>(),
                )),
                #[cfg(feature = "large_string")]
                TextArray::String64(a) => Some((
                    a.offsets.as_ptr() as *const u8,
                    a.offsets.len() * std::mem::size_of::<u64>(),
                )),
                _ => None,
            },
            _ => None,
        }
    }

    /// Returns the null count of the array
    pub fn null_count(&self) -> usize {
        match self {
            Array::NumericArray(inner) => match inner {
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(a) => a.null_count(),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(a) => a.null_count(),
                NumericArray::Int32(a) => a.null_count(),
                NumericArray::Int64(a) => a.null_count(),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(a) => a.null_count(),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(a) => a.null_count(),
                NumericArray::UInt32(a) => a.null_count(),
                NumericArray::UInt64(a) => a.null_count(),
                NumericArray::Float32(a) => a.null_count(),
                NumericArray::Float64(a) => a.null_count(),
                NumericArray::Null => 0,
            },
            Array::BooleanArray(a) => a.null_count(),
            Array::TextArray(inner) => match inner {
                TextArray::String32(a) => a.null_count(),
                #[cfg(feature = "large_string")]
                TextArray::String64(a) => a.null_count(),
                #[cfg(feature = "default_categorical_8")]
                TextArray::Categorical8(a) => a.null_count(),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(a) => a.null_count(),
                #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
                TextArray::Categorical32(a) => a.null_count(),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(a) => a.null_count(),
                TextArray::Null => 0,
            },
            #[cfg(feature = "datetime")]
            Array::TemporalArray(inner) => match inner {
                TemporalArray::Datetime32(a) => a.null_count(),
                TemporalArray::Datetime64(a) => a.null_count(),
                TemporalArray::Null => 0,
            },
            Array::Null => 0,
        }
    }

    /// Appends all values (and null mask if present) from `other` into `self`.
    ///
    /// Panics if the two arrays are of different variants or incompatible types.
    ///
    /// This function uses copy-on-write semantics for arrays wrapped in `Arc`.
    /// If `self` is the only owner of its data, appends are performed in place without copying the first array.
    /// If the array data is shared (`Arc` reference count > 1), the data is first cloned
    /// (so the mutation does not affect other owners), and the append is then performed on the unique copy.
    /// The second array is allocated into the buffer, which is standard.
    pub fn concat_array(&mut self, other: &Self) {
        match (self, other) {
            (Array::NumericArray(lhs), Array::NumericArray(rhs)) => lhs.append_array(rhs),
            (Array::BooleanArray(a), Array::BooleanArray(b)) => Arc::make_mut(a).append_array(b),
            (Array::TextArray(lhs), Array::TextArray(rhs)) => lhs.append_array(rhs),
            #[cfg(feature = "datetime")]
            (Array::TemporalArray(lhs), Array::TemporalArray(rhs)) => lhs.append_array(rhs),
            (Array::Null, Array::Null) => (),
            (lhs, rhs) => panic!(
                "Cannot append {:?} into {:?}",
                rhs.arrow_type(),
                lhs.arrow_type()
            ),
        }
    }

    /// Appends rows `[offset..offset+len)` from another array into self.
    /// Extends data and null masks directly from the source range.
    pub fn concat_array_range(&mut self, other: &Self, offset: usize, len: usize) -> Result<(), MinarrowError> {
        match (self, other) {
            (Array::NumericArray(lhs), Array::NumericArray(rhs)) => lhs.append_range(rhs, offset, len),
            (Array::BooleanArray(a), Array::BooleanArray(b)) => Arc::make_mut(a).append_range(b, offset, len),
            (Array::TextArray(lhs), Array::TextArray(rhs)) => lhs.append_range(rhs, offset, len),
            #[cfg(feature = "datetime")]
            (Array::TemporalArray(lhs), Array::TemporalArray(rhs)) => lhs.append_range(rhs, offset, len),
            (Array::Null, Array::Null) => Ok(()),
            (lhs, rhs) => Err(MinarrowError::TypeError {
                from: "Array",
                to: "Array",
                message: Some(format!(
                    "Cannot append_range {:?} into {:?}",
                    rhs.arrow_type(),
                    lhs.arrow_type()
                )),
            }),
        }
    }

    /// Inserts all values (and null mask if present) from `other` into `self` at the specified index.
    ///
    /// This is an **O(n)** operation.
    ///
    /// Returns an error if the two arrays are of different variants or incompatible types,
    /// or if the index is out of bounds.
    pub fn insert_rows(&mut self, index: usize, other: &Self) -> Result<(), MinarrowError> {
        match (self, other) {
            (Array::NumericArray(lhs), Array::NumericArray(rhs)) => lhs.insert_rows(index, rhs),
            (Array::BooleanArray(a), Array::BooleanArray(b)) => {
                Arc::make_mut(a).insert_rows(index, b)
            }
            (Array::TextArray(lhs), Array::TextArray(rhs)) => lhs.insert_rows(index, rhs),
            #[cfg(feature = "datetime")]
            (Array::TemporalArray(lhs), Array::TemporalArray(rhs)) => lhs.insert_rows(index, rhs),
            (Array::Null, Array::Null) => Ok(()),
            (lhs, rhs) => Err(MinarrowError::TypeError {
                from: "Array",
                to: "Array",
                message: Some(format!(
                    "Cannot insert {} into {}: incompatible types",
                    rhs.arrow_type(),
                    lhs.arrow_type()
                )),
            }),
        }
    }

    /// Splits the Array at the specified index, consuming self and returning a SuperArray
    /// with two FieldArray chunks.
    ///
    /// Splits the underlying buffers (via vec `.split_off()`), allocating new storage for the second half.
    /// More efficient than cloning the entire array but requires allocation.
    #[cfg(feature = "chunked")]
    pub fn split(self, index: usize, field: &Arc<Field>) -> Result<SuperArray, MinarrowError> {
        match self {
            Array::NumericArray(arr) => {
                let (left, right) = arr.split(index)?;
                Ok(SuperArray::from_field_array_chunks(vec![
                    FieldArray::new((**field).clone(), Array::NumericArray(left)),
                    FieldArray::new((**field).clone(), Array::NumericArray(right)),
                ]))
            }
            Array::TextArray(arr) => {
                let (left, right) = arr.split(index)?;
                Ok(SuperArray::from_field_array_chunks(vec![
                    FieldArray::new((**field).clone(), Array::TextArray(left)),
                    FieldArray::new((**field).clone(), Array::TextArray(right)),
                ]))
            }
            Array::BooleanArray(arr) => {
                let (left, right) = arr.split(index)?;
                Ok(SuperArray::from_field_array_chunks(vec![
                    FieldArray::new((**field).clone(), Array::BooleanArray(left)),
                    FieldArray::new((**field).clone(), Array::BooleanArray(right)),
                ]))
            }
            #[cfg(feature = "datetime")]
            Array::TemporalArray(arr) => {
                let (left, right) = arr.split(index)?;
                Ok(SuperArray::from_field_array_chunks(vec![
                    FieldArray::new((**field).clone(), Array::TemporalArray(left)),
                    FieldArray::new((**field).clone(), Array::TemporalArray(right)),
                ]))
            }
            Array::Null => Err(MinarrowError::IndexError(
                "Cannot split Null array".to_string(),
            )),
        }
    }

    // ===========================================================
    // Apache Arrow bridge - tested under tests/apache_arrow.rs
    // ===========================================================
    //
    // For specific logical types such as Timestamp/Time/Duration/Interval,
    // wrap the array in a `FieldArray` with the desired `Field` and
    // call `FieldArray::to_apache_arrow()`.

    /// Build an arrow-rs `ArrayRef`, deriving a `Field` from the array shape.
    ///
    /// Panics on FFI failure. For a fallible variant returning
    /// `Result<_, MinarrowError>`, see [`Array::try_to_apache_arrow`].
    ///
    /// For Timestamp/Time/Duration/Interval, wrap in a `FieldArray` with the
    /// desired `Field` and use `FieldArray::to_apache_arrow()`.
    #[cfg(feature = "cast_arrow")]
    #[inline]
    pub fn to_apache_arrow(&self, name: &str) -> ArrayRef {
        self.try_to_apache_arrow(name)
            .expect("Array::to_apache_arrow failed")
    }

    /// Fallible variant of [`Array::to_apache_arrow`].
    #[cfg(feature = "cast_arrow")]
    pub fn try_to_apache_arrow(&self, name: &str) -> Result<ArrayRef, MinarrowError> {
        let field = Field::from_array(name, self, None);
        crate::ffi::arrow_rs::export(Arc::new(self.clone()), Schema::from(vec![field]))
    }

    // ===========================================================
    // Polars bridge - tested under tests/polars.rs
    // ===========================================================
    //
    // For specific logical types, wrap in a `FieldArray` with the desired
    // `Field` and call `FieldArray::to_polars()`.

    /// Build a Polars Series, deriving a `Field` from the array shape.
    ///
    /// Panics on FFI failure. For a fallible variant, see
    /// [`Array::try_to_polars`].
    ///
    /// For Timestamp/Time/Duration/Interval, wrap in a `FieldArray` with the
    /// desired `Field` and use `FieldArray::to_polars()`.
    #[cfg(feature = "cast_polars")]
    pub fn to_polars(&self, name: &str) -> polars::prelude::Series {
        self.try_to_polars(name)
            .expect("Array::to_polars failed")
    }

    /// Fallible variant of [`Array::to_polars`].
    #[cfg(feature = "cast_polars")]
    pub fn try_to_polars(&self, name: &str) -> Result<polars::prelude::Series, MinarrowError> {
        // Map physical Datetime variants to a sensible Arrow logical type for
        // export; specific Timestamp/Time/Duration/Interval semantics need a
        // FieldArray with an explicit Field.
        #[cfg(feature = "datetime")]
        use crate::{TemporalArray, TimeUnit, ffi::arrow_dtype::ArrowType};

        let field = match self {
            #[cfg(feature = "datetime")]
            Array::TemporalArray(TemporalArray::Datetime32(a)) => {
                let ty = match a.time_unit {
                    TimeUnit::Days => ArrowType::Date32,
                    TimeUnit::Seconds => ArrowType::Time32(TimeUnit::Seconds),
                    TimeUnit::Milliseconds => ArrowType::Time32(TimeUnit::Milliseconds),
                    _ => ArrowType::Date32,
                };
                Field::new(name.to_string(), ty, a.is_nullable(), None)
            }
            #[cfg(feature = "datetime")]
            Array::TemporalArray(TemporalArray::Datetime64(a)) => {
                let ty = match a.time_unit {
                    TimeUnit::Milliseconds => ArrowType::Date64,
                    TimeUnit::Seconds => ArrowType::Timestamp(TimeUnit::Seconds, None),
                    TimeUnit::Microseconds => ArrowType::Timestamp(TimeUnit::Microseconds, None),
                    TimeUnit::Nanoseconds => ArrowType::Timestamp(TimeUnit::Nanoseconds, None),
                    TimeUnit::Days => ArrowType::Date64,
                };
                Field::new(name.to_string(), ty, a.is_nullable(), None)
            }
            _ => Field::from_array(name.to_string(), self, None),
        };

        crate::ffi::polars::export(
            Arc::new(self.clone()),
            name,
            Schema::from(vec![field]),
        )
    }
    // ===========================================================
    // Apache Arrow / Polars import (`from_*`)
    // ===========================================================

    /// Import an arrow-rs `ArrayRef` into a Minarrow `Array`.
    ///
    /// The recovered `Field` (dtype + nullable + metadata) is dropped. Use
    /// [`crate::FieldArray::from_apache_arrow`] to preserve it.
    ///
    /// Panics on FFI failure. For a fallible variant, see
    /// [`Array::try_from_apache_arrow`].
    #[cfg(feature = "cast_arrow")]
    #[inline]
    pub fn from_apache_arrow(arr: &arrow::array::ArrayRef) -> Array {
        Self::try_from_apache_arrow(arr).expect("Array::from_apache_arrow failed")
    }

    /// Fallible variant of [`Array::from_apache_arrow`].
    #[cfg(feature = "cast_arrow")]
    pub fn try_from_apache_arrow(
        arr: &arrow::array::ArrayRef,
    ) -> Result<Array, MinarrowError> {
        let (array_arc, _field) = crate::ffi::arrow_rs::import(arr)?;
        Ok(Arc::try_unwrap(array_arc).unwrap_or_else(|arc| (*arc).clone()))
    }

    /// Import a Polars `Series` into a Minarrow `Array`.
    ///
    /// A polars `Series` is inherently multi-chunked; the canonical mapping
    /// is `Series` <-> [`crate::SuperArray`]. This helper routes through
    /// [`crate::SuperArray::from_polars`] and then **consolidates** the
    /// chunks into a single contiguous, 64-byte aligned buffer. The series
    /// name and recovered Field metadata are dropped on the way through;
    /// use [`crate::FieldArray::from_polars`] to preserve them.
    ///
    /// ## Performance note
    /// Two separate costs to be aware of:
    ///
    /// 1. **Alignment copy**: Polars data is typically 8-byte aligned (per
    ///    the Arrow spec default), while Minarrow uses 64-byte aligned
    ///    `Vec64<T>` buffers for SIMD. Most of the time this results in a
    ///    memory copy to realign on import, unless the source data happens
    ///    to be pre-aligned to 64 bytes. The FFI hand-off itself is
    ///    pointer-level zero-copy; the realignment is done by
    ///    `Buffer::from_shared` when the source isn't 64-byte aligned.
    ///
    /// 2. **Consolidation copy**: Multi-chunk Series are merged into a
    ///    single contiguous buffer, which is a second O(n) allocation and
    ///    copy pass. Single-chunk Series (e.g. after `s.rechunk()` on the
    ///    caller side) skip this step. The consolidation itself is cheap
    ///    on Linux when the `vmap64` feature is enabled.
    ///
    /// In practice you should expect at least one full allocation + copy
    /// when importing polars data into an `Array`. If you would like to
    /// preserve the original chunk boundaries and avoid the consolidation
    /// step, use [`crate::SuperArray::from_polars`] directly - though the
    /// alignment copy will still occur per chunk that isn't pre-aligned.
    ///
    /// Panics on FFI failure. For a fallible variant, see
    /// [`Array::try_from_polars`].
    #[cfg(feature = "cast_polars")]
    #[inline]
    pub fn from_polars(s: &polars::prelude::Series) -> Array {
        Self::try_from_polars(s).expect("Array::from_polars failed")
    }

    /// Fallible variant of [`Array::from_polars`].
    #[cfg(feature = "cast_polars")]
    pub fn try_from_polars(s: &polars::prelude::Series) -> Result<Array, MinarrowError> {
        use crate::SuperArray;
        use crate::traits::consolidate::Consolidate;
        Ok(SuperArray::try_from_polars(s)?.consolidate())
    }
}

/// Reinterpret-cast a `&[U]` window to `&[T]` when the caller has separately
/// established that `U` and `T` are layout-compatible. Used by variant
/// dispatch sites in this crate that already pattern-match on the concrete
/// inner type.
#[inline(always)]
pub(crate) fn cast_slice<'a, U, T>(data: &'a [U], offset: usize, len: usize) -> Option<&'a [T]> {
    assert_eq!(
        std::mem::size_of::<U>(),
        std::mem::size_of::<T>(),
        "cast_slice: size mismatch between U and T"
    );
    assert_eq!(
        std::mem::align_of::<U>(),
        std::mem::align_of::<T>(),
        "cast_slice: alignment mismatch between U and T"
    );
    if offset.checked_add(len)? > data.len() {
        return None;
    }
    Some(unsafe { &*(&data[offset..offset + len] as *const [U] as *const [T]) })
}

//
// From<Vec64<T>> for Array - enables ergonomic array construction from Vec64
//

impl From<Vec64<f64>> for Array {
    fn from(vec: Vec64<f64>) -> Self {
        Array::from_float64(FloatArray::from_vec64(vec, None).into())
    }
}

impl From<Vec64<f32>> for Array {
    fn from(vec: Vec64<f32>) -> Self {
        Array::from_float32(FloatArray::from_vec64(vec, None).into())
    }
}

impl From<Vec64<i64>> for Array {
    fn from(vec: Vec64<i64>) -> Self {
        Array::from_int64(IntegerArray::from_vec64(vec, None).into())
    }
}

impl From<Vec64<i32>> for Array {
    fn from(vec: Vec64<i32>) -> Self {
        Array::from_int32(IntegerArray::from_vec64(vec, None).into())
    }
}

impl From<Vec64<u64>> for Array {
    fn from(vec: Vec64<u64>) -> Self {
        Array::from_uint64(IntegerArray::from_vec64(vec, None).into())
    }
}

impl From<Vec64<u32>> for Array {
    fn from(vec: Vec64<u32>) -> Self {
        Array::from_uint32(IntegerArray::from_vec64(vec, None).into())
    }
}

#[cfg(feature = "extended_numeric_types")]
impl From<Vec64<i16>> for Array {
    fn from(vec: Vec64<i16>) -> Self {
        Array::from_int16(IntegerArray::from_vec64(vec, None).into())
    }
}

#[cfg(feature = "extended_numeric_types")]
impl From<Vec64<i8>> for Array {
    fn from(vec: Vec64<i8>) -> Self {
        Array::from_int8(IntegerArray::from_vec64(vec, None).into())
    }
}

#[cfg(feature = "extended_numeric_types")]
impl From<Vec64<u16>> for Array {
    fn from(vec: Vec64<u16>) -> Self {
        Array::from_uint16(IntegerArray::from_vec64(vec, None).into())
    }
}

#[cfg(feature = "extended_numeric_types")]
impl From<Vec64<u8>> for Array {
    fn from(vec: Vec64<u8>) -> Self {
        Array::from_uint8(IntegerArray::from_vec64(vec, None).into())
    }
}

impl Display for Array {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Array::Null => writeln!(f, "Array<Null>\n[null]"),
            Array::BooleanArray(arr) => {
                writeln!(f, "Array<Boolean>")?;
                Display::fmt(arr, f)
            }
            Array::NumericArray(arr) => {
                writeln!(f, "Array<Numeric>")?;
                Display::fmt(arr, f)
            }
            #[cfg(feature = "datetime")]
            Array::TemporalArray(arr) => {
                writeln!(f, "Array<Temporal>")?;
                Display::fmt(arr, f)
            }
            Array::TextArray(arr) => {
                writeln!(f, "Array<Text>")?;
                Display::fmt(arr, f)
            }
        }
    }
}

#[inline(always)]
fn clear_bit_in_place(bits: &mut Vec64<u8>, i: usize) {
    let byte = i >> 3;
    bits[byte] &= !(1u8 << (i & 7));
}

// String option extraction for Vec64 - produces owned strings
pub fn extract_string_option_values64_owned<T: AsRef<str>>(
    options: Vec64<Option<T>>,
) -> (Vec64<String>, Option<Bitmask>) {
    let len = options.len();
    let mut values = Vec64::with_capacity(len);

    // Start with all bits valid; clear for nulls
    let mut null_bytes = Vec64::with_capacity((len + 7) / 8);
    null_bytes.resize((len + 7) / 8, 0xFFu8);

    let mut has_nulls = false;

    for (i, opt) in options.into_iter().enumerate() {
        match opt {
            Some(s) => values.push(s.as_ref().to_string()),
            None => {
                values.push(String::new());
                clear_bit_in_place(&mut null_bytes, i);
                has_nulls = true;
            }
        }
    }

    let mask = if has_nulls {
        Some(Bitmask::from_bytes(null_bytes, len))
    } else {
        None
    };
    (values, mask)
}

// ===== categoricals (borrowed &str) =====
pub fn extract_categorical_option_values64(
    options: Vec64<Option<&str>>,
) -> (Vec64<&str>, Option<Bitmask>) {
    let len = options.len();
    let mut values: Vec64<&str> = Vec64::with_capacity(len);

    // Start with all bits valid; clear for nulls
    let mut null_bytes = Vec64::with_capacity((len + 7) / 8);
    null_bytes.resize((len + 7) / 8, 0xFFu8);

    let mut has_nulls = false;

    for (i, opt) in options.into_iter().enumerate() {
        match opt {
            Some(s) => values.push(s),
            None => {
                values.push(""); // sentinel; masked out
                clear_bit_in_place(&mut null_bytes, i);
                has_nulls = true;
            }
        }
    }

    let mask = if has_nulls {
        Some(Bitmask::from_bytes(null_bytes, len))
    } else {
        None
    };
    (values, mask)
}
// Generic numeric option extraction for Vec64
pub fn extract_option_values64<T: Default + Copy>(
    options: Vec64<Option<T>>,
) -> (Vec64<T>, Option<Bitmask>) {
    let len = options.len();
    let mut values = Vec64::with_capacity(len);

    // Start with all bits valid; clear for nulls
    let mut null_bytes = Vec64::with_capacity((len + 7) / 8);
    null_bytes.resize((len + 7) / 8, 0xFFu8);

    let mut has_nulls = false;

    for (i, opt) in options.into_iter().enumerate() {
        match opt {
            Some(v) => values.push(v),
            None => {
                values.push(T::default());
                clear_bit_in_place(&mut null_bytes, i);
                has_nulls = true;
            }
        }
    }

    let mask = if has_nulls {
        Some(Bitmask::from_bytes(null_bytes, len))
    } else {
        None
    };
    (values, mask)
}

#[allow(dead_code)]
pub fn extract_categorical_option_values64_owned(
    options: Vec64<Option<&str>>,
) -> (Vec64<&str>, Option<Bitmask>) {
    extract_categorical_option_values64(options)
}

// Helper macro to detect if any element is None
#[macro_export]
macro_rules! has_nulls {
    () => { false };
    (None $(, $rest:expr)*) => { true };
    (None) => { true };
    ($first:expr $(, $rest:expr)*) => { has_nulls!($($rest),*) };
    ($first:expr) => { false };
}

// ======== numeric ========

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! arr_i8 {
    // Literal array ref `&[a, b, c]` + Bitmask
    (&[ $($x:expr),+ $(,)? ] ; $mask:expr) => {{
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_int8($crate::IntegerArray::<i8>::from_vec64(temp_vec, Some($mask)))
    }};
    // Literal array ref `&[a, b, c]`
    (&[ $($x:expr),+ $(,)? ]) => {{
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_int8($crate::IntegerArray::<i8>::from_vec64(temp_vec, None))
    }};
    // Values + Bitmask, semicolon-separated
    ($v:expr; $mask:expr) => {
        $crate::Array::from_int8($crate::IntegerArray::<i8>::from_vec64($v.into(), Some($mask)))
    };
    // Handle Vec64 or `&[i8]` input via `Into<Vec64<i8>>`
    ($v:expr) => {
        $crate::Array::from_int8($crate::IntegerArray::<i8>::from_vec64($v.into(), None))
    };
    // Literal elements + Bitmask
    ($($x:expr),+ ; $mask:expr) => {{
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_int8($crate::IntegerArray::<i8>::from_vec64(temp_vec, Some($mask)))
    }};
    // Handle literal arrays
    ($($x:expr),+ $(,)?) => {{
        // Check if any element is None by trying to match patterns
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_int8($crate::IntegerArray::<i8>::from_vec64(temp_vec, None))
    }};
    // Handle empty arrays
    () => {
        $crate::Array::from_int8($crate::IntegerArray::<i8>::from_vec64(vec64![], None))
    };
}

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! arr_i16 {
    (&[ $($x:expr),+ $(,)? ] ; $mask:expr) => {{
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_int16($crate::IntegerArray::<i16>::from_vec64(temp_vec, Some($mask)))
    }};
    (&[ $($x:expr),+ $(,)? ]) => {{
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_int16($crate::IntegerArray::<i16>::from_vec64(temp_vec, None))
    }};
    ($v:expr; $mask:expr) => {
        $crate::Array::from_int16($crate::IntegerArray::<i16>::from_vec64($v.into(), Some($mask)))
    };
    ($v:expr) => {
        $crate::Array::from_int16($crate::IntegerArray::<i16>::from_vec64($v.into(), None))
    };
    ($($x:expr),+ ; $mask:expr) => {{
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_int16($crate::IntegerArray::<i16>::from_vec64(temp_vec, Some($mask)))
    }};
    ($($x:expr),+ $(,)?) => {{

        let temp_vec = vec64![$($x),+];
        $crate::Array::from_int16($crate::IntegerArray::<i16>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_int16($crate::IntegerArray::<i16>::from_vec64(vec64![], None))
    };
}

#[macro_export]
macro_rules! arr_i32 {
    (&[ $($x:expr),+ $(,)? ] ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_int32($crate::IntegerArray::<i32>::from_vec64(temp_vec, Some($mask)))
    }};
    (&[ $($x:expr),+ $(,)? ]) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_int32($crate::IntegerArray::<i32>::from_vec64(temp_vec, None))
    }};
    ($v:expr; $mask:expr) => {
        $crate::Array::from_int32($crate::IntegerArray::<i32>::from_vec64($v.into(), Some($mask)))
    };
    ($v:expr) => {
        $crate::Array::from_int32($crate::IntegerArray::<i32>::from_vec64($v.into(), None))
    };
    ($($x:expr),+ ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_int32($crate::IntegerArray::<i32>::from_vec64(temp_vec, Some($mask)))
    }};
    ($($x:expr),+ $(,)?) => {{

        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_int32($crate::IntegerArray::<i32>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_int32($crate::IntegerArray::<i32>::from_vec64(vec64![], None))
    };
}

#[macro_export]
macro_rules! arr_i64 {
    (&[ $($x:expr),+ $(,)? ] ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_int64($crate::IntegerArray::<i64>::from_vec64(temp_vec, Some($mask)))
    }};
    (&[ $($x:expr),+ $(,)? ]) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_int64($crate::IntegerArray::<i64>::from_vec64(temp_vec, None))
    }};
    ($v:expr; $mask:expr) => {
        $crate::Array::from_int64($crate::IntegerArray::<i64>::from_vec64($v.into(), Some($mask)))
    };
    ($v:expr) => {
        $crate::Array::from_int64($crate::IntegerArray::<i64>::from_vec64($v.into(), None))
    };
    ($($x:expr),+ ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_int64($crate::IntegerArray::<i64>::from_vec64(temp_vec, Some($mask)))
    }};
    ($($x:expr),+ $(,)?) => {{

        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_int64($crate::IntegerArray::<i64>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_int64($crate::IntegerArray::<i64>::from_vec64(vec64![], None))
    };
}

// ======== Datetime (i32) ========

/// Build an `Array` of i32-backed datetimes. The time unit is
/// required and precedes the values, separated by `;`.
///
/// ```ignore
/// use minarrow::ffi::arrow_dtype::TimeUnit;
/// let a = arr_dt32![TimeUnit::Seconds; 1_768_521_600, 1_775_865_600];
/// let b = arr_dt32![TimeUnit::Milliseconds; vec64![1, 2, 3]];
/// ```
#[cfg(feature = "datetime")]
#[macro_export]
macro_rules! arr_dt32 {
    ($unit:expr; $v:expr) => {
        $crate::Array::from_datetime_i32($crate::DatetimeArray::<i32>::from_vec64(
            $v, None, Some($unit),
        ))
    };
    ($unit:expr; $($x:expr),+ $(,)?) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_datetime_i32($crate::DatetimeArray::<i32>::from_vec64(
            temp_vec, None, Some($unit),
        ))
    }};
    ($unit:expr;) => {
        $crate::Array::from_datetime_i32($crate::DatetimeArray::<i32>::from_vec64(
            $crate::Vec64::new(),
            None,
            Some($unit),
        ))
    };
}

// ======== Datetime (i64) ========

/// Build an `Array` of i64-backed datetimes. The time unit is
/// required and precedes the values, separated by `;`.
///
/// ```ignore
/// use minarrow::ffi::arrow_dtype::TimeUnit;
/// let a = arr_dt64![TimeUnit::Seconds; 1_768_521_600, 1_775_865_600];
/// let b = arr_dt64![TimeUnit::Milliseconds; vec64![1, 2, 3]];
/// ```
#[cfg(feature = "datetime")]
#[macro_export]
macro_rules! arr_dt64 {
    ($unit:expr; $v:expr) => {
        $crate::Array::from_datetime_i64($crate::DatetimeArray::<i64>::from_vec64(
            $v, None, Some($unit),
        ))
    };
    ($unit:expr; $($x:expr),+ $(,)?) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_datetime_i64($crate::DatetimeArray::<i64>::from_vec64(
            temp_vec, None, Some($unit),
        ))
    }};
    ($unit:expr;) => {
        $crate::Array::from_datetime_i64($crate::DatetimeArray::<i64>::from_vec64(
            $crate::Vec64::new(),
            None,
            Some($unit),
        ))
    };
}

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! arr_u8 {
    (&[ $($x:expr),+ $(,)? ] ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_uint8($crate::IntegerArray::<u8>::from_vec64(temp_vec, Some($mask)))
    }};
    (&[ $($x:expr),+ $(,)? ]) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_uint8($crate::IntegerArray::<u8>::from_vec64(temp_vec, None))
    }};
    ($v:expr; $mask:expr) => {
        $crate::Array::from_uint8($crate::IntegerArray::<u8>::from_vec64($v.into(), Some($mask)))
    };
    ($v:expr) => {
        $crate::Array::from_uint8($crate::IntegerArray::<u8>::from_vec64($v.into(), None))
    };
    ($($x:expr),+ ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_uint8($crate::IntegerArray::<u8>::from_vec64(temp_vec, Some($mask)))
    }};
    ($($x:expr),+ $(,)?) => {{

        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_uint8($crate::IntegerArray::<u8>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_uint8($crate::IntegerArray::<u8>::from_vec64(vec64![], None))
    };
}

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! arr_u16 {
    (&[ $($x:expr),+ $(,)? ] ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_uint16($crate::IntegerArray::<u16>::from_vec64(temp_vec, Some($mask)))
    }};
    (&[ $($x:expr),+ $(,)? ]) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_uint16($crate::IntegerArray::<u16>::from_vec64(temp_vec, None))
    }};
    ($v:expr; $mask:expr) => {
        $crate::Array::from_uint16($crate::IntegerArray::<u16>::from_vec64($v.into(), Some($mask)))
    };
    ($v:expr) => {
        $crate::Array::from_uint16($crate::IntegerArray::<u16>::from_vec64($v.into(), None))
    };
    ($($x:expr),+ ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_uint16($crate::IntegerArray::<u16>::from_vec64(temp_vec, Some($mask)))
    }};
    ($($x:expr),+ $(,)?) => {{

        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_uint16($crate::IntegerArray::<u16>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_uint16($crate::IntegerArray::<u16>::from_vec64(vec64![], None))
    };
}

#[macro_export]
macro_rules! arr_u32 {
    (&[ $($x:expr),+ $(,)? ] ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_uint32($crate::IntegerArray::<u32>::from_vec64(temp_vec, Some($mask)))
    }};
    (&[ $($x:expr),+ $(,)? ]) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_uint32($crate::IntegerArray::<u32>::from_vec64(temp_vec, None))
    }};
    ($v:expr; $mask:expr) => {
        $crate::Array::from_uint32($crate::IntegerArray::<u32>::from_vec64($v.into(), Some($mask)))
    };
    ($v:expr) => {
        $crate::Array::from_uint32($crate::IntegerArray::<u32>::from_vec64($v.into(), None))
    };
    ($($x:expr),+ ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_uint32($crate::IntegerArray::<u32>::from_vec64(temp_vec, Some($mask)))
    }};
    ($($x:expr),+ $(,)?) => {{

        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_uint32($crate::IntegerArray::<u32>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_uint32($crate::IntegerArray::<u32>::from_vec64(vec64![], None))
    };
}

#[macro_export]
macro_rules! arr_u64 {
    (&[ $($x:expr),+ $(,)? ] ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_uint64($crate::IntegerArray::<u64>::from_vec64(temp_vec, Some($mask)))
    }};
    (&[ $($x:expr),+ $(,)? ]) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_uint64($crate::IntegerArray::<u64>::from_vec64(temp_vec, None))
    }};
    ($v:expr; $mask:expr) => {
        $crate::Array::from_uint64($crate::IntegerArray::<u64>::from_vec64($v.into(), Some($mask)))
    };
    ($v:expr) => {
        $crate::Array::from_uint64($crate::IntegerArray::<u64>::from_vec64($v.into(), None))
    };
    ($($x:expr),+ ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_uint64($crate::IntegerArray::<u64>::from_vec64(temp_vec, Some($mask)))
    }};
    ($($x:expr),+ $(,)?) => {{

        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_uint64($crate::IntegerArray::<u64>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_uint64($crate::IntegerArray::<u64>::from_vec64(vec64![], None))
    };
}

// ======== Float types ========

#[macro_export]
macro_rules! arr_f32 {
    (&[ $($x:expr),+ $(,)? ] ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_float32($crate::FloatArray::<f32>::from_vec64(temp_vec, Some($mask)))
    }};
    (&[ $($x:expr),+ $(,)? ]) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_float32($crate::FloatArray::<f32>::from_vec64(temp_vec, None))
    }};
    ($v:expr; $mask:expr) => {
        $crate::Array::from_float32($crate::FloatArray::<f32>::from_vec64($v.into(), Some($mask)))
    };
    ($v:expr) => {
        $crate::Array::from_float32($crate::FloatArray::<f32>::from_vec64($v.into(), None))
    };
    ($($x:expr),+ ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_float32($crate::FloatArray::<f32>::from_vec64(temp_vec, Some($mask)))
    }};
    ($($x:expr),+ $(,)?) => {{

        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_float32($crate::FloatArray::<f32>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_float32($crate::FloatArray::<f32>::from_vec64(vec64![], None))
    };
}

#[macro_export]
macro_rules! arr_f64 {
    (&[ $($x:expr),+ $(,)? ] ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_float64($crate::FloatArray::<f64>::from_vec64(temp_vec, Some($mask)))
    }};
    (&[ $($x:expr),+ $(,)? ]) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_float64($crate::FloatArray::<f64>::from_vec64(temp_vec, None))
    }};
    ($v:expr; $mask:expr) => {
        $crate::Array::from_float64($crate::FloatArray::<f64>::from_vec64($v.into(), Some($mask)))
    };
    ($v:expr) => {
            $crate::Array::from_float64($crate::FloatArray::<f64>::from_vec64($v.into(), None))
        };
    ($($x:expr),+ ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_float64($crate::FloatArray::<f64>::from_vec64(temp_vec, Some($mask)))
    }};
    ($($x:expr),+ $(,)?) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_float64($crate::FloatArray::<f64>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_float64($crate::FloatArray::<f64>::from_vec64(vec64![], None))
    };
}

// ======== Boolean ========

#[macro_export]
macro_rules! arr_bool {
    (&[ $($x:expr),+ $(,)? ] ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_bool($crate::BooleanArray::from_vec64(temp_vec, Some($mask)))
    }};
    (&[ $($x:expr),+ $(,)? ]) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_bool($crate::BooleanArray::from_vec64(temp_vec, None))
    }};
    ($v:expr; $mask:expr) => {
        $crate::Array::from_bool($crate::BooleanArray::from_vec64($v.into(), Some($mask)))
    };
    ($v:expr) => {
        $crate::Array::from_bool($crate::BooleanArray::from_vec64($v.into(), None))
    };
    ($($x:expr),+ ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_bool($crate::BooleanArray::from_vec64(temp_vec, Some($mask)))
    }};
    ($($x:expr),+ $(,)?) => {{

        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_bool($crate::BooleanArray::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_bool($crate::BooleanArray::from_vec64(vec64![], None))
    };
}

// ======== String ========

#[macro_export]
macro_rules! arr_str32 {
    (&[ $($x:expr),+ $(,)? ] ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_string32($crate::StringArray::<u32>::from_vec64(temp_vec, Some($mask)))
    }};
    (&[ $($x:expr),+ $(,)? ]) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_string32($crate::StringArray::<u32>::from_vec64(temp_vec, None))
    }};
    ($v:expr; $mask:expr) => {
        $crate::Array::from_string32($crate::StringArray::<u32>::from_vec64($v.into(), Some($mask)))
    };
    ($v:expr) => {
        $crate::Array::from_string32($crate::StringArray::<u32>::from_vec64($v.into(), None))
    };
    ($($x:expr),+ ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_string32($crate::StringArray::<u32>::from_vec64(temp_vec, Some($mask)))
    }};
    ($($x:expr),+ $(,)?) => {{

        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_string32($crate::StringArray::<u32>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_string32($crate::StringArray::<u32>::from_vec64(vec64![], None))
    };
}

#[cfg(feature = "large_string")]
#[macro_export]
macro_rules! arr_str64 {
    (&[ $($x:expr),+ $(,)? ] ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_string64($crate::StringArray::<u64>::from_vec64(temp_vec, Some($mask)))
    }};
    (&[ $($x:expr),+ $(,)? ]) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_string64($crate::StringArray::<u64>::from_vec64(temp_vec, None))
    }};
    ($v:expr; $mask:expr) => {
        $crate::Array::from_string64($crate::StringArray::<u64>::from_vec64($v.into(), Some($mask)))
    };
    ($v:expr) => {
        $crate::Array::from_string64($crate::StringArray::<u64>::from_vec64($v.into(), None))
    };
    ($($x:expr),+ ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_string64($crate::StringArray::<u64>::from_vec64(temp_vec, Some($mask)))
    }};
    ($($x:expr),+ $(,)?) => {{

        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_string64($crate::StringArray::<u64>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_string64($crate::StringArray::<u64>::from_vec64(vec64![], None))
    };
}

// ======== Categorical ========

#[cfg(feature = "default_categorical_8")]
#[macro_export]
macro_rules! arr_cat8 {
    (&[ $($x:expr),+ $(,)? ] ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_categorical8($crate::CategoricalArray::<u8>::from_vec64(temp_vec, Some($mask)))
    }};
    (&[ $($x:expr),+ $(,)? ]) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_categorical8($crate::CategoricalArray::<u8>::from_vec64(temp_vec, None))
    }};
    ($v:expr; $mask:expr) => {
        $crate::Array::from_categorical8($crate::CategoricalArray::<u8>::from_vec64($v.into(), Some($mask)))
    };
    ($v:expr) => {
        $crate::Array::from_categorical8($crate::CategoricalArray::<u8>::from_vec64($v.into(), None))
    };
    ($($x:expr),+ ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_categorical8($crate::CategoricalArray::<u8>::from_vec64(temp_vec, Some($mask)))
    }};
    ($($x:expr),+ $(,)?) => {{

        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_categorical8($crate::CategoricalArray::<u8>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_categorical8($crate::CategoricalArray::<u8>::from_vec64(vec64![], None))
    };
}

#[cfg(feature = "extended_categorical")]
#[macro_export]
macro_rules! arr_cat16 {
    (&[ $($x:expr),+ $(,)? ] ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_categorical16($crate::CategoricalArray::<u16>::from_vec64(temp_vec, Some($mask)))
    }};
    (&[ $($x:expr),+ $(,)? ]) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_categorical16($crate::CategoricalArray::<u16>::from_vec64(temp_vec, None))
    }};
    ($v:expr; $mask:expr) => {
        $crate::Array::from_categorical16($crate::CategoricalArray::<u16>::from_vec64($v.into(), Some($mask)))
    };
    ($v:expr) => {
        $crate::Array::from_categorical16($crate::CategoricalArray::<u16>::from_vec64($v.into(), None))
    };
    ($($x:expr),+ ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_categorical16($crate::CategoricalArray::<u16>::from_vec64(temp_vec, Some($mask)))
    }};
    ($($x:expr),+ $(,)?) => {{

        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_categorical16($crate::CategoricalArray::<u16>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_categorical16($crate::CategoricalArray::<u16>::from_vec64(vec64![], None))
    };
}

#[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
#[macro_export]
macro_rules! arr_cat32 {
    (&[ $($x:expr),+ $(,)? ] ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_categorical32($crate::CategoricalArray::<u32>::from_vec64(temp_vec, Some($mask)))
    }};
    (&[ $($x:expr),+ $(,)? ]) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_categorical32($crate::CategoricalArray::<u32>::from_vec64(temp_vec, None))
    }};
    ($v:expr; $mask:expr) => {
        $crate::Array::from_categorical32($crate::CategoricalArray::<u32>::from_vec64($v.into(), Some($mask)))
    };
    ($v:expr) => {
        $crate::Array::from_categorical32($crate::CategoricalArray::<u32>::from_vec64($v.into(), None))
    };
    ($($x:expr),+ ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_categorical32($crate::CategoricalArray::<u32>::from_vec64(temp_vec, Some($mask)))
    }};
    ($($x:expr),+ $(,)?) => {{

        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_categorical32($crate::CategoricalArray::<u32>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_categorical32($crate::CategoricalArray::<u32>::from_vec64(vec64![], None))
    };
}

#[cfg(feature = "extended_categorical")]
#[macro_export]
macro_rules! arr_cat64 {
    (&[ $($x:expr),+ $(,)? ] ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_categorical64($crate::CategoricalArray::<u64>::from_vec64(temp_vec, Some($mask)))
    }};
    (&[ $($x:expr),+ $(,)? ]) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_categorical64($crate::CategoricalArray::<u64>::from_vec64(temp_vec, None))
    }};
    ($v:expr; $mask:expr) => {
        $crate::Array::from_categorical64($crate::CategoricalArray::<u64>::from_vec64($v.into(), Some($mask)))
    };
    ($v:expr) => {
        $crate::Array::from_categorical64($crate::CategoricalArray::<u64>::from_vec64($v.into(), None))
    };
    ($($x:expr),+ ; $mask:expr) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_categorical64($crate::CategoricalArray::<u64>::from_vec64(temp_vec, Some($mask)))
    }};
    ($($x:expr),+ $(,)?) => {{

        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_categorical64($crate::CategoricalArray::<u64>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_categorical64($crate::CategoricalArray::<u64>::from_vec64(vec64![], None))
    };
}

// ======== Integer (signed) ========

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! arr_i8_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_option_values64($v);
        $crate::Array::from_int8($crate::IntegerArray::<i8>::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_int8($crate::IntegerArray::<i8>::from_vec64(vals, mask))
    }};
    () => {{
        use $crate::vec64;
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_int8($crate::IntegerArray::<i8>::from_vec64(vals, mask))
    }};
}

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! arr_i16_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_option_values64($v);
        $crate::Array::from_int16($crate::IntegerArray::<i16>::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_int16($crate::IntegerArray::<i16>::from_vec64(vals, mask))
    }};
    () => {{
        use $crate::vec64;
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_int16($crate::IntegerArray::<i16>::from_vec64(vals, mask))
    }};
}

#[macro_export]
macro_rules! arr_i32_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_option_values64($v);
        $crate::Array::from_int32($crate::IntegerArray::<i32>::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_int32($crate::IntegerArray::<i32>::from_vec64(vals, mask))
    }};
    () => {{
        use $crate::vec64;
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_int32($crate::IntegerArray::<i32>::from_vec64(vals, mask))
    }};
}

#[macro_export]
macro_rules! arr_i64_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_option_values64($v);
        $crate::Array::from_int64($crate::IntegerArray::<i64>::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_int64($crate::IntegerArray::<i64>::from_vec64(vals, mask))
    }};
    () => {{
        use $crate::vec64;
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_int64($crate::IntegerArray::<i64>::from_vec64(vals, mask))
    }};
}

// ======== Integer (unsigned) ========

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! arr_u8_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_option_values64($v);
        $crate::Array::from_uint8($crate::IntegerArray::<u8>::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_uint8($crate::IntegerArray::<u8>::from_vec64(vals, mask))
    }};
    () => {{
        use $crate::vec64;
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_uint8($crate::IntegerArray::<u8>::from_vec64(vals, mask))
    }};
}

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! arr_u16_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_option_values64($v);
        $crate::Array::from_uint16($crate::IntegerArray::<u16>::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_uint16($crate::IntegerArray::<u16>::from_vec64(vals, mask))
    }};
    () => {{
        use $crate::vec64;
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_uint16($crate::IntegerArray::<u16>::from_vec64(vals, mask))
    }};
}

#[macro_export]
macro_rules! arr_u32_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_option_values64($v);
        $crate::Array::from_uint32($crate::IntegerArray::<u32>::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_uint32($crate::IntegerArray::<u32>::from_vec64(vals, mask))
    }};
    () => {{
        use $crate::vec64;
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_uint32($crate::IntegerArray::<u32>::from_vec64(vals, mask))
    }};
}

#[macro_export]
macro_rules! arr_u64_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_option_values64($v);
        $crate::Array::from_uint64($crate::IntegerArray::<u64>::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_uint64($crate::IntegerArray::<u64>::from_vec64(vals, mask))
    }};
    () => {{
        use $crate::vec64;
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_uint64($crate::IntegerArray::<u64>::from_vec64(vals, mask))
    }};
}

// ======== Float ========

#[macro_export]
macro_rules! arr_f32_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_option_values64($v);
        $crate::Array::from_float32($crate::FloatArray::<f32>::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_float32($crate::FloatArray::<f32>::from_vec64(vals, mask))
    }};
    () => {{
        use $crate::vec64;
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_float32($crate::FloatArray::<f32>::from_vec64(vals, mask))
    }};
}

#[macro_export]
macro_rules! arr_f64_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_option_values64($v);
        $crate::Array::from_float64($crate::FloatArray::<f64>::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_float64($crate::FloatArray::<f64>::from_vec64(vals, mask))
    }};
    () => {{
        use $crate::vec64;
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_float64($crate::FloatArray::<f64>::from_vec64(vals, mask))
    }};
}

// ======== Boolean ========

#[macro_export]
macro_rules! arr_bool_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_option_values64($v);
        $crate::Array::from_bool($crate::BooleanArray::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_bool($crate::BooleanArray::from_vec64(vals, mask))
    }};
    () => {{
        use $crate::vec64;
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_bool($crate::BooleanArray::from_vec64(vals, mask))
    }};
}

// ======== String ========

#[macro_export]
macro_rules! arr_str32_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_string_option_values64_owned($v);
        $crate::Array::from_string32($crate::StringArray::<u32>::from_vec64_owned(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_string_option_values64_owned(temp_vec);
        $crate::Array::from_string32($crate::StringArray::<u32>::from_vec64_owned(vals, mask))
    }};
    () => {{
        use $crate::vec64;
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_string_option_values64_owned(temp_vec);
        $crate::Array::from_string32($crate::StringArray::<u32>::from_vec64_owned(vals, mask))
    }};
}

#[cfg(feature = "large_string")]
#[macro_export]
macro_rules! arr_str64_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_string_option_values64_owned($v);
        $crate::Array::from_string64($crate::StringArray::<u64>::from_vec64_owned(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_string_option_values64_owned(temp_vec);
        $crate::Array::from_string64($crate::StringArray::<u64>::from_vec64_owned(vals, mask))
    }};
    () => {{
        use $crate::vec64;
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_string_option_values64_owned(temp_vec);
        $crate::Array::from_string64($crate::StringArray::<u64>::from_vec64_owned(vals, mask))
    }};
}

// ======== Categorical ========

#[cfg(feature = "default_categorical_8")]
#[macro_export]
macro_rules! arr_cat8_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_categorical_option_values64($v);
        $crate::Array::from_categorical8($crate::CategoricalArray::<u8>::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_categorical_option_values64(temp_vec);
        $crate::Array::from_categorical8($crate::CategoricalArray::<u8>::from_vec64(vals, mask))
    }};
    () => {{
        use $crate::vec64;
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_categorical_option_values64(temp_vec);
        $crate::Array::from_categorical8($crate::CategoricalArray::<u8>::from_vec64(vals, mask))
    }};
}

#[cfg(feature = "extended_categorical")]
#[macro_export]
macro_rules! arr_cat16_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_categorical_option_values64($v);
        $crate::Array::from_categorical16($crate::CategoricalArray::<u16>::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_categorical_option_values64(temp_vec);
        $crate::Array::from_categorical16($crate::CategoricalArray::<u16>::from_vec64(vals, mask))
    }};
    () => {{
        use $crate::vec64;
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_categorical_option_values64(temp_vec);
        $crate::Array::from_categorical16($crate::CategoricalArray::<u16>::from_vec64(vals, mask))
    }};
}

#[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
#[macro_export]
macro_rules! arr_cat32_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_categorical_option_values64($v);
        $crate::Array::from_categorical32($crate::CategoricalArray::<u32>::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_categorical_option_values64(temp_vec);
        $crate::Array::from_categorical32($crate::CategoricalArray::<u32>::from_vec64(vals, mask))
    }};
    () => {{
        use $crate::vec64;
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_categorical_option_values64(temp_vec);
        $crate::Array::from_categorical32($crate::CategoricalArray::<u32>::from_vec64(vals, mask))
    }};
}

#[cfg(feature = "extended_categorical")]
#[macro_export]
macro_rules! arr_cat64_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_categorical_option_values64($v);
        $crate::Array::from_categorical64($crate::CategoricalArray::<u64>::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        use $crate::vec64;
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_categorical_option_values64(temp_vec);
        $crate::Array::from_categorical64($crate::CategoricalArray::<u64>::from_vec64(vals, mask))
    }};
    () => {{
        use $crate::vec64;
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_categorical_option_values64(temp_vec);
        $crate::Array::from_categorical64($crate::CategoricalArray::<u64>::from_vec64(vals, mask))
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structs::variants::boolean::BooleanArray;
    use crate::structs::variants::categorical::CategoricalArray;
    use crate::structs::variants::float::FloatArray;
    use crate::structs::variants::integer::IntegerArray;
    use crate::structs::variants::string::StringArray;
    use crate::traits::masked_array::MaskedArray;

    #[test]
    fn test_array_len_and_null() {
        assert_eq!(Array::Null.len(), 0);

        let arr = Array::from_int32(IntegerArray::<i32>::default());
        assert_eq!(arr.len(), 0);

        let mut arr = Array::from_int32(IntegerArray::<i32>::default());
        if let Array::NumericArray(NumericArray::Int32(ref mut a)) = arr {
            let a_mut = Arc::get_mut(a).expect("Array not uniquely owned");
            a_mut.push(7);
            a_mut.push(42);
        }
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn test_array_arrow_type() {
        assert_eq!(Array::Null.arrow_type(), ArrowType::Null);
        assert_eq!(
            Array::from_float32(FloatArray::<f32>::default()).arrow_type(),
            ArrowType::Float32
        );
        assert_eq!(
            Array::from_string32(StringArray::default()).arrow_type(),
            ArrowType::String
        );
        #[cfg(feature = "large_string")]
        assert_eq!(
            Array::from_string64(StringArray::default()).arrow_type(),
            ArrowType::LargeString
        );
        assert_eq!(
            Array::from_bool(BooleanArray::default()).arrow_type(),
            ArrowType::Boolean
        );

        #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
        {
            let dict32 = Array::from_categorical32(CategoricalArray::<u32>::default());
            assert_eq!(
                dict32.arrow_type(),
                ArrowType::Dictionary(CategoricalIndexType::UInt32)
            );
        }
    }

    #[test]
    fn test_array_is_nullable() {
        assert!(Array::Null.is_nullable());

        let arr = Array::from_int64(IntegerArray::<i64>::default());
        assert!(!arr.is_nullable());

        let mut arr = Array::from_int64(IntegerArray::<i64>::default());
        if let Array::NumericArray(NumericArray::Int64(ref mut a)) = arr {
            // only succeeds if the Arc is uniquely owned
            let a_mut = Arc::get_mut(a).expect("Array not uniquely owned");
            a_mut.push(100);
            a_mut.push(200);
            a_mut.push_null();
        }
        assert!(arr.is_nullable());

        let arr = Array::from_string32(StringArray::default());
        assert!(!arr.is_nullable());
    }

    #[test]
    fn test_data_ptr_and_len_for_primitives() {
        let mut arr = Array::from_int32(IntegerArray::<i32>::default());
        if let Array::NumericArray(NumericArray::Int32(ref mut a)) = arr {
            // Must have unique ownership to mutate the Arc contents.
            let a_mut = Arc::get_mut(a).expect("Array not uniquely owned");
            a_mut.push(123);
            a_mut.push(456);
        }
        let (ptr, len, sz) = arr.data_ptr_and_byte_len();
        assert!(!ptr.is_null());
        assert_eq!(len, 2);
        assert_eq!(sz, std::mem::size_of::<i32>());
    }

    #[test]
    fn test_data_ptr_and_len_for_str() {
        let mut str = StringArray::default();
        str.push_str("hello");
        str.push_str("world");
        let arr = Array::from_string32(str);
        let (ptr, len, sz) = arr.data_ptr_and_byte_len();
        assert!(!ptr.is_null());
        // byte length, not string count
        assert_eq!(len, 10);
        assert_eq!(sz, 1);
    }

    #[test]
    fn test_data_ptr_and_len_for_bool() {
        let mut bools = BooleanArray::default();
        for _ in 0..10 {
            bools.push(true);
        }
        let arr = Array::from_bool(bools);
        let (ptr, len, sz) = arr.data_ptr_and_byte_len();
        assert!(!ptr.is_null());
        assert_eq!(len, 10); // number of bits = logical elements
        assert_eq!(sz, 1);
    }

    #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
    #[test]
    fn test_data_ptr_and_len_for_dictionary() {
        let mut dict = CategoricalArray::<u32>::default();
        dict.push_str("a");
        dict.push_str("b");
        dict.push_str("a");
        let arr = Array::from_categorical32(dict);
        let (ptr, len, sz) = arr.data_ptr_and_byte_len();
        assert!(!ptr.is_null());
        assert_eq!(len, 3);
        assert_eq!(sz, std::mem::size_of::<u32>());
    }

    #[test]
    fn test_null_mask_ptr_and_len() {
        // Null variant has no mask
        assert!(Array::Null.null_mask_ptr_and_byte_len().is_none());

        let mut arr = IntegerArray::<i32>::default();
        arr.push(5);
        arr.push_null();
        let arr = Array::from_int32(arr);
        let mask = arr.null_mask_ptr_and_byte_len();
        assert!(mask.is_some());
        let (ptr, len) = mask.unwrap();
        assert!(!ptr.is_null());
        assert!(len > 0);

        let arr = Array::from_float64(FloatArray::<f64>::default());
        assert!(arr.null_mask_ptr_and_byte_len().is_none());
    }

    #[test]
    fn test_offsets_ptr_and_len() {
        let arr = Array::from_int32(IntegerArray::<i32>::default());
        assert!(arr.offsets_ptr_and_len().is_none());

        let mut str = StringArray::default();
        str.push_str("a");
        str.push_str("bc");
        let arr = Array::from_string32(str);
        let opt = arr.offsets_ptr_and_len();
        assert!(opt.is_some());
        let (ptr, len) = opt.unwrap();
        assert!(!ptr.is_null());
        assert_eq!(len, 12); // 3 offsets (u32) == 12 bytes
    }

    #[test]
    fn test_enum_variant_consistency() {
        let i32arr = IntegerArray::<i32>::default();
        let a = Array::from_int32(i32arr.clone());
        if let Array::NumericArray(NumericArray::Int32(ref arr2)) = a {
            assert_eq!(arr2.as_ref(), &i32arr);
        } else {
            panic!("Not the right variant");
        }
    }

    #[test]
    fn test_default_variant_is_null() {
        let a = Array::default();
        assert!(matches!(a, Array::Null));
        assert_eq!(a.len(), 0);
        assert!(a.is_nullable());
    }

    #[test]
    fn test_array_enum_slice() {
        use crate::{Array, ArrayVT};

        let mut bool_arr = BooleanArray::default();
        bool_arr.push(true);
        bool_arr.push(false);
        bool_arr.push(true);
        bool_arr.push(false);

        let array = Array::from_bool(bool_arr);
        let view: ArrayVT = (&array, 1, 2);

        match view.0 {
            Array::BooleanArray(inner) => {
                assert_eq!(inner.get(view.1), Some(false));
                assert_eq!(inner.get(view.1 + 1), Some(true));
            }
            _ => panic!("Expected Bool variant"),
        }
    }

    #[test]
    fn test_num_from_int_array() {
        let arr = IntegerArray::<i32>::from_slice(&[1, 2, 3]);
        let array = Array::from_int32(arr.clone());
        let out = array.num();
        match out {
            NumericArray::Int32(ref a) => assert_eq!(a.data, arr.data),
            _ => panic!("Expected Int32"),
        }
    }

    #[test]
    fn test_num_from_bool_array() {
        let mut arr = BooleanArray::default();
        arr.push(true);
        arr.push(false);
        arr.push_null();
        let array = Array::from_bool(arr.clone());
        let out = array.num();
        match out {
            NumericArray::Int32(ref a) => assert_eq!(&a.data[..], &[1, 0, 0]),
            _ => panic!("Expected Int32"),
        }
    }

    #[test]
    fn test_num_from_string_array() {
        let arr = StringArray::<u32>::from_slice(&["123", "xyz", ""]);
        let array = Array::from_string32(arr.clone());
        let out = array.num();
        match out {
            NumericArray::Int32(ref a) => {
                // "123" parses, "xyz" and "" are invalid, thus 0 and marked null.
                assert_eq!(&a.data[..], &[123, 0, 0]);
                let mask = a.null_mask.as_ref().expect("Should have a null mask");
                assert_eq!(mask.get(0), true);
                assert_eq!(mask.get(1), false);
                assert_eq!(mask.get(2), false);
            }
            _ => panic!("Expected Int32"),
        }
    }

    #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
    #[test]
    fn test_num_from_categorical_array() {
        let arr = StringArray::<u32>::from_slice(&["42", "hi", "999"]);
        let cat = arr.to_categorical_array();
        let array = Array::from_categorical32(cat.clone());
        let out = array.num();
        match out {
            NumericArray::Int32(ref a) => {
                // unique_values: ["42", "hi", "999"]
                // .data indices: [0, 0, 1, 2, ...]
                // Only "42" and "999" parse as i32
                let expected_vals: Vec<i32> = cat
                    .unique_values()
                    .iter()
                    .map(|s| s.parse::<i32>().unwrap_or(0))
                    .collect();
                let expected_mask: Vec<bool> = cat
                    .unique_values()
                    .iter()
                    .map(|s| s.parse::<i32>().is_ok())
                    .collect();
                // a.data contains, for each code, expected_vals[code]
                for (ix, &cat_idx) in cat.data.iter().enumerate() {
                    assert_eq!(a.data[ix], expected_vals[cat_idx as usize]);
                    let mask = a.null_mask.as_ref().unwrap();
                    assert_eq!(mask.get(ix), expected_mask[cat_idx as usize]);
                }
            }
            _ => panic!("Expected Int32"),
        }
    }

    #[test]
    fn test_str_from_bool_array() {
        let mut arr = BooleanArray::default();
        arr.push(true);
        arr.push(false);
        arr.push_null();
        let array = Array::from_bool(arr);
        let out = array.str();
        match out {
            TextArray::String32(ref s) => {
                let got: Vec<String> = (0..s.len())
                    .map(|i| s.get_str(i).unwrap_or("").to_string())
                    .collect();
                assert_eq!(&got[..], &["true", "false", ""]);
            }
            _ => panic!("Expected String32"),
        }
    }

    #[test]
    fn test_str_from_int_array() {
        let arr = IntegerArray::<i32>::from_slice(&[5, 0, -10]);
        let array = Array::from_int32(arr);
        let out = array.str();
        match out {
            TextArray::String32(ref s) => {
                let got: Vec<_> = (0..s.len())
                    .map(|i| s.get_str(i).unwrap_or("").to_string())
                    .collect();
                assert_eq!(&got[..], &["5", "0", "-10"]);
            }
            _ => panic!("Expected String32"),
        }
    }

    #[test]
    fn test_bool_from_int_array() {
        let arr = IntegerArray::<i32>::from_slice(&[0, 1, -2, 0]);
        let array = Array::from_int32(arr);
        let out = array.bool();
        let values: Vec<_> = (0..out.len()).map(|i| out.get(i)).collect();
        assert_eq!(
            &values[..],
            &[Some(false), Some(true), Some(true), Some(false)]
        );
    }

    #[test]
    fn test_bool_from_string_array() {
        let arr = StringArray::<u32>::from_slice(&["True", "0", "false", "abc", ""]);
        let array = Array::from_string32(arr);
        let out = array.bool();
        let values: Vec<_> = (0..out.len()).map(|i| out.get(i)).collect();
        assert_eq!(
            &values[..],
            &[
                Some(true),
                Some(false),
                Some(false),
                Some(true),
                Some(false)
            ]
        );
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn test_num_from_datetime_i32() {
        use crate::TimeUnit;

        let dt = DatetimeArray::<i32>::from_slice(&[123, 456, 789], Some(TimeUnit::Milliseconds));
        let array = Array::from_datetime_i32(dt.clone());
        let out = array.num();
        match out {
            NumericArray::Int32(ref a) => assert_eq!(&a.data[..], &[123, 456, 789]),
            _ => panic!("Expected Int32"),
        }
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn test_dt_from_int_array() {
        let arr = IntegerArray::<i32>::from_slice(&[1000, 2000]);
        let array = Array::from_int32(arr);
        let out = array.dt();
        match out {
            TemporalArray::Datetime64(ref dt) => assert_eq!(&dt.data[..], &[1000, 2000]),
            _ => panic!("Expected DatetimeI64"),
        }
    }

    #[cfg(feature = "datetime_ops")]
    #[test]
    fn test_dt_from_text_array_parsing() {
        let arr = StringArray::<u32>::from_slice(&[
            "2023-01-01T00:00:00Z",
            "foo",
            "2020-06-30T12:00:00Z",
        ]);
        let array = Array::from_string32(arr);
        println!("{:?}", array);
        let out = array.dt();
        println!("{:?}", out);
        match out {
            TemporalArray::Datetime64(ref dt) => {
                assert_eq!(dt.len(), 3);
                let valid = dt.null_mask.as_ref().unwrap();
                assert!(valid.get(0), "First date should be valid");
                assert!(!valid.get(1), "Second date ('foo') should be invalid/null");
                assert!(valid.get(2), "Third date should be valid");
            }
            _ => panic!("Expected DatetimeI64"),
        }
    }

    #[test]
    fn test_null_cases() {
        let array = Array::Null;
        assert!(array.try_num().is_err());
        assert!(array.try_str().is_err());
        assert!(array.try_bool().is_err());
        #[cfg(feature = "datetime")]
        assert!(array.try_dt().is_err());
    }

    // ── typed accessor tests ─────────────────────────────────────────

    #[test]
    fn test_num_i32_accessor() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(10);
        let array = Array::from_int32(arr);
        let inner = array.num().i32();
        assert_eq!(inner.data[0], 10);
    }

    #[test]
    fn test_num_try_i32_converts_float() {
        let mut arr = FloatArray::<f64>::default();
        arr.push(3.0);
        let array = Array::from_float64(arr);
        let inner = array.num().try_i32().unwrap();
        assert_eq!(inner.data[0], 3);
    }

    #[test]
    fn test_num_f64_accessor() {
        let mut arr = FloatArray::<f64>::default();
        arr.push(3.14);
        let array = Array::from_float64(arr);
        let inner = array.num().f64();
        assert!((inner.data[0] - 3.14).abs() < f64::EPSILON);
    }

    #[test]
    fn test_str_str32_accessor() {
        let arr = StringArray::<u32>::from_slice(&["hello", "world"]);
        let array = Array::from_string32(arr);
        let inner = array.str().str32();
        assert_eq!(inner.get_str(0), Some("hello"));
    }

    #[test]
    fn test_str_try_str32_converts_int() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(42);
        let array = Array::from_int32(arr);
        let inner = array.str().try_str32().unwrap();
        assert_eq!(inner.get_str(0), Some("42"));
    }

    #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
    #[test]
    fn test_str_cat32_accessor() {
        let arr = CategoricalArray::<u32>::from_vec(vec!["a", "b", "a"], None);
        let array = Array::from_categorical32(arr);
        let inner = array.str().cat32();
        assert_eq!(inner.get_str(0), Some("a"));
    }

    #[test]
    fn test_try_accessors_null_array() {
        let array = Array::Null;
        assert!(array.try_num().is_err());
        assert!(array.try_str().is_err());
        assert!(array.try_bool().is_err());
    }

    // ── value_to_string tests ─────────────────────────────────────────

    #[test]
    fn test_value_to_string_integer() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(42);
        arr.push(-7);
        let array = Array::from_int32(arr);
        assert_eq!(array.value_to_string(0), "42");
        assert_eq!(array.value_to_string(1), "-7");
    }

    #[test]
    fn test_value_to_string_float() {
        let mut arr = FloatArray::<f64>::default();
        arr.push(3.14);
        let array = Array::from_float64(arr);
        assert_eq!(array.value_to_string(0), "3.14");
    }

    #[test]
    fn test_value_to_string_string() {
        let arr = StringArray::<u32>::from_slice(&["hello"]);
        let array = Array::from_string32(arr);
        assert_eq!(array.value_to_string(0), "hello");
    }

    #[test]
    fn test_value_to_string_bool() {
        let arr = BooleanArray::from_slice(&[true, false]);
        let array = Array::from_bool(arr);
        assert_eq!(array.value_to_string(0), "true");
        assert_eq!(array.value_to_string(1), "false");
    }

    #[test]
    fn test_value_to_string_null_variant() {
        let array = Array::Null;
        assert_eq!(array.value_to_string(0), "null");
    }

    #[test]
    fn test_value_to_string_null_value() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(10);
        arr.push(20);
        arr.set_null_mask(Some(Bitmask::from_bools(&[true, false])));
        let array = Array::from_int32(arr);
        assert_eq!(array.value_to_string(0), "10");
        assert_eq!(array.value_to_string(1), "null");
    }

    // ── compare_at tests ─────────────────────────────────────────────

    #[test]
    fn test_compare_at_ascending() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(1);
        arr.push(2);
        arr.push(3);
        let array = Array::from_int32(arr);
        assert_eq!(array.compare_at(0, 1), std::cmp::Ordering::Less);
        assert_eq!(array.compare_at(2, 1), std::cmp::Ordering::Greater);
    }

    #[test]
    fn test_compare_at_equal() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(5);
        arr.push(5);
        let array = Array::from_int32(arr);
        assert_eq!(array.compare_at(0, 1), std::cmp::Ordering::Equal);
    }

    #[test]
    fn test_compare_at_null_handling() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(1);
        arr.push(2);
        arr.push(3);
        arr.set_null_mask(Some(Bitmask::from_bools(&[true, false, true])));
        let array = Array::from_int32(arr);
        // null > non-null
        assert_eq!(array.compare_at(1, 0), std::cmp::Ordering::Greater);
        // non-null < null
        assert_eq!(array.compare_at(0, 1), std::cmp::Ordering::Less);
    }

    #[test]
    fn test_compare_at_both_null() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(1);
        arr.push(2);
        arr.set_null_mask(Some(Bitmask::from_bools(&[false, false])));
        let array = Array::from_int32(arr);
        assert_eq!(array.compare_at(0, 1), std::cmp::Ordering::Equal);
    }

    #[test]
    fn test_compare_at_float_nan() {
        let mut arr = FloatArray::<f64>::default();
        arr.push(1.0);
        arr.push(f64::NAN);
        let array = Array::from_float64(arr);
        // NaN sorts after all values with total_cmp
        assert_eq!(array.compare_at(0, 1), std::cmp::Ordering::Less);
        // NaN == NaN with total_cmp
        let mut arr2 = FloatArray::<f64>::default();
        arr2.push(f64::NAN);
        arr2.push(f64::NAN);
        let array2 = Array::from_float64(arr2);
        assert_eq!(array2.compare_at(0, 1), std::cmp::Ordering::Equal);
    }

    #[test]
    fn test_compare_at_string() {
        let arr = StringArray::<u32>::from_slice(&["apple", "banana", "apple"]);
        let array = Array::from_string32(arr);
        assert_eq!(array.compare_at(0, 1), std::cmp::Ordering::Less);
        assert_eq!(array.compare_at(0, 2), std::cmp::Ordering::Equal);
    }

    #[test]
    fn test_compare_at_bool() {
        let arr = BooleanArray::from_slice(&[false, true]);
        let array = Array::from_bool(arr);
        assert_eq!(array.compare_at(0, 1), std::cmp::Ordering::Less);
    }

    // ── hash_element_at tests ────────────────────────────────────────

    #[cfg(feature = "hash")]
    #[test]
    fn test_hash_element_at_same_value() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut arr = IntegerArray::<i32>::default();
        arr.push(42);
        arr.push(42);
        let array = Array::from_int32(arr);

        let mut h1 = DefaultHasher::new();
        array.hash_element_at(0, &mut h1);
        let mut h2 = DefaultHasher::new();
        array.hash_element_at(1, &mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[cfg(feature = "hash")]
    #[test]
    fn test_hash_element_at_different_value() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut arr = IntegerArray::<i32>::default();
        arr.push(1);
        arr.push(2);
        let array = Array::from_int32(arr);

        let mut h1 = DefaultHasher::new();
        array.hash_element_at(0, &mut h1);
        let mut h2 = DefaultHasher::new();
        array.hash_element_at(1, &mut h2);
        assert_ne!(h1.finish(), h2.finish());
    }

    #[cfg(feature = "hash")]
    #[test]
    fn test_hash_element_at_null_consistent() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut arr = IntegerArray::<i32>::default();
        arr.push(10);
        arr.push(20);
        arr.set_null_mask(Some(Bitmask::from_bools(&[false, false])));
        let array = Array::from_int32(arr);

        let mut h1 = DefaultHasher::new();
        array.hash_element_at(0, &mut h1);
        let mut h2 = DefaultHasher::new();
        array.hash_element_at(1, &mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[cfg(feature = "hash")]
    #[test]
    fn test_hash_element_at_float_to_bits() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut arr = FloatArray::<f64>::default();
        arr.push(3.14);
        arr.push(3.14);
        let array = Array::from_float64(arr);

        let mut h1 = DefaultHasher::new();
        array.hash_element_at(0, &mut h1);
        let mut h2 = DefaultHasher::new();
        array.hash_element_at(1, &mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }
}

#[cfg(test)]
mod macro_tests {
    use crate::{Array, Bitmask, MaskedArray, NumericArray, TextArray, Vec64, vec64};

    // helper for checking null masks
    fn assert_mask(mask: &Option<Bitmask>, expected: &[bool]) {
        if expected.iter().all(|&b| b) {
            assert!(mask.is_none(), "Expected no null mask");
        } else {
            let m = mask.as_ref().expect("Expected Some(null_mask)");
            for (i, &val) in expected.iter().enumerate() {
                assert_eq!(m.get(i), val, "Mask mismatch at position {}", i);
            }
        }
    }

    // ===== numeric types =====

    #[test]
    fn arr_i32_vec64_dense() {
        let v = vec64![1i32, 2, 3];
        let arr = arr_i32!(v);
        if let Array::NumericArray(NumericArray::Int32(a)) = arr {
            assert_eq!(a.data.as_slice(), &[1, 2, 3]);
            assert_mask(&a.null_mask, &[true, true, true]);
        } else {
            panic!("arr_i32!(Vec64) wrong variant");
        }
    }

    #[test]
    fn arr_i32_vec64_opt() {
        let v: Vec64<Option<i32>> = vec64![Some(1i32), None::<i32>, Some(3)];
        let arr = arr_i32_opt!(v);
        if let Array::NumericArray(NumericArray::Int32(a)) = arr {
            assert_eq!(a.data.as_slice(), &[1, 0, 3]);
            assert_mask(&a.null_mask, &[true, false, true]);
        } else {
            panic!("arr_i32_opt!(Vec64<Option>) wrong variant");
        }
    }

    #[test]
    fn arr_f64_vec64_dense() {
        let v = vec64![1.1f64, 2.2, 3.3];
        let arr = arr_f64!(v);
        if let Array::NumericArray(NumericArray::Float64(a)) = arr {
            assert_eq!(a.data.as_slice(), &[1.1, 2.2, 3.3]);
            assert_mask(&a.null_mask, &[true, true, true]);
        } else {
            panic!("arr_f64!(Vec64) wrong variant");
        }
    }

    #[test]
    fn arr_f64_vec64_opt() {
        let v = vec64![Some(1.5f64), None::<f64>, Some(-2.5)];
        let arr = arr_f64_opt!(v);
        if let Array::NumericArray(NumericArray::Float64(a)) = arr {
            assert_eq!(a.data.as_slice(), &[1.5, 0.0, -2.5]);
            assert_mask(&a.null_mask, &[true, false, true]);
        } else {
            panic!("arr_f64_opt!(Vec64<Option>) wrong variant");
        }
    }

    // ===== bool =====

    #[test]
    fn arr_bool_vec64_dense() {
        let v = vec64![true, false, true];
        let arr = arr_bool!(v);
        if let Array::BooleanArray(a) = arr {
            assert_eq!(a.get(0), Some(true));
            assert_eq!(a.get(1), Some(false));
            assert_eq!(a.get(2), Some(true));
            assert_mask(&a.null_mask, &[true, true, true]);
        } else {
            panic!("arr_bool!(Vec64) wrong variant");
        }
    }

    #[test]
    fn arr_bool_vec64_opt() {
        let v = vec64![Some(true), None::<bool>, Some(false)];
        let arr = arr_bool_opt!(v);
        if let Array::BooleanArray(a) = arr {
            assert_eq!(a.get(0), Some(true));
            assert_eq!(a.get(1), None);
            assert_eq!(a.get(2), Some(false));
            assert_mask(&a.null_mask, &[true, false, true]);
        } else {
            panic!("arr_bool_opt!(Vec64<Option>) wrong variant");
        }
    }

    // ===== string =====

    #[test]
    fn arr_str32_vec64_dense() {
        let v = vec64!["a", "b", "c"];
        let arr = arr_str32!(v);
        if let Array::TextArray(TextArray::String32(a)) = arr {
            assert_eq!(a.get_str(0), Some("a"));
            assert_eq!(a.get_str(1), Some("b"));
            assert_eq!(a.get_str(2), Some("c"));
            assert_mask(&a.null_mask, &[true, true, true]);
        } else {
            panic!("arr_str32!(Vec64) wrong variant");
        }
    }

    #[test]
    fn arr_str32_vec64_opt() {
        let v = vec64![Some("x"), None::<&str>, Some("y")];
        let arr = arr_str32_opt!(v);
        if let Array::TextArray(TextArray::String32(a)) = arr {
            assert_eq!(a.get_str(0), Some("x"));
            assert_eq!(a.get_str(1), None);
            assert_eq!(a.get_str(2), Some("y"));
            assert_mask(&a.null_mask, &[true, false, true]);
        } else {
            panic!("arr_str32_opt!(Vec64<Option>) wrong variant");
        }
    }

    // ===== categorical =====

    #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
    #[test]
    fn arr_cat32_vec64_dense() {
        let v = vec64!["red", "green", "red"];
        let arr = arr_cat32!(v);
        if let Array::TextArray(TextArray::Categorical32(a)) = arr {
            assert_eq!(a.get_str(0), Some("red"));
            assert_eq!(a.get_str(1), Some("green"));
            assert_eq!(a.get_str(2), Some("red"));
            assert_mask(&a.null_mask, &[true, true, true]);
        } else {
            panic!("arr_cat32!(Vec64) wrong variant");
        }
    }

    #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
    #[test]
    fn arr_cat32_vec64_opt() {
        let v = vec64![Some("red"), None::<&str>, Some("blue")];
        let arr = arr_cat32_opt!(v);
        if let Array::TextArray(TextArray::Categorical32(a)) = arr {
            assert_eq!(a.get_str(0), Some("red"));
            assert_eq!(a.get_str(1), None);
            assert_eq!(a.get_str(2), Some("blue"));
            assert_mask(&a.null_mask, &[true, false, true]);
        } else {
            panic!("arr_cat32_opt!(Vec64<Option>) wrong variant");
        }
    }

    // ===== All numeric types =====

    #[cfg(feature = "extended_numeric_types")]
    #[test]
    fn test_all_integer_types() {
        // i8
        let arr = arr_i8!(1i8, 2, 3);
        if let Array::NumericArray(NumericArray::Int8(a)) = arr {
            assert_eq!(a.data.as_slice(), &[1, 2, 3]);
        } else {
            panic!("Wrong variant");
        }

        let arr = arr_i8_opt!(Some(1i8), None::<i8>, Some(3));
        if let Array::NumericArray(NumericArray::Int8(a)) = arr {
            assert_eq!(a.data.as_slice(), &[1, 0, 3]);
            assert_mask(&a.null_mask, &[true, false, true]);
        } else {
            panic!("Wrong variant");
        }

        // i16
        let arr = arr_i16!(100i16, 200, 300);
        if let Array::NumericArray(NumericArray::Int16(a)) = arr {
            assert_eq!(a.data.as_slice(), &[100, 200, 300]);
        } else {
            panic!("Wrong variant");
        }

        // u8
        let arr = arr_u8!(10u8, 20, 30);
        if let Array::NumericArray(NumericArray::UInt8(a)) = arr {
            assert_eq!(a.data.as_slice(), &[10, 20, 30]);
        } else {
            panic!("Wrong variant");
        }

        // u16
        let arr = arr_u16!(1000u16, 2000, 3000);
        if let Array::NumericArray(NumericArray::UInt16(a)) = arr {
            assert_eq!(a.data.as_slice(), &[1000, 2000, 3000]);
        } else {
            panic!("Wrong variant");
        }
    }

    #[test]
    fn test_standard_integer_types() {
        // i32
        let arr = arr_i32!(vec64![1i32, 2, 3]);
        if let Array::NumericArray(NumericArray::Int32(a)) = arr {
            assert_eq!(a.data.as_slice(), &[1, 2, 3]);
        } else {
            panic!("Wrong variant");
        }

        let arr = arr_i32_opt!(vec64![Some(1i32), None::<i32>, Some(3)]);
        if let Array::NumericArray(NumericArray::Int32(a)) = arr {
            assert_mask(&a.null_mask, &[true, false, true]);
        } else {
            panic!("Wrong variant");
        }

        // i64
        let arr = arr_i64!(vec64![100i64, 200, 300]);
        if let Array::NumericArray(NumericArray::Int64(a)) = arr {
            assert_eq!(a.data.as_slice(), &[100, 200, 300]);
        } else {
            panic!("Wrong variant");
        }

        // u32
        let arr = arr_u32!(vec64![1000u32, 2000, 3000]);
        if let Array::NumericArray(NumericArray::UInt32(a)) = arr {
            assert_eq!(a.data.as_slice(), &[1000, 2000, 3000]);
        } else {
            panic!("Wrong variant");
        }

        // u64
        let arr = arr_u64!(vec64![10000u64, 20000, 30000]);
        if let Array::NumericArray(NumericArray::UInt64(a)) = arr {
            assert_eq!(a.data.as_slice(), &[10000, 20000, 30000]);
        } else {
            panic!("Wrong variant");
        }
    }

    #[test]
    fn test_float_types() {
        // f32
        let arr = arr_f32!(vec64![1.1f32, 2.2, 3.3]);
        if let Array::NumericArray(NumericArray::Float32(a)) = arr {
            assert_eq!(a.data.as_slice(), &[1.1, 2.2, 3.3]);
        } else {
            panic!("Wrong variant");
        }

        let arr = arr_f32_opt!(vec64![Some(1.5f32), None::<f32>, Some(-2.5)]);
        if let Array::NumericArray(NumericArray::Float32(a)) = arr {
            assert_mask(&a.null_mask, &[true, false, true]);
        } else {
            panic!("Wrong variant");
        }

        // f64
        let arr = arr_f64!(vec64![10.1f64, 20.2, 30.3]);
        if let Array::NumericArray(NumericArray::Float64(a)) = arr {
            assert_eq!(a.data.as_slice(), &[10.1, 20.2, 30.3]);
        } else {
            panic!("Wrong variant");
        }
    }

    #[test]
    fn test_boolean_types() {
        let arr = arr_bool!(vec64![true, false, true]);
        if let Array::BooleanArray(a) = arr {
            assert_eq!(a.get(0), Some(true));
            assert_eq!(a.get(1), Some(false));
            assert_eq!(a.get(2), Some(true));
        } else {
            panic!("Wrong variant");
        }

        let arr = arr_bool_opt!(vec64![Some(true), None::<bool>, Some(false)]);
        if let Array::BooleanArray(a) = arr {
            assert_mask(&a.null_mask, &[true, false, true]);
        } else {
            panic!("Wrong variant");
        }
    }

    #[test]
    fn test_string_types() {
        let arr = arr_str32!(vec64!["hello", "world", "test"]);
        if let Array::TextArray(TextArray::String32(a)) = arr {
            assert_eq!(a.get_str(0), Some("hello"));
            assert_eq!(a.get_str(1), Some("world"));
            assert_eq!(a.get_str(2), Some("test"));
        } else {
            panic!("Wrong variant");
        }

        let arr = arr_str32_opt!(vec64![Some("x"), None::<&str>, Some("y")]);
        if let Array::TextArray(TextArray::String32(a)) = arr {
            assert_eq!(a.get_str(0), Some("x"));
            assert_eq!(a.get_str(1), None);
            assert_eq!(a.get_str(2), Some("y"));
            assert_mask(&a.null_mask, &[true, false, true]);
        } else {
            panic!("Wrong variant");
        }

        #[cfg(feature = "large_string")]
        {
            let arr = arr_str64!(vec64!["large", "string", "test"]);
            if let Array::TextArray(TextArray::String64(a)) = arr {
                assert_eq!(a.get_str(0), Some("large"));
            } else {
                panic!("Wrong variant");
            }
        }
    }

    #[test]
    fn test_categorical_types() {
        #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
        {
            let arr = arr_cat32!(vec64!["red", "green", "red"]);
            if let Array::TextArray(TextArray::Categorical32(a)) = arr {
                assert_eq!(a.get_str(0), Some("red"));
                assert_eq!(a.get_str(1), Some("green"));
                assert_eq!(a.get_str(2), Some("red"));
            } else {
                panic!("Wrong variant");
            }

            let arr = arr_cat32_opt!(vec64![Some("red"), None::<&str>, Some("blue")]);
            if let Array::TextArray(TextArray::Categorical32(a)) = arr {
                assert_eq!(a.get_str(0), Some("red"));
                assert_eq!(a.get_str(1), None);
                assert_eq!(a.get_str(2), Some("blue"));
                assert_mask(&a.null_mask, &[true, false, true]);
            } else {
                panic!("Wrong variant");
            }
        }

        #[cfg(feature = "extended_categorical")]
        {
            let arr = arr_cat8!(vec64!["a", "b", "a"]);
            if let Array::TextArray(TextArray::Categorical8(a)) = arr {
                assert_eq!(a.get_str(0), Some("a"));
            } else {
                panic!("Wrong variant");
            }

            let arr = arr_cat16!(vec64!["x", "y", "x"]);
            if let Array::TextArray(TextArray::Categorical16(a)) = arr {
                assert_eq!(a.get_str(0), Some("x"));
            } else {
                panic!("Wrong variant");
            }

            let arr = arr_cat64!(vec64!["alpha", "beta", "alpha"]);
            if let Array::TextArray(TextArray::Categorical64(a)) = arr {
                assert_eq!(a.get_str(0), Some("alpha"));
            } else {
                panic!("Wrong variant");
            }
        }
    }

    #[test]
    fn test_empty_arrays() {
        let arr = arr_i32!(vec64![]);
        if let Array::NumericArray(NumericArray::Int32(a)) = arr {
            assert_eq!(a.len(), 0);
        } else {
            panic!("Wrong variant");
        }

        let arr = arr_str32!(vec64![]);
        if let Array::TextArray(TextArray::String32(a)) = arr {
            assert_eq!(a.len(), 0);
        } else {
            panic!("Wrong variant");
        }
    }

    #[test]
    fn test_all_nulls() {
        let arr = arr_i32_opt!(vec64![None::<i32>, None::<i32>, None::<i32>]);
        if let Array::NumericArray(NumericArray::Int32(a)) = arr {
            assert_eq!(a.null_count(), 3);
            assert_mask(&a.null_mask, &[false, false, false]);
        } else {
            panic!("Wrong variant");
        }

        let arr = arr_str32_opt!(vec64![None::<&str>, None::<&str>, None::<&str>]);
        if let Array::TextArray(TextArray::String32(a)) = arr {
            assert_eq!(a.null_count(), 3);
        } else {
            panic!("Wrong variant");
        }
    }

    #[test]
    fn test_literal_syntax_no_nulls() {
        // Numeric types
        let arr = arr_i32![1, 2, 3, 4];
        if let Array::NumericArray(NumericArray::Int32(a)) = arr {
            assert_eq!(a.data.as_slice(), &[1, 2, 3, 4]);
            assert!(a.null_mask.is_none());
        } else {
            panic!("Wrong variant");
        }

        let arr = arr_f64![0.5, 1.5, 2.5];
        if let Array::NumericArray(NumericArray::Float64(a)) = arr {
            assert_eq!(a.data.as_slice(), &[0.5, 1.5, 2.5]);
            assert!(a.null_mask.is_none());
        } else {
            panic!("Wrong variant");
        }

        // Boolean
        let arr = arr_bool![true, false, true];
        if let Array::BooleanArray(a) = arr {
            assert_eq!(a.get(0), Some(true));
            assert_eq!(a.get(1), Some(false));
            assert_eq!(a.get(2), Some(true));
            assert!(a.null_mask.is_none());
        } else {
            panic!("Wrong variant");
        }

        // String
        let arr = arr_str32!["a", "b", "c"];
        if let Array::TextArray(TextArray::String32(a)) = arr {
            assert_eq!(a.get_str(0), Some("a"));
            assert_eq!(a.get_str(1), Some("b"));
            assert_eq!(a.get_str(2), Some("c"));
            assert!(a.null_mask.is_none());
        } else {
            panic!("Wrong variant");
        }

        // Categorical
        #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
        {
            let arr = arr_cat32!["x", "y", "x", "z"];
            if let Array::TextArray(TextArray::Categorical32(a)) = arr {
                assert_eq!(a.get_str(0), Some("x"));
                assert_eq!(a.get_str(1), Some("y"));
                assert_eq!(a.get_str(2), Some("x"));
                assert_eq!(a.get_str(3), Some("z"));
                assert!(a.null_mask.is_none());
            } else {
                panic!("Wrong variant");
            }
        }
    }

    #[test]
    fn test_literal_syntax_with_nulls() {
        // Test explicit null handling macros
        let arr = arr_i32_opt![Some(1), None::<i32>, Some(3)];
        if let Array::NumericArray(NumericArray::Int32(a)) = arr {
            assert_eq!(a.data.as_slice(), &[1, 0, 3]);
            assert!(a.null_mask.is_some());
            assert_eq!(a.get(0), Some(1));
            assert_eq!(a.get(1), None);
            assert_eq!(a.get(2), Some(3));
        } else {
            panic!("Wrong variant");
        }

        let arr = arr_str32_opt![Some("hello"), None::<&str>, Some("world")];
        if let Array::TextArray(TextArray::String32(a)) = arr {
            assert_eq!(a.get_str(0), Some("hello"));
            assert_eq!(a.get_str(1), None);
            assert_eq!(a.get_str(2), Some("world"));
            assert!(a.null_mask.is_some());
        } else {
            panic!("Wrong variant");
        }

        let arr = arr_bool_opt![Some(true), None::<bool>, Some(false)];
        if let Array::BooleanArray(a) = arr {
            assert_eq!(a.get(0), Some(true));
            assert_eq!(a.get(1), None);
            assert_eq!(a.get(2), Some(false));
            assert!(a.null_mask.is_some());
        } else {
            panic!("Wrong variant");
        }

        #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
        {
            let arr = arr_cat32_opt![Some("red"), None::<&str>, Some("blue")];
            if let Array::TextArray(TextArray::Categorical32(a)) = arr {
                assert_eq!(a.get_str(0), Some("red"));
                assert_eq!(a.get_str(1), None);
                assert_eq!(a.get_str(2), Some("blue"));
                assert!(a.null_mask.is_some());
            } else {
                panic!("Wrong variant");
            }
        }
    }

    #[test]
    fn test_single_elements() {
        // TODO: This currently needs vec64 wrapping
        let arr = arr_i32![vec64![42]];
        if let Array::NumericArray(NumericArray::Int32(a)) = arr {
            assert_eq!(a.data.as_slice(), &[42]);
            assert!(a.null_mask.is_none());
        } else {
            panic!("Wrong variant");
        }

        let arr = arr_str32![vec64!["hello"]];
        if let Array::TextArray(TextArray::String32(a)) = arr {
            assert_eq!(a.get_str(0), Some("hello"));
            assert!(a.null_mask.is_none());
        } else {
            panic!("Wrong variant");
        }
    }

    #[test]
    fn test_mixed_usage() {
        // Test that both syntaxes work
        let v = vec64![1, 2, 3];
        let arr1 = arr_i32!(v);
        let arr2 = arr_i32![1, 2, 3];

        if let (
            Array::NumericArray(NumericArray::Int32(a1)),
            Array::NumericArray(NumericArray::Int32(a2)),
        ) = (arr1, arr2)
        {
            assert_eq!(a1.data.as_slice(), a2.data.as_slice());
        } else {
            panic!("Wrong variants");
        }
    }
}

impl Shape for Array {
    fn shape(&self) -> ShapeDim {
        match self {
            Array::NumericArray(numeric_array) => numeric_array.shape(),
            Array::TextArray(text_array) => text_array.shape(),
            #[cfg(feature = "datetime")]
            Array::TemporalArray(temporal_array) => temporal_array.shape(),
            Array::BooleanArray(boolean_array) => boolean_array.shape(),
            Array::Null => ShapeDim::Rank0(0),
        }
    }
}

impl Concatenate for Array {
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        match (self, other) {
            (Array::NumericArray(a), Array::NumericArray(b)) => {
                Ok(Array::NumericArray(a.concat(b)?))
            }
            (Array::TextArray(a), Array::TextArray(b)) => Ok(Array::TextArray(a.concat(b)?)),
            #[cfg(feature = "datetime")]
            (Array::TemporalArray(a), Array::TemporalArray(b)) => {
                Ok(Array::TemporalArray(a.concat(b)?))
            }
            (Array::BooleanArray(a), Array::BooleanArray(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(Array::BooleanArray(Arc::new(a.concat(b)?)))
            }
            (Array::Null, Array::Null) => Ok(Array::Null),
            (lhs, rhs) => Err(MinarrowError::IncompatibleTypeError {
                from: "Array",
                to: "Array",
                message: Some(format!(
                    "Cannot concatenate mismatched Array categories: {} and {}",
                    array_category_name(&lhs),
                    array_category_name(&rhs)
                )),
            }),
        }
    }
}

/// Helper function to get the category name for error messages
fn array_category_name(arr: &Array) -> &'static str {
    match arr {
        Array::NumericArray(_) => "NumericArray",
        Array::TextArray(_) => "TextArray",
        #[cfg(feature = "datetime")]
        Array::TemporalArray(_) => "TemporalArray",
        Array::BooleanArray(_) => "BooleanArray",
        Array::Null => "Null",
    }
}

// =
// RowSelection Implementation
// =

#[cfg(all(feature = "select", feature = "views"))]
impl crate::traits::selection::RowSelection for Array {
    type View = ArrayV;

    /// Select rows by index or range, returning an ArrayV (view)
    ///
    /// For contiguous selections (ranges), creates a zero-copy view.
    /// For non-contiguous selections (index arrays), gathers into a new array.
    fn r<S: crate::traits::selection::DataSelector>(&self, selection: S) -> ArrayV {
        if selection.is_contiguous() {
            // Contiguous selection (ranges): create a view
            let indices = selection.resolve_indices(self.len());
            if indices.is_empty() {
                return ArrayV::new(self.clone(), 0, 0);
            }
            let offset = indices[0];
            let len = indices.len();
            ArrayV::new(self.clone(), offset, len)
        } else {
            // Non-contiguous selection (index arrays): gather into new array
            // For now, create a view over the whole array and use its gather
            let full_view = ArrayV::new(self.clone(), 0, self.len());
            let indices = selection.resolve_indices(self.len());
            let gathered = full_view.gather_indices(&indices);
            ArrayV::new(gathered, 0, indices.len())
        }
    }

    fn get_row_count(&self) -> usize {
        self.len()
    }
}

#[cfg(test)]
mod concat_tests {
    use super::*;
    use crate::{IntegerArray, StringArray};

    #[test]
    fn test_array_concat_numeric() {
        let arr1 = Array::from_int32(IntegerArray::from_slice(&[1, 2, 3]));
        let arr2 = Array::from_int32(IntegerArray::from_slice(&[4, 5, 6]));

        let result = arr1.concat(arr2).unwrap();

        match result {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.len(), 6);
                assert_eq!(arr.data.as_slice(), &[1, 2, 3, 4, 5, 6]);
            }
            _ => panic!("Expected Int32 array"),
        }
    }

    #[test]
    fn test_array_concat_text() {
        let arr1 = Array::from_string32(StringArray::from_slice(&["a", "b"]));
        let arr2 = Array::from_string32(StringArray::from_slice(&["c", "d"]));

        let result = arr1.concat(arr2).unwrap();

        match result {
            Array::TextArray(TextArray::String32(arr)) => {
                assert_eq!(arr.len(), 4);
                assert_eq!(arr.get_str(0), Some("a"));
                assert_eq!(arr.get_str(1), Some("b"));
                assert_eq!(arr.get_str(2), Some("c"));
                assert_eq!(arr.get_str(3), Some("d"));
            }
            _ => panic!("Expected String32 array"),
        }
    }

    #[test]
    fn test_array_concat_boolean() {
        let arr1 = Array::from_bool(BooleanArray::from_slice(&[true, false, true]));
        let arr2 = Array::from_bool(BooleanArray::from_slice(&[false, true]));

        let result = arr1.concat(arr2).unwrap();

        match result {
            Array::BooleanArray(arr) => {
                assert_eq!(arr.len(), 5);
                assert_eq!(arr.get(0), Some(true));
                assert_eq!(arr.get(1), Some(false));
                assert_eq!(arr.get(2), Some(true));
                assert_eq!(arr.get(3), Some(false));
                assert_eq!(arr.get(4), Some(true));
            }
            _ => panic!("Expected BooleanArray"),
        }
    }

    #[test]
    fn test_array_concat_mismatched_types() {
        let arr1 = Array::from_int32(IntegerArray::from_slice(&[1, 2, 3]));
        let arr2 = Array::from_string32(StringArray::from_slice(&["a", "b"]));

        let result = arr1.concat(arr2);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            MinarrowError::IncompatibleTypeError { .. }
        ));
    }

    #[test]
    fn test_array_concat_null() {
        let arr1 = Array::Null;
        let arr2 = Array::Null;

        let result = arr1.concat(arr2).unwrap();

        assert!(matches!(result, Array::Null));
    }
}

#[cfg(test)]
mod arr_macro_extensions_tests {
    use crate::{Array, Bitmask, NumericArray, TextArray, Vec64, vec64};

    fn expect_int32_mask(arr: Array) -> (Vec<i32>, Option<Bitmask>) {
        match arr {
            Array::NumericArray(NumericArray::Int32(a)) => {
                (a.data.as_slice().to_vec(), a.null_mask.clone())
            }
            _ => panic!("expected Int32 Array"),
        }
    }

    fn expect_f64_mask(arr: Array) -> (Vec<f64>, Option<Bitmask>) {
        match arr {
            Array::NumericArray(NumericArray::Float64(a)) => {
                (a.data.as_slice().to_vec(), a.null_mask.clone())
            }
            _ => panic!("expected Float64 Array"),
        }
    }

    fn expect_bool_mask(arr: Array) -> (Vec<bool>, Option<Bitmask>) {
        match arr {
            Array::BooleanArray(a) => {
                let data = (0..a.data.len()).map(|i| a.data.get(i)).collect();
                (data, a.null_mask.clone())
            }
            _ => panic!("expected Boolean Array"),
        }
    }

    #[test]
    fn arr_f64_accepts_slice_dense() {
        let s: &[f64] = &[1.0, 2.0, 3.0];
        let (data, mask) = expect_f64_mask(arr_f64!(s));
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
        assert!(mask.is_none());
    }

    #[test]
    fn arr_f64_accepts_vec64_dense() {
        let v: Vec64<f64> = vec64![1.0, 2.0, 3.0];
        let (data, mask) = expect_f64_mask(arr_f64!(v));
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
        assert!(mask.is_none());
    }

    #[test]
    fn arr_f64_literal_elements_unchanged() {
        let (data, mask) = expect_f64_mask(arr_f64![1.0, 2.0, 3.0]);
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
        assert!(mask.is_none());
    }

    #[test]
    fn arr_f64_accepts_literal_array_ref() {
        let (data, mask) = expect_f64_mask(arr_f64!(&[1.0, 2.0, 3.0]));
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
        assert!(mask.is_none());
    }

    #[test]
    fn arr_f64_slice_with_mask() {
        let s: &[f64] = &[1.0, 2.0, 3.0];
        let m = Bitmask::from_bools(&[true, false, true]);
        let (data, mask) = expect_f64_mask(arr_f64!(s; m));
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
        let m = mask.expect("expected mask");
        assert_eq!(m.get(0), true);
        assert_eq!(m.get(1), false);
        assert_eq!(m.get(2), true);
    }

    #[test]
    fn arr_f64_vec64_with_mask() {
        let v: Vec64<f64> = vec64![10.0, 20.0, 30.0];
        let m = Bitmask::from_bools(&[true, true, false]);
        let (data, mask) = expect_f64_mask(arr_f64!(v; m));
        assert_eq!(data, vec![10.0, 20.0, 30.0]);
        let m = mask.expect("expected mask");
        assert_eq!(m.get(2), false);
    }

    #[test]
    fn arr_f64_literal_elements_with_mask() {
        let m = Bitmask::from_bools(&[true, false, true]);
        let (data, mask) = expect_f64_mask(arr_f64!(1.0, 2.0, 3.0 ; m));
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
        assert_eq!(mask.expect("expected mask").get(1), false);
    }

    #[test]
    fn arr_f64_literal_array_ref_with_mask() {
        let m = Bitmask::from_bools(&[true, false]);
        let (data, mask) = expect_f64_mask(arr_f64!(&[1.0, 2.0]; m));
        assert_eq!(data, vec![1.0, 2.0]);
        assert_eq!(mask.expect("expected mask").get(1), false);
    }

    #[test]
    fn arr_i32_slice_and_mask() {
        let s: &[i32] = &[10, 20, 30];
        let m = Bitmask::from_bools(&[true, false, true]);
        let (data, mask) = expect_int32_mask(arr_i32!(s; m));
        assert_eq!(data, vec![10, 20, 30]);
        assert_eq!(mask.expect("expected mask").get(1), false);
    }

    #[test]
    fn arr_bool_slice_and_mask() {
        let s: &[bool] = &[true, false, true];
        let m = Bitmask::from_bools(&[true, true, false]);
        let (data, mask) = expect_bool_mask(arr_bool!(s; m));
        assert_eq!(data, vec![true, false, true]);
        assert_eq!(mask.expect("expected mask").get(2), false);
    }

    #[test]
    fn arr_str32_slice_and_mask() {
        let s: &[&str] = &["alpha", "beta", "gamma"];
        let m = Bitmask::from_bools(&[true, false, true]);
        let arr = arr_str32!(s; m);
        match arr {
            Array::TextArray(TextArray::String32(a)) => {
                assert_eq!(a.get_str(0), Some("alpha"));
                assert_eq!(a.get_str(2), Some("gamma"));
                let mask = a.null_mask.as_ref().expect("expected mask");
                assert_eq!(mask.get(1), false);
            }
            _ => panic!("expected String32 Array"),
        }
    }

    #[cfg(any(feature = "default_categorical_8", not(feature = "default_categorical_8")))]
    #[test]
    fn arr_cat32_slice_and_mask() {
        let s: &[&str] = &["red", "green", "red"];
        let m = Bitmask::from_bools(&[true, false, true]);
        let arr = arr_cat32!(s; m);
        match arr {
            Array::TextArray(TextArray::Categorical32(a)) => {
                assert_eq!(a.get_str(0), Some("red"));
                let mask = a.null_mask.as_ref().expect("expected mask");
                assert_eq!(mask.get(1), false);
            }
            _ => panic!("expected Categorical32 Array"),
        }
    }
}
