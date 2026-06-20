// Copyright 2025-2026 Peter Garfield Bower. All Rights Reserved.
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

//! # **Scalar Module** - *Single Value Container*
//!
//! Contains the Scalar type for holding a single value.
//!
//! ## Purpose
//! - Supports numeric, text, temporal, and null variants.  
//! - Used for unifying type signatures and other cases when one would like
//! to match to one of a range of possible values.

use std::convert::From;

#[cfg(feature = "scalar_type")]
use num_traits::Pow;

#[cfg(feature = "datetime")]
#[cfg(feature = "scalar_type")]
use crate::DatetimeArray;
#[cfg(feature = "scalar_type")]
use crate::{Array, Bitmask, BooleanArray, FloatArray, IntegerArray, MaskedArray, StringArray};

/// # Scalar
///
/// Scalar literals (single values) covering all supported types.
///
/// ## Description
/// - Useful when unifying type signatures.
/// - Includes accessor methods to avoid needing to match to a known type.
///   These also downcast to that type, including for e.g., string operations.
/// - There are also `try_<type>` methods that can be used to attempt it gracefully
/// without the risk of panicking.
#[cfg(feature = "scalar_type")]
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Scalar {
    #[default]
    Null,
    Boolean(bool),
    // Signed integers
    #[cfg(feature = "extended_numeric_types")]
    Int8(i8),
    #[cfg(feature = "extended_numeric_types")]
    Int16(i16),
    Int32(i32),
    Int64(i64),
    // Unsigned integers
    #[cfg(feature = "extended_numeric_types")]
    UInt8(u8),
    #[cfg(feature = "extended_numeric_types")]
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    // Floats
    Float32(f32),
    Float64(f64),
    // String strings
    String32(String),
    #[cfg(feature = "large_string")]
    String64(String),
    #[cfg(feature = "datetime")]
    Datetime32(i32),
    #[cfg(feature = "datetime")]
    Datetime64(i64),
    #[cfg(feature = "datetime")]
    Interval,
}

#[cfg(feature = "scalar_type")]
impl Scalar {
    /// Casts the value to a bool
    ///
    /// # Behaviour:
    ///  - any non-zero value becomes True
    ///  - strings convert `true`, `t`, `1`, `false`, `f`, and `0` to bool
    /// on a case insensitive basis
    /// - Unsuccessful casts return None
    ///
    /// # Safety
    /// - Panics on failure.
    /// - Consider the try variant for a safe alternative
    ///
    #[inline]
    pub fn bool(&self) -> bool {
        match self {
            Scalar::Boolean(v) => *v,
            Scalar::Int32(v) => *v != 0,
            Scalar::Int64(v) => *v != 0,
            Scalar::UInt32(v) => *v != 0,
            Scalar::UInt64(v) => *v != 0,
            Scalar::Float32(v) => *v != 0.0,
            Scalar::Float64(v) => *v != 0.0,
            Scalar::Null => panic!("Cannot convert Null to bool"),
            Scalar::String32(s) => {
                let s = s.trim();
                if s.is_empty() {
                    panic!("Cannot convert empty string to bool")
                } else if s.eq_ignore_ascii_case("true") || s.eq_ignore_ascii_case("t") || s == "1"
                {
                    true
                } else if s.eq_ignore_ascii_case("false") || s.eq_ignore_ascii_case("f") || s == "0"
                {
                    false
                } else {
                    panic!("Cannot convert string '{s}' to bool")
                }
            }
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => {
                let s = s.trim();
                if s.is_empty() {
                    panic!("Cannot convert empty string to bool")
                } else if s.eq_ignore_ascii_case("true") || s.eq_ignore_ascii_case("t") || s == "1"
                {
                    true
                } else if s.eq_ignore_ascii_case("false") || s.eq_ignore_ascii_case("f") || s == "0"
                {
                    false
                } else {
                    panic!("Cannot convert string '{s}' to bool")
                }
            }
            #[cfg(feature = "datetime")]
            Scalar::Datetime32(v) => *v != 0,
            #[cfg(feature = "datetime")]
            Scalar::Datetime64(v) => *v != 0,
            #[cfg(feature = "datetime")]
            Scalar::Interval => panic!("Cannot convert Interval to bool"),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int8(v) => *v != 0,
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int16(v) => *v != 0,
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt8(v) => *v != 0,
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt16(v) => *v != 0,
        }
    }

    /// Converts the scalar to an `i8` value
    ///
    /// - Panics on failure.
    /// - Consider the try variant for a safe alternative
    #[cfg(feature = "extended_numeric_types")]
    #[inline]
    pub fn i8(&self) -> i8 {
        match self {
            Scalar::Int8(v) => *v,
            Scalar::Int16(v) => i8::try_from(*v).expect("i16 out of range for i8"),
            Scalar::Int32(v) => i8::try_from(*v).expect("i32 out of range for i8"),
            Scalar::Int64(v) => i8::try_from(*v).expect("i64 out of range for i8"),
            Scalar::UInt8(v) => i8::try_from(*v).expect("u8 out of range for i8"),
            Scalar::UInt16(v) => i8::try_from(*v).expect("u16 out of range for i8"),
            Scalar::UInt32(v) => i8::try_from(*v).expect("u32 out of range for i8"),
            Scalar::UInt64(v) => i8::try_from(*v).expect("u64 out of range for i8"),
            Scalar::Float32(v) => i8::try_from(*v as i32).expect("f32 out of range for i8"),
            Scalar::Float64(v) => i8::try_from(*v as i32).expect("f64 out of range for i8"),
            Scalar::Null => panic!("Cannot convert Null to i8"),
            Scalar::String32(s) => s.parse::<i8>().expect("Cannot parse string as i8"),
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => s.parse::<i8>().expect("Cannot parse string as i8"),
            #[cfg(feature = "datetime")]
            Scalar::Datetime32(v) => i8::try_from(*v).expect("u32 out of range for i8"),
            #[cfg(feature = "datetime")]
            Scalar::Datetime64(v) => i8::try_from(*v).expect("u64 out of range for i8"),
            #[cfg(feature = "datetime")]
            Scalar::Interval => panic!("Cannot convert Interval to i8"),
            Scalar::Boolean(b) => {
                if *b {
                    1
                } else {
                    0
                }
            }
        }
    }

    /// Converts the scalar to an `i16` value
    ///
    /// - Panics on failure.
    /// - Consider the try variant for a safe alternative
    #[cfg(feature = "extended_numeric_types")]
    #[inline]
    pub fn i16(&self) -> i16 {
        match self {
            Scalar::Int8(v) => *v as i16,
            Scalar::Int16(v) => *v,
            Scalar::Int32(v) => i16::try_from(*v).expect("i32 out of range for i16"),
            Scalar::Int64(v) => i16::try_from(*v).expect("i64 out of range for i16"),
            Scalar::UInt8(v) => *v as i16,
            Scalar::UInt16(v) => i16::try_from(*v).expect("u16 out of range for i16"),
            Scalar::UInt32(v) => i16::try_from(*v).expect("u32 out of range for i16"),
            Scalar::UInt64(v) => i16::try_from(*v).expect("u64 out of range for i16"),
            Scalar::Float32(v) => i16::try_from(*v as i32).expect("f32 out of range for i16"),
            Scalar::Float64(v) => i16::try_from(*v as i32).expect("f64 out of range for i16"),
            Scalar::Null => panic!("Cannot convert Null to i16"),
            Scalar::String32(s) => s.parse::<i16>().expect("Cannot parse string as i16"),
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => s.parse::<i16>().expect("Cannot parse string as i16"),
            #[cfg(feature = "datetime")]
            Scalar::Datetime32(v) => i16::try_from(*v).expect("u32 out of range for i16"),
            #[cfg(feature = "datetime")]
            Scalar::Datetime64(v) => i16::try_from(*v).expect("u64 out of range for i16"),
            #[cfg(feature = "datetime")]
            Scalar::Interval => panic!("Cannot convert Interval to i16"),
            Scalar::Boolean(b) => {
                if *b {
                    1
                } else {
                    0
                }
            }
        }
    }

    /// Converts the scalar to an `i32` value
    ///
    /// - Panics on failure.
    /// - Consider the try variant for a safe alternative
    #[inline]
    pub fn i32(&self) -> i32 {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int8(v) => *v as i32,
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int16(v) => *v as i32,
            Scalar::Int32(v) => *v,
            Scalar::Int64(v) => i32::try_from(*v).expect("i64 out of range for i32"),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt8(v) => *v as i32,
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt16(v) => *v as i32,
            Scalar::UInt32(v) => i32::try_from(*v).expect("u32 out of range for i32"),
            Scalar::UInt64(v) => i32::try_from(*v).expect("u64 out of range for i32"),
            Scalar::Float32(v) => *v as i32,
            Scalar::Float64(v) => *v as i32,
            Scalar::Null => panic!("Cannot convert Null to i32"),
            Scalar::String32(s) => s.parse::<i32>().expect("Cannot parse string as i32"),
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => s.parse::<i32>().expect("Cannot parse string as i32"),
            #[cfg(feature = "datetime")]
            Scalar::Datetime32(v) => *v,
            #[cfg(feature = "datetime")]
            Scalar::Datetime64(v) => i32::try_from(*v).expect("u64 out of range for i32"),
            #[cfg(feature = "datetime")]
            Scalar::Interval => panic!("Cannot convert Interval to i32"),
            Scalar::Boolean(b) => {
                if *b {
                    1
                } else {
                    0
                }
            }
        }
    }

    /// Converts the scalar to an `i64` value
    ///
    /// - Panics on failure.
    /// - Consider the try variant for a safe alternative
    #[inline]
    pub fn i64(&self) -> i64 {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int8(v) => *v as i64,
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int16(v) => *v as i64,
            Scalar::Int32(v) => *v as i64,
            Scalar::Int64(v) => *v,
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt8(v) => *v as i64,
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt16(v) => *v as i64,
            Scalar::UInt32(v) => *v as i64,
            Scalar::UInt64(v) => {
                if *v <= i64::MAX as u64 {
                    *v as i64
                } else {
                    panic!("u64 out of range for i64")
                }
            }
            Scalar::Float32(v) => *v as i64,
            Scalar::Float64(v) => *v as i64,
            Scalar::Null => panic!("Cannot convert Null to i64"),
            Scalar::String32(s) => s.parse::<i64>().expect("Cannot parse string as i64"),
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => s.parse::<i64>().expect("Cannot parse string as i64"),
            #[cfg(feature = "datetime")]
            Scalar::Datetime32(v) => *v as i64,
            #[cfg(feature = "datetime")]
            Scalar::Datetime64(v) => i64::try_from(*v).expect("u64 out of range for i64"),
            #[cfg(feature = "datetime")]
            Scalar::Interval => panic!("Cannot convert Interval to i64"),
            Scalar::Boolean(b) => {
                if *b {
                    1
                } else {
                    0
                }
            }
        }
    }

    /// Converts the scalar to an `u8` value
    ///
    /// - Panics on failure.
    /// - Consider the try variant for a safe alternative
    #[inline]
    pub fn u8(&self) -> u8 {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int8(v) => u8::try_from(*v).expect("i8 out of range for u8"),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int16(v) => u8::try_from(*v).expect("i16 out of range for u8"),
            Scalar::Int32(v) => u8::try_from(*v).expect("i32 out of range for u8"),
            Scalar::Int64(v) => u8::try_from(*v).expect("i64 out of range for u8"),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt8(v) => *v,
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt16(v) => u8::try_from(*v).expect("u16 out of range for u8"),
            Scalar::UInt32(v) => u8::try_from(*v).expect("u32 out of range for u8"),
            Scalar::UInt64(v) => u8::try_from(*v).expect("u64 out of range for u8"),
            Scalar::Float32(v) => u8::try_from(*v as i32).expect("f32 out of range for u8"),
            Scalar::Float64(v) => u8::try_from(*v as i32).expect("f64 out of range for u8"),
            Scalar::Null => panic!("Cannot convert Null to u8"),
            Scalar::String32(s) => s.parse::<u8>().expect("Cannot parse string as u8"),
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => s.parse::<u8>().expect("Cannot parse string as u8"),
            #[cfg(feature = "datetime")]
            Scalar::Datetime32(v) => u8::try_from(*v).expect("u32 out of range for u8"),
            #[cfg(feature = "datetime")]
            Scalar::Datetime64(v) => u8::try_from(*v).expect("u64 out of range for u8"),
            #[cfg(feature = "datetime")]
            Scalar::Interval => panic!("Cannot convert Interval to u8"),
            Scalar::Boolean(b) => {
                if *b {
                    1
                } else {
                    0
                }
            }
        }
    }

    /// Converts the scalar to an `u16` value
    ///
    /// - Panics on failure.
    /// - Consider the try variant for a safe alternative
    #[inline]
    pub fn u16(&self) -> u16 {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int8(v) => u16::try_from(*v).expect("i8 out of range for u16"),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int16(v) => u16::try_from(*v).expect("i16 out of range for u16"),
            Scalar::Int32(v) => u16::try_from(*v).expect("i32 out of range for u16"),
            Scalar::Int64(v) => u16::try_from(*v).expect("i64 out of range for u16"),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt8(v) => *v as u16,
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt16(v) => *v,
            Scalar::UInt32(v) => u16::try_from(*v).expect("u32 out of range for u16"),
            Scalar::UInt64(v) => u16::try_from(*v).expect("u64 out of range for u16"),
            Scalar::Float32(v) => u16::try_from(*v as i32).expect("f32 out of range for u16"),
            Scalar::Float64(v) => u16::try_from(*v as i32).expect("f64 out of range for u16"),
            Scalar::Null => panic!("Cannot convert Null to u16"),
            Scalar::String32(s) => s.parse::<u16>().expect("Cannot parse string as u16"),
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => s.parse::<u16>().expect("Cannot parse string as u16"),
            #[cfg(feature = "datetime")]
            Scalar::Datetime32(v) => u16::try_from(*v).expect("u32 out of range for u16"),
            #[cfg(feature = "datetime")]
            Scalar::Datetime64(v) => u16::try_from(*v).expect("u64 out of range for u16"),
            #[cfg(feature = "datetime")]
            Scalar::Interval => panic!("Cannot convert Interval to u16"),
            Scalar::Boolean(b) => {
                if *b {
                    1
                } else {
                    0
                }
            }
        }
    }

    /// Converts the scalar to an `u32` value
    ///
    /// - Panics on failure.
    /// - Consider the try variant for a safe alternative
    #[inline]
    pub fn u32(&self) -> u32 {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int8(v) => u32::try_from(*v).expect("i8 out of range for u32"),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int16(v) => u32::try_from(*v).expect("i16 out of range for u32"),
            Scalar::Int32(v) => u32::try_from(*v).expect("i32 out of range for u32"),
            Scalar::Int64(v) => u32::try_from(*v).expect("i64 out of range for u32"),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt8(v) => *v as u32,
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt16(v) => *v as u32,
            Scalar::UInt32(v) => *v,
            Scalar::UInt64(v) => u32::try_from(*v).expect("u64 out of range for u32"),
            Scalar::Float32(v) => *v as u32,
            Scalar::Float64(v) => *v as u32,
            Scalar::Null => panic!("Cannot convert Null to u32"),
            Scalar::String32(s) => s.parse::<u32>().expect("Cannot parse string as u32"),
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => s.parse::<u32>().expect("Cannot parse string as u32"),
            #[cfg(feature = "datetime")]
            Scalar::Datetime32(v) => u32::try_from(*v).expect("u32 out of range for i32"),
            #[cfg(feature = "datetime")]
            Scalar::Datetime64(v) => u32::try_from(*v).expect("u64 out of range for i32"),
            #[cfg(feature = "datetime")]
            Scalar::Interval => panic!("Cannot convert Interval to u32"),
            Scalar::Boolean(b) => {
                if *b {
                    1
                } else {
                    0
                }
            }
        }
    }

    /// Converts the scalar to an `u64` value
    ///
    /// - Panics on failure.
    /// - Consider the try variant for a safe alternative
    #[inline]
    pub fn u64(&self) -> u64 {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int8(v) => {
                if *v >= 0 {
                    *v as u64
                } else {
                    panic!("i8 out of range for u64")
                }
            }
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int16(v) => {
                if *v >= 0 {
                    *v as u64
                } else {
                    panic!("i16 out of range for u64")
                }
            }
            Scalar::Int32(v) => {
                if *v >= 0 {
                    *v as u64
                } else {
                    panic!("i32 out of range for u64")
                }
            }
            Scalar::Int64(v) => {
                if *v >= 0 {
                    *v as u64
                } else {
                    panic!("i64 out of range for u64")
                }
            }
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt8(v) => *v as u64,
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt16(v) => *v as u64,
            Scalar::UInt32(v) => *v as u64,
            Scalar::UInt64(v) => *v,
            Scalar::Float32(v) => {
                if *v >= 0.0 {
                    *v as u64
                } else {
                    panic!("f32 out of range for u64")
                }
            }
            Scalar::Float64(v) => {
                if *v >= 0.0 {
                    *v as u64
                } else {
                    panic!("f64 out of range for u64")
                }
            }
            Scalar::Null => panic!("Cannot convert Null to u64"),
            Scalar::String32(s) => s.parse::<u64>().expect("Cannot parse string as u64"),
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => s.parse::<u64>().expect("Cannot parse string as u64"),
            #[cfg(feature = "datetime")]
            Scalar::Datetime32(v) => *v as u64,
            #[cfg(feature = "datetime")]
            Scalar::Datetime64(v) => *v as u64,
            #[cfg(feature = "datetime")]
            Scalar::Interval => panic!("Cannot convert Interval to u64"),
            Scalar::Boolean(b) => {
                if *b {
                    1
                } else {
                    0
                }
            }
        }
    }

    /// Converts the scalar to an `f32` value
    ///
    /// - Panics on failure.
    /// - Consider the try variant for a safe alternative
    #[inline]
    pub fn f32(&self) -> f32 {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int8(v) => *v as f32,
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int16(v) => *v as f32,
            Scalar::Int32(v) => *v as f32,
            Scalar::Int64(v) => *v as f32,
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt8(v) => *v as f32,
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt16(v) => *v as f32,
            Scalar::UInt32(v) => *v as f32,
            Scalar::UInt64(v) => *v as f32,
            Scalar::Float32(v) => *v,
            Scalar::Float64(v) => *v as f32,
            Scalar::Boolean(v) => {
                if *v {
                    1.0
                } else {
                    0.0
                }
            }
            Scalar::Null => panic!("Cannot convert Null to f32"),
            Scalar::String32(s) => s.parse::<f32>().expect("Cannot parse string as f32"),
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => s.parse::<f32>().expect("Cannot parse string as f32"),
            #[cfg(feature = "datetime")]
            Scalar::Datetime32(v) => *v as f32,
            #[cfg(feature = "datetime")]
            Scalar::Datetime64(v) => *v as f32,
            #[cfg(feature = "datetime")]
            Scalar::Interval => panic!("Cannot convert Interval to f32"),
        }
    }

    /// Converts the scalar to a `float` value
    ///
    /// - Panics on failure.
    /// - Consider the try variant for a safe alternative
    #[inline]
    pub fn f64(&self) -> f64 {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int8(v) => *v as f64,
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int16(v) => *v as f64,
            Scalar::Int32(v) => *v as f64,
            Scalar::Int64(v) => *v as f64,
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt8(v) => *v as f64,
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt16(v) => *v as f64,
            Scalar::UInt32(v) => *v as f64,
            Scalar::UInt64(v) => *v as f64,
            Scalar::Float32(v) => *v as f64,
            Scalar::Float64(v) => *v,
            Scalar::Boolean(v) => {
                if *v {
                    1.0
                } else {
                    0.0
                }
            }
            Scalar::Null => panic!("Cannot convert Null to f64"),
            Scalar::String32(s) => s.parse::<f64>().expect("Cannot parse string as f64"),
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => s.parse::<f64>().expect("Cannot parse string as f64"),
            #[cfg(feature = "datetime")]
            Scalar::Datetime32(v) => *v as f64,
            #[cfg(feature = "datetime")]
            Scalar::Datetime64(v) => *v as f64,
            #[cfg(feature = "datetime")]
            Scalar::Interval => panic!("Cannot convert Interval to f64"),
        }
    }

    /// Converts the scalar to a `String` value
    ///
    /// - Panics on failure.
    /// - Consider the try variant for a safe alternative
    #[inline]
    pub fn str(&self) -> String {
        match self {
            Scalar::String32(s) => s.clone(),
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => s.clone(),
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
            Scalar::Null => panic!("Cannot convert Null to String"),
            #[cfg(feature = "datetime")]
            Scalar::Datetime32(v) => v.to_string(),
            #[cfg(feature = "datetime")]
            Scalar::Datetime64(v) => v.to_string(),
            #[cfg(feature = "datetime")]
            Scalar::Interval => panic!("Cannot convert Interval to String"),
        }
    }

    /// Converts the scalar to an `u32` representing time since epoch
    ///
    /// - Panics on failure.
    /// - Consider the try variant for a safe alternative
    #[cfg(feature = "datetime")]
    #[inline]
    pub fn dt32(&self) -> u32 {
        match self {
            Scalar::Datetime32(v) => {
                if *v >= 0 {
                    *v as u32
                } else {
                    panic!("Scalar::Datetime32's i32 out of range for dt32 (negative value)")
                }
            }
            Scalar::UInt32(v) => *v,
            Scalar::Int32(v) => {
                if *v >= 0 {
                    *v as u32
                } else {
                    panic!("i32 out of range for dt32 (negative value)")
                }
            }
            Scalar::Null => panic!("Cannot convert Null to dt32"),
            Scalar::Boolean(b) => {
                if *b {
                    1
                } else {
                    0
                }
            }
            Scalar::Int64(v) => {
                if *v >= 0 && *v <= u32::MAX as i64 {
                    *v as u32
                } else {
                    panic!("i64 out of range for dt32")
                }
            }
            Scalar::UInt64(v) => {
                if *v <= u32::MAX as u64 {
                    *v as u32
                } else {
                    panic!("u64 out of range for dt32")
                }
            }
            Scalar::Float32(v) => {
                if *v >= 0.0 && *v <= u32::MAX as f32 {
                    *v as u32
                } else {
                    panic!("f32 out of range for dt32")
                }
            }
            Scalar::Float64(v) => {
                if *v >= 0.0 && *v <= u32::MAX as f64 {
                    *v as u32
                } else {
                    panic!("f64 out of range for dt32")
                }
            }
            Scalar::String32(s) => s.parse::<u32>().expect("Cannot parse string as dt32"),
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => s.parse::<u32>().expect("Cannot parse string as dt32"),
            Scalar::Datetime64(v) => {
                if *v <= u32::MAX as i64 {
                    *v as u32
                } else {
                    panic!("Datetime64 out of range for dt32")
                }
            }
            Scalar::Interval => panic!("Cannot convert Interval to dt32"),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int8(v) => {
                if *v >= 0 {
                    *v as u32
                } else {
                    panic!("i8 out of range for dt32 (negative value)")
                }
            }
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int16(v) => {
                if *v >= 0 {
                    *v as u32
                } else {
                    panic!("i16 out of range for dt32 (negative value)")
                }
            }
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt8(v) => *v as u32,
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt16(v) => *v as u32,
        }
    }

    /// Converts the scalar to an `u64` value
    ///
    /// - Panics on failure.
    /// - Consider the try variant for a safe alternative
    #[cfg(feature = "datetime")]
    #[inline]
    pub fn dt64(&self) -> u64 {
        match self {
            Scalar::Datetime64(v) => {
                if *v >= 0 {
                    *v as u64
                } else {
                    panic!("Scalar::Datetime64's i64 out of range for dt64 (negative value)")
                }
            }
            Scalar::UInt64(v) => *v,
            Scalar::Int64(v) => {
                if *v >= 0 {
                    *v as u64
                } else {
                    panic!("i64 out of range for dt64 (negative value)")
                }
            }
            Scalar::Null => panic!("Cannot convert Null to dt64"),
            Scalar::Boolean(b) => {
                if *b {
                    1
                } else {
                    0
                }
            }
            Scalar::Int32(v) => {
                if *v >= 0 {
                    *v as u64
                } else {
                    panic!("i32 out of range for dt64 (negative value)")
                }
            }
            Scalar::UInt32(v) => *v as u64,
            Scalar::Float32(v) => {
                if *v >= 0.0 {
                    *v as u64
                } else {
                    panic!("f32 out of range for dt64")
                }
            }
            Scalar::Float64(v) => {
                if *v >= 0.0 {
                    *v as u64
                } else {
                    panic!("f64 out of range for dt64")
                }
            }
            Scalar::String32(s) => s.parse::<u64>().expect("Cannot parse string as dt64"),
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => s.parse::<u64>().expect("Cannot parse string as dt64"),
            Scalar::Datetime32(v) => *v as u64,
            Scalar::Interval => panic!("Cannot convert Interval to dt64"),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int8(v) => {
                if *v >= 0 {
                    *v as u64
                } else {
                    panic!("i8 out of range for dt64 (negative value)")
                }
            }
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int16(v) => {
                if *v >= 0 {
                    *v as u64
                } else {
                    panic!("i16 out of range for dt64 (negative value)")
                }
            }
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt8(v) => *v as u64,
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt16(v) => *v as u64,
        }
    }

    /// Tries to convert the scalar to an `bool` value
    #[inline]
    pub fn try_bool(&self) -> Option<bool> {
        match self {
            Scalar::Boolean(v) => Some(*v),
            Scalar::Int32(v) => Some(*v != 0),
            Scalar::Int64(v) => Some(*v != 0),
            Scalar::UInt32(v) => Some(*v != 0),
            Scalar::UInt64(v) => Some(*v != 0),
            Scalar::Float32(v) => Some(*v != 0.0),
            Scalar::Float64(v) => Some(*v != 0.0),
            Scalar::Null => None,
            Scalar::String32(s) => {
                let s = s.trim();
                if s.is_empty() {
                    None
                } else if s.eq_ignore_ascii_case("true") || s.eq_ignore_ascii_case("t") || s == "1"
                {
                    Some(true)
                } else if s.eq_ignore_ascii_case("false") || s.eq_ignore_ascii_case("f") || s == "0"
                {
                    Some(false)
                } else {
                    None
                }
            }
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => {
                let s = s.trim();
                if s.is_empty() {
                    None
                } else if s.eq_ignore_ascii_case("true") || s.eq_ignore_ascii_case("t") || s == "1"
                {
                    Some(true)
                } else if s.eq_ignore_ascii_case("false") || s.eq_ignore_ascii_case("f") || s == "0"
                {
                    Some(false)
                } else {
                    None
                }
            }
            #[cfg(feature = "datetime")]
            Scalar::Datetime32(v) => Some(*v != 0),
            #[cfg(feature = "datetime")]
            Scalar::Datetime64(v) => Some(*v != 0),
            #[cfg(feature = "datetime")]
            Scalar::Interval => None,
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int8(v) => Some(*v != 0),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int16(v) => Some(*v != 0),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt8(v) => Some(*v != 0),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt16(v) => Some(*v != 0),
        }
    }

    /// Tries to convert the scalar to an `i8` value
    #[cfg(feature = "extended_numeric_types")]
    #[inline]
    pub fn try_i8(&self) -> Option<i8> {
        match self {
            Scalar::Int8(v) => Some(*v),
            Scalar::Int16(v) => i8::try_from(*v).ok(),
            Scalar::Int32(v) => i8::try_from(*v).ok(),
            Scalar::Int64(v) => i8::try_from(*v).ok(),
            Scalar::UInt8(v) => i8::try_from(*v).ok(),
            Scalar::UInt16(v) => i8::try_from(*v).ok(),
            Scalar::UInt32(v) => i8::try_from(*v).ok(),
            Scalar::UInt64(v) => i8::try_from(*v).ok(),
            Scalar::Float32(v) => i8::try_from(*v as i32).ok(),
            Scalar::Float64(v) => i8::try_from(*v as i32).ok(),
            Scalar::Null => None,
            Scalar::String32(s) => s.parse::<i8>().ok(),
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => s.parse::<i8>().ok(),
            #[cfg(feature = "datetime")]
            Scalar::Datetime32(v) => i8::try_from(*v).ok(),
            #[cfg(feature = "datetime")]
            Scalar::Datetime64(v) => i8::try_from(*v).ok(),
            #[cfg(feature = "datetime")]
            Scalar::Interval => None,
            Scalar::Boolean(b) => Some(if *b { 1 } else { 0 }),
        }
    }

    /// Tries to convert the scalar to an `i16` value
    #[cfg(feature = "extended_numeric_types")]
    #[inline]
    pub fn try_i16(&self) -> Option<i16> {
        match self {
            Scalar::Int8(v) => Some(*v as i16),
            Scalar::Int16(v) => Some(*v),
            Scalar::Int32(v) => i16::try_from(*v).ok(),
            Scalar::Int64(v) => i16::try_from(*v).ok(),
            Scalar::UInt8(v) => Some(*v as i16),
            Scalar::UInt16(v) => i16::try_from(*v).ok(),
            Scalar::UInt32(v) => i16::try_from(*v).ok(),
            Scalar::UInt64(v) => i16::try_from(*v).ok(),
            Scalar::Float32(v) => i16::try_from(*v as i32).ok(),
            Scalar::Float64(v) => i16::try_from(*v as i32).ok(),
            Scalar::Null => None,
            Scalar::String32(s) => s.parse::<i16>().ok(),
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => s.parse::<i16>().ok(),
            #[cfg(feature = "datetime")]
            Scalar::Datetime32(v) => i16::try_from(*v).ok(),
            #[cfg(feature = "datetime")]
            Scalar::Datetime64(v) => i16::try_from(*v).ok(),
            #[cfg(feature = "datetime")]
            Scalar::Interval => None,
            Scalar::Boolean(b) => Some(if *b { 1 } else { 0 }),
        }
    }

    /// Tries to convert the scalar to an `i32` value
    #[inline]
    pub fn try_i32(&self) -> Option<i32> {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int8(v) => Some(*v as i32),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int16(v) => Some(*v as i32),
            Scalar::Int32(v) => Some(*v),
            Scalar::Int64(v) => i32::try_from(*v).ok(),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt8(v) => Some(*v as i32),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt16(v) => Some(*v as i32),
            Scalar::UInt32(v) => i32::try_from(*v).ok(),
            Scalar::UInt64(v) => i32::try_from(*v).ok(),
            Scalar::Float32(v) => Some(*v as i32),
            Scalar::Float64(v) => Some(*v as i32),
            Scalar::Null => None,
            Scalar::String32(s) => s.parse::<i32>().ok(),
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => s.parse::<i32>().ok(),
            #[cfg(feature = "datetime")]
            Scalar::Datetime32(v) => i32::try_from(*v).ok(),
            #[cfg(feature = "datetime")]
            Scalar::Datetime64(v) => i32::try_from(*v).ok(),
            #[cfg(feature = "datetime")]
            Scalar::Interval => None,
            Scalar::Boolean(b) => Some(if *b { 1 } else { 0 }),
        }
    }

    /// Tries to convert the scalar to an `i64` value
    #[inline]
    pub fn try_i64(&self) -> Option<i64> {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int8(v) => Some(*v as i64),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int16(v) => Some(*v as i64),
            Scalar::Int32(v) => Some(*v as i64),
            Scalar::Int64(v) => Some(*v),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt8(v) => Some(*v as i64),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt16(v) => Some(*v as i64),
            Scalar::UInt32(v) => Some(*v as i64),
            Scalar::UInt64(v) => {
                if *v <= i64::MAX as u64 {
                    Some(*v as i64)
                } else {
                    None
                }
            }
            Scalar::Float32(v) => Some(*v as i64),
            Scalar::Float64(v) => Some(*v as i64),
            Scalar::Null => None,
            Scalar::String32(s) => s.parse::<i64>().ok(),
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => s.parse::<i64>().ok(),
            #[cfg(feature = "datetime")]
            Scalar::Datetime32(v) => Some(*v as i64),
            #[cfg(feature = "datetime")]
            Scalar::Datetime64(v) => i64::try_from(*v).ok(),
            #[cfg(feature = "datetime")]
            Scalar::Interval => None,
            Scalar::Boolean(b) => Some(if *b { 1 } else { 0 }),
        }
    }

    /// Tries to convert the scalar to an `u8` value
    #[inline]
    pub fn try_u8(&self) -> Option<u8> {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int8(v) => u8::try_from(*v).ok(),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int16(v) => u8::try_from(*v).ok(),
            Scalar::Int32(v) => u8::try_from(*v).ok(),
            Scalar::Int64(v) => u8::try_from(*v).ok(),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt8(v) => Some(*v),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt16(v) => u8::try_from(*v).ok(),
            Scalar::UInt32(v) => u8::try_from(*v).ok(),
            Scalar::UInt64(v) => u8::try_from(*v).ok(),
            Scalar::Float32(v) => u8::try_from(*v as i32).ok(),
            Scalar::Float64(v) => u8::try_from(*v as i32).ok(),
            Scalar::Null => None,
            Scalar::String32(s) => s.parse::<u8>().ok(),
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => s.parse::<u8>().ok(),
            #[cfg(feature = "datetime")]
            Scalar::Datetime32(v) => u8::try_from(*v).ok(),
            #[cfg(feature = "datetime")]
            Scalar::Datetime64(v) => u8::try_from(*v).ok(),
            #[cfg(feature = "datetime")]
            Scalar::Interval => None,
            Scalar::Boolean(b) => Some(if *b { 1 } else { 0 }),
        }
    }

    /// Tries to convert the scalar to an `u16` value
    #[inline]
    pub fn try_u16(&self) -> Option<u16> {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int8(v) => u16::try_from(*v).ok(),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int16(v) => u16::try_from(*v).ok(),
            Scalar::Int32(v) => u16::try_from(*v).ok(),
            Scalar::Int64(v) => u16::try_from(*v).ok(),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt8(v) => Some(*v as u16),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt16(v) => Some(*v),
            Scalar::UInt32(v) => u16::try_from(*v).ok(),
            Scalar::UInt64(v) => u16::try_from(*v).ok(),
            Scalar::Float32(v) => u16::try_from(*v as i32).ok(),
            Scalar::Float64(v) => u16::try_from(*v as i32).ok(),
            Scalar::Null => None,
            Scalar::String32(s) => s.parse::<u16>().ok(),
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => s.parse::<u16>().ok(),
            #[cfg(feature = "datetime")]
            Scalar::Datetime32(v) => u16::try_from(*v).ok(),
            #[cfg(feature = "datetime")]
            Scalar::Datetime64(v) => u16::try_from(*v).ok(),
            #[cfg(feature = "datetime")]
            Scalar::Interval => None,
            Scalar::Boolean(b) => Some(if *b { 1 } else { 0 }),
        }
    }

    /// Tries to convert the scalar to an `u32` value
    #[inline]
    pub fn try_u32(&self) -> Option<u32> {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int8(v) => u32::try_from(*v).ok(),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int16(v) => u32::try_from(*v).ok(),
            Scalar::Int32(v) => u32::try_from(*v).ok(),
            Scalar::Int64(v) => u32::try_from(*v).ok(),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt8(v) => Some(*v as u32),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt16(v) => Some(*v as u32),
            Scalar::UInt32(v) => Some(*v),
            Scalar::UInt64(v) => u32::try_from(*v).ok(),
            Scalar::Float32(v) => Some(*v as u32),
            Scalar::Float64(v) => Some(*v as u32),
            Scalar::Null => None,
            Scalar::String32(s) => s.parse::<u32>().ok(),
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => s.parse::<u32>().ok(),
            #[cfg(feature = "datetime")]
            Scalar::Datetime32(v) => u32::try_from(*v).ok(),
            #[cfg(feature = "datetime")]
            Scalar::Datetime64(v) => u32::try_from(*v).ok(),
            #[cfg(feature = "datetime")]
            Scalar::Interval => None,
            Scalar::Boolean(b) => Some(if *b { 1 } else { 0 }),
        }
    }

    /// Tries to convert the scalar to an `u64` value
    #[inline]
    pub fn try_u64(&self) -> Option<u64> {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int8(v) => {
                if *v >= 0 {
                    Some(*v as u64)
                } else {
                    None
                }
            }
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int16(v) => {
                if *v >= 0 {
                    Some(*v as u64)
                } else {
                    None
                }
            }
            Scalar::Int32(v) => {
                if *v >= 0 {
                    Some(*v as u64)
                } else {
                    None
                }
            }
            Scalar::Int64(v) => {
                if *v >= 0 {
                    Some(*v as u64)
                } else {
                    None
                }
            }
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt8(v) => Some(*v as u64),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt16(v) => Some(*v as u64),
            Scalar::UInt32(v) => Some(*v as u64),
            Scalar::UInt64(v) => Some(*v),
            Scalar::Float32(v) => {
                if *v >= 0.0 {
                    Some(*v as u64)
                } else {
                    None
                }
            }
            Scalar::Float64(v) => {
                if *v >= 0.0 {
                    Some(*v as u64)
                } else {
                    None
                }
            }
            Scalar::Null => None,
            Scalar::String32(s) => s.parse::<u64>().ok(),
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => s.parse::<u64>().ok(),
            #[cfg(feature = "datetime")]
            Scalar::Datetime32(v) => {
                if *v >= 0 {
                    Some(*v as u64)
                } else {
                    None
                }
            }
            #[cfg(feature = "datetime")]
            Scalar::Datetime64(v) => {
                if *v >= 0 {
                    Some(*v as u64)
                } else {
                    None
                }
            }
            #[cfg(feature = "datetime")]
            Scalar::Interval => None,
            Scalar::Boolean(b) => Some(if *b { 1 } else { 0 }),
        }
    }

    /// Tries to convert the scalar to an `f32` value
    #[inline]
    pub fn try_f32(&self) -> Option<f32> {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int8(v) => Some(*v as f32),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int16(v) => Some(*v as f32),
            Scalar::Int32(v) => Some(*v as f32),
            Scalar::Int64(v) => Some(*v as f32),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt8(v) => Some(*v as f32),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt16(v) => Some(*v as f32),
            Scalar::UInt32(v) => Some(*v as f32),
            Scalar::UInt64(v) => Some(*v as f32),
            Scalar::Float32(v) => Some(*v),
            Scalar::Float64(v) => Some(*v as f32),
            Scalar::Boolean(v) => Some(if *v { 1.0 } else { 0.0 }),
            Scalar::Null => None,
            Scalar::String32(s) => s.parse::<f32>().ok(),
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => s.parse::<f32>().ok(),
            #[cfg(feature = "datetime")]
            Scalar::Datetime32(v) => Some(*v as f32),
            #[cfg(feature = "datetime")]
            Scalar::Datetime64(v) => Some(*v as f32),
            #[cfg(feature = "datetime")]
            Scalar::Interval => None,
        }
    }

    /// Tries to convert the scalar to an `f64` value
    #[inline]
    pub fn try_f64(&self) -> Option<f64> {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int8(v) => Some(*v as f64),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int16(v) => Some(*v as f64),
            Scalar::Int32(v) => Some(*v as f64),
            Scalar::Int64(v) => Some(*v as f64),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt8(v) => Some(*v as f64),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt16(v) => Some(*v as f64),
            Scalar::UInt32(v) => Some(*v as f64),
            Scalar::UInt64(v) => Some(*v as f64),
            Scalar::Float32(v) => Some(*v as f64),
            Scalar::Float64(v) => Some(*v),
            Scalar::Boolean(v) => Some(if *v { 1.0 } else { 0.0 }),
            Scalar::Null => None,
            Scalar::String32(s) => s.parse::<f64>().ok(),
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => s.parse::<f64>().ok(),
            #[cfg(feature = "datetime")]
            Scalar::Datetime32(v) => Some(*v as f64),
            #[cfg(feature = "datetime")]
            Scalar::Datetime64(v) => Some(*v as f64),
            #[cfg(feature = "datetime")]
            Scalar::Interval => None,
        }
    }

    /// Tries to convert the scalar to a `String` value
    #[inline]
    pub fn try_str(&self) -> Option<String> {
        match self {
            Scalar::String32(s) => Some(s.clone()),
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => Some(s.clone()),
            Scalar::Boolean(v) => Some(v.to_string()),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int8(v) => Some(v.to_string()),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int16(v) => Some(v.to_string()),
            Scalar::Int32(v) => Some(v.to_string()),
            Scalar::Int64(v) => Some(v.to_string()),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt8(v) => Some(v.to_string()),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt16(v) => Some(v.to_string()),
            Scalar::UInt32(v) => Some(v.to_string()),
            Scalar::UInt64(v) => Some(v.to_string()),
            Scalar::Float32(v) => Some(v.to_string()),
            Scalar::Float64(v) => Some(v.to_string()),
            Scalar::Null => None,
            #[cfg(feature = "datetime")]
            Scalar::Datetime32(v) => Some(v.to_string()),
            #[cfg(feature = "datetime")]
            Scalar::Datetime64(v) => Some(v.to_string()),
            #[cfg(feature = "datetime")]
            Scalar::Interval => None,
        }
    }

    /// Tries to convert the scalar to a `u32` time since epoch value
    #[cfg(feature = "datetime")]
    #[inline]
    pub fn try_dt32(&self) -> Option<u32> {
        match self {
            Scalar::Datetime32(v) => {
                if *v >= 0 {
                    Some(*v as u32)
                } else {
                    None
                }
            }
            Scalar::UInt32(v) => Some(*v),
            Scalar::Int32(v) => {
                if *v >= 0 {
                    Some(*v as u32)
                } else {
                    None
                }
            }
            Scalar::Null => None,
            Scalar::Boolean(b) => Some(if *b { 1 } else { 0 }),
            Scalar::Int64(v) => {
                if *v >= 0 && *v <= u32::MAX as i64 {
                    Some(*v as u32)
                } else {
                    None
                }
            }
            Scalar::UInt64(v) => {
                if *v <= u32::MAX as u64 {
                    Some(*v as u32)
                } else {
                    None
                }
            }
            Scalar::Float32(v) => {
                if *v >= 0.0 && *v <= u32::MAX as f32 {
                    Some(*v as u32)
                } else {
                    None
                }
            }
            Scalar::Float64(v) => {
                if *v >= 0.0 && *v <= u32::MAX as f64 {
                    Some(*v as u32)
                } else {
                    None
                }
            }
            Scalar::String32(s) => s.parse::<u32>().ok(),
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => s.parse::<u32>().ok(),
            Scalar::Datetime64(v) => {
                if *v >= 0 && *v <= u32::MAX as i64 {
                    Some(*v as u32)
                } else {
                    None
                }
            }
            Scalar::Interval => None,
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int8(v) => {
                if *v >= 0 {
                    Some(*v as u32)
                } else {
                    None
                }
            }
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int16(v) => {
                if *v >= 0 {
                    Some(*v as u32)
                } else {
                    None
                }
            }
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt8(v) => Some(*v as u32),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt16(v) => Some(*v as u32),
        }
    }

    /// Tries to convert the scalar to a `u64` time since epoch value
    #[cfg(feature = "datetime")]
    #[inline]
    pub fn try_dt64(&self) -> Option<u64> {
        match self {
            Scalar::Datetime64(v) => {
                if *v >= 0 {
                    Some(*v as u64)
                } else {
                    None
                }
            }
            Scalar::UInt64(v) => Some(*v),
            Scalar::Int64(v) => {
                if *v >= 0 {
                    Some(*v as u64)
                } else {
                    None
                }
            }
            Scalar::Null => None,
            Scalar::Boolean(b) => Some(if *b { 1 } else { 0 }),
            Scalar::Int32(v) => {
                if *v >= 0 {
                    Some(*v as u64)
                } else {
                    None
                }
            }
            Scalar::UInt32(v) => Some(*v as u64),
            Scalar::Float32(v) => {
                if *v >= 0.0 {
                    Some(*v as u64)
                } else {
                    None
                }
            }
            Scalar::Float64(v) => {
                if *v >= 0.0 {
                    Some(*v as u64)
                } else {
                    None
                }
            }
            Scalar::String32(s) => s.parse::<u64>().ok(),
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => s.parse::<u64>().ok(),
            Scalar::Datetime32(v) => Some(*v as u64),
            Scalar::Interval => None,
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int8(v) => {
                if *v >= 0 {
                    Some(*v as u64)
                } else {
                    None
                }
            }
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int16(v) => {
                if *v >= 0 {
                    Some(*v as u64)
                } else {
                    None
                }
            }
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt8(v) => Some(*v as u64),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt16(v) => Some(*v as u64),
        }
    }

    /// Converts a scalar Value to an Array by repeating the scalar `len` times.
    pub fn array_from_value(self, len: usize) -> Array {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int8(v) => {
                let mut arr = IntegerArray::<i8>::with_capacity(len, false);
                for _ in 0..len {
                    arr.push(v);
                }
                Array::from_int8(arr)
            }
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int16(v) => {
                let mut arr = IntegerArray::<i16>::with_capacity(len, false);
                for _ in 0..len {
                    arr.push(v);
                }
                Array::from_int16(arr)
            }
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt8(v) => {
                let mut arr = IntegerArray::<u8>::with_capacity(len, false);
                for _ in 0..len {
                    arr.push(v);
                }
                Array::from_uint8(arr)
            }
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt16(v) => {
                let mut arr = IntegerArray::<u16>::with_capacity(len, false);
                for _ in 0..len {
                    arr.push(v);
                }
                Array::from_uint16(arr)
            }
            Scalar::Int32(v) => {
                let mut arr = IntegerArray::<i32>::with_capacity(len, false);
                for _ in 0..len {
                    arr.push(v);
                }
                Array::from_int32(arr)
            }
            Scalar::Int64(v) => {
                let mut arr = IntegerArray::<i64>::with_capacity(len, false);
                for _ in 0..len {
                    arr.push(v);
                }
                Array::from_int64(arr)
            }
            Scalar::UInt32(v) => {
                let mut arr = IntegerArray::<u32>::with_capacity(len, false);
                for _ in 0..len {
                    arr.push(v);
                }
                Array::from_uint32(arr)
            }
            Scalar::UInt64(v) => {
                let mut arr = IntegerArray::<u64>::with_capacity(len, false);
                for _ in 0..len {
                    arr.push(v);
                }
                Array::from_uint64(arr)
            }
            Scalar::Float32(v) => {
                let mut arr = FloatArray::<f32>::with_capacity(len, false);
                for _ in 0..len {
                    arr.push(v);
                }
                Array::from_float32(arr)
            }
            Scalar::Float64(v) => {
                let mut arr = FloatArray::<f64>::with_capacity(len, false);
                for _ in 0..len {
                    arr.push(v);
                }
                Array::from_float64(arr)
            }
            Scalar::Boolean(v) => {
                let mut arr = BooleanArray::with_capacity(len, false);
                for _ in 0..len {
                    arr.push(v);
                }
                Array::from_bool(arr)
            }
            Scalar::String32(s) => {
                let str_len = s.len();
                let mut arr = StringArray::with_capacity(len, len * str_len, false);
                for _ in 0..len {
                    arr.push_str(s.as_str());
                }
                Array::from_string32(arr)
            }
            #[cfg(feature = "large_string")]
            Scalar::String64(s) => {
                let str_len = s.len();
                let mut arr = StringArray::with_capacity(len, len * str_len, false);
                for _ in 0..len {
                    arr.push_str(s.as_str());
                }
                Array::from_string64(arr)
            }
            Scalar::Null => {
                // Allocate with null mask for len elements (all null)
                let arr = BooleanArray::new(
                    Bitmask::new_set_all(len, false),
                    Some(Bitmask::new_set_all(len, false)),
                );
                Array::from_bool(arr)
            }
            #[cfg(feature = "datetime")]
            Scalar::Datetime32(v) => {
                let mut arr = DatetimeArray::<i32>::with_capacity(len, false, None);
                for _ in 0..len {
                    arr.push(v.try_into().unwrap());
                }
                Array::from_datetime_i32(arr)
            }
            #[cfg(feature = "datetime")]
            Scalar::Datetime64(v) => {
                let mut arr = DatetimeArray::<i64>::with_capacity(len, false, None);
                for _ in 0..len {
                    arr.push(v.try_into().unwrap());
                }
                Array::from_datetime_i64(arr)
            }
            #[cfg(feature = "datetime")]
            Scalar::Interval => unimplemented!(),
        }
    }
}

#[cfg(feature = "hash")]
impl Eq for Scalar {}

#[cfg(feature = "hash")]
impl std::hash::Hash for Scalar {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Scalar::Null => {}
            Scalar::Boolean(v) => v.hash(state),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int8(v) => v.hash(state),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::Int16(v) => v.hash(state),
            Scalar::Int32(v) => v.hash(state),
            Scalar::Int64(v) => v.hash(state),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt8(v) => v.hash(state),
            #[cfg(feature = "extended_numeric_types")]
            Scalar::UInt16(v) => v.hash(state),
            Scalar::UInt32(v) => v.hash(state),
            Scalar::UInt64(v) => v.hash(state),
            Scalar::Float32(v) => v.to_bits().hash(state),
            Scalar::Float64(v) => v.to_bits().hash(state),
            Scalar::String32(v) => v.hash(state),
            #[cfg(feature = "large_string")]
            Scalar::String64(v) => v.hash(state),
            #[cfg(feature = "datetime")]
            Scalar::Datetime32(v) => v.hash(state),
            #[cfg(feature = "datetime")]
            Scalar::Datetime64(v) => v.hash(state),
            #[cfg(feature = "datetime")]
            Scalar::Interval => {}
        }
    }
}

#[cfg(feature = "scalar_type")]
macro_rules! impl_scalar_from {
    ($variant:ident: $($t:ty),+ $(,)?) => {
        $(
            impl From<$t> for Scalar {
                #[inline] fn from(v: $t) -> Self { Scalar::$variant(v) }
            }
        )+
    };
}

// boolean
#[cfg(feature = "scalar_type")]
impl_scalar_from!(Boolean: bool);

// signed ints
#[cfg(all(feature = "scalar_type", feature = "extended_numeric_types"))]
impl_scalar_from!(Int8:  i8);

#[cfg(all(feature = "scalar_type", feature = "extended_numeric_types"))]
impl_scalar_from!(Int16: i16);

#[cfg(feature = "scalar_type")]
impl_scalar_from!(Int32: i32);

#[cfg(feature = "scalar_type")]
impl_scalar_from!(Int64: i64);

// unsigned ints
#[cfg(all(feature = "scalar_type", feature = "extended_numeric_types"))]
impl_scalar_from!(UInt8:  u8);

#[cfg(all(feature = "scalar_type", feature = "extended_numeric_types"))]
impl_scalar_from!(UInt16: u16);

#[cfg(feature = "scalar_type")]
impl_scalar_from!(UInt32: u32);

#[cfg(feature = "scalar_type")]
impl_scalar_from!(UInt64: u64);

// floats
#[cfg(feature = "scalar_type")]
impl_scalar_from!(Float32: f32);

#[cfg(feature = "scalar_type")]
impl_scalar_from!(Float64: f64);

// string types
#[cfg(all(feature = "scalar_type", not(feature = "large_string")))]
impl From<String> for Scalar {
    #[inline]
    fn from(v: String) -> Self {
        Scalar::String32(v)
    }
}

#[cfg(all(feature = "scalar_type", feature = "large_string"))]
impl From<String> for Scalar {
    #[inline]
    fn from(v: String) -> Self {
        Scalar::String64(v)
    }
}

#[cfg(feature = "scalar_type")]
impl From<&str> for Scalar {
    #[inline]
    fn from(v: &str) -> Self {
        Scalar::String32(v.to_owned())
    }
}

#[cfg(feature = "scalar_type")]
use std::ops::{Add, Mul, Sub};

#[cfg(feature = "scalar_type")]
impl Add for Scalar {
    type Output = Scalar;

    fn add(self, rhs: Scalar) -> Scalar {
        use Scalar::*;

        match (self, rhs) {
            // Nulls propagate
            (Null, _) | (_, Null) => Null,

            // Float promotion
            (Float64(a), b) => Float64(a + b.f64()),
            (a, Float64(b)) => Float64(a.f64() + b),
            (Float32(a), b) => Float32(a + b.f32()),
            (a, Float32(b)) => Float32(a.f32() + b),

            // Int64 handling
            (Int64(a), b) => Int64(a + b.i64()),
            (a, Int64(b)) => Int64(a.i64() + b),

            // Int32 handling
            (Int32(a), b) => Int32(a + b.i32()),
            (a, Int32(b)) => Int32(a.i32() + b),

            // UInt64 handling
            (UInt64(a), b) => UInt64(a + b.u64()),
            (a, UInt64(b)) => UInt64(a.u64() + b),

            // UInt32 handling
            (UInt32(a), b) => UInt32(a + b.u32()),
            (a, UInt32(b)) => UInt32(a.u32() + b),

            // Boolean as integers
            (Boolean(a), b) => Int32((if a { 1 } else { 0 }) + b.i32()),
            (a, Boolean(b)) => Int32(a.i32() + if b { 1 } else { 0 }),

            // String concatenation
            (String32(a), String32(b)) => String32(a + &b),
            #[cfg(feature = "large_string")]
            (String64(a), String64(b)) => String64(a + &b),

            // String + number coercion
            (String32(a), b) => String32(a + &b.str()),
            (a, String32(b)) => String32(a.str() + &b),

            // Datetime as arithmetic (int-based)
            #[cfg(feature = "datetime")]
            (Datetime32(a), b) => Datetime32(a + b.i32()),
            #[cfg(feature = "datetime")]
            (a, Datetime32(b)) => Datetime32(a.i32() + b),

            #[cfg(feature = "datetime")]
            (Datetime64(a), b) => Datetime64(a + b.i64()),
            #[cfg(feature = "datetime")]
            (a, Datetime64(b)) => Datetime64(a.i64() + b),

            #[cfg(feature = "datetime")]
            (Interval, _) | (_, Interval) => panic!("Cannot add Interval"),

            // Extended numeric types
            #[cfg(feature = "extended_numeric_types")]
            (Int8(a), b) => Int8(a.wrapping_add(b.i8())),
            #[cfg(feature = "extended_numeric_types")]
            (a, Int8(b)) => Int8(a.i8().wrapping_add(b)),

            #[cfg(feature = "extended_numeric_types")]
            (Int16(a), b) => Int16(a.wrapping_add(b.i16())),
            #[cfg(feature = "extended_numeric_types")]
            (a, Int16(b)) => Int16(a.i16().wrapping_add(b)),

            #[cfg(feature = "extended_numeric_types")]
            (UInt8(a), b) => UInt8(a.wrapping_add(b.u8())),
            #[cfg(feature = "extended_numeric_types")]
            (a, UInt8(b)) => UInt8(a.u8().wrapping_add(b)),

            #[cfg(feature = "extended_numeric_types")]
            (UInt16(a), b) => UInt16(a.wrapping_add(b.u16())),
            #[cfg(feature = "extended_numeric_types")]
            (a, UInt16(b)) => UInt16(a.u16().wrapping_add(b)),
        }
    }
}

#[cfg(feature = "scalar_type")]
impl Sub for Scalar {
    type Output = Scalar;

    fn sub(self, rhs: Scalar) -> Scalar {
        use Scalar::*;

        match (self, rhs) {
            (Null, _) | (_, Null) => Null,

            (Float64(a), b) => Float64(a - b.f64()),
            (a, Float64(b)) => Float64(a.f64() - b),
            (Float32(a), b) => Float32(a - b.f32()),
            (a, Float32(b)) => Float32(a.f32() - b),

            (Int64(a), b) => Int64(a - b.i64()),
            (a, Int64(b)) => Int64(a.i64() - b),
            (Int32(a), b) => Int32(a - b.i32()),
            (a, Int32(b)) => Int32(a.i32() - b),

            (UInt64(a), b) => UInt64(a - b.u64()),
            (a, UInt64(b)) => UInt64(a.u64() - b),
            (UInt32(a), b) => UInt32(a - b.u32()),
            (a, UInt32(b)) => UInt32(a.u32() - b),

            (Boolean(a), b) => Int32((if a { 1 } else { 0 }) - b.i32()),
            (a, Boolean(b)) => Int32(a.i32() - if b { 1 } else { 0 }),

            #[cfg(feature = "datetime")]
            (Datetime32(a), b) => Datetime32(a - b.i32()),
            #[cfg(feature = "datetime")]
            (a, Datetime32(b)) => Datetime32(a.i32() - b),
            #[cfg(feature = "datetime")]
            (Datetime64(a), b) => Datetime64(a - b.i64()),
            #[cfg(feature = "datetime")]
            (a, Datetime64(b)) => Datetime64(a.i64() - b),
            #[cfg(feature = "datetime")]
            (Interval, _) | (_, Interval) => panic!("Cannot subtract Interval"),

            #[cfg(feature = "extended_numeric_types")]
            (Int8(a), b) => Int8(a.wrapping_sub(b.i8())),
            #[cfg(feature = "extended_numeric_types")]
            (a, Int8(b)) => Int8(a.i8().wrapping_sub(b)),

            #[cfg(feature = "extended_numeric_types")]
            (Int16(a), b) => Int16(a.wrapping_sub(b.i16())),
            #[cfg(feature = "extended_numeric_types")]
            (a, Int16(b)) => Int16(a.i16().wrapping_sub(b)),

            #[cfg(feature = "extended_numeric_types")]
            (UInt8(a), b) => UInt8(a.wrapping_sub(b.u8())),
            #[cfg(feature = "extended_numeric_types")]
            (a, UInt8(b)) => UInt8(a.u8().wrapping_sub(b)),

            #[cfg(feature = "extended_numeric_types")]
            (UInt16(a), b) => UInt16(a.wrapping_sub(b.u16())),
            #[cfg(feature = "extended_numeric_types")]
            (a, UInt16(b)) => UInt16(a.u16().wrapping_sub(b)),

            // String subtraction - removes first instance from lhs
            (String32(lhs), String32(rhs)) => String32(lhs.replacen(&rhs, "", 1)),
            #[cfg(feature = "large_string")]
            (String64(lhs), String64(rhs)) => String64(lhs.replacen(&rhs, "", 1)),
            #[cfg(feature = "large_string")]
            (String32(lhs), String64(rhs)) => String64(lhs.replacen(&rhs, "", 1)),
            #[cfg(feature = "large_string")]
            (String64(lhs), String32(rhs)) => String64(lhs.replacen(&rhs, "", 1)),
        }
    }
}

#[cfg(feature = "scalar_type")]
impl Mul for Scalar {
    type Output = Scalar;

    fn mul(self, rhs: Scalar) -> Scalar {
        use Scalar::*;

        match (self, rhs) {
            (Null, _) | (_, Null) => Null,

            (Float64(a), b) => Float64(a * b.f64()),
            (a, Float64(b)) => Float64(a.f64() * b),
            (Float32(a), b) => Float32(a * b.f32()),
            (a, Float32(b)) => Float32(a.f32() * b),

            (Int64(a), b) => Int64(a * b.i64()),
            (a, Int64(b)) => Int64(a.i64() * b),
            (Int32(a), b) => Int32(a * b.i32()),
            (a, Int32(b)) => Int32(a.i32() * b),

            (UInt64(a), b) => UInt64(a * b.u64()),
            (a, UInt64(b)) => UInt64(a.u64() * b),
            (UInt32(a), b) => UInt32(a * b.u32()),
            (a, UInt32(b)) => UInt32(a.u32() * b),

            (Boolean(a), b) => Int32((if a { 1 } else { 0 }) * b.i32()),
            (a, Boolean(b)) => Int32(a.i32() * if b { 1 } else { 0 }),

            #[cfg(feature = "datetime")]
            (Datetime32(_), _) | (_, Datetime32(_)) => panic!("Cannot multiply datetimes"),
            #[cfg(feature = "datetime")]
            (Datetime64(_), _) | (_, Datetime64(_)) => panic!("Cannot multiply datetimes"),
            #[cfg(feature = "datetime")]
            (Interval, _) | (_, Interval) => panic!("Cannot multiply Interval"),

            #[cfg(feature = "extended_numeric_types")]
            (Int8(a), b) => Int8(a.wrapping_mul(b.i8())),
            #[cfg(feature = "extended_numeric_types")]
            (a, Int8(b)) => Int8(a.i8().wrapping_mul(b)),

            #[cfg(feature = "extended_numeric_types")]
            (Int16(a), b) => Int16(a.wrapping_mul(b.i16())),
            #[cfg(feature = "extended_numeric_types")]
            (a, Int16(b)) => Int16(a.i16().wrapping_mul(b)),

            #[cfg(feature = "extended_numeric_types")]
            (UInt8(a), b) => UInt8(a.wrapping_mul(b.u8())),
            #[cfg(feature = "extended_numeric_types")]
            (a, UInt8(b)) => UInt8(a.u8().wrapping_mul(b)),

            #[cfg(feature = "extended_numeric_types")]
            (UInt16(a), b) => UInt16(a.wrapping_mul(b.u16())),
            #[cfg(feature = "extended_numeric_types")]
            (a, UInt16(b)) => UInt16(a.u16().wrapping_mul(b)),

            // String handling - Adds the rhs string to every line of lhs
            (String32(a), String32(b)) => {
                let appended = a
                    .lines()
                    .map(|line| format!("{line}{b}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                String32(appended)
            }
            #[cfg(feature = "large_string")]
            (String64(a), String64(b)) => {
                let appended = a
                    .lines()
                    .map(|line| format!("{line}{b}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                String64(appended)
            }

            #[cfg(feature = "large_string")]
            (String32(a), String64(b)) => {
                let appended = a
                    .lines()
                    .map(|line| format!("{line}{b}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                String64(appended)
            }

            #[cfg(feature = "large_string")]
            (String64(a), String32(b)) => {
                let appended = a
                    .lines()
                    .map(|line| format!("{line}{b}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                String64(appended)
            }
        }
    }
}

#[cfg(feature = "scalar_type")]
impl Pow<Scalar> for Scalar {
    type Output = Scalar;

    fn pow(self, rhs: Scalar) -> Scalar {
        use Scalar::*;

        match (self, rhs) {
            // Nulls propagate
            (Null, _) | (_, Null) => Null,

            #[cfg(feature = "datetime")]
            (Interval, _) => panic!("Cannot exponentiate Interval"),
            #[cfg(feature = "datetime")]
            (_, Interval) => panic!("Cannot exponentiate Interval"),

            // Datetime - explicitly panic for all
            #[cfg(feature = "datetime")]
            (Datetime32(_), _) => panic!("Cannot exponentiate datetimes"),
            #[cfg(feature = "datetime")]
            (_, Datetime32(_)) => panic!("Cannot exponentiate datetimes"),
            #[cfg(feature = "datetime")]
            (Datetime64(_), _) => panic!("Cannot exponentiate datetimes"),
            #[cfg(feature = "datetime")]
            (_, Datetime64(_)) => panic!("Cannot exponentiate datetimes"),

            // Boolean
            (Boolean(a), b) => {
                let base = if a { 1 } else { 0 };
                Int32(base.pow(b.u32()))
            }
            (a, Boolean(b)) => {
                let exp = if b { 1 } else { 0 };
                Int32(a.i32().pow(exp))
            }

            // String + String - append rhs to each word in lhs
            (String32(a), String32(b)) => {
                let appended = a
                    .split_whitespace()
                    .map(|word| format!("{word}{b}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                String32(appended)
            }

            #[cfg(feature = "large_string")]
            (String64(a), String64(b)) => {
                let appended = a
                    .split_whitespace()
                    .map(|word| format!("{word}{b}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                String64(appended)
            }

            #[cfg(feature = "large_string")]
            (String32(a), String64(b)) => {
                let appended = a
                    .split_whitespace()
                    .map(|word| format!("{word}{b}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                String64(appended)
            }

            #[cfg(feature = "large_string")]
            (String64(a), String32(b)) => {
                let appended = a
                    .split_whitespace()
                    .map(|word| format!("{word}{b}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                String64(appended)
            }

            // Attempt to parse string as f64
            (String32(a), b) => {
                let base = Scalar::Float64(
                    a.parse::<f64>()
                        .expect("Exponent: Cannot parse string as f64 base"),
                );
                base.pow(b)
            }
            #[cfg(feature = "large_string")]
            (String64(a), b) => {
                let base = Scalar::Float64(
                    a.parse::<f64>()
                        .expect("Exponent: Cannot parse string as f64 base"),
                );
                base.pow(b)
            }
            (a, String32(b)) => {
                let exp = Scalar::Float64(
                    b.parse::<f64>()
                        .expect("Exponent: Cannot parse string as f64 exponent"),
                );
                a.pow(exp)
            }
            #[cfg(feature = "large_string")]
            (a, String64(b)) => {
                let exp = Scalar::Float64(
                    b.parse::<f64>()
                        .expect("Exponent: Cannot parse string as f64 exponent"),
                );
                a.pow(exp)
            }

            // Floats
            (Float64(a), b) => Float64(a.powf(b.f64())),
            (a, Float64(b)) => Float64(a.f64().powf(b)),
            (Float32(a), b) => Float32(a.powf(b.f32())),
            (a, Float32(b)) => Float32(a.f32().powf(b)),

            // Signed integers
            (Int64(a), b) => Int64(a.pow(b.u32())),
            (Int32(a), b) => Int32(a.pow(b.u32())),

            #[cfg(feature = "extended_numeric_types")]
            (Int8(a), b) => Int8(a.wrapping_pow(b.u32())),
            #[cfg(feature = "extended_numeric_types")]
            (Int16(a), b) => Int16(a.wrapping_pow(b.u32())),

            // Unsigned integers
            (UInt64(a), b) => UInt64(a.pow(b.u32())),
            (UInt32(a), b) => UInt32(a.pow(b.u32())),

            #[cfg(feature = "extended_numeric_types")]
            (UInt8(a), b) => UInt8(a.wrapping_pow(b.u32())),
            #[cfg(feature = "extended_numeric_types")]
            (UInt16(a), b) => UInt16(a.wrapping_pow(b.u32())),
        }
    }
}

#[cfg(feature = "scalar_type")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Scalar::{Float32, Float64, Int32, String32, String64};

    #[test]
    fn test_bool() {
        assert_eq!(Scalar::Boolean(true).bool(), true);
        assert_eq!(Scalar::Boolean(false).bool(), false);
        assert_eq!(Scalar::Int32(1).bool(), true);
        assert_eq!(Scalar::Int32(0).bool(), false);
        assert_eq!(Scalar::Int64(-1).bool(), true);
        assert_eq!(Scalar::UInt32(10).bool(), true);
        assert_eq!(Scalar::UInt64(0).bool(), false);
        assert_eq!(Scalar::Float32(1.5).bool(), true);
        assert_eq!(Scalar::Float64(0.0).bool(), false);
        assert_eq!(Scalar::String32("true".to_owned()).bool(), true);
        assert_eq!(Scalar::String32("FaLse".to_owned()).bool(), false);
        assert_eq!(Scalar::String32("  t ".to_owned()).bool(), true);
        assert_eq!(Scalar::String32("0".to_owned()).bool(), false);

        #[cfg(feature = "datetime")]
        {
            assert_eq!(Scalar::Datetime32(0).bool(), false);
            assert_eq!(Scalar::Datetime64(1).bool(), true);
        }

        #[cfg(feature = "extended_numeric_types")]
        {
            assert_eq!(Scalar::Int8(-1).bool(), true);
            assert_eq!(Scalar::Int16(0).bool(), false);
            assert_eq!(Scalar::UInt8(1).bool(), true);
            assert_eq!(Scalar::UInt16(0).bool(), false);
        }
    }

    #[test]
    #[should_panic(expected = "Cannot convert Null to bool")]
    fn test_bool_null_panics() {
        Scalar::Null.bool();
    }

    #[test]
    #[should_panic(expected = "Cannot convert empty string to bool")]
    fn test_bool_empty_string_panics() {
        Scalar::String32("  ".to_owned()).bool();
    }

    #[test]
    #[should_panic]
    fn test_bool_invalid_string_panics() {
        Scalar::String32("notabool".to_owned()).bool();
    }

    #[cfg(feature = "extended_numeric_types")]
    #[test]
    fn test_i8_all() {
        assert_eq!(Scalar::Int8(-128).i8(), -128);
        assert_eq!(Scalar::Int16(127).i8(), 127);
        assert_eq!(Scalar::Int32(100).i8(), 100);
        assert_eq!(Scalar::Int64(1).i8(), 1);
        assert_eq!(Scalar::UInt8(1).i8(), 1);
        assert_eq!(Scalar::UInt16(2).i8(), 2);
        assert_eq!(Scalar::UInt32(10).i8(), 10);
        assert_eq!(Scalar::UInt64(20).i8(), 20);
        assert_eq!(Scalar::Float32(123.0).i8(), 123);
        assert_eq!(Scalar::Float64(120.0).i8(), 120);
        assert_eq!(Scalar::String32("-7".into()).i8(), -7);
        assert_eq!(Scalar::Boolean(true).i8(), 1);
        assert_eq!(Scalar::Boolean(false).i8(), 0);

        #[cfg(feature = "datetime")]
        {
            assert_eq!(Scalar::Datetime32(1).i8(), 1);
            assert_eq!(Scalar::Datetime64(2).i8(), 2);
        }
    }

    #[cfg(feature = "extended_numeric_types")]
    #[test]
    #[should_panic(expected = "i16 out of range for i8")]
    fn test_i8_i16_overflow() {
        Scalar::Int16(128).i8();
    }

    #[cfg(feature = "extended_numeric_types")]
    #[test]
    #[should_panic(expected = "Cannot parse string as i8")]
    fn test_i8_invalid_string() {
        Scalar::String32("foo".into()).i8();
    }

    #[cfg(feature = "extended_numeric_types")]
    #[test]
    #[should_panic(expected = "Cannot convert Null to i8")]
    fn test_i8_null_panics() {
        Scalar::Null.i8();
    }

    #[cfg(feature = "extended_numeric_types")]
    #[test]
    fn test_i16_all() {
        assert_eq!(Scalar::Int8(-7).i16(), -7);
        assert_eq!(Scalar::Int16(7).i16(), 7);
        assert_eq!(Scalar::Int32(128).i16(), 128);
        assert_eq!(Scalar::UInt8(5).i16(), 5);
        assert_eq!(Scalar::Float32(5.0).i16(), 5);
        assert_eq!(Scalar::String32("9".into()).i16(), 9);
        assert_eq!(Scalar::Boolean(true).i16(), 1);
    }

    #[cfg(feature = "extended_numeric_types")]
    #[test]
    #[should_panic]
    fn test_i16_invalid_string() {
        Scalar::String32("foo".into()).i16();
    }

    #[test]
    fn test_i32_all() {
        assert_eq!(Scalar::Int32(7).i32(), 7);
        assert_eq!(Scalar::Int64(9).i32(), 9);
        assert_eq!(Scalar::UInt32(15).i32(), 15);
        assert_eq!(Scalar::UInt64(20).i32(), 20);
        assert_eq!(Scalar::Float32(7.9).i32(), 7);
        assert_eq!(Scalar::Float64(8.9).i32(), 8);
        assert_eq!(Scalar::String32("10".into()).i32(), 10);
        assert_eq!(Scalar::Boolean(false).i32(), 0);

        #[cfg(feature = "extended_numeric_types")]
        {
            assert_eq!(Scalar::Int8(3).i32(), 3);
            assert_eq!(Scalar::Int16(5).i32(), 5);
            assert_eq!(Scalar::UInt8(5).i32(), 5);
            assert_eq!(Scalar::UInt16(5).i32(), 5);
        }
    }

    #[test]
    #[should_panic(expected = "Cannot parse string as i32")]
    fn test_i32_invalid_string() {
        Scalar::String32("not_a_number".to_string()).i32();
    }

    #[test]
    #[should_panic(expected = "Cannot convert Null to i32")]
    fn test_i32_null_panics() {
        Scalar::Null.i32();
    }

    #[test]
    fn test_i64_all() {
        assert_eq!(Scalar::Int64(-12).i64(), -12);
        assert_eq!(Scalar::Int32(7).i64(), 7);
        assert_eq!(Scalar::UInt32(100).i64(), 100);
        assert_eq!(Scalar::UInt64(127).i64(), 127);
        assert_eq!(Scalar::Float32(1.2).i64(), 1);
        assert_eq!(Scalar::String32("99".into()).i64(), 99);
        assert_eq!(Scalar::Boolean(true).i64(), 1);

        #[cfg(feature = "extended_numeric_types")]
        {
            assert_eq!(Scalar::Int8(-1).i64(), -1);
            assert_eq!(Scalar::Int16(2).i64(), 2);
            assert_eq!(Scalar::UInt8(10).i64(), 10);
            assert_eq!(Scalar::UInt16(20).i64(), 20);
        }
    }

    #[test]
    #[should_panic]
    fn test_i64_string_invalid() {
        Scalar::String32("oops".to_string()).i64();
    }

    #[test]
    fn test_u8_all() {
        #[cfg(feature = "extended_numeric_types")]
        assert_eq!(Scalar::UInt8(15).u8(), 15);
        #[cfg(feature = "extended_numeric_types")]
        assert_eq!(Scalar::UInt16(15).u8(), 15);
        assert_eq!(Scalar::UInt32(15).u8(), 15);
        assert_eq!(Scalar::UInt64(15).u8(), 15);
        assert_eq!(Scalar::Int32(3).u8(), 3);
        assert_eq!(Scalar::Float32(6.0).u8(), 6);
        assert_eq!(Scalar::String32("7".into()).u8(), 7);
        assert_eq!(Scalar::Boolean(true).u8(), 1);
    }

    #[test]
    #[should_panic(expected = "Cannot parse string as u8")]
    fn test_u8_string_invalid() {
        Scalar::String32("abc".to_string()).u8();
    }

    #[test]
    fn test_f32_and_f64() {
        assert_eq!(Scalar::Float32(1.25).f32(), 1.25);
        assert_eq!(Scalar::Float64(7.5).f64(), 7.5);
        assert_eq!(Scalar::Int32(9).f32(), 9.0);
        assert_eq!(Scalar::Boolean(true).f64(), 1.0);
        assert_eq!(Scalar::String32("1.0".into()).f32(), 1.0);
    }

    #[test]
    #[should_panic(expected = "Cannot convert Null to f32")]
    fn test_f32_null_panics() {
        Scalar::Null.f32();
    }

    #[test]
    #[should_panic(expected = "Cannot parse string as f32")]
    fn test_f32_invalid_string_panics() {
        Scalar::String32("notafloat".into()).f32();
    }

    #[test]
    fn test_str() {
        assert_eq!(Scalar::String32("foo".into()).str(), "foo");
        assert_eq!(Scalar::Boolean(true).str(), "true");
        assert_eq!(Scalar::Float32(1.0).str(), "1");
        assert_eq!(Scalar::Int64(-5).str(), "-5");

        #[cfg(feature = "extended_numeric_types")]
        {
            assert_eq!(Scalar::Int8(-3).str(), "-3");
            assert_eq!(Scalar::UInt8(3).str(), "3");
        }
        #[cfg(feature = "datetime")]
        {
            assert_eq!(Scalar::Datetime32(55).str(), "55");
        }
    }

    #[test]
    #[should_panic(expected = "Cannot convert Null to String")]
    fn test_str_null_panics() {
        Scalar::Null.str();
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn test_dt32_dt64() {
        assert_eq!(Scalar::Datetime32(5).dt32(), 5);
        assert_eq!(Scalar::Datetime64(99).dt64(), 99);
        assert_eq!(Scalar::UInt32(5).dt32(), 5);
        assert_eq!(Scalar::Int32(2).dt32(), 2);
        assert_eq!(Scalar::UInt64(25).dt64(), 25);
        assert_eq!(Scalar::Int64(5).dt64(), 5);
        assert_eq!(Scalar::String32("7".into()).dt32(), 7);
        assert_eq!(Scalar::String32("8".into()).dt64(), 8);
    }

    #[cfg(feature = "datetime")]
    #[test]
    #[should_panic(expected = "Cannot convert Null to dt32")]
    fn test_dt32_null_panics() {
        Scalar::Null.dt32();
    }

    #[cfg(feature = "datetime")]
    #[test]
    #[should_panic(expected = "Cannot convert Null to dt64")]
    fn test_dt64_null_panics() {
        Scalar::Null.dt64();
    }

    #[cfg(feature = "datetime")]
    #[test]
    #[should_panic(expected = "i32 out of range for dt32 (negative value)")]
    fn test_dt32_negative_int_panics() {
        Scalar::Int32(-1).dt32();
    }

    #[cfg(feature = "datetime")]
    #[test]
    #[should_panic(expected = "i64 out of range for dt64 (negative value)")]
    fn test_dt64_negative_int_panics() {
        Scalar::Int64(-1).dt64();
    }

    #[cfg(feature = "datetime")]
    #[test]
    #[should_panic]
    fn test_str_interval_panics() {
        Scalar::Interval.str();
    }

    #[test]
    #[should_panic]
    fn test_any_null_panics() {
        Scalar::Null.u64();
    }

    #[cfg(feature = "hash")]
    #[test]
    fn test_scalar_hash_same_value_same_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        fn hash_one(s: &Scalar) -> u64 {
            let mut h = DefaultHasher::new();
            s.hash(&mut h);
            h.finish()
        }

        assert_eq!(hash_one(&Int32(42)), hash_one(&Int32(42)));
        assert_eq!(hash_one(&Float64(3.14)), hash_one(&Float64(3.14)));
        assert_eq!(
            hash_one(&String32("hello".into())),
            hash_one(&String32("hello".into()))
        );
        assert_eq!(hash_one(&Scalar::Null), hash_one(&Scalar::Null));
    }

    #[cfg(feature = "hash")]
    #[test]
    fn test_scalar_hash_different_value_different_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        fn hash_one(s: &Scalar) -> u64 {
            let mut h = DefaultHasher::new();
            s.hash(&mut h);
            h.finish()
        }

        assert_ne!(hash_one(&Int32(1)), hash_one(&Int32(2)));
        assert_ne!(hash_one(&Float64(1.0)), hash_one(&Float64(2.0)));
    }

    #[cfg(feature = "hash")]
    #[test]
    fn test_scalar_hash_float_to_bits_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        fn hash_one(s: &Scalar) -> u64 {
            let mut h = DefaultHasher::new();
            s.hash(&mut h);
            h.finish()
        }

        // NaN == NaN under to_bits() convention
        assert_eq!(hash_one(&Float64(f64::NAN)), hash_one(&Float64(f64::NAN)));
        assert_eq!(hash_one(&Float32(f32::NAN)), hash_one(&Float32(f32::NAN)));
    }

    #[cfg(feature = "hash")]
    #[test]
    fn test_scalar_eq() {
        assert_eq!(Int32(5), Int32(5));
        assert_ne!(Int32(5), Int32(6));
        assert_eq!(Scalar::Null, Scalar::Null);
    }

    #[cfg(test)]
    #[cfg(feature = "scalar_type")]
    mod try_scalar_tests {
        use super::*;

        #[test]
        fn test_try_bool_success_and_none() {
            assert_eq!(Scalar::Boolean(true).try_bool(), Some(true));
            assert_eq!(Scalar::Int32(0).try_bool(), Some(false));
            assert_eq!(Scalar::Int64(7).try_bool(), Some(true));
            assert_eq!(Scalar::UInt32(0).try_bool(), Some(false));
            assert_eq!(Scalar::UInt64(1).try_bool(), Some(true));
            assert_eq!(Scalar::Float32(0.0).try_bool(), Some(false));
            assert_eq!(Scalar::Float64(-3.2).try_bool(), Some(true));
            assert_eq!(Scalar::Null.try_bool(), None);
            assert_eq!(Scalar::String32("t".into()).try_bool(), Some(true));
            assert_eq!(Scalar::String32("False".into()).try_bool(), Some(false));
            assert_eq!(Scalar::String32("".into()).try_bool(), None);
            assert_eq!(Scalar::String32("bad".into()).try_bool(), None);

            #[cfg(feature = "large_string")]
            {
                assert_eq!(Scalar::String64("T".into()).try_bool(), Some(true));
                assert_eq!(Scalar::String64("0".into()).try_bool(), Some(false));
                assert_eq!(Scalar::String64("".into()).try_bool(), None);
                assert_eq!(Scalar::String64("off".into()).try_bool(), None);
            }

            #[cfg(feature = "datetime")]
            {
                assert_eq!(Scalar::Datetime32(0).try_bool(), Some(false));
                assert_eq!(Scalar::Datetime64(2).try_bool(), Some(true));
                assert_eq!(Scalar::Interval.try_bool(), None);
            }
            #[cfg(feature = "extended_numeric_types")]
            {
                assert_eq!(Scalar::Int8(0).try_bool(), Some(false));
                assert_eq!(Scalar::Int8(5).try_bool(), Some(true));
                assert_eq!(Scalar::Int16(-3).try_bool(), Some(true));
                assert_eq!(Scalar::UInt8(0).try_bool(), Some(false));
                assert_eq!(Scalar::UInt16(1).try_bool(), Some(true));
            }
        }

        #[cfg(feature = "extended_numeric_types")]
        #[test]
        fn test_try_i8_all_cases() {
            assert_eq!(Scalar::Int8(3).try_i8(), Some(3));
            assert_eq!(Scalar::Int16(127).try_i8(), Some(127));
            assert_eq!(Scalar::Int16(128).try_i8(), None);
            assert_eq!(Scalar::Int32(-128).try_i8(), Some(-128));
            assert_eq!(Scalar::Int32(200).try_i8(), None);
            assert_eq!(Scalar::Int64(-5).try_i8(), Some(-5));
            assert_eq!(Scalar::UInt8(15).try_i8(), Some(15));
            assert_eq!(Scalar::UInt16(200).try_i8(), None);
            assert_eq!(Scalar::Float32(12.0).try_i8(), Some(12));
            assert_eq!(Scalar::Float64(10.5).try_i8(), Some(10));
            assert_eq!(Scalar::Null.try_i8(), None);
            assert_eq!(Scalar::String32("44".into()).try_i8(), Some(44));
            assert_eq!(Scalar::String32("err".into()).try_i8(), None);
            assert_eq!(Scalar::Boolean(true).try_i8(), Some(1));
            assert_eq!(Scalar::Boolean(false).try_i8(), Some(0));

            #[cfg(feature = "large_string")]
            {
                assert_eq!(Scalar::String64("100".into()).try_i8(), Some(100));
                assert_eq!(Scalar::String64("bad".into()).try_i8(), None);
            }
            #[cfg(feature = "datetime")]
            {
                assert_eq!(Scalar::Datetime32(2).try_i8(), Some(2));
                assert_eq!(Scalar::Datetime64(127).try_i8(), Some(127));
                assert_eq!(Scalar::Interval.try_i8(), None);
            }
        }

        #[cfg(feature = "extended_numeric_types")]
        #[test]
        fn test_try_i16_all_cases() {
            assert_eq!(Scalar::Int8(-7).try_i16(), Some(-7));
            assert_eq!(Scalar::Int16(7).try_i16(), Some(7));
            assert_eq!(Scalar::Int32(32767).try_i16(), Some(32767));
            assert_eq!(Scalar::Int32(40000).try_i16(), None);
            assert_eq!(Scalar::Int64(-1).try_i16(), Some(-1));
            assert_eq!(Scalar::Int64(40000).try_i16(), None);
            assert_eq!(Scalar::UInt8(5).try_i16(), Some(5));
            assert_eq!(Scalar::UInt16(32767).try_i16(), Some(32767));
            assert_eq!(Scalar::UInt32(70000).try_i16(), None);
            assert_eq!(Scalar::UInt64(32767).try_i16(), Some(32767));
            assert_eq!(Scalar::UInt64(40000).try_i16(), None);
            assert_eq!(Scalar::Float32(5.0).try_i16(), Some(5));
            assert_eq!(Scalar::Float32(40000.0).try_i16(), None);
            assert_eq!(Scalar::Float64(-7.9).try_i16(), Some(-7));
            assert_eq!(Scalar::Null.try_i16(), None);
            assert_eq!(Scalar::String32("9".into()).try_i16(), Some(9));
            assert_eq!(Scalar::String32("bad".into()).try_i16(), None);
            assert_eq!(Scalar::Boolean(true).try_i16(), Some(1));

            #[cfg(feature = "large_string")]
            {
                assert_eq!(Scalar::String64("10".into()).try_i16(), Some(10));
                assert_eq!(Scalar::String64("bad".into()).try_i16(), None);
            }
            #[cfg(feature = "datetime")]
            {
                assert_eq!(Scalar::Datetime32(4).try_i16(), Some(4));
                assert_eq!(Scalar::Datetime64(99).try_i16(), Some(99));
                assert_eq!(Scalar::Interval.try_i16(), None);
            }
        }

        #[test]
        fn test_try_i32_all_cases() {
            assert_eq!(Scalar::Int32(7).try_i32(), Some(7));
            assert_eq!(Scalar::Int64(12).try_i32(), Some(12));
            assert_eq!(Scalar::Int64(i64::from(i32::MAX) + 1).try_i32(), None);
            assert_eq!(Scalar::UInt32(10).try_i32(), Some(10));
            assert_eq!(Scalar::UInt64(100).try_i32(), Some(100));
            assert_eq!(Scalar::UInt64(u64::from(u32::MAX) + 1).try_i32(), None);
            assert_eq!(Scalar::Float32(11.7).try_i32(), Some(11));
            assert_eq!(Scalar::Float64(8.9).try_i32(), Some(8));
            assert_eq!(Scalar::Null.try_i32(), None);
            assert_eq!(Scalar::String32("10".into()).try_i32(), Some(10));
            assert_eq!(Scalar::String32("bad".into()).try_i32(), None);
            assert_eq!(Scalar::Boolean(true).try_i32(), Some(1));

            #[cfg(feature = "extended_numeric_types")]
            {
                assert_eq!(Scalar::Int8(3).try_i32(), Some(3));
                assert_eq!(Scalar::Int16(5).try_i32(), Some(5));
                assert_eq!(Scalar::UInt8(5).try_i32(), Some(5));
                assert_eq!(Scalar::UInt16(5).try_i32(), Some(5));
            }
            #[cfg(feature = "large_string")]
            {
                assert_eq!(Scalar::String64("10".into()).try_i32(), Some(10));
                assert_eq!(Scalar::String64("bad".into()).try_i32(), None);
            }
            #[cfg(feature = "datetime")]
            {
                assert_eq!(Scalar::Datetime32(5).try_i32(), Some(5));
                assert_eq!(Scalar::Datetime64(7).try_i32(), Some(7));
                assert_eq!(Scalar::Interval.try_i32(), None);
            }
        }

        #[test]
        fn test_try_i64_all_cases() {
            assert_eq!(Scalar::Int32(2).try_i64(), Some(2));
            assert_eq!(Scalar::Int64(7).try_i64(), Some(7));
            assert_eq!(Scalar::UInt32(5).try_i64(), Some(5));
            assert_eq!(Scalar::UInt64(127).try_i64(), Some(127));
            assert_eq!(Scalar::UInt64(u64::MAX).try_i64(), None);
            assert_eq!(Scalar::Float32(1.2).try_i64(), Some(1));
            assert_eq!(Scalar::Float64(-1.2).try_i64(), Some(-1));
            assert_eq!(Scalar::Null.try_i64(), None);
            assert_eq!(Scalar::String32("99".into()).try_i64(), Some(99));
            assert_eq!(Scalar::String32("bad".into()).try_i64(), None);
            assert_eq!(Scalar::Boolean(true).try_i64(), Some(1));

            #[cfg(feature = "extended_numeric_types")]
            {
                assert_eq!(Scalar::Int8(-1).try_i64(), Some(-1));
                assert_eq!(Scalar::Int16(2).try_i64(), Some(2));
                assert_eq!(Scalar::UInt8(10).try_i64(), Some(10));
                assert_eq!(Scalar::UInt16(20).try_i64(), Some(20));
            }
            #[cfg(feature = "large_string")]
            {
                assert_eq!(Scalar::String64("10".into()).try_i64(), Some(10));
                assert_eq!(Scalar::String64("bad".into()).try_i64(), None);
            }
            #[cfg(feature = "datetime")]
            {
                assert_eq!(Scalar::Datetime32(4).try_i64(), Some(4));
                assert_eq!(Scalar::Datetime64(99).try_i64(), Some(99));
                assert_eq!(Scalar::Interval.try_i64(), None);
            }
        }

        #[test]
        fn test_try_u8_all_cases() {
            #[cfg(feature = "extended_numeric_types")]
            assert_eq!(Scalar::UInt8(15).try_u8(), Some(15));
            #[cfg(feature = "extended_numeric_types")]
            assert_eq!(Scalar::UInt16(15).try_u8(), Some(15));
            #[cfg(feature = "extended_numeric_types")]
            assert_eq!(Scalar::UInt16(256).try_u8(), None);
            assert_eq!(Scalar::UInt32(200).try_u8(), Some(200));
            assert_eq!(Scalar::Int32(3).try_u8(), Some(3));
            assert_eq!(Scalar::Int32(-1).try_u8(), None);
            assert_eq!(Scalar::Float32(6.0).try_u8(), Some(6));
            assert_eq!(Scalar::Float32(300.0).try_u8(), None);
            assert_eq!(Scalar::String32("7".into()).try_u8(), Some(7));
            assert_eq!(Scalar::String32("bad".into()).try_u8(), None);
            assert_eq!(Scalar::Boolean(true).try_u8(), Some(1));
            assert_eq!(Scalar::Null.try_u8(), None);

            #[cfg(feature = "extended_numeric_types")]
            {
                assert_eq!(Scalar::Int8(7).try_u8(), Some(7));
                assert_eq!(Scalar::Int8(-1).try_u8(), None);
                assert_eq!(Scalar::Int16(7).try_u8(), Some(7));
                assert_eq!(Scalar::Int16(-1).try_u8(), None);
            }
            #[cfg(feature = "large_string")]
            {
                assert_eq!(Scalar::String64("10".into()).try_u8(), Some(10));
                assert_eq!(Scalar::String64("bad".into()).try_u8(), None);
            }
            #[cfg(feature = "datetime")]
            {
                assert_eq!(Scalar::Datetime32(2).try_u8(), Some(2));
                assert_eq!(Scalar::Datetime64(7).try_u8(), Some(7));
                assert_eq!(Scalar::Interval.try_u8(), None);
            }
        }

        #[test]
        fn test_try_f32_and_try_f64_all_cases() {
            assert_eq!(Scalar::Float32(1.25).try_f32(), Some(1.25));
            assert_eq!(Scalar::Float64(7.5).try_f64(), Some(7.5));
            assert_eq!(Scalar::Int32(9).try_f32(), Some(9.0));
            assert_eq!(Scalar::Boolean(true).try_f64(), Some(1.0));
            assert_eq!(Scalar::Boolean(false).try_f64(), Some(0.0));
            assert_eq!(Scalar::String32("1.0".into()).try_f32(), Some(1.0));
            assert_eq!(Scalar::String32("bad".into()).try_f32(), None);
            assert_eq!(Scalar::Null.try_f32(), None);
            assert_eq!(Scalar::Null.try_f64(), None);

            #[cfg(feature = "extended_numeric_types")]
            {
                assert_eq!(Scalar::Int8(4).try_f32(), Some(4.0));
                assert_eq!(Scalar::Int16(-9).try_f64(), Some(-9.0));
            }
            #[cfg(feature = "large_string")]
            {
                assert_eq!(Scalar::String64("5.5".into()).try_f32(), Some(5.5));
                assert_eq!(Scalar::String64("oops".into()).try_f64(), None);
            }
            #[cfg(feature = "datetime")]
            {
                assert_eq!(Scalar::Datetime32(4).try_f32(), Some(4.0));
                assert_eq!(Scalar::Datetime64(9).try_f64(), Some(9.0));
                assert_eq!(Scalar::Interval.try_f32(), None);
            }
        }

        #[test]
        fn test_try_str() {
            assert_eq!(
                Scalar::String32("foo".into()).try_str(),
                Some("foo".to_string())
            );
            assert_eq!(Scalar::Boolean(true).try_str(), Some("true".to_string()));
            assert_eq!(Scalar::Float32(1.0).try_str(), Some("1".to_string()));
            assert_eq!(Scalar::Int64(-5).try_str(), Some("-5".to_string()));
            assert_eq!(Scalar::Null.try_str(), None);

            #[cfg(feature = "extended_numeric_types")]
            {
                assert_eq!(Scalar::Int8(-3).try_str(), Some("-3".to_string()));
                assert_eq!(Scalar::UInt8(3).try_str(), Some("3".to_string()));
            }
            #[cfg(feature = "datetime")]
            {
                assert_eq!(Scalar::Datetime32(55).try_str(), Some("55".to_string()));
                assert_eq!(Scalar::Interval.try_str(), None);
            }
            #[cfg(feature = "large_string")]
            {
                assert_eq!(
                    Scalar::String64("bar".into()).try_str(),
                    Some("bar".to_string())
                );
            }
        }

        #[cfg(feature = "datetime")]
        #[test]
        fn test_try_dt32_and_try_dt64() {
            assert_eq!(Scalar::Datetime32(5).try_dt32(), Some(5));
            assert_eq!(Scalar::UInt32(99).try_dt32(), Some(99));
            assert_eq!(Scalar::Int32(7).try_dt32(), Some(7));
            assert_eq!(Scalar::Int32(-1).try_dt32(), None);
            assert_eq!(Scalar::Int64(8).try_dt32(), Some(8));
            assert_eq!(Scalar::Int64(-1).try_dt32(), None);
            assert_eq!(Scalar::UInt64(10).try_dt32(), Some(10));
            assert_eq!(Scalar::UInt64(u64::from(u32::MAX) + 1).try_dt32(), None);
            assert_eq!(Scalar::Float32(12.0).try_dt32(), Some(12));
            assert_eq!(Scalar::Float32(-1.0).try_dt32(), None);
            assert_eq!(Scalar::Float32(u32::MAX as f32 + 1000.0).try_dt32(), None);
            assert_eq!(Scalar::Float64(44.0).try_dt32(), Some(44));
            assert_eq!(Scalar::Float64(-5.0).try_dt32(), None);
            assert_eq!(Scalar::String32("33".into()).try_dt32(), Some(33));
            assert_eq!(Scalar::String32("bad".into()).try_dt32(), None);
            assert_eq!(Scalar::Null.try_dt32(), None);
            assert_eq!(Scalar::Boolean(true).try_dt32(), Some(1));
            assert_eq!(Scalar::Boolean(false).try_dt32(), Some(0));
            assert_eq!(Scalar::Datetime64(1).try_dt32(), Some(1));
            assert_eq!(Scalar::Datetime64(i64::from(u32::MAX) + 1).try_dt32(), None);
            assert_eq!(Scalar::Interval.try_dt32(), None);
            #[cfg(feature = "extended_numeric_types")]
            assert_eq!(Scalar::Int8(3).try_dt32(), Some(3));
            #[cfg(feature = "extended_numeric_types")]
            assert_eq!(Scalar::Int8(-1).try_dt32(), None);
            #[cfg(feature = "extended_numeric_types")]
            assert_eq!(Scalar::Int16(4).try_dt32(), Some(4));
            #[cfg(feature = "extended_numeric_types")]
            assert_eq!(Scalar::Int16(-1).try_dt32(), None);
            #[cfg(feature = "extended_numeric_types")]
            assert_eq!(Scalar::UInt8(5).try_dt32(), Some(5));
            #[cfg(feature = "extended_numeric_types")]
            assert_eq!(Scalar::UInt16(6).try_dt32(), Some(6));
        }

        #[cfg(feature = "datetime")]
        #[test]
        fn test_try_dt64() {
            assert_eq!(Scalar::Datetime64(100).try_dt64(), Some(100));
            assert_eq!(Scalar::UInt64(7).try_dt64(), Some(7));
            assert_eq!(Scalar::Int64(6).try_dt64(), Some(6));
            assert_eq!(Scalar::Int64(-1).try_dt64(), None);
            assert_eq!(Scalar::Int32(4).try_dt64(), Some(4));
            assert_eq!(Scalar::Int32(-4).try_dt64(), None);
            assert_eq!(Scalar::UInt32(33).try_dt64(), Some(33));
            assert_eq!(Scalar::Float32(3.0).try_dt64(), Some(3));
            assert_eq!(Scalar::Float32(-3.0).try_dt64(), None);
            assert_eq!(Scalar::Float64(10.0).try_dt64(), Some(10));
            assert_eq!(Scalar::Float64(-5.0).try_dt64(), None);
            assert_eq!(Scalar::String32("8".into()).try_dt64(), Some(8));
            assert_eq!(Scalar::String32("notnum".into()).try_dt64(), None);
            assert_eq!(Scalar::Null.try_dt64(), None);
            assert_eq!(Scalar::Boolean(true).try_dt64(), Some(1));
            assert_eq!(Scalar::Boolean(false).try_dt64(), Some(0));
            assert_eq!(Scalar::Datetime32(12).try_dt64(), Some(12));
            assert_eq!(Scalar::Interval.try_dt64(), None);
            #[cfg(feature = "extended_numeric_types")]
            assert_eq!(Scalar::Int8(3).try_dt64(), Some(3));
            #[cfg(feature = "extended_numeric_types")]
            assert_eq!(Scalar::Int8(-1).try_dt64(), None);
            #[cfg(feature = "extended_numeric_types")]
            assert_eq!(Scalar::Int16(4).try_dt64(), Some(4));
            #[cfg(feature = "extended_numeric_types")]
            assert_eq!(Scalar::Int16(-1).try_dt64(), None);
            #[cfg(feature = "extended_numeric_types")]
            assert_eq!(Scalar::UInt8(5).try_dt64(), Some(5));
            #[cfg(feature = "extended_numeric_types")]
            assert_eq!(Scalar::UInt16(6).try_dt64(), Some(6));
            #[cfg(feature = "large_string")]
            {
                assert_eq!(Scalar::String64("101".into()).try_dt64(), Some(101));
                assert_eq!(Scalar::String64("notnum".into()).try_dt64(), None);
            }
        }
    }

    #[test]
    fn test_add_strings() {
        assert_eq!(
            String32("foo".into()) + String32("bar".into()),
            String32("foobar".into())
        );
    }

    #[cfg(feature = "large_string")]
    #[test]
    fn test_add_string32_and_string64() {
        assert_eq!(
            String32("abc".into()) + String64("123".into()),
            String32("abc123".into())
        );
        assert_eq!(
            String32("xyz".into()) + String32("789".into()),
            String32("xyz789".into())
        );
    }

    #[test]
    fn test_sub_string_first_instance() {
        assert_eq!(
            String32("hello world world".into()) - String32("world".into()),
            String32("hello  world".into())
        );
    }

    #[cfg(feature = "large_string")]
    #[test]
    fn test_sub_string_mixed_variants() {
        assert_eq!(
            String32("foo bar bar".into()) - String64("bar".into()),
            String64("foo  bar".into())
        );
        assert_eq!(
            String64("bar bar bar".into()) - String32("bar".into()),
            String64(" bar bar".into())
        );
    }

    #[test]
    fn test_mul_append_to_each_line() {
        let lhs = "a\nb\nc";
        let rhs = "#";
        let expected = "a#\nb#\nc#";

        assert_eq!(
            String32(lhs.into()) * String32(rhs.into()),
            String32(expected.into())
        );
    }

    #[cfg(feature = "large_string")]
    #[test]
    fn test_mul_string64_append_to_each_line() {
        let lhs = "x\ny\nz";
        let rhs = "@";
        let expected = "x@\ny@\nz@";

        assert_eq!(
            String64(lhs.into()) * String64(rhs.into()),
            String64(expected.into())
        );
    }

    #[test]
    fn test_pow_append_to_each_word() {
        let lhs = "a b c";
        let rhs = "!";

        assert_eq!(
            String32(lhs.into()).pow(String32(rhs.into())),
            String32("a! b! c!".into())
        );
    }

    #[cfg(feature = "large_string")]
    #[test]
    fn test_pow_string64_append_to_each_word() {
        let lhs = "one two";
        let rhs = "+";

        assert_eq!(
            String64(lhs.into()).pow(String64(rhs.into())),
            String64("one+ two+".into())
        );
    }

    #[cfg(feature = "large_string")]
    #[test]
    fn test_pow_mixed_string32_string64() {
        let lhs = "hi there";
        let rhs = "?";

        assert_eq!(
            String32(lhs.into()).pow(String64(rhs.into())),
            String64("hi? there?".into())
        );

        assert_eq!(
            String64(lhs.into()).pow(String32(rhs.into())),
            String64("hi? there?".into())
        );
    }

    #[test]
    fn test_pow_parse_string_as_float() {
        assert_eq!(String32("9.0".into()).pow(Float32(0.5)), Float64(3.0));
        assert_eq!(Float64(16.0).pow(String32("0.25".into())), Float64(2.0));
    }

    #[test]
    #[should_panic(expected = "Exponent: Cannot parse string as f64")]
    fn test_pow_parse_invalid_string_base() {
        let _ = String32("abc".into()).pow(Int32(2));
    }

    #[test]
    #[should_panic(expected = "Exponent: Cannot parse string as f64")]
    fn test_pow_parse_invalid_string_exponent() {
        let _ = Int32(2).pow(String32("notanumber".into()));
    }
}
