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

//! # **Error Module** - Custom *Minarrow* Error Type
//!
//! Defines the unified error type for Minarrow.
//!
//! Also includes a KernelError type for this crate and downstream SIMD-kernels
//!
//! ## Covers
//! - Array length mismatches, overflow, lossy casts, null handling,
//! type incompatibility, and invalid conversions.  
//! - Implements `Display` for readable output and `Error` for integration
//! with standard Rust error handling.

use std::error::Error;
use std::fmt;

#[cfg(feature = "shared_dict")]
use crate::structs::dictionary::DictionaryError;

/// Catch all error types for `Minarrow`
#[derive(Debug, PartialEq)]
pub enum MinarrowError {
    ColumnLengthMismatch {
        col: usize,
        expected: usize,
        found: usize,
    },
    Overflow {
        value: String,
        target: &'static str,
    },
    LossyCast {
        value: String,
        target: &'static str,
    },
    TypeError {
        from: &'static str,
        to: &'static str,
        message: Option<String>,
    },
    NullError {
        message: Option<String>,
    },
    IncompatibleTypeError {
        from: &'static str,
        to: &'static str,
        message: Option<String>,
    },
    KernelError(Option<String>),
    ShapeError {
        message: String,
    },
    NotImplemented {
        feature: String,
    },
    IndexError(String),
    /// External bridge failure surfaced by `to_apache_arrow` / `to_polars` /
    /// `from_apache_arrow` / `from_polars` or any of their `try_*` siblings.
    /// `source` identifies the foreign library (e.g. `"arrow-rs"`, `"polars"`)
    /// so the same variant can carry errors from either bridge.
    BridgeError {
        source: &'static str,
        message: String,
    },
    /// Raised when a categorical operation would exceed the index type's
    /// cardinality, e.g. a 257th unique value on a `u8` categorical.
    DictionaryOverflow,
}

#[cfg(feature = "shared_dict")]
impl From<DictionaryError> for MinarrowError {
    fn from(source: DictionaryError) -> Self {
        match source {
            DictionaryError::Overflow => MinarrowError::DictionaryOverflow,
        }
    }
}

impl fmt::Display for MinarrowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MinarrowError::ColumnLengthMismatch {
                col,
                expected,
                found,
            } => {
                write!(
                    f,
                    "Column length mismatch in column {}: expected {}, found {}.",
                    col, expected, found
                )
            }
            MinarrowError::Overflow { value, target } => {
                write!(
                    f,
                    "Overflow: value '{}' cannot be represented in type '{}'.",
                    value, target
                )
            }
            MinarrowError::LossyCast { value, target } => {
                write!(
                    f,
                    "Lossy cast: value '{}' loses precision or cannot be exactly represented as '{}'.",
                    value, target
                )
            }
            MinarrowError::TypeError { from, to, message } => {
                if let Some(msg) = message {
                    write!(
                        f,
                        "Type error: cannot cast from '{}' to '{}': {}",
                        from, to, msg
                    )
                } else {
                    write!(f, "Type error: cannot cast from '{}' to '{}'.", from, to)
                }
            }
            MinarrowError::NullError { message } => {
                if let Some(msg) = message {
                    write!(f, "Null error: {}", msg)
                } else {
                    write!(f, "Null error: nulls cannot be represented in target type.")
                }
            }
            MinarrowError::IncompatibleTypeError { from, to, message } => {
                if let Some(msg) = message {
                    write!(
                        f,
                        "Incompatible type error: cannot convert from '{}' to '{}': {}",
                        from, to, msg
                    )
                } else {
                    write!(
                        f,
                        "Incompatible type error: cannot convert from '{}' to '{}'.",
                        from, to
                    )
                }
            }
            MinarrowError::KernelError(message) => {
                if let Some(msg) = message {
                    write!(f, "Kernel error: {}", msg)
                } else {
                    write!(f, "Kernel error")
                }
            }
            MinarrowError::ShapeError { message } => {
                write!(f, "Shape error: {}", message)
            }
            MinarrowError::NotImplemented { feature } => {
                write!(f, "Not implemented: {}", feature)
            }
            MinarrowError::IndexError(message) => {
                write!(f, "Index error: {}", message)
            }
            MinarrowError::BridgeError { source, message } => {
                write!(f, "Bridge error ({}): {}", source, message)
            }
            MinarrowError::DictionaryOverflow => {
                write!(
                    f,
                    "Categorical dictionary overflow: new value would exceed the index type's cardinality"
                )
            }
        }
    }
}

impl Error for MinarrowError {}

#[cfg(feature = "cast_arrow")]
impl From<arrow::error::ArrowError> for MinarrowError {
    fn from(e: arrow::error::ArrowError) -> Self {
        MinarrowError::BridgeError {
            source: "arrow-rs",
            message: e.to_string(),
        }
    }
}

#[cfg(feature = "cast_polars")]
impl From<polars::error::PolarsError> for MinarrowError {
    fn from(e: polars::error::PolarsError) -> Self {
        MinarrowError::BridgeError {
            source: "polars",
            message: e.to_string(),
        }
    }
}

/// Error type for all kernel operations.
///
/// Each variant includes a contextual message string providing specific details
/// about the error condition, enabling precise debugging and error reporting.
#[derive(Debug, Clone)]
pub enum KernelError {
    /// Data type mismatch between operands or unsupported type combinations.
    TypeMismatch(String),

    /// Array length mismatch between operands.
    LengthMismatch(String),

    /// Broadcast Error often due to data structure shape.
    BroadcastingError(String),

    /// Invalid operator for the given operands or context.
    OperatorMismatch(String),

    /// Unsupported data type for the requested operation.
    UnsupportedType(String),

    /// Column or field not found in structured data.
    ColumnNotFound(String),

    /// Invalid arguments provided to kernel function.
    InvalidArguments(String),

    /// Planning or configuration error.
    Plan(String),

    /// Array index or memory access out of bounds.
    OutOfBounds(String),

    /// Division by zero or similar mathematical errors.
    DivideByZero(String),
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelError::TypeMismatch(msg) => write!(f, "Type mismatch: {}", msg),
            KernelError::LengthMismatch(msg) => write!(f, "Length mismatch: {}", msg),
            KernelError::OperatorMismatch(msg) => write!(f, "Operator mismatch: {}", msg),
            KernelError::BroadcastingError(msg) => write!(f, "Shape Error: {}", msg),
            KernelError::UnsupportedType(msg) => write!(f, "Unsupported type: {}", msg),
            KernelError::ColumnNotFound(msg) => write!(f, "Column not found: {}", msg),
            KernelError::InvalidArguments(msg) => write!(f, "Invalid arguments: {}", msg),
            KernelError::Plan(msg) => write!(f, "Planning error: {}", msg),
            KernelError::OutOfBounds(msg) => write!(f, "Out of bounds: {}", msg),
            KernelError::DivideByZero(msg) => write!(f, "Divide by Zero error: {}", msg),
        }
    }
}

impl Error for KernelError {}

impl From<KernelError> for MinarrowError {
    fn from(err: KernelError) -> Self {
        MinarrowError::KernelError(Some(err.to_string()))
    }
}

/// Creates a formatted error message for length mismatches between left-hand side (LHS) and right-hand side (RHS) arrays.
///
/// # Arguments
/// * `fname` - Function name where the mismatch occurred
/// * `lhs` - Length of the left-hand side array
/// * `rhs` - Length of the right-hand side array
///
/// # Returns
/// A formatted error message string
pub fn log_length_mismatch(fname: String, lhs: usize, rhs: usize) -> String {
    return format!("{} => Length mismatch: LHS {} RHS {}", fname, lhs, rhs);
}
