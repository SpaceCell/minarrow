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

//! # **Field Module** - *Arrow-compliant Column Metadata Tagging*
//!
//! Defines column-level schema metadata for `Minarrow`.
//!
//! A `Field` captures a column’s name, logical Arrow data type,
//! nullability, and optional lightweight metadata.  
//!
//! Primarily used in table schemas and during Arrow FFI export to ensure
//! correct logical typing (especially for temporal data).  
//!
//! This module contains only the schema description - it does not hold
//! any row data. Pair with `FieldArray` to bind a schema to actual values.

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "datetime")]
use crate::TemporalArray;
use crate::ffi::arrow_dtype::{ArrowType, CategoricalIndexType};
use crate::{Array, MaskedArray, NumericArray, TextArray};

/// Global counter for unnamed fields
static UNNAMED_FIELD_COUNTER: AtomicUsize = AtomicUsize::new(1);

/// # Field
///
/// ## Description
/// `Field` struct supporting:
/// - Array metadata such as type, name, nullability, etc.
/// - Light metadata, e.g. a few key-value pairs.
/// - Later `Schema` construction during Arrow FFI.
///
/// ### Tips:
/// - `Field` is *cloned often*, so it is best kept any metadata
///   lightweight to avoid performance penalties. `SuperTable` wraps it in Arc.
/// - For `Datetime` arrays, `Field` carries the logical `Arrow` type.
///   The physical type remains a single integer-backed `Datetime`, while
///   the logical type specifies its intended semantics.
///     i.e.:
///     `Date32`
///     `Date64`
///     `Time32(TimeUnit)`
///     `Time64(TimeUnit)`
///     `Duration32(TimeUnit)`
///     `Duration64(TimeUnit)`
///     `Timestamp(TimeUnit)`
///     `Interval(IntervalUnit)`
///
/// - This ensures that when sent over Arrow C-FFI (or `to_apache_arrow()`),
/// it converts to the correct external type. Whilst, avoiding proliferating many
/// specialised types prematurely, keeping the API and binary size minimal.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Field {
    pub name: String,
    pub dtype: ArrowType,
    pub nullable: bool,
    pub metadata: BTreeMap<String, String>,
}

impl Field {
    /// Constructs a new `Field`. If the provided name is empty or only whitespace,
    /// a globally unique name like `UnnamedField1` will generate.
    pub fn new<T: Into<String>>(
        name: T,
        dtype: ArrowType,
        nullable: bool,
        metadata: Option<BTreeMap<String, String>>,
    ) -> Self {
        let mut name = name.into();
        if name.trim().is_empty() {
            let id = UNNAMED_FIELD_COUNTER.fetch_add(1, Ordering::Relaxed);
            name = format!("UnnamedField{}", id);
        }

        Field {
            name,
            dtype,
            nullable,
            metadata: metadata.unwrap_or_default(),
        }
    }

    /// Constructs a new `Field` from an `Array` enum instance.
    /// Derives the dtype and nullability directly from the inner array.
    ///
    /// For `Duration`, `Time`, `Timestamp` and `Interval` types, use `Field::new()`.
    pub fn from_array(
        name: impl Into<String>,
        array: &Array,
        metadata: Option<BTreeMap<String, String>>,
    ) -> Self {
        let name = name.into();
        let metadata = metadata.unwrap_or_default();

        match array {
            Array::NumericArray(inner) => match inner {
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(a) => {
                    Field::new(name, ArrowType::Int8, a.is_nullable(), Some(metadata))
                }
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(a) => {
                    Field::new(name, ArrowType::Int16, a.is_nullable(), Some(metadata))
                }
                NumericArray::Int32(a) => {
                    Field::new(name, ArrowType::Int32, a.is_nullable(), Some(metadata))
                }
                NumericArray::Int64(a) => {
                    Field::new(name, ArrowType::Int64, a.is_nullable(), Some(metadata))
                }
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(a) => {
                    Field::new(name, ArrowType::UInt8, a.is_nullable(), Some(metadata))
                }
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(a) => {
                    Field::new(name, ArrowType::UInt16, a.is_nullable(), Some(metadata))
                }
                NumericArray::UInt32(a) => {
                    Field::new(name, ArrowType::UInt32, a.is_nullable(), Some(metadata))
                }
                NumericArray::UInt64(a) => {
                    Field::new(name, ArrowType::UInt64, a.is_nullable(), Some(metadata))
                }
                NumericArray::Float32(a) => {
                    Field::new(name, ArrowType::Float32, a.is_nullable(), Some(metadata))
                }
                NumericArray::Float64(a) => {
                    Field::new(name, ArrowType::Float64, a.is_nullable(), Some(metadata))
                }
                NumericArray::Null => Field::new(name, ArrowType::Null, false, Some(metadata)),
            },
            Array::BooleanArray(a) => {
                Field::new(name, ArrowType::Boolean, a.is_nullable(), Some(metadata))
            }
            Array::TextArray(inner) => match inner {
                TextArray::String32(a) => {
                    Field::new(name, ArrowType::String, a.is_nullable(), Some(metadata))
                }
                #[cfg(feature = "large_string")]
                TextArray::String64(a) => Field::new(
                    name,
                    ArrowType::LargeString,
                    a.is_nullable(),
                    Some(metadata),
                ),
                #[cfg(feature = "default_categorical_8")]
                TextArray::Categorical8(a) => Field::new(
                    name,
                    ArrowType::Dictionary(CategoricalIndexType::UInt8),
                    a.is_nullable(),
                    Some(metadata),
                ),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(a) => Field::new(
                    name,
                    ArrowType::Dictionary(CategoricalIndexType::UInt16),
                    a.is_nullable(),
                    Some(metadata),
                ),
                #[cfg(any(
                    not(feature = "default_categorical_8"),
                    feature = "extended_categorical"
                ))]
                TextArray::Categorical32(a) => Field::new(
                    name,
                    ArrowType::Dictionary(CategoricalIndexType::UInt32),
                    a.is_nullable(),
                    Some(metadata),
                ),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(a) => Field::new(
                    name,
                    ArrowType::Dictionary(CategoricalIndexType::UInt64),
                    a.is_nullable(),
                    Some(metadata),
                ),
                TextArray::Null => Field::new(name, ArrowType::Null, false, Some(metadata)),
            },
            #[cfg(feature = "datetime")]
            Array::TemporalArray(inner) => match inner {
                TemporalArray::Datetime32(a) => {
                    println!(
                        "Warning: Datetime requires creating fields via `Field::new` and setting the desired arrow logical type.\nSetting ArrowType::Date32. If you need a `Timestamp`, `Duration`, or `Time` field, please use `Field::new`."
                    );
                    return Field::new(name, ArrowType::Date32, a.is_nullable(), Some(metadata));
                }
                TemporalArray::Datetime64(a) => {
                    println!(
                        "Warning: Datetime requires creating fields via `Field::new` and setting the desired arrow logical type.\nSetting ArrowType::Date64. If you need a `Timestamp`, `Duration`, or `Time` field, please use `Field::new`."
                    );
                    Field::new(name, ArrowType::Date64, a.is_nullable(), Some(metadata))
                }
                TemporalArray::Null => Field::new(name, ArrowType::Null, false, Some(metadata)),
            },
            Array::Null => Field::new(name, ArrowType::Null, false, Some(metadata)),
        }
    }
}

impl Display for Field {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Field \"{}\": {}{}",
            self.name,
            self.dtype,
            if self.nullable { " (nullable)" } else { "" }
        )?;

        if !self.metadata.is_empty() {
            write!(f, " [metadata: ")?;
            for (i, (k, v)) in self.metadata.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}=\"{}\"", k, v)?;
            }
            write!(f, "]")?;
        }

        Ok(())
    }
}

impl From<Arc<Field>> for Field {
    fn from(arc: Arc<Field>) -> Self {
        Arc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::arrow_dtype::ArrowType;

    #[test]
    fn test_field_new_and_metadata() {
        let field = Field::new("foo", ArrowType::String, true, None);
        assert_eq!(field.name, "foo");
        assert_eq!(field.dtype, ArrowType::String);
        assert!(field.metadata.is_empty());

        let mut meta = BTreeMap::new();
        meta.insert("k".to_string(), "v".to_string());
        let field2 = Field::new("bar", ArrowType::Int64, false, Some(meta.clone()));
        assert_eq!(field2.metadata, meta);
    }

    #[test]
    fn test_field_unnamed_autonaming() {
        let f1 = Field::new("", ArrowType::Int32, false, None);
        let f2 = Field::new("   ", ArrowType::Int32, false, None);
        assert!(f1.name.starts_with("UnnamedField"));
        assert!(f2.name.starts_with("UnnamedField"));
        assert_ne!(f1.name, f2.name);
    }
}
