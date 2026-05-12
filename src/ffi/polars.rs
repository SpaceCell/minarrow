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

//! # **Polars Bridge** - *Adapter between Minarrow's C Data Interface and `polars` / `polars_arrow`*
//!
//! Thin reinterpret layer over [`crate::ffi::arrow_c_ffi::export_to_c`] and
//! [`crate::ffi::arrow_c_ffi::import_from_c_owned`]. polars_arrow's FFI structs
//! are layout-compatible with Minarrow's C ABI structs, so the export side is
//! a pointer cast plus a hand-off to `polars_arrow::ffi::import_array_from_c`,
//! and the import side is the symmetric step.
//!
//! `arrow_type_to_polars_dtype` exists because polars_arrow's
//! `import_array_from_c` needs the dtype as a separate argument rather than
//! decoding it from the schema's format string. We map our `ArrowType` enum
//! directly to avoid round-tripping through any unsupported FFI types.
//!
//! Used by `Array::to_polars`, `FieldArray::to_polars`, `Table::to_polars`
//! and their `from_*` siblings (and the `Super*` chunked equivalents).
//!
//! Gated by the `cast_polars` feature.
//!
//! ## Polars round-trip behaviour
//!
//! Most Arrow logical types pass through polars unchanged. A few logical types
//! are normalised by polars's internal representation on the way through. These
//! conversions are **lossless** - the same logical value is preserved, null
//! mask survives, and dtype/payload are rescaled together - but the type
//! label and byte payload differ on the return trip. Each is covered by
//! a dedicated round-trip test that asserts the promotion explicitly so any
//! future polars behaviour change surfaces immediately.
//!
//! | Sent into polars                    | Returned from polars                | Notes                                              |
//! |-------------------------------------|--------------------------------------|----------------------------------------------------|
//! | `Date64` (i64 ms-since-epoch)       | `Timestamp(Milliseconds, None)`     | Payload byte-identical (same i64 ms).              |
//! | `Time32(Seconds)` (i32)             | `Time64(Nanoseconds)` (i64)         | Payload rescaled: `n * 1_000_000_000`.             |
//! | `Time32(Milliseconds)` (i32)        | `Time64(Nanoseconds)` (i64)         | Payload rescaled: `n * 1_000_000`.                 |
//! | `Time64(Microseconds)` (i64)        | `Time64(Nanoseconds)` (i64)         | Payload rescaled: `n * 1_000`.                     |
//! | `String` / `LargeString` (utf8)     | may return as the other offset width| Polars normalises to its preferred offset width;   |
//! |                                     |                                     | bytes/text content preserved.                      |
//!
//! All other tested logical types - `Int8/16/32/64`, `UInt8/16/32/64`,
//! `Float32/64`, `Boolean`, `Date32`, `Time64(Nanoseconds)`, all `Timestamp`
//! units with and without a timezone, `Duration32`, `Duration64`, and
//! `Dictionary` (categorical) - return with their dtype and payload bytes
//! unchanged.
//!
//! ## Null masks
//!
//! Polars preserves the null mask in every direction we test, including
//! across the type promotions above. Validity bits sit on the underlying
//! Arrow array and travel through `Series::to_arrow` / `Series::from_arrow`
//! along with the data buffer.

use std::sync::Arc;

use polars::prelude::{CompatLevel, Series};

use crate::enums::error::MinarrowError;
use crate::ffi::arrow_c_ffi::{ArrowArray, ArrowSchema, export_to_c, import_from_c_owned};
use crate::ffi::arrow_dtype::ArrowType;
use crate::ffi::schema::Schema;
use crate::{Array, Field};

/// Export a Minarrow array to a polars `Series`.
///
/// `schema.fields[0]` supplies the logical type for the export. The Series
/// name is taken from `name` (Series carry their own name, separate from
/// schema field names).
pub fn export(
    array: Arc<Array>,
    name: &str,
    schema: Schema,
) -> Result<Series, MinarrowError> {
    let field_dtype = schema.fields[0].dtype.clone();
    let (c_arr, c_schema) = export_to_c(array, schema);

    // Move ArrowArray contents into polars_arrow ownership, then free the
    // boxes allocated by `export_to_c`. The schema box is not consumed by
    // polars_arrow (we build the dtype manually below), so we run its
    // release callback explicitly before freeing.
    let arr_ptr = c_arr as *mut polars_arrow::ffi::ArrowArray;
    let arr_val = unsafe { std::ptr::read(arr_ptr) };
    unsafe {
        drop(Box::from_raw(c_arr));
        if let Some(release) = (*c_schema).release {
            release(c_schema);
        }
        drop(Box::from_raw(c_schema));
    }

    let a2_dtype = arrow_type_to_polars_dtype(&field_dtype);

    let a2_array = unsafe {
        polars_arrow::ffi::import_array_from_c(arr_val, a2_dtype)
    }
    .map_err(|e| MinarrowError::BridgeError {
        source: "polars_arrow",
        message: format!("import_array_from_c: {e}"),
    })?;

    Ok(Series::from_arrow(name.into(), a2_array)?)
}

/// Import a polars `Series` into a Minarrow `(Arc<Array>, Field)`.
///
/// Reads chunk 0 of the Series. Callers wanting all chunks should use the
/// `SuperArray::from_polars` path which iterates over `s.n_chunks()`.
///
/// The recovered `Field` has its `name` overridden with `s.name()` so the
/// Series name round-trips cleanly.
pub fn import(s: &Series) -> Result<(Arc<Array>, Field), MinarrowError> {
    let arr2 = s.to_arrow(0, CompatLevel::oldest());
    import_chunk(s.name().as_str(), s.null_count() > 0, arr2)
}

/// Import a single polars_arrow `Array` (one chunk) with caller-supplied
/// name and nullability into a Minarrow `(Arc<Array>, Field)`.
///
/// Used by `SuperArray::from_polars` / `SuperTable::from_polars` to consume
/// per-chunk arrow arrays without rebuilding intermediate Series.
pub fn import_chunk(
    name: &str,
    nullable: bool,
    arr2: Box<dyn polars_arrow::array::Array>,
) -> Result<(Arc<Array>, Field), MinarrowError> {
    let dtype = arr2.dtype().clone();
    let pa_arr = polars_arrow::ffi::export_array_to_c(arr2);
    let pa_field =
        polars_arrow::datatypes::Field::new(name.into(), dtype, nullable);
    let pa_sch = polars_arrow::ffi::export_field_to_c(&pa_field);

    let arr_ptr = Box::into_raw(Box::new(pa_arr)) as *mut ArrowArray;
    let sch_ptr = Box::into_raw(Box::new(pa_sch)) as *mut ArrowSchema;
    let arr_box = unsafe { Box::from_raw(arr_ptr) };
    let sch_box = unsafe { Box::from_raw(sch_ptr) };

    let (array, mut field) = unsafe { import_from_c_owned(arr_box, sch_box) };
    field.name = name.to_string();
    Ok((array, field))
}

/// Maps a Minarrow `ArrowType` to the corresponding polars_arrow
/// `ArrowDataType`. Required because polars_arrow's
/// `import_array_from_c` takes the dtype as a parameter rather than
/// decoding it from the FFI schema's format string.
fn arrow_type_to_polars_dtype(dtype: &ArrowType) -> polars_arrow::datatypes::ArrowDataType {
    use crate::ffi::arrow_dtype::CategoricalIndexType;

    match dtype {
        ArrowType::Null => polars_arrow::datatypes::ArrowDataType::Null,
        ArrowType::Boolean => polars_arrow::datatypes::ArrowDataType::Boolean,

        #[cfg(feature = "extended_numeric_types")]
        ArrowType::Int8 => polars_arrow::datatypes::ArrowDataType::Int8,
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::Int16 => polars_arrow::datatypes::ArrowDataType::Int16,
        ArrowType::Int32 => polars_arrow::datatypes::ArrowDataType::Int32,
        ArrowType::Int64 => polars_arrow::datatypes::ArrowDataType::Int64,

        #[cfg(feature = "extended_numeric_types")]
        ArrowType::UInt8 => polars_arrow::datatypes::ArrowDataType::UInt8,
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::UInt16 => polars_arrow::datatypes::ArrowDataType::UInt16,
        ArrowType::UInt32 => polars_arrow::datatypes::ArrowDataType::UInt32,
        ArrowType::UInt64 => polars_arrow::datatypes::ArrowDataType::UInt64,

        ArrowType::Float32 => polars_arrow::datatypes::ArrowDataType::Float32,
        ArrowType::Float64 => polars_arrow::datatypes::ArrowDataType::Float64,

        ArrowType::String => polars_arrow::datatypes::ArrowDataType::Utf8,
        #[cfg(feature = "large_string")]
        ArrowType::LargeString => polars_arrow::datatypes::ArrowDataType::LargeUtf8,
        // Utf8View arrays are reshaped to regular Utf8 during the minarrow
        // import path, so map to Utf8 here.
        ArrowType::Utf8View => polars_arrow::datatypes::ArrowDataType::Utf8,

        #[cfg(feature = "datetime")]
        ArrowType::Date32 => polars_arrow::datatypes::ArrowDataType::Date32,
        #[cfg(feature = "datetime")]
        ArrowType::Date64 => polars_arrow::datatypes::ArrowDataType::Date64,

        #[cfg(feature = "datetime")]
        ArrowType::Time32(u) => {
            polars_arrow::datatypes::ArrowDataType::Time32(match u {
                crate::TimeUnit::Seconds => polars_arrow::datatypes::TimeUnit::Second,
                crate::TimeUnit::Milliseconds => {
                    polars_arrow::datatypes::TimeUnit::Millisecond
                }
                _ => panic!("Time32 supports Seconds or Milliseconds only"),
            })
        }
        #[cfg(feature = "datetime")]
        ArrowType::Time64(u) => {
            polars_arrow::datatypes::ArrowDataType::Time64(match u {
                crate::TimeUnit::Microseconds => {
                    polars_arrow::datatypes::TimeUnit::Microsecond
                }
                crate::TimeUnit::Nanoseconds => {
                    polars_arrow::datatypes::TimeUnit::Nanosecond
                }
                _ => panic!("Time64 supports Microseconds or Nanoseconds only"),
            })
        }
        #[cfg(feature = "datetime")]
        ArrowType::Duration32(u) => {
            polars_arrow::datatypes::ArrowDataType::Duration(match u {
                crate::TimeUnit::Seconds => polars_arrow::datatypes::TimeUnit::Second,
                crate::TimeUnit::Milliseconds => {
                    polars_arrow::datatypes::TimeUnit::Millisecond
                }
                _ => panic!("Duration32 supports Seconds or Milliseconds only"),
            })
        }
        #[cfg(feature = "datetime")]
        ArrowType::Duration64(u) => {
            polars_arrow::datatypes::ArrowDataType::Duration(match u {
                crate::TimeUnit::Microseconds => {
                    polars_arrow::datatypes::TimeUnit::Microsecond
                }
                crate::TimeUnit::Nanoseconds => {
                    polars_arrow::datatypes::TimeUnit::Nanosecond
                }
                _ => panic!("Duration64 supports Microseconds or Nanoseconds only"),
            })
        }
        #[cfg(feature = "datetime")]
        ArrowType::Timestamp(u, tz) => polars_arrow::datatypes::ArrowDataType::Timestamp(
            match u {
                crate::TimeUnit::Seconds => polars_arrow::datatypes::TimeUnit::Second,
                crate::TimeUnit::Milliseconds => {
                    polars_arrow::datatypes::TimeUnit::Millisecond
                }
                crate::TimeUnit::Microseconds => {
                    polars_arrow::datatypes::TimeUnit::Microsecond
                }
                crate::TimeUnit::Nanoseconds => {
                    polars_arrow::datatypes::TimeUnit::Nanosecond
                }
                crate::TimeUnit::Days => panic!("Timestamp(Days) is invalid"),
            },
            tz.as_ref().map(|s| s.as_str().into()),
        ),
        #[cfg(feature = "datetime")]
        ArrowType::Interval(iu) => {
            polars_arrow::datatypes::ArrowDataType::Interval(match iu {
                crate::IntervalUnit::YearMonth => {
                    polars_arrow::datatypes::IntervalUnit::YearMonth
                }
                crate::IntervalUnit::DaysTime => {
                    polars_arrow::datatypes::IntervalUnit::DayTime
                }
                crate::IntervalUnit::MonthDaysNs => {
                    polars_arrow::datatypes::IntervalUnit::MonthDayNano
                }
            })
        }

        ArrowType::Dictionary(idx) => {
            let key: polars_arrow::datatypes::IntegerType = match idx {
                #[cfg(feature = "default_categorical_8")]
                CategoricalIndexType::UInt8 => {
                    polars_arrow::datatypes::IntegerType::UInt8
                }
                #[cfg(feature = "extended_categorical")]
                CategoricalIndexType::UInt16 => {
                    polars_arrow::datatypes::IntegerType::UInt16
                }
                #[cfg(any(
                    not(feature = "default_categorical_8"),
                    feature = "extended_categorical"
                ))]
                CategoricalIndexType::UInt32 => {
                    polars_arrow::datatypes::IntegerType::UInt32
                }
                #[cfg(feature = "extended_categorical")]
                CategoricalIndexType::UInt64 => {
                    polars_arrow::datatypes::IntegerType::UInt64
                }
            };
            polars_arrow::datatypes::ArrowDataType::Dictionary(
                key,
                Box::new(polars_arrow::datatypes::ArrowDataType::Utf8),
                false,
            )
        }
    }
}
