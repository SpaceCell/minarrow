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

//! # **FieldArray Module** - *De-Facto *Column* Array type w' Tagged Arrow Metadata*
//!
//! Couples a `Field` (i.e., array-level schema metadata) with an immutable `Array` of values.
//!
//! Used as the primary column representation in `Minarrow` tables, ensuring
//! schema and data remain consistent.
//!
//! Supports creation from raw components or by inferring schema from arrays,
//! and is the unit transferred over Arrow FFI or to external libraries
//! such as Apache Arrow or Polars.
//!
//! ## Apache Arrow / Polars bridges (`cast_arrow` / `cast_polars` features)
//!
//! - `to_apache_arrow()` / `to_polars()` export with the stored `Field`, so
//!   Timestamp/Time/Duration/Interval logical types round-trip correctly.
//! - `from_apache_arrow(name, &ArrayRef)` and `from_polars(&Series)` recover
//!   dtype, nullability, and any custom field metadata from the FFI schema.
//!   For polars, the column name comes from `s.name()`.
//! - Each panicking method has a `try_*` sibling returning
//!   `Result<_, MinarrowError>`. `&Series` can also be converted via
//!   `(&series).into()` for ergonomic call sites.

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

#[cfg(feature = "cast_arrow")]
use arrow::array::ArrayRef;
#[cfg(feature = "cast_polars")]
use polars::series::Series;

#[cfg(all(feature = "select", feature = "views"))]
use crate::ArrayV;
#[cfg(feature = "views")]
use crate::aliases::FieldAVT;
use crate::enums::error::MinarrowError;
use crate::enums::shape_dim::ShapeDim;
use crate::ffi::arrow_dtype::ArrowType;
use crate::ffi::arrow_dtype::CategoricalIndexType;
use crate::traits::concatenate::Concatenate;
#[cfg(all(feature = "select", feature = "views"))]
use crate::traits::selection::{DataSelector, RowSelection};
use crate::traits::shape::Shape;
use crate::{Array, Field, NumericArray, TextArray};
#[cfg(feature = "datetime")]
use crate::{TemporalArray, TimeUnit};

/// # FieldArray
///
/// Named and typed data column with associated array values.
///
/// ## Role
/// - Combines a `Field` with an immutable `Array` instance.
/// - `FieldArray` integrates naturally into a `Table`, where immutability enforces row-length guarantees.
/// It can also serve as a self-documenting array and is required when sending `Minarrow` data
/// over FFI to `Apache Arrow`. In such cases, it's worth ensuring the correct logical `Datetime` Arrow type
/// is built when constructing the `Field`, as this determines the `Arrow` type on the receiving side.
///
/// ## Examples
/// ```rust
/// use minarrow::{Array, Field, FieldArray, MaskedArray};
/// use minarrow::ffi::arrow_dtype::ArrowType;
/// use minarrow::structs::variants::integer::IntegerArray;
///
/// // Build a typed array
/// let mut ints = IntegerArray::<i32>::default();
/// ints.push(1);
/// ints.push(2);
/// let arr = Array::from_int32(ints);
///
/// // Construct with a Field and Array
/// let field = Field::new("id", ArrowType::Int32, false, None);
/// let fa = FieldArray::new(field, arr);
///
/// assert_eq!(fa.field.name, "id");
/// assert_eq!(fa.arrow_type(), ArrowType::Int32);
/// assert_eq!(fa.len(), 2);
///
/// // Take an owned slice [offset..offset+len)
/// let sub = fa.slice_clone(0, 1);
/// assert_eq!(sub.len(), 1);
/// ```
///
/// The `fa_*` macros are a concise construction shorthand:
/// ```rust
/// use minarrow::{fa_i32, fa_i32_opt, fa_str32};
/// use minarrow::ffi::arrow_dtype::ArrowType;
///
/// // From literal values - type and nullability inferred
/// let fa = fa_i32!("id", 1, 2);
/// assert_eq!(fa.field.name, "id");
/// assert_eq!(fa.arrow_type(), ArrowType::Int32);
/// assert_eq!(fa.len(), 2);
///
/// // Nullable variant with Option values
/// let fa = fa_i32_opt!("score", Some(10i32), None::<i32>, Some(30));
/// assert_eq!(fa.len(), 3);
/// assert!(fa.field.nullable);
/// assert_eq!(fa.null_count(), 1);
///
/// // String columns
/// let names = fa_str32!("name", "alice", "bob");
/// assert_eq!(names.field.name, "name");
/// assert_eq!(names.len(), 2);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FieldArray {
    /// Array metadata
    pub field: Arc<Field>,

    /// The array's inner payload is wrapped in Arc for immutability
    /// so it can safely share across threads.
    /// When part of a Table *(or higher-dimensional structure)*,
    /// immutability also upholds shape constraints.
    pub array: Array,

    /// Null count for the immutable array to support skipping null-mask
    /// operations when it's `0`, and/or related strategies.
    pub null_count: usize,
}

impl FieldArray {
    /// Constructs a new `FieldArray` from an existing `Field` and `Array`.
    pub fn new(field: Field, array: Array) -> Self {
        let null_count = array.null_count();
        FieldArray {
            field: field.into(),
            array,
            null_count,
        }
    }

    /// Constructs a new `FieldArray` from an existing `Arc<Field>` and `Array`.
    pub fn new_arc(field: Arc<Field>, array: Array) -> Self {
        let null_count = array.null_count();
        FieldArray {
            field: field,
            array,
            null_count,
        }
    }

    /// Constructs a new `FieldArray` from a name and any supported typed array,
    /// automatically wrapping as `Array` and inferring type and nullability.
    pub fn from_arr<N, A>(name: N, arr: A) -> Self
    where
        N: Into<String>,
        A: Into<Array>,
    {
        let array: Array = arr.into();
        let dtype = array.arrow_type();
        let nullable = array.is_nullable();
        let field = Field::new(name, dtype, nullable, None);
        FieldArray::new(field, array)
    }

    /// Constructs a new `FieldArray` from raw field components and an `Array`.
    pub fn from_parts<T: Into<String>>(
        field_name: T,
        dtype: ArrowType,
        nullable: Option<bool>,
        metadata: Option<BTreeMap<String, String>>,
        array: Array,
    ) -> Self {
        let null_count = array.null_count();
        let field = Field {
            name: field_name.into(),
            dtype,
            nullable: nullable.unwrap_or_else(|| array.is_nullable()),
            metadata: metadata.unwrap_or_default(),
        };
        FieldArray {
            field: field.into(),
            array: array.into(),
            null_count,
        }
    }

    pub fn len(&self) -> usize {
        self.array.len()
    }

    /// Removes the rows in `[start, end)`, shifting later rows left, and
    /// refreshes the cached null count.
    ///
    /// # Panics
    /// Panics if `start > end` or `end > len`.
    pub fn delete_range(&mut self, start: usize, end: usize) {
        self.array.delete_range(start, end);
        self.null_count = self.array.null_count();
    }

    pub fn is_empty(&self) -> bool {
        self.array.len() == 0
    }

    pub fn arrow_type(&self) -> ArrowType {
        self.field.dtype.clone()
    }

    /// Returns a new FieldArray with updated timezone metadata.
    ///
    /// The underlying timestamp data (always UTC) remains unchanged. Only the timezone
    /// metadata in the Field's ArrowType is updated for interpretation/display purposes.
    ///
    /// # Arguments
    /// * `tz` - Timezone string in Arrow format (IANA like "America/New_York" or offset like "+05:00")
    ///
    /// # Errors
    /// Returns an error if the array is not a Timestamp type.
    #[cfg(feature = "datetime")]
    pub fn tz(&self, tz: &str) -> Result<Self, MinarrowError> {
        match &self.field.dtype {
            ArrowType::Timestamp(unit, _) => {
                let mut new_field = (*self.field).clone();
                new_field.dtype = ArrowType::Timestamp(*unit, Some(tz.to_string()));
                Ok(FieldArray {
                    field: Arc::new(new_field),
                    array: self.array.clone(),
                    null_count: self.null_count,
                })
            }
            _ => Err(MinarrowError::TypeError {
                from: "FieldArray",
                to: "Timestamp",
                message: Some("tz() requires a Timestamp type".to_string()),
            }),
        }
    }

    /// Returns a new FieldArray with timezone metadata set to "UTC".
    ///
    /// The underlying timestamp data (always UTC) remains unchanged. Only the timezone
    /// metadata in the Field's ArrowType is updated.
    ///
    /// # Errors
    /// Returns an error if the array is not a Timestamp type.
    #[cfg(feature = "datetime")]
    pub fn utc(&self) -> Result<Self, MinarrowError> {
        self.tz("UTC")
    }

    /// Returns a zero-copy view (`FieldArraySlice`) into the window `[offset, offset+len)`.
    ///
    /// The returned object holds references into the original `FieldArray`.
    ///
    /// The `(&Array, Offset, WindowLength), &Field)` `FieldArraySlice` pattern here
    /// is a once-off we avoid recommending.
    #[cfg(feature = "views")]
    #[inline]
    pub fn view(&self, offset: usize, len: usize) -> FieldAVT<'_> {
        ((&self.array, offset, len), &self.field)
    }

    /// Returns a new owned FieldArray with array sliced `[offset, offset+len)`.
    pub fn slice_clone(&self, offset: usize, len: usize) -> Self {
        let array: Array = self.array.slice_clone(offset, len).into();
        let null_count = array.null_count();
        FieldArray {
            field: self.field.clone(),
            array: array.into(),
            null_count,
        }
    }

    /// Updates the cached null_count from the underlying array.
    /// Should be called after any mutation of the array that could change null count.
    #[inline]
    pub fn refresh_null_count(&mut self) {
        self.null_count = self.array.null_count();
    }

    /// Returns the cached null count.
    /// This is kept in sync with the underlying array via refresh_null_count().
    #[inline]
    pub fn null_count(&self) -> usize {
        self.null_count
    }

    /// Concatenates another FieldArray's data into this one using copy-on-write semantics.
    /// If this FieldArray's array has Arc reference count > 1, the data is cloned first.
    /// Both FieldArrays must have compatible types. Updates the cached null_count.
    pub fn concat_field_array(&mut self, other: &FieldArray) {
        self.array.concat_array(&other.array);
        self.refresh_null_count();
    }

    /// Appends rows `[offset..offset+len)` from another FieldArray into self.
    /// Extends data directly from the source's backing buffer.
    pub fn concat_range(&mut self, other: &FieldArray, offset: usize, len: usize) -> Result<(), MinarrowError> {
        self.array.concat_array_range(&other.array, offset, len)?;
        self.refresh_null_count();
        Ok(())
    }

    /// Provides mutable access to the underlying array with automatic null_count refresh.
    /// Uses copy-on-write semantics - clones array data if Arc reference count > 1.
    /// Use this for operations that may change the null count.
    pub fn with_array_mut<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Array) -> R,
    {
        let result = f(&mut self.array);
        self.refresh_null_count();
        result
    }

    /// Export this field+array over Arrow C-FFI and import into arrow-rs.
    ///
    /// Carries the stored `Field` through the conversion, preserving logical
    /// types such as Timestamp/Time/Duration/Interval.
    ///
    /// Panics on FFI failure. For a fallible variant, see
    /// [`FieldArray::try_to_apache_arrow`].
    #[cfg(feature = "cast_arrow")]
    #[inline]
    pub fn to_apache_arrow(&self) -> ArrayRef {
        self.try_to_apache_arrow()
            .expect("FieldArray::to_apache_arrow failed")
    }

    /// Fallible variant of [`FieldArray::to_apache_arrow`].
    #[cfg(feature = "cast_arrow")]
    #[inline]
    pub fn try_to_apache_arrow(&self) -> Result<ArrayRef, MinarrowError> {
        crate::ffi::arrow_rs::export(
            Arc::new(self.array.clone()),
            crate::ffi::schema::Schema::from(vec![(*self.field).clone()]),
        )
    }

    // ** The below polars function is tested tests/polars.rs **

    /// Casts the FieldArray to a polars Series, preserving the stored `Field`.
    ///
    /// Panics on FFI failure. For a fallible variant, see
    /// [`FieldArray::try_to_polars`].
    #[cfg(feature = "cast_polars")]
    pub fn to_polars(&self) -> Series {
        self.try_to_polars().expect("FieldArray::to_polars failed")
    }

    /// Fallible variant of [`FieldArray::to_polars`].
    #[cfg(feature = "cast_polars")]
    pub fn try_to_polars(&self) -> Result<Series, MinarrowError> {
        crate::ffi::polars::export(
            Arc::new(self.array.clone()),
            self.field.name.as_str(),
            crate::ffi::schema::Schema::from(vec![(*self.field).clone()]),
        )
    }

    // ===========================================================
    // Apache Arrow / Polars import (`from_*`)
    // ===========================================================

    /// Build a `FieldArray` from an arrow-rs `ArrayRef` and a column name.
    ///
    /// arrow-rs `ArrayRef` does not carry a column name (only `Schema`/`Field` do),
    /// so the caller must provide one. Dtype, nullability, and any custom metadata
    /// are recovered from the FFI schema.
    ///
    /// Panics on FFI failure. For a fallible variant, see
    /// [`FieldArray::try_from_apache_arrow`].
    #[cfg(feature = "cast_arrow")]
    #[inline]
    pub fn from_apache_arrow(
        name: impl Into<String>,
        arr: &arrow::array::ArrayRef,
    ) -> FieldArray {
        Self::try_from_apache_arrow(name, arr)
            .expect("FieldArray::from_apache_arrow failed")
    }

    /// Fallible variant of [`FieldArray::from_apache_arrow`].
    #[cfg(feature = "cast_arrow")]
    pub fn try_from_apache_arrow(
        name: impl Into<String>,
        arr: &arrow::array::ArrayRef,
    ) -> Result<FieldArray, MinarrowError> {
        let (array_arc, mut field) = crate::ffi::arrow_rs::import(arr)?;
        field.name = name.into();
        let array = Arc::try_unwrap(array_arc).unwrap_or_else(|arc| (*arc).clone());
        Ok(FieldArray::new(field, array))
    }

    /// Build a `FieldArray` from a Polars `Series`.
    ///
    /// A polars `Series` is inherently multi-chunked; the canonical mapping
    /// is `Series` <-> [`crate::SuperArray`]. This helper routes through
    /// [`crate::SuperArray::from_polars`] and then **consolidates** the
    /// chunks into a single contiguous, 64-byte aligned buffer.
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
    /// when importing polars data into a `FieldArray`. If you would like to
    /// preserve the original chunk boundaries and avoid the consolidation
    /// step, use [`crate::SuperArray::from_polars`] directly - though the
    /// alignment copy will still occur per chunk that isn't pre-aligned.
    ///
    /// Name comes from `s.name()`; dtype, nullability, and any custom
    /// metadata are recovered from the FFI schema.
    ///
    /// Panics on FFI failure. For a fallible variant, see
    /// [`FieldArray::try_from_polars`].
    #[cfg(feature = "cast_polars")]
    #[inline]
    pub fn from_polars(s: &polars::prelude::Series) -> FieldArray {
        Self::try_from_polars(s).expect("FieldArray::from_polars failed")
    }

    /// Fallible variant of [`FieldArray::from_polars`].
    #[cfg(feature = "cast_polars")]
    pub fn try_from_polars(
        s: &polars::prelude::Series,
    ) -> Result<FieldArray, MinarrowError> {
        use crate::SuperArray;
        use crate::traits::consolidate::Consolidate;
        let sa = SuperArray::try_from_polars(s)?;
        let field = sa.field_ref().clone();
        let array = sa.consolidate();
        Ok(FieldArray::new(field, array))
    }
}

// Provides `(&series).into()` ergonomics for building a FieldArray.
// Panics on FFI failure - users who want the fallible flavour should call
// the named `FieldArray::try_from_polars(&s)?` method instead.
#[cfg(feature = "cast_polars")]
impl From<&polars::prelude::Series> for FieldArray {
    fn from(s: &polars::prelude::Series) -> Self {
        FieldArray::from_polars(s)
    }
}

/// Creates a new basic field array based on a name and an existing array
pub fn field_array<T: Into<String>>(name: T, array: Array) -> FieldArray {
    let dtype = array.arrow_type();
    let nullable = array.is_nullable();
    let field = Field::new(name, dtype, nullable, None);
    FieldArray::new(field, array)
}

/// Helper to create a proper Field for an Array with correct type, mask, and metadata
pub fn create_field_for_array(
    name: &str,
    array: &Array,
    other_array: Option<&Array>,
    metadata: Option<std::collections::BTreeMap<String, String>>,
) -> Field {
    let arrow_type = match array {
        Array::NumericArray(num_arr) => match num_arr {
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
        Array::TextArray(text_arr) => match text_arr {
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
        Array::TemporalArray(temp_arr) => match temp_arr {
            TemporalArray::Datetime32(dt_arr) => match &dt_arr.time_unit {
                TimeUnit::Days => ArrowType::Date32,
                unit => ArrowType::Time32(unit.clone()),
            },
            TemporalArray::Datetime64(dt_arr) => match &dt_arr.time_unit {
                TimeUnit::Milliseconds => ArrowType::Date64,
                TimeUnit::Microseconds | TimeUnit::Nanoseconds => {
                    ArrowType::Time64(dt_arr.time_unit.clone())
                }
                unit => ArrowType::Timestamp(unit.clone(), None), // TODO: extract timezone from metadata
            },
            TemporalArray::Null => ArrowType::Null,
        },
        Array::BooleanArray(_) => ArrowType::Boolean,
        Array::Null => ArrowType::Null,
    };

    let has_mask = array.null_mask().is_some()
        || other_array.map_or(false, |other| other.null_mask().is_some());

    Field::new(name, arrow_type, has_mask, metadata)
}

impl Display for FieldArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "\nFieldArray \"{}\" [{} values] (dtype: {:?})",
            self.field.name,
            self.array.len(),
            self.field.dtype
        )?;

        // For Timestamp types with timezone, use custom printing
        #[cfg(feature = "datetime")]
        if let ArrowType::Timestamp(_unit, Some(ref tz)) = self.field.dtype {
            return format_field_array_with_timezone(f, self, tz);
        }

        self.array.fmt(f)
    }
}

#[cfg(feature = "datetime")]
fn format_field_array_with_timezone(
    f: &mut Formatter<'_>,
    field_array: &FieldArray,
    timezone: &str,
) -> std::fmt::Result {
    use crate::traits::print::MAX_PREVIEW;
    use crate::{Array, TemporalArray};

    let arr = &field_array.array;
    let len = arr.len();
    let nulls = arr.null_count();

    if let Array::TemporalArray(TemporalArray::Datetime64(dt)) = arr {
        writeln!(
            f,
            "DatetimeArray [{} values] (dtype: datetime[{:?}], timezone: {}, nulls: {})",
            len, dt.time_unit, timezone, nulls
        )?;

        write!(f, "[")?;
        for i in 0..usize::min(len, MAX_PREVIEW) {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", format_datetime_with_tz(dt.as_ref(), i, timezone))?;
        }
        if len > MAX_PREVIEW {
            write!(f, ", ...")?;
        }
        writeln!(f, "]")
    } else if let Array::TemporalArray(TemporalArray::Datetime32(dt)) = arr {
        writeln!(
            f,
            "DatetimeArray [{} values] (dtype: datetime[{:?}], timezone: {}, nulls: {})",
            len, dt.time_unit, timezone, nulls
        )?;

        write!(f, "[")?;
        for i in 0..usize::min(len, MAX_PREVIEW) {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", format_datetime_with_tz(dt.as_ref(), i, timezone))?;
        }
        if len > MAX_PREVIEW {
            write!(f, ", ...")?;
        }
        writeln!(f, "]")
    } else {
        field_array.array.fmt(f)
    }
}

#[cfg(feature = "datetime")]
fn format_datetime_with_tz<T>(arr: &crate::DatetimeArray<T>, idx: usize, timezone: &str) -> String
where
    T: crate::Integer + std::fmt::Display,
{
    use crate::traits::print::format_datetime_value;
    format_datetime_value(arr, idx, Some(timezone))
}

impl Shape for FieldArray {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank1(self.len())
    }
}

impl Concatenate for FieldArray {
    /// Concatenates two FieldArrays, consuming both.
    ///
    /// # Requirements
    /// - Both FieldArrays must have matching field metadata:
    ///   - Same name
    ///   - Same dtype
    ///   - Same nullability
    ///
    /// # Returns
    /// A new FieldArray with the concatenated array data
    ///
    /// # Errors
    /// - `IncompatibleTypeError` if field metadata doesn't match
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        // Validate field compatibility
        if self.field.name != other.field.name {
            return Err(MinarrowError::IncompatibleTypeError {
                from: "FieldArray",
                to: "FieldArray",
                message: Some(format!(
                    "Field name mismatch: '{}' vs '{}'",
                    self.field.name, other.field.name
                )),
            });
        }

        if self.field.dtype != other.field.dtype {
            return Err(MinarrowError::IncompatibleTypeError {
                from: "FieldArray",
                to: "FieldArray",
                message: Some(format!(
                    "Field '{}' dtype mismatch: {:?} vs {:?}",
                    self.field.name, self.field.dtype, other.field.dtype
                )),
            });
        }

        if self.field.nullable != other.field.nullable {
            return Err(MinarrowError::IncompatibleTypeError {
                from: "FieldArray",
                to: "FieldArray",
                message: Some(format!(
                    "Field '{}' nullable mismatch: {} vs {}",
                    self.field.name, self.field.nullable, other.field.nullable
                )),
            });
        }

        // Concatenate the underlying arrays
        let concatenated_array = self.array.concat(other.array)?;
        let null_count = concatenated_array.null_count();

        // Create result FieldArray with the same field metadata
        Ok(FieldArray {
            field: self.field,
            array: concatenated_array,
            null_count,
        })
    }
}

// ===== Selection Trait Implementation =====

#[cfg(all(feature = "select", feature = "views"))]
impl RowSelection for FieldArray {
    type View = ArrayV;

    fn r<S: DataSelector>(&self, selection: S) -> ArrayV {
        if selection.is_contiguous() {
            // Contiguous selection (ranges): adjust offset and len
            let indices = selection.resolve_indices(self.array.len());
            if indices.is_empty() {
                return ArrayV::new(self.array.clone(), 0, 0);
            }
            ArrayV::new(self.array.clone(), indices[0], indices.len())
        } else {
            // Non-contiguous selection (index arrays): gather into new array
            let view = ArrayV::from(self.array.clone());
            let indices = selection.resolve_indices(self.array.len());
            let gathered_array = view.gather_indices(&indices);
            ArrayV::new(gathered_array, 0, indices.len())
        }
    }

    fn get_row_count(&self) -> usize {
        self.array.len()
    }
}

// ============================================================
// fa_* macros - FieldArray creation from name + array values
// ============================================================
// Each fa_<variant> wraps the corresponding arr_<variant>,
// producing a FieldArray with the given name and inferred schema.
//
// Usage:
//   fa_i32!("col", 1, 2, 3)        // from literals
//   fa_i32!("col", my_vec64)        // from a Vec64
//   fa_i32!("col")                  // empty array
//   fa_i32_opt!("col", Some(1), None, Some(3))  // nullable
// ============================================================

// ======== numeric ========

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! fa_i8 {
    ($name:expr, @vec64 $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i8!($v; $mask))
    };
    ($name:expr, @slice $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i8!($v; $mask))
    };
    ($name:expr, $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i8!($v; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ ; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i8!($first, $($rest),+ ; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i8!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i8!($v))
    };
    ($name:expr, @slice $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i8!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_i8!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_i8!())
    }};
}

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! fa_i16 {
    ($name:expr, @vec64 $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i16!($v; $mask))
    };
    ($name:expr, @slice $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i16!($v; $mask))
    };
    ($name:expr, $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i16!($v; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ ; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i16!($first, $($rest),+ ; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i16!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i16!($v))
    };
    ($name:expr, @slice $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i16!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_i16!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_i16!())
    }};
}

#[macro_export]
macro_rules! fa_i32 {
    ($name:expr, @vec64 $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i32!($v; $mask))
    };
    ($name:expr, @slice $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i32!($v; $mask))
    };
    ($name:expr, $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i32!($v; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ ; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i32!($first, $($rest),+ ; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i32!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i32!($v))
    };
    ($name:expr, @slice $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i32!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_i32!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_i32!())
    }};
}

#[macro_export]
macro_rules! fa_i64 {
    ($name:expr, @vec64 $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i64!($v; $mask))
    };
    ($name:expr, @slice $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i64!($v; $mask))
    };
    ($name:expr, $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i64!($v; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ ; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i64!($first, $($rest),+ ; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i64!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i64!($v))
    };
    ($name:expr, @slice $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i64!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_i64!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_i64!())
    }};
}

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! fa_u8 {
    ($name:expr, @vec64 $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u8!($v; $mask))
    };
    ($name:expr, @slice $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u8!($v; $mask))
    };
    ($name:expr, $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u8!($v; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ ; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u8!($first, $($rest),+ ; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u8!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u8!($v))
    };
    ($name:expr, @slice $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u8!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_u8!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_u8!())
    }};
}

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! fa_u16 {
    ($name:expr, @vec64 $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u16!($v; $mask))
    };
    ($name:expr, @slice $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u16!($v; $mask))
    };
    ($name:expr, $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u16!($v; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ ; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u16!($first, $($rest),+ ; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u16!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u16!($v))
    };
    ($name:expr, @slice $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u16!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_u16!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_u16!())
    }};
}

#[macro_export]
macro_rules! fa_u32 {
    ($name:expr, @vec64 $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u32!($v; $mask))
    };
    ($name:expr, @slice $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u32!($v; $mask))
    };
    ($name:expr, $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u32!($v; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ ; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u32!($first, $($rest),+ ; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u32!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u32!($v))
    };
    ($name:expr, @slice $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u32!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_u32!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_u32!())
    }};
}

#[macro_export]
macro_rules! fa_u64 {
    ($name:expr, @vec64 $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u64!($v; $mask))
    };
    ($name:expr, @slice $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u64!($v; $mask))
    };
    ($name:expr, $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u64!($v; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ ; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u64!($first, $($rest),+ ; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u64!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u64!($v))
    };
    ($name:expr, @slice $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u64!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_u64!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_u64!())
    }};
}

// ======== Float types ========

#[macro_export]
macro_rules! fa_f32 {
    ($name:expr, @vec64 $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_f32!($v; $mask))
    };
    ($name:expr, @slice $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_f32!($v; $mask))
    };
    ($name:expr, $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_f32!($v; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ ; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_f32!($first, $($rest),+ ; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_f32!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_f32!($v))
    };
    ($name:expr, @slice $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_f32!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_f32!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_f32!())
    }};
}

#[macro_export]
macro_rules! fa_f64 {
    ($name:expr, @vec64 $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_f64!($v; $mask))
    };
    ($name:expr, @slice $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_f64!($v; $mask))
    };
    ($name:expr, $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_f64!($v; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ ; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_f64!($first, $($rest),+ ; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_f64!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_f64!($v))
    };
    ($name:expr, @slice $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_f64!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_f64!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_f64!())
    }};
}

// ======== Boolean ========

#[macro_export]
macro_rules! fa_bool {
    ($name:expr, @vec64 $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_bool!($v; $mask))
    };
    ($name:expr, @slice $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_bool!($v; $mask))
    };
    ($name:expr, $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_bool!($v; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ ; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_bool!($first, $($rest),+ ; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_bool!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_bool!($v))
    };
    ($name:expr, @slice $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_bool!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_bool!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_bool!())
    }};
}

// ======== String ========

#[macro_export]
macro_rules! fa_str32 {
    ($name:expr, @vec64 $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_str32!($v; $mask))
    };
    ($name:expr, @slice $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_str32!($v; $mask))
    };
    ($name:expr, $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_str32!($v; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ ; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_str32!($first, $($rest),+ ; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_str32!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_str32!($v))
    };
    ($name:expr, @slice $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_str32!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_str32!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_str32!())
    }};
}

#[cfg(feature = "large_string")]
#[macro_export]
macro_rules! fa_str64 {
    ($name:expr, @vec64 $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_str64!($v; $mask))
    };
    ($name:expr, @slice $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_str64!($v; $mask))
    };
    ($name:expr, $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_str64!($v; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ ; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_str64!($first, $($rest),+ ; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_str64!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_str64!($v))
    };
    ($name:expr, @slice $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_str64!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_str64!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_str64!())
    }};
}

// ======== Categorical ========

#[cfg(feature = "default_categorical_8")]
#[macro_export]
macro_rules! fa_cat8 {
    ($name:expr, @vec64 $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat8!($v; $mask))
    };
    ($name:expr, @slice $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat8!($v; $mask))
    };
    ($name:expr, $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat8!($v; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ ; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat8!($first, $($rest),+ ; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat8!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat8!($v))
    };
    ($name:expr, @slice $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat8!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_cat8!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_cat8!())
    }};
}

#[cfg(feature = "extended_categorical")]
#[macro_export]
macro_rules! fa_cat16 {
    ($name:expr, @vec64 $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat16!($v; $mask))
    };
    ($name:expr, @slice $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat16!($v; $mask))
    };
    ($name:expr, $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat16!($v; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ ; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat16!($first, $($rest),+ ; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat16!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat16!($v))
    };
    ($name:expr, @slice $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat16!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_cat16!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_cat16!())
    }};
}

#[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
#[macro_export]
macro_rules! fa_cat32 {
    ($name:expr, @vec64 $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat32!($v; $mask))
    };
    ($name:expr, @slice $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat32!($v; $mask))
    };
    ($name:expr, $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat32!($v; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ ; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat32!($first, $($rest),+ ; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat32!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat32!($v))
    };
    ($name:expr, @slice $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat32!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_cat32!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_cat32!())
    }};
}

#[cfg(feature = "extended_categorical")]
#[macro_export]
macro_rules! fa_cat64 {
    ($name:expr, @vec64 $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat64!($v; $mask))
    };
    ($name:expr, @slice $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat64!($v; $mask))
    };
    ($name:expr, $v:expr; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat64!($v; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ ; $mask:expr) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat64!($first, $($rest),+ ; $mask))
    };
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat64!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat64!($v))
    };
    ($name:expr, @slice $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat64!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_cat64!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_cat64!())
    }};
}

// ======== fa_*_opt variants - nullable FieldArray creation ========

// ======== Integer (signed) ========

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! fa_i8_opt {
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i8_opt!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i8_opt!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_i8_opt!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_i8_opt!())
    }};
}

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! fa_i16_opt {
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i16_opt!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i16_opt!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_i16_opt!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_i16_opt!())
    }};
}

#[macro_export]
macro_rules! fa_i32_opt {
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i32_opt!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i32_opt!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_i32_opt!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_i32_opt!())
    }};
}

#[macro_export]
macro_rules! fa_i64_opt {
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i64_opt!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_i64_opt!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_i64_opt!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_i64_opt!())
    }};
}

// ======== Integer (unsigned) ========

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! fa_u8_opt {
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u8_opt!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u8_opt!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_u8_opt!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_u8_opt!())
    }};
}

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! fa_u16_opt {
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u16_opt!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u16_opt!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_u16_opt!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_u16_opt!())
    }};
}

#[macro_export]
macro_rules! fa_u32_opt {
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u32_opt!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u32_opt!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_u32_opt!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_u32_opt!())
    }};
}

#[macro_export]
macro_rules! fa_u64_opt {
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u64_opt!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_u64_opt!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_u64_opt!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_u64_opt!())
    }};
}

// ======== Float ========

#[macro_export]
macro_rules! fa_f32_opt {
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_f32_opt!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_f32_opt!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_f32_opt!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_f32_opt!())
    }};
}

#[macro_export]
macro_rules! fa_f64_opt {
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_f64_opt!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_f64_opt!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_f64_opt!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_f64_opt!())
    }};
}

// ======== Boolean ========

#[macro_export]
macro_rules! fa_bool_opt {
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_bool_opt!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_bool_opt!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_bool_opt!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_bool_opt!())
    }};
}

// ======== String ========

#[macro_export]
macro_rules! fa_str32_opt {
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_str32_opt!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_str32_opt!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_str32_opt!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_str32_opt!())
    }};
}

#[cfg(feature = "large_string")]
#[macro_export]
macro_rules! fa_str64_opt {
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_str64_opt!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_str64_opt!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_str64_opt!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_str64_opt!())
    }};
}

// ======== Categorical ========

#[cfg(feature = "default_categorical_8")]
#[macro_export]
macro_rules! fa_cat8_opt {
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat8_opt!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat8_opt!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_cat8_opt!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_cat8_opt!())
    }};
}

#[cfg(feature = "extended_categorical")]
#[macro_export]
macro_rules! fa_cat16_opt {
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat16_opt!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat16_opt!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_cat16_opt!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_cat16_opt!())
    }};
}

#[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
#[macro_export]
macro_rules! fa_cat32_opt {
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat32_opt!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat32_opt!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_cat32_opt!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_cat32_opt!())
    }};
}

#[cfg(feature = "extended_categorical")]
#[macro_export]
macro_rules! fa_cat64_opt {
    ($name:expr, $first:expr, $($rest:expr),+ $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat64_opt!($first, $($rest),+))
    };
    ($name:expr, @vec64 $v:expr $(,)?) => {
        $crate::FieldArray::from_arr($name, $crate::arr_cat64_opt!($v))
    };
    ($name:expr, $v:expr $(,)?) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_cat64_opt!(vec64![$v]))
    }};
    ($name:expr) => {{
        use $crate::vec64;
        $crate::FieldArray::from_arr($name, $crate::arr_cat64_opt!())
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structs::variants::integer::IntegerArray;
    use crate::traits::masked_array::MaskedArray;

    #[test]
    fn test_field_array_basic_construction() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(1);
        arr.push(2);
        let array = Array::from_int32(arr);

        let field = Field::new("my_col", ArrowType::Int32, false, None);
        let field_array = FieldArray::new(field.clone(), array.clone());

        assert_eq!(field_array.len(), 2);
        assert_eq!(field_array.field, field.into());
        assert_eq!(field_array.array, array.into());
    }

    #[test]
    fn test_field_array_from_parts_infers_nullability() {
        let mut arr = IntegerArray::<i64>::default();
        arr.push(10);
        arr.push_null(); // makes it nullable
        let array = Array::from_int64(arr);

        let field_array =
            FieldArray::from_parts("nullable_col", ArrowType::Int64, None, None, array.clone());

        assert_eq!(field_array.field.name, "nullable_col");
        assert_eq!(field_array.field.dtype, ArrowType::Int64);
        assert_eq!(field_array.field.nullable, true);
        assert_eq!(field_array.len(), 2);
        assert_eq!(field_array.array, array.into());
    }

    #[cfg(feature = "views")]
    #[test]
    fn test_field_array_slice() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(10);
        arr.push(20);
        arr.push(30);

        let fa = field_array("x", Array::from_int32(arr));
        let view = fa.view(1, 2);
        assert_eq!(view.1.name, "x");
        assert_eq!(view.0.2, 2);
        assert_eq!(view.0.1, 1);
        assert_eq!(view.0.2, 2);
        assert_eq!(view.0.0.len(), 3);
    }

    #[test]
    fn test_null_count_cache_sync_concat() {
        // Create first FieldArray with nulls
        let mut arr1 = IntegerArray::<i32>::default();
        arr1.push(1);
        arr1.push_null();
        arr1.push(3);
        let mut fa1 = field_array("test", Array::from_int32(arr1));
        assert_eq!(fa1.null_count(), 1);

        // Create second FieldArray with nulls
        let mut arr2 = IntegerArray::<i32>::default();
        arr2.push_null();
        arr2.push(5);
        let fa2 = field_array("test", Array::from_int32(arr2));
        assert_eq!(fa2.null_count(), 1);

        // Concatenate and verify null_count cache is updated
        fa1.concat_field_array(&fa2);
        assert_eq!(fa1.len(), 5);
        assert_eq!(fa1.null_count(), 2); // Should be 2 nulls total
    }

    #[test]
    fn test_null_count_cache_sync_with_array_mut() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(1);
        arr.push(2);
        let mut fa = field_array("test", Array::from_int32(arr));
        assert_eq!(fa.null_count(), 0);

        // Mutate through with_array_mut to add nulls
        fa.with_array_mut(|array| {
            array.concat_array(&Array::from_int32({
                let mut new_arr = IntegerArray::<i32>::default();
                new_arr.push_null();
                new_arr.push_null();
                new_arr
            }));
        });

        assert_eq!(fa.len(), 4);
        assert_eq!(fa.null_count(), 2); // Cache should be refreshed automatically
    }

    #[test]
    fn test_refresh_null_count() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(1);
        arr.push(2);
        let mut fa = field_array("test", Array::from_int32(arr));
        assert_eq!(fa.null_count(), 0);

        // Manually mutate underlying array (simulating external mutation)
        if let Array::NumericArray(crate::NumericArray::Int32(int_arr)) = &mut fa.array {
            use crate::traits::masked_array::MaskedArray;
            std::sync::Arc::make_mut(int_arr).push_null();
        }

        // Cache is now stale
        assert_eq!(fa.null_count, 0); // Cached value still 0
        assert_eq!(fa.array.null_count(), 1); // Actual value is 1

        // Refresh the cache
        fa.refresh_null_count();
        assert_eq!(fa.null_count(), 1); // Cache now updated
    }
}

#[cfg(test)]
mod concat_tests {
    use super::*;
    use crate::structs::variants::integer::IntegerArray;
    use crate::traits::concatenate::Concatenate;
    use crate::traits::masked_array::MaskedArray;

    #[test]
    fn test_field_array_concat_basic() {
        let arr1 = IntegerArray::<i32>::from_slice(&[1, 2, 3]);
        let fa1 = field_array("numbers", Array::from_int32(arr1));

        let arr2 = IntegerArray::<i32>::from_slice(&[4, 5, 6]);
        let fa2 = field_array("numbers", Array::from_int32(arr2));

        let result = fa1.concat(fa2).unwrap();

        assert_eq!(result.len(), 6);
        assert_eq!(result.field.name, "numbers");
        assert_eq!(result.field.dtype, ArrowType::Int32);

        if let Array::NumericArray(crate::NumericArray::Int32(arr)) = result.array {
            assert_eq!(arr.len(), 6);
            assert_eq!(arr.get(0), Some(1));
            assert_eq!(arr.get(5), Some(6));
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[test]
    fn test_field_array_concat_with_nulls() {
        let mut arr1 = IntegerArray::<i32>::with_capacity(3, true);
        arr1.push(10);
        arr1.push_null();
        arr1.push(30);
        let fa1 = FieldArray::from_parts(
            "data",
            ArrowType::Int32,
            Some(true),
            None,
            Array::from_int32(arr1),
        );

        let mut arr2 = IntegerArray::<i32>::with_capacity(2, true);
        arr2.push_null();
        arr2.push(50);
        let fa2 = FieldArray::from_parts(
            "data",
            ArrowType::Int32,
            Some(true),
            None,
            Array::from_int32(arr2),
        );

        let result = fa1.concat(fa2).unwrap();

        assert_eq!(result.len(), 5);
        assert_eq!(result.null_count(), 2);

        if let Array::NumericArray(crate::NumericArray::Int32(arr)) = result.array {
            assert_eq!(arr.get(0), Some(10));
            assert_eq!(arr.get(1), None);
            assert_eq!(arr.get(2), Some(30));
            assert_eq!(arr.get(3), None);
            assert_eq!(arr.get(4), Some(50));
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[test]
    fn test_field_array_concat_name_mismatch() {
        let arr1 = IntegerArray::<i32>::from_slice(&[1, 2]);
        let fa1 = field_array("col_a", Array::from_int32(arr1));

        let arr2 = IntegerArray::<i32>::from_slice(&[3, 4]);
        let fa2 = field_array("col_b", Array::from_int32(arr2));

        let result = fa1.concat(fa2);
        assert!(result.is_err());

        if let Err(MinarrowError::IncompatibleTypeError { message, .. }) = result {
            assert!(message.unwrap().contains("Field name mismatch"));
        } else {
            panic!("Expected IncompatibleTypeError");
        }
    }

    #[test]
    fn test_field_array_concat_dtype_mismatch() {
        let arr1 = IntegerArray::<i32>::from_slice(&[1, 2]);
        let fa1 = field_array("data", Array::from_int32(arr1));

        let arr2 = crate::FloatArray::<f64>::from_slice(&[3.0, 4.0]);
        let fa2 = field_array("data", Array::from_float64(arr2));

        let result = fa1.concat(fa2);
        assert!(result.is_err());

        if let Err(MinarrowError::IncompatibleTypeError { message, .. }) = result {
            assert!(message.unwrap().contains("dtype mismatch"));
        } else {
            panic!("Expected IncompatibleTypeError");
        }
    }

    #[test]
    fn test_field_array_concat_nullable_mismatch() {
        let arr1 = IntegerArray::<i32>::from_slice(&[1, 2]);
        let fa1 = FieldArray::from_parts(
            "data",
            ArrowType::Int32,
            Some(false),
            None,
            Array::from_int32(arr1),
        );

        let mut arr2 = IntegerArray::<i32>::with_capacity(2, true);
        arr2.push(3);
        arr2.push(4);
        let fa2 = FieldArray::from_parts(
            "data",
            ArrowType::Int32,
            Some(true),
            None,
            Array::from_int32(arr2),
        );

        let result = fa1.concat(fa2);
        assert!(result.is_err());

        if let Err(MinarrowError::IncompatibleTypeError { message, .. }) = result {
            assert!(message.unwrap().contains("nullable mismatch"));
        } else {
            panic!("Expected IncompatibleTypeError");
        }
    }
}

#[cfg(test)]
mod fa_macro_tests {
    use crate::ffi::arrow_dtype::ArrowType;

    #[test]
    fn test_fa_i32_literals() {
        let fa = fa_i32!("age", 10, 20, 30);
        assert_eq!(fa.field.name, "age");
        assert_eq!(fa.field.dtype, ArrowType::Int32);
        assert_eq!(fa.len(), 3);
        assert!(!fa.field.nullable);
    }

    #[test]
    fn test_fa_i64_literals() {
        let fa = fa_i64!("big", 100, 200);
        assert_eq!(fa.field.name, "big");
        assert_eq!(fa.field.dtype, ArrowType::Int64);
        assert_eq!(fa.len(), 2);
    }

    #[test]
    fn test_fa_u32_literals() {
        let fa = fa_u32!("count", 1, 2, 3, 4);
        assert_eq!(fa.field.name, "count");
        assert_eq!(fa.field.dtype, ArrowType::UInt32);
        assert_eq!(fa.len(), 4);
    }

    #[test]
    fn test_fa_u64_literals() {
        let fa = fa_u64!("id", 99, 100);
        assert_eq!(fa.field.name, "id");
        assert_eq!(fa.field.dtype, ArrowType::UInt64);
        assert_eq!(fa.len(), 2);
    }

    #[test]
    fn test_fa_f32_literals() {
        let fa = fa_f32!("temp", 1.0, 2.5, 3.7);
        assert_eq!(fa.field.name, "temp");
        assert_eq!(fa.field.dtype, ArrowType::Float32);
        assert_eq!(fa.len(), 3);
    }

    #[test]
    fn test_fa_f64_literals() {
        let fa = fa_f64!("price", 9.99, 19.99);
        assert_eq!(fa.field.name, "price");
        assert_eq!(fa.field.dtype, ArrowType::Float64);
        assert_eq!(fa.len(), 2);
    }

    #[test]
    fn test_fa_bool_literals() {
        let fa = fa_bool!("active", true, false, true);
        assert_eq!(fa.field.name, "active");
        assert_eq!(fa.field.dtype, ArrowType::Boolean);
        assert_eq!(fa.len(), 3);
    }

    #[test]
    fn test_fa_str32_literals() {
        let fa = fa_str32!("name", "alice", "bob", "charlie");
        assert_eq!(fa.field.name, "name");
        assert_eq!(fa.field.dtype, ArrowType::String);
        assert_eq!(fa.len(), 3);
    }

    #[cfg(feature = "large_string")]
    #[test]
    fn test_fa_str64_literals() {
        let fa = fa_str64!("label", "x", "y");
        assert_eq!(fa.field.name, "label");
        assert_eq!(fa.field.dtype, ArrowType::LargeString);
        assert_eq!(fa.len(), 2);
    }

    #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
    #[test]
    fn test_fa_cat32_literals() {
        use crate::ffi::arrow_dtype::CategoricalIndexType;
        let fa = fa_cat32!("colour", "red", "green", "blue");
        assert_eq!(fa.field.name, "colour");
        assert_eq!(fa.field.dtype, ArrowType::Dictionary(CategoricalIndexType::UInt32));
        assert_eq!(fa.len(), 3);
    }

    #[test]
    fn test_fa_i32_empty() {
        let fa = fa_i32!("empty");
        assert_eq!(fa.field.name, "empty");
        assert_eq!(fa.field.dtype, ArrowType::Int32);
        assert_eq!(fa.len(), 0);
    }

    #[test]
    fn test_fa_i32_opt_with_nulls() {
        let fa = fa_i32_opt!("score", Some(1i32), None::<i32>, Some(3));
        assert_eq!(fa.field.name, "score");
        assert_eq!(fa.field.dtype, ArrowType::Int32);
        assert_eq!(fa.len(), 3);
        assert!(fa.field.nullable);
        assert_eq!(fa.null_count(), 1);
    }

    #[test]
    fn test_fa_f64_opt_with_nulls() {
        let fa = fa_f64_opt!("weight", Some(1.5f64), None::<f64>);
        assert_eq!(fa.field.name, "weight");
        assert_eq!(fa.field.dtype, ArrowType::Float64);
        assert_eq!(fa.len(), 2);
        assert!(fa.field.nullable);
        assert_eq!(fa.null_count(), 1);
    }

    #[test]
    fn test_fa_bool_opt_with_nulls() {
        let fa = fa_bool_opt!("flag", Some(true), None::<bool>, Some(false));
        assert_eq!(fa.field.name, "flag");
        assert_eq!(fa.field.dtype, ArrowType::Boolean);
        assert_eq!(fa.len(), 3);
        assert!(fa.field.nullable);
        assert_eq!(fa.null_count(), 1);
    }

    #[test]
    fn test_fa_str32_opt_with_nulls() {
        let fa = fa_str32_opt!("city", Some("london"), None::<&str>, Some("paris"));
        assert_eq!(fa.field.name, "city");
        assert_eq!(fa.field.dtype, ArrowType::String);
        assert_eq!(fa.len(), 3);
        assert!(fa.field.nullable);
        assert_eq!(fa.null_count(), 1);
    }

    #[test]
    fn test_fa_i32_single_value() {
        let fa = fa_i32!("data", 42);
        assert_eq!(fa.field.name, "data");
        assert_eq!(fa.len(), 1);
    }

    #[test]
    fn test_fa_i32_trailing_comma() {
        let fa = fa_i32!("data", 5, 6, 7,);
        assert_eq!(fa.field.name, "data");
        assert_eq!(fa.len(), 3);
    }
}

#[cfg(test)]
mod fa_macro_extensions_tests {
    use crate::{Array, Bitmask, NumericArray, Vec64, vec64};

    fn f64_values_and_mask(fa: &crate::FieldArray) -> (Vec<f64>, Option<Bitmask>) {
        match &fa.array {
            Array::NumericArray(NumericArray::Float64(a)) => {
                (a.data.as_slice().to_vec(), a.null_mask.clone())
            }
            _ => panic!("expected Float64 FieldArray"),
        }
    }

    #[test]
    fn fa_f64_vec64_marker_with_mask() {
        let v: Vec64<f64> = vec64![1.0, 2.0, 3.0];
        let m = Bitmask::from_bools(&[true, false, true]);
        let fa = fa_f64!("a", @vec64 v; m);
        assert_eq!(fa.field.name, "a");
        let (data, mask) = f64_values_and_mask(&fa);
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
        assert_eq!(mask.expect("expected mask").get(1), false);
    }

    #[test]
    fn fa_f64_slice_marker_with_mask() {
        let s: &[f64] = &[1.0, 2.0, 3.0];
        let m = Bitmask::from_bools(&[false, true, true]);
        let fa = fa_f64!("a", @slice s; m);
        let (data, mask) = f64_values_and_mask(&fa);
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
        assert_eq!(mask.expect("expected mask").get(0), false);
    }

    #[test]
    fn fa_f64_bare_values_with_mask() {
        let s: &[f64] = &[10.0, 20.0, 30.0];
        let m = Bitmask::from_bools(&[true, true, false]);
        let fa = fa_f64!("a", s; m);
        let (data, mask) = f64_values_and_mask(&fa);
        assert_eq!(data, vec![10.0, 20.0, 30.0]);
        assert_eq!(mask.expect("expected mask").get(2), false);
    }

    #[test]
    fn fa_f64_literal_multi_with_mask() {
        let m = Bitmask::from_bools(&[true, false, true]);
        let fa = fa_f64!("a", 1.0, 2.0, 3.0 ; m);
        let (data, mask) = f64_values_and_mask(&fa);
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
        assert_eq!(mask.expect("expected mask").get(1), false);
    }

    #[test]
    fn fa_f64_slice_marker_no_mask() {
        let s: &[f64] = &[1.0, 2.0];
        let fa = fa_f64!("a", @slice s);
        let (data, mask) = f64_values_and_mask(&fa);
        assert_eq!(data, vec![1.0, 2.0]);
        assert!(mask.is_none());
    }

    #[test]
    fn fa_f64_existing_forms_unchanged() {
        let single = fa_f64!("a", 42.0);
        assert_eq!(single.len(), 1);

        let multi = fa_f64!("a", 1.0, 2.0, 3.0);
        assert_eq!(multi.len(), 3);

        let v: Vec64<f64> = vec64![5.0, 6.0];
        let from_vec = fa_f64!("a", @vec64 v);
        assert_eq!(from_vec.len(), 2);

        let empty = fa_f64!("a");
        assert_eq!(empty.len(), 0);
    }
}
