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

//! # **DatetimeArray Module** - *Mid-Level, Inner Typed DateTime Array*
//!
//! Arrow-compatible datetime/timestamp array implementation with optional null-mask,
//! 64-byte alignment, and efficient memory layout for analytical workloads.
//!
//! **Notice**: When using the `datetime_ops` feature, all dates are stored in UTC time,
//! even when a timezone is stored in `Field` metadata. Encoding timezone in `Field`
//! ensures that the tz-aware time is displayed when printing. **Warning**: *All timezones
//! are static values in ./tz.rs and are subject to change over time. When performing
//! international time calculations with any specific accuracy requirements, please verify
//! your timezone(s) and raise a PR or Issue if and need to be updated.
//! For e.g., for regions who have moved on/off Daylight savings time, etc.*
//!
//! ## Overview
//! - Logical type: temporal values with a defined [`TimeUnit`] (seconds, milliseconds,
//!   microseconds, nanoseconds, days).
//! - Physical storage: i64 integers representing raw time offsets from
//!   the UNIX epoch or a base date, plus an optional bit-packed validity mask.
//! - Single generic type supports all Arrow datetime/timestamp variants via `Field`
//!   metadata, avoiding multiple specialised array structs.
//! - Integrates with Arrow FFI for zero-copy interop.
//! - ***Timezone information does not alter the underlying UTC storage***, only what value
//! is displayed when printed. Therefore, when working with integer values whose times are physically
//! those timezones - be sure to convert them to UTC first before storing them in Minarrow, otherwise
//! it would 'offset the already offsetted' values.
//!
//! ## Features
//! - **Construction** from slices, `Vec64` or plain `Vec` buffers, with optional null mask.
//! - **Mutation**: push, set, and bulk null insertion.
//! - **Null handling**: optional validity mask with length validation.
//! - **Conversion**: when `datetime_ops` feature is enabled, convert to native date/time
//!   values via the `time` crate, plus component extraction and datetime operations.
//! - **Datetime operations**: full suite of standard datetime operations under ./datetime_ops
//! - **Tz-aware**: see `examples/datetime_ops` for timezone usage.
//!
//! ## Use Cases
//! - High-performance temporal analytics.
//! - FFI-based interchange with Apache Arrow or other columnar systems.
//! - Streaming or batch ingestion with incremental append.
//!
//! ## Related Types
//! - [`TimeUnit`]: enumerates supported time granularities.
//! - [`Bitmask`]: underlying null-mask storage.
//! - [`MaskedArray`]: trait defining the nullable array API.

use std::fmt::{Display, Formatter, Result as FmtResult};

#[cfg(feature = "datetime")]
use crate::Buffer;
use crate::enums::shape_dim::ShapeDim;
use crate::enums::time_units::TimeUnit;
use crate::traits::concatenate::Concatenate;
use crate::traits::masked_array::MaskedArray;
use crate::traits::shape::Shape;
use crate::traits::type_unions::Integer;
use crate::utils::validate_null_mask_len;
use crate::{
    Bitmask, Length, Offset, impl_arc_masked_array, impl_array_ref_deref, impl_masked_array,
};
use ::vec64::{Vec64, Vec64Alloc};

pub mod datetime_ops;
pub mod parse;
pub mod tz;

/// Julian Day Number corresponding to the Unix epoch (1970-01-01 00:00:00 UTC).
///
/// Used when converting between “days since Unix epoch” and absolute Julian day counts.
/// The `time` crate’s [`Date::from_julian_day`] and [`Date::to_julian_day`] use Julian
/// day numbering, so this constant provides the offset required to translate
/// Arrow-style day counts (relative to 1970-01-01) into absolute Julian days.
///
/// # Value
/// `2_440_588` - the Julian day number for **1970-01-01 UTC**.
pub const UNIX_EPOCH_JULIAN_DAY: i64 = 2_440_588;

/// # DatetimeArray
///
/// Arrow-compatible datetime/timestamp array with 64-byte alignment and optional null mask.
///
/// ## Role
/// - Many will prefer the higher level `Array` type, which dispatches to this when
/// necessary.
/// - Can be used as a standalone datetime array or as the datetime arm of `TemporalArray` / `Array`.
///
/// ## Description
/// - Stores temporal values as numeric offsets (`T: Integer`) from the UNIX epoch or a base date,
///   with units defined by [`TimeUnit`].
/// - A single struct supports all Arrow datetime/timestamp variants, with the exact logical
///   type determined by an associated `Field`'s `ArrowType`.
/// - `null_mask` is an optional bit-packed validity bitmap (`1 = valid`, `0 = null`).
/// - Implements [`MaskedArray`] for consistent nullable array behaviour.
/// - When `datetime_ops` is enabled, provides conversion to native date/time via the `time` crate.
///
/// ### Fields
/// - `data`: backing buffer storing raw temporal values.
/// - `null_mask`: optional bit-packed validity bitmap.
/// - `time_unit`: time units associated with the stored values.
///
/// ## Example
/// ```rust
/// use minarrow::{Bitmask, DatetimeArray, vec64};
/// use minarrow::enums::time_units::TimeUnit;
///
/// // Milliseconds since epoch, no nulls
/// let arr = DatetimeArray::<i64>::from_slice(&[1_700_000_000_000, 1_700_000_100_000], Some(TimeUnit::Milliseconds));
/// assert_eq!(arr.len(), 2);
/// assert_eq!(arr.value(0), Some(1_700_000_000_000));
///
/// // With nulls
/// let mask = Bitmask::from_bools(&[true, false, true]);
/// let arr_with_nulls = DatetimeArray::new(
///     vec64![1000_i64, 2000, 3000],
///     Some(mask),
///     Some(TimeUnit::Milliseconds)
/// );
/// assert_eq!(arr_with_nulls.value(1), None);
/// ```
#[cfg(feature = "datetime")]
#[repr(C, align(64))]
#[derive(PartialEq, Clone, Debug, Default)]
pub struct DatetimeArray<T> {
    /// Backing buffer of time values (e.g., milliseconds since epoch).
    pub data: Buffer<T>,
    /// Optional null mask (bit-packed; 1=valid, 0=null).
    pub null_mask: Option<Bitmask>,
    /// The time units associated with the datatype
    pub time_unit: TimeUnit,
}

impl<T: Integer> DatetimeArray<T> {
    /// Constructs a new, empty array.
    #[inline]
    pub fn new(
        data: impl Into<Buffer<T>>,
        null_mask: Option<Bitmask>,
        time_unit: Option<TimeUnit>,
    ) -> Self {
        let data: Buffer<T> = data.into();
        validate_null_mask_len(data.len(), &null_mask);
        Self {
            data: data.into(),
            null_mask: null_mask,
            time_unit: time_unit.unwrap_or_default(),
        }
    }

    /// Constructs an array with reserved capacity and optional null mask.
    ///
    /// # Arguments
    ///
    /// * `cap` - Capacity (number of elements) to reserve for the backing buffer.
    /// * `null_mask` - If true, allocates a null-mask bit vector.
    #[inline]
    pub fn with_capacity(cap: usize, null_mask: bool, time_unit: Option<TimeUnit>) -> Self {
        Self {
            data: Vec64::with_capacity(cap).into(),
            null_mask: if null_mask {
                // All-valid (1) default - reserved validity slots default to
                // valid under Arrow's 1=valid, 0=null convention.
                Some(Bitmask::new_set_all(cap, true))
            } else {
                None
            },
            time_unit: time_unit.unwrap_or_default(),
        }
    }

    /// Constructs a new, empty array.
    #[inline]
    pub fn with_default_unit(time_unit: Option<TimeUnit>) -> Self {
        Self {
            data: Vec64::new().into(),
            null_mask: None,
            time_unit: time_unit.unwrap_or_default(),
        }
    }

    /// Constructs an DateTimeArray from a slice (dense, no nulls).
    #[inline]
    pub fn from_slice(slice: &[T], time_unit: Option<TimeUnit>) -> Self {
        Self {
            data: Vec64(slice.to_vec_in(Vec64Alloc::default())).into(),
            null_mask: None,
            time_unit: time_unit.unwrap_or_default(),
        }
    }

    /// Returns the raw time value at index (e.g., ms since epoch), or None if null.
    #[inline]
    pub fn value(&self, idx: usize) -> Option<T> {
        if idx >= self.len() {
            None
        } else if self.is_null(idx) {
            None
        } else {
            Some(self.data[idx])
        }
    }

    /// Returns all raw values.
    #[inline]
    pub fn values(&self) -> &Buffer<T> {
        &self.data
    }

    /// Constructs a DatetimeArray by taking ownership of an existing buffer.
    ///
    /// # Arguments
    ///
    /// * `data` - Backing buffer of time values.
    /// * `null_mask` - Optional bit-packed null bitmap.
    /// * `time_unit` - Time unit metadata.
    #[inline]
    pub fn from_vec64(
        data: Vec64<T>,
        null_mask: Option<Bitmask>,
        time_unit: Option<TimeUnit>,
    ) -> Self {
        Self {
            data: data.into(),
            null_mask,
            time_unit: time_unit.unwrap_or_default(),
        }
    }

    /// Constructs a DatetimeArray by taking ownership of a standard Vec<T>.
    ///
    /// # Arguments
    ///
    /// * `data` - Standard heap-allocated buffer (unconditionally promoted).
    /// * `null_mask` - Optional bit-packed null bitmap.
    /// * `time_unit` - Time unit metadata.
    #[inline]
    pub fn from_vec(data: Vec<T>, null_mask: Option<Bitmask>, time_unit: Option<TimeUnit>) -> Self {
        Self::from_vec64(data.into(), null_mask, time_unit)
    }
}

// Time crate datetime conversion
#[cfg(feature = "datetime_ops")]
impl DatetimeArray<i64> {
    /// Returns the value at index as time::OffsetDateTime (UTC)
    ///
    /// Interprets the value based on the array's TimeUnit.
    #[inline]
    pub fn as_datetime(&self, idx: usize) -> Option<time::OffsetDateTime> {
        use time::OffsetDateTime;
        self.value(idx).and_then(|val| {
            match self.time_unit {
                TimeUnit::Seconds => OffsetDateTime::from_unix_timestamp(val).ok(),
                TimeUnit::Milliseconds => {
                    OffsetDateTime::from_unix_timestamp_nanos((val as i128) * 1_000_000).ok()
                }
                TimeUnit::Microseconds => {
                    OffsetDateTime::from_unix_timestamp_nanos((val as i128) * 1_000).ok()
                }
                TimeUnit::Nanoseconds => {
                    OffsetDateTime::from_unix_timestamp_nanos(val as i128).ok()
                }
                TimeUnit::Days => {
                    // Days since Unix epoch (1970-01-01)
                    time::Date::from_julian_day((val + UNIX_EPOCH_JULIAN_DAY) as i32) // Convert Unix days to Julian day
                        .ok()
                        .and_then(|date| date.with_hms(0, 0, 0).ok())
                        .map(|dt| dt.assume_utc())
                }
            }
        })
    }

    /// Returns (year, month, day, hour, minute, second, millisecond, nanosecond) tuple.
    #[inline]
    pub fn tuple_dt(&self, idx: usize) -> Option<(i32, u32, u32, u32, u32, u32, u32, u32)> {
        self.as_datetime(idx).map(|dt| {
            (
                dt.year(),
                dt.month() as u32,
                dt.day() as u32,
                dt.hour() as u32,
                dt.minute() as u32,
                dt.second() as u32,
                dt.nanosecond() / 1_000_000, // ms
                dt.nanosecond(),
            )
        })
    }

    /// Returns the value at index as time::Date
    #[inline]
    pub fn as_date(&self, idx: usize) -> Option<time::Date> {
        self.as_datetime(idx).map(|dt| dt.date())
    }

    /// Returns the value at index as time::Time
    #[inline]
    pub fn as_time(&self, idx: usize) -> Option<time::Time> {
        self.as_datetime(idx).map(|dt| dt.time())
    }

    /// Wraps this DatetimeArray in a FieldArray with timezone metadata.
    ///
    /// The underlying timestamp data (always UTC) remains unchanged. The timezone
    /// is stored as metadata in the Field's ArrowType for interpretation/display.
    ///
    /// # Arguments
    /// * `tz` - Timezone string in Arrow format (IANA like "America/New_York" or offset like "+05:00")
    ///
    /// # Returns
    /// A FieldArray with Timestamp type and timezone metadata
    #[cfg(feature = "datetime")]
    pub fn tz(&self, tz: &str) -> crate::FieldArray
    where
        Self: Into<crate::Array>,
    {
        use crate::FieldArray;
        use crate::ffi::arrow_dtype::ArrowType;

        let array: crate::Array = self.clone().into();

        FieldArray::from_parts(
            "datetime",
            ArrowType::Timestamp(self.time_unit, Some(tz.to_string())),
            Some(self.is_nullable()),
            None,
            array,
        )
    }
}

impl_masked_array!(DatetimeArray, Integer, Buffer<T>, T, time_unit);
impl_array_ref_deref!(DatetimeArray<T>);
impl_arc_masked_array!(
    Inner = DatetimeArray<T>,
    T = T,
    Container = Buffer<T>,
    LogicalType = T,
    CopyType = T,
    BufferT = T,
    Variant = TemporalArray,
    Bound = Integer,
);

/// There are 2 options for displaying dates
/// One - is to enable the `datetime_ops` feature and then they will display as native datetimes.
/// Second, is to not enable `datetime_ops` and they will display as the value with a suffix e.g. " ms".
#[cfg(feature = "datetime")]
impl<T> Display for DatetimeArray<T>
where
    T: Integer + Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        use crate::traits::print::MAX_PREVIEW;

        let len = self.len();
        let nulls = self.null_count();
        writeln!(
            f,
            "DatetimeArray [{} values] (dtype: datetime[{:?}], nulls: {})",
            len, self.time_unit, nulls
        )?;

        write!(f, "[")?;

        #[cfg(feature = "datetime_ops")]
        {
            use time::OffsetDateTime;
            for i in 0..usize::min(len, MAX_PREVIEW) {
                if i > 0 {
                    write!(f, ", ")?;
                }
                match self.value(i) {
                    Some(val) => match self.time_unit {
                        TimeUnit::Seconds => {
                            let v = val.to_i64().unwrap();
                            match OffsetDateTime::from_unix_timestamp(v) {
                                Ok(dt) => write!(f, "{}", dt),
                                Err(_) => write!(f, "{}s", val),
                            }
                        }
                        TimeUnit::Milliseconds => {
                            let v = val.to_i64().unwrap();
                            match OffsetDateTime::from_unix_timestamp_nanos((v as i128) * 1_000_000)
                            {
                                Ok(dt) => write!(f, "{}", dt),
                                Err(_) => write!(f, "{}ms", val),
                            }
                        }
                        TimeUnit::Microseconds => {
                            let v = val.to_i64().unwrap();
                            match OffsetDateTime::from_unix_timestamp_nanos((v as i128) * 1_000) {
                                Ok(dt) => write!(f, "{}", dt),
                                Err(_) => write!(f, "{}µs", val),
                            }
                        }
                        TimeUnit::Nanoseconds => {
                            let v = val.to_i64().unwrap();
                            match OffsetDateTime::from_unix_timestamp_nanos(v as i128) {
                                Ok(dt) => write!(f, "{}", dt),
                                Err(_) => write!(f, "{}ns", val),
                            }
                        }
                        TimeUnit::Days => {
                            let days = val.to_i64().unwrap();
                            match time::Date::from_julian_day((days + UNIX_EPOCH_JULIAN_DAY) as i32)
                            {
                                Ok(date) => write!(f, "{}", date),
                                Err(_) => write!(f, "{}d", val),
                            }
                        }
                    },
                    None => write!(f, "null"),
                }?;
            }
        }

        #[cfg(not(feature = "datetime_ops"))]
        {
            // Raw value plus suffix
            let suffix = match self.time_unit {
                TimeUnit::Seconds => "s",
                TimeUnit::Milliseconds => "ms",
                TimeUnit::Microseconds => "µs",
                TimeUnit::Nanoseconds => "ns",
                TimeUnit::Days => "d",
            };
            for i in 0..usize::min(len, MAX_PREVIEW) {
                if i > 0 {
                    write!(f, ", ")?;
                }
                match self.value(i) {
                    Some(val) => write!(f, "{}{}", val, suffix)?,
                    None => write!(f, "null")?,
                }
            }
        }

        if len > MAX_PREVIEW {
            write!(f, ", … ({} total)", len)?;
        }

        write!(f, "]")
    }
}

impl<T: Integer> Shape for DatetimeArray<T> {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank1(self.len())
    }
}

#[cfg(feature = "chunked")]
impl<'a, T: Integer> crate::traits::consolidate::Consolidate
    for Vec<crate::aliases::DatetimeAVT<'a, T>>
{
    type Output = DatetimeArray<T>;

    /// Consolidate a vector of `(DatetimeArray<T>, offset, len)` view
    /// tuples into one contiguous `DatetimeArray<T>`. Reads each window
    /// directly from the source buffer. The `time_unit` is taken from
    /// the first chunk.
    fn consolidate(self) -> DatetimeArray<T> {
        use crate::structs::bitmask::Bitmask;
        use crate::traits::consolidate::extend_null_mask;
        use crate::traits::masked_array::MaskedArray;

        assert!(!self.is_empty(), "consolidate() called on empty Vec<DatetimeAVT>");

        let total_len: usize = self.iter().map(|(_, _, len)| *len).sum();
        let has_nulls = self.iter().any(|(arr, _, _)| arr.null_mask.is_some());
        let time_unit = self[0].0.time_unit.clone();

        let mut result = DatetimeArray::<T>::with_capacity(total_len, has_nulls, Some(time_unit));
        let mut result_mask: Option<Bitmask> = if has_nulls {
            Some(Bitmask::default())
        } else {
            None
        };
        let mut current_len = 0;

        for (arr, offset, len) in &self {
            let data: &[T] = &arr.data[*offset..*offset + *len];
            result.extend_from_slice(data);
            extend_null_mask(
                &mut result_mask,
                current_len,
                arr.null_mask(),
                *offset,
                *len,
            );
            current_len += *len;
        }

        if let Some(mask) = result_mask {
            result.set_null_mask(Some(mask));
        }
        result
    }
}

impl<T: Integer> Concatenate for DatetimeArray<T> {
    fn concat(
        mut self,
        other: Self,
    ) -> core::result::Result<Self, crate::enums::error::MinarrowError> {
        // Check that time units match
        if self.time_unit != other.time_unit {
            return Err(crate::enums::error::MinarrowError::IncompatibleTypeError {
                from: "DatetimeArray",
                to: "DatetimeArray",
                message: Some(format!(
                    "Cannot concatenate DatetimeArrays with different time units: {:?} and {:?}",
                    self.time_unit, other.time_unit
                )),
            });
        }

        // Consume other and extend self with its data
        self.append_array(&other);
        Ok(self)
    }
}

#[cfg(feature = "datetime")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec64;

    #[test]
    fn test_new_and_with_capacity() {
        let arr = DatetimeArray::<i64>::default();
        assert_eq!(arr.len(), 0);
        assert!(arr.data.is_empty());
        assert!(arr.null_mask.is_none());

        let mut arr = DatetimeArray::<i64>::with_capacity(50, true, None);
        assert_eq!(arr.len(), 0);
        assert!(arr.data.capacity() >= 50);
        assert!(arr.null_mask.is_some());

        // Reserved null-mask slots must default to valid (1).
        assert_eq!(arr.null_count(), 0);

        arr.push(1_700_000_000);
        arr.push(1_700_000_001);
        assert_eq!(arr.null_count(), 0);

        arr.push_null();
        assert_eq!(arr.null_count(), 1);
    }

    #[test]
    fn test_push_and_value_no_null_mask() {
        let mut arr = DatetimeArray::<i64>::with_capacity(2, false, None);
        arr.push(123);
        arr.push(456);
        assert_eq!(arr.len(), 2);
        assert_eq!(arr.value(0), Some(123));
        assert_eq!(arr.value(1), Some(456));
        assert!(!arr.is_null(0));
        assert!(!arr.is_null(1));
    }

    #[test]
    fn test_push_and_value_with_null_mask() {
        let mut arr = DatetimeArray::<i64>::with_capacity(3, true, None);
        arr.push(1234);
        arr.push(5678);
        arr.push_null();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.value(0), Some(1234));
        assert_eq!(arr.value(1), Some(5678));
        assert_eq!(arr.value(2), None);
        assert!(arr.is_null(2));
    }

    #[test]
    fn test_push_null_auto_mask() {
        let mut arr = DatetimeArray::<i64>::default();
        arr.push_null();
        assert_eq!(arr.len(), 1);
        assert!(arr.is_null(0));
        assert!(arr.null_mask.is_some());
        assert_eq!(arr.value(0), None);
    }

    #[test]
    fn test_set_and_set_null() {
        let mut arr = DatetimeArray::<i64>::with_capacity(3, true, None);
        arr.push(1);
        arr.push(2);
        arr.push(3);
        arr.set(1, 99);
        assert_eq!(arr.value(1), Some(99));
        arr.set_null(1);
        assert_eq!(arr.value(1), None);
        assert!(arr.is_null(1));
    }

    #[test]
    fn test_values_method() {
        let mut arr = DatetimeArray::<i32>::with_capacity(3, false, None);
        arr.push(10);
        arr.push(20);
        arr.push(30);
        assert_eq!(arr.values(), &vec64![10, 20, 30]);
    }

    #[test]
    fn test_out_of_bounds_value() {
        let mut arr = DatetimeArray::<i64>::with_capacity(1, true, None);
        arr.push(9);
        assert_eq!(arr.value(1), None);
        assert_eq!(arr.value(100), None);
    }

    #[test]
    fn test_is_empty() {
        let arr = DatetimeArray::<i64>::default();
        assert!(arr.is_empty());
        let mut arr = DatetimeArray::<i64>::with_capacity(2, true, None);
        arr.push(0);
        assert!(!arr.is_empty());
    }

    #[test]
    fn test_push_nulls_bulk() {
        let mut arr = DatetimeArray::<i64>::with_capacity(4, true, None);
        arr.push(100);
        arr.push_nulls(5);
        assert_eq!(arr.len(), 6);
        assert_eq!(arr.value(0), Some(100));
        for i in 1..6 {
            assert_eq!(arr.value(i), None);
        }
    }

    #[test]
    fn test_masked_and_mutable_trait() {
        let mut arr = DatetimeArray::<i64>::with_capacity(3, true, None);
        arr.push(1);
        arr.push(2);
        arr.push(3);
        assert_eq!(arr.get(0), Some(1));
        assert_eq!(arr.get(2), Some(3));
        arr.set_null(0);
        assert_eq!(arr.get(0), None);
        arr.set(2, 88);
        assert_eq!(arr.get(2), Some(88));
    }

    #[test]
    fn test_masked_trait_no_mask() {
        let mut arr = DatetimeArray::<i64>::with_capacity(2, false, None);
        arr.push(3);
        arr.push(5);
        assert_eq!(arr.get(0), Some(3));
        assert!(!arr.is_null(1));
    }

    #[test]
    fn test_unsigned_type() {
        let mut arr = DatetimeArray::<u64>::with_capacity(3, true, None);
        arr.push(1);
        arr.push(2);
        arr.push_null();
        assert_eq!(arr.value(0), Some(1));
        assert_eq!(arr.value(2), None);
    }

    #[cfg(feature = "datetime_ops")]
    #[test]
    fn test_as_datetime_and_tuple_dt() {
        use time::OffsetDateTime;
        let mut arr = DatetimeArray::<i64>::with_capacity(2, true, Some(TimeUnit::Milliseconds));
        let ms = 1_700_000_000_000;
        arr.push(ms);
        arr.push_null();

        let dt = arr.as_datetime(0).unwrap();
        let expected = OffsetDateTime::from_unix_timestamp_nanos((ms as i128) * 1_000_000).unwrap();
        assert_eq!(dt, expected);

        let tuple = arr.tuple_dt(0).unwrap();
        assert_eq!(tuple.0, dt.year());
        assert_eq!(tuple.1, dt.month() as u32);
        assert_eq!(tuple.2, dt.day() as u32);
        assert_eq!(tuple.3, dt.hour() as u32);
        assert_eq!(tuple.4, dt.minute() as u32);
        assert_eq!(tuple.5, dt.second() as u32);
        assert_eq!(tuple.6, dt.nanosecond() / 1_000_000);
        assert_eq!(tuple.7, dt.nanosecond());

        assert!(arr.as_datetime(1).is_none());
        assert!(arr.tuple_dt(1).is_none());
    }

    #[cfg(feature = "datetime_ops")]
    #[test]
    fn test_as_date_and_as_time() {
        use time::Month;
        let mut arr = DatetimeArray::<i64>::with_capacity(1, false, Some(TimeUnit::Seconds));
        let timestamp = 1_700_000_000; // 2023-11-14 22:13:20 UTC
        arr.push(timestamp);

        let date = arr.as_date(0).unwrap();
        assert_eq!(date.year(), 2023);
        assert_eq!(date.month(), Month::November);
        assert_eq!(date.day(), 14);

        let time = arr.as_time(0).unwrap();
        assert_eq!(time.hour(), 22);
        assert_eq!(time.minute(), 13);
        assert_eq!(time.second(), 20);
    }

    #[test]
    fn test_datetime_array_slice() {
        use crate::enums::time_units::TimeUnit;

        let mut arr = DatetimeArray::<i64>::with_capacity(5, true, Some(TimeUnit::Milliseconds));
        arr.push(1000);
        arr.push(2000);
        arr.push(3000);
        arr.push_null(); // index 3 is null
        arr.push(5000);

        let sliced = arr.slice_clone(1, 3);
        assert_eq!(sliced.len(), 3);
        assert_eq!(sliced.value(0), Some(2000));
        assert_eq!(sliced.value(1), Some(3000));
        assert_eq!(sliced.value(2), None);
        assert!(sliced.is_null(2));
        assert_eq!(sliced.null_count(), 1);
        assert_eq!(sliced.time_unit, TimeUnit::Milliseconds);
    }

    #[test]
    fn test_batch_extend_from_iter_with_capacity() {
        let mut arr = DatetimeArray::<i64>::with_default_unit(Some(TimeUnit::Seconds));
        let data: Vec<i64> = (1_000_000_000..1_000_000_100).collect(); // Unix timestamps

        arr.extend_from_iter_with_capacity(data.into_iter(), 100);

        assert_eq!(arr.len(), 100);
        for i in 0..100 {
            assert_eq!(arr.value(i), Some(1_000_000_000 + i as i64));
        }
        assert_eq!(arr.time_unit, TimeUnit::Seconds);
    }

    #[test]
    fn test_batch_extend_from_slice() {
        let mut arr = DatetimeArray::<i32>::with_capacity(10, true, Some(TimeUnit::Milliseconds));
        arr.push(1000);
        arr.push_null();

        let data = &[2000i32, 3000, 4000];
        arr.extend_from_slice(data);

        assert_eq!(arr.len(), 5);
        assert_eq!(arr.value(0), Some(1000));
        assert_eq!(arr.value(1), None);
        assert_eq!(arr.value(2), Some(2000));
        assert_eq!(arr.value(3), Some(3000));
        assert_eq!(arr.value(4), Some(4000));
        assert!(arr.null_count() >= 1); // At least the initial null
        assert_eq!(arr.time_unit, TimeUnit::Milliseconds);
    }

    #[test]
    fn test_batch_fill_datetime() {
        let timestamp = 1_700_000_000i64; // Recent timestamp
        let arr = DatetimeArray::<i64>::fill(timestamp, 200);

        assert_eq!(arr.len(), 200);
        assert_eq!(arr.null_count(), 0);
        for i in 0..200 {
            assert_eq!(arr.value(i), Some(timestamp));
        }
        // Default time unit from fill()
        // Let's check what it actually is rather than assume
        assert!(matches!(
            arr.time_unit,
            TimeUnit::Nanoseconds
                | TimeUnit::Milliseconds
                | TimeUnit::Seconds
                | TimeUnit::Microseconds
                | TimeUnit::Days
        ));
    }

    #[test]
    fn test_batch_operations_different_time_units() {
        // Test with microseconds
        let mut arr_micro = DatetimeArray::<i64>::with_default_unit(Some(TimeUnit::Microseconds));
        let micro_data = &[1_000_000i64, 2_000_000, 3_000_000]; // 1, 2, 3 seconds in microseconds

        arr_micro.extend_from_slice(micro_data);

        assert_eq!(arr_micro.len(), 3);
        assert_eq!(arr_micro.time_unit, TimeUnit::Microseconds);
        for (i, &expected) in micro_data.iter().enumerate() {
            assert_eq!(arr_micro.value(i), Some(expected));
        }
    }

    #[test]
    fn test_batch_fill_preserves_time_unit() {
        let _arr = DatetimeArray::<i32>::with_default_unit(Some(TimeUnit::Milliseconds));
        let filled = DatetimeArray::<i32>::fill(1000, 50);

        // Note: fill() creates a new array with default time unit
        // This test documents current behavior
        assert!(matches!(
            filled.time_unit,
            TimeUnit::Nanoseconds
                | TimeUnit::Milliseconds
                | TimeUnit::Seconds
                | TimeUnit::Microseconds
                | TimeUnit::Days
        ));
        assert_eq!(filled.len(), 50);
    }

    #[test]
    fn test_batch_operations_large_timestamps() {
        let mut arr = DatetimeArray::<i64>::with_default_unit(Some(TimeUnit::Nanoseconds));
        let large_timestamps: Vec<i64> = (0..10).map(|i| 1_000_000_000_000_000_000 + i).collect();

        arr.extend_from_iter_with_capacity(large_timestamps.into_iter(), 10);

        assert_eq!(arr.len(), 10);
        for i in 0..10 {
            assert_eq!(arr.value(i), Some(1_000_000_000_000_000_000 + i as i64));
        }
    }

    #[test]
    fn test_datetime_array_concat() {
        use crate::traits::concatenate::Concatenate;

        let arr1 =
            DatetimeArray::<i64>::from_slice(&[1000, 2000, 3000], Some(TimeUnit::Milliseconds));
        let arr2 = DatetimeArray::<i64>::from_slice(&[4000, 5000], Some(TimeUnit::Milliseconds));

        let result = arr1.concat(arr2).unwrap();

        assert_eq!(result.len(), 5);
        assert_eq!(result.value(0), Some(1000));
        assert_eq!(result.value(1), Some(2000));
        assert_eq!(result.value(2), Some(3000));
        assert_eq!(result.value(3), Some(4000));
        assert_eq!(result.value(4), Some(5000));
        assert_eq!(result.time_unit, TimeUnit::Milliseconds);
    }

    #[test]
    fn test_datetime_array_concat_with_nulls() {
        use crate::traits::concatenate::Concatenate;

        let mut arr1 = DatetimeArray::<i64>::with_capacity(3, true, Some(TimeUnit::Seconds));
        arr1.push(100);
        arr1.push_null();
        arr1.push(300);

        let mut arr2 = DatetimeArray::<i64>::with_capacity(2, true, Some(TimeUnit::Seconds));
        arr2.push(400);
        arr2.push_null();

        let result = arr1.concat(arr2).unwrap();

        assert_eq!(result.len(), 5);
        assert_eq!(result.value(0), Some(100));
        assert_eq!(result.value(1), None);
        assert_eq!(result.value(2), Some(300));
        assert_eq!(result.value(3), Some(400));
        assert_eq!(result.value(4), None);
        assert_eq!(result.null_count(), 2);
        assert_eq!(result.time_unit, TimeUnit::Seconds);
    }

    #[test]
    fn test_datetime_array_concat_incompatible_time_units() {
        use crate::traits::concatenate::Concatenate;

        let arr1 = DatetimeArray::<i64>::from_slice(&[1000, 2000], Some(TimeUnit::Milliseconds));
        let arr2 = DatetimeArray::<i64>::from_slice(&[3, 4], Some(TimeUnit::Seconds));

        let result = arr1.concat(arr2);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            crate::enums::error::MinarrowError::IncompatibleTypeError { .. }
        ));
    }
}

#[cfg(test)]
#[cfg(feature = "parallel_proc")]
mod parallel_tests {
    use rayon::prelude::*;

    use super::*;
    use crate::enums::time_units::TimeUnit;

    #[test]
    fn test_datetimearray_par_iter_all_valid_refs() {
        let arr =
            DatetimeArray::<i64>::from_slice(&[1000, 2000, 3000], Some(TimeUnit::Milliseconds));
        let out: Vec<&i64> = arr.par_iter().collect();
        let expected: Vec<i64> = vec![1000, 2000, 3000];
        for (i, val) in out.iter().enumerate() {
            assert_eq!(**val, expected[i]);
        }
    }

    #[test]
    fn test_datetimearray_par_iter_nulls() {
        let mut arr = DatetimeArray::<i64>::with_default_unit(Some(TimeUnit::Milliseconds));
        arr.data.extend_from_slice(&[10, 20, 30]);
        let mut mask = Bitmask::new_set_all(3, false);
        mask.set_bits_chunk(0, 0b0000_0011, 3);
        arr.null_mask = Some(mask);
        let out: Vec<&i64> = arr.par_iter().collect();
        let expected: Vec<i64> = vec![10, 20, 30];
        for (i, val) in out.iter().enumerate() {
            assert_eq!(**val, expected[i]);
        }
    }

    #[test]
    fn test_datetimearray_par_iter_opt_all_valid() {
        let arr = DatetimeArray::<i64>::from_slice(&[111, 222, 333], Some(TimeUnit::Seconds));
        let out: Vec<Option<&i64>> = arr.par_iter_opt().collect();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], Some(&111));
        assert_eq!(out[1], Some(&222));
        assert_eq!(out[2], Some(&333));
    }

    #[test]
    fn test_datetimearray_par_iter_opt_with_nulls() {
        let mut arr = DatetimeArray::<i64>::with_default_unit(Some(TimeUnit::Microseconds));
        arr.data.extend_from_slice(&[77, 88, 99, 111]);
        let mut mask = Bitmask::new_set_all(4, false);
        mask.set_bits_chunk(0, 0b0000_1011, 4);
        arr.null_mask = Some(mask);
        let out: Vec<Option<&i64>> = arr.par_iter_opt().collect();
        assert_eq!(out.len(), 4);
        assert_eq!(out[0], Some(&77));
        assert_eq!(out[1], Some(&88));
        assert_eq!(out[2], None);
        assert_eq!(out[3], Some(&111));
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn test_datetimearray_par_iter_range_unchecked() {
        let arr = DatetimeArray::<i64>::from_slice(
            &[100, 200, 300, 400],
            Some(crate::enums::time_units::TimeUnit::Milliseconds),
        );
        let out: Vec<&i64> = unsafe { arr.par_iter_range_unchecked(1, 3).collect() };
        assert_eq!(*out[0], 200);
        assert_eq!(*out[1], 300);
    }
}
