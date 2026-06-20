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

//! # **TemporalArrayView Module** - *Windowed View over a TemporalArray*
//!
//! `TemporalArrayV` is a **read-only, windowed view** over a [`TemporalArray`].
//! It exposes a zero-copy slice `[offset .. offset + len)` for fast, indexable
//! access to datetime/timestamp data.
//!
//! ## Role
//! - Let APIs accept either a full `TemporalArray` or a pre-sliced view.
//! - Avoid deep copies while enabling per-window operations and previews.
//! - Optionally cache per-window null counts to speed up repeated scans.
//!
//! ## Behaviour
//! - Works with 32-bit and 64-bit datetime variants behind `TemporalArray`.
//! - Accessors - [`get_i64`](TemporalArrayV::get_i64) and
//!   [`get_i32`](TemporalArrayV::get_i32) (the latter returns `None` for 64-bit).
//! - Slicing yields another borrowed view; buffers are not cloned.
//!
//! ## Threading
//! - Not thread-safe: uses `Cell` to cache the window’s null count.
//! - For parallel use, create per-thread views via [`slice`](TemporalArrayV::slice).
//!
//! ## Interop
//! - Convert to an owned `TemporalArray` of the window with
//!   [`to_temporal_array`](TemporalArrayV::to_temporal_array).
//! - Lift to `Array` with [`inner_array`](TemporalArrayV::inner_array) for enum-level APIs.
//!
//! ## Invariants
//! - `offset + len <= array.len()`
//! - `len` is the logical element count of this view.

use std::fmt::{self, Debug, Display, Formatter};
use std::sync::OnceLock;

use crate::enums::error::MinarrowError;
use crate::enums::shape_dim::ShapeDim;
use crate::traits::concatenate::Concatenate;
use crate::traits::print::MAX_PREVIEW;
use crate::traits::shape::Shape;
use crate::{Array, ArrayV, BitmaskV, MaskedArray, TemporalArray};

/// # TemporalArrayView
///
/// Read-only, zero-copy view over a `[offset .. offset + len)` window of a
/// [`TemporalArray`].
///
/// ## Purpose
/// - Return an indexable subrange without cloning buffers.
/// - Optionally cache per-window null counts for faster repeated passes.
///
/// ## Behaviour
/// - Supports 32-bit and 64-bit datetime storage behind `TemporalArray`.
/// - Provides upcast helpers - [`get_i64`](Self::get_i64) and
///   [`get_i32`](Self::get_i32).
/// - Further slicing yields another borrowed view.
///
/// ## Fields
/// - `array`: backing [`TemporalArray`] (enum over temporal types).
/// - `offset`: starting index into the backing array.
/// - `len`: logical number of elements in the view.
/// - `null_count`: cached `Option<usize>` for this window (internal).
///
/// ## Notes
/// - Not thread-safe due to `Cell`. Create per-thread views with [`slice`](Self::slice).
/// - Use [`to_temporal_array`](Self::to_temporal_array) to materialise the window.
#[derive(Clone, PartialEq)]
pub struct TemporalArrayV {
    /// The **outer array** that this view is derived from - we retain a reference to it. 
    /// Importantly, this is the ***full array*** - not the *view*, and thus should not be 
    /// accessed as though it were the view subset.
    pub array: TemporalArray,
    /// The index offset from 0 that for where this view starts from the outer array
    pub offset: usize,
    /// The length of the array view
    len: usize,
    /// How many nulls are in the TemporalArrayView
    /// At construction, this is None, unless constructed via new_nc. When one uses '.null_count()',
    /// the first time it will calculate it (quickly) using Bitmask popcount, and then from that 
    /// point onwards the null count is a cached value. 
    null_count: OnceLock<usize>,
}

impl TemporalArrayV {
    /// Creates a new `TemporalArrayView` with the given offset and length.
    pub fn new(array: TemporalArray, offset: usize, len: usize) -> Self {
        assert!(
            offset + len <= array.len(),
            "TemporalArrayView: window out of bounds (offset + len = {}, array.len = {})",
            offset + len,
            array.len()
        );
        Self {
            array,
            offset,
            len,
            null_count: OnceLock::new(),
        }
    }

    /// Creates a new `TemporalArrayView` with a precomputed null count.
    pub fn new_nc(
        array: TemporalArray,
        offset: usize,
        len: usize,
        null_count: usize,
    ) -> Self {
        assert!(
            offset + len <= array.len(),
            "TemporalArrayView: window out of bounds (offset + len = {}, array.len = {})",
            offset + len,
            array.len()
        );
        let lock = OnceLock::new();
        let _ = lock.set(null_count); // Pre-initialize with the provided count
        Self {
            array,
            offset,
            len,
            null_count: lock,
        }
    }

    /// Returns `true` if the view is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the value at logical index `i` within the window as i64 (upcasts i32 variant).
    #[inline]
    pub fn get_i64(&self, i: usize) -> Option<i64> {
        if i >= self.len {
            return None;
        }
        let phys_idx = self.offset + i;
        match &self.array {
            TemporalArray::Datetime32(arr) => arr.get(phys_idx).map(|v| v as i64),
            TemporalArray::Datetime64(arr) => arr.get(phys_idx),
            TemporalArray::Null => None,
        }
    }

    /// Returns the value at logical index `i` as i32 (None for Datetime64).
    #[inline]
    pub fn get_i32(&self, i: usize) -> Option<i32> {
        if i >= self.len {
            return None;
        }
        let phys_idx = self.offset + i;
        match &self.array {
            TemporalArray::Datetime32(arr) => arr.get(phys_idx),
            TemporalArray::Datetime64(_) => None,
            TemporalArray::Null => None,
        }
    }

    /// Returns a sliced `TemporalArrayView` from the current view.
    #[inline]
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        assert!(
            offset + len <= self.len,
            "TemporalArrayView::slice: out of bounds"
        );
        Self {
            array: self.array.clone(),
            offset: self.offset + offset,
            len,
            null_count: OnceLock::new(),
        }
    }

    /// Returns the length of the window
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the full backing array wrapped as an `Array` enum, ignoring the view's offset and length.
    ///
    /// Use this to access inner array methods. The returned array is the unwindowed original.
    #[inline]
    pub fn inner_array(&self) -> Array {
        Array::TemporalArray(self.array.clone()) // Arc clone for data buffer
    }

    /// Converts the view into an owned `TemporalArray` for the window.
    ///
    /// If the view covers the entire backing array, returns a cheap Arc clone
    /// of the original variant. Otherwise deep-copies the window via
    /// `slice_clone` through `inner_array`.
    pub fn to_temporal_array(&self) -> TemporalArray {
        if self.offset == 0 && self.len == self.array.len() {
            return self.array.clone();
        }
        self.inner_array().slice_clone(self.offset, self.len).dt()
    }

    /// Returns the end index of the view.
    #[inline]
    pub fn end(&self) -> usize {
        self.offset + self.len
    }

    /// Returns the view as a tuple `(array, offset, len)`
    #[inline]
    pub fn as_tuple(&self) -> (TemporalArray, usize, usize) {
        (self.array.clone(), self.offset, self.len)
    }

    /// Returns the number of nulls in the view.
    #[inline]
    pub fn null_count(&self) -> usize {
        *self
            .null_count
            .get_or_init(|| match self.array.null_mask() {
                Some(mask) => mask.view(self.offset, self.len).count_zeros(),
                None => 0,
            })
    }

    /// Returns true when the windowed view holds at least one null.
    ///
    /// Reads through `null_count`, so the cached value is trusted when set
    /// and the full popcount is only paid on the first call that observes
    /// this view.
    #[inline]
    pub fn has_nulls(&self) -> bool {
        self.null_count() > 0
    }

    /// Returns the null mask as a windowed `BitmaskView`.
    #[inline]
    pub fn null_mask_view(&self) -> Option<BitmaskV<'_>> {
        self.array
            .null_mask()
            .map(|mask| mask.view(self.offset, self.len))
    }

    /// Sets the cached null count for the view.
    #[inline]
    pub fn set_null_count(&self, count: usize) -> Result<(), usize> {
        self.null_count.set(count).map_err(|_| count)
    }
}

impl From<TemporalArray> for TemporalArrayV {
    fn from(array: TemporalArray) -> Self {
        let len = array.len();
        TemporalArrayV {
            array,
            offset: 0,
            len,
            null_count: OnceLock::new(),
        }
    }
}

impl From<Array> for TemporalArrayV {
    fn from(array: Array) -> Self {
        match array {
            Array::TemporalArray(arr) => {
                let len = arr.len();
                TemporalArrayV {
                    array: arr,
                    offset: 0,
                    len,
                    null_count: OnceLock::new(),
                }
            }
            _ => panic!("Array is not a TemporalArray"),
        }
    }
}

impl From<ArrayV> for TemporalArrayV {
    /// Converts an `ArrayView` to a `TemporalArrayView`, panicking if the array is not temporal.
    fn from(view: ArrayV) -> Self {
        let (array, offset, len) = view.as_tuple();
        match array {
            Array::TemporalArray(inner) => Self {
                array: inner,
                offset,
                len,
                null_count: OnceLock::new(),
            },
            _ => panic!("From<ArrayView>: expected TemporalArray variant"),
        }
    }
}

impl Debug for TemporalArrayV {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("TemporalArrayView")
            .field("offset", &self.offset)
            .field("len", &self.len)
            .field("array", &self.array)
            .field("cached_null_count", &self.null_count.get())
            .finish()
    }
}

impl Display for TemporalArrayV {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let dtype = match &self.array {
            TemporalArray::Datetime32(_) => "Datetime32<i32>",
            TemporalArray::Datetime64(_) => "Datetime64<i64>",
            TemporalArray::Null => "Null",
        };

        writeln!(
            f,
            "TemporalArrayView<{dtype}> [{} values] (offset: {}, nulls: {})",
            self.len(),
            self.offset,
            self.null_count()
        )?;

        let max = self.len().min(MAX_PREVIEW);

        #[cfg(feature = "datetime_ops")]
        {
            use time::OffsetDateTime;

            use crate::TimeUnit;
            let unit = match &self.array {
                TemporalArray::Datetime32(arr) => &arr.time_unit,
                TemporalArray::Datetime64(arr) => &arr.time_unit,
                TemporalArray::Null => &TimeUnit::Milliseconds,
            };
            for i in 0..max {
                match self.get_i64(i) {
                    Some(val) => match unit {
                        TimeUnit::Seconds => match OffsetDateTime::from_unix_timestamp(val) {
                            Ok(dt) => writeln!(f, "  {dt}")?,
                            Err(_) => writeln!(f, "  {val}s")?,
                        },
                        TimeUnit::Milliseconds => {
                            match OffsetDateTime::from_unix_timestamp_nanos(
                                (val as i128) * 1_000_000,
                            ) {
                                Ok(dt) => writeln!(f, "  {dt}")?,
                                Err(_) => writeln!(f, "  {val}ms")?,
                            }
                        }
                        TimeUnit::Microseconds => {
                            match OffsetDateTime::from_unix_timestamp_nanos((val as i128) * 1_000) {
                                Ok(dt) => writeln!(f, "  {dt}")?,
                                Err(_) => writeln!(f, "  {val}µs")?,
                            }
                        }
                        TimeUnit::Nanoseconds => {
                            match OffsetDateTime::from_unix_timestamp_nanos(val as i128) {
                                Ok(dt) => writeln!(f, "  {dt}")?,
                                Err(_) => writeln!(f, "  {val}ns")?,
                            }
                        }
                        TimeUnit::Days => {
                            use crate::structs::variants::datetime::UNIX_EPOCH_JULIAN_DAY;

                            let days = val;
                            match time::Date::from_julian_day((days + UNIX_EPOCH_JULIAN_DAY) as i32)
                            {
                                Ok(d) => writeln!(f, "  {d}")?,
                                Err(_) => writeln!(f, "  {days}d")?,
                            }
                        }
                    },
                    None => writeln!(f, "  null")?,
                }
            }
        }

        #[cfg(not(feature = "datetime_ops"))]
        {
            use crate::TimeUnit;

            let unit = match &self.array {
                TemporalArray::Datetime32(arr) => &arr.time_unit,
                TemporalArray::Datetime64(arr) => &arr.time_unit,
                TemporalArray::Null => &TimeUnit::Milliseconds,
            };
            let suffix = match unit {
                TimeUnit::Seconds => "s",
                TimeUnit::Milliseconds => "ms",
                TimeUnit::Microseconds => "µs",
                TimeUnit::Nanoseconds => "ns",
                TimeUnit::Days => "d",
            };
            for i in 0..max {
                match self.get_i64(i) {
                    Some(val) => writeln!(f, "  {}{}", val, suffix)?,
                    None => writeln!(f, "  null")?,
                }
            }
        }

        if self.len() > MAX_PREVIEW {
            writeln!(f, "  ... ({} more)", self.len() - MAX_PREVIEW)?;
        }

        Ok(())
    }
}

#[cfg(feature = "datetime_ops")]
use crate::DatetimeOps;

#[cfg(feature = "datetime_ops")]
use crate::enums::time_units::TimeUnit;

#[cfg(feature = "datetime_ops")]
use time::Duration;

#[cfg(feature = "datetime_ops")]
use crate::structs::variants::{boolean::BooleanArray, integer::IntegerArray};

#[cfg(feature = "datetime_ops")]
use crate::DatetimeArray;

#[cfg(feature = "datetime_ops")]
use num_traits::{FromPrimitive, ToPrimitive};

#[cfg(feature = "datetime_ops")]
use std::sync::Arc;

/// Macro to dispatch a windowed component extraction over both temporal variants.
///
/// Iterates `self.offset .. self.offset + self.len` on the inner `DatetimeArray`,
/// pushing the extracted component into a fresh `IntegerArray<i32>`.
#[cfg(feature = "datetime_ops")]
macro_rules! windowed_component {
    ($self:expr, $extract:expr) => {{
        let offset = $self.offset;
        let len = $self.len();
        match &$self.array {
            TemporalArray::Datetime32(arr) => {
                windowed_component_inner!(arr, offset, len, $extract)
            }
            TemporalArray::Datetime64(arr) => {
                windowed_component_inner!(arr, offset, len, $extract)
            }
            TemporalArray::Null => IntegerArray::default(),
        }
    }};
}

#[cfg(feature = "datetime_ops")]
macro_rules! windowed_component_inner {
    ($arr:expr, $offset:expr, $len:expr, $extract:expr) => {{
        let mut result = IntegerArray::with_capacity($len, $arr.is_nullable());
        for i in $offset..$offset + $len {
            if $arr.is_null(i) {
                result.push_null();
            } else if let Some(val_i64) = $arr.data[i].to_i64() {
                if let Some(dt) = DatetimeArray::<i64>::i64_to_datetime(val_i64, $arr.time_unit) {
                    #[allow(clippy::redundant_closure_call)]
                    result.push(($extract)(&dt));
                } else {
                    result.push_null();
                }
            } else {
                result.push_null();
            }
        }
        result
    }};
}

/// Macro to dispatch a windowed Self-returning operation over both temporal variants.
#[cfg(feature = "datetime_ops")]
macro_rules! windowed_self_op {
    ($self:expr, $method:ident $(, $arg:expr)*) => {{
        let offset = $self.offset;
        let len = $self.len();
        match &$self.array {
            TemporalArray::Datetime32(arr) => {
                let sliced = arr.slice_clone(offset, len);
                let result = sliced.$method($($arg),*)?;
                Ok(TemporalArrayV::from(TemporalArray::Datetime32(Arc::new(result))))
            }
            TemporalArray::Datetime64(arr) => {
                let sliced = arr.slice_clone(offset, len);
                let result = sliced.$method($($arg),*)?;
                Ok(TemporalArrayV::from(TemporalArray::Datetime64(Arc::new(result))))
            }
            TemporalArray::Null => Err(MinarrowError::NullError { message: None }),
        }
    }};
}

/// Macro for infallible Self-returning operations.
#[cfg(feature = "datetime_ops")]
macro_rules! windowed_self_op_infallible {
    ($self:expr, $method:ident $(, $arg:expr)*) => {{
        let offset = $self.offset;
        let len = $self.len();
        match &$self.array {
            TemporalArray::Datetime32(arr) => {
                let sliced = arr.slice_clone(offset, len);
                let result = sliced.$method($($arg),*);
                TemporalArrayV::from(TemporalArray::Datetime32(Arc::new(result)))
            }
            TemporalArray::Datetime64(arr) => {
                let sliced = arr.slice_clone(offset, len);
                let result = sliced.$method($($arg),*);
                TemporalArrayV::from(TemporalArray::Datetime64(Arc::new(result)))
            }
            TemporalArray::Null => TemporalArrayV::from(TemporalArray::Null),
        }
    }};
}

#[cfg(feature = "datetime_ops")]
use crate::TimePeriod;

#[cfg(feature = "datetime_ops")]
impl DatetimeOps for TemporalArrayV {
    // Component Extraction - iterate only the windowed range

    fn year(&self) -> IntegerArray<i32> {
        windowed_component!(self, |dt: &time::OffsetDateTime| dt.year())
    }

    fn month(&self) -> IntegerArray<i32> {
        windowed_component!(self, |dt: &time::OffsetDateTime| dt.month() as i32)
    }

    fn day(&self) -> IntegerArray<i32> {
        windowed_component!(self, |dt: &time::OffsetDateTime| dt.day() as i32)
    }

    fn hour(&self) -> IntegerArray<i32> {
        windowed_component!(self, |dt: &time::OffsetDateTime| dt.hour() as i32)
    }

    fn minute(&self) -> IntegerArray<i32> {
        windowed_component!(self, |dt: &time::OffsetDateTime| dt.minute() as i32)
    }

    fn second(&self) -> IntegerArray<i32> {
        windowed_component!(self, |dt: &time::OffsetDateTime| dt.second() as i32)
    }

    fn weekday(&self) -> IntegerArray<i32> {
        windowed_component!(
            self,
            |dt: &time::OffsetDateTime| dt.weekday().number_from_sunday() as i32
        )
    }

    fn day_of_year(&self) -> IntegerArray<i32> {
        windowed_component!(self, |dt: &time::OffsetDateTime| dt.ordinal() as i32)
    }

    fn iso_week(&self) -> IntegerArray<i32> {
        windowed_component!(self, |dt: &time::OffsetDateTime| dt.iso_week() as i32)
    }

    fn quarter(&self) -> IntegerArray<i32> {
        windowed_component!(self, |dt: &time::OffsetDateTime| {
            let month = dt.month() as i32;
            ((month - 1) / 3) + 1
        })
    }

    fn week_of_year(&self) -> IntegerArray<i32> {
        windowed_component!(self, |dt: &time::OffsetDateTime| {
            let day_of_year = dt.ordinal() as i32;
            let weekday = dt.weekday().number_from_sunday() as i32;
            (day_of_year + 7 - weekday) / 7
        })
    }

    fn is_leap_year(&self) -> BooleanArray<()> {
        let offset = self.offset;
        let len = self.len();
        match &self.array {
            TemporalArray::Datetime32(arr) => is_leap_year_windowed(arr, offset, len),
            TemporalArray::Datetime64(arr) => is_leap_year_windowed(arr, offset, len),
            TemporalArray::Null => BooleanArray::default(),
        }
    }

    // Arithmetic - slice the windowed range, delegate, wrap result as view

    fn add_duration(&self, duration: Duration) -> Result<Self, MinarrowError> {
        windowed_self_op!(self, add_duration, duration)
    }

    fn sub_duration(&self, duration: Duration) -> Result<Self, MinarrowError> {
        windowed_self_op!(self, sub_duration, duration)
    }

    fn add_days(&self, days: i64) -> Result<Self, MinarrowError> {
        windowed_self_op!(self, add_days, days)
    }

    fn add_months(&self, months: i32) -> Result<Self, MinarrowError> {
        windowed_self_op!(self, add_months, months)
    }

    fn add_years(&self, years: i32) -> Result<Self, MinarrowError> {
        windowed_self_op!(self, add_years, years)
    }

    // Comparison - slice both operands, delegate

    fn diff(&self, other: &Self, unit: TimeUnit) -> Result<IntegerArray<i64>, MinarrowError> {
        let self_arr = self.to_temporal_array();
        let other_arr = other.to_temporal_array();
        self_arr.diff(&other_arr, unit)
    }

    fn abs_diff(&self, other: &Self, unit: TimeUnit) -> Result<IntegerArray<i64>, MinarrowError> {
        let self_arr = self.to_temporal_array();
        let other_arr = other.to_temporal_array();
        self_arr.abs_diff(&other_arr, unit)
    }

    fn is_before(&self, other: &Self) -> Result<BooleanArray<()>, MinarrowError> {
        let self_arr = self.to_temporal_array();
        let other_arr = other.to_temporal_array();
        self_arr.is_before(&other_arr)
    }

    fn is_after(&self, other: &Self) -> Result<BooleanArray<()>, MinarrowError> {
        let self_arr = self.to_temporal_array();
        let other_arr = other.to_temporal_array();
        self_arr.is_after(&other_arr)
    }

    fn between(&self, start: &Self, end: &Self) -> Result<BooleanArray<()>, MinarrowError> {
        let self_arr = self.to_temporal_array();
        let start_arr = start.to_temporal_array();
        let end_arr = end.to_temporal_array();
        self_arr.between(&start_arr, &end_arr)
    }

    // Truncation - slice the windowed range, delegate

    fn truncate<P: Into<TimePeriod>>(&self, period: P) -> Self {
        let period: TimePeriod = period.into();
        windowed_self_op_infallible!(self, truncate, period)
    }

    fn us(&self) -> Self {
        windowed_self_op_infallible!(self, us)
    }

    fn ms(&self) -> Self {
        windowed_self_op_infallible!(self, ms)
    }

    fn sec(&self) -> Self {
        windowed_self_op_infallible!(self, sec)
    }

    fn min(&self) -> Self {
        windowed_self_op_infallible!(self, min)
    }

    fn hr(&self) -> Self {
        windowed_self_op_infallible!(self, hr)
    }

    fn week(&self) -> Self {
        windowed_self_op_infallible!(self, week)
    }

    // Type Casting

    fn cast_time_unit(&self, new_unit: TimeUnit) -> Result<Self, MinarrowError> {
        windowed_self_op!(self, cast_time_unit, new_unit)
    }
}

/// Helper for windowed `is_leap_year` over a generic `DatetimeArray<T>`.
#[cfg(feature = "datetime_ops")]
fn is_leap_year_windowed<T: crate::Integer + FromPrimitive>(
    arr: &DatetimeArray<T>,
    offset: usize,
    len: usize,
) -> BooleanArray<()> {
    let mut result = BooleanArray::with_capacity(len, arr.is_nullable());
    for i in offset..offset + len {
        if arr.is_null(i) {
            result.push_null();
        } else if let Some(val_i64) = arr.data[i].to_i64() {
            if let Some(dt) = DatetimeArray::<i64>::i64_to_datetime(val_i64, arr.time_unit) {
                result.push(time::util::is_leap_year(dt.year()));
            } else {
                result.push_null();
            }
        } else {
            result.push_null();
        }
    }
    result
}

impl Shape for TemporalArrayV {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank1(self.len())
    }
}

impl Concatenate for TemporalArrayV {
    /// Concatenates two temporal array views by materialising both to owned temporal arrays,
    /// concatenating them, and wrapping the result back in a view.
    ///
    /// # Notes
    /// - This operation copies data from both views to create owned temporal arrays.
    /// - The resulting view has offset=0 and length equal to the combined length.
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        // Materialise both views to owned temporal arrays
        let self_array = self.to_temporal_array();
        let other_array = other.to_temporal_array();

        // Concatenate the owned temporal arrays
        let concatenated = self_array.concat(other_array)?;

        // Wrap the result in a new view
        Ok(TemporalArrayV::from(concatenated))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{Bitmask, DatetimeArray, TemporalArray};

    #[test]
    fn test_temporal_array_view_basic_indexing_and_slice() {
        let arr = DatetimeArray::<i64>::from_slice(&[10_000, 20_000, 30_000, 40_000], None);
        let temporal = TemporalArray::Datetime64(Arc::new(arr));
        let view = TemporalArrayV::new(temporal, 1, 2);

        assert_eq!(view.len(), 2);
        assert_eq!(view.offset, 1);
        assert_eq!(view.get_i64(0), Some(20_000));
        assert_eq!(view.get_i64(1), Some(30_000));
        assert_eq!(view.get_i64(2), None);

        let sub = view.slice(1, 1);
        assert_eq!(sub.len(), 1);
        assert_eq!(sub.get_i64(0), Some(30_000));
        assert_eq!(sub.get_i64(1), None);
    }

    #[test]
    fn test_temporal_array_view_null_count_and_cache() {
        let mut arr = DatetimeArray::<i64>::from_slice(&[1, 2, 3, 4], None);
        let mut mask = Bitmask::new_set_all(4, true);
        mask.set(2, false);
        arr.null_mask = Some(mask);

        let temporal = TemporalArray::Datetime64(Arc::new(arr));
        let view = TemporalArrayV::new(temporal, 0, 4);
        assert_eq!(view.null_count(), 1, "Null count should detect one null");
        assert_eq!(view.null_count(), 1);

        let view2 = view.slice(0, 2);
        assert_eq!(view2.null_count(), 0);
        let view3 = view.slice(2, 2);
        assert_eq!(view3.null_count(), 1);
    }

    #[test]
    fn test_temporal_array_view_with_supplied_null_count() {
        let arr = DatetimeArray::<i64>::from_slice(&[5, 6], None);
        let temporal = TemporalArray::Datetime64(Arc::new(arr));
        let view = TemporalArrayV::new_nc(temporal, 0, 2, 99);
        assert_eq!(view.null_count(), 99);
        // Trying to set again should fail since it\'s already initialized
        assert!(view.set_null_count(101).is_err());
        // Still returns original value
        assert_eq!(view.null_count(), 99);
    }

    #[test]
    fn test_temporal_array_view_to_temporal_array_and_as_tuple() {
        let arr = DatetimeArray::<i64>::from_slice(&(10..20).collect::<Vec<_>>(), None);
        let temporal = TemporalArray::Datetime64(Arc::new(arr));
        let view = TemporalArrayV::new(temporal.clone(), 4, 3);
        let arr2 = view.to_temporal_array();
        if let TemporalArray::Datetime64(a2) = arr2 {
            assert_eq!(&a2.data[..], &[14, 15, 16]);
        } else {
            panic!("Unexpected variant");
        }
        let tup = view.as_tuple();
        assert_eq!(&tup.0, &temporal);
        assert_eq!(tup.1, 4);
        assert_eq!(tup.2, 3);
    }

    #[test]
    fn test_temporal_array_view_null_mask_view() {
        let mut arr = DatetimeArray::<i64>::from_slice(&[2, 4, 6], None);
        let mut mask = Bitmask::new_set_all(3, true);
        mask.set(0, false);
        arr.null_mask = Some(mask);

        let temporal = TemporalArray::Datetime64(Arc::new(arr));
        let view = TemporalArrayV::new(temporal, 1, 2);
        let mask_view = view.null_mask_view().expect("Should have mask");
        assert_eq!(mask_view.len(), 2);
        assert!(mask_view.get(0));
        assert!(mask_view.get(1));
    }

    #[test]
    fn test_temporal_array_view_from_temporal_array_and_array() {
        let arr = DatetimeArray::<i64>::from_slice(&[1, 2], None);
        let temporal = TemporalArray::Datetime64(Arc::new(arr));
        let view_from_temporal = TemporalArrayV::from(temporal.clone());
        assert_eq!(view_from_temporal.len(), 2);
        assert_eq!(view_from_temporal.get_i64(0), Some(1));

        let array = Array::TemporalArray(temporal);
        let view_from_array = TemporalArrayV::from(array);
        assert_eq!(view_from_array.len(), 2);
        assert_eq!(view_from_array.get_i64(1), Some(2));
    }

    #[test]
    #[should_panic(expected = "Array is not a TemporalArray")]
    fn test_temporal_array_view_from_array_panics_on_wrong_variant() {
        let array = Array::Null;
        let _view = TemporalArrayV::from(array);
    }

    #[test]
    fn test_to_temporal_array_full_coverage_shares_arc() {
        let arr = DatetimeArray::<i64>::from_slice(&[10, 20, 30], None);
        let src = Arc::new(arr);
        let temporal = TemporalArray::Datetime64(src.clone());

        let view = TemporalArrayV::new(temporal, 0, 3);
        let out = view.to_temporal_array();

        match out {
            TemporalArray::Datetime64(out_arc) => assert!(
                Arc::ptr_eq(&src, &out_arc),
                "full-coverage to_temporal_array should share the underlying Arc"
            ),
            _ => panic!("expected Datetime64 variant"),
        }
    }

    #[test]
    fn test_to_temporal_array_windowed_copies() {
        let arr = DatetimeArray::<i64>::from_slice(&[10, 20, 30], None);
        let src = Arc::new(arr);
        let temporal = TemporalArray::Datetime64(src.clone());

        let view = TemporalArrayV::new(temporal, 1, 2);
        let out = view.to_temporal_array();

        match out {
            TemporalArray::Datetime64(out_arc) => assert!(
                !Arc::ptr_eq(&src, &out_arc),
                "windowed to_temporal_array must allocate a fresh buffer"
            ),
            _ => panic!("expected Datetime64 variant"),
        }
    }
}
