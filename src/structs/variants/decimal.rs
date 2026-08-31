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

//! # DecimalArray - Fixed-Point Decimal Array
//!
//! Arrow-compatible, 64-byte-aligned decimal array for exact numeric values
//! with configurable precision and scale (for e.g., monetary, accounting,
//! and high-precision numeric columns) where floating-point approximation
//! is not acceptable.
//!
//! ## Representation
//! The array stores unscaled integer values in a 64-byte-aligned `Buffer<T>`,
//! where `T` is one of `i32`, `i64`, or `i128` (Decimal32, Decimal64, or
//! Decimal128 respectively). An optional Arrow-style null mask tracks valid
//! vs. null entries.
//!
//! ## Precision and Scale
//! - `precision` - total number of significant digits.
//! - `scale` - how many of those digits follow the decimal point. Positive
//!   scale shifts the decimal point left, negative scale appends trailing zeros.
//! - For example, a raw value of 12345 with scale=2 represents 123.45.
//!
//! ## Construction and Metadata
//! All constructors require `precision` and `scale` parameters to ensure
//! metadata consistency. Once created, the array propagates these values
//! through slicing, splitting, and concatenation operations. The
//! `append_array` method validates precision and scale match before combining
//! arrays, since mismatched decimal semantics would produce incorrect results.
//!
//! ## Null handling
//! Nullness uses Arrow's 1-bit convention (1 = valid, 0 = null) via `Bitmask`,
//! consistent with all other Minarrow array types.
//!
//! ## Display
//! The `Display` impl formats values scale-aware: inserting a decimal point
//! at the position determined by scale, using padding where needed, and
//! honouring null entries as the string "null".
//!
//! ## Example
//! ```ignore
//! use minarrow::DecimalArray;
//!
//! let arr = DecimalArray::<i64>::from_slice(&[12345, 67890], 10, 2);
//! assert_eq!(format!("{}", arr), "DecimalArray [2 values] (precision: 10, scale: 2, nulls: 0)\n[123.45, 678.90]");
//! ```

use std::fmt::{Display, Formatter};

use crate::enums::shape_dim::ShapeDim;
use crate::traits::concatenate::Concatenate;
use crate::traits::print::MAX_PREVIEW;
use crate::traits::shape::Shape;
use crate::traits::type_unions::Integer;
use crate::ffi::arrow_dtype::ArrowType;
use crate::{Bitmask, Buffer, Length, MaskedArray, Offset, impl_arc_masked_array, impl_array_ref_deref};
use vec64::Vec64;

/// # DecimalArray
///
/// Arrow-compatible, 64-byte-aligned fixed-point decimal array with optional null mask.
///
/// Represents exact numeric values with configurable precision and scale, backed
/// by an integer buffer (for e.g., monetary, accounting, and high-precision
/// numeric data) where floating-point approximation is not acceptable.
///
/// ## Fields
/// - `data`: backing buffer of unscaled integer values (`Buffer<T>`).
/// - `null_mask`: optional bit-packed bitmap (`1 = valid`, `0 = null`).
/// - `precision`: total number of significant digits (u8).
/// - `scale`: digits after the decimal point (i8). Positive scale shifts
///   the point left, negative scale appends trailing zeros.
///
/// ## Type parameters
/// `T` is one of `i32` (Decimal32), `i64` (Decimal64), or `i128` (Decimal128),
/// determining the maximum value range that can be represented.
///
/// ## Example
/// ```ignore
/// use minarrow::{DecimalArray, MaskedArray};
///
/// let arr = DecimalArray::<i64>::from_slice(&[12345, -67890], 10, 2);
/// assert_eq!(arr.len(), 2);
/// assert_eq!(arr.precision, 10);
/// assert_eq!(arr.scale, 2);
/// ```
#[derive(PartialEq, Clone, Debug)]
pub struct DecimalArray<T> {
    /// Backing buffer of unscaled integer values (Arrow-compatible).
    pub data: Buffer<T>,
    /// Optional null mask (bit-packed, 1=valid, 0=null).
    pub null_mask: Option<Bitmask>,
    /// Total number of significant digits.
    pub precision: u8,
    /// Digits after the decimal point. Positive shifts the point left,
    /// negative appends trailing zeros.
    pub scale: i8,
}

impl<T: Default> Default for DecimalArray<T> {
    fn default() -> Self {
        Self {
            data: Buffer::default(),
            null_mask: None,
            precision: 0,
            scale: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

impl<T: Integer> DecimalArray<T> {
    /// Constructs a new array from existing data and null mask with the given
    /// precision and scale.
    #[inline]
    pub fn new(
        data: impl Into<Buffer<T>>,
        null_mask: Option<Bitmask>,
        precision: u8,
        scale: i8,
    ) -> Self {
        let data: Buffer<T> = data.into();
        crate::utils::validate_null_mask_len(data.len(), &null_mask);
        Self {
            data,
            null_mask,
            precision,
            scale,
        }
    }

    /// Constructs an array with reserved capacity and optional null mask.
    ///
    /// ## Arguments
    /// - `cap` - capacity (number of elements) to reserve for the backing buffer.
    /// - `nullable` - if true, allocates a null-mask bit vector.
    /// - `precision` - total significant digits.
    /// - `scale` - digits after the decimal point.
    #[inline]
    pub fn with_capacity(cap: usize, nullable: bool, precision: u8, scale: i8) -> Self {
        Self {
            data: Vec64::with_capacity(cap).into(),
            null_mask: if nullable {
                Some(Bitmask::new_set_all(cap, true))
            } else {
                None
            },
            precision,
            scale,
        }
    }

    /// Constructs a contiguous DecimalArray from a slice with the given
    /// precision and scale. The null mask must be applied after construction
    /// if needed.
    #[inline]
    pub fn from_slice(slice: &[T], precision: u8, scale: i8) -> Self {
        Self {
            data: Vec64(slice.to_vec_in(vec64::Vec64Alloc::default())).into(),
            null_mask: None,
            precision,
            scale,
        }
    }

    /// Constructs from an already 64-byte-aligned buffer, taking ownership
    /// without copying.
    #[inline]
    pub fn from_vec64(
        data: Vec64<T>,
        null_mask: Option<Bitmask>,
        precision: u8,
        scale: i8,
    ) -> Self {
        Self {
            data: data.into(),
            null_mask,
            precision,
            scale,
        }
    }

    /// Constructs from a standard `Vec<T>`, aligning to 64-byte boundaries
    /// where needed.
    #[inline]
    pub fn from_vec(data: Vec<T>, precision: u8, scale: i8) -> Self {
        Self::from_vec64(data.into(), None, precision, scale)
    }

    /// Returns the `ArrowType` for this decimal array, determined by the
    /// backing integer width (i32, i64, or i128).
    pub fn arrow_type(&self) -> ArrowType {
        use std::any::TypeId;
        let tid = TypeId::of::<T>();
        if tid == TypeId::of::<i32>() {
            ArrowType::Decimal32(self.precision, self.scale)
        } else if tid == TypeId::of::<i64>() {
            ArrowType::Decimal64(self.precision, self.scale)
        } else if tid == TypeId::of::<i128>() {
            ArrowType::Decimal128(self.precision, self.scale)
        } else {
            panic!("DecimalArray::arrow_type: unsupported backing type")
        }
    }
}

// ---------------------------------------------------------------------------
// MaskedArray implementation
// ---------------------------------------------------------------------------
//
// DecimalArray implements MaskedArray manually because precision and scale
// metadata must propagate through slice_clone, split, and be validated in
// append_array. The impl_masked_array! macro supports only one extra field
// via its $extra_field parameter, which is insufficient for two metadata fields.

impl<T: Integer> MaskedArray for DecimalArray<T> {
    type T = T;
    type Container = Buffer<T>;
    type LogicalType = T;
    type CopyType<'a> = T where Self: 'a;

    fn data(&self) -> &Buffer<T> {
        &self.data
    }

    fn delete_range(&mut self, start: usize, end: usize) {
        self.data.delete_range(start, end);
        if let Some(mask) = &mut self.null_mask {
            mask.delete_range(start, end);
        }
    }

    #[inline]
    fn push(&mut self, value: T) {
        self.data_mut().push(value);
        let idx = self.len() - 1;
        if let Some(nm) = &mut self.null_mask {
            nm.set(idx, true);
        }
    }

    #[inline(always)]
    unsafe fn push_unchecked(&mut self, value: T) {
        let idx = self.len();
        unsafe {
            self.set_unchecked(idx, value);
            if let Some(mask) = self.null_mask_mut() {
                mask.set_unchecked(idx, true);
            }
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }

    fn null_mask(&self) -> Option<&Bitmask> {
        self.null_mask.as_ref()
    }

    /// Returns a copy of `[offset, offset+len)` with precision and scale
    /// propagated from the source array.
    fn slice_clone(&self, offset: usize, len: usize) -> Self {
        assert!(offset + len <= self.data.len(), "slice out of bounds");

        let data = Vec64::from_slice(&self.data[offset..offset + len]);
        let null_mask = self.null_mask.as_ref().map(|m| m.slice_clone(offset, len));

        Self {
            data: data.into(),
            null_mask,
            precision: self.precision,
            scale: self.scale,
        }
    }

    #[inline(always)]
    fn tuple_ref<'a>(&'a self, offset: Offset, len: Length) -> (&'a Self, Offset, Length) {
        (self, offset, len)
    }

    fn null_mask_mut(&mut self) -> Option<&mut Bitmask> {
        self.null_mask.as_mut()
    }

    fn set_null_mask(&mut self, mask: Option<Bitmask>) {
        self.null_mask = mask;
    }

    fn data_mut(&mut self) -> &mut Buffer<T> {
        &mut self.data
    }

    #[inline]
    fn iter(&self) -> impl Iterator<Item = T> + '_
    where
        T: Copy,
    {
        (0..self.len()).map(move |i| self.data()[i])
    }

    #[inline]
    fn iter_opt(&self) -> impl Iterator<Item = Option<T>> + '_
    where
        T: Copy,
    {
        (0..self.len()).map(move |i| {
            if self.is_null(i) {
                None
            } else {
                Some(self.data()[i])
            }
        })
    }

    #[inline]
    fn iter_range(&self, offset: usize, len: usize) -> impl Iterator<Item = T> + '_
    where
        T: Copy,
    {
        (offset..offset + len).map(move |i| self.data()[i])
    }

    #[inline]
    fn iter_opt_range(&self, offset: usize, len: usize) -> impl Iterator<Item = Option<T>> + '_
    where
        T: Copy,
    {
        (offset..offset + len).map(move |i| {
            if self.is_null(i) {
                None
            } else {
                Some(self.data()[i])
            }
        })
    }

    #[inline]
    fn get(&self, idx: usize) -> Option<T> {
        if idx >= self.len() {
            return None;
        }
        if self.is_null(idx) {
            None
        } else {
            self.data().get(idx).copied()
        }
    }

    #[inline(always)]
    unsafe fn get_unchecked(&self, idx: usize) -> Option<T> {
        if let Some(mask) = self.null_mask() {
            if !mask.get(idx) {
                return None;
            }
        }
        Some(unsafe { *self.data().get_unchecked(idx) })
    }

    #[inline]
    fn set(&mut self, idx: usize, value: T) {
        assert!(idx < self.len(), "index out of bounds");
        let data = self.data_mut().as_mut_slice();
        data[idx] = value;
        if let Some(mask) = self.null_mask_mut() {
            mask.set(idx, true);
        }
    }

    #[inline(always)]
    unsafe fn set_unchecked(&mut self, idx: usize, value: T) {
        let data = self.data_mut().as_mut_slice();
        data[idx] = value;
        if let Some(mask) = self.null_mask_mut() {
            unsafe { mask.set_unchecked(idx, true) };
        }
    }

    fn resize(&mut self, n: usize, value: T) {
        self.data.resize(n, value)
    }

    /// Appends all values and null mask from `other` to `self`.
    ///
    /// Panics if precision or scale differ between the two arrays, because
    /// concatenating values with different decimal semantics produces
    /// incorrect results.
    fn append_array(&mut self, other: &Self) {
        assert!(
            self.precision == other.precision && self.scale == other.scale,
            "DecimalArray::append_array: precision/scale mismatch (self: p={} s={}, other: p={} s={})",
            self.precision, self.scale, other.precision, other.scale
        );

        let orig_len = self.len();
        let other_len = other.len();

        if other_len == 0 {
            return;
        }

        self.data_mut().extend_from_slice(other.data());

        match (self.null_mask_mut(), other.null_mask()) {
            (Some(self_mask), Some(other_mask)) => {
                self_mask.extend_from_bitmask(other_mask);
            }
            (Some(self_mask), None) => {
                for i in orig_len..(orig_len + other_len) {
                    self_mask.set(i, true);
                }
            }
            (None, Some(other_mask)) => {
                let mut mask = Bitmask::new_set_all(orig_len + other_len, true);
                for i in 0..other_len {
                    mask.set(orig_len + i, other_mask.get(i));
                }
                self.set_null_mask(Some(mask));
            }
            (None, None) => {}
        }
    }

    fn append_range(
        &mut self,
        other: &Self,
        offset: usize,
        len: usize,
    ) -> Result<(), crate::enums::error::MinarrowError> {
        if len == 0 {
            return Ok(());
        }
        if offset + len > other.len() {
            return Err(crate::enums::error::MinarrowError::IndexError(format!(
                "append_range: offset {} + len {} exceeds source length {}",
                offset,
                len,
                other.len()
            )));
        }
        assert!(
            self.precision == other.precision && self.scale == other.scale,
            "DecimalArray::append_range: precision/scale mismatch (self: p={} s={}, other: p={} s={})",
            self.precision, self.scale, other.precision, other.scale
        );

        let orig_len = self.len();
        self.data_mut()
            .extend_from_slice(&other.data()[offset..offset + len]);

        match (self.null_mask_mut(), other.null_mask()) {
            (Some(self_mask), Some(other_mask)) => {
                self_mask.extend_from_bitmask_range(other_mask, offset, len);
            }
            (Some(self_mask), None) => {
                self_mask.resize(orig_len + len, true);
            }
            (None, Some(other_mask)) => {
                let mut mask = Bitmask::new_set_all(orig_len, true);
                mask.extend_from_bitmask_range(other_mask, offset, len);
                self.set_null_mask(Some(mask));
            }
            (None, None) => {}
        }
        Ok(())
    }

    fn insert_rows(
        &mut self,
        index: usize,
        other: &Self,
    ) -> Result<(), crate::enums::error::MinarrowError> {
        let orig_len = self.len();
        let other_len = other.len();

        if index > orig_len {
            return Err(crate::enums::error::MinarrowError::IndexError(format!(
                "Index {} out of bounds for array of length {}",
                index, orig_len
            )));
        }
        if other_len == 0 {
            return Ok(());
        }

        assert!(
            self.precision == other.precision && self.scale == other.scale,
            "DecimalArray::insert_rows: precision/scale mismatch (self: p={} s={}, other: p={} s={})",
            self.precision, self.scale, other.precision, other.scale
        );

        self.data.resize(orig_len + other_len, Default::default());
        for i in (index..orig_len).rev() {
            unsafe {
                let val = *self.data.as_ref().get_unchecked(i);
                *self.data.as_mut().get_unchecked_mut(i + other_len) = val;
            }
        }
        for i in 0..other_len {
            unsafe {
                let val = *other.data.as_ref().get_unchecked(i);
                *self.data.as_mut().get_unchecked_mut(index + i) = val;
            }
        }

        match (self.null_mask_mut(), other.null_mask()) {
            (Some(self_mask), Some(other_mask)) => {
                self_mask.resize(orig_len + other_len, true);
                for i in (index..orig_len).rev() {
                    unsafe {
                        let bit = self_mask.get_unchecked(i);
                        self_mask.set_unchecked(i + other_len, bit);
                    }
                }
                for i in 0..other_len {
                    unsafe {
                        let bit = other_mask.get_unchecked(i);
                        self_mask.set_unchecked(index + i, bit);
                    }
                }
            }
            (Some(self_mask), None) => {
                self_mask.resize(orig_len + other_len, true);
                for i in (index..orig_len).rev() {
                    unsafe {
                        let bit = self_mask.get_unchecked(i);
                        self_mask.set_unchecked(i + other_len, bit);
                    }
                }
                for i in index..(index + other_len) {
                    unsafe {
                        self_mask.set_unchecked(i, true);
                    }
                }
            }
            (None, Some(other_mask)) => {
                let mut mask = Bitmask::new_set_all(orig_len + other_len, true);
                for i in 0..other_len {
                    unsafe {
                        let bit = other_mask.get_unchecked(i);
                        mask.set_unchecked(index + i, bit);
                    }
                }
                self.set_null_mask(Some(mask));
            }
            (None, None) => {}
        }
        Ok(())
    }

    /// Splits this array at the specified index, consuming self and returning
    /// two arrays. Both halves inherit the source's precision and scale.
    fn split(
        mut self,
        index: usize,
    ) -> Result<(Self, Self), crate::enums::error::MinarrowError> {
        let len = self.len();
        if index == 0 || index >= len {
            return Err(crate::enums::error::MinarrowError::IndexError(format!(
                "Split index {} must be > 0 and < array length {}",
                index, len
            )));
        }

        let precision = self.precision;
        let scale = self.scale;

        let after_data = self.data.split_off(index);
        let after_mask = if let Some(ref mut mask) = self.null_mask {
            Some(mask.split_off(index))
        } else {
            None
        };

        let before = Self {
            data: self.data,
            null_mask: self.null_mask,
            precision,
            scale,
        };
        let after = Self {
            data: after_data,
            null_mask: after_mask,
            precision,
            scale,
        };

        Ok((before, after))
    }

    fn extend_from_iter_with_capacity<I>(&mut self, iter: I, additional_capacity: usize)
    where
        I: Iterator<Item = T>,
    {
        self.data.reserve(additional_capacity);
        let values: Vec<T> = iter.collect();
        let start_len = self.len();
        self.data.resize(start_len + values.len(), Default::default());
        for (i, value) in values.iter().enumerate() {
            unsafe { self.set_unchecked(start_len + i, *value) };
        }
    }

    fn extend_from_slice(&mut self, slice: &[T]) {
        let start_len = self.len();
        self.data.reserve(slice.len());
        self.data.resize(start_len + slice.len(), Default::default());
        for (i, value) in slice.iter().enumerate() {
            unsafe { self.set_unchecked(start_len + i, *value) };
        }
    }

    /// Creates a new array filled with the specified value repeated `count` times.
    ///
    /// The resulting array has precision=0 and scale=0 because the MaskedArray
    /// trait signature does not carry decimal metadata. Use the constructors
    /// for arrays with specific precision and scale.
    fn fill(value: T, count: usize) -> Self {
        let mut array = Self::default();
        array.data.reserve(count);
        array.data.resize(count, Default::default());
        for i in 0..count {
            unsafe { array.set_unchecked(i, value) };
        }
        array
    }
}

// ---------------------------------------------------------------------------
// AsRef / AsMut / Deref / DerefMut
// ---------------------------------------------------------------------------

impl_array_ref_deref!(DecimalArray<T>);
impl_arc_masked_array!(
    Inner = DecimalArray<T>,
    T = T,
    Container = Buffer<T>,
    LogicalType = T,
    CopyType = T,
    BufferT = T,
    Variant = NumericArray,
    Bound = Integer,
);

// ---------------------------------------------------------------------------
// Display - scale-aware formatting
// ---------------------------------------------------------------------------

/// Formats an unscaled integer value as a decimal string, applying the given scale.
///
/// Applies the decimal point at the position determined by scale. When scale is
/// zero, returns the integer without a decimal point. When scale is positive,
/// inserts the decimal point `scale` digits from the right, padding with leading
/// zeros if the value has fewer digits than scale. When scale is negative,
/// appends `|scale|` trailing zeros.
fn format_decimal_value<T: Integer + Display>(raw: T, scale: i8) -> String {
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

    // Pad with leading zeros when the digit count is <= scale
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

impl<T> Display for DecimalArray<T>
where
    T: Integer + Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let len = self.len();
        let nulls = self.null_count();

        writeln!(
            f,
            "DecimalArray [{} values] (precision: {}, scale: {}, nulls: {})",
            len, self.precision, self.scale, nulls
        )?;

        write!(f, "[")?;

        for i in 0..usize::min(len, MAX_PREVIEW) {
            if i > 0 {
                write!(f, ", ")?;
            }

            match self.get(i) {
                Some(val) => write!(f, "{}", format_decimal_value(val, self.scale))?,
                None => write!(f, "null")?,
            }
        }

        if len > MAX_PREVIEW {
            write!(f, ", ... ({} total)", len)?;
        }

        write!(f, "]")
    }
}

// ---------------------------------------------------------------------------
// Shape
// ---------------------------------------------------------------------------

impl<T: Integer> Shape for DecimalArray<T> {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank1(self.len())
    }
}

// ---------------------------------------------------------------------------
// Concatenate - validates matching scale
// ---------------------------------------------------------------------------

impl<T: Integer> Concatenate for DecimalArray<T> {
    fn concat(
        mut self,
        other: Self,
    ) -> core::result::Result<Self, crate::enums::error::MinarrowError> {
        if self.scale != other.scale {
            return Err(crate::enums::error::MinarrowError::IncompatibleTypeError {
                from: "DecimalArray",
                to: "DecimalArray",
                message: Some(format!(
                    "scale mismatch: {} vs {}",
                    self.scale, other.scale
                )),
            });
        }
        // Take the wider precision to accommodate both operands
        self.precision = self.precision.max(other.precision);
        self.append_array(&other);
        Ok(self)
    }
}

// ---------------------------------------------------------------------------
// Widening conversions (lossless)
// ---------------------------------------------------------------------------

impl From<DecimalArray<i32>> for DecimalArray<i64> {
    fn from(src: DecimalArray<i32>) -> Self {
        let data: Vec64<i64> = src.data.iter().map(|&v| v as i64).collect();
        DecimalArray {
            data: data.into(),
            null_mask: src.null_mask,
            precision: src.precision,
            scale: src.scale,
        }
    }
}

impl From<DecimalArray<i64>> for DecimalArray<i128> {
    fn from(src: DecimalArray<i64>) -> Self {
        let data: Vec64<i128> = src.data.iter().map(|&v| v as i128).collect();
        DecimalArray {
            data: data.into(),
            null_mask: src.null_mask,
            precision: src.precision,
            scale: src.scale,
        }
    }
}

// ---------------------------------------------------------------------------
// Parallel iterators (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "parallel_proc")]
impl<T> DecimalArray<T>
where
    T: Integer + Send + Sync + Copy + 'static,
{
    /// Parallel iterator over the backing buffer.
    ///
    /// Yields `&T` for every entry, regardless of null status. Consult the
    /// null-mask separately for null awareness.
    #[inline]
    pub fn par_iter(&self) -> rayon::slice::Iter<'_, T> {
        self.data.par_iter()
    }

    /// Parallel iterator with null awareness.
    ///
    /// Yields `None` for null entries and `Some(&T)` for valid values.
    #[inline]
    pub fn par_iter_opt(
        &self,
    ) -> impl rayon::prelude::ParallelIterator<Item = Option<&T>> + '_ {
        use rayon::prelude::*;
        let nmask = self.null_mask.as_ref();
        self.data.par_iter().enumerate().map(move |(idx, val)| {
            if nmask.map(|m| !m.get(idx)).unwrap_or(false) {
                None
            } else {
                Some(val)
            }
        })
    }

    /// Parallel mutable iterator over the backing buffer.
    ///
    /// Zero-copy iteration giving mutable access to underlying values.
    #[inline]
    pub fn par_iter_mut(&mut self) -> rayon::slice::IterMut<'_, T> {
        self.data.par_iter_mut()
    }

    /// Parallel iterator over a range of rows with null awareness.
    ///
    /// Iterates in parallel over the window `[start, end)`, yielding `None`
    /// for null entries and `Some(&T)` for valid values.
    #[inline]
    pub fn par_iter_range(
        &self,
        start: usize,
        end: usize,
    ) -> impl rayon::prelude::ParallelIterator<Item = Option<&T>> + '_
    where
        for<'r> &'r T: Send,
    {
        use rayon::prelude::*;
        let nmask = self.null_mask.as_ref();
        let data = &self.data;
        debug_assert!(start <= end && end <= data.len());
        (start..end).into_par_iter().map(move |i| {
            if nmask.map(|m| !m.get(i)).unwrap_or(false) {
                None
            } else {
                Some(&data[i])
            }
        })
    }

    /// Parallel iterator over a range of rows with null awareness.
    ///
    /// Iterates in parallel over the window `[start, end)`, yielding `None`
    /// for null entries and `Some(&T)` for valid values.
    pub fn par_iter_range_opt(
        &self,
        start: usize,
        end: usize,
    ) -> impl rayon::prelude::ParallelIterator<Item = Option<&T>> + '_ {
        use rayon::prelude::*;
        let nmask = self.null_mask.as_ref();
        let data = &self.data;
        (start..end).into_par_iter().map(move |i| {
            if nmask.map(|m| !m.get(i)).unwrap_or(false) {
                None
            } else {
                Some(&data[i])
            }
        })
    }

    /// Unchecked parallel iterator over a range of rows.
    ///
    /// Iterates in parallel over the window `[start, end)` without bounds checks,
    /// yielding `&T` for every entry regardless of null status.
    /// Caller must ensure `[start, end)` is within bounds.
    #[inline]
    pub unsafe fn par_iter_range_unchecked(
        &self,
        start: usize,
        end: usize,
    ) -> impl rayon::prelude::ParallelIterator<Item = &T> + '_ {
        use rayon::prelude::*;
        let data = &self.data;
        (start..end)
            .into_par_iter()
            .map(move |i| unsafe { data.get_unchecked(i) })
    }

    /// Unchecked parallel iterator over a range with null awareness.
    ///
    /// Iterates in parallel over the window `[start, end)` without bounds checks
    /// but honouring the null-mask, yielding `None` for null entries and
    /// `Some(&T)` for valid values. Caller must ensure `[start, end)` is within bounds.
    #[inline]
    pub unsafe fn par_iter_range_opt_unchecked(
        &self,
        start: usize,
        end: usize,
    ) -> impl rayon::prelude::ParallelIterator<Item = Option<&T>> + '_
    where
        for<'r> &'r T: Send,
    {
        use rayon::prelude::*;
        let nmask = self.null_mask.as_ref();
        let data = &self.data;
        (start..end).into_par_iter().map(move |i| unsafe {
            if nmask.map(|m| !m.get_unchecked(i)).unwrap_or(false) {
                None
            } else {
                Some(data.get_unchecked(i))
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structs::bitmask::Bitmask;
    use crate::traits::masked_array::MaskedArray;
    use crate::vec64;

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_default() {
        let arr = DecimalArray::<i32>::default();
        assert_eq!(arr.data.len(), 0);
        assert!(arr.null_mask.is_none());
        assert_eq!(arr.precision, 0);
        assert_eq!(arr.scale, 0);
    }

    #[test]
    fn test_with_capacity() {
        let mut arr = DecimalArray::<i64>::with_capacity(8, true, 10, 2);
        assert_eq!(arr.data.len(), 0);
        assert!(arr.data.capacity() >= 8);
        assert!(arr.null_mask.is_some());
        assert_eq!(arr.precision, 10);
        assert_eq!(arr.scale, 2);
        assert_eq!(arr.null_count(), 0);

        arr.push(100);
        arr.push(200);
        assert_eq!(arr.null_count(), 0);

        arr.push_null();
        assert_eq!(arr.null_count(), 1);
    }

    #[test]
    fn test_from_slice() {
        let arr = DecimalArray::<i32>::from_slice(&[12345, 67890], 10, 2);
        assert_eq!(arr.len(), 2);
        assert_eq!(arr.precision, 10);
        assert_eq!(arr.scale, 2);
        assert_eq!(arr.get(0), Some(12345));
        assert_eq!(arr.get(1), Some(67890));
    }

    #[test]
    fn test_from_vec() {
        let arr = DecimalArray::<i64>::from_vec(vec![100, 200, 300], 5, 1);
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.precision, 5);
        assert_eq!(arr.scale, 1);
    }

    #[test]
    fn test_new_with_null_mask() {
        let data = vec64![10i32, 20, 30];
        let mut mask = Bitmask::new_set_all(3, true);
        mask.set(1, false);
        let arr = DecimalArray::new(data, Some(mask), 5, 2);
        assert_eq!(arr.get(0), Some(10));
        assert_eq!(arr.get(1), None);
        assert_eq!(arr.get(2), Some(30));
    }

    // -----------------------------------------------------------------------
    // Push, get, set
    // -----------------------------------------------------------------------

    #[test]
    fn test_push_and_get_no_null_mask() {
        let mut arr = DecimalArray::<i64>::with_capacity(4, false, 10, 2);
        arr.push(12345);
        arr.push(-67890);
        assert_eq!(arr.get(0), Some(12345));
        assert_eq!(arr.get(1), Some(-67890));
        assert!(!arr.is_null(0));
        assert!(!arr.is_null(1));
    }

    #[test]
    fn test_push_and_get_with_null_mask() {
        let mut arr = DecimalArray::<i32>::with_capacity(3, true, 5, 1);
        arr.push(42);
        arr.push_null();
        arr.push(7);
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.get(0), Some(42));
        assert_eq!(arr.get(1), None);
        assert_eq!(arr.get(2), Some(7));
        assert!(!arr.is_null(0));
        assert!(arr.is_null(1));
        assert!(!arr.is_null(2));
    }

    #[test]
    fn test_push_null_auto_mask() {
        let mut arr = DecimalArray::<i32>::from_slice(&[], 10, 2);
        arr.push_null();
        assert_eq!(arr.data, vec64![0]);
        assert!(arr.is_null(0));
        assert!(arr.null_mask.is_some());
    }

    #[test]
    fn test_set_and_set_null() {
        let mut arr = DecimalArray::<i32>::with_capacity(3, true, 10, 2);
        arr.push(100);
        arr.push(200);
        arr.push(300);
        arr.set(1, 222);
        assert_eq!(arr.get(1), Some(222));
        arr.set_null(2);
        assert_eq!(arr.get(2), None);
        assert!(arr.is_null(2));
    }

    #[test]
    fn test_out_of_bounds() {
        let arr = DecimalArray::<i64>::from_slice(&[], 10, 2);
        assert_eq!(arr.get(0), None);
        assert_eq!(arr.get(100), None);
    }

    // -----------------------------------------------------------------------
    // Bulk operations
    // -----------------------------------------------------------------------

    #[test]
    fn test_bulk_push_nulls() {
        let mut arr = DecimalArray::<i32>::with_capacity(8, true, 10, 2);
        arr.push(19);
        arr.push_nulls(3);
        assert_eq!(arr.len(), 4);
        assert_eq!(arr.get(0), Some(19));
        assert_eq!(arr.get(1), None);
        assert_eq!(arr.get(3), None);
        assert!(arr.is_null(2));
    }

    #[test]
    fn test_extend_from_slice() {
        let mut arr = DecimalArray::<i32>::with_capacity(10, false, 10, 2);
        arr.push(100);
        arr.extend_from_slice(&[200, 300, 400]);
        assert_eq!(arr.len(), 4);
        assert_eq!(arr.get(0), Some(100));
        assert_eq!(arr.get(3), Some(400));
    }

    #[test]
    fn test_extend_from_iter_with_capacity() {
        let mut arr = DecimalArray::<i64>::with_capacity(3, false, 10, 2);
        arr.extend_from_iter_with_capacity(vec![10i64, 20, 30].into_iter(), 3);
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.get(0), Some(10));
        assert_eq!(arr.get(2), Some(30));
    }

    #[test]
    fn test_fill() {
        let arr = DecimalArray::<i32>::fill(42, 100);
        assert_eq!(arr.len(), 100);
        for i in 0..100 {
            assert_eq!(arr.get(i), Some(42));
        }
    }

    // -----------------------------------------------------------------------
    // Slice clone - precision/scale propagation
    // -----------------------------------------------------------------------

    #[test]
    fn test_slice_clone_propagates_precision_scale() {
        let mut arr = DecimalArray::<i32>::from_slice(&[10, 20, 30, 40, 50], 10, 3);
        arr.null_mask = Some(Bitmask::new_set_all(5, true));
        arr.set_null(3);

        let sliced = arr.slice_clone(1, 3);
        assert_eq!(sliced.len(), 3);
        assert_eq!(sliced.precision, 10);
        assert_eq!(sliced.scale, 3);
        assert_eq!(sliced.get(0), Some(20));
        assert_eq!(sliced.get(1), Some(30));
        assert_eq!(sliced.get(2), None);
        assert_eq!(sliced.null_count(), 1);
    }

    // -----------------------------------------------------------------------
    // Append array - precision/scale validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_append_array_matching_metadata() {
        let mut arr1 = DecimalArray::<i32>::from_slice(&[10, 20, 30], 10, 2);
        let arr2 = DecimalArray::<i32>::from_slice(&[40, 50], 10, 2);
        arr1.append_array(&arr2);
        assert_eq!(arr1.len(), 5);
        assert_eq!(arr1.get(0), Some(10));
        assert_eq!(arr1.get(4), Some(50));
    }

    #[test]
    #[should_panic(expected = "precision/scale mismatch")]
    fn test_append_array_precision_mismatch_panics() {
        let mut arr1 = DecimalArray::<i32>::from_slice(&[10], 10, 2);
        let arr2 = DecimalArray::<i32>::from_slice(&[20], 5, 2);
        arr1.append_array(&arr2);
    }

    #[test]
    #[should_panic(expected = "precision/scale mismatch")]
    fn test_append_array_scale_mismatch_panics() {
        let mut arr1 = DecimalArray::<i32>::from_slice(&[10], 10, 2);
        let arr2 = DecimalArray::<i32>::from_slice(&[20], 10, 3);
        arr1.append_array(&arr2);
    }

    #[test]
    fn test_append_array_with_nulls() {
        let mut arr1 = DecimalArray::<i32>::with_capacity(3, true, 10, 2);
        arr1.push(60);
        arr1.push_null();
        arr1.push(70);

        let mut arr2 = DecimalArray::<i32>::with_capacity(2, true, 10, 2);
        arr2.push_null();
        arr2.push(80);

        arr1.append_array(&arr2);
        assert_eq!(arr1.len(), 5);
        let vals: Vec<Option<i32>> = (0..arr1.len()).map(|i| arr1.get(i)).collect();
        assert_eq!(vals, vec![Some(60), None, Some(70), None, Some(80)]);
        assert_eq!(arr1.null_count(), 2);
    }

    // -----------------------------------------------------------------------
    // Split - precision/scale propagation
    // -----------------------------------------------------------------------

    #[test]
    fn test_split_propagates_precision_scale() {
        let arr = DecimalArray::<i64>::from_slice(&[100, 200, 300, 400], 12, 4);
        let (left, right) = arr.split(2).unwrap();
        assert_eq!(left.len(), 2);
        assert_eq!(right.len(), 2);
        assert_eq!(left.precision, 12);
        assert_eq!(left.scale, 4);
        assert_eq!(right.precision, 12);
        assert_eq!(right.scale, 4);
        assert_eq!(left.get(0), Some(100));
        assert_eq!(right.get(0), Some(300));
    }

    // -----------------------------------------------------------------------
    // Delete range
    // -----------------------------------------------------------------------

    #[test]
    fn test_delete_range() {
        let mut arr = DecimalArray::<i32>::from_slice(&[10, 20, 30, 40, 50], 10, 2);
        arr.delete_range(1, 3);
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.get(0), Some(10));
        assert_eq!(arr.get(1), Some(40));
        assert_eq!(arr.get(2), Some(50));
    }

    // -----------------------------------------------------------------------
    // Insert rows
    // -----------------------------------------------------------------------

    #[test]
    fn test_insert_rows() {
        let mut arr = DecimalArray::<i32>::from_slice(&[10, 40, 50], 10, 2);
        let insert = DecimalArray::<i32>::from_slice(&[20, 30], 10, 2);
        arr.insert_rows(1, &insert).unwrap();
        assert_eq!(arr.len(), 5);
        let vals: Vec<Option<i32>> = (0..5).map(|i| arr.get(i)).collect();
        assert_eq!(
            vals,
            vec![Some(10), Some(20), Some(30), Some(40), Some(50)]
        );
    }

    // -----------------------------------------------------------------------
    // Display - scale-aware formatting
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_decimal_value_positive_scale() {
        assert_eq!(format_decimal_value(12345i64, 2), "123.45");
        assert_eq!(format_decimal_value(-12345i64, 2), "-123.45");
        assert_eq!(format_decimal_value(5i32, 2), "0.05");
        assert_eq!(format_decimal_value(50i32, 2), "0.50");
        assert_eq!(format_decimal_value(0i32, 3), "0.000");
        assert_eq!(format_decimal_value(-1i64, 1), "-0.1");
    }

    #[test]
    fn test_format_decimal_value_zero_scale() {
        assert_eq!(format_decimal_value(12345i64, 0), "12345");
        assert_eq!(format_decimal_value(-99i32, 0), "-99");
        assert_eq!(format_decimal_value(0i32, 0), "0");
    }

    #[test]
    fn test_format_decimal_value_negative_scale() {
        assert_eq!(format_decimal_value(12345i64, -2), "1234500");
        assert_eq!(format_decimal_value(-5i32, -3), "-5000");
        assert_eq!(format_decimal_value(0i32, -1), "00");
    }

    #[test]
    fn test_display_positive_scale() {
        let arr = DecimalArray::<i64>::from_slice(&[12345, -67890, 5], 10, 2);
        let display = format!("{}", arr);
        assert!(display.contains("123.45"));
        assert!(display.contains("-678.90"));
        assert!(display.contains("0.05"));
        assert!(display.contains("precision: 10"));
        assert!(display.contains("scale: 2"));
    }

    #[test]
    fn test_display_zero_scale() {
        let arr = DecimalArray::<i32>::from_slice(&[100, 200], 5, 0);
        let display = format!("{}", arr);
        assert!(display.contains("100"));
        assert!(display.contains("200"));
    }

    #[test]
    fn test_display_negative_scale() {
        let arr = DecimalArray::<i32>::from_slice(&[12345], 10, -2);
        let display = format!("{}", arr);
        assert!(display.contains("1234500"));
    }

    #[test]
    fn test_display_with_nulls() {
        let mut arr = DecimalArray::<i32>::with_capacity(3, true, 10, 2);
        arr.push(12345);
        arr.push_null();
        arr.push(67890);
        let display = format!("{}", arr);
        assert!(display.contains("null"));
        assert!(display.contains("nulls: 1"));
    }

    // -----------------------------------------------------------------------
    // Shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_shape() {
        let arr = DecimalArray::<i32>::from_slice(&[1, 2, 3], 5, 0);
        assert_eq!(arr.shape(), ShapeDim::Rank1(3));
    }

    // -----------------------------------------------------------------------
    // Concatenate
    // -----------------------------------------------------------------------

    #[test]
    fn test_concat_matching_scale() {
        let arr1 = DecimalArray::<i32>::from_slice(&[100, 200], 10, 2);
        let arr2 = DecimalArray::<i32>::from_slice(&[300, 400], 10, 2);
        let result = arr1.concat(arr2).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result.get(0), Some(100));
        assert_eq!(result.get(3), Some(400));
        assert_eq!(result.precision, 10);
        assert_eq!(result.scale, 2);
    }

    #[test]
    fn test_concat_takes_max_precision() {
        let arr1 = DecimalArray::<i64>::from_slice(&[100], 8, 2);
        let arr2 = DecimalArray::<i64>::from_slice(&[200], 12, 2);
        let result = arr1.concat(arr2).unwrap();
        assert_eq!(result.precision, 12);
    }

    #[test]
    fn test_concat_mismatched_scale_errors() {
        let arr1 = DecimalArray::<i32>::from_slice(&[100], 10, 2);
        let arr2 = DecimalArray::<i32>::from_slice(&[200], 10, 3);
        let result = arr1.concat(arr2);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{}", err).contains("scale mismatch"),
            "Expected scale mismatch error, got: {}",
            err
        );
    }

    #[test]
    fn test_concat_with_nulls() {
        let mut arr1 = DecimalArray::<i32>::with_capacity(2, true, 10, 2);
        arr1.push(10);
        arr1.push_null();

        let mut arr2 = DecimalArray::<i32>::with_capacity(2, true, 10, 2);
        arr2.push_null();
        arr2.push(40);

        let result = arr1.concat(arr2).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result.get(0), Some(10));
        assert_eq!(result.get(1), None);
        assert_eq!(result.get(2), None);
        assert_eq!(result.get(3), Some(40));
        assert_eq!(result.null_count(), 2);
    }

    // -----------------------------------------------------------------------
    // Is empty and len
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_empty_and_len() {
        let mut arr = DecimalArray::<i32>::from_slice(&[], 10, 2);
        assert!(arr.is_empty());
        arr.push(1);
        assert!(!arr.is_empty());
        assert_eq!(arr.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Null mask replace
    // -----------------------------------------------------------------------

    #[test]
    fn test_null_mask_replace() {
        let mut arr = DecimalArray::<i32>::from_slice(&[9], 10, 2);
        let mut mask = Bitmask::new_set_all(1, false);
        mask.set(0, true);
        arr.set_null_mask(Some(mask));
        assert!(!arr.is_null(0));
    }

    // -----------------------------------------------------------------------
    // i128 type
    // -----------------------------------------------------------------------

    #[test]
    fn test_decimal128_basic() {
        let mut arr = DecimalArray::<i128>::with_capacity(4, true, 38, 10);
        arr.push(123456789012345);
        arr.push(-987654321098765);
        arr.push_null();
        arr.push(0);
        assert_eq!(arr.len(), 4);
        assert_eq!(arr.get(0), Some(123456789012345));
        assert_eq!(arr.get(1), Some(-987654321098765));
        assert_eq!(arr.get(2), None);
        assert_eq!(arr.get(3), Some(0));
        assert_eq!(arr.precision, 38);
        assert_eq!(arr.scale, 10);
    }

    #[test]
    fn test_decimal128_from_slice() {
        let arr = DecimalArray::<i128>::from_slice(&[100i128, 200, 300], 38, 18);
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.precision, 38);
        assert_eq!(arr.scale, 18);
    }

    #[test]
    fn test_decimal128_display() {
        let arr = DecimalArray::<i128>::from_slice(&[12345i128], 38, 2);
        let display = format!("{}", arr);
        assert!(display.contains("123.45"));
    }

    #[test]
    fn test_decimal128_concat() {
        let arr1 = DecimalArray::<i128>::from_slice(&[100], 38, 10);
        let arr2 = DecimalArray::<i128>::from_slice(&[200], 38, 10);
        let result = arr1.concat(arr2).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result.get(0), Some(100));
        assert_eq!(result.get(1), Some(200));
    }

    #[test]
    fn test_decimal128_slice_clone() {
        let arr = DecimalArray::<i128>::from_slice(&[10, 20, 30, 40], 38, 5);
        let sliced = arr.slice_clone(1, 2);
        assert_eq!(sliced.len(), 2);
        assert_eq!(sliced.precision, 38);
        assert_eq!(sliced.scale, 5);
        assert_eq!(sliced.get(0), Some(20));
        assert_eq!(sliced.get(1), Some(30));
    }

    // -----------------------------------------------------------------------
    // i128 trait impls
    // -----------------------------------------------------------------------

    #[test]
    fn test_i128_integer_trait() {
        use crate::traits::type_unions::Integer;
        let val: i128 = 42;
        assert_eq!(val.to_usize(), 42usize);
        assert_eq!(i128::from_usize(99), 99i128);
    }

    #[test]
    fn test_i128_numeric_trait() {
        use crate::traits::type_unions::Numeric;
        fn is_numeric<T: Numeric>(_: T) {}
        is_numeric(42i128);
    }

    #[test]
    fn test_i128_primitive_trait() {
        use crate::traits::type_unions::Primitive;
        fn is_primitive<T: Primitive>(_: T) {}
        is_primitive(42i128);
    }

    // -----------------------------------------------------------------------
    // DecimalArray<i32> and DecimalArray<i64> also compile
    // -----------------------------------------------------------------------

    #[test]
    fn test_decimal32_basic() {
        let arr = DecimalArray::<i32>::from_slice(&[12345, -67890], 9, 4);
        assert_eq!(arr.len(), 2);
        assert_eq!(arr.get(0), Some(12345));
        assert_eq!(arr.get(1), Some(-67890));
        assert_eq!(arr.precision, 9);
        assert_eq!(arr.scale, 4);
    }

    #[test]
    fn test_decimal64_basic() {
        let arr = DecimalArray::<i64>::from_slice(&[1234567890, -9876543210], 18, 6);
        assert_eq!(arr.len(), 2);
        assert_eq!(arr.get(0), Some(1234567890));
        assert_eq!(arr.get(1), Some(-9876543210));
        assert_eq!(arr.precision, 18);
        assert_eq!(arr.scale, 6);
    }

    // -----------------------------------------------------------------------
    // Extend from slice with nulls
    // -----------------------------------------------------------------------

    #[test]
    fn test_extend_from_slice_with_nulls() {
        let mut arr = DecimalArray::<i32>::with_capacity(10, true, 10, 2);
        arr.push(100);
        arr.push_null();

        arr.extend_from_slice(&[200, 300, 400]);

        assert_eq!(arr.len(), 5);
        assert_eq!(arr.get(0), Some(100));
        assert_eq!(arr.get(1), None);
        assert_eq!(arr.get(2), Some(200));
        assert_eq!(arr.get(3), Some(300));
        assert_eq!(arr.get(4), Some(400));
    }

    // -----------------------------------------------------------------------
    // Append range
    // -----------------------------------------------------------------------

    #[test]
    fn test_append_range() {
        let mut arr1 = DecimalArray::<i32>::from_slice(&[10, 20], 10, 2);
        let arr2 = DecimalArray::<i32>::from_slice(&[30, 40, 50, 60], 10, 2);
        arr1.append_range(&arr2, 1, 2).unwrap();
        assert_eq!(arr1.len(), 4);
        assert_eq!(arr1.get(2), Some(40));
        assert_eq!(arr1.get(3), Some(50));
    }

    #[test]
    fn test_append_range_out_of_bounds() {
        let mut arr1 = DecimalArray::<i32>::from_slice(&[10], 10, 2);
        let arr2 = DecimalArray::<i32>::from_slice(&[20, 30], 10, 2);
        let result = arr1.append_range(&arr2, 1, 5);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // arrow_type
    // -----------------------------------------------------------------------

    #[test]
    fn test_arrow_type_decimal32() {
        use crate::ffi::arrow_dtype::ArrowType;
        let arr = DecimalArray::<i32>::from_slice(&[1], 9, 2);
        assert_eq!(arr.arrow_type(), ArrowType::Decimal32(9, 2));
    }

    #[test]
    fn test_arrow_type_decimal64() {
        use crate::ffi::arrow_dtype::ArrowType;
        let arr = DecimalArray::<i64>::from_slice(&[1], 18, 4);
        assert_eq!(arr.arrow_type(), ArrowType::Decimal64(18, 4));
    }

    #[test]
    fn test_arrow_type_decimal128() {
        use crate::ffi::arrow_dtype::ArrowType;
        let arr = DecimalArray::<i128>::from_slice(&[1], 38, 10);
        assert_eq!(arr.arrow_type(), ArrowType::Decimal128(38, 10));
    }

    // -----------------------------------------------------------------------
    // Widening From impls
    // -----------------------------------------------------------------------

    #[test]
    fn test_widen_decimal32_to_decimal64() {
        let arr32 = DecimalArray::<i32>::from_slice(&[12345, -67890], 9, 2);
        let arr64 = DecimalArray::<i64>::from(arr32);
        assert_eq!(arr64.len(), 2);
        assert_eq!(arr64.get(0), Some(12345i64));
        assert_eq!(arr64.get(1), Some(-67890i64));
        assert_eq!(arr64.precision, 9);
        assert_eq!(arr64.scale, 2);
    }

    #[test]
    fn test_widen_decimal64_to_decimal128() {
        let arr64 = DecimalArray::<i64>::from_slice(&[999999999999, -1], 18, 6);
        let arr128 = DecimalArray::<i128>::from(arr64);
        assert_eq!(arr128.len(), 2);
        assert_eq!(arr128.get(0), Some(999999999999i128));
        assert_eq!(arr128.get(1), Some(-1i128));
        assert_eq!(arr128.precision, 18);
        assert_eq!(arr128.scale, 6);
    }

    #[test]
    fn test_widen_preserves_null_mask() {
        let mut arr32 = DecimalArray::<i32>::with_capacity(3, true, 9, 2);
        arr32.push(100);
        arr32.push_null();
        arr32.push(300);
        let arr64 = DecimalArray::<i64>::from(arr32);
        assert_eq!(arr64.len(), 3);
        assert_eq!(arr64.get(0), Some(100i64));
        assert_eq!(arr64.get(1), None);
        assert_eq!(arr64.get(2), Some(300i64));
        assert_eq!(arr64.null_count(), 1);
    }

    // -----------------------------------------------------------------------
    // Arc<DecimalArray> MaskedArray delegation
    // -----------------------------------------------------------------------

    #[test]
    fn test_arc_masked_array_delegation() {
        let mut arc_arr = std::sync::Arc::new(
            DecimalArray::<i32>::from_slice(&[10, 20, 30], 9, 2),
        );
        assert_eq!(arc_arr.len(), 3);
        assert_eq!(arc_arr.get(0), Some(10));

        // Copy-on-write push
        arc_arr.push(40);
        assert_eq!(arc_arr.len(), 4);
        assert_eq!(arc_arr.get(3), Some(40));
    }
}
