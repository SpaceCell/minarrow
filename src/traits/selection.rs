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

//! # **Selection Traits** - *Selection across dimensions*
//!
//! Traits for field and data selection that enable polymorphic methods
//! across Table, TableV, Cube, and future types.
//!
//! ## Architecture
//! - **FieldSelector**: Input types that can specify field selection (e.g., `&str`, `usize`)
//! - **DataSelector**: Input types that can specify data selection (e.g., `usize`, ranges)
//! - **FieldSelection**: Capability trait for types that support field selection
//! - **DataSelection**: Capability trait for types that support data selection
//! - **Selection2D**: Combined 2D selection (FieldSelection + DataSelection)
//! - **Selection3D**: Future Extension for 3D selection
//! - **Selection4D**: Future Extension for 4D selection

use crate::Field;
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo};
use std::sync::Arc;

// Input types that can be passed to selection methods
// ===================================================
// These traits are implemented on user-facing input types like `&str`, `usize`, and ranges.
// They convert user input (e.g., `table.f("name")` or `table.d(0..10)`) into index vectors.
// These are "what the user writes" when selecting.

/// Trait for types that can specify a field selection (named/schema dimension)
pub trait FieldSelector {
    /// Resolve this selection to field indices for the given fields
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize>;

    /// Produce an owned version of this selector.
    fn to_owned(&self) -> Box<dyn FieldSelector + Send + Sync>;
}

/// Trait for types that can specify a data selection (index-based dimension)
pub trait DataSelector {
    /// Resolve this selection to indices (within the given count)
    fn resolve_indices(&self, count: usize) -> Vec<usize>;

    /// Returns true if this selector represents a contiguous range.
    /// Range types (Range, RangeFrom, etc.) return true.
    /// Index arrays (&[usize], Vec<usize>) return false.
    fn is_contiguous(&self) -> bool {
        false // Default: assume non-contiguous
    }

    /// Resolve this selector against one axis of `dim_size`, producing a
    /// `(start, end, collapse)` span. Ranges window the axis and a single
    /// index collapses it. Panics when the selection falls outside the
    /// axis. Index arrays have no strided-view representation, so they
    /// are rejected - gather through [`RowSelection::r`] instead.
    #[cfg(feature = "ndarray")]
    fn resolve_axis(&self, dim_size: usize) -> (usize, usize, bool) {
        // usize::MAX so out-of-range indices survive to the bounds check
        // below rather than being silently filtered.
        let indices = self.resolve_indices(usize::MAX);
        assert_eq!(
            indices.len(), 1,
            "axis selection: index arrays take a single index or a contiguous range; gather with r() instead"
        );
        assert!(
            indices[0] < dim_size,
            "axis selection: index {} out of bounds (size {})", indices[0], dim_size
        );
        (indices[0], indices[0] + 1, true)
    }
}

// These traits are implemented on structures like Table, ArrayV, etc.
// They define what selection methods are available on each structure.
// These are "what the structure can do" for selection operations.

/// Trait for types that support field/column selection
///
/// Associated types determine what each access pattern returns:
/// - `View`: multi-field selection result, e.g. TableV for Table, Cube for Cube
/// - `DataView`: single column by index, e.g. ArrayV
/// - `Field`: single field by name via `.get()`, e.g. FieldArray for Table, Arc<Table> for Cube
pub trait ColumnSelection {
    /// The view type returned by multi-field selection e.g. TableV
    type View;
    /// A single column view by index e.g. ArrayV
    type ColumnView;
    /// An owned single field by name via `.get()` e.g. FieldArray for Table, Arc<Table> for Cube
    type ColumnOwned;

    /// Select fields/columns by name, index, or range
    ///
    /// Shorthand alias for `col`
    ///
    /// # Examples
    /// ```ignore
    /// table.c("age")           // single column by name
    /// table.c(&["a", "b"])     // multiple columns by name
    /// table.c(0)               // single column by index
    /// table.c(0..3)            // columns by range
    /// ```
    fn c<S: FieldSelector>(&self, selection: S) -> Self::View;

    /// Select a single column by name
    ///
    /// Named form of `c` - use `c` for selection by index, range,
    /// or multiple names.
    ///
    /// # Examples
    /// ```ignore
    /// table.col("age")         // single column by name
    /// table.c(&["a", "b"])     // multiple columns by name
    /// table.c(0)               // single column by index
    /// table.c(0..3)            // columns by range
    /// ```
    fn col(&self, name: &str) -> Self::View {
        self.c(name)
    }

    /// Get a single field by name, returning an owned value.
    fn get(&self, field: &str) -> Option<Self::ColumnOwned>;

    /// Get a single column view by index
    fn col_ix(&self, idx: usize) -> Option<Self::ColumnView>;

    /// Get all columns as views
    fn col_vec(&self) -> Vec<Self::ColumnView>;

    /// Get the fields for field resolution
    fn get_cols(&self) -> Vec<Arc<Field>>;
}

/// Trait for types that support row/data selection
pub trait RowSelection {
    /// The view type returned by selection operations
    type View;

    /// Select rows by index or range
    ///
    /// Shorthand alias for `row`
    ///
    /// # Examples
    /// ```ignore
    /// table.r(5)               // single row
    /// table.r(&[1, 3, 5])      // specific rows
    /// table.r(0..10)           // row range
    /// ```
    fn r<S: DataSelector>(&self, selection: S) -> Self::View;

    /// Select a single row by index
    ///
    /// Named form of `r` - use `r` for selection by range or
    /// index array.
    ///
    /// # Examples
    /// ```ignore
    /// table.row(5)             // single row
    /// table.r(&[1, 3, 5])      // specific rows
    /// table.r(0..10)           // row range
    /// ```
    fn row(&self, idx: usize) -> Self::View {
        self.r(idx)
    }

    /// Get the count for data resolution
    fn get_row_count(&self) -> usize;
}

/// Trait for types that support selection across any axis combination.
///
/// The N-dimensional member of the selection family, alongside
/// [`ColumnSelection`] and [`RowSelection`]. Each axis takes any
/// [`DataSelector`] - a single index collapses the dimension, and a
/// contiguous range keeps it.
///
/// # Examples
/// ```ignore
/// arr.s(&[&(1..4), &2])        // rows 1..4 of column 2 (2D)
/// arr.s(nd![1..4, 2])          // the same through the nd! macro
/// arr.s(nd![0..2, 1, 0..3])    // mixed selection (3D)
/// ```
#[cfg(feature = "ndarray")]
pub trait AxisSelection {
    /// The view type returned by axis selection e.g. NdArrayV
    type View;

    /// Select along every axis at once, one [`DataSelector`] per axis
    ///
    /// Shorthand alias for `select`
    ///
    /// A single index collapses its dimension, and a contiguous range
    /// keeps it.
    ///
    /// # Examples
    /// ```ignore
    /// arr.s(nd![1..4, 2])          // rows 1..4 of column 2 (2D)
    /// arr.s(nd![0..2, 1, 0..3])    // mixed selection (3D)
    /// arr.s(nd![.., 5])            // full range on axis 0
    /// arr.s(&[&(1..4), &2])        // without the nd! macro
    /// ```
    fn s(&self, selection: &[&dyn DataSelector]) -> Self::View;

    /// Select along every axis at once, one [`DataSelector`] per axis
    ///
    /// A single index collapses its dimension, and a contiguous range
    /// keeps it.
    ///
    /// # Examples
    /// ```ignore
    /// arr.select(nd![1..4, 2])          // rows 1..4 of column 2 (2D)
    /// arr.select(nd![0..2, 1, 0..3])    // mixed selection (3D)
    /// arr.select(nd![.., 5])            // full range on axis 0
    /// ```
    fn select(&self, selection: &[&dyn DataSelector]) -> Self::View {
        self.s(selection)
    }

    /// Get the axis count for selection resolution
    fn get_axis_count(&self) -> usize;
}

/// Combined trait for 2D selection (field + data dimensions)
///
/// This trait is automatically implemented for any type that implements
/// both `ColumnSelection` and `RowSelection` with the same `View` type.
pub trait Selection2D: ColumnSelection + RowSelection {}

/// Blanket implementation for any type that implements both traits
impl<T> Selection2D for T where
    T: ColumnSelection + RowSelection<View = <T as ColumnSelection>::View>
{
}

// These allow users to pass names, indices, and ranges when selecting fields.
// For example: table.c("age"), table.c(&["name", "age"]), table.c(0..3)

/// Single field by name
impl FieldSelector for &str {
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize> {
        fields
            .iter()
            .position(|f| f.name == *self)
            .into_iter()
            .collect()
    }
    fn to_owned(&self) -> Box<dyn FieldSelector + Send + Sync> {
        Box::new(vec![self.to_string()])
    }
}

/// Multiple fields by names
impl FieldSelector for &[&str] {
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize> {
        self.iter()
            .filter_map(|name| fields.iter().position(|f| f.name == *name))
            .collect()
    }
    fn to_owned(&self) -> Box<dyn FieldSelector + Send + Sync> {
        Box::new(self.iter().map(|s| s.to_string()).collect::<Vec<String>>())
    }
}

/// Multiple fields by names (array reference)
impl<const N: usize> FieldSelector for &[&str; N] {
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize> {
        self.iter()
            .filter_map(|name| fields.iter().position(|f| f.name == *name))
            .collect()
    }
    fn to_owned(&self) -> Box<dyn FieldSelector + Send + Sync> {
        Box::new(self.iter().map(|s| s.to_string()).collect::<Vec<String>>())
    }
}

/// Multiple fields by names (Vec of borrowed str)
impl FieldSelector for Vec<&str> {
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize> {
        self.iter()
            .filter_map(|name| fields.iter().position(|f| f.name == *name))
            .collect()
    }
    fn to_owned(&self) -> Box<dyn FieldSelector + Send + Sync> {
        Box::new(self.iter().map(|s| s.to_string()).collect::<Vec<String>>())
    }
}

/// Multiple fields by names (Vec of owned String)
impl FieldSelector for Vec<String> {
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize> {
        self.iter()
            .filter_map(|name| fields.iter().position(|f| f.name.as_str() == name.as_str()))
            .collect()
    }
    fn to_owned(&self) -> Box<dyn FieldSelector + Send + Sync> {
        Box::new(self.clone())
    }
}

/// Single field by index
impl FieldSelector for usize {
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize> {
        if *self < fields.len() {
            vec![*self]
        } else {
            Vec::new()
        }
    }
    fn to_owned(&self) -> Box<dyn FieldSelector + Send + Sync> {
        Box::new(vec![*self])
    }
}

/// Multiple fields by indices
impl FieldSelector for &[usize] {
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize> {
        self.iter()
            .copied()
            .filter(|&idx| idx < fields.len())
            .collect()
    }
    fn to_owned(&self) -> Box<dyn FieldSelector + Send + Sync> {
        Box::new(self.to_vec())
    }
}

/// Multiple fields by indices (array reference)
impl<const N: usize> FieldSelector for &[usize; N] {
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize> {
        self.iter()
            .copied()
            .filter(|&idx| idx < fields.len())
            .collect()
    }
    fn to_owned(&self) -> Box<dyn FieldSelector + Send + Sync> {
        Box::new(self.to_vec())
    }
}

/// Multiple fields by indices (Vec)
impl FieldSelector for Vec<usize> {
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize> {
        self.iter()
            .copied()
            .filter(|&idx| idx < fields.len())
            .collect()
    }
    fn to_owned(&self) -> Box<dyn FieldSelector + Send + Sync> {
        Box::new(self.clone())
    }
}

/// Field range selection
impl FieldSelector for Range<usize> {
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize> {
        let end = self.end.min(fields.len());
        (self.start..end).collect()
    }
    fn to_owned(&self) -> Box<dyn FieldSelector + Send + Sync> {
        Box::new(self.clone())
    }
}

/// Field range from selection
impl FieldSelector for RangeFrom<usize> {
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize> {
        (self.start..fields.len()).collect()
    }
    fn to_owned(&self) -> Box<dyn FieldSelector + Send + Sync> {
        Box::new(self.clone())
    }
}

/// Field range to selection
impl FieldSelector for RangeTo<usize> {
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize> {
        let end = self.end.min(fields.len());
        (0..end).collect()
    }
    fn to_owned(&self) -> Box<dyn FieldSelector + Send + Sync> {
        Box::new(self.clone())
    }
}

/// Field full range selection
impl FieldSelector for RangeFull {
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize> {
        (0..fields.len()).collect()
    }
    fn to_owned(&self) -> Box<dyn FieldSelector + Send + Sync> {
        Box::new(..)
    }
}

/// Field inclusive range selection
impl FieldSelector for RangeInclusive<usize> {
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize> {
        let start = *self.start();
        let end = (*self.end() + 1).min(fields.len());
        (start..end).collect()
    }
    fn to_owned(&self) -> Box<dyn FieldSelector + Send + Sync> {
        Box::new(self.clone())
    }
}

/// Boxed field selector for owned selection
impl FieldSelector for Box<dyn FieldSelector + Send + Sync> {
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize> {
        (**self).resolve_fields(fields)
    }
    fn to_owned(&self) -> Box<dyn FieldSelector + Send + Sync> {
        (**self).to_owned()
    }
}

// These allow users to pass indices and ranges when selecting data (rows, time, etc.).
// For example: table.r(5), table.r(&[1, 3, 5]), table.r(0..10)

/// Single data index
impl DataSelector for usize {
    fn resolve_indices(&self, count: usize) -> Vec<usize> {
        if *self < count {
            vec![*self]
        } else {
            Vec::new()
        }
    }

    /// A single index is a contiguous window of length one.
    fn is_contiguous(&self) -> bool {
        true
    }

    #[cfg(feature = "ndarray")]
    fn resolve_axis(&self, dim_size: usize) -> (usize, usize, bool) {
        assert!(
            *self < dim_size,
            "axis selection: index {} out of bounds (size {})", self, dim_size
        );
        (*self, *self + 1, true)
    }
}

/// Single data index from a plain integer literal. Negative values
/// resolve to nothing.
impl DataSelector for i32 {
    fn resolve_indices(&self, count: usize) -> Vec<usize> {
        if *self >= 0 && (*self as usize) < count {
            vec![*self as usize]
        } else {
            Vec::new()
        }
    }

    /// A single index is a contiguous window of length one.
    fn is_contiguous(&self) -> bool {
        true
    }

    #[cfg(feature = "ndarray")]
    fn resolve_axis(&self, dim_size: usize) -> (usize, usize, bool) {
        assert!(
            *self >= 0 && (*self as usize) < dim_size,
            "axis selection: index {} out of bounds (size {})", self, dim_size
        );
        (*self as usize, *self as usize + 1, true)
    }
}

/// Multiple data indices
impl DataSelector for &[usize] {
    fn resolve_indices(&self, count: usize) -> Vec<usize> {
        self.iter().copied().filter(|&idx| idx < count).collect()
    }
}

/// Multiple data indices (array reference)
impl<const N: usize> DataSelector for &[usize; N] {
    fn resolve_indices(&self, count: usize) -> Vec<usize> {
        self.iter().copied().filter(|&idx| idx < count).collect()
    }
}

/// Multiple data indices (Vec)
impl DataSelector for Vec<usize> {
    fn resolve_indices(&self, count: usize) -> Vec<usize> {
        self.iter().copied().filter(|&idx| idx < count).collect()
    }
}

/// Data range selection
impl DataSelector for Range<usize> {
    fn resolve_indices(&self, count: usize) -> Vec<usize> {
        let end = self.end.min(count);
        (self.start..end).collect()
    }

    fn is_contiguous(&self) -> bool {
        true
    }

    #[cfg(feature = "ndarray")]
    fn resolve_axis(&self, dim_size: usize) -> (usize, usize, bool) {
        assert!(
            self.start <= self.end && self.end <= dim_size,
            "axis selection: range {}..{} out of bounds (size {})",
            self.start, self.end, dim_size
        );
        (self.start, self.end, false)
    }
}

/// Data range selection from plain integer literals. Negative bounds
/// clamp to zero.
impl DataSelector for Range<i32> {
    fn resolve_indices(&self, count: usize) -> Vec<usize> {
        let start = self.start.max(0) as usize;
        let end = (self.end.max(0) as usize).min(count);
        (start..end).collect()
    }

    fn is_contiguous(&self) -> bool {
        true
    }

    #[cfg(feature = "ndarray")]
    fn resolve_axis(&self, dim_size: usize) -> (usize, usize, bool) {
        assert!(
            self.start >= 0 && self.end >= self.start && (self.end as usize) <= dim_size,
            "axis selection: range {}..{} out of bounds (size {})",
            self.start, self.end, dim_size
        );
        (self.start as usize, self.end as usize, false)
    }
}

/// Data range from selection
impl DataSelector for RangeFrom<usize> {
    fn resolve_indices(&self, count: usize) -> Vec<usize> {
        (self.start..count).collect()
    }

    fn is_contiguous(&self) -> bool {
        true
    }

    #[cfg(feature = "ndarray")]
    fn resolve_axis(&self, dim_size: usize) -> (usize, usize, bool) {
        assert!(
            self.start <= dim_size,
            "axis selection: range {}.. out of bounds (size {})", self.start, dim_size
        );
        (self.start, dim_size, false)
    }
}

/// Data range from selection from a plain integer literal.
impl DataSelector for RangeFrom<i32> {
    fn resolve_indices(&self, count: usize) -> Vec<usize> {
        let start = self.start.max(0) as usize;
        (start..count).collect()
    }

    fn is_contiguous(&self) -> bool {
        true
    }

    #[cfg(feature = "ndarray")]
    fn resolve_axis(&self, dim_size: usize) -> (usize, usize, bool) {
        assert!(
            self.start >= 0 && (self.start as usize) <= dim_size,
            "axis selection: range {}.. out of bounds (size {})", self.start, dim_size
        );
        (self.start as usize, dim_size, false)
    }
}

/// Data range to selection
impl DataSelector for RangeTo<usize> {
    fn resolve_indices(&self, count: usize) -> Vec<usize> {
        let end = self.end.min(count);
        (0..end).collect()
    }

    fn is_contiguous(&self) -> bool {
        true
    }

    #[cfg(feature = "ndarray")]
    fn resolve_axis(&self, dim_size: usize) -> (usize, usize, bool) {
        assert!(
            self.end <= dim_size,
            "axis selection: range ..{} out of bounds (size {})", self.end, dim_size
        );
        (0, self.end, false)
    }
}

/// Data range to selection from a plain integer literal. Negative bounds
/// clamp to zero.
impl DataSelector for RangeTo<i32> {
    fn resolve_indices(&self, count: usize) -> Vec<usize> {
        let end = (self.end.max(0) as usize).min(count);
        (0..end).collect()
    }

    fn is_contiguous(&self) -> bool {
        true
    }

    #[cfg(feature = "ndarray")]
    fn resolve_axis(&self, dim_size: usize) -> (usize, usize, bool) {
        assert!(
            self.end >= 0 && (self.end as usize) <= dim_size,
            "axis selection: range ..{} out of bounds (size {})", self.end, dim_size
        );
        (0, self.end as usize, false)
    }
}

/// Data full range selection
impl DataSelector for RangeFull {
    fn resolve_indices(&self, count: usize) -> Vec<usize> {
        (0..count).collect()
    }

    fn is_contiguous(&self) -> bool {
        true
    }

    #[cfg(feature = "ndarray")]
    fn resolve_axis(&self, dim_size: usize) -> (usize, usize, bool) {
        (0, dim_size, false)
    }
}

/// Data inclusive range selection
impl DataSelector for RangeInclusive<usize> {
    fn resolve_indices(&self, count: usize) -> Vec<usize> {
        let start = *self.start();
        let end = (*self.end() + 1).min(count);
        (start..end).collect()
    }

    fn is_contiguous(&self) -> bool {
        true
    }

    #[cfg(feature = "ndarray")]
    fn resolve_axis(&self, dim_size: usize) -> (usize, usize, bool) {
        assert!(
            self.start() <= self.end() && *self.end() < dim_size,
            "axis selection: range {}..={} out of bounds (size {})",
            self.start(), self.end(), dim_size
        );
        (*self.start(), *self.end() + 1, false)
    }
}

/// Data inclusive range selection from plain integer literals. Negative
/// bounds clamp to zero.
impl DataSelector for RangeInclusive<i32> {
    fn resolve_indices(&self, count: usize) -> Vec<usize> {
        if *self.end() < 0 {
            return Vec::new();
        }
        let start = (*self.start()).max(0) as usize;
        let end = (*self.end() as usize + 1).min(count);
        (start..end).collect()
    }

    fn is_contiguous(&self) -> bool {
        true
    }

    #[cfg(feature = "ndarray")]
    fn resolve_axis(&self, dim_size: usize) -> (usize, usize, bool) {
        assert!(
            *self.start() >= 0 && self.start() <= self.end() && (*self.end() as usize) < dim_size,
            "axis selection: range {}..={} out of bounds (size {})",
            self.start(), self.end(), dim_size
        );
        (*self.start() as usize, *self.end() as usize + 1, false)
    }
}
