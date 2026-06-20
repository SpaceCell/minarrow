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

//! # **TableV Module** - *Windowed View over a Table*
//!
//! `TableV` is a **row-aligned, zero-copy view** over a `Table`, holding the
//! window `[offset .. offset+len)` across all columns.
//!
//! ## Purpose
//! - Stream or batch rows without copying full columns.
//! - Do per-partition / per-window work with stable schema metadata (`Field`s).
//! - Glue for chunked processing alongside `SuperTableV` and `ArrayV`, but focused
//! on windowed zero-copy access within a single Table batch.
//!
//! ## Behaviour
//! - Columns are exposed as `ArrayV` windows with the same `(offset, len)`.
//! - Schema (`Field`s) is retained and shared; column data is not copied.
//! - Slicing a `TableV` (`from_self`) is O(1) - metadata-only.
//! - `to_table()` materialises an owned `Table` copy of the window.
//!
//! ## When to use
//! - Sliding windows, micro-batching, streaming sinks/sources.
//! - Thread-local views where each worker consumes a row range.
//! - Building higher-level chunked tables (`SuperTableV`) from base tables.
//!
//! ## Example
//! ```rust
//! use minarrow::{Table, TableV, fa_i32};
//! use minarrow::ColumnSelection;  // Re-exported at crate root
//! use std::sync::Arc;
//!
//! // Build a simple 2-column table
//! let a = fa_i32!("a", 1, 2, 3, 4, 5);
//! let b = fa_i32!("b", 10, 20, 30, 40, 50);
//! let mut tbl = Table::new("T".to_string(), vec![a,b].into());
//!
//! // View rows 1..4 (3 rows total)
//! let tv = TableV::from_table(tbl, 1, 3);
//! assert_eq!(tv.n_rows(), 3);
//! assert_eq!(tv.n_cols(), 2);
//! // Access a column window via ColumnSelection trait
//! let col0 = tv.col_ix(0).unwrap();
//! assert_eq!(col0.get::<minarrow::IntegerArray<i32>>(0), Some(2));
//!
//! // Materialise the window as an owned Table copy
//! let owned = tv.to_table();
//! assert_eq!(owned.n_rows, 3);
//! ```

use std::fmt::{Display, Formatter};
use std::sync::Arc;

#[cfg(feature = "select")]
use crate::Array;
#[cfg(feature = "views")]
use crate::ArrayV;
use crate::enums::error::MinarrowError;
use crate::enums::shape_dim::ShapeDim;
use crate::traits::concatenate::Concatenate;
use crate::traits::print::{
    MAX_PREVIEW, print_ellipsis_row, print_header_row, print_rule, value_to_string,
};
#[cfg(feature = "select")]
use crate::traits::selection::{ColumnSelection, DataSelector, FieldSelector, RowSelection};
use crate::traits::shape::Shape;
use crate::{Field, FieldArray, Table};

/// # TableView
///
/// Row-aligned view into a `Table` over `[offset .. offset+len)`.
///
/// ## Fields
/// - `name`: table name, propagated from the parent `Table`.
/// - `fields`: shared schema, one `Arc<Field>` per column.
/// - `cols`: column windows as `ArrayV`, all with the same `(offset, len)`.
/// - `offset`: starting row, relative to parent table.
/// - `len`: number of rows in the window.
/// - `active_col_selection`: optional column selection indices for lazy
///   column filtering. When set, access methods transparently return only
///   the selected columns. Row windowing is handled by `offset`/`len`.
///
/// ## Notes
/// - Construction from a `Table`/`Arc<Table>` is zero-copy for column data.
/// - Use `from_self` to take sub-windows cheaply.
/// - Use `to_table()` to materialise an owned copy of just this window.
/// - Column helpers (`col`, `col_ix`, `col_vec`) provide ergonomic access.
/// - When `active_col_selection` is set, all access methods filter transparently.
///   The underlying `fields` and `cols` vectors retain the full schema so that
///   selections can be composed or cleared without reallocating.
#[derive(Debug, Clone, PartialEq)]
pub struct TableV {
    /// Table name
    pub name: String,
    /// Full schema - one entry per column in the parent table
    pub fields: Vec<Arc<Field>>,
    /// Full column windows - one ArrayV per column, all with the same (offset, len)
    pub cols: Vec<ArrayV>,
    /// Row offset from start of parent table
    pub offset: usize,
    /// Number of rows in the window
    pub len: usize,
    /// Lazy column selection. When Some, only these indices into `fields`/`cols`
    /// are considered active. Access methods, iteration, and materialisation
    /// all filter through this selection transparently.
    #[cfg(feature = "select")]
    pub active_col_selection: Option<Vec<usize>>,
}

impl TableV {
    /// Creates a new `TableView` over `table[offset .. offset+len)`.
    /// Provides a non-owning view into a subrange of the table.
    #[inline]
    pub fn from_table(table: Table, offset: usize, len: usize) -> Self {
        let mut fields = Vec::with_capacity(table.cols.len());
        let mut cols = Vec::with_capacity(table.cols.len());

        for fa in &table.cols {
            fields.push(fa.field.clone());
            cols.push(ArrayV::new(fa.array.clone(), offset, len));
        }

        Self {
            name: table.name.clone(),
            fields,
            cols,
            offset,
            len,
            #[cfg(feature = "select")]
            active_col_selection: None,
        }
    }

    /// Creates a new `TableView` over `table[offset .. offset+len)`.
    /// Provides a non-owning view into a subrange of the table.
    #[inline]
    pub fn from_arc_table(table: Arc<Table>, offset: usize, len: usize) -> Self {
        let mut fields = Vec::with_capacity(table.cols.len());
        let mut cols = Vec::with_capacity(table.cols.len());

        for fa in &table.cols {
            fields.push(fa.field.clone());
            cols.push(ArrayV::new(fa.array.clone(), offset, len));
        }

        Self {
            name: table.name.clone(),
            fields,
            cols,
            offset,
            len,
            #[cfg(feature = "select")]
            active_col_selection: None,
        }
    }

    /// Derives a subwindow from this `TableView`, adjusted by `offset` and `len`.
    /// Preserves the active column selection from the parent view.
    #[inline]
    pub fn from_self(&self, offset: usize, len: usize) -> Self {
        assert!(
            len <= self.len && offset <= self.len - len,
            "TableView::from_self: slice out of bounds (offset = {offset}, len = {len}, view len = {})",
            self.len
        );

        let mut fields = Vec::with_capacity(self.cols.len());
        let mut cols = Vec::with_capacity(self.cols.len());

        for (field, array_window) in self.fields.iter().zip(self.cols.iter()) {
            let w = array_window.as_tuple();
            fields.push(field.clone());
            cols.push(ArrayV::new(
                w.0,          // &Array
                w.1 + offset, // adjusted offset
                len,          // subwindow length
            ));
        }

        TableV {
            name: self.name.clone(),
            fields,
            cols,
            offset: self.offset + offset,
            len,
            #[cfg(feature = "select")]
            active_col_selection: self.active_col_selection.clone(),
        }
    }

    /// Clear the active column selection, making all columns active again.
    #[cfg(feature = "select")]
    #[inline]
    pub fn clear_col_selection(&mut self) {
        self.active_col_selection = None;
    }

    /// Returns the active column indices, or all indices if no selection is set.
    #[inline]
    pub fn active_col_indices(&self) -> Vec<usize> {
        #[cfg(feature = "select")]
        if let Some(indices) = &self.active_col_selection {
            return indices.clone();
        }
        (0..self.fields.len()).collect()
    }

    /// Returns true if a column selection is active.
    #[inline]
    pub fn has_col_selection(&self) -> bool {
        #[cfg(feature = "select")]
        { self.active_col_selection.is_some() }
        #[cfg(not(feature = "select"))]
        { false }
    }

    /// Resolve an active column index to the raw index into fields/cols.
    /// When no selection is active, the index is used directly.
    #[inline]
    fn resolve_col_index(&self, idx: usize) -> Option<usize> {
        #[cfg(feature = "select")]
        if let Some(indices) = &self.active_col_selection {
            return indices.get(idx).copied();
        }
        if idx < self.fields.len() { Some(idx) } else { None }
    }

    /// Returns true if the window contains no rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_rows() == 0
    }

    /// True when the view spans the entirety of its backing table:
    /// every column window spans its backing array (offset == 0, length
    /// matches), and no column projection is active. When true,
    /// `to_table()` produces an Arc-bumped Table with no per-column
    /// buffer copy. When false the view is genuinely windowed or
    /// projected and `to_table()` reallocates each column buffer.
    #[inline]
    pub fn spans_backing(&self) -> bool {
        #[cfg(feature = "select")]
        {
            if self.active_col_selection.is_some() {
                return false;
            }
        }
        self.cols.iter().all(|c| c.spans_backing())
    }

    /// Returns the exclusive end row index of the window.
    #[inline]
    pub fn end(&self) -> usize {
        self.offset + self.len
    }

    /// Returns the number of active columns in the table window.
    #[inline]
    pub fn n_cols(&self) -> usize {
        #[cfg(feature = "select")]
        if let Some(indices) = &self.active_col_selection {
            return indices.len();
        }
        self.cols.len()
    }

    /// Returns the number of rows in the window.
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.len
    }

    /// Returns the number of rows in the window.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the name of the table.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the names of all active columns.
    #[inline]
    pub fn col_names(&self) -> Vec<&str> {
        #[cfg(feature = "select")]
        if let Some(indices) = &self.active_col_selection {
            return indices
                .iter()
                .filter_map(|&i| self.fields.get(i).map(|f| f.name.as_str()))
                .collect();
        }
        self.fields.iter().map(|f| f.name.as_str()).collect()
    }

    /// Rename columns in place. Each pair is (old_name, new_name).
    ///
    /// Only searches active columns when a selection is set.
    /// Returns an error if any old name is not found among active columns.
    /// This is metadata-only - array data is not touched.
    #[cfg(feature = "select")]
    pub fn rename_columns(
        &mut self,
        mapping: &[(&str, &str)],
    ) -> Result<(), MinarrowError> {
        let active = self.active_col_indices();
        for &(old, _) in mapping {
            if !active.iter().any(|&i| self.fields.get(i).is_some_and(|f| f.name == old)) {
                return Err(MinarrowError::IndexError(format!(
                    "rename_columns: column '{}' not found",
                    old
                )));
            }
        }
        for &idx in &active {
            if let Some(field_arc) = self.fields.get_mut(idx) {
                for &(old, new) in mapping {
                    if field_arc.name == old {
                        match Arc::get_mut(field_arc) {
                            Some(f) => f.name = new.to_string(),
                            None => {
                                let f = field_arc.as_ref();
                                *field_arc = Arc::new(Field::new(
                                    new,
                                    f.dtype.clone(),
                                    f.nullable,
                                    if f.metadata.is_empty() {
                                        None
                                    } else {
                                        Some(f.metadata.clone())
                                    },
                                ));
                            }
                        }
                        break;
                    }
                }
            }
        }
        Ok(())
    }

    /// Returns the position of a column by name within the active selection.
    #[inline]
    pub fn col_name_index(&self, name: &str) -> Option<usize> {
        #[cfg(feature = "select")]
        if let Some(indices) = &self.active_col_selection {
            return indices
                .iter()
                .position(|&i| self.fields.get(i).is_some_and(|f| f.name == name));
        }
        self.fields.iter().position(|f| f.name == name)
    }

    /// Returns the window of the column at the given active index.
    #[inline]
    pub fn col_window(&self, idx: usize) -> Option<ArrayV> {
        let raw = self.resolve_col_index(idx)?;
        self.cols.get(raw).map(|av| {
            let (array, offset, len) = &av.as_tuple();
            ArrayV::new(array.clone(), *offset, *len)
        })
    }

    /// Returns the name of the column at the given active index.
    #[inline]
    pub fn col_name(&self, idx: usize) -> Option<&str> {
        let raw = self.resolve_col_index(idx)?;
        self.fields.get(raw).map(|f| f.name.as_str())
    }

    /// Returns a `Vec<bool>` indicating which active columns are nullable.
    #[inline]
    pub fn nullable_cols(&self) -> Vec<bool> {
        self.active_col_indices()
            .iter()
            .filter_map(|&i| self.fields.get(i).map(|f| f.nullable))
            .collect()
    }

    /// Materialise the view into an owned Table. Respects active column selection.
    ///
    /// Each column delegates to `ArrayV::to_array()`, which avoids a buffer copy
    /// when the column window already spans its full backing array. Genuinely
    /// row-windowed columns fall through to `slice_clone` inside `to_array`.
    pub fn to_table(&self) -> Table {
        let col_indices = self.active_col_indices();
        let cols: Vec<_> = col_indices
            .iter()
            .filter_map(|&col_idx| {
                let field = self.fields.get(col_idx)?;
                let window = self.cols.get(col_idx)?;

                let array = window.to_array();
                let null_count = array.null_count();

                Some(FieldArray {
                    field: field.clone(),
                    array,
                    null_count,
                })
            })
            .collect();

        let n_rows = self.len;
        Table::build(cols, n_rows, self.name.clone())
    }

    /// Apply a transformation to each active column view, producing a new table.
    ///
    /// The closure receives the field and column view directly without
    /// materialisation. To pass a column through unchanged, materialise
    /// with `view.to_array()`. To transform, operate on the view and
    /// return the result as a FieldArray.
    pub fn apply_cols<E>(
        &self,
        mut f: impl FnMut(&Arc<Field>, &ArrayV) -> Result<FieldArray, E>,
    ) -> Result<Table, E> {
        let indices = self.active_col_indices();
        let cols = indices
            .iter()
            .filter_map(|&i| {
                let field = self.fields.get(i)?;
                let col_view = self.cols.get(i)?;
                Some(f(field, col_view))
            })
            .collect::<Result<Vec<_>, E>>()?;
        Ok(Table::new(self.name.clone(), Some(cols)))
    }

    /// Gather specific rows from an ArrayV window
    #[cfg(feature = "select")]
    fn gather_rows_from_window(&self, window: &ArrayV, row_indices: &[usize]) -> Option<Array> {
        use crate::{
            Array, BooleanArray, CategoricalArray, FloatArray, IntegerArray, MaskedArray,
            NumericArray, StringArray, TextArray,
        };
        #[cfg(feature = "datetime")]
        use crate::{DatetimeArray, TemporalArray};

        let result = match &window.array {
            Array::Null => return None,
            Array::NumericArray(num_arr) => match num_arr {
                NumericArray::Int32(_) => {
                    let mut new_arr = IntegerArray::<i32>::with_capacity(row_indices.len(), true);
                    for &idx in row_indices {
                        if let Some(val) = window.get::<IntegerArray<i32>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_int32(new_arr)
                }
                NumericArray::Int64(_) => {
                    let mut new_arr = IntegerArray::<i64>::with_capacity(row_indices.len(), true);
                    for &idx in row_indices {
                        if let Some(val) = window.get::<IntegerArray<i64>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_int64(new_arr)
                }
                NumericArray::UInt32(_) => {
                    let mut new_arr = IntegerArray::<u32>::with_capacity(row_indices.len(), true);
                    for &idx in row_indices {
                        if let Some(val) = window.get::<IntegerArray<u32>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_uint32(new_arr)
                }
                NumericArray::UInt64(_) => {
                    let mut new_arr = IntegerArray::<u64>::with_capacity(row_indices.len(), true);
                    for &idx in row_indices {
                        if let Some(val) = window.get::<IntegerArray<u64>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_uint64(new_arr)
                }
                NumericArray::Float32(_) => {
                    let mut new_arr = FloatArray::<f32>::with_capacity(row_indices.len(), true);
                    for &idx in row_indices {
                        if let Some(val) = window.get::<FloatArray<f32>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_float32(new_arr)
                }
                NumericArray::Float64(_) => {
                    let mut new_arr = FloatArray::<f64>::with_capacity(row_indices.len(), true);
                    for &idx in row_indices {
                        if let Some(val) = window.get::<FloatArray<f64>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_float64(new_arr)
                }
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(_) => {
                    let mut new_arr = IntegerArray::<i8>::with_capacity(row_indices.len(), true);
                    for &idx in row_indices {
                        if let Some(val) = window.get::<IntegerArray<i8>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_int8(new_arr)
                }
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(_) => {
                    let mut new_arr = IntegerArray::<i16>::with_capacity(row_indices.len(), true);
                    for &idx in row_indices {
                        if let Some(val) = window.get::<IntegerArray<i16>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_int16(new_arr)
                }
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(_) => {
                    let mut new_arr = IntegerArray::<u8>::with_capacity(row_indices.len(), true);
                    for &idx in row_indices {
                        if let Some(val) = window.get::<IntegerArray<u8>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_uint8(new_arr)
                }
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(_) => {
                    let mut new_arr = IntegerArray::<u16>::with_capacity(row_indices.len(), true);
                    for &idx in row_indices {
                        if let Some(val) = window.get::<IntegerArray<u16>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_uint16(new_arr)
                }
                NumericArray::Null => return None,
            },
            Array::TextArray(text_arr) => match text_arr {
                TextArray::String32(_) => {
                    let mut new_arr = StringArray::<u32>::default();
                    for &idx in row_indices {
                        if let Some(val) = window.get_str(idx) {
                            new_arr.push(val.to_string());
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_string32(new_arr)
                }
                #[cfg(feature = "large_string")]
                TextArray::String64(_) => {
                    let mut new_arr = StringArray::<u64>::default();
                    for &idx in row_indices {
                        if let Some(val) = window.get_str(idx) {
                            new_arr.push(val.to_string());
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_string64(new_arr)
                }
                #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
                TextArray::Categorical32(_) => {
                    use crate::{Bitmask, Vec64};
                    use std::collections::HashMap;

                    let mut codes = Vec64::<u32>::with_capacity(row_indices.len());
                    let mut value_map = HashMap::<String, u32>::new();
                    let mut mask = Bitmask::new_set_all(row_indices.len(), true);

                    for (i, &idx) in row_indices.iter().enumerate() {
                        if let Some(val) = window.get_str(idx) {
                            let code = if let Some(&existing_code) = value_map.get(val) {
                                existing_code
                            } else {
                                let new_code = value_map.len() as u32;
                                value_map.insert(val.to_string(), new_code);
                                new_code
                            };
                            codes.push(code);
                        } else {
                            codes.push(0);
                            mask.set_false(i);
                        }
                    }

                    let mut unique_values = Vec64::<String>::with_capacity(value_map.len());
                    unique_values.resize(value_map.len(), String::new());
                    for (val, code) in value_map {
                        unique_values[code as usize] = val;
                    }

                    let null_mask = if mask.all_set() { None } else { Some(mask) };

                    let new_arr = CategoricalArray::<u32>::new(codes, unique_values, null_mask);
                    Array::from_categorical32(new_arr)
                }
                #[cfg(feature = "default_categorical_8")]
                TextArray::Categorical8(_) => {
                    use crate::{Bitmask, Vec64};
                    use std::collections::HashMap;

                    let mut codes = Vec64::<u8>::with_capacity(row_indices.len());
                    let mut value_map = HashMap::<String, u8>::new();
                    let mut mask = Bitmask::new_set_all(row_indices.len(), true);

                    for (i, &idx) in row_indices.iter().enumerate() {
                        if let Some(val) = window.get_str(idx) {
                            let code = if let Some(&existing_code) = value_map.get(val) {
                                existing_code
                            } else {
                                let new_code = value_map.len() as u8;
                                value_map.insert(val.to_string(), new_code);
                                new_code
                            };
                            codes.push(code);
                        } else {
                            codes.push(0);
                            mask.set_false(i);
                        }
                    }

                    let mut unique_values = Vec64::<String>::with_capacity(value_map.len());
                    unique_values.resize(value_map.len(), String::new());
                    for (val, code) in value_map {
                        unique_values[code as usize] = val;
                    }

                    let null_mask = if mask.all_set() { None } else { Some(mask) };

                    let new_arr = CategoricalArray::<u8>::new(codes, unique_values, null_mask);
                    Array::from_categorical8(new_arr)
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(_) => {
                    use crate::{Bitmask, Vec64};
                    use std::collections::HashMap;

                    let mut codes = Vec64::<u16>::with_capacity(row_indices.len());
                    let mut value_map = HashMap::<String, u16>::new();
                    let mut mask = Bitmask::new_set_all(row_indices.len(), true);

                    for (i, &idx) in row_indices.iter().enumerate() {
                        if let Some(val) = window.get_str(idx) {
                            let code = if let Some(&existing_code) = value_map.get(val) {
                                existing_code
                            } else {
                                let new_code = value_map.len() as u16;
                                value_map.insert(val.to_string(), new_code);
                                new_code
                            };
                            codes.push(code);
                        } else {
                            codes.push(0);
                            mask.set_false(i);
                        }
                    }

                    let mut unique_values = Vec64::<String>::with_capacity(value_map.len());
                    unique_values.resize(value_map.len(), String::new());
                    for (val, code) in value_map {
                        unique_values[code as usize] = val;
                    }

                    let null_mask = if mask.all_set() { None } else { Some(mask) };

                    let new_arr = CategoricalArray::<u16>::new(codes, unique_values, null_mask);
                    Array::from_categorical16(new_arr)
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(_) => {
                    use crate::{Bitmask, Vec64};
                    use std::collections::HashMap;

                    let mut codes = Vec64::<u64>::with_capacity(row_indices.len());
                    let mut value_map = HashMap::<String, u64>::new();
                    let mut mask = Bitmask::new_set_all(row_indices.len(), true);

                    for (i, &idx) in row_indices.iter().enumerate() {
                        if let Some(val) = window.get_str(idx) {
                            let code = if let Some(&existing_code) = value_map.get(val) {
                                existing_code
                            } else {
                                let new_code = value_map.len() as u64;
                                value_map.insert(val.to_string(), new_code);
                                new_code
                            };
                            codes.push(code);
                        } else {
                            codes.push(0);
                            mask.set_false(i);
                        }
                    }

                    let mut unique_values = Vec64::<String>::with_capacity(value_map.len());
                    unique_values.resize(value_map.len(), String::new());
                    for (val, code) in value_map {
                        unique_values[code as usize] = val;
                    }

                    let null_mask = if mask.all_set() { None } else { Some(mask) };

                    let new_arr = CategoricalArray::<u64>::new(codes, unique_values, null_mask);
                    Array::from_categorical64(new_arr)
                }
                TextArray::Null => return None,
            },
            Array::BooleanArray(_) => {
                let mut new_arr = BooleanArray::with_capacity(row_indices.len(), true);
                for &idx in row_indices {
                    if let Some(val) = window.get::<BooleanArray<()>>(idx) {
                        new_arr.push(val);
                    } else {
                        new_arr.push_null();
                    }
                }
                Array::from_bool(new_arr)
            }
            #[cfg(feature = "datetime")]
            Array::TemporalArray(temp_arr) => match temp_arr {
                TemporalArray::Datetime32(arr) => {
                    let mut new_arr = DatetimeArray::<i32>::with_capacity(
                        row_indices.len(),
                        true,
                        Some(arr.time_unit),
                    );
                    for &idx in row_indices {
                        if let Some(val) = window.get::<DatetimeArray<i32>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_datetime_i32(new_arr)
                }
                TemporalArray::Datetime64(arr) => {
                    let mut new_arr = DatetimeArray::<i64>::with_capacity(
                        row_indices.len(),
                        true,
                        Some(arr.time_unit),
                    );
                    for &idx in row_indices {
                        if let Some(val) = window.get::<DatetimeArray<i64>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_datetime_i64(new_arr)
                }
                TemporalArray::Null => return None,
            },
        };

        Some(result)
    }

    /// Converts a column window into an owned `FieldArray`, slicing the array and copying data.
    /// Copies the data.
    pub fn extract_column(field: &Field, window: &ArrayV) -> FieldArray {
        let w = window.as_tuple();
        let sliced = w.0.slice_clone(w.1, w.2);
        let null_count = sliced.null_count();
        FieldArray {
            field: field.clone().into(),
            array: sliced,
            null_count,
        }
    }

    /// Gather specific row indices from this view into a materialised Table.
    /// Indices are relative to this view's window.
    #[cfg(feature = "select")]
    pub fn gather_rows(&self, indices: &[usize]) -> Table {
        if indices.is_empty() {
            return Table::new(self.name.clone(), Some(vec![]));
        }

        let active = self.active_col_indices();
        let cols: Vec<_> = active
            .iter()
            .filter_map(|&col_idx| {
                let field = self.fields.get(col_idx)?;
                let window = self.cols.get(col_idx)?;
                let gathered_array = self.gather_rows_from_window(window, indices)?;
                let null_count = gathered_array.null_count();

                Some(FieldArray {
                    field: field.clone(),
                    array: gathered_array,
                    null_count,
                })
            })
            .collect();

        Table::build(cols, indices.len(), self.name.clone())
    }
}

impl Display for TableV {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let n_rows = self.n_rows();
        let n_cols = self.n_cols();

        if n_cols == 0 {
            return writeln!(f, "TableView \"{}\" [0 rows × 0 cols] – empty", self.name);
        }

        // Determine which rows to display
        let row_indices: Vec<usize> = if n_rows <= MAX_PREVIEW {
            (0..n_rows).collect()
        } else {
            let mut idx = (0..10).collect::<Vec<_>>();
            idx.extend((n_rows - 10)..n_rows);
            idx
        };

        // Build column headers and track widths
        let mut headers: Vec<String> = Vec::with_capacity(n_cols);
        let mut widths: Vec<usize> = Vec::with_capacity(n_cols);

        for col_idx in 0..n_cols {
            if let Some(_col_view) = self.cols.get(col_idx) {
                let hdr = if let Some(f) = self.fields.get(col_idx) {
                    format!("{}:{:?}", f.name, f.dtype)
                } else {
                    "unknown".to_string()
                };
                widths.push(hdr.len());
                headers.push(hdr);
            }
        }

        // Build matrix of cell strings
        let mut rows: Vec<Vec<String>> = Vec::with_capacity(row_indices.len());

        for &row_idx in &row_indices {
            let mut row: Vec<String> = Vec::with_capacity(n_cols);

            for col_idx in 0..n_cols {
                if let Some(col_view) = self.cols.get(col_idx) {
                    let val = value_to_string(&col_view.array, row_idx);
                    widths[col_idx] = widths[col_idx].max(val.len());
                    row.push(val);
                } else {
                    row.push("·".to_string());
                }
            }
            rows.push(row);
        }

        // Calculate idx column width
        let max_idx = n_rows.saturating_sub(1);
        let idx_width = usize::max(
            3, // "idx"
            (max_idx as f64).log10().floor() as usize + 1,
        );

        // Render header
        writeln!(
            f,
            "TableView \"{}\" [{} rows × {} cols]",
            self.name, n_rows, n_cols
        )?;
        print_rule(f, idx_width, &widths)?;
        print_header_row(f, idx_width, &headers, &widths)?;
        print_rule(f, idx_width, &widths)?;

        // Render body
        for (i, cells) in rows.iter().enumerate() {
            let row_idx = row_indices[i];

            write!(f, "| {idx:>w$} |", idx = row_idx, w = idx_width)?;
            for (col_idx, cell) in cells.iter().enumerate() {
                write!(f, " {val:^w$} |", val = cell, w = widths[col_idx])?;
            }
            writeln!(f)?;
            if i == 9 && n_rows > MAX_PREVIEW {
                print_ellipsis_row(f, idx_width, &widths)?;
            }
        }
        print_rule(f, idx_width, &widths)
    }
}

impl Shape for TableV {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank2 {
            rows: self.n_rows(),
            cols: self.n_cols(),
        }
    }
}

impl Concatenate for TableV {
    /// Concatenates two table views by materialising both to owned tables,
    /// concatenating them, and wrapping the result back in a view.
    ///
    /// # Notes
    /// - This operation copies data from both views to create owned tables.
    /// - The resulting view has offset=0 and length equal to the combined length.
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        // Materialise both views to owned tables
        let self_table = self.to_table();
        let other_table = other.to_table();

        // Concatenate the owned tables
        let concatenated = self_table.concat(other_table)?;

        // Wrap the result in a new view
        Ok(TableV::from(concatenated))
    }
}

// From implementations for conversion between Table and TableV

/// Table -> TableV conversion
impl From<Table> for TableV {
    fn from(table: Table) -> Self {
        let fields: Vec<Arc<Field>> = table.cols.iter().map(|fa| fa.field.clone()).collect();

        let cols: Vec<ArrayV> = table.cols.into_iter().map(|fa| ArrayV::from(fa)).collect();

        TableV {
            name: table.name,
            fields,
            cols,
            offset: 0,
            len: table.n_rows,
            #[cfg(feature = "select")]
            active_col_selection: None,
        }
    }
}

/// Arc<Table> -> TableV conversion. Creates a zero-copy view over the Arc'd table.
impl From<Arc<Table>> for TableV {
    fn from(table: Arc<Table>) -> Self {
        let fields: Vec<Arc<Field>> = table.cols.iter().map(|fa| fa.field.clone()).collect();
        let cols: Vec<ArrayV> = table.cols.iter().map(|fa| ArrayV::from(fa)).collect();
        let len = table.n_rows;

        TableV {
            name: table.name.clone(),
            fields,
            cols,
            offset: 0,
            len,
            #[cfg(feature = "select")]
            active_col_selection: None,
        }
    }
}

/// TableV -> Table conversion. Respects active column selection.
impl From<TableV> for Table {
    fn from(view: TableV) -> Self {
        view.to_table()
    }
}

// ===== Selection Trait Implementations =====

#[cfg(feature = "select")]
impl ColumnSelection for TableV {
    type View = TableV;
    type ColumnView = ArrayV;
    type ColumnOwned = FieldArray;

    fn c<S: FieldSelector>(&self, selection: S) -> TableV {
        // Resolve against the full field set so indices are always raw
        let resolved = selection.resolve_fields(&self.fields);

        // Compose with existing selection if present
        let new_selection = match &self.active_col_selection {
            Some(current) => {
                let current_set: std::collections::HashSet<usize> =
                    current.iter().copied().collect();
                resolved.into_iter().filter(|i| current_set.contains(i)).collect()
            }
            None => resolved,
        };

        TableV {
            name: self.name.clone(),
            fields: self.fields.clone(),
            cols: self.cols.clone(),
            offset: self.offset,
            len: self.len,
            active_col_selection: Some(new_selection),
        }
    }

    fn get(&self, field: &str) -> Option<FieldArray> {
        let idx = self.fields.iter().position(|f| f.name == field)?;
        let raw = self.resolve_col_index(idx)?;
        let col = self.cols.get(raw)?;
        let field_ref = self.fields.get(idx)?;
        // If the view covers the full array, use it without copying
        if col.offset == 0 && col.len() == col.array.len() {
            Some(FieldArray::new_arc(field_ref.clone(), col.array.clone()))
        } else {
            // Is a subview - we need to materialise
            Some(FieldArray::new_arc(field_ref.clone(), col.to_array()))
        }
    }

    fn col_ix(&self, idx: usize) -> Option<ArrayV> {
        let raw = self.resolve_col_index(idx)?;
        self.cols.get(raw).cloned()
    }

    fn col_vec(&self) -> Vec<ArrayV> {
        self.active_col_indices()
            .iter()
            .filter_map(|&i| self.cols.get(i).cloned())
            .collect()
    }

    fn get_cols(&self) -> Vec<Arc<Field>> {
        self.active_col_indices()
            .iter()
            .filter_map(|&i| self.fields.get(i).cloned())
            .collect()
    }
}

#[cfg(feature = "select")]
impl RowSelection for TableV {
    type View = TableV;

    fn r<S: DataSelector>(&self, selection: S) -> TableV {
        if selection.is_contiguous() {
            // Contiguous selection (ranges): derive a properly windowed subview
            let indices = selection.resolve_indices(self.len);
            if indices.is_empty() {
                return self.from_self(0, 0);
            }
            self.from_self(indices[0], indices.len())
        } else {
            // Non-contiguous selection (index arrays): materialise into Table
            let indices = selection.resolve_indices(self.len);
            let materialised_table = self.gather_rows(&indices);
            TableV::from(materialised_table)
        }
    }

    fn get_row_count(&self) -> usize {
        self.len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fa_i32;
    use crate::structs::table::Table;
    #[cfg(feature = "select")]
    use crate::traits::selection::ColumnSelection;

    #[test]
    fn test_table_slice_from_table_and_access() {
        let fa_a = fa_i32!("a", 1, 2, 3, 4, 5);
        let fa_b = fa_i32!("b", 10, 20, 30, 40, 50);

        let mut tbl = Table::new_empty();
        tbl.add_col(fa_a);
        tbl.add_col(fa_b);
        tbl.name = "TestTable".to_string();

        // Slice rows 1..4 (offset 1, len 3)
        let slice = TableV::from_table(tbl, 1, 3);

        assert_eq!(slice.offset, 1);
        assert_eq!(slice.len, 3);
        assert_eq!(slice.n_rows(), 3);
        assert_eq!(slice.n_cols(), 2);
        assert_eq!(slice.name(), "TestTable");
        assert_eq!(slice.fields[0].name, "a");
        assert_eq!(slice.fields[1].name, "b");
        assert_eq!(slice.col_names(), vec!["a", "b"]);
        assert_eq!(slice.col_name_index("b"), Some(1));
        assert!(!slice.is_empty());
        assert_eq!(slice.end(), 4);
    }

    #[cfg(feature = "select")]
    #[test]
    fn test_table_view_selection_trait() {
        let fa_a = fa_i32!("a", 1, 2, 3, 4, 5);

        let mut tbl = Table::new_empty();
        tbl.add_col(fa_a);

        let slice = TableV::from_table(tbl, 0, 5);

        // Test ColumnSelection trait methods
        assert_eq!(slice.col("a").n_cols(), 1);
        assert_eq!(slice.col("nonexistent").n_cols(), 0);
    }

    #[test]
    fn test_table_slice_empty() {
        let tbl = Table::new_empty();
        let slice = TableV::from_table(tbl, 0, 0);
        assert_eq!(slice.n_cols(), 0);
        assert_eq!(slice.n_rows(), 0);
        assert!(slice.is_empty());
    }

    #[cfg(feature = "select")]
    #[test]
    fn test_tablev_row_selection_to_table_column_lengths() {
        use crate::{NumericArray, traits::selection::RowSelection};

        let fa_a = fa_i32!("a", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        let fa_b = fa_i32!("b", 10, 20, 30, 40, 50, 60, 70, 80, 90, 100);

        let mut tbl = Table::new_empty();
        tbl.add_col(fa_a);
        tbl.add_col(fa_b);

        // Create a view of the full table, then select rows via r()
        let full_view = TableV::from_table(tbl, 0, 10);

        // Select first 5 rows from view
        let selected = full_view.r(0..5);
        assert_eq!(selected.len(), 5);
        let result = selected.to_table();
        assert_eq!(result.n_rows(), 5);
        for col in &result.cols {
            assert_eq!(
                col.array.len(),
                5,
                "Column '{}' should have 5 elements",
                col.field.name
            );
        }

        // Select from an already-sliced view
        let partial_view = TableV::from_table(result.clone(), 0, 5);
        let sub_selected = partial_view.r(1..4);
        assert_eq!(sub_selected.len(), 3);
        let sub_result = sub_selected.to_table();
        assert_eq!(sub_result.n_rows(), 3);
        for col in &sub_result.cols {
            assert_eq!(col.array.len(), 3);
        }
        // Verify values: column "a" should be [2, 3, 4]
        match &sub_result.cols[0].array {
            Array::NumericArray(NumericArray::Int32(a)) => {
                let vals: Vec<i32> = (0..a.len()).map(|i| *a.get(i).unwrap()).collect();
                assert_eq!(vals, vec![2, 3, 4]);
            }
            _ => panic!("unexpected type"),
        }

        // Chained selection: view.r().r() should work correctly
        let chained = full_view.r(2..8).r(1..4);
        assert_eq!(chained.len(), 3);
        let chained_result = chained.to_table();
        assert_eq!(chained_result.n_rows(), 3);
        for col in &chained_result.cols {
            assert_eq!(col.array.len(), 3);
        }
        // Values: indices 2..8 gives [3,4,5,6,7,8], then 1..4 gives [4,5,6]
        match &chained_result.cols[0].array {
            Array::NumericArray(NumericArray::Int32(a)) => {
                let vals: Vec<i32> = (0..a.len()).map(|i| *a.get(i).unwrap()).collect();
                assert_eq!(vals, vec![4, 5, 6]);
            }
            _ => panic!("unexpected type"),
        }

        // Empty selection
        let empty = full_view.r(0..0);
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.to_table().n_rows(), 0);
    }

    /// Pull out the inner `Arc<IntegerArray<i32>>` from a column for pointer-identity
    /// checks. Asserts the column is the expected variant.
    #[cfg(test)]
    fn i32_arc(table: &Table, col_idx: usize) -> Arc<crate::IntegerArray<i32>> {
        use crate::NumericArray;
        match &table.cols[col_idx].array {
            Array::NumericArray(NumericArray::Int32(arc)) => arc.clone(),
            other => panic!("expected Int32 column, got {:?}", other),
        }
    }

    #[test]
    fn test_to_table_full_coverage_shares_arc() {
        let fa_a = fa_i32!("a", 1, 2, 3, 4, 5);
        let fa_b = fa_i32!("b", 10, 20, 30, 40, 50);

        let mut tbl = Table::new_empty();
        tbl.add_col(fa_a);
        tbl.add_col(fa_b);

        let src_a = i32_arc(&tbl, 0);
        let src_b = i32_arc(&tbl, 1);

        // Full-coverage view: offset=0, len matches each column's full length.
        let view = TableV::from_table(tbl, 0, 5);
        let materialised = view.to_table();

        let out_a = i32_arc(&materialised, 0);
        let out_b = i32_arc(&materialised, 1);

        // Same allocation reused, not a fresh buffer copy.
        assert!(Arc::ptr_eq(&src_a, &out_a), "column a should share the underlying Arc on full-coverage to_table");
        assert!(Arc::ptr_eq(&src_b, &out_b), "column b should share the underlying Arc on full-coverage to_table");
    }

    #[test]
    fn test_to_table_row_windowed_copies() {
        let fa_a = fa_i32!("a", 1, 2, 3, 4, 5);

        let mut tbl = Table::new_empty();
        tbl.add_col(fa_a);

        let src_a = i32_arc(&tbl, 0);

        // Genuine row window forces per-column slice_clone (no fast path).
        let view = TableV::from_table(tbl, 1, 3);
        let materialised = view.to_table();
        let out_a = i32_arc(&materialised, 0);

        assert!(!Arc::ptr_eq(&src_a, &out_a), "row-windowed to_table must not alias the source buffer");
        assert_eq!(materialised.n_rows(), 3);
    }

    #[cfg(feature = "select")]
    #[test]
    fn test_to_table_column_projection_kept_columns_share_arc() {
        let fa_a = fa_i32!("a", 1, 2, 3, 4, 5);
        let fa_b = fa_i32!("b", 10, 20, 30, 40, 50);
        let fa_c = fa_i32!("c", 100, 200, 300, 400, 500);

        let mut tbl = Table::new_empty();
        tbl.add_col(fa_a);
        tbl.add_col(fa_b);
        tbl.add_col(fa_c);

        let src_a = i32_arc(&tbl, 0);
        let src_c = i32_arc(&tbl, 2);

        // Drop column "b" via projection. Rows are still full coverage on the kept
        // columns, so a and c should both bypass the buffer copy.
        let view = TableV::from_table(tbl, 0, 5).c(vec!["a", "c"]);
        let materialised = view.to_table();

        assert_eq!(materialised.n_cols(), 2);
        let out_a = i32_arc(&materialised, 0);
        let out_c = i32_arc(&materialised, 1);

        assert!(Arc::ptr_eq(&src_a, &out_a), "kept column a should share Arc under projection");
        assert!(Arc::ptr_eq(&src_c, &out_c), "kept column c should share Arc under projection");
    }
}
