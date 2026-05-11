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

//! # **Table Module** - *Standard Table ("RecordBatch") for Columnar Analytics and Data Engineering*
//!
//! Columnar data container pairing a fixed-length set of rows
//! with named, typed `FieldArray` columns.
//!
//! Equivalent in role to Apache Arrow’s `RecordBatch`, with
//! guaranteed column length consistency and optional table name.
//!
//! Great for in-memory analytics, transformation pipelines,
//! and zero-copy FFI interchange.
//!
//! Cast into *Polars* dataframe via `.to_polars()` or *Apache Arrow* RecordBatch via `.to_apache_arrow()`,
//! zero-copy, via the `cast_polars` and `cast_arrow` features. Round-trip the
//! other direction with `Table::from_polars(&df)` or `Table::from_apache_arrow(&rb)`,
//! or call `(&df).into()` / `(&rb).into()`. Each has a `try_*` sibling returning
//! `Result<_, MinarrowError>`.

use std::fmt::{Display, Formatter};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "cast_arrow")]
use arrow::array::RecordBatch;
use log::warn;
#[cfg(feature = "cast_polars")]
use polars::frame::DataFrame;
#[cfg(feature = "cast_polars")]
use polars::prelude::Column;
#[cfg(feature = "parallel_proc")]
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator};

use super::field_array::FieldArray;
#[cfg(all(feature = "views", feature = "select"))]
use crate::ArrayV;
use crate::Field;
#[cfg(feature = "chunked")]
use crate::SuperTable;
#[cfg(feature = "views")]
use crate::{BitmaskV, NumericArrayV, TableV, TextArrayV};
use crate::enums::{error::MinarrowError, shape_dim::ShapeDim};
#[cfg(feature = "chunked")]
use crate::traits::consolidate::Consolidate;
#[cfg(all(feature = "views", feature = "select"))]
use crate::traits::selection::{ColumnSelection, DataSelector, FieldSelector, RowSelection};
use crate::traits::{
    concatenate::Concatenate,
    print::{MAX_PREVIEW, print_ellipsis_row, print_header_row, print_rule, value_to_string},
    shape::Shape,
};

// Global counter for unnamed table instances
static UNNAMED_COUNTER: AtomicUsize = AtomicUsize::new(1);

/// # Table
///
/// # Description
/// - Standard columnar table with named columns (`FieldArray`),
/// a fixed number of rows, and an optional logical table name.
/// - All columns are required to be equal length and have consistent schema.
/// - Supports zero-copy slicing, efficient iteration, and bulk operations.
/// - Equivalent to the `RecordBatch` in *Apache Arrow*.
///
/// # Structure
/// - `cols`: A vector of `FieldArray`, each representing a column with metadata and data.
/// - `n_rows`: The logical number of rows (guaranteed equal for all columns).
/// - `name`: Optional logical name or alias for this table instance.
///
/// # Usage
/// - Use `Table` as a general-purpose, in-memory columnar data container.
/// - Good for analytics, and transformation pipelines.
/// - For batched/partitioned tables, see [`SuperTable`] or windowed/chunked abstractions.
/// - Cast into *Polars* dataframe via `.to_polars()` or *Apache Arrow* via `.to_apache_arrow()`
/// - FFI-compatible
///
/// # Notes
/// - Table instances are typically lightweight to clone and pass by value.
/// - For mutation, construct a new table or replace individual columns as needed.
/// - There is an alias `RecordBatch` under [crate::aliases::RecordBatch]
///
/// # Example
/// ```rust
/// use minarrow::{fa_i32, fa_str32, Print, Table};
///
/// let col1 = fa_i32!("numbers", 1, 2, 3);
/// let col2 = fa_str32!("letters", "x", "y", "z");
///
/// let mut tbl = Table::new("Demo".into(), vec![col1, col2].into());
/// tbl.print();
/// ```
#[repr(C, align(64))]
#[derive(Default, PartialEq, Clone, Debug)]
pub struct Table {
    /// FieldArrays representing named columns.
    pub cols: Vec<FieldArray>,
    /// Number of rows in the table.
    pub n_rows: usize,
    /// Table name
    pub name: String,
    /// Schema-level metadata as key-value pairs.
    /// Captures metadata that Arrow producers like PyArrow embed
    /// in the top-level ArrowSchema.metadata, e.g. pandas categorical ordering.
    #[cfg(feature = "table_metadata")]
    pub metadata: std::collections::BTreeMap<String, String>,
}

impl Table {
    /// Internal constructor handling the conditional metadata field.
    /// All code paths that build a `Table` from parts should go through here
    /// so the `#[cfg]` lives in one place.
    #[inline(always)]
    pub(crate) fn build(cols: Vec<FieldArray>, n_rows: usize, name: String) -> Self {
        Self {
            cols,
            n_rows,
            name,
            #[cfg(feature = "table_metadata")]
            metadata: std::collections::BTreeMap::new(),
        }
    }

    /// Constructs a new Table with a specified name and optional columns.
    /// If `cols` is provided, the number of rows will be inferred from the first column.
    pub fn new(name: String, cols: Option<Vec<FieldArray>>) -> Self {
        let cols = cols.unwrap_or_else(Vec::new);
        let n_rows = cols.first().map(|col| col.len()).unwrap_or(0);

        let name = if name.trim().is_empty() {
            let id = UNNAMED_COUNTER.fetch_add(1, Ordering::Relaxed);
            format!("UnnamedTable{}", id)
        } else {
            name
        };

        Self::build(cols, n_rows, name)
    }

    /// Constructs a new Table with schema-level metadata.
    ///
    /// Use this when importing data from Arrow producers that embed metadata
    /// in the top-level schema, e.g. pandas categorical ordering via PyArrow.
    #[cfg(feature = "table_metadata")]
    pub fn new_with_metadata(
        name: String,
        cols: Option<Vec<FieldArray>>,
        metadata: std::collections::BTreeMap<String, String>,
    ) -> Self {
        let mut table = Self::new(name, cols);
        table.metadata = metadata;
        table
    }

    /// Returns a reference to the schema-level metadata.
    #[cfg(feature = "table_metadata")]
    pub fn metadata(&self) -> &std::collections::BTreeMap<String, String> {
        &self.metadata
    }

    /// Constructs a new, empty Table with a globally unique name.
    pub fn new_empty() -> Self {
        let id = UNNAMED_COUNTER.fetch_add(1, Ordering::Relaxed);
        let name = format!("UnnamedTable{}", id);
        Self::build(Vec::new(), 0, name)
    }

    /// Build a Table from an Arena and its collected array regions.
    ///
    /// Freezes the arena into a SharedBuffer, then reconstructs each
    /// column as a zero-copy view into the shared allocation. This is
    /// the read-side complement to `Arena::write_slices` and friends.
    ///
    /// Typical use: IPC or streaming ingestion where batch sizes are
    /// known from message headers, allowing a single arena allocation
    /// per batch.
    #[cfg(feature = "arena")]
    pub fn from_arena(
        name: String,
        schema: &[Arc<Field>],
        arena: crate::structs::arena::Arena,
        regions: Vec<crate::structs::arena::AAMaker>,
        n_rows: usize,
    ) -> Self {
        let shared = arena.freeze();
        let cols: Vec<FieldArray> = schema
            .iter()
            .zip(regions)
            .map(|(field, region)| {
                let array = region.to_array(&field.dtype, &shared, n_rows);
                let null_count = array.null_count();
                FieldArray {
                    field: field.clone(),
                    array,
                    null_count,
                }
            })
            .collect();

        Self::build(cols, n_rows, name)
    }

    /// Adds a column with a name.
    pub fn add_col(&mut self, field_array: FieldArray) {
        let array_len = field_array.len();
        if self.cols.is_empty() {
            self.n_rows = array_len;
        } else {
            assert!(self.n_rows == array_len, "Column length mismatch");
        }
        self.cols.push(field_array);
    }

    /// Builds a schema via the underlying field arrays
    pub fn schema(&self) -> Vec<Arc<Field>> {
        let mut vec = Vec::new();
        for fa in &self.cols {
            vec.push(fa.field.clone())
        }
        vec
    }

    /// Returns the number of columns.
    pub fn n_cols(&self) -> usize {
        self.cols.len()
    }

    /// Returns the number of rows.
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Returns true if the table is empty (no columns or no rows).
    pub fn is_empty(&self) -> bool {
        self.n_cols() == 0 || self.n_rows == 0
    }

    /// Returns the list of column names.
    pub fn col_names(&self) -> Vec<&str> {
        self.cols.iter().map(|fa| fa.field.name.as_str()).collect()
    }

    /// Rename columns in place. Each pair is (old_name, new_name).
    ///
    /// Returns an error if any old name is not found.
    /// This is metadata-only - array data is not touched.
    pub fn rename_columns(
        &mut self,
        mapping: &[(&str, &str)],
    ) -> Result<(), MinarrowError> {
        for &(old, _) in mapping {
            if !self.cols.iter().any(|fa| fa.field.name == old) {
                return Err(MinarrowError::IndexError(format!(
                    "rename_columns: column '{}' not found",
                    old
                )));
            }
        }
        for col in &mut self.cols {
            for &(old, new) in mapping {
                if col.field.name == old {
                    let f = &col.field;
                    col.field = Arc::new(Field::new(
                        new,
                        f.dtype.clone(),
                        f.nullable,
                        if f.metadata.is_empty() {
                            None
                        } else {
                            Some(f.metadata.clone())
                        },
                    ));
                    break;
                }
            }
        }
        Ok(())
    }

    /// Returns the index of a column by name.
    pub fn col_name_index(&self, name: &str) -> Option<usize> {
        self.cols.iter().position(|fa| fa.field.name == name)
    }

    /// Resolve a named column to a `NumericArrayV`.
    #[cfg(feature = "views")]
    pub fn col_numeric(&self, name: &str) -> Result<NumericArrayV, MinarrowError> {
        let idx = self.col_name_index(name)
            .ok_or_else(|| MinarrowError::IndexError(format!("column '{}' not found", name)))?;
        let num = self.cols[idx].array.num_ref()?;
        Ok(NumericArrayV::from(num.clone()))
    }

    /// Resolve a named column to a `TextArrayV`.
    #[cfg(feature = "views")]
    pub fn col_text(&self, name: &str) -> Result<TextArrayV, MinarrowError> {
        let idx = self.col_name_index(name)
            .ok_or_else(|| MinarrowError::IndexError(format!("column '{}' not found", name)))?;
        let ta = self.cols[idx].array.str_ref()?;
        Ok(TextArrayV::from(ta.clone()))
    }

    /// Resolve a named column to a `BitmaskV`.
    #[cfg(feature = "views")]
    pub fn col_bitmask(&self, name: &str) -> Result<BitmaskV, MinarrowError> {
        let idx = self.col_name_index(name)
            .ok_or_else(|| MinarrowError::IndexError(format!("column '{}' not found", name)))?;
        let ba = self.cols[idx].array.bool_ref()?;
        Ok(BitmaskV::new(ba.data.clone(), 0, ba.len))
    }

    /// Removes a column by name.
    pub fn remove_col(&mut self, name: &str) -> bool {
        if let Some(idx) = self.col_name_index(name) {
            self.cols.remove(idx);
            self.recalc_n_rows();
            true
        } else {
            false
        }
    }

    /// Removes a column by index.
    pub fn remove_col_at(&mut self, idx: usize) -> bool {
        if idx < self.cols.len() {
            self.cols.remove(idx);
            self.recalc_n_rows();
            true
        } else {
            false
        }
    }

    /// Clears all columns and resets row count.
    pub fn clear(&mut self) {
        self.cols.clear();
        self.n_rows = 0;
    }

    /// Checks if a column with the given name exists.
    pub fn has_col(&self, name: &str) -> bool {
        self.col_name_index(name).is_some()
    }

    /// Returns all columns as a slice.
    pub fn cols(&self) -> &[FieldArray] {
        &self.cols
    }

    /// Returns mutable reference to all columns.
    pub fn cols_mut(&mut self) -> &mut [FieldArray] {
        &mut self.cols
    }

    // Keeps total rows cache up to date
    fn recalc_n_rows(&mut self) {
        if let Some(col) = self.cols.first() {
            self.n_rows = col.len();
        } else {
            self.n_rows = 0;
        }
    }

    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, FieldArray> {
        self.cols.iter()
    }
    #[inline]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, FieldArray> {
        self.cols.iter_mut()
    }

    #[inline]
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.n_rows()
    }

    /// Returns a new owned `Table` containing rows `[offset, offset+len)`.
    ///
    /// All columns are deeply copied, but only for the affected row(s).
    pub fn slice_clone(&self, offset: usize, len: usize) -> Self {
        assert!(offset <= self.n_rows, "offset out of bounds");
        assert!(offset + len <= self.n_rows, "slice window out of bounds");
        let cols: Vec<FieldArray> = self
            .cols
            .iter()
            .map(|fa| fa.slice_clone(offset, len))
            .collect();
        let name = format!("{}[{}, {})", self.name, offset, offset + len);
        #[allow(unused_mut)]
        let mut table = Table::build(cols, len, name);
        #[cfg(feature = "table_metadata")]
        {
            table.metadata = self.metadata.clone();
        }
        table
    }

    /// Returns a zero-copy view over rows `[offset, offset+len)`.
    /// This view borrows from the parent table and does not copy data.
    #[cfg(feature = "views")]
    pub fn slice(&self, offset: usize, len: usize) -> TableV {
        assert!(offset <= self.n_rows, "offset out of bounds");
        assert!(offset + len <= self.n_rows, "slice window out of bounds");
        TableV::from_table(self.clone(), offset, len)
    }

    /// Maps a function over a single column by name, returning the result.
    /// Returns None if the column doesn't exist.
    pub fn map_col<T, F>(&self, col_name: &str, func: F) -> Option<T>
    where
        F: FnOnce(&FieldArray) -> T,
    {
        self.cols
            .iter()
            .find(|c| c.field.name == col_name)
            .map(func)
    }

    /// Maps a function over multiple columns by name, returning a Vec of results.
    /// Warns if any requested columns are missing.
    pub fn map_cols_by_name<T, F>(&self, col_names: &[&str], mut func: F) -> Vec<T>
    where
        F: FnMut(&FieldArray) -> T,
    {
        let mut results = Vec::with_capacity(col_names.len());
        for name in col_names {
            match self.cols.iter().find(|c| c.field.name == *name) {
                Some(col) => results.push(func(col)),
                None => {
                    warn!(
                        "Column '{}' not found in table '{}'",
                        name, self.name
                    );
                }
            }
        }
        results
    }

    /// Maps a function over multiple columns by index, returning a Vec of results.
    /// Warns if any requested indices are out of bounds.
    pub fn map_cols_by_index<T, F>(&self, indices: &[usize], mut func: F) -> Vec<T>
    where
        F: FnMut(&FieldArray) -> T,
    {
        let mut results = Vec::with_capacity(indices.len());
        for &idx in indices {
            match self.cols.get(idx) {
                Some(col) => results.push(func(col)),
                None => {
                    warn!(
                        "Column index {} out of bounds in table '{}' (has {} columns)",
                        idx,
                        self.name,
                        self.n_cols()
                    );
                }
            }
        }
        results
    }

    /// Maps a function over all columns, returning a Vec of results.
    pub fn map_all_cols<T, F>(&self, func: F) -> Vec<T>
    where
        F: FnMut(&FieldArray) -> T,
    {
        self.cols.iter().map(func).collect()
    }

    /// Apply a transformation to each column, producing a new table.
    ///
    /// The closure receives each FieldArray and returns a transformed FieldArray.
    /// To pass a column through unchanged, clone it. The closure can dispatch
    /// on `Array` variant to handle numeric, text, temporal, and boolean columns
    /// differently.
    pub fn apply_cols<E>(
        &self,
        mut f: impl FnMut(&FieldArray) -> Result<FieldArray, E>,
    ) -> Result<Table, E> {
        let cols = self
            .cols
            .iter()
            .map(|fa| f(fa))
            .collect::<Result<Vec<_>, E>>()?;
        Ok(Table::new(self.name.clone(), Some(cols)))
    }

    /// Inserts rows from another table at the specified index.
    ///
    /// This is an **O(n)** operation where n is the number of rows after the insertion point.
    ///
    /// # Arguments
    /// * `index` - Position before which to insert (0 = prepend, n_rows = append)
    /// * `other` - Table to insert
    ///
    /// # Requirements
    /// - Both tables must have the same number of columns
    /// - Column names, types, and nullability must match in order
    /// - `index` must be <= `self.n_rows()`
    ///
    /// # Errors
    /// - `IndexError` if index > n_rows
    /// - `IncompatibleTypeError` if column schemas don't match
    pub fn insert_rows(&mut self, index: usize, other: &Self) -> Result<(), MinarrowError> {
        // Validate index
        if index > self.n_rows {
            return Err(MinarrowError::IndexError(format!(
                "Index {} out of bounds for table with {} rows",
                index, self.n_rows
            )));
        }

        // Check column count
        if self.n_cols() != other.n_cols() {
            return Err(MinarrowError::IncompatibleTypeError {
                from: "Table",
                to: "Table",
                message: Some(format!(
                    "Cannot insert tables with different column counts: {} vs {}",
                    self.n_cols(),
                    other.n_cols()
                )),
            });
        }

        // If both tables are empty, nothing to do
        if self.n_cols() == 0 {
            return Ok(());
        }

        // Validate column schemas and insert into each column
        for (col_idx, (self_col, other_col)) in
            self.cols.iter_mut().zip(other.cols.iter()).enumerate()
        {
            // Check field compatibility
            if self_col.field.name != other_col.field.name {
                return Err(MinarrowError::IncompatibleTypeError {
                    from: "Table",
                    to: "Table",
                    message: Some(format!(
                        "Column {} name mismatch: '{}' vs '{}'",
                        col_idx, self_col.field.name, other_col.field.name
                    )),
                });
            }

            if self_col.field.dtype != other_col.field.dtype {
                return Err(MinarrowError::IncompatibleTypeError {
                    from: "Table",
                    to: "Table",
                    message: Some(format!(
                        "Column '{}' type mismatch: {:?} vs {:?}",
                        self_col.field.name, self_col.field.dtype, other_col.field.dtype
                    )),
                });
            }

            if self_col.field.nullable != other_col.field.nullable {
                return Err(MinarrowError::IncompatibleTypeError {
                    from: "Table",
                    to: "Table",
                    message: Some(format!(
                        "Column '{}' nullable mismatch: {} vs {}",
                        self_col.field.name, self_col.field.nullable, other_col.field.nullable
                    )),
                });
            }

            // Insert into this column's array
            self_col.array.insert_rows(index, &other_col.array)?;

            // Update null count
            self_col.null_count = self_col.array.null_count();
        }

        // Update row count
        self.n_rows += other.n_rows;

        Ok(())
    }

    /// Splits the Table at the specified row index, consuming self and returning a SuperTable
    /// with two Table batches.
    ///
    /// Splits the underlying buffers, allocating new storage for the second half.
    #[cfg(feature = "chunked")]
    pub fn split(self, index: usize) -> Result<SuperTable, MinarrowError> {
        if index == 0 || index >= self.n_rows {
            return Err(MinarrowError::IndexError(format!(
                "Split index {} out of valid range (0, {})",
                index, self.n_rows
            )));
        }

        // Split each column
        let mut left_cols = Vec::with_capacity(self.cols.len());
        let mut right_cols = Vec::with_capacity(self.cols.len());

        for col in self.cols {
            let split_result = col.array.split(index, &col.field)?;
            let field = col.field.clone();

            // Extract the two arrays from the SuperArray
            let mut chunks = split_result.into_chunks();
            let right_array = chunks.pop().expect("split should produce 2 chunks");
            let left_array = chunks.pop().expect("split should produce 2 chunks");

            // Reconstruct FieldArrays with the original field
            let left_field = FieldArray {
                field: field.clone(),
                array: left_array,
                null_count: 0, // Will be recomputed if needed
            };
            let right_field = FieldArray {
                field,
                array: right_array,
                null_count: 0, // Will be recomputed if needed
            };

            left_cols.push(left_field);
            right_cols.push(right_field);
        }

        let left_table = Table::build(left_cols, index, format!("{}_left", self.name));
        let right_table = Table::build(
            right_cols,
            self.n_rows - index,
            format!("{}_right", self.name),
        );
        #[cfg(feature = "table_metadata")]
        let left_table = {
            let mut t = left_table;
            t.metadata = self.metadata.clone();
            t
        };
        #[cfg(feature = "table_metadata")]
        let right_table = {
            let mut t = right_table;
            t.metadata = self.metadata.clone();
            t
        };

        Ok(SuperTable::from_batches(
            vec![Arc::new(left_table), Arc::new(right_table)],
            Some(self.name),
        ))
    }
}

impl Table {
    #[cfg(feature = "parallel_proc")]
    #[inline]
    pub fn par_iter(&self) -> rayon::slice::Iter<'_, FieldArray> {
        self.cols.par_iter()
    }

    #[cfg(feature = "parallel_proc")]
    #[inline]
    pub fn par_iter_mut(&mut self) -> rayon::slice::IterMut<'_, FieldArray> {
        self.cols.par_iter_mut()
    }

    /// Export each column to arrow-rs `ArrayRef` and build a `RecordBatch`.
    ///
    /// The Arrow schema is derived from the imported array dtypes while
    /// preserving the original field names and nullability flags.
    ///
    /// Panics on FFI failure or empty table. For a fallible variant, see
    /// [`Table::try_to_apache_arrow`].
    #[cfg(feature = "cast_arrow")]
    #[inline]
    pub fn to_apache_arrow(&self) -> RecordBatch {
        self.try_to_apache_arrow()
            .expect("Table::to_apache_arrow failed")
    }

    /// Fallible variant of [`Table::to_apache_arrow`].
    #[cfg(feature = "cast_arrow")]
    pub fn try_to_apache_arrow(&self) -> Result<RecordBatch, MinarrowError> {
        use arrow::array::ArrayRef;
        if self.cols.is_empty() {
            return Err(MinarrowError::ShapeError {
                message: "cannot build RecordBatch from an empty Table".to_string(),
            });
        }

        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(self.cols.len());
        for col in &self.cols {
            arrays.push(col.try_to_apache_arrow()?);
        }

        let mut fields = Vec::with_capacity(self.cols.len());
        for (i, col) in self.cols.iter().enumerate() {
            let dt = arrays[i].data_type().clone();
            fields.push(arrow_schema::Field::new(
                col.field.name.clone(),
                dt,
                col.field.nullable,
            ));
        }
        let schema = Arc::new(arrow_schema::Schema::new(fields));

        Ok(RecordBatch::try_new(schema, arrays)?)
    }

    // ** The below polars function is tested tests/polars.rs **

    /// Casts the table to a Polars DataFrame.
    ///
    /// Panics on FFI failure. For a fallible variant, see
    /// [`Table::try_to_polars`].
    #[cfg(feature = "cast_polars")]
    pub fn to_polars(&self) -> DataFrame {
        self.try_to_polars().expect("Table::to_polars failed")
    }

    /// Fallible variant of [`Table::to_polars`].
    #[cfg(feature = "cast_polars")]
    pub fn try_to_polars(&self) -> Result<DataFrame, MinarrowError> {
        let mut cols = Vec::with_capacity(self.cols.len());
        for fa in &self.cols {
            let series = fa.try_to_polars()?;
            cols.push(Column::new(fa.field.name.clone().into(), series));
        }
        Ok(DataFrame::new(self.n_rows, cols)?)
    }

    // ===========================================================
    // Apache Arrow / Polars import (`from_*`)
    // ===========================================================

    /// Build a `Table` from an arrow-rs `RecordBatch`. Column names, dtypes,
    /// nullability, and any custom field metadata are recovered from the
    /// RecordBatch schema. The Table name will be empty; assign one via
    /// `Table::new` if needed.
    ///
    /// Panics on FFI failure. For a fallible variant, see
    /// [`Table::try_from_apache_arrow`].
    #[cfg(feature = "cast_arrow")]
    #[inline]
    pub fn from_apache_arrow(rb: &RecordBatch) -> Table {
        Self::try_from_apache_arrow(rb).expect("Table::from_apache_arrow failed")
    }

    /// Fallible variant of [`Table::from_apache_arrow`].
    #[cfg(feature = "cast_arrow")]
    pub fn try_from_apache_arrow(rb: &RecordBatch) -> Result<Table, MinarrowError> {
        let schema = rb.schema();
        let mut cols = Vec::with_capacity(rb.num_columns());
        for (i, col) in rb.columns().iter().enumerate() {
            let arr_field = schema.field(i);
            let fa = FieldArray::try_from_apache_arrow(arr_field.name(), col)?;
            cols.push(fa);
        }
        Ok(Table::new(String::new(), Some(cols)))
    }

    /// Build a `Table` from a Polars `DataFrame`. Column names, dtypes,
    /// nullability, and any custom Series metadata are recovered.
    ///
    /// Panics on FFI failure. For a fallible variant, see
    /// [`Table::try_from_polars`].
    #[cfg(feature = "cast_polars")]
    #[inline]
    pub fn from_polars(df: &DataFrame) -> Table {
        Self::try_from_polars(df).expect("Table::from_polars failed")
    }

    /// Fallible variant of [`Table::from_polars`].
    #[cfg(feature = "cast_polars")]
    pub fn try_from_polars(df: &DataFrame) -> Result<Table, MinarrowError> {
        let columns = df.columns();
        let mut cols = Vec::with_capacity(columns.len());
        for column in columns {
            // Polars Column owns its data either as a Series or scalar; materialise
            // into a single chunked Series so we can hit the polars_arrow FFI path.
            let series = column.as_materialized_series();
            let fa = FieldArray::try_from_polars(series)?;
            cols.push(fa);
        }
        Ok(Table::new(String::new(), Some(cols)))
    }
}

#[cfg(feature = "cast_arrow")]
impl From<&RecordBatch> for Table {
    fn from(rb: &RecordBatch) -> Self {
        Table::from_apache_arrow(rb)
    }
}

#[cfg(feature = "cast_polars")]
impl From<&DataFrame> for Table {
    fn from(df: &DataFrame) -> Self {
        Table::from_polars(df)
    }
}

impl<'a> IntoIterator for &'a Table {
    type Item = &'a FieldArray;
    type IntoIter = std::slice::Iter<'a, FieldArray>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.cols.iter()
    }
}

impl<'a> IntoIterator for &'a mut Table {
    type Item = &'a mut FieldArray;
    type IntoIter = std::slice::IterMut<'a, FieldArray>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.cols.iter_mut()
    }
}

impl IntoIterator for Table {
    type Item = FieldArray;
    type IntoIter = <Vec<FieldArray> as IntoIterator>::IntoIter;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.cols.into_iter()
    }
}

impl Shape for Table {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank2 {
            rows: self.n_rows(),
            cols: self.n_cols(),
        }
    }
}

impl Concatenate for Table {
    /// Concatenates two tables vertically (row-wise).
    ///
    /// # Requirements
    /// - Both tables must have the same number of columns
    /// - Column names, types, and nullability must match in order
    ///
    /// # Returns
    /// A new Table with rows from `self` followed by rows from `other`
    ///
    /// # Errors
    /// - `IncompatibleTypeError` if column schemas don't match
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        // Check column count
        if self.n_cols() != other.n_cols() {
            return Err(MinarrowError::IncompatibleTypeError {
                from: "Table",
                to: "Table",
                message: Some(format!(
                    "Cannot concatenate tables with different column counts: {} vs {}",
                    self.n_cols(),
                    other.n_cols()
                )),
            });
        }

        // If both tables are empty, return empty table
        if self.n_cols() == 0 {
            return Ok(Table::new(format!("{}+{}", self.name, other.name), None));
        }

        // Validate column schemas match and concatenate arrays
        let mut result_cols = Vec::with_capacity(self.n_cols());

        for (col_idx, (self_col, other_col)) in self
            .cols
            .into_iter()
            .zip(other.cols.into_iter())
            .enumerate()
        {
            // Check field compatibility
            if self_col.field.name != other_col.field.name {
                return Err(MinarrowError::IncompatibleTypeError {
                    from: "Table",
                    to: "Table",
                    message: Some(format!(
                        "Column {} name mismatch: '{}' vs '{}'",
                        col_idx, self_col.field.name, other_col.field.name
                    )),
                });
            }

            if self_col.field.dtype != other_col.field.dtype {
                return Err(MinarrowError::IncompatibleTypeError {
                    from: "Table",
                    to: "Table",
                    message: Some(format!(
                        "Column '{}' type mismatch: {:?} vs {:?}",
                        self_col.field.name, self_col.field.dtype, other_col.field.dtype
                    )),
                });
            }

            if self_col.field.nullable != other_col.field.nullable {
                return Err(MinarrowError::IncompatibleTypeError {
                    from: "Table",
                    to: "Table",
                    message: Some(format!(
                        "Column '{}' nullable mismatch: {} vs {}",
                        self_col.field.name, self_col.field.nullable, other_col.field.nullable
                    )),
                });
            }

            // Concatenate arrays
            let concatenated_array = self_col.array.concat(other_col.array)?;
            let null_count = concatenated_array.null_count();

            // Create new FieldArray with concatenated data
            result_cols.push(FieldArray {
                field: self_col.field.clone(),
                array: concatenated_array,
                null_count,
            });
        }

        // Create result table
        let n_rows = result_cols.first().map(|c| c.len()).unwrap_or(0);
        let name = format!("{}+{}", self.name, other.name);
        let table = Table::build(result_cols, n_rows, name);
        #[cfg(feature = "table_metadata")]
        let table = {
            let mut t = table;
            t.metadata = self.metadata;
            t
        };

        Ok(table)
    }
}

#[cfg(feature = "chunked")]
impl Consolidate for Vec<Table> {
    type Output = Table;

    /// Consolidate a vector of tables with the same schema into a single table.
    ///
    /// Returns an empty table if the input is empty. For a single table,
    /// returns it directly without copying.
    ///
    /// When the `arena` feature is enabled, all column buffers are written
    /// into a single allocation then sliced into typed views, reducing
    /// allocation count from O(columns) to O(1). The resulting buffers
    /// are SharedBuffer-backed; mutations trigger copy-on-write.
    ///
    /// Without the `arena` feature, falls back to per-column concat.
    fn consolidate(self) -> Table {
        if self.is_empty() {
            return Table::new_empty();
        }
        if self.len() == 1 {
            return self.into_iter().next().unwrap();
        }

        #[cfg(feature = "arena")]
        {
            let name = self[0].name.clone();
            let refs: Vec<&Table> = self.iter().collect();
            crate::structs::arena::consolidate_tables_arena(&refs, name)
        }
        #[cfg(not(feature = "arena"))]
        {
            consolidate_vec_concat(self)
        }
    }
}

#[cfg(feature = "chunked")]
#[cfg(not(feature = "arena"))]
fn consolidate_vec_concat(tables: Vec<Table>) -> Table {
    let n_cols = tables[0].cols.len();
    let mut unified_cols = Vec::with_capacity(n_cols);

    for col_idx in 0..n_cols {
        let field = tables[0].cols[col_idx].field.clone();
        let mut arr = tables[0].cols[col_idx].array.clone();
        for table in tables.iter().skip(1) {
            arr.concat_array(&table.cols[col_idx].array);
        }
        let null_count = arr.null_count();
        unified_cols.push(FieldArray {
            field,
            array: arr,
            null_count,
        });
    }

    let n_rows = unified_cols.first().map(|c| c.len()).unwrap_or(0);
    let name = tables[0].name.clone();
    let table = Table::build(unified_cols, n_rows, name);
    #[cfg(feature = "table_metadata")]
    {
        let mut t = table;
        t.metadata = tables[0].metadata.clone();
        t
    }
    #[cfg(not(feature = "table_metadata"))]
    table
}

impl Display for Table {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.cols.is_empty() {
            return writeln!(f, "Table  \"{}\" [0 rows × 0 cols] – empty", self.name);
        }

        // Gather column metadata & cell strings (with null handling)
        let row_indices: Vec<usize> = if self.n_rows <= MAX_PREVIEW {
            (0..self.n_rows).collect()
        } else {
            let mut idx = (0..10).collect::<Vec<_>>();
            idx.extend((self.n_rows - 10)..self.n_rows);
            idx
        };

        // column header strings and tracked widths
        let mut headers: Vec<String> = Vec::with_capacity(self.cols.len());
        let mut widths: Vec<usize> = Vec::with_capacity(self.cols.len());

        for col in &self.cols {
            let hdr = format!("{}:{:?}", col.field.name, col.field.dtype);
            widths.push(hdr.len());
            headers.push(hdr);
        }

        // matrix of cell strings
        let mut rows: Vec<Vec<String>> = Vec::with_capacity(row_indices.len());

        for &row_idx in &row_indices {
            let mut row: Vec<String> = Vec::with_capacity(self.cols.len());

            for (col_idx, col) in self.cols.iter().enumerate() {
                let val = value_to_string(&col.array, row_idx);
                widths[col_idx] = widths[col_idx].max(val.len());
                row.push(val);
            }
            rows.push(row);
        }

        // row-index column (“idx”)
        let idx_width = usize::max(
            3, // “idx”
            ((self.n_rows - 1) as f64).log10().floor() as usize + 1,
        );

        // Render header
        writeln!(
            f,
            "Table \"{}\" [{} rows × {} cols]",
            self.name,
            self.n_rows,
            self.cols.len()
        )?;
        print_rule(f, idx_width, &widths)?;
        print_header_row(f, idx_width, &headers, &widths)?;
        print_rule(f, idx_width, &widths)?;

        // Render body
        for (logical_row, cells) in rows.iter().enumerate() {
            let physical_row = row_indices[logical_row];
            write!(f, "| {idx:>w$} |", idx = physical_row, w = idx_width)?;
            for (col_idx, cell) in cells.iter().enumerate() {
                write!(f, " {val:^w$} |", val = cell, w = widths[col_idx])?;
            }
            writeln!(f)?;
            if logical_row == 9 && self.n_rows > MAX_PREVIEW {
                print_ellipsis_row(f, idx_width, &widths)?;
            }
        }
        print_rule(f, idx_width, &widths)
    }
}

// ===== Selection Trait Implementations =====

#[cfg(all(feature = "views", feature = "select"))]
impl ColumnSelection for Table {
    type View = TableV;
    type ColumnView = ArrayV;
    type ColumnOwned = FieldArray;

    fn c<S: FieldSelector>(&self, selection: S) -> TableV {
        let all_fields: Vec<Arc<Field>> = self.cols.iter().map(|fa| fa.field.clone()).collect();
        let col_indices = selection.resolve_fields(&all_fields);

        // If selecting all columns, create a full view without filtering
        if col_indices.len() == all_fields.len() {
            return TableV {
                name: self.name.clone(),
                fields: all_fields,
                cols: self.cols.iter().map(|fa| ArrayV::from(fa.clone())).collect(),
                offset: 0,
                len: self.n_rows,
                active_col_selection: None,
            };
        }

        // Create a view with only the selected columns
        let selected_fields: Vec<Arc<Field>> = col_indices
            .iter()
            .filter_map(|&i| self.cols.get(i).map(|fa| fa.field.clone()))
            .collect();
        let selected_cols: Vec<ArrayV> = col_indices
            .iter()
            .filter_map(|&i| self.cols.get(i).map(|fa| ArrayV::from(fa.clone())))
            .collect();

        TableV {
            name: self.name.clone(),
            fields: selected_fields,
            cols: selected_cols,
            offset: 0,
            len: self.n_rows,
            active_col_selection: None,
        }
    }

    fn get(&self, field: &str) -> Option<FieldArray> {
        self.col_name_index(field).map(|idx| self.cols[idx].clone())
    }

    fn col_ix(&self, idx: usize) -> Option<ArrayV> {
        self.cols.get(idx).map(|fa| ArrayV::from(fa.clone()))
    }

    fn col_vec(&self) -> Vec<ArrayV> {
        self.cols
            .iter()
            .map(|fa| ArrayV::from(fa.clone()))
            .collect()
    }

    fn get_cols(&self) -> Vec<Arc<Field>> {
        self.cols.iter().map(|fa| fa.field.clone()).collect()
    }
}

#[cfg(all(feature = "views", feature = "select"))]
impl RowSelection for Table {
    type View = TableV;

    fn r<S: DataSelector>(&self, selection: S) -> TableV {
        if selection.is_contiguous() {
            // Contiguous selection (ranges): create a properly windowed view
            let indices = selection.resolve_indices(self.n_rows);
            if indices.is_empty() {
                return TableV::from_table(self.clone(), 0, 0);
            }
            let new_offset = indices[0];
            let new_len = indices.len();
            TableV::from_table(self.clone(), new_offset, new_len)
        } else {
            // Non-contiguous selection (index arrays): materialise
            let indices = selection.resolve_indices(self.n_rows);
            let table_v = TableV::from(self.clone());
            let materialised_table = table_v.gather_rows(&indices);
            TableV::from(materialised_table)
        }
    }

    fn get_row_count(&self) -> usize {
        self.n_rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structs::field_array::field_array;
    use crate::traits::masked_array::MaskedArray;
    use crate::{fa_bool, fa_i32, fa_i64, fa_u32};
    #[cfg(all(feature = "views", feature = "select"))]
    use crate::traits::selection::ColumnSelection;
    use crate::{Array, BooleanArray, IntegerArray, NumericArray};

    #[test]
    fn test_new_table() {
        let t = Table::new_empty();
        assert_eq!(t.n_cols(), 0);
        assert_eq!(t.n_rows(), 0);
        assert!(t.is_empty());
    }

    #[test]
    fn test_add_and_get_columns() {
        let mut t = Table::new_empty();
        t.add_col(fa_i32!("ints", 1, 2));
        t.add_col(fa_bool!("bools", true, false));

        assert_eq!(t.n_cols(), 2);
        assert_eq!(t.n_rows(), 2);
        assert!(!t.is_empty());

        // Test column access via cols()
        assert!(t.cols().get(0).is_some());
        assert!(t.cols().get(1).is_some());
        assert!(t.cols().get(2).is_none());
        assert_eq!(t.col_names(), vec!["ints", "bools"]);

        // Test column by name via col_name_index
        let idx = t.col_name_index("ints").unwrap();
        let col = t.cols().get(idx).unwrap();
        match &col.array {
            Array::NumericArray(NumericArray::Int32(a)) => assert_eq!(a.len(), 2),
            _ => panic!("ints column type mismatch"),
        }
    }

    #[cfg(all(feature = "views", feature = "select"))]
    #[test]
    fn test_column_selection_trait() {
        let mut t = Table::new_empty();
        t.add_col(fa_i32!("ints", 1, 2));
        t.add_col(fa_bool!("bools", true, false));

        // Test ColumnSelection trait methods
        assert!(t.col_ix(0).is_some());
        assert!(t.col_ix(1).is_some());
        assert!(t.col_ix(2).is_none());

        // col() returns TableV, col_ix(0) gets the single column as ArrayV
        let col_view = t.col("ints");
        assert_eq!(col_view.cols.len(), 1); // Column found
        let av = col_view.col_ix(0).unwrap();
        assert_eq!(col_view.fields[0].name, "ints");
        match &av.array {
            Array::NumericArray(NumericArray::Int32(a)) => assert_eq!(a.len(), 2),
            _ => panic!("ints column type mismatch"),
        }
    }

    #[test]
    #[should_panic(expected = "Column length mismatch")]
    fn test_column_length_mismatch_panics() {
        let mut t = Table::new_empty();
        t.add_col(fa_i32!("ints", 1, 2, 3));
        // This should panic due to mismatched row count
        t.add_col(fa_bool!("bools", true, false));
    }

    #[test]
    fn test_column_index_and_has_column() {
        let mut t = Table::new_empty();
        t.add_col(fa_i64!("foo"));
        assert_eq!(t.col_name_index("foo"), Some(0));
        assert_eq!(t.col_name_index("bar"), None);
        assert!(t.has_col("foo"));
        assert!(!t.has_col("bar"));
    }

    #[test]
    fn test_remove_column_by_name_and_index() {
        let mut t = Table::new_empty();
        t.add_col(fa_u32!("a", 10, 20));
        t.add_col(fa_bool!("b", true, false));

        assert!(t.remove_col("a"));
        assert!(!t.has_col("a"));
        assert_eq!(t.n_cols(), 1);

        assert!(t.remove_col_at(0));
        assert_eq!(t.n_cols(), 0);
        assert_eq!(t.n_rows(), 0);

        // Removing non-existent column
        assert!(!t.remove_col("not_there"));
        assert!(!t.remove_col_at(5));
    }

    #[test]
    fn test_clear() {
        let mut t = Table::new_empty();
        t.add_col(fa_i32!("x", 42));
        assert!(!t.is_empty());
        t.clear();
        assert!(t.is_empty());
        assert_eq!(t.n_cols(), 0);
        assert_eq!(t.n_rows(), 0);
    }

    #[test]
    fn test_columns() {
        let mut t = Table::new_empty();
        t.add_col(fa_i32!("c", 7));
        {
            let cols = t.cols();
            assert_eq!(cols.len(), 1);
        }
    }

    #[test]
    fn test_table_iter() {
        let mut t = Table::new_empty();
        t.add_col(fa_i32!("a", 1));
        t.add_col(fa_bool!("b", true));

        let names: Vec<_> = t.iter().map(|fa| fa.field.name.as_str()).collect();
        assert_eq!(names, ["a", "b"]);

        let names2: Vec<_> = (&t).into_iter().map(|fa| fa.field.name.as_str()).collect();
        assert_eq!(names2, ["a", "b"]);
    }

    #[cfg(feature = "views")]
    #[test]
    fn test_table_slice_and_slice() {
        let mut t = Table::new("foo".into(), None);
        t.add_col(fa_i32!("ints", 1, 2, 3));
        t.add_col(fa_bool!("bools", true, false, true));

        let sliced = t.slice_clone(1, 2);
        assert_eq!(sliced.n_rows(), 2);
        // Access column by name index
        let idx = sliced.col_name_index("ints").unwrap();
        assert_eq!(sliced.cols().get(idx).unwrap().array.len(), 2);

        let view = t.slice(1, 2);
        assert_eq!(view.n_rows(), 2);
        // TableV is a zero-copy view - underlying array still has full length
        // The view's logical length is accessed via n_rows()
        assert!(view.col_name_index("bools").is_some());

        // // Zero-copy: view.table == &t via the underlying arrays
        // for (orig, sliced) in t.cols.iter().zip(view.cols.iter()) {
        //     use std::sync::Arc;

        //     assert!(Arc::ptr_eq(&orig.field, &sliced.field), "FieldArc pointer mismatch");
        // }
    }

    #[test]
    fn test_map_cols_by_name() {
        let mut t = Table::new_empty();
        t.add_col(fa_i32!("a", 1, 2));
        t.add_col(fa_i32!("b", 3, 4));

        // Test with all valid names
        let results = t.map_cols_by_name(&["a", "b"], |fa| fa.field.name.clone());
        assert_eq!(results, vec!["a", "b"]);

        // Test with missing column (will warn but skip)
        let results = t.map_cols_by_name(&["a", "missing", "b"], |fa| fa.field.name.clone());
        assert_eq!(results, vec!["a", "b"]);
    }

    #[test]
    fn test_map_cols_by_index() {
        let mut t = Table::new_empty();
        t.add_col(fa_i32!("a", 1, 2));
        t.add_col(fa_i32!("b", 3, 4));

        // Test with all valid indices
        let results = t.map_cols_by_index(&[0, 1], |fa| fa.field.name.clone());
        assert_eq!(results, vec!["a", "b"]);

        // Test with out-of-bounds index (will warn but skip)
        let results = t.map_cols_by_index(&[0, 5, 1], |fa| fa.field.name.clone());
        assert_eq!(results, vec!["a", "b"]);
    }

    #[test]
    fn test_table_insert_rows_prepend() {
        let mut t1 = Table::new_empty();
        t1.add_col(fa_i32!("a", 1, 2));
        t1.add_col(fa_i32!("b", 10, 20));

        let mut t2 = Table::new_empty();
        t2.add_col(fa_i32!("a", 99));
        t2.add_col(fa_i32!("b", 88));

        t1.insert_rows(0, &t2).unwrap();

        assert_eq!(t1.n_rows(), 3);
        match &t1.cols[0].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.data.as_slice(), &[99, 1, 2]);
            }
            _ => panic!("wrong type"),
        }
        match &t1.cols[1].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.data.as_slice(), &[88, 10, 20]);
            }
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn test_table_insert_rows_middle() {
        let mut t1 = Table::new_empty();
        t1.add_col(fa_i32!("a", 1, 2, 3));
        t1.add_col(fa_i32!("b", 10, 20, 30));

        let mut t2 = Table::new_empty();
        t2.add_col(fa_i32!("a", 99, 88));
        t2.add_col(fa_i32!("b", 77, 66));

        t1.insert_rows(1, &t2).unwrap();

        assert_eq!(t1.n_rows(), 5);
        match &t1.cols[0].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.data.as_slice(), &[1, 99, 88, 2, 3]);
            }
            _ => panic!("wrong type"),
        }
        match &t1.cols[1].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.data.as_slice(), &[10, 77, 66, 20, 30]);
            }
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn test_table_insert_rows_append() {
        let mut t1 = Table::new_empty();
        t1.add_col(fa_i32!("a", 1, 2));

        let mut t2 = Table::new_empty();
        t2.add_col(fa_i32!("a", 3, 4));

        t1.insert_rows(2, &t2).unwrap();

        assert_eq!(t1.n_rows(), 4);
        match &t1.cols[0].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.data.as_slice(), &[1, 2, 3, 4]);
            }
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn test_table_insert_rows_schema_mismatch() {
        let mut t1 = Table::new_empty();
        t1.add_col(fa_i32!("a"));

        let mut t2 = Table::new_empty();
        t2.add_col(fa_i32!("b"));

        let result = t1.insert_rows(0, &t2);
        assert!(result.is_err());
    }

    #[test]
    fn test_table_insert_rows_out_of_bounds() {
        let mut t1 = Table::new_empty();
        t1.add_col(fa_i32!("a", 1));

        let t2 = Table::new_empty();
        let result = t1.insert_rows(10, &t2);
        assert!(result.is_err());
    }

    #[cfg(feature = "chunked")]
    #[test]
    fn test_table_split_basic() {
        let mut t = Table::new_empty();
        t.add_col(fa_i32!("a", 1, 2, 3, 4));
        t.add_col(fa_i32!("b", 10, 20, 30, 40));

        let super_table = t.split(2).unwrap();

        assert_eq!(super_table.n_batches(), 2);
        assert_eq!(super_table.batches[0].n_rows(), 2);
        assert_eq!(super_table.batches[1].n_rows(), 2);

        match &super_table.batches[0].cols[0].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.data.as_slice(), &[1, 2]);
            }
            _ => panic!("wrong type"),
        }

        match &super_table.batches[1].cols[0].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.data.as_slice(), &[3, 4]);
            }
            _ => panic!("wrong type"),
        }
    }

    #[cfg(feature = "chunked")]
    #[test]
    fn test_table_split_invalid_index() {
        let mut t1 = Table::new_empty();
        t1.add_col(fa_i32!("a", 1, 2));
        assert!(t1.split(0).is_err());

        let mut t2 = Table::new_empty();
        t2.add_col(fa_i32!("a", 1, 2));
        assert!(t2.split(2).is_err());

        let mut t3 = Table::new_empty();
        t3.add_col(fa_i32!("a", 1, 2));
        assert!(t3.split(10).is_err());
    }

    #[cfg(all(feature = "views", feature = "select"))]
    #[test]
    fn test_row_selection_to_table_column_lengths() {
        use crate::traits::selection::RowSelection;

        let mut ids = IntegerArray::<i32>::default();
        let mut flags = BooleanArray::default();
        for i in 0..10 {
            ids.push(i + 1);
            flags.push(i % 2 == 0);
        }

        let mut t = Table::new_empty();
        t.add_col(field_array("ids", Array::from_int32(ids)));
        t.add_col(field_array("flags", Array::from_bool(flags)));
        assert_eq!(t.n_rows(), 10);

        // Contiguous range selection via r()
        let result = t.r(0..5).to_table();
        assert_eq!(result.n_rows(), 5);
        for col in &result.cols {
            assert_eq!(
                col.array.len(),
                5,
                "Column '{}' has {} elements after r(0..5), expected 5",
                col.field.name,
                col.array.len()
            );
        }

        // Offset range
        let result = t.r(3..7).to_table();
        assert_eq!(result.n_rows(), 4);
        for col in &result.cols {
            assert_eq!(col.array.len(), 4);
        }
        // Verify values: ids should be [4, 5, 6, 7]
        match &result.cols[0].array {
            Array::NumericArray(NumericArray::Int32(a)) => {
                let vals: Vec<i32> = (0..a.len()).map(|i| a.get(i).unwrap()).collect();
                assert_eq!(vals, vec![4, 5, 6, 7]);
            }
            _ => panic!("unexpected type"),
        }

        // Equivalence with slice
        let via_r = t.r(2..8).to_table();
        let via_slice = t.slice(2, 6).to_table();
        assert_eq!(via_r.n_rows(), via_slice.n_rows());
        for (r_col, s_col) in via_r.cols.iter().zip(via_slice.cols.iter()) {
            assert_eq!(r_col.array.len(), s_col.array.len());
        }

        // Empty selection
        let result = t.r(0..0).to_table();
        assert_eq!(result.n_rows(), 0);
        for col in &result.cols {
            assert_eq!(col.array.len(), 0);
        }
    }

    // --- Table::from_arena tests ---

    #[cfg(feature = "arena")]
    mod arena_tests {
        use crate::Bitmask;
        use crate::ffi::arrow_dtype::ArrowType;
        use crate::structs::arena::{AAMaker, Arena};
        use crate::structs::field::Field;
        use crate::structs::table::Table;
        use crate::traits::masked_array::MaskedArray;
        use std::sync::Arc;

        #[test]
        fn test_from_arena_integer_and_float() {
            let ids: Vec<i32> = vec![10, 20, 30];
            let prices: Vec<f64> = vec![1.5, 2.5, 3.5];

            let mut arena = Arena::with_capacity(4096);
            let r_ids = arena.push_slice(&ids);
            let r_prices = arena.push_slice(&prices);

            let schema = vec![
                Arc::new(Field::new("id", ArrowType::Int32, false, None)),
                Arc::new(Field::new("price", ArrowType::Float64, false, None)),
            ];
            let regions = vec![
                AAMaker::Primitive {
                    data: r_ids,
                    mask: None,
                },
                AAMaker::Primitive {
                    data: r_prices,
                    mask: None,
                },
            ];

            let table = Table::from_arena("test".into(), &schema, arena, regions, 3);
            assert_eq!(table.n_rows(), 3);
            assert_eq!(table.n_cols(), 2);
            assert_eq!(table.cols[0].field.name, "id");
            assert_eq!(table.cols[1].field.name, "price");

            // Verify values
            if let crate::Array::NumericArray(crate::NumericArray::Int32(a)) = &table.cols[0].array
            {
                assert_eq!(a.get(0), Some(10));
                assert_eq!(a.get(2), Some(30));
            } else {
                panic!("Expected Int32 array");
            }

            if let crate::Array::NumericArray(crate::NumericArray::Float64(a)) =
                &table.cols[1].array
            {
                assert_eq!(a.get(0), Some(1.5));
                assert_eq!(a.get(2), Some(3.5));
            } else {
                panic!("Expected Float64 array");
            }
        }

        #[test]
        fn test_from_arena_string_columns() {
            let strings = ["hello", "world", "foo"];
            let mut offsets: Vec<u32> = Vec::with_capacity(4);
            let mut data: Vec<u8> = Vec::new();
            offsets.push(0);
            for s in &strings {
                data.extend_from_slice(s.as_bytes());
                offsets.push(data.len() as u32);
            }

            let mut arena = Arena::with_capacity(4096);
            let r_offsets = arena.push_slice(&offsets);
            let r_data = arena.push_slice(&data);

            let schema = vec![Arc::new(Field::new("text", ArrowType::String, true, None))];
            let regions = vec![AAMaker::String {
                offsets: r_offsets,
                data: r_data,
                mask: None,
            }];

            let table = Table::from_arena("str_test".into(), &schema, arena, regions, 3);
            assert_eq!(table.n_rows(), 3);

            if let crate::Array::TextArray(
                crate::enums::collections::text_array::TextArray::String32(a),
            ) = &table.cols[0].array
            {
                assert_eq!(a.get_str(0), Some("hello"));
                assert_eq!(a.get_str(1), Some("world"));
                assert_eq!(a.get_str(2), Some("foo"));
            } else {
                panic!("Expected String32 array");
            }
        }

        #[test]
        fn test_from_arena_nullable_columns() {
            let values: Vec<i64> = vec![100, 200, 300, 400];
            let mut mask = Bitmask::new_set_all(4, true);
            mask.set(1, false); // second value is null
            mask.set(3, false); // fourth value is null

            let mut arena = Arena::with_capacity(4096);
            let r_data = arena.push_slice(&values);
            let r_mask = arena.push_bitmask(&mask);

            let schema = vec![Arc::new(Field::new("vals", ArrowType::Int64, true, None))];
            let regions = vec![AAMaker::Primitive {
                data: r_data,
                mask: Some(r_mask),
            }];

            let table = Table::from_arena("nullable".into(), &schema, arena, regions, 4);
            assert_eq!(table.n_rows(), 4);
            assert_eq!(table.cols[0].null_count, 2);

            if let crate::Array::NumericArray(crate::NumericArray::Int64(a)) = &table.cols[0].array
            {
                assert_eq!(a.get(0), Some(100));
                assert_eq!(a.get(1), None);
                assert_eq!(a.get(2), Some(300));
                assert_eq!(a.get(3), None);
            } else {
                panic!("Expected Int64 array");
            }
        }

        #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
        #[test]
        fn test_from_arena_boolean_and_categorical() {
            use crate::ffi::arrow_dtype::CategoricalIndexType;
            use vec64::Vec64;

            // Boolean column: true, false, true
            let mut bool_data = Bitmask::new_set_all(3, true);
            bool_data.set(1, false);

            let mut arena = Arena::with_capacity(4096);
            let r_bool = arena.push_bitmask(&bool_data);

            // Categorical column
            let indices: Vec<u32> = vec![0, 1, 0];
            let r_cat_idx = arena.push_slice(&indices);

            let mut unique = Vec64::new();
            unique.push("cat_a".to_string());
            unique.push("cat_b".to_string());

            let schema = vec![
                Arc::new(Field::new("flag", ArrowType::Boolean, false, None)),
                Arc::new(Field::new(
                    "category",
                    ArrowType::Dictionary(CategoricalIndexType::UInt32),
                    false,
                    None,
                )),
            ];
            let regions = vec![
                AAMaker::Boolean {
                    data: r_bool,
                    mask: None,
                },
                AAMaker::Categorical {
                    indices: r_cat_idx,
                    mask: None,
                    unique_values: unique,
                },
            ];

            let table = Table::from_arena("mixed".into(), &schema, arena, regions, 3);
            assert_eq!(table.n_rows(), 3);
            assert_eq!(table.n_cols(), 2);

            if let crate::Array::BooleanArray(a) = &table.cols[0].array {
                assert_eq!(a.get(0), Some(true));
                assert_eq!(a.get(1), Some(false));
                assert_eq!(a.get(2), Some(true));
            } else {
                panic!("Expected BooleanArray");
            }

            if let crate::Array::TextArray(
                crate::enums::collections::text_array::TextArray::Categorical32(a),
            ) = &table.cols[1].array
            {
                assert_eq!(a.get_str(0), Some("cat_a"));
                assert_eq!(a.get_str(1), Some("cat_b"));
                assert_eq!(a.get_str(2), Some("cat_a"));
            } else {
                panic!("Expected Categorical32 array");
            }
        }

        #[test]
        fn test_from_arena_shared_buffer_backed() {
            let col1: Vec<i32> = vec![1, 2, 3];
            let col2: Vec<f64> = vec![4.0, 5.0, 6.0];

            let mut arena = Arena::with_capacity(4096);
            let r1 = arena.push_slice(&col1);
            let r2 = arena.push_slice(&col2);

            let schema = vec![
                Arc::new(Field::new("a", ArrowType::Int32, false, None)),
                Arc::new(Field::new("b", ArrowType::Float64, false, None)),
            ];
            let regions = vec![
                AAMaker::Primitive {
                    data: r1,
                    mask: None,
                },
                AAMaker::Primitive {
                    data: r2,
                    mask: None,
                },
            ];

            let table = Table::from_arena("shared".into(), &schema, arena, regions, 3);

            // Verify all buffers are SharedBuffer-backed
            if let crate::Array::NumericArray(crate::NumericArray::Int32(a)) = &table.cols[0].array
            {
                assert!(a.data.is_shared());
            } else {
                panic!("Expected Int32");
            }
            if let crate::Array::NumericArray(crate::NumericArray::Float64(a)) =
                &table.cols[1].array
            {
                assert!(a.data.is_shared());
            } else {
                panic!("Expected Float64");
            }
        }
    }
}

#[cfg(test)]
#[cfg(feature = "parallel_proc")]
mod parallel_column_tests {
    use rayon::prelude::*;

    use super::*;
    use crate::{fa_bool, fa_i32};

    #[test]
    fn test_table_par_iter_column_names() {
        let mut table = Table::new_empty();
        table.add_col(fa_i32!("id", 1));
        table.add_col(fa_bool!("flag", true));

        let mut names: Vec<&str> = table.par_iter().map(|fa| fa.field.name.as_str()).collect();
        names.sort_unstable(); // Ensure deterministic order for assert
        assert_eq!(names, vec!["flag", "id"]);
    }
}
