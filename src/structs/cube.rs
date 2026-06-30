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

//! # **Cube Module** - *3D Type for Advanced Analysis*
//!
//! An optional collection type that groups multiple [`Table`]s into a logical 3rd dimension
//! (e.g., **time snapshots** or a **category** axis). Gated behind the `cube` feature.
//!
//! ## Purpose
//! - Compare tables at the same schema/grain without aggregating away detail.
//! - Organise sequences of tables for sliding windows, diffs, or rollups.
//!
//! ## Behaviour
//! - All tables must share the **same schema** (names and Arrow dtypes).
//! - `n_rows` is tracked per table; helpers to add/remove tables/columns.
//! - Zero-copy windowing via views when `views` feature is enabled.
//! - Optional parallel iteration with `parallel_proc`.
//! - Despite the name, rows do not have to be equal between each table
//!
//! ## Interop
//! - Uses the project’s [`Table`], [`FieldArray`], and (optionally) [`TableV`] / `CubeV`.
//! - Arrow interop is inherited from the underlying arrays/fields.
//!
//! ## Status
//! Feature-gated and **WIP/unstable**. APIs may evolve.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "parallel_proc")]
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator};

use super::field_array::FieldArray;
#[cfg(feature = "views")]
use crate::TableV;
#[cfg(feature = "views")]
use crate::aliases::CubeV;
use crate::enums::{error::MinarrowError, shape_dim::ShapeDim};
use crate::ffi::arrow_dtype::ArrowType;
use crate::traits::{concatenate::Concatenate, shape::Shape};
use crate::{Field, Table};

// Global counter for unnamed cube instances
static UNNAMED_COUNTER: AtomicUsize = AtomicUsize::new(1);

/// # Cube - 3D Type for Advanced Analysis Use Cases
///
/// Holds a vector of tables unified by some value, often `Time`,
/// for special indexing. Useful for data analysis.
///
/// ## Purpose
/// Useful when the tables represent discrete time snapshots,
/// or a category dimension. This enables comparing data without losing
/// the underlying grain through aggregation, whilst still supporting that.
///
/// ## Description
/// **This is an optional extra enabled by the `cube` feature,
/// and is not part of the *`Apache Arrow`* framework**.
///
/// ### Under Development
/// ⚠️ **Unstable API and WIP: expect future development. Breaking changes will be minimised,
/// but avoid using this in production unless you are ready to wear API adjustments**.
#[repr(C, align(64))]
#[derive(Default, PartialEq, Clone, Debug)]
pub struct Cube {
    /// Table entries forming the cube (rectangular prism)
    pub tables: Vec<Arc<Table>>,

    /// Cube name
    pub name: String,
    // Third-dimensional index column names
    // It's a vec, as there are cases where one will
    // want to compound the index using time.
    pub third_dim_index: Option<Vec<String>>,
    // O(1) lookup from group key to position in the tables vec.
    // Maintained alongside tables for fast group resolution across batches.
    pub resolver: HashMap<String, usize>,
}

impl Cube {
    /// Constructs a new Cube with a specified name and optional set of columns.
    /// If `cols` is provided, the columns are used to create the first table.
    /// The number of rows will be inferred from the first column.
    /// If the name is empty or whitespace, a unique default name is assigned.
    /// If no columns are provided, the Cube will be empty.
    pub fn new(
        name: impl Into<String>,
        cols: Option<Vec<FieldArray>>,
        third_dim_index: Option<Vec<String>>,
    ) -> Self {
        let name = name.into();
        let name = if name.trim().is_empty() {
            let id = UNNAMED_COUNTER.fetch_add(1, Ordering::Relaxed);
            format!("UnnamedCube{}", id)
        } else {
            name
        };

        let mut tables = Vec::new();
        let mut resolver = HashMap::new();

        if let Some(cols) = cols {
            let table = Table::new(name.clone(), Some(cols));
            resolver.insert(table.name.clone(), 0);
            tables.push(Arc::new(table));
        }

        let cube = Self {
            tables,
            name,
            third_dim_index,
            resolver,
        };
        cube.validate_third_dim_index();
        cube
    }

    /// Constructs a Cube from a pre-built tables vec. Builds the resolver
    /// automatically from table names.
    ///
    /// `index_by` specifies which column names form the third-dimension index,
    pub fn from_tables(tables: Vec<Table>, name: String, index_by: Option<Vec<String>>) -> Self {
        let arc_tables: Vec<Arc<Table>> = tables.into_iter().map(Arc::new).collect();
        let mut resolver = HashMap::new();
        for (i, t) in arc_tables.iter().enumerate() {
            resolver.insert(t.name.clone(), i);
        }
        Self {
            tables: arc_tables,
            name,
            third_dim_index: index_by,
            resolver,
        }
    }

    /// Constructs a new, empty Cube with a globally unique name.
    pub fn new_empty() -> Self {
        let id = UNNAMED_COUNTER.fetch_add(1, Ordering::Relaxed);
        let name = format!("UnnamedCube{}", id);
        Self {
            tables: Vec::new(),
            name,
            third_dim_index: None,
            resolver: HashMap::new(),
        }
    }

    /// Adds a table to the cube, keyed by the table's name.
    pub fn add_table(&mut self, table: Table) {
        if !self.tables.is_empty() {
            let existing_fields: HashMap<String, ArrowType> = self.tables[0]
                .cols()
                .iter()
                .map(|col| (col.field.name.clone(), col.field.dtype.clone()))
                .collect();

            for col in table.cols() {
                let field = &col.field;
                match existing_fields.get(&field.name) {
                    Some(existing_dtype) => assert_eq!(
                        existing_dtype, &field.dtype,
                        "Error: Schema mismatch between existing and new tables for Cube."
                    ),
                    None => panic!(
                        "New table has field '{}' with datatype '{}' not present in existing tables.",
                        field.name, field.dtype
                    ),
                }
            }
        }

        let idx = self.tables.len();
        self.resolver.insert(table.name.clone(), idx);
        self.tables.push(Arc::new(table));
    }

    /// Gets the schemea from the first table as representative
    /// of the rest
    pub fn schema(&self) -> Vec<Arc<Field>> {
        self.tables[0].schema()
    }

    /// Returns the number of tables.
    pub fn n_tables(&self) -> usize {
        self.tables.len()
    }

    /// Returns the number of tables.
    ///
    /// Alias for n_tables
    pub fn len(&self) -> usize {
        self.n_tables()
    }

    /// Returns the number of rows per table
    pub fn n_rows(&self) -> Vec<usize> {
        self.tables.iter().map(|t| t.n_rows()).collect()
    }

    /// Returns the number of columns.
    pub fn n_cols(&self) -> usize {
        self.tables[0].n_cols()
    }

    /// Returns true if the cube is empty (no tables, no columns or no rows).
    pub fn is_empty(&self) -> bool {
        self.n_tables() == 0 || self.n_cols() == 0 || self.tables.iter().all(|t| t.n_rows() == 0)
    }

    /// Returns a reference to a table by index.
    pub fn table(&self, idx: usize) -> Option<&Arc<Table>> {
        self.tables.get(idx)
    }

    /// Returns a mutable reference to a table by index (copy-on-write via Arc::make_mut).
    pub fn table_mut(&mut self, idx: usize) -> Option<&mut Table> {
        self.tables.get_mut(idx).map(|arc| Arc::make_mut(arc))
    }

    /// Returns the names of all tables in the cube.
    pub fn table_names(&self) -> Vec<&str> {
        self.tables.iter().map(|t| t.name.as_str()).collect()
    }

    /// Returns the index of a table by name.
    pub fn table_index(&self, name: &str) -> Option<usize> {
        self.tables.iter().position(|t| t.name == name)
    }

    /// Checks if a table with the given name exists.
    pub fn has_table(&self, name: &str) -> bool {
        self.table_index(name).is_some()
    }

    /// Removes a table by index.
    pub fn remove_table_at(&mut self, idx: usize) -> bool {
        if idx < self.tables.len() {
            let removed = self.tables.remove(idx);
            self.resolver.remove(&removed.name);
            // Rebuild resolver indices for entries after the removed position
            for (_, pos) in self.resolver.iter_mut() {
                if *pos > idx {
                    *pos -= 1;
                }
            }
            true
        } else {
            false
        }
    }

    /// Removes a table by name.
    pub fn remove_table(&mut self, name: &str) -> bool {
        if let Some(idx) = self.resolver.remove(name) {
            self.tables.remove(idx);
            // Rebuild resolver indices for entries after the removed position
            for (_, pos) in self.resolver.iter_mut() {
                if *pos > idx {
                    *pos -= 1;
                }
            }
            true
        } else {
            false
        }
    }

    /// Clears all tables.
    pub fn clear(&mut self) {
        self.tables.clear();
        self.resolver.clear();
        self.third_dim_index = None;
    }

    /// Returns an immutable reference to all tables.
    pub fn tables(&self) -> &[Arc<Table>] {
        &self.tables
    }

    /// Returns a mutable reference to the tables vec.
    pub fn tables_mut(&mut self) -> &mut Vec<Arc<Table>> {
        &mut self.tables
    }

    /// Returns an iterator over all tables.
    #[inline]
    pub fn iter_tables(&self) -> std::slice::Iter<'_, Arc<Table>> {
        self.tables.iter()
    }
    #[inline]
    pub fn iter_tables_mut(&mut self) -> std::slice::IterMut<'_, Arc<Table>> {
        self.tables.iter_mut()
    }

    /// Returns the list of column names (from the first table).
    pub fn col_names(&self) -> Vec<&str> {
        if self.tables.is_empty() {
            Vec::new()
        } else {
            self.tables[0].col_names()
        }
    }

    /// Returns the index of a column by name (from the first table).
    pub fn col_name_index(&self, name: &str) -> Option<usize> {
        if self.tables.is_empty() {
            None
        } else {
            self.tables[0].col_name_index(name)
        }
    }

    /// Returns true if the cube has a column of the given name (in all tables).
    pub fn has_col(&self, name: &str) -> bool {
        self.tables.iter().all(|t| t.has_col(name))
    }

    /// Returns all columns (FieldArrays) with the given index across all tables.
    pub fn col_ix(&self, idx: usize) -> Option<Vec<&FieldArray>> {
        if self.tables.is_empty() || idx >= self.n_cols() {
            None
        } else {
            Some(
                self.tables
                    .iter()
                    .filter_map(|t| t.cols().get(idx))
                    .collect(),
            )
        }
    }

    /// Returns all columns (FieldArrays) with the given name across all tables.
    pub fn col_by_name(&self, name: &str) -> Option<Vec<&FieldArray>> {
        if !self.has_col(name) {
            None
        } else {
            Some(
                self.tables
                    .iter()
                    .filter_map(|t| t.col_name_index(name).and_then(|idx| t.cols().get(idx)))
                    .collect(),
            )
        }
    }

    /// Returns all columns for all tables as Vec<Vec<&FieldArray>>.
    pub fn col_vec(&self) -> Vec<Vec<&FieldArray>> {
        self.tables
            .iter()
            .map(|t| t.cols().iter().collect())
            .collect()
    }

    /// Removes a column by name from all tables. Returns true if removed from all.
    pub fn remove_col(&mut self, name: &str) -> bool {
        let mut all_removed = true;
        for t in &mut self.tables {
            if !Arc::make_mut(t).remove_col(name) {
                all_removed = false;
            }
        }
        all_removed
    }

    /// Removes a column by index from all tables. Returns true if removed from all.
    pub fn remove_col_at(&mut self, idx: usize) -> bool {
        let mut all_removed = true;
        for t in &mut self.tables {
            if !Arc::make_mut(t).remove_col_at(idx) {
                all_removed = false;
            }
        }
        all_removed
    }

    /// Resolve a group key to its table position.
    pub fn resolve(&self, key: &str) -> Option<usize> {
        self.resolver.get(key).copied()
    }

    /// Rebuild the resolver from the current tables vec.
    pub fn rebuild_resolver(&mut self) {
        self.resolver.clear();
        for (i, t) in self.tables.iter().enumerate() {
            self.resolver.insert(t.name.clone(), i);
        }
    }

    /// Returns an iterator over the specified column across all tables.
    #[inline]
    pub fn iter_cols(&self, col_idx: usize) -> Option<impl Iterator<Item = &FieldArray>> {
        if col_idx < self.n_cols() {
            Some(
                self.tables
                    .iter()
                    .filter_map(move |t| t.cols().get(col_idx)),
            )
        } else {
            None
        }
    }

    /// Returns an iterator over the named column across all tables.
    #[inline]
    pub fn iter_cols_by_name<'a>(
        &'a self,
        name: &'a str,
    ) -> Option<impl Iterator<Item = &'a FieldArray> + 'a> {
        if self.has_col(name) {
            Some(
                self.tables
                    .iter()
                    .filter_map(move |t| t.col_name_index(name).and_then(|idx| t.cols().get(idx))),
            )
        } else {
            None
        }
    }

    /// Sets the third dimensional index
    pub fn set_third_dim_index<S: Into<String>>(&mut self, cols: Vec<S>) {
        self.third_dim_index = Some(cols.into_iter().map(|s| s.into()).collect());
    }

    /// Retrieves the third dimensional index
    pub fn third_dim_index(&self) -> Option<&[String]> {
        self.third_dim_index.as_deref()
    }

    /// Confirms that the third dimension index exists in the schema
    fn validate_third_dim_index(&self) {
        if let Some(ref indices) = self.third_dim_index {
            for col_name in indices {
                assert!(
                    self.has_col(col_name),
                    "Index column '{}' not found in all tables",
                    col_name
                );
            }
        }
    }

    /// Returns a new owned Cube containing rows `[offset, offset+len)` for all tables.
    #[cfg(feature = "views")]
    pub fn slice_clone(&self, offset: usize, len: usize) -> Self {
        assert!(!self.tables.is_empty(), "No tables to slice");
        for t in &self.tables {
            assert!(
                offset + len <= t.n_rows(),
                "slice window out of bounds for one or more tables"
            );
        }
        let tables: Vec<Arc<Table>> = self
            .tables
            .iter()
            .map(|t| Arc::new(t.slice_clone(offset, len)))
            .collect();
        let mut resolver = HashMap::new();
        for (i, t) in tables.iter().enumerate() {
            resolver.insert(t.name.clone(), i);
        }
        let name = format!("{}[{}, {})", self.name, offset, offset + len);
        Cube {
            tables,
            name,
            third_dim_index: self.third_dim_index.clone(),
            resolver,
        }
    }

    /// Returns a zero-copy view over rows `[offset, offset+len)` for all tables.
    #[cfg(feature = "views")]
    pub fn slice(&self, offset: usize, len: usize) -> CubeV {
        assert!(!self.tables.is_empty(), "No tables to slice");
        for t in &self.tables {
            assert!(
                offset + len <= t.n_rows(),
                "slice window out of bounds for one or more tables"
            );
        }
        self.tables
            .iter()
            .map(|t| TableV::from_table((**t).clone(), offset, len))
            .collect()
    }

    /// Returns a parallel iterator over all tables.
    #[cfg(feature = "parallel_proc")]
    #[inline]
    pub fn par_iter_tables(&self) -> rayon::slice::Iter<'_, Arc<Table>> {
        self.tables.par_iter()
    }

    /// Returns a parallel mutable iterator over all tables.
    #[cfg(feature = "parallel_proc")]
    #[inline]
    pub fn par_iter_tables_mut(&mut self) -> rayon::slice::IterMut<'_, Arc<Table>> {
        self.tables.par_iter_mut()
    }
}

impl<'a> IntoIterator for &'a Cube {
    type Item = &'a Arc<Table>;
    type IntoIter = std::slice::Iter<'a, Arc<Table>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.tables.iter()
    }
}

impl<'a> IntoIterator for &'a mut Cube {
    type Item = &'a mut Arc<Table>;
    type IntoIter = std::slice::IterMut<'a, Arc<Table>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.tables.iter_mut()
    }
}

impl IntoIterator for Cube {
    type Item = Arc<Table>;
    type IntoIter = <Vec<Arc<Table>> as IntoIterator>::IntoIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.tables.into_iter()
    }
}

impl Shape for Cube {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Collection(self.tables.iter().map(|t| t.shape()).collect())
    }
}

impl Concatenate for Cube {
    /// Concatenates two cubes by appending all tables from `other` to `self`.
    ///
    /// # Requirements
    /// - Both cubes must have the same schema (column names and types)
    ///
    /// # Returns
    /// A new Cube containing all tables from `self` followed by all tables from `other`
    ///
    /// # Errors
    /// - `IncompatibleTypeError` if schemas don't match
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        // If both cubes are empty, return empty cube
        if self.tables.is_empty() && other.tables.is_empty() {
            return Ok(Cube::new(
                format!("{}+{}", self.name, other.name),
                None,
                None,
            ));
        }

        // If one is empty, return the other
        if self.tables.is_empty() {
            let mut result = other;
            result.name = format!("{}+{}", self.name, result.name);
            return Ok(result);
        }
        if other.tables.is_empty() {
            let mut result = self;
            result.name = format!("{}+{}", result.name, other.name);
            return Ok(result);
        }

        // Validate schemas match between first tables
        let self_schema: HashMap<String, ArrowType> = self.tables[0]
            .cols()
            .iter()
            .map(|col| (col.field.name.clone(), col.field.dtype.clone()))
            .collect();

        let other_schema: HashMap<String, ArrowType> = other.tables[0]
            .cols()
            .iter()
            .map(|col| (col.field.name.clone(), col.field.dtype.clone()))
            .collect();

        // Check column count
        if self_schema.len() != other_schema.len() {
            return Err(MinarrowError::IncompatibleTypeError {
                from: "Cube",
                to: "Cube",
                message: Some(format!(
                    "Cannot concatenate cubes with different column counts: {} vs {}",
                    self_schema.len(),
                    other_schema.len()
                )),
            });
        }

        // Check schema compatibility
        for (col_name, col_type) in &self_schema {
            match other_schema.get(col_name) {
                Some(other_type) if other_type == col_type => {}
                Some(other_type) => {
                    return Err(MinarrowError::IncompatibleTypeError {
                        from: "Cube",
                        to: "Cube",
                        message: Some(format!(
                            "Column '{}' type mismatch: {:?} vs {:?}",
                            col_name, col_type, other_type
                        )),
                    });
                }
                None => {
                    return Err(MinarrowError::IncompatibleTypeError {
                        from: "Cube",
                        to: "Cube",
                        message: Some(format!(
                            "Column '{}' present in first cube but not in second",
                            col_name
                        )),
                    });
                }
            }
        }

        // Concatenate tables
        let mut result_tables = self.tables;
        result_tables.extend(other.tables);
        let mut resolver = HashMap::new();
        for (i, t) in result_tables.iter().enumerate() {
            resolver.insert(t.name.clone(), i);
        }

        Ok(Cube {
            tables: result_tables,
            name: format!("{}+{}", self.name, other.name),
            third_dim_index: self.third_dim_index.clone(),
            resolver,
        })
    }
}

// ColumnSelection for Cube - selects columns from each sub-table,
// always preserving third_dim_index columns since they are structural.
#[cfg(all(feature = "views", feature = "select"))]
impl crate::traits::selection::ColumnSelection for Cube {
    type View = Cube;
    type ColumnView = Vec<crate::ArrayV>;
    type ColumnOwned = Arc<Table>;

    fn c<S: crate::traits::selection::FieldSelector>(&self, selection: S) -> Cube {
        if self.tables.is_empty() {
            return self.clone();
        }

        let fields = self.schema();
        let mut indices = selection.resolve_fields(&fields);

        // Always preserve third_dim_index columns - they are structural, not data
        if let Some(ref keys) = self.third_dim_index {
            for key_name in keys {
                if let Some(idx) = self.col_name_index(key_name) {
                    if !indices.contains(&idx) {
                        indices.push(idx);
                    }
                }
            }
        }
        indices.sort();
        indices.dedup();

        // If selecting all columns, return a clone without rebuilding
        if indices.len() == fields.len() {
            return self.clone();
        }

        // Build new Cube with only selected columns from each sub-table
        let tables: Vec<Arc<Table>> = self
            .tables
            .iter()
            .map(|t| {
                let cols: Vec<FieldArray> = indices
                    .iter()
                    .filter_map(|&i| t.cols().get(i).cloned())
                    .collect();
                Arc::new(Table::new(t.name.clone(), Some(cols)))
            })
            .collect();

        let mut resolver = HashMap::new();
        for (i, t) in tables.iter().enumerate() {
            resolver.insert(t.name.clone(), i);
        }
        Cube {
            tables,
            name: self.name.clone(),
            third_dim_index: self.third_dim_index.clone(),
            resolver,
        }
    }

    fn get(&self, field: &str) -> Option<Arc<Table>> {
        let idx = self.resolver.get(field)?;
        self.tables.get(*idx).cloned()
    }

    fn col_ix(&self, idx: usize) -> Option<Self::ColumnView> {
        if self.tables.is_empty() || idx >= self.n_cols() {
            return None;
        }
        Some(
            self.tables
                .iter()
                .filter_map(|t| {
                    let fa = t.cols().get(idx)?;
                    Some(crate::ArrayV::new(fa.array.clone(), 0, fa.len()))
                })
                .collect(),
        )
    }

    fn col_vec(&self) -> Vec<Self::ColumnView> {
        if self.tables.is_empty() {
            return Vec::new();
        }
        (0..self.n_cols())
            .filter_map(|i| {
                // Use ColumnSelection::col_ix, not the inherent method
                crate::traits::selection::ColumnSelection::col_ix(self, i)
            })
            .collect()
    }

    fn get_cols(&self) -> Vec<Arc<Field>> {
        if self.tables.is_empty() {
            Vec::new()
        } else {
            self.schema()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structs::field_array::field_array;
    use crate::traits::masked_array::MaskedArray;
    use crate::traits::selection::ColumnSelection;
    use crate::{Array, BooleanArray, IntegerArray, NumericArray};

    fn build_test_table(name: &str, vals: &[i32], bools: &[bool]) -> Table {
        let mut int_arr = IntegerArray::<i32>::default();
        for v in vals {
            int_arr.push(*v);
        }
        let mut bool_arr = BooleanArray::default();
        for b in bools {
            bool_arr.push(*b);
        }
        let mut t = Table::new(name.to_string(), None);
        t.add_col(field_array("ints", Array::from_int32(int_arr)));
        t.add_col(field_array("bools", Array::from_bool(bool_arr)));
        t
    }

    #[test]
    fn test_new_cube_empty() {
        let c = Cube::new_empty();
        assert_eq!(c.n_tables(), 0);
        assert_eq!(c.n_rows().len(), 0);
        assert!(c.is_empty());
        assert!(c.tables().is_empty());
    }

    #[test]
    fn test_add_and_get_tables_and_columns() {
        let mut c = Cube::new_empty();
        let t1 = build_test_table("t1", &[1, 2], &[true, false]);
        let t2 = build_test_table("t2", &[3, 4], &[false, true]);

        c.add_table(t1.clone());
        c.add_table(t2.clone());

        assert_eq!(c.n_tables(), 2);
        assert_eq!(c.n_rows(), vec![2, 2]);
        assert!(!c.is_empty());

        // Table access
        assert_eq!(c.table_names(), vec!["t1", "t2"]);
        assert!(c.has_table("t1"));
        assert!(!c.has_table("notthere"));
        assert_eq!(c.table_index("t2"), Some(1));
        assert_eq!(c.table(0).unwrap().name, "t1");
        assert_eq!(c.table(1).unwrap().name, "t2");

        // Column access across tables
        assert_eq!(c.n_cols(), 2);
        assert_eq!(c.col_names(), vec!["ints", "bools"]);
        assert!(c.has_col("ints"));
        assert!(!c.has_col("nonexistent"));

        let cols_by_idx = c.col_ix(0).unwrap();
        assert_eq!(cols_by_idx.len(), 2);
        assert_eq!(cols_by_idx[0].field.name, "ints");
        assert_eq!(cols_by_idx[1].field.name, "ints");

        let cols_by_name = c.col_by_name("bools").unwrap();
        assert_eq!(cols_by_name.len(), 2);
        assert_eq!(cols_by_name[0].field.name, "bools");

        // Iterating columns across tables
        let mut seen: Vec<i32> = Vec::new();
        for col in c.iter_cols_by_name("ints").unwrap() {
            match &col.array {
                Array::NumericArray(NumericArray::Int32(arr)) => seen.push(arr.get(0).unwrap()),
                _ => panic!("Type mismatch"),
            }
        }
        assert_eq!(seen, vec![1, 3]);
    }

    #[test]
    #[should_panic]
    fn test_add_table_schema_mismatch_panics() {
        let mut c = Cube::new_empty();
        let mut t1 = Table::new("t1", None);
        let mut arr = IntegerArray::<i32>::default();
        arr.push(1);
        t1.add_col(field_array("ints", Array::from_int32(arr)));
        c.add_table(t1);

        let mut t2 = Table::new("t2", None);
        let mut arr2 = IntegerArray::<i32>::default();
        arr2.push(2);
        t2.add_col(field_array("other", Array::from_int32(arr2))); // Different column name
        c.add_table(t2); // Should panic
    }

    #[test]
    fn test_remove_table_by_index_and_name() {
        let mut c = Cube::new_empty();
        let t1 = build_test_table("t1", &[1, 2], &[true, false]);
        let t2 = build_test_table("t2", &[3, 4], &[false, true]);
        c.add_table(t1.clone());
        c.add_table(t2.clone());

        assert!(c.remove_table("t1"));
        assert_eq!(c.n_tables(), 1);
        assert!(!c.has_table("t1"));

        assert!(c.remove_table_at(0));
        assert_eq!(c.n_tables(), 0);

        // Remove non-existent
        assert!(!c.remove_table("not_there"));
        assert!(!c.remove_table_at(5));
    }

    #[test]
    fn test_remove_and_clear_column_across_tables() {
        let mut c = Cube::new_empty();
        let t1 = build_test_table("t1", &[1, 2], &[true, false]);
        let t2 = build_test_table("t2", &[3, 4], &[false, true]);
        c.add_table(t1.clone());
        c.add_table(t2.clone());

        // Remove by name
        assert!(c.remove_col("ints"));
        assert!(!c.has_col("ints"));
        assert_eq!(c.n_cols(), 1);

        // Remove by index
        assert!(c.remove_col_at(0));
        assert_eq!(c.n_cols(), 0);

        // Remove non-existent column
        assert!(!c.remove_col("doesnotexist"));
        assert!(!c.remove_col_at(10));
    }

    #[test]
    fn test_clear_cube() {
        let mut c = Cube::new_empty();
        let t = build_test_table("t1", &[1, 2, 3], &[true, false, true]);
        c.add_table(t);
        assert!(!c.is_empty());
        c.clear();
        assert!(c.is_empty());
        assert_eq!(c.n_tables(), 0);
        assert_eq!(c.n_rows().len(), 0);
    }

    #[test]
    fn test_iter_tables_and_into_iter() {
        let mut c = Cube::new_empty();
        let t1 = build_test_table("t1", &[1], &[true]);
        let t2 = build_test_table("t2", &[2], &[false]);
        c.add_table(t1.clone());
        c.add_table(t2.clone());

        let names: Vec<_> = c.iter_tables().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["t1", "t2"]);

        let names2: Vec<_> = (&c).into_iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names2, ["t1", "t2"]);
    }

    #[test]
    fn test_col_vec_method_and_schema() {
        let mut c = Cube::new_empty();
        let t1 = build_test_table("t1", &[1, 2], &[true, false]);
        let t2 = build_test_table("t2", &[3, 4], &[false, true]);
        c.add_table(t1);
        c.add_table(t2);

        let cols = c.col_vec();
        assert_eq!(cols.len(), 2); // Two tables
        assert_eq!(cols[0].len(), 2); // Each has two columns

        let schema = c.schema();
        assert_eq!(schema[0].name, "ints");
        assert_eq!(schema[1].name, "bools");
    }

    #[test]
    fn test_iter_cols_and_col_variants() {
        let mut c = Cube::new_empty();
        let t1 = build_test_table("t1", &[1, 2], &[true, false]);
        let t2 = build_test_table("t2", &[3, 4], &[false, true]);
        c.add_table(t1.clone());
        c.add_table(t2.clone());

        // By index
        let ints: Vec<i32> = c
            .iter_cols(0)
            .unwrap()
            .map(|col| match &col.array {
                Array::NumericArray(NumericArray::Int32(arr)) => arr.get(1).unwrap(),
                _ => panic!("Type mismatch"),
            })
            .collect();
        assert_eq!(ints, vec![2, 4]);

        // By name
        let bools: Vec<bool> = c
            .iter_cols_by_name("bools")
            .unwrap()
            .map(|col| match &col.array {
                Array::BooleanArray(arr) => arr.get(0).unwrap(),
                _ => panic!("Type mismatch"),
            })
            .collect();
        assert_eq!(bools, vec![true, false]);
    }

    // Add test for new() if implemented
    #[test]
    fn test_cube_new_named() {
        let mut int_arr = IntegerArray::<i32>::default();
        int_arr.push(42);
        let mut bool_arr = BooleanArray::default();
        bool_arr.push(true);
        let cols = vec![
            field_array("x", Array::from_int32(int_arr)),
            field_array("flag", Array::from_bool(bool_arr)),
        ];
        let table = Table::new("single".to_string(), Some(cols.clone()));
        let cube = Cube {
            tables: vec![Arc::new(table)],
            name: "test".to_string(),
            third_dim_index: Some(vec!["timestamp".to_string()]),
            resolver: HashMap::from([("single".to_string(), 0)]),
        };
        assert_eq!(cube.n_tables(), 1);
        assert_eq!(cube.n_cols(), 2);
        assert_eq!(cube.name, "test");
        assert_eq!(cube.third_dim_index().unwrap(), &["timestamp"]);
    }

    #[cfg(feature = "views")]
    #[test]
    fn test_cube_slice_and_slice_clone() {
        use crate::structs::field_array::field_array;
        use crate::{Array, BooleanArray, IntegerArray};

        let mut col1 = IntegerArray::<i32>::default();
        col1.push(1);
        col1.push(2);
        col1.push(3);
        let mut col2 = BooleanArray::default();
        col2.push(true);
        col2.push(false);
        col2.push(true);

        let mut t1 = Table::new("snap1", None);
        t1.add_col(field_array("ints", Array::from_int32(col1)));
        t1.add_col(field_array("bools", Array::from_bool(col2)));

        let mut t2 = Table::new("snap2", None);
        let mut col3 = IntegerArray::<i32>::default();
        col3.push(11);
        col3.push(12);
        col3.push(13);
        let mut col4 = BooleanArray::default();
        col4.push(false);
        col4.push(true);
        col4.push(false);
        t2.add_col(field_array("ints", Array::from_int32(col3)));
        t2.add_col(field_array("bools", Array::from_bool(col4)));

        let mut cube = Cube::new_empty();
        cube.add_table(t1);
        cube.add_table(t2);

        let view = cube.slice(1, 2); // Vec<TableView<'_>>
        assert_eq!(view.len(), 2); // Two tables
        assert_eq!(view[0].n_rows(), 2); // First table window length is 2
        assert_eq!(view[1].n_rows(), 2); // Second table window length is 2

        assert_eq!(view[0].col("bools").col_ix(0).unwrap().len(), 2); // arrayview length is 2
        assert_eq!(view[1].col("bools").col_ix(0).unwrap().len(), 2); // arrayview length is 2
    }

    #[cfg(feature = "parallel_proc")]
    #[test]
    fn test_cube_par_iter_tables() {
        use rayon::prelude::*;
        let mut cube = Cube::new_empty();
        let t1 = build_test_table("a", &[1, 2], &[true, false]);
        let t2 = build_test_table("b", &[3, 4], &[false, true]);
        cube.add_table(t1);
        cube.add_table(t2);

        let mut names: Vec<&str> = cube.par_iter_tables().map(|t| t.name.as_str()).collect();
        names.sort_unstable();
        assert_eq!(names, vec!["a", "b"]);
    }
}
