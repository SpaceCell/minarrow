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

//! # **SuperTable** - *Holds multiple Tables for chunked data partitioning, streaming + fast memIO*
//!
//! SuperTable groups multiple `Table` batches under a shared schema.
//!
//! ## Purpose
//! - Treats an ordered sequence of `Table` batches as one dataset.
//! - Preserves per-batch independence while allowing unified export over Arrow FFI.
//! - Enables both bounded and unbounded (Live / streaming) workflows.
//!
//! ## Behaviour
//! - All batches must have identical column definitions (`Field` metadata).
//! - Row counts may differ between batches, but not between inner arrays.
//! - When sent via Arrow FFI, exposed as a single logical table.
//! - Supports concatenation into a materialised `Table` on demand.
//!
//! ## Typical Uses
//! - Partitioned storage readers (e.g., multiple Arrow IPC files).
//! - Streaming ingestion into append-only datasets.
//! - Windowed or mini-batch analytics.
//! - Incremental build-up of tables for later unification.
//!
//! ## Apache Arrow / Polars bridges (`cast_arrow` / `cast_polars` features)
//! - `to_apache_arrow()` exports each batch as an arrow-rs `RecordBatch`.
//! - `to_polars()` builds a polars `DataFrame` whose columns are chunked Series
//!   mirroring the SuperTable's batches.
//! - `from_apache_arrow(&[RecordBatch])` and `from_polars(&DataFrame)`
//!   recover schema, dtypes, and chunk boundaries on import.
//! - Each panicking method has a `try_*` sibling returning `Result<_, MinarrowError>`.
//! - `&[RecordBatch]` and `&DataFrame` can be converted via `.into()` for
//!   ergonomic call sites.

use std::fmt::{Display, Formatter};

use std::iter::FromIterator;
use std::sync::Arc;

use crate::enums::{error::MinarrowError, shape_dim::ShapeDim};
use crate::structs::chunked::super_array::RechunkStrategy;
#[cfg(feature = "shared_dict")]
use crate::structs::dictionary::CategoryManagerT;
use crate::structs::field::Field;
use crate::structs::field_array::FieldArray;
use crate::structs::table::Table;
#[cfg(feature = "size")]
use crate::traits::byte_size::ByteSize;
use crate::traits::concatenate::Concatenate;
use crate::traits::consolidate::Consolidate;
use crate::traits::shape::Shape;
#[cfg(feature = "views")]
use crate::{SuperTableV, TableV};

/// # SuperTable
///
/// Higher-order container representing a sequence of `Table` batches with consistent schema.
///
/// ## Overview
/// - Each batch is a `Table` (record batch) with identical column metadata.
/// - Stored as `Vec<Arc<Table>>`, preserving order and schema consistency.
/// - Row counts per batch may vary, but are consistent across all Table columns.
/// - When exported via Arrow FFI, the batches are viewed as a single logical table.
/// - Useful for open-ended streams, partitioned datasets, or
///   other scenarios where batches are processed independently.
///
/// ## Fields
/// - `batches`: ordered collection of `Table` batches.
/// - `schema`: cached schema from the first batch for fast access.
/// - `n_rows`: total row count across all batches.
/// - `name`: super table name.
///
/// ## Use cases
/// - Streaming and mini-batch processing.
/// - Reading multiple Arrow IPC/memory-mapped files as one dataset.
/// - Parallel or windowed in-memory analytics.
/// - Incremental table construction where batches arrive over time.
#[derive(Clone, Debug)]
pub struct SuperTable {
    pub batches: Vec<Arc<Table>>,
    pub schema: Vec<Arc<Field>>,
    pub n_rows: usize,
    pub name: String,
    /// One shared category dictionary per column, present only when the
    /// `shared_dict` feature is on and the column type is categorical.
    /// Sibling batches pushed into the same `SuperTable` all share each
    /// column's dictionary, so codes are mutually meaningful across them.
    /// New values arrive via `push(batch)`.
    #[cfg(feature = "shared_dict")]
    pub(crate) category_managers: Vec<Option<CategoryManagerT>>,
}

impl PartialEq for SuperTable {
    /// Equality compares the table elements minus the category manager.
    fn eq(&self, other: &Self) -> bool {
        self.batches == other.batches
            && self.schema == other.schema
            && self.n_rows == other.n_rows
            && self.name == other.name
    }
}

impl SuperTable {
    /// Creates a new empty BatchedTable with a specified name.
    pub fn new(name: String) -> Self {
        Self {
            batches: Vec::new(),
            schema: Vec::new(),
            n_rows: 0,
            name,
            #[cfg(feature = "shared_dict")]
            category_managers: Vec::new(),
        }
    }

    /// Builds from a collection of Table batches.
    ///
    /// Panics if column count or field metadata are inconsistent.
    pub fn from_batches(batches: Vec<Arc<Table>>, name_override: Option<String>) -> Self {
        if batches.is_empty() {
            return Self::new("Unnamed".into());
        }

        let name = name_override.unwrap_or_else(|| batches[0].name.clone());
        let schema: Vec<Arc<Field>> = batches[0].cols.iter().map(|fa| fa.field.clone()).collect();
        let n_cols = schema.len();
        let mut total_rows = 0usize;

        // Validate all batches.
        for (b_idx, batch) in batches.iter().enumerate() {
            assert_eq!(
                batch.n_cols(),
                n_cols,
                "Batch {b_idx} column-count mismatch"
            );
            for col_idx in 0..n_cols {
                let field = &schema[col_idx];
                let fa = &batch.cols[col_idx];
                assert_eq!(
                    &fa.field, field,
                    "Batch {b_idx} col {col_idx} schema mismatch"
                );
            }
            total_rows += batch.n_rows;
        }

        #[cfg_attr(not(feature = "shared_dict"), allow(unused_mut))]
        let mut st = Self {
            batches,
            schema,
            n_rows: total_rows,
            name,
            #[cfg(feature = "shared_dict")]
            category_managers: Vec::new(),
        };
        #[cfg(feature = "shared_dict")]
        st.rebuild_category_managers();
        st
    }

    /// Append a new Table batch. Panics on schema mismatch.
    ///
    /// When the `shared_dict` feature is on, each categorical column is
    /// routed through this SuperTable's `CategoryManagerT` for that
    /// column. The first batch seeds the manager from its own dictionary;
    /// subsequent batches merge their values into the existing manager and
    /// rebind their dictionaries to the resulting shared snapshot.
    pub fn push(&mut self, batch: Arc<Table>) {
        if self.batches.is_empty() {
            self.schema = batch.cols.iter().map(|fa| fa.field.clone()).collect();
            #[cfg(feature = "shared_dict")]
            {
                self.category_managers = vec![None; self.schema.len()];
            }
        }
        let n_cols = self.schema.len();
        assert_eq!(batch.n_cols(), n_cols, "Pushed batch column-count mismatch");
        for col_idx in 0..n_cols {
            let field = &self.schema[col_idx];
            let fa = &batch.cols[col_idx];
            assert_eq!(
                &fa.field, field,
                "Pushed batch col {col_idx} schema mismatch"
            );
        }

        #[cfg_attr(not(feature = "shared_dict"), allow(unused_mut))]
        let mut batch = batch;
        #[cfg(feature = "shared_dict")]
        self.add_dict_categories(&mut batch);

        self.n_rows += batch.n_rows;
        self.batches.push(batch);
    }

    /// Borrow the column's `CategoryManagerT`, or `None` if the column
    /// is not categorical or `col_idx` is out of bounds.
    ///
    /// This accessor is for sharing-association: two `CategoricalArray`s
    /// whose dictionaries point at the same `Arc<DictionaryInner>` (check
    /// via `Dictionary::shares_with`) are in the same sharing group.
    /// New values arrive through `push(batch)`, not through this borrow.
    #[cfg(feature = "shared_dict")]
    #[inline]
    pub fn category_manager(&self, col_idx: usize) -> Option<&CategoryManagerT> {
        self.category_managers.get(col_idx)?.as_ref()
    }

    /// Rebuild every column's category manager from the current batches.
    /// The first batch's dictionary seeds each manager; subsequent batches
    /// are merged in by the same logic as `push`.
    #[cfg(feature = "shared_dict")]
    pub(crate) fn rebuild_category_managers(&mut self) {
        if self.batches.is_empty() {
            self.category_managers.clear();
            return;
        }
        let n_cols = self.schema.len();
        self.category_managers = vec![None; n_cols];

        let batches = std::mem::take(&mut self.batches);
        let mut rebuilt: Vec<Arc<Table>> = Vec::with_capacity(batches.len());
        for batch in batches {
            let mut batch = batch;
            self.add_dict_categories(&mut batch);
            rebuilt.push(batch);
        }
        self.batches = rebuilt;
    }

    /// Route each categorical column in the incoming batch through this
    /// SuperTable's column manager: merge codes, remap if shifted, and
    /// rebind the batch's dictionary to the shared snapshot.
    #[cfg(feature = "shared_dict")]
    fn add_dict_categories(&mut self, incoming: &mut Arc<Table>) {
        let table = Arc::make_mut(incoming);
        if self.category_managers.len() < table.cols.len() {
            self.category_managers.resize_with(table.cols.len(), || None);
        }
        for (col_idx, fa) in table.cols.iter_mut().enumerate() {
            CategoryManagerT::add_remap_cats(
                &mut self.category_managers[col_idx],
                std::iter::once(&mut fa.array),
            );
        }
    }

    /// Inserts rows from another SuperTable (or Table) at the specified index.
    ///
    /// This is an **O(n)** operation where n is the number of rows in the batch
    /// containing the insertion point.
    ///
    /// # Arguments
    /// * `index` - Global row position before which to insert (0 = prepend, n_rows = append)
    /// * `other` - SuperTable or Table to insert (via `Into<SuperTable>`)
    ///
    /// # Requirements
    /// - Schema (column names, types, nullability) must match
    /// - `index` must be <= `self.n_rows`
    ///
    /// # Strategy
    /// Finds the batch containing the insertion point, splits it at that position, then
    /// inserts other's batches in between the split halves. This redistributes rows across
    /// batches while preserving chunked structure.
    ///
    /// # Errors
    /// - `IndexError` if index > n_rows
    /// - Schema mismatch if field metadata doesn't match
    pub fn insert_rows(
        &mut self,
        index: usize,
        other: impl Into<SuperTable>,
    ) -> Result<(), MinarrowError> {
        let other = other.into();

        // Validate index
        if index > self.n_rows {
            return Err(MinarrowError::IndexError(format!(
                "Index {} out of bounds for SuperTable with {} rows",
                index, self.n_rows
            )));
        }

        // If other is empty, nothing to do
        if other.n_rows == 0 {
            return Ok(());
        }

        // Validate schema match
        if !self.batches.is_empty() {
            if self.schema.len() != other.schema.len() {
                return Err(MinarrowError::IncompatibleTypeError {
                    from: "SuperTable",
                    to: "SuperTable",
                    message: Some(format!(
                        "Column count mismatch: {} vs {}",
                        self.schema.len(),
                        other.schema.len()
                    )),
                });
            }

            for (col_idx, (self_field, other_field)) in
                self.schema.iter().zip(other.schema.iter()).enumerate()
            {
                if self_field.name != other_field.name {
                    return Err(MinarrowError::IncompatibleTypeError {
                        from: "SuperTable",
                        to: "SuperTable",
                        message: Some(format!(
                            "Column {} name mismatch: '{}' vs '{}'",
                            col_idx, self_field.name, other_field.name
                        )),
                    });
                }

                if self_field.dtype != other_field.dtype {
                    return Err(MinarrowError::IncompatibleTypeError {
                        from: "SuperTable",
                        to: "SuperTable",
                        message: Some(format!(
                            "Column '{}' type mismatch: {:?} vs {:?}",
                            self_field.name, self_field.dtype, other_field.dtype
                        )),
                    });
                }

                if self_field.nullable != other_field.nullable {
                    return Err(MinarrowError::IncompatibleTypeError {
                        from: "SuperTable",
                        to: "SuperTable",
                        message: Some(format!(
                            "Column '{}' nullable mismatch: {} vs {}",
                            self_field.name, self_field.nullable, other_field.nullable
                        )),
                    });
                }
            }
        }

        // Handle empty self - just append other's batches
        if self.batches.is_empty() {
            self.batches = other.batches;
            self.schema = other.schema;
            self.n_rows = other.n_rows;
            return Ok(());
        }

        // Find which batch contains the insertion index
        let mut cumulative = 0;
        let mut target_idx = 0;
        let mut local_index = index;

        for (idx, batch) in self.batches.iter().enumerate() {
            let batch_rows = batch.n_rows;

            if index <= cumulative + batch_rows {
                target_idx = idx;
                local_index = index - cumulative;
                break;
            }

            cumulative += batch_rows;
        }

        let target_batch_rows = self.batches[target_idx].n_rows;

        // Handle edge cases: prepend or append to a batch without splitting
        if local_index == 0 {
            // Insert before target batch
            let mut new_batches = Vec::with_capacity(self.batches.len() + other.batches.len());
            new_batches.extend(self.batches.drain(0..target_idx));
            new_batches.extend(other.batches.into_iter());
            new_batches.extend(self.batches.drain(..));
            self.batches = new_batches;
            self.n_rows += other.n_rows;
        } else if local_index == target_batch_rows {
            // Insert after target batch
            let mut new_batches = Vec::with_capacity(self.batches.len() + other.batches.len());
            new_batches.extend(self.batches.drain(0..=target_idx));
            new_batches.extend(other.batches.into_iter());
            new_batches.extend(self.batches.drain(..));
            self.batches = new_batches;
            self.n_rows += other.n_rows;
        } else {
            // Split the target batch at the insertion point
            let target_batch = self.batches.remove(target_idx);
            let target_table = Arc::try_unwrap(target_batch).unwrap_or_else(|arc| (*arc).clone());
            let mut split_batches = target_table.split(local_index)?;

            // Build new batch list: batches before target + left batch + other's batches + right batch + remaining batches
            let mut new_batches = Vec::with_capacity(self.batches.len() + other.batches.len() + 2);
            new_batches.extend(self.batches.drain(0..target_idx));
            new_batches.extend(split_batches.batches.drain(0..1));
            new_batches.extend(other.batches.into_iter());
            new_batches.extend(split_batches.batches.drain(..));
            new_batches.extend(self.batches.drain(..));

            self.batches = new_batches;
            self.n_rows += other.n_rows;
        }

        Ok(())
    }

    // API

    #[inline]
    pub fn n_cols(&self) -> usize {
        self.schema.len()
    }

    // TODO: Add test, confirm null case

    /// Returns the columns of the Super Table
    ///
    /// Holds an assumption that all inner tables have the same fields
    #[inline]
    pub fn cols(&self) -> Vec<Arc<Field>> {
        self.batches[0]
            .cols()
            .iter()
            .map(|x| x.field.clone())
            .collect()
    }

    /// Number of rows across all batches.
    #[cfg(not(feature = "lbuffer"))]
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of rows across all batches.
    #[cfg(feature = "lbuffer")]
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.batches.iter().map(|b| b.n_rows()).sum()
    }

    #[inline]
    pub fn n_batches(&self) -> usize {
        self.batches.len()
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.n_rows()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_rows == 0
    }
    #[inline]
    pub fn schema(&self) -> &[Arc<Field>] {
        &self.schema
    }
    #[inline]
    pub fn batches(&self) -> &[Arc<Table>] {
        &self.batches
    }
    #[inline]
    pub fn batch(&self, idx: usize) -> Option<&Arc<Table>> {
        self.batches.get(idx)
    }

    /// Returns the schema-level metadata from the first batch, or an empty map
    /// if there are no batches.
    #[cfg(feature = "table_metadata")]
    pub fn metadata(&self) -> &std::collections::BTreeMap<String, String> {
        static EMPTY: std::sync::LazyLock<std::collections::BTreeMap<String, String>> =
            std::sync::LazyLock::new(std::collections::BTreeMap::new);
        self.batches.first().map(|b| b.metadata()).unwrap_or(&EMPTY)
    }

    // Return a new BatchedTable over a sub-range of rows.
    #[cfg(feature = "views")]
    pub fn view(&self, offset: usize, len: usize) -> SuperTableV {
        assert!(offset + len <= self.n_rows, "slice out of bounds");
        let mut slices = Vec::<TableV>::new();
        let mut remaining = len;
        let mut global_row = offset;

        for batch in &self.batches {
            if global_row >= batch.n_rows {
                global_row -= batch.n_rows;
                continue;
            }
            let take = (batch.n_rows - global_row).min(remaining);
            slices.push(TableV::from_arc_table(batch.clone(), global_row, take));
            global_row = 0;
            remaining -= take;
            if remaining == 0 {
                break;
            }
        }
        SuperTableV { slices, len }
    }

    #[cfg(feature = "views")]
    pub fn from_views(slices: &[TableV], name: String) -> Self {
        assert!(!slices.is_empty(), "from_slices: no slices provided");
        let n_cols = slices[0].n_cols();
        let mut batches = Vec::with_capacity(slices.len());
        let mut total_rows = 0usize;
        for slice in slices {
            let table = slice.to_table();
            assert_eq!(table.n_cols(), n_cols, "Batch column-count mismatch");
            total_rows += table.n_rows;
            batches.push(table.into());
        }
        let schema = slices[0].fields.iter().cloned().collect();
        #[cfg_attr(not(feature = "shared_dict"), allow(unused_mut))]
        let mut st = Self {
            batches,
            schema,
            n_rows: total_rows,
            name,
            #[cfg(feature = "shared_dict")]
            category_managers: Vec::new(),
        };
        #[cfg(feature = "shared_dict")]
        st.rebuild_category_managers();
        st
    }

    /// Rechunks the table according to the specified strategy.
    ///
    /// Redistributes rows across batches using an efficient incremental approach
    /// that avoids full materialization:
    /// - `Count(n)`: Creates batches of `n` rows (last batch may be smaller)
    /// - `Auto`: Uses a default size of 8192 rows
    /// - `Memory(bytes)`: Targets a specific memory size per batch
    ///
    /// # Arguments
    /// * `strategy` - The rechunking strategy to use
    ///
    /// # Errors
    /// - Returns `IndexError` if `Count(0)` is specified
    /// - Returns `IndexError` if memory-based calculation results in 0 chunk size
    ///
    /// # Example
    /// ```ignore
    /// // Rechunk into 1024-row batches
    /// table.rechunk(RechunkStrategy::Count(1024))?;
    ///
    /// // Rechunk with default size
    /// table.rechunk(RechunkStrategy::Auto)?;
    ///
    /// // Target 64KB per batch
    /// table.rechunk(RechunkStrategy::Memory(65536))?;
    /// ```
    pub fn rechunk(&mut self, strategy: RechunkStrategy) -> Result<(), MinarrowError> {
        if self.batches.is_empty() || self.n_rows == 0 {
            return Ok(());
        }

        // Determine chunk size based on strategy
        let chunk_size = match strategy {
            RechunkStrategy::Count(size) => {
                if size == 0 {
                    return Err(MinarrowError::IndexError(
                        "Count chunk size must be greater than 0".to_string(),
                    ));
                }
                size
            }
            RechunkStrategy::Auto => 8192,
            #[cfg(feature = "size")]
            RechunkStrategy::Memory(bytes_per_chunk) => {
                let total_bytes = self.est_bytes();
                let total_rows = self.n_rows;

                if total_bytes == 0 {
                    return Err(MinarrowError::IndexError(
                        "Cannot rechunk: table has 0 estimated bytes".to_string(),
                    ));
                }

                ((bytes_per_chunk * total_rows) / total_bytes).max(1)
            }
        };

        // Fast path: single batch already at target size
        if self.batches.len() == 1 && self.batches[0].n_rows == chunk_size {
            return Ok(());
        }

        let mut new_batches = Vec::new();
        let mut accumulator: Option<Table> = None;

        // Process each existing batch
        for batch_arc in self.batches.drain(..) {
            let batch = Arc::try_unwrap(batch_arc).unwrap_or_else(|arc| (*arc).clone());
            let mut remaining = batch;

            while remaining.n_rows > 0 {
                if let Some(ref mut acc) = accumulator {
                    let acc_rows = acc.n_rows;
                    let needed = chunk_size - acc_rows;

                    if remaining.n_rows <= needed {
                        // Entire remaining batch fits in accumulator
                        *acc = acc.clone().concat(remaining)?;

                        // If accumulator is now full, emit it
                        if acc.n_rows == chunk_size {
                            new_batches.push(Arc::new(accumulator.take().unwrap()));
                        }
                        break; // consumed remaining
                    } else {
                        // Split remaining batch to complete accumulator
                        let split_result = remaining.split(needed)?;
                        let mut parts = split_result.batches;
                        let to_add =
                            Arc::try_unwrap(parts.remove(0)).unwrap_or_else(|arc| (*arc).clone());
                        remaining =
                            Arc::try_unwrap(parts.remove(0)).unwrap_or_else(|arc| (*arc).clone());

                        // Complete and emit the accumulator
                        *acc = acc.clone().concat(to_add)?;
                        new_batches.push(Arc::new(accumulator.take().unwrap()));
                    }
                } else {
                    // No accumulator - start processing remaining
                    if remaining.n_rows == chunk_size {
                        // Exact fit - use remaining as-is
                        new_batches.push(Arc::new(remaining));
                        break;
                    } else if remaining.n_rows > chunk_size {
                        // Split off one chunk_size portion
                        let split_result = remaining.split(chunk_size)?;
                        let mut parts = split_result.batches;
                        new_batches.push(parts.remove(0));
                        remaining =
                            Arc::try_unwrap(parts.remove(0)).unwrap_or_else(|arc| (*arc).clone());
                    } else {
                        // Remaining becomes new accumulator
                        accumulator = Some(remaining);
                        break;
                    }
                }
            }
        }

        // Emit any remaining accumulator as final batch
        if let Some(final_batch) = accumulator {
            new_batches.push(Arc::new(final_batch));
        }

        self.batches = new_batches;
        Ok(())
    }

    /// Rechunks only the first `up_to_row` rows, leaving the rest untouched.
    ///
    /// This is useful for streaming scenarios where new data is being appended
    /// and you want to rechunk stable data while leaving recent additions alone.
    ///
    /// # Arguments
    /// * `up_to_row` - Rechunk only rows before this index
    /// * `strategy` - The rechunking strategy to use
    ///
    /// # Errors
    /// - Returns `IndexError` if `up_to_row` is greater than total row count
    /// - Returns same errors as `rechunk()` for invalid strategies
    ///
    /// # Example
    /// ```ignore
    /// // Rechunk first 1000 rows, leave the rest untouched
    /// table.rechunk_to(1000, RechunkStrategy::Count(512))?;
    /// ```
    pub fn rechunk_to(
        &mut self,
        up_to_row: usize,
        strategy: RechunkStrategy,
    ) -> Result<(), MinarrowError> {
        let total_rows = self.n_rows;

        if up_to_row > total_rows {
            return Err(MinarrowError::IndexError(format!(
                "rechunk_to row {} out of bounds for table with {} rows",
                up_to_row, total_rows
            )));
        }

        if up_to_row == 0 || self.batches.is_empty() {
            return Ok(());
        }

        if up_to_row == total_rows {
            // Rechunk everything
            return self.rechunk(strategy);
        }

        // Find which batches contain the data up to up_to_row
        let mut current_offset = 0;
        let mut split_point = 0;

        for (i, batch) in self.batches.iter().enumerate() {
            let batch_end = current_offset + batch.n_rows;
            if batch_end > up_to_row {
                split_point = i;
                break;
            }
            current_offset = batch_end;
        }

        // Extract batches to rechunk and batches to keep
        let mut to_rechunk = self.batches.drain(..=split_point).collect::<Vec<_>>();
        let keep_batches = self.batches.drain(..).collect::<Vec<_>>();

        // If the split batch needs to be divided
        if current_offset < up_to_row {
            let split_batch_arc = to_rechunk.pop().unwrap();
            let split_batch = Arc::try_unwrap(split_batch_arc).unwrap_or_else(|arc| (*arc).clone());
            let split_at = up_to_row - current_offset;

            let split_result = split_batch.split(split_at)?;
            let mut parts = split_result.batches;
            to_rechunk.push(parts.remove(0));
            self.batches.push(parts.remove(0));
        }

        // Rechunk the selected portion
        self.batches.extend(keep_batches);
        // from_batches infers schema from the batches, second param is name
        let mut temp = SuperTable::from_batches(to_rechunk.into(), Some(self.name.clone()));
        temp.rechunk(strategy)?;

        // Reconstruct rechunked portion + untouched portion
        let mut result = temp.batches;
        result.extend(self.batches.drain(..));
        self.batches = result;

        // Recalculate n_rows
        self.n_rows = self.batches.iter().map(|b| b.n_rows).sum();

        Ok(())
    }
}

impl Default for SuperTable {
    fn default() -> Self {
        Self::new("Unnamed".into())
    }
}

impl FromIterator<Table> for SuperTable {
    fn from_iter<T: IntoIterator<Item = Table>>(iter: T) -> Self {
        let batches: Vec<Arc<Table>> = iter.into_iter().map(|x| x.into()).collect();
        SuperTable::from_batches(batches, None)
    }
}

impl From<SuperTable> for Vec<Table> {
    fn from(super_table: SuperTable) -> Self {
        super_table
            .batches
            .into_iter()
            .map(|arc| Arc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone()))
            .collect()
    }
}

impl Shape for SuperTable {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank2 {
            rows: self.n_rows(),
            cols: self.n_cols(),
        }
    }
}

impl Consolidate for SuperTable {
    type Output = Table;

    /// Consolidates all batches into a single contiguous `Table`.
    ///
    /// Materialises all rows from all batches into one table.
    /// Use this when you need contiguous memory for operations or
    /// APIs that require single buffers.
    ///
    /// Uses `self.name` for the resulting table. Rename afterwards if needed.
    ///
    /// When the `arena` feature is enabled, all column buffers are written
    /// into a single allocation then sliced into typed views, reducing
    /// allocation count from O(columns) to O(1). The resulting buffers
    /// are SharedBuffer-backed; mutations trigger copy-on-write.
    ///
    /// Without the `arena` feature, falls back to per-column concat.
    ///
    /// # Panics
    /// A zero-batch SuperTable consolidates to a zero-row Table carrying
    /// the declared schema; an empty schema yields a column-less Table.
    fn consolidate(self) -> Table {
        if self.batches.is_empty() {
            let cols: Vec<FieldArray> = self
                .schema
                .iter()
                .map(|field| {
                    let array = crate::Array::from_arrow_dtype(&field.dtype);
                    FieldArray::new_arc(field.clone(), array)
                })
                .collect();
            return Table::build(cols, 0, self.name);
        }
        #[cfg(feature = "arena")]
        {
            self.consolidate_arena()
        }
        #[cfg(not(feature = "arena"))]
        {
            self.consolidate_concat()
        }
    }
}

impl SuperTable {
    /// Concat-based consolidation: appends batches per column via `concat_array`.
    /// The zero-batch path is handled at the trait wrapper above.
    #[cfg_attr(feature = "arena", allow(dead_code))]
    fn consolidate_concat(self) -> Table {
        let n_cols = self.schema.len();
        let mut unified_cols = Vec::with_capacity(n_cols);

        for col_idx in 0..n_cols {
            let field = self.schema[col_idx].clone();
            let mut arr = self.batches[0].cols[col_idx].array.clone();
            for batch in self.batches.iter().skip(1) {
                arr.concat_array(&batch.cols[col_idx].array);
            }
            let null_count = arr.null_count();
            unified_cols.push(FieldArray {
                field,
                array: arr.clone(),
                null_count,
            });
        }

        #[allow(unused_mut)]
        let mut table = Table::build(unified_cols, self.n_rows, self.name);
        #[cfg(feature = "table_metadata")]
        {
            table.metadata = self.batches[0].metadata.clone();
        }
        table
    }

    /// Arena-based consolidation: writes all column buffers into a single
    /// allocation, then slices typed views from the frozen SharedBuffer.
    /// Reduces allocation count from O(columns) to O(1). The zero-batch
    /// path is handled at the trait wrapper above.
    #[cfg(feature = "arena")]
    fn consolidate_arena(self) -> Table {
        // Fast path: single batch, no copying needed
        if self.batches.len() == 1 {
            let batch = self.batches.into_iter().next().unwrap();
            let mut table = Arc::try_unwrap(batch).unwrap_or_else(|arc| (*arc).clone());
            table.name = self.name;
            return table;
        }

        let refs: Vec<&Table> = self.batches.iter().map(|b| b.as_ref()).collect();
        crate::structs::arena::consolidate_tables_arena(&refs, self.name)
    }
}

impl Concatenate for SuperTable {
    /// Concatenates two SuperTables by appending all batches from `other` to `self`.
    ///
    /// # Requirements
    /// - Both SuperTables must have the same schema (column names and types)
    ///
    /// # Returns
    /// A new SuperTable containing all batches from `self` followed by all batches from `other`
    ///
    /// # Errors
    /// - `IncompatibleTypeError` if schemas don't match
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        // If both are empty, return empty
        if self.batches.is_empty() && other.batches.is_empty() {
            return Ok(SuperTable::new(format!("{}+{}", self.name, other.name)));
        }

        // If one is empty, return the other
        if self.batches.is_empty() {
            let mut result = other;
            result.name = format!("{}+{}", self.name, result.name);
            return Ok(result);
        }
        if other.batches.is_empty() {
            let mut result = self;
            result.name = format!("{}+{}", result.name, other.name);
            return Ok(result);
        }

        // Validate schemas match
        if self.schema.len() != other.schema.len() {
            return Err(MinarrowError::IncompatibleTypeError {
                from: "SuperTable",
                to: "SuperTable",
                message: Some(format!(
                    "Cannot concatenate SuperTables with different column counts: {} vs {}",
                    self.schema.len(),
                    other.schema.len()
                )),
            });
        }

        // Check schema compatibility field by field
        for (col_idx, (self_field, other_field)) in
            self.schema.iter().zip(other.schema.iter()).enumerate()
        {
            if self_field.name != other_field.name {
                return Err(MinarrowError::IncompatibleTypeError {
                    from: "SuperTable",
                    to: "SuperTable",
                    message: Some(format!(
                        "Column {} name mismatch: '{}' vs '{}'",
                        col_idx, self_field.name, other_field.name
                    )),
                });
            }

            if self_field.dtype != other_field.dtype {
                return Err(MinarrowError::IncompatibleTypeError {
                    from: "SuperTable",
                    to: "SuperTable",
                    message: Some(format!(
                        "Column '{}' type mismatch: {:?} vs {:?}",
                        self_field.name, self_field.dtype, other_field.dtype
                    )),
                });
            }

            if self_field.nullable != other_field.nullable {
                return Err(MinarrowError::IncompatibleTypeError {
                    from: "SuperTable",
                    to: "SuperTable",
                    message: Some(format!(
                        "Column '{}' nullable mismatch: {} vs {}",
                        self_field.name, self_field.nullable, other_field.nullable
                    )),
                });
            }
        }

        // Concatenate batches
        let mut result_batches = self.batches;
        result_batches.extend(other.batches);
        let total_rows = self.n_rows + other.n_rows;

        #[cfg_attr(not(feature = "shared_dict"), allow(unused_mut))]
        let mut st = SuperTable {
            batches: result_batches,
            schema: self.schema,
            n_rows: total_rows,
            name: format!("{}+{}", self.name, other.name),
            #[cfg(feature = "shared_dict")]
            category_managers: Vec::new(),
        };
        #[cfg(feature = "shared_dict")]
        st.rebuild_category_managers();
        Ok(st)
    }
}

impl Display for SuperTable {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "SuperTable \"{}\" [{} rows, {} columns, {} batches]",
            self.name,
            self.n_rows,
            self.schema.len(),
            self.batches.len()
        )?;

        for (batch_idx, batch) in self.batches.iter().enumerate() {
            writeln!(
                f,
                "  ├─ Batch {batch_idx}: {} rows, {} columns",
                batch.n_rows,
                batch.n_cols()
            )?;
            for (col_idx, col) in batch.cols.iter().enumerate() {
                let indent = "    │ ";
                writeln!(
                    f,
                    "{indent}Col {col_idx}: \"{}\" (dtype: {}, nulls: {})",
                    col.field.name, col.field.dtype, col.null_count
                )?;
                for line in format!("{}", col.array).lines() {
                    writeln!(f, "{indent}  {line}")?;
                }
            }
        }

        Ok(())
    }
}

// ===========================================================
// Apache Arrow / Polars bridges for SuperTable
// ===========================================================

impl SuperTable {
    /// Export each batch as an arrow-rs `RecordBatch`.
    ///
    /// Panics on FFI failure. For a fallible variant, see
    /// [`SuperTable::try_to_apache_arrow`].
    #[cfg(feature = "cast_arrow")]
    #[inline]
    pub fn to_apache_arrow(&self) -> Vec<arrow::array::RecordBatch> {
        self.try_to_apache_arrow()
            .expect("SuperTable::to_apache_arrow failed")
    }

    /// Fallible variant of [`SuperTable::to_apache_arrow`].
    #[cfg(feature = "cast_arrow")]
    pub fn try_to_apache_arrow(
        &self,
    ) -> Result<Vec<arrow::array::RecordBatch>, MinarrowError> {
        let mut out = Vec::with_capacity(self.batches.len());
        for batch in &self.batches {
            out.push(batch.try_to_apache_arrow()?);
        }
        Ok(out)
    }

    /// Build a polars `DataFrame` whose columns are chunked Series mirroring
    /// the SuperTable's batches.
    ///
    /// Panics on FFI failure. For a fallible variant, see
    /// [`SuperTable::try_to_polars`].
    #[cfg(feature = "cast_polars")]
    #[inline]
    pub fn to_polars(&self) -> polars::frame::DataFrame {
        self.try_to_polars()
            .expect("SuperTable::to_polars failed")
    }

    /// Fallible variant of [`SuperTable::to_polars`].
    #[cfg(feature = "cast_polars")]
    pub fn try_to_polars(
        &self,
    ) -> Result<polars::frame::DataFrame, MinarrowError> {
        use polars::prelude::Column;
        if self.batches.is_empty() {
            // Build an empty DataFrame matching the schema.
            return Ok(polars::frame::DataFrame::default());
        }

        let n_cols = self.batches[0].n_cols();
        let mut col_series: Vec<polars::prelude::Series> =
            Vec::with_capacity(n_cols);

        // Per column, fold per-batch Series via `append` so chunks survive.
        for col_idx in 0..n_cols {
            let mut iter = self.batches.iter();
            let first_batch = iter.next().unwrap();
            let mut acc = first_batch.cols[col_idx].try_to_polars()?;
            for batch in iter {
                let s = batch.cols[col_idx].try_to_polars()?;
                acc.append(&s)?;
            }
            col_series.push(acc);
        }

        let cols: Vec<Column> = col_series
            .into_iter()
            .map(|s| Column::new(s.name().clone(), s))
            .collect();
        Ok(polars::frame::DataFrame::new(self.n_rows, cols)?)
    }

    /// Build a `SuperTable` from a slice of arrow-rs `RecordBatch` values.
    /// All batches must share the same schema.
    ///
    /// Panics on FFI failure. For a fallible variant, see
    /// [`SuperTable::try_from_apache_arrow`].
    #[cfg(feature = "cast_arrow")]
    #[inline]
    pub fn from_apache_arrow(
        batches: &[arrow::array::RecordBatch],
    ) -> SuperTable {
        Self::try_from_apache_arrow(batches)
            .expect("SuperTable::from_apache_arrow failed")
    }

    /// Fallible variant of [`SuperTable::from_apache_arrow`].
    #[cfg(feature = "cast_arrow")]
    pub fn try_from_apache_arrow(
        batches: &[arrow::array::RecordBatch],
    ) -> Result<SuperTable, MinarrowError> {
        if batches.is_empty() {
            return Ok(SuperTable::new(String::new()));
        }
        let mut tables = Vec::with_capacity(batches.len());
        for rb in batches {
            tables.push(Arc::new(Table::try_from_apache_arrow(rb)?));
        }
        Ok(SuperTable::from_batches(tables, None))
    }

    /// Build a `SuperTable` from a Polars `DataFrame`, preserving per-column
    /// chunk boundaries as separate batches.
    ///
    /// All columns must share the same number of chunks for this to map cleanly
    /// to batches; if chunks are misaligned across columns we re-chunk the
    /// DataFrame to a single chunk first as a fallback.
    ///
    /// ## Performance note
    /// Polars data is typically 8-byte aligned (per the Arrow spec default),
    /// while Minarrow uses 64-byte aligned `Vec64<T>` buffers for SIMD.
    /// Most of the time this results in a memory copy per chunk to realign
    /// on import, unless the source data happens to be pre-aligned to 64
    /// bytes. The FFI hand-off itself is pointer-level zero-copy; the
    /// realignment is done by `Buffer::from_shared` when the source isn't
    /// 64-byte aligned. No consolidation copy is performed unless the
    /// chunk-misalignment fallback path is taken.
    ///
    /// Panics on FFI failure. For a fallible variant, see
    /// [`SuperTable::try_from_polars`].
    #[cfg(feature = "cast_polars")]
    #[inline]
    pub fn from_polars(df: &polars::frame::DataFrame) -> SuperTable {
        Self::try_from_polars(df)
            .expect("SuperTable::from_polars failed")
    }

    /// Fallible variant of [`SuperTable::from_polars`].
    #[cfg(feature = "cast_polars")]
    pub fn try_from_polars(
        df: &polars::frame::DataFrame,
    ) -> Result<SuperTable, MinarrowError> {
        let columns = df.columns();
        if columns.is_empty() {
            return Ok(SuperTable::new(String::new()));
        }

        // Materialise each column to a Series and read its chunk count.
        let series: Vec<&polars::prelude::Series> = columns
            .iter()
            .map(|c| c.as_materialized_series())
            .collect();
        let n_chunks = series[0].n_chunks();
        let aligned = series.iter().all(|s| s.n_chunks() == n_chunks);

        if !aligned {
            // Misaligned chunks: fall back to a single-batch SuperTable built
            // from a rechunked DataFrame. We rebuild via Table::from_polars
            // which calls per-column FieldArray::try_from_polars (which itself
            // uses chunk 0; for misaligned multi-chunk Series, we rechunk).
            let mut rechunked = df.clone();
            rechunked.align_chunks_par();
            let table = Table::try_from_polars(&rechunked)?;
            return Ok(SuperTable::from_batches(vec![Arc::new(table)], None));
        }

        // Aligned: build one Table per chunk index by importing the i'th
        // chunk of each Series directly through the polars bridge.
        use polars::prelude::CompatLevel;
        let mut batches = Vec::with_capacity(n_chunks);
        for chunk_idx in 0..n_chunks {
            let mut cols = Vec::with_capacity(series.len());
            for s in &series {
                let arr2 = s.to_arrow(chunk_idx, CompatLevel::oldest());
                let (array_arc, field) = crate::ffi::polars::import_chunk(
                    s.name().as_str(),
                    s.null_count() > 0,
                    arr2,
                )?;
                let array =
                    Arc::try_unwrap(array_arc).unwrap_or_else(|arc| (*arc).clone());
                cols.push(FieldArray::new(field, array));
            }
            batches.push(Arc::new(Table::new(String::new(), Some(cols))));
        }
        Ok(SuperTable::from_batches(batches, None))
    }
}

#[cfg(feature = "cast_arrow")]
impl From<&[arrow::array::RecordBatch]> for SuperTable {
    fn from(batches: &[arrow::array::RecordBatch]) -> Self {
        SuperTable::from_apache_arrow(batches)
    }
}

#[cfg(feature = "cast_polars")]
impl From<&polars::frame::DataFrame> for SuperTable {
    fn from(df: &polars::frame::DataFrame) -> Self {
        SuperTable::from_polars(df)
    }
}

#[cfg(feature = "views")]
impl From<SuperTableV> for SuperTable {
    fn from(super_table_v: SuperTableV) -> Self {
        if super_table_v.is_empty() {
            return SuperTable::new("".to_string());
        }
        SuperTable::from_views(&super_table_v.slices, "SuperTable".to_string())
    }
}

/// Ergonomic constructor for a [`SuperTable`] from named table batches.
///
/// Each batch argument may be a `Table` or `Arc<Table>`; both flow
/// through `.into()` into the required `Arc<Table>`. When the first
/// argument is a string literal it becomes the SuperTable name;
/// otherwise the first batch's name is used.
///
/// # Forms
/// - `st!("name", batch1, batch2, ...)` - named SuperTable from batches.
/// - `st!(batch1, batch2, ...)` - name derived from the first batch.
/// - `st!("name")` - empty SuperTable with the supplied name.
///
/// # Example
/// ```
/// use minarrow::{fa_i32, st, tbl};
///
/// let a = tbl!("batch", fa_i32!("x", 1, 2));
/// let b = tbl!("batch", fa_i32!("x", 3, 4));
/// let s = st!("rolling", a, b);
/// assert_eq!(s.name, "rolling");
/// assert_eq!(s.n_rows, 4);
/// ```
///
/// # Note
/// The named form requires a string literal for the name. For a
/// dynamic `String` name use `SuperTable::from_batches(..., Some(name))`
/// directly.
#[macro_export]
macro_rules! st {
    ($name:literal, $($t:expr),+ $(,)?) => {
        $crate::SuperTable::from_batches(
            ::std::vec::Vec::from([$(::std::convert::Into::<::std::sync::Arc<$crate::Table>>::into($t)),+]),
            ::std::option::Option::Some(::std::string::String::from($name)),
        )
    };
    ($name:literal) => {
        $crate::SuperTable::new(::std::string::String::from($name))
    };
    ($($t:expr),+ $(,)?) => {
        $crate::SuperTable::from_batches(
            ::std::vec::Vec::from([$(::std::convert::Into::<::std::sync::Arc<$crate::Table>>::into($t)),+]),
            ::std::option::Option::None,
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::arrow_dtype::ArrowType;
    use crate::{fa_bool, fa_f64, fa_i32, fa_i64, fa_str32};
    #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
    use crate::fa_cat32;
    use crate::{Array, Field, FieldArray, MaskedArray, NumericArray, Table};

    fn table(cols: Vec<FieldArray>) -> Table {
        let n_rows = cols[0].len();
        for c in &cols {
            assert_eq!(c.len(), n_rows, "all columns must have same len for Table");
        }
        Table::build(cols, n_rows, "batch".to_string())
    }

    #[test]
    fn test_empty_and_default() {
        let t = SuperTable::default();
        assert!(t.is_empty());
        assert_eq!(t.n_cols(), 0);
        assert_eq!(t.n_batches(), 0);
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn test_from_batches_basic() {
        let col1 = fa_i32!("a", 1, 2, 3);
        let col2 = fa_i32!("b", 10, 11, 12);
        let col3 = fa_i32!("a", 4, 5);
        let col4 = fa_i32!("b", 13, 14);
        let batch1 = Arc::new(table(vec![col1.clone(), col2.clone()]));
        let batch2 = Arc::new(table(vec![col3.clone(), col4.clone()]));
        let batches = vec![batch1, batch2].into();

        let t = SuperTable::from_batches(batches, None);
        assert_eq!(t.n_cols(), 2);
        assert_eq!(t.n_batches(), 2);
        assert_eq!(t.len(), 5);
        assert_eq!(t.schema()[0].name, "a");
        assert_eq!(t.schema()[1].name, "b");
        assert_eq!(t.batches()[0].cols[0], col1);
        assert_eq!(t.batches()[1].cols[1], col4);
    }

    #[test]
    #[should_panic(expected = "column-count mismatch")]
    fn test_from_batches_col_count_mismatch() {
        let batch1 = Arc::new(table(vec![fa_i32!("a", 1, 2)]));
        let batch2 = Arc::new(table(vec![fa_i32!("a", 3, 4), fa_i32!("b", 5, 6)]));
        SuperTable::from_batches(vec![batch1, batch2].into(), None);
    }

    #[test]
    #[should_panic(expected = "schema mismatch")]
    fn test_from_batches_schema_mismatch() {
        let batch1 = Arc::new(table(vec![fa_i32!("a", 1, 2)]));
        let mut wrong = fa_i32!("z", 3, 4);
        let mut mismatched_field = (*wrong.field).clone();
        mismatched_field.dtype = ArrowType::Int32;
        wrong.field = Arc::new(mismatched_field);
        let batch2 = Arc::new(table(vec![wrong]));
        SuperTable::from_batches(vec![batch1, batch2].into(), None);
    }

    #[test]
    fn test_push_and_consolidate() {
        let mut t = SuperTable::default();
        t.push(Arc::new(table(vec![fa_i32!("x", 1, 2), fa_i32!("y", 3, 4)])));
        t.push(Arc::new(table(vec![fa_i32!("x", 5), fa_i32!("y", 6)])));
        assert_eq!(t.n_cols(), 2);
        assert_eq!(t.n_batches(), 2);
        assert_eq!(t.len(), 3);
        let tab = t.consolidate();
        // consolidate() uses self.name; default SuperTable name is "Unnamed"
        assert_eq!(tab.name, "Unnamed");
        assert_eq!(tab.n_rows, 3);
        assert_eq!(tab.cols[0].field.name, "x");
        assert_eq!(tab.cols[1].field.name, "y");
    }

    #[test]
    #[should_panic(expected = "column-count mismatch")]
    fn test_push_col_count_mismatch() {
        let mut t = SuperTable::default();
        t.push(Arc::new(table(vec![fa_i32!("a", 1, 2)])));
        t.push(Arc::new(table(vec![fa_i32!("a", 3, 4), fa_i32!("b", 5, 6)])));
    }

    #[cfg(feature = "views")]
    #[test]
    fn test_slice_and_owned_table() {
        let batch1 = Arc::new(table(vec![fa_i32!("q", 1, 2, 3), fa_i32!("w", 4, 5, 6)]));
        let batch2 = Arc::new(table(vec![fa_i32!("q", 7, 8), fa_i32!("w", 9, 10)]));
        let t = SuperTable::from_batches(vec![batch1, batch2].into(), None);

        // Slice rows 2..5 (3 rows), crossing the batch boundary
        let slice = t.view(2, 3);
        assert_eq!(slice.len, 3);
        assert_eq!(slice.slices.len(), 2);

        let owned = slice.consolidate();
        assert_eq!(owned.name, "part");
        assert_eq!(owned.n_rows, 3);
        assert_eq!(owned.cols[0].field.name, "q");
        assert_eq!(owned.cols[1].field.name, "w");

        let arr = &owned.cols[0].array;
        if let Array::NumericArray(NumericArray::Int32(ints)) = arr {
            assert_eq!(ints.data.as_slice(), &[3, 7, 8]);
        } else {
            panic!("expected Int32 array");
        }

        let arr = &owned.cols[1].array;
        if let Array::NumericArray(NumericArray::Int32(ints)) = arr {
            assert_eq!(ints.data.as_slice(), &[6, 9, 10]);
        } else {
            panic!("expected Int32 array");
        }
    }

    #[test]
    fn test_schema_and_batch_access() {
        let t = SuperTable::from_batches(vec![Arc::new(table(vec![fa_i32!("alpha", 1, 2)]))], None);
        assert_eq!(t.n_cols(), 1);
        assert_eq!(t.schema()[0].name, "alpha");
        assert!(t.batch(0).is_some());
        assert!(t.batch(5).is_none());
        assert_eq!(t.batches().len(), 1);
    }

    #[cfg(feature = "views")]
    #[test]
    fn test_from_slices() {
        let batch1 = Arc::new(table(vec![fa_i32!("x", 1, 2), fa_i32!("y", 3, 4)]));
        let batch2 = Arc::new(table(vec![fa_i32!("x", 5, 6), fa_i32!("y", 7, 8)]));
        let t = SuperTable::from_batches(vec![batch1.clone(), batch2.clone()], None);

        // Break into 4 slices of 1 row each
        let mut table_slices = Vec::new();
        for i in 0..t.len() {
            let bts = t.view(i, 1);
            for ts in bts.slices.clone() {
                table_slices.push(ts);
            }
        }

        // Reconstruct a new batched table
        let rebuilt = SuperTable::from_views(&table_slices, "rebuilt".to_string());

        assert_eq!(rebuilt.n_cols(), t.n_cols());
        assert_eq!(rebuilt.len(), t.len());

        // Schema should match
        for (left, right) in rebuilt.schema().iter().zip(t.schema()) {
            assert_eq!(left.name, right.name);
            assert_eq!(left.dtype, right.dtype);
        }

        // Validate data for each column
        let expected_x = [1, 2, 5, 6];
        let expected_y = [3, 4, 7, 8];
        let consolidated = rebuilt.consolidate();
        for (col_idx, expected) in [expected_x.as_slice(), expected_y.as_slice()]
            .iter()
            .enumerate()
        {
            let arr = consolidated.cols[col_idx].array.clone();
            if let Array::NumericArray(NumericArray::Int32(ints)) = arr {
                assert_eq!(ints.data.as_slice(), *expected);
            } else {
                panic!("unexpected array type at col {col_idx}");
            }
        }
    }

    #[test]
    fn test_insert_rows_into_first_batch() {
        let batch1 = Arc::new(table(vec![fa_i32!("a", 1, 2, 3), fa_i32!("b", 10, 20, 30)]));
        let batch2 = Arc::new(table(vec![fa_i32!("a", 4, 5), fa_i32!("b", 40, 50)]));
        let mut st = SuperTable::from_batches(vec![batch1, batch2], None);

        let insert_batch = Arc::new(table(vec![fa_i32!("a", 99), fa_i32!("b", 88)]));
        let insert_st = SuperTable::from_batches(vec![insert_batch], None);

        st.insert_rows(1, insert_st).unwrap();

        assert_eq!(st.n_rows(), 6);
        assert_eq!(st.n_batches(), 4);

        let materialised = st.consolidate();
        match &materialised.cols[0].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.data.as_slice(), &[1, 99, 2, 3, 4, 5]);
            }
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn test_insert_rows_into_second_batch() {
        let batch1 = Arc::new(table(vec![fa_i32!("a", 1, 2)]));
        let batch2 = Arc::new(table(vec![fa_i32!("a", 3, 4, 5)]));
        let mut st = SuperTable::from_batches(vec![batch1, batch2], None);

        let insert_batch = Arc::new(table(vec![fa_i32!("a", 99, 88)]));
        let insert_st = SuperTable::from_batches(vec![insert_batch], None);

        st.insert_rows(3, insert_st).unwrap();

        assert_eq!(st.n_rows(), 7);

        let materialised = st.consolidate();
        match &materialised.cols[0].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.data.as_slice(), &[1, 2, 3, 99, 88, 4, 5]);
            }
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn test_insert_rows_prepend() {
        let batch1 = Arc::new(table(vec![fa_i32!("a", 1, 2, 3)]));
        let mut st = SuperTable::from_batches(vec![batch1], None);

        let insert_batch = Arc::new(table(vec![fa_i32!("a", 99)]));
        let insert_st = SuperTable::from_batches(vec![insert_batch], None);

        st.insert_rows(0, insert_st).unwrap();

        assert_eq!(st.n_rows(), 4);

        let materialised = st.consolidate();
        match &materialised.cols[0].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.data.as_slice(), &[99, 1, 2, 3]);
            }
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn test_insert_rows_append() {
        let batch1 = Arc::new(table(vec![fa_i32!("a", 1, 2)]));
        let batch2 = Arc::new(table(vec![fa_i32!("a", 3, 4)]));
        let mut st = SuperTable::from_batches(vec![batch1, batch2], None);

        let insert_batch = Arc::new(table(vec![fa_i32!("a", 99)]));
        let insert_st = SuperTable::from_batches(vec![insert_batch], None);

        st.insert_rows(4, insert_st).unwrap();

        assert_eq!(st.n_rows(), 5);

        let materialised = st.consolidate();
        match &materialised.cols[0].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.data.as_slice(), &[1, 2, 3, 4, 99]);
            }
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn test_insert_rows_schema_mismatch() {
        let batch1 = Arc::new(table(vec![fa_i32!("a", 1, 2)]));
        let mut st = SuperTable::from_batches(vec![batch1], None);

        let wrong_batch = Arc::new(table(vec![fa_i32!("b", 99)]));
        let wrong_st = SuperTable::from_batches(vec![wrong_batch], None);

        let result = st.insert_rows(0, wrong_st);
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_rows_out_of_bounds() {
        let batch1 = Arc::new(table(vec![fa_i32!("a", 1, 2)]));
        let mut st = SuperTable::from_batches(vec![batch1], None);

        let insert_batch = Arc::new(table(vec![fa_i32!("a", 99)]));
        let insert_st = SuperTable::from_batches(vec![insert_batch], None);

        let result = st.insert_rows(10, insert_st);
        assert!(result.is_err());
    }

    #[test]
    fn test_rechunk_uniform() {
        // Create a SuperTable with 3 batches of varying sizes
        let batch1 = Arc::new(table(vec![fa_i32!("x", 1, 2, 3), fa_i32!("y", 10, 20, 30)]));
        let batch2 = Arc::new(table(vec![
            fa_i32!("x", 4, 5, 6, 7),
            fa_i32!("y", 40, 50, 60, 70),
        ]));
        let batch3 = Arc::new(table(vec![fa_i32!("x", 8, 9), fa_i32!("y", 80, 90)]));
        let mut st = SuperTable::from_batches(vec![batch1, batch2, batch3], None);

        // Total rows: 3 + 4 + 2 = 9 rows
        assert_eq!(st.n_rows(), 9);
        assert_eq!(st.n_batches(), 3);

        // Rechunk to batches of 4 rows each
        st.rechunk(RechunkStrategy::Count(4)).unwrap();

        // Should have 3 batches: [4 rows, 4 rows, 1 row]
        assert_eq!(st.n_batches(), 3);
        assert_eq!(st.batch(0).unwrap().n_rows, 4);
        assert_eq!(st.batch(1).unwrap().n_rows, 4);
        assert_eq!(st.batch(2).unwrap().n_rows, 1);
        assert_eq!(st.n_rows(), 9);
    }

    #[test]
    fn test_rechunk_auto() {
        // Create a SuperTable with many rows spread across small batches
        let mut batches = Vec::new();
        for i in 0..100 {
            let vals: Vec<i32> = vec![i * 10, i * 10 + 1];
            let arr = Array::from_int32(crate::IntegerArray::<i32>::from_slice(&vals));
            let field = Field::new("col", ArrowType::Int32, false, None);
            batches.push(Arc::new(table(vec![FieldArray::new(field, arr)])));
        }
        let mut st = SuperTable::from_batches(batches.into(), None);

        // Total rows: 100 batches * 2 rows = 200 rows
        assert_eq!(st.n_rows(), 200);
        assert_eq!(st.n_batches(), 100);

        // Rechunk with Auto strategy (default 8192 rows per batch)
        st.rechunk(RechunkStrategy::Auto).unwrap();

        // Should consolidate to 1 batch since 200 < 8192
        assert_eq!(st.n_batches(), 1);
        assert_eq!(st.batch(0).unwrap().n_rows, 200);
        assert_eq!(st.n_rows(), 200);
    }

    #[test]
    #[cfg(feature = "size")]
    fn test_rechunk_by_memory() {
        // Create a SuperTable with i32 data
        let batch1 = Arc::new(table(vec![fa_i32!("a", 1, 2, 3, 4), fa_i32!("b", 5, 6, 7, 8)]));
        let batch2 = Arc::new(table(vec![
            fa_i32!("a", 9, 10, 11, 12),
            fa_i32!("b", 13, 14, 15, 16),
        ]));
        let mut st = SuperTable::from_batches(vec![batch1, batch2], None);

        assert_eq!(st.n_rows(), 8);
        assert_eq!(st.n_batches(), 2);

        // Use a larger memory target to get predictable chunking
        // The actual byte size includes overhead beyond raw data
        st.rechunk(RechunkStrategy::Memory(64)).unwrap();

        // Should rechunk into batches
        assert!(st.n_batches() >= 1);
        assert_eq!(st.n_rows(), 8);

        // Verify data integrity after rechunking
        let materialized = st.consolidate();
        assert_eq!(materialized.n_rows, 8);
    }

    #[test]
    fn test_rechunk_uniform_zero_error() {
        let batch1 = Arc::new(table(vec![fa_i32!("x", 1, 2, 3)]));
        let mut st = SuperTable::from_batches(vec![batch1], None);

        let result = st.rechunk(RechunkStrategy::Count(0));
        assert!(result.is_err());
        if let Err(MinarrowError::IndexError(msg)) = result {
            assert!(msg.contains("Count chunk size must be greater than 0"));
        } else {
            panic!("Expected IndexError for zero chunk size");
        }
    }

    #[test]
    fn test_rechunk_empty_table() {
        let mut st = SuperTable::default();
        assert!(st.is_empty());

        // Rechunking an empty table should succeed and remain empty
        st.rechunk(RechunkStrategy::Auto).unwrap();
        assert!(st.is_empty());
        assert_eq!(st.n_batches(), 0);

        st.rechunk(RechunkStrategy::Count(10)).unwrap();
        assert!(st.is_empty());
        assert_eq!(st.n_batches(), 0);
    }

    #[test]
    fn test_rechunk_preserves_data_order() {
        // Create batches with sequential data
        let batch1 = Arc::new(table(vec![fa_i32!("num", 1, 2, 3)]));
        let batch2 = Arc::new(table(vec![fa_i32!("num", 4, 5, 6, 7)]));
        let batch3 = Arc::new(table(vec![fa_i32!("num", 8, 9)]));
        let mut st = SuperTable::from_batches(vec![batch1, batch2, batch3], None);

        assert_eq!(st.n_rows(), 9);

        // Rechunk with different size
        st.rechunk(RechunkStrategy::Count(5)).unwrap();

        // Materialize to verify order is preserved
        let materialized = st.consolidate();
        match &materialized.cols[0].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.data.as_slice(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
            }
            _ => panic!("Expected Int32 array"),
        }
    }

    // --- consolidate_arena tests ---

    #[test]
    fn test_consolidate_arena_integer_and_float() {
        let b1 = Arc::new(table(vec![
            fa_i32!("id", 1, 2, 3),
            fa_f64!("val", 1.5, 2.5, 3.5),
        ]));
        let b2 = Arc::new(table(vec![
            fa_i32!("id", 4, 5),
            fa_f64!("val", 4.5, 5.5),
        ]));
        let st = SuperTable::from_batches(vec![b1, b2], None);
        let result = st.consolidate();

        assert_eq!(result.n_rows, 5);
        match &result.cols[0].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.data.as_slice(), &[1, 2, 3, 4, 5]);
            }
            _ => panic!("Expected Int32"),
        }
        match &result.cols[1].array {
            Array::NumericArray(NumericArray::Float64(arr)) => {
                assert_eq!(arr.data.as_slice(), &[1.5, 2.5, 3.5, 4.5, 5.5]);
            }
            _ => panic!("Expected Float64"),
        }
    }

    #[test]
    fn test_consolidate_arena_string_columns() {
        let b1 = Arc::new(table(vec![fa_str32!("name", "hello", "world")]));
        let b2 = Arc::new(table(vec![fa_str32!("name", "foo", "bar", "baz")]));
        let st = SuperTable::from_batches(vec![b1, b2], None);
        let result = st.consolidate();

        assert_eq!(result.n_rows, 5);
        match &result.cols[0].array {
            Array::TextArray(crate::TextArray::String32(arr)) => {
                assert_eq!(
                    arr.offsets.len(),
                    6,
                    "offsets should have n_rows+1 elements"
                );
                assert_eq!(arr.len(), 5);
                assert_eq!(arr.get_str(0), Some("hello"));
                assert_eq!(arr.get_str(1), Some("world"));
                assert_eq!(arr.get_str(2), Some("foo"));
                assert_eq!(arr.get_str(3), Some("bar"));
                assert_eq!(arr.get_str(4), Some("baz"));
            }
            _ => panic!("Expected String32"),
        }
    }

    #[test]
    fn test_consolidate_arena_nullable_columns() {
        use crate::{Bitmask, Buffer, IntegerArray};

        let fa_nullable = |name: &str, vals: &[i32], nulls: &[bool]| -> FieldArray {
            let mask = Bitmask::from_bools(nulls);
            let arr = Array::from_int32(IntegerArray::<i32>::new(
                Buffer::from_slice(vals),
                Some(mask),
            ));
            FieldArray::new(
                Field::new(name.to_string(), ArrowType::Int32, true, None),
                arr,
            )
        };

        // batch 1: index 1 is null
        let b1 = Arc::new(table(vec![fa_nullable(
            "x",
            &[10, 0, 30],
            &[true, false, true],
        )]));
        // batch 2: index 0 is null
        let b2 = Arc::new(table(vec![fa_nullable("x", &[0, 50], &[false, true])]));
        let st = SuperTable::from_batches(vec![b1, b2], None);
        let result = st.consolidate();

        assert_eq!(result.n_rows, 5);
        match &result.cols[0].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.get(0), Some(10));
                assert_eq!(arr.get(1), None); // null
                assert_eq!(arr.get(2), Some(30));
                assert_eq!(arr.get(3), None); // null
                assert_eq!(arr.get(4), Some(50));
            }
            _ => panic!("Expected Int32"),
        }
    }

    #[test]
    fn test_consolidate_arena_boolean_columns() {
        let b1 = Arc::new(table(vec![fa_bool!("flag", true, false, true)]));
        let b2 = Arc::new(table(vec![fa_bool!("flag", false, true)]));
        let st = SuperTable::from_batches(vec![b1, b2], None);
        let result = st.consolidate();

        assert_eq!(result.n_rows, 5);
        match &result.cols[0].array {
            Array::BooleanArray(arr) => {
                assert_eq!(arr.data.get(0), true);
                assert_eq!(arr.data.get(1), false);
                assert_eq!(arr.data.get(2), true);
                assert_eq!(arr.data.get(3), false);
                assert_eq!(arr.data.get(4), true);
            }
            _ => panic!("Expected BooleanArray"),
        }
    }

    #[test]
    fn test_consolidate_arena_mixed_types() {
        let b1 = Arc::new(table(vec![
            fa_i64!("id", 1, 2),
            fa_f64!("score", 9.5, 8.0),
            fa_str32!("name", "alice", "bob"),
            fa_bool!("active", true, false),
        ]));
        let b2 = Arc::new(table(vec![
            fa_i64!("id", 3),
            fa_f64!("score", 7.0),
            fa_str32!("name", "charlie"),
            fa_bool!("active", true),
        ]));
        let st = SuperTable::from_batches(vec![b1, b2], None);
        let result = st.consolidate();

        assert_eq!(result.n_rows, 3);
        assert_eq!(result.cols.len(), 4);

        // Verify int64 column
        match &result.cols[0].array {
            Array::NumericArray(NumericArray::Int64(arr)) => {
                assert_eq!(arr.data.as_slice(), &[1i64, 2, 3]);
            }
            _ => panic!("Expected Int64"),
        }
        // Verify float64 column
        match &result.cols[1].array {
            Array::NumericArray(NumericArray::Float64(arr)) => {
                assert_eq!(arr.data.as_slice(), &[9.5, 8.0, 7.0]);
            }
            _ => panic!("Expected Float64"),
        }
        // Verify string column
        match &result.cols[2].array {
            Array::TextArray(crate::TextArray::String32(arr)) => {
                assert_eq!(arr.get_str(0), Some("alice"));
                assert_eq!(arr.get_str(1), Some("bob"));
                assert_eq!(arr.get_str(2), Some("charlie"));
            }
            _ => panic!("Expected String32"),
        }
        // Verify boolean column
        match &result.cols[3].array {
            Array::BooleanArray(arr) => {
                assert_eq!(arr.data.get(0), true);
                assert_eq!(arr.data.get(1), false);
                assert_eq!(arr.data.get(2), true);
            }
            _ => panic!("Expected BooleanArray"),
        }
    }

    #[test]
    #[cfg(feature = "arena")]
    fn test_consolidate_arena_shared_buffers() {
        let b1 = Arc::new(table(vec![fa_i32!("x", 1, 2, 3)]));
        let b2 = Arc::new(table(vec![fa_i32!("x", 4, 5)]));
        let st = SuperTable::from_batches(vec![b1, b2], None);
        let result = st.consolidate();

        // All output buffers should be SharedBuffer-backed
        match &result.cols[0].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert!(arr.data.is_shared(), "Data buffer should be shared");
            }
            _ => panic!("Expected Int32"),
        }
    }

    #[test]
    fn test_consolidate_arena_single_batch() {
        let batch = Arc::new(table(vec![fa_i32!("x", 10, 20, 30)]));
        let st = SuperTable::from_batches(vec![batch], None);
        let result = st.consolidate();

        assert_eq!(result.n_rows, 3);
        match &result.cols[0].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.data.as_slice(), &[10, 20, 30]);
            }
            _ => panic!("Expected Int32"),
        }
    }

    #[test]
    /// Verifies that arena-based and concat-based consolidation produce
    /// identical results for integer, float, and string columns.
    #[cfg(feature = "arena")]
    fn test_consolidate_arena_equivalence_with_consolidate() {
        // Build identical SuperTables for both paths
        let make_st = || {
            let b1 = Arc::new(table(vec![
                fa_i32!("id", 1, 2, 3),
                fa_f64!("val", 1.5, 2.5, 3.5),
                fa_str32!("label", "a", "bb", "ccc"),
            ]));
            let b2 = Arc::new(table(vec![
                fa_i32!("id", 4, 5),
                fa_f64!("val", 4.5, 5.5),
                fa_str32!("label", "dddd", "e"),
            ]));
            SuperTable::from_batches(vec![b1, b2], None)
        };

        let result_concat = make_st().consolidate_concat();
        let result_arena = make_st().consolidate_arena();

        assert_eq!(result_concat.n_rows, result_arena.n_rows);
        assert_eq!(result_concat.cols.len(), result_arena.cols.len());

        // Compare integer column
        match (&result_concat.cols[0].array, &result_arena.cols[0].array) {
            (
                Array::NumericArray(NumericArray::Int32(a)),
                Array::NumericArray(NumericArray::Int32(b)),
            ) => {
                assert_eq!(a.data.as_slice(), b.data.as_slice());
            }
            _ => panic!("Mismatched types for column 0"),
        }
        // Compare float column
        match (&result_concat.cols[1].array, &result_arena.cols[1].array) {
            (
                Array::NumericArray(NumericArray::Float64(a)),
                Array::NumericArray(NumericArray::Float64(b)),
            ) => {
                assert_eq!(a.data.as_slice(), b.data.as_slice());
            }
            _ => panic!("Mismatched types for column 1"),
        }
        // Compare string column
        match (&result_concat.cols[2].array, &result_arena.cols[2].array) {
            (
                Array::TextArray(crate::TextArray::String32(a)),
                Array::TextArray(crate::TextArray::String32(b)),
            ) => {
                for i in 0..5 {
                    assert_eq!(a.get_str(i), b.get_str(i), "String mismatch at index {i}");
                }
            }
            _ => panic!("Mismatched types for column 2"),
        }
    }

    #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
    #[test]
    fn test_consolidate_arena_categorical() {
        let b1 = Arc::new(table(vec![fa_cat32!("cat", "a", "b", "a")]));
        let b2 = Arc::new(table(vec![fa_cat32!("cat", "a", "b")]));
        let st = SuperTable::from_batches(vec![b1, b2], None);
        let result = st.consolidate();

        assert_eq!(result.n_rows, 5);
        match &result.cols[0].array {
            Array::TextArray(crate::TextArray::Categorical32(arr)) => {
                assert_eq!(arr.get_str(0), Some("a"));
                assert_eq!(arr.get_str(1), Some("b"));
                assert_eq!(arr.get_str(2), Some("a"));
                assert_eq!(arr.get_str(3), Some("a"));
                assert_eq!(arr.get_str(4), Some("b"));
            }
            _ => panic!("Expected Categorical32"),
        }
    }

    #[test]
    fn test_consolidate_arena_nullable_strings() {
        use crate::{Bitmask, StringArray};

        let fa_nullable_str = |name: &str, vals: &[&str], nulls: &[bool]| -> FieldArray {
            let mask = Bitmask::from_bools(nulls);
            let arr = Array::from_string32(StringArray::<u32>::from_vec(vals.to_vec(), Some(mask)));
            FieldArray::new(
                Field::new(name.to_string(), ArrowType::String, true, None),
                arr,
            )
        };

        let b1 = Arc::new(table(vec![fa_nullable_str(
            "s",
            &["hello", "", "world"],
            &[true, false, true],
        )]));
        let b2 = Arc::new(table(vec![fa_nullable_str(
            "s",
            &["", "bar"],
            &[false, true],
        )]));
        let st = SuperTable::from_batches(vec![b1, b2], None);
        let result = st.consolidate();

        assert_eq!(result.n_rows, 5);
        match &result.cols[0].array {
            Array::TextArray(crate::TextArray::String32(arr)) => {
                assert!(!arr.is_null(0));
                assert!(arr.is_null(1));
                assert!(!arr.is_null(2));
                assert!(arr.is_null(3));
                assert!(!arr.is_null(4));
                assert_eq!(arr.get_str(0), Some("hello"));
                assert_eq!(arr.get_str(2), Some("world"));
                assert_eq!(arr.get_str(4), Some("bar"));
            }
            _ => panic!("Expected String32"),
        }
    }

    #[test]
    fn test_consolidate_arena_three_batches() {
        let b1 = Arc::new(table(vec![fa_i32!("x", 1, 2)]));
        let b2 = Arc::new(table(vec![fa_i32!("x", 3)]));
        let b3 = Arc::new(table(vec![fa_i32!("x", 4, 5, 6)]));
        let st = SuperTable::from_batches(vec![b1, b2, b3], None);
        let result = st.consolidate();

        assert_eq!(result.n_rows, 6);
        match &result.cols[0].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.data.as_slice(), &[1, 2, 3, 4, 5, 6]);
            }
            _ => panic!("Expected Int32"),
        }
    }

    #[test]
    fn test_consolidate_arena_preserves_name() {
        let b1 = Arc::new(table(vec![fa_i32!("x", 1, 2)]));
        let b2 = Arc::new(table(vec![fa_i32!("x", 3)]));
        let mut st = SuperTable::from_batches(vec![b1, b2], None);
        st.name = "my_table".to_string();
        let result = st.consolidate();
        assert_eq!(result.name, "my_table");
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn test_consolidate_arena_datetime() {
        use crate::enums::collections::temporal_array::TemporalArray;
        use crate::{DatetimeArray, TimeUnit};

        let fa_dt = |name: &str, vals: &[i64]| -> FieldArray {
            let arr =
                Array::TemporalArray(TemporalArray::Datetime64(Arc::new(DatetimeArray::new(
                    crate::Buffer::from_vec64(vals.into()),
                    None,
                    Some(TimeUnit::Milliseconds),
                ))));
            FieldArray::new(
                Field::new(
                    name.to_string(),
                    ArrowType::Timestamp(TimeUnit::Milliseconds, None),
                    false,
                    None,
                ),
                arr,
            )
        };

        let b1 = Arc::new(table(vec![fa_dt("ts", &[1000, 2000, 3000])]));
        let b2 = Arc::new(table(vec![fa_dt("ts", &[4000, 5000])]));
        let st = SuperTable::from_batches(vec![b1, b2], None);
        let result = st.consolidate();

        assert_eq!(result.n_rows, 5);
        match &result.cols[0].array {
            Array::TemporalArray(TemporalArray::Datetime64(arr)) => {
                assert_eq!(arr.data.as_slice(), &[1000i64, 2000, 3000, 4000, 5000]);
                assert_eq!(arr.time_unit, TimeUnit::Milliseconds);
            }
            _ => panic!("Expected Datetime64"),
        }
    }
}

#[cfg(test)]
mod st_macro_tests {
    use std::sync::Arc;

    use crate::{fa_i32, tbl, SuperTable, Table};

    fn make_batch(name: &str, ids: &[i32]) -> Table {
        tbl!("batch", fa_i32!(name, @slice ids))
    }

    #[test]
    fn st_named_from_tables() {
        let a = make_batch("x", &[1, 2]);
        let b = make_batch("x", &[3, 4, 5]);
        let st: SuperTable = st!("rolling", a, b);
        assert_eq!(st.name, "rolling");
        assert_eq!(st.batches.len(), 2);
        assert_eq!(st.n_rows, 5);
    }

    #[test]
    fn st_auto_name_uses_first_batch_name() {
        let a = make_batch("x", &[1, 2]);
        let b = make_batch("x", &[3]);
        let st: SuperTable = st!(a, b);
        assert_eq!(st.name, "batch");
        assert_eq!(st.batches.len(), 2);
        assert_eq!(st.n_rows, 3);
    }

    #[test]
    fn st_accepts_arc_table() {
        let a: Arc<Table> = Arc::new(make_batch("x", &[1, 2]));
        let b: Arc<Table> = Arc::new(make_batch("x", &[3, 4]));
        let st: SuperTable = st!("arc", a, b);
        assert_eq!(st.name, "arc");
        assert_eq!(st.n_rows, 4);
    }

    #[test]
    fn st_name_only_builds_empty_supertable() {
        let st: SuperTable = st!("scratch");
        assert_eq!(st.name, "scratch");
        assert_eq!(st.batches.len(), 0);
        assert_eq!(st.n_rows, 0);
    }

    #[test]
    fn st_single_batch_named() {
        let a = make_batch("x", &[7, 8, 9]);
        let st: SuperTable = st!("solo", a);
        assert_eq!(st.batches.len(), 1);
        assert_eq!(st.n_rows, 3);
    }

    #[test]
    fn st_trailing_comma_accepted() {
        let a = make_batch("x", &[1]);
        let b = make_batch("x", &[2]);
        let st: SuperTable = st!("trail", a, b,);
        assert_eq!(st.batches.len(), 2);
    }
}
