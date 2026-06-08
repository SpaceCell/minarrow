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

//! # **SuperArray** - *Holds multiple arrays for chunked data partitioning, streaming + fast memIO*
//!
//! Contains SuperArray, a higher-order container representing a logical column split into
//! multiple immutable `Array` chunks with optional shared field metadata.
//!
//! ## Overview
//! - Equivalent to Apache Arrow's `ChunkedArray`.
//! - Stores an ordered list of `Array` segments with shared field metadata.
//! - Chunk lengths may vary.
//! - A solid fit for append-only patterns, partitioned storage, and streaming data ingestion.
//!
//! ## Field Metadata
//! - Field metadata is stored once at the SuperArray level, not per-chunk.
//! - Use `from_arrays()` when you don't need field metadata (e.g., Dam consolidation)
//! - Use `from_arrays_with_field()` when field metadata is required
//!
//! ## Apache Arrow / Polars bridges (`cast_arrow` / `cast_polars` features)
//! - `to_apache_arrow()` exports each chunk as an arrow-rs `ArrayRef`.
//! - `to_polars()` builds a polars `Series` whose internal chunks mirror the SuperArray.
//! - `from_apache_arrow(name, &[ArrayRef])` and `from_polars(&Series)`
//!   recover dtype + field metadata across chunks.
//! - Each panicking method has a `try_*` sibling returning `Result<_, MinarrowError>`.
//! - `&Series` can be converted via `(&series).into()` for ergonomic call sites.

use std::fmt::{Display, Formatter};
use std::iter::FromIterator;
use std::sync::Arc;

#[cfg(feature = "views")]
use crate::ArrayV;
#[cfg(feature = "views")]
use crate::SuperArrayV;
use crate::enums::{error::MinarrowError, shape_dim::ShapeDim};
use crate::ffi::arrow_dtype::ArrowType;
#[cfg(feature = "shared_dict")]
use crate::structs::dictionary::CategoryManagerT;
#[cfg(feature = "size")]
use crate::traits::byte_size::ByteSize;
use crate::traits::consolidate::Consolidate;
use crate::traits::{concatenate::Concatenate, shape::Shape};
use crate::{Array, Field, FieldArray};

/// Strategy for rechunking arrays and tables.
///
/// Defines how to redistribute data across chunks/batches.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RechunkStrategy {
    /// Rechunk into uniform chunks of the specified element/row count.
    Count(usize),
    /// Rechunk targeting a specific memory size per chunk in bytes.
    #[cfg(feature = "size")]
    Memory(usize),
    /// Rechunk using a default size of 8192 elements/rows.
    Auto,
}

/// # SuperArray
///
/// Higher-order container for multiple immutable `Array` chunks with optional shared field metadata.
///
/// ## Description
/// - Stores an ordered sequence of `Array` chunks with a single optional `Field` for all.
/// - Equivalent to Apache Arrow's `ChunkedArray` when sent over FFI, where it is treated
///   as a single logical column.
/// - It can also serve as an unbounded or continuously growing
///   collection of segments, making it useful for streaming ingestion and partitioned storage.
/// - Chunk lengths may vary without restriction.
///
/// ## Field Metadata
/// - Field metadata is stored once at the SuperArray level.
/// - For streaming consolidation (e.g., Dam output), field may be `None`.
/// - Use `field()` to access metadata optionally, `field_ref()` when metadata is required.
///
/// ## Example
/// ```ignore
/// // From raw arrays without field metadata
/// let sa = SuperArray::from_arrays(vec![arr1, arr2]);
/// assert!(sa.field().is_none());
///
/// // From arrays with field metadata
/// let sa = SuperArray::from_arrays_with_field(
///     vec![arr1, arr2],
///     Field::new("col", ArrowType::Int32, false, None)
/// );
/// assert_eq!(sa.field().unwrap().name, "col");
///
/// // From FieldArrays (extracts field from first)
/// let sa = SuperArray::from_field_array_chunks(vec![fa1, fa2]);
/// assert_eq!(sa.field().unwrap().name, fa1.field.name);
/// ```
#[derive(Clone, Debug)]
pub struct SuperArray {
    /// The underlying array chunks.
    pub chunks: Vec<Array>,
    /// Optional field metadata, shared by all chunks.
    pub field: Option<Arc<Field>>,
    /// Optional null counts per chunk. If present, must have same length as `chunks`.
    pub null_counts: Option<Vec<usize>>,
    /// Shared category dictionary for this column when its type is
    /// categorical and the `shared_dict` feature is on. Sibling chunks
    /// pushed into the same `SuperArray` all share this manager's
    /// dictionary, so codes are mutually meaningful across them. New
    /// values arrive via `push(chunk)`.
    #[cfg(feature = "shared_dict")]
    pub(crate) category_manager: Option<CategoryManagerT>,
}

impl PartialEq for SuperArray {
    /// Equality compares chunk data, field metadata, and null counts. The
    /// `category_manager` is derived from `chunks` and not compared.
    fn eq(&self, other: &Self) -> bool {
        self.chunks == other.chunks
            && self.field == other.field
            && self.null_counts == other.null_counts
    }
}

impl SuperArray {
    // Constructors

    /// Constructs an empty SuperArray with no field metadata.
    #[inline]
    pub fn new() -> Self {
        Self {
            chunks: Vec::new(),
            field: None,
            null_counts: None,
            #[cfg(feature = "shared_dict")]
            category_manager: None,
        }
    }

    /// Constructs a SuperArray from raw `Array` chunks without field metadata.
    ///
    /// Use this for streaming consolidation patterns where field metadata is not needed.
    ///
    /// # Panics
    /// Panics if chunks have mismatched types.
    pub fn from_arrays(chunks: Vec<Array>) -> Self {
        if chunks.len() > 1 {
            let dtype = chunks[0].arrow_type();
            for (i, chunk) in chunks.iter().enumerate().skip(1) {
                assert_eq!(
                    chunk.arrow_type(),
                    dtype,
                    "Chunk {i} ArrowType mismatch (expected {:?}, got {:?})",
                    dtype,
                    chunk.arrow_type()
                );
            }
        }
        #[cfg_attr(not(feature = "shared_dict"), allow(unused_mut))]
        let mut sa = Self {
            chunks,
            field: None,
            null_counts: None,
            #[cfg(feature = "shared_dict")]
            category_manager: None,
        };
        #[cfg(feature = "shared_dict")]
        sa.rebuild_category_manager();
        sa
    }

    /// Constructs a SuperArray from raw `Array` chunks with field metadata.
    ///
    /// The field metadata applies to all chunks (they represent the same logical column).
    ///
    /// # Panics
    /// Panics if chunks have mismatched types or don't match the field type.
    pub fn from_arrays_with_field(chunks: Vec<Array>, field: impl Into<Arc<Field>>) -> Self {
        let field = field.into();

        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(
                chunk.arrow_type(),
                field.dtype,
                "Chunk {i} ArrowType mismatch (expected {:?}, got {:?})",
                field.dtype,
                chunk.arrow_type()
            );
        }

        #[cfg_attr(not(feature = "shared_dict"), allow(unused_mut))]
        let mut sa = Self {
            chunks,
            field: Some(field),
            null_counts: None,
            #[cfg(feature = "shared_dict")]
            category_manager: None,
        };
        #[cfg(feature = "shared_dict")]
        sa.rebuild_category_manager();
        sa
    }

    /// Constructs a SuperArray from raw `Array` chunks with null counts.
    ///
    /// # Panics
    /// Panics if chunks have mismatched types or null_counts length doesn't match chunks length.
    pub fn from_arrays_nc(chunks: Vec<Array>, null_counts: Vec<usize>) -> Self {
        assert_eq!(
            chunks.len(),
            null_counts.len(),
            "null_counts length ({}) must match chunks length ({})",
            null_counts.len(),
            chunks.len()
        );

        if chunks.len() > 1 {
            let dtype = chunks[0].arrow_type();
            for (i, chunk) in chunks.iter().enumerate().skip(1) {
                assert_eq!(
                    chunk.arrow_type(),
                    dtype,
                    "Chunk {i} ArrowType mismatch (expected {:?}, got {:?})",
                    dtype,
                    chunk.arrow_type()
                );
            }
        }

        #[cfg_attr(not(feature = "shared_dict"), allow(unused_mut))]
        let mut sa = Self {
            chunks,
            field: None,
            null_counts: Some(null_counts),
            #[cfg(feature = "shared_dict")]
            category_manager: None,
        };
        #[cfg(feature = "shared_dict")]
        sa.rebuild_category_manager();
        sa
    }

    /// Constructs a SuperArray from `FieldArray` chunks.
    ///
    /// Extracts field metadata and null counts from the chunks.
    ///
    /// # Panics
    /// Panics if chunks is empty or metadata/type/nullable mismatch is found.
    pub fn from_field_array_chunks(chunks: Vec<FieldArray>) -> Self {
        assert!(
            !chunks.is_empty(),
            "from_field_array_chunks: input chunks cannot be empty"
        );

        let field = chunks[0].field.clone();

        for (i, fa) in chunks.iter().enumerate().skip(1) {
            assert_eq!(
                fa.field.dtype, field.dtype,
                "Chunk {i} ArrowType mismatch (expected {:?}, got {:?})",
                field.dtype, fa.field.dtype
            );
            assert_eq!(
                fa.field.nullable, field.nullable,
                "Chunk {i} nullability mismatch"
            );
            assert_eq!(
                fa.field.name, field.name,
                "Chunk {i} field name mismatch (expected '{}', got '{}')",
                field.name, fa.field.name
            );
        }

        let null_counts: Vec<usize> = chunks.iter().map(|fa| fa.null_count).collect();
        let arrays = chunks.into_iter().map(|fa| fa.array).collect();

        #[cfg_attr(not(feature = "shared_dict"), allow(unused_mut))]
        let mut sa = Self {
            chunks: arrays,
            field: Some(field),
            null_counts: Some(null_counts),
            #[cfg(feature = "shared_dict")]
            category_manager: None,
        };
        #[cfg(feature = "shared_dict")]
        sa.rebuild_category_manager();
        sa
    }

    /// Construct from `Vec<FieldArray>`.
    ///
    /// Alias for `from_field_array_chunks`.
    pub fn from_chunks(chunks: Vec<FieldArray>) -> Self {
        Self::from_field_array_chunks(chunks)
    }

    /// Materialises a `SuperArray` from an existing slice of `ArrayView` tuples,
    /// using the provided field metadata (applied to all slices).
    ///
    /// Panics if the slice list is empty, or if any slice's type or nullability
    /// does not match the provided field.
    #[cfg(feature = "views")]
    pub fn from_slices(slices: &[ArrayV], field: Arc<Field>) -> Self {
        assert!(!slices.is_empty(), "from_slices requires non-empty slice");

        let mut arrays = Vec::with_capacity(slices.len());
        let mut null_counts = Vec::with_capacity(slices.len());
        for (i, view) in slices.iter().enumerate() {
            assert_eq!(
                view.array.arrow_type(),
                field.dtype,
                "Slice {i} ArrowType does not match field"
            );
            assert_eq!(
                view.array.is_nullable(),
                field.nullable,
                "Slice {i} nullability does not match field"
            );
            arrays.push(view.array.slice_clone(view.offset, view.len()));
            null_counts.push(view.null_count());
        }

        #[cfg_attr(not(feature = "shared_dict"), allow(unused_mut))]
        let mut sa = Self {
            chunks: arrays,
            field: Some(field),
            null_counts: Some(null_counts),
            #[cfg(feature = "shared_dict")]
            category_manager: None,
        };
        #[cfg(feature = "shared_dict")]
        sa.rebuild_category_manager();
        sa
    }

    /// Returns a zero-copy view of this chunked array over the window `[offset..offset+len)`.
    ///
    /// If the chunks are fragmented in memory, access patterns may result in
    /// degraded cache locality and reduced SIMD optimisation.
    ///
    /// # Panics
    /// Panics if field metadata is not present.
    #[cfg(feature = "views")]
    pub fn slice(&self, offset: usize, len: usize) -> SuperArrayV {
        assert!(offset + len <= self.len(), "slice out of bounds");
        let field = self.field.clone().expect("slice() requires field metadata");

        let mut remaining = len;
        let mut off = offset;
        let mut slices = Vec::new();

        for chunk in &self.chunks {
            let this_len = chunk.len();
            if off >= this_len {
                off -= this_len;
                continue;
            }

            let take = remaining.min(this_len - off);
            slices.push(ArrayV::new(chunk.clone(), off, take));
            remaining -= take;

            if remaining == 0 {
                break;
            }
            off = 0;
        }

        SuperArrayV { slices, len, field }
    }

    // Metadata

    /// Returns the field metadata if present.
    #[inline]
    pub fn field(&self) -> Option<&Field> {
        self.field.as_deref()
    }

    /// Returns the field metadata, panicking if not present.
    ///
    /// Use this when field metadata is required (e.g., for schema operations).
    #[inline]
    pub fn field_ref(&self) -> &Field {
        self.field
            .as_ref()
            .expect("field_ref() called but SuperArray has no field metadata")
    }

    /// Returns `true` if this SuperArray has field metadata.
    #[inline]
    pub fn has_field(&self) -> bool {
        self.field.is_some()
    }

    /// Returns the Arc-wrapped field if present.
    #[inline]
    pub fn field_arc(&self) -> Option<&Arc<Field>> {
        self.field.as_ref()
    }

    /// Returns the Arrow physical type from the first chunk.
    ///
    /// Falls back to field metadata if no chunks present.
    ///
    /// # Panics
    /// Panics if both chunks and field are empty/None.
    #[inline]
    pub fn arrow_type(&self) -> ArrowType {
        if let Some(chunk) = self.chunks.first() {
            chunk.arrow_type()
        } else if let Some(ref field) = self.field {
            field.dtype.clone()
        } else {
            panic!("arrow_type() called on empty SuperArray with no field")
        }
    }

    /// Returns the nullability flag.
    ///
    /// # Panics
    /// Panics if field metadata is not present.
    #[inline]
    pub fn is_nullable(&self) -> bool {
        self.field_ref().nullable
    }

    /// Returns the number of logical chunks.
    #[inline]
    pub fn n_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Returns total logical length (sum of all chunk lengths).
    pub fn len(&self) -> usize {
        self.chunks.iter().map(|c| c.len()).sum()
    }

    /// Returns true if the array has no chunks or all chunks are empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_chunks() == 0 || self.len() == 0
    }

    // Chunk Access

    /// Returns a reference to a specific chunk, if it exists.
    #[inline]
    pub fn chunk(&self, idx: usize) -> Option<&Array> {
        self.chunks.get(idx)
    }

    /// Returns the null count for a specific chunk, if available.
    #[inline]
    pub fn chunk_null_count(&self, idx: usize) -> Option<usize> {
        self.null_counts
            .as_ref()
            .and_then(|nc| nc.get(idx).copied())
    }

    // Mutation

    /// Appends a raw array chunk.
    ///
    /// If null counts are being tracked, the null count is computed from the
    /// array's null_mask. If you already know the null count, use
    /// `push_with_null_count()` to avoid recomputation.
    ///
    /// # Panics
    /// Panics if the chunk type doesn't match existing chunks or field.
    pub fn push(&mut self, chunk: Array) {
        if let Some(first) = self.chunks.first() {
            assert_eq!(
                chunk.arrow_type(),
                first.arrow_type(),
                "Chunk ArrowType mismatch"
            );
        } else if let Some(ref field) = self.field {
            assert_eq!(
                chunk.arrow_type(),
                field.dtype,
                "Chunk ArrowType mismatch with field"
            );
        }
        // If tracking null counts, compute from the array's null_mask
        if let Some(ref mut nc) = self.null_counts {
            nc.push(chunk.null_count());
        }
        #[cfg_attr(not(feature = "shared_dict"), allow(unused_mut))]
        let mut chunk = chunk;
        #[cfg(feature = "shared_dict")]
        CategoryManagerT::add_remap_cats(
            &mut self.category_manager,
            std::iter::once(&mut chunk),
        );
        self.chunks.push(chunk);
    }

    /// Appends a raw array chunk with its null count.
    ///
    /// When the null count is already known this is slightly faster than `push`
    pub fn push_with_null_count(&mut self, chunk: Array, null_count: usize) {
        if let Some(first) = self.chunks.first() {
            assert_eq!(
                chunk.arrow_type(),
                first.arrow_type(),
                "Chunk ArrowType mismatch"
            );
        } else if let Some(ref field) = self.field {
            assert_eq!(
                chunk.arrow_type(),
                field.dtype,
                "Chunk ArrowType mismatch with field"
            );
        }
        #[cfg_attr(not(feature = "shared_dict"), allow(unused_mut))]
        let mut chunk = chunk;
        #[cfg(feature = "shared_dict")]
        CategoryManagerT::add_remap_cats(
            &mut self.category_manager,
            std::iter::once(&mut chunk),
        );
        self.chunks.push(chunk);
        if let Some(ref mut nc) = self.null_counts {
            nc.push(null_count);
        } else {
            self.null_counts = Some(vec![null_count]);
        }
    }

    /// Validates and appends a FieldArray chunk.
    ///
    /// If this SuperArray has no field metadata yet, it will be set from the chunk.
    ///
    /// # Panics
    /// If the chunk does not match the expected type, nullability, or field name.
    pub fn push_field_array(&mut self, chunk: FieldArray) {
        if let Some(ref field) = self.field {
            assert_eq!(chunk.field.dtype, field.dtype, "Chunk ArrowType mismatch");
            assert_eq!(
                chunk.field.nullable, field.nullable,
                "Chunk nullability mismatch"
            );
            assert_eq!(chunk.field.name, field.name, "Chunk field name mismatch");
        } else if !self.chunks.is_empty() {
            assert_eq!(
                chunk.array.arrow_type(),
                self.chunks[0].arrow_type(),
                "Chunk ArrowType mismatch"
            );
        }

        // Set field from first push if not already set
        if self.field.is_none() {
            self.field = Some(chunk.field.clone());
        }

        #[cfg_attr(not(feature = "shared_dict"), allow(unused_mut))]
        let mut array = chunk.array;
        #[cfg(feature = "shared_dict")]
        CategoryManagerT::add_remap_cats(
            &mut self.category_manager,
            std::iter::once(&mut array),
        );
        self.chunks.push(array);
        if let Some(ref mut nc) = self.null_counts {
            nc.push(chunk.null_count);
        } else {
            self.null_counts = Some(vec![chunk.null_count]);
        }
    }

    /// Inserts rows from another SuperArray (or Array) at the specified index.
    ///
    /// This is an **O(n)** operation where n is the number of elements in the chunk
    /// containing the insertion point.
    ///
    /// # Arguments
    /// * `index` - Global row position before which to insert (0 = prepend, len() = append)
    /// * `other` - SuperArray or Array to insert (via `Into<SuperArray>`)
    ///
    /// # Requirements
    /// - Array types must match
    /// - `index` must be <= `self.len()`
    ///
    /// # Strategy
    /// Finds the chunk containing the insertion point and inserts all of `other`'s data
    /// into that chunk. This may cause the target chunk to grow significantly.
    ///
    /// # Errors
    /// - `IndexError` if index > len()
    /// - Type mismatch if array types don't match
    pub fn insert_rows(
        &mut self,
        index: usize,
        other: impl Into<SuperArray>,
    ) -> Result<(), MinarrowError> {
        let other = other.into();
        let total_len = self.len();

        // Validate index
        if index > total_len {
            return Err(MinarrowError::IndexError(format!(
                "Index {} out of bounds for SuperArray of length {}",
                index, total_len
            )));
        }

        // If other is empty, nothing to do
        if other.is_empty() {
            return Ok(());
        }

        // Validate type match
        if !self.chunks.is_empty() && !other.chunks.is_empty() {
            let self_type = self.chunks[0].arrow_type();
            let other_type = other.chunks[0].arrow_type();

            if self_type != other_type {
                return Err(MinarrowError::IncompatibleTypeError {
                    from: "SuperArray",
                    to: "SuperArray",
                    message: Some(format!(
                        "Type mismatch: {:?} vs {:?}",
                        self_type, other_type
                    )),
                });
            }
        }

        // Handle empty self - just take other's data
        if self.chunks.is_empty() {
            self.chunks = other.chunks;
            self.field = other.field;
            self.null_counts = other.null_counts;
            return Ok(());
        }

        // Find which chunk contains the insertion index
        let mut cumulative = 0;
        let mut chunk_idx = None;
        for (i, chunk) in self.chunks.iter().enumerate() {
            let chunk_len = chunk.len();

            // Check if index falls within this chunk or at its boundary
            if index <= cumulative + chunk_len {
                chunk_idx = Some((i, index - cumulative));
                break;
            }

            cumulative += chunk_len;
        }

        let (target_idx, local_index) = chunk_idx.ok_or_else(|| {
            MinarrowError::IndexError(format!("Failed to find chunk for index {}", index))
        })?;

        let target_chunk_len = self.chunks[target_idx].len();

        // Get field for split operations (required by Array::split)
        let field = self.field.clone().unwrap_or_else(|| {
            Arc::new(Field::new(
                "data",
                self.chunks[0].arrow_type(),
                self.chunks[0].is_nullable(),
                None,
            ))
        });

        // Handle edge cases: prepend or append to a chunk without splitting
        if local_index == 0 {
            // Insert before target chunk
            let mut new_chunks = Vec::with_capacity(self.chunks.len() + other.chunks.len());
            new_chunks.extend(self.chunks.drain(0..target_idx));
            new_chunks.extend(other.chunks.into_iter());
            new_chunks.extend(self.chunks.drain(..));
            self.chunks = new_chunks;
            // Note: null_counts tracking is invalidated by this operation
            self.null_counts = None;
        } else if local_index == target_chunk_len {
            // Insert after target chunk
            let mut new_chunks = Vec::with_capacity(self.chunks.len() + other.chunks.len());
            new_chunks.extend(self.chunks.drain(0..=target_idx));
            new_chunks.extend(other.chunks.into_iter());
            new_chunks.extend(self.chunks.drain(..));
            self.chunks = new_chunks;
            // Note: null_counts tracking is invalidated by this operation
            self.null_counts = None;
        } else {
            // Split the target chunk at the insertion point
            let target_chunk = self.chunks.remove(target_idx);
            let split_result = target_chunk.split(local_index, &field)?;
            let split_arrays: Vec<Array> = split_result.chunks;

            // Build new chunk list: left chunk + other's chunks + right chunk
            let mut new_chunks = Vec::with_capacity(self.chunks.len() + other.chunks.len() + 2);

            // Add chunks before target
            new_chunks.extend(self.chunks.drain(0..target_idx));

            // Add left half of split
            new_chunks.push(split_arrays[0].clone());

            // Add other's chunks
            new_chunks.extend(other.chunks.into_iter());

            // Add right half of split
            new_chunks.push(split_arrays[1].clone());

            // Add remaining chunks after target
            new_chunks.extend(self.chunks.drain(..));

            self.chunks = new_chunks;
            // Note: null_counts tracking is invalidated by this operation
            self.null_counts = None;
        }

        Ok(())
    }

    /// Rechunks the array according to the specified strategy.
    ///
    /// Redistributes data across chunks using an efficient incremental approach
    /// that avoids full materialisation:
    /// - `Count(n)`: Creates chunks of `n` elements. The last chunk may be smaller.
    /// - `Auto`: Uses a default size of 8192 elements
    /// - `Memory(bytes)`: Targets a specific memory size per chunk
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
    /// // Rechunk into 1024-element chunks
    /// array.rechunk(RechunkStrategy::Count(1024))?;
    ///
    /// // Rechunk with default size
    /// array.rechunk(RechunkStrategy::Auto)?;
    ///
    /// // Target 64KB per chunk
    /// array.rechunk(RechunkStrategy::Memory(65536))?;
    /// ```
    pub fn rechunk(&mut self, strategy: RechunkStrategy) -> Result<(), MinarrowError> {
        if self.chunks.is_empty() || self.len() == 0 {
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
                let total_len = self.len();

                if total_bytes == 0 {
                    return Err(MinarrowError::IndexError(
                        "Cannot rechunk: array has 0 estimated bytes".to_string(),
                    ));
                }

                ((bytes_per_chunk * total_len) / total_bytes).max(1)
            }
        };

        // Fast path: single chunk already at target size
        if self.chunks.len() == 1 && self.chunks[0].len() == chunk_size {
            return Ok(());
        }

        // Get or create field for split operations
        let field = self.field.clone().unwrap_or_else(|| {
            Arc::new(Field::new(
                "data",
                self.chunks[0].arrow_type(),
                self.chunks[0].is_nullable(),
                None,
            ))
        });

        let mut new_chunks: Vec<Array> = Vec::new();
        let mut accumulator: Option<Array> = None;

        // Process each existing chunk
        for chunk in self.chunks.drain(..) {
            let mut remaining = chunk;

            while remaining.len() > 0 {
                if let Some(ref mut acc) = accumulator {
                    let acc_len = acc.len();
                    let needed = chunk_size - acc_len;

                    if remaining.len() <= needed {
                        // Entire remaining chunk fits in accumulator
                        *acc = acc.clone().concat(remaining)?;

                        // If accumulator is now full, emit it
                        if acc.len() == chunk_size {
                            new_chunks.push(accumulator.take().unwrap());
                        }
                        break; // consumed remaining
                    } else {
                        // Split remaining chunk to complete accumulator
                        let split_result = remaining.split(needed, &field)?;
                        let parts = split_result.chunks;
                        let to_add = parts[0].clone();
                        remaining = parts[1].clone();

                        // Complete and emit the accumulator
                        *acc = acc.clone().concat(to_add)?;
                        new_chunks.push(accumulator.take().unwrap());
                    }
                } else {
                    // No accumulator - start processing remaining
                    if remaining.len() == chunk_size {
                        // Exact fit - use remaining as-is
                        new_chunks.push(remaining);
                        break;
                    } else if remaining.len() > chunk_size {
                        // Split off one chunk_size portion
                        let split_result = remaining.split(chunk_size, &field)?;
                        let parts = split_result.chunks;
                        new_chunks.push(parts[0].clone());
                        remaining = parts[1].clone();
                    } else {
                        // Remaining becomes new accumulator
                        accumulator = Some(remaining);
                        break;
                    }
                }
            }
        }

        // Emit any remaining accumulator as final chunk
        if let Some(final_chunk) = accumulator {
            new_chunks.push(final_chunk);
        }

        self.chunks = new_chunks;
        // Null counts are invalidated by rechunking
        self.null_counts = None;
        Ok(())
    }

    /// Rechunks only the first `up_to_index` elements, leaving the rest untouched.
    ///
    /// This is useful for streaming scenarios where new data is being appended
    /// and you want to rechunk stable data while leaving recent additions alone.
    ///
    /// # Arguments
    /// * `up_to_index` - Rechunk only elements before this index
    /// * `strategy` - The rechunking strategy to use
    ///
    /// # Errors
    /// - Returns `IndexError` if `up_to_index` is greater than array length
    /// - Returns same errors as `rechunk()` for invalid strategies
    ///
    /// # Example
    /// ```ignore
    /// // Rechunk first 1000 elements, leave the rest untouched
    /// array.rechunk_to(1000, RechunkStrategy::Count(512))?;
    /// ```
    pub fn rechunk_to(
        &mut self,
        up_to_index: usize,
        strategy: RechunkStrategy,
    ) -> Result<(), MinarrowError> {
        let total_len = self.len();

        if up_to_index > total_len {
            return Err(MinarrowError::IndexError(format!(
                "rechunk_to index {} out of bounds for array of length {}",
                up_to_index, total_len
            )));
        }

        if up_to_index == 0 || self.chunks.is_empty() {
            return Ok(());
        }

        if up_to_index == total_len {
            // Rechunk everything
            return self.rechunk(strategy);
        }

        // Get or create field for split operations
        let field = self.field.clone().unwrap_or_else(|| {
            Arc::new(Field::new(
                "data",
                self.chunks[0].arrow_type(),
                self.chunks[0].is_nullable(),
                None,
            ))
        });

        // Find which chunks contain the data up to up_to_index
        let mut current_offset = 0;
        let mut split_point = 0;

        for (i, chunk) in self.chunks.iter().enumerate() {
            let chunk_end = current_offset + chunk.len();
            if chunk_end > up_to_index {
                split_point = i;
                break;
            }
            current_offset = chunk_end;
        }

        // Extract chunks to rechunk and chunks to keep
        let mut to_rechunk: Vec<Array> = self.chunks.drain(..=split_point).collect();
        let keep_chunks: Vec<Array> = self.chunks.drain(..).collect();

        // If the split chunk needs to be divided
        if current_offset < up_to_index {
            let split_chunk = to_rechunk.pop().unwrap();
            let split_at = up_to_index - current_offset;

            let split_result = split_chunk.split(split_at, &field)?;
            let parts = split_result.chunks;
            to_rechunk.push(parts[0].clone());
            self.chunks.push(parts[1].clone());
        }

        // Rechunk the selected portion
        self.chunks.extend(keep_chunks);
        let mut temp = SuperArray::from_arrays(to_rechunk);
        temp.field = self.field.clone();
        temp.rechunk(strategy)?;

        // Reconstruct rechunked portion + untouched portion
        let mut result = temp.chunks;
        result.extend(self.chunks.drain(..));
        self.chunks = result;
        // Null counts are invalidated
        self.null_counts = None;

        Ok(())
    }

    /// Consumes the SuperArray and returns the underlying chunks.
    #[inline]
    pub fn into_chunks(self) -> Vec<Array> {
        self.chunks
    }

    /// Returns a reference to the underlying chunks.
    #[inline]
    pub fn chunks(&self) -> &[Array] {
        &self.chunks
    }

    /// Borrow the column's `CategoryManagerT`, or `None` if the column
    /// is not categorical or no chunks have been pushed yet.
    ///
    /// This accessor is for sharing-association: two `CategoricalArray`s
    /// whose dictionaries point at the same `Arc<DictionaryInner>` (check
    /// via `Dictionary::shares_with`) are in the same sharing group.
    /// New values arrive through `push(chunk)`, not through this borrow.
    #[cfg(feature = "shared_dict")]
    #[inline]
    pub fn category_manager(&self) -> Option<&CategoryManagerT> {
        self.category_manager.as_ref()
    }

    /// Rebuild `category_manager` from the current `chunks`. Used by
    /// constructors that take a `Vec<Array>` upfront and need to set up
    /// the manager afterwards. The first chunk's dictionary is taken as
    /// the starting state; subsequent chunks' dictionaries are folded in
    /// by the same logic as `push`.
    #[cfg(feature = "shared_dict")]
    pub(crate) fn rebuild_category_manager(&mut self) {
        self.category_manager = None;
        if self.chunks.is_empty() {
            return;
        }
        let mut chunks = std::mem::take(&mut self.chunks);
        CategoryManagerT::add_remap_cats(
            &mut self.category_manager,
            chunks.iter_mut(),
        );
        self.chunks = chunks;
    }
}

impl Default for SuperArray {
    fn default() -> Self {
        Self::new()
    }
}

// Vec<Array> -> SuperArray
//
// Multiple chunks without field metadata
impl From<Vec<FieldArray>> for SuperArray {
    fn from(arrays: Vec<FieldArray>) -> Self {
        SuperArray::from_field_array_chunks(arrays)
    }
}

// Vec<Array> -> SuperArray
//
// Multiple chunks without field metadata - Catch all case
impl FromIterator<FieldArray> for SuperArray {
    fn from_iter<T: IntoIterator<Item = FieldArray>>(iter: T) -> Self {
        let chunks: Vec<FieldArray> = iter.into_iter().collect();
        Self::from_field_array_chunks(chunks)
    }
}

// FieldArray -> SuperArray
// Single chunk with field metadata
impl From<FieldArray> for SuperArray {
    fn from(fa: FieldArray) -> Self {
        #[cfg_attr(not(feature = "shared_dict"), allow(unused_mut))]
        let mut sa = SuperArray {
            chunks: vec![fa.array],
            field: Some(fa.field),
            null_counts: Some(vec![fa.null_count]),
            #[cfg(feature = "shared_dict")]
            category_manager: None,
        };
        #[cfg(feature = "shared_dict")]
        sa.rebuild_category_manager();
        sa
    }
}

// Array -> SuperArray
//
// Single chunk without field metadata
impl From<Array> for SuperArray {
    fn from(array: Array) -> Self {
        #[cfg_attr(not(feature = "shared_dict"), allow(unused_mut))]
        let mut sa = SuperArray {
            chunks: vec![array],
            field: None,
            null_counts: None,
            #[cfg(feature = "shared_dict")]
            category_manager: None,
        };
        #[cfg(feature = "shared_dict")]
        sa.rebuild_category_manager();
        sa
    }
}

// Vec<Array> -> SuperArray
//
// Multiple chunks without field metadata
impl From<Vec<Array>> for SuperArray {
    fn from(arrays: Vec<Array>) -> Self {
        SuperArray::from_arrays(arrays)
    }
}

// Vec<Array> -> SuperArray
//
// Catch all iterator
impl FromIterator<Array> for SuperArray {
    fn from_iter<T: IntoIterator<Item = Array>>(iter: T) -> Self {
        let chunks: Vec<Array> = iter.into_iter().collect();
        Self::from_arrays(chunks)
    }
}

impl Shape for SuperArray {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank1(self.len())
    }
}

impl Consolidate for SuperArray {
    type Output = Array;

    /// Consolidates all chunks into a single contiguous `Array`.
    ///
    /// Materialises all rows from all chunks into one contiguous buffer.
    /// Use this when you need contiguous memory for operations or
    /// APIs that require single buffers.
    ///
    /// When the `arena` feature is enabled, all buffers are written
    /// into a single allocation then sliced into typed views. The
    /// resulting buffers are SharedBuffer-backed; mutations trigger
    /// copy-on-write.
    ///
    /// Without the `arena` feature, falls back to concat fold.
    ///
    /// A zero-chunk SuperArray consolidates to a zero-row Array of the
    /// declared field's dtype, or `Array::Null` when no field is set.
    fn consolidate(self) -> Array {
        if self.chunks.is_empty() {
            return match self.field.as_ref() {
                Some(f) => Array::from_arrow_dtype(&f.dtype),
                None => Array::Null,
            };
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

impl SuperArray {
    /// Concat-based consolidation: folds chunks via `Array::concat`. The
    /// zero-chunk path is handled at the trait wrapper above.
    #[cfg_attr(feature = "arena", allow(dead_code))]
    fn consolidate_concat(self) -> Array {
        self.chunks
            .into_iter()
            .reduce(|acc, arr| acc.concat(arr).expect("Failed to concatenate arrays"))
            .expect("Expected at least one array")
    }

    /// Arena-based consolidation: writes all buffers into a single
    /// allocation, then slices typed views from the frozen SharedBuffer.
    /// The zero-chunk path is handled at the trait wrapper above.
    #[cfg(feature = "arena")]
    fn consolidate_arena(self) -> Array {
        if self.chunks.len() == 1 {
            return self.chunks.into_iter().next().unwrap();
        }

        let dtype = self.chunks[0].arrow_type();
        let refs: Vec<&Array> = self.chunks.iter().collect();
        crate::structs::arena::consolidate_array_arena(&refs, &dtype)
    }
}

impl Concatenate for SuperArray {
    /// Concatenates two SuperArrays by appending all chunks from `other` to `self`.
    ///
    /// # Requirements
    /// - Both SuperArrays must have compatible types
    ///
    /// # Returns
    /// A new SuperArray containing all chunks from `self` followed by all chunks from `other`
    ///
    /// # Errors
    /// - `IncompatibleTypeError` if array types don't match
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        // If both are empty, return empty
        if self.chunks.is_empty() && other.chunks.is_empty() {
            return Ok(SuperArray::new());
        }

        // If one is empty, return the other
        if self.chunks.is_empty() {
            return Ok(other);
        }
        if other.chunks.is_empty() {
            return Ok(self);
        }

        // Validate types match
        let self_type = self.chunks[0].arrow_type();
        let other_type = other.chunks[0].arrow_type();

        if self_type != other_type {
            return Err(MinarrowError::IncompatibleTypeError {
                from: "SuperArray",
                to: "SuperArray",
                message: Some(format!(
                    "Type mismatch: {:?} vs {:?}",
                    self_type, other_type
                )),
            });
        }

        // Concatenate chunks
        let mut result_chunks = self.chunks;
        result_chunks.extend(other.chunks);

        // Merge null counts if both have them
        let null_counts = match (self.null_counts, other.null_counts) {
            (Some(mut self_nc), Some(other_nc)) => {
                self_nc.extend(other_nc);
                Some(self_nc)
            }
            _ => None,
        };

        #[cfg_attr(not(feature = "shared_dict"), allow(unused_mut))]
        let mut sa = SuperArray {
            chunks: result_chunks,
            field: self.field.or(other.field),
            null_counts,
            #[cfg(feature = "shared_dict")]
            category_manager: None,
        };
        #[cfg(feature = "shared_dict")]
        sa.rebuild_category_manager();
        Ok(sa)
    }
}

// ===========================================================
// Apache Arrow / Polars bridges for SuperArray
// ===========================================================

impl SuperArray {
    /// Export each chunk as an arrow-rs `ArrayRef`. The resulting `Vec` has the
    /// same length as `self.n_chunks()`. Field metadata travels with each chunk
    /// when the SuperArray was built with one.
    ///
    /// Panics on FFI failure. For a fallible variant, see
    /// [`SuperArray::try_to_apache_arrow`].
    #[cfg(feature = "cast_arrow")]
    #[inline]
    pub fn to_apache_arrow(&self) -> Vec<arrow::array::ArrayRef> {
        self.try_to_apache_arrow()
            .expect("SuperArray::to_apache_arrow failed")
    }

    /// Fallible variant of [`SuperArray::to_apache_arrow`].
    #[cfg(feature = "cast_arrow")]
    pub fn try_to_apache_arrow(
        &self,
    ) -> Result<Vec<arrow::array::ArrayRef>, MinarrowError> {
        let mut out = Vec::with_capacity(self.chunks.len());
        // When a field is present, use it so logical types (Timestamp/Time/etc)
        // are preserved across chunks. Otherwise derive per chunk from shape.
        match self.field.as_deref() {
            Some(field) => {
                for chunk in &self.chunks {
                    out.push(crate::ffi::arrow_rs::export(
                        Arc::new(chunk.clone()),
                        crate::ffi::schema::Schema::from(vec![field.clone()]),
                    )?);
                }
            }
            None => {
                for chunk in &self.chunks {
                    out.push(chunk.try_to_apache_arrow("")?);
                }
            }
        }
        Ok(out)
    }

    /// Build a polars `Series` whose internal arrow chunks mirror the
    /// SuperArray's chunks. Requires a stored field for the name.
    ///
    /// Panics on FFI failure or missing field metadata. For a fallible variant,
    /// see [`SuperArray::try_to_polars`].
    #[cfg(feature = "cast_polars")]
    #[inline]
    pub fn to_polars(&self) -> polars::prelude::Series {
        self.try_to_polars().expect("SuperArray::to_polars failed")
    }

    /// Fallible variant of [`SuperArray::to_polars`].
    #[cfg(feature = "cast_polars")]
    pub fn try_to_polars(
        &self,
    ) -> Result<polars::prelude::Series, MinarrowError> {
        let field = self.field.as_deref().ok_or_else(|| MinarrowError::TypeError {
            from: "SuperArray",
            to: "polars::Series",
            message: Some("SuperArray has no field metadata; assign one before exporting".into()),
        })?;
        let schema = crate::ffi::schema::Schema::from(vec![field.clone()]);

        if self.chunks.is_empty() {
            // Build an empty Series with the right dtype via an empty chunk.
            return crate::ffi::polars::export(
                Arc::new(Array::Null),
                &field.name,
                schema,
            );
        }

        // Build a Series per chunk, then concatenate via polars to preserve chunks.
        let mut iter = self.chunks.iter();
        let first = crate::ffi::polars::export(
            Arc::new(iter.next().unwrap().clone()),
            &field.name,
            schema.clone(),
        )?;
        let mut acc = first;
        for chunk in iter {
            let s = crate::ffi::polars::export(
                Arc::new(chunk.clone()),
                &field.name,
                schema.clone(),
            )?;
            acc.append(&s)?;
        }
        Ok(acc)
    }

    /// Build a `SuperArray` from arrow-rs chunks sharing a single column name.
    ///
    /// The schema of each chunk must match. Dtype, nullability, and any custom
    /// metadata are recovered from the first chunk.
    ///
    /// Panics on FFI failure. For a fallible variant, see
    /// [`SuperArray::try_from_apache_arrow`].
    #[cfg(feature = "cast_arrow")]
    #[inline]
    pub fn from_apache_arrow(
        name: impl Into<String>,
        chunks: &[arrow::array::ArrayRef],
    ) -> SuperArray {
        Self::try_from_apache_arrow(name, chunks)
            .expect("SuperArray::from_apache_arrow failed")
    }

    /// Fallible variant of [`SuperArray::from_apache_arrow`].
    #[cfg(feature = "cast_arrow")]
    pub fn try_from_apache_arrow(
        name: impl Into<String>,
        chunks: &[arrow::array::ArrayRef],
    ) -> Result<SuperArray, MinarrowError> {
        let name: String = name.into();
        if chunks.is_empty() {
            return Ok(SuperArray::new());
        }
        let mut field_arrays = Vec::with_capacity(chunks.len());
        for c in chunks {
            field_arrays.push(FieldArray::try_from_apache_arrow(name.clone(), c)?);
        }
        Ok(SuperArray::from_field_array_chunks(field_arrays))
    }

    /// Build a `SuperArray` from a Polars `Series`, preserving the Series'
    /// internal chunk boundaries.
    ///
    /// ## Performance note
    /// Polars data is typically 8-byte aligned (per the Arrow spec default),
    /// while Minarrow uses 64-byte aligned `Vec64<T>` buffers for SIMD.
    /// Most of the time this results in a memory copy per chunk to realign
    /// on import, unless the source data happens to be pre-aligned to 64
    /// bytes. The FFI hand-off itself is pointer-level zero-copy; the
    /// realignment is done by `Buffer::from_shared` when the source isn't
    /// 64-byte aligned. No consolidation copy is performed - chunk
    /// boundaries are preserved as-is.
    ///
    /// Panics on FFI failure. For a fallible variant, see
    /// [`SuperArray::try_from_polars`].
    #[cfg(feature = "cast_polars")]
    #[inline]
    pub fn from_polars(s: &polars::prelude::Series) -> SuperArray {
        Self::try_from_polars(s).expect("SuperArray::from_polars failed")
    }

    /// Fallible variant of [`SuperArray::from_polars`].
    #[cfg(feature = "cast_polars")]
    pub fn try_from_polars(
        s: &polars::prelude::Series,
    ) -> Result<SuperArray, MinarrowError> {
        use polars::prelude::CompatLevel;
        let n = s.n_chunks();
        if n == 0 {
            return Ok(SuperArray::new());
        }
        let name = s.name().as_str().to_string();
        let nullable = s.null_count() > 0;

        let mut field_arrays = Vec::with_capacity(n);
        for i in 0..n {
            let arr2 = s.to_arrow(i, CompatLevel::oldest());
            let (array_arc, field) =
                crate::ffi::polars::import_chunk(&name, nullable, arr2)?;
            let array = Arc::try_unwrap(array_arc).unwrap_or_else(|arc| (*arc).clone());
            field_arrays.push(FieldArray::new(field, array));
        }
        Ok(SuperArray::from_field_array_chunks(field_arrays))
    }
}

#[cfg(feature = "cast_polars")]
impl From<&polars::prelude::Series> for SuperArray {
    fn from(s: &polars::prelude::Series) -> Self {
        SuperArray::from_polars(s)
    }
}

impl Display for SuperArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let name = self
            .field
            .as_ref()
            .map(|f| f.name.as_str())
            .unwrap_or("<unnamed>");
        let dtype = if let Some(chunk) = self.chunks.first() {
            format!("{:?}", chunk.arrow_type())
        } else if let Some(ref field) = self.field {
            format!("{:?}", field.dtype)
        } else {
            "<empty>".to_string()
        };

        writeln!(
            f,
            "SuperArray \"{}\" [{} rows, {} chunks] (dtype: {})",
            name,
            self.len(),
            self.n_chunks(),
            dtype
        )?;

        for (i, chunk) in self.chunks.iter().enumerate() {
            let null_count = self
                .null_counts
                .as_ref()
                .and_then(|nc| nc.get(i).copied())
                .map(|n| n.to_string())
                .unwrap_or_else(|| "?".to_string());
            writeln!(
                f,
                "  ├─ Chunk {i}: {} rows, nulls: {}",
                chunk.len(),
                null_count
            )?;
            let indent = "    │ ";
            for line in format!("{}", chunk).lines() {
                writeln!(f, "{indent}{line}")?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "views")]
    use crate::NumericArray;
    use crate::ffi::arrow_dtype::ArrowType;
    use crate::{Array, Field, Vec64, fa_i32};

    #[allow(dead_code)]
    fn field(name: &str, dtype: ArrowType, nullable: bool) -> Field {
        Field {
            name: name.to_string(),
            dtype,
            nullable,
            metadata: Default::default(),
        }
    }

    fn int_array(data: &[i32]) -> Array {
        Array::from_int32(crate::IntegerArray::<i32> {
            data: Vec64::from_slice(data).into(),
            null_mask: None,
        })
    }

    #[test]
    fn test_new_and_push_array() {
        let mut ca = SuperArray::new();
        assert_eq!(ca.n_chunks(), 0);
        ca.push(int_array(&[1, 2, 3]));
        assert_eq!(ca.n_chunks(), 1);
        assert_eq!(ca.len(), 3);
        ca.push(int_array(&[4, 5]));
        assert_eq!(ca.n_chunks(), 2);
        assert_eq!(ca.len(), 5);
    }

    #[test]
    fn test_new_and_push_field_array() {
        let mut ca = SuperArray::new();
        assert_eq!(ca.n_chunks(), 0);
        ca.push_field_array(fa_i32!("a", 1, 2, 3));
        assert_eq!(ca.n_chunks(), 1);
        assert_eq!(ca.len(), 3);
        assert!(ca.field().is_some());
        ca.push_field_array(fa_i32!("a", 4, 5));
        assert_eq!(ca.n_chunks(), 2);
        assert_eq!(ca.len(), 5);
    }

    #[test]
    #[should_panic(expected = "Chunk ArrowType mismatch")]
    fn test_type_mismatch() {
        let mut ca = SuperArray::new();
        ca.push(int_array(&[1, 2, 3]));
        let wrong = Array::from_float64(crate::FloatArray::<f64> {
            data: Vec64::from_slice(&[1.0, 2.0]).into(),
            null_mask: None,
        });
        ca.push(wrong);
    }

    #[test]
    #[should_panic(expected = "Chunk field name mismatch")]
    fn test_name_mismatch() {
        let mut ca = SuperArray::new();
        ca.push_field_array(fa_i32!("a", 1, 2, 3));
        ca.push_field_array(fa_i32!("b", 4, 5)); // wrong name
    }

    #[test]
    fn test_from_field_array_chunks() {
        let c1 = fa_i32!("a", 1, 2, 3);
        let c2 = fa_i32!("a", 4);
        let ca = SuperArray::from_field_array_chunks(vec![c1.clone(), c2.clone()]);
        assert_eq!(ca.n_chunks(), 2);
        assert_eq!(ca.len(), 4);
        assert_eq!(ca.field().unwrap().name, "a");
    }

    #[test]
    fn test_from_arrays() {
        let a1 = int_array(&[1, 2, 3]);
        let a2 = int_array(&[4, 5]);
        let ca = SuperArray::from_arrays(vec![a1, a2]);
        assert_eq!(ca.n_chunks(), 2);
        assert_eq!(ca.len(), 5);
        assert!(ca.field().is_none());
    }

    #[test]
    #[should_panic(expected = "from_field_array_chunks: input chunks cannot be empty")]
    fn test_from_field_array_chunks_empty() {
        let _ = SuperArray::from_field_array_chunks(Vec::new());
    }

    #[cfg(feature = "views")]
    #[test]
    fn test_slice_and_materialise() {
        use crate::NumericArray;

        let c1 = fa_i32!("a", 10, 20, 30);
        let c2 = fa_i32!("a", 40, 50);
        let ca = SuperArray::from_field_array_chunks(vec![c1.clone(), c2.clone()]);
        let sl = ca.slice(2, 3);
        assert_eq!(sl.len, 3);
        let arr = sl.consolidate();
        if let Array::NumericArray(NumericArray::Int32(ia)) = arr {
            assert_eq!(&*ia.data, &[30, 40, 50]);
        } else {
            panic!("Expected Int32");
        }
    }

    #[cfg(feature = "views")]
    #[test]
    fn test_from_slices() {
        let c1 = fa_i32!("a", 10, 20, 30);
        let c2 = fa_i32!("a", 40, 50);
        let ca = SuperArray::from_field_array_chunks(vec![c1.clone(), c2.clone()]);

        let sl = ca.slice(1, 4);
        let slices = &sl.slices;
        let field = c1.field.clone();
        let ca2 = SuperArray::from_slices(slices, field);
        assert_eq!(ca2.n_chunks(), 2);
        assert_eq!(ca2.len(), 4);
    }

    #[test]
    fn test_is_empty_and_default() {
        let ca = SuperArray::default();
        assert!(ca.is_empty());
        let ca2 = SuperArray::from_chunks(vec![fa_i32!("a", 1)]);
        assert!(!ca2.is_empty());
    }

    #[test]
    fn test_metadata_accessors() {
        let ca = SuperArray::from_chunks(vec![fa_i32!("z", 1, 2, 3, 4)]);
        assert_eq!(ca.arrow_type(), ArrowType::Int32);
        assert!(!ca.is_nullable());
        assert_eq!(ca.field().unwrap().name, "z");
        assert_eq!(ca.chunks().len(), 1);
    }

    #[cfg(feature = "views")]
    #[test]
    fn test_insert_rows_into_first_chunk() {
        let mut ca = SuperArray::from_chunks(vec![fa_i32!("a", 1, 2, 3), fa_i32!("a", 4, 5)]);

        let other = SuperArray::from_arrays(vec![int_array(&[99, 88])]);

        ca.insert_rows(1, other).unwrap();

        assert_eq!(ca.len(), 7);
        assert_eq!(ca.n_chunks(), 4);

        let result = ca.consolidate();
        if let Array::NumericArray(NumericArray::Int32(ia)) = result {
            assert_eq!(&*ia.data, &[1, 99, 88, 2, 3, 4, 5]);
        } else {
            panic!("Expected Int32");
        }
    }

    #[cfg(feature = "views")]
    #[test]
    fn test_insert_rows_into_second_chunk() {
        let mut ca = SuperArray::from_chunks(vec![fa_i32!("a", 1, 2), fa_i32!("a", 3, 4, 5)]);

        let other = SuperArray::from_arrays(vec![int_array(&[99])]);

        ca.insert_rows(3, other).unwrap();

        assert_eq!(ca.len(), 6);
        assert_eq!(ca.n_chunks(), 4);

        let result = ca.consolidate();
        if let Array::NumericArray(NumericArray::Int32(ia)) = result {
            assert_eq!(&*ia.data, &[1, 2, 3, 99, 4, 5]);
        } else {
            panic!("Expected Int32");
        }
    }

    #[cfg(feature = "views")]
    #[test]
    fn test_insert_rows_prepend() {
        let mut ca = SuperArray::from_chunks(vec![fa_i32!("a", 1, 2, 3)]);

        let other = SuperArray::from_arrays(vec![int_array(&[99])]);

        ca.insert_rows(0, other).unwrap();

        assert_eq!(ca.len(), 4);

        let result = ca.consolidate();
        if let Array::NumericArray(NumericArray::Int32(ia)) = result {
            assert_eq!(&*ia.data, &[99, 1, 2, 3]);
        } else {
            panic!("Expected Int32");
        }
    }

    #[cfg(feature = "views")]
    #[test]
    fn test_insert_rows_append() {
        let mut ca = SuperArray::from_chunks(vec![fa_i32!("a", 1, 2, 3)]);

        let other = SuperArray::from_arrays(vec![int_array(&[99])]);

        ca.insert_rows(3, other).unwrap();

        assert_eq!(ca.len(), 4);

        let result = ca.consolidate();
        if let Array::NumericArray(NumericArray::Int32(ia)) = result {
            assert_eq!(&*ia.data, &[1, 2, 3, 99]);
        } else {
            panic!("Expected Int32");
        }
    }

    #[test]
    fn test_insert_rows_type_mismatch() {
        let mut ca = SuperArray::from_arrays(vec![int_array(&[1, 2, 3])]);

        let wrong_type = Array::from_float64(crate::FloatArray::<f64> {
            data: Vec64::from_slice(&[99.0]).into(),
            null_mask: None,
        });
        let other = SuperArray::from_arrays(vec![wrong_type]);

        let result = ca.insert_rows(0, other);
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_rows_out_of_bounds() {
        let mut ca = SuperArray::from_arrays(vec![int_array(&[1, 2, 3])]);
        let other = SuperArray::from_arrays(vec![int_array(&[99])]);

        let result = ca.insert_rows(10, other);
        assert!(result.is_err());
    }

    #[test]
    fn test_rechunk_uniform() {
        let mut ca = SuperArray::from_chunks(vec![
            fa_i32!("a", 1, 2, 3),
            fa_i32!("a", 4, 5),
            fa_i32!("a", 6, 7, 8, 9),
        ]);

        ca.rechunk(RechunkStrategy::Count(3)).unwrap();

        assert_eq!(ca.n_chunks(), 3);
        assert_eq!(ca.len(), 9);
        assert_eq!(ca.chunks[0].len(), 3);
        assert_eq!(ca.chunks[1].len(), 3);
        assert_eq!(ca.chunks[2].len(), 3);

        let result = ca.consolidate();
        if let Array::NumericArray(NumericArray::Int32(ia)) = result {
            assert_eq!(&*ia.data, &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        } else {
            panic!("Expected Int32");
        }
    }

    #[test]
    fn test_rechunk_auto() {
        let mut ca = SuperArray::from_chunks(vec![fa_i32!("a", 1, 2, 3), fa_i32!("a", 4, 5)]);

        ca.rechunk(RechunkStrategy::Auto).unwrap();

        assert_eq!(ca.n_chunks(), 1);
        assert_eq!(ca.len(), 5);
    }

    #[test]
    #[cfg(feature = "size")]
    fn test_rechunk_by_memory() {
        let mut ca = SuperArray::from_chunks(vec![
            fa_i32!("a", 1, 2, 3, 4, 5, 6, 7, 8),
            fa_i32!("a", 9, 10, 11, 12),
        ]);

        // i32 is 4 bytes, so 16 bytes = 4 elements
        ca.rechunk(RechunkStrategy::Memory(16)).unwrap();

        assert_eq!(ca.len(), 12);
        assert!(ca.n_chunks() >= 3);

        let result = ca.consolidate();
        if let Array::NumericArray(NumericArray::Int32(ia)) = result {
            assert_eq!(&*ia.data, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        } else {
            panic!("Expected Int32");
        }
    }

    #[test]
    fn test_rechunk_uniform_zero_error() {
        let mut ca = SuperArray::from_chunks(vec![fa_i32!("a", 1, 2, 3)]);

        let result = ca.rechunk(RechunkStrategy::Count(0));
        assert!(result.is_err());
    }

    #[test]
    fn test_rechunk_empty_array() {
        let mut ca = SuperArray::new();
        ca.rechunk(RechunkStrategy::Auto).unwrap();
        assert_eq!(ca.n_chunks(), 0);
    }

    #[test]
    fn test_rechunk_preserves_data_order() {
        let mut ca = SuperArray::from_chunks(vec![
            fa_i32!("a", 10, 20),
            fa_i32!("a", 30),
            fa_i32!("a", 40, 50, 60),
        ]);

        ca.rechunk(RechunkStrategy::Count(2)).unwrap();

        let result = ca.consolidate();
        if let Array::NumericArray(NumericArray::Int32(ia)) = result {
            assert_eq!(&*ia.data, &[10, 20, 30, 40, 50, 60]);
        } else {
            panic!("Expected Int32");
        }
    }

    /// Pushing categorical chunks into a SuperArray installs the shared
    /// dictionary on the first push and merges subsequent chunks into
    /// it. Every chunk holds a clone of the same `Dictionary` handle
    /// (Arc-shared), so a `shares_with` check is true across them, and
    /// growth observed at any one chunk is immediately visible at all
    /// others (single atomic store).
    #[cfg(all(
        feature = "shared_dict",
        any(not(feature = "default_categorical_8"), feature = "extended_categorical")
    ))]
    #[test]
    fn test_shared_dict_add_across_pushes() {
        use crate::TextArray;
        use crate::arr_cat32;

        let chunk1: Array = arr_cat32!("a", "b", "a");
        let chunk2: Array = arr_cat32!("b", "c", "a");

        let mut sa = SuperArray::new();
        sa.push(chunk1);
        sa.push(chunk2);

        let d0 = match &sa.chunks[0] {
            Array::TextArray(TextArray::Categorical32(a)) => a.dictionary.clone(),
            _ => panic!("expected Categorical32"),
        };
        let d1 = match &sa.chunks[1] {
            Array::TextArray(TextArray::Categorical32(a)) => a.dictionary.clone(),
            _ => panic!("expected Categorical32"),
        };
        // Both chunks hold a clone of the same `Dictionary` handle - the
        // single atomic store backing the sharing group.
        assert!(d0.shares_with(&d1));
        // Every chunk observes the union of all interned strings,
        // including those added after its own push.
        assert_eq!(d0.values(), &["a", "b", "c"]);
        assert_eq!(d1.values(), &["a", "b", "c"]);

        let chunk3: Array = arr_cat32!("a", "b");
        sa.push(chunk3);
        let d2 = match &sa.chunks[2] {
            Array::TextArray(TextArray::Categorical32(a)) => a.dictionary.clone(),
            _ => panic!("expected Categorical32"),
        };
        assert!(d0.shares_with(&d2));
        assert_eq!(d2.values(), &["a", "b", "c"]);

        // category_manager returns Some; interning known values is idempotent.
        let dispatch = sa.category_manager().expect("dispatch present after push");
        match dispatch {
            CategoryManagerT::U32(d) => {
                assert_eq!(d.add_cat("a"), Ok(0));
                assert_eq!(d.add_cat("b"), Ok(1));
                assert_eq!(d.add_cat("c"), Ok(2));
            }
            _ => panic!("expected U32 dispatch"),
        }
    }
}
