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

//! # **SuperNdArray** - *Chunked N-dimensional array*
//!
//! The N-dimensional equivalent of [`SuperTable`](crate::SuperTable): groups multiple `NdArray`
//! batches under a shared rank and trailing shape. Batches vary only in the
//! size of their leading axis (axis 0), so element access, iteration, and
//! slicing work transparently across chunk boundaries.
//!
//! ## Typical uses
//! - Streaming sensor or telemetry data where each chunk is a temporal batch.
//! - Partitioned reads from storage where batches arrive independently.
//! - Memory-bounded ingestion where you want to keep batches separate
//!   until consolidation is needed.
//!
//! ## Relationship to XArray
//! `XArray` wraps NdArray, NdArrayV, and SuperNdArray behind a single type,
//! adding named dimensions and coordinate labels. Use SuperNdArray directly
//! when you need raw chunked storage without label overhead.

use std::fmt;

use crate::enums::error::MinarrowError;
use crate::enums::shape_dim::ShapeDim;
use crate::structs::chunked::super_array::RechunkStrategy;
use crate::structs::ndarray::NdArray;
use crate::traits::concatenate::Concatenate;
use crate::traits::consolidate::Consolidate;
use crate::traits::shape::Shape;
use crate::traits::type_unions::Float;
use crate::structs::ndarray::NdArrayIter;
#[cfg(feature = "views")]
use crate::Vec64;
#[cfg(all(feature = "views", feature = "select"))]
use crate::structs::ndarray::gather_obs_impl;
#[cfg(all(feature = "views", feature = "select"))]
use crate::traits::selection::{AxisSelection, DataSelector, RowSelection};
#[cfg(feature = "views")]
use crate::structs::views::chunked::super_ndarray_view::SuperNdArrayV;
#[cfg(feature = "views")]
use crate::structs::views::ndarray_view::NdArrayV;

#[cfg(feature = "parallel_proc")]
use rayon::prelude::*;

/// Chunked N-dimensional array.
///
/// All batches share the same rank and non-leading dimensions.
/// Only the leading axis (axis 0) may differ between batches.
/// Access by global index is transparent - the chunk boundary
/// is resolved internally.
#[derive(Clone)]
pub struct SuperNdArray<T> {
    pub batches: Vec<NdArray<T>>,
    ndim: usize,
    inner_shape: Vec<usize>,
    pub name: String,
}

impl<T: Float> SuperNdArray<T> {
    /// Create an empty SuperNdArray.
    pub fn new(name: impl Into<String>) -> Self {
        SuperNdArray {
            batches: Vec::new(),
            ndim: 0,
            inner_shape: Vec::new(),
            name: name.into(),
        }
    }

    /// Create from existing batches. Panics if shapes are incompatible.
    pub fn from_batches(batches: Vec<NdArray<T>>, name: impl Into<String>) -> Self {
        if batches.is_empty() {
            return Self::new(name);
        }
        let ndim = batches[0].ndim();
        let inner_shape: Vec<usize> = batches[0].shape()[1..].to_vec();
        for (i, chunk) in batches.iter().enumerate().skip(1) {
            assert_eq!(
                chunk.ndim(), ndim,
                "SuperNdArray: chunk {} has rank {} but expected {}", i, chunk.ndim(), ndim
            );
            assert_eq!(
                &chunk.shape()[1..], inner_shape.as_slice(),
                "SuperNdArray: chunk {} inner shape mismatch", i
            );
        }
        SuperNdArray { batches, ndim, inner_shape, name: name.into() }
    }

    /// Append a chunk. Validates shape compatibility.
    pub fn push(&mut self, chunk: NdArray<T>) {
        if self.batches.is_empty() {
            self.ndim = chunk.ndim();
            self.inner_shape = chunk.shape()[1..].to_vec();
        } else {
            assert_eq!(
                chunk.ndim(), self.ndim,
                "SuperNdArray::push: rank {} does not match expected {}", chunk.ndim(), self.ndim
            );
            assert_eq!(
                &chunk.shape()[1..], self.inner_shape.as_slice(),
                "SuperNdArray::push: inner shape {:?} does not match expected {:?}",
                &chunk.shape()[1..], self.inner_shape
            );
        }
        self.batches.push(chunk);
    }

    // ****************************************************************
    // Introspection
    // ****************************************************************

    /// Number of batches.
    #[inline]
    pub fn n_batches(&self) -> usize { self.batches.len() }

    /// Shared rank.
    #[inline]
    pub fn ndim(&self) -> usize { self.ndim }

    /// Total logical elements across all batches i.e. the product of shape.
    #[inline]
    pub fn len(&self) -> usize {
        self.batches.iter().map(|c| c.len()).sum()
    }

    #[inline]
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Total leading-axis (axis 0) observations across all batches.
    #[inline]
    pub fn n_obs(&self) -> usize {
        self.batches.iter().map(|c| c.shape()[0]).sum()
    }

    /// Dimensions shared across all batches i.e. shape[1..].
    #[inline]
    pub fn inner_shape(&self) -> &[usize] { &self.inner_shape }

    /// Get batch by index.
    #[inline]
    pub fn batch(&self, idx: usize) -> Option<&NdArray<T>> { self.batches.get(idx) }

    /// The constituent batches.
    #[inline]
    pub fn batches(&self) -> &[NdArray<T>] { &self.batches }

    /// Consume into the constituent batches.
    #[inline]
    pub fn into_batches(self) -> Vec<NdArray<T>> { self.batches }

    /// Logical shape as if consolidated. Returns a temporary vec, not a
    /// slice reference, since the full shape doesn't exist as a contiguous
    /// field - axis 0 is the sum across batches.
    pub fn shape(&self) -> Vec<usize> {
        let mut s = vec![self.n_obs()];
        s.extend_from_slice(&self.inner_shape);
        s
    }

    /// Strides of the first batch. Strides above axis 0 scale with each
    /// batch's own leading-axis length, so these describe the first batch
    /// only and are not shared across batches. Use them for per-batch
    /// access rather than for the consolidated array.
    pub fn strides(&self) -> &[usize] {
        if self.batches.is_empty() { return &[]; }
        self.batches[0].strides()
    }

    /// Column slice from a 2D chunked array. Consolidates the column
    /// across all batches into a contiguous allocation.
    pub fn col(&self, c: usize) -> Vec<T> {
        assert_eq!(self.ndim, 2, "col() requires a 2D array");
        let mut result = Vec::with_capacity(self.n_obs());
        for batch in &self.batches {
            result.extend_from_slice(batch.col(c));
        }
        result
    }

    /// Returns a zero-copy [`SuperNdArrayV`] over `[offset .. offset + len)`
    /// axis-0 observations, spanning batch boundaries. Each overlapped batch
    /// contributes a windowed slice view.
    #[cfg(feature = "views")]
    pub fn slice(&self, mut offset: usize, mut len: usize) -> SuperNdArrayV<T> {
        assert!(
            offset + len <= self.n_obs(),
            "SuperNdArray::slice: window [{}, {}) out of bounds (n_obs {})",
            offset, offset + len, self.n_obs()
        );

        let mut slices = Vec::new();
        for batch in &self.batches {
            let base_obs = batch.shape()[0];
            if offset >= base_obs {
                offset -= base_obs;
                continue;
            }

            let take = (base_obs - offset).min(len);
            let mut window_shape = vec![take];
            window_shape.extend_from_slice(&batch.shape()[1..]);
            slices.push(NdArrayV::new(
                batch.clone(),
                offset * batch.strides()[0],
                &window_shape,
                batch.strides(),
            ));

            len -= take;
            if len == 0 {
                break;
            }
            offset = 0;
        }

        SuperNdArrayV::from_slices(slices, self.ndim, self.inner_shape.clone(), self.name.clone())
    }

    /// Zero-copy view of a single observation (axis-0 element)
    /// across batch boundaries.
    ///
    /// Returns an (N-1)-dimensional `NdArrayV` view. For a 2D chunked
    /// array with shape `[n, m]`, returns a 1D view of shape `[m]`. For 3D
    /// `[n, m, k]`, returns 2D `[m, k]`. Requires rank 2 or higher - for
    /// scalar access on a 1D array use `get(&[i])`.
    #[cfg(feature = "views")]
    pub fn obs(&self, idx: usize) -> NdArrayV<T> {
        let (chunk_idx, local) = self.resolve(idx);
        self.batches[chunk_idx].obs(local)
    }

    /// BLAS row count (2D).
    #[inline]
    pub fn m(&self) -> i32 {
        assert_eq!(self.ndim, 2, "m() requires a 2D array");
        self.n_obs() as i32
    }

    /// BLAS column count (2D).
    #[inline]
    pub fn n(&self) -> i32 {
        assert_eq!(self.ndim, 2, "n() requires a 2D array");
        self.inner_shape[0] as i32
    }

    /// BLAS leading dimension of the first batch (2D). The leading dimension
    /// equals each batch's own row count, so this value applies to the first
    /// batch only, not to the whole chunked array.
    #[inline]
    pub fn lda(&self) -> i32 {
        assert_eq!(self.ndim, 2, "lda() requires a 2D array");
        if self.batches.is_empty() { return 0; }
        self.batches[0].lda()
    }

    // ****************************************************************
    // Global element access
    // ****************************************************************

    /// Get element by global N-dimensional index, transparently resolving
    /// which chunk contains it. The first index is the global axis-0 position.
    pub fn get(&self, indices: &[usize]) -> T {
        let (chunk_idx, local_row) = self.resolve(indices[0]);
        let mut local = indices.to_vec();
        local[0] = local_row;
        self.batches[chunk_idx].get(&local)
    }

    /// Set element by global index. Triggers copy-on-write if the
    /// target chunk's buffer is shared.
    pub fn set(&mut self, indices: &[usize], value: T) {
        let (chunk_idx, local_row) = self.resolve(indices[0]);
        let mut local = indices.to_vec();
        local[0] = local_row;
        self.batches[chunk_idx].set(&local, value);
    }

    /// Resolve a global axis-0 index to (batch_index, local_index).
    /// Returns an error if the index is out of bounds.
    pub(crate) fn try_resolve(&self, global: usize) -> Result<(usize, usize), MinarrowError> {
        let mut remaining = global;
        for (i, chunk) in self.batches.iter().enumerate() {
            let n = chunk.shape()[0];
            if remaining < n {
                return Ok((i, remaining));
            }
            remaining -= n;
        }
        Err(MinarrowError::IndexError(
            format!("global index {} out of bounds (n_obs {})", global, self.n_obs())
        ))
    }

    /// Resolve a global axis-0 index to (batch_index, local_index).
    /// Panics if the index is out of bounds.
    fn resolve(&self, global: usize) -> (usize, usize) {
        self.try_resolve(global).unwrap_or_else(|e| panic!("{}", e))
    }

    // ****************************************************************
    // Iteration
    // ****************************************************************

    /// Iterate over batches.
    #[inline]
    pub fn iter_batches(&self) -> std::slice::Iter<'_, NdArray<T>> {
        self.batches.iter()
    }

    /// Mutable iteration over batches.
    #[inline]
    pub fn iter_batches_mut(&mut self) -> std::slice::IterMut<'_, NdArray<T>> {
        self.batches.iter_mut()
    }

    /// Parallel iteration over batches.
    #[cfg(feature = "parallel_proc")]
    #[inline]
    pub fn par_iter_batches(&self) -> rayon::slice::Iter<'_, NdArray<T>>
    where
        T: Send + Sync,
    {
        self.batches.par_iter()
    }

    /// Parallel iterator over axis-0 observations across all batches.
    /// Each item is the global observation index and a zero-copy
    /// `NdArrayV` view. Batch boundaries are resolved transparently.
    #[cfg(all(feature = "parallel_proc", feature = "views"))]
    pub fn par_iter_obs(&self) -> impl rayon::iter::ParallelIterator<Item = (usize, NdArrayV<T>)> + '_
    where
        T: Send + Sync,
    {
        use rayon::prelude::*;
        let n_obs = self.n_obs();
        (0..n_obs).into_par_iter().map(move |i| (i, self.obs(i)))
    }

    // ****************************************************************
    // Apply
    // ****************************************************************

    /// Apply a function to every logical element, returning a new chunked
    /// array with the same batch boundaries and name. The closure brings
    /// the computation, so any kernel under `kernels` runs through this
    /// entry point.
    pub fn apply(&self, f: impl Fn(T) -> T) -> SuperNdArray<T> {
        let batches: Vec<NdArray<T>> = self.batches.iter().map(|b| b.apply(&f)).collect();
        SuperNdArray::from_batches(batches, self.name.clone())
    }

    /// Apply a function to every logical element in place, with no
    /// allocation. Copy-on-write triggers per batch when views share a
    /// batch's buffer.
    pub fn apply_mut(&mut self, f: impl Fn(T) -> T) {
        for batch in &mut self.batches {
            batch.apply_mut(&f);
        }
    }

    /// Apply a function to every 1D lane along the given axis, collapsing
    /// that axis. Each lane arrives as an [`NdArrayV`] and the closure
    /// returns one value for it, mirroring [`NdArray::apply_axis`].
    ///
    /// Lanes along axis 1 and above live inside single batches, so the
    /// result keeps this array's chunk boundaries. Lanes along axis 0
    /// cross batch boundaries, so each gathers into a contiguous buffer
    /// and the result holds one batch with the trailing shape.
    #[cfg(feature = "views")]
    pub fn apply_axis(&self, axis: usize, mut f: impl FnMut(NdArrayV<T>) -> T) -> SuperNdArray<T> {
        assert!(self.ndim >= 2, "apply_axis requires a 2D or higher array");
        assert!(
            axis < self.ndim,
            "apply_axis: axis {} out of bounds for {}D array", axis, self.ndim
        );

        if axis > 0 {
            let batches: Vec<NdArray<T>> = self
                .batches
                .iter()
                .map(|b| b.apply_axis(axis, &mut f))
                .collect();
            return SuperNdArray::from_batches(batches, self.name.clone());
        }

        // Axis-0 lanes span batches. Walk the inner positions in
        // column-major order, gathering each lane batch by batch.
        let n_obs = self.n_obs();
        let inner_positions: usize = self.inner_shape.iter().product();
        let mut results = Vec64::with_capacity(inner_positions);
        let mut inner_idx = vec![0usize; self.inner_shape.len()];
        let mut lane_idx = vec![0usize; self.ndim];
        for _ in 0..inner_positions {
            let mut lane = Vec64::with_capacity(n_obs);
            lane_idx[1..].copy_from_slice(&inner_idx);
            for batch in &self.batches {
                let obs = batch.shape()[0];
                for i in 0..obs {
                    lane_idx[0] = i;
                    lane.push(batch.get(&lane_idx));
                }
            }
            let lane_arr = NdArray::from_slice(&lane, &[n_obs]);
            results.push(f(lane_arr.as_view()));

            let mut carry = true;
            for d in 0..inner_idx.len() {
                if carry {
                    inner_idx[d] += 1;
                    if inner_idx[d] < self.inner_shape[d] {
                        carry = false;
                    } else {
                        inner_idx[d] = 0;
                    }
                }
            }
        }
        SuperNdArray::from_batches(
            vec![NdArray::from_slice(&results, &self.inner_shape)],
            self.name.clone(),
        )
    }

    /// Apply a transformation to each batch, producing a new chunked array.
    ///
    /// The closure receives each batch and returns a transformed batch,
    /// mirroring `Table::apply_cols` with the batch as the unit of work.
    /// Returned batches must share rank and trailing shape.
    pub fn apply_batches<E>(
        &self,
        mut f: impl FnMut(&NdArray<T>) -> Result<NdArray<T>, E>,
    ) -> Result<SuperNdArray<T>, E> {
        let batches = self
            .batches
            .iter()
            .map(|b| f(b))
            .collect::<Result<Vec<_>, E>>()?;
        Ok(SuperNdArray::from_batches(batches, self.name.clone()))
    }

    /// Apply an in-place transformation to each batch, with no allocation
    /// beyond what the closure itself performs.
    pub fn apply_batches_mut<E>(
        &mut self,
        mut f: impl FnMut(&mut NdArray<T>) -> Result<(), E>,
    ) -> Result<(), E> {
        for batch in &mut self.batches {
            f(batch)?;
        }
        Ok(())
    }

    // ****************************************************************
    // Rechunking
    // ****************************************************************

    /// Redistribute data across batches.
    ///
    /// - `Count(n)` - uniform batches of n leading-axis elements
    /// - `Auto` - default chunk size of 8192 leading-axis elements
    /// - `Memory(bytes)` - target byte size per chunk (requires `size` feature)
    pub fn rechunk(&mut self, strategy: RechunkStrategy) -> Result<(), MinarrowError> {
        if self.batches.is_empty() || self.n_obs() == 0 {
            return Ok(());
        }

        let chunk_size = match strategy {
            RechunkStrategy::Count(size) => {
                if size == 0 {
                    return Err(MinarrowError::IndexError(
                        "rechunk: chunk size must be > 0".to_string(),
                    ));
                }
                size
            }
            RechunkStrategy::Auto => 8192,
            #[cfg(feature = "size")]
            RechunkStrategy::Memory(target_bytes) => {
                use crate::traits::byte_size::ByteSize;
                if target_bytes == 0 {
                    return Err(MinarrowError::IndexError(
                        "rechunk: target bytes must be > 0".to_string(),
                    ));
                }
                // Estimate rows per chunk from the first chunk's byte density
                let first = &self.batches[0];
                let bytes_per_row = first.est_bytes().max(1) / first.shape()[0].max(1);
                (target_bytes / bytes_per_row.max(1)).max(1)
            }
        };

        // Capture state before moving self
        let total_obs = self.n_obs();
        let ndim = self.ndim;
        let inner_shape = self.inner_shape.clone();
        let name = self.name.clone();

        // Consolidate then re-split
        let old = std::mem::replace(self, SuperNdArray::new(&name));
        let consolidated = old.consolidate();
        let strides = consolidated.strides().to_vec();
        let buf = consolidated.as_slice();

        let mut new_batches = Vec::with_capacity(total_obs.div_ceil(chunk_size));
        let mut row_offset = 0;

        while row_offset < total_obs {
            let n = (total_obs - row_offset).min(chunk_size);
            let mut chunk_shape = vec![n];
            chunk_shape.extend_from_slice(&inner_shape);

            if ndim <= 1 {
                let chunk = NdArray::from_slice(&buf[row_offset..row_offset + n], &chunk_shape);
                new_batches.push(chunk);
            } else {
                let mut chunk = NdArray::new(&chunk_shape);
                let dst_stride = chunk.strides()[1];
                let n_cols: usize = inner_shape.iter().product::<usize>().max(1);
                let dst = chunk.as_mut_slice();
                for c in 0..n_cols {
                    let src_start = c * strides[1] + row_offset;
                    let dst_start = c * dst_stride;
                    dst[dst_start..dst_start + n]
                        .copy_from_slice(&buf[src_start..src_start + n]);
                }
                new_batches.push(chunk);
            }
            row_offset += n;
        }

        *self = SuperNdArray { ndim, inner_shape, batches: new_batches, name };
        Ok(())
    }
}

// *** Axis selection: snd.s(nd![1..4, 2]) *************************

/// Selection across every axis at once. The axis-0 selection windows
/// across batch boundaries, and trailing-axis selections narrow each
/// slice. Zero-copy. Delegates to the full-window [`SuperNdArrayV`].
///
/// An axis-0 single index keeps the dimension as a one-observation
/// window, since a chunked view cannot collapse its chunking axis - use
/// `obs` to collapse a single observation. Trailing single indices
/// collapse their dimensions as usual.
#[cfg(all(feature = "views", feature = "select"))]
impl<T: Float> AxisSelection for SuperNdArray<T> {
    type View = SuperNdArrayV<T>;

    fn s(&self, selection: &[&dyn DataSelector]) -> SuperNdArrayV<T> {
        self.slice(0, self.n_obs()).s(selection)
    }

    fn get_axis_count(&self) -> usize {
        self.ndim
    }
}

// *** Row selection: snd.r(0..10) *********************************

/// Axis-0 observation selection across batch boundaries. Contiguous
/// ranges return a zero-copy [`SuperNdArrayV`] window. Index arrays
/// gather the selected observations into one owned batch wrapped in a
/// single-slice view.
#[cfg(all(feature = "views", feature = "select"))]
impl<T: Float> RowSelection for SuperNdArray<T> {
    type View = SuperNdArrayV<T>;

    fn r<S: DataSelector>(&self, selection: S) -> SuperNdArrayV<T> {
        if self.batches.is_empty() {
            return SuperNdArrayV::from_slices(
                Vec::new(),
                self.ndim,
                self.inner_shape.clone(),
                self.name.clone(),
            );
        }
        let indices = selection.resolve_indices(self.n_obs());
        if selection.is_contiguous() {
            let start = indices.first().copied().unwrap_or(0);
            return self.slice(start, indices.len());
        }
        let gathered = gather_obs_impl(
            &indices,
            &self.shape(),
            Some(self.name.clone()),
            |idx| self.get(idx),
        );
        SuperNdArrayV::from_slices(
            vec![NdArrayV::from_ndarray(gathered)],
            self.ndim,
            self.inner_shape.clone(),
            self.name.clone(),
        )
    }

    fn get_row_count(&self) -> usize {
        self.n_obs()
    }
}

// ****************************************************************
// IntoIterator - element-wise iteration across batches
// ****************************************************************

/// Iterating a SuperNdArray yields `T` values in column-major order,
/// seamlessly crossing chunk boundaries.
impl<'a, T: Float> IntoIterator for &'a SuperNdArray<T> {
    type Item = T;
    type IntoIter = SuperNdArrayIter<'a, T>;

    fn into_iter(self) -> SuperNdArrayIter<'a, T> {
        SuperNdArrayIter {
            batches: &self.batches,
            chunk_idx: 0,
            inner: None,
        }
    }
}

/// Iterator that chains across chunk boundaries transparently.
pub struct SuperNdArrayIter<'a, T> {
    batches: &'a [NdArray<T>],
    chunk_idx: usize,
    inner: Option<NdArrayIter<'a, T>>,
}

impl<'a, T: Float> Iterator for SuperNdArrayIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        loop {
            if let Some(ref mut it) = self.inner {
                if let Some(v) = it.next() {
                    return Some(v);
                }
            }
            // Advance to next chunk
            if self.chunk_idx >= self.batches.len() {
                return None;
            }
            self.inner = Some((&self.batches[self.chunk_idx]).into_iter());
            self.chunk_idx += 1;
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let inner_remaining = self.inner.as_ref().map_or(0, |it| it.len());
        let remaining_batches: usize = self.batches[self.chunk_idx..]
            .iter()
            .map(|c| c.len())
            .sum();
        let total = inner_remaining + remaining_batches;
        (total, Some(total))
    }
}

impl<'a, T: Float> ExactSizeIterator for SuperNdArrayIter<'a, T> {}

// ****************************************************************
// Consolidate
// ****************************************************************

impl<T: Float> Consolidate for SuperNdArray<T> {
    type Output = NdArray<T>;

    /// Materialise into a single contiguous column-major [`NdArray`].
    /// Each source batch reads at its own strides and writes straight
    /// into the result, with unit-stride axis-0 runs taking the
    /// memcpy path.
    fn consolidate(self) -> NdArray<T> {
        if self.batches.is_empty() {
            let mut result = NdArray::new(&[0]);
            result.name = Some(self.name);
            return result;
        }
        if self.batches.len() == 1 {
            let mut result = self.batches.into_iter().next().unwrap();
            if !result.is_contiguous() {
                result = result.to_contiguous();
            }
            result.name = Some(self.name);
            return result;
        }

        let total_axis0: usize = self.batches.iter().map(|c| c.shape()[0]).sum();
        let mut full_shape = vec![total_axis0];
        full_shape.extend_from_slice(&self.inner_shape);

        let mut result = NdArray::new(&full_shape);

        if self.ndim == 1 {
            let dst = result.as_mut_slice();
            let mut pos = 0;
            for chunk in &self.batches {
                let n = chunk.shape()[0];
                let s0 = chunk.strides()[0];
                let src = chunk.as_slice();
                if s0 == 1 {
                    dst[pos..pos + n].copy_from_slice(&src[..n]);
                } else {
                    for i in 0..n {
                        dst[pos + i] = src[i * s0];
                    }
                }
                pos += n;
            }
        } else {
            // The result is contiguous column-major, so column c starts at
            // c * strides[1] for c in 0..product(inner_shape), and the
            // axis-0 rows of every chunk interleave one column at a time.
            // Column c's base offset within each source chunk decomposes
            // across that chunk's own outer dims, which for a contiguous
            // chunk reduces to c * strides[1].
            let dst_stride = result.strides()[1];
            let n_cols: usize = self.inner_shape.iter().product();
            let dst = result.as_mut_slice();
            let mut row_offset = 0;
            for chunk in &self.batches {
                let chunk_obs = chunk.shape()[0];
                let s0 = chunk.strides()[0];
                let src = chunk.as_slice();
                for c in 0..n_cols {
                    let mut src_start = 0;
                    let mut rem = c;
                    for d in 1..chunk.ndim() {
                        src_start += (rem % chunk.shape()[d]) * chunk.strides()[d];
                        rem /= chunk.shape()[d];
                    }
                    let dst_start = c * dst_stride + row_offset;
                    if s0 == 1 {
                        dst[dst_start..dst_start + chunk_obs]
                            .copy_from_slice(&src[src_start..src_start + chunk_obs]);
                    } else {
                        for i in 0..chunk_obs {
                            dst[dst_start + i] = src[src_start + i * s0];
                        }
                    }
                }
                row_offset += chunk_obs;
            }
        }

        result.name = Some(self.name);
        result
    }
}

// ****************************************************************
// Trait implementations
// ****************************************************************

/// Logical equality. Two chunked arrays are equal when they share the same
/// rank, trailing shape, and observation count, and hold the same values in
/// logical order. Chunk boundaries and the name do not affect equality.
///
/// The comparison walks one logical column at a time, chaining each side's
/// batches independently, so arrays split into different chunks still line
/// up without materialising either side. Each batch reads at its own
/// strides, so any layout compares in place.
impl<T: Float> PartialEq for SuperNdArray<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.ndim != other.ndim
            || self.inner_shape != other.inner_shape
            || self.n_obs() != other.n_obs()
        {
            return false;
        }
        let n_cols: usize = self.inner_shape.iter().product();
        for c in 0..n_cols {
            let lhs = self.batches.iter().flat_map(|b| {
                let obs = b.shape()[0];
                let s0 = b.strides()[0];
                // Column c's base offset decomposes across the outer dims,
                // which for a contiguous batch reduces to c * strides[1].
                let mut off = 0;
                let mut rem = c;
                for d in 1..b.ndim() {
                    off += (rem % b.shape()[d]) * b.strides()[d];
                    rem /= b.shape()[d];
                }
                let buf = b.as_slice();
                (0..obs).map(move |i| buf[off + i * s0])
            });
            let rhs = other.batches.iter().flat_map(|b| {
                let obs = b.shape()[0];
                let s0 = b.strides()[0];
                let mut off = 0;
                let mut rem = c;
                for d in 1..b.ndim() {
                    off += (rem % b.shape()[d]) * b.strides()[d];
                    rem /= b.shape()[d];
                }
                let buf = b.as_slice();
                (0..obs).map(move |i| buf[off + i * s0])
            });
            if !lhs.eq(rhs) {
                return false;
            }
        }
        true
    }
}

impl<T: Float> Shape for SuperNdArray<T> {
    fn shape(&self) -> ShapeDim {
        let obs = self.n_obs();
        match self.ndim {
            0 | 1 => ShapeDim::Rank1(obs),
            2 => ShapeDim::Rank2 { rows: obs, cols: self.inner_shape[0] },
            _ => {
                let mut full = vec![obs];
                full.extend_from_slice(&self.inner_shape);
                ShapeDim::RankN(full)
            }
        }
    }
}

impl<T: Float> Concatenate for SuperNdArray<T> {
    fn concat(mut self, other: Self) -> Result<Self, MinarrowError> {
        if self.batches.is_empty() { return Ok(other); }
        if other.batches.is_empty() { return Ok(self); }
        if self.ndim != other.ndim {
            return Err(MinarrowError::IncompatibleTypeError {
                from: "SuperNdArray", to: "SuperNdArray",
                message: Some(format!("rank {} vs {}", self.ndim, other.ndim)),
            });
        }
        if self.inner_shape != other.inner_shape {
            return Err(MinarrowError::IncompatibleTypeError {
                from: "SuperNdArray", to: "SuperNdArray",
                message: Some(format!(
                    "inner shape {:?} vs {:?}", self.inner_shape, other.inner_shape
                )),
            });
        }
        self.batches.extend(other.batches);
        Ok(self)
    }
}

impl<T: Float> fmt::Debug for SuperNdArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "SuperNdArray '{}': {} batches, {}D, shape {:?}, {} elements",
            self.name, self.n_batches(), self.ndim, self.shape(), self.len()
        )
    }
}

// ****************************************************************
// Tests
// ****************************************************************

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Buffer;
    #[cfg(all(feature = "views", feature = "select"))]
    use crate::nd;

    // *** Row selection and apply *************************************

    #[cfg(all(feature = "views", feature = "select"))]
    #[test]
    fn axis_selection_windows_and_narrows() {
        // col0 = [1..5], col1 = [10..50] across two batches.
        let snd = SuperNdArray::from_batches(
            vec![
                NdArray::from_slice(&[1.0, 2.0, 10.0, 20.0], &[2, 2]),
                NdArray::from_slice(&[3.0, 4.0, 5.0, 30.0, 40.0, 50.0], &[3, 2]),
            ],
            "data",
        );
        // Axis-0 range crosses the batch boundary, trailing index collapses.
        let v = snd.s(nd![1..4, 1]);
        assert_eq!(v.ndim(), 1);
        assert_eq!(v.n_obs(), 3);
        assert_eq!(v.get(&[0]), 20.0);
        assert_eq!(v.get(&[2]), 40.0);
        // Axis-0 single index keeps the dimension as a one-observation window.
        let one = snd.s(nd![3, 0..2]);
        assert_eq!(one.n_obs(), 1);
        assert_eq!(one.get(&[0, 0]), 4.0);
        assert_eq!(one.get(&[0, 1]), 40.0);
    }

    #[cfg(all(feature = "views", feature = "select"))]
    #[test]
    fn row_selection_contiguous_crosses_batches() {
        let snd = SuperNdArray::from_batches(
            vec![
                NdArray::from_slice(&[1.0, 2.0, 10.0, 20.0], &[2, 2]),
                NdArray::from_slice(&[3.0, 4.0, 5.0, 30.0, 40.0, 50.0], &[3, 2]),
            ],
            "data",
        );
        let v = snd.r(1..4);
        assert_eq!(v.n_slices(), 2);
        assert_eq!(v.n_obs(), 3);
        assert_eq!(v.get(&[0, 0]), 2.0);
        assert_eq!(v.get(&[2, 1]), 40.0);
    }

    #[cfg(all(feature = "views", feature = "select"))]
    #[test]
    fn row_selection_gathers_across_batches() {
        let snd = SuperNdArray::from_batches(
            vec![
                NdArray::from_slice(&[1.0, 2.0], &[2]),
                NdArray::from_slice(&[3.0, 4.0], &[2]),
            ],
            "data",
        );
        let v = snd.r(&[3, 0]);
        assert_eq!(v.n_slices(), 1);
        let vals: Vec<f64> = (&v).into_iter().collect();
        assert_eq!(vals, vec![4.0, 1.0]);
    }

    #[test]
    fn apply_preserves_chunking() {
        let snd = SuperNdArray::from_batches(
            vec![
                NdArray::from_slice(&[1.0, 2.0], &[2]),
                NdArray::from_slice(&[3.0], &[1]),
            ],
            "stream",
        );
        let out = snd.apply(|x| x * 10.0);
        assert_eq!(out.n_batches(), 2);
        assert_eq!(out.name, "stream");
        let vals: Vec<f64> = (&out).into_iter().collect();
        assert_eq!(vals, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn apply_mut_in_place() {
        let mut snd = SuperNdArray::from_batches(
            vec![
                NdArray::from_slice(&[1.0, 2.0], &[2]),
                NdArray::from_slice(&[3.0], &[1]),
            ],
            "stream",
        );
        snd.apply_mut(|x| x + 1.0);
        assert_eq!(snd.n_batches(), 2);
        let vals: Vec<f64> = (&snd).into_iter().collect();
        assert_eq!(vals, vec![2.0, 3.0, 4.0]);
    }

    #[cfg(feature = "views")]
    #[test]
    fn apply_axis_zero_crosses_batches() {
        // col0 = [1, 2, 3, 4], col1 = [10, 20, 30, 40] across two batches.
        let snd = SuperNdArray::from_batches(
            vec![
                NdArray::from_slice(&[1.0, 2.0, 10.0, 20.0], &[2, 2]),
                NdArray::from_slice(&[3.0, 4.0, 30.0, 40.0], &[2, 2]),
            ],
            "data",
        );
        let sums = snd.apply_axis(0, |lane| (&lane).into_iter().sum());
        assert_eq!(sums.n_batches(), 1);
        assert_eq!(sums.shape(), vec![2]);
        assert_eq!(sums.get(&[0]), 10.0);
        assert_eq!(sums.get(&[1]), 100.0);
    }

    #[cfg(feature = "views")]
    #[test]
    fn apply_axis_inner_preserves_chunking() {
        let snd = SuperNdArray::from_batches(
            vec![
                NdArray::from_slice(&[1.0, 2.0, 10.0, 20.0], &[2, 2]),
                NdArray::from_slice(&[3.0, 30.0], &[1, 2]),
            ],
            "data",
        );
        // Sum each row lane (axis 1) - per-batch, boundaries kept.
        let row_sums = snd.apply_axis(1, |lane| (&lane).into_iter().sum());
        assert_eq!(row_sums.n_batches(), 2);
        assert_eq!(row_sums.shape(), vec![3]);
        assert_eq!(row_sums.get(&[0]), 11.0);
        assert_eq!(row_sums.get(&[1]), 22.0);
        assert_eq!(row_sums.get(&[2]), 33.0);
    }

    #[test]
    fn apply_batches_transforms_each_batch() {
        let snd = SuperNdArray::from_batches(
            vec![
                NdArray::from_slice(&[1.0, 2.0], &[2]),
                NdArray::from_slice(&[3.0], &[1]),
            ],
            "stream",
        );
        let out = snd
            .apply_batches(|b| Ok::<_, MinarrowError>(b.apply(|x| x - 1.0)))
            .unwrap();
        assert_eq!(out.n_batches(), 2);
        let vals: Vec<f64> = (&out).into_iter().collect();
        assert_eq!(vals, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn apply_batches_propagates_errors() {
        let snd = SuperNdArray::from_batches(
            vec![NdArray::from_slice(&[1.0], &[1])],
            "stream",
        );
        let result = snd.apply_batches(|_| {
            Err::<NdArray<f64>, MinarrowError>(MinarrowError::KernelError(Some(
                "boom".to_string(),
            )))
        });
        assert!(result.is_err());
    }

    #[test]
    fn apply_batches_mut_in_place() {
        let mut snd = SuperNdArray::from_batches(
            vec![
                NdArray::from_slice(&[1.0, 2.0], &[2]),
                NdArray::from_slice(&[3.0], &[1]),
            ],
            "stream",
        );
        snd.apply_batches_mut(|b| {
            b.apply_mut(|x| x * 3.0);
            Ok::<_, MinarrowError>(())
        })
        .unwrap();
        let vals: Vec<f64> = (&snd).into_iter().collect();
        assert_eq!(vals, vec![3.0, 6.0, 9.0]);
    }

    #[test]
    fn empty() {
        let snd = SuperNdArray::<f64>::new("empty");
        assert!(snd.is_empty());
        assert_eq!(snd.n_batches(), 0);
    }

    #[test]
    fn from_batches_1d() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = NdArray::from_slice(&[4.0, 5.0], &[2]);
        let snd = SuperNdArray::from_batches(vec![a, b], "test");
        assert_eq!(snd.n_batches(), 2);
        assert_eq!(snd.len(), 5);
        assert_eq!(snd.n_obs(), 5);
        assert_eq!(snd.ndim(), 1);
    }

    #[test]
    fn from_batches_2d() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = NdArray::from_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[3, 2]);
        let snd = SuperNdArray::from_batches(vec![a, b], "data");
        assert_eq!(snd.n_batches(), 2);
        assert_eq!(snd.len(), 10); // 2*2 + 3*2 = 10 total elements
        assert_eq!(snd.n_obs(), 5); // 2 + 3 = 5 leading-axis observations
        assert_eq!(snd.inner_shape(), &[2]);
    }

    #[test]
    fn push_and_validate() {
        let mut snd = SuperNdArray::new("stream");
        snd.push(NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
        snd.push(NdArray::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]));
        assert_eq!(snd.n_batches(), 2);
        assert_eq!(snd.len(), 8); // 2*2 + 2*2 = 8 total elements
        assert_eq!(snd.n_obs(), 4);
    }

    #[test]
    #[should_panic(expected = "rank")]
    fn push_rank_mismatch() {
        let mut snd = SuperNdArray::new("bad");
        snd.push(NdArray::from_slice(&[1.0, 2.0], &[2]));
        snd.push(NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
    }

    #[test]
    #[should_panic(expected = "inner shape")]
    fn push_trailing_shape_mismatch() {
        let mut snd = SuperNdArray::new("bad");
        snd.push(NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
        snd.push(NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    }

    // *** Global element access ***************************************

    #[test]
    fn global_get_1d() {
        let snd = SuperNdArray::from_batches(vec![
            NdArray::from_slice(&[10.0, 20.0, 30.0], &[3]),
            NdArray::from_slice(&[40.0, 50.0], &[2]),
        ], "test");
        assert_eq!(snd.get(&[0]), 10.0);
        assert_eq!(snd.get(&[2]), 30.0);
        assert_eq!(snd.get(&[3]), 40.0);
        assert_eq!(snd.get(&[4]), 50.0);
    }

    #[test]
    fn global_get_2d() {
        // chunk0: 2x2, col0=[1,2], col1=[3,4]
        // chunk1: 3x2, col0=[5,6,7], col1=[8,9,10]
        let snd = SuperNdArray::from_batches(vec![
            NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
            NdArray::from_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[3, 2]),
        ], "data");
        // Global row 0, col 0 -> chunk0 row 0 col 0
        assert_eq!(snd.get(&[0, 0]), 1.0);
        // Global row 1, col 1 -> chunk0 row 1 col 1
        assert_eq!(snd.get(&[1, 1]), 4.0);
        // Global row 2, col 0 -> chunk1 row 0 col 0
        assert_eq!(snd.get(&[2, 0]), 5.0);
        // Global row 4, col 1 -> chunk1 row 2 col 1
        assert_eq!(snd.get(&[4, 1]), 10.0);
    }

    #[test]
    fn global_set_2d() {
        let mut snd = SuperNdArray::from_batches(vec![
            NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
            NdArray::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]),
        ], "mut");
        snd.set(&[3, 1], 99.0);
        assert_eq!(snd.get(&[3, 1]), 99.0);
    }

    // *** Iteration ***************************************************

    #[test]
    fn iter_1d_across_batches() {
        let snd = SuperNdArray::from_batches(vec![
            NdArray::from_slice(&[1.0, 2.0, 3.0], &[3]),
            NdArray::from_slice(&[4.0, 5.0], &[2]),
        ], "test");
        let vals: Vec<f64> = (&snd).into_iter().collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn iter_2d_across_batches() {
        let snd = SuperNdArray::from_batches(vec![
            NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
            NdArray::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]),
        ], "data");
        let vals: Vec<f64> = (&snd).into_iter().collect();
        // chunk0 col-major: [1,2,3,4], chunk1 col-major: [5,6,7,8]
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn iter_exact_size() {
        let snd = SuperNdArray::from_batches(vec![
            NdArray::from_slice(&[1.0, 2.0, 3.0], &[3]),
            NdArray::from_slice(&[4.0, 5.0], &[2]),
        ], "test");
        let iter = (&snd).into_iter();
        assert_eq!(iter.len(), 5);
    }

    // *** Consolidate *************************************************

    #[test]
    fn consolidate_1d() {
        let snd = SuperNdArray::from_batches(vec![
            NdArray::from_slice(&[1.0, 2.0, 3.0], &[3]),
            NdArray::from_slice(&[4.0, 5.0], &[2]),
        ], "test");
        let result = snd.consolidate();
        assert_eq!(result.shape(), &[5]);
        let vals: Vec<f64> = (&result).into_iter().collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn consolidate_2d() {
        let snd = SuperNdArray::from_batches(vec![
            NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
            NdArray::from_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[3, 2]),
        ], "data");
        let result = snd.consolidate();
        assert_eq!(result.shape(), &[5, 2]);
        assert_eq!(result.col(0), &[1.0, 2.0, 5.0, 6.0, 7.0]);
        assert_eq!(result.col(1), &[3.0, 4.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    fn consolidate_single_chunk() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let snd = SuperNdArray::from_batches(vec![a.clone()], "one");
        let result = snd.consolidate();
        // The single-batch shortcut carries the chunked array's name.
        assert_eq!(result.name.as_deref(), Some("one"));
        let mut expected = a;
        expected.name = Some("one".to_string());
        assert_eq!(result, expected);
    }

    #[test]
    fn batch_accessors() {
        let snd = SuperNdArray::from_batches(
            vec![
                NdArray::from_slice(&[1.0, 2.0], &[2]),
                NdArray::from_slice(&[3.0], &[1]),
            ],
            "ticks",
        );
        assert_eq!(snd.batches().len(), 2);
        assert_eq!(snd.batch(1).unwrap().get(&[0]), 3.0);
        assert!(snd.batch(2).is_none());
        let batches = snd.into_batches();
        assert_eq!(batches.len(), 2);
    }

    #[cfg(feature = "views")]
    #[test]
    fn view_carries_name() {
        let snd = SuperNdArray::from_batches(
            vec![
                NdArray::from_slice(&[1.0, 2.0], &[2]),
                NdArray::from_slice(&[3.0, 4.0], &[2]),
            ],
            "ticks",
        );
        let window = snd.slice(1, 2);
        assert_eq!(window.name(), "ticks");
        assert_eq!(window.consolidate().name.as_deref(), Some("ticks"));
    }

    #[test]
    fn consolidate_3d() {
        // chunk A [2,2,2] with logical column-major values 1..=8,
        // chunk B [1,2,2] with logical column-major values 9..=12.
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
        let b = NdArray::from_slice(&[9.0, 10.0, 11.0, 12.0], &[1, 2, 2]);
        let snd = SuperNdArray::from_batches(vec![a, b], "cube");
        let c = snd.consolidate();
        assert_eq!(c.shape(), &[3, 2, 2]);
        // Axis-0 rows interleave one higher-dimensional slice at a time.
        assert_eq!(c.get(&[0, 0, 0]), 1.0);
        assert_eq!(c.get(&[1, 0, 0]), 2.0);
        assert_eq!(c.get(&[2, 0, 0]), 9.0);
        assert_eq!(c.get(&[0, 1, 0]), 3.0);
        assert_eq!(c.get(&[2, 1, 0]), 10.0);
        assert_eq!(c.get(&[1, 1, 1]), 8.0);
        assert_eq!(c.get(&[2, 0, 1]), 11.0);
        assert_eq!(c.get(&[2, 1, 1]), 12.0);
    }

    #[test]
    fn eq_ignores_chunking_and_name() {
        // Same logical [3,2] values, chunked and named differently.
        let single = SuperNdArray::from_batches(
            vec![NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])],
            "single",
        );
        let split = SuperNdArray::from_batches(
            vec![
                NdArray::from_slice(&[1.0, 4.0], &[1, 2]),
                NdArray::from_slice(&[2.0, 3.0, 5.0, 6.0], &[2, 2]),
            ],
            "split",
        );
        assert_eq!(single, split);

        // A changed value breaks equality.
        let different = SuperNdArray::from_batches(
            vec![NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 99.0], &[3, 2])],
            "single",
        );
        assert_ne!(single, different);
    }

    #[test]
    fn eq_chunk_invariant_3d() {
        // Same logical [3,2,2] values, one whole chunk versus a [1,2,2] and
        // a [2,2,2] split, exercising the rank-3 column walk in equality.
        let whole = SuperNdArray::from_batches(
            vec![NdArray::from_slice(
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                &[3, 2, 2],
            )],
            "whole",
        );
        let split = SuperNdArray::from_batches(
            vec![
                NdArray::from_slice(&[1.0, 4.0, 7.0, 10.0], &[1, 2, 2]),
                NdArray::from_slice(&[2.0, 3.0, 5.0, 6.0, 8.0, 9.0, 11.0, 12.0], &[2, 2, 2]),
            ],
            "split",
        );
        assert_eq!(whole, split);
    }

    // *** Rechunk *****************************************************

    #[test]
    fn rechunk_count() {
        let mut snd = SuperNdArray::from_batches(vec![
            NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]),
        ], "test");
        snd.rechunk(RechunkStrategy::Count(2)).unwrap();
        assert_eq!(snd.n_batches(), 3); // 2 + 2 + 1
        assert_eq!(snd.len(), 5);
        // Verify data integrity
        let vals: Vec<f64> = (&snd).into_iter().collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn rechunk_2d() {
        let mut snd = SuperNdArray::from_batches(vec![
            NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]),
        ], "test");
        snd.rechunk(RechunkStrategy::Count(2)).unwrap();
        assert_eq!(snd.n_batches(), 2); // 2 + 1
        assert_eq!(snd.len(), 6); // 3*2 = 6 total elements
        assert_eq!(snd.n_obs(), 3);
        // Verify data
        assert_eq!(snd.get(&[0, 0]), 1.0);
        assert_eq!(snd.get(&[2, 1]), 6.0);
    }

    #[test]
    fn rechunk_auto() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let mut snd = SuperNdArray::from_batches(vec![
            NdArray::from_slice(&data, &[100]),
        ], "big");
        snd.rechunk(RechunkStrategy::Auto).unwrap();
        // Auto = 8192 rows, 100 < 8192, so still 1 chunk
        assert_eq!(snd.n_batches(), 1);
    }

    // *** Concat ******************************************************

    #[test]
    fn concat_super_ndarrays() {
        let a = SuperNdArray::from_batches(
            vec![NdArray::from_slice(&[1.0, 2.0], &[2])], "a"
        );
        let b = SuperNdArray::from_batches(
            vec![NdArray::from_slice(&[3.0, 4.0], &[2])], "b"
        );
        let c = a.concat(b).unwrap();
        assert_eq!(c.n_batches(), 2);
        assert_eq!(c.len(), 4);
    }

    // *** Shape *******************************************************

    #[test]
    fn shape_trait() {
        let snd = SuperNdArray::from_batches(vec![
            NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
            NdArray::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]),
        ], "test");
        assert_eq!(Shape::shape(&snd), ShapeDim::Rank2 { rows: 4, cols: 2 });
    }

    #[test]
    fn shape_method() {
        let snd = SuperNdArray::from_batches(vec![
            NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
            NdArray::from_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[3, 2]),
        ], "test");
        assert_eq!(snd.shape(), vec![5, 2]);
    }

    // *** Strided batches *********************************************

    #[test]
    fn consolidate_non_contiguous_single_batch() {
        // Row-major strides [2, 1] on shape [2, 2] give logical
        // col0 = [1, 3] and col1 = [2, 4].
        let nd = NdArray::from_buffer(
            Buffer::from_slice(&[1.0, 2.0, 3.0, 4.0]),
            &[2, 2],
            &[2, 1],
        );
        assert!(!nd.is_contiguous());
        let snd = SuperNdArray::from_batches(vec![nd], "strided");
        let result = snd.consolidate();
        assert!(result.is_contiguous());
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.col(0), &[1.0, 3.0]);
        assert_eq!(result.col(1), &[2.0, 4.0]);
    }

    #[test]
    fn consolidate_mixed_layout_batches() {
        let compact = NdArray::from_slice(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
        // Row-major batch holding logical col0 = [1, 3], col1 = [2, 4].
        let strided = NdArray::from_buffer(
            Buffer::from_slice(&[1.0, 2.0, 3.0, 4.0]),
            &[2, 2],
            &[2, 1],
        );
        let snd = SuperNdArray::from_batches(vec![compact, strided], "mixed");
        let result = snd.consolidate();
        assert_eq!(result.shape(), &[4, 2]);
        assert_eq!(result.col(0), &[10.0, 20.0, 1.0, 3.0]);
        assert_eq!(result.col(1), &[30.0, 40.0, 2.0, 4.0]);
    }

    #[test]
    fn eq_strided_batch_matches_contiguous() {
        // Row-major batch holding logical col0 = [1, 3], col1 = [2, 4],
        // compared against the same values chunked row by row.
        let strided = SuperNdArray::from_batches(
            vec![NdArray::from_buffer(
                Buffer::from_slice(&[1.0, 2.0, 3.0, 4.0]),
                &[2, 2],
                &[2, 1],
            )],
            "strided",
        );
        let compact = SuperNdArray::from_batches(
            vec![
                NdArray::from_slice(&[1.0, 2.0], &[1, 2]),
                NdArray::from_slice(&[3.0, 4.0], &[1, 2]),
            ],
            "compact",
        );
        assert_eq!(strided, compact);

        let different = SuperNdArray::from_batches(
            vec![NdArray::from_slice(&[1.0, 3.0, 2.0, 99.0], &[2, 2])],
            "different",
        );
        assert_ne!(strided, different);
    }

    // *** Bounds and observation access *******************************

    #[cfg(feature = "views")]
    #[test]
    fn obs_resolves_across_chunk_boundary() {
        let snd = SuperNdArray::from_batches(vec![
            NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
            NdArray::from_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[3, 2]),
        ], "data");
        // Global observation 2 is the second batch's first row.
        let o = snd.obs(2);
        assert_eq!(o.shape(), &[2]);
        assert_eq!(o.get(&[0]), 5.0);
        assert_eq!(o.get(&[1]), 8.0);
    }

    #[cfg(feature = "views")]
    #[test]
    #[should_panic(expected = "global index 5 out of bounds (n_obs 5)")]
    fn obs_out_of_bounds() {
        let snd = SuperNdArray::from_batches(vec![
            NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
            NdArray::from_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[3, 2]),
        ], "data");
        snd.obs(5);
    }

    #[test]
    #[should_panic(expected = "global index 9 out of bounds (n_obs 5)")]
    fn get_out_of_bounds() {
        let snd = SuperNdArray::from_batches(vec![
            NdArray::from_slice(&[1.0, 2.0, 3.0], &[3]),
            NdArray::from_slice(&[4.0, 5.0], &[2]),
        ], "test");
        snd.get(&[9]);
    }

    #[test]
    #[should_panic(expected = "global index 4 out of bounds (n_obs 4)")]
    fn set_out_of_bounds() {
        let mut snd = SuperNdArray::from_batches(vec![
            NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
            NdArray::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]),
        ], "mut");
        snd.set(&[4, 0], 1.0);
    }

    #[cfg(feature = "views")]
    #[test]
    #[should_panic(expected = "window [3, 7) out of bounds (n_obs 5)")]
    fn slice_beyond_n_obs() {
        let snd = SuperNdArray::from_batches(vec![
            NdArray::from_slice(&[1.0, 2.0, 3.0], &[3]),
            NdArray::from_slice(&[4.0, 5.0], &[2]),
        ], "test");
        snd.slice(3, 4);
    }

    #[test]
    fn col_concatenates_across_batches() {
        let snd = SuperNdArray::from_batches(vec![
            NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
            NdArray::from_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[3, 2]),
        ], "data");
        assert_eq!(snd.col(0), vec![1.0, 2.0, 5.0, 6.0, 7.0]);
        assert_eq!(snd.col(1), vec![3.0, 4.0, 8.0, 9.0, 10.0]);
    }

    // *** Construction edge cases *************************************

    #[test]
    fn from_batches_empty_then_push_adopts_shape() {
        let mut snd = SuperNdArray::<f64>::from_batches(vec![], "fresh");
        assert_eq!(snd.n_batches(), 0);
        assert_eq!(snd.ndim(), 0);
        assert!(snd.is_empty());
        snd.push(NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
        assert_eq!(snd.ndim(), 2);
        assert_eq!(snd.inner_shape(), &[2]);
        assert_eq!(snd.n_obs(), 2);
    }

    #[test]
    #[should_panic(expected = "has rank 2 but expected 1")]
    fn from_batches_rank_mismatch() {
        SuperNdArray::from_batches(vec![
            NdArray::from_slice(&[1.0, 2.0], &[2]),
            NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
        ], "bad");
    }

    #[test]
    #[should_panic(expected = "chunk 1 inner shape mismatch")]
    fn from_batches_inner_shape_mismatch() {
        SuperNdArray::from_batches(vec![
            NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
            NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
        ], "bad");
    }

    #[test]
    fn concat_rank_mismatch_errors() {
        let a = SuperNdArray::from_batches(
            vec![NdArray::from_slice(&[1.0, 2.0], &[2])], "a"
        );
        let b = SuperNdArray::from_batches(
            vec![NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2])], "b"
        );
        let err = a.concat(b).unwrap_err();
        assert!(format!("{}", err).contains("rank 1 vs 2"));
    }

    #[test]
    fn concat_inner_shape_mismatch_errors() {
        let a = SuperNdArray::from_batches(
            vec![NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2])], "a"
        );
        let b = SuperNdArray::from_batches(
            vec![NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])], "b"
        );
        let err = a.concat(b).unwrap_err();
        assert!(format!("{}", err).contains("inner shape [2] vs [3]"));
    }

    // *** Rechunk and empty batches ***********************************

    #[test]
    fn rechunk_3d_count() {
        let data: Vec<f64> = (1..=16).map(|x| x as f64).collect();
        let mut snd = SuperNdArray::from_batches(
            vec![NdArray::from_slice(&data, &[4, 2, 2])], "cube"
        );
        snd.rechunk(RechunkStrategy::Count(3)).unwrap();
        assert_eq!(snd.n_batches(), 2); // 3 + 1
        assert_eq!(snd.batch(0).unwrap().shape(), &[3, 2, 2]);
        assert_eq!(snd.batch(1).unwrap().shape(), &[1, 2, 2]);
        assert_eq!(snd.get(&[0, 0, 0]), 1.0);
        assert_eq!(snd.get(&[2, 1, 0]), 7.0);
        assert_eq!(snd.get(&[3, 0, 1]), 12.0);
        assert_eq!(snd.get(&[3, 1, 1]), 16.0);
    }

    #[test]
    fn zero_observation_batch() {
        let snd = SuperNdArray::from_batches(vec![
            NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
            NdArray::from_slice(&[], &[0, 2]),
            NdArray::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]),
        ], "gappy");
        assert_eq!(snd.n_obs(), 4);
        let vals: Vec<f64> = (&snd).into_iter().collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let result = snd.consolidate();
        assert_eq!(result.shape(), &[4, 2]);
        assert_eq!(result.col(0), &[1.0, 2.0, 5.0, 6.0]);
        assert_eq!(result.col(1), &[3.0, 4.0, 7.0, 8.0]);
    }
}
