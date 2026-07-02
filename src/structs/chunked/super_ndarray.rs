//! # **SuperNdArray** - *Chunked N-dimensional array*
//!
//! The N-dimensional equivalent of [`SuperTable`]: groups multiple `NdArray`
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

use std::sync::Arc;

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
    pub batches: Vec<Arc<NdArray<T>>>,
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
        let batches = batches.into_iter().map(Arc::new).collect();
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
        self.batches.push(Arc::new(chunk));
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

    /// Get chunk by index.
    #[inline]
    pub fn chunk(&self, idx: usize) -> Option<&Arc<NdArray<T>>> { self.batches.get(idx) }

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

    /// Zero-copy view of a single observation (axis-0 element)
    /// across batch boundaries.
    ///
    /// Returns an (N-1)-dimensional `NdArrayV` view. For a 2D chunked
    /// array with shape [n, m], returns a 1D view of shape [m]. For 3D
    /// [n, m, k], returns 2D [m, k]. Requires rank 2 or higher - for
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
        Arc::make_mut(&mut self.batches[chunk_idx]).set(&local, value);
    }

    /// Resolve a global axis-0 index to (batch_index, local_index).
    /// Returns an error if the index is out of bounds.
    pub fn try_resolve(&self, global: usize) -> Result<(usize, usize), MinarrowError> {
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
        self.try_resolve(global).unwrap()
    }

    // ****************************************************************
    // Iteration
    // ****************************************************************

    /// Iterate over batches.
    #[inline]
    pub fn iter_batches(&self) -> std::slice::Iter<'_, Arc<NdArray<T>>> {
        self.batches.iter()
    }

    /// Mutable iteration over batches.
    #[inline]
    pub fn iter_batches_mut(&mut self) -> std::slice::IterMut<'_, Arc<NdArray<T>>> {
        self.batches.iter_mut()
    }

    /// Parallel iteration over batches.
    #[cfg(feature = "parallel_proc")]
    #[inline]
    pub fn par_iter_batches(&self) -> rayon::slice::Iter<'_, Arc<NdArray<T>>> {
        self.batches.par_iter()
    }

    /// Parallel iterator over axis-0 observations across all batches.
    /// Each item is the global observation index and a zero-copy
    /// `NdArrayV` view. Batch boundaries are resolved transparently.
    #[cfg(all(feature = "parallel_proc", feature = "views"))]
    pub fn par_iter_obs(&self) -> impl rayon::iter::ParallelIterator<Item = (usize, NdArrayV<T>)> + '_ {
        use rayon::prelude::*;
        let n_obs = self.n_obs();
        (0..n_obs).into_par_iter().map(move |i| (i, self.obs(i)))
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

        let mut new_batches = Vec::with_capacity((total_obs + chunk_size - 1) / chunk_size);
        let mut row_offset = 0;

        while row_offset < total_obs {
            let n = (total_obs - row_offset).min(chunk_size);
            let mut chunk_shape = vec![n];
            chunk_shape.extend_from_slice(&inner_shape);

            if ndim <= 1 {
                let chunk = NdArray::from_slice(&buf[row_offset..row_offset + n], &chunk_shape);
                new_batches.push(Arc::new(chunk));
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
                new_batches.push(Arc::new(chunk));
            }
            row_offset += n;
        }

        *self = SuperNdArray { ndim, inner_shape, batches: new_batches, name };
        Ok(())
    }
}

// ****************************************************************
// IntoIterator - element-wise iteration across batches
// ****************************************************************

/// Iterating a SuperNdArray yields f64 values in column-major order,
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
    batches: &'a [Arc<NdArray<T>>],
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
            self.inner = Some(self.batches[self.chunk_idx].as_ref().into_iter());
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

    fn consolidate(self) -> NdArray<T> {
        if self.batches.is_empty() {
            return NdArray::new(&[0]);
        }
        if self.batches.len() == 1 {
            let arc = self.batches.into_iter().next().unwrap();
            return Arc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone());
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
                dst[pos..pos + n].copy_from_slice(&chunk.as_slice()[..n]);
                pos += n;
            }
        } else {
            // Column-major layout places each higher-dimensional slice at
            // c * strides[1] for c in 0..product(inner_shape), so the axis-0
            // rows of every chunk interleave one column at a time. This holds
            // for any rank of two or more, since the strides above the leading
            // axis are exact multiples of strides[1].
            let dst_stride = result.strides()[1];
            let n_cols: usize = self.inner_shape.iter().product();
            let dst = result.as_mut_slice();
            let mut row_offset = 0;
            for chunk in &self.batches {
                let chunk_obs = chunk.shape()[0];
                let src_stride = chunk.strides()[1];
                let src = chunk.as_slice();
                for c in 0..n_cols {
                    let src_start = c * src_stride;
                    let dst_start = c * dst_stride + row_offset;
                    dst[dst_start..dst_start + chunk_obs]
                        .copy_from_slice(&src[src_start..src_start + chunk_obs]);
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
/// up without materialising either side.
impl<T: Float> PartialEq for SuperNdArray<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.ndim != other.ndim
            || self.inner_shape != other.inner_shape
            || self.n_obs() != other.n_obs()
        {
            return false;
        }
        // Column-major layout places a chunk's column c in a contiguous
        // axis-0 run at c * strides[1], or the whole leading run in 1D.
        let n_cols: usize = self.inner_shape.iter().product();
        for c in 0..n_cols {
            let lhs = self.batches.iter().flat_map(|b| {
                let obs = b.shape()[0];
                let start = if b.ndim() <= 1 { 0 } else { c * b.strides()[1] };
                b.as_slice()[start..start + obs].iter()
            });
            let rhs = other.batches.iter().flat_map(|b| {
                let obs = b.shape()[0];
                let start = if b.ndim() <= 1 { 0 } else { c * b.strides()[1] };
                b.as_slice()[start..start + obs].iter()
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
        assert_eq!(result, a);
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
}
