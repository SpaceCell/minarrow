//! # **NdArrayV** - *Zero-copy view into an NdArray*
//!
//! Holds a clone of the parent `NdArray`, kept alive through the array's
//! shared internal buffer, plus its own offset and dimension metadata.
//! Views can have different shapes and strides from the parent, enabling
//! slicing, axis selection, transposition, and axis permutation without
//! copying data.

use std::fmt;
use std::ops::Index;

#[cfg(feature = "matrix")]
use crate::enums::error::MinarrowError;
use crate::enums::shape_dim::ShapeDim;
use crate::structs::ndarray::{AxisSel, IntoSlice, NdArray, NdArrayIter, NdDims, offset_of_impl};
#[cfg(feature = "select")]
use crate::structs::ndarray::gather_obs_impl;
#[cfg(feature = "select")]
use crate::traits::selection::{DataSelector, RowSelection};
use crate::traits::shape::Shape;
use crate::traits::type_unions::Float;
use crate::Vec64;

#[cfg(feature = "matrix")]
use crate::structs::matrix::Matrix;

/// Zero-copy view into an [`NdArray`].
///
/// Holds a clone of the parent `NdArray`, kept alive through the array's
/// shared internal buffer, with its own offset and dimension metadata.
/// This enables slicing, axis selection, transposition, and axis
/// permutation without copying the underlying data.
#[derive(Clone)]
pub struct NdArrayV<T> {
    source: NdArray<T>,
    offset: usize,
    dims: NdDims,
}

impl<T: Float> NdArrayV<T> {
    /// Create a view over an NdArray with the given offset and dimensions.
    pub fn new(source: NdArray<T>, offset: usize, shape: &[usize], strides: &[usize]) -> Self {
        NdArrayV {
            source,
            offset,
            dims: NdDims::from_shape_and_strides(shape, strides),
        }
    }

    /// Create a full view over an NdArray with the same shape and strides.
    pub fn from_ndarray(source: NdArray<T>) -> Self {
        let dims = source.dims.clone();
        NdArrayV { source, offset: 0, dims }
    }

    /// Number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize { self.dims.ndim() }

    /// Shape as a slice.
    #[inline]
    pub fn shape(&self) -> &[usize] { self.dims.shape() }

    /// Strides as a slice.
    #[inline]
    pub fn strides(&self) -> &[usize] { self.dims.strides() }

    /// Total logical element count.
    #[inline]
    pub fn len(&self) -> usize { self.dims.len() }

    /// True if empty.
    #[inline]
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Get element by N-dimensional index.
    #[inline]
    pub fn get(&self, indices: &[usize]) -> T {
        let off = self.offset_of(indices);
        self.source.data.as_slice()[off]
    }

    /// Compute flat offset.
    #[inline]
    fn offset_of(&self, indices: &[usize]) -> usize {
        self.offset + offset_of_impl(indices, self.dims.shape(), self.dims.strides())
    }

    /// Immutable column slice for 2D views. Panics if ndim != 2.
    #[inline]
    pub fn col(&self, col: usize) -> &[T] {
        let shape = self.dims.shape();
        assert_eq!(shape.len(), 2, "col() requires a 2D view");
        debug_assert!(col < shape[1]);
        let stride = self.dims.strides()[1];
        let start = self.offset + col * stride;
        &self.source.data.as_slice()[start..start + shape[0]]
    }

    /// All columns as slices. 2D only.
    pub fn columns(&self) -> Vec<&[T]> {
        let shape = self.dims.shape();
        assert_eq!(shape.len(), 2, "columns() requires a 2D view");
        let stride = self.dims.strides()[1];
        let n_rows = shape[0];
        let buf = self.source.data.as_slice();
        (0..shape[1])
            .map(|c| &buf[self.offset + c * stride..self.offset + c * stride + n_rows])
            .collect()
    }

    // *** BLAS/LAPACK compatibility (2D) **************************

    /// BLAS row count. 2D only.
    #[inline]
    pub fn m(&self) -> i32 {
        assert_eq!(self.ndim(), 2, "m() requires a 2D view");
        self.dims.shape()[0] as i32
    }

    /// BLAS column count. 2D only.
    #[inline]
    pub fn n(&self) -> i32 {
        assert_eq!(self.ndim(), 2, "n() requires a 2D view");
        self.dims.shape()[1] as i32
    }

    /// BLAS leading dimension. 2D only.
    #[inline]
    pub fn lda(&self) -> i32 {
        assert_eq!(self.ndim(), 2, "lda() requires a 2D view");
        self.dims.strides()[1] as i32
    }

    // *** Slicing *************************************************

    /// Zero-copy view of a single observation (axis-0 element).
    ///
    /// Returns an (N-1)-dimensional view. For a 2D view with shape
    /// [n, m], returns a 1D view of shape [m]. For 3D [n, m, k],
    /// returns 2D [m, k]. Requires rank 2 or higher - a 1D view has no
    /// sub-array observations, so scalar access goes through `get(&[i])`.
    pub fn obs(&self, idx: usize) -> NdArrayV<T> {
        let shape = self.dims.shape();
        let strides = self.dims.strides();
        assert!(
            shape.len() >= 2,
            "obs() requires a 2D or higher view, use get(&[i]) for scalar access on 1D"
        );
        debug_assert!(idx < shape[0], "obs: index {} out of bounds for axis 0 (size {})", idx, shape[0]);

        let new_offset = self.offset + idx * strides[0];
        NdArrayV::new(self.source.clone(), new_offset, &shape[1..], &strides[1..])
    }

    /// Slice this view, producing a sub-view. Zero-copy - shares the
    /// same backing buffer, just adjusts offset and dims.
    pub fn slice<S: IntoSlice>(&self, sel: S) -> NdArrayV<T> {
        let axes = sel.into_slice();
        let shape = self.dims.shape();
        let strides = self.dims.strides();
        assert_eq!(
            axes.len(), shape.len(),
            "slice(): expected {} axes, got {}", shape.len(), axes.len()
        );

        let mut new_offset = self.offset;
        let mut new_shape = Vec::with_capacity(shape.len());
        let mut new_strides = Vec::with_capacity(shape.len());

        for (d, ax) in axes.iter().enumerate() {
            match ax {
                AxisSel::Index(i) => {
                    debug_assert!(*i < shape[d]);
                    new_offset += i * strides[d];
                }
                AxisSel::Range(start, end) => {
                    debug_assert!(*end <= shape[d]);
                    new_offset += start * strides[d];
                    new_shape.push(end - start);
                    new_strides.push(strides[d]);
                }
            }
        }

        if new_shape.is_empty() {
            new_shape.push(1);
            new_strides.push(1);
        }

        NdArrayV::new(self.source.clone(), new_offset, &new_shape, &new_strides)
    }

    // *** Axis manipulation ***************************************

    /// Transposed view with the axis order reversed. Zero-copy - only
    /// the shape and stride metadata reorder. A 1D view returns itself
    /// unchanged.
    pub fn transpose(&self) -> NdArrayV<T> {
        let shape: Vec<usize> = self.dims.shape().iter().rev().copied().collect();
        let strides: Vec<usize> = self.dims.strides().iter().rev().copied().collect();
        NdArrayV::new(self.source.clone(), self.offset, &shape, &strides)
    }

    /// View with axes reordered by the given permutation. Zero-copy.
    ///
    /// `perm[d]` names the source axis that becomes axis `d` of the
    /// result. Panics unless `perm` is a permutation of `0..ndim`.
    pub fn permute_axes(&self, perm: &[usize]) -> NdArrayV<T> {
        let shape = self.dims.shape();
        let strides = self.dims.strides();
        let ndim = shape.len();
        assert_eq!(
            perm.len(), ndim,
            "permute_axes: expected {} axes, got {}", ndim, perm.len()
        );
        let mut seen = vec![false; ndim];
        for &ax in perm {
            assert!(ax < ndim, "permute_axes: axis {} out of bounds for {}D view", ax, ndim);
            assert!(!seen[ax], "permute_axes: axis {} repeated", ax);
            seen[ax] = true;
        }
        let new_shape: Vec<usize> = perm.iter().map(|&ax| shape[ax]).collect();
        let new_strides: Vec<usize> = perm.iter().map(|&ax| strides[ax]).collect();
        NdArrayV::new(self.source.clone(), self.offset, &new_shape, &new_strides)
    }

    /// View with two axes swapped. Zero-copy.
    pub fn swap_axes(&self, a: usize, b: usize) -> NdArrayV<T> {
        let ndim = self.dims.ndim();
        assert!(
            a < ndim && b < ndim,
            "swap_axes: axes ({}, {}) out of bounds for {}D view", a, b, ndim
        );
        let mut shape = self.dims.shape().to_vec();
        let mut strides = self.dims.strides().to_vec();
        shape.swap(a, b);
        strides.swap(a, b);
        NdArrayV::new(self.source.clone(), self.offset, &shape, &strides)
    }

    // *** Materialisation *****************************************

    /// Materialise this view as an owned NdArray.
    pub fn to_ndarray(&self) -> NdArray<T> {
        let flat: Vec64<T> = self.into_iter().collect();
        let mut arr = NdArray::from_slice(&flat, self.dims.shape());
        arr.name = self.source.name.clone();
        arr
    }

    // *** Parallel iteration (rayon) ******************************

    /// Parallel iterator over axis-0 observations. Each item is the
    /// observation index and a zero-copy `NdArrayV` view.
    #[cfg(feature = "parallel_proc")]
    pub fn par_iter_obs(&self) -> impl rayon::iter::ParallelIterator<Item = (usize, NdArrayV<T>)> + '_ {
        use rayon::prelude::*;
        let n_obs = self.dims.shape()[0];
        (0..n_obs).into_par_iter().map(move |i| (i, self.obs(i)))
    }
}

/// Materialise a 2D view as a Matrix.
#[cfg(feature = "matrix")]
impl NdArrayV<f64> {
    /// Materialise a 2D view as a Matrix.
    pub fn to_matrix(&self) -> Result<Matrix, MinarrowError> {
        self.to_ndarray().to_matrix()
    }
}

impl<T: Float> NdArrayV<T> {
    /// Apply a function to every logical element, materialising a new
    /// compact [`NdArray`] with this view's shape and the parent's name.
    pub fn apply(&self, f: impl Fn(T) -> T) -> NdArray<T> {
        let flat: Vec64<T> = self.into_iter().map(f).collect();
        let mut result = NdArray::from_slice(&flat, self.dims.shape());
        result.name = self.source.name.clone();
        result
    }
}

// *** Row selection: view.r(0..10) ********************************

/// Axis-0 observation selection over a view. Contiguous ranges narrow
/// the window zero-copy. Index arrays gather the selected observations
/// into an owned array wrapped in a full view.
#[cfg(feature = "select")]
impl<T: Float> RowSelection for NdArrayV<T> {
    type View = NdArrayV<T>;

    fn r<S: DataSelector>(&self, selection: S) -> NdArrayV<T> {
        let n_obs = self.dims.shape()[0];
        let indices = selection.resolve_indices(n_obs);
        if selection.is_contiguous() {
            let start = indices.first().copied().unwrap_or(0);
            let mut sel = vec![AxisSel::Range(start, start + indices.len())];
            for d in 1..self.ndim() {
                sel.push(AxisSel::Range(0, self.dims.shape()[d]));
            }
            return self.slice(sel);
        }
        NdArrayV::from_ndarray(gather_obs_impl(
            &indices,
            self.dims.shape(),
            self.source.name.clone(),
            |idx| self.get(idx),
        ))
    }

    fn get_row_count(&self) -> usize {
        self.dims.shape()[0]
    }
}

// *** IntoIterator ************************************************

/// Iterating a view works the same as iterating an NdArray: contiguous
/// runs along axis 0, cache-friendly, no per-element offset arithmetic.
impl<'a, T: Float> IntoIterator for &'a NdArrayV<T> {
    type Item = T;
    type IntoIter = NdArrayIter<'a, T>;

    fn into_iter(self) -> NdArrayIter<'a, T> {
        let shape = self.dims.shape();
        let strides = self.dims.strides();
        let n_inner = shape[0];
        let n_runs: usize = shape[1..].iter().product();

        let mut run_offsets = Vec::with_capacity(n_runs);
        if shape.len() <= 1 {
            run_offsets.push(self.offset);
        } else {
            let outer_shape = &shape[1..];
            let outer_strides = &strides[1..];
            let mut outer_indices = vec![0usize; outer_shape.len()];
            for _ in 0..n_runs {
                let off: usize = self.offset + outer_indices.iter()
                    .zip(outer_strides.iter())
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
                run_offsets.push(off);
                let mut carry = true;
                for d in 0..outer_shape.len() {
                    if carry {
                        outer_indices[d] += 1;
                        if outer_indices[d] < outer_shape[d] {
                            carry = false;
                        } else {
                            outer_indices[d] = 0;
                        }
                    }
                }
            }
        }

        NdArrayIter {
            buf: self.source.data.as_slice(),
            n_inner,
            inner_stride: strides[0],
            run_offsets,
            run_idx: 0,
            inner_idx: 0,
            total: self.len(),
            yielded: 0,
        }
    }
}

// *** Trait implementations ***************************************

impl<T: Float> Shape for NdArrayV<T> {
    fn shape(&self) -> ShapeDim {
        match self.dims.ndim() {
            1 => ShapeDim::Rank1(self.dims.shape()[0]),
            2 => ShapeDim::Rank2 {
                rows: self.dims.shape()[0],
                cols: self.dims.shape()[1],
            },
            _ => ShapeDim::RankN(self.dims.shape().to_vec()),
        }
    }
}

impl<T: Float> PartialEq for NdArrayV<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.dims.shape() != other.dims.shape() { return false; }
        self.into_iter()
            .zip(other.into_iter())
            .all(|(a, b)| a == b)
    }
}

// *** Tuple indexing **********************************************

impl<T: Float> Index<(usize,)> for NdArrayV<T> {
    type Output = T;
    #[inline]
    fn index(&self, (i,): (usize,)) -> &T {
        &self.source.data.as_slice()[self.offset_of(&[i])]
    }
}

impl<T: Float> Index<(usize, usize)> for NdArrayV<T> {
    type Output = T;
    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &T {
        &self.source.data.as_slice()[self.offset_of(&[i, j])]
    }
}

impl<T: Float> Index<(usize, usize, usize)> for NdArrayV<T> {
    type Output = T;
    #[inline]
    fn index(&self, (i, j, k): (usize, usize, usize)) -> &T {
        &self.source.data.as_slice()[self.offset_of(&[i, j, k])]
    }
}

impl<T: Float> Index<(usize, usize, usize, usize)> for NdArrayV<T> {
    type Output = T;
    #[inline]
    fn index(&self, (i, j, k, l): (usize, usize, usize, usize)) -> &T {
        &self.source.data.as_slice()[self.offset_of(&[i, j, k, l])]
    }
}

impl<T: Float> Index<(usize, usize, usize, usize, usize)> for NdArrayV<T> {
    type Output = T;
    #[inline]
    fn index(&self, (i, j, k, l, m): (usize, usize, usize, usize, usize)) -> &T {
        &self.source.data.as_slice()[self.offset_of(&[i, j, k, l, m])]
    }
}

// *** Debug *******************************************************

impl<T: Float> fmt::Debug for NdArrayV<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "NdArrayV: {:?} [{}D, offset={}]",
            self.dims.shape(), self.ndim(), self.offset,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enums::shape_dim::ShapeDim;

    #[test]
    fn from_ndarray() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let v = NdArrayV::from_ndarray(a);
        assert_eq!(v.shape(), &[3, 2]);
        assert_eq!(v.len(), 6);
        assert_eq!(v[(0, 0)], 1.0);
        assert_eq!(v[(2, 1)], 6.0);
    }

    #[test]
    fn col_access() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let v = NdArrayV::from_ndarray(a);
        assert_eq!(v.col(0), &[1.0, 2.0, 3.0]);
        assert_eq!(v.col(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn columns() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let v = NdArrayV::from_ndarray(a);
        let cols = v.columns();
        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0], &[1.0, 2.0, 3.0]);
        assert_eq!(cols[1], &[4.0, 5.0, 6.0]);
    }

    #[test]
    #[cfg(feature = "select")]
    fn row_selection_on_view() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let v = a.as_view();
        // Contiguous narrows zero-copy.
        let sub = v.r(1..3);
        assert_eq!(sub.shape(), &[2, 2]);
        assert_eq!(sub.get(&[0, 1]), 5.0);
        // Index arrays gather in order, and the source is unaffected.
        let picked = v.r(&[2, 0]);
        assert_eq!(picked.get(&[0, 0]), 3.0);
        assert_eq!(picked.get(&[1, 1]), 4.0);
    }

    #[test]
    fn apply_on_view() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let out = a.as_view().apply(|x| x + 1.0);
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.get(&[1, 1]), 5.0);
    }

    #[test]
    fn obs_access() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let v = NdArrayV::from_ndarray(a);
        let obs0: Vec<f64> = (&v.obs(0)).into_iter().collect();
        let obs2: Vec<f64> = (&v.obs(2)).into_iter().collect();
        assert_eq!(obs0, vec![1.0, 4.0]);
        assert_eq!(obs2, vec![3.0, 6.0]);
    }

    #[test]
    fn iteration() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let v = NdArrayV::from_ndarray(a);
        let vals: Vec<f64> = (&v).into_iter().collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn iteration_2d() {
        let data: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let a = NdArray::from_slice(&data, &[10, 2]);
        let v = NdArrayV::from_ndarray(a);
        let vals: Vec<f64> = (&v).into_iter().collect();
        assert_eq!(vals.len(), 20);
        assert_eq!(&vals[..10], &data[..10]);
        assert_eq!(&vals[10..], &data[10..]);
    }

    #[test]
    fn with_offset() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let stride = a.strides()[1];
        // View into column 1 only as a 1D view
        let v = NdArrayV::new(a.clone(), stride, &[3], &[1]);
        assert_eq!(v.shape(), &[3]);
        assert_eq!(v[(0,)], 4.0);
        assert_eq!(v[(1,)], 5.0);
        assert_eq!(v[(2,)], 6.0);
    }

    #[test]
    fn to_ndarray() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let v = NdArrayV::from_ndarray(a);
        let b = v.to_ndarray();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.col(0), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn transpose_2d_view() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let t = NdArrayV::from_ndarray(a).transpose();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t[(0, 0)], 1.0);
        assert_eq!(t[(1, 0)], 4.0);
        assert_eq!(t[(0, 2)], 3.0);
        assert_eq!(t[(1, 2)], 6.0);
        // Materialising the transposed view matches the owned transpose.
        let owned = t.to_ndarray();
        assert_eq!(owned.shape(), &[2, 3]);
        assert_eq!(owned.get(&[1, 1]), 5.0);
    }

    #[test]
    fn transpose_3d_view_matches_materialised() {
        let data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
        let a = NdArray::from_slice(&data, &[2, 3, 4]);
        let t = a.as_view().transpose();
        assert_eq!(t.shape(), &[4, 3, 2]);
        let materialised = a.transpose();
        for i in 0..4 {
            for j in 0..3 {
                for k in 0..2 {
                    assert_eq!(t[(i, j, k)], materialised.get(&[i, j, k]));
                }
            }
        }
    }

    #[test]
    fn permute_axes_view() {
        let data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
        let a = NdArray::from_slice(&data, &[2, 3, 4]);
        let p = a.as_view().permute_axes(&[2, 0, 1]);
        assert_eq!(p.shape(), &[4, 2, 3]);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(p[(k, i, j)], a.get(&[i, j, k]));
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "permute_axes")]
    fn permute_axes_rejects_repeat() {
        let a = NdArray::<f64>::new(&[2, 3, 4]);
        let _ = a.as_view().permute_axes(&[0, 0, 1]);
    }

    #[test]
    fn swap_axes_view() {
        let data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
        let a = NdArray::from_slice(&data, &[2, 3, 4]);
        let s = a.as_view().swap_axes(0, 2);
        assert_eq!(s.shape(), &[4, 3, 2]);
        assert_eq!(s[(3, 2, 1)], a.get(&[1, 2, 3]));
        assert_eq!(s[(0, 0, 0)], a.get(&[0, 0, 0]));
    }

    #[test]
    #[should_panic(expected = "obs()")]
    fn obs_on_1d_panics() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let _ = a.as_view().obs(1);
    }

    #[cfg(feature = "matrix")]
    #[test]
    fn to_matrix() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let v = NdArrayV::from_ndarray(a);
        let mat = v.to_matrix().unwrap();
        assert_eq!(mat.n_rows, 3);
        assert_eq!(mat.n_cols, 2);
    }

    #[test]
    fn blas_params() {
        let a = NdArray::<f64>::new(&[10, 5]);
        let v = NdArrayV::from_ndarray(a);
        assert_eq!(v.m(), 10);
        assert_eq!(v.n(), 5);
        assert_eq!(v.lda(), 10);
    }

    #[test]
    fn eq() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let v1 = NdArrayV::from_ndarray(a.clone());
        let v2 = NdArrayV::from_ndarray(a);
        assert_eq!(v1, v2);
    }

    #[test]
    fn shape_trait() {
        let a = NdArray::<f64>::new(&[3, 4]);
        let v = NdArrayV::from_ndarray(a);
        assert_eq!(Shape::shape(&v), ShapeDim::Rank2 { rows: 3, cols: 4 });
    }
}
