//! # **NdArrayV** - *Zero-copy view into an NdArray*
//!
//! Holds an `Arc<NdArray>` to keep the parent alive, plus its own offset
//! and dimension metadata. Views can have different shapes and strides
//! from the parent, enabling slicing and axis selection without copying
//! data.

use std::fmt;
use std::ops::Index;
use std::sync::Arc;

#[cfg(feature = "matrix")]
use crate::enums::error::MinarrowError;
use crate::enums::shape_dim::ShapeDim;
use crate::structs::ndarray::{AxisSel, IntoSlice, NdArray, NdArrayIter, NdDims, offset_of_impl};
use crate::traits::shape::Shape;
use crate::Vec64;

#[cfg(feature = "matrix")]
use crate::structs::matrix::Matrix;

/// Zero-copy view into an [`NdArray`].
///
/// Holds an `Arc<NdArray>` to keep the parent buffer alive, with its
/// own offset and dimension metadata. This enables slicing and axis
/// selection without copying the underlying data.
#[derive(Clone)]
pub struct NdArrayV {
    source: Arc<NdArray>,
    offset: usize,
    dims: NdDims,
}

impl NdArrayV {
    /// Create a view over an NdArray with the given offset and dimensions.
    pub fn new(source: Arc<NdArray>, offset: usize, shape: &[usize], strides: &[usize]) -> Self {
        NdArrayV {
            source,
            offset,
            dims: NdDims::from_shape_and_strides(shape, strides),
        }
    }

    /// Create a full view over an NdArray with the same shape and strides.
    pub fn from_ndarray(source: Arc<NdArray>) -> Self {
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
    pub fn get(&self, indices: &[usize]) -> f64 {
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
    pub fn col(&self, col: usize) -> &[f64] {
        let shape = self.dims.shape();
        assert_eq!(shape.len(), 2, "col() requires a 2D view");
        debug_assert!(col < shape[1]);
        let stride = self.dims.strides()[1];
        let start = self.offset + col * stride;
        &self.source.data.as_slice()[start..start + shape[0]]
    }

    /// All columns as slices. 2D only.
    pub fn columns(&self) -> Vec<&[f64]> {
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
    /// returns 2D [m, k]. For 1D, returns a single-element view.
    pub fn obs(&self, idx: usize) -> NdArrayV {
        let shape = self.dims.shape();
        let strides = self.dims.strides();
        debug_assert!(idx < shape[0], "obs: index {} out of bounds for axis 0 (size {})", idx, shape[0]);

        let new_offset = self.offset + idx * strides[0];
        if shape.len() <= 1 {
            NdArrayV::new(self.source.clone(), new_offset, &[1], &[1])
        } else {
            NdArrayV::new(self.source.clone(), new_offset, &shape[1..], &strides[1..])
        }
    }

    /// Slice this view, producing a sub-view. Zero-copy - shares the
    /// same backing Arc<NdArray>, just adjusts offset and dims.
    pub fn slice<S: IntoSlice>(&self, sel: S) -> NdArrayV {
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

    // *** Materialisation *****************************************

    /// Materialise this view as an owned NdArray.
    pub fn to_ndarray(&self) -> NdArray {
        let flat: Vec64<f64> = self.into_iter().collect();
        let mut arr = NdArray::from_slice(&flat, self.dims.shape());
        arr.name = self.source.name.clone();
        arr
    }

    /// Materialise a 2D view as a Matrix.
    #[cfg(feature = "matrix")]
    pub fn to_matrix(&self) -> Result<Matrix, MinarrowError> {
        self.to_ndarray().to_matrix()
    }

    // *** Parallel iteration (rayon) ******************************

    /// Parallel iterator over axis-0 observations. Each item is the
    /// observation index and a zero-copy `NdArrayV` view.
    #[cfg(feature = "parallel_proc")]
    pub fn par_iter_obs(&self) -> impl rayon::iter::ParallelIterator<Item = (usize, NdArrayV)> + '_ {
        use rayon::prelude::*;
        let n_obs = self.dims.shape()[0];
        (0..n_obs).into_par_iter().map(move |i| (i, self.obs(i)))
    }
}

// *** IntoIterator ************************************************

/// Iterating a view works the same as iterating an NdArray: contiguous
/// runs along axis 0, cache-friendly, no per-element offset arithmetic.
impl<'a> IntoIterator for &'a NdArrayV {
    type Item = f64;
    type IntoIter = NdArrayIter<'a>;

    fn into_iter(self) -> NdArrayIter<'a> {
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

impl Shape for NdArrayV {
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

impl PartialEq for NdArrayV {
    fn eq(&self, other: &Self) -> bool {
        if self.dims.shape() != other.dims.shape() { return false; }
        self.into_iter()
            .zip(other.into_iter())
            .all(|(a, b)| a == b)
    }
}

// *** Tuple indexing **********************************************

impl Index<(usize,)> for NdArrayV {
    type Output = f64;
    #[inline]
    fn index(&self, (i,): (usize,)) -> &f64 {
        &self.source.data.as_slice()[self.offset_of(&[i])]
    }
}

impl Index<(usize, usize)> for NdArrayV {
    type Output = f64;
    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.source.data.as_slice()[self.offset_of(&[i, j])]
    }
}

impl Index<(usize, usize, usize)> for NdArrayV {
    type Output = f64;
    #[inline]
    fn index(&self, (i, j, k): (usize, usize, usize)) -> &f64 {
        &self.source.data.as_slice()[self.offset_of(&[i, j, k])]
    }
}

impl Index<(usize, usize, usize, usize)> for NdArrayV {
    type Output = f64;
    #[inline]
    fn index(&self, (i, j, k, l): (usize, usize, usize, usize)) -> &f64 {
        &self.source.data.as_slice()[self.offset_of(&[i, j, k, l])]
    }
}

impl Index<(usize, usize, usize, usize, usize)> for NdArrayV {
    type Output = f64;
    #[inline]
    fn index(&self, (i, j, k, l, m): (usize, usize, usize, usize, usize)) -> &f64 {
        &self.source.data.as_slice()[self.offset_of(&[i, j, k, l, m])]
    }
}

// *** Debug *******************************************************

impl fmt::Debug for NdArrayV {
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
        let a = Arc::new(NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]));
        let v = NdArrayV::from_ndarray(a);
        assert_eq!(v.shape(), &[3, 2]);
        assert_eq!(v.len(), 6);
        assert_eq!(v[(0, 0)], 1.0);
        assert_eq!(v[(2, 1)], 6.0);
    }

    #[test]
    fn col_access() {
        let a = Arc::new(NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]));
        let v = NdArrayV::from_ndarray(a);
        assert_eq!(v.col(0), &[1.0, 2.0, 3.0]);
        assert_eq!(v.col(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn columns() {
        let a = Arc::new(NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]));
        let v = NdArrayV::from_ndarray(a);
        let cols = v.columns();
        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0], &[1.0, 2.0, 3.0]);
        assert_eq!(cols[1], &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn obs_access() {
        let a = Arc::new(NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]));
        let v = NdArrayV::from_ndarray(a);
        let obs0: Vec<f64> = (&v.obs(0)).into_iter().collect();
        let obs2: Vec<f64> = (&v.obs(2)).into_iter().collect();
        assert_eq!(obs0, vec![1.0, 4.0]);
        assert_eq!(obs2, vec![3.0, 6.0]);
    }

    #[test]
    fn iteration() {
        let a = Arc::new(NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]));
        let v = NdArrayV::from_ndarray(a);
        let vals: Vec<f64> = (&v).into_iter().collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn iteration_with_padding() {
        let data: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let a = Arc::new(NdArray::from_slice(&data, &[10, 2]));
        let v = NdArrayV::from_ndarray(a);
        let vals: Vec<f64> = (&v).into_iter().collect();
        assert_eq!(vals.len(), 20);
        assert_eq!(&vals[..10], &data[..10]);
        assert_eq!(&vals[10..], &data[10..]);
    }

    #[test]
    fn with_offset() {
        let a = Arc::new(NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]));
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
        let a = Arc::new(NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]));
        let v = NdArrayV::from_ndarray(a);
        let b = v.to_ndarray();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.col(0), &[1.0, 2.0, 3.0]);
    }

    #[cfg(feature = "matrix")]
    #[test]
    fn to_matrix() {
        let a = Arc::new(NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]));
        let v = NdArrayV::from_ndarray(a);
        let mat = v.to_matrix().unwrap();
        assert_eq!(mat.n_rows, 3);
        assert_eq!(mat.n_cols, 2);
    }

    #[test]
    fn blas_params() {
        let a = Arc::new(NdArray::new(&[10, 5]));
        let v = NdArrayV::from_ndarray(a);
        assert_eq!(v.m(), 10);
        assert_eq!(v.n(), 5);
        assert_eq!(v.lda(), 16);
    }

    #[test]
    fn eq() {
        let a = Arc::new(NdArray::from_slice(&[1.0, 2.0, 3.0], &[3]));
        let v1 = NdArrayV::from_ndarray(a.clone());
        let v2 = NdArrayV::from_ndarray(a);
        assert_eq!(v1, v2);
    }

    #[test]
    fn shape_trait() {
        let a = Arc::new(NdArray::new(&[3, 4]));
        let v = NdArrayV::from_ndarray(a);
        assert_eq!(Shape::shape(&v), ShapeDim::Rank2 { rows: 3, cols: 4 });
    }
}
