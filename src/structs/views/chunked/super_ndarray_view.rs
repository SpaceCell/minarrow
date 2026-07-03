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

//! # **SuperNdArrayView Module** - *Chunked, Windowed View over N-dimensional Batches*
//!
//! `SuperNdArrayV` is a borrowed, chunked view over a single logical
//! N-dimensional array, exposing an arbitrary `[offset .. offset + len)`
//! axis-0 window that may span multiple underlying batches. It presents
//! those batches as one continuous logical range without copying the
//! underlying memory.
//!
//! ## Role
//! - Represents a batch window over a [`SuperNdArray`], for streaming,
//!   batching, or region-wise processing of tensor data such as sensor
//!   frames landing in chunks.
//! - The N-dimensional counterpart of [`SuperArrayV`](crate::SuperArrayV).
//!
//! ## Interop
//! - Constructed by [`SuperNdArray::slice`] or [`From<SuperNdArray>`].
//! - Materialises to a contiguous [`NdArray`] via [`Consolidate`].
//!
//! ## Invariants
//! - `slices` are ordered, non-overlapping axis-0 windows sharing rank and
//!   trailing shape.
//! - `n_obs` is the logical axis-0 observation count of this view.

use std::fmt;

use crate::enums::error::MinarrowError;
use crate::enums::shape_dim::ShapeDim;
use crate::structs::chunked::super_ndarray::SuperNdArray;
use crate::structs::ndarray::NdArray;
#[cfg(feature = "select")]
use crate::structs::ndarray::gather_obs_impl;
use crate::structs::views::ndarray_view::NdArrayV;
use crate::traits::concatenate::Concatenate;
#[cfg(feature = "select")]
use crate::traits::selection::{AxisSelection, DataSelector, RowSelection};
use crate::traits::consolidate::Consolidate;
use crate::traits::shape::Shape;
use crate::traits::type_unions::Float;
use crate::Vec64;

/// Borrowed view over an arbitrary `[offset .. offset + len)` axis-0 window
/// of a [`SuperNdArray`], spanning batch boundaries without copying.
///
/// ## Fields
/// - `slices`: constituent [`NdArrayV`] windows spanning the range, ordered
///   and sharing rank and trailing shape.
/// - Rank and trailing shape are cached so an empty window keeps its
///   dimensionality.
#[derive(Clone)]
pub struct SuperNdArrayV<T> {
    pub slices: Vec<NdArrayV<T>>,
    ndim: usize,
    inner_shape: Vec<usize>,
}

impl<T: Float> SuperNdArrayV<T> {
    /// Assemble from ordered axis-0 window slices. Panics if the slices
    /// disagree on rank or trailing shape.
    pub fn from_slices(slices: Vec<NdArrayV<T>>, ndim: usize, inner_shape: Vec<usize>) -> Self {
        for (i, s) in slices.iter().enumerate() {
            assert_eq!(
                s.ndim(), ndim,
                "SuperNdArrayV: slice {} has rank {} but expected {}", i, s.ndim(), ndim
            );
            assert_eq!(
                &s.shape()[1..], inner_shape.as_slice(),
                "SuperNdArrayV: slice {} inner shape mismatch", i
            );
        }
        SuperNdArrayV { slices, ndim, inner_shape }
    }

    /// Number of constituent slices.
    #[inline]
    pub fn n_slices(&self) -> usize { self.slices.len() }

    /// Shared rank.
    #[inline]
    pub fn ndim(&self) -> usize { self.ndim }

    /// Dimensions shared across all slices i.e. shape[1..].
    #[inline]
    pub fn inner_shape(&self) -> &[usize] { &self.inner_shape }

    /// Total axis-0 observations across all slices.
    #[inline]
    pub fn n_obs(&self) -> usize {
        self.slices.iter().map(|s| s.shape()[0]).sum()
    }

    /// Total logical elements across all slices.
    #[inline]
    pub fn len(&self) -> usize {
        self.slices.iter().map(|s| s.len()).sum()
    }

    #[inline]
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Logical shape as if consolidated. Axis 0 is the sum across slices.
    pub fn shape(&self) -> Vec<usize> {
        let mut s = vec![self.n_obs()];
        s.extend_from_slice(&self.inner_shape);
        s
    }

    /// Iterator over the constituent slice views.
    #[inline]
    pub fn chunks(&self) -> impl Iterator<Item = &NdArrayV<T>> {
        self.slices.iter()
    }

    /// Returns a sub-window of this view over `[offset .. offset + len)`
    /// axis-0 observations. Zero-copy - the new view narrows each
    /// constituent slice as needed.
    pub fn slice(&self, mut offset: usize, mut len: usize) -> Self {
        assert!(offset + len <= self.n_obs(), "slice out of bounds");

        let mut slices = Vec::new();
        for view in &self.slices {
            let base_obs = view.shape()[0];
            if offset >= base_obs {
                offset -= base_obs;
                continue;
            }

            let take = (base_obs - offset).min(len);
            let mut window_shape = vec![take];
            window_shape.extend_from_slice(&view.shape()[1..]);
            slices.push(NdArrayV::new(
                view.source.clone(),
                view.offset + offset * view.strides()[0],
                &window_shape,
                view.strides(),
            ));

            len -= take;
            if len == 0 {
                break;
            }
            offset = 0;
        }

        SuperNdArrayV {
            slices,
            ndim: self.ndim,
            inner_shape: self.inner_shape.clone(),
        }
    }

    /// Zero-copy view of a single observation (axis-0 element), resolving
    /// which slice contains it. Returns an (N-1)-dimensional view.
    pub fn obs(&self, mut idx: usize) -> NdArrayV<T> {
        for slice in &self.slices {
            let n = slice.shape()[0];
            if idx < n {
                return slice.obs(idx);
            }
            idx -= n;
        }
        panic!("obs: index out of bounds (n_obs {})", self.n_obs());
    }

    /// Get element by global N-dimensional index. The first index is the
    /// global axis-0 position across slices.
    pub fn get(&self, indices: &[usize]) -> T {
        let mut local = indices.to_vec();
        for slice in &self.slices {
            let n = slice.shape()[0];
            if local[0] < n {
                return slice.get(&local);
            }
            local[0] -= n;
        }
        panic!("get: index out of bounds (n_obs {})", self.n_obs());
    }

    /// Apply a function to every logical element, materialising a new
    /// compact [`NdArray`] with this window's shape.
    pub fn apply(&self, f: impl Fn(T) -> T) -> NdArray<T> {
        self.clone().consolidate().apply(f)
    }
}

// *** Axis selection: view.s((1..4, 2)) ***************************

/// Selection across every axis at once over a chunked window. The
/// axis-0 selection narrows the window, and trailing-axis selections
/// narrow each slice. Zero-copy. An axis-0 single index keeps the
/// dimension as a one-observation window - use `obs` to collapse.
/// The resulting rank and trailing shape derive from the selections,
/// so an empty window keeps its dimensionality.
#[cfg(feature = "select")]
impl<T: Float> AxisSelection for SuperNdArrayV<T> {
    type View = SuperNdArrayV<T>;

    fn s(&self, selection: &[&dyn DataSelector]) -> SuperNdArrayV<T> {
        assert_eq!(
            selection.len(), self.ndim,
            "s(): expected {} axes, got {}", self.ndim, selection.len()
        );
        let (start, end, _) = selection[0].resolve_axis(self.n_obs());
        let window = self.slice(start, end - start);
        if self.ndim == 1 {
            return window;
        }

        let inner = &selection[1..];
        let mut inner_shape = Vec::new();
        for (d, sel) in inner.iter().enumerate() {
            let (start, end, collapse) = sel.resolve_axis(window.inner_shape()[d]);
            if !collapse {
                inner_shape.push(end - start);
            }
        }
        let ndim = 1 + inner_shape.len();

        let slices: Vec<NdArrayV<T>> = window
            .slices
            .iter()
            .map(|sv| {
                let full = 0..sv.shape()[0];
                let mut refs: Vec<&dyn DataSelector> = vec![&full];
                refs.extend_from_slice(inner);
                sv.slice(&refs)
            })
            .collect();
        SuperNdArrayV::from_slices(slices, ndim, inner_shape)
    }

    fn get_axis_count(&self) -> usize {
        self.ndim()
    }
}

// *** Row selection: view.r(0..10) ********************************

/// Axis-0 observation selection over a chunked window. Contiguous ranges
/// narrow the window zero-copy. Index arrays gather into one owned batch
/// wrapped in a single-slice view.
#[cfg(feature = "select")]
impl<T: Float> RowSelection for SuperNdArrayV<T> {
    type View = SuperNdArrayV<T>;

    fn r<S: DataSelector>(&self, selection: S) -> SuperNdArrayV<T> {
        if self.slices.is_empty() {
            return SuperNdArrayV::from_slices(Vec::new(), self.ndim, self.inner_shape.clone());
        }
        let indices = selection.resolve_indices(self.n_obs());
        if selection.is_contiguous() {
            let start = indices.first().copied().unwrap_or(0);
            return self.slice(start, indices.len());
        }
        let gathered = gather_obs_impl(&indices, &self.shape(), None, |idx| self.get(idx));
        SuperNdArrayV::from_slices(
            vec![NdArrayV::from_ndarray(gathered)],
            self.ndim,
            self.inner_shape.clone(),
        )
    }

    fn get_row_count(&self) -> usize {
        self.n_obs()
    }
}

// *** IntoIterator ************************************************

/// Iterating a chunked view yields values in column-major order within
/// each slice, crossing slice boundaries in order.
impl<'a, T: Float> IntoIterator for &'a SuperNdArrayV<T> {
    type Item = T;
    type IntoIter = Box<dyn Iterator<Item = T> + 'a>;

    fn into_iter(self) -> Self::IntoIter {
        Box::new(self.slices.iter().flat_map(|s| s.into_iter()))
    }
}

// *** Trait implementations ***************************************

impl<T: Float> Shape for SuperNdArrayV<T> {
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

impl<T: Float> Consolidate for SuperNdArrayV<T> {
    type Output = NdArray<T>;

    /// Materialise the window into a contiguous compact [`NdArray`],
    /// interleaving each slice's axis-0 rows one column at a time.
    fn consolidate(self) -> NdArray<T> {
        if self.slices.is_empty() {
            return NdArray::new(&[0]);
        }

        let full_shape = self.shape();
        let total_obs = full_shape[0];
        let n_cols: usize = self.inner_shape.iter().product::<usize>().max(1);

        // Each slice's iterator yields column-major logical values, so its
        // column runs arrive in order and interleave by column index.
        let per_slice: Vec<(usize, Vec<T>)> = self
            .slices
            .iter()
            .map(|s| (s.shape()[0], s.into_iter().collect()))
            .collect();

        let mut flat: Vec64<T> = Vec64::with_capacity(total_obs * n_cols);
        for c in 0..n_cols {
            for (obs, elems) in &per_slice {
                flat.extend_from_slice(&elems[c * obs..(c + 1) * obs]);
            }
        }
        NdArray::from_slice(&flat, &full_shape)
    }
}

impl<T: Float> Concatenate for SuperNdArrayV<T> {
    /// Concatenates two chunked views along axis 0 by appending the other
    /// view's slices. Zero-copy - both views' slices carry across.
    fn concat(mut self, other: Self) -> Result<Self, MinarrowError> {
        if self.slices.is_empty() {
            return Ok(other);
        }
        if other.slices.is_empty() {
            return Ok(self);
        }
        if self.ndim != other.ndim || self.inner_shape != other.inner_shape {
            return Err(MinarrowError::IncompatibleTypeError {
                from: "SuperNdArrayV",
                to: "SuperNdArrayV",
                message: Some(format!(
                    "shape {:?} vs {:?}", self.shape(), other.shape()
                )),
            });
        }
        self.slices.extend(other.slices);
        Ok(self)
    }
}

/// Logical equality over shape and values in logical order. Slice
/// boundaries do not affect equality.
impl<T: Float> PartialEq for SuperNdArrayV<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.ndim != other.ndim
            || self.inner_shape != other.inner_shape
            || self.n_obs() != other.n_obs()
        {
            return false;
        }
        let a = self.clone().consolidate();
        let b = other.clone().consolidate();
        (&a).into_iter().eq((&b).into_iter())
    }
}

impl<T: Float> fmt::Debug for SuperNdArrayV<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SuperNdArrayV: {} slices, {}D, shape {:?}, {} elements",
            self.n_slices(),
            self.ndim,
            self.shape(),
            self.len()
        )
    }
}

/// SuperNdArray -> SuperNdArrayV conversion. Each batch becomes a full
/// axis-0 slice view, keeping the parent batches alive through each
/// batch's shared internal buffer.
impl<T: Float> From<SuperNdArray<T>> for SuperNdArrayV<T> {
    fn from(super_nd: SuperNdArray<T>) -> Self {
        let ndim = super_nd.ndim();
        let inner_shape = super_nd.inner_shape().to_vec();
        let slices: Vec<NdArrayV<T>> = super_nd
            .batches
            .iter()
            .map(|b| NdArrayV::from_ndarray(b.clone()))
            .collect();
        SuperNdArrayV { slices, ndim, inner_shape }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_batch_2d() -> SuperNdArray<f64> {
        SuperNdArray::from_batches(
            vec![
                NdArray::from_slice(&[1.0, 2.0, 10.0, 20.0], &[2, 2]),
                NdArray::from_slice(&[3.0, 4.0, 5.0, 30.0, 40.0, 50.0], &[3, 2]),
            ],
            "data",
        )
    }

    #[test]
    fn full_view_from_super() {
        let snd = two_batch_2d();
        let v = SuperNdArrayV::from(snd);
        assert_eq!(v.n_slices(), 2);
        assert_eq!(v.n_obs(), 5);
        assert_eq!(v.shape(), vec![5, 2]);
        assert_eq!(v.get(&[0, 0]), 1.0);
        assert_eq!(v.get(&[2, 0]), 3.0);
        assert_eq!(v.get(&[4, 1]), 50.0);
    }

    #[test]
    fn window_spans_batches() {
        let snd = two_batch_2d();
        // Rows 1..4 span the batch boundary at row 2.
        let v = snd.slice(1, 3);
        assert_eq!(v.n_slices(), 2);
        assert_eq!(v.n_obs(), 3);
        assert_eq!(v.get(&[0, 0]), 2.0);
        assert_eq!(v.get(&[1, 0]), 3.0);
        assert_eq!(v.get(&[2, 1]), 40.0);
    }

    #[test]
    fn sub_window() {
        let snd = two_batch_2d();
        let v = snd.slice(0, 5);
        let sub = v.slice(1, 3);
        assert_eq!(sub.n_obs(), 3);
        assert_eq!(sub.get(&[0, 0]), 2.0);
        assert_eq!(sub.get(&[2, 0]), 4.0);
    }

    #[test]
    fn obs_across_boundary() {
        let snd = two_batch_2d();
        let v = snd.slice(0, 5);
        let o = v.obs(3);
        assert_eq!(o.shape(), &[2]);
        assert_eq!(o.get(&[0]), 4.0);
        assert_eq!(o.get(&[1]), 40.0);
    }

    #[test]
    fn consolidate_window() {
        let snd = two_batch_2d();
        let v = snd.slice(1, 3);
        let nd = v.consolidate();
        assert_eq!(nd.shape(), &[3, 2]);
        assert!(nd.is_contiguous());
        assert_eq!(nd.col(0), &[2.0, 3.0, 4.0]);
        assert_eq!(nd.col(1), &[20.0, 30.0, 40.0]);
    }

    #[test]
    fn iteration_crosses_slices() {
        let snd = SuperNdArray::from_batches(
            vec![
                NdArray::from_slice(&[1.0, 2.0], &[2]),
                NdArray::from_slice(&[3.0], &[1]),
            ],
            "1d",
        );
        let v = snd.slice(0, 3);
        let vals: Vec<f64> = (&v).into_iter().collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn eq_ignores_slice_boundaries() {
        let snd = two_batch_2d();
        let whole = snd.slice(0, 5);
        let single = SuperNdArrayV::from(SuperNdArray::from_batches(
            vec![NdArray::from_slice(
                &[1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
                &[5, 2],
            )],
            "one",
        ));
        assert_eq!(whole, single);
    }

    #[cfg(feature = "select")]
    #[test]
    fn row_selection_on_window() {
        let snd = two_batch_2d();
        let v = snd.slice(0, 5);
        // Contiguous sub-selection narrows zero-copy.
        let sub = v.r(1..4);
        assert_eq!(sub.n_obs(), 3);
        assert_eq!(sub.get(&[0, 0]), 2.0);
        // Index selection gathers in order.
        let picked = v.r(&[4, 0]);
        assert_eq!(picked.n_slices(), 1);
        assert_eq!(picked.get(&[0, 1]), 50.0);
        assert_eq!(picked.get(&[1, 0]), 1.0);
    }

    #[test]
    fn apply_materialises_window() {
        let snd = two_batch_2d();
        let out = snd.slice(1, 3).apply(|x| x * 2.0);
        assert_eq!(out.shape(), &[3, 2]);
        assert_eq!(out.get(&[0, 0]), 4.0);
        assert_eq!(out.get(&[2, 1]), 80.0);
    }

    #[test]
    fn concat_appends_slices() {
        let snd = two_batch_2d();
        let a = snd.slice(0, 2);
        let b = snd.slice(2, 3);
        let joined = a.concat(b).unwrap();
        assert_eq!(joined.n_obs(), 5);
        assert_eq!(joined.get(&[4, 1]), 50.0);
    }

    #[cfg(feature = "broadcast")]
    #[test]
    fn native_operators() {
        let snd = two_batch_2d();
        let a = snd.slice(1, 3);
        let b = snd.slice(1, 3);
        let sum = (a + b).unwrap();
        assert_eq!(sum.shape(), &[3, 2]);
        assert_eq!(sum.get(&[0, 0]), 4.0);
        assert_eq!(sum.get(&[2, 1]), 80.0);
    }

    #[cfg(all(feature = "broadcast", feature = "value_type"))]
    #[test]
    fn value_window_and_pair() {
        use std::sync::Arc;
        use crate::Value;

        let snd = two_batch_2d();
        let v = Value::SuperNdArray(Arc::new(snd));
        // Value::slice windows across batch boundaries into a view variant.
        let window = v.slice(1, 3);
        let Value::SuperNdArrayView(w) = &window else {
            panic!("expected Value::SuperNdArrayView");
        };
        assert_eq!(w.n_obs(), 3);
        assert_eq!(w.get(&[0, 0]), 2.0);

        let sum = (window.clone() + window).unwrap();
        let Value::NdArray(nd) = sum else {
            panic!("expected Value::NdArray");
        };
        assert_eq!(nd.get(&[2, 1]), 80.0);
    }
}
