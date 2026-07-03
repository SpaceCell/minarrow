//! # **NdArray** - *N-dimensional dense array for scientific and statistical computing*
//!
//! Comparable to NumPy's `ndarray` or Rust's `ndarray` crate, but backed
//! by Minarrow's [`Buffer<T>`], so external memory such as an imported
//! DLPack tensor wraps without copying. Conversions into the padded and
//! columnar types re-lay data for their layouts.
//!
//! ## Intent
//! NdArray is the landing container for n-dimensional numeric data - off a
//! network, sensor feed, or file - and the bridge to Python and GPU
//! frameworks over DLPack, or a basis for building on in Rust. 
//! It is not a 'Python NdArray-equivalent' compute library. Kernels, BLAS, and
//! deep learning frameworks operate on the data it carries.
//! It's therefore here to make it easier and more consistent to have all the required
//! data structures in one place for bridging rather than needing to collate separate 
//! sources for different data structures.
//! 
//! ## Element type
//! Generic over `T: Float`, which covers `f32` and `f64`. Rank is determined
//! at runtime rather than through generics, and dimensions 1D-5D are stored
//! inline to avoid heap allocation in the common case.
//!
//! ## Layout
//! Compact column-major with no inter-dimension padding, so the buffer is
//! fully contiguous and hands to DLPack consumers as a dense tensor. The
//! allocation start is 64-byte aligned through `Vec64`, which serves
//! whole-buffer SIMD over the flattened data. Padded per-column alignment
//! for BLAS/LAPACK column access is [`Matrix`]'s job, and `to_matrix`
//! re-lays data into that form.
//! 
//! # SIMD only in 1D
//! Notably, this is different to the other structures in Minarrow, which optimise
//! for columnar SIMD. NdArray does not - except when the array is laid out flat in 1D.
//! The reasoning is that SIMD is for fast decisions via CPU. With Nd dimensions,
//! typically one will use a GPU which has different optimisation mechanisms that don't
//! benefit from 64-byte padding, which then otherwise causes issues with Python libraries
//! like PyTorch that may re-allocate in order to achieve a contiguous layout.
//!
//! ## Null handling
//! Missing values are represented as NaN. BLAS/LAPACK and IEEE 754 arithmetic
//! propagate NaN through results, so nulls flow through compute without
//! special-case logic.
//!
//! ## Interop
//! - [`Matrix`] to 2D NdArray - zero-copy, the padded stride carries through (f64).
//! - 2D NdArray to [`Matrix`] - re-lays into the padded column layout, zero-copy
//!   when the stride already matches (f64).
//! - 2D NdArray to [`Table`] - each column copies into its own 64-byte aligned
//!   `FloatArray` (f64).
//! - 1D NdArray to [`Array`] - moves the buffer into a `FloatArray<f64>`.
//! - [`Table`] to NdArray via [`TryFrom`] - copies, converting nulls to NaN (f64).
//! - DLPack export and import for zero-copy sharing with PyTorch, JAX,
//!   TensorFlow (f32 and f64).

use std::fmt;
use std::ops::{Index, IndexMut, Range, RangeFrom, RangeFull, RangeTo};
use std::sync::Arc;

use crate::enums::error::MinarrowError;
use crate::enums::shape_dim::ShapeDim;
use crate::structs::buffer::Buffer;
#[cfg(all(feature = "views", feature = "select"))]
use crate::traits::selection::{AxisSelection, DataSelector, RowSelection};
use crate::traits::type_unions::Float;
use crate::traits::{concatenate::Concatenate, shape::Shape};
use crate::{Array, ArrowType, Field, FieldArray, FloatArray, NumericArray, Table, Vec64};

#[cfg(feature = "matrix")]
use crate::structs::matrix::{Matrix, aligned_stride};
#[cfg(feature = "views")]
use crate::structs::views::ndarray_view::NdArrayV;
#[cfg(feature = "dlpack")]
use crate::ffi::dlpack::{export_to_dlpack, DLPackTensor};

// ****************************************************************
// NdArray
// ****************************************************************

/// N-dimensional dense array of float values.
///
/// Backed by [`Buffer<T>`] for zero-copy interop with external memory.
/// Compact column-major layout with a 64-byte aligned allocation start.
///
/// See the [module-level documentation](self) for design rationale.
#[derive(Clone, PartialEq)]
pub struct NdArray<T> {
    pub(crate) data: Arc<Buffer<T>>,
    pub(crate) dims: NdDims,
    pub name: Option<String>,
}

// *** Construction ************************************************

impl<T: Float> NdArray<T> {
    /// Create a zeroed NdArray with the given shape, column-major strides.
    pub fn new(shape: &[usize]) -> Self {
        let dims = NdDims::from_shape(shape);
        let total = buffer_len(dims.shape(), dims.strides());
        let mut v = Vec64::with_capacity(total);
        v.0.resize(total, T::default());
        NdArray { data: Arc::new(Buffer::from_vec64(v)), dims, name: None }
    }

    /// Create a zeroed NdArray with a name.
    pub fn new_named(shape: &[usize], name: impl Into<String>) -> Self {
        let mut arr = Self::new(shape);
        arr.name = Some(name.into());
        arr
    }

    /// Create from a flat column-major slice and shape.
    ///
    /// The slice holds `product(shape)` logical elements in column-major
    /// order, matching the compact layout, so the data copies straight in.
    pub fn from_slice(data: &[T], shape: &[usize]) -> Self {
        let logical_len: usize = shape.iter().product();
        assert_eq!(
            data.len(), logical_len,
            "NdArray::from_slice: data length {} does not match shape product {}",
            data.len(), logical_len
        );
        let dims = NdDims::from_shape(shape);
        NdArray {
            data: Arc::new(Buffer::from_slice(data)),
            dims,
            name: None,
        }
    }

    /// Create from a pre-owned `Buffer<f64>` with explicit shape and strides.
    ///
    /// The buffer must already contain `buffer_len(shape, strides)` elements
    /// in the correct strided layout.
    pub fn from_buffer(data: Buffer<T>, shape: &[usize], strides: &[usize]) -> Self {
        let required = buffer_len(shape, strides);
        assert!(
            data.len() >= required,
            "NdArray::from_buffer: buffer has {} elements but shape requires {}",
            data.len(), required
        );
        let dims = NdDims::from_shape_and_strides(shape, strides);
        NdArray { data: Arc::new(data), dims, name: None }
    }

    /// Create an NdArray filled with a constant value.
    pub fn fill(shape: &[usize], value: T) -> Self {
        let dims = NdDims::from_shape(shape);
        let total = buffer_len(dims.shape(), dims.strides());
        let mut v = Vec64::with_capacity(total);
        v.0.resize(total, value);
        NdArray { data: Arc::new(Buffer::from_vec64(v)), dims, name: None }
    }

    /// Create an NdArray of ones.
    pub fn ones(shape: &[usize]) -> Self {
        Self::fill(shape, T::one())
    }

    /// Create a 2D identity matrix.
    pub fn eye(n: usize) -> Self {
        let mut arr = Self::new(&[n, n]);
        let stride = arr.dims.strides()[1];
        let buf = Arc::make_mut(&mut arr.data).as_mut_slice();
        for i in 0..n {
            buf[i * stride + i] = T::one();
        }
        arr
    }

    /// Create a 1D NdArray with evenly spaced values in `[start, end]`.
    pub fn linspace(start: T, end: T, n: usize) -> Self {
        assert!(n >= 2, "linspace requires at least 2 points");
        let step = (end - start) / T::from(n - 1).unwrap();
        let v: Vec64<T> = (0..n).map(|i| start + step * T::from(i).unwrap()).collect();
        NdArray {
            data: Arc::new(Buffer::from_vec64(v)),
            dims: NdDims::from_shape(&[n]),
            name: None,
        }
    }

    /// Create a 1D NdArray with values `start, start+step, start+2*step, ...`
    /// for `n` elements.
    pub fn arange(start: T, step: T, n: usize) -> Self {
        let v: Vec64<T> = (0..n).map(|i| start + step * T::from(i).unwrap()).collect();
        NdArray {
            data: Arc::new(Buffer::from_vec64(v)),
            dims: NdDims::from_shape(&[n]),
            name: None,
        }
    }

    // *** Shape and introspection *********************************

    /// Number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize { self.dims.ndim() }

    /// Shape as a slice of dimension sizes.
    #[inline]
    pub fn shape(&self) -> &[usize] { self.dims.shape() }

    /// Strides as a slice of element offsets per dimension.
    #[inline]
    pub fn strides(&self) -> &[usize] { self.dims.strides() }

    /// Total number of logical elements i.e. the product of shape.
    #[inline]
    pub fn len(&self) -> usize { self.dims.len() }

    /// True if any dimension is zero.
    #[inline]
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// True if the buffer layout matches compact column-major strides,
    /// with no transposition or non-standard stride pattern.
    pub fn is_contiguous(&self) -> bool {
        let shape = self.dims.shape();
        let strides = self.dims.strides();
        let mut expected = 1;
        for d in 0..shape.len() {
            if strides[d] != expected {
                return false;
            }
            expected *= shape[d];
        }
        true
    }

    /// True if any element is NaN.
    pub fn has_nan(&self) -> bool {
        self.into_iter().any(|v| v.is_nan())
    }

    /// Count of NaN elements.
    pub fn nan_count(&self) -> usize {
        self.into_iter().filter(|v| v.is_nan()).count()
    }

    // *** Element access ******************************************

    /// Get element by N-dimensional index. Panics if out of bounds.
    #[inline]
    pub fn get(&self, indices: &[usize]) -> T {
        self.data.as_slice()[self.offset_of(indices)]
    }

    /// Set element by N-dimensional index. Panics if out of bounds.
    /// Triggers copy-on-write when views share the buffer.
    #[inline]
    pub fn set(&mut self, indices: &[usize], value: T) {
        let off = self.offset_of(indices);
        Arc::make_mut(&mut self.data).as_mut_slice()[off] = value;
    }

    /// Compute flat buffer offset for an N-dimensional index.
    #[inline]
    pub(crate) fn offset_of(&self, indices: &[usize]) -> usize {
        offset_of_impl(indices, self.dims.shape(), self.dims.strides())
    }

    // *** Metadata ************************************************

    /// Set the array name.
    #[inline]
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
    }

    /// Immutable reference to the full flat buffer.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    /// Mutable reference to the full flat buffer. Triggers copy-on-write
    /// when views share the buffer.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        Arc::make_mut(&mut self.data).as_mut_slice()
    }

    /// Fill every logical element with a value.
    pub fn fill_with(&mut self, value: T) {
        // For contiguous arrays, fill the whole buffer
        if self.is_contiguous() {
            Arc::make_mut(&mut self.data).as_mut_slice().fill(value);
            return;
        }
        // Walk logical positions for non-contiguous layouts
        let offsets: Vec<usize> = {
            let shape = self.dims.shape();
            let strides = self.dims.strides();
            let mut result = Vec::with_capacity(self.len());
            let mut indices = vec![0usize; shape.len()];
            for _ in 0..self.len() {
                result.push(indices.iter().zip(strides).map(|(&i, &s)| i * s).sum());
                let mut carry = true;
                for d in 0..shape.len() {
                    if carry {
                        indices[d] += 1;
                        if indices[d] < shape[d] { carry = false; } else { indices[d] = 0; }
                    }
                }
            }
            result
        };
        let buf = Arc::make_mut(&mut self.data).as_mut_slice();
        for off in offsets { buf[off] = value; }
    }

    // *** 2D axis access (column-major) ***************************

    /// Immutable column slice for a 2D array. Panics if ndim != 2.
    #[inline]
    pub fn col(&self, col: usize) -> &[T] {
        let shape = self.dims.shape();
        assert_eq!(shape.len(), 2, "col() requires a 2D array");
        debug_assert!(col < shape[1], "Column index out of bounds");
        let stride = self.dims.strides()[1];
        let start = col * stride;
        &self.data.as_slice()[start..start + shape[0]]
    }

    /// Mutable column slice for a 2D array. Panics if ndim != 2.
    /// Triggers copy-on-write if the buffer is shared.
    #[inline]
    pub fn col_mut(&mut self, col: usize) -> &mut [T] {
        let shape = self.dims.shape();
        assert_eq!(shape.len(), 2, "col_mut() requires a 2D array");
        debug_assert!(col < shape[1], "Column index out of bounds");
        let stride = self.dims.strides()[1];
        let n_rows = shape[0];
        let start = col * stride;
        &mut Arc::make_mut(&mut self.data).as_mut_slice()[start..start + n_rows]
    }

    /// All columns as immutable slices. 2D only.
    pub fn columns(&self) -> Vec<&[T]> {
        let shape = self.dims.shape();
        assert_eq!(shape.len(), 2, "columns() requires a 2D array");
        let stride = self.dims.strides()[1];
        let n_rows = shape[0];
        let buf = self.data.as_slice();
        (0..shape[1])
            .map(|c| &buf[c * stride..c * stride + n_rows])
            .collect()
    }

    /// All columns as mutable slices. 2D only.
    pub fn columns_mut(&mut self) -> Vec<&mut [T]> {
        let shape = self.dims.shape();
        assert_eq!(shape.len(), 2, "columns_mut() requires a 2D array");
        let n_rows = shape[0];
        let n_cols = shape[1];
        let stride = self.dims.strides()[1];
        let ptr = Arc::make_mut(&mut self.data).as_mut_slice().as_mut_ptr();
        let mut result = Vec::with_capacity(n_cols);
        for c in 0..n_cols {
            let start = c * stride;
            // SAFETY: each slice is within bounds and non-overlapping,
            // we have exclusive &mut access.
            unsafe {
                let col_ptr = ptr.add(start);
                result.push(std::slice::from_raw_parts_mut(col_ptr, n_rows));
            }
        }
        result
    }

    /// Wrap as a full `NdArrayV` view for repeated observation access.
    ///
    /// Call `.obs(i)` on the returned view to get individual observations.
    /// The view holds the parent through the array's shared internal
    /// buffer, so this is a refcount bump.
    #[cfg(feature = "views")]
    pub fn as_view(&self) -> NdArrayV<T> {
        NdArrayV::from_ndarray(self.clone())
    }

    /// Zero-copy view of a single observation (axis-0 element).
    ///
    /// Returns an (N-1)-dimensional `NdArrayV` view. For a 2D array
    /// with shape [n, m], returns a 1D view of shape [m]. For 3D
    /// [n, m, k], returns 2D [m, k]. Requires rank 2 or higher - for
    /// scalar access on a 1D array use `get(&[i])`.
    ///
    /// For repeated access in a loop, prefer `nd.as_view()` then call
    /// `.obs()` on the view to avoid re-wrapping each time.
    #[cfg(feature = "views")]
    pub fn obs(&self, idx: usize) -> NdArrayV<T> {
        self.as_view().obs(idx)
    }

    // *** BLAS/LAPACK compatibility (2D) **************************

    /// Number of rows as i32. Panics if ndim != 2.
    #[inline]
    pub fn m(&self) -> i32 {
        assert_eq!(self.ndim(), 2, "m() requires a 2D array");
        self.dims.shape()[0] as i32
    }

    /// Number of columns as i32. Panics if ndim != 2.
    #[inline]
    pub fn n(&self) -> i32 {
        assert_eq!(self.ndim(), 2, "n() requires a 2D array");
        self.dims.shape()[1] as i32
    }

    /// Leading dimension for BLAS. Panics if ndim != 2.
    #[inline]
    pub fn lda(&self) -> i32 {
        assert_eq!(self.ndim(), 2, "lda() requires a 2D array");
        self.dims.strides()[1] as i32
    }

    // *** Reshape *************************************************

    /// Reshape to a new shape. Returns a new NdArray with re-laid out data.
    ///
    /// The total number of logical elements must match. Data is copied
    /// from logical element order into the new strided layout.
    pub fn reshape(&self, new_shape: &[usize]) -> Result<NdArray<T>, MinarrowError> {
        let new_len: usize = new_shape.iter().product();
        if new_len != self.len() {
            return Err(MinarrowError::ShapeError {
                message: format!(
                    "reshape: cannot reshape array of size {} into shape {:?}",
                    self.len(), new_shape
                ),
            });
        }
        let flat: Vec64<T> = self.into_iter().collect();
        Ok(NdArray::from_slice(&flat, new_shape))
    }

    /// Transpose by reversing the axis order. Returns a new NdArray with
    /// re-laid out data. A 1D array copies through unchanged.
    ///
    /// For a zero-copy transposed view, call `as_view()` and use the
    /// view's `transpose()`.
    pub fn transpose(&self) -> NdArray<T> {
        let shape = self.dims.shape();
        let ndim = shape.len();
        if ndim <= 1 {
            let mut result = self.to_contiguous();
            result.name = self.name.clone();
            return result;
        }
        if ndim == 2 {
            let (n_rows, n_cols) = (shape[0], shape[1]);
            let new_shape = [n_cols, n_rows];
            let mut result = NdArray::new(&new_shape);
            let src_stride = self.dims.strides()[1];
            let dst_stride = result.dims.strides()[1];
            let src = self.data.as_slice();
            let dst = Arc::make_mut(&mut result.data).as_mut_slice();
            for r in 0..n_rows {
                for c in 0..n_cols {
                    dst[r * dst_stride + c] = src[c * src_stride + r];
                }
            }
            result.name = self.name.clone();
            return result;
        }

        // General N-D. Walking the result's logical positions in column-major
        // order reads the source at the reversed index, which is the source
        // walked with reversed strides.
        let new_shape: Vec<usize> = shape.iter().rev().copied().collect();
        let rev_strides: Vec<usize> = self.dims.strides().iter().rev().copied().collect();
        let src = self.data.as_slice();
        let total = self.len();
        let mut buf = Vec64::with_capacity(total);
        let mut indices = vec![0usize; ndim];
        for _ in 0..total {
            let offset: usize = indices.iter()
                .zip(rev_strides.iter())
                .map(|(&i, &s)| i * s)
                .sum();
            buf.push(src[offset]);
            let mut carry = true;
            for d in 0..ndim {
                if carry {
                    indices[d] += 1;
                    if indices[d] < new_shape[d] {
                        carry = false;
                    } else {
                        indices[d] = 0;
                    }
                }
            }
        }
        NdArray {
            data: Arc::new(Buffer::from_vec64(buf)),
            dims: NdDims::from_shape(&new_shape),
            name: self.name.clone(),
        }
    }

    /// Flatten to a contiguous 1D array.
    pub fn flatten(&self) -> NdArray<T> {
        let flat: Vec64<T> = self.into_iter().collect();
        let n = flat.len();
        NdArray {
            data: Arc::new(Buffer::from_vec64(flat)),
            dims: NdDims::from_shape(&[n]),
            name: self.name.clone(),
        }
    }

    /// If the array has non-standard strides, re-lay out into default
    /// column-major contiguous form.
    pub fn to_contiguous(&self) -> NdArray<T> {
        if self.is_contiguous() {
            return self.clone();
        }
        NdArray::from_slice(
            &self.into_iter().collect::<Vec64<T>>(),
            self.shape(),
        )
    }

    // *** Slicing: arr.slice(nd![1..4, 2..5]) *************************

    /// Slice this array along any combination of axes.
    ///
    /// Each axis takes any [`DataSelector`] - a single index collapses
    /// that dimension, and a contiguous range keeps it. Returns a
    /// zero-copy `NdArrayV` view that holds the parent through the
    /// array's shared internal buffer, so this is a refcount bump,
    /// matching `as_view` and `obs`.
    ///
    /// # Examples
    /// ```ignore
    /// arr.slice(&[&2])              // single index on 1D
    /// arr.slice(&[&(1..4)])         // range on axis 0
    /// arr.slice(nd![1..4, 2..5])    // range on both axes (2D)
    /// arr.slice(nd![0..3, 2])       // range on axis 0, single on axis 1
    /// arr.slice(nd![1, 0..4, 3])    // mixed for 3D
    /// ```
    #[cfg(all(feature = "views", feature = "select"))]
    pub fn slice(&self, selection: &[&dyn DataSelector]) -> NdArrayV<T> {
        assert_eq!(
            selection.len(), self.ndim(),
            "slice(): expected {} axes, got {}", self.ndim(), selection.len()
        );

        let shape = self.dims.shape();
        let strides = self.dims.strides();

        // Compute the new offset, shape, and strides
        let mut new_offset: usize = 0;
        let mut new_shape = Vec::with_capacity(self.ndim());
        let mut new_strides = Vec::with_capacity(self.ndim());

        for (d, sel) in selection.iter().enumerate() {
            let (start, end, collapse) = sel.resolve_axis(shape[d]);
            debug_assert!(
                end <= shape[d],
                "slice(): end {} out of bounds for axis {} (size {})", end, d, shape[d]
            );
            new_offset += start * strides[d];
            if !collapse {
                new_shape.push(end - start);
                new_strides.push(strides[d]);
            }
        }

        // If all axes were single indices, return a 1-element 1D view
        if new_shape.is_empty() {
            new_shape.push(1);
            new_strides.push(1);
        }

        NdArrayV::new(self.clone(), new_offset, &new_shape, &new_strides)
    }

    // *** Apply ***************************************************

    /// Apply a function to every logical element, returning a new compact
    /// array with this array's shape and name. The closure brings the
    /// computation, so any kernel under `kernels` runs through this
    /// entry point.
    pub fn apply(&self, f: impl Fn(T) -> T) -> NdArray<T> {
        let flat: Vec64<T> = self.into_iter().map(f).collect();
        let mut result = NdArray::from_slice(&flat, self.shape());
        result.name = self.name.clone();
        result
    }

    /// Apply a function to every logical element in place, with no
    /// allocation. Copy-on-write triggers first when views share the
    /// buffer.
    pub fn apply_mut(&mut self, f: impl Fn(T) -> T) {
        if self.is_contiguous() {
            for v in Arc::make_mut(&mut self.data).as_mut_slice() {
                *v = f(*v);
            }
            return;
        }
        // Walk logical positions for non-contiguous layouts so stride
        // padding stays untouched.
        let shape = self.dims.shape().to_vec();
        let strides = self.dims.strides().to_vec();
        let total = self.len();
        let buf = Arc::make_mut(&mut self.data).as_mut_slice();
        let mut indices = vec![0usize; shape.len()];
        for _ in 0..total {
            let offset: usize = indices.iter().zip(strides.iter()).map(|(&i, &s)| i * s).sum();
            buf[offset] = f(buf[offset]);
            let mut carry = true;
            for d in 0..shape.len() {
                if carry {
                    indices[d] += 1;
                    if indices[d] < shape[d] {
                        carry = false;
                    } else {
                        indices[d] = 0;
                    }
                }
            }
        }
    }

    /// Apply a function to every 1D lane along the given axis, collapsing
    /// that axis. Each lane arrives as a zero-copy [`NdArrayV`] and the
    /// closure returns one value for it, so the output shape drops `axis`.
    /// Requires rank 2 or higher - a 1D array is itself a single lane.
    #[cfg(feature = "views")]
    pub fn apply_axis(&self, axis: usize, mut f: impl FnMut(NdArrayV<T>) -> T) -> NdArray<T> {
        let shape = self.dims.shape();
        let strides = self.dims.strides();
        let ndim = shape.len();
        assert!(ndim >= 2, "apply_axis requires a 2D or higher array");
        assert!(axis < ndim, "apply_axis: axis {} out of bounds for {}D array", axis, ndim);

        let out_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .filter(|(d, _)| *d != axis)
            .map(|(_, &s)| s)
            .collect();
        let out_dims: Vec<usize> = (0..ndim).filter(|&d| d != axis).collect();
        let total: usize = out_shape.iter().product();

        let lane_shape = [shape[axis]];
        let lane_strides = [strides[axis]];

        // Walk the output positions in column-major order. Each position
        // anchors one lane's base offset in the source.
        let mut flat = Vec64::with_capacity(total);
        let mut indices = vec![0usize; out_shape.len()];
        for _ in 0..total {
            let offset: usize = indices
                .iter()
                .zip(out_dims.iter())
                .map(|(&i, &d)| i * strides[d])
                .sum();
            let lane = NdArrayV::new(self.clone(), offset, &lane_shape, &lane_strides);
            flat.push(f(lane));
            let mut carry = true;
            for d in 0..out_shape.len() {
                if carry {
                    indices[d] += 1;
                    if indices[d] < out_shape[d] {
                        carry = false;
                    } else {
                        indices[d] = 0;
                    }
                }
            }
        }
        let mut result = NdArray::from_slice(&flat, &out_shape);
        result.name = self.name.clone();
        result
    }

    // *** Conversions *********************************************

    /// Export as a DLPack tensor for zero-copy sharing with PyTorch,
    /// NumPy, JAX, and other DLPack-compatible frameworks.
    ///
    /// Returns a `DLPackTensor` that manages the lifecycle. Drop it to
    /// release, or call `.into_raw()` to transfer ownership to an FFI
    /// consumer such as a PyCapsule.
    #[cfg(feature = "dlpack")]
    pub fn to_dlpack(self) -> DLPackTensor {
        export_to_dlpack(self)
    }

    // *** Parallel iteration (rayon) ******************************

    /// Parallel iterator over the underlying buffer. Rayon splits
    /// the contiguous data into chunks across threads automatically.
    #[cfg(feature = "parallel_proc")]
    pub fn par_iter(&self) -> rayon::slice::Iter<'_, T> {
        use rayon::prelude::*;
        self.data.as_slice().par_iter()
    }

    /// Parallel chunks of the underlying buffer. Each chunk is a
    /// contiguous `&[f64]` slice that rayon distributes across threads.
    #[cfg(feature = "parallel_proc")]
    pub fn par_chunks(&self, chunk_size: usize) -> rayon::slice::Chunks<'_, T> {
        use rayon::prelude::*;
        self.data.as_slice().par_chunks(chunk_size)
    }

    /// Parallel iterator over axis-0 observations. Each item is the
    /// observation index and a zero-copy `NdArrayV` view.
    #[cfg(all(feature = "parallel_proc", feature = "views"))]
    pub fn par_iter_obs(&self) -> impl rayon::iter::ParallelIterator<Item = (usize, NdArrayV<T>)> + '_ {
        use rayon::prelude::*;
        let n_obs = self.dims.shape()[0];
        (0..n_obs).into_par_iter().map(move |i| (i, self.obs(i)))
    }
}

// *** f64 conversions to the Array/Table/Matrix enum boundary *****

impl NdArray<f64> {
    /// Convert a 2D NdArray to a Table.
    ///
    /// Each column is copied into its own 64-byte aligned `FloatArray<f64>`,
    /// since the compact tensor layout does not place column starts on
    /// alignment boundaries. `fields` must have exactly `n_cols` entries.
    pub fn to_table(self, fields: Vec<Field>) -> Result<Table, MinarrowError> {
        let shape = self.dims.shape();
        if shape.len() != 2 {
            return Err(MinarrowError::ShapeError {
                message: format!("to_table requires a 2D array, got {}D", shape.len()),
            });
        }
        let n_rows = shape[0];
        let n_cols = shape[1];
        if fields.len() != n_cols {
            return Err(MinarrowError::ShapeError {
                message: format!(
                    "to_table: expected {} fields for {} columns, got {}",
                    n_cols, n_cols, fields.len()
                ),
            });
        }
        let stride = self.dims.strides()[1];
        let name = self.name;
        let buf = self.data.as_slice();

        let mut cols = Vec::with_capacity(n_cols);
        for (i, field) in fields.into_iter().enumerate() {
            let col_start = i * stride;
            let col: Buffer<f64> = Buffer::from_slice(&buf[col_start..col_start + n_rows]);
            let float_arr = FloatArray::new(col, None);
            let array = Array::NumericArray(NumericArray::Float64(Arc::new(float_arr)));
            cols.push(FieldArray::new(field, array));
        }

        Ok(Table::new(name.unwrap_or_default(), Some(cols)))
    }

    /// Convert a 2D NdArray to a Table with auto-generated column names.
    pub fn to_table_gen(self) -> Result<Table, MinarrowError> {
        let n_cols = if self.ndim() == 2 {
            self.dims.shape()[1]
        } else {
            return Err(MinarrowError::ShapeError {
                message: format!("to_table_gen requires a 2D array, got {}D", self.ndim()),
            });
        };
        let fields: Vec<Field> = (0..n_cols)
            .map(|i| Field::new(
                format!("col_{}", i),
                ArrowType::Float64,
                false,
                None,
            ))
            .collect();
        self.to_table(fields)
    }

    /// Convert a 1D NdArray to an Array (FloatArray<f64>).
    pub fn to_array(self) -> Result<Array, MinarrowError> {
        if self.ndim() != 1 {
            return Err(MinarrowError::ShapeError {
                message: format!("to_array requires a 1D array, got {}D", self.ndim()),
            });
        }
        let buffer = Arc::try_unwrap(self.data).unwrap_or_else(|arc| (*arc).clone());
        let float_arr = FloatArray::new(buffer, None);
        Ok(Array::NumericArray(NumericArray::Float64(Arc::new(float_arr))))
    }

    /// Convert a 2D NdArray to a Matrix.
    ///
    /// Matrix pads each column to a 64-byte boundary for BLAS/LAPACK and
    /// SIMD access, so compact tensor data is re-laid out into the padded
    /// form. An array whose stride already matches the padded layout, such
    /// as one built from a Matrix, moves across zero-copy.
    #[cfg(feature = "matrix")]
    pub fn to_matrix(self) -> Result<Matrix, MinarrowError> {
        let shape = self.dims.shape();
        if shape.len() != 2 {
            return Err(MinarrowError::ShapeError {
                message: format!("to_matrix requires a 2D array, got {}D", shape.len()),
            });
        }
        let n_rows = shape[0];
        let n_cols = shape[1];
        let strides = self.dims.strides();

        if strides[0] == 1 && strides[1] == aligned_stride(n_rows) {
            return Ok(Matrix {
                n_rows,
                n_cols,
                stride: strides[1],
                data: Arc::try_unwrap(self.data).unwrap_or_else(|arc| (*arc).clone()),
                name: self.name,
            });
        }

        let name = self.name.clone();
        let compact: Vec64<f64> = self.into_iter().collect();
        Ok(Matrix::from_f64_unaligned(&compact, n_rows, n_cols, name))
    }
}

// ****************************************************************
// NdDims - internal dimension storage
// ****************************************************************

/// Internal storage for shape and strides.
///
/// Inline arrays for 1D-5D to avoid heap allocation; the `Dn` variant
/// handles 6+ dimensions via boxed slices.
#[derive(Clone, PartialEq)]
pub(crate) enum NdDims {
    D1 { shape: [usize; 1], strides: [usize; 1] },
    D2 { shape: [usize; 2], strides: [usize; 2] },
    D3 { shape: [usize; 3], strides: [usize; 3] },
    D4 { shape: [usize; 4], strides: [usize; 4] },
    D5 { shape: [usize; 5], strides: [usize; 5] },
    Dn { shape: Box<[usize]>, strides: Box<[usize]> },
}

impl NdDims {
    /// Build dims from a shape slice, computing compact column-major
    /// strides.
    pub(crate) fn from_shape(shape: &[usize]) -> Self {
        let strides = col_major_strides(shape);
        Self::from_shape_and_strides(shape, &strides)
    }

    /// Build dims from explicit shape and strides.
    pub(crate) fn from_shape_and_strides(shape: &[usize], strides: &[usize]) -> Self {
        debug_assert_eq!(shape.len(), strides.len());
        match shape.len() {
            1 => NdDims::D1 {
                shape: [shape[0]],
                strides: [strides[0]],
            },
            2 => NdDims::D2 {
                shape: [shape[0], shape[1]],
                strides: [strides[0], strides[1]],
            },
            3 => NdDims::D3 {
                shape: [shape[0], shape[1], shape[2]],
                strides: [strides[0], strides[1], strides[2]],
            },
            4 => NdDims::D4 {
                shape: [shape[0], shape[1], shape[2], shape[3]],
                strides: [strides[0], strides[1], strides[2], strides[3]],
            },
            5 => NdDims::D5 {
                shape: [shape[0], shape[1], shape[2], shape[3], shape[4]],
                strides: [strides[0], strides[1], strides[2], strides[3], strides[4]],
            },
            _ => NdDims::Dn {
                shape: shape.into(),
                strides: strides.into(),
            },
        }
    }

    /// Number of dimensions.
    #[inline]
    pub(crate) fn ndim(&self) -> usize {
        match self {
            NdDims::D1 { .. } => 1,
            NdDims::D2 { .. } => 2,
            NdDims::D3 { .. } => 3,
            NdDims::D4 { .. } => 4,
            NdDims::D5 { .. } => 5,
            NdDims::Dn { shape, .. } => shape.len(),
        }
    }

    /// Shape as a slice.
    #[inline]
    pub(crate) fn shape(&self) -> &[usize] {
        match self {
            NdDims::D1 { shape, .. } => shape,
            NdDims::D2 { shape, .. } => shape,
            NdDims::D3 { shape, .. } => shape,
            NdDims::D4 { shape, .. } => shape,
            NdDims::D5 { shape, .. } => shape,
            NdDims::Dn { shape, .. } => shape,
        }
    }

    /// Strides as a slice.
    #[inline]
    pub(crate) fn strides(&self) -> &[usize] {
        match self {
            NdDims::D1 { strides, .. } => strides,
            NdDims::D2 { strides, .. } => strides,
            NdDims::D3 { strides, .. } => strides,
            NdDims::D4 { strides, .. } => strides,
            NdDims::D5 { strides, .. } => strides,
            NdDims::Dn { strides, .. } => strides,
        }
    }

    /// Total logical element count i.e. the product of all dimensions.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.shape().iter().product()
    }
}

// ****************************************************************
// Axis selection input
// ****************************************************************

/// Build an axis-selection slice from mixed indices and ranges.
///
/// Each entry is any [`DataSelector`](crate::DataSelector) - a single
/// index collapses the dimension, and a contiguous range keeps it.
///
/// # Example
/// ```ignore
/// arr.slice(nd![0..3, 2, 1..4])
/// ```
#[macro_export]
macro_rules! nd {
    ($($sel:expr),+ $(,)?) => {
        &[$(&$sel as &dyn $crate::traits::selection::DataSelector),+]
    };
}

// ****************************************************************
// Stride computation
// ****************************************************************

/// Compute compact column-major strides with no inter-dimension padding.
///
/// For a shape `[a, b, c]`, the strides are `[1, a, a * b]`. The buffer is
/// fully contiguous, so DLPack consumers receive a dense tensor and range
/// indexing over the outermost axis reads logical data with no gaps. The
/// backing allocation start remains 64-byte aligned through `Vec64`.
pub(crate) fn col_major_strides(shape: &[usize]) -> Vec<usize> {
    assert!(!shape.is_empty(), "NdArray: shape must have at least one dimension");
    let mut strides = Vec::with_capacity(shape.len());
    strides.push(1);
    for d in 1..shape.len() {
        strides.push(strides[d - 1] * shape[d - 1]);
    }
    strides
}

/// Total buffer length required for a given shape and strides.
pub(crate) fn buffer_len(shape: &[usize], strides: &[usize]) -> usize {
    if shape.is_empty() || shape.iter().any(|&d| d == 0) {
        return 0;
    }
    // The last element is at sum((shape[d]-1) * strides[d]) for all d.
    // Buffer must hold one past that.
    let max_offset: usize = shape.iter()
        .zip(strides.iter())
        .map(|(&s, &st)| (s - 1) * st)
        .sum();
    max_offset + 1
}

// ****************************************************************
// IntoIterator
// ****************************************************************

/// Iterating an NdArray yields f64 values in column-major order,
/// walking contiguous runs along axis 0 (the innermost dimension)
/// and advancing through higher dimensions. Each column/slice is
/// a sequential cache-friendly read with no per-element arithmetic.
impl<'a, T: Float> IntoIterator for &'a NdArray<T> {
    type Item = T;
    type IntoIter = NdArrayIter<'a, T>;

    #[inline]
    fn into_iter(self) -> NdArrayIter<'a, T> {
        let shape = self.dims.shape();
        let strides = self.dims.strides();
        let n_inner = shape[0];

        // Number of contiguous runs = product of all dims except axis 0
        let n_runs: usize = shape[1..].iter().product();

        // Build the starting offset of each contiguous run.
        // For 1D there is one run at offset 0.
        // For 2D these are just [0, stride1, 2*stride1, ...].
        // For N-D we walk the outer indices in column-major order.
        let mut run_offsets = Vec::with_capacity(n_runs);
        if shape.len() <= 1 {
            run_offsets.push(0);
        } else {
            let outer_shape = &shape[1..];
            let outer_strides = &strides[1..];
            let mut outer_indices = vec![0usize; outer_shape.len()];
            for _ in 0..n_runs {
                let off: usize = outer_indices.iter()
                    .zip(outer_strides.iter())
                    .map(|(&i, &s)| i * s)
                    .sum();
                run_offsets.push(off);
                // Advance outer indices (column-major)
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
            buf: self.data.as_slice(),
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

/// Consuming iterator - collects logical elements then iterates.
impl<T: Float> IntoIterator for NdArray<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    #[inline]
    fn into_iter(self) -> std::vec::IntoIter<T> {
        let v: Vec<T> = (&self).into_iter().collect();
        v.into_iter()
    }
}

/// Iterator over NdArray elements in column-major order.
///
/// When `inner_stride` is 1 (the normal case), walks contiguous runs
/// along axis 0 with sequential memory reads. When `inner_stride` > 1
/// (e.g. after collapsing axis 0 via slicing), steps through strided
/// elements within each run.
pub struct NdArrayIter<'a, T> {
    pub(crate) buf: &'a [T],
    pub(crate) n_inner: usize,
    pub(crate) inner_stride: usize,
    pub(crate) run_offsets: Vec<usize>,
    pub(crate) run_idx: usize,
    pub(crate) inner_idx: usize,
    pub(crate) total: usize,
    pub(crate) yielded: usize,
}

impl<'a, T: Float> Iterator for NdArrayIter<'a, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.yielded >= self.total { return None; }

        let val = self.buf[self.run_offsets[self.run_idx] + self.inner_idx * self.inner_stride];
        self.yielded += 1;
        self.inner_idx += 1;
        if self.inner_idx >= self.n_inner {
            self.inner_idx = 0;
            self.run_idx += 1;
        }
        Some(val)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.total - self.yielded;
        (r, Some(r))
    }
}

impl<'a, T: Float> ExactSizeIterator for NdArrayIter<'a, T> {}

// ****************************************************************
// Internal helpers
// ****************************************************************

/// Compute flat buffer offset for an N-dimensional index.
#[inline]
pub(crate) fn offset_of_impl(indices: &[usize], shape: &[usize], strides: &[usize]) -> usize {
    debug_assert_eq!(indices.len(), shape.len());
    let mut offset = 0;
    for d in 0..shape.len() {
        debug_assert!(
            indices[d] < shape[d],
            "NdArray: index {} out of bounds for dim {} (size {})",
            indices[d], d, shape[d]
        );
        offset += indices[d] * strides[d];
    }
    offset
}

// ****************************************************************
// Trait implementations
// ****************************************************************

impl<T: Float> Shape for NdArray<T> {
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

impl<T: Float> Concatenate for NdArray<T> {
    /// Concatenate along axis 0. All other dimensions must match.
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        let s1 = self.dims.shape();
        let s2 = other.dims.shape();
        if s1.len() != s2.len() {
            return Err(MinarrowError::IncompatibleTypeError {
                from: "NdArray",
                to: "NdArray",
                message: Some(format!(
                    "Cannot concatenate {}D and {}D arrays", s1.len(), s2.len()
                )),
            });
        }
        for d in 1..s1.len() {
            if s1[d] != s2[d] {
                return Err(MinarrowError::IncompatibleTypeError {
                    from: "NdArray",
                    to: "NdArray",
                    message: Some(format!(
                        "Dimension {} mismatch: {} vs {}", d, s1[d], s2[d]
                    )),
                });
            }
        }

        let mut new_shape: Vec<usize> = s1.to_vec();
        new_shape[0] += s2[0];

        // Fast path for 2D: interleave columns
        if s1.len() == 2 {
            let new_dims = NdDims::from_shape(&new_shape);
            let new_stride = new_dims.strides()[1];
            let total = buffer_len(&new_shape, new_dims.strides());
            let mut buf = Vec64::with_capacity(total);
            buf.0.resize(total, T::default());
            let dst = buf.as_mut_slice();
            for c in 0..s1[1] {
                let dst_start = c * new_stride;
                dst[dst_start..dst_start + s1[0]].copy_from_slice(self.col(c));
                dst[dst_start + s1[0]..dst_start + s1[0] + s2[0]].copy_from_slice(other.col(c));
            }
            return Ok(NdArray {
                data: Arc::new(Buffer::from_vec64(buf)),
                dims: new_dims,
                name: None,
            });
        }

        // General case: collect logical elements and re-lay out
        let mut flat = Vec64::with_capacity(self.len() + other.len());
        flat.extend(&self);
        flat.extend(&other);
        Ok(NdArray::from_slice(&flat, &new_shape))
    }
}

// *** Axis selection: arr.s((1..4, 2)) ****************************

/// Selection across every axis at once, delegating to `slice`. Single
/// indices collapse their dimension, and contiguous ranges keep it.
/// Zero-copy.
#[cfg(all(feature = "views", feature = "select"))]
impl<T: Float> AxisSelection for NdArray<T> {
    type View = NdArrayV<T>;

    fn s(&self, selection: &[&dyn DataSelector]) -> NdArrayV<T> {
        self.slice(selection)
    }

    fn get_axis_count(&self) -> usize {
        self.ndim()
    }
}

// *** Row selection: arr.r(0..10) *********************************

/// Axis-0 observation selection. Contiguous ranges return a zero-copy
/// window view. Index arrays gather the selected observations into an
/// owned array wrapped in a full view, matching `Table`'s behaviour.
#[cfg(all(feature = "views", feature = "select"))]
impl<T: Float> RowSelection for NdArray<T> {
    type View = NdArrayV<T>;

    fn r<S: DataSelector>(&self, selection: S) -> NdArrayV<T> {
        let n_obs = self.shape()[0];
        let indices = selection.resolve_indices(n_obs);
        if selection.is_contiguous() {
            let start = indices.first().copied().unwrap_or(0);
            let ranges: Vec<Range<usize>> = std::iter::once(start..start + indices.len())
                .chain(self.shape()[1..].iter().map(|&n| 0..n))
                .collect();
            let refs: Vec<&dyn DataSelector> = ranges.iter().map(|r| r as _).collect();
            return self.slice(&refs);
        }
        NdArrayV::from_ndarray(gather_obs_impl(
            &indices,
            self.shape(),
            self.name.clone(),
            |idx| self.get(idx),
        ))
    }

    fn get_row_count(&self) -> usize {
        self.shape()[0]
    }
}

/// Materialise selected axis-0 observations into a compact owned array.
/// Walks the output positions in column-major order, reading each source
/// element through the provided accessor, so any stride layout gathers
/// correctly.
#[cfg(all(feature = "views", feature = "select"))]
pub(crate) fn gather_obs_impl<T: Float>(
    indices: &[usize],
    shape: &[usize],
    name: Option<String>,
    get: impl Fn(&[usize]) -> T,
) -> NdArray<T> {
    let mut out_shape = shape.to_vec();
    out_shape[0] = indices.len();
    let total: usize = out_shape.iter().product();

    let mut flat = Vec64::with_capacity(total);
    let ndim = shape.len();
    let mut idx = vec![0usize; ndim];
    let inner_runs: usize = shape[1..].iter().product::<usize>().max(1);
    for _ in 0..inner_runs {
        for &obs in indices {
            idx[0] = obs;
            flat.push(get(&idx));
        }
        // Advance the inner multi-index in column-major order.
        let mut carry = true;
        for d in 1..ndim {
            if carry {
                idx[d] += 1;
                if idx[d] < shape[d] {
                    carry = false;
                } else {
                    idx[d] = 0;
                }
            }
        }
    }
    let mut result = NdArray::from_slice(&flat, &out_shape);
    result.name = name;
    result
}

// *** Bracket indexing: arr[col][row] ******************************

/// `arr[i]` selects along the outermost stored axis.
///
/// For 1D, returns a single-element slice.
/// For 2D (column-major), `arr[col]` returns the contiguous column
/// as `&[f64]`, so `arr[col][row]` gives `&f64`.
impl<T: Float> Index<usize> for NdArray<T> {
    type Output = [T];

    #[inline]
    fn index(&self, idx: usize) -> &[T] {
        let shape = self.dims.shape();
        let strides = self.dims.strides();
        match shape.len() {
            1 => {
                debug_assert!(idx < shape[0], "index out of bounds");
                &self.data.as_slice()[idx..idx + 1]
            }
            2 => {
                debug_assert!(idx < shape[1], "column index out of bounds");
                let start = idx * strides[1];
                &self.data.as_slice()[start..start + shape[0]]
            }
            n => {
                // Index the outermost axis (last), return the contiguous inner slab
                assert!(
                    self.is_contiguous(),
                    "outermost-axis indexing on 3D+ requires a contiguous layout, use slice() for strided access"
                );
                let last = n - 1;
                debug_assert!(idx < shape[last], "index out of bounds for axis {}", last);
                let start = idx * strides[last];
                &self.data.as_slice()[start..start + strides[last]]
            }
        }
    }
}

impl<T: Float> IndexMut<usize> for NdArray<T> {
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut [T] {
        let shape = self.dims.shape().to_vec();
        let strides = self.dims.strides().to_vec();
        match shape.len() {
            1 => {
                debug_assert!(idx < shape[0], "index out of bounds");
                &mut Arc::make_mut(&mut self.data).as_mut_slice()[idx..idx + 1]
            }
            2 => {
                debug_assert!(idx < shape[1], "column index out of bounds");
                let start = idx * strides[1];
                let n_rows = shape[0];
                &mut Arc::make_mut(&mut self.data).as_mut_slice()[start..start + n_rows]
            }
            n => {
                assert!(
                    self.is_contiguous(),
                    "outermost-axis indexing on 3D+ requires a contiguous layout, use slice() for strided access"
                );
                let last = n - 1;
                debug_assert!(idx < shape[last], "index out of bounds for axis {}", last);
                let start = idx * strides[last];
                &mut Arc::make_mut(&mut self.data).as_mut_slice()[start..start + strides[last]]
            }
        }
    }
}

// *** Range indexing: arr[1..4] ************************************

/// `arr[start..end]` selects a contiguous range along the outermost axis.
///
/// For 1D, returns the element slice directly.
/// For 2D and above, selects a range of outermost-axis entries as one
/// contiguous slab of logical data. Requires a contiguous layout, since a
/// padded or transposed stride pattern has no gap-free slab to return.
/// Non-contiguous arrays panic with guidance to use `slice()`.
impl<T: Float> Index<Range<usize>> for NdArray<T> {
    type Output = [T];

    #[inline]
    fn index(&self, range: Range<usize>) -> &[T] {
        let shape = self.dims.shape();
        let strides = self.dims.strides();
        match shape.len() {
            1 => &self.data.as_slice()[range],
            _ => {
                assert!(
                    self.is_contiguous(),
                    "range indexing requires a contiguous layout, use slice() for strided access"
                );
                let last = shape.len() - 1;
                debug_assert!(range.end <= shape[last], "range end out of bounds");
                let start = range.start * strides[last];
                let end = range.end * strides[last];
                &self.data.as_slice()[start..end]
            }
        }
    }
}

impl<T: Float> IndexMut<Range<usize>> for NdArray<T> {
    #[inline]
    fn index_mut(&mut self, range: Range<usize>) -> &mut [T] {
        let shape = self.dims.shape().to_vec();
        let strides = self.dims.strides().to_vec();
        match shape.len() {
            1 => &mut Arc::make_mut(&mut self.data).as_mut_slice()[range],
            _ => {
                assert!(
                    self.is_contiguous(),
                    "range indexing requires a contiguous layout, use slice() for strided access"
                );
                let last = shape.len() - 1;
                debug_assert!(range.end <= shape[last], "range end out of bounds");
                let start = range.start * strides[last];
                let end = range.end * strides[last];
                &mut Arc::make_mut(&mut self.data).as_mut_slice()[start..end]
            }
        }
    }
}

impl<T: Float> Index<RangeFrom<usize>> for NdArray<T> {
    type Output = [T];

    #[inline]
    fn index(&self, range: RangeFrom<usize>) -> &[T] {
        let last_dim = self.dims.shape().len() - 1;
        let end = self.dims.shape()[last_dim];
        &self[range.start..end]
    }
}

impl<T: Float> Index<RangeTo<usize>> for NdArray<T> {
    type Output = [T];

    #[inline]
    fn index(&self, range: RangeTo<usize>) -> &[T] {
        &self[0..range.end]
    }
}

impl<T: Float> Index<RangeFull> for NdArray<T> {
    type Output = [T];

    #[inline]
    fn index(&self, _: RangeFull) -> &[T] {
        let last_dim = self.dims.shape().len() - 1;
        let end = self.dims.shape()[last_dim.max(0)];
        &self[0..end]
    }
}

// *** Tuple indexing **********************************************

impl<T: Float> Index<(usize,)> for NdArray<T> {
    type Output = T;
    #[inline]
    fn index(&self, (i,): (usize,)) -> &T {
        &self.data.as_slice()[self.offset_of(&[i])]
    }
}

impl<T: Float> Index<(usize, usize)> for NdArray<T> {
    type Output = T;
    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &T {
        &self.data.as_slice()[self.offset_of(&[i, j])]
    }
}

impl<T: Float> Index<(usize, usize, usize)> for NdArray<T> {
    type Output = T;
    #[inline]
    fn index(&self, (i, j, k): (usize, usize, usize)) -> &T {
        &self.data.as_slice()[self.offset_of(&[i, j, k])]
    }
}

impl<T: Float> Index<(usize, usize, usize, usize)> for NdArray<T> {
    type Output = T;
    #[inline]
    fn index(&self, (i, j, k, l): (usize, usize, usize, usize)) -> &T {
        &self.data.as_slice()[self.offset_of(&[i, j, k, l])]
    }
}

impl<T: Float> Index<(usize, usize, usize, usize, usize)> for NdArray<T> {
    type Output = T;
    #[inline]
    fn index(&self, (i, j, k, l, m): (usize, usize, usize, usize, usize)) -> &T {
        &self.data.as_slice()[self.offset_of(&[i, j, k, l, m])]
    }
}

impl<T: Float> IndexMut<(usize,)> for NdArray<T> {
    #[inline]
    fn index_mut(&mut self, (i,): (usize,)) -> &mut T {
        let off = self.offset_of(&[i]);
        &mut Arc::make_mut(&mut self.data).as_mut_slice()[off]
    }
}

impl<T: Float> IndexMut<(usize, usize)> for NdArray<T> {
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        let off = self.offset_of(&[i, j]);
        &mut Arc::make_mut(&mut self.data).as_mut_slice()[off]
    }
}

impl<T: Float> IndexMut<(usize, usize, usize)> for NdArray<T> {
    #[inline]
    fn index_mut(&mut self, (i, j, k): (usize, usize, usize)) -> &mut T {
        let off = self.offset_of(&[i, j, k]);
        &mut Arc::make_mut(&mut self.data).as_mut_slice()[off]
    }
}

impl<T: Float> IndexMut<(usize, usize, usize, usize)> for NdArray<T> {
    #[inline]
    fn index_mut(&mut self, (i, j, k, l): (usize, usize, usize, usize)) -> &mut T {
        let off = self.offset_of(&[i, j, k, l]);
        &mut Arc::make_mut(&mut self.data).as_mut_slice()[off]
    }
}

impl<T: Float> IndexMut<(usize, usize, usize, usize, usize)> for NdArray<T> {
    #[inline]
    fn index_mut(&mut self, (i, j, k, l, m): (usize, usize, usize, usize, usize)) -> &mut T {
        let off = self.offset_of(&[i, j, k, l, m]);
        &mut Arc::make_mut(&mut self.data).as_mut_slice()[off]
    }
}

// *** From conversions ********************************************

/// 1D from a flat slice.
impl<T: Float> From<&[T]> for NdArray<T> {
    fn from(data: &[T]) -> Self {
        NdArray {
            data: Arc::new(Buffer::from_slice(data)),
            dims: NdDims::from_shape(&[data.len()]),
            name: None,
        }
    }
}

/// 1D from owned Vec64.
impl<T: Float> From<Vec64<T>> for NdArray<T> {
    fn from(v: Vec64<T>) -> Self {
        let n = v.len();
        NdArray {
            data: Arc::new(Buffer::from_vec64(v)),
            dims: NdDims::from_shape(&[n]),
            name: None,
        }
    }
}

/// 2D from column vectors.
impl<T: Float> From<&[Vec<T>]> for NdArray<T> {
    fn from(columns: &[Vec<T>]) -> Self {
        let n_cols = columns.len();
        if n_cols == 0 {
            return NdArray::new(&[0, 0]);
        }
        let n_rows = columns[0].len();
        for col in columns {
            assert_eq!(col.len(), n_rows, "Column length mismatch");
        }
        let shape = [n_rows, n_cols];
        let dims = NdDims::from_shape(&shape);
        let stride = dims.strides()[1];
        let total = buffer_len(&shape, dims.strides());
        let mut buf = Vec64::with_capacity(total);
        buf.0.resize(total, T::default());
        for (c, col) in columns.iter().enumerate() {
            let start = c * stride;
            buf.as_mut_slice()[start..start + n_rows].copy_from_slice(col);
        }
        NdArray { data: Arc::new(Buffer::from_vec64(buf)), dims, name: None }
    }
}

/// 2D from FloatArray columns.
impl<T: Float> From<&[FloatArray<T>]> for NdArray<T> {
    fn from(columns: &[FloatArray<T>]) -> Self {
        let n_cols = columns.len();
        if n_cols == 0 {
            return NdArray::new(&[0, 0]);
        }
        let n_rows = columns[0].data.len();
        for col in columns {
            assert_eq!(col.data.len(), n_rows, "Column length mismatch");
        }
        let shape = [n_rows, n_cols];
        let dims = NdDims::from_shape(&shape);
        let stride = dims.strides()[1];
        let total = buffer_len(&shape, dims.strides());
        let mut buf = Vec64::with_capacity(total);
        buf.0.resize(total, T::default());
        for (c, col) in columns.iter().enumerate() {
            let start = c * stride;
            buf.as_mut_slice()[start..start + n_rows].copy_from_slice(col.data.as_slice());
        }
        NdArray { data: Arc::new(Buffer::from_vec64(buf)), dims, name: None }
    }
}

/// From Matrix - zero-copy, moves the Buffer straight across. The Matrix's
/// padded column stride carries through, so the resulting array reports
/// non-contiguous. Call `to_contiguous` to re-lay out compactly.
#[cfg(feature = "matrix")]
impl From<Matrix> for NdArray<f64> {
    fn from(mat: Matrix) -> Self {
        let shape = [mat.n_rows, mat.n_cols];
        let strides = [1, mat.stride];
        NdArray {
            data: Arc::new(mat.data),
            dims: NdDims::from_shape_and_strides(&shape, &strides),
            name: mat.name,
        }
    }
}

/// TryFrom Table - extracts numeric columns, converts nulls to NaN.
impl TryFrom<&Table> for NdArray<f64> {
    type Error = MinarrowError;

    fn try_from(table: &Table) -> Result<Self, Self::Error> {
        let n_cols = table.n_cols();
        let n_rows = table.n_rows;
        if n_cols == 0 {
            return Ok(NdArray::new(&[0, 0]));
        }

        let shape = [n_rows, n_cols];
        let dims = NdDims::from_shape(&shape);
        let stride = dims.strides()[1];
        let total = buffer_len(&shape, dims.strides());
        let mut buf = Vec64::with_capacity(total);
        buf.0.resize(total, 0.0);

        for (col_idx, fa) in table.cols.iter().enumerate() {
            let numeric = fa.array.try_num().map_err(|_| MinarrowError::TypeError {
                from: "non-numeric",
                to: "Float64",
                message: Some(format!("column {} is not numeric", col_idx)),
            })?;
            let f64_arr = numeric.try_f64()?;
            if f64_arr.data.len() != n_rows {
                return Err(MinarrowError::ColumnLengthMismatch {
                    col: col_idx,
                    expected: n_rows,
                    found: f64_arr.data.len(),
                });
            }

            let start = col_idx * stride;
            let src = f64_arr.data.as_slice();
            let dst = &mut buf.as_mut_slice()[start..start + n_rows];

            // Copy data, converting nulls to NaN
            match f64_arr.null_mask.as_ref() {
                Some(mask) => {
                    for i in 0..n_rows {
                        dst[i] = if mask.get(i) { src[i] } else { f64::NAN };
                    }
                }
                None => dst.copy_from_slice(src),
            }
        }

        let name = if table.name.is_empty() { None } else { Some(table.name.clone()) };
        Ok(NdArray { data: Arc::new(Buffer::from_vec64(buf)), dims, name })
    }
}

impl TryFrom<Table> for NdArray<f64> {
    type Error = MinarrowError;
    fn try_from(table: Table) -> Result<Self, Self::Error> {
        NdArray::try_from(&table)
    }
}

// *** Debug *******************************************************

impl<T: Float> fmt::Debug for NdArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "NdArray{}: {:?} [{}D, col-major]",
            self.name.as_deref().map_or(String::new(), |n| format!(" '{}'", n)),
            self.dims.shape(),
            self.ndim(),
        )?;
        if self.ndim() == 2 {
            let shape = self.dims.shape();
            let max_rows = shape[0].min(6);
            let max_cols = shape[1].min(8);
            for r in 0..max_rows {
                write!(f, "\n[")?;
                for c in 0..max_cols {
                    write!(f, " {:8.4}", self.get(&[r, c]).to_f64().unwrap_or(f64::NAN))?;
                    if c < max_cols - 1 { write!(f, ",")?; }
                }
                if shape[1] > 8 { write!(f, " ...")?; }
                write!(f, " ]")?;
            }
            if shape[0] > 6 { write!(f, "\n...")?; }
        } else if self.ndim() == 1 {
            let n = self.dims.shape()[0].min(10);
            write!(f, "\n[")?;
            for i in 0..n {
                write!(f, " {:8.4}", self.get(&[i]).to_f64().unwrap_or(f64::NAN))?;
                if i < n - 1 { write!(f, ",")?; }
            }
            if self.dims.shape()[0] > 10 { write!(f, " ...")?; }
            write!(f, " ]")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StringArray;
    use crate::structs::bitmask::Bitmask;

    // *** Row selection and apply *************************************

    #[cfg(all(feature = "views", feature = "select"))]
    #[test]
    fn axis_selection_rank10() {
        let shape = [3, 4, 2, 5, 3, 2, 4, 3, 2, 5];
        let len: usize = shape.iter().product();
        let data: Vec<f64> = (0..len).map(|i| i as f64).collect();
        let a = NdArray::from_slice(&data, &shape);

        // Mixed ranges, single indices, and a full range across all ten axes.
        let v = a.s(nd![1..3, 0..2, 1, 2..5, .., 0..1, 1..3, 2, 0..2, 3..5]);
        assert_eq!(v.shape(), &[2, 2, 3, 3, 1, 2, 2, 2]);
        assert_eq!(
            v.get(&[0, 0, 0, 0, 0, 0, 0, 0]),
            a.get(&[1, 0, 1, 2, 0, 0, 1, 2, 0, 3])
        );
        assert_eq!(
            v.get(&[1, 1, 2, 2, 0, 1, 1, 1]),
            a.get(&[2, 1, 1, 4, 2, 0, 2, 2, 1, 4])
        );
    }

    #[cfg(all(feature = "views", feature = "select"))]
    #[test]
    fn axis_selection_runtime_rank() {
        // Selections built at runtime for ranks beyond literal syntax.
        let mut shape = vec![1usize; 100];
        shape[0] = 3;
        shape[10] = 4;
        shape[50] = 5;
        shape[99] = 2;
        let len: usize = shape.iter().product();
        let data: Vec<f64> = (0..len).map(|i| i as f64).collect();
        let a = NdArray::from_slice(&data, &shape);

        // Full range on every axis, then narrow three and collapse one.
        let mut sels: Vec<Box<dyn DataSelector>> =
            shape.iter().map(|&n| Box::new(0..n) as Box<dyn DataSelector>).collect();
        sels[0] = Box::new(1..3);
        sels[10] = Box::new(2usize);
        sels[50] = Box::new(1..4);
        let refs: Vec<&dyn DataSelector> = sels.iter().map(|s| s.as_ref()).collect();

        let v = a.s(&refs);
        assert_eq!(v.ndim(), 99);
        assert_eq!(v.shape()[0], 2);
        assert_eq!(v.shape()[49], 3);
        assert_eq!(v.shape()[98], 2);

        let mut view_idx = vec![0usize; 99];
        view_idx[0] = 1;
        view_idx[49] = 2;
        view_idx[98] = 1;
        let mut source_idx = vec![0usize; 100];
        source_idx[0] = 2;
        source_idx[10] = 2;
        source_idx[50] = 3;
        source_idx[99] = 1;
        assert_eq!(v.get(&view_idx), a.get(&source_idx));
    }

    #[cfg(all(feature = "views", feature = "select"))]
    #[test]
    fn axis_selection_trait() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        // Range keeps the axis, index collapses it.
        let v = a.s(nd![1..3, 1]);
        assert_eq!(v.shape(), &[2]);
        assert_eq!(v.get(&[0]), 5.0);
        assert_eq!(v.get(&[1]), 6.0);
        // Selection composes on the view.
        let sub = a.s(nd![0..3, 0..2]).s(nd![2, 0..2]);
        assert_eq!(sub.shape(), &[2]);
        assert_eq!(sub.get(&[0]), 3.0);
        assert_eq!(sub.get(&[1]), 6.0);
        assert_eq!(a.get_axis_count(), 2);
        assert_eq!(sub.get_axis_count(), 1);
    }

    #[cfg(all(feature = "views", feature = "select"))]
    #[test]
    fn row_selection_contiguous() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let v = a.r(1..3);
        assert_eq!(v.shape(), &[2, 2]);
        assert_eq!(v.get(&[0, 0]), 2.0);
        assert_eq!(v.get(&[1, 1]), 6.0);
        // The row alias selects a single observation.
        let single = a.row(2);
        assert_eq!(single.shape(), &[1, 2]);
        assert_eq!(single.get(&[0, 1]), 6.0);
    }

    #[cfg(all(feature = "views", feature = "select"))]
    #[test]
    fn row_selection_gathers_indices() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let v = a.r(&[2, 0]);
        assert_eq!(v.shape(), &[2, 2]);
        // Gathered rows follow selection order.
        assert_eq!(v.get(&[0, 0]), 3.0);
        assert_eq!(v.get(&[0, 1]), 6.0);
        assert_eq!(v.get(&[1, 0]), 1.0);
        assert_eq!(v.get(&[1, 1]), 4.0);
    }

    #[test]
    fn apply_maps_elements() {
        let mut a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        a.set_name("m");
        let b = a.apply(|x| x * 10.0);
        assert_eq!(b.get(&[1, 1]), 40.0);
        assert_eq!(b.name.as_deref(), Some("m"));
        // The source is untouched.
        assert_eq!(a.get(&[1, 1]), 4.0);
    }

    #[test]
    fn apply_mut_in_place() {
        let mut a = NdArray::from_slice(&[1.0, 2.0, 3.0], &[3]);
        a.apply_mut(|x| x + 0.5);
        assert_eq!((&a).into_iter().collect::<Vec<f64>>(), vec![1.5, 2.5, 3.5]);
    }

    #[cfg(feature = "matrix")]
    #[test]
    fn apply_mut_non_contiguous_touches_logical_only() {
        // A Matrix-imported array carries stride padding. The logical walk
        // mutates only real elements.
        let mat = Matrix::from_f64_unaligned(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2, None);
        let mut a = NdArray::from(mat);
        assert!(!a.is_contiguous());
        a.apply_mut(|x| x * 2.0);
        assert_eq!(a.get(&[0, 0]), 2.0);
        assert_eq!(a.get(&[2, 1]), 12.0);
    }

    #[cfg(feature = "views")]
    #[test]
    fn apply_axis_collapses_axis() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        // Sum each column lane (axis 0) - output shape [2].
        let col_sums = a.apply_axis(0, |lane| (&lane).into_iter().sum());
        assert_eq!(col_sums.shape(), &[2]);
        assert_eq!(col_sums.get(&[0]), 6.0);
        assert_eq!(col_sums.get(&[1]), 15.0);
        // Sum each row lane (axis 1) - output shape [3].
        let row_sums = a.apply_axis(1, |lane| (&lane).into_iter().sum());
        assert_eq!(row_sums.shape(), &[3]);
        assert_eq!(row_sums.get(&[0]), 5.0);
        assert_eq!(row_sums.get(&[2]), 9.0);
    }

    #[cfg(feature = "views")]
    #[test]
    fn apply_axis_3d() {
        let data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
        let a = NdArray::from_slice(&data, &[2, 3, 4]);
        let maxes = a.apply_axis(1, |lane| {
            (&lane).into_iter().fold(f64::MIN, f64::max)
        });
        assert_eq!(maxes.shape(), &[2, 4]);
        // Lane over axis 1 at [0, .., 0] holds values at [0,j,0].
        assert_eq!(maxes.get(&[0, 0]), a.get(&[0, 2, 0]));
        assert_eq!(maxes.get(&[1, 3]), a.get(&[1, 2, 3]));
    }

    // ****************************************************************
    // Construction
    // ****************************************************************

    #[test]
    fn new_zeroed_1d() {
        let a = NdArray::<f64>::new(&[5]);
        assert_eq!(a.ndim(), 1);
        assert_eq!(a.shape(), &[5]);
        assert_eq!(a.len(), 5);
        assert!(!a.is_empty());
        for v in &a { assert_eq!(v, 0.0); }
    }

    #[test]
    fn new_zeroed_2d() {
        let a = NdArray::<f64>::new(&[3, 4]);
        assert_eq!(a.ndim(), 2);
        assert_eq!(a.shape(), &[3, 4]);
        assert_eq!(a.len(), 12);
        for v in &a { assert_eq!(v, 0.0); }
    }

    #[test]
    fn new_zeroed_3d() {
        let a = NdArray::<f64>::new(&[2, 3, 4]);
        assert_eq!(a.ndim(), 3);
        assert_eq!(a.shape(), &[2, 3, 4]);
        assert_eq!(a.len(), 24);
    }

    #[test]
    fn new_zeroed_5d() {
        let a = NdArray::<f64>::new(&[2, 3, 4, 5, 6]);
        assert_eq!(a.ndim(), 5);
        assert_eq!(a.len(), 720);
    }

    #[test]
    fn new_zeroed_6d_heap() {
        let a = NdArray::<f64>::new(&[2, 3, 2, 2, 2, 2]);
        assert_eq!(a.ndim(), 6);
        assert_eq!(a.len(), 96);
    }

    #[test]
    fn new_named() {
        let a = NdArray::<f64>::new_named(&[3, 3], "covariance");
        assert_eq!(a.name.as_deref(), Some("covariance"));
    }

    #[test]
    fn from_slice_1d() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let a = NdArray::from_slice(&data, &[5]);
        assert_eq!(a.len(), 5);
        let vals: Vec<f64> = (&a).into_iter().collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn from_slice_2d_column_major() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = NdArray::from_slice(&data, &[3, 2]);
        assert_eq!(a.get(&[0, 0]), 1.0);
        assert_eq!(a.get(&[1, 0]), 2.0);
        assert_eq!(a.get(&[2, 0]), 3.0);
        assert_eq!(a.get(&[0, 1]), 4.0);
        assert_eq!(a.get(&[1, 1]), 5.0);
        assert_eq!(a.get(&[2, 1]), 6.0);
    }

    #[test]
    fn fill_and_ones() {
        let a = NdArray::fill(&[2, 3], 7.0);
        assert_eq!(a.len(), 6);
        for v in &a { assert_eq!(v, 7.0); }

        let b = NdArray::<f64>::ones(&[4]);
        for v in &b { assert_eq!(v, 1.0); }
    }

    #[test]
    fn eye_identity() {
        let a = NdArray::<f64>::eye(3);
        assert_eq!(a.shape(), &[3, 3]);
        assert_eq!(a.get(&[0, 0]), 1.0);
        assert_eq!(a.get(&[1, 1]), 1.0);
        assert_eq!(a.get(&[2, 2]), 1.0);
        assert_eq!(a.get(&[0, 1]), 0.0);
        assert_eq!(a.get(&[1, 0]), 0.0);
    }

    #[test]
    fn linspace_basic() {
        let a = NdArray::<f64>::linspace(0.0, 1.0, 5);
        assert_eq!(a.shape(), &[5]);
        assert_eq!(a.get(&[0]), 0.0);
        assert_eq!(a.get(&[4]), 1.0);
        assert!((a.get(&[2]) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn arange_basic() {
        let a = NdArray::arange(0.0, 0.5, 4);
        assert_eq!(a.shape(), &[4]);
        assert_eq!(a.get(&[0]), 0.0);
        assert_eq!(a.get(&[1]), 0.5);
        assert_eq!(a.get(&[2]), 1.0);
        assert_eq!(a.get(&[3]), 1.5);
    }

    // ****************************************************************
    // Element access and indexing
    // ****************************************************************

    #[test]
    fn get_set_1d() {
        let mut a = NdArray::new(&[3]);
        a.set(&[0], 10.0);
        a.set(&[1], 20.0);
        a.set(&[2], 30.0);
        assert_eq!(a.get(&[0]), 10.0);
        assert_eq!(a.get(&[1]), 20.0);
        assert_eq!(a.get(&[2]), 30.0);
    }

    #[test]
    fn tuple_index_1d() {
        let a = NdArray::from_slice(&[10.0, 20.0, 30.0], &[3]);
        assert_eq!(a[(0,)], 10.0);
        assert_eq!(a[(1,)], 20.0);
        assert_eq!(a[(2,)], 30.0);
    }

    #[test]
    fn tuple_index_2d() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = NdArray::from_slice(&data, &[3, 2]);
        assert_eq!(a[(0, 0)], 1.0);
        assert_eq!(a[(2, 0)], 3.0);
        assert_eq!(a[(0, 1)], 4.0);
        assert_eq!(a[(2, 1)], 6.0);
    }

    #[test]
    fn tuple_index_3d() {
        let data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let a = NdArray::from_slice(&data, &[2, 3, 2]);
        assert_eq!(a[(0, 0, 0)], 1.0);
        assert_eq!(a[(1, 0, 0)], 2.0);
        assert_eq!(a[(0, 1, 0)], 3.0);
        assert_eq!(a[(1, 1, 0)], 4.0);
        assert_eq!(a[(0, 2, 0)], 5.0);
        assert_eq!(a[(1, 2, 0)], 6.0);
        assert_eq!(a[(0, 0, 1)], 7.0);
        assert_eq!(a[(1, 2, 1)], 12.0);
    }

    #[test]
    fn index_mut_2d() {
        let mut a = NdArray::new(&[2, 2]);
        a[(0, 0)] = 1.0;
        a[(1, 0)] = 2.0;
        a[(0, 1)] = 3.0;
        a[(1, 1)] = 4.0;
        assert_eq!(a[(0, 0)], 1.0);
        assert_eq!(a[(1, 1)], 4.0);
    }

    // ****************************************************************
    // Iteration
    // ****************************************************************

    #[test]
    fn iter_1d() {
        let a = NdArray::from_slice(&[10.0, 20.0, 30.0], &[3]);
        let vals: Vec<f64> = (&a).into_iter().collect();
        assert_eq!(vals, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn iter_2d_column_major_order() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let vals: Vec<f64> = (&a).into_iter().collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn iter_2d_compact_layout() {
        // Compact strides place columns back to back, so iteration reads
        // the buffer straight through.
        let data: Vec<f64> = (1..=30).map(|x| x as f64).collect();
        let a = NdArray::from_slice(&data, &[10, 3]);
        assert_eq!(a.strides()[1], 10);
        let vals: Vec<f64> = (&a).into_iter().collect();
        assert_eq!(vals.len(), 30);
        assert_eq!(&vals[..10], &data[..10]);
        assert_eq!(&vals[10..20], &data[10..20]);
        assert_eq!(&vals[20..30], &data[20..30]);
    }

    #[test]
    fn iter_3d_column_major_order() {
        let data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
        let a = NdArray::from_slice(&data, &[2, 3, 4]);
        let vals: Vec<f64> = (&a).into_iter().collect();
        assert_eq!(vals.len(), 24);
        assert_eq!(vals, data);
    }

    #[test]
    fn iter_exact_size() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let iter = (&a).into_iter();
        assert_eq!(iter.len(), 6);
    }

    #[test]
    fn consuming_into_iter() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let vals: Vec<f64> = a.into_iter().collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    // ****************************************************************
    // Shape introspection
    // ****************************************************************

    #[test]
    fn is_contiguous_default() {
        let a = NdArray::<f64>::new(&[3, 4]);
        assert!(a.is_contiguous());
    }

    #[test]
    fn shape_trait_1d() {
        let a = NdArray::<f64>::new(&[5]);
        assert_eq!(Shape::shape(&a), ShapeDim::Rank1(5));
    }

    #[test]
    fn shape_trait_2d() {
        let a = NdArray::<f64>::new(&[3, 4]);
        assert_eq!(Shape::shape(&a), ShapeDim::Rank2 { rows: 3, cols: 4 });
    }

    #[test]
    fn shape_trait_3d() {
        let a = NdArray::<f64>::new(&[2, 3, 4]);
        assert_eq!(Shape::shape(&a), ShapeDim::RankN(vec![2, 3, 4]));
    }

    // ****************************************************************
    // NaN handling
    // ****************************************************************

    #[test]
    fn has_nan_false() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0], &[3]);
        assert!(!a.has_nan());
        assert_eq!(a.nan_count(), 0);
    }

    #[test]
    fn has_nan_true() {
        let a = NdArray::from_slice(&[1.0, f64::NAN, 3.0], &[3]);
        assert!(a.has_nan());
        assert_eq!(a.nan_count(), 1);
    }

    #[test]
    fn has_nan_2d() {
        let a = NdArray::from_slice(&[1.0, 2.0, f64::NAN, 4.0, 5.0, f64::NAN], &[3, 2]);
        assert!(a.has_nan());
        assert_eq!(a.nan_count(), 2);
    }

    // ****************************************************************
    // 2D axis access
    // ****************************************************************

    #[test]
    fn col_access() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        assert_eq!(a.col(0), &[1.0, 2.0, 3.0]);
        assert_eq!(a.col(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn col_mut_access() {
        let mut a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        a.col_mut(0)[1] = 99.0;
        assert_eq!(a.col(0), &[1.0, 99.0, 3.0]);
    }

    #[test]
    fn columns_access() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let cols = a.columns();
        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0], &[1.0, 2.0, 3.0]);
        assert_eq!(cols[1], &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn columns_mut_access() {
        let mut a = NdArray::new(&[3, 2]);
        {
            let mut cols = a.columns_mut();
            cols[0].copy_from_slice(&[1.0, 2.0, 3.0]);
            cols[1].copy_from_slice(&[4.0, 5.0, 6.0]);
        }
        assert_eq!(a.col(0), &[1.0, 2.0, 3.0]);
        assert_eq!(a.col(1), &[4.0, 5.0, 6.0]);
    }

    #[cfg(feature = "views")]
    #[test]
    fn obs_access() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let v = a.as_view();
        let obs0: Vec<f64> = (&v.obs(0)).into_iter().collect();
        let obs1: Vec<f64> = (&v.obs(1)).into_iter().collect();
        let obs2: Vec<f64> = (&v.obs(2)).into_iter().collect();
        assert_eq!(obs0, vec![1.0, 4.0]);
        assert_eq!(obs1, vec![2.0, 5.0]);
        assert_eq!(obs2, vec![3.0, 6.0]);

        // Single-shot .obs() on NdArray also works
        assert_eq!((&a.obs(1)).into_iter().collect::<Vec<f64>>(), vec![2.0, 5.0]);

        // 3D obs returns a 2D view
        let b = NdArray::from_slice(&(1..=24).map(|x| x as f64).collect::<Vec<_>>(), &[2, 3, 4]);
        let obs0_3d = b.obs(0);
        assert_eq!(obs0_3d.shape(), &[3, 4]);
    }

    #[test]
    fn col_access_2d() {
        let data: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let a = NdArray::from_slice(&data, &[10, 2]);
        assert_eq!(a.col(0).len(), 10);
        assert_eq!(a.col(1).len(), 10);
        assert_eq!(a.col(0), &data[..10]);
        assert_eq!(a.col(1), &data[10..20]);
    }

    // ****************************************************************
    // BLAS compatibility
    // ****************************************************************

    #[test]
    fn blas_params() {
        let a = NdArray::<f64>::new(&[10, 5]);
        assert_eq!(a.m(), 10);
        assert_eq!(a.n(), 5);
        assert_eq!(a.lda(), 10);
    }

    #[test]
    fn blas_params_aligned_rows() {
        let a = NdArray::<f64>::new(&[8, 3]);
        assert_eq!(a.m(), 8);
        assert_eq!(a.n(), 3);
        assert_eq!(a.lda(), 8);
    }

    // ****************************************************************
    // Compact strides
    // ****************************************************************

    #[test]
    fn compact_strides_2d() {
        for n_rows in 1..=20 {
            let a = NdArray::<f64>::new(&[n_rows, 3]);
            assert_eq!(a.strides()[1], n_rows);
            assert!(a.is_contiguous());
        }
    }

    #[test]
    fn compact_strides_3d() {
        let a = NdArray::<f64>::new(&[10, 3, 4]);
        let strides = a.strides();
        assert_eq!(strides[0], 1);
        assert_eq!(strides[1], 10);
        assert_eq!(strides[2], 10 * 3);
        assert!(a.is_contiguous());
    }

    // ****************************************************************
    // Reshape and transform
    // ****************************************************************

    #[test]
    fn reshape_1d_to_2d() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]);
        let b = a.reshape(&[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(&[0, 0]), 1.0);
        assert_eq!(b.get(&[1, 0]), 2.0);
        assert_eq!(b.get(&[2, 0]), 3.0);
        assert_eq!(b.get(&[0, 1]), 4.0);
    }

    #[test]
    fn reshape_size_mismatch() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0], &[3]);
        assert!(a.reshape(&[2, 2]).is_err());
    }

    #[test]
    fn transpose_2d() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let t = a.transpose();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.get(&[0, 0]), 1.0);
        assert_eq!(t.get(&[1, 0]), 4.0);
        assert_eq!(t.get(&[0, 1]), 2.0);
        assert_eq!(t.get(&[1, 1]), 5.0);
        assert_eq!(t.get(&[0, 2]), 3.0);
        assert_eq!(t.get(&[1, 2]), 6.0);
    }

    #[test]
    fn transpose_3d_reverses_axes() {
        let data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
        let a = NdArray::from_slice(&data, &[2, 3, 4]);
        let t = a.transpose();
        assert_eq!(t.shape(), &[4, 3, 2]);
        assert!(t.is_contiguous());
        // Every element lands at its reversed index.
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(t.get(&[k, j, i]), a.get(&[i, j, k]));
                }
            }
        }
    }

    #[test]
    fn transpose_1d_copies_through() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let t = a.transpose();
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t, a);
    }

    #[test]
    fn flatten() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let flat = a.flatten();
        assert_eq!(flat.shape(), &[6]);
        let vals: Vec<f64> = (&flat).into_iter().collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn to_contiguous_noop() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = a.to_contiguous();
        assert_eq!(a, b);
    }

    #[test]
    fn fill_with_contiguous() {
        let mut a = NdArray::new(&[3, 2]);
        a.fill_with(42.0);
        for v in &a { assert_eq!(v, 42.0); }
    }

    // ****************************************************************
    // From conversions
    // ****************************************************************

    #[test]
    fn from_f64_slice() {
        let a = NdArray::from(&[1.0, 2.0, 3.0][..]);
        assert_eq!(a.ndim(), 1);
        assert_eq!(a.len(), 3);
        assert_eq!(a[(0,)], 1.0);
    }

    #[test]
    fn from_vec64() {
        let v: Vec64<f64> = vec![10.0, 20.0].into_iter().collect();
        let a = NdArray::from(v);
        assert_eq!(a.ndim(), 1);
        assert_eq!(a[(0,)], 10.0);
        assert_eq!(a[(1,)], 20.0);
    }

    #[test]
    fn from_column_vecs() {
        let cols = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let a = NdArray::from(cols.as_slice());
        assert_eq!(a.shape(), &[3, 2]);
        assert_eq!(a.col(0), &[1.0, 2.0, 3.0]);
        assert_eq!(a.col(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn from_float_arrays() {
        let c0 = FloatArray::from_slice(&[1.0, 2.0]);
        let c1 = FloatArray::from_slice(&[3.0, 4.0]);
        let a = NdArray::from([c0, c1].as_slice());
        assert_eq!(a.shape(), &[2, 2]);
        assert_eq!(a.col(0), &[1.0, 2.0]);
        assert_eq!(a.col(1), &[3.0, 4.0]);
    }

    #[test]
    fn from_buffer_explicit_strides() {
        let mut buf = Vec64::with_capacity(16);
        buf.0.resize(16, 0.0);
        buf[0] = 1.0; buf[1] = 2.0; buf[2] = 3.0;
        buf[8] = 4.0; buf[9] = 5.0; buf[10] = 6.0;
        let a = NdArray::from_buffer(Buffer::from_vec64(buf), &[3, 2], &[1, 8]);
        assert_eq!(a.col(0), &[1.0, 2.0, 3.0]);
        assert_eq!(a.col(1), &[4.0, 5.0, 6.0]);
    }

    // ****************************************************************
    // Matrix interop
    // ****************************************************************

    #[cfg(feature = "matrix")]
    #[test]
    fn from_matrix() {
        let mat = Matrix::from_f64_unaligned(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2, Some("m".into()));
        let a = NdArray::from(mat);
        assert_eq!(a.shape(), &[3, 2]);
        assert_eq!(a.name.as_deref(), Some("m"));
        assert_eq!(a.col(0), &[1.0, 2.0, 3.0]);
        assert_eq!(a.col(1), &[4.0, 5.0, 6.0]);
    }

    #[cfg(feature = "matrix")]
    #[test]
    fn to_matrix_roundtrip() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = NdArray::from_slice(&data, &[3, 2]);
        let mat = a.to_matrix().unwrap();
        assert_eq!(mat.n_rows, 3);
        assert_eq!(mat.n_cols, 2);
        assert_eq!(mat.col(0), &[1.0, 2.0, 3.0]);
        assert_eq!(mat.col(1), &[4.0, 5.0, 6.0]);
    }

    #[cfg(feature = "matrix")]
    #[test]
    fn to_matrix_non_2d_fails() {
        let a = NdArray::new(&[5]);
        assert!(a.to_matrix().is_err());
    }

    // ****************************************************************
    // Table interop
    // ****************************************************************

    fn make_numeric_table() -> Table {
        let c0 = FieldArray::from_arr("x", Array::NumericArray(
            NumericArray::Float64(Arc::new(FloatArray::from_slice(&[1.0, 2.0, 3.0])))
        ));
        let c1 = FieldArray::from_arr("y", Array::NumericArray(
            NumericArray::Float64(Arc::new(FloatArray::from_slice(&[4.0, 5.0, 6.0])))
        ));
        Table::new("data".to_string(), Some(vec![c0, c1]))
    }

    #[test]
    fn try_from_table() {
        let table = make_numeric_table();
        let a = NdArray::try_from(&table).unwrap();
        assert_eq!(a.shape(), &[3, 2]);
        assert_eq!(a.col(0), &[1.0, 2.0, 3.0]);
        assert_eq!(a.col(1), &[4.0, 5.0, 6.0]);
        assert_eq!(a.name.as_deref(), Some("data"));
    }

    #[test]
    fn try_from_table_with_nulls_converts_to_nan() {
        let mut mask = Bitmask::new_set_all(3, true);
        mask.set(1, false);
        let arr = FloatArray::new(Buffer::from_slice(&[10.0, 0.0, 30.0]), Some(mask));
        let c0 = FieldArray::from_arr("v", Array::NumericArray(
            NumericArray::Float64(Arc::new(arr))
        ));
        let table = Table::new("nulls".to_string(), Some(vec![c0]));
        let a = NdArray::try_from(&table).unwrap();
        assert_eq!(a.get(&[0, 0]), 10.0);
        assert!(a.get(&[1, 0]).is_nan());
        assert_eq!(a.get(&[2, 0]), 30.0);
    }

    #[test]
    fn try_from_table_coerces_unparseable_text_to_nan() {
        // TryFrom<&Table> uses the library's lenient numeric cast. Text that
        // does not parse as a number coerces to nulls, which surface as NaN
        // in the dense NdArray rather than failing the conversion.
        let c0 = FieldArray::from_arr("name", Array::from_string32(
            StringArray::from_slice(&["a", "b"])
        ));
        let table = Table::new("text".to_string(), Some(vec![c0]));
        let a = NdArray::try_from(&table).unwrap();
        assert_eq!(a.shape(), &[2, 1]);
        assert!(a.get(&[0, 0]).is_nan());
        assert!(a.get(&[1, 0]).is_nan());
    }

    #[test]
    fn to_table_roundtrip() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = NdArray::from_slice(&data, &[3, 2]);
        let fields = vec![
            Field::new("x", ArrowType::Float64, false, None),
            Field::new("y", ArrowType::Float64, false, None),
        ];
        let table = a.to_table(fields).unwrap();
        assert_eq!(table.n_rows(), 3);
        assert_eq!(table.n_cols(), 2);
        assert_eq!(table.col_names(), vec!["x", "y"]);
        let col0 = table.cols[0].array.num().f64();
        assert_eq!(col0.data.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn to_table_gen_auto_names() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let table = a.to_table_gen().unwrap();
        assert_eq!(table.col_names(), vec!["col_0", "col_1"]);
    }

    #[test]
    fn to_table_non_2d_fails() {
        let a = NdArray::new(&[5]);
        assert!(a.to_table_gen().is_err());
    }

    #[test]
    fn to_array_1d() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let arr = a.to_array().unwrap();
        let f = arr.num().f64();
        assert_eq!(f.data.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn to_array_non_1d_fails() {
        let a = NdArray::new(&[2, 3]);
        assert!(a.to_array().is_err());
    }

    // ****************************************************************
    // Concatenate
    // ****************************************************************

    #[test]
    fn concat_1d() {
        let a = NdArray::from_slice(&[1.0, 2.0], &[2]);
        let b = NdArray::from_slice(&[3.0, 4.0, 5.0], &[3]);
        let c = a.concat(b).unwrap();
        assert_eq!(c.shape(), &[5]);
        let vals: Vec<f64> = (&c).into_iter().collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn concat_2d() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = NdArray::from_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[3, 2]);
        let c = a.concat(b).unwrap();
        assert_eq!(c.shape(), &[5, 2]);
        assert_eq!(c.col(0), &[1.0, 2.0, 5.0, 6.0, 7.0]);
        assert_eq!(c.col(1), &[3.0, 4.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    fn concat_dimension_mismatch_fails() {
        let a = NdArray::<f64>::new(&[3, 2]);
        let b = NdArray::new(&[3, 3]);
        assert!(a.concat(b).is_err());
    }

    #[test]
    fn concat_rank_mismatch_fails() {
        let a = NdArray::<f64>::new(&[3]);
        let b = NdArray::new(&[3, 2]);
        assert!(a.concat(b).is_err());
    }

    // ****************************************************************
    // Clone and PartialEq
    // ****************************************************************

    #[test]
    fn clone_and_eq() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn ne_different_data() {
        let a = NdArray::from_slice(&[1.0, 2.0], &[2]);
        let b = NdArray::from_slice(&[1.0, 3.0], &[2]);
        assert_ne!(a, b);
    }

    #[test]
    fn ne_different_shape() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let b = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert_ne!(a, b);
    }

    // ****************************************************************
    // Debug formatting
    // ****************************************************************

    #[test]
    fn debug_1d() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let s = format!("{:?}", a);
        assert!(s.contains("[3]"));
        assert!(s.contains("1D"));
    }

    #[test]
    fn debug_2d_named() {
        let a = NdArray::<f64>::new_named(&[2, 3], "test");
        let s = format!("{:?}", a);
        assert!(s.contains("'test'"));
        assert!(s.contains("[2, 3]"));
        assert!(s.contains("2D"));
    }

    // ****************************************************************
    // Edge cases
    // ****************************************************************

    #[test]
    fn empty_array() {
        let a = NdArray::new(&[0, 5]);
        assert!(a.is_empty());
        assert_eq!(a.len(), 0);
        let vals: Vec<f64> = (&a).into_iter().collect();
        assert!(vals.is_empty());
    }

    #[test]
    fn single_element() {
        let a = NdArray::from_slice(&[42.0], &[1]);
        assert_eq!(a.len(), 1);
        assert_eq!(a[(0,)], 42.0);
        let vals: Vec<f64> = (&a).into_iter().collect();
        assert_eq!(vals, vec![42.0]);
    }

    #[test]
    fn single_element_2d() {
        let a = NdArray::from_slice(&[42.0], &[1, 1]);
        assert_eq!(a[(0, 0)], 42.0);
    }

    #[test]
    fn large_array_iteration_count() {
        let n = 1000;
        let a = NdArray::<f64>::ones(&[n, 100]);
        let count = (&a).into_iter().count();
        assert_eq!(count, n * 100);
    }

    // ****************************************************************
    // Bracket indexing: arr[col][row]
    // ****************************************************************

    #[test]
    fn bracket_index_1d() {
        let a = NdArray::from_slice(&[10.0, 20.0, 30.0], &[3]);
        assert_eq!(a[0], [10.0]);
        assert_eq!(a[2], [30.0]);
    }

    #[test]
    fn bracket_index_2d_column() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        assert_eq!(a[0], [1.0, 2.0, 3.0]);
        assert_eq!(a[1], [4.0, 5.0, 6.0]);
    }

    #[test]
    fn bracket_index_2d_chained() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        // arr[col][row]
        assert_eq!(a[0][0], 1.0);
        assert_eq!(a[0][2], 3.0);
        assert_eq!(a[1][0], 4.0);
        assert_eq!(a[1][2], 6.0);
    }

    #[test]
    fn bracket_index_mut_2d() {
        let mut a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        a[0][1] = 99.0;
        assert_eq!(a[0][1], 99.0);
        assert_eq!(a[0], [1.0, 99.0, 3.0]);
    }

    // ****************************************************************
    // Range indexing: arr[1..3]
    // ****************************************************************

    #[test]
    fn range_index_1d() {
        let a = NdArray::from_slice(&[10.0, 20.0, 30.0, 40.0], &[4]);
        assert_eq!(a[1..3], [20.0, 30.0]);
    }

    #[test]
    fn range_index_2d_columns() {
        // Selecting a range of columns returns the exact logical data,
        // since the compact layout has no padding between columns.
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let slab = &a[0..2];
        assert_eq!(slab, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let col1 = &a[1..2];
        assert_eq!(col1, &[4.0, 5.0, 6.0]);
    }

    #[cfg(feature = "matrix")]
    #[test]
    #[should_panic(expected = "contiguous")]
    fn range_index_non_contiguous_panics() {
        // A Matrix-imported array carries the padded stride, so range
        // indexing has no gap-free slab to return and panics.
        let mat = Matrix::from_f64_unaligned(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2, None,
        );
        let a = NdArray::from(mat);
        if a.is_contiguous() {
            // Padding only appears when rows are off the alignment boundary,
            // which holds for 3 rows. Guard the premise.
            panic!("premise failed: expected non-contiguous import");
        }
        let _ = &a[0..2];
    }

    #[cfg(feature = "matrix")]
    #[test]
    fn to_matrix_repacks_compact_layout() {
        // Compact tensor data re-lays into Matrix's padded column layout.
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let mat = a.to_matrix().unwrap();
        assert_eq!(mat.n_rows, 3);
        assert_eq!(mat.n_cols, 2);
        assert_eq!(mat.stride, 8);
        assert_eq!(&mat.data.as_slice()[..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&mat.data.as_slice()[8..11], &[4.0, 5.0, 6.0]);
    }

    #[cfg(feature = "matrix")]
    #[test]
    fn matrix_roundtrip_via_contiguous() {
        // Matrix -> NdArray is zero-copy with the padded stride carried
        // through. to_contiguous compacts, and to_matrix re-pads.
        let mat = Matrix::from_f64_unaligned(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2, None,
        );
        let a = NdArray::from(mat);
        assert!(!a.is_contiguous());
        assert_eq!(a.get(&[2, 1]), 6.0);
        let compact = a.to_contiguous();
        assert!(compact.is_contiguous());
        assert_eq!(&compact[0..2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let back = compact.to_matrix().unwrap();
        assert_eq!(back.stride, 8);
        assert_eq!(&back.data.as_slice()[8..11], &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn range_from_index() {
        let a = NdArray::from_slice(&[10.0, 20.0, 30.0, 40.0], &[4]);
        assert_eq!(a[2..], [30.0, 40.0]);
    }

    #[test]
    fn range_to_index() {
        let a = NdArray::from_slice(&[10.0, 20.0, 30.0, 40.0], &[4]);
        assert_eq!(a[..2], [10.0, 20.0]);
    }

    #[test]
    fn range_full_index() {
        let a = NdArray::from_slice(&[10.0, 20.0, 30.0], &[3]);
        assert_eq!(a[..], [10.0, 20.0, 30.0]);
    }

    // ****************************************************************
    // Slicing: arr.slice((1..4, 2..5))
    // ****************************************************************

    #[cfg(feature = "views")]
    #[test]
    fn slice_1d_single_index() {
        let a = NdArray::from_slice(&[10.0, 20.0, 30.0], &[3]);
        let v = a.slice(&[&1]);
        assert_eq!(v.shape(), &[1]);
        assert_eq!(v[(0,)], 20.0);
    }

    #[cfg(feature = "views")]
    #[test]
    fn slice_1d_range() {
        let a = NdArray::from_slice(&[10.0, 20.0, 30.0, 40.0], &[4]);
        let v = a.slice(&[&(1..3)]);
        assert_eq!(v.shape(), &[2]);
        assert_eq!(v[(0,)], 20.0);
        assert_eq!(v[(1,)], 30.0);
    }

    #[cfg(feature = "views")]
    #[test]
    fn slice_2d_row_range_single_col() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        // Rows 0..2 of column 1
        let v = a.slice(nd![0..2, 1]);
        assert_eq!(v.shape(), &[2]);
        assert_eq!(v[(0,)], 4.0);
        assert_eq!(v[(1,)], 5.0);
    }

    #[cfg(feature = "views")]
    #[test]
    fn slice_2d_single_row_col_range() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        // Row 1, columns 0..2 - collapses row axis
        let v = a.slice(nd![1, 0..2]);
        assert_eq!(v.shape(), &[2]);
        // Should get row 1 values: a[(1,0)]=2.0, a[(1,1)]=5.0
        let vals: Vec<f64> = (&v).into_iter().collect();
        assert_eq!(vals, vec![2.0, 5.0]);
    }

    #[cfg(feature = "views")]
    #[test]
    fn slice_2d_both_ranges() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        // Rows 0..2, columns 0..2 - sub-matrix
        let v = a.slice(nd![0..2, 0..2]);
        assert_eq!(v.shape(), &[2, 2]);
        assert_eq!(v[(0, 0)], 1.0);
        assert_eq!(v[(1, 0)], 2.0);
        assert_eq!(v[(0, 1)], 4.0);
        assert_eq!(v[(1, 1)], 5.0);
    }

    #[cfg(feature = "views")]
    #[test]
    fn slice_2d_both_indices_scalar() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        // Single element as 1D view
        let v = a.slice(nd![2, 1]);
        assert_eq!(v.shape(), &[1]);
        assert_eq!(v[(0,)], 6.0);
    }

    #[cfg(feature = "views")]
    #[test]
    fn slice_3d_mixed() {
        // 2x3x4 array
        let data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
        let a = NdArray::from_slice(&data, &[2, 3, 4]);
        // All rows, column 1, slices 0..2
        let v = a.slice(nd![0..2, 1, 0..2]);
        assert_eq!(v.shape(), &[2, 2]);
        // a[(0,1,0)]=3, a[(1,1,0)]=4, a[(0,1,1)]=9, a[(1,1,1)]=10
        assert_eq!(v[(0, 0)], 3.0);
        assert_eq!(v[(1, 0)], 4.0);
        assert_eq!(v[(0, 1)], 9.0);
        assert_eq!(v[(1, 1)], 10.0);
    }

    #[cfg(feature = "views")]
    #[test]
    fn slice_with_nd_macro() {
        let data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
        let a = NdArray::from_slice(&data, &[2, 3, 4]);
        let v = a.slice(nd![0..2, 0..3, 0..4]);
        assert_eq!(v.shape(), &[2, 3, 4]);
        assert_eq!(v.len(), 24);
    }

    #[cfg(feature = "views")]
    #[test]
    fn slice_preserves_data_through_iteration() {
        let a = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let v = a.slice(nd![1..3, 0..2]);
        // Sub-matrix: rows 1..3, cols 0..2
        let vals: Vec<f64> = (&v).into_iter().collect();
        assert_eq!(vals, vec![2.0, 3.0, 5.0, 6.0]);
    }

    #[cfg(feature = "views")]
    #[test]
    fn slice_column_window() {
        // Slice rows 2..5 from column 1 of a [10, 2] array
        let data: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let a = NdArray::from_slice(&data, &[10, 2]);
        let v = a.slice(nd![2..5, 1]);
        assert_eq!(v.shape(), &[3]);
        assert_eq!(v[(0,)], 13.0);
        assert_eq!(v[(1,)], 14.0);
        assert_eq!(v[(2,)], 15.0);
    }
}
