//! # **XArray** - *Labelled N-dimensional array for indexed data*
//!
//! Wraps an `NdArray` with named dimensions and optional coordinate labels per axis,
//! enabling selection by label rather than raw position.
//!
//! Where NumPy and NdArray address "give me element [3, 7]", XArray addresses
//! "give me latitude 51.5, longitude -0.1" or "all observations between
//! timestamps 100 and 200". Useful in geospatial, climate, sensor, and any
//! domain where axes carry physical meaning.
//!
//! ## Storage
//! Internally holds one of three modes - owned [`NdArray`], zero-copy
//! [`NdArrayV`] view, or chunked [`SuperNdArray`] - behind a single type.
//! There is no XArrayV or SuperXArray; the storage variant is transparent.
//!
//! ## Quick reference
//! ```ignore
//! let xa = XArray::new(data, &["observation", "feature"]);
//! xa.ax("feature");                       // axis metadata
//! xa.dim("feature");                      // axis position -> 1
//! xa.at("feature", 2.0);                 // select by coordinate value
//! xa.between("observation", 0.0, 50.0);  // coordinate range
//! xa.select(&[("lat", &(0..3))]);        // positional selection
//! ```

use std::fmt;

use crate::enums::error::MinarrowError;
use crate::enums::shape_dim::ShapeDim;
use crate::ffi::arrow_dtype::ArrowType;
use crate::structs::ndarray::NdArray;
#[cfg(all(feature = "views", feature = "select"))]
use crate::traits::selection::{AxisSelection, DataSelector};
#[cfg(all(feature = "views", feature = "select"))]
use std::ops::Range;
use crate::traits::type_unions::Float;
use crate::traits::{concatenate::Concatenate, shape::Shape};
use crate::{Array, Field, StringArray, Table};

#[cfg(feature = "views")]
use crate::structs::views::ndarray_view::NdArrayV;

#[cfg(feature = "chunked")]
use crate::structs::chunked::super_ndarray::SuperNdArray;
#[cfg(feature = "chunked")]
use crate::traits::consolidate::Consolidate;

// ****************************************************************
// Dispatch macros
// ****************************************************************

/// Dispatch to the inner NdArray/NdArrayV/SuperNdArray.
#[cfg(all(feature = "views", feature = "chunked"))]
macro_rules! delegate {
    ($self:expr, $method:ident ( $($arg:expr),* )) => {
        match &$self.data {
            NdArrayE::Owned(nd) => nd.$method($($arg),*),
            NdArrayE::View(v) => v.$method($($arg),*),
            NdArrayE::Chunked(snd) => snd.$method($($arg),*),
        }
    };
}

#[cfg(all(feature = "views", not(feature = "chunked")))]
macro_rules! delegate {
    ($self:expr, $method:ident ( $($arg:expr),* )) => {
        match &$self.data {
            NdArrayE::Owned(nd) => nd.$method($($arg),*),
            NdArrayE::View(v) => v.$method($($arg),*),
        }
    };
}

#[cfg(all(not(feature = "views"), feature = "chunked"))]
macro_rules! delegate {
    ($self:expr, $method:ident ( $($arg:expr),* )) => {
        match &$self.data {
            NdArrayE::Owned(nd) => nd.$method($($arg),*),
            NdArrayE::Chunked(snd) => snd.$method($($arg),*),
        }
    };
}

#[cfg(not(any(feature = "views", feature = "chunked")))]
macro_rules! delegate {
    ($self:expr, $method:ident ( $($arg:expr),* )) => {
        match &$self.data {
            NdArrayE::Owned(nd) => nd.$method($($arg),*),
        }
    };
}

// ****************************************************************
// XArray
// ****************************************************************

/// Labelled N-dimensional array with named dimensions and coordinate-based indexing.
///
/// XArray wraps an [`NdArray`], [`NdArrayV`], or [`SuperNdArray`] with per-axis
/// names and optional coordinate labels, enabling selection by value rather
/// than raw position. The storage variant is chosen at construction time and
/// is transparent to the caller.
///
/// # Construction
/// ```
/// use minarrow::structs::ndarray::NdArray;
/// use minarrow::structs::xarray::{XArray, Axis};
///
/// // Name the dimensions, no coordinate labels
/// let data = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
/// let xa = XArray::new(data, &["station", "measurement"]);
/// assert_eq!(xa.dim("measurement"), 1);
/// ```
///
/// # Coordinate-based selection
/// Assign coordinate labels to an axis, then select by value:
/// ```
/// # use minarrow::structs::ndarray::NdArray;
/// # use minarrow::structs::xarray::XArray;
/// # use minarrow::arr_f64;
/// // 5 stations, 2 measurements each
/// let data = NdArray::from_slice(
///     &[10.0, 20.0, 30.0, 40.0, 50.0, 1.0, 2.0, 3.0, 4.0, 5.0],
///     &[5, 2],
/// );
/// let mut xa = XArray::new(data, &["station", "measurement"]);
///
/// // Label the station axis with latitude values
/// xa.assign_coords("station", arr_f64![-33.8, 35.7, 51.5, 40.7, -22.9]);
///
/// // Select stations between latitudes 35 and 52
/// let subset = xa.between("station", 35.0, 52.0);
/// assert_eq!(subset.shape(), vec![3, 2]); // 3 stations matched
/// ```
///
/// # Positional selection
/// Select by axis name and index/range without coordinates:
/// ```
/// # use minarrow::structs::ndarray::NdArray;
/// # use minarrow::structs::xarray::XArray;
/// let data = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
/// let xa = XArray::new(data, &["obs", "feat"]);
///
/// // Range on one axis - keeps the dimension
/// let sub = xa.select(&[("obs", &(0..2))]);
/// assert_eq!(sub.shape(), vec![2, 2]);
///
/// // Single index collapses the dimension
/// let col0 = xa.select(&[("feat", &0)]);
/// assert_eq!(col0.ndim(), 1);
/// ```
#[derive(Clone)]
pub struct XArray<T> {
    data: NdArrayE<T>,
    axes: Vec<Axis>,
}

impl<T: Float> XArray<T> {
    /// Create with named dimensions, no coordinates.
    pub fn new(data: NdArray<T>, dim_names: &[&str]) -> Self {
        assert_eq!(
            data.ndim(), dim_names.len(),
            "XArray: {} dim names for {}D array", dim_names.len(), data.ndim()
        );
        let axes = dim_names.iter().map(|n| Axis::named(*n)).collect();
        XArray { data: NdArrayE::Owned(data), axes }
    }

    /// Create with fully specified axes.
    pub fn with_axes(data: NdArray<T>, axes: Vec<Axis>) -> Self {
        assert_eq!(
            data.ndim(), axes.len(),
            "XArray: {} axes for {}D array", axes.len(), data.ndim()
        );
        for (i, ax) in axes.iter().enumerate() {
            if let Some(ref coords) = ax.coords {
                assert_eq!(
                    coords.len(), data.shape()[i],
                    "XArray: axis '{}' has {} coords but dimension size is {}",
                    ax.name, coords.len(), data.shape()[i]
                );
            }
        }
        XArray { data: NdArrayE::Owned(data), axes }
    }

    /// Create from NdArray with auto-generated dim names.
    pub fn from_ndarray(data: NdArray<T>) -> Self {
        let axes = (0..data.ndim())
            .map(|i| Axis::named(format!("dim_{}", i)))
            .collect();
        XArray { data: NdArrayE::Owned(data), axes }
    }

    /// Wrap a SuperNdArray with axes.
    #[cfg(feature = "chunked")]
    pub fn from_batched(data: SuperNdArray<T>, dim_names: &[&str]) -> Self {
        assert_eq!(
            data.ndim(), dim_names.len(),
            "XArray: {} dim names for {}D batched array", dim_names.len(), data.ndim()
        );
        let axes = dim_names.iter().map(|n| Axis::named(*n)).collect();
        XArray { data: NdArrayE::Chunked(data), axes }
    }

    /// Wrap an NdArrayV view with axes (zero-copy).
    #[cfg(feature = "views")]
    pub fn from_view(view: NdArrayV<T>, axes: Vec<Axis>) -> Self {
        assert_eq!(
            view.ndim(), axes.len(),
            "XArray: {} axes for {}D view", axes.len(), view.ndim()
        );
        XArray { data: NdArrayE::View(view), axes }
    }

    // ****************************************************************
    // Axis access
    // ****************************************************************

    /// Get axis by name. Returns None if not found.
    pub fn try_ax(&self, name: &str) -> Option<&Axis> {
        self.axes.iter().find(|a| a.name == name)
    }

    /// Get axis by name. Panics if not found.
    pub fn ax(&self, name: &str) -> &Axis {
        self.try_ax(name)
            .unwrap_or_else(|| panic!("XArray: no axis named '{}'", name))
    }

    /// Get the position of a named axis. Returns None if not found.
    pub fn try_dim(&self, name: &str) -> Option<usize> {
        self.axes.iter().position(|a| a.name == name)
    }

    /// Get the position of a named axis. Panics if not found.
    pub fn dim(&self, name: &str) -> usize {
        self.try_dim(name)
            .unwrap_or_else(|| panic!("XArray: no axis named '{}'", name))
    }

    /// All axes.
    #[inline]
    pub fn axes(&self) -> &[Axis] { &self.axes }

    /// Dim names.
    pub fn dim_names(&self) -> Vec<&str> {
        self.axes.iter().map(|a| a.name.as_str()).collect()
    }

    // ****************************************************************
    // Delegated data access
    // ****************************************************************

    /// Consume and return the inner NdArray, consolidating if batched,
    /// materialising if a view.
    pub fn into_ndarray(self) -> NdArray<T> {
        match self.data {
            NdArrayE::Owned(nd) => nd,
            #[cfg(feature = "views")]
            NdArrayE::View(v) => v.to_ndarray(),
            #[cfg(feature = "chunked")]
            NdArrayE::Chunked(snd) => {
                snd.consolidate()
            }
        }
    }

    /// Materialise to owned NdArray if currently a view or batched.
    pub fn to_owned(&self) -> XArray<T> {
        match &self.data {
            NdArrayE::Owned(_) => self.clone(),
            #[cfg(feature = "views")]
            NdArrayE::View(v) => XArray {
                data: NdArrayE::Owned(v.to_ndarray()),
                axes: self.axes.clone(),
            },
            #[cfg(feature = "chunked")]
            NdArrayE::Chunked(snd) => {
                XArray {
                    data: NdArrayE::Owned(snd.clone().consolidate()),
                    axes: self.axes.clone(),
                }
            }
        }
    }

    /// True if backed by a single owned NdArray.
    pub fn is_owned(&self) -> bool {
        matches!(&self.data, NdArrayE::Owned(_))
    }

    /// Borrow the inner storage for crate-internal dispatch.
    #[cfg(feature = "broadcast")]
    #[inline]
    pub(crate) fn storage(&self) -> &NdArrayE<T> {
        &self.data
    }

    /// Assemble from storage and axes for crate-internal construction.
    #[cfg(feature = "broadcast")]
    #[inline]
    pub(crate) fn from_storage(data: NdArrayE<T>, axes: Vec<Axis>) -> Self {
        XArray { data, axes }
    }

    #[inline]
    pub fn ndim(&self) -> usize { delegate!(self, ndim()) }

    pub fn shape(&self) -> Vec<usize> {
        match &self.data {
            NdArrayE::Owned(nd) => nd.shape().to_vec(),
            #[cfg(feature = "views")]
            NdArrayE::View(v) => v.shape().to_vec(),
            #[cfg(feature = "chunked")]
            NdArrayE::Chunked(snd) => snd.shape(),
        }
    }

    #[inline]
    pub fn strides(&self) -> &[usize] { delegate!(self, strides()) }

    #[inline]
    pub fn len(&self) -> usize { delegate!(self, len()) }

    #[inline]
    pub fn is_empty(&self) -> bool { delegate!(self, is_empty()) }

    /// Zero-copy view of a single observation (axis-0 element).
    ///
    /// Returns an (N-1)-dimensional `NdArrayV` view regardless of
    /// the inner storage mode.
    #[cfg(feature = "views")]
    pub fn obs(&self, idx: usize) -> NdArrayV<T> {
        match &self.data {
            NdArrayE::Owned(nd) => nd.as_view().obs(idx),
            #[cfg(feature = "views")]
            NdArrayE::View(v) => v.obs(idx),
            #[cfg(feature = "chunked")]
            NdArrayE::Chunked(snd) => snd.obs(idx),
        }
    }

    #[inline]
    pub fn m(&self) -> i32 { delegate!(self, m()) }

    #[inline]
    pub fn n(&self) -> i32 { delegate!(self, n()) }

    #[inline]
    pub fn lda(&self) -> i32 { delegate!(self, lda()) }

    /// Single element access.
    pub fn get(&self, indices: &[usize]) -> T { delegate!(self, get(indices)) }

    /// Mutable element access. Only works on owned or batched data.
    /// Triggers copy-on-write if the owned array is shared with views.
    pub fn set(&mut self, indices: &[usize], value: T) {
        match &mut self.data {
            NdArrayE::Owned(nd) => nd.set(indices, value),
            #[cfg(feature = "views")]
            NdArrayE::View(_) => panic!("XArray: cannot mutate a view"),
            #[cfg(feature = "chunked")]
            NdArrayE::Chunked(snd) => snd.set(indices, value),
        }
    }

    /// Parallel iterator over axis-0 observations. Each item is the
    /// observation index and a zero-copy `NdArrayV` view.
    #[cfg(all(feature = "parallel_proc", feature = "views"))]
    pub fn par_iter_obs(&self) -> impl rayon::iter::ParallelIterator<Item = (usize, NdArrayV<T>)> + '_ {
        use rayon::prelude::*;
        let n_obs = self.shape()[0];
        (0..n_obs).into_par_iter().map(move |i| (i, self.obs(i)))
    }

    // ****************************************************************
    // Apply
    // ****************************************************************

    /// Apply a function to every logical element, returning a new labelled
    /// array with the same axes. Chunked storage keeps its batch
    /// boundaries. View storage materialises to owned.
    pub fn apply(&self, f: impl Fn(T) -> T) -> XArray<T> {
        let data = match &self.data {
            NdArrayE::Owned(nd) => NdArrayE::Owned(nd.apply(f)),
            #[cfg(feature = "views")]
            NdArrayE::View(v) => NdArrayE::Owned(v.apply(f)),
            #[cfg(feature = "chunked")]
            NdArrayE::Chunked(snd) => NdArrayE::Chunked(snd.apply(f)),
        };
        XArray { data, axes: self.axes.clone() }
    }

    /// Apply a function to every logical element in place, with no
    /// allocation. Only works on owned or batched data, matching `set`.
    pub fn apply_mut(&mut self, f: impl Fn(T) -> T) {
        match &mut self.data {
            NdArrayE::Owned(nd) => nd.apply_mut(f),
            #[cfg(feature = "views")]
            NdArrayE::View(_) => panic!("XArray: cannot mutate a view"),
            #[cfg(feature = "chunked")]
            NdArrayE::Chunked(snd) => snd.apply_mut(f),
        }
    }

    // ****************************************************************
    // Label management
    // ****************************************************************

    /// Rename an axis.
    pub fn rename_dim(&mut self, old: &str, new: &str) {
        let idx = self.dim(old);
        self.axes[idx].name = new.to_string();
    }

    /// Assign or replace coordinates for a named axis.
    pub fn assign_coords(&mut self, dim_name: &str, coords: Array) {
        let idx = self.dim(dim_name);
        let dim_size = self.shape()[idx];
        assert_eq!(
            coords.len(), dim_size,
            "XArray: coords length {} does not match axis '{}' size {}",
            coords.len(), dim_name, dim_size
        );
        self.axes[idx].coords = Some(coords);
    }

    /// Remove coordinates from a named axis.
    pub fn drop_coords(&mut self, dim_name: &str) {
        let idx = self.dim(dim_name);
        self.axes[idx].coords = None;
    }

    // ****************************************************************
    // Positional selection: .select()
    // ****************************************************************

    /// Select sub-arrays by named axis positions. Returns zero-copy XArray
    /// backed by NdArrayV.
    ///
    /// Single indices collapse that dimension. Ranges keep it.
    ///
    /// # Examples
    /// ```ignore
    /// xa.select(&[("lat", &(0..3))])                  // single axis range
    /// xa.select(&[("lat", &(0..3)), ("lon", &2)])     // multi-axis mixed
    /// ```
    #[cfg(all(feature = "views", feature = "select"))]
    pub fn select(&self, selection: &[(&str, &dyn DataSelector)]) -> XArray<T> {
        let shape = self.shape();

        // Default: full range on every axis
        let full_ranges: Vec<Range<usize>> = shape.iter().map(|&n| 0..n).collect();
        let mut refs: Vec<&dyn DataSelector> = full_ranges.iter().map(|r| r as _).collect();
        let mut new_axes: Vec<Option<Axis>> =
            self.axes.iter().cloned().map(Some).collect();

        // Apply named selections. Collapsed axes drop their labels, and
        // windowed axes carry their coordinates through, narrowed.
        for (name, sel) in selection {
            let idx = self.dim(name);
            refs[idx] = *sel;
            let (start, end, collapse) = sel.resolve_axis(shape[idx]);
            if collapse {
                new_axes[idx] = None;
            } else {
                let mut narrowed = Axis::named(&self.axes[idx].name);
                if let Some(ref coords) = self.axes[idx].coords {
                    narrowed.coords = Some(coords.slice_clone(start, end - start));
                }
                new_axes[idx] = Some(narrowed);
            }
        }

        let view = self.slice(&refs);
        let result_axes: Vec<Axis> = new_axes.into_iter().flatten().collect();
        XArray { data: NdArrayE::View(view), axes: result_axes }
    }

    /// Slice the underlying NdArray/NdArrayV positionally. Zero-copy for
    /// owned and view storage. Chunked storage consolidates first, since a
    /// single strided view cannot span separate batch allocations.
    /// For named axis selection, use `.select()` instead.
    #[cfg(all(feature = "views", feature = "select"))]
    pub fn slice(&self, selection: &[&dyn DataSelector]) -> NdArrayV<T> {
        match &self.data {
            NdArrayE::Owned(nd) => nd.slice(selection),
            NdArrayE::View(v) => v.slice(selection),
            #[cfg(feature = "chunked")]
            NdArrayE::Chunked(snd) => {
                snd.clone().consolidate().slice(selection)
            }
        }
    }

    // ****************************************************************
    // Coordinate value selection: .at() and .between()
    // ****************************************************************

    /// Select a single position on a named axis by coordinate value.
    /// Collapses that dimension. Returns an error if the value is not found.
    #[cfg(all(feature = "views", feature = "select"))]
    pub fn try_at(&self, dim_name: &str, value: f64) -> Result<XArray<T>, MinarrowError> {
        let dim_idx = self.dim(dim_name);
        let pos = self.try_find_coord_pos(dim_idx, value)?;

        let shape = self.shape();
        let full_ranges: Vec<Range<usize>> = shape.iter().map(|&n| 0..n).collect();
        let mut refs: Vec<&dyn DataSelector> = full_ranges.iter().map(|r| r as _).collect();
        refs[dim_idx] = &pos;
        let new_axes: Vec<Axis> = self
            .axes
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != dim_idx)
            .map(|(_, ax)| ax.clone())
            .collect();

        let view = self.slice(&refs);
        Ok(XArray { data: NdArrayE::View(view), axes: new_axes })
    }

    /// Select a single position by coordinate value. Panics if not found.
    #[cfg(all(feature = "views", feature = "select"))]
    pub fn at(&self, dim_name: &str, value: f64) -> XArray<T> {
        self.try_at(dim_name, value)
            .unwrap_or_else(|e| panic!("{}", e))
    }

    /// Select a range by coordinate value bounds (inclusive).
    /// Returns an error if no values fall in the range.
    #[cfg(all(feature = "views", feature = "select"))]
    pub fn try_between(&self, dim_name: &str, low: f64, high: f64) -> Result<XArray<T>, MinarrowError> {
        let dim_idx = self.dim(dim_name);
        let (start, end) = self.try_find_coord_range(dim_idx, low, high)?;

        let shape = self.shape();
        let full_ranges: Vec<Range<usize>> = shape.iter().map(|&n| 0..n).collect();
        let mut refs: Vec<&dyn DataSelector> = full_ranges.iter().map(|r| r as _).collect();
        let window = start..end;
        refs[dim_idx] = &window;
        let new_axes: Vec<Axis> = self
            .axes
            .iter()
            .enumerate()
            .map(|(i, ax)| {
                if i == dim_idx {
                    let mut narrowed = Axis::named(&ax.name);
                    if let Some(ref coords) = ax.coords {
                        narrowed.coords = Some(coords.slice_clone(start, end - start));
                    }
                    narrowed
                } else {
                    ax.clone()
                }
            })
            .collect();

        let view = self.slice(&refs);
        Ok(XArray { data: NdArrayE::View(view), axes: new_axes })
    }

    /// Select a range by coordinate value bounds. Panics if no values match.
    #[cfg(all(feature = "views", feature = "select"))]
    pub fn between(&self, dim_name: &str, low: f64, high: f64) -> XArray<T> {
        self.try_between(dim_name, low, high)
            .unwrap_or_else(|e| panic!("{}", e))
    }

    // ****************************************************************
    // Axis operations
    // ****************************************************************

    /// Transpose (2D only). Reorders axes by name.
    pub fn transpose(&self, dim_order: &[&str]) -> Result<XArray<T>, MinarrowError> {
        if self.ndim() != 2 {
            return Err(MinarrowError::ShapeError {
                message: format!("transpose requires 2D, got {}D", self.ndim()),
            });
        }
        if dim_order.len() != 2 {
            return Err(MinarrowError::ShapeError {
                message: "transpose: expected 2 dim names".to_string(),
            });
        }
        let inner = match &self.data {
            NdArrayE::Owned(nd) => nd.transpose(),
            #[cfg(feature = "views")]
            NdArrayE::View(v) => v.transpose().to_ndarray(),
            #[cfg(feature = "chunked")]
            NdArrayE::Chunked(snd) => {
                snd.clone().consolidate().transpose()
            }
        };
        let new_axes = dim_order.iter()
            .map(|name| self.ax(name).clone())
            .collect();
        Ok(XArray { data: NdArrayE::Owned(inner), axes: new_axes })
    }

    // ****************************************************************
    // Internal
    // ****************************************************************

    /// Find the position of a coordinate value on an axis.
    fn try_find_coord_pos(&self, dim_idx: usize, value: f64) -> Result<usize, MinarrowError> {
        let ax = &self.axes[dim_idx];
        let coords = ax.coords.as_ref().ok_or_else(|| MinarrowError::ShapeError {
            message: format!("axis '{}' has no coordinates for value lookup", ax.name),
        })?;
        let f64_arr = coords.try_num()
            .map_err(|_| MinarrowError::TypeError {
                from: "non-numeric", to: "Float64",
                message: Some(format!("axis '{}' coords are not numeric", ax.name)),
            })?
            .try_f64()
            .map_err(|_| MinarrowError::TypeError {
                from: "numeric", to: "Float64",
                message: Some(format!("axis '{}' coords cannot convert to f64", ax.name)),
            })?;
        let data = f64_arr.data.as_slice();
        for i in 0..data.len() {
            if data[i] == value { return Ok(i); }
        }
        Err(MinarrowError::IndexError(
            format!("value {} not found on axis '{}'", value, ax.name)
        ))
    }

    /// Find the start/end positions for a coordinate range on an axis.
    fn try_find_coord_range(&self, dim_idx: usize, low: f64, high: f64) -> Result<(usize, usize), MinarrowError> {
        let ax = &self.axes[dim_idx];
        let coords = ax.coords.as_ref().ok_or_else(|| MinarrowError::ShapeError {
            message: format!("axis '{}' has no coordinates for range lookup", ax.name),
        })?;
        let f64_arr = coords.try_num()
            .map_err(|_| MinarrowError::TypeError {
                from: "non-numeric", to: "Float64",
                message: Some(format!("axis '{}' coords are not numeric", ax.name)),
            })?
            .try_f64()
            .map_err(|_| MinarrowError::TypeError {
                from: "numeric", to: "Float64",
                message: Some(format!("axis '{}' coords cannot convert to f64", ax.name)),
            })?;
        let data = f64_arr.data.as_slice();
        let mut start = data.len();
        let mut end = 0;
        for i in 0..data.len() {
            if data[i] >= low && data[i] <= high {
                if i < start { start = i; }
                if i + 1 > end { end = i + 1; }
            }
        }
        if start >= end {
            return Err(MinarrowError::IndexError(
                format!("no values in [{}, {}] on axis '{}'", low, high, ax.name)
            ));
        }
        Ok((start, end))
    }
}

impl XArray<f64> {
    /// Convert a 2D XArray to a Table. Uses axis 1 coords as column names
    /// if available, otherwise generates names from the dim name.
    pub fn to_table(self) -> Result<Table, MinarrowError> {
        if self.ndim() != 2 {
            return Err(MinarrowError::ShapeError {
                message: format!("to_table requires 2D, got {}D", self.ndim()),
            });
        }
        let n_cols = self.shape()[1];
        let fields: Vec<Field> = if let Some(ref coords) = self.axes[1].coords {
            (0..n_cols).map(|i| {
                let name = coords.value_to_string(i);
                Field::new(name, ArrowType::Float64, false, None)
            }).collect()
        } else {
            (0..n_cols).map(|i| {
                Field::new(
                    format!("{}_{}", self.axes[1].name, i),
                    ArrowType::Float64, false, None,
                )
            }).collect()
        };
        self.into_ndarray().to_table(fields)
    }
}

// ****************************************************************
// NdArrayE - owned or view storage
// ****************************************************************

/// Internal storage: either owned NdArray or zero-copy NdArrayV.
/// Enables XArray to be a single type regardless of ownership.
#[derive(Clone)]
pub(crate) enum NdArrayE<T> {
    /// The array's own shared buffer makes clones a refcount bump and
    /// lets views borrow the parent zero-copy, with copy-on-write
    /// mutation.
    Owned(NdArray<T>),
    #[cfg(feature = "views")]
    View(NdArrayV<T>),
    #[cfg(feature = "chunked")]
    Chunked(SuperNdArray<T>),
}

// ****************************************************************
// Axis
// ****************************************************************

/// A named dimension with optional coordinate labels.
///
/// The coords array, when present, must have the same length as the
/// corresponding NdArray dimension. Coordinates may be stored as any
/// Minarrow Array type, but value-based selection currently resolves
/// floating-point coordinates only.
#[derive(Clone, Debug, PartialEq)]
pub struct Axis {
    pub name: String,
    pub coords: Option<Array>,
}

impl Axis {
    /// Named axis without coordinates.
    pub fn named(name: impl Into<String>) -> Self {
        Axis { name: name.into(), coords: None }
    }

    /// Named axis with coordinate labels.
    pub fn with_coords(name: impl Into<String>, coords: Array) -> Self {
        Axis { name: name.into(), coords: Some(coords) }
    }
}

// ****************************************************************
// Trait implementations
// ****************************************************************

/// Positional selection across every axis at once, delegating to `slice`.
/// The result is an unlabelled view. For named-axis selection with the
/// labels carried through, use `.select()`.
#[cfg(all(feature = "views", feature = "select"))]
impl<T: Float> AxisSelection for XArray<T> {
    type View = NdArrayV<T>;

    fn s(&self, selection: &[&dyn DataSelector]) -> NdArrayV<T> {
        self.slice(selection)
    }

    fn get_axis_count(&self) -> usize {
        self.ndim()
    }
}

impl<T: Float> Shape for XArray<T> {
    fn shape(&self) -> ShapeDim {
        match &self.data {
            NdArrayE::Owned(nd) => Shape::shape(nd),
            #[cfg(feature = "views")]
            NdArrayE::View(v) => Shape::shape(v),
            #[cfg(feature = "chunked")]
            NdArrayE::Chunked(snd) => Shape::shape(snd),
        }
    }
}

impl<T: Float> Concatenate for XArray<T> {
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        if self.axes.len() != other.axes.len() {
            return Err(MinarrowError::IncompatibleTypeError {
                from: "XArray", to: "XArray",
                message: Some(format!("{} dims vs {} dims", self.axes.len(), other.axes.len())),
            });
        }
        for (a, b) in self.axes.iter().zip(other.axes.iter()) {
            if a.name != b.name {
                return Err(MinarrowError::IncompatibleTypeError {
                    from: "XArray", to: "XArray",
                    message: Some(format!("dim name mismatch: '{}' vs '{}'", a.name, b.name)),
                });
            }
        }

        // Destructure to avoid cloning axes
        let XArray { data: self_data, axes: self_axes } = self;
        let XArray { data: other_data, axes: other_axes } = other;

        let to_ndarray = |data: NdArrayE<T>| -> NdArray<T> {
            match data {
                NdArrayE::Owned(nd) => nd,
                #[cfg(feature = "views")]
                NdArrayE::View(v) => v.to_ndarray(),
                #[cfg(feature = "chunked")]
                NdArrayE::Chunked(snd) => {
                    snd.consolidate()
                }
            }
        };
        let self_nd = to_ndarray(self_data);
        let other_nd = to_ndarray(other_data);
        let new_data = self_nd.concat(other_nd)?;

        // Merge axis 0 coords if both present, keep other axes unchanged
        let mut new_axes = Vec::with_capacity(self_axes.len());
        for (i, (a, b)) in self_axes.into_iter().zip(other_axes.into_iter()).enumerate() {
            if i == 0 {
                let merged_coords = match (a.coords, b.coords) {
                    (Some(ca), Some(cb)) => Some(ca.concat(cb)?),
                    _ => None,
                };
                new_axes.push(Axis { name: a.name, coords: merged_coords });
            } else {
                new_axes.push(a);
            }
        }

        Ok(XArray { data: NdArrayE::Owned(new_data), axes: new_axes })
    }
}

impl<'a, T: Float> IntoIterator for &'a XArray<T> {
    type Item = T;
    type IntoIter = Box<dyn Iterator<Item = T> + 'a>;

    fn into_iter(self) -> Self::IntoIter {
        match &self.data {
            NdArrayE::Owned(nd) => Box::new(nd.into_iter()),
            #[cfg(feature = "views")]
            NdArrayE::View(v) => Box::new(v.into_iter()),
            #[cfg(feature = "chunked")]
            NdArrayE::Chunked(snd) => Box::new(snd.into_iter()),
        }
    }
}

impl<T: Float> PartialEq for XArray<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.axes != other.axes { return false; }
        if self.shape() != other.shape() { return false; }
        self.into_iter().zip(other.into_iter()).all(|(a, b)| a == b)
    }
}

impl<T: Float> fmt::Debug for XArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shape = self.shape();
        let dims: Vec<String> = self.axes.iter()
            .zip(shape.iter())
            .map(|(ax, &size)| {
                let label = if ax.coords.is_some() { " (labelled)" } else { "" };
                format!("{}={}{}", ax.name, size, label)
            })
            .collect();
        let storage = if self.is_owned() { "owned" } else { "view" };
        write!(f, "XArray [{}] ({})", dims.join(", "), storage)
    }
}


// ****************************************************************
// TryFrom<Table>
// ****************************************************************

impl TryFrom<Table> for XArray<f64> {
    type Error = MinarrowError;

    fn try_from(table: Table) -> Result<Self, Self::Error> {
        let col_names: Vec<String> = table.col_names().iter().map(|s| s.to_string()).collect();
        let table_name = table.name.clone();
        let data = NdArray::try_from(&table)?;

        let obs_axis = Axis::named(
            if table_name.is_empty() { "observation" } else { &table_name }
        );
        let feat_axis = Axis::with_coords(
            "feature",
            Array::from_string32(StringArray::from_slice(
                &col_names.iter().map(|s| s.as_str()).collect::<Vec<_>>()
            )),
        );

        Ok(XArray { data: NdArrayE::Owned(data), axes: vec![obs_axis, feat_axis] })
    }
}

// ****************************************************************
// Tests
// ****************************************************************

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(all(feature = "views", feature = "select"))]
    use crate::nd;
    use std::sync::Arc;
    use crate::{FloatArray, NumericArray, FieldArray};

    #[cfg(all(feature = "views", feature = "select"))]
    #[test]
    fn axis_selection_positional() {
        let xa = XArray::new(
            NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]),
            &["obs", "feat"],
        );
        let v = xa.s(nd![0..2, 1]);
        assert_eq!(v.shape(), &[2]);
        assert_eq!(v.get(&[0]), 4.0);
        assert_eq!(v.get(&[1]), 5.0);
    }

    #[test]
    fn apply_preserves_axes() {
        let xa = XArray::new(
            NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
            &["obs", "feat"],
        );
        let out = xa.apply(|x| x * 2.0);
        assert_eq!(out.dim_names(), vec!["obs", "feat"]);
        assert_eq!(out.get(&[1, 1]), 8.0);
        assert_eq!(xa.get(&[1, 1]), 4.0);
    }

    #[test]
    fn apply_mut_owned() {
        let mut xa = XArray::new(NdArray::from_slice(&[1.0, 2.0], &[2]), &["t"]);
        xa.apply_mut(|x| x + 10.0);
        assert_eq!(xa.get(&[0]), 11.0);
        assert_eq!(xa.get(&[1]), 12.0);
    }

    #[cfg(feature = "views")]
    #[test]
    #[should_panic(expected = "cannot mutate a view")]
    fn apply_mut_view_panics() {
        let base = XArray::new(NdArray::from_slice(&[1.0, 2.0], &[2]), &["t"]);
        let mut view = base.select(&[("t", &(0..2))]);
        view.apply_mut(|x| x + 1.0);
    }

    fn make_2d() -> NdArray<f64> {
        NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])
    }

    #[test]
    fn f32_element_type() {
        let data = NdArray::<f32>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let xa = XArray::new(data, &["obs", "feat"]);
        assert_eq!(xa.get(&[0, 0]), 1.0f32);
        assert_eq!(xa.get(&[2, 1]), 6.0f32);
        let sub = xa.select(&[("obs", &(0..2))]);
        assert_eq!(sub.shape(), vec![2, 2]);
        assert_eq!(sub.get(&[1, 1]), 5.0f32);
    }

    fn float_coords(vals: &[f64]) -> Array {
        Array::NumericArray(NumericArray::Float64(
            Arc::new(FloatArray::from_slice(vals))
        ))
    }

    // *** Construction ************************************************

    #[test]
    fn new_with_dim_names() {
        let xa = XArray::new(make_2d(), &["obs", "feat"]);
        assert_eq!(xa.ndim(), 2);
        assert_eq!(xa.dim_names(), vec!["obs", "feat"]);
    }

    #[test]
    fn from_ndarray_auto_names() {
        let xa = XArray::from_ndarray(make_2d());
        assert_eq!(xa.dim_names(), vec!["dim_0", "dim_1"]);
    }

    #[test]
    fn with_axes_and_coords() {
        let axes = vec![
            Axis::named("obs"),
            Axis::with_coords("feat", float_coords(&[10.0, 20.0])),
        ];
        let xa = XArray::with_axes(make_2d(), axes);
        assert!(xa.ax("feat").coords.is_some());
        assert!(xa.ax("obs").coords.is_none());
    }

    // *** Axis access *************************************************

    #[test]
    fn ax_and_dim() {
        let xa = XArray::new(make_2d(), &["obs", "feat"]);
        assert_eq!(xa.ax("feat").name, "feat");
        assert_eq!(xa.dim("obs"), 0);
        assert_eq!(xa.dim("feat"), 1);
    }

    #[test]
    fn try_ax_and_try_dim() {
        let xa = XArray::new(make_2d(), &["obs", "feat"]);
        assert!(xa.try_ax("feat").is_some());
        assert!(xa.try_ax("missing").is_none());
        assert_eq!(xa.try_dim("obs"), Some(0));
        assert_eq!(xa.try_dim("missing"), None);
    }

    // *** Delegation **************************************************

    #[test]
    fn delegates_shape_and_access() {
        let xa = XArray::new(make_2d(), &["obs", "feat"]);
        assert_eq!(xa.shape(), vec![3, 2]);
        assert_eq!(xa.len(), 6);
        assert_eq!(xa.get(&[0, 0]), 1.0);
        let obs0: Vec<f64> = (&xa.obs(0)).into_iter().collect();
        assert_eq!(obs0, vec![1.0, 4.0]);
    }

    #[test]
    fn iteration() {
        let xa = XArray::new(make_2d(), &["obs", "feat"]);
        let vals: Vec<f64> = (&xa).into_iter().collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // *** Label management ********************************************

    #[test]
    fn rename_dim() {
        let mut xa = XArray::new(make_2d(), &["obs", "feat"]);
        xa.rename_dim("feat", "variable");
        assert_eq!(xa.dim("variable"), 1);
    }

    #[test]
    fn assign_and_drop_coords() {
        let mut xa = XArray::new(make_2d(), &["obs", "feat"]);
        assert!(xa.ax("feat").coords.is_none());
        xa.assign_coords("feat", float_coords(&[10.0, 20.0]));
        assert!(xa.ax("feat").coords.is_some());
        xa.drop_coords("feat");
        assert!(xa.ax("feat").coords.is_none());
    }

    // *** Positional selection ****************************************

    #[cfg(feature = "views")]
    #[test]
    fn select_range() {
        let xa = XArray::new(make_2d(), &["obs", "feat"]);
        let result = xa.select(&[("obs", &(0..2))]);
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.dim_names(), vec!["obs", "feat"]);
        assert!(!result.is_owned());
    }

    #[cfg(feature = "views")]
    #[test]
    fn select_index_collapses() {
        let xa = XArray::new(make_2d(), &["obs", "feat"]);
        let result = xa.select(&[("feat", &0)]);
        assert_eq!(result.ndim(), 1);
        assert_eq!(result.dim_names(), vec!["obs"]);
        let vals: Vec<f64> = (&result).into_iter().collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[cfg(feature = "views")]
    #[test]
    fn select_multi_axis() {
        let xa = XArray::new(make_2d(), &["obs", "feat"]);
        let result = xa.select(&[("obs", &(1..3)), ("feat", &1)]);
        assert_eq!(result.ndim(), 1);
        assert_eq!(result.dim_names(), vec!["obs"]);
        let vals: Vec<f64> = (&result).into_iter().collect();
        assert_eq!(vals, vec![5.0, 6.0]);
    }

    #[cfg(feature = "views")]
    #[test]
    fn select_preserves_coords() {
        let mut xa = XArray::new(
            NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]),
            &["obs", "feat"],
        );
        xa.assign_coords("obs", float_coords(&[10.0, 20.0, 30.0, 40.0]));
        let result = xa.select(&[("obs", &(1..3))]);
        let coords = result.ax("obs").coords.as_ref().unwrap();
        assert_eq!(coords.len(), 2);
    }

    // *** Coordinate selection ****************************************

    #[cfg(feature = "views")]
    #[test]
    fn at_by_coord() {
        let mut xa = XArray::new(make_2d(), &["obs", "feat"]);
        xa.assign_coords("feat", float_coords(&[10.0, 20.0]));
        let result = xa.at("feat", 20.0);
        assert_eq!(result.ndim(), 1);
        assert_eq!(result.dim_names(), vec!["obs"]);
        let vals: Vec<f64> = (&result).into_iter().collect();
        assert_eq!(vals, vec![4.0, 5.0, 6.0]);
    }

    #[cfg(feature = "views")]
    #[test]
    fn try_at_not_found() {
        let mut xa = XArray::new(make_2d(), &["obs", "feat"]);
        xa.assign_coords("feat", float_coords(&[10.0, 20.0]));
        assert!(xa.try_at("feat", 99.0).is_err());
    }

    #[cfg(feature = "views")]
    #[test]
    fn between_coord_range() {
        let data = NdArray::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            &[5, 2],
        );
        let mut xa = XArray::new(data, &["obs", "feat"]);
        xa.assign_coords("obs", float_coords(&[0.0, 1.0, 2.0, 3.0, 4.0]));
        let result = xa.between("obs", 1.0, 3.0);
        assert_eq!(result.shape(), vec![3, 2]);
        let vals: Vec<f64> = (&result).into_iter().take(3).collect();
        assert_eq!(vals, vec![2.0, 3.0, 4.0]);
    }

    #[cfg(feature = "views")]
    #[test]
    fn try_between_empty_range() {
        let mut xa = XArray::new(make_2d(), &["obs", "feat"]);
        xa.assign_coords("obs", float_coords(&[0.0, 1.0, 2.0]));
        assert!(xa.try_between("obs", 100.0, 200.0).is_err());
    }

    // *** Transpose ***************************************************

    #[test]
    fn transpose_reorders_axes() {
        let xa = XArray::new(make_2d(), &["obs", "feat"]);
        let t = xa.transpose(&["feat", "obs"]).unwrap();
        assert_eq!(t.dim_names(), vec!["feat", "obs"]);
        assert_eq!(t.shape(), &[2, 3]);
    }

    // *** Concatenate *************************************************

    #[test]
    fn concat_matching_dims() {
        let a = XArray::new(
            NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
            &["obs", "feat"],
        );
        let b = XArray::new(
            NdArray::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]),
            &["obs", "feat"],
        );
        let c = a.concat(b).unwrap();
        assert_eq!(c.shape(), &[4, 2]);
        assert_eq!(c.dim_names(), vec!["obs", "feat"]);
    }

    #[test]
    fn concat_merges_axis0_coords() {
        let mut a = XArray::new(
            NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
            &["obs", "feat"],
        );
        a.assign_coords("obs", float_coords(&[0.0, 1.0]));
        let mut b = XArray::new(
            NdArray::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]),
            &["obs", "feat"],
        );
        b.assign_coords("obs", float_coords(&[2.0, 3.0]));
        let c = a.concat(b).unwrap();
        let coords = c.ax("obs").coords.as_ref().unwrap();
        assert_eq!(coords.len(), 4);
    }

    #[test]
    fn concat_mismatched_dims_fails() {
        let a = XArray::new(make_2d(), &["obs", "feat"]);
        let b = XArray::new(make_2d(), &["obs", "variable"]);
        assert!(a.concat(b).is_err());
    }

    // *** Table roundtrip *********************************************

    #[test]
    fn try_from_table() {
        let c0 = FieldArray::from_arr("height", Array::NumericArray(
            NumericArray::Float64(Arc::new(FloatArray::from_slice(&[175.0, 168.0])))
        ));
        let c1 = FieldArray::from_arr("weight", Array::NumericArray(
            NumericArray::Float64(Arc::new(FloatArray::from_slice(&[82.0, 71.0])))
        ));
        let table = Table::new("patients".to_string(), Some(vec![c0, c1]));
        let xa = XArray::try_from(table).unwrap();
        assert_eq!(xa.dim_names(), vec!["patients", "feature"]);
        assert!(xa.ax("feature").coords.is_some());
    }

    #[test]
    fn to_table_with_coord_names() {
        let mut xa = XArray::new(make_2d(), &["obs", "feat"]);
        xa.assign_coords("feat", Array::from_string32(
            StringArray::from_slice(&["height", "weight"])
        ));
        let table = xa.to_table().unwrap();
        assert_eq!(table.col_names(), vec!["height", "weight"]);
    }

    // *** Misc ********************************************************

    #[test]
    fn clone_and_eq() {
        let xa = XArray::new(make_2d(), &["obs", "feat"]);
        assert_eq!(xa, xa.clone());
    }

    #[test]
    fn debug_format() {
        let mut xa = XArray::new(make_2d(), &["obs", "feat"]);
        xa.assign_coords("feat", float_coords(&[10.0, 20.0]));
        let s = format!("{:?}", xa);
        assert!(s.contains("obs=3"));
        assert!(s.contains("feat=2 (labelled)"));
        assert!(s.contains("owned"));
    }

    #[cfg(feature = "views")]
    #[test]
    fn select_produces_view_debug() {
        let xa = XArray::new(make_2d(), &["obs", "feat"]);
        let result = xa.select(&[("obs", &(0..2))]);
        assert!(!result.is_owned());
        assert!(format!("{:?}", result).contains("view"));
    }

    #[test]
    fn to_owned_from_view() {
        #[cfg(feature = "views")]
        {
            let xa = XArray::new(make_2d(), &["obs", "feat"]);
            let view = xa.select(&[("obs", &(0..2))]);
            assert!(!view.is_owned());
            let owned = view.to_owned();
            assert!(owned.is_owned());
            assert_eq!(owned.shape(), &[2, 2]);
        }
    }

    #[test]
    fn three_dimensional() {
        let data = NdArray::from_slice(
            &(1..=24).map(|x| x as f64).collect::<Vec<_>>(),
            &[2, 3, 4],
        );
        let xa = XArray::new(data, &["obs", "feat", "time"]);
        assert_eq!(xa.ndim(), 3);
        assert_eq!(xa.dim("time"), 2);
    }
}
