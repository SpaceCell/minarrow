//! # `XArray`
//!
//! A labelled N-dimensional array with named axes and optional coordinates.
//!
//! `XArray` associates an [`NdArray`] with dimension names and, optionally,
//! coordinate labels for each axis. It supports both positional selection and
//! coordinate-based queries without changing the underlying numerical model.
//!
//! Where `NdArray` addresses values by position, such as `[3, 7]`, `XArray`
//! can address them using domain values such as latitude `51.5`, ticker `"NQ"`,
//! or timestamps within a specified interval.
//!
//! This representation is well-suited to spatial, climate, sensor, time-series, and
//! other datasets whose dimensions carry semantic meaning.
//!
//! ## Storage
//!
//! An `XArray` contains either an owned [`NdArray`] or a zero-copy [`NdArrayV`]
//! view behind the same public interface. Selections can therefore remain
//! labelled `XArray` values without materialising or copying the selected data.
//!
//! ## Usage
//!
//! `ignore
//! let xa = XArray::new(data, &["observation", "feature"]);
//!
//! xa.ax("feature");                       // Return axis metadata.
//! xa.dim("feature");                      // Resolve the axis position: 1.
//! xa.at("feature", 2.0);                  // Select a numeric coordinate.
//! xa.at("ticker", "NQ");                  // Select a string coordinate.
//! xa.at("time", 1_700_000_000_000i64);   // Select a datetime tick.
//! xa.nearest("time", ts);                 // Select the closest coordinate.
//! xa.between("observation", 0.0, 50.0);   // Select a coordinate interval.
//! xa.select(&[("lat", &(0..3))]);         // Select by positional range.
//! `

use std::fmt;

use crate::enums::error::MinarrowError;
use crate::enums::shape_dim::ShapeDim;
use crate::ffi::arrow_dtype::ArrowType;
use crate::structs::ndarray::NdArray;
#[cfg(all(feature = "views", feature = "select"))]
use crate::traits::selection::{AxisSelection, DataSelector};
#[cfg(all(feature = "views", feature = "select", feature = "scalar_type"))]
use crate::Scalar;
#[cfg(all(feature = "views", feature = "select", feature = "scalar_type"))]
use crate::{NumericArray, TextArray};
#[cfg(all(
    feature = "views",
    feature = "select",
    feature = "scalar_type",
    feature = "datetime"
))]
use crate::TemporalArray;
#[cfg(all(feature = "views", feature = "select"))]
use std::ops::Range;
use crate::traits::type_unions::Float;
use crate::traits::{concatenate::Concatenate, shape::Shape};
use crate::{Array, Field, StringArray, Table};

#[cfg(feature = "views")]
use crate::structs::views::ndarray_view::NdArrayV;

// ****************************************************************
// Dispatch macros
// ****************************************************************

/// Dispatch to the inner NdArray/NdArrayV.
#[cfg(feature = "views")]
macro_rules! delegate {
    ($self:expr, $method:ident ( $($arg:expr),* )) => {
        match &$self.data {
            NdArrayE::Owned(nd) => nd.$method($($arg),*),
            NdArrayE::View(v) => v.$method($($arg),*),
        }
    };
}

#[cfg(not(feature = "views"))]
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
/// XArray wraps an [`NdArray`] or [`NdArrayV`] with per-axis names and optional
/// coordinate labels, enabling selection by value rather than raw position.
/// Owned data and selection views expose the same container interface.
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

    /// Wrap an NdArrayV view with axes (zero-copy).
    #[cfg(feature = "views")]
    pub fn from_view(view: NdArrayV<T>, axes: Vec<Axis>) -> Self {
        assert_eq!(
            view.ndim(), axes.len(),
            "XArray: {} axes for {}D view", axes.len(), view.ndim()
        );
        for (i, ax) in axes.iter().enumerate() {
            if let Some(ref coords) = ax.coords {
                assert_eq!(
                    coords.len(), view.shape()[i],
                    "XArray: axis '{}' has {} coords but dimension size is {}",
                    ax.name, coords.len(), view.shape()[i]
                );
            }
        }
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

    /// Consume and return the inner NdArray, materialising if it is a view.
    pub fn into_ndarray(self) -> NdArray<T> {
        match self.data {
            NdArrayE::Owned(nd) => nd,
            #[cfg(feature = "views")]
            NdArrayE::View(v) => v.to_ndarray(),
        }
    }

    /// Materialise to an owned NdArray if currently a view.
    pub fn to_owned(&self) -> XArray<T> {
        match &self.data {
            NdArrayE::Owned(_) => self.clone(),
            #[cfg(feature = "views")]
            NdArrayE::View(v) => XArray {
                data: NdArrayE::Owned(v.to_ndarray()),
                axes: self.axes.clone(),
            },
        }
    }

    /// Borrow the data as a zero-copy NdArrayV, regardless of whether this
    /// XArray currently owns its NdArray or already holds a view.
    #[cfg(feature = "views")]
    pub fn as_view(&self) -> NdArrayV<T> {
        match &self.data {
            NdArrayE::Owned(nd) => nd.as_view(),
            NdArrayE::View(v) => v.clone(),
        }
    }

    /// True if backed by a single owned NdArray.
    pub fn is_owned(&self) -> bool {
        matches!(&self.data, NdArrayE::Owned(_))
    }

    /// Borrow the inner storage for crate-internal dispatch.
    #[cfg(feature = "size")]
    #[inline]
    pub(crate) fn storage(&self) -> &NdArrayE<T> {
        &self.data
    }

    #[inline]
    pub fn ndim(&self) -> usize { delegate!(self, ndim()) }

    pub fn shape(&self) -> Vec<usize> {
        match &self.data {
            NdArrayE::Owned(nd) => nd.shape().to_vec(),
            #[cfg(feature = "views")]
            NdArrayE::View(v) => v.shape().to_vec(),
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

    /// Mutable element access. Triggers copy-on-write if the owned array is
    /// shared with views; an XArray backed by a view cannot be mutated.
    pub fn set(&mut self, indices: &[usize], value: T) {
        match &mut self.data {
            NdArrayE::Owned(nd) => nd.set(indices, value),
            #[cfg(feature = "views")]
            NdArrayE::View(_) => panic!("XArray: cannot mutate a view"),
        }
    }

    /// Parallel iterator over axis-0 observations. Each item is the
    /// observation index and a zero-copy `NdArrayV` view.
    #[cfg(all(feature = "parallel_proc", feature = "views"))]
    pub fn par_iter_obs(&self) -> impl rayon::iter::ParallelIterator<Item = (usize, NdArrayV<T>)> + '_
    where
        T: Send + Sync,
    {
        use rayon::prelude::*;
        assert!(self.ndim() >= 2, "par_iter_obs() requires a 2D or higher array");
        let n_obs = self.shape()[0];
        (0..n_obs).into_par_iter().map(move |i| (i, self.obs(i)))
    }

    // ****************************************************************
    // Apply
    // ****************************************************************

    /// Apply a function to every logical element, returning a new labelled
    /// array with the same axes. View storage materialises to owned.
    pub fn apply(&self, f: impl Fn(T) -> T) -> XArray<T> {
        let data = match &self.data {
            NdArrayE::Owned(nd) => NdArrayE::Owned(nd.apply(f)),
            #[cfg(feature = "views")]
            NdArrayE::View(v) => NdArrayE::Owned(v.apply(f)),
        };
        XArray { data, axes: self.axes.clone() }
    }

    /// Apply a function to every logical element in place. An XArray backed
    /// by a view cannot be mutated, matching `set`.
    pub fn apply_mut(&mut self, f: impl Fn(T) -> T) {
        match &mut self.data {
            NdArrayE::Owned(nd) => nd.apply_mut(f),
            #[cfg(feature = "views")]
            NdArrayE::View(_) => panic!("XArray: cannot mutate a view"),
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

    /// Slice the underlying NdArray/NdArrayV positionally and zero-copy.
    /// For named axis selection, use `.select()` instead.
    #[cfg(all(feature = "views", feature = "select"))]
    pub fn slice(&self, selection: &[&dyn DataSelector]) -> NdArrayV<T> {
        match &self.data {
            NdArrayE::Owned(nd) => nd.slice(selection),
            NdArrayE::View(v) => v.slice(selection),
        }
    }

    // ****************************************************************
    // Coordinate value selection: .at() and .between()
    // ****************************************************************

    /// Select a single position on a named axis by coordinate value.
    /// Accepts numeric, string, and datetime values. Collapses that
    /// dimension. Returns an error if the value is not found. Float
    /// coordinates match by IEEE equality, so NaN never matches and
    /// derived values may miss - `nearest` tolerates rounding.
    #[cfg(all(feature = "views", feature = "select", feature = "scalar_type"))]
    pub fn try_at(&self, dim_name: &str, value: impl Into<Scalar>) -> Result<XArray<T>, MinarrowError> {
        let dim_idx = self.dim(dim_name);
        let pos = self.axes[dim_idx].try_coord_pos(&value.into())?;

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
    #[cfg(all(feature = "views", feature = "select", feature = "scalar_type"))]
    pub fn at(&self, dim_name: &str, value: impl Into<Scalar>) -> XArray<T> {
        self.try_at(dim_name, value)
            .unwrap_or_else(|e| panic!("{}", e))
    }

    /// Select a range by coordinate value bounds (inclusive).
    /// Accepts numeric, string, and datetime bounds. Returns an error
    /// if no values fall in the range, or if the matching coordinates
    /// do not form a contiguous run i.e. the axis is not monotonic over
    /// the requested bounds - sort the axis or gather by position for
    /// unsorted coordinates.
    #[cfg(all(feature = "views", feature = "select", feature = "scalar_type"))]
    pub fn try_between(
        &self,
        dim_name: &str,
        low: impl Into<Scalar>,
        high: impl Into<Scalar>,
    ) -> Result<XArray<T>, MinarrowError> {
        let dim_idx = self.dim(dim_name);
        let (start, end) = self.axes[dim_idx].try_coord_range(&low.into(), &high.into())?;

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

    /// Select a range by coordinate value bounds. Panics if no values
    /// match, or if the axis is not monotonic over the requested bounds.
    #[cfg(all(feature = "views", feature = "select", feature = "scalar_type"))]
    pub fn between(
        &self,
        dim_name: &str,
        low: impl Into<Scalar>,
        high: impl Into<Scalar>,
    ) -> XArray<T> {
        self.try_between(dim_name, low, high)
            .unwrap_or_else(|e| panic!("{}", e))
    }

    /// Select the position whose coordinate is closest to `value` on a
    /// named axis. Collapses that dimension. Numeric and datetime
    /// coordinates only, since text has no distance metric. Returns an
    /// error if the axis has no comparable coordinates.
    #[cfg(all(feature = "views", feature = "select", feature = "scalar_type"))]
    pub fn try_nearest(&self, dim_name: &str, value: impl Into<Scalar>) -> Result<XArray<T>, MinarrowError> {
        let dim_idx = self.dim(dim_name);
        let pos = self.axes[dim_idx].try_coord_nearest(&value.into())?;

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

    /// Select the position whose coordinate is closest to `value`.
    /// Panics if the axis has no comparable coordinates.
    #[cfg(all(feature = "views", feature = "select", feature = "scalar_type"))]
    pub fn nearest(&self, dim_name: &str, value: impl Into<Scalar>) -> XArray<T> {
        self.try_nearest(dim_name, value)
            .unwrap_or_else(|e| panic!("{}", e))
    }

    // ****************************************************************
    // Axis operations
    // ****************************************************************

    /// Transpose (2D only). Reorders axes by name, so the result's
    /// dimensions arrive in `dim_order`. Passing the current order
    /// returns the array unchanged.
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
        if dim_order[0] == dim_order[1] {
            return Err(MinarrowError::ShapeError {
                message: format!(
                    "transpose: dim names must be distinct, got '{}' twice", dim_order[0]
                ),
            });
        }
        for name in dim_order {
            if self.try_dim(name).is_none() {
                return Err(MinarrowError::ShapeError {
                    message: format!(
                        "transpose: no axis named '{}', axes are {:?}",
                        name,
                        self.dim_names()
                    ),
                });
            }
        }
        // The current order is a no-op, so the data only transposes when
        // the axes actually swap.
        if dim_order[0] == self.axes[0].name {
            return Ok(self.clone());
        }
        let inner = match &self.data {
            NdArrayE::Owned(nd) => nd.transpose(),
            #[cfg(feature = "views")]
            NdArrayE::View(v) => v.transpose().to_ndarray(),
        };
        let new_axes = dim_order.iter()
            .map(|name| self.ax(name).clone())
            .collect();
        Ok(XArray { data: NdArrayE::Owned(inner), axes: new_axes })
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
        self.into_ndarray().to_table(Some(fields))
    }
}

// ****************************************************************
// NdArrayEnum - owned or view storage
// ****************************************************************

/// Either owned NdArray or zero-copy NdArrayV.
/// Enables XArray to be a single type regardless of ownership.
#[derive(Clone)]
pub(crate) enum NdArrayE<T> {
    /// The array's own shared buffer makes clones a refcount bump and
    /// lets views borrow the parent zero-copy, with copy-on-write
    /// mutation.
    Owned(NdArray<T>),
    #[cfg(feature = "views")]
    View(NdArrayV<T>),
}

// ****************************************************************
// Axis
// ****************************************************************

/// A named dimension with optional coordinate labels.
///
/// The coords array, when present, must have the same length as the
/// corresponding NdArray dimension. Coordinates may be stored as any
/// Minarrow Array type. Value-based selection resolves numeric, string,
/// and datetime coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct Axis {
    pub name: String,
    pub coords: Option<Array>,
}

/// Scans a typed coordinate slice, widening `start`/`end` over positions
/// within the inclusive `[lo, hi]` window and counting the matches, so the
/// caller can detect a span polluted by out-of-range coordinates.
#[cfg(all(feature = "views", feature = "select", feature = "scalar_type"))]
macro_rules! coord_window {
    ($data:expr, $lo:expr, $hi:expr, $domain:ty, $start:ident, $end:ident, $count:ident) => {
        for (i, &v) in $data.iter().enumerate() {
            if (v as $domain) >= $lo && (v as $domain) <= $hi {
                if i < $start { $start = i; }
                if i + 1 > $end { $end = i + 1; }
                $count += 1;
            }
        }
    };
}

/// Scans string-valued coordinates, widening `start`/`end` over positions
/// within the inclusive lexicographic `[lo, hi]` window and counting the
/// matches, so the caller can detect a span polluted by out-of-range
/// coordinates.
#[cfg(all(feature = "views", feature = "select", feature = "scalar_type"))]
macro_rules! coord_window_str {
    ($arr:expr, $n:expr, $lo:expr, $hi:expr, $start:ident, $end:ident, $count:ident) => {
        for i in 0..$n {
            if let Some(s) = $arr.get_str(i) {
                if s >= $lo && s <= $hi {
                    if i < $start { $start = i; }
                    if i + 1 > $end { $end = i + 1; }
                    $count += 1;
                }
            }
        }
    };
}

/// Scans a typed coordinate slice for the position with the smallest
/// distance to the target. Ties resolve to the earliest position.
#[cfg(all(feature = "views", feature = "select", feature = "scalar_type"))]
macro_rules! coord_nearest {
    ($data:expr, |$v:ident| $dist:expr) => {{
        let mut best = None;
        for (i, &$v) in $data.iter().enumerate() {
            let dist = $dist;
            if best.is_none_or(|(_, bd)| dist < bd) {
                best = Some((i, dist));
            }
        }
        best.map(|(i, _)| i)
    }};
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

    /// Position of a coordinate value on this axis. String and
    /// categorical coordinates match by string equality, datetime
    /// coordinates by tick value in the axis's time unit, and numeric
    /// coordinates in their native domain (i64, u64, or f64).
    #[cfg(all(feature = "views", feature = "select", feature = "scalar_type"))]
    pub(crate) fn try_coord_pos(&self, value: &Scalar) -> Result<usize, MinarrowError> {
        let coords = self.coords.as_ref().ok_or_else(|| MinarrowError::ShapeError {
            message: format!("axis '{}' has no coordinates for value lookup", self.name),
        })?;
        let n = coords.len();
        let found = match coords {
            Array::TextArray(text) => {
                let target = value.try_str().ok_or_else(|| MinarrowError::TypeError {
                    from: "scalar", to: "string",
                    message: Some(format!("axis '{}' holds text coordinates", self.name)),
                })?;
                let target = target.as_str();
                match text {
                    TextArray::String32(a) => (0..n).find(|&i| a.get_str(i) == Some(target)),
                    #[cfg(feature = "large_string")]
                    TextArray::String64(a) => (0..n).find(|&i| a.get_str(i) == Some(target)),
                    #[cfg(feature = "default_categorical_8")]
                    TextArray::Categorical8(a) => (0..n).find(|&i| a.get_str(i) == Some(target)),
                    #[cfg(feature = "extended_categorical")]
                    TextArray::Categorical16(a) => (0..n).find(|&i| a.get_str(i) == Some(target)),
                    #[cfg(any(
                        not(feature = "default_categorical_8"),
                        feature = "extended_categorical"
                    ))]
                    TextArray::Categorical32(a) => (0..n).find(|&i| a.get_str(i) == Some(target)),
                    #[cfg(feature = "extended_categorical")]
                    TextArray::Categorical64(a) => (0..n).find(|&i| a.get_str(i) == Some(target)),
                    TextArray::Null => None,
                }
            }
            Array::NumericArray(num) => match num {
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(a) =>
                    value.try_i64().and_then(|t| a.data.iter().position(|&v| v as i64 == t)),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(a) =>
                    value.try_i64().and_then(|t| a.data.iter().position(|&v| v as i64 == t)),
                NumericArray::Int32(a) =>
                    value.try_i64().and_then(|t| a.data.iter().position(|&v| v as i64 == t)),
                NumericArray::Int64(a) =>
                    value.try_i64().and_then(|t| a.data.iter().position(|&v| v == t)),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(a) =>
                    value.try_u64().and_then(|t| a.data.iter().position(|&v| v as u64 == t)),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(a) =>
                    value.try_u64().and_then(|t| a.data.iter().position(|&v| v as u64 == t)),
                NumericArray::UInt32(a) =>
                    value.try_u64().and_then(|t| a.data.iter().position(|&v| v as u64 == t)),
                NumericArray::UInt64(a) =>
                    value.try_u64().and_then(|t| a.data.iter().position(|&v| v == t)),
                NumericArray::Float32(a) =>
                    value.try_f64().and_then(|t| a.data.iter().position(|&v| v as f64 == t)),
                NumericArray::Float64(a) =>
                    value.try_f64().and_then(|t| a.data.iter().position(|&v| v == t)),
                NumericArray::Null => None,
            },
            #[cfg(feature = "datetime")]
            Array::TemporalArray(temporal) => {
                let target = value.try_i64().ok_or_else(|| MinarrowError::TypeError {
                    from: "scalar", to: "datetime ticks",
                    message: Some(format!("axis '{}' holds datetime coordinates", self.name)),
                })?;
                match temporal {
                    TemporalArray::Datetime32(a) => a.data.iter().position(|&t| t as i64 == target),
                    TemporalArray::Datetime64(a) => a.data.iter().position(|&t| t == target),
                    TemporalArray::Null => None,
                }
            }
            _ => return Err(MinarrowError::TypeError {
                from: "coordinate array", to: "comparable value",
                message: Some(format!(
                    "axis '{}' coordinates do not support value lookup", self.name
                )),
            }),
        };
        found.ok_or_else(|| MinarrowError::IndexError(format!(
            "value {:?} not found on axis '{}'", value, self.name
        )))
    }

    /// Start and end positions covering all coordinates within the
    /// inclusive `[low, high]` bounds. String and categorical coordinates
    /// compare lexicographically, datetime coordinates by tick value, and
    /// numeric coordinates in their native domain (i64, u64, or f64).
    #[cfg(all(feature = "views", feature = "select", feature = "scalar_type"))]
    pub(crate) fn try_coord_range(
        &self,
        low: &Scalar,
        high: &Scalar,
    ) -> Result<(usize, usize), MinarrowError> {
        let coords = self.coords.as_ref().ok_or_else(|| MinarrowError::ShapeError {
            message: format!("axis '{}' has no coordinates for range lookup", self.name),
        })?;
        let n = coords.len();
        let mut start = n;
        let mut end = 0;
        let mut count = 0usize;
        match coords {
            Array::TextArray(text) => {
                let lo = low.try_str().ok_or_else(|| MinarrowError::TypeError {
                    from: "scalar", to: "string",
                    message: Some(format!("axis '{}' holds text coordinates", self.name)),
                })?;
                let hi = high.try_str().ok_or_else(|| MinarrowError::TypeError {
                    from: "scalar", to: "string",
                    message: Some(format!("axis '{}' holds text coordinates", self.name)),
                })?;
                let (lo, hi) = (lo.as_str(), hi.as_str());
                match text {
                    TextArray::String32(a) => { coord_window_str!(a, n, lo, hi, start, end, count); }
                    #[cfg(feature = "large_string")]
                    TextArray::String64(a) => { coord_window_str!(a, n, lo, hi, start, end, count); }
                    #[cfg(feature = "default_categorical_8")]
                    TextArray::Categorical8(a) => { coord_window_str!(a, n, lo, hi, start, end, count); }
                    #[cfg(feature = "extended_categorical")]
                    TextArray::Categorical16(a) => { coord_window_str!(a, n, lo, hi, start, end, count); }
                    #[cfg(any(
                        not(feature = "default_categorical_8"),
                        feature = "extended_categorical"
                    ))]
                    TextArray::Categorical32(a) => { coord_window_str!(a, n, lo, hi, start, end, count); }
                    #[cfg(feature = "extended_categorical")]
                    TextArray::Categorical64(a) => { coord_window_str!(a, n, lo, hi, start, end, count); }
                    TextArray::Null => {}
                }
            }
            Array::NumericArray(num) => match num {
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(a) => if let (Some(lo), Some(hi)) = (low.try_i64(), high.try_i64()) {
                    coord_window!(a.data, lo, hi, i64, start, end, count);
                },
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(a) => if let (Some(lo), Some(hi)) = (low.try_i64(), high.try_i64()) {
                    coord_window!(a.data, lo, hi, i64, start, end, count);
                },
                NumericArray::Int32(a) => if let (Some(lo), Some(hi)) = (low.try_i64(), high.try_i64()) {
                    coord_window!(a.data, lo, hi, i64, start, end, count);
                },
                NumericArray::Int64(a) => if let (Some(lo), Some(hi)) = (low.try_i64(), high.try_i64()) {
                    coord_window!(a.data, lo, hi, i64, start, end, count);
                },
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(a) => if let (Some(lo), Some(hi)) = (low.try_u64(), high.try_u64()) {
                    coord_window!(a.data, lo, hi, u64, start, end, count);
                },
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(a) => if let (Some(lo), Some(hi)) = (low.try_u64(), high.try_u64()) {
                    coord_window!(a.data, lo, hi, u64, start, end, count);
                },
                NumericArray::UInt32(a) => if let (Some(lo), Some(hi)) = (low.try_u64(), high.try_u64()) {
                    coord_window!(a.data, lo, hi, u64, start, end, count);
                },
                NumericArray::UInt64(a) => if let (Some(lo), Some(hi)) = (low.try_u64(), high.try_u64()) {
                    coord_window!(a.data, lo, hi, u64, start, end, count);
                },
                NumericArray::Float32(a) => if let (Some(lo), Some(hi)) = (low.try_f64(), high.try_f64()) {
                    coord_window!(a.data, lo, hi, f64, start, end, count);
                },
                NumericArray::Float64(a) => if let (Some(lo), Some(hi)) = (low.try_f64(), high.try_f64()) {
                    coord_window!(a.data, lo, hi, f64, start, end, count);
                },
                NumericArray::Null => {}
            },
            #[cfg(feature = "datetime")]
            Array::TemporalArray(temporal) => {
                let lo = low.try_i64().ok_or_else(|| MinarrowError::TypeError {
                    from: "scalar", to: "datetime ticks",
                    message: Some(format!("axis '{}' holds datetime coordinates", self.name)),
                })?;
                let hi = high.try_i64().ok_or_else(|| MinarrowError::TypeError {
                    from: "scalar", to: "datetime ticks",
                    message: Some(format!("axis '{}' holds datetime coordinates", self.name)),
                })?;
                match temporal {
                    TemporalArray::Datetime32(a) => { coord_window!(a.data, lo, hi, i64, start, end, count); }
                    TemporalArray::Datetime64(a) => { coord_window!(a.data, lo, hi, i64, start, end, count); }
                    TemporalArray::Null => {}
                }
            }
            _ => return Err(MinarrowError::TypeError {
                from: "coordinate array", to: "comparable value",
                message: Some(format!(
                    "axis '{}' coordinates do not support range lookup", self.name
                )),
            }),
        }
        if start >= end {
            return Err(MinarrowError::IndexError(format!(
                "no values in [{:?}, {:?}] on axis '{}'", low, high, self.name
            )));
        }
        // A span wider than its match count contains coordinates outside
        // the bounds, so a window would return wrong rows.
        if end - start != count {
            return Err(MinarrowError::IndexError(format!(
                "coordinates on axis '{}' are not monotonic over [{:?}, {:?}], sort the axis or gather by position instead",
                self.name, low, high
            )));
        }
        Ok((start, end))
    }

    /// Position of the coordinate closest to `value`. Numeric coordinates
    /// measure distance in their native domain (i64, u64, or f64), and
    /// datetime coordinates as tick difference. String and categorical
    /// coordinates have no distance metric and return an error - use
    /// exact lookup instead. Ties resolve to the earliest position.
    #[cfg(all(feature = "views", feature = "select", feature = "scalar_type"))]
    pub(crate) fn try_coord_nearest(&self, value: &Scalar) -> Result<usize, MinarrowError> {
        let coords = self.coords.as_ref().ok_or_else(|| MinarrowError::ShapeError {
            message: format!("axis '{}' has no coordinates for nearest lookup", self.name),
        })?;
        let found = match coords {
            Array::TextArray(_) => return Err(MinarrowError::TypeError {
                from: "text", to: "numeric",
                message: Some(format!(
                    "axis '{}' holds text coordinates with no distance metric - use at",
                    self.name
                )),
            }),
            Array::NumericArray(num) => match num {
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(a) =>
                    value.try_i64().and_then(|t| coord_nearest!(a.data, |v| (v as i64).abs_diff(t))),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(a) =>
                    value.try_i64().and_then(|t| coord_nearest!(a.data, |v| (v as i64).abs_diff(t))),
                NumericArray::Int32(a) =>
                    value.try_i64().and_then(|t| coord_nearest!(a.data, |v| (v as i64).abs_diff(t))),
                NumericArray::Int64(a) =>
                    value.try_i64().and_then(|t| coord_nearest!(a.data, |v| v.abs_diff(t))),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(a) =>
                    value.try_u64().and_then(|t| coord_nearest!(a.data, |v| (v as u64).abs_diff(t))),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(a) =>
                    value.try_u64().and_then(|t| coord_nearest!(a.data, |v| (v as u64).abs_diff(t))),
                NumericArray::UInt32(a) =>
                    value.try_u64().and_then(|t| coord_nearest!(a.data, |v| (v as u64).abs_diff(t))),
                NumericArray::UInt64(a) =>
                    value.try_u64().and_then(|t| coord_nearest!(a.data, |v| v.abs_diff(t))),
                NumericArray::Float32(a) =>
                    value.try_f64().and_then(|t| coord_nearest!(a.data, |v| (v as f64 - t).abs())),
                NumericArray::Float64(a) =>
                    value.try_f64().and_then(|t| coord_nearest!(a.data, |v| (v - t).abs())),
                NumericArray::Null => None,
            },
            #[cfg(feature = "datetime")]
            Array::TemporalArray(temporal) => {
                let target = value.try_i64().ok_or_else(|| MinarrowError::TypeError {
                    from: "scalar", to: "datetime ticks",
                    message: Some(format!("axis '{}' holds datetime coordinates", self.name)),
                })?;
                match temporal {
                    TemporalArray::Datetime32(a) =>
                        coord_nearest!(a.data, |t| (t as i64).abs_diff(target)),
                    TemporalArray::Datetime64(a) =>
                        coord_nearest!(a.data, |t| t.abs_diff(target)),
                    TemporalArray::Null => None,
                }
            }
            _ => return Err(MinarrowError::TypeError {
                from: "coordinate array", to: "comparable value",
                message: Some(format!(
                    "axis '{}' coordinates do not support nearest lookup", self.name
                )),
            }),
        };
        found.ok_or_else(|| MinarrowError::IndexError(format!(
            "axis '{}' has no comparable coordinates", self.name
        )))
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
            }
        };
        let self_nd = to_ndarray(self_data);
        let other_nd = to_ndarray(other_data);
        let new_data = self_nd.concat(other_nd)?;

        // Axis 0 coords concatenate when both sides carry them. Non-0
        // axes describe the same positions on both sides, so their
        // coords must agree, and a side missing them adopts the other's.
        let mut new_axes = Vec::with_capacity(self_axes.len());
        for (i, (a, b)) in self_axes.into_iter().zip(other_axes.into_iter()).enumerate() {
            if i == 0 {
                let merged_coords = match (a.coords, b.coords) {
                    (Some(ca), Some(cb)) => Some(ca.concat(cb)?),
                    (None, None) => None,
                    _ => {
                        return Err(MinarrowError::IncompatibleTypeError {
                            from: "XArray",
                            to: "XArray",
                            message: Some(format!(
                                "axis '{}' is labelled on one side only, assign_coords or drop_coords first",
                                a.name
                            )),
                        });
                    }
                };
                new_axes.push(Axis { name: a.name, coords: merged_coords });
            } else {
                let coords = match (a.coords, b.coords) {
                    (Some(ca), Some(cb)) => {
                        if ca != cb {
                            return Err(MinarrowError::IncompatibleTypeError {
                                from: "XArray",
                                to: "XArray",
                                message: Some(format!(
                                    "axis '{}' coordinates differ between the two sides",
                                    a.name
                                )),
                            });
                        }
                        Some(ca)
                    }
                    (Some(ca), None) => Some(ca),
                    (None, Some(cb)) => Some(cb),
                    (None, None) => None,
                };
                new_axes.push(Axis { name: a.name, coords });
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
        let storage = match &self.data {
            NdArrayE::Owned(_) => "owned",
            #[cfg(feature = "views")]
            NdArrayE::View(_) => "view",
        };
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
    fn rank_zero_scalar() {
        let xa = XArray::new(NdArray::from_slice(&[5.0], &[]), &[]);
        assert_eq!(xa.ndim(), 0);
        assert_eq!(xa.shape(), Vec::<usize>::new());
        assert!(xa.dim_names().is_empty());
        assert_eq!(xa.len(), 1);
        assert_eq!(xa.get(&[]), 5.0);
    }

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

    #[cfg(all(feature = "views", feature = "select", feature = "scalar_type"))]
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

    #[cfg(all(feature = "views", feature = "select", feature = "scalar_type"))]
    #[test]
    fn try_at_not_found() {
        let mut xa = XArray::new(make_2d(), &["obs", "feat"]);
        xa.assign_coords("feat", float_coords(&[10.0, 20.0]));
        assert!(xa.try_at("feat", 99.0).is_err());
    }

    #[cfg(all(feature = "views", feature = "select", feature = "scalar_type"))]
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

    #[cfg(all(feature = "views", feature = "select", feature = "scalar_type"))]
    #[test]
    fn try_between_empty_range() {
        let mut xa = XArray::new(make_2d(), &["obs", "feat"]);
        xa.assign_coords("obs", float_coords(&[0.0, 1.0, 2.0]));
        assert!(xa.try_between("obs", 100.0, 200.0).is_err());
    }

    #[cfg(all(feature = "views", feature = "select", feature = "scalar_type"))]
    #[test]
    fn at_string_coords() {
        use crate::arr_str32;
        let mut xa = XArray::new(make_2d(), &["ticker", "feat"]);
        xa.assign_coords("ticker", arr_str32!(&["ES", "NQ", "CL"]));
        let result = xa.at("ticker", "NQ");
        assert_eq!(result.ndim(), 1);
        assert_eq!(result.dim_names(), vec!["feat"]);
        let vals: Vec<f64> = (&result).into_iter().collect();
        assert_eq!(vals, vec![2.0, 5.0]);
        assert!(xa.try_at("ticker", "ZB").is_err());
    }

    #[cfg(all(
        feature = "views",
        feature = "select",
        feature = "scalar_type",
        any(not(feature = "default_categorical_8"), feature = "extended_categorical")
    ))]
    #[test]
    fn at_categorical_coords() {
        use crate::arr_cat32;
        let mut xa = XArray::new(make_2d(), &["ticker", "feat"]);
        xa.assign_coords("ticker", arr_cat32!(&["ES", "NQ", "ES"]));
        let result = xa.at("ticker", "NQ");
        assert_eq!(result.ndim(), 1);
        let vals: Vec<f64> = (&result).into_iter().collect();
        assert_eq!(vals, vec![2.0, 5.0]);
        assert!(xa.try_at("ticker", "ZB").is_err());
    }

    #[cfg(all(feature = "views", feature = "select", feature = "scalar_type"))]
    #[test]
    fn coord_lookup_integer_coords() {
        use crate::arr_i64;
        let mut xa = XArray::new(make_2d(), &["obs", "feat"]);
        xa.assign_coords("obs", arr_i64![100, 200, 400]);
        let result = xa.at("obs", 200);
        let vals: Vec<f64> = (&result).into_iter().collect();
        assert_eq!(vals, vec![2.0, 5.0]);
        let ranged = xa.between("obs", 150, 450);
        assert_eq!(ranged.shape(), vec![2, 2]);
        let near = xa.nearest("obs", 230);
        let vals: Vec<f64> = (&near).into_iter().collect();
        assert_eq!(vals, vec![2.0, 5.0]);
    }

    #[cfg(all(feature = "views", feature = "select", feature = "scalar_type"))]
    #[test]
    fn between_string_coords() {
        use crate::arr_str32;
        let mut xa = XArray::new(make_2d(), &["ticker", "feat"]);
        xa.assign_coords("ticker", arr_str32!(&["alpha", "bravo", "charlie"]));
        // Lexicographic inclusive bounds.
        let result = xa.between("ticker", "bravo", "delta");
        assert_eq!(result.shape(), vec![2, 2]);
        let vals: Vec<f64> = (&result).into_iter().take(2).collect();
        assert_eq!(vals, vec![2.0, 3.0]);
    }

    #[cfg(all(
        feature = "views",
        feature = "select",
        feature = "scalar_type",
        feature = "datetime"
    ))]
    #[test]
    fn at_datetime_coords() {
        use crate::arr_dt64;
        use crate::enums::time_units::TimeUnit;
        let mut xa = XArray::new(make_2d(), &["time", "feat"]);
        xa.assign_coords("time", arr_dt64!(TimeUnit::Milliseconds; 1_000, 2_000, 3_000));
        let result = xa.at("time", 2_000);
        assert_eq!(result.ndim(), 1);
        let vals: Vec<f64> = (&result).into_iter().collect();
        assert_eq!(vals, vec![2.0, 5.0]);
        assert!(xa.try_at("time", 5_000).is_err());
    }

    #[cfg(all(
        feature = "views",
        feature = "select",
        feature = "scalar_type",
        feature = "datetime"
    ))]
    #[test]
    fn between_datetime_coords() {
        use crate::arr_dt64;
        use crate::enums::time_units::TimeUnit;
        let mut xa = XArray::new(make_2d(), &["time", "feat"]);
        xa.assign_coords("time", arr_dt64!(TimeUnit::Milliseconds; 1_000, 2_000, 3_000));
        let result = xa.between("time", 1_500, 3_500);
        assert_eq!(result.shape(), vec![2, 2]);
        let vals: Vec<f64> = (&result).into_iter().take(2).collect();
        assert_eq!(vals, vec![2.0, 3.0]);
    }

    #[cfg(all(
        feature = "views",
        feature = "select",
        feature = "scalar_type",
        feature = "datetime"
    ))]
    #[test]
    fn nearest_datetime_coords() {
        use crate::arr_dt64;
        use crate::enums::time_units::TimeUnit;
        let mut xa = XArray::new(make_2d(), &["time", "feat"]);
        xa.assign_coords("time", arr_dt64!(TimeUnit::Milliseconds; 1_000, 2_000, 4_000));
        let result = xa.nearest("time", 2_700);
        let vals: Vec<f64> = (&result).into_iter().collect();
        assert_eq!(vals, vec![2.0, 5.0]);
    }

    #[cfg(all(feature = "views", feature = "select", feature = "scalar_type"))]
    #[test]
    fn nearest_numeric_coords() {
        let mut xa = XArray::new(make_2d(), &["obs", "feat"]);
        xa.assign_coords("obs", float_coords(&[10.0, 20.0, 40.0]));
        let result = xa.nearest("obs", 24.0);
        assert_eq!(result.ndim(), 1);
        let vals: Vec<f64> = (&result).into_iter().collect();
        assert_eq!(vals, vec![2.0, 5.0]);
        // Ties resolve to the earliest position.
        let tied = xa.nearest("obs", 30.0);
        let vals: Vec<f64> = (&tied).into_iter().collect();
        assert_eq!(vals, vec![2.0, 5.0]);
    }

    #[cfg(all(feature = "views", feature = "select", feature = "scalar_type"))]
    #[test]
    fn nearest_text_coords_rejected() {
        use crate::arr_str32;
        let mut xa = XArray::new(make_2d(), &["ticker", "feat"]);
        xa.assign_coords("ticker", arr_str32!(&["a", "b", "c"]));
        assert!(xa.try_nearest("ticker", "b").is_err());
    }

    #[test]
    fn axis_equality_includes_coords() {
        let a = Axis::with_coords("obs", float_coords(&[1.0, 2.0]));
        let b = Axis::with_coords("obs", float_coords(&[1.0, 2.0]));
        let c = Axis::with_coords("obs", float_coords(&[1.0, 3.0]));
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, Axis::named("obs"));
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

    #[test]
    fn transpose_current_order_is_identity() {
        let xa = XArray::new(make_2d(), &["obs", "feat"]);
        let same = xa.transpose(&["obs", "feat"]).unwrap();
        assert_eq!(same.dim_names(), vec!["obs", "feat"]);
        assert_eq!(same.shape(), vec![3, 2]);
        assert_eq!(same.get(&[2, 1]), xa.get(&[2, 1]));
    }

    #[test]
    fn transpose_reversed_moves_data() {
        let xa = XArray::new(make_2d(), &["obs", "feat"]);
        let t = xa.transpose(&["feat", "obs"]).unwrap();
        assert_eq!(t.dim_names(), vec!["feat", "obs"]);
        assert_eq!(t.get(&[0, 2]), xa.get(&[2, 0]));
        assert_eq!(t.get(&[1, 0]), xa.get(&[0, 1]));
    }

    #[test]
    fn transpose_duplicate_dim_errors() {
        let xa = XArray::new(make_2d(), &["obs", "feat"]);
        let err = xa.transpose(&["obs", "obs"]).unwrap_err();
        assert!(err.to_string().contains("must be distinct"));
    }

    #[test]
    fn transpose_unknown_dim_errors() {
        let xa = XArray::new(make_2d(), &["obs", "feat"]);
        let err = xa.transpose(&["obs", "missing"]).unwrap_err();
        assert!(err.to_string().contains("no axis named 'missing'"));
    }

    #[cfg(all(feature = "views", feature = "select", feature = "scalar_type"))]
    #[test]
    fn try_between_unsorted_coords_errors() {
        // The covering span 0..3 includes the out-of-bounds 100.0, so a
        // window would return wrong rows.
        let mut xa = XArray::new(make_2d(), &["obs", "feat"]);
        xa.assign_coords("obs", float_coords(&[20.0, 100.0, 21.0]));
        let err = xa.try_between("obs", 15.0, 25.0).unwrap_err();
        assert!(err.to_string().contains("not monotonic"));

        // Sorted coordinates over the same bounds window cleanly.
        let mut sorted = XArray::new(make_2d(), &["obs", "feat"]);
        sorted.assign_coords("obs", float_coords(&[20.0, 21.0, 100.0]));
        let result = sorted.try_between("obs", 15.0, 25.0).unwrap();
        assert_eq!(result.shape(), vec![2, 2]);
    }

    #[test]
    fn concat_axis0_coords_on_one_side_fails() {
        let mut a = XArray::new(
            NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
            &["obs", "feat"],
        );
        a.assign_coords("obs", float_coords(&[0.0, 1.0]));
        let b = XArray::new(
            NdArray::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]),
            &["obs", "feat"],
        );
        let err = a.concat(b).unwrap_err();
        assert!(err.to_string().contains("labelled on one side"));
    }

    #[test]
    fn concat_non_zero_axis_coord_mismatch_fails() {
        let mut a = XArray::new(
            NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
            &["obs", "feat"],
        );
        a.assign_coords("feat", float_coords(&[10.0, 20.0]));
        let mut b = XArray::new(
            NdArray::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]),
            &["obs", "feat"],
        );
        b.assign_coords("feat", float_coords(&[10.0, 30.0]));
        let err = a.concat(b).unwrap_err();
        assert!(err.to_string().contains("coordinates differ"));
    }

    #[test]
    fn concat_non_zero_axis_adopts_coords() {
        let mut a = XArray::new(
            NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
            &["obs", "feat"],
        );
        a.assign_coords("feat", float_coords(&[10.0, 20.0]));
        let b = XArray::new(
            NdArray::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]),
            &["obs", "feat"],
        );
        let c = a.concat(b).unwrap();
        let coords = c.ax("feat").coords.as_ref().unwrap();
        assert_eq!(*coords, float_coords(&[10.0, 20.0]));
    }

    #[cfg(feature = "views")]
    #[test]
    #[should_panic(expected = "coords but dimension size is")]
    fn from_view_coord_length_mismatch_panics() {
        let nd = make_2d();
        let axes = vec![
            Axis::named("obs"),
            Axis::with_coords("feat", float_coords(&[10.0])),
        ];
        let _ = XArray::from_view(nd.as_view(), axes);
    }

}
