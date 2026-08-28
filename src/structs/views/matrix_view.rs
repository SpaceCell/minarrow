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

//! # MatrixV - Windowed Matrix View
//!
//! `MatrixV` represents a row window over a [`Matrix`], covering rows
//! `[offset .. offset + len)` across all columns while sharing the underlying
//! column-major buffer.
//!
//! ## Purpose
//!
//! Matrix windows allow row ranges to be partitioned or processed in batches
//! without copying or repacking the stride-aligned storage required by BLAS and
//! LAPACK, which are well-established packages for numerical linear algebra routines. 
//! Each view retains the backing matrix through `Arc<Matrix>` and stores
//! only its row offset and length.
//!
//! ## Behaviour
//!
//! - Column storage remains contiguous within the selected row range.
//!   [`MatrixV::col`] returns a `&[f64]` for one column.
//! - [`MatrixV::from_self`] creates a sub-window without copying data.
//! - [`MatrixV::to_matrix`] materialises the window as a new owned [`Matrix`],
//!   including any padding required for the new row count.
//! - Columns are positional and have no names or null masks. Column selection
//!   is not supported. Use [`MatrixV::col`] for individual columns or convert
//!   to `Table` for named-column operations.
//!
//! ## Example
//!
//! ```rust
//! use minarrow::{Matrix, MatrixV, mat};
//!
//! let m = mat![[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]];
//!
//! // Rows 1..3 from both columns, sharing the backing matrix.
//! let v = MatrixV::from_matrix(m, 1, 2);
//! assert_eq!(v.n_rows(), 2);
//! assert_eq!(v.n_cols(), 2);
//! assert_eq!(v.col(0), &[2.0, 3.0]);
//! assert_eq!(v.col(1), &[20.0, 30.0]);
//!
//! // Materialise the window as an owned matrix.
//! let owned = v.to_matrix();
//! assert_eq!(owned.n_rows, 2);
//! ```
use std::fmt;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use crate::aliases::MatrixVT;
use crate::enums::error::MinarrowError;
use crate::enums::shape_dim::ShapeDim;
use crate::structs::matrix::Matrix;
use crate::traits::concatenate::Concatenate;
use crate::traits::print::print_float_grid;
use crate::traits::shape::Shape;
#[cfg(feature = "select")]
use crate::traits::selection::{DataSelector, RowSelection};

/// # MatrixView
///
/// Row window over a [`Matrix`] covering rows `[offset .. offset + len)`.
///
/// ## Fields
///
/// - `matrix`: backing matrix shared through `Arc`.
/// - `offset`: index of the first row in the backing matrix.
/// - `len`: number of rows in the window.
///
/// ## Behaviour
///
/// - Construction and re-windowing do not copy column data. Use
///   [`MatrixV::to_matrix`] to materialise the window as an independent
///   [`Matrix`].
/// - [`MatrixV::as_strided`] returns `(data, lda)` for BLAS and LAPACK, with
///   `data` positioned at the first row of the window.
/// - [`MatrixV::as_tuple`] returns `(data, offset, length, stride)` for kernels
///   that operate without the view type.
/// - Columns are positional and have no names or null masks. Column selection
///   is not supported. Use [`MatrixV::col`] for individual columns or convert
///   to `Table` for named-column operations.
#[derive(Clone)]
pub struct MatrixV {
    /// Backing matrix shared by all views over the same data.
    pub matrix: Arc<Matrix>,
    /// Index of the first row in the backing matrix.
    pub offset: usize,
    /// Number of rows in the window.
    pub len: usize,
}

impl MatrixV {
    /// Creates a view over rows `[offset .. offset + len)` of the matrix.
    ///
    /// The matrix is moved into shared `Arc` storage for this view and any
    /// subsequent views derived from it.
    #[inline]
    pub fn from_matrix(matrix: Matrix, offset: usize, len: usize) -> Self {
        Self::from_arc_matrix(Arc::new(matrix), offset, len)
    }

    /// Creates a zero-copy view over `matrix[offset .. offset+len)` rows of an
    /// already-shared matrix.    /// Creates a view over rows `[offset .. offset + len)` of a shared matrix.
    #[inline]
    pub fn from_arc_matrix(matrix: Arc<Matrix>, offset: usize, len: usize) -> Self {
        assert!(
            offset <= matrix.n_rows && len <= matrix.n_rows - offset,
            "MatrixV::from_arc_matrix: window [{}, {}) out of bounds for a matrix of {} rows",
            offset,
            offset.saturating_add(len),
            matrix.n_rows
        );
        MatrixV {
            matrix,
            offset,
            len,
        }
    }

    /// Creates a sub-window with `offset` relative to this view.
    #[inline]
    pub fn from_self(&self, offset: usize, len: usize) -> Self {
        assert!(
            offset <= self.len && len <= self.len - offset,
            "MatrixV::from_self: window [{}, {}) out of bounds for a view of {} rows",
            offset,
            offset.saturating_add(len),
            self.len
        );
        MatrixV {
            matrix: self.matrix.clone(),
            offset: self.offset + offset,
            len,
        }
    }

    /// Returns the number of rows in the window.
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.len
    }

    /// Returns the number of columns in the backing matrix.
    ///
    /// Windowing affects rows only, so every backing column remains present.
    #[inline]
    pub fn n_cols(&self) -> usize {
        self.matrix.n_cols
    }

    /// Returns the number of rows in the window.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the window has no rows or the backing matrix has no
    /// columns.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0 || self.matrix.n_cols == 0
    }

    /// Returns the exclusive end row of the window in backing-matrix
    /// coordinates.
    #[inline]
    pub fn end(&self) -> usize {
        self.offset + self.len
    }

    /// Returns `true` if the window covers every row of the backing matrix.
    ///
    /// Full-span views can be materialised by cloning the backing matrix rather
    /// than constructing a new row window.
    #[inline]
    pub fn spans_backing(&self) -> bool {
        self.offset == 0 && self.len == self.matrix.n_rows
    }

    /// Returns the backing matrix name.
    #[inline]
    pub fn name(&self) -> Option<&str> {
        self.matrix.name.as_deref()
    }

    /// Returns the backing matrix column stride in elements.
    ///
    /// This is the BLAS leading dimension (`lda`).
    #[inline]
    pub fn stride(&self) -> usize {
        self.matrix.stride
    }

    /// Returns column `col` over the current row window.
    ///
    /// The returned slice is contiguous. Panics if `col` is outside the
    /// backing matrix.
    #[inline]
    pub fn col(&self, col: usize) -> &[f64] {
        assert!(
            col < self.matrix.n_cols,
            "MatrixV::col: column {} out of bounds for {} columns",
            col,
            self.matrix.n_cols
        );
        let start = col * self.matrix.stride + self.offset;
        &self.matrix.data.as_slice()[start..start + self.len]
    }

    /// Returns all column windows in column order.
    pub fn columns(&self) -> Vec<&[f64]> {
        (0..self.matrix.n_cols).map(|c| self.col(c)).collect()
    }

    /// Returns the value at `(row, col)`.
    ///
    /// `row` is relative to the current window.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        debug_assert!(row < self.len, "Row out of bounds");
        self.matrix.get(self.offset + row, col)
    }

    /// Returns one window-relative row as an owned `Vec<f64>`.
    #[inline]
    pub fn row(&self, row: usize) -> Vec<f64> {
        debug_assert!(row < self.len, "Row out of bounds");
        self.matrix.row(self.offset + row)
    }

    /// Returns the view as `(data, offset, length, stride)`.
    ///
    /// `data` references the complete backing buffer. `offset` and `length`
    /// identify the current row window, and `stride` gives the physical spacing
    /// between columns.
    #[inline]
    pub fn as_tuple(&self) -> MatrixVT<'_> {
        (
            self.matrix.data.as_slice(),
            self.offset,
            self.len,
            self.matrix.stride,
        )
    }

    // ********************** BLAS/LAPACK Compatibility **************

    /// Returns the window row count as `i32` for BLAS and LAPACK calls.
    #[inline]
    pub fn m(&self) -> i32 {
        self.len as i32
    }

    /// Returns the column count as `i32` for BLAS and LAPACK calls.
    #[inline]
    pub fn n(&self) -> i32 {
        self.matrix.n_cols as i32
    }

    /// Returns the BLAS leading dimension (`lda`).
    ///
    /// Row windowing does not change the physical spacing between backing
    /// columns.
    #[inline]
    pub fn lda(&self) -> i32 {
        self.matrix.stride as i32
    }

    /// Returns the window as `(data, lda)` for BLAS and LAPACK.
    ///
    /// `data` begins at the first row of the window. Element `(i, j)` is located
    /// at `data[j * lda + i]` for the `m() × n()` view.
    #[inline]
    pub fn as_strided(&self) -> (&[f64], i32) {
        (
            &self.matrix.data.as_slice()[self.offset..],
            self.matrix.stride as i32,
        )
    }

    /// Returns mutable strided storage for BLAS and LAPACK routines.
    ///
    /// `Arc::make_mut` provides copy-on-write behaviour when the backing matrix
    /// is shared. Mutations affect only the resulting backing allocation after
    /// any required copy.
    #[inline]
    pub fn as_mut_strided(&mut self) -> (&mut [f64], i32) {
        let lda = self.matrix.stride as i32;
        let offset = self.offset;
        (
            &mut Arc::make_mut(&mut self.matrix).as_mut_slice()[offset..],
            lda,
        )
    }

    /// Materialises the window as an owned `Matrix`.
    ///
    /// Partial windows are copied into a new 64-byte-aligned column-major
    /// buffer with stride appropriate to the new row count. Full-span windows
    /// use the backing matrix's `Clone` implementation.
    pub fn to_matrix(&self) -> Matrix {
        if self.spans_backing() {
            return (*self.matrix).clone();
        }
        let mut out = Matrix::new(self.len, self.matrix.n_cols, self.matrix.name.clone());
        for col in 0..self.matrix.n_cols {
            out.col_mut(col).copy_from_slice(self.col(col));
        }
        out
    }

    /// Gathers window-relative rows into a new owned `Matrix`.
    ///
    /// Output rows follow the order of `indices`. Panics if any index is outside
    /// the current window.
    pub fn gather_rows(&self, indices: &[usize]) -> Matrix {
        let absolute: Vec<usize> = indices
            .iter()
            .map(|&i| {
                assert!(
                    i < self.len,
                    "MatrixV::gather_rows: row {} out of bounds for a view of {} rows",
                    i,
                    self.len
                );
                self.offset + i
            })
            .collect();
        self.matrix.extract_rows(&absolute)
    }
}

impl Shape for MatrixV {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank2 {
            rows: self.n_rows(),
            cols: self.n_cols(),
        }
    }
}
impl Concatenate for MatrixV {
    /// Concatenates two row windows into one owned matrix.
    ///
    /// Both windows must have the same number of columns. The result is returned
    /// as a full-span view over the concatenated matrix.
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        let joined = self.to_matrix().concat(other.to_matrix())?;
        Ok(MatrixV::from(joined))
    }
}

/// Equality is based on shape and values rather than backing storage.
impl PartialEq for MatrixV {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len
            && self.n_cols() == other.n_cols()
            && (0..self.n_cols()).all(|c| self.col(c) == other.col(c))
    }
}

impl fmt::Debug for MatrixV {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MatrixV{}: {} × {} [col-major, rows {}..{} of {}]",
            self.matrix
                .name
                .as_deref()
                .map_or(String::new(), |n| format!(" '{}'", n)),
            self.len,
            self.n_cols(),
            self.offset,
            self.end(),
            self.matrix.n_rows
        )
    }
}

impl Display for MatrixV {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let n_rows = self.n_rows();
        let n_cols = self.n_cols();
        match &self.matrix.name {
            Some(name) => writeln!(
                f,
                "MatrixView \"{}\" [{} rows × {} cols, f64] (rows {}..{} of {})",
                name,
                n_rows,
                n_cols,
                self.offset,
                self.end(),
                self.matrix.n_rows
            )?,
            None => writeln!(
                f,
                "MatrixView [{} rows × {} cols, f64] (rows {}..{} of {})",
                n_rows,
                n_cols,
                self.offset,
                self.end(),
                self.matrix.n_rows
            )?,
        }
        if n_cols == 0 {
            return Ok(());
        }
        let headers: Vec<String> = (0..n_cols).map(|c| format!("col_{c}")).collect();
        print_float_grid(f, &headers, n_rows, |row, col| self.get(row, col))
    }
}

/// Converts an owned `Matrix` to a full-span `MatrixV`.
impl From<Matrix> for MatrixV {
    fn from(matrix: Matrix) -> Self {
        let len = matrix.n_rows;
        MatrixV::from_matrix(matrix, 0, len)
    }
}

/// Converts a shared `Arc<Matrix>` to a full-span `MatrixV`.
impl From<Arc<Matrix>> for MatrixV {
    fn from(matrix: Arc<Matrix>) -> Self {
        let len = matrix.n_rows;
        MatrixV::from_arc_matrix(matrix, 0, len)
    }
}

/// Materialises a `MatrixV` as an owned `Matrix`.
impl From<MatrixV> for Matrix {
    fn from(view: MatrixV) -> Self {
        view.to_matrix()
    }
}

// ===== Selection Trait Implementations =====

/// Row selection over a matrix.
///
/// Contiguous selections return a view over the existing row range. Non-
/// contiguous selections gather the requested rows into a new owned matrix and
/// return a full-span view over it.
#[cfg(feature = "select")]
impl RowSelection for Matrix {
    type View = MatrixV;

    fn r<S: DataSelector>(&self, selection: S) -> MatrixV {
        let indices = selection.resolve_indices(self.n_rows);
        if selection.is_contiguous() {
            let start = indices.first().copied().unwrap_or(0);
            return MatrixV::from_matrix(self.clone(), start, indices.len());
        }
        MatrixV::from(self.extract_rows(&indices))
    }

    fn get_row_count(&self) -> usize {
        self.n_rows
    }
}

/// Row selection over an existing matrix view.
///
/// Contiguous selections return a sub-window over the same backing matrix.
/// Non-contiguous selections gather the requested rows into a new owned matrix
/// and return a full-span view over it.
#[cfg(feature = "select")]
impl RowSelection for MatrixV {
    type View = MatrixV;

    fn r<S: DataSelector>(&self, selection: S) -> MatrixV {
        let indices = selection.resolve_indices(self.len);
        if selection.is_contiguous() {
            let start = indices.first().copied().unwrap_or(0);
            return self.from_self(start, indices.len());
        }
        MatrixV::from(self.gather_rows(&indices))
    }

    fn get_row_count(&self) -> usize {
        self.len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat;

    /// A 5-row, 3-column matrix whose values encode `row + 10 * col`.
    fn sample() -> Matrix {
        Matrix::from_f64_unaligned(
            &[
                0.0, 1.0, 2.0, 3.0, 4.0, // col 0
                10.0, 11.0, 12.0, 13.0, 14.0, // col 1
                20.0, 21.0, 22.0, 23.0, 24.0, // col 2
            ],
            5,
            3,
            Some("m".to_string()),
        )
    }

    #[test]
    fn from_matrix_windows_rows_and_keeps_every_column() {
        let v = MatrixV::from_matrix(sample(), 1, 3);
        assert_eq!(v.offset, 1);
        assert_eq!(v.len, 3);
        assert_eq!(v.n_rows(), 3);
        assert_eq!(v.len(), 3);
        assert_eq!(v.n_cols(), 3);
        assert_eq!(v.end(), 4);
        assert_eq!(v.name(), Some("m"));
        assert_eq!(v.stride(), 8);
        assert!(!v.is_empty());
        assert!(!v.spans_backing());
    }

    #[test]
    fn from_arc_matrix_shares_the_backing_allocation() {
        let arc = Arc::new(sample());
        let v = MatrixV::from_arc_matrix(arc.clone(), 2, 2);
        assert!(Arc::ptr_eq(&arc, &v.matrix));
        assert_eq!(v.col(0), &[2.0, 3.0]);
        assert_eq!(v.col(2), &[22.0, 23.0]);
    }

    #[test]
    fn whole_matrix_window_spans_backing() {
        let v = MatrixV::from(sample());
        assert!(v.spans_backing());
        assert_eq!(v.n_rows(), 5);
        assert_eq!(v.end(), 5);
    }

    #[test]
    fn from_self_rewindows_relative_to_the_view() {
        let v = MatrixV::from_matrix(sample(), 1, 4);
        let inner = v.from_self(1, 2);
        assert_eq!(inner.offset, 2);
        assert_eq!(inner.len, 2);
        assert_eq!(inner.col(0), &[2.0, 3.0]);
        // Multiple re-windowing operations compose against the same backing matrix.
        let innermost = inner.from_self(1, 1);
        assert_eq!(innermost.offset, 3);
        assert_eq!(innermost.col(1), &[13.0]);
    }

    #[test]
    #[should_panic(expected = "MatrixV::from_arc_matrix: window [3, 6) out of bounds")]
    fn construction_past_the_backing_row_count_panics() {
        let _ = MatrixV::from_matrix(sample(), 3, 3);
    }

    #[test]
    #[should_panic(expected = "MatrixV::from_self: window [1, 4) out of bounds")]
    fn rewindow_past_the_view_row_count_panics() {
        let v = MatrixV::from_matrix(sample(), 1, 3);
        let _ = v.from_self(1, 3);
    }

    #[test]
    #[should_panic(expected = "MatrixV::col: column 3 out of bounds")]
    fn column_index_past_the_column_count_panics() {
        let v = MatrixV::from_matrix(sample(), 0, 5);
        let _ = v.col(3);
    }

    #[test]
    fn col_windows_are_sub_slices_of_the_backing_buffer() {
        let arc = Arc::new(sample());
        let v = MatrixV::from_arc_matrix(arc.clone(), 1, 3);
        let backing = arc.data.as_slice().as_ptr_range();
        for c in 0..v.n_cols() {
            let window = v.col(c).as_ptr_range();
            assert!(
                backing.contains(&window.start) && window.end <= backing.end,
                "column {c} window must lie inside the backing buffer"
            );
        }
    }

    #[test]
    fn to_matrix_copies_the_window_and_matches_the_source_rows() {
        let src = sample();
        let v = MatrixV::from_matrix(src.clone(), 1, 3);
        let owned = v.to_matrix();

        assert_eq!(owned.n_rows, 3);
        assert_eq!(owned.n_cols, 3);
        assert_eq!(owned.name.as_deref(), Some("m"));
        for c in 0..3 {
            assert_eq!(owned.col(c), v.col(c));
            assert_eq!(owned.col(c), &src.col(c)[1..4]);
        }
        // The materialised matrix is independent of the source and backing allocation.
        assert!(!std::ptr::eq(
            owned.as_slice().as_ptr(),
            v.matrix.as_slice().as_ptr()
        ));
    }

    #[test]
    fn to_matrix_round_trips_through_a_view() {
        let v = MatrixV::from_matrix(sample(), 1, 3);
        let round_tripped = MatrixV::from(v.to_matrix());
        assert_eq!(round_tripped, v);
    }

    #[test]
    fn columns_and_row_read_the_window() {
        let v = MatrixV::from_matrix(sample(), 2, 2);
        let cols = v.columns();
        assert_eq!(cols.len(), 3);
        assert_eq!(cols[0], &[2.0, 3.0]);
        assert_eq!(cols[1], &[12.0, 13.0]);
        assert_eq!(v.get(0, 1), 12.0);
        assert_eq!(v.row(1), vec![3.0, 13.0, 23.0]);
    }

    #[test]
    fn as_tuple_describes_the_window_over_the_flat_buffer() {
        let m = sample();
        let v = MatrixV::from_matrix(m.clone(), 1, 3);
        let (data, offset, len, stride) = v.as_tuple();

        assert_eq!(offset, 1);
        assert_eq!(len, 3);
        assert_eq!(stride, 8);
        // Column count is derived from buffer length and stride.
        assert_eq!(data.len() / stride, 3);
        // Column j of the window is accessible through the tuple.
        for j in 0..3 {
            let start = j * stride + offset;
            assert_eq!(&data[start..start + len], v.col(j));
        }

        // A full matrix produces the same tuple shape (zero offset) so kernels
        // can accept either form without knowing which was passed.
        let (whole, offset, len, stride) = m.as_tuple();
        assert_eq!((offset, len, stride), (0, 5, 8));
        assert_eq!(whole.len() / stride, 3);
    }

    #[test]
    fn as_strided_hands_blas_the_window_start_and_leading_dimension() {
        let v = MatrixV::from_matrix(sample(), 1, 3);
        let (a, lda) = v.as_strided();

        assert_eq!(lda, 8);
        assert_eq!((v.m(), v.n()), (3, 3));
        assert_eq!(v.lda(), lda);
        // Element (i, j) is at index j * lda + i in the BLAS layout.
        for j in 0..v.n_cols() {
            for i in 0..v.n_rows() {
                assert_eq!(a[j * lda as usize + i], v.get(i, j));
            }
        }
        // The last column's data is within the borrowed slice.
        assert!((v.n_cols() - 1) * lda as usize + v.n_rows() <= a.len());
    }

    #[test]
    fn as_mut_strided_writes_through_to_the_backing_matrix() {
        let backing = Arc::new(sample());
        let mut v = MatrixV::from_arc_matrix(backing.clone(), 1, 3);

        let (a, lda) = v.as_mut_strided();
        a[lda as usize] = 99.0;

        // The backing matrix is shared. Arc::make_mut copied the buffer. The write
        // affects only this view's copy, leaving the original unchanged.
        assert_eq!(v.get(0, 1), 99.0);
        assert_eq!(backing.get(1, 1), 11.0);
        assert!(!Arc::ptr_eq(&backing, &v.matrix));

        // With sole ownership of the backing matrix, writes modify it directly.
        let mut sole = MatrixV::from_matrix(sample(), 1, 3);
        let backing_ptr = sole.matrix.as_slice().as_ptr();
        let (a, _) = sole.as_mut_strided();
        a[0] = 42.0;
        assert_eq!(sole.matrix.get(1, 0), 42.0);
        assert_eq!(sole.matrix.as_slice().as_ptr(), backing_ptr);
    }

    #[test]
    fn gather_rows_reads_window_relative_indices() {
        let v = MatrixV::from_matrix(sample(), 1, 4);
        let gathered = v.gather_rows(&[2, 0]);
        assert_eq!(gathered.n_rows, 2);
        assert_eq!(gathered.col(0), &[3.0, 1.0]);
        assert_eq!(gathered.col(2), &[23.0, 21.0]);
    }

    #[test]
    #[should_panic(expected = "MatrixV::gather_rows: row 4 out of bounds")]
    fn gather_rows_past_the_window_panics() {
        let v = MatrixV::from_matrix(sample(), 1, 4);
        let _ = v.gather_rows(&[4]);
    }

    #[test]
    fn empty_window_reads_as_empty() {
        let v = MatrixV::from_matrix(sample(), 2, 0);
        assert!(v.is_empty());
        assert_eq!(v.n_rows(), 0);
        assert!(v.col(0).is_empty());
        assert_eq!(v.to_matrix().n_rows, 0);
    }

    #[test]
    fn shape_reports_the_window() {
        let v = MatrixV::from_matrix(sample(), 1, 3);
        assert_eq!(Shape::shape(&v), ShapeDim::Rank2 { rows: 3, cols: 3 });
    }

    #[test]
    fn equality_compares_window_contents_not_backing_matrices() {
        let a = MatrixV::from_matrix(sample(), 1, 3);
        let b = MatrixV::from(a.to_matrix());
        assert_eq!(a, b);
        assert_ne!(a, MatrixV::from_matrix(sample(), 0, 3));
    }

    #[test]
    fn concat_stacks_two_windows_into_one_owned_matrix() {
        let m = sample();
        let top = MatrixV::from_matrix(m.clone(), 0, 2);
        let bottom = MatrixV::from_matrix(m, 3, 2);
        let joined = top.concat(bottom).unwrap();
        assert_eq!(joined.n_rows(), 4);
        assert_eq!(joined.n_cols(), 3);
        assert!(joined.spans_backing());
        assert_eq!(joined.col(0), &[0.0, 1.0, 3.0, 4.0]);
        assert_eq!(joined.col(2), &[20.0, 21.0, 23.0, 24.0]);
    }

    #[test]
    fn concat_rejects_a_column_count_mismatch() {
        let a = MatrixV::from(mat![[1.0, 2.0]]);
        let b = MatrixV::from(mat![[1.0, 2.0], [3.0, 4.0]]);
        assert!(a.concat(b).is_err());
    }

    #[test]
    fn display_states_the_window_over_its_backing_matrix() {
        let v = MatrixV::from_matrix(sample(), 1, 2);
        let rendered = format!("{v}");
        assert!(
            rendered.starts_with("MatrixView \"m\" [2 rows × 3 cols, f64] (rows 1..3 of 5)"),
            "unexpected header: {rendered}"
        );
        assert!(rendered.contains("col_0"));
        assert!(rendered.contains("11"));
        // Only rows in the window are rendered, not the full backing matrix.
        assert!(!rendered.contains("14"));
    }

    #[test]
    fn debug_states_the_window_over_its_backing_matrix() {
        let v = MatrixV::from_matrix(sample(), 1, 2);
        assert_eq!(
            format!("{v:?}"),
            "MatrixV 'm': 2 × 3 [col-major, rows 1..3 of 5]"
        );
    }

    #[cfg(feature = "select")]
    #[test]
    fn row_selection_on_a_matrix_windows_a_contiguous_range() {
        let selected = sample().r(1..4);
        assert_eq!(selected.n_rows(), 3);
        assert_eq!(selected.offset, 1);
        assert_eq!(selected.col(0), &[1.0, 2.0, 3.0]);
    }

    #[cfg(feature = "select")]
    #[test]
    fn row_selection_on_a_matrix_moves_no_data() {
        let m = sample();
        let backing = m.as_slice().as_ptr();
        let selected = m.r(1..4);
        assert_eq!(
            selected.matrix.as_slice().as_ptr(),
            backing,
            "a range selection must share the matrix buffer, not copy it"
        );
    }

    #[cfg(feature = "select")]
    #[test]
    fn row_selection_on_a_matrix_gathers_an_index_array() {
        let selected = sample().r(&[3usize, 0][..]);
        assert_eq!(selected.n_rows(), 2);
        assert!(selected.spans_backing());
        assert_eq!(selected.col(1), &[13.0, 10.0]);
    }

    #[cfg(feature = "select")]
    #[test]
    fn row_selection_on_a_view_composes_windows() {
        let v = MatrixV::from_matrix(sample(), 1, 4);
        let chained = v.r(1..4).r(0..2);
        assert_eq!(chained.offset, 2);
        assert_eq!(chained.n_rows(), 2);
        assert_eq!(chained.col(0), &[2.0, 3.0]);
        assert_eq!(chained.get_row_count(), 2);

        // Non-contiguous selections gather rows into a new matrix.
        let picked = v.r(&[3usize, 1][..]);
        assert_eq!(picked.col(0), &[4.0, 2.0]);
    }

    #[cfg(feature = "size")]
    #[test]
    fn est_bytes_scales_with_the_window() {
        use crate::ByteSize;

        let full = MatrixV::from(sample());
        let half = MatrixV::from_matrix(sample(), 0, 2);
        assert!(half.est_bytes() < full.est_bytes());
        assert_eq!(half.est_bytes(), full.est_bytes() * 2 / 5);
    }
}
