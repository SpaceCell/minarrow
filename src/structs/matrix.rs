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

//! # **Matrix Module** - *De-facto Matrix Memory Layout for BLAS/LAPACK ecosystem compatibility*
//!
//! Dense column-major matrix type for high-performance linear algebra.
//! BLAS/LAPACK compatible with built-inconversions from `Table` data.

use std::fmt;
use std::sync::Arc;

use crate::enums::error::MinarrowError;
use crate::enums::shape_dim::ShapeDim;
use crate::structs::buffer::Buffer;
use crate::traits::{concatenate::Concatenate, shape::Shape};
use crate::{Array, Field, FieldArray, FloatArray, NumericArray, Table, Vec64};
#[cfg(feature = "views")]
use crate::{SharedBuffer, TableV};

/// # Matrix
///
/// Column-major dense matrix.
///
/// ### Description
/// This struct is compatible with Arrow, LAPACK, BLAS, and all
/// column-major numeric routines.
///
/// **This is an optional extra enabled by the `matrix` feature,
/// and is not part of the *`Apache Arrow`* framework**.
///
/// ### Properties
/// - `n_rows`: Logical number of rows.
/// - `n_cols`: Number of columns.
/// - `stride`: Physical elements per column in the buffer. Padded to 8-element
///   (64-byte) boundaries so every column starts SIMD-aligned. This is the
///   BLAS leading dimension (lda). Always `>= n_rows`.
/// - `data`: Flat buffer in column-major order with stride padding.
/// - `name`: Optional matrix name for diagnostics.
///
/// ### Null handling
/// - It is dense - nulls can be represented through `f64::NAN`
/// - However this is not always reliable, as a single *NaN* can affect vectorised
/// calculations when integrating with various frameworks.
///
/// ### Layout trade-offs vs standard Arrow tables
///
/// Matrix is shaped for OLAP-style batch compute, designed to hand memory to
/// LAPACK without repacking. For streaming or append-heavy workloads, stay in
/// a Table and promote to Matrix only at the boundary where LAPACK access is
/// needed (see [`TableV::try_as_matrix_zc`]).
///
/// | Operation | Matrix | Arrow Table |
/// |---|---|---|
/// | Cross-column scan at a shared row index (e.g. dot product across columns) | Strided access across the flat buffer; recovering spatial locality requires a transpose | Each column lives in its own allocation, so cross-column access spans distinct arenas |
/// | Hand the buffer to BLAS/LAPACK expecting `(ptr, lda)` | Native - pass `(as_slice(), stride)` directly to `dgemm`, `dsyev`, and their peers | Requires a repack into a contiguous stride-aligned buffer before dispatch |
/// | Grow the row count | Bounded by the pre-allocated stride budget; exceeding it triggers a full reallocation and re-layout of every column | Amortised O(1) per column via `Buffer::push` |
/// | Drop or reorder columns | Metadata-only while callers track which columns are live; a physical reorder rebuilds the flat buffer | Metadata change on the `FieldArray` list |
///
/// Both layouts deliver fast per-column SIMD scans. They differ on row-append
/// flexibility and on how cheaply the buffer can be handed to LAPACK.
#[repr(C, align(64))]
#[derive(Clone, PartialEq)]
pub struct Matrix {
    pub n_rows: usize,
    pub n_cols: usize,
    /// Physical column stride in elements, padded so each column is 64-byte aligned.
    pub stride: usize,
    /// Backing storage. `Buffer<f64>` carries either an owned `Vec64<f64>` or a
    /// shared view into an existing allocation. The shared variant enables
    /// zero-copy construction from upstream `TableV` arenas via
    /// `TableV::try_as_matrix_zc` without sacrificing the dense column-major
    /// stride layout that BLAS/LAPACK expects.
    pub data: Buffer<f64>,
    pub name: Option<String>,
}

/// Number of f64 elements per 64-byte alignment boundary.
const ALIGN_ELEMS: usize = 64 / std::mem::size_of::<f64>(); // 8

/// Round up to next multiple of ALIGN_ELEMS for 64-byte column alignment.
#[inline]
const fn aligned_stride(n_rows: usize) -> usize {
    (n_rows + ALIGN_ELEMS - 1) & !(ALIGN_ELEMS - 1)
}

impl Matrix {
    /// Constructs a new dense Matrix with shape and optional name.
    /// Data buffer is zeroed. Columns are padded to 64-byte alignment.
    pub fn new(n_rows: usize, n_cols: usize, name: Option<String>) -> Self {
        let stride = aligned_stride(n_rows);
        let len = stride * n_cols;
        let mut vec = Vec64::with_capacity(len);
        vec.resize(len, 0.0);
        let data = Buffer::from_vec64(vec);
        Matrix { n_rows, n_cols, stride, data, name }
    }

    /// Constructs a Matrix from a pre-padded Vec64 buffer.
    /// The buffer must already have `stride * n_cols` elements with the correct
    /// stride layout. Use `from_f64_unaligned` if your data is unpadded.
    pub fn from_f64_aligned(data: Vec64<f64>, n_rows: usize, n_cols: usize, name: Option<String>) -> Self {
        let stride = aligned_stride(n_rows);
        assert_eq!(
            data.len(),
            stride * n_cols,
            "Matrix: padded buffer length does not match stride * n_cols"
        );
        Matrix { n_rows, n_cols, stride, data: Buffer::from_vec64(data), name }
    }

    /// Constructs a Matrix from a flat column-major buffer without stride padding.
    /// The data is re-laid out with 64-byte aligned column padding.
    pub fn from_f64_unaligned(src: &[f64], n_rows: usize, n_cols: usize, name: Option<String>) -> Self {
        assert_eq!(
            src.len(),
            n_rows * n_cols,
            "Matrix shape does not match buffer length"
        );
        let stride = aligned_stride(n_rows);
        if stride == n_rows {
            // No padding needed, take ownership directly
            let vec = Vec64::from(src);
            return Matrix { n_rows, n_cols, stride, data: Buffer::from_vec64(vec), name };
        }
        // Re-layout with padding between columns
        let mut vec = Vec64::with_capacity(stride * n_cols);
        vec.resize(stride * n_cols, 0.0);
        for col in 0..n_cols {
            let src_start = col * n_rows;
            let dst_start = col * stride;
            vec.as_mut_slice()[dst_start..dst_start + n_rows]
                .copy_from_slice(&src[src_start..src_start + n_rows]);
        }
        Matrix { n_rows, n_cols, stride, data: Buffer::from_vec64(vec), name }
    }

    /// Constructs a Matrix from a slice of `FloatArray<f64>` columns. Each
    /// column becomes one Matrix column; null masks are rejected so Matrix
    /// stays dense.
    ///
    /// `impl AsRef<[FloatArray<f64>]>` accepts any contiguous layout at the
    /// call-site: `&[FloatArray<f64>]`, `&Vec<FloatArray<f64>>`,
    /// `[FloatArray<f64>; N]`, etc. 
    ///
    /// All columns must have the same length. Columns are copied into a
    /// 64-byte-aligned column-major buffer with stride padding.
    ///
    /// For existing `Table` / `TableV` containers prefer `Matrix::try_from`.
    /// This constructor targets direct construction from already-built column
    /// arrays.
    pub fn try_from_cols(
        cols: impl AsRef<[FloatArray<f64>]>,
        name: Option<String>,
    ) -> Result<Self, MinarrowError> {
        let columns = cols.as_ref();
        if columns.is_empty() {
            return Err(MinarrowError::ShapeError {
                message: "Matrix::try_from_cols requires at least one column".into(),
            });
        }

        let n_rows = columns[0].data.len();
        for (i, col) in columns.iter().enumerate() {
            if col.data.len() != n_rows {
                return Err(MinarrowError::ColumnLengthMismatch {
                    col: i,
                    expected: n_rows,
                    found: col.data.len(),
                });
            }
            // `None` short-circuits without touching the mask. 
            // A present-but-empty mask pays a single popcount.
            if col.null_mask.as_ref().map_or(false, |m| m.has_nulls()) {
                return Err(MinarrowError::NullError {
                    message: Some(format!(
                        "Matrix::try_from_cols: column {i} contains null values; Matrix requires dense data"
                    )),
                });
            }
        }

        let n_cols = columns.len();
        let stride = aligned_stride(n_rows);
        let pad = stride - n_rows;
        let mut vec = Vec64::with_capacity(stride * n_cols);

        // Each column is a guaranteed-dense slice, so
        // extend_from_slice is a straight memcpy into the column-major buffer.
        for col in columns {
            let col_slice: &[f64] = col.data.as_slice();
            vec.extend_from_slice(col_slice);
            if pad > 0 {
                vec.extend_from_slice(&[0.0; ALIGN_ELEMS][..pad]);
            }
        }
        Ok(Matrix { n_rows, n_cols, stride, data: Buffer::from_vec64(vec), name })
    }

    /// Returns the value at (row, col) (0-based). Panics if out of bounds.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        debug_assert!(row < self.n_rows, "Row out of bounds");
        debug_assert!(col < self.n_cols, "Col out of bounds");
        self.data.as_slice()[col * self.stride + row]
    }

    /// Sets the value at (row, col) (0-based). Panics if out of bounds.
    /// Triggers copy-on-write if the backing buffer is currently shared.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        debug_assert!(row < self.n_rows, "Row out of bounds");
        debug_assert!(col < self.n_cols, "Col out of bounds");
        self.data.as_mut_slice()[col * self.stride + row] = value;
    }

    /// Returns true if the matrix is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_rows == 0 || self.n_cols == 0
    }

    /// Returns the logical number of elements (n_rows * n_cols), not including padding.
    #[inline]
    pub fn len(&self) -> usize {
        self.n_rows * self.n_cols
    }

    /// Returns an immutable reference to the full flat buffer including padding.
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        self.data.as_slice()
    }

    /// Returns a mutable reference to the full flat buffer including padding.
    /// Triggers copy-on-write if the backing buffer is currently shared.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        self.data.as_mut_slice()
    }

    /// Returns a view of the matrix as a slice of columns (logical rows only, no padding).
    pub fn columns(&self) -> Vec<&[f64]> {
        let slice = self.data.as_slice();
        (0..self.n_cols)
            .map(|col| &slice[(col * self.stride)..(col * self.stride + self.n_rows)])
            .collect()
    }

    /// Returns a vector of mutable slices, each corresponding to a column.
    /// Triggers copy-on-write if the backing buffer is currently shared.
    pub fn columns_mut(&mut self) -> Vec<&mut [f64]> {
        let n_rows = self.n_rows;
        let stride = self.stride;
        let n_cols = self.n_cols;
        // The public fields permit a caller to set stride < n_rows, which
        // would produce aliasing &mut slices below. Enforce the non-overlap
        // invariant at the point that materialises the slices so the unsafe
        // block keeps its non-aliasing guarantee.
        assert!(
            stride >= n_rows,
            "Matrix::columns_mut: stride ({stride}) < n_rows ({n_rows}); column slices would alias"
        );
        let total = stride.checked_mul(n_cols).expect("Matrix::columns_mut: stride * n_cols overflow");
        assert!(
            self.data.len() >= total,
            "Matrix::columns_mut: data buffer shorter than stride * n_cols"
        );
        let ptr = self.data.as_mut_slice().as_mut_ptr();
        let mut result = Vec::with_capacity(n_cols);

        for col in 0..n_cols {
            let start = col * stride;
            // SAFETY: stride >= n_rows ensures successive column slices do
            // not overlap, the data-length check ensures every slice is in
            // bounds, and we hold exclusive &mut self for the whole vec.
            unsafe {
                let col_ptr = ptr.add(start);
                let slice = std::slice::from_raw_parts_mut(col_ptr, n_rows);
                result.push(slice);
            }
        }
        result
    }

    /// Returns a single column as a slice, panics if col out of bounds.
    #[inline]
    pub fn col(&self, col: usize) -> &[f64] {
        assert!(col < self.n_cols, "Col out of bounds");
        let slice = self.data.as_slice();
        &slice[(col * self.stride)..(col * self.stride + self.n_rows)]
    }

    /// Returns a single column as a mutable slice, panics if col out of bounds.
    /// Triggers copy-on-write if the backing buffer is currently shared.
    #[inline]
    pub fn col_mut(&mut self, col: usize) -> &mut [f64] {
        assert!(col < self.n_cols, "Col out of bounds");
        let start = col * self.stride;
        &mut self.data.as_mut_slice()[start..start + self.n_rows]
    }

    /// Returns a single row as an owned Vec.
    #[inline]
    pub fn row(&self, row: usize) -> Vec<f64> {
        debug_assert!(row < self.n_rows, "Row out of bounds");
        let slice = self.data.as_slice();
        (0..self.n_cols).map(|col| slice[col * self.stride + row]).collect()
    }

    /// Transpose this matrix, returning a new Matrix with rows and columns swapped.
    /// The result has stride-aligned columns for SIMD access.
    pub fn transpose(&self) -> Self {
        let mut dst = Matrix::new(self.n_cols, self.n_rows, self.name.clone());
        let src = self.data.as_slice();
        let dst_stride = dst.stride;
        let dst_slice = dst.data.as_mut_slice();
        for j in 0..self.n_cols {
            for i in 0..self.n_rows {
                dst_slice[i * dst_stride + j] = src[j * self.stride + i];
            }
        }
        dst
    }

    /// Extract rows by index into a new Matrix.
    /// Column stride alignment is maintained in the result.
    pub fn extract_rows(&self, indices: &[usize]) -> Self {
        let n_new = indices.len();
        let mut dst = Matrix::new(n_new, self.n_cols, self.name.clone());
        for j in 0..self.n_cols {
            let src_col = self.col(j);
            let dst_col = dst.col_mut(j);
            for (k, &idx) in indices.iter().enumerate() {
                dst_col[k] = src_col[idx];
            }
        }
        dst
    }

    /// Sets the matrix name.
    #[inline]
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
    }

    /// Returns the number of columns.
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Returns the number of rows.
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    // ********************** BLAS/LAPACK Compatibility **************

    /// Number of rows as i32 for BLAS parameter passing.
    #[inline]
    pub fn m(&self) -> i32 {
        self.n_rows as i32
    }

    /// Number of columns as i32 for BLAS parameter passing.
    #[inline]
    pub fn n(&self) -> i32 {
        self.n_cols as i32
    }

    /// Leading dimension for BLAS - equals stride, which is n_rows padded to
    /// 64-byte alignment. Pass this as the `lda` parameter to all BLAS/LAPACK calls.
    #[inline]
    pub fn lda(&self) -> i32 {
        self.stride as i32
    }

    /// Borrow the matrix as a `(data, lda)` pair suitable for BLAS/LAPACK
    /// routines. The leading dimension travels with the slice because the
    /// two are meaningless apart - both describe the same stride-aligned
    /// column-major layout.
    ///
    /// Packaging them together also avoids a borrow-checker conflict when
    /// handing a matrix to a function that wants both values: reading
    /// `self.stride` produces a Copy `i32`, so no outstanding borrow of the
    /// matrix lingers once the tuple is returned.
    #[inline]
    pub fn as_strided(&self) -> (&[f64], i32) {
        (self.data.as_slice(), self.stride as i32)
    }

    /// Mutable counterpart to [`as_strided`]. Returns `(data, lda)` where
    /// `data` is a `&mut [f64]` view of the backing buffer (triggering
    /// copy-on-write if the buffer is currently shared). The leading
    /// dimension is read as a Copy value before the mutable borrow is
    /// created, so the returned `i32` carries no borrow of `self`.
    #[inline]
    pub fn as_mut_strided(&mut self) -> (&mut [f64], i32) {
        let lda = self.stride as i32;
        (self.data.as_mut_slice(), lda)
    }

    // ********************** Table conversion **********************

    /// Convert this Matrix into a Table with zero-copy column sharing.
    ///
    /// The matrix data buffer is frozen into a SharedBuffer (or reused, if it
    /// was already shared), and each column becomes a FloatArray backed by a
    /// window into that shared allocation. No data is copied.
    ///
    /// `fields` must have exactly `n_cols` entries, providing the name and
    /// metadata for each column.
    pub fn to_table(self, fields: Vec<Field>) -> Result<Table, MinarrowError> {
        if fields.len() != self.n_cols {
            return Err(MinarrowError::ShapeError {
                message: format!(
                    "to_table: expected {} fields for {} columns, got {}",
                    self.n_cols, self.n_cols, fields.len()
                ),
            });
        }

        let n_rows = self.n_rows;
        let n_cols = self.n_cols;
        let stride = self.stride;
        let name = self.name;

        // Surface the backing SharedBuffer and its element-level base offset.
        // Owned data is frozen into a new SharedBuffer; already-shared data
        // reuses its owner so to_table stays zero-copy across repeat calls.
        // SAFETY: f64 has no drop logic or interior invariants.
        let (shared, base_offset, _len) = unsafe { self.data.into_shared_parts() };

        let mut cols = Vec::with_capacity(n_cols);
        for (i, field) in fields.into_iter().enumerate() {
            // Each column starts at base_offset + i * stride elements.
            // base_offset is always 64-byte aligned for owned data and carried
            // through for shared data so downstream SIMD stays aligned.
            let col_offset = base_offset + i * stride;
            let buf: Buffer<f64> = Buffer::from_shared_column(shared.clone(), col_offset, n_rows);
            let float_arr = FloatArray::new(buf, None);
            let array = Array::NumericArray(NumericArray::Float64(Arc::new(float_arr)));
            cols.push(FieldArray::new(field, array));
        }

        Ok(Table::new(name.unwrap_or_default(), Some(cols)))
    }

    /// Convert this Matrix into a Table using auto-generated column names
    /// (col_0, col_1, ...).
    pub fn to_table_gen(self) -> Table {
        let n_cols = self.n_cols;
        let fields: Vec<Field> = (0..n_cols)
            .map(|i| Field::new(format!("col_{}", i), crate::ffi::arrow_dtype::ArrowType::Float64, false, None))
            .collect();
        self.to_table(fields).unwrap()
    }

}

#[cfg(feature = "views")]
impl TableV {
    /// Attempt a zero-copy construction of a `Matrix` from this view.
    ///
    /// Succeeds when the columns are already laid out in memory the way a
    /// Matrix expects: every active column is a dense, null-free
    /// `FloatArray<f64>` whose `Buffer` is a `Shared` view into one common
    /// `SharedBuffer`, with column `i` starting at `base + i * stride` elements
    /// (where `stride = aligned_stride(n_rows)`). When that holds, the Matrix
    /// borrows the existing column-major stride-aligned allocation directly.
    ///
    /// This layout is currently supported via `SharedBuffer`-backed Table
    /// construction: build a single 64-byte-aligned allocation, hand each
    /// column a `Buffer::from_shared_column` view at the right offset, and
    /// wrap those in `FieldArray`s. Tables produced that way can be fed to
    /// BLAS/LAPACK statistics code without materialising a fresh Matrix.
    ///
    /// Any other layout (owned buffers, mixed allocations, non-aligned column
    /// spacing, nulls, non-f64 columns) returns `Err` - callers that want a
    /// Matrix regardless should fall back to `Matrix::try_from(&table)`, which
    /// copies.
    ///
    /// The returned Matrix holds a refcount on the shared allocation; the
    /// source TableV and Table remain independently usable. Mutating the
    /// returned Matrix triggers copy-on-write via `Buffer` and does not
    /// affect the original shared allocation.
    pub fn try_as_matrix_zc(&self) -> Result<Matrix, MinarrowError> {
        let n_rows = self.n_rows();
        let n_cols = self.n_cols();
        if n_cols == 0 {
            return Err(MinarrowError::ShapeError {
                message: "try_as_matrix_zc: TableV has no active columns".into(),
            });
        }

        let stride = aligned_stride(n_rows);
        let active = self.active_col_indices();

        let mut base_owner: Option<SharedBuffer> = None;
        let mut base_offset: usize = 0;
        let mut owner_elem_len: usize = 0;

        // Every column must be a dense null-free f64 view into
        // the same SharedBuffer, with column_i starting at base + i * stride.
        // ArrayV::new enforces offset+len <= array.len at construction so the
        // window bound is already invariant; ArrayV::null_count() covers the
        // windowed null check and caches on first use.
        for (pos, &raw_idx) in active.iter().enumerate() {
            let arrayv = &self.cols[raw_idx];
            let fa = arrayv.array.num_ref()?.f64_ref()?;
            if arrayv.null_count() > 0 {
                return Err(MinarrowError::NullError {
                    message: Some(format!(
                        "try_as_matrix_zc: column {raw_idx} contains null values"
                    )),
                });
            }
            let (owner, buf_offset, _) = fa.data.shared_parts().ok_or_else(|| {
                MinarrowError::ShapeError {
                    message: format!(
                        "try_as_matrix_zc: column {raw_idx} is owned, not a shared view"
                    ),
                }
            })?;
            let col_elem_offset = buf_offset + arrayv.offset;

            match &base_owner {
                None => {
                    base_owner = Some(owner.clone());
                    base_offset = col_elem_offset;
                    owner_elem_len = owner.len() / std::mem::size_of::<f64>();
                }
                Some(base) => {
                    if !SharedBuffer::ptr_eq(base, owner) {
                        return Err(MinarrowError::ShapeError {
                            message: format!(
                                "try_as_matrix_zc: column {raw_idx} lives in a different allocation"
                            ),
                        });
                    }
                    let expected = base_offset + pos * stride;
                    if col_elem_offset != expected {
                        return Err(MinarrowError::ShapeError {
                            message: format!(
                                "try_as_matrix_zc: column {raw_idx} offset {col_elem_offset} does \
                                 not match expected stride layout {expected}"
                            ),
                        });
                    }
                }
            }
        }

        // The Matrix's backing buffer spans stride * n_cols elements including
        // trailing pad on the last column. Confirm the shared allocation
        // actually covers that region before handing out a view.
        let total_elems = stride * n_cols;
        if base_offset + total_elems > owner_elem_len {
            return Err(MinarrowError::ShapeError {
                message: format!(
                    "try_as_matrix_zc: shared allocation has {owner_elem_len} elements, \
                     needs {} for column-major layout of {n_rows}x{n_cols} at stride {stride}",
                    base_offset + total_elems
                ),
            });
        }

        let data = Buffer::from_shared_column(base_owner.unwrap(), base_offset, total_elems);
        let name = if self.name().is_empty() { None } else { Some(self.name().to_string()) };
        Ok(Matrix { n_rows, n_cols, stride, data, name })
    }
}

impl Shape for Matrix {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank2 {
            rows: self.n_rows(),
            cols: self.n_cols(),
        }
    }
}

impl Concatenate for Matrix {
    /// Concatenates two matrices vertically (i.e., row-wise stacking).
    ///
    /// # Requirements
    /// - Both matrices must have the same number of columns
    ///
    /// # Returns
    /// A new Matrix with rows from `self` followed by rows from `other`
    ///
    /// # Errors
    /// - `IncompatibleTypeError` if column counts don't match
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        // Check column count
        if self.n_cols != other.n_cols {
            return Err(MinarrowError::IncompatibleTypeError {
                from: "Matrix",
                to: "Matrix",
                message: Some(format!(
                    "Cannot concatenate matrices with different column counts: {} vs {}",
                    self.n_cols, other.n_cols
                )),
            });
        }

        // Handle empty matrices
        if self.is_empty() && other.is_empty() {
            return Ok(Matrix::new(
                0,
                0,
                None,
            ));
        }

        let result_n_rows = self.n_rows + other.n_rows;
        let result_n_cols = self.n_cols;
        let result_stride = aligned_stride(result_n_rows);
        let pad = result_stride - result_n_rows;
        let mut result_vec = Vec64::with_capacity(result_stride * result_n_cols);

        for col in 0..result_n_cols {
            result_vec.extend_from_slice(self.col(col));
            result_vec.extend_from_slice(other.col(col));
            if pad > 0 {
                result_vec.extend_from_slice(&[0.0; ALIGN_ELEMS][..pad]);
            }
        }

        Ok(Matrix {
            n_rows: result_n_rows,
            n_cols: result_n_cols,
            stride: result_stride,
            data: Buffer::from_vec64(result_vec),
            name: None,
        })
    }
}

// Pretty print
impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Matrix{}: {} × {} [col-major]",
            self.name.as_deref().map_or(String::new(), |n| format!(" '{}'", n)),
            self.n_rows, self.n_cols
        )?;
        for row in 0..self.n_rows.min(6) {
            // Print up to 6 rows
            write!(f, "\n[")?;
            for col in 0..self.n_cols.min(8) {
                // Print up to 8 cols
                write!(f, " {:8.4}", self.get(row, col))?;
                if col != self.n_cols - 1 {
                    write!(f, ",")?;
                }
            }
            if self.n_cols > 8 {
                write!(f, " ...")?;
            }
            write!(f, " ]")?;
        }
        if self.n_rows > 6 {
            write!(f, "\n...")?;
        }
        Ok(())
    }
}

// From Vec<FloatArray<f64>> to unnamed Matrix
impl From<Vec<FloatArray<f64>>> for Matrix {
    fn from(columns: Vec<FloatArray<f64>>) -> Self {
        let n_cols = columns.len();
        let n_rows = columns.first().map(|c| c.data.len()).unwrap_or(0);
        let stride = aligned_stride(n_rows);
        let pad = stride - n_rows;
        for col in &columns {
            assert_eq!(col.data.len(), n_rows, "Column length mismatch");
        }
        let mut vec = Vec64::with_capacity(stride * n_cols);
        for col in &columns {
            vec.extend_from_slice(&col.data);
            if pad > 0 {
                vec.extend_from_slice(&[0.0; ALIGN_ELEMS][..pad]);
            }
        }
        Matrix { n_rows, n_cols, stride, data: Buffer::from_vec64(vec), name: None }
    }
}

// From (Vec<FloatArray<f64>>, String) to named Matrix
impl From<(Vec<FloatArray<f64>>, String)> for Matrix {
    fn from((columns, name): (Vec<FloatArray<f64>>, String)) -> Self {
        let mut mat = Matrix::from(columns);
        mat.name = Some(name);
        mat
    }
}

// From &[FloatArray<f64>] to unnamed Matrix
impl From<&[FloatArray<f64>]> for Matrix {
    fn from(columns: &[FloatArray<f64>]) -> Self {
        let n_cols = columns.len();
        let n_rows = columns.first().map(|c| c.data.len()).unwrap_or(0);
        let stride = aligned_stride(n_rows);
        let pad = stride - n_rows;
        for col in columns {
            assert_eq!(col.data.len(), n_rows, "Column length mismatch");
        }
        let mut vec = Vec64::with_capacity(stride * n_cols);
        for col in columns {
            vec.extend_from_slice(&col.data);
            if pad > 0 {
                vec.extend_from_slice(&[0.0; ALIGN_ELEMS][..pad]);
            }
        }
        Matrix { n_rows, n_cols, stride, data: Buffer::from_vec64(vec), name: None }
    }
}

impl TryFrom<&Table> for Matrix {
    type Error = MinarrowError;

    fn try_from(table: &Table) -> Result<Self, Self::Error> {
        let name = if table.name.is_empty() { None } else { Some(table.name.clone()) };
        let n_cols = table.n_cols();
        let n_rows = table.n_rows;
        let stride = aligned_stride(n_rows);
        let pad = stride - n_rows;

        let mut vec = Vec64::with_capacity(stride * n_cols);
        for (col_idx, fa) in table.cols.iter().enumerate() {
            let numeric = fa.array.num_ref().map_err(|_| MinarrowError::TypeError {
                from: "non-numeric",
                to: "Float64",
                message: Some(format!("column {} is not numeric", col_idx)),
            })?;
            let f64_arr = numeric.clone().f64()?;
            if f64_arr.data.len() != n_rows {
                return Err(MinarrowError::ColumnLengthMismatch {
                    col: col_idx,
                    expected: n_rows,
                    found: f64_arr.data.len(),
                });
            }
            vec.extend_from_slice(f64_arr.data.as_slice());
            if pad > 0 {
                vec.extend_from_slice(&[0.0; ALIGN_ELEMS][..pad]);
            }
        }

        Ok(Matrix { n_rows, n_cols, stride, data: Buffer::from_vec64(vec), name })
    }
}

impl TryFrom<Table> for Matrix {
    type Error = MinarrowError;

    fn try_from(table: Table) -> Result<Self, Self::Error> {
        Matrix::try_from(&table)
    }
}

// From &[Vec<f64>] (Vec-of-cols) to unnamed Matrix
impl From<&[Vec<f64>]> for Matrix {
    fn from(columns: &[Vec<f64>]) -> Self {
        let n_cols = columns.len();
        let n_rows = columns.first().map(|c| c.len()).unwrap_or(0);
        let stride = aligned_stride(n_rows);
        let pad = stride - n_rows;
        for col in columns {
            assert_eq!(col.len(), n_rows, "Column length mismatch");
        }
        let mut vec = Vec64::with_capacity(stride * n_cols);
        for col in columns {
            vec.extend_from_slice(col);
            if pad > 0 {
                vec.extend_from_slice(&[0.0; ALIGN_ELEMS][..pad]);
            }
        }
        Matrix { n_rows, n_cols, stride, data: Buffer::from_vec64(vec), name: None }
    }
}

// From flat unpadded slice with shape - re-lays out with stride padding
impl<'a> From<(&'a [f64], usize, usize, Option<String>)> for Matrix {
    fn from((slice, n_rows, n_cols, name): (&'a [f64], usize, usize, Option<String>)) -> Self {
        assert_eq!(slice.len(), n_rows * n_cols, "Slice shape mismatch");
        Matrix::from_f64_unaligned(slice, n_rows, n_cols, name)
    }
}

#[cfg(feature = "views")]
impl TryFrom<&TableV> for Matrix {
    type Error = MinarrowError;

    /// Materialise a `TableV` window into a column-major `Matrix`. Honours
    /// `active_col_selection` and slices each column at the view's offset, so
    /// only the windowed rows are copied. Columns must be `FloatArray<f64>`;
    /// callers needing other numeric types should convert before constructing
    /// the view.
    fn try_from(view: &TableV) -> Result<Self, Self::Error> {
        let name = if view.name().is_empty() { None } else { Some(view.name().to_string()) };
        let n_rows = view.n_rows();
        let active = view.active_col_indices();
        let n_cols = active.len();
        let stride = aligned_stride(n_rows);
        let pad = stride - n_rows;

        let mut vec = Vec64::with_capacity(stride * n_cols);
        for &col_idx in &active {
            let (array, offset, len) = view.cols[col_idx].as_tuple_ref();
            let fa = array.num_ref()?.f64_ref()?;
            // ArrayV::new bounds-checks offset+len at construction.
            vec.extend_from_slice(&fa.data.as_slice()[offset..offset + len]);
            if pad > 0 {
                vec.extend_from_slice(&[0.0; ALIGN_ELEMS][..pad]);
            }
        }
        Ok(Matrix { n_rows, n_cols, stride, data: Buffer::from_vec64(vec), name })
    }
}

#[cfg(feature = "views")]
impl TableV {
    /// Materialise this view as an owned `Matrix`, copying column data once
    /// into a 64-byte-aligned column-major buffer. For zero-copy access into a
    /// shared allocation, see [`TableV::try_as_matrix_zc`].
    pub fn try_as_matrix(&self) -> Result<Matrix, MinarrowError> {
        Matrix::try_from(self)
    }
}

// ********************** Iterators ***********************

impl<'a> IntoIterator for &'a Matrix {
    type Item = &'a f64;
    type IntoIter = std::slice::Iter<'a, f64>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.as_slice().iter()
    }
}

impl<'a> IntoIterator for &'a mut Matrix {
    type Item = &'a mut f64;
    type IntoIter = std::slice::IterMut<'a, f64>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.as_mut_slice().iter_mut()
    }
}

impl IntoIterator for Matrix {
    type Item = f64;
    type IntoIter = <Vec64<f64> as IntoIterator>::IntoIter;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

/// Ergonomic constructor for a [`Matrix`] from `f64` column data.
///
/// Both forms build through [`Matrix::try_from_cols`] and unwrap the
/// result, so a length mismatch panics with the same error shape as the
/// underlying call.
///
/// # Forms
/// - Literal columns: `mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]` produces
///   a 3-row, 2-column matrix.
/// - Slice-of-slices expression: `mat!(&[&[1.0, 2.0, 3.0][..], &[4.0, 5.0, 6.0][..]])`
///   produces the equivalent matrix from an existing `&[&[f64]]` value.
///
/// # Example
/// ```
/// use minarrow::mat;
/// let m = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
/// assert_eq!((m.n_rows, m.n_cols), (3, 2));
/// ```
#[cfg(feature = "matrix")]
#[macro_export]
macro_rules! mat {
    // Literal column form: one or more bracketed column literals.
    ($([ $($x:expr),+ $(,)? ]),+ $(,)?) => {{
        $crate::Matrix::try_from_cols(
            &[
                $( $crate::FloatArray::<f64>::from_slice(&[$($x),+]) ),+
            ][..],
            None,
        ).unwrap()
    }};

    // Slice-of-slices form: a single expression evaluating to `&[&[f64]]`.
    ($cols:expr $(,)?) => {{
        let cols_ref: &[&[f64]] = $cols;
        $crate::Matrix::try_from_cols(
            cols_ref
                .iter()
                .map(|c| $crate::FloatArray::<f64>::from_slice(*c))
                .collect::<::std::vec::Vec<_>>()
                .as_slice(),
            None,
        ).unwrap()
    }};
}

#[cfg(all(test, feature = "views"))]
mod try_as_matrix_zc_tests {
    use super::*;

    #[test]
    fn round_trip_zero_copy() {
        // Build an owned matrix, hand it out via to_table so the backing Vec64
        // becomes a SharedBuffer, then pull it back through TableV as ZC.
        let src = Matrix::try_from_cols(&[
            FloatArray::from_slice(&[1.0_f64, 2.0, 3.0, 4.0, 5.0]),
            FloatArray::from_slice(&[6.0_f64, 7.0, 8.0, 9.0, 10.0]),
            FloatArray::from_slice(&[11.0_f64, 12.0, 13.0, 14.0, 15.0]),
        ], Some("m".into())).unwrap();
        let expected: Vec<f64> = src.data.as_slice().to_vec();
        let stride_src = src.stride;
        let table = src.to_table_gen();
        let view = TableV::from_table(table, 0, 5);

        let zc = view.try_as_matrix_zc().expect("zero-copy should succeed");
        assert_eq!(zc.n_rows, 5);
        assert_eq!(zc.n_cols, 3);
        assert_eq!(zc.stride, stride_src);
        assert_eq!(zc.data.as_slice(), expected.as_slice());
        assert!(zc.data.is_shared(), "ZC result must reuse the shared allocation");
    }

    #[test]
    fn cow_on_mutate_preserves_source() {
        // Mutating the ZC matrix must copy-on-write and leave the source table
        // untouched.
        let src = Matrix::try_from_cols(&[
            FloatArray::from_slice(&[1.0_f64, 2.0, 3.0]),
            FloatArray::from_slice(&[4.0_f64, 5.0, 6.0]),
        ], None).unwrap();
        let table = src.to_table_gen();
        let view = TableV::from_table(table.clone(), 0, 3);
        let mut zc = view.try_as_matrix_zc().unwrap();
        zc.set(0, 0, 99.0);
        assert_eq!(zc.get(0, 0), 99.0);
        assert!(!zc.data.is_shared(), "mutation must trigger COW");

        // Source table columns still see the original values.
        let original_table_first = {
            let tv = TableV::from_table(table, 0, 3);
            let zc2 = tv.try_as_matrix_zc().unwrap();
            zc2.get(0, 0)
        };
        assert_eq!(original_table_first, 1.0);
    }

    #[test]
    fn rejects_owned_buffer_columns() {
        // Table built directly from FieldArrays (not via Matrix::to_table) has
        // independent owned buffers, which never satisfy the ZC layout.
        let a = crate::fa_f64!("a", 1.0, 2.0, 3.0);
        let b = crate::fa_f64!("b", 4.0, 5.0, 6.0);
        let table = Table::new("t".into(), Some(vec![a, b]));
        let view = TableV::from_table(table, 0, 3);
        let err = view.try_as_matrix_zc().expect_err("owned columns cannot be zero-copy");
        assert!(
            matches!(err, MinarrowError::ShapeError { .. }),
            "expected ShapeError, got {err:?}"
        );
    }

    #[test]
    fn rejects_null_masked_columns() {
        let src = Matrix::try_from_cols(&[
            FloatArray::from_slice(&[1.0_f64, 2.0, 3.0]),
            FloatArray::from_slice(&[4.0_f64, 5.0, 6.0]),
        ], None).unwrap();
        let mut table = src.to_table_gen();

        // Inject a null into col 0. Buffer::clone preserves Shared storage, so
        // the data stays in the original allocation; only the null mask
        // changes. This exercises the windowed null check in the ZC path.
        if let crate::Array::NumericArray(crate::NumericArray::Float64(arc_fa)) =
            &mut table.cols[0].array
        {
            let mut fa = (**arc_fa).clone();
            let mut mask = crate::Bitmask::new_set_all(3, true);
            mask.set(0, false);
            fa.null_mask = Some(mask);
            *arc_fa = std::sync::Arc::new(fa);
        } else {
            panic!("expected Float64 column");
        }

        let view = TableV::from_table(table, 0, 3);
        let err = view.try_as_matrix_zc().expect_err("column with nulls must reject");
        assert!(
            matches!(err, MinarrowError::NullError { .. }),
            "expected NullError, got {err:?}"
        );
    }

    #[test]
    fn rejects_empty_view() {
        let table = Table::new("t".into(), Some(Vec::new()));
        let view = TableV::from_table(table, 0, 0);
        let err = view.try_as_matrix_zc().expect_err("empty view must reject");
        assert!(matches!(err, MinarrowError::ShapeError { .. }));
    }

    #[test]
    fn rejects_non_f64_columns() {
        // Use fa_i32! for a shared-free integer column. The f64_ref cast fails
        // first, so the ZC path surfaces a TypeError before the owned check.
        let a = crate::fa_i32!("a", 1, 2, 3);
        let table = Table::new("t".into(), Some(vec![a]));
        let view = TableV::from_table(table, 0, 3);
        let err = view.try_as_matrix_zc().expect_err("non-f64 column must reject");
        assert!(matches!(err, MinarrowError::TypeError { .. }));
    }

    #[test]
    fn rejects_column_order_mismatch() {
        // Two independent matrices produce two independent SharedBuffers.
        // Swapping a column from matrix B into a table built from matrix A
        // trips the ptr_eq check: different allocations, same column layout.
        let m_a = Matrix::try_from_cols(&[
            FloatArray::from_slice(&[1.0_f64, 2.0, 3.0]),
            FloatArray::from_slice(&[4.0_f64, 5.0, 6.0]),
        ], None).unwrap();
        let m_b = Matrix::try_from_cols(&[
            FloatArray::from_slice(&[10.0_f64, 20.0, 30.0]),
            FloatArray::from_slice(&[40.0_f64, 50.0, 60.0]),
        ], None).unwrap();
        let mut table_a = m_a.to_table_gen();
        let table_b = m_b.to_table_gen();
        // Replace col 1 of table_a with col 0 of table_b.
        table_a.cols[1] = table_b.cols[0].clone();
        table_a.n_rows = 3;

        let view = TableV::from_table(table_a, 0, 3);
        let err = view.try_as_matrix_zc().expect_err("mixed allocations must reject");
        assert!(matches!(err, MinarrowError::ShapeError { .. }));
    }

    #[test]
    fn rejects_column_offset_misalignment() {
        // Reordering columns in a shared-buffer table breaks the column-major
        // stride invariant. Col 2 ends up at raw position 0 (offset = 2*stride),
        // col 0 ends up at raw position 2 (offset = 0), so the second pass of
        // the ZC loop computes expected = 2*stride + stride but sees offset
        // 1*stride from the unchanged middle column.
        let src = Matrix::try_from_cols(&[
            FloatArray::from_slice(&[1.0_f64, 2.0, 3.0]),
            FloatArray::from_slice(&[4.0_f64, 5.0, 6.0]),
            FloatArray::from_slice(&[7.0_f64, 8.0, 9.0]),
        ], None).unwrap();
        let mut table = src.to_table_gen();
        table.cols.swap(0, 2);

        let view = TableV::from_table(table, 0, 3);
        let err = view.try_as_matrix_zc().expect_err("reordered columns must reject");
        assert!(matches!(err, MinarrowError::ShapeError { .. }));
    }

    // ---- try_as_matrix (copy path) ----

    #[test]
    fn try_as_matrix_copies_owned_columns() {
        // fa_f64! produces independently-owned column buffers, so the ZC path
        // would reject - but the copy path lays them out into a fresh aligned
        // buffer and succeeds.
        let a = crate::fa_f64!("a", 1.0, 2.0, 3.0);
        let b = crate::fa_f64!("b", 4.0, 5.0, 6.0);
        let table = Table::new("t".into(), Some(vec![a, b]));
        let view = TableV::from_table(table, 0, 3);
        let m = view.try_as_matrix().expect("copy path always works for f64 columns");
        assert_eq!(m.n_rows, 3);
        assert_eq!(m.n_cols, 2);
        assert_eq!(m.col(0), &[1.0, 2.0, 3.0]);
        assert_eq!(m.col(1), &[4.0, 5.0, 6.0]);
        assert!(!m.data.is_shared(), "try_as_matrix produces owned data");
    }

    #[test]
    fn try_as_matrix_respects_window_offset() {
        // Slicing the view to rows 1..4 must copy only those rows into the
        // matrix, not the full underlying column.
        let a = crate::fa_f64!("a", 10.0, 11.0, 12.0, 13.0, 14.0);
        let b = crate::fa_f64!("b", 20.0, 21.0, 22.0, 23.0, 24.0);
        let table = Table::new("t".into(), Some(vec![a, b]));
        let view = TableV::from_table(table, 1, 3);
        let m = view.try_as_matrix().unwrap();
        assert_eq!(m.n_rows, 3);
        assert_eq!(m.col(0), &[11.0, 12.0, 13.0]);
        assert_eq!(m.col(1), &[21.0, 22.0, 23.0]);
    }

    #[test]
    fn try_as_matrix_rejects_non_f64() {
        let a = crate::fa_i32!("a", 1, 2, 3);
        let table = Table::new("t".into(), Some(vec![a]));
        let view = TableV::from_table(table, 0, 3);
        let err = view.try_as_matrix().expect_err("non-f64 column must reject");
        assert!(matches!(err, MinarrowError::TypeError { .. }));
    }
}

#[cfg(test)]
mod mat_macro_tests {
    #[test]
    fn literal_columns_form_builds_3x2() {
        let m = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert_eq!(m.n_rows, 3);
        assert_eq!(m.n_cols, 2);
        assert_eq!(m.col(0), &[1.0, 2.0, 3.0]);
        assert_eq!(m.col(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn slice_of_slices_form_builds_equivalent_matrix() {
        let cols: &[&[f64]] = &[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]];
        let m = mat!(cols);
        let expected = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        assert_eq!(m.n_rows, expected.n_rows);
        assert_eq!(m.n_cols, expected.n_cols);
        for c in 0..m.n_cols {
            assert_eq!(m.col(c), expected.col(c));
        }
    }

    #[test]
    #[should_panic(expected = "ColumnLengthMismatch")]
    fn literal_columns_length_mismatch_panics() {
        let _ = mat![[1.0, 2.0, 3.0], [4.0, 5.0]];
    }

    #[test]
    #[should_panic(expected = "ColumnLengthMismatch")]
    fn slice_of_slices_length_mismatch_panics() {
        let cols: &[&[f64]] = &[&[1.0, 2.0, 3.0], &[4.0, 5.0]];
        let _ = mat!(cols);
    }
}
