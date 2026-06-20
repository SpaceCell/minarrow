// Copyright 2025-2026 Peter Garfield Bower. All Rights Reserved.
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

//! # Shape Trait Module
//!
//! Unified way to describe the dimensionality “shape” of any `Value`.  
//!
//! Shapes recursively support scalars, arrays, tables,
//! tuples, chunked data, and arbitrary nested collections,
//! but include standard accessors for 1d, 2d, etc. to avoid penalising you.

use crate::enums::shape_dim::ShapeDim;

/// Shape trait.
///
/// Returns a recursively-describable `Shape` for the receiver.
///
/// Includes accessor types for common use cases e.g., shape_1d, shape_2d,
/// which are automatic provided the implementor implements `shape`.
pub trait Shape {
    /// Returns arbitrary Shape dimension for any data shape
    fn shape(&self) -> ShapeDim;

    /// Returns the first dimension shape
    ///
    /// Exists to bypass a match on `ShapeDim` for `Array` shaped types
    fn shape_1d(&self) -> usize {
        match self.shape() {
            ShapeDim::Rank0(n) => n,
            ShapeDim::Rank1(n) => n,
            ShapeDim::Rank2 { rows, .. } => rows,
            ShapeDim::Rank3 { x, .. } => x,
            ShapeDim::Rank4 { a, .. } => a,
            ShapeDim::RankN(dims) => *dims.get(0).unwrap_or(&1),
            ShapeDim::Collection(items) => items.iter().map(|x| x.shape_1d()).sum(),
            ShapeDim::Dictionary { .. } => panic!("shape_1d: incompatible Dictionary shape"),
            ShapeDim::Unknown => panic!("shape_1d: incompatible Unknown shape"),
        }
    }

    /// Returns the first and second dimension shapes
    ///
    /// Exists to bypass a match on `ShapeDim` for `Table` shaped types
    fn shape_2d(&self) -> (usize, usize) {
        match self.shape() {
            ShapeDim::Rank0(n) => (n, 1),
            ShapeDim::Rank1(n) => (n, 1),
            ShapeDim::Rank2 { rows, cols } => (rows, cols),
            ShapeDim::Rank3 { x, y, .. } => (x, y),
            ShapeDim::Rank4 { a, b, .. } => (a, b),
            ShapeDim::RankN(dims) => (*dims.get(0).unwrap_or(&1), *dims.get(1).unwrap_or(&1)),
            ShapeDim::Collection(items) => {
                let mut total_rows = 0usize;
                let mut ref_cols: Option<usize> = None;

                for item in items {
                    let (rows, cols) = item.shape_2d();
                    total_rows += rows;

                    match ref_cols {
                        None => ref_cols = Some(cols),
                        Some(prev) if prev == cols => {}
                        Some(prev) => panic!(
                            "shape_2d: column mismatch in Collection: {} vs {}",
                            prev, cols
                        ),
                    }
                }

                (total_rows, ref_cols.unwrap_or(1))
            }
            ShapeDim::Dictionary { .. } => panic!("shape_2d: incompatible Dictionary shape"),
            ShapeDim::Unknown => panic!("shape_2d: incompatible Unknown shape"),
        }
    }

    /// Returns the first, second and third dimension shapes
    ///
    /// Exists to bypass a match on `ShapeDim` for 3D types
    fn shape_3d(&self) -> (usize, usize, usize) {
        match self.shape() {
            ShapeDim::Rank0(n) => (n, 1, 1),
            ShapeDim::Rank1(n) => (n, 1, 1),
            ShapeDim::Rank2 { rows, cols } => (rows, cols, 1),
            ShapeDim::Rank3 { x, y, z } => (x, y, z),
            ShapeDim::Rank4 { a, b, c, .. } => (a, b, c),
            ShapeDim::RankN(dims) => (
                *dims.get(0).unwrap_or(&1),
                *dims.get(1).unwrap_or(&1),
                *dims.get(2).unwrap_or(&1),
            ),
            ShapeDim::Collection(items) => {
                let mut total_a = 0usize;
                let mut ref_b: Option<usize> = None;
                let mut ref_c: Option<usize> = None;

                for item in items {
                    let (a, b, c) = item.shape_3d();
                    total_a += a;

                    match ref_b {
                        None => ref_b = Some(b),
                        Some(prev) if prev == b => {}
                        Some(prev) => panic!(
                            "shape_3d: 2nd dim mismatch in Collection: {} vs {}",
                            prev, b
                        ),
                    }

                    match ref_c {
                        None => ref_c = Some(c),
                        Some(prev) if prev == c => {}
                        Some(prev) => panic!(
                            "shape_3d: 3rd dim mismatch in Collection: {} vs {}",
                            prev, c
                        ),
                    }
                }

                (total_a, ref_b.unwrap_or(1), ref_c.unwrap_or(1))
            }
            ShapeDim::Dictionary { .. } => panic!("shape_3d: incompatible Dictionary shape"),
            ShapeDim::Unknown => panic!("shape_3d: incompatible Unknown shape"),
        }
    }

    /// Returns the first, second, third and fourth dimension shapes
    ///
    /// Exists to bypass a match on `ShapeDim` for 4D types
    fn shape_4d(&self) -> (usize, usize, usize, usize) {
        match self.shape() {
            ShapeDim::Rank0(n) => (n, 1, 1, 1),
            ShapeDim::Rank1(n) => (n, 1, 1, 1),
            ShapeDim::Rank2 { rows, cols } => (rows, cols, 1, 1),
            ShapeDim::Rank3 { x, y, z } => (x, y, z, 1),
            ShapeDim::Rank4 { a, b, c, d } => (a, b, c, d),
            ShapeDim::RankN(dims) => (
                *dims.get(0).unwrap_or(&1),
                *dims.get(1).unwrap_or(&1),
                *dims.get(2).unwrap_or(&1),
                *dims.get(3).unwrap_or(&1),
            ),
            ShapeDim::Collection(items) => {
                let mut total_a = 0usize;
                let mut ref_b: Option<usize> = None;
                let mut ref_c: Option<usize> = None;
                let mut ref_d: Option<usize> = None;

                for item in items {
                    let (a, b, c, d) = item.shape_4d();
                    total_a += a;

                    match ref_b {
                        None => ref_b = Some(b),
                        Some(prev) if prev == b => {}
                        Some(prev) => panic!("shape_4d: 2nd dim mismatch: {} vs {}", prev, b),
                    }

                    match ref_c {
                        None => ref_c = Some(c),
                        Some(prev) if prev == c => {}
                        Some(prev) => panic!("shape_4d: 3rd dim mismatch: {} vs {}", prev, c),
                    }

                    match ref_d {
                        None => ref_d = Some(d),
                        Some(prev) if prev == d => {}
                        Some(prev) => panic!("shape_4d: 4th dim mismatch: {} vs {}", prev, d),
                    }
                }

                (
                    total_a,
                    ref_b.unwrap_or(1),
                    ref_c.unwrap_or(1),
                    ref_d.unwrap_or(1),
                )
            }
            ShapeDim::Dictionary { .. } => panic!("shape_4d: incompatible Dictionary shape"),
            ShapeDim::Unknown => panic!("shape_4d: incompatible Unknown shape"),
        }
    }
}
