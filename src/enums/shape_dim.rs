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

//! # ShapeDim Enum Module
//!
//! Companion to [crate::traits::shape::Shape];
//!
//! Contains all supported `Shape` variants.

use crate::traits::shape::Shape;

/// Recursively-describable dimensional rank for any `Value`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeDim {
    /// Rank-0 - must always be `1`
    Rank0(usize),

    /// Array row count
    Rank1(usize),

    /// Relational table with row/column counts.
    Rank2 { rows: usize, cols: usize },

    /// 3d object
    Rank3 { x: usize, y: usize, z: usize },

    /// 4d Object
    Rank4 {
        a: usize,
        b: usize,
        c: usize,
        d: usize,
    },

    /// N-dimensional tensor.
    RankN(Vec<usize>),

    /// Dictionary shape
    Dictionary {
        // Number of keys
        n_keys: usize,
        // Number of values for each key
        n_values: Vec<usize>,
    },

    /// Heterogeneous ordered collection.
    /// Covers lists, tuples, cubes (with varying row-counts) and user-custom chunked types.
    ///
    /// The order is significant; for tuples it is the fixed arity order.
    Collection(Vec<ShapeDim>),

    /// Shape could not be determined.
    Unknown,
}

/// Implement `Shape` for `ShapeDim` so recursive calls like `item.shape_3d()`
/// compile when iterating `Collection(Vec<ShapeDim>)`.
impl Shape for ShapeDim {
    fn shape(&self) -> ShapeDim {
        self.clone()
    }
}
