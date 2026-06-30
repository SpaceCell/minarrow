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

//! # Combine Trait Module
//!
//! Builds a new container of the same shape from a vector of elements,
//! inheriting `&self`'s name, metadata, and grouping keys.
//!
//! Implementations:
//! - `Table`: `Element = FieldArray`, validates a consistent row count.
//! - `TableV`: `Element = FieldArray`, produces an owned `Table`.
//! - `Cube`: `Element = Table`, element count must match the group count.
//!
//! Differs from [`Concatenate`]: `concat` joins two same-shape instances
//! along their length axis. `combine` produces a new instance by
//! substituting elements into a template.
//!
//! [`Concatenate`]: crate::Concatenate

#[cfg(feature = "cube")]
use crate::Cube;
#[cfg(feature = "views")]
use crate::TableV;
use crate::enums::error::MinarrowError;
use crate::{FieldArray, Table};

/// Build a new container from `elements`, inheriting `&self`'s name,
/// metadata, and grouping keys. Views produce owned output without
/// materialising.
pub trait Combine {
    /// Element type the container is composed of.
    type Element;
    /// Owned container shape produced by `combine`.
    type Output;

    /// Build the output from `elements`, inheriting `self`'s name,
    /// metadata, and any grouping state.
    fn combine(&self, elements: Vec<Self::Element>) -> Result<Self::Output, MinarrowError>;
}

impl Combine for Table {
    type Element = FieldArray;
    type Output = Table;

    fn combine(&self, elements: Vec<Self::Element>) -> Result<Self::Output, MinarrowError> {
        let n_rows = elements.first().map(|fa| fa.array.len()).unwrap_or(0);
        for (i, fa) in elements.iter().enumerate() {
            if fa.array.len() != n_rows {
                return Err(MinarrowError::ColumnLengthMismatch {
                    col: i,
                    expected: n_rows,
                    found: fa.array.len(),
                });
            }
        }
        #[allow(unused_mut)]
        let mut table = Table::build(elements, n_rows, self.name.clone());
        #[cfg(feature = "table_metadata")]
        {
            table.metadata = self.metadata.clone();
        }
        Ok(table)
    }
}

#[cfg(feature = "views")]
impl Combine for TableV {
    type Element = FieldArray;
    type Output = Table;

    fn combine(&self, elements: Vec<Self::Element>) -> Result<Self::Output, MinarrowError> {
        let n_rows = elements.first().map(|fa| fa.array.len()).unwrap_or(0);
        for (i, fa) in elements.iter().enumerate() {
            if fa.array.len() != n_rows {
                return Err(MinarrowError::ColumnLengthMismatch {
                    col: i,
                    expected: n_rows,
                    found: fa.array.len(),
                });
            }
        }
        Ok(Table::build(elements, n_rows, self.name.clone()))
    }
}

#[cfg(feature = "cube")]
impl Combine for Cube {
    type Element = Table;
    type Output = Cube;

    fn combine(&self, elements: Vec<Self::Element>) -> Result<Self::Output, MinarrowError> {
        if elements.len() != self.tables.len() {
            return Err(MinarrowError::ShapeError {
                message: format!(
                    "Cube::combine: expected {} elements (one per group), got {}",
                    self.tables.len(),
                    elements.len()
                ),
            });
        }
        Ok(Cube::from_tables(
            elements,
            self.name.clone(),
            self.third_dim_index.clone(),
        ))
    }
}
