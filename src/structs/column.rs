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

//! # **Column Module** - *Lazy Column Reference*
//!
//! Provides the `Column` type for referencing columns by name without requiring
//! explicit Field construction. Used for ergonomic column selection APIs.
//!
//! # Example
//! ```rust
//! use minarrow::{column, Column};
//!
//! // Create column references
//! let col_a = column("employee_id");
//! let col_b = Column::new("salary");
//! ```

/// Lazy reference to a column by name
///
/// `Column` wraps a string column name and defers resolution until selection time.
/// This separates user intent ("I want column A") from runtime data (Field with dtype, metadata).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Column {
    name: String,
}

impl Column {
    /// Create a new column reference
    ///
    /// # Example
    /// ```rust
    /// use minarrow::Column;
    ///
    /// let col = Column::new("employee_id");
    /// assert_eq!(col.name(), "employee_id");
    /// ```
    pub fn new(name: impl Into<String>) -> Self {
        Column { name: name.into() }
    }

    /// Get the column name
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Consume and return the column name as a String
    #[inline]
    pub fn into_name(self) -> String {
        self.name
    }
}

/// User-facing constructor for column references
///
/// # Example
/// ```rust
/// use minarrow::column;
///
/// let col = column("employee_id");
/// assert_eq!(col.name(), "employee_id");
/// ```
pub fn column(name: impl Into<String>) -> Column {
    Column::new(name)
}

impl From<&str> for Column {
    fn from(name: &str) -> Self {
        Column::new(name)
    }
}

impl From<String> for Column {
    fn from(name: String) -> Self {
        Column::new(name)
    }
}

impl AsRef<str> for Column {
    fn as_ref(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_creation() {
        let col_a = column("employee_id");
        assert_eq!(col_a.name(), "employee_id");
    }

    #[test]
    fn test_column_new() {
        let col = Column::new("salary");
        assert_eq!(col.name(), "salary");
    }

    #[test]
    fn test_column_from_str() {
        let col_a: Column = "salary".into();
        assert_eq!(col_a.name(), "salary");
    }

    #[test]
    fn test_column_from_string() {
        let name = String::from("department");
        let col_a: Column = name.into();
        assert_eq!(col_a.name(), "department");
    }

    #[test]
    fn test_column_into_name() {
        let col = column("age");
        let name = col.into_name();
        assert_eq!(name, "age");
    }

    #[test]
    fn test_column_equality() {
        let col_a = column("id");
        let col_b = column("id");
        let col_c = column("name");

        assert_eq!(col_a, col_b);
        assert_ne!(col_a, col_c);
    }

    #[test]
    fn test_column_as_ref() {
        let col = column("test");
        let s: &str = col.as_ref();
        assert_eq!(s, "test");
    }
}
