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

//! Integration tests for pandas-style selection on Table and TableV

#![cfg(feature = "select")]

use minarrow::traits::selection::{ColumnSelection, RowSelection};
use minarrow::{Array, FieldArray, IntegerArray, MaskedArray, StringArray, Table};

#[test]
fn test_table_column_selection_by_names() {
    let table = create_test_table();

    // Select columns by names
    let view = table.c(&["name", "value"]);

    assert_eq!(view.n_cols(), 2);
    assert_eq!(view.col_names(), vec!["name", "value"]);
    assert_eq!(view.n_rows(), 5);
}

#[test]
fn test_table_column_selection_by_indices() {
    let table = create_test_table();

    // Select columns by indices
    let view = table.c(&[0, 2]);

    assert_eq!(view.n_cols(), 2);
    assert_eq!(view.n_rows(), 5);
}

#[test]
fn test_table_column_selection_by_range() {
    let table = create_test_table();

    // Select columns by range
    let view = table.c(0..2);

    assert_eq!(view.n_cols(), 2);
    assert_eq!(view.n_rows(), 5);
}

#[test]
fn test_table_row_selection_by_range() {
    let table = create_test_table();

    // Select rows by range
    let view = table.r(1..4);

    assert_eq!(view.n_rows(), 3);
    assert_eq!(view.n_cols(), 3);
}

#[test]
fn test_table_row_selection_by_indices() {
    let table = create_test_table();

    // Select specific rows
    let view = table.r(&[0, 2, 4]);

    assert_eq!(view.n_rows(), 3);
    assert_eq!(view.n_cols(), 3);
}

#[test]
fn test_chained_column_then_row_selection() {
    let table = create_test_table();

    // Select columns first, then rows
    let view = table.c(&["id", "value"]).r(1..4);

    assert_eq!(view.n_cols(), 2);
    assert_eq!(view.n_rows(), 3);

    // Verify column names are correct
    let names = view.col_names();
    assert_eq!(names, vec!["id", "value"]);
}

#[test]
fn test_chained_row_then_column_selection() {
    let table = create_test_table();

    // Select rows first, then columns
    let view = table.r(1..4).c(&["name"]);

    assert_eq!(view.n_rows(), 3);
    assert_eq!(view.n_cols(), 1);
    assert_eq!(view.col_names(), vec!["name"]);
}

#[test]
fn test_tablev_column_refinement() {
    let table = create_test_table();

    // Create initial view with all columns
    let view1 = table.c(&["id", "name", "value"]);
    assert_eq!(view1.n_cols(), 3);

    // Refine to fewer columns
    let view2 = view1.c(&["name", "value"]);
    assert_eq!(view2.n_cols(), 2);
    assert_eq!(view2.col_names(), vec!["name", "value"]);
}

#[test]
fn test_tablev_row_refinement() {
    let table = create_test_table();

    // Create initial view with rows 0..5
    let view1 = table.r(0..5);
    assert_eq!(view1.n_rows(), 5);

    // Refine to rows 1..3 (which maps to physical rows 1..3)
    let view2 = view1.r(&[1, 2]);
    assert_eq!(view2.n_rows(), 2);
}

#[test]
fn test_tablev_methods_respect_column_selection() {
    let table = create_test_table();
    let view = table.c(&["id", "value"]);

    // Test n_cols
    assert_eq!(view.n_cols(), 2);

    // Test col_names
    assert_eq!(view.col_names(), vec!["id", "value"]);

    // Test col_ix by index (logical index within selection)
    let col0 = view.col_ix(0);
    assert!(col0.is_some());

    let col1 = view.col_ix(1);
    assert!(col1.is_some());

    // Logical index 2 doesn't exist (only 2 columns selected)
    let col2 = view.col_ix(2);
    assert!(col2.is_none());

    // Test col by name (alias for c)
    assert_eq!(view.col("id").col_ix(0).is_some(), true);
    assert_eq!(view.col("value").col_ix(0).is_some(), true);
    assert_eq!(view.col("name").col_ix(0).is_none(), true); // Not in selection
}

#[test]
fn test_tablev_methods_respect_row_selection() {
    let table = create_test_table();
    let view = table.r(&[1, 3]);

    // Test n_rows
    assert_eq!(view.n_rows(), 2);
}

#[test]
fn test_to_table_respects_selections() {
    let table = create_test_table();

    // Create view with column and row selections
    let view = table.c(&["id", "value"]).r(1..4);

    // Convert back to table
    let materialised = view.to_table();

    assert_eq!(materialised.n_cols(), 2);
    assert_eq!(materialised.n_rows, 3);
}

// ===== Tests for selection API =====

#[test]
fn test_column_selection_methods() {
    let table = create_test_table();

    // Test .c() method - column selection by names
    let view1 = table.c(&["name", "value"]);
    assert_eq!(view1.n_cols(), 2);
    assert_eq!(view1.col_names(), vec!["name", "value"]);

    // Test .c() method - column selection by indices
    let view2 = table.c(&[0, 2]);
    assert_eq!(view2.n_cols(), 2);

    // Test .col() alias (same as .c())
    let view3 = table.col("id");
    assert_eq!(view3.n_cols(), 1);
    assert_eq!(view3.col_names(), vec!["id"]);
}

#[test]
fn test_row_selection_methods() {
    let table = create_test_table();

    // Test .r() method - row selection by range
    let view1 = table.r(1..4);
    assert_eq!(view1.n_rows(), 3);

    // Test .r() method - row selection by indices
    let view2 = table.r(&[0, 2, 4]);
    assert_eq!(view2.n_rows(), 3);
}

#[test]
fn test_chained_selection() {
    let table = create_test_table();

    // Chain column and row selection
    let view1 = table.c(&["id", "value"]).r(1..4);
    assert_eq!(view1.n_cols(), 2);
    assert_eq!(view1.n_rows(), 3);

    // Chain row and column selection
    let view2 = table.r(1..4).c(&["name"]);
    assert_eq!(view2.n_cols(), 1);
    assert_eq!(view2.n_rows(), 3);
}

#[test]
fn test_tablev_new_api_refinement() {
    let table = create_test_table();

    // Create initial view with .f() and .d()
    let view1 = table.c(&["id", "name", "value"]).r(0..5);
    assert_eq!(view1.n_cols(), 3);
    assert_eq!(view1.n_rows(), 5);

    // Refine using new API
    let view2 = view1.c(&["name", "value"]).r(&[1, 2, 3]);
    assert_eq!(view2.n_cols(), 2);
    assert_eq!(view2.n_rows(), 3);
    assert_eq!(view2.col_names(), vec!["name", "value"]);
}

#[test]
fn test_api_compatibility() {
    let table = create_test_table();

    // Old API
    let view_old = table.c(&["id", "value"]).r(1..4);

    // New API (should produce identical results)
    let view_new = table.c(&["id", "value"]).r(1..4);

    assert_eq!(view_old.n_cols(), view_new.n_cols());
    assert_eq!(view_old.n_rows(), view_new.n_rows());
    assert_eq!(view_old.col_names(), view_new.col_names());
}

// Helper function to create a test table
fn create_test_table() -> Table {
    // Create id column (0, 1, 2, 3, 4)
    let mut id_arr = IntegerArray::<i32>::default();
    for i in 0..5 {
        id_arr.push(i);
    }

    // Create name column
    let mut name_arr = StringArray::<u32>::default();
    for i in 0..5 {
        name_arr.push(format!("name{}", i));
    }

    // Create value column
    let mut value_arr = IntegerArray::<i64>::default();
    for i in 0..5 {
        value_arr.push((i * 10) as i64);
    }

    let col_id = FieldArray::from_arr("id", Array::from_int32(id_arr));
    let col_name = FieldArray::from_arr("name", Array::from_string32(name_arr));
    let col_value = FieldArray::from_arr("value", Array::from_int64(value_arr));

    Table::new(
        "TestTable".to_string(),
        Some(vec![col_id, col_name, col_value]),
    )
}

// DataSelector::to_owned and the boxed DataSelector impl

use minarrow::traits::selection::DataSelector;

/// Asserts that the owned selector matches the original on indices and contiguity.
fn assert_owned_matches<S: DataSelector>(selector: S, count: usize) {
    let owned = selector.to_owned();
    assert_eq!(owned.resolve_indices(count), selector.resolve_indices(count));
    assert_eq!(owned.is_contiguous(), selector.is_contiguous());
}

#[test]
fn test_data_selector_to_owned_single_index() {
    assert_owned_matches(2usize, 5);
    assert_owned_matches(2i32, 5);
    assert_eq!(DataSelector::to_owned(&3usize).resolve_indices(5), vec![3]);
}

#[test]
fn test_data_selector_to_owned_index_lists() {
    let slice: &[usize] = &[1, 3, 9];
    assert_owned_matches(slice, 5);
    assert_owned_matches(&[0usize, 4], 5);
    assert_owned_matches(vec![4usize, 2, 7], 5);
    assert!(!DataSelector::to_owned(&vec![1usize, 2]).is_contiguous());
}

#[test]
fn test_data_selector_to_owned_ranges() {
    assert_owned_matches(1usize..4, 5);
    assert_owned_matches(1i32..4, 5);
    assert_owned_matches(2usize.., 5);
    assert_owned_matches(2i32.., 5);
    assert_owned_matches(..3usize, 5);
    assert_owned_matches(..3i32, 5);
    assert_owned_matches(.., 5);
    assert_owned_matches(1usize..=3, 5);
    assert_owned_matches(1i32..=3, 5);
    assert!(DataSelector::to_owned(&(1usize..4)).is_contiguous());
}

#[test]
fn test_data_selector_boxed_impl_forwards() {
    let boxed: Box<dyn DataSelector + Send + Sync> = Box::new(1usize..4);
    assert_eq!(boxed.resolve_indices(5), vec![1, 2, 3]);
    assert!(boxed.is_contiguous());

    let boxed_list: Box<dyn DataSelector + Send + Sync> = Box::new(vec![0usize, 2]);
    assert_eq!(boxed_list.resolve_indices(5), vec![0, 2]);
    assert!(!boxed_list.is_contiguous());

    // A boxed selector clones into another boxed selector with the same behaviour.
    let again = DataSelector::to_owned(&boxed);
    assert_eq!(again.resolve_indices(5), vec![1, 2, 3]);
    assert!(again.is_contiguous());
}

#[test]
fn test_data_selector_boxed_usable_with_row_selection() {
    let table = create_test_table();
    let boxed: Box<dyn DataSelector + Send + Sync> = DataSelector::to_owned(&(1usize..3));
    let view = table.r(boxed);
    assert_eq!(view.n_rows(), 2);
}

#[cfg(feature = "ndarray")]
#[test]
fn test_data_selector_boxed_resolve_axis_forwards() {
    let range: Box<dyn DataSelector + Send + Sync> = Box::new(1usize..4);
    assert_eq!(range.resolve_axis(5), (1, 4, false));
    let single: Box<dyn DataSelector + Send + Sync> = Box::new(2usize);
    assert_eq!(single.resolve_axis(5), (2, 3, true));
    let full: Box<dyn DataSelector + Send + Sync> = DataSelector::to_owned(&..);
    assert_eq!(full.resolve_axis(5), (0, 5, false));
}
