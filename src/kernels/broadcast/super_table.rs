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

use crate::enums::error::MinarrowError;
use crate::enums::operators::ArithmeticOperator;
use crate::kernels::broadcast::array::broadcast_array_to_table;
use crate::kernels::broadcast::table::{
    broadcast_table_to_array, broadcast_table_to_scalar, broadcast_table_with_operator,
};
use crate::{SuperTable, SuperTableV, Value};
use std::sync::Arc;

#[cfg(feature = "scalar_type")]
use crate::Scalar;

#[cfg(any(feature = "views", feature = "scalar_type"))]
use crate::{Array, FieldArray};

#[cfg(feature = "views")]
use crate::{ArrayV, NumericArrayV, TextArrayV};

#[cfg(all(feature = "views", feature = "datetime"))]
use crate::TemporalArrayV;

/// General super table broadcasting function that supports all arithmetic operators
#[cfg(feature = "chunked")]
pub fn broadcast_super_table_with_operator(
    op: ArithmeticOperator,
    lhs: impl Into<SuperTableV>,
    rhs: impl Into<SuperTableV>,
) -> Result<SuperTable, MinarrowError> {
    let lhs_table: SuperTableV = lhs.into();
    let rhs_table: SuperTableV = rhs.into();

    // Ensure tables have same number of chunks
    if lhs_table.slices.len() != rhs_table.slices.len() {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "SuperTable chunk count mismatch: {} vs {}",
                lhs_table.slices.len(),
                rhs_table.slices.len()
            ),
        });
    }

    let mut result_tables = Vec::new();

    for (lhs_slice, rhs_slice) in lhs_table.slices.iter().zip(rhs_table.slices.iter()) {
        // Convert slices to full tables for broadcasting
        let lhs_table = lhs_slice.to_table();
        let rhs_table = rhs_slice.to_table();

        // Broadcast using general table routing
        let result_table = broadcast_table_with_operator(op, lhs_table, rhs_table)?;
        result_tables.push(result_table);
    }

    Ok(SuperTable::from_batches(
        result_tables.into_iter().map(Arc::new).collect(),
        None,
    ))
}

/// Broadcast SuperTable to Scalar - apply scalar to each batch
#[cfg(all(feature = "chunked", feature = "scalar_type"))]
pub fn broadcast_supertable_to_scalar(
    op: ArithmeticOperator,
    super_table: &SuperTable,
    scalar: &Scalar,
) -> Result<SuperTable, MinarrowError> {
    let new_tables: Result<Vec<_>, _> = super_table
        .batches
        .iter()
        .map(|table| broadcast_table_to_scalar(op, table, scalar).map(Arc::new))
        .collect();
    Ok(SuperTable::from_batches(
        new_tables?,
        Some(super_table.name.clone()),
    ))
}

/// Broadcast SuperTable to Array - apply array to each batch
#[cfg(feature = "chunked")]
pub fn broadcast_supertable_to_array(
    op: ArithmeticOperator,
    super_table: &SuperTable,
    array: &Array,
) -> Result<SuperTable, MinarrowError> {
    let new_tables: Result<Vec<_>, _> = super_table
        .batches
        .iter()
        .map(|table| broadcast_table_to_array(op, table, array).map(Arc::new))
        .collect();
    Ok(SuperTable::from_batches(
        new_tables?,
        Some(super_table.name.clone()),
    ))
}

/// Broadcast FieldArray to SuperTable - apply field array's inner array to each batch
#[cfg(feature = "chunked")]
pub fn broadcast_fieldarray_to_supertable(
    op: ArithmeticOperator,
    field_array: &FieldArray,
    super_table: &SuperTable,
) -> Result<SuperTable, MinarrowError> {
    let new_tables: Result<Vec<_>, _> = super_table
        .batches
        .iter()
        .map(|table| broadcast_array_to_table(op, &field_array.array, table).map(Arc::new))
        .collect();
    Ok(SuperTable::from_batches(
        new_tables?,
        Some(super_table.name.clone()),
    ))
}

/// Broadcast SuperTable to FieldArray - apply field array's inner array to each batch
#[cfg(feature = "chunked")]
pub fn broadcast_supertable_to_fieldarray(
    op: ArithmeticOperator,
    super_table: &SuperTable,
    field_array: &FieldArray,
) -> Result<SuperTable, MinarrowError> {
    let new_tables: Result<Vec<_>, _> = super_table
        .batches
        .iter()
        .map(|table| broadcast_table_to_array(op, table, &field_array.array).map(Arc::new))
        .collect();
    Ok(SuperTable::from_batches(
        new_tables?,
        Some(super_table.name.clone()),
    ))
}

/// Broadcast ArrayView to SuperTable - convert view to array and apply to each batch
#[cfg(all(feature = "chunked", feature = "views"))]
pub fn broadcast_arrayview_to_supertable(
    op: ArithmeticOperator,
    array_view: &ArrayV,
    super_table: &SuperTable,
) -> Result<SuperTable, MinarrowError> {
    let array = array_view.to_array();
    let new_tables: Result<Vec<_>, _> = super_table
        .batches
        .iter()
        .map(|table| broadcast_array_to_table(op, &array, table).map(Arc::new))
        .collect();
    Ok(SuperTable::from_batches(
        new_tables?,
        Some(super_table.name.clone()),
    ))
}

/// Broadcast SuperTable to ArrayView - convert view to array and apply to each batch
#[cfg(all(feature = "chunked", feature = "views"))]
pub fn broadcast_supertable_to_arrayview(
    op: ArithmeticOperator,
    super_table: &SuperTable,
    array_view: &ArrayV,
) -> Result<SuperTable, MinarrowError> {
    let array = array_view.to_array();
    let new_tables: Result<Vec<_>, _> = super_table
        .batches
        .iter()
        .map(|table| broadcast_table_to_array(op, table, &array).map(Arc::new))
        .collect();
    Ok(SuperTable::from_batches(
        new_tables?,
        Some(super_table.name.clone()),
    ))
}

/// Broadcast NumericArrayView to SuperTable - convert view to array and apply to each batch
#[cfg(all(feature = "chunked", feature = "views"))]
pub fn broadcast_numericarrayview_to_supertable(
    op: ArithmeticOperator,
    numeric_view: &NumericArrayV,
    super_table: &SuperTable,
) -> Result<SuperTable, MinarrowError> {
    let array = Array::NumericArray(numeric_view.array.clone());
    let new_tables: Result<Vec<_>, _> = super_table
        .batches
        .iter()
        .map(|table| broadcast_array_to_table(op, &array, table).map(Arc::new))
        .collect();
    Ok(SuperTable::from_batches(
        new_tables?,
        Some(super_table.name.clone()),
    ))
}

/// Broadcast SuperTable to NumericArrayView - convert view to array and apply to each batch
#[cfg(all(feature = "chunked", feature = "views"))]
pub fn broadcast_supertable_to_numeric_arrayview(
    op: ArithmeticOperator,
    super_table: &SuperTable,
    numeric_view: &NumericArrayV,
) -> Result<SuperTable, MinarrowError> {
    let array = Array::NumericArray(numeric_view.array.clone());
    let new_tables: Result<Vec<_>, _> = super_table
        .batches
        .iter()
        .map(|table| broadcast_table_to_array(op, table, &array).map(Arc::new))
        .collect();
    Ok(SuperTable::from_batches(
        new_tables?,
        Some(super_table.name.clone()),
    ))
}

/// Broadcast TextArrayView to SuperTable - convert view to array and apply to each batch
#[cfg(all(feature = "chunked", feature = "views"))]
pub fn broadcast_textarrayview_to_supertable(
    op: ArithmeticOperator,
    text_view: &TextArrayV,
    super_table: &SuperTable,
) -> Result<SuperTable, MinarrowError> {
    let array = Array::TextArray(text_view.array.clone());
    let new_tables: Result<Vec<_>, _> = super_table
        .batches
        .iter()
        .map(|table| broadcast_array_to_table(op, &array, table).map(Arc::new))
        .collect();
    Ok(SuperTable::from_batches(
        new_tables?,
        Some(super_table.name.clone()),
    ))
}

/// Broadcast SuperTable to TextArrayView - convert view to array and apply to each batch
#[cfg(all(feature = "chunked", feature = "views"))]
pub fn broadcast_supertable_to_text_arrayview(
    op: ArithmeticOperator,
    super_table: &SuperTable,
    text_view: &TextArrayV,
) -> Result<SuperTable, MinarrowError> {
    let array = Array::TextArray(text_view.array.clone());
    let new_tables: Result<Vec<_>, _> = super_table
        .batches
        .iter()
        .map(|table| broadcast_table_to_array(op, table, &array).map(Arc::new))
        .collect();
    Ok(SuperTable::from_batches(
        new_tables?,
        Some(super_table.name.clone()),
    ))
}

/// Broadcast TemporalArrayView to SuperTable - convert view to array and apply to each batch
#[cfg(all(feature = "chunked", feature = "views", feature = "datetime"))]
pub fn broadcast_temporalarrayview_to_supertable(
    op: ArithmeticOperator,
    temporal_view: &TemporalArrayV,
    super_table: &SuperTable,
) -> Result<SuperTable, MinarrowError> {
    let array = Array::TemporalArray(temporal_view.array.clone());
    let new_tables: Result<Vec<_>, _> = super_table
        .batches
        .iter()
        .map(|table| broadcast_array_to_table(op, &array, table).map(Arc::new))
        .collect();
    Ok(SuperTable::from_batches(
        new_tables?,
        Some(super_table.name.clone()),
    ))
}

/// Broadcast SuperTable to TemporalArrayView - convert view to array and apply to each batch
#[cfg(all(feature = "chunked", feature = "views", feature = "datetime"))]
pub fn broadcast_supertable_to_temporal_arrayview(
    op: ArithmeticOperator,
    super_table: &SuperTable,
    temporal_view: &TemporalArrayV,
) -> Result<SuperTable, MinarrowError> {
    let array = Array::TemporalArray(temporal_view.array.clone());
    let new_tables: Result<Vec<_>, _> = super_table
        .batches
        .iter()
        .map(|table| broadcast_table_to_array(op, table, &array).map(Arc::new))
        .collect();
    Ok(SuperTable::from_batches(
        new_tables?,
        Some(super_table.name.clone()),
    ))
}

/// Broadcast SuperArray to SuperTable - chunks must align
#[cfg(feature = "chunked")]
pub fn broadcast_superarray_to_supertable(
    op: ArithmeticOperator,
    super_array: &crate::SuperArray,
    super_table: &SuperTable,
) -> Result<SuperTable, MinarrowError> {
    // Verify chunks align
    if super_array.n_chunks() != super_table.batches.len() {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "SuperArray has {} chunks but SuperTable has {} batches",
                super_array.n_chunks(),
                super_table.batches.len()
            ),
        });
    }

    // Broadcast each chunk with corresponding table
    let mut result_tables = Vec::with_capacity(super_table.batches.len());
    for (i, table) in super_table.batches.iter().enumerate() {
        let chunk = &super_array.chunks()[i];
        let broadcasted = broadcast_array_to_table(op, chunk, table)?;
        result_tables.push(Arc::new(broadcasted));
    }

    Ok(SuperTable::from_batches(
        result_tables,
        Some(super_table.name.clone()),
    ))
}

/// Broadcast SuperTable to SuperArray - chunks must align (reverse order)
#[cfg(feature = "chunked")]
pub fn broadcast_supertable_to_superarray(
    op: ArithmeticOperator,
    super_table: &SuperTable,
    super_array: &crate::SuperArray,
) -> Result<SuperTable, MinarrowError> {
    if super_table.batches.len() != super_array.n_chunks() {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "SuperTable has {} batches but SuperArray has {} chunks",
                super_table.batches.len(),
                super_array.n_chunks()
            ),
        });
    }

    let mut result_tables = Vec::with_capacity(super_table.batches.len());
    for (i, table) in super_table.batches.iter().enumerate() {
        let chunk = &super_array.chunks()[i];
        let broadcasted = broadcast_table_to_array(op, table, chunk)?;
        result_tables.push(Arc::new(broadcasted));
    }

    Ok(SuperTable::from_batches(
        result_tables,
        Some(super_table.name.clone()),
    ))
}

/// Broadcast SuperArray to SuperTableView - chunks must align
#[cfg(all(feature = "chunked", feature = "views"))]
pub fn broadcast_superarray_to_supertableview(
    op: ArithmeticOperator,
    super_array: &crate::SuperArray,
    super_table_view: &SuperTableV,
) -> Result<SuperTable, MinarrowError> {
    // Verify chunks align
    if super_array.n_chunks() != super_table_view.slices.len() {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "SuperArray has {} chunks but SuperTableView has {} slices",
                super_array.n_chunks(),
                super_table_view.slices.len()
            ),
        });
    }

    // Broadcast each chunk with corresponding table view
    let mut result_tables = Vec::with_capacity(super_table_view.slices.len());
    for (i, table_view) in super_table_view.slices.iter().enumerate() {
        let chunk = &super_array.chunks()[i];
        let table = table_view.to_table();
        let broadcasted = broadcast_array_to_table(op, chunk, &table)?;
        result_tables.push(Arc::new(broadcasted));
    }

    Ok(SuperTable::from_batches(
        result_tables,
        Some("broadcasted".to_string()),
    ))
}

/// Broadcast SuperTableView to SuperArray - chunks must align (reverse order)
#[cfg(all(feature = "chunked", feature = "views"))]
pub fn broadcast_supertableview_to_superarray(
    op: ArithmeticOperator,
    super_table_view: &SuperTableV,
    super_array: &crate::SuperArray,
) -> Result<SuperTable, MinarrowError> {
    if super_table_view.slices.len() != super_array.n_chunks() {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "SuperTableView has {} slices but SuperArray has {} chunks",
                super_table_view.slices.len(),
                super_array.n_chunks()
            ),
        });
    }

    let mut result_tables = Vec::with_capacity(super_table_view.slices.len());
    for (i, table_view) in super_table_view.slices.iter().enumerate() {
        let chunk = &super_array.chunks()[i];
        let table = table_view.to_table();
        let broadcasted = broadcast_table_to_array(op, &table, chunk)?;
        result_tables.push(Arc::new(broadcasted));
    }

    Ok(SuperTable::from_batches(
        result_tables,
        Some("broadcasted".to_string()),
    ))
}

/// Broadcast SuperArrayView to SuperTable - chunks must align
#[cfg(all(feature = "chunked", feature = "views"))]
pub fn broadcast_superarrayview_to_supertable(
    op: ArithmeticOperator,
    super_array_view: &crate::SuperArrayV,
    super_table: &SuperTable,
) -> Result<SuperTable, MinarrowError> {
    if super_array_view.slices.len() != super_table.batches.len() {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "SuperArrayView has {} slices but SuperTable has {} batches",
                super_array_view.slices.len(),
                super_table.batches.len()
            ),
        });
    }

    let mut result_tables = Vec::with_capacity(super_table.batches.len());
    for (i, table) in super_table.batches.iter().enumerate() {
        let array_view = &super_array_view.slices[i];
        let array = array_view.to_array();
        let broadcasted = broadcast_array_to_table(op, &array, table)?;
        result_tables.push(Arc::new(broadcasted));
    }

    Ok(SuperTable::from_batches(
        result_tables,
        Some(super_table.name.clone()),
    ))
}

/// Broadcast SuperTable to SuperArrayView - chunks must align (reverse order)
#[cfg(all(feature = "chunked", feature = "views"))]
pub fn broadcast_supertable_to_superarrayview(
    op: ArithmeticOperator,
    super_table: &SuperTable,
    super_array_view: &crate::SuperArrayV,
) -> Result<SuperTable, MinarrowError> {
    if super_table.batches.len() != super_array_view.slices.len() {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "SuperTable has {} batches but SuperArrayView has {} slices",
                super_table.batches.len(),
                super_array_view.slices.len()
            ),
        });
    }

    let mut result_tables = Vec::with_capacity(super_table.batches.len());
    for (i, table) in super_table.batches.iter().enumerate() {
        let array_view = &super_array_view.slices[i];
        let array = array_view.to_array();
        let broadcasted = broadcast_table_to_array(op, table, &array)?;
        result_tables.push(Arc::new(broadcasted));
    }

    Ok(SuperTable::from_batches(
        result_tables,
        Some(super_table.name.clone()),
    ))
}

/// Broadcast SuperArrayView to SuperTableView - chunks must align
#[cfg(all(feature = "chunked", feature = "views"))]
pub fn broadcast_superarrayview_to_supertableview(
    op: ArithmeticOperator,
    super_array_view: &crate::SuperArrayV,
    super_table_view: &SuperTableV,
) -> Result<SuperTable, MinarrowError> {
    if super_array_view.slices.len() != super_table_view.slices.len() {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "SuperArrayView has {} slices but SuperTableView has {} slices",
                super_array_view.slices.len(),
                super_table_view.slices.len()
            ),
        });
    }

    let mut result_tables = Vec::with_capacity(super_table_view.slices.len());
    for (i, table_view) in super_table_view.slices.iter().enumerate() {
        let array_view = &super_array_view.slices[i];
        let array = array_view.to_array();
        let table = table_view.to_table();
        let broadcasted = broadcast_array_to_table(op, &array, &table)?;
        result_tables.push(Arc::new(broadcasted));
    }

    Ok(SuperTable::from_batches(
        result_tables,
        Some("broadcasted".to_string()),
    ))
}

/// Broadcast SuperTableView to SuperArrayView - chunks must align (reverse order)
#[cfg(all(feature = "chunked", feature = "views"))]
pub fn broadcast_supertableview_to_superarrayview(
    op: ArithmeticOperator,
    super_table_view: &SuperTableV,
    super_array_view: &crate::SuperArrayV,
) -> Result<SuperTable, MinarrowError> {
    if super_table_view.slices.len() != super_array_view.slices.len() {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "SuperTableView has {} slices but SuperArrayView has {} slices",
                super_table_view.slices.len(),
                super_array_view.slices.len()
            ),
        });
    }

    let mut result_tables = Vec::with_capacity(super_table_view.slices.len());
    for (i, table_view) in super_table_view.slices.iter().enumerate() {
        let array_view = &super_array_view.slices[i];
        let array = array_view.to_array();
        let table = table_view.to_table();
        let broadcasted = broadcast_table_to_array(op, &table, &array)?;
        result_tables.push(Arc::new(broadcasted));
    }

    Ok(SuperTable::from_batches(
        result_tables,
        Some("broadcasted".to_string()),
    ))
}

/// Helper function for TableView-SuperTable broadcasting - convert TableView to Table then to SuperTable
#[cfg(all(feature = "views", feature = "chunked"))]
pub fn broadcast_tableview_to_supertable(
    op: ArithmeticOperator,
    table_view: &crate::TableV,
    super_table: &SuperTable,
) -> Result<Value, MinarrowError> {
    use crate::Value;
    use crate::kernels::broadcast::broadcast_value;
    let table = table_view.to_table();
    let single_table_super =
        SuperTable::from_batches(vec![Arc::new(table)], Some(super_table.name.clone()));
    let result = match (
        Value::SuperTable(single_table_super.into()),
        Value::SuperTable(super_table.clone().into()),
    ) {
        (a, b) => broadcast_value(op, a, b)?,
    };
    Ok(result)
}

/// Helper function for SuperTable-TableView broadcasting - convert TableView to Table then to SuperTable
#[cfg(all(feature = "views", feature = "chunked"))]
pub fn broadcast_supertable_to_tableview(
    op: ArithmeticOperator,
    super_table: &SuperTable,
    table_view: &crate::TableV,
) -> Result<Value, MinarrowError> {
    use crate::Value;
    use crate::kernels::broadcast::broadcast_value;
    let table = table_view.to_table();
    let single_table_super =
        SuperTable::from_batches(vec![Arc::new(table)], Some(super_table.name.clone()));
    let result = match (
        Value::SuperTable(super_table.clone().into()),
        Value::SuperTable(single_table_super.into()),
    ) {
        (a, b) => broadcast_value(op, a, b)?,
    };
    Ok(result)
}

/// Helper function for TableView-SuperTableView broadcasting - convert TableView to Table -> SuperTable -> SuperTableView
#[cfg(all(feature = "views", feature = "chunked"))]
pub fn broadcast_tableview_to_supertableview(
    op: ArithmeticOperator,
    table_view: &crate::TableV,
    super_table_view: &SuperTableV,
) -> Result<Value, MinarrowError> {
    use crate::Value;
    use crate::kernels::broadcast::broadcast_value;
    let table = table_view.to_table();
    let single_table_super =
        SuperTable::from_batches(vec![Arc::new(table)], Some("TempSuper".to_string()));
    let single_super_view = single_table_super.view(0, single_table_super.n_rows);
    let result = match (
        Value::SuperTableView(single_super_view.into()),
        Value::SuperTableView(super_table_view.clone().into()),
    ) {
        (a, b) => broadcast_value(op, a, b)?,
    };
    Ok(result)
}

/// Helper function for SuperTableView-TableView broadcasting - convert TableView to Table -> SuperTable -> SuperTableView
#[cfg(all(feature = "views", feature = "chunked"))]
pub fn broadcast_supertableview_to_tableview(
    op: ArithmeticOperator,
    super_table_view: &SuperTableV,
    table_view: &crate::TableV,
) -> Result<Value, MinarrowError> {
    use crate::Value;
    use crate::kernels::broadcast::broadcast_value;
    let table = table_view.to_table();
    let single_table_super =
        SuperTable::from_batches(vec![Arc::new(table)], Some("TempSuper".to_string()));
    let single_super_view = single_table_super.view(0, single_table_super.n_rows);
    let result = match (
        Value::SuperTableView(super_table_view.clone().into()),
        Value::SuperTableView(single_super_view.into()),
    ) {
        (a, b) => broadcast_value(op, a, b)?,
    };
    Ok(result)
}

#[cfg(all(test, feature = "chunked"))]
mod tests {
    use super::*;
    use crate::ffi::arrow_dtype::ArrowType;
    use crate::{Array, Field, FieldArray, IntegerArray, NumericArray, Table, vec64};

    fn create_field_array(name: &str, vals: &[i32]) -> FieldArray {
        let arr = Array::from_int32(IntegerArray::from_slice(&vec64![vals[0], vals[1], vals[2]]));
        let field = Field::new(name.to_string(), ArrowType::Int32, false, None);
        FieldArray::new(field, arr)
    }

    fn create_test_table(name: &str, data1: &[i32], data2: &[i32]) -> Table {
        Table::build(
            vec![
                create_field_array("col1", data1),
                create_field_array("col2", data2),
            ],
            3,
            name.to_string(),
        )
    }

    fn create_super_table(batches: Vec<Table>) -> SuperTable {
        SuperTable::from_batches(
            batches.into_iter().map(Arc::new).collect(),
            Some("test_super_table".to_string()),
        )
    }

    #[test]
    fn test_super_table_add() {
        // Create two SuperTables with 2 batches each
        let batch1_lhs = create_test_table("batch1", &[1, 2, 3], &[10, 20, 30]);
        let batch2_lhs = create_test_table("batch2", &[4, 5, 6], &[40, 50, 60]);
        let super_table_lhs = create_super_table(vec![batch1_lhs, batch2_lhs]);

        let batch1_rhs = create_test_table("batch1", &[1, 1, 1], &[5, 5, 5]);
        let batch2_rhs = create_test_table("batch2", &[2, 2, 2], &[10, 10, 10]);
        let super_table_rhs = create_super_table(vec![batch1_rhs, batch2_rhs]);

        let result = broadcast_super_table_with_operator(
            ArithmeticOperator::Add,
            super_table_lhs,
            super_table_rhs,
        )
        .unwrap();

        assert_eq!(result.n_batches(), 2);
        assert_eq!(result.n_rows(), 6);
        assert_eq!(result.n_cols(), 2);

        // Check first batch: [1,2,3] + [1,1,1] = [2,3,4]
        let batch1 = result.batch(0).unwrap();
        if let Array::NumericArray(NumericArray::Int32(arr)) = &batch1.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[2, 3, 4]);
        } else {
            panic!("Expected Int32 array");
        }

        // Check second batch col1: [4,5,6] + [2,2,2] = [6,7,8]
        let batch2 = result.batch(1).unwrap();
        if let Array::NumericArray(NumericArray::Int32(arr)) = &batch2.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[6, 7, 8]);
        } else {
            panic!("Expected Int32 array");
        }

        // Check second batch col2: [40,50,60] + [10,10,10] = [50,60,70]
        if let Array::NumericArray(NumericArray::Int32(arr)) = &batch2.cols[1].array {
            assert_eq!(arr.data.as_slice(), &[50, 60, 70]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[test]
    fn test_super_table_subtract() {
        let batch1_lhs = create_test_table("batch1", &[10, 20, 30], &[100, 200, 300]);
        let super_table_lhs = create_super_table(vec![batch1_lhs]);

        let batch1_rhs = create_test_table("batch1", &[1, 2, 3], &[10, 20, 30]);
        let super_table_rhs = create_super_table(vec![batch1_rhs]);

        let result = broadcast_super_table_with_operator(
            ArithmeticOperator::Subtract,
            super_table_lhs,
            super_table_rhs,
        )
        .unwrap();

        assert_eq!(result.n_batches(), 1);

        let batch = result.batch(0).unwrap();
        if let Array::NumericArray(NumericArray::Int32(arr)) = &batch.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[9, 18, 27]); // [10,20,30] - [1,2,3]
        } else {
            panic!("Expected Int32 array");
        }

        if let Array::NumericArray(NumericArray::Int32(arr)) = &batch.cols[1].array {
            assert_eq!(arr.data.as_slice(), &[90, 180, 270]); // [100,200,300] - [10,20,30]
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[test]
    fn test_super_table_multiply() {
        let batch1 = create_test_table("batch1", &[2, 3, 4], &[5, 6, 7]);
        let super_table_lhs = create_super_table(vec![batch1]);

        let batch2 = create_test_table("batch1", &[10, 10, 10], &[2, 2, 2]);
        let super_table_rhs = create_super_table(vec![batch2]);

        let result = broadcast_super_table_with_operator(
            ArithmeticOperator::Multiply,
            super_table_lhs,
            super_table_rhs,
        )
        .unwrap();

        let batch = result.batch(0).unwrap();
        if let Array::NumericArray(NumericArray::Int32(arr)) = &batch.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[20, 30, 40]); // [2,3,4] * [10,10,10]
        } else {
            panic!("Expected Int32 array");
        }

        if let Array::NumericArray(NumericArray::Int32(arr)) = &batch.cols[1].array {
            assert_eq!(arr.data.as_slice(), &[10, 12, 14]); // [5,6,7] * [2,2,2]
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[test]
    fn test_super_table_divide() {
        let batch1 = create_test_table("batch1", &[20, 30, 40], &[100, 200, 300]);
        let super_table_lhs = create_super_table(vec![batch1]);

        let batch2 = create_test_table("batch1", &[2, 3, 4], &[10, 20, 30]);
        let super_table_rhs = create_super_table(vec![batch2]);

        let result = broadcast_super_table_with_operator(
            ArithmeticOperator::Divide,
            super_table_lhs,
            super_table_rhs,
        )
        .unwrap();

        let batch = result.batch(0).unwrap();
        if let Array::NumericArray(NumericArray::Int32(arr)) = &batch.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[10, 10, 10]); // [20,30,40] / [2,3,4]
        } else {
            panic!("Expected Int32 array");
        }

        if let Array::NumericArray(NumericArray::Int32(arr)) = &batch.cols[1].array {
            assert_eq!(arr.data.as_slice(), &[10, 10, 10]); // [100,200,300] / [10,20,30]
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[test]
    fn test_super_table_chunk_count_mismatch() {
        // Create SuperTables with different numbers of batches
        let batch1 = create_test_table("batch1", &[1, 2, 3], &[10, 20, 30]);
        let super_table_lhs = create_super_table(vec![batch1]);

        let batch3 = create_test_table("batch1", &[1, 1, 1], &[5, 5, 5]);
        let batch4 = create_test_table("batch2", &[2, 2, 2], &[10, 10, 10]);
        let super_table_rhs = create_super_table(vec![batch3, batch4]);

        // This should return an error because lhs has 1 batch and rhs has 2 batches
        let result = broadcast_super_table_with_operator(
            ArithmeticOperator::Add,
            super_table_lhs,
            super_table_rhs,
        );

        assert!(result.is_err());
        if let Err(MinarrowError::ShapeError { message }) = result {
            assert!(message.contains("chunk count mismatch"));
        } else {
            panic!("Expected ShapeError with chunk count mismatch message");
        }
    }

    #[test]
    fn test_super_table_multiple_batches() {
        // Test with 3 batches to ensure all are processed correctly
        let batch1_lhs = create_test_table("batch1", &[1, 2, 3], &[10, 20, 30]);
        let batch2_lhs = create_test_table("batch2", &[4, 5, 6], &[40, 50, 60]);
        let batch3_lhs = create_test_table("batch3", &[7, 8, 9], &[70, 80, 90]);
        let super_table_lhs = create_super_table(vec![batch1_lhs, batch2_lhs, batch3_lhs]);

        let batch1_rhs = create_test_table("batch1", &[1, 1, 1], &[1, 1, 1]);
        let batch2_rhs = create_test_table("batch2", &[2, 2, 2], &[2, 2, 2]);
        let batch3_rhs = create_test_table("batch3", &[3, 3, 3], &[3, 3, 3]);
        let super_table_rhs = create_super_table(vec![batch1_rhs, batch2_rhs, batch3_rhs]);

        let result = broadcast_super_table_with_operator(
            ArithmeticOperator::Add,
            super_table_lhs,
            super_table_rhs,
        )
        .unwrap();

        assert_eq!(result.n_batches(), 3);
        assert_eq!(result.n_rows(), 9);

        // Check each batch
        let batch1 = result.batch(0).unwrap();
        if let Array::NumericArray(NumericArray::Int32(arr)) = &batch1.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[2, 3, 4]); // [1,2,3] + [1,1,1]
        } else {
            panic!("Expected Int32 array");
        }

        let batch2 = result.batch(1).unwrap();
        if let Array::NumericArray(NumericArray::Int32(arr)) = &batch2.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[6, 7, 8]); // [4,5,6] + [2,2,2]
        } else {
            panic!("Expected Int32 array");
        }

        let batch3 = result.batch(2).unwrap();
        if let Array::NumericArray(NumericArray::Int32(arr)) = &batch3.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[10, 11, 12]); // [7,8,9] + [3,3,3]
        } else {
            panic!("Expected Int32 array");
        }
    }
}
