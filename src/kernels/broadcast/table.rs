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

use std::sync::Arc;

#[cfg(feature = "scalar_type")]
use crate::Scalar;
use crate::enums::error::{KernelError, MinarrowError};
use crate::enums::operators::ArithmeticOperator;
use crate::kernels::broadcast::array::broadcast_array_add;
use crate::kernels::broadcast::broadcast_value;
use crate::kernels::broadcast::table_view::broadcast_tableview_to_arrayview;
use crate::kernels::routing::arithmetic::resolve_binary_arithmetic;
use crate::structs::field_array::create_field_for_array;
use crate::{Array, ArrayV, Bitmask, Field, FieldArray, Table, TableV, Value};
#[cfg(feature = "chunked")]
use crate::{SuperArray, SuperArrayV, SuperTable, SuperTableV};

/// General table broadcasting function that supports all arithmetic operators
pub fn broadcast_table_with_operator(
    op: ArithmeticOperator,
    table_l: impl Into<TableV>,
    table_r: impl Into<TableV>,
) -> Result<Table, MinarrowError> {
    let table_l: TableV = table_l.into();
    let table_r: TableV = table_r.into();

    // Ensure tables have same number of columns
    if table_l.cols.len() != table_r.cols.len() {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "Table column count mismatch: {} vs {}",
                table_l.cols.len(),
                table_r.cols.len()
            ),
        });
    }

    let mut result_field_arrays = Vec::new();

    for (i, (col_l, col_r)) in table_l.cols.iter().zip(table_r.cols.iter()).enumerate() {
        // Route through array broadcasting
        let result_array = resolve_binary_arithmetic(op, col_l.clone(), col_r.clone(), None)?;

        // Create new FieldArray with the field schema from the left table
        let result_field_array = FieldArray::new(table_l.fields[i].as_ref().clone(), result_array);
        result_field_arrays.push(result_field_array);
    }

    Ok(Table::new(table_l.name.clone(), Some(result_field_arrays)))
}

/// Broadcasts addition over table columns element-wise
///
/// Both tables must have the same number of columns and rows.
/// Addition is applied column-wise between corresponding columns.
pub fn broadcast_table_add(
    lhs: impl Into<TableV>,
    rhs: impl Into<TableV>,
    null_mask: Option<Arc<Bitmask>>,
) -> Result<Table, KernelError> {
    let lhs_table: TableV = lhs.into();
    let rhs_table: TableV = rhs.into();

    // Check shape compatibility
    if lhs_table.cols.len() != rhs_table.cols.len() {
        return Err(KernelError::BroadcastingError(format!(
            "Table column count mismatch: LHS {} cols, RHS {} cols",
            lhs_table.cols.len(),
            rhs_table.cols.len()
        )));
    }

    if lhs_table.len != rhs_table.len {
        return Err(KernelError::BroadcastingError(format!(
            "Table row count mismatch: LHS {} rows, RHS {} rows",
            lhs_table.len, rhs_table.len
        )));
    }

    // Apply addition column-wise
    let mut result_cols = Vec::with_capacity(lhs_table.cols.len());

    for (i, (lhs_col, rhs_col)) in lhs_table.cols.iter().zip(rhs_table.cols.iter()).enumerate() {
        let result_array =
            broadcast_array_add(lhs_col.clone(), rhs_col.clone(), null_mask.as_deref()).map_err(
                |e| KernelError::BroadcastingError(format!("Column {} addition failed: {}", i, e)),
            )?;

        // Create FieldArray with name from left table
        let field_name = if i < lhs_table.fields.len() {
            lhs_table.fields[i].name.clone()
        } else {
            format!("col_{}", i)
        };

        let field_dtype = if i < lhs_table.fields.len() {
            lhs_table.fields[i].dtype.clone()
        } else {
            result_array.arrow_type()
        };

        let field = Field::new(
            field_name,
            field_dtype,
            result_array.null_mask().is_some(), // nullable based on result array
            None,                               // metadata
        );
        let field_array = FieldArray::new(field, result_array);

        result_cols.push(field_array);
    }

    // Create result table with same name as left table
    Ok(Table::new(lhs_table.name.clone(), Some(result_cols)))
}

/// Broadcasts addition over SuperTable chunks (batched tables)
///
/// Both SuperTables must have the same number of chunks and compatible shapes.
/// Addition is applied chunk-wise between corresponding table chunks.
#[cfg(feature = "chunked")]
pub fn broadcast_super_table_add(
    lhs: impl Into<SuperTableV>,
    rhs: impl Into<SuperTableV>,
    null_mask: Option<Arc<Bitmask>>,
) -> Result<SuperTable, KernelError> {
    let lhs_table: SuperTableV = lhs.into();
    let rhs_table: SuperTableV = rhs.into();

    // Check chunk count compatibility
    if lhs_table.slices.len() != rhs_table.slices.len() {
        return Err(KernelError::BroadcastingError(format!(
            "SuperTable chunk count mismatch: LHS {} chunks, RHS {} chunks",
            lhs_table.slices.len(),
            rhs_table.slices.len()
        )));
    }

    // Apply addition chunk-wise
    let mut result_tables = Vec::with_capacity(lhs_table.slices.len());

    for (i, (lhs_chunk, rhs_chunk)) in lhs_table
        .slices
        .iter()
        .zip(rhs_table.slices.iter())
        .enumerate()
    {
        let result_table =
            broadcast_table_add(lhs_chunk.clone(), rhs_chunk.clone(), null_mask.clone()).map_err(
                |e| KernelError::BroadcastingError(format!("Chunk {} addition failed: {}", i, e)),
            )?;

        result_tables.push(Arc::new(result_table));
    }

    // Create result SuperTable - use name from first slice if available
    let name = if !lhs_table.slices.is_empty() && !lhs_table.slices[0].name.is_empty() {
        lhs_table.slices[0].name.clone()
    } else {
        "SuperTable".to_string()
    };
    Ok(SuperTable::from_batches(result_tables, Some(name)))
}

/// Helper function for table-array broadcasting - apply table columns to array
pub fn broadcast_table_to_array(
    op: ArithmeticOperator,
    table: &Table,
    array: &Array,
) -> Result<Table, MinarrowError> {
    let new_cols: Result<Vec<_>, _> = table
        .cols
        .iter()
        .map(|field_array| {
            let col_array = &field_array.array;
            let result_array = match (
                Value::Array(Arc::new(col_array.clone())),
                Value::Array(Arc::new(array.clone())),
            ) {
                (a, b) => broadcast_value(op, a, b)?,
            };

            match result_array {
                Value::Array(result_array) => {
                    let result_array = Arc::unwrap_or_clone(result_array);
                    // Preserve original field metadata but update type if needed
                    let new_field = create_field_for_array(
                        &field_array.field.name,
                        &result_array,
                        Some(&array),
                        Some(field_array.field.metadata.clone()),
                    );
                    Ok(FieldArray::new(new_field, result_array))
                }
                _ => Err(MinarrowError::TypeError {
                    from: "table-array broadcasting",
                    to: "Array result",
                    message: Some("Expected Array result from broadcasting".to_string()),
                }),
            }
        })
        .collect();

    let table_out = Table::new(table.name.clone(), Some(new_cols?));
    #[cfg(feature = "table_metadata")]
    {
        let mut t = table_out;
        t.metadata = table.metadata.clone();
        return Ok(t);
    }
    #[cfg(not(feature = "table_metadata"))]
    Ok(table_out)
}

/// Helper function for table-scalar broadcasting - apply table columns to scalar
#[cfg(feature = "scalar_type")]
pub fn broadcast_table_to_scalar(
    op: ArithmeticOperator,
    table: &Table,
    scalar: &Scalar,
) -> Result<Table, MinarrowError> {
    let new_cols: Result<Vec<_>, _> = table
        .cols
        .iter()
        .map(|field_array| {
            let col_array = &field_array.array;
            let result_array = match (
                Value::Array(Arc::new(col_array.clone())),
                Value::Scalar(scalar.clone()),
            ) {
                (a, b) => broadcast_value(op, a, b)?,
            };

            match result_array {
                Value::Array(result_array) => {
                    let result_array = Arc::unwrap_or_clone(result_array);
                    // Preserve original field metadata but update type if needed
                    let new_field = create_field_for_array(
                        &field_array.field.name,
                        &result_array,
                        Some(&col_array),
                        Some(field_array.field.metadata.clone()),
                    );
                    Ok(FieldArray::new(new_field, result_array))
                }
                _ => Err(MinarrowError::TypeError {
                    from: "table-scalar broadcasting",
                    to: "Array result",
                    message: Some("Expected Array result from broadcasting".to_string()),
                }),
            }
        })
        .collect();

    let table_out = Table::new(table.name.clone(), Some(new_cols?));
    #[cfg(feature = "table_metadata")]
    {
        let mut t = table_out;
        t.metadata = table.metadata.clone();
        return Ok(t);
    }
    #[cfg(not(feature = "table_metadata"))]
    Ok(table_out)
}

/// Helper function for table-arrayview broadcasting - work directly with view without conversion
#[cfg(feature = "views")]
pub fn broadcast_table_to_arrayview(
    op: ArithmeticOperator,
    table: &Table,
    array_view: &ArrayV,
) -> Result<Table, MinarrowError> {
    // Work directly with the view's underlying array and window bounds
    let new_cols: Result<Vec<_>, _> = table
        .cols
        .iter()
        .map(|field_array| {
            let col_array = &field_array.array;
            // Create a view of the column array that matches the input view's window
            let col_view = ArrayV::new(col_array.clone(), array_view.offset, array_view.len());
            let result_array = match (
                Value::ArrayView(Arc::new(col_view)),
                Value::ArrayView(Arc::new(array_view.clone())),
            ) {
                (a, b) => broadcast_value(op, a, b)?,
            };

            match result_array {
                Value::Array(result_array) => {
                    let result_array = Arc::unwrap_or_clone(result_array);
                    let new_field = create_field_for_array(
                        &field_array.field.name,
                        &result_array,
                        Some(&array_view.array),
                        Some(field_array.field.metadata.clone()),
                    );
                    Ok(FieldArray::new(new_field, result_array))
                }
                _ => Err(MinarrowError::TypeError {
                    from: "table-arrayview broadcasting",
                    to: "Array result",
                    message: Some("Expected Array result from view broadcasting".to_string()),
                }),
            }
        })
        .collect();

    let table_out = Table::new(table.name.clone(), Some(new_cols?));
    #[cfg(feature = "table_metadata")]
    {
        let mut t = table_out;
        t.metadata = table.metadata.clone();
        return Ok(t);
    }
    #[cfg(not(feature = "table_metadata"))]
    Ok(table_out)
}

/// Helper function for Table-SuperArrayView broadcasting - promote Table to aligned SuperTableView
#[cfg(all(feature = "chunked", feature = "views"))]
pub fn broadcast_table_to_superarrayview(
    op: ArithmeticOperator,
    table: &Table,
    super_array_view: &SuperArrayV,
) -> Result<SuperTableV, MinarrowError> {
    // 1. Validate lengths match
    if table.n_rows != super_array_view.len {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "Table rows ({}) does not match SuperArrayView length ({})",
                table.n_rows, super_array_view.len
            ),
        });
    }

    // 2. Promote Table to SuperTableView with aligned chunking
    let mut current_offset = 0;
    let mut table_slices = Vec::new();

    for array_slice in super_array_view.slices.iter() {
        let chunk_len = array_slice.len();
        let table_slice = TableV::from_table(table.clone(), current_offset, chunk_len);
        table_slices.push(table_slice);
        current_offset += chunk_len;
    }

    let aligned_super_table = SuperTableV {
        slices: table_slices,
        len: table.n_rows,
    };

    // 3. Broadcast per chunk using indexed loops
    let mut result_slices = Vec::new();
    for i in 0..aligned_super_table.slices.len() {
        let table_slice = &aligned_super_table.slices[i];
        let array_slice = &super_array_view.slices[i];
        let slice_result = broadcast_tableview_to_arrayview(op, table_slice, array_slice)?;
        result_slices.push(slice_result);
    }

    Ok(SuperTableV {
        slices: result_slices,
        len: super_array_view.len,
    })
}

/// Helper function for Table-SuperArray broadcasting - broadcast table against each chunk
#[cfg(feature = "chunked")]
pub fn broadcast_table_to_superarray(
    op: ArithmeticOperator,
    table: &Table,
    super_array: &SuperArray,
) -> Result<SuperArray, MinarrowError> {
    let new_chunks: Result<Vec<_>, _> = super_array
        .chunks()
        .iter()
        .map(|chunk| {
            let chunk_array = chunk;
            let result_table = broadcast_table_to_array(op, table, chunk_array)?;
            // Convert result table back to a FieldArray chunk with matching structure
            if result_table.cols.len() == 1 {
                Ok(result_table.cols[0].clone())
            } else {
                Err(MinarrowError::ShapeError {
                    message: "Table-SuperArray broadcasting should result in single column"
                        .to_string(),
                })
            }
        })
        .collect();

    Ok(SuperArray::from_chunks(new_chunks?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::selection::ColumnSelection;
    use crate::{Array, FieldArray, IntegerArray, vec64};

    fn create_test_table(name: &str, data1: &[i32], data2: &[i32]) -> Table {
        let col1 = FieldArray::from_arr(
            "col1",
            Array::from_int32(IntegerArray::from_slice(&vec64![
                data1[0], data1[1], data1[2]
            ])),
        );
        let col2 = FieldArray::from_arr(
            "col2",
            Array::from_int32(IntegerArray::from_slice(&vec64![
                data2[0], data2[1], data2[2]
            ])),
        );

        Table::new(name.to_string(), Some(vec![col1, col2]))
    }

    #[test]
    fn test_table_plus_table() {
        let table1 = create_test_table("table1", &[1, 2, 3], &[10, 20, 30]);
        let table2 = create_test_table("table2", &[4, 5, 6], &[40, 50, 60]);

        let result = broadcast_table_add(table1, table2, None).unwrap();

        assert_eq!(result.n_cols(), 2);
        assert_eq!(result.n_rows(), 3);
        assert_eq!(result.name, "table1"); // Takes name from left table

        // Check first column: [1,2,3] + [4,5,6] = [5,7,9]
        if let Some(col1) = result.col_ix(0) {
            if let crate::Array::NumericArray(crate::NumericArray::Int32(arr)) = &col1.array {
                assert_eq!(arr.data.as_slice(), &[5, 7, 9]);
            } else {
                panic!("Expected Int32 array in first column");
            }
        } else {
            panic!("Could not get first column");
        }

        // Check second column: [10,20,30] + [40,50,60] = [50,70,90]
        if let Some(col2) = result.col_ix(1) {
            if let crate::Array::NumericArray(crate::NumericArray::Int32(arr)) = &col2.array {
                assert_eq!(arr.data.as_slice(), &[50, 70, 90]);
            } else {
                panic!("Expected Int32 array in second column");
            }
        } else {
            panic!("Could not get second column");
        }
    }

    #[test]
    #[should_panic(expected = "column count mismatch")]
    fn test_mismatched_column_count() {
        let col1 = FieldArray::from_arr(
            "col1",
            Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3])),
        );
        let table1 = Table::new("table1".to_string(), Some(vec![col1])); // 1 column

        let table2 = create_test_table("table2", &[4, 5, 6], &[40, 50, 60]); // 2 columns

        let _ = broadcast_table_add(table1, table2, None).unwrap();
    }

    #[test]
    #[should_panic(expected = "row count mismatch")]
    fn test_mismatched_row_count() {
        let col1 = FieldArray::from_arr(
            "col1",
            Array::from_int32(IntegerArray::from_slice(&vec64![1, 2])),
        );
        let col2 = FieldArray::from_arr(
            "col2",
            Array::from_int32(IntegerArray::from_slice(&vec64![10, 20])),
        );
        let table1 = Table::new("table1".to_string(), Some(vec![col1, col2])); // 2 rows

        let table2 = create_test_table("table2", &[4, 5, 6], &[40, 50, 60]); // 3 rows

        let _ = broadcast_table_add(table1, table2, None).unwrap();
    }

    #[test]
    fn test_broadcast_table_with_operator_multiply() {
        let table1 = create_test_table("table1", &[2, 3, 4], &[5, 6, 7]);
        let table2 = create_test_table("table2", &[10, 10, 10], &[2, 2, 2]);

        let result =
            broadcast_table_with_operator(ArithmeticOperator::Multiply, table1, table2).unwrap();

        // col1: [2,3,4] * [10,10,10] = [20,30,40]
        if let crate::Array::NumericArray(crate::NumericArray::Int32(arr)) = &result.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[20, 30, 40]);
        } else {
            panic!("Expected Int32 array");
        }

        // col2: [5,6,7] * [2,2,2] = [10,12,14]
        if let crate::Array::NumericArray(crate::NumericArray::Int32(arr)) = &result.cols[1].array {
            assert_eq!(arr.data.as_slice(), &[10, 12, 14]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[test]
    fn test_broadcast_table_to_array() {
        let table = create_test_table("table1", &[10, 20, 30], &[100, 200, 300]);
        let arr = Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3]));

        let result = broadcast_table_to_array(ArithmeticOperator::Add, &table, &arr).unwrap();

        // col1: [10,20,30] + [1,2,3] = [11,22,33]
        if let crate::Array::NumericArray(crate::NumericArray::Int32(arr)) = &result.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[11, 22, 33]);
        } else {
            panic!("Expected Int32 array");
        }

        // col2: [100,200,300] + [1,2,3] = [101,202,303]
        if let crate::Array::NumericArray(crate::NumericArray::Int32(arr)) = &result.cols[1].array {
            assert_eq!(arr.data.as_slice(), &[101, 202, 303]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[cfg(feature = "scalar_type")]
    #[test]
    fn test_broadcast_table_to_scalar() {
        let table = create_test_table("table1", &[10, 20, 30], &[100, 200, 300]);
        let scalar = Scalar::Int32(5);

        let result =
            broadcast_table_to_scalar(ArithmeticOperator::Multiply, &table, &scalar).unwrap();

        // col1: [10,20,30] * 5 = [50,100,150]
        if let crate::Array::NumericArray(crate::NumericArray::Int32(arr)) = &result.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[50, 100, 150]);
        } else {
            panic!("Expected Int32 array");
        }

        // col2: [100,200,300] * 5 = [500,1000,1500]
        if let crate::Array::NumericArray(crate::NumericArray::Int32(arr)) = &result.cols[1].array {
            assert_eq!(arr.data.as_slice(), &[500, 1000, 1500]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[cfg(feature = "views")]
    #[test]
    fn test_broadcast_table_to_arrayview() {
        let table = create_test_table("table1", &[10, 20, 30], &[100, 200, 300]);
        let arr = Array::from_int32(IntegerArray::from_slice(&vec64![2, 3, 4]));
        let array_view = ArrayV::from(arr);

        let result =
            broadcast_table_to_arrayview(ArithmeticOperator::Subtract, &table, &array_view)
                .unwrap();

        // col1: [10,20,30] - [2,3,4] = [8,17,26]
        if let crate::Array::NumericArray(crate::NumericArray::Int32(arr)) = &result.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[8, 17, 26]);
        } else {
            panic!("Expected Int32 array");
        }

        // col2: [100,200,300] - [2,3,4] = [98,197,296]
        if let crate::Array::NumericArray(crate::NumericArray::Int32(arr)) = &result.cols[1].array {
            assert_eq!(arr.data.as_slice(), &[98, 197, 296]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[cfg(feature = "chunked")]
    #[test]
    fn test_broadcast_super_table_add() {
        let table1 = create_test_table("table1", &[1, 2, 3], &[10, 20, 30]);
        let table2 = create_test_table("table2", &[4, 5, 6], &[40, 50, 60]);
        let table3 = create_test_table("table3", &[7, 8, 9], &[70, 80, 90]);
        let table4 = create_test_table("table4", &[1, 1, 1], &[2, 2, 2]);

        let super_table1 = SuperTableV {
            slices: vec![
                TableV::from_table(table1, 0, 3),
                TableV::from_table(table2, 0, 3),
            ],
            len: 6,
        };

        let super_table2 = SuperTableV {
            slices: vec![
                TableV::from_table(table3, 0, 3),
                TableV::from_table(table4, 0, 3),
            ],
            len: 6,
        };

        let result = broadcast_super_table_add(super_table1, super_table2, None).unwrap();

        assert_eq!(result.batches.len(), 2);

        // First batch: table1 + table3
        // col1: [1,2,3] + [7,8,9] = [8,10,12]
        if let crate::Array::NumericArray(crate::NumericArray::Int32(arr)) =
            &result.batches[0].cols[0].array
        {
            assert_eq!(arr.data.as_slice(), &[8, 10, 12]);
        } else {
            panic!("Expected Int32 array");
        }

        // col2: [10,20,30] + [70,80,90] = [80,100,120]
        if let crate::Array::NumericArray(crate::NumericArray::Int32(arr)) =
            &result.batches[0].cols[1].array
        {
            assert_eq!(arr.data.as_slice(), &[80, 100, 120]);
        } else {
            panic!("Expected Int32 array");
        }

        // Second batch: table2 + table4
        // col1: [4,5,6] + [1,1,1] = [5,6,7]
        if let crate::Array::NumericArray(crate::NumericArray::Int32(arr)) =
            &result.batches[1].cols[0].array
        {
            assert_eq!(arr.data.as_slice(), &[5, 6, 7]);
        } else {
            panic!("Expected Int32 array");
        }

        // col2: [40,50,60] + [2,2,2] = [42,52,62]
        if let crate::Array::NumericArray(crate::NumericArray::Int32(arr)) =
            &result.batches[1].cols[1].array
        {
            assert_eq!(arr.data.as_slice(), &[42, 52, 62]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[cfg(feature = "chunked")]
    #[test]
    fn test_broadcast_table_to_superarray() {
        use crate::ffi::arrow_dtype::ArrowType;

        // Table with 3 rows to match each chunk size
        let table = Table::build(
            vec![FieldArray::new(
                Field::new("col1".to_string(), ArrowType::Int32, false, None),
                Array::from_int32(IntegerArray::from_slice(&vec64![2, 3, 4])),
            )],
            3,
            "test".to_string(),
        );

        let field = Field::new("data".to_string(), ArrowType::Int32, false, None);
        let chunks = vec![
            FieldArray::new(
                field.clone(),
                Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30])),
            ),
            FieldArray::new(
                field.clone(),
                Array::from_int32(IntegerArray::from_slice(&vec64![40, 50, 60])),
            ),
        ];
        let super_array = SuperArray::from_chunks(chunks);

        let result =
            broadcast_table_to_superarray(ArithmeticOperator::Add, &table, &super_array).unwrap();

        assert_eq!(result.chunks().len(), 2);

        // Both chunks: [2,3,4] + [10,20,30] = [12,23,34] and [2,3,4] + [40,50,60] = [42,53,64]
        if let crate::Array::NumericArray(crate::NumericArray::Int32(arr)) = &result.chunks()[0] {
            assert_eq!(arr.data.as_slice(), &[12, 23, 34]);
        } else {
            panic!("Expected Int32 array");
        }

        if let crate::Array::NumericArray(crate::NumericArray::Int32(arr)) = &result.chunks()[1] {
            assert_eq!(arr.data.as_slice(), &[42, 53, 64]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[cfg(all(feature = "chunked", feature = "views"))]
    #[test]
    fn test_broadcast_table_to_superarrayview() {
        use crate::ffi::arrow_dtype::ArrowType;

        let table = Table::build(
            vec![FieldArray::new(
                Field::new("col1".to_string(), ArrowType::Int32, false, None),
                Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3, 4, 5, 6])),
            )],
            6,
            "test".to_string(),
        );

        let arr = Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30, 40, 50, 60]));
        let field = Field::new("data".to_string(), ArrowType::Int32, false, None);

        let slices = vec![
            ArrayV::from(arr.clone()).slice(0, 3),
            ArrayV::from(arr.clone()).slice(3, 3),
        ];
        let super_array_view = SuperArrayV {
            slices,
            field: Arc::new(field),
            len: 6,
        };

        let result = broadcast_table_to_superarrayview(
            ArithmeticOperator::Multiply,
            &table,
            &super_array_view,
        )
        .unwrap();

        assert_eq!(result.slices.len(), 2);
        assert_eq!(result.len, 6);

        // First slice: [1,2,3] * [10,20,30] = [10,40,90]
        let slice1 = result.slices[0].to_table();
        if let crate::Array::NumericArray(crate::NumericArray::Int32(arr)) = &slice1.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[10, 40, 90]);
        } else {
            panic!("Expected Int32 array");
        }

        // Second slice: [4,5,6] * [40,50,60] = [160,250,360]
        let slice2 = result.slices[1].to_table();
        if let crate::Array::NumericArray(crate::NumericArray::Int32(arr)) = &slice2.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[160, 250, 360]);
        } else {
            panic!("Expected Int32 array");
        }
    }
}
