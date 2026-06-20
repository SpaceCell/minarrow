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

use std::sync::Arc;

use crate::enums::error::MinarrowError;
use crate::enums::operators::ArithmeticOperator;
use crate::kernels::broadcast::broadcast_value;
use crate::kernels::routing::arithmetic::resolve_binary_arithmetic;
use crate::{ArrayV, FieldArray, Scalar, SuperArrayV, SuperTableV, Table, TableV, Value};

/// Helper function for TableView-TableView broadcasting - work directly with views
#[cfg(feature = "views")]
pub fn broadcast_tableview_to_tableview(
    op: ArithmeticOperator,
    table_view_l: &TableV,
    table_view_r: &TableV,
) -> Result<Table, MinarrowError> {
    // Ensure tables have same number of columns
    if table_view_l.cols.len() != table_view_r.cols.len() {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "TableView column count mismatch: {} vs {}",
                table_view_l.cols.len(),
                table_view_r.cols.len()
            ),
        });
    }

    let mut result_field_arrays = Vec::new();

    // No conversion needed
    for ((array_view_l, field_l), array_view_r) in table_view_l
        .cols
        .iter()
        .zip(&table_view_l.fields)
        .zip(table_view_r.cols.iter())
    {
        // Route through array broadcasting using the ArrayViews directly
        let result_array =
            resolve_binary_arithmetic(op, array_view_l.clone(), array_view_r.clone(), None)?;

        // Create new FieldArray with result
        let result_field_array = FieldArray::new(field_l.as_ref().clone(), result_array);
        result_field_arrays.push(result_field_array);
    }

    Ok(Table::new("".to_string(), Some(result_field_arrays)))
}

/// Helper function for tableview-scalar broadcasting - work directly with views
#[cfg(all(feature = "scalar_type", feature = "views"))]
pub fn broadcast_tableview_to_scalar(
    op: ArithmeticOperator,
    table_view: &TableV,
    scalar: &Scalar,
) -> Result<Table, MinarrowError> {
    // Broadcast each column view with scalar directly
    let new_cols: Result<Vec<_>, _> = table_view
        .cols
        .iter()
        .map(|col_view| {
            // Broadcast scalar with the column directly
            let scalar_value = Value::Scalar(scalar.clone());

            // Broadcast with the column view
            let result = broadcast_value(
                op,
                Value::ArrayView(Arc::new(col_view.clone())),
                scalar_value,
            )?;

            match result {
                Value::Array(arr) => Ok(Arc::unwrap_or_clone(arr)),
                _ => Err(MinarrowError::TypeError {
                    from: "tableview-scalar broadcasting",
                    to: "Array result",
                    message: Some("Expected Array result from broadcasting".to_string()),
                }),
            }
        })
        .collect();

    // Create FieldArrays from the result arrays
    let field_arrays: Vec<FieldArray> = table_view
        .fields
        .iter()
        .zip(new_cols?)
        .map(|(field, array)| FieldArray::new_arc(field.clone(), array))
        .collect();

    Ok(Table::new(table_view.name.clone(), Some(field_arrays)))
}

/// Helper function for tableview-arrayview broadcasting - work directly with views
#[cfg(feature = "views")]
pub fn broadcast_tableview_to_arrayview(
    op: ArithmeticOperator,
    table_view: &TableV,
    array_view: &ArrayV,
) -> Result<TableV, MinarrowError> {
    let new_cols: Result<Vec<_>, _> = table_view
        .cols
        .iter()
        .map(|col_view| {
            let result_array = match (
                Value::ArrayView(Arc::new(col_view.clone())),
                Value::ArrayView(Arc::new(array_view.clone())),
            ) {
                (a, b) => broadcast_value(op, a, b)?,
            };

            match result_array {
                Value::Array(result_array) => Ok(ArrayV::from(Arc::unwrap_or_clone(result_array))),
                _ => Err(MinarrowError::TypeError {
                    from: "tableview-arrayview broadcasting",
                    to: "ArrayView result",
                    message: Some("Expected Array result from broadcasting".to_string()),
                }),
            }
        })
        .collect();

    Ok(TableV {
        name: table_view.name.clone(),
        fields: table_view.fields.clone(),
        cols: new_cols?,
        offset: table_view.offset,
        len: table_view.len,
        #[cfg(feature = "select")]
        active_col_selection: None,
    })
}

/// Helper function for TableView-SuperArrayView broadcasting - promote TableView to aligned SuperTableView
#[cfg(all(feature = "chunked", feature = "views"))]
pub fn broadcast_tableview_to_superarrayview(
    op: ArithmeticOperator,
    table_view: &TableV,
    super_array_view: &SuperArrayV,
) -> Result<SuperTableV, MinarrowError> {
    // 1. Validate lengths match
    if table_view.len != super_array_view.len {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "TableView length ({}) does not match SuperArrayView length ({})",
                table_view.len, super_array_view.len
            ),
        });
    }

    // 2. Promote TableView to SuperTableView with aligned chunking
    let mut current_offset = 0;
    let mut table_slices = Vec::new();

    for array_slice in super_array_view.slices.iter() {
        let chunk_len = array_slice.len();
        let table_slice = table_view.from_self(current_offset, chunk_len);
        table_slices.push(table_slice);
        current_offset += chunk_len;
    }

    let aligned_super_table = SuperTableV {
        slices: table_slices,
        len: table_view.len,
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

#[cfg(all(test, feature = "views"))]
mod tests {
    use super::*;
    use crate::ffi::arrow_dtype::ArrowType;
    use crate::{Array, Field, FieldArray, IntegerArray, NumericArray, Table, vec64};

    #[test]
    fn test_tableview_to_tableview_add() {
        // Create two tables
        let arr1 = Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3]));
        let arr2 = Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30]));
        let table1 = Table::build(
            vec![
                FieldArray::new(
                    Field::new("col1".to_string(), ArrowType::Int32, false, None),
                    arr1,
                ),
                FieldArray::new(
                    Field::new("col2".to_string(), ArrowType::Int32, false, None),
                    arr2,
                ),
            ],
            3,
            "test".to_string(),
        );
        let table_view1 = TableV::from_table(table1, 0, 3);

        let arr3 = Array::from_int32(IntegerArray::from_slice(&vec64![5, 5, 5]));
        let arr4 = Array::from_int32(IntegerArray::from_slice(&vec64![100, 100, 100]));
        let table2 = Table::build(
            vec![
                FieldArray::new(
                    Field::new("col1".to_string(), ArrowType::Int32, false, None),
                    arr3,
                ),
                FieldArray::new(
                    Field::new("col2".to_string(), ArrowType::Int32, false, None),
                    arr4,
                ),
            ],
            3,
            "test".to_string(),
        );
        let table_view2 = TableV::from_table(table2, 0, 3);

        let result =
            broadcast_tableview_to_tableview(ArithmeticOperator::Add, &table_view1, &table_view2)
                .unwrap();

        assert_eq!(result.n_rows, 3);
        assert_eq!(result.n_cols(), 2);

        // col1: [1,2,3] + [5,5,5] = [6,7,8]
        if let Array::NumericArray(NumericArray::Int32(arr)) = &result.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[6, 7, 8]);
        } else {
            panic!("Expected Int32 array");
        }

        // col2: [10,20,30] + [100,100,100] = [110,120,130]
        if let Array::NumericArray(NumericArray::Int32(arr)) = &result.cols[1].array {
            assert_eq!(arr.data.as_slice(), &[110, 120, 130]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[test]
    fn test_tableview_to_tableview_column_mismatch() {
        // Create tables with different numbers of columns
        let arr1 = Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3]));
        let table1 = Table::build(
            vec![FieldArray::new(
                Field::new("col1".to_string(), ArrowType::Int32, false, None),
                arr1,
            )],
            3,
            "test".to_string(),
        );
        let table_view1 = TableV::from_table(table1, 0, 3);

        let arr2 = Array::from_int32(IntegerArray::from_slice(&vec64![5, 5, 5]));
        let arr3 = Array::from_int32(IntegerArray::from_slice(&vec64![10, 10, 10]));
        let table2 = Table::build(
            vec![
                FieldArray::new(
                    Field::new("col1".to_string(), ArrowType::Int32, false, None),
                    arr2,
                ),
                FieldArray::new(
                    Field::new("col2".to_string(), ArrowType::Int32, false, None),
                    arr3,
                ),
            ],
            3,
            "test".to_string(),
        );
        let table_view2 = TableV::from_table(table2, 0, 3);

        let result =
            broadcast_tableview_to_tableview(ArithmeticOperator::Add, &table_view1, &table_view2);

        assert!(result.is_err());
        if let Err(MinarrowError::ShapeError { message }) = result {
            assert!(message.contains("column count mismatch"));
        } else {
            panic!("Expected ShapeError");
        }
    }

    #[cfg(feature = "scalar_type")]
    #[test]
    fn test_tableview_to_scalar_multiply() {
        let arr1 = Array::from_int32(IntegerArray::from_slice(&vec64![2, 3, 4]));
        let arr2 = Array::from_int32(IntegerArray::from_slice(&vec64![5, 6, 7]));
        let table = Table::build(
            vec![
                FieldArray::new(
                    Field::new("col1".to_string(), ArrowType::Int32, false, None),
                    arr1,
                ),
                FieldArray::new(
                    Field::new("col2".to_string(), ArrowType::Int32, false, None),
                    arr2,
                ),
            ],
            3,
            "test".to_string(),
        );
        let table_view = TableV::from_table(table, 0, 3);

        let scalar = Scalar::Int32(10);

        let result =
            broadcast_tableview_to_scalar(ArithmeticOperator::Multiply, &table_view, &scalar)
                .unwrap();

        // col1: [2,3,4] * 10 = [20,30,40]
        if let Array::NumericArray(NumericArray::Int32(arr)) = &result.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[20, 30, 40]);
        } else {
            panic!("Expected Int32 array");
        }

        // col2: [5,6,7] * 10 = [50,60,70]
        if let Array::NumericArray(NumericArray::Int32(arr)) = &result.cols[1].array {
            assert_eq!(arr.data.as_slice(), &[50, 60, 70]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[test]
    fn test_tableview_to_arrayview_subtract() {
        let arr1 = Array::from_int32(IntegerArray::from_slice(&vec64![100, 200, 300]));
        let table = Table::build(
            vec![FieldArray::new(
                Field::new("col1".to_string(), ArrowType::Int32, false, None),
                arr1,
            )],
            3,
            "test".to_string(),
        );
        let table_view = TableV::from_table(table, 0, 3);

        let arr2 = Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30]));
        let array_view = ArrayV::from(arr2);

        let result = broadcast_tableview_to_arrayview(
            ArithmeticOperator::Subtract,
            &table_view,
            &array_view,
        )
        .unwrap();

        assert_eq!(result.len, 3);

        // [100,200,300] - [10,20,30] = [90,180,270]
        let result_table = result.to_table();
        if let Array::NumericArray(NumericArray::Int32(arr)) = &result_table.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[90, 180, 270]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[cfg(feature = "chunked")]
    #[test]
    fn test_tableview_to_superarrayview() {
        use crate::SuperArrayV;
        use std::sync::Arc;

        // Create table with 6 rows
        let arr1 = Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3, 4, 5, 6]));
        let table = Table::build(
            vec![FieldArray::new(
                Field::new("col1".to_string(), ArrowType::Int32, false, None),
                arr1,
            )],
            6,
            "test".to_string(),
        );
        let table_view = TableV::from_table(table, 0, 6);

        // Create SuperArrayView with 2 chunks of 3 elements each
        let field = Field::new("data".to_string(), ArrowType::Int32, false, None);
        let arr = Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30, 40, 50, 60]));

        let slices = vec![
            ArrayV::from(arr.clone()).slice(0, 3),
            ArrayV::from(arr.clone()).slice(3, 3),
        ];
        let super_array_view = SuperArrayV {
            slices,
            field: Arc::new(field),
            len: 6,
        };

        let result = broadcast_tableview_to_superarrayview(
            ArithmeticOperator::Multiply,
            &table_view,
            &super_array_view,
        )
        .unwrap();

        assert_eq!(result.len, 6);
        assert_eq!(result.slices.len(), 2);

        // First slice: [1,2,3] * [10,20,30] = [10,40,90]
        let slice1 = result.slices[0].to_table();
        if let Array::NumericArray(NumericArray::Int32(arr)) = &slice1.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[10, 40, 90]);
        } else {
            panic!("Expected Int32 array");
        }

        // Second slice: [4,5,6] * [40,50,60] = [160,250,360]
        let slice2 = result.slices[1].to_table();
        if let Array::NumericArray(NumericArray::Int32(arr)) = &slice2.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[160, 250, 360]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[cfg(feature = "chunked")]
    #[test]
    fn test_tableview_to_superarrayview_length_mismatch() {
        use crate::{FieldArray as FA, SuperArray, SuperArrayV};

        let arr1 = Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3, 4, 5]));
        let table = Table::build(
            vec![FieldArray::new(
                Field::new("col1".to_string(), ArrowType::Int32, false, None),
                arr1,
            )],
            5,
            "test".to_string(),
        );
        let table_view = TableV::from_table(table, 0, 5);

        // Create SuperArrayView with 6 elements (mismatch)
        let fa1 = FA::new(
            Field::new("test".to_string(), ArrowType::Int32, false, None),
            Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30])),
        );
        let fa2 = FA::new(
            Field::new("test".to_string(), ArrowType::Int32, false, None),
            Array::from_int32(IntegerArray::from_slice(&vec64![40, 50, 60])),
        );
        let super_array = SuperArray::from_chunks(vec![fa1, fa2]);
        let super_array_view = SuperArrayV::from(super_array);

        let result = broadcast_tableview_to_superarrayview(
            ArithmeticOperator::Add,
            &table_view,
            &super_array_view,
        );

        assert!(result.is_err());
        if let Err(MinarrowError::ShapeError { message }) = result {
            assert!(message.contains("does not match"));
        } else {
            panic!("Expected ShapeError");
        }
    }
}
