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

#[cfg(feature = "chunked")]
use crate::SuperTableV;
use crate::enums::error::MinarrowError;
use crate::enums::operators::ArithmeticOperator;
use crate::kernels::broadcast::broadcast_value;
use crate::structs::field_array::create_field_for_array;
use std::sync::Arc;

use crate::{ArrayV, FieldArray, Table, TableV, Value};

/// Helper function for arrayview-table broadcasting - work with view
#[cfg(feature = "views")]
pub fn broadcast_arrayview_to_table(
    op: ArithmeticOperator,
    array_view: &ArrayV,
    table: &Table,
) -> Result<Table, MinarrowError> {
    // Extract array and window once to avoid repeated struct clones
    let (array, offset, len) = array_view.as_tuple_ref();

    // Work directly with the ArrayView by broadcasting with each column
    let new_cols: Result<Vec<_>, _> = table
        .cols
        .iter()
        .map(|field_array| {
            // Create lightweight ArrayView from shared reference - only one Array clone per iteration
            let view = ArrayV::new(array.clone(), offset, len);
            let result = broadcast_value(
                op,
                Value::ArrayView(Arc::new(view)),
                Value::Array(Arc::new(field_array.array.clone())),
            )?;

            match result {
                Value::Array(arr) => Ok(Arc::unwrap_or_clone(arr)),
                _ => Err(MinarrowError::TypeError {
                    from: "arrayview-table broadcasting",
                    to: "Array result",
                    message: Some("Expected Array result from broadcasting".to_string()),
                }),
            }
        })
        .collect();

    // Create new FieldArrays from the result arrays
    let field_arrays: Vec<FieldArray> = table
        .cols
        .iter()
        .zip(new_cols?)
        .map(|(original_field_array, array)| {
            FieldArray::new_arc(original_field_array.field.clone(), array)
        })
        .collect();

    let table_out = Table::new(table.name.clone(), Some(field_arrays));
    #[cfg(feature = "table_metadata")]
    {
        let mut t = table_out;
        t.metadata = table.metadata.clone();
        return Ok(t);
    }
    #[cfg(not(feature = "table_metadata"))]
    Ok(table_out)
}

/// Helper function for arrayview-tableview broadcasting - work with views
#[cfg(feature = "views")]
pub fn broadcast_arrayview_to_tableview(
    op: ArithmeticOperator,
    array_view: &ArrayV,
    table_view: &TableV,
) -> Result<Table, MinarrowError> {
    // Extract array and window once to avoid repeated struct clones
    let (array, offset, len) = array_view.as_tuple_ref();

    let new_cols: Result<Vec<_>, _> = table_view
        .cols
        .iter()
        .zip(table_view.fields.iter())
        .map(|(col_view, field)| {
            // Create lightweight ArrayView from shared reference
            let view = ArrayV::new(array.clone(), offset, len);
            let result_array = broadcast_value(
                op,
                Value::ArrayView(Arc::new(view)),
                Value::ArrayView(Arc::new(col_view.clone())),
            )?;

            match result_array {
                Value::Array(result_array) => {
                    let result_array = Arc::unwrap_or_clone(result_array);
                    let new_field = create_field_for_array(
                        &field.name,
                        &result_array,
                        Some(array),
                        Some(field.metadata.clone()),
                    );
                    Ok(FieldArray::new(new_field, result_array))
                }
                _ => Err(MinarrowError::TypeError {
                    from: "arrayview-tableview broadcasting",
                    to: "Array result",
                    message: Some("Expected Array result from view broadcasting".to_string()),
                }),
            }
        })
        .collect();

    Ok(Table::new(table_view.name.clone(), Some(new_cols?)))
}

/// Helper function for ArrayView-SuperTableView broadcasting - work per chunk by slicing the existing ArrayView
#[cfg(feature = "views")]
pub fn broadcast_arrayview_to_supertableview(
    op: ArithmeticOperator,
    array_view: &ArrayV,
    super_table_view: &SuperTableV,
) -> Result<SuperTableV, MinarrowError> {
    // Validation: ArrayView length must match SuperTableView total length
    if array_view.len() != super_table_view.len {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "ArrayView length ({}) does not match SuperTableView length ({})",
                array_view.len(),
                super_table_view.len
            ),
        });
    }

    let mut current_offset = 0;
    let mut result_slices = Vec::new();

    for table_slice in super_table_view.slices.iter() {
        // Slice the existing ArrayView to match this table slice's size
        let aligned_array_view = array_view.slice(current_offset, table_slice.len);

        // Broadcast the aligned array view with this table slice
        let slice_result = broadcast_arrayview_to_tableview(op, &aligned_array_view, table_slice)?;
        let n_rows = slice_result.n_rows;
        result_slices.push(TableV::from_table(slice_result, 0, n_rows));
        current_offset += table_slice.len;
    }

    Ok(SuperTableV {
        slices: result_slices,
        len: super_table_view.len,
    })
}

#[cfg(all(test, feature = "views"))]
mod tests {
    use super::*;
    use crate::ffi::arrow_dtype::ArrowType;
    use crate::{Array, Field, FieldArray, IntegerArray, NumericArray, Table, vec64};

    #[test]
    fn test_arrayview_to_table_add() {
        // Create an array view: [1, 2, 3]
        let arr = Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3]));
        let array_view = ArrayV::from(arr);

        // Create a table with 2 columns
        let arr1 = Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30]));
        let arr2 = Array::from_int32(IntegerArray::from_slice(&vec64![100, 200, 300]));
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

        let result =
            broadcast_arrayview_to_table(ArithmeticOperator::Add, &array_view, &table).unwrap();

        assert_eq!(result.n_rows, 3);
        assert_eq!(result.n_cols(), 2);

        // col1: [10,20,30] + [1,2,3] = [11,22,33]
        if let Array::NumericArray(NumericArray::Int32(arr)) = &result.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[11, 22, 33]);
        } else {
            panic!("Expected Int32 array");
        }

        // col2: [100,200,300] + [1,2,3] = [101,202,303]
        if let Array::NumericArray(NumericArray::Int32(arr)) = &result.cols[1].array {
            assert_eq!(arr.data.as_slice(), &[101, 202, 303]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[test]
    fn test_arrayview_to_tableview_multiply() {
        // Create an array view: [2, 3, 4]
        let arr = Array::from_int32(IntegerArray::from_slice(&vec64![2, 3, 4]));
        let array_view = ArrayV::from(arr);

        // Create a table and table view
        let arr1 = Array::from_int32(IntegerArray::from_slice(&vec64![10, 10, 10]));
        let table = Table::build(
            vec![FieldArray::new(
                Field::new("col1".to_string(), ArrowType::Int32, false, None),
                arr1,
            )],
            3,
            "test".to_string(),
        );
        let table_view = TableV::from_table(table, 0, 3);

        let result = broadcast_arrayview_to_tableview(
            ArithmeticOperator::Multiply,
            &array_view,
            &table_view,
        )
        .unwrap();

        assert_eq!(result.n_rows, 3);

        // [10,10,10] * [2,3,4] = [20,30,40]
        if let Array::NumericArray(NumericArray::Int32(arr)) = &result.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[20, 30, 40]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[test]
    fn test_arrayview_to_tableview_subtract() {
        // Create an array view: [5, 5, 5]
        let arr = Array::from_int32(IntegerArray::from_slice(&vec64![5, 5, 5]));
        let array_view = ArrayV::from(arr);

        // Create a table view
        let arr1 = Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30]));
        let arr2 = Array::from_int32(IntegerArray::from_slice(&vec64![100, 200, 300]));
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

        let result = broadcast_arrayview_to_tableview(
            ArithmeticOperator::Subtract,
            &array_view,
            &table_view,
        )
        .unwrap();

        // col1: [5,5,5] - [10,20,30] = [-5,-15,-25]
        if let Array::NumericArray(NumericArray::Int32(arr)) = &result.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[-5, -15, -25]);
        } else {
            panic!("Expected Int32 array");
        }

        // col2: [5,5,5] - [100,200,300] = [-95,-195,-295]
        if let Array::NumericArray(NumericArray::Int32(arr)) = &result.cols[1].array {
            assert_eq!(arr.data.as_slice(), &[-95, -195, -295]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[cfg(feature = "chunked")]
    #[test]
    fn test_arrayview_to_supertableview() {
        use crate::SuperTableV;

        // Create an array view: [1, 2, 3, 4, 5, 6]
        let arr = Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3, 4, 5, 6]));
        let array_view = ArrayV::from(arr);

        // Create SuperTableView with 2 slices
        let table1_arr = Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30]));
        let table1 = Table::build(
            vec![FieldArray::new(
                Field::new("col1".to_string(), ArrowType::Int32, false, None),
                table1_arr,
            )],
            3,
            "test".to_string(),
        );
        let table_view1 = TableV::from_table(table1, 0, 3);

        let table2_arr = Array::from_int32(IntegerArray::from_slice(&vec64![40, 50, 60]));
        let table2 = Table::build(
            vec![FieldArray::new(
                Field::new("col1".to_string(), ArrowType::Int32, false, None),
                table2_arr,
            )],
            3,
            "test".to_string(),
        );
        let table_view2 = TableV::from_table(table2, 0, 3);

        let super_table_view = SuperTableV {
            slices: vec![table_view1, table_view2],
            len: 6,
        };

        let result = broadcast_arrayview_to_supertableview(
            ArithmeticOperator::Add,
            &array_view,
            &super_table_view,
        )
        .unwrap();

        assert_eq!(result.len, 6);
        assert_eq!(result.slices.len(), 2);

        // First slice: [10,20,30] + [1,2,3] = [11,22,33]
        let slice1 = result.slices[0].to_table();
        if let Array::NumericArray(NumericArray::Int32(arr)) = &slice1.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[11, 22, 33]);
        } else {
            panic!("Expected Int32 array");
        }

        // Second slice: [40,50,60] + [4,5,6] = [44,55,66]
        let slice2 = result.slices[1].to_table();
        if let Array::NumericArray(NumericArray::Int32(arr)) = &slice2.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[44, 55, 66]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[cfg(feature = "chunked")]
    #[test]
    fn test_arrayview_to_supertableview_length_mismatch() {
        use crate::SuperTableV;

        // Create an array view with 5 elements
        let arr = Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3, 4, 5]));
        let array_view = ArrayV::from(arr);

        // Create a SuperTableView with 6 total rows (mismatch)
        let arr1 = Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30]));
        let table1 = Table::build(
            vec![FieldArray::new(
                Field::new("col1".to_string(), ArrowType::Int32, false, None),
                arr1,
            )],
            3,
            "test".to_string(),
        );
        let table_view1 = TableV::from_table(table1, 0, 3);

        let arr2 = Array::from_int32(IntegerArray::from_slice(&vec64![40, 50, 60]));
        let table2 = Table::build(
            vec![FieldArray::new(
                Field::new("col1".to_string(), ArrowType::Int32, false, None),
                arr2,
            )],
            3,
            "test".to_string(),
        );
        let table_view2 = TableV::from_table(table2, 0, 3);

        let super_table_view = SuperTableV {
            slices: vec![table_view1, table_view2],
            len: 6,
        };

        let result = broadcast_arrayview_to_supertableview(
            ArithmeticOperator::Add,
            &array_view,
            &super_table_view,
        );

        assert!(result.is_err());
        if let Err(MinarrowError::ShapeError { message }) = result {
            assert!(message.contains("does not match"));
        } else {
            panic!("Expected ShapeError");
        }
    }
}
