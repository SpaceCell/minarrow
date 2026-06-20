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
use crate::kernels::broadcast::array_view::broadcast_arrayview_to_tableview;
use crate::kernels::broadcast::broadcast_value;
use crate::kernels::broadcast::table_view::{
    broadcast_tableview_to_arrayview, broadcast_tableview_to_tableview,
};
use crate::{Array, ArrayV, Scalar, SuperArrayV, SuperTableV, Table, TableV, Value};

/// Helper function for supertableview-scalar broadcasting - convert to table, broadcast, return as table
#[cfg(all(feature = "scalar_type", feature = "chunked", feature = "views"))]
pub fn broadcast_supertableview_to_scalar(
    op: ArithmeticOperator,
    super_table_view: &SuperTableV,
    scalar: &Scalar,
) -> Result<SuperTableV, MinarrowError> {
    // Recursively broadcast each table slice to scalar, keeping as SuperTableView
    let result_slices: Result<Vec<_>, _> = super_table_view
        .slices
        .iter()
        .map(|table_slice| {
            let result = broadcast_value(
                op,
                Value::TableView(Arc::new(table_slice.clone())),
                Value::Scalar(scalar.clone()),
            )?;
            match result {
                Value::Table(table) => {
                    let table = Arc::unwrap_or_clone(table);
                    let n_rows = table.n_rows;
                    Ok(TableV::from_table(table, 0, n_rows))
                }
                _ => Err(MinarrowError::TypeError {
                    from: "supertableview-scalar broadcasting",
                    to: "TableView result",
                    message: Some("Expected Table result from broadcasting".to_string()),
                }),
            }
        })
        .collect();

    Ok(SuperTableV {
        slices: result_slices?,
        len: super_table_view.len,
    })
}

/// Helper function for SuperTableView-ArrayView broadcasting - work per chunk by slicing the existing ArrayView
#[cfg(feature = "views")]
pub fn broadcast_supertableview_to_arrayview(
    op: ArithmeticOperator,
    super_table_view: &SuperTableV,
    array_view: &ArrayV,
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
        // Create an aligned view from the existing ArrayView's underlying array
        // Account for the ArrayView's existing offset
        let aligned_array_view = ArrayV::new(
            array_view.array.clone(),
            array_view.offset + current_offset,
            table_slice.len,
        );

        // Broadcast this table slice with the aligned array view
        let slice_result = broadcast_tableview_to_arrayview(op, table_slice, &aligned_array_view)?;
        result_slices.push(slice_result);
        current_offset += table_slice.len;
    }

    Ok(SuperTableV {
        slices: result_slices,
        len: super_table_view.len,
    })
}

/// Helper function for SuperArrayView-Table broadcasting - promote Table to aligned SuperTableView
#[cfg(all(feature = "chunked", feature = "views"))]
pub fn broadcast_superarrayview_to_table(
    op: ArithmeticOperator,
    super_array_view: &SuperArrayV,
    table: &Table,
) -> Result<SuperTableV, MinarrowError> {
    // 1. Validate lengths match
    if super_array_view.len != table.n_rows {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "SuperArrayView length ({}) does not match Table rows ({})",
                super_array_view.len, table.n_rows
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
    for i in 0..super_array_view.slices.len() {
        let array_slice = &super_array_view.slices[i];
        let table_slice = &aligned_super_table.slices[i];
        let slice_result_table = broadcast_arrayview_to_tableview(op, array_slice, table_slice)?;
        let n_rows = slice_result_table.n_rows;
        result_slices.push(TableV::from_table(slice_result_table, 0, n_rows));
    }

    Ok(SuperTableV {
        slices: result_slices,
        len: super_array_view.len,
    })
}

/// Helper function for SuperTableView-Array broadcasting - create aligned array views for each table slice
#[cfg(all(feature = "chunked", feature = "views"))]
pub fn broadcast_supertableview_to_array(
    op: ArithmeticOperator,
    super_table_view: &SuperTableV,
    array: &Array,
) -> Result<SuperTableV, MinarrowError> {
    let mut current_offset = 0;
    let mut result_slices = Vec::new();

    for table_slice in super_table_view.slices.iter() {
        // Create an array view that matches this table slice's size
        let array_view = ArrayV::new(array.clone(), current_offset, table_slice.len);

        // Broadcast this table slice with the aligned array view
        let slice_result = broadcast_tableview_to_arrayview(op, table_slice, &array_view)?;
        result_slices.push(slice_result);
        current_offset += table_slice.len;
    }

    Ok(SuperTableV {
        slices: result_slices,
        len: super_table_view.len,
    })
}

/// Helper function for Table-SuperTableView broadcasting - promote Table to SuperTableView with aligned chunking
#[cfg(all(feature = "chunked", feature = "views"))]
pub fn broadcast_table_to_supertableview(
    op: ArithmeticOperator,
    table: &Table,
    super_table_view: &SuperTableV,
) -> Result<SuperTableV, MinarrowError> {
    // Validate lengths match
    if table.n_rows != super_table_view.len {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "Table rows ({}) does not match SuperTableView length ({})",
                table.n_rows, super_table_view.len
            ),
        });
    }

    let mut current_offset = 0;
    let mut result_slices = Vec::new();

    for table_slice in super_table_view.slices.iter() {
        let table_view = TableV::from_table(table.clone(), current_offset, table_slice.len);
        let result = broadcast_tableview_to_tableview(op, &table_view, table_slice)?;
        // Convert the resulting Table back to a TableView
        result_slices.push(TableV::from_table(result, 0, table_slice.len));
        current_offset += table_slice.len;
    }

    Ok(SuperTableV {
        slices: result_slices,
        len: super_table_view.len,
    })
}

/// Helper function for SuperTableView-Table broadcasting - promote Table to SuperTableView with aligned chunking
#[cfg(all(feature = "chunked", feature = "views"))]
pub fn broadcast_supertableview_to_table(
    op: ArithmeticOperator,
    super_table_view: &SuperTableV,
    table: &Table,
) -> Result<SuperTableV, MinarrowError> {
    // Validate lengths match
    if super_table_view.len != table.n_rows {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "SuperTableView length ({}) does not match Table rows ({})",
                super_table_view.len, table.n_rows
            ),
        });
    }

    let mut current_offset = 0;
    let mut result_slices = Vec::new();

    for table_slice in super_table_view.slices.iter() {
        let table_view = TableV::from_table(table.clone(), current_offset, table_slice.len);
        let result = broadcast_tableview_to_tableview(op, table_slice, &table_view)?;
        // Convert the resulting Table back to a TableView
        result_slices.push(TableV::from_table(result, 0, table_slice.len));
        current_offset += table_slice.len;
    }

    Ok(SuperTableV {
        slices: result_slices,
        len: super_table_view.len,
    })
}

#[cfg(all(test, feature = "chunked", feature = "views"))]
mod tests {
    use super::*;
    use crate::ffi::arrow_dtype::ArrowType;
    use crate::{Array, Field, FieldArray, IntegerArray, NumericArray, Table, vec64};

    #[cfg(feature = "scalar_type")]
    #[test]
    fn test_supertableview_to_scalar_add() {
        // Create SuperTableView with 2 slices
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

        let arr2 = Array::from_int32(IntegerArray::from_slice(&vec64![4, 5, 6]));
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

        let scalar = Scalar::Int32(10);

        let result =
            broadcast_supertableview_to_scalar(ArithmeticOperator::Add, &super_table_view, &scalar)
                .unwrap();

        assert_eq!(result.len, 6);
        assert_eq!(result.slices.len(), 2);

        // First slice: [1,2,3] + 10 = [11,12,13]
        let slice1 = result.slices[0].to_table();
        if let Array::NumericArray(NumericArray::Int32(arr)) = &slice1.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[11, 12, 13]);
        } else {
            panic!("Expected Int32 array");
        }

        // Second slice: [4,5,6] + 10 = [14,15,16]
        let slice2 = result.slices[1].to_table();
        if let Array::NumericArray(NumericArray::Int32(arr)) = &slice2.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[14, 15, 16]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[test]
    fn test_supertableview_to_arrayview_multiply() {
        // Create SuperTableView with 2 slices
        let arr1 = Array::from_int32(IntegerArray::from_slice(&vec64![2, 3, 4]));
        let table1 = Table::build(
            vec![FieldArray::new(
                Field::new("col1".to_string(), ArrowType::Int32, false, None),
                arr1,
            )],
            3,
            "test".to_string(),
        );
        let table_view1 = TableV::from_table(table1, 0, 3);

        let arr2 = Array::from_int32(IntegerArray::from_slice(&vec64![5, 6, 7]));
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

        // Create ArrayView: [10, 10, 10, 10, 10, 10]
        let arr = Array::from_int32(IntegerArray::from_slice(&vec64![10, 10, 10, 10, 10, 10]));
        let array_view = ArrayV::from(arr);

        let result = broadcast_supertableview_to_arrayview(
            ArithmeticOperator::Multiply,
            &super_table_view,
            &array_view,
        )
        .unwrap();

        assert_eq!(result.len, 6);
        assert_eq!(result.slices.len(), 2);

        // First slice: [2,3,4] * [10,10,10] = [20,30,40]
        let slice1 = result.slices[0].to_table();
        if let Array::NumericArray(NumericArray::Int32(arr)) = &slice1.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[20, 30, 40]);
        } else {
            panic!("Expected Int32 array");
        }

        // Second slice: [5,6,7] * [10,10,10] = [50,60,70]
        let slice2 = result.slices[1].to_table();
        if let Array::NumericArray(NumericArray::Int32(arr)) = &slice2.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[50, 60, 70]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[test]
    fn test_supertableview_to_arrayview_length_mismatch() {
        // Create SuperTableView with 6 elements
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

        let arr2 = Array::from_int32(IntegerArray::from_slice(&vec64![4, 5, 6]));
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

        // Create ArrayView with 5 elements (mismatch)
        let arr = Array::from_int32(IntegerArray::from_slice(&vec64![10, 10, 10, 10, 10]));
        let array_view = ArrayV::from(arr);

        let result = broadcast_supertableview_to_arrayview(
            ArithmeticOperator::Add,
            &super_table_view,
            &array_view,
        );

        assert!(result.is_err());
        if let Err(MinarrowError::ShapeError { message }) = result {
            assert!(message.contains("does not match"));
        } else {
            panic!("Expected ShapeError");
        }
    }

    #[test]
    fn test_superarrayview_to_table_subtract() {
        use crate::{SuperArray, SuperArrayV};

        // Create SuperArrayView with 2 chunks: [100, 200, 300], [400, 500, 600]
        let fa1 = FieldArray::new(
            Field::new("test".to_string(), ArrowType::Int32, false, None),
            Array::from_int32(IntegerArray::from_slice(&vec64![100, 200, 300])),
        );
        let fa2 = FieldArray::new(
            Field::new("test".to_string(), ArrowType::Int32, false, None),
            Array::from_int32(IntegerArray::from_slice(&vec64![400, 500, 600])),
        );
        let super_array = SuperArray::from_chunks(vec![fa1, fa2]);
        let super_array_view = SuperArrayV::from(super_array);

        // Create Table: [[10, 20, 30, 40, 50, 60]]
        let arr = Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30, 40, 50, 60]));
        let table = Table::build(
            vec![FieldArray::new(
                Field::new("col1".to_string(), ArrowType::Int32, false, None),
                arr,
            )],
            6,
            "test".to_string(),
        );

        let result = broadcast_superarrayview_to_table(
            ArithmeticOperator::Subtract,
            &super_array_view,
            &table,
        )
        .unwrap();

        assert_eq!(result.len, 6);
        assert_eq!(result.slices.len(), 2);

        // First slice: [100,200,300] - [10,20,30] = [90,180,270]
        let slice1 = result.slices[0].to_table();
        if let Array::NumericArray(NumericArray::Int32(arr)) = &slice1.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[90, 180, 270]);
        } else {
            panic!("Expected Int32 array");
        }

        // Second slice: [400,500,600] - [40,50,60] = [360,450,540]
        let slice2 = result.slices[1].to_table();
        if let Array::NumericArray(NumericArray::Int32(arr)) = &slice2.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[360, 450, 540]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[test]
    fn test_superarrayview_to_table_length_mismatch() {
        use crate::{SuperArray, SuperArrayV};

        // Create SuperArrayView with 6 elements
        let fa1 = FieldArray::new(
            Field::new("test".to_string(), ArrowType::Int32, false, None),
            Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3])),
        );
        let fa2 = FieldArray::new(
            Field::new("test".to_string(), ArrowType::Int32, false, None),
            Array::from_int32(IntegerArray::from_slice(&vec64![4, 5, 6])),
        );
        let super_array = SuperArray::from_chunks(vec![fa1, fa2]);
        let super_array_view = SuperArrayV::from(super_array);

        // Create Table with 5 rows (mismatch)
        let arr = Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30, 40, 50]));
        let table = Table::build(
            vec![FieldArray::new(
                Field::new("col1".to_string(), ArrowType::Int32, false, None),
                arr,
            )],
            5,
            "test".to_string(),
        );

        let result =
            broadcast_superarrayview_to_table(ArithmeticOperator::Add, &super_array_view, &table);

        assert!(result.is_err());
        if let Err(MinarrowError::ShapeError { message }) = result {
            assert!(message.contains("does not match"));
        } else {
            panic!("Expected ShapeError");
        }
    }

    #[test]
    fn test_supertableview_to_array_divide() {
        // Create SuperTableView with 2 slices
        let arr1 = Array::from_int32(IntegerArray::from_slice(&vec64![100, 200, 300]));
        let table1 = Table::build(
            vec![FieldArray::new(
                Field::new("col1".to_string(), ArrowType::Int32, false, None),
                arr1,
            )],
            3,
            "test".to_string(),
        );
        let table_view1 = TableV::from_table(table1, 0, 3);

        let arr2 = Array::from_int32(IntegerArray::from_slice(&vec64![400, 500, 600]));
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

        // Create Array: [10, 20, 30, 40, 50, 60]
        let arr = Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30, 40, 50, 60]));

        let result =
            broadcast_supertableview_to_array(ArithmeticOperator::Divide, &super_table_view, &arr)
                .unwrap();

        assert_eq!(result.len, 6);
        assert_eq!(result.slices.len(), 2);

        // First slice: [100,200,300] / [10,20,30] = [10,10,10]
        let slice1 = result.slices[0].to_table();
        if let Array::NumericArray(NumericArray::Int32(arr)) = &slice1.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[10, 10, 10]);
        } else {
            panic!("Expected Int32 array");
        }

        // Second slice: [400,500,600] / [40,50,60] = [10,10,10]
        let slice2 = result.slices[1].to_table();
        if let Array::NumericArray(NumericArray::Int32(arr)) = &slice2.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[10, 10, 10]);
        } else {
            panic!("Expected Int32 array");
        }
    }
}
