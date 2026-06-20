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
use crate::{FieldArray, Value};

#[cfg(all(feature = "chunked", feature = "views"))]
use crate::kernels::broadcast::array_view::broadcast_arrayview_to_tableview;
#[cfg(all(feature = "chunked", feature = "views"))]
use crate::kernels::broadcast::table_view::broadcast_tableview_to_arrayview;
#[cfg(all(feature = "chunked", feature = "views"))]
use crate::{ArrayV, SuperArray, SuperArrayV, SuperTableV, TableV};

/// Helper function for FieldArray-SuperArrayView broadcasting - promote FieldArray to SuperArrayView
#[cfg(all(feature = "chunked", feature = "views"))]
pub fn broadcast_fieldarray_to_superarrayview(
    op: ArithmeticOperator,
    field_array: &FieldArray,
    super_array_view: &SuperArrayV,
) -> Result<Value, MinarrowError> {
    // Convert FieldArray to SuperArray then to SuperArrayView
    let l_super_array = SuperArray::from_chunks(vec![field_array.clone()]);
    let l_super_array_view = l_super_array.slice(0, l_super_array.len());
    let result = match (
        Value::SuperArrayView(Arc::new(l_super_array_view.into())),
        Value::SuperArrayView(Arc::new(super_array_view.clone().into())),
    ) {
        (a, b) => broadcast_value(op, a, b)?,
    };
    Ok(result)
}

/// Helper function for SuperArrayView-FieldArray broadcasting - promote FieldArray to SuperArrayView
#[cfg(all(feature = "chunked", feature = "views"))]
pub fn broadcast_superarrayview_to_fieldarray(
    op: ArithmeticOperator,
    super_array_view: &SuperArrayV,
    field_array: &FieldArray,
) -> Result<Value, MinarrowError> {
    // Convert FieldArray to SuperArray then to SuperArrayView
    let r_super_array = SuperArray::from_chunks(vec![field_array.clone()]);
    let r_super_array_view = r_super_array.slice(0, r_super_array.len());
    let result = match (
        Value::SuperArrayView(Arc::new(super_array_view.clone().into())),
        Value::SuperArrayView(Arc::new(r_super_array_view.into())),
    ) {
        (a, b) => broadcast_value(op, a, b)?,
    };
    Ok(result)
}

/// Helper function for FieldArray-SuperTableView broadcasting - chunk the array to match super table view structure
#[cfg(all(feature = "chunked", feature = "views"))]
pub fn broadcast_fieldarray_to_supertableview(
    op: ArithmeticOperator,
    field_array: &FieldArray,
    super_table_view: &SuperTableV,
) -> Result<SuperTableV, MinarrowError> {
    // Check total lengths match
    if field_array.array.len() != super_table_view.len {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "FieldArray length ({}) does not match SuperTableView length ({})",
                field_array.array.len(),
                super_table_view.len
            ),
        });
    }

    // Chunk the array to match super table view structure and broadcast per chunk
    let mut current_offset = 0;
    let mut result_slices = Vec::new();

    for table_slice in super_table_view.slices.iter() {
        // Create an array view matching this table slice's size
        let array_view = ArrayV::new(field_array.array.clone(), current_offset, table_slice.len);

        // Broadcast the array view with this table slice
        let slice_result_table = broadcast_arrayview_to_tableview(op, &array_view, table_slice)?;
        result_slices.push(TableV::from_table(slice_result_table, 0, table_slice.len));
        current_offset += table_slice.len;
    }

    Ok(SuperTableV {
        slices: result_slices,
        len: super_table_view.len,
    })
}

/// Helper function for SuperTableView-FieldArray broadcasting - chunk the array to match super table view structure
#[cfg(all(feature = "chunked", feature = "views"))]
pub fn broadcast_supertableview_to_fieldarray(
    op: ArithmeticOperator,
    super_table_view: &SuperTableV,
    field_array: &FieldArray,
) -> Result<SuperTableV, MinarrowError> {
    // Check total lengths match
    if super_table_view.len != field_array.array.len() {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "SuperTableView length ({}) does not match FieldArray length ({})",
                super_table_view.len,
                field_array.array.len()
            ),
        });
    }

    // Chunk the array to match super table view structure and broadcast per chunk
    let mut current_offset = 0;
    let mut result_slices = Vec::new();

    for table_slice in super_table_view.slices.iter() {
        // Create an array view matching this table slice's size
        let array_view = ArrayV::new(field_array.array.clone(), current_offset, table_slice.len);

        // Broadcast this table slice with the array view
        let slice_result = broadcast_tableview_to_arrayview(op, table_slice, &array_view)?;
        result_slices.push(slice_result);
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

    #[test]
    fn test_fieldarray_to_supertableview() {
        // Create a FieldArray with 6 elements
        let field_array = FieldArray::new(
            Field::new("data".to_string(), ArrowType::Int32, false, None),
            Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30, 40, 50, 60])),
        );

        // Create SuperTableView with 2 slices
        let table1 = Table::build(
            vec![FieldArray::new(
                Field::new("col1".to_string(), ArrowType::Int32, false, None),
                Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3])),
            )],
            3,
            "test".to_string(),
        );
        let table_view1 = TableV::from_table(table1, 0, 3);

        let table2 = Table::build(
            vec![FieldArray::new(
                Field::new("col1".to_string(), ArrowType::Int32, false, None),
                Array::from_int32(IntegerArray::from_slice(&vec64![4, 5, 6])),
            )],
            3,
            "test".to_string(),
        );
        let table_view2 = TableV::from_table(table2, 0, 3);

        let super_table_view = SuperTableV {
            slices: vec![table_view1, table_view2],
            len: 6,
        };

        let result = broadcast_fieldarray_to_supertableview(
            ArithmeticOperator::Add,
            &field_array,
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

    #[test]
    fn test_supertableview_to_fieldarray() {
        // Create SuperTableView with 2 slices
        let table1 = Table::build(
            vec![FieldArray::new(
                Field::new("col1".to_string(), ArrowType::Int32, false, None),
                Array::from_int32(IntegerArray::from_slice(&vec64![100, 200, 300])),
            )],
            3,
            "test".to_string(),
        );
        let table_view1 = TableV::from_table(table1, 0, 3);

        let table2 = Table::build(
            vec![FieldArray::new(
                Field::new("col1".to_string(), ArrowType::Int32, false, None),
                Array::from_int32(IntegerArray::from_slice(&vec64![400, 500, 600])),
            )],
            3,
            "test".to_string(),
        );
        let table_view2 = TableV::from_table(table2, 0, 3);

        let super_table_view = SuperTableV {
            slices: vec![table_view1, table_view2],
            len: 6,
        };

        // Create a FieldArray with 6 elements
        let field_array = FieldArray::new(
            Field::new("data".to_string(), ArrowType::Int32, false, None),
            Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3, 4, 5, 6])),
        );

        let result = broadcast_supertableview_to_fieldarray(
            ArithmeticOperator::Subtract,
            &super_table_view,
            &field_array,
        )
        .unwrap();

        assert_eq!(result.len, 6);

        // First slice: [100,200,300] - [1,2,3] = [99,198,297]
        let slice1 = result.slices[0].to_table();
        if let Array::NumericArray(NumericArray::Int32(arr)) = &slice1.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[99, 198, 297]);
        } else {
            panic!("Expected Int32 array");
        }

        // Second slice: [400,500,600] - [4,5,6] = [396,495,594]
        let slice2 = result.slices[1].to_table();
        if let Array::NumericArray(NumericArray::Int32(arr)) = &slice2.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[396, 495, 594]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[test]
    fn test_fieldarray_to_supertableview_length_mismatch() {
        // Create a FieldArray with 5 elements (mismatch)
        let field_array = FieldArray::new(
            Field::new("data".to_string(), ArrowType::Int32, false, None),
            Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30, 40, 50])),
        );

        // Create SuperTableView with 6 total rows
        let table1 = Table::build(
            vec![FieldArray::new(
                Field::new("col1".to_string(), ArrowType::Int32, false, None),
                Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3])),
            )],
            3,
            "test".to_string(),
        );
        let table_view1 = TableV::from_table(table1, 0, 3);

        let table2 = Table::build(
            vec![FieldArray::new(
                Field::new("col1".to_string(), ArrowType::Int32, false, None),
                Array::from_int32(IntegerArray::from_slice(&vec64![4, 5, 6])),
            )],
            3,
            "test".to_string(),
        );
        let table_view2 = TableV::from_table(table2, 0, 3);

        let super_table_view = SuperTableV {
            slices: vec![table_view1, table_view2],
            len: 6,
        };

        let result = broadcast_fieldarray_to_supertableview(
            ArithmeticOperator::Add,
            &field_array,
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
