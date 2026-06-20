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
use crate::kernels::broadcast::array_view::broadcast_arrayview_to_tableview;
use crate::{SuperArrayV, SuperTableV, TableV};

/// Helper function for SuperArrayView-TableView broadcasting - promote TableView to aligned SuperTableView
#[cfg(all(feature = "chunked", feature = "views"))]
pub fn broadcast_superarrayview_to_tableview(
    op: ArithmeticOperator,
    super_array_view: &SuperArrayV,
    table_view: &TableV,
) -> Result<SuperTableV, MinarrowError> {
    // 1. Validate lengths match
    if super_array_view.len != table_view.len {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "SuperArrayView length ({}) does not match TableView length ({})",
                super_array_view.len, table_view.len
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

#[cfg(all(test, feature = "chunked", feature = "views"))]
mod tests {
    use super::*;
    use crate::ffi::arrow_dtype::ArrowType;
    use crate::{Array, Field, FieldArray, IntegerArray, NumericArray, SuperArray, Table, vec64};

    fn create_super_array_view(chunks: Vec<&[i32]>) -> SuperArrayV {
        let field_arrays: Vec<FieldArray> = chunks
            .iter()
            .map(|chunk| {
                let arr = Array::from_int32(IntegerArray::from_slice(chunk));
                let field = Field::new("test_col".to_string(), ArrowType::Int32, false, None);
                FieldArray::new(field, arr)
            })
            .collect();

        let super_array = SuperArray::from_chunks(field_arrays);
        SuperArrayV::from(super_array)
    }

    #[test]
    fn test_superarrayview_to_tableview_single_chunk() {
        // Simple test: single chunk
        // SuperArrayView: [1, 2, 3]
        let super_array_view = create_super_array_view(vec![&[1, 2, 3]]);

        // TableView: [[10, 20, 30]] (1 column, 3 rows)
        let arr1 = Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30]));
        let table = Table::build(
            vec![FieldArray::new(
                Field::new("col1".to_string(), ArrowType::Int32, false, None),
                arr1,
            )],
            3,
            "test".to_string(),
        );
        let table_view = TableV::from_table(table, 0, 3);

        let result = broadcast_superarrayview_to_tableview(
            ArithmeticOperator::Add,
            &super_array_view,
            &table_view,
        )
        .unwrap();

        assert_eq!(result.len, 3);
        assert_eq!(result.slices.len(), 1);

        // Expected: [1,2,3] + [10,20,30] = [11,22,33]
        let result_table = result.slices[0].to_table();
        if let Array::NumericArray(NumericArray::Int32(arr)) = &result_table.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[11, 22, 33]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[test]
    fn test_superarrayview_to_tableview_length_mismatch() {
        // Create SuperArrayView with 3 elements
        let super_array_view = create_super_array_view(vec![&[1, 2, 3]]);

        // Create TableView with 5 rows (mismatched)
        let arr1 = Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30, 40, 50]));
        let table = Table::build(
            vec![FieldArray::new(
                Field::new("col1".to_string(), ArrowType::Int32, false, None),
                arr1,
            )],
            5,
            "test".to_string(),
        );
        let table_view = TableV::from_table(table, 0, 5);

        let result = broadcast_superarrayview_to_tableview(
            ArithmeticOperator::Add,
            &super_array_view,
            &table_view,
        );

        assert!(result.is_err());
        if let Err(MinarrowError::ShapeError { message }) = result {
            assert!(message.contains("does not match"));
        } else {
            panic!("Expected ShapeError with length mismatch message");
        }
    }
}
