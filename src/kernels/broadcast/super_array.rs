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

#[cfg(all(feature = "chunked", feature = "scalar_type"))]
use crate::Scalar;
use crate::enums::error::KernelError;
#[cfg(feature = "chunked")]
use crate::enums::error::MinarrowError;
#[cfg(feature = "chunked")]
use crate::enums::operators::ArithmeticOperator;
use crate::kernels::broadcast::array::{broadcast_array_add, broadcast_array_to_table};
use crate::kernels::broadcast::broadcast_value;
use crate::kernels::routing::arithmetic::resolve_binary_arithmetic;
use crate::traits::shape::Shape;
use crate::{Bitmask, FieldArray, SuperArray, SuperArrayV, Table, Value};

/// Broadcasts addition over all child array chunks
pub fn broadcast_super_array_add(
    lhs: impl Into<SuperArrayV>,
    rhs: impl Into<SuperArrayV>,
    null_mask_override: Option<Arc<Bitmask>>,
) -> Result<SuperArray, KernelError> {
    let lhs_arr: SuperArrayV = lhs.into();
    let rhs_arr: SuperArrayV = rhs.into();
    let mut super_array: SuperArray = SuperArray::default();
    for (i, lhs_chunk) in lhs_arr.chunks().enumerate() {
        let rhs_chunk = &rhs_arr.slices[i];
        let len_lhs = lhs_arr.slices[i].len();
        let len_rhs = rhs_arr.slices[i].len();
        if len_lhs != len_rhs {
            return Err(KernelError::BroadcastingError(format!(
                "Super Array broadcasting error for - Chunk: LHS {len_lhs} RHS {len_rhs}, Shape: LHS {:?} RHS {:?}",
                lhs_arr.shape_1d(),
                rhs_arr.shape_1d()
            )));
        }
        let mask = match null_mask_override {
            None => {
                let lhs_null_mask = lhs_chunk.null_mask_view();
                let rhs_null_mask = rhs_chunk.null_mask_view();
                let masks = (lhs_null_mask, rhs_null_mask);
                let common_mask: Option<Arc<Bitmask>> = match masks {
                    (None, None) => None,
                    (None, Some(rhs_bm)) => Some(Arc::new(rhs_bm.bitmask.clone())),
                    (Some(lhs_bm), None) => Some(Arc::new(lhs_bm.bitmask.clone())),
                    (Some(lhs_bm), Some(rhs_bm)) => {
                        Some(lhs_bm.bitmask.union(rhs_bm.bitmask).into())
                    }
                };
                common_mask
            }
            Some(ref m) => Some(m.clone()),
        };
        let arr_res = broadcast_array_add(lhs_chunk.clone(), rhs_chunk.clone(), mask.as_deref());
        let arr = match arr_res {
            Ok(arr) => arr,
            Err(e) => {
                return Err(KernelError::BroadcastingError(format!(
                    "Super Array broadcasting error for - Error: {e}, Chunk: LHS {len_lhs} RHS {len_rhs}, Shape: LHS {:?} RHS {:?}",
                    lhs_arr.shape_1d(),
                    rhs_arr.shape_1d()
                )));
            }
        };
        // TODO: Metadata clone has potential to be heavily than should be required here.
        // Push the result array into the SuperArray
        super_array.push(arr);
    }
    Ok(super_array)
}

/// Helper function for SuperArray-Scalar broadcasting - broadcast each chunk against scalar
#[cfg(all(feature = "chunked", feature = "scalar_type"))]
pub fn broadcast_superarray_to_scalar(
    op: ArithmeticOperator,
    super_array: &SuperArray,
    scalar: &Scalar,
) -> Result<SuperArray, MinarrowError> {
    let result_chunks: Result<Vec<_>, _> = super_array
        .chunks()
        .iter()
        .map(|chunk| {
            let chunk_result = broadcast_value(
                op,
                Value::Array(Arc::new(chunk.clone())),
                Value::Scalar(scalar.clone()),
            )?;
            match chunk_result {
                Value::Array(arr) => Ok(FieldArray::new(
                    super_array.field_ref().clone(),
                    Arc::unwrap_or_clone(arr),
                )),
                _ => Err(MinarrowError::TypeError {
                    from: "Array chunk + Scalar",
                    to: "Array",
                    message: Some("Expected Array result from chunk operation".to_string()),
                }),
            }
        })
        .collect();

    Ok(SuperArray::from_chunks(result_chunks?))
}

/// Helper function for SuperArrayView-Scalar broadcasting - broadcast each view slice against scalar
#[cfg(all(feature = "chunked", feature = "scalar_type", feature = "views"))]
pub fn broadcast_superarrayview_to_scalar(
    op: ArithmeticOperator,
    super_array_view: &SuperArrayV,
    scalar: &Scalar,
) -> Result<SuperArray, MinarrowError> {
    let result_chunks: Result<Vec<_>, _> = super_array_view
        .slices
        .iter()
        .map(|slice| {
            let chunk_result = broadcast_value(
                op,
                Value::ArrayView(Arc::new(slice.clone())),
                Value::Scalar(scalar.clone()),
            )?;
            match chunk_result {
                Value::Array(arr) => Ok(FieldArray::new(
                    (*super_array_view.field).clone(),
                    Arc::unwrap_or_clone(arr),
                )),
                _ => Err(MinarrowError::TypeError {
                    from: "ArrayView chunk + Scalar",
                    to: "Array",
                    message: Some("Expected Array result from chunk operation".to_string()),
                }),
            }
        })
        .collect();

    Ok(SuperArray::from_chunks(result_chunks?))
}

/// Helper function for SuperArray-Table broadcasting - broadcast each chunk against table
#[cfg(feature = "chunked")]
pub fn broadcast_superarray_to_table(
    op: ArithmeticOperator,
    super_array: &SuperArray,
    table: &Table,
) -> Result<SuperArray, MinarrowError> {
    let new_chunks: Result<Vec<_>, _> = super_array
        .chunks()
        .iter()
        .map(|chunk| {
            let result_table = broadcast_array_to_table(op, chunk, table)?;
            // Convert result table back to a FieldArray chunk with matching structure
            if result_table.cols.len() == 1 {
                Ok(result_table.cols[0].clone())
            } else {
                Err(MinarrowError::ShapeError {
                    message: "SuperArray-Table broadcasting should result in single column"
                        .to_string(),
                })
            }
        })
        .collect();

    Ok(SuperArray::from_chunks(new_chunks?))
}

#[cfg(feature = "chunked")]
/// Routes SuperArray arithmetic operations to correct broadcast function
pub fn route_super_array_broadcast(
    op: ArithmeticOperator,
    lhs: impl Into<SuperArrayV>,
    rhs: impl Into<SuperArrayV>,
    null_mask_override: Option<Arc<Bitmask>>,
) -> Result<SuperArray, MinarrowError> {
    use SuperArray;

    // LHS and RHS as Super Array Views
    let lhs_arr: SuperArrayV = lhs.into();
    let rhs_arr: SuperArrayV = rhs.into();
    let mut super_array: SuperArray = SuperArray::default();

    // TODO: Parallelise
    // Iterate over each chunk
    for (i, lhs_chunk) in lhs_arr.chunks().enumerate() {
        let rhs_chunk = &rhs_arr.slices[i];

        // Get their length and confirm equal and consistent shapes
        let len_lhs = lhs_arr.slices[i].len();
        let len_rhs = rhs_arr.slices[i].len();

        if len_lhs != len_rhs {
            return Err(MinarrowError::ShapeError {
                message: format!(
                    "Super Array broadcasting error for {:?} - Chunk: LHS {len_lhs} RHS {len_rhs}, Shape: LHS {:?} RHS {:?}",
                    op,
                    lhs_arr.shape_1d(),
                    rhs_arr.shape_1d()
                ),
            });
        }

        // Produce a common null mask
        let mask = match null_mask_override {
            None => {
                let lhs_null_mask = lhs_chunk.null_mask_view();
                let rhs_null_mask = rhs_chunk.null_mask_view();
                let masks = (lhs_null_mask, rhs_null_mask);
                let common_mask: Option<Arc<Bitmask>> = match masks {
                    (None, None) => None,
                    (None, Some(rhs_bm)) => Some(Arc::new(rhs_bm.bitmask.clone())),
                    (Some(lhs_bm), None) => Some(Arc::new(lhs_bm.bitmask.clone())),
                    (Some(lhs_bm), Some(rhs_bm)) => {
                        Some(lhs_bm.bitmask.union(rhs_bm.bitmask).into())
                    }
                };
                common_mask
            }
            Some(ref m) => Some(m.clone()),
        };

        // Resolve the arithmetic on a per chunk basis
        let arr_res =
            resolve_binary_arithmetic(op, lhs_chunk.clone(), rhs_chunk.clone(), mask.as_deref());
        let arr = match arr_res {
            Ok(arr) => arr,
            Err(e) => {
                return Err(MinarrowError::KernelError(Some(format!(
                    "Super Array broadcasting error for {:?} - Error: {}, Chunk: LHS {len_lhs} RHS {len_rhs}, Shape: LHS {:?} RHS {:?}",
                    op,
                    e,
                    lhs_arr.shape_1d(),
                    rhs_arr.shape_1d()
                ))));
            }
        };

        super_array.push(arr);
    }
    Ok(super_array)
}

/// Helper function for ArrayView-SuperArray broadcasting - work with view directly by creating aligned views
#[cfg(all(feature = "views", feature = "chunked"))]
pub fn broadcast_arrayview_to_superarray(
    op: ArithmeticOperator,
    array_view: &crate::ArrayV,
    super_array: &SuperArray,
) -> Result<SuperArray, MinarrowError> {
    // Validate lengths match
    if array_view.len() != super_array.len() {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "ArrayView length ({}) does not match SuperArray length ({})",
                array_view.len(),
                super_array.len()
            ),
        });
    }

    // Broadcast per chunk using the view's window
    let mut result_chunks = Vec::new();
    let mut current_offset = 0;

    for chunk in super_array.chunks() {
        // Create a view into the array matching this chunk's size
        let array_slice = array_view.slice(current_offset, chunk.len());

        // Broadcast the array slice with this chunk
        let result = match (
            Value::ArrayView(Arc::new(array_slice)),
            Value::Array(Arc::new(chunk.clone())),
        ) {
            (a, b) => broadcast_value(op, a, b)?,
        };

        match result {
            Value::Array(arr) => {
                let field_array = FieldArray::new_arc(
                    super_array.field.clone().unwrap(),
                    Arc::unwrap_or_clone(arr),
                );
                result_chunks.push(field_array);
            }
            _ => {
                return Err(MinarrowError::TypeError {
                    from: "arrayview-superarray broadcasting",
                    to: "Array result",
                    message: Some("Expected Array result from broadcasting".to_string()),
                });
            }
        }
        current_offset += chunk.len();
    }

    Ok(SuperArray::from_field_array_chunks(result_chunks))
}

/// Helper function for SuperArray-ArrayView broadcasting - work with view directly by creating aligned views
#[cfg(all(feature = "views", feature = "chunked"))]
pub fn broadcast_superarray_to_arrayview(
    op: ArithmeticOperator,
    super_array: &SuperArray,
    array_view: &crate::ArrayV,
) -> Result<SuperArray, MinarrowError> {
    // Validate lengths match
    if super_array.len() != array_view.len() {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "SuperArray length ({}) does not match ArrayView length ({})",
                super_array.len(),
                array_view.len()
            ),
        });
    }

    // Broadcast per chunk using the view's window
    let mut result_chunks = Vec::new();
    let mut current_offset = 0;

    for chunk in super_array.chunks() {
        // Create a view into the array matching this chunk's size
        let array_slice = array_view.slice(current_offset, chunk.len());

        // Broadcast this chunk with the array slice
        let result = match (
            Value::Array(Arc::new(chunk.clone())),
            Value::ArrayView(Arc::new(array_slice)),
        ) {
            (a, b) => broadcast_value(op, a, b)?,
        };

        match result {
            Value::Array(arr) => {
                let field_array = FieldArray::new_arc(
                    super_array.field.clone().unwrap(),
                    Arc::unwrap_or_clone(arr),
                );
                result_chunks.push(field_array);
            }
            _ => {
                return Err(MinarrowError::TypeError {
                    from: "superarray-arrayview broadcasting",
                    to: "Array result",
                    message: Some("Expected Array result from broadcasting".to_string()),
                });
            }
        }
        current_offset += chunk.len();
    }

    Ok(SuperArray::from_field_array_chunks(result_chunks))
}

/// Helper function for ArrayView-SuperArrayView broadcasting - work with views directly
#[cfg(all(feature = "views", feature = "chunked"))]
pub fn broadcast_arrayview_to_superarrayview(
    op: ArithmeticOperator,
    array_view: &crate::ArrayV,
    super_array_view: &SuperArrayV,
) -> Result<SuperArray, MinarrowError> {
    // Validate lengths match
    if array_view.len() != super_array_view.len {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "ArrayView length ({}) does not match SuperArrayView length ({})",
                array_view.len(),
                super_array_view.len
            ),
        });
    }

    // Broadcast per chunk using views
    let mut result_chunks = Vec::new();
    let mut current_offset = 0;

    for slice in super_array_view.slices.iter() {
        // Create a view into the array matching this slice's size
        let array_slice = array_view.slice(current_offset, slice.len());

        // Broadcast the array slice with this super array slice
        let result = match (
            Value::ArrayView(Arc::new(array_slice)),
            Value::ArrayView(Arc::new(slice.clone())),
        ) {
            (a, b) => broadcast_value(op, a, b)?,
        };

        match result {
            Value::Array(arr) => {
                let field_array =
                    FieldArray::new_arc(super_array_view.field.clone(), Arc::unwrap_or_clone(arr));
                result_chunks.push(field_array);
            }
            _ => {
                return Err(MinarrowError::TypeError {
                    from: "arrayview-superarrayview broadcasting",
                    to: "Array result",
                    message: Some("Expected Array result from broadcasting".to_string()),
                });
            }
        }
        current_offset += slice.len();
    }

    Ok(SuperArray::from_field_array_chunks(result_chunks))
}

/// Helper function for SuperArrayView-ArrayView broadcasting - work with views directly
#[cfg(all(feature = "views", feature = "chunked"))]
pub fn broadcast_superarrayview_to_arrayview(
    op: ArithmeticOperator,
    super_array_view: &SuperArrayV,
    array_view: &crate::ArrayV,
) -> Result<SuperArray, MinarrowError> {
    // Validate lengths match
    if super_array_view.len != array_view.len() {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "SuperArrayView length ({}) does not match ArrayView length ({})",
                super_array_view.len,
                array_view.len()
            ),
        });
    }

    // Broadcast per chunk using views
    let mut result_chunks = Vec::new();
    let mut current_offset = 0;

    for slice in super_array_view.slices.iter() {
        // Create a view into the array matching this slice's size
        let array_slice = array_view.slice(current_offset, slice.len());

        // Broadcast this super array slice with the array slice
        let result = match (
            Value::ArrayView(Arc::new(slice.clone())),
            Value::ArrayView(Arc::new(array_slice)),
        ) {
            (a, b) => broadcast_value(op, a, b)?,
        };

        match result {
            Value::Array(arr) => {
                let field_array =
                    FieldArray::new_arc(super_array_view.field.clone(), Arc::unwrap_or_clone(arr));
                result_chunks.push(field_array);
            }
            _ => {
                return Err(MinarrowError::TypeError {
                    from: "superarrayview-arrayview broadcasting",
                    to: "Array result",
                    message: Some("Expected Array result from broadcasting".to_string()),
                });
            }
        }
        current_offset += slice.len();
    }

    Ok(SuperArray::from_field_array_chunks(result_chunks))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::arrow_dtype::ArrowType;
    use crate::{Array, ArrayV, Field, IntegerArray, NumericArray, vec64};

    #[test]
    fn test_array_plus_scalar() {
        let arr1 = Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3]));
        let scalar = Array::from_int32(IntegerArray::from_slice(&vec64![5]));

        let result =
            broadcast_array_add(ArrayV::new(arr1, 0, 3), ArrayV::new(scalar, 0, 1), None).unwrap();

        if let Array::NumericArray(NumericArray::Int32(result_arr)) = result {
            assert_eq!(result_arr.data.as_slice(), &[6, 7, 8]);
        } else {
            panic!("Expected Int32 result");
        }
    }

    #[test]
    fn test_scalar_plus_array() {
        let scalar = Array::from_int32(IntegerArray::from_slice(&vec64![5]));
        let arr2 = Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3]));

        let result =
            broadcast_array_add(ArrayV::new(scalar, 0, 1), ArrayV::new(arr2, 0, 3), None).unwrap();

        if let Array::NumericArray(NumericArray::Int32(result_arr)) = result {
            assert_eq!(result_arr.data.as_slice(), &[6, 7, 8]);
        } else {
            panic!("Expected Int32 result");
        }
    }

    #[test]
    fn test_array_plus_array() {
        let arr1 = Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3]));
        let arr2 = Array::from_int32(IntegerArray::from_slice(&vec64![4, 5, 6]));

        let result =
            broadcast_array_add(ArrayV::new(arr1, 0, 3), ArrayV::new(arr2, 0, 3), None).unwrap();

        if let Array::NumericArray(NumericArray::Int32(result_arr)) = result {
            assert_eq!(result_arr.data.as_slice(), &[5, 7, 9]);
        } else {
            panic!("Expected Int32 result");
        }
    }

    #[cfg(feature = "chunked")]
    #[test]
    fn test_broadcast_super_array_add() {
        // Create SuperArray with 2 chunks: [1, 2, 3], [4, 5, 6]
        let fa1 = FieldArray::new(
            Field::new("test".to_string(), ArrowType::Int32, false, None),
            Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3])),
        );
        let fa2 = FieldArray::new(
            Field::new("test".to_string(), ArrowType::Int32, false, None),
            Array::from_int32(IntegerArray::from_slice(&vec64![4, 5, 6])),
        );
        let super_array1 = SuperArray::from_chunks(vec![fa1, fa2]);

        // Create second SuperArray: [10, 10, 10], [20, 20, 20]
        let fa3 = FieldArray::new(
            Field::new("test".to_string(), ArrowType::Int32, false, None),
            Array::from_int32(IntegerArray::from_slice(&vec64![10, 10, 10])),
        );
        let fa4 = FieldArray::new(
            Field::new("test".to_string(), ArrowType::Int32, false, None),
            Array::from_int32(IntegerArray::from_slice(&vec64![20, 20, 20])),
        );
        let super_array2 = SuperArray::from_chunks(vec![fa3, fa4]);

        let result = broadcast_super_array_add(super_array1, super_array2, None).unwrap();

        assert_eq!(result.chunks().len(), 2);

        // First chunk: [1,2,3] + [10,10,10] = [11,12,13]
        if let Array::NumericArray(NumericArray::Int32(arr)) = &result.chunks()[0] {
            assert_eq!(arr.data.as_slice(), &[11, 12, 13]);
        } else {
            panic!("Expected Int32 array");
        }

        // Second chunk: [4,5,6] + [20,20,20] = [24,25,26]
        if let Array::NumericArray(NumericArray::Int32(arr)) = &result.chunks()[1] {
            assert_eq!(arr.data.as_slice(), &[24, 25, 26]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[cfg(feature = "chunked")]
    #[test]
    fn test_broadcast_super_array_add_length_mismatch() {
        // Create SuperArray with mismatched chunk lengths
        let fa1 = FieldArray::new(
            Field::new("test".to_string(), ArrowType::Int32, false, None),
            Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3])),
        );
        let super_array1 = SuperArray::from_chunks(vec![fa1]);

        let fa2 = FieldArray::new(
            Field::new("test".to_string(), ArrowType::Int32, false, None),
            Array::from_int32(IntegerArray::from_slice(&vec64![10, 10])), // Different length
        );
        let super_array2 = SuperArray::from_chunks(vec![fa2]);

        let result = broadcast_super_array_add(super_array1, super_array2, None);

        assert!(result.is_err());
        if let Err(KernelError::BroadcastingError(msg)) = result {
            assert!(msg.contains("Super Array broadcasting error"));
        } else {
            panic!("Expected BroadcastingError");
        }
    }

    #[cfg(feature = "chunked")]
    #[test]
    fn test_route_super_array_broadcast_multiply() {
        // Create SuperArray with 2 chunks
        let fa1 = FieldArray::new(
            Field::new("test".to_string(), ArrowType::Int32, false, None),
            Array::from_int32(IntegerArray::from_slice(&vec64![2, 3, 4])),
        );
        let fa2 = FieldArray::new(
            Field::new("test".to_string(), ArrowType::Int32, false, None),
            Array::from_int32(IntegerArray::from_slice(&vec64![5, 6, 7])),
        );
        let super_array1 = SuperArray::from_chunks(vec![fa1, fa2]);

        let fa3 = FieldArray::new(
            Field::new("test".to_string(), ArrowType::Int32, false, None),
            Array::from_int32(IntegerArray::from_slice(&vec64![10, 10, 10])),
        );
        let fa4 = FieldArray::new(
            Field::new("test".to_string(), ArrowType::Int32, false, None),
            Array::from_int32(IntegerArray::from_slice(&vec64![2, 2, 2])),
        );
        let super_array2 = SuperArray::from_chunks(vec![fa3, fa4]);

        let result = route_super_array_broadcast(
            ArithmeticOperator::Multiply,
            super_array1,
            super_array2,
            None,
        )
        .unwrap();

        // First chunk: [2,3,4] * [10,10,10] = [20,30,40]
        if let Array::NumericArray(NumericArray::Int32(arr)) = &result.chunks()[0] {
            assert_eq!(arr.data.as_slice(), &[20, 30, 40]);
        } else {
            panic!("Expected Int32 array");
        }

        // Second chunk: [5,6,7] * [2,2,2] = [10,12,14]
        if let Array::NumericArray(NumericArray::Int32(arr)) = &result.chunks()[1] {
            assert_eq!(arr.data.as_slice(), &[10, 12, 14]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[cfg(feature = "chunked")]
    #[test]
    fn test_route_super_array_broadcast_divide() {
        let fa1 = FieldArray::new(
            Field::new("test".to_string(), ArrowType::Int32, false, None),
            Array::from_int32(IntegerArray::from_slice(&vec64![100, 200, 300])),
        );
        let super_array1 = SuperArray::from_chunks(vec![fa1]);

        let fa2 = FieldArray::new(
            Field::new("test".to_string(), ArrowType::Int32, false, None),
            Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30])),
        );
        let super_array2 = SuperArray::from_chunks(vec![fa2]);

        let result = route_super_array_broadcast(
            ArithmeticOperator::Divide,
            super_array1,
            super_array2,
            None,
        )
        .unwrap();

        // [100,200,300] / [10,20,30] = [10,10,10]
        if let Array::NumericArray(NumericArray::Int32(arr)) = &result.chunks()[0] {
            assert_eq!(arr.data.as_slice(), &[10, 10, 10]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[cfg(feature = "chunked")]
    #[test]
    fn test_broadcast_superarray_to_table() {
        use crate::Table;

        // Create SuperArray with 2 chunks: [1, 2, 3], [4, 5, 6]
        let fa1 = FieldArray::new(
            Field::new("test".to_string(), ArrowType::Int32, false, None),
            Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3])),
        );
        let fa2 = FieldArray::new(
            Field::new("test".to_string(), ArrowType::Int32, false, None),
            Array::from_int32(IntegerArray::from_slice(&vec64![4, 5, 6])),
        );
        let super_array = SuperArray::from_chunks(vec![fa1, fa2]);

        // Create a single-column table: [[10, 20, 30]]
        let table_arr = Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30]));
        let table = Table::build(
            vec![FieldArray::new(
                Field::new("col1".to_string(), ArrowType::Int32, false, None),
                table_arr,
            )],
            3,
            "test".to_string(),
        );

        let result =
            broadcast_superarray_to_table(ArithmeticOperator::Add, &super_array, &table).unwrap();

        assert_eq!(result.chunks().len(), 2);

        // First chunk: [1,2,3] + [10,20,30] = [11,22,33]
        if let Array::NumericArray(NumericArray::Int32(arr)) = &result.chunks()[0] {
            assert_eq!(arr.data.as_slice(), &[11, 22, 33]);
        } else {
            panic!("Expected Int32 array");
        }

        // Second chunk: [4,5,6] + [10,20,30] = [14,25,36]
        if let Array::NumericArray(NumericArray::Int32(arr)) = &result.chunks()[1] {
            assert_eq!(arr.data.as_slice(), &[14, 25, 36]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[cfg(feature = "chunked")]
    #[test]
    fn test_route_super_array_broadcast_subtract() {
        let fa1 = FieldArray::new(
            Field::new("test".to_string(), ArrowType::Int32, false, None),
            Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30])),
        );
        let fa2 = FieldArray::new(
            Field::new("test".to_string(), ArrowType::Int32, false, None),
            Array::from_int32(IntegerArray::from_slice(&vec64![100, 200, 300])),
        );
        let super_array1 = SuperArray::from_chunks(vec![fa1, fa2]);

        let fa3 = FieldArray::new(
            Field::new("test".to_string(), ArrowType::Int32, false, None),
            Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3])),
        );
        let fa4 = FieldArray::new(
            Field::new("test".to_string(), ArrowType::Int32, false, None),
            Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30])),
        );
        let super_array2 = SuperArray::from_chunks(vec![fa3, fa4]);

        let result = route_super_array_broadcast(
            ArithmeticOperator::Subtract,
            super_array1,
            super_array2,
            None,
        )
        .unwrap();

        // First chunk: [10,20,30] - [1,2,3] = [9,18,27]
        if let Array::NumericArray(NumericArray::Int32(arr)) = &result.chunks()[0] {
            assert_eq!(arr.data.as_slice(), &[9, 18, 27]);
        } else {
            panic!("Expected Int32 array");
        }

        // Second chunk: [100,200,300] - [10,20,30] = [90,180,270]
        if let Array::NumericArray(NumericArray::Int32(arr)) = &result.chunks()[1] {
            assert_eq!(arr.data.as_slice(), &[90, 180, 270]);
        } else {
            panic!("Expected Int32 array");
        }
    }
}
