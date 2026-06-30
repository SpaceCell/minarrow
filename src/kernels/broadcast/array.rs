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

#[cfg(feature = "cube")]
use crate::Cube;
#[cfg(feature = "chunked")]
use crate::SuperTable;
#[cfg(all(feature = "chunked", feature = "views"))]
use crate::SuperTableV;
use crate::enums::error::MinarrowError;
use crate::enums::operators::ArithmeticOperator;
use crate::kernels::broadcast::array_view::broadcast_arrayview_to_tableview;
use crate::kernels::broadcast::broadcast_value;
use crate::kernels::routing::arithmetic::resolve_binary_arithmetic;
use crate::structs::field_array::create_field_for_array;
use crate::{Array, ArrayV, Bitmask, FieldArray, Table, Value};
#[cfg(feature = "scalar_type")]
use crate::{BooleanArray, DatetimeArray, FloatArray, IntegerArray, Scalar, StringArray};
use std::sync::Arc;

/// Broadcast addition: `lhs + rhs` with automatic scalar expansion.
///
/// If one operand has length 1 and the other has length N, the scalar
/// operand will be broadcast (repeated) to match the array operand's length.
///
/// # Examples
/// - Array + Scalar: `[1, 2, 3] + [5] = [6, 7, 8]`
/// - Scalar + Array: `[5] + [1, 2, 3] = [6, 7, 8]`
/// - Array + Array: `[1, 2, 3] + [4, 5, 6] = [5, 7, 9]`
///
/// # Errors
/// - Returns `KernelError::LengthMismatch` if lengths are incompatible
/// - Returns `KernelError::UnsupportedType` for unsupported type combinations
pub fn broadcast_array_add(
    lhs: impl Into<ArrayV>,
    rhs: impl Into<ArrayV>,
    null_mask_override: Option<&Bitmask>,
) -> Result<Array, MinarrowError> {
    resolve_binary_arithmetic(
        ArithmeticOperator::Add,
        lhs.into(),
        rhs.into(),
        null_mask_override,
    )
}

/// Broadcast division: `lhs / rhs` with automatic scalar expansion.
///
/// If one operand has length 1 and the other has length N, the scalar
/// operand will be broadcast (repeated) to match the array operand's length.
///
/// # Examples
/// - Array / Scalar: `[10, 20, 30] / [2] = [5, 10, 15]`
/// - Scalar / Array: `[100] / [2, 4, 5] = [50, 25, 20]`
/// - Array / Array: `[10, 20, 30] / [2, 4, 5] = [5, 5, 6]`
///
/// # Errors
/// - Returns `KernelError::LengthMismatch` if lengths are incompatible
/// - Returns `KernelError::UnsupportedType` for unsupported type combinations
/// - Returns `KernelError::DivideByZero` for division by zero (integer arrays)
pub fn broadcast_array_div(
    lhs: impl Into<ArrayV>,
    rhs: impl Into<ArrayV>,
    null_mask: Option<&Bitmask>,
) -> Result<Array, MinarrowError> {
    resolve_binary_arithmetic(
        ArithmeticOperator::Divide,
        lhs.into(),
        rhs.into(),
        null_mask,
    )
}

/// Broadcast multiplication: `lhs * rhs` with automatic scalar expansion.
///
/// If one operand has length 1 and the other has length N, the scalar
/// operand will be broadcast (repeated) to match the array operand's length.
///
/// # Examples
/// - Array * Scalar: `[1, 2, 3] * [5] = [5, 10, 15]`
/// - Scalar * Array: `[5] * [1, 2, 3] = [5, 10, 15]`
/// - Array * Array: `[1, 2, 3] * [4, 5, 6] = [4, 10, 18]`
///
/// # Errors
/// - Returns `KernelError::LengthMismatch` if lengths are incompatible
/// - Returns `KernelError::UnsupportedType` for unsupported type combinations
pub fn broadcast_array_mul(
    lhs: impl Into<ArrayV>,
    rhs: impl Into<ArrayV>,
    null_mask: Option<&Bitmask>,
) -> Result<Array, MinarrowError> {
    resolve_binary_arithmetic(
        ArithmeticOperator::Multiply,
        lhs.into(),
        rhs.into(),
        null_mask,
    )
}

/// Broadcast subtraction: `lhs - rhs` with automatic scalar expansion.
///
/// If one operand has length 1 and the other has length N, the scalar
/// operand will be broadcast (repeated) to match the array operand's length.
///
/// # Examples
/// - Array - Scalar: `[5, 6, 7] - [2] = [3, 4, 5]`
/// - Scalar - Array: `[10] - [1, 2, 3] = [9, 8, 7]`
/// - Array - Array: `[5, 6, 7] - [1, 2, 3] = [4, 4, 4]`
///
/// # Errors
/// - Returns `KernelError::LengthMismatch` if lengths are incompatible
/// - Returns `KernelError::UnsupportedType` for unsupported type combinations
pub fn broadcast_array_sub(
    lhs: impl Into<ArrayV>,
    rhs: impl Into<ArrayV>,
    null_mask: Option<&Bitmask>,
) -> Result<Array, MinarrowError> {
    resolve_binary_arithmetic(
        ArithmeticOperator::Subtract,
        lhs.into(),
        rhs.into(),
        null_mask,
    )
}

/// Helper function for array-scalar broadcasting - convert scalar to array, then broadcast
#[cfg(feature = "scalar_type")]
pub fn broadcast_array_to_scalar(
    op: ArithmeticOperator,
    array: &Array,
    scalar: &Scalar,
) -> Result<Array, MinarrowError> {
    // Convert scalar to single-element array
    let scalar_array = match scalar {
        Scalar::Int32(val) => Array::from_int32(IntegerArray::from_slice(&[*val])),
        Scalar::Int64(val) => Array::from_int64(IntegerArray::from_slice(&[*val])),
        Scalar::Float32(val) => Array::from_float32(FloatArray::from_slice(&[*val])),
        Scalar::Float64(val) => Array::from_float64(FloatArray::from_slice(&[*val])),
        Scalar::String32(val) => Array::from_string32(StringArray::from_slice(&[val.as_str()])),
        #[cfg(feature = "large_string")]
        Scalar::String64(val) => Array::from_string32(StringArray::from_slice(&[val.as_str()])),
        Scalar::Boolean(val) => Array::from_bool(BooleanArray::from_slice(&[*val])),
        #[cfg(feature = "extended_numeric_types")]
        Scalar::Int8(val) => Array::from_int8(IntegerArray::from_slice(&[*val])),
        #[cfg(feature = "extended_numeric_types")]
        Scalar::Int16(val) => Array::from_int16(IntegerArray::from_slice(&[*val])),
        Scalar::UInt32(val) => Array::from_uint32(IntegerArray::from_slice(&[*val])),
        Scalar::UInt64(val) => Array::from_uint64(IntegerArray::from_slice(&[*val])),
        #[cfg(feature = "extended_numeric_types")]
        Scalar::UInt8(val) => Array::from_uint8(IntegerArray::from_slice(&[*val])),
        #[cfg(feature = "extended_numeric_types")]
        Scalar::UInt16(val) => Array::from_uint16(IntegerArray::from_slice(&[*val])),

        #[cfg(feature = "datetime")]
        Scalar::Datetime32(val) => {
            Array::from_datetime_i32(DatetimeArray::from_slice(&[*val], None))
        }
        #[cfg(feature = "datetime")]
        Scalar::Datetime64(val) => {
            Array::from_datetime_i64(DatetimeArray::from_slice(&[*val], None))
        }
        Scalar::Null => Array::Null,
        #[cfg(feature = "datetime")]
        Scalar::Interval => {
            return Err(MinarrowError::NotImplemented {
                feature: "Interval scalar broadcasting not yet supported".to_string(),
            });
        }
    };

    // Broadcast the array with the scalar array (scalar expansion will happen automatically)
    resolve_binary_arithmetic(op, array.clone(), scalar_array, None)
}

/// Helper function for array-table broadcasting - apply array to each column
pub fn broadcast_array_to_table(
    op: ArithmeticOperator,
    array: &Array,
    table: &Table,
) -> Result<Table, MinarrowError> {
    let new_cols: Result<Vec<_>, _> = table
        .cols
        .iter()
        .map(|field_array| {
            let col_array = &field_array.array;
            let result_array = match (
                Value::Array(Arc::new(array.clone())),
                Value::Array(Arc::new(col_array.clone())),
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
                    from: "array-table broadcasting",
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

/// Helper function for Array-SuperTable broadcasting - broadcast array to each table batch
#[cfg(feature = "chunked")]
pub fn broadcast_array_to_supertable(
    op: ArithmeticOperator,
    array: &Array,
    super_table: &SuperTable,
) -> Result<SuperTable, MinarrowError> {
    let new_tables: Result<Vec<_>, _> = super_table
        .batches
        .iter()
        .map(|table| broadcast_array_to_table(op, array, table).map(Arc::new))
        .collect();
    Ok(SuperTable::from_batches(
        new_tables?,
        Some(super_table.name.clone()),
    ))
}

/// Helper function for Array-Cube broadcasting - broadcast array to each table in cube
#[cfg(feature = "cube")]
pub fn broadcast_array_to_cube(
    op: ArithmeticOperator,
    array: &Array,
    cube: &Cube,
) -> Result<Cube, MinarrowError> {
    let mut result_tables = Vec::with_capacity(cube.tables.len());
    for table in &cube.tables {
        let broadcasted = broadcast_array_to_table(op, array, table)?;
        result_tables.push(broadcasted);
    }
    Ok(Cube::from_tables(
        result_tables,
        cube.name.clone(),
        cube.third_dim_index.clone(),
    ))
}

/// Helper function for Array-Tuple2 broadcasting - broadcast array to each tuple element
pub fn broadcast_array_to_tuple2(
    op: ArithmeticOperator,
    array: &Array,
    tuple: (Arc<Value>, Arc<Value>),
) -> Result<(Arc<Value>, Arc<Value>), MinarrowError> {
    let res1 = broadcast_value(
        op,
        Value::Array(Arc::new(array.clone())),
        Arc::unwrap_or_clone(tuple.0),
    )?;
    let res2 = broadcast_value(
        op,
        Value::Array(Arc::new(array.clone())),
        Arc::unwrap_or_clone(tuple.1),
    )?;
    Ok((Arc::new(res1), Arc::new(res2)))
}

/// Helper function for Array-Tuple3 broadcasting
pub fn broadcast_array_to_tuple3(
    op: ArithmeticOperator,
    array: &Array,
    tuple: (Arc<Value>, Arc<Value>, Arc<Value>),
) -> Result<(Arc<Value>, Arc<Value>, Arc<Value>), MinarrowError> {
    let res1 = broadcast_value(
        op,
        Value::Array(Arc::new(array.clone())),
        Arc::unwrap_or_clone(tuple.0),
    )?;
    let res2 = broadcast_value(
        op,
        Value::Array(Arc::new(array.clone())),
        Arc::unwrap_or_clone(tuple.1),
    )?;
    let res3 = broadcast_value(
        op,
        Value::Array(Arc::new(array.clone())),
        Arc::unwrap_or_clone(tuple.2),
    )?;
    Ok((Arc::new(res1), Arc::new(res2), Arc::new(res3)))
}

/// Helper function for Array-Tuple4 broadcasting
pub fn broadcast_array_to_tuple4(
    op: ArithmeticOperator,
    array: &Array,
    tuple: (Arc<Value>, Arc<Value>, Arc<Value>, Arc<Value>),
) -> Result<(Arc<Value>, Arc<Value>, Arc<Value>, Arc<Value>), MinarrowError> {
    let res1 = broadcast_value(
        op,
        Value::Array(Arc::new(array.clone())),
        Arc::unwrap_or_clone(tuple.0),
    )?;
    let res2 = broadcast_value(
        op,
        Value::Array(Arc::new(array.clone())),
        Arc::unwrap_or_clone(tuple.1),
    )?;
    let res3 = broadcast_value(
        op,
        Value::Array(Arc::new(array.clone())),
        Arc::unwrap_or_clone(tuple.2),
    )?;
    let res4 = broadcast_value(
        op,
        Value::Array(Arc::new(array.clone())),
        Arc::unwrap_or_clone(tuple.3),
    )?;
    Ok((
        Arc::new(res1),
        Arc::new(res2),
        Arc::new(res3),
        Arc::new(res4),
    ))
}

/// Helper function for Array-Tuple5 broadcasting
pub fn broadcast_array_to_tuple5(
    op: ArithmeticOperator,
    array: &Array,
    tuple: (Arc<Value>, Arc<Value>, Arc<Value>, Arc<Value>, Arc<Value>),
) -> Result<(Arc<Value>, Arc<Value>, Arc<Value>, Arc<Value>, Arc<Value>), MinarrowError> {
    let res1 = broadcast_value(
        op,
        Value::Array(Arc::new(array.clone())),
        Arc::unwrap_or_clone(tuple.0),
    )?;
    let res2 = broadcast_value(
        op,
        Value::Array(Arc::new(array.clone())),
        Arc::unwrap_or_clone(tuple.1),
    )?;
    let res3 = broadcast_value(
        op,
        Value::Array(Arc::new(array.clone())),
        Arc::unwrap_or_clone(tuple.2),
    )?;
    let res4 = broadcast_value(
        op,
        Value::Array(Arc::new(array.clone())),
        Arc::unwrap_or_clone(tuple.3),
    )?;
    let res5 = broadcast_value(
        op,
        Value::Array(Arc::new(array.clone())),
        Arc::unwrap_or_clone(tuple.4),
    )?;
    Ok((
        Arc::new(res1),
        Arc::new(res2),
        Arc::new(res3),
        Arc::new(res4),
        Arc::new(res5),
    ))
}

/// Helper function for Array-Tuple6 broadcasting
pub fn broadcast_array_to_tuple6(
    op: ArithmeticOperator,
    array: &Array,
    tuple: (
        Arc<Value>,
        Arc<Value>,
        Arc<Value>,
        Arc<Value>,
        Arc<Value>,
        Arc<Value>,
    ),
) -> Result<
    (
        Arc<Value>,
        Arc<Value>,
        Arc<Value>,
        Arc<Value>,
        Arc<Value>,
        Arc<Value>,
    ),
    MinarrowError,
> {
    let res1 = broadcast_value(
        op,
        Value::Array(Arc::new(array.clone())),
        Arc::unwrap_or_clone(tuple.0),
    )?;
    let res2 = broadcast_value(
        op,
        Value::Array(Arc::new(array.clone())),
        Arc::unwrap_or_clone(tuple.1),
    )?;
    let res3 = broadcast_value(
        op,
        Value::Array(Arc::new(array.clone())),
        Arc::unwrap_or_clone(tuple.2),
    )?;
    let res4 = broadcast_value(
        op,
        Value::Array(Arc::new(array.clone())),
        Arc::unwrap_or_clone(tuple.3),
    )?;
    let res5 = broadcast_value(
        op,
        Value::Array(Arc::new(array.clone())),
        Arc::unwrap_or_clone(tuple.4),
    )?;
    let res6 = broadcast_value(
        op,
        Value::Array(Arc::new(array.clone())),
        Arc::unwrap_or_clone(tuple.5),
    )?;
    Ok((
        Arc::new(res1),
        Arc::new(res2),
        Arc::new(res3),
        Arc::new(res4),
        Arc::new(res5),
        Arc::new(res6),
    ))
}

/// Helper function for Array-SuperTableView broadcasting - create aligned array views for each table slice
#[cfg(all(feature = "chunked", feature = "views"))]
pub fn broadcast_array_to_supertableview(
    op: ArithmeticOperator,
    array: &Array,
    super_table_view: &SuperTableV,
) -> Result<SuperTableV, MinarrowError> {
    let mut current_offset = 0;
    let mut result_slices = Vec::new();

    for table_slice in super_table_view.slices.iter() {
        // Create an array view that matches this table slice's size

        use crate::TableV;
        let array_view = ArrayV::new(array.clone(), current_offset, table_slice.len);

        // Broadcast the aligned array view with this table slice
        let slice_result_table = broadcast_arrayview_to_tableview(op, &array_view, table_slice)?;
        let n_rows = slice_result_table.n_rows;
        result_slices.push(TableV::from_table(slice_result_table, 0, n_rows));
        current_offset += table_slice.len;
    }

    Ok(SuperTableV {
        slices: result_slices,
        len: super_table_view.len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::arrow_dtype::ArrowType;
    use crate::{Array, Field, IntegerArray, NumericArray, vec64};

    #[test]
    fn test_broadcast_array_add() {
        // Array + Array: [1, 2, 3] + [4, 5, 6] = [5, 7, 9]
        let arr1 = Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3]));
        let arr2 = Array::from_int32(IntegerArray::from_slice(&vec64![4, 5, 6]));

        let result = broadcast_array_add(arr1, arr2, None).unwrap();

        if let Array::NumericArray(NumericArray::Int32(arr)) = result {
            assert_eq!(arr.data.as_slice(), &[5, 7, 9]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[test]
    fn test_broadcast_array_add_scalar_expansion() {
        // Scalar expansion: [1, 2, 3] + [10] = [11, 12, 13]
        let arr1 = Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3]));
        let arr2 = Array::from_int32(IntegerArray::from_slice(&vec64![10]));

        let result = broadcast_array_add(arr1, arr2, None).unwrap();

        if let Array::NumericArray(NumericArray::Int32(arr)) = result {
            assert_eq!(arr.data.as_slice(), &[11, 12, 13]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[test]
    fn test_broadcast_array_sub() {
        // Array - Array: [10, 20, 30] - [1, 2, 3] = [9, 18, 27]
        let arr1 = Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30]));
        let arr2 = Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3]));

        let result = broadcast_array_sub(arr1, arr2, None).unwrap();

        if let Array::NumericArray(NumericArray::Int32(arr)) = result {
            assert_eq!(arr.data.as_slice(), &[9, 18, 27]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[test]
    fn test_broadcast_array_mul() {
        // Array * Array: [2, 3, 4] * [5, 6, 7] = [10, 18, 28]
        let arr1 = Array::from_int32(IntegerArray::from_slice(&vec64![2, 3, 4]));
        let arr2 = Array::from_int32(IntegerArray::from_slice(&vec64![5, 6, 7]));

        let result = broadcast_array_mul(arr1, arr2, None).unwrap();

        if let Array::NumericArray(NumericArray::Int32(arr)) = result {
            assert_eq!(arr.data.as_slice(), &[10, 18, 28]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[test]
    fn test_broadcast_array_div() {
        // Array / Array: [100, 200, 300] / [10, 20, 30] = [10, 10, 10]
        let arr1 = Array::from_int32(IntegerArray::from_slice(&vec64![100, 200, 300]));
        let arr2 = Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30]));

        let result = broadcast_array_div(arr1, arr2, None).unwrap();

        if let Array::NumericArray(NumericArray::Int32(arr)) = result {
            assert_eq!(arr.data.as_slice(), &[10, 10, 10]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[test]
    fn test_broadcast_array_to_table() {
        // Broadcast array [1, 2, 3] to a 2-column table
        let arr = Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3]));

        let table_arr1 = Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30]));
        let table_arr2 = Array::from_int32(IntegerArray::from_slice(&vec64![100, 200, 300]));
        let table = Table::build(
            vec![
                FieldArray::new(
                    Field::new("col1".to_string(), ArrowType::Int32, false, None),
                    table_arr1,
                ),
                FieldArray::new(
                    Field::new("col2".to_string(), ArrowType::Int32, false, None),
                    table_arr2,
                ),
            ],
            3,
            "test".to_string(),
        );

        let result = broadcast_array_to_table(ArithmeticOperator::Add, &arr, &table).unwrap();

        assert_eq!(result.n_rows, 3);
        assert_eq!(result.n_cols(), 2);

        // col1: [10,20,30] + [1,2,3] = [11,22,33]
        if let Array::NumericArray(NumericArray::Int32(arr)) = &result.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[11, 22, 33]);
        } else {
            panic!("Expected Int32 array in col1");
        }

        // col2: [100,200,300] + [1,2,3] = [101,202,303]
        if let Array::NumericArray(NumericArray::Int32(arr)) = &result.cols[1].array {
            assert_eq!(arr.data.as_slice(), &[101, 202, 303]);
        } else {
            panic!("Expected Int32 array in col2");
        }
    }

    #[test]
    fn test_broadcast_array_to_table_multiply() {
        // Broadcast array [2, 3, 4] to table with multiply operation
        let arr = Array::from_int32(IntegerArray::from_slice(&vec64![2, 3, 4]));

        let table_arr = Array::from_int32(IntegerArray::from_slice(&vec64![10, 10, 10]));
        let table = Table::build(
            vec![FieldArray::new(
                Field::new("col1".to_string(), ArrowType::Int32, false, None),
                table_arr,
            )],
            3,
            "test".to_string(),
        );

        let result = broadcast_array_to_table(ArithmeticOperator::Multiply, &arr, &table).unwrap();

        // [10,10,10] * [2,3,4] = [20,30,40]
        if let Array::NumericArray(NumericArray::Int32(arr)) = &result.cols[0].array {
            assert_eq!(arr.data.as_slice(), &[20, 30, 40]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[cfg(all(feature = "chunked", feature = "views"))]
    #[test]
    fn test_broadcast_array_to_supertableview() {
        use crate::{SuperTableV, TableV};

        // Create array: [1, 2, 3, 4, 5, 6]
        let arr = Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3, 4, 5, 6]));

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

        let result =
            broadcast_array_to_supertableview(ArithmeticOperator::Add, &arr, &super_table_view)
                .unwrap();

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

    #[cfg(feature = "scalar_type")]
    #[test]
    fn test_broadcast_array_to_scalar() {
        use crate::Scalar;

        // Test array [10, 20, 30] * scalar 2 = [20, 40, 60]
        let arr = Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30]));
        let scalar = Scalar::Int32(2);

        let result =
            broadcast_array_to_scalar(ArithmeticOperator::Multiply, &arr, &scalar).unwrap();

        if let Array::NumericArray(NumericArray::Int32(arr)) = result {
            assert_eq!(arr.data.as_slice(), &[20, 40, 60]);
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[cfg(feature = "chunked")]
    #[test]
    fn test_broadcast_array_to_supertable() {
        // Create array: [1, 2, 3]
        let arr = Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3]));

        // Create SuperTable with 2 batches
        let table1_arr = Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30]));
        let table1 = Table::build(
            vec![FieldArray::new(
                Field::new("col1".to_string(), ArrowType::Int32, false, None),
                table1_arr,
            )],
            3,
            "batch1".to_string(),
        );

        let table2_arr = Array::from_int32(IntegerArray::from_slice(&vec64![100, 200, 300]));
        let table2 = Table::build(
            vec![FieldArray::new(
                Field::new("col1".to_string(), ArrowType::Int32, false, None),
                table2_arr,
            )],
            3,
            "batch2".to_string(),
        );

        let super_table = SuperTable::from_batches(
            vec![Arc::new(table1), Arc::new(table2)],
            Some("test_super".to_string()),
        );

        let result =
            broadcast_array_to_supertable(ArithmeticOperator::Add, &arr, &super_table).unwrap();

        assert_eq!(result.batches.len(), 2);

        // First batch: [10,20,30] + [1,2,3] = [11,22,33]
        if let Array::NumericArray(NumericArray::Int32(arr)) = &result.batches[0].cols[0].array {
            assert_eq!(arr.data.as_slice(), &[11, 22, 33]);
        } else {
            panic!("Expected Int32 array in first batch");
        }

        // Second batch: [100,200,300] + [1,2,3] = [101,202,303]
        if let Array::NumericArray(NumericArray::Int32(arr)) = &result.batches[1].cols[0].array {
            assert_eq!(arr.data.as_slice(), &[101, 202, 303]);
        } else {
            panic!("Expected Int32 array in second batch");
        }
    }

    #[cfg(feature = "cube")]
    #[test]
    fn test_broadcast_array_to_cube() {
        use crate::Cube;

        // Create array: [1, 2, 3]
        let arr = Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3]));

        // Create Cube with 2 tables
        let table1_arr = Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30]));
        let table1 = Table::build(
            vec![FieldArray::new(
                Field::new("col1".to_string(), ArrowType::Int32, false, None),
                table1_arr,
            )],
            3,
            "table1".to_string(),
        );

        let table2_arr = Array::from_int32(IntegerArray::from_slice(&vec64![100, 200, 300]));
        let table2 = Table::build(
            vec![FieldArray::new(
                Field::new("col1".to_string(), ArrowType::Int32, false, None),
                table2_arr,
            )],
            3,
            "table2".to_string(),
        );

        let cube = Cube::from_tables(vec![table1, table2], "test_cube".to_string(), None);

        let result = broadcast_array_to_cube(ArithmeticOperator::Subtract, &arr, &cube).unwrap();

        assert_eq!(result.tables.len(), 2);
        assert_eq!(result.name, "test_cube");

        // First table: [1,2,3] - [10,20,30] = [-9,-18,-27]
        if let Array::NumericArray(NumericArray::Int32(arr)) = &result.tables[0].cols[0].array {
            assert_eq!(arr.data.as_slice(), &[-9, -18, -27]);
        } else {
            panic!("Expected Int32 array in first table");
        }

        // Second table: [1,2,3] - [100,200,300] = [-99,-198,-297]
        if let Array::NumericArray(NumericArray::Int32(arr)) = &result.tables[1].cols[0].array {
            assert_eq!(arr.data.as_slice(), &[-99, -198, -297]);
        } else {
            panic!("Expected Int32 array in second table");
        }
    }

    #[test]
    fn test_broadcast_array_to_tuple2() {
        // Test array [5, 10, 15] with tuple ([1,2,3], [10,20,30])
        let arr = Array::from_int32(IntegerArray::from_slice(&vec64![5, 10, 15]));
        let arr1 = Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3]));
        let arr2 = Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30]));

        let tuple = (
            Arc::new(Value::Array(Arc::new(arr1))),
            Arc::new(Value::Array(Arc::new(arr2))),
        );

        let result = broadcast_array_to_tuple2(ArithmeticOperator::Add, &arr, tuple).unwrap();

        // First element: [1,2,3] + [5,10,15] = [6,12,18]
        if let Value::Array(arc_arr) = &*result.0 {
            if let Array::NumericArray(NumericArray::Int32(arr)) = arc_arr.as_ref() {
                assert_eq!(arr.data.as_slice(), &[6, 12, 18]);
            } else {
                panic!("Expected Int32 array in first element");
            }
        } else {
            panic!("Expected Array value");
        }

        // Second element: [10,20,30] + [5,10,15] = [15,30,45]
        if let Value::Array(arc_arr) = &*result.1 {
            if let Array::NumericArray(NumericArray::Int32(arr)) = arc_arr.as_ref() {
                assert_eq!(arr.data.as_slice(), &[15, 30, 45]);
            } else {
                panic!("Expected Int32 array in second element");
            }
        } else {
            panic!("Expected Array value");
        }
    }

    #[test]
    fn test_broadcast_array_to_tuple3() {
        // Test array [2, 3, 4] * tuple
        let arr = Array::from_int32(IntegerArray::from_slice(&vec64![2, 3, 4]));
        let arr1 = Array::from_int32(IntegerArray::from_slice(&vec64![10, 10, 10]));
        let arr2 = Array::from_int32(IntegerArray::from_slice(&vec64![5, 5, 5]));
        let arr3 = Array::from_int32(IntegerArray::from_slice(&vec64![1, 1, 1]));

        let tuple = (
            Arc::new(Value::Array(Arc::new(arr1))),
            Arc::new(Value::Array(Arc::new(arr2))),
            Arc::new(Value::Array(Arc::new(arr3))),
        );

        let result = broadcast_array_to_tuple3(ArithmeticOperator::Multiply, &arr, tuple).unwrap();

        // [10,10,10] * [2,3,4] = [20,30,40]
        if let Value::Array(arc_arr) = &*result.0 {
            if let Array::NumericArray(NumericArray::Int32(arr)) = arc_arr.as_ref() {
                assert_eq!(arr.data.as_slice(), &[20, 30, 40]);
            } else {
                panic!("Expected Int32 array in first element");
            }
        } else {
            panic!("Expected Array value");
        }

        // [5,5,5] * [2,3,4] = [10,15,20]
        if let Value::Array(arc_arr) = &*result.1 {
            if let Array::NumericArray(NumericArray::Int32(arr)) = arc_arr.as_ref() {
                assert_eq!(arr.data.as_slice(), &[10, 15, 20]);
            } else {
                panic!("Expected Int32 array in second element");
            }
        } else {
            panic!("Expected Array value");
        }

        // [1,1,1] * [2,3,4] = [2,3,4]
        if let Value::Array(arc_arr) = &*result.2 {
            if let Array::NumericArray(NumericArray::Int32(arr)) = arc_arr.as_ref() {
                assert_eq!(arr.data.as_slice(), &[2, 3, 4]);
            } else {
                panic!("Expected Int32 array in third element");
            }
        } else {
            panic!("Expected Array value");
        }
    }

    #[test]
    fn test_broadcast_array_to_tuple4() {
        // Test array [1, 1, 1] + tuple of 4 elements
        let arr = Array::from_int32(IntegerArray::from_slice(&vec64![1, 1, 1]));
        let arr1 = Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30]));
        let arr2 = Array::from_int32(IntegerArray::from_slice(&vec64![100, 200, 300]));
        let arr3 = Array::from_int32(IntegerArray::from_slice(&vec64![5, 10, 15]));
        let arr4 = Array::from_int32(IntegerArray::from_slice(&vec64![2, 4, 6]));

        let tuple = (
            Arc::new(Value::Array(Arc::new(arr1))),
            Arc::new(Value::Array(Arc::new(arr2))),
            Arc::new(Value::Array(Arc::new(arr3))),
            Arc::new(Value::Array(Arc::new(arr4))),
        );

        let result = broadcast_array_to_tuple4(ArithmeticOperator::Add, &arr, tuple).unwrap();

        // Verify all 4 elements
        if let Value::Array(arc_arr) = &*result.0 {
            if let Array::NumericArray(NumericArray::Int32(arr)) = arc_arr.as_ref() {
                assert_eq!(arr.data.as_slice(), &[11, 21, 31]);
            } else {
                panic!("Expected Int32 array in element 0");
            }
        } else {
            panic!("Expected Array value");
        }

        if let Value::Array(arc_arr) = &*result.1 {
            if let Array::NumericArray(NumericArray::Int32(arr)) = arc_arr.as_ref() {
                assert_eq!(arr.data.as_slice(), &[101, 201, 301]);
            } else {
                panic!("Expected Int32 array in element 1");
            }
        } else {
            panic!("Expected Array value");
        }

        if let Value::Array(arc_arr) = &*result.2 {
            if let Array::NumericArray(NumericArray::Int32(arr)) = arc_arr.as_ref() {
                assert_eq!(arr.data.as_slice(), &[6, 11, 16]);
            } else {
                panic!("Expected Int32 array in element 2");
            }
        } else {
            panic!("Expected Array value");
        }

        if let Value::Array(arc_arr) = &*result.3 {
            if let Array::NumericArray(NumericArray::Int32(arr)) = arc_arr.as_ref() {
                assert_eq!(arr.data.as_slice(), &[3, 5, 7]);
            } else {
                panic!("Expected Int32 array in element 3");
            }
        } else {
            panic!("Expected Array value");
        }
    }

    #[test]
    fn test_broadcast_array_to_tuple5() {
        // Test array [10, 10, 10] * tuple of 5 elements
        let arr = Array::from_int32(IntegerArray::from_slice(&vec64![10, 10, 10]));
        let tuple = (
            Arc::new(Value::Array(Arc::new(Array::from_int32(
                IntegerArray::from_slice(&vec64![1, 2, 3]),
            )))),
            Arc::new(Value::Array(Arc::new(Array::from_int32(
                IntegerArray::from_slice(&vec64![2, 3, 4]),
            )))),
            Arc::new(Value::Array(Arc::new(Array::from_int32(
                IntegerArray::from_slice(&vec64![3, 4, 5]),
            )))),
            Arc::new(Value::Array(Arc::new(Array::from_int32(
                IntegerArray::from_slice(&vec64![4, 5, 6]),
            )))),
            Arc::new(Value::Array(Arc::new(Array::from_int32(
                IntegerArray::from_slice(&vec64![5, 6, 7]),
            )))),
        );

        let result = broadcast_array_to_tuple5(ArithmeticOperator::Multiply, &arr, tuple).unwrap();

        // [1,2,3] * [10,10,10] = [10,20,30]
        if let Value::Array(arc_arr) = &*result.0 {
            if let Array::NumericArray(NumericArray::Int32(arr)) = arc_arr.as_ref() {
                assert_eq!(arr.data.as_slice(), &[10, 20, 30]);
            } else {
                panic!("Expected Int32 array in element 0");
            }
        } else {
            panic!("Expected Array value");
        }

        // [5,6,7] * [10,10,10] = [50,60,70]
        if let Value::Array(arc_arr) = &*result.4 {
            if let Array::NumericArray(NumericArray::Int32(arr)) = arc_arr.as_ref() {
                assert_eq!(arr.data.as_slice(), &[50, 60, 70]);
            } else {
                panic!("Expected Int32 array in element 4");
            }
        } else {
            panic!("Expected Array value");
        }
    }

    #[test]
    fn test_broadcast_array_to_tuple6() {
        // Test array [5, 5, 5] - tuple of 6 elements
        let arr = Array::from_int32(IntegerArray::from_slice(&vec64![5, 5, 5]));
        let tuple = (
            Arc::new(Value::Array(Arc::new(Array::from_int32(
                IntegerArray::from_slice(&vec64![10, 10, 10]),
            )))),
            Arc::new(Value::Array(Arc::new(Array::from_int32(
                IntegerArray::from_slice(&vec64![20, 20, 20]),
            )))),
            Arc::new(Value::Array(Arc::new(Array::from_int32(
                IntegerArray::from_slice(&vec64![15, 15, 15]),
            )))),
            Arc::new(Value::Array(Arc::new(Array::from_int32(
                IntegerArray::from_slice(&vec64![8, 8, 8]),
            )))),
            Arc::new(Value::Array(Arc::new(Array::from_int32(
                IntegerArray::from_slice(&vec64![12, 12, 12]),
            )))),
            Arc::new(Value::Array(Arc::new(Array::from_int32(
                IntegerArray::from_slice(&vec64![6, 6, 6]),
            )))),
        );

        let result = broadcast_array_to_tuple6(ArithmeticOperator::Subtract, &arr, tuple).unwrap();

        // [5,5,5] - [10,10,10] = [-5,-5,-5]
        if let Value::Array(arc_arr) = &*result.0 {
            if let Array::NumericArray(NumericArray::Int32(arr)) = arc_arr.as_ref() {
                assert_eq!(arr.data.as_slice(), &[-5, -5, -5]);
            } else {
                panic!("Expected Int32 array in element 0");
            }
        } else {
            panic!("Expected Array value");
        }

        // [5,5,5] - [6,6,6] = [-1,-1,-1]
        if let Value::Array(arc_arr) = &*result.5 {
            if let Array::NumericArray(NumericArray::Int32(arr)) = arc_arr.as_ref() {
                assert_eq!(arr.data.as_slice(), &[-1, -1, -1]);
            } else {
                panic!("Expected Int32 array in element 5");
            }
        } else {
            panic!("Expected Array value");
        }
    }
}
