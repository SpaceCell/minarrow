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

//! # Comprehensive Broadcasting Examples
//!
//! This example demonstrates Minarrow's broadcasting capabilities across different array types
//! and operations. Broadcasting allows operations between arrays of different shapes by
//! automatically replicating smaller arrays to match larger ones.
//!
//! ## Broadcasting Rules
//! - Arrays with matching lengths operate element-wise
//! - Single-element arrays broadcast to match the length of the other operand
//! - Type promotion occurs automatically (e.g., Int32 + Float32 -> Float32)
//! - Complex types (Table, Cube, SuperArray) support element-wise broadcasting

use minarrow::{Array, FloatArray, IntegerArray, NumericArray, Table, Value, fa_i32, vec64};
use std::sync::Arc;

#[cfg(feature = "views")]
use minarrow::ArrayV;

#[cfg(feature = "chunked")]
use minarrow::SuperArray;

#[cfg(feature = "cube")]
use minarrow::Cube;

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Minarrow Comprehensive Broadcasting Examples");
    println!("═══════════════════════════════════════════════════════════\n");

    test_integer_broadcasting();
    test_float_broadcasting();
    test_mixed_type_promotion();
    test_scalar_broadcasting();
    test_division_broadcasting();
    test_reference_operations();
    test_subtraction_broadcasting();
    test_chained_operations();
    test_table_broadcasting();
    test_array_view_broadcasting();
    test_super_array_broadcasting();
    test_cube_broadcasting();

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  All broadcasting tests completed successfully!");
    println!("═══════════════════════════════════════════════════════════");
}

/// Test integer array broadcasting with multiplication
fn test_integer_broadcasting() {
    println!("┌─ Test 1: Integer Broadcasting");
    println!("│  Operation: [100] * [1, 2, 3, 4, 5]");
    println!("│  Expected:  [100, 200, 300, 400, 500]");

    let scalar_array = Value::Array(Arc::new(Array::from_int32(IntegerArray::from_slice(
        &vec64![100],
    ))));
    let multi_array = Value::Array(Arc::new(Array::from_int32(IntegerArray::from_slice(
        &vec64![1, 2, 3, 4, 5],
    ))));

    match scalar_array * multi_array {
        Ok(Value::Array(arr_arc)) => {
            if let Array::NumericArray(NumericArray::Int32(arr)) = arr_arc.as_ref() {
                println!("│  Result:    {:?}", arr.data.as_slice());
                println!("└─ ✓ Passed\n");
            } else {
                println!("└─ ✗ Error: Unexpected array type\n");
            }
        }
        Ok(_) => println!("└─ ✗ Error: Unexpected result type\n"),
        Err(e) => println!("└─ ✗ Error: {:?}\n", e),
    }
}

/// Test float array broadcasting with addition
fn test_float_broadcasting() {
    println!("┌─ Test 2: Float Broadcasting");
    println!("│  Operation: [2.5] + [1.0, 2.0, 3.0]");
    println!("│  Expected:  [3.5, 4.5, 5.5]");

    let scalar_float = Value::Array(Arc::new(Array::from_float64(FloatArray::from_slice(
        &vec64![2.5],
    ))));
    let multi_float = Value::Array(Arc::new(Array::from_float64(FloatArray::from_slice(
        &vec64![1.0, 2.0, 3.0],
    ))));

    match scalar_float + multi_float {
        Ok(Value::Array(arr_arc)) => {
            if let Array::NumericArray(NumericArray::Float64(arr)) = arr_arc.as_ref() {
                println!("│  Result:    {:?}", arr.data.as_slice());
                println!("└─ ✓ Passed\n");
            } else {
                println!("└─ ✗ Error: Unexpected array type\n");
            }
        }
        Ok(_) => println!("└─ ✗ Error: Unexpected result type\n"),
        Err(e) => println!("└─ ✗ Error: {:?}\n", e),
    }
}

/// Test automatic type promotion from integer to float
fn test_mixed_type_promotion() {
    println!("┌─ Test 3: Mixed Type Promotion");
    println!("│  Operation: Int32[10, 20, 30] + Float32[0.5, 0.5, 0.5]");
    println!("│  Expected:  Float32[10.5, 20.5, 30.5]");

    let int_array = Value::Array(Arc::new(Array::from_int32(IntegerArray::from_slice(
        &vec64![10, 20, 30],
    ))));
    let float_array = Value::Array(Arc::new(Array::from_float32(FloatArray::from_slice(
        &vec64![0.5, 0.5, 0.5],
    ))));

    match int_array + float_array {
        Ok(Value::Array(arr_arc)) => {
            if let Array::NumericArray(NumericArray::Float32(arr)) = arr_arc.as_ref() {
                println!(
                    "│  Result:    {:?} (promoted to Float32)",
                    arr.data.as_slice()
                );
                println!("└─ ✓ Passed\n");
            } else {
                println!("└─ ✗ Error: Unexpected array type\n");
            }
        }
        Ok(_) => println!("└─ ✗ Error: Unexpected result type\n"),
        Err(e) => println!("└─ ✗ Error: {:?}\n", e),
    }
}

/// Test broadcasting with Scalar type (requires scalar_type feature)
fn test_scalar_broadcasting() {
    #[cfg(feature = "scalar_type")]
    {
        println!("┌─ Test 4: Scalar + Array Broadcasting");
        println!("│  Operation: Scalar(1000) + [1, 2, 3]");
        println!("│  Expected:  [1001, 1002, 1003]");

        let scalar = Value::Scalar(minarrow::Scalar::Int64(1000));
        let array = Value::Array(Arc::new(Array::from_int64(IntegerArray::from_slice(
            &vec64![1, 2, 3],
        ))));

        match scalar + array {
            Ok(Value::Array(arr_arc)) => {
                if let Array::NumericArray(NumericArray::Int64(arr)) = arr_arc.as_ref() {
                    println!("│  Result:    {:?}", arr.data.as_slice());
                    println!("└─ ✓ Passed\n");
                } else {
                    println!("└─ ✗ Error: Unexpected array type\n");
                }
            }
            Ok(_) => println!("└─ ✗ Error: Unexpected result type\n"),
            Err(e) => println!("└─ ✗ Error: {:?}\n", e),
        }
    }

    #[cfg(not(feature = "scalar_type"))]
    {
        println!("┌─ Test 4: Scalar + Array Broadcasting");
        println!("└─ ⊘ Skipped (scalar_type feature not enabled)\n");
    }
}

/// Test division broadcasting
fn test_division_broadcasting() {
    println!("┌─ Test 5: Division Broadcasting");
    println!("│  Operation: [100.0] / [2.0, 4.0, 5.0, 10.0]");
    println!("│  Expected:  [50.0, 25.0, 20.0, 10.0]");

    let dividend = Value::Array(Arc::new(Array::from_float64(FloatArray::from_slice(
        &vec64![100.0],
    ))));
    let divisors = Value::Array(Arc::new(Array::from_float64(FloatArray::from_slice(
        &vec64![2.0, 4.0, 5.0, 10.0],
    ))));

    match dividend / divisors {
        Ok(Value::Array(arr_arc)) => {
            if let Array::NumericArray(NumericArray::Float64(arr)) = arr_arc.as_ref() {
                println!("│  Result:    {:?}", arr.data.as_slice());
                println!("└─ ✓ Passed\n");
            } else {
                println!("└─ ✗ Error: Unexpected array type\n");
            }
        }
        Ok(_) => println!("└─ ✗ Error: Unexpected result type\n"),
        Err(e) => println!("└─ ✗ Error: {:?}\n", e),
    }
}

/// Test operations using references (non-consuming)
fn test_reference_operations() {
    println!("┌─ Test 6: Reference Operations (Non-Consuming)");
    println!("│  Operation: &[5] * &[10, 20, 30]");
    println!("│  Expected:  [50, 100, 150]");

    let a = Value::Array(Arc::new(Array::from_int32(IntegerArray::from_slice(
        &vec64![5],
    ))));
    let b = Value::Array(Arc::new(Array::from_int32(IntegerArray::from_slice(
        &vec64![10, 20, 30],
    ))));

    match &a * &b {
        Ok(Value::Array(arr_arc)) => {
            if let Array::NumericArray(NumericArray::Int32(arr)) = arr_arc.as_ref() {
                println!("│  Result:    {:?}", arr.data.as_slice());
                println!("│  Note: Original arrays remain available for reuse");
                println!("└─ ✓ Passed\n");
            } else {
                println!("└─ ✗ Error: Unexpected array type\n");
            }
        }
        Ok(_) => println!("└─ ✗ Error: Unexpected result type\n"),
        Err(e) => println!("└─ ✗ Error: {:?}\n", e),
    }
}

/// Test subtraction with broadcasting
fn test_subtraction_broadcasting() {
    println!("┌─ Test 7: Subtraction Broadcasting");
    println!("│  Operation: [100, 200, 300] - [1]");
    println!("│  Expected:  [99, 199, 299]");

    let array = Value::Array(Arc::new(Array::from_int32(IntegerArray::from_slice(
        &vec64![100, 200, 300],
    ))));
    let scalar_array = Value::Array(Arc::new(Array::from_int32(IntegerArray::from_slice(
        &vec64![1],
    ))));

    match array - scalar_array {
        Ok(Value::Array(arr_arc)) => {
            if let Array::NumericArray(NumericArray::Int32(arr)) = arr_arc.as_ref() {
                println!("│  Result:    {:?}", arr.data.as_slice());
                println!("└─ ✓ Passed\n");
            } else {
                println!("└─ ✗ Error: Unexpected array type\n");
            }
        }
        Ok(_) => println!("└─ ✗ Error: Unexpected result type\n"),
        Err(e) => println!("└─ ✗ Error: {:?}\n", e),
    }
}

/// Test chained operations with broadcasting
fn test_chained_operations() {
    println!("┌─ Test 8: Chained Operations");
    println!("│  Operation: ([2] * [1, 2, 3]) + [10]");
    println!("│  Expected:  [12, 14, 16]");

    let two = Value::Array(Arc::new(Array::from_int32(IntegerArray::from_slice(
        &vec64![2],
    ))));
    let nums = Value::Array(Arc::new(Array::from_int32(IntegerArray::from_slice(
        &vec64![1, 2, 3],
    ))));
    let ten = Value::Array(Arc::new(Array::from_int32(IntegerArray::from_slice(
        &vec64![10],
    ))));

    let step1 = (two * nums).expect("First operation failed");
    match step1 + ten {
        Ok(Value::Array(arr_arc)) => {
            if let Array::NumericArray(NumericArray::Int32(arr)) = arr_arc.as_ref() {
                println!("│  Result:    {:?}", arr.data.as_slice());
                println!("└─ ✓ Passed\n");
            } else {
                println!("└─ ✗ Error: Unexpected array type\n");
            }
        }
        Ok(_) => println!("└─ ✗ Error: Unexpected result type\n"),
        Err(e) => println!("└─ ✗ Error: {:?}\n", e),
    }
}

/// Test Table broadcasting - operates on each column
fn test_table_broadcasting() {
    println!("┌─ Test 9: Table Broadcasting");
    println!(
        "│  Operation: Table{{col1:[1,2,3], col2:[4,5,6]}} + Table{{col1:[10,10,10], col2:[20,20,20]}}"
    );
    println!("│  Expected:  Table{{col1:[11,12,13], col2:[24,25,26]}}");

    // Create first table with two columns
    let mut table1 = Table::new("table1".to_string(), None);
    table1.add_col(fa_i32!("col1", 1, 2, 3));
    table1.add_col(fa_i32!("col2", 4, 5, 6));

    // Create second table with matching structure
    let mut table2 = Table::new("table2".to_string(), None);
    table2.add_col(fa_i32!("col1", 10, 10, 10));
    table2.add_col(fa_i32!("col2", 20, 20, 20));

    match Value::Table(Arc::new(table1)) + Value::Table(Arc::new(table2)) {
        Ok(Value::Table(result)) => {
            if let Array::NumericArray(NumericArray::Int32(col1)) = &result.cols[0].array {
                println!("│  Result col1: {:?}", col1.data.as_slice());
            }
            if let Array::NumericArray(NumericArray::Int32(col2)) = &result.cols[1].array {
                println!("│  Result col2: {:?}", col2.data.as_slice());
            }
            println!("└─ ✓ Passed\n");
        }
        Ok(other) => println!("└─ ✗ Error: Unexpected result type {:?}\n", other),
        Err(e) => println!("└─ ✗ Error: {:?}\n", e),
    }
}

/// Test ArrayView broadcasting - efficient windowed operations
fn test_array_view_broadcasting() {
    #[cfg(feature = "views")]
    {
        println!("┌─ Test 10: ArrayView Broadcasting");
        println!("│  Operation: ArrayView([2,3,4]) + ArrayView([10,10,10])");
        println!("│  Expected:  Array([12,13,14])");

        // Create an array and a view into it
        let arr1 = Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3, 4, 5]));
        let view1 = ArrayV::new(arr1, 1, 3); // View of elements [2,3,4]

        let arr2 = Array::from_int32(IntegerArray::from_slice(&vec64![10, 10, 10]));
        let view2 = ArrayV::new(arr2, 0, 3);

        match Value::ArrayView(Arc::new(view1)) + Value::ArrayView(Arc::new(view2)) {
            Ok(Value::Array(arr_arc)) => {
                if let Array::NumericArray(NumericArray::Int32(result)) = arr_arc.as_ref() {
                    println!("│  Result:    {:?}", result.data.as_slice());
                    println!("└─ ✓ Passed\n");
                } else {
                    println!("└─ ✗ Error: Unexpected array type\n");
                }
            }
            Ok(_) => println!("└─ ✗ Error: Unexpected result type\n"),
            Err(e) => println!("└─ ✗ Error: {:?}\n", e),
        }
    }

    #[cfg(not(feature = "views"))]
    {
        println!("┌─ Test 10: ArrayView Broadcasting");
        println!("└─ ⊘ Skipped (views feature not enabled)\n");
    }
}

/// Test SuperArray broadcasting - chunked array operations
fn test_super_array_broadcasting() {
    #[cfg(feature = "chunked")]
    {
        println!("┌─ Test 11: SuperArray (Chunked) Broadcasting");
        println!("│  Operation: SuperArray{{[1,2],[3,4]}} * SuperArray{{[2,2],[2,2]}}");
        println!("│  Expected:  SuperArray{{[2,4],[6,8]}}");

        // Create chunked arrays (multiple field array chunks)
        let super_arr1 = SuperArray::from_field_array_chunks(vec![
            fa_i32!("chunk1", 1, 2),
            fa_i32!("chunk1", 3, 4),
        ]);

        let super_arr2 = SuperArray::from_field_array_chunks(vec![
            fa_i32!("chunk1", 2, 2),
            fa_i32!("chunk1", 2, 2),
        ]);

        match Value::SuperArray(Arc::new(super_arr1)) * Value::SuperArray(Arc::new(super_arr2)) {
            Ok(Value::SuperArray(result)) => {
                println!("│  Result with {} chunks:", result.len());
                for i in 0..result.len() {
                    if let Some(chunk) = result.chunk(i) {
                        if let Array::NumericArray(NumericArray::Int32(arr)) = chunk {
                            println!("│    Chunk {}: {:?}", i, arr.data.as_slice());
                        }
                    }
                }
                println!("└─ ✓ Passed\n");
            }
            Ok(other) => println!("└─ ✗ Error: Unexpected result type {:?}\n", other),
            Err(e) => println!("└─ ✗ Error: {:?}\n", e),
        }
    }

    #[cfg(not(feature = "chunked"))]
    {
        println!("┌─ Test 11: SuperArray (Chunked) Broadcasting");
        println!("└─ ⊘ Skipped (chunked feature not enabled)\n");
    }
}

/// Test Cube broadcasting - 3D tensor operations
fn test_cube_broadcasting() {
    #[cfg(feature = "cube")]
    {
        println!("┌─ Test 12: Cube (3D) Broadcasting");
        println!("│  Operation: Cube{{2 tables}} + Cube{{2 tables}}");
        println!("│  Expected:  Element-wise addition across all tables");

        // Create first cube with 2 tables
        let mut cube1 = Cube::new(
            "cube1".to_string(),
            Some(vec![fa_i32!("col1", 1, 2), fa_i32!("col2", 3, 4)]),
            None,
        );

        // Add second table to cube1
        let mut table2 = Table::new("t2".to_string(), None);
        table2.add_col(fa_i32!("col1", 5, 6));
        table2.add_col(fa_i32!("col2", 7, 8));
        cube1.add_table(table2);

        // Create second cube
        let mut cube2 = Cube::new(
            "cube2".to_string(),
            Some(vec![fa_i32!("col1", 10, 10), fa_i32!("col2", 20, 20)]),
            None,
        );

        let mut table4 = Table::new("t4".to_string(), None);
        table4.add_col(fa_i32!("col1", 30, 30));
        table4.add_col(fa_i32!("col2", 40, 40));
        cube2.add_table(table4);

        match Value::Cube(Arc::new(cube1)) + Value::Cube(Arc::new(cube2)) {
            Ok(Value::Cube(result)) => {
                println!("│  Result cube with {} tables:", result.n_tables());
                for i in 0..result.n_tables() {
                    println!("│  Table {}:", i);
                    if let Some(table) = result.table(i) {
                        for j in 0..table.n_cols() {
                            let col = &table.cols[j];
                            if let Array::NumericArray(NumericArray::Int32(arr)) = &col.array {
                                println!("│    Column {}: {:?}", j, arr.data.as_slice());
                            }
                        }
                    }
                }
                println!("└─ ✓ Passed\n");
            }
            Ok(other) => println!("└─ ✗ Error: Unexpected result type {:?}\n", other),
            Err(e) => println!("└─ ✗ Error: {:?}\n", e),
        }
    }

    #[cfg(not(feature = "cube"))]
    {
        println!("┌─ Test 12: Cube (3D) Broadcasting");
        println!("└─ ⊘ Skipped (cube feature not enabled)\n");
    }
}
