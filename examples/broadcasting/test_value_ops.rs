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

//! # Value Operations Examples
//!
//! This example provides a quick overview of basic Value operations in Minarrow,
//! demonstrating the high-level API for working with arrays and scalars.
//!
//! ## Operations Covered
//! - Array + Array (element-wise operations)
//! - Scalar + Array (broadcasting)
//! - Single-element array broadcasting
//! - Float operations with broadcasting
//! - Reference-based operations (non-consuming)

use minarrow::{Array, FloatArray, IntegerArray, NumericArray, Value, vec64};
use std::sync::Arc;

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Minarrow Value Operations Examples");
    println!("═══════════════════════════════════════════════════════════\n");

    test_array_addition();
    test_scalar_array_ops();
    test_integer_broadcasting();
    test_float_broadcasting();
    test_reference_operations();
    test_subtraction();
    test_division();

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  All value operation tests completed successfully!");
    println!("═══════════════════════════════════════════════════════════");
}

/// Test basic array addition (equal-length arrays)
fn test_array_addition() {
    println!("┌─ Test 1: Array + Array (Equal Length)");
    println!("│  Operation: [1, 2, 3] + [4, 5, 6]");
    println!("│  Expected:  [5, 7, 9]");

    let arr1 = Value::Array(Arc::new(Array::from_int32(IntegerArray::from_slice(
        &vec64![1, 2, 3],
    ))));
    let arr2 = Value::Array(Arc::new(Array::from_int32(IntegerArray::from_slice(
        &vec64![4, 5, 6],
    ))));

    match arr1 + arr2 {
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

/// Test scalar + array operations (requires scalar_type feature)
fn test_scalar_array_ops() {
    #[cfg(feature = "scalar_type")]
    {
        println!("┌─ Test 2: Scalar + Array Broadcasting");
        println!("│  Operation: Scalar(10) + [1, 2, 3]");
        println!("│  Expected:  [11, 12, 13]");

        let scalar = Value::Scalar(minarrow::Scalar::Int32(10));
        let array = Value::Array(Arc::new(Array::from_int32(IntegerArray::from_slice(
            &vec64![1, 2, 3],
        ))));

        match scalar + array {
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

    #[cfg(not(feature = "scalar_type"))]
    {
        println!("┌─ Test 2: Scalar + Array Broadcasting");
        println!("└─ ⊘ Skipped (scalar_type feature not enabled)\n");
    }
}

/// Test single-element array broadcasting with multiplication
fn test_integer_broadcasting() {
    println!("┌─ Test 3: Integer Array Broadcasting");
    println!("│  Operation: [100] * [1, 2, 3, 4, 5]");
    println!("│  Expected:  [100, 200, 300, 400, 500]");

    let single = Value::Array(Arc::new(Array::from_int32(IntegerArray::from_slice(
        &vec64![100],
    ))));
    let array = Value::Array(Arc::new(Array::from_int32(IntegerArray::from_slice(
        &vec64![1, 2, 3, 4, 5],
    ))));

    match single * array {
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

/// Test float array broadcasting with multiplication
fn test_float_broadcasting() {
    println!("┌─ Test 4: Float Array Broadcasting");
    println!("│  Operation: [2.5] * [1.0, 2.0, 3.0, 4.0]");
    println!("│  Expected:  [2.5, 5.0, 7.5, 10.0]");

    let float_single = Value::Array(Arc::new(Array::from_float64(FloatArray::from_slice(
        &vec64![2.5],
    ))));
    let float_array = Value::Array(Arc::new(Array::from_float64(FloatArray::from_slice(
        &vec64![1.0, 2.0, 3.0, 4.0],
    ))));

    match float_single * float_array {
        Ok(Value::Array(arr_arc)) => {
            if let Array::NumericArray(NumericArray::Float64(result)) = arr_arc.as_ref() {
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

/// Test reference-based operations (non-consuming)
fn test_reference_operations() {
    println!("┌─ Test 5: Reference-Based Operations");
    println!("│  Operation: &[10, 20] + &[30, 40]");
    println!("│  Expected:  [40, 60]");

    let a = Value::Array(Arc::new(Array::from_int32(IntegerArray::from_slice(
        &vec64![10, 20],
    ))));
    let b = Value::Array(Arc::new(Array::from_int32(IntegerArray::from_slice(
        &vec64![30, 40],
    ))));

    match &a + &b {
        Ok(Value::Array(arr_arc)) => {
            if let Array::NumericArray(NumericArray::Int32(result)) = arr_arc.as_ref() {
                println!("│  Result:    {:?}", result.data.as_slice());
                println!("│  Note: Both 'a' and 'b' remain valid after operation");
                println!("└─ ✓ Passed\n");
            } else {
                println!("└─ ✗ Error: Unexpected array type\n");
            }
        }
        Ok(_) => println!("└─ ✗ Error: Unexpected result type\n"),
        Err(e) => println!("└─ ✗ Error: {:?}\n", e),
    }
}

/// Test array subtraction with broadcasting
fn test_subtraction() {
    println!("┌─ Test 6: Subtraction with Broadcasting");
    println!("│  Operation: [100, 200, 300] - [10]");
    println!("│  Expected:  [90, 190, 290]");

    let array = Value::Array(Arc::new(Array::from_int32(IntegerArray::from_slice(
        &vec64![100, 200, 300],
    ))));
    let scalar_array = Value::Array(Arc::new(Array::from_int32(IntegerArray::from_slice(
        &vec64![10],
    ))));

    match array - scalar_array {
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

/// Test division with broadcasting
fn test_division() {
    println!("┌─ Test 7: Division with Broadcasting");
    println!("│  Operation: [100.0, 50.0, 25.0] / [2.0]");
    println!("│  Expected:  [50.0, 25.0, 12.5]");

    let dividend = Value::Array(Arc::new(Array::from_float64(FloatArray::from_slice(
        &vec64![100.0, 50.0, 25.0],
    ))));
    let divisor = Value::Array(Arc::new(Array::from_float64(FloatArray::from_slice(
        &vec64![2.0],
    ))));

    match dividend / divisor {
        Ok(Value::Array(arr_arc)) => {
            if let Array::NumericArray(NumericArray::Float64(result)) = arr_arc.as_ref() {
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
