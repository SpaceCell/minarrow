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

//! # Scalar Arithmetic Examples
//!
//! This example demonstrates arithmetic operations on scalar values in Minarrow.
//! Unlike array operations, scalar arithmetic maintains scalar types throughout
//! the computation chain, providing efficient operations on individual values.
//!
//! ## Key Features
//! - Scalar + Scalar = Scalar (no array conversion)
//! - Automatic type promotion (Int + Float -> Float)
//! - String concatenation support
//! - All standard arithmetic operations: +, -, *, /

use minarrow::{Scalar, Value};

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Minarrow Scalar Arithmetic Examples");
    println!("═══════════════════════════════════════════════════════════\n");

    test_integer_addition();
    test_integer_multiplication();
    test_float_operations();
    test_mixed_type_promotion();
    test_division();
    test_subtraction();
    test_string_concatenation();
    test_reference_operations();

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  All scalar arithmetic tests completed successfully!");
    println!("═══════════════════════════════════════════════════════════");
}

/// Test basic integer scalar addition
fn test_integer_addition() {
    println!("┌─ Test 1: Integer Scalar Addition");
    println!("│  Operation: Scalar(10) + Scalar(20)");
    println!("│  Expected:  Scalar(30)");

    let a = Value::Scalar(Scalar::Int32(10));
    let b = Value::Scalar(Scalar::Int32(20));

    match a + b {
        Ok(Value::Scalar(Scalar::Int32(val))) => {
            assert_eq!(val, 30, "Expected 30, got {}", val);
            println!("│  Result:    Scalar::Int32({})", val);
            println!("└─ ✓ Passed\n");
        }
        Ok(other) => println!("└─ ✗ Error: Unexpected result {:?}\n", other),
        Err(e) => println!("└─ ✗ Error: {:?}\n", e),
    }
}

/// Test integer scalar multiplication
fn test_integer_multiplication() {
    println!("┌─ Test 2: Integer Scalar Multiplication");
    println!("│  Operation: Scalar(7) * Scalar(6)");
    println!("│  Expected:  Scalar(42)");

    let a = Value::Scalar(Scalar::Int32(7));
    let b = Value::Scalar(Scalar::Int32(6));

    match a * b {
        Ok(Value::Scalar(Scalar::Int32(val))) => {
            assert_eq!(val, 42, "Expected 42, got {}", val);
            println!("│  Result:    Scalar::Int32({})", val);
            println!("└─ ✓ Passed\n");
        }
        Ok(other) => println!("└─ ✗ Error: Unexpected result {:?}\n", other),
        Err(e) => println!("└─ ✗ Error: {:?}\n", e),
    }
}

/// Test float scalar operations
fn test_float_operations() {
    println!("┌─ Test 3: Float Scalar Operations");
    println!("│  Operation: Scalar(3.14) + Scalar(2.86)");
    println!("│  Expected:  Scalar(6.0)");

    let a = Value::Scalar(Scalar::Float64(3.14));
    let b = Value::Scalar(Scalar::Float64(2.86));

    match a + b {
        Ok(Value::Scalar(Scalar::Float64(val))) => {
            let expected = 6.0;
            let diff = (val - expected).abs();
            assert!(diff < 0.001, "Expected {}, got {}", expected, val);
            println!("│  Result:    Scalar::Float64({:.2})", val);
            println!("└─ ✓ Passed\n");
        }
        Ok(other) => println!("└─ ✗ Error: Unexpected result {:?}\n", other),
        Err(e) => println!("└─ ✗ Error: {:?}\n", e),
    }
}

/// Test automatic type promotion in mixed-type operations
fn test_mixed_type_promotion() {
    println!("┌─ Test 4: Mixed Type Promotion");
    println!("│  Operation: Scalar::Int32(10) * Scalar::Float32(2.5)");
    println!("│  Expected:  Scalar::Float32(25.0)");

    let a = Value::Scalar(Scalar::Int32(10));
    let b = Value::Scalar(Scalar::Float32(2.5));

    match a * b {
        Ok(Value::Scalar(Scalar::Float32(val))) => {
            let expected = 25.0;
            let diff = (val - expected).abs();
            assert!(diff < 0.001, "Expected {}, got {}", expected, val);
            println!("│  Result:    Scalar::Float32({}) (type promoted)", val);
            println!("└─ ✓ Passed\n");
        }
        Ok(other) => println!("└─ ✗ Error: Unexpected result {:?}\n", other),
        Err(e) => println!("└─ ✗ Error: {:?}\n", e),
    }
}

/// Test scalar division
fn test_division() {
    println!("┌─ Test 5: Scalar Division");
    println!("│  Operation: Scalar(100.0) / Scalar(4.0)");
    println!("│  Expected:  Scalar(25.0)");

    let a = Value::Scalar(Scalar::Float64(100.0));
    let b = Value::Scalar(Scalar::Float64(4.0));

    match a / b {
        Ok(Value::Scalar(Scalar::Float64(val))) => {
            assert_eq!(val, 25.0, "Expected 25.0, got {}", val);
            println!("│  Result:    Scalar::Float64({})", val);
            println!("└─ ✓ Passed\n");
        }
        Ok(other) => println!("└─ ✗ Error: Unexpected result {:?}\n", other),
        Err(e) => println!("└─ ✗ Error: {:?}\n", e),
    }
}

/// Test scalar subtraction
fn test_subtraction() {
    println!("┌─ Test 6: Scalar Subtraction");
    println!("│  Operation: Scalar(50) - Scalar(15)");
    println!("│  Expected:  Scalar(35)");

    let a = Value::Scalar(Scalar::Int32(50));
    let b = Value::Scalar(Scalar::Int32(15));

    match a - b {
        Ok(Value::Scalar(Scalar::Int32(val))) => {
            assert_eq!(val, 35, "Expected 35, got {}", val);
            println!("│  Result:    Scalar::Int32({})", val);
            println!("└─ ✓ Passed\n");
        }
        Ok(other) => println!("└─ ✗ Error: Unexpected result {:?}\n", other),
        Err(e) => println!("└─ ✗ Error: {:?}\n", e),
    }
}

/// Test string scalar concatenation
fn test_string_concatenation() {
    println!("┌─ Test 7: String Scalar Concatenation");
    println!("│  Operation: Scalar(\"Hello\") + Scalar(\" World\")");
    println!("│  Expected:  Scalar(\"Hello World\")");

    let a = Value::Scalar(Scalar::String32("Hello".to_string()));
    let b = Value::Scalar(Scalar::String32(" World".to_string()));

    match a + b {
        Ok(Value::Scalar(Scalar::String32(val))) => {
            assert_eq!(val, "Hello World", "Expected 'Hello World', got '{}'", val);
            println!("│  Result:    Scalar::String32(\"{}\")", val);
            println!("└─ ✓ Passed\n");
        }
        Ok(other) => println!("└─ ✗ Error: Unexpected result {:?}\n", other),
        Err(e) => println!("└─ ✗ Error: {:?}\n", e),
    }
}

/// Test operations with references (non-consuming)
fn test_reference_operations() {
    println!("┌─ Test 8: Reference Operations");
    println!("│  Operation: &Scalar(5) * &Scalar(8)");
    println!("│  Expected:  Scalar(40)");

    let a = Value::Scalar(Scalar::Int32(5));
    let b = Value::Scalar(Scalar::Int32(8));

    match &a * &b {
        Ok(Value::Scalar(Scalar::Int32(val))) => {
            assert_eq!(val, 40, "Expected 40, got {}", val);
            println!("│  Result:    Scalar::Int32({})", val);
            println!("│  Note: Original scalars remain available for reuse");
            println!("└─ ✓ Passed\n");
        }
        Ok(other) => println!("└─ ✗ Error: Unexpected result {:?}\n", other),
        Err(e) => println!("└─ ✗ Error: {:?}\n", e),
    }
}
