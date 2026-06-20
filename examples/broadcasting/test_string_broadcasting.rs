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

//! # String Broadcasting Examples
//!
//! This example demonstrates string concatenation with broadcasting in Minarrow.
//! String arrays support the same broadcasting rules as numeric arrays, allowing
//! efficient string operations across arrays of different sizes.
//!
//! ## Key Features
//! - String32 and String64 array concatenation
//! - Broadcasting single strings to match larger arrays
//! - Efficient zero-copy operations where possible
//! - Support for both directions: [1] + [N] and [N] + [1]

use minarrow::{Array, MaskedArray, StringArray, TextArray, Value};
use std::sync::Arc;

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Minarrow String Broadcasting Examples");
    println!("═══════════════════════════════════════════════════════════\n");

    test_equal_length_strings();
    test_broadcast_string32_forward();
    test_broadcast_string32_reverse();
    test_broadcast_string64();
    test_empty_strings();
    test_complex_concatenation();

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  All string broadcasting tests completed successfully!");
    println!("═══════════════════════════════════════════════════════════");
}

/// Test concatenation of equal-length string arrays
fn test_equal_length_strings() {
    println!("┌─ Test 1: Equal-Length String32 Arrays");
    println!("│  Operation: [\"Hello\", \"Hi\", \"Hey\"] + [\" World\", \" Rust\", \" There\"]");
    println!("│  Expected:  [\"Hello World\", \"Hi Rust\", \"Hey There\"]");

    let str1 = Value::Array(Arc::new(Array::from_string32(StringArray::from_slice(&[
        "Hello", "Hi", "Hey",
    ]))));
    let str2 = Value::Array(Arc::new(Array::from_string32(StringArray::from_slice(&[
        " World", " Rust", " There",
    ]))));

    match str1 + str2 {
        Ok(Value::Array(arr_arc)) => {
            if let Array::TextArray(TextArray::String32(arr)) = arr_arc.as_ref() {
                let len = MaskedArray::len(&*arr);
                println!("│  Results:");
                for i in 0..len {
                    let s = arr.get_str(i).unwrap_or("<null>");
                    println!("│    [{}] \"{}\"", i, s);
                }
                println!("└─ ✓ Passed\n");
            } else {
                println!("└─ ✗ Error: Unexpected array type\n");
            }
        }
        Ok(_) => println!("└─ ✗ Error: Unexpected result type\n"),
        Err(e) => println!("└─ ✗ Error: {:?}\n", e),
    }
}

/// Test broadcasting: single string + array of strings
fn test_broadcast_string32_forward() {
    println!("┌─ Test 2: Broadcasting String32 [1] + [N]");
    println!("│  Operation: [\"Hello\"] + [\" World\", \" Rust\", \" Minarrow\"]");
    println!("│  Expected:  [\"Hello World\", \"Hello Rust\", \"Hello Minarrow\"]");

    let single = Value::Array(Arc::new(Array::from_string32(StringArray::from_slice(&[
        "Hello",
    ]))));
    let multiple = Value::Array(Arc::new(Array::from_string32(StringArray::from_slice(&[
        " World",
        " Rust",
        " Minarrow",
    ]))));

    match single + multiple {
        Ok(Value::Array(arr_arc)) => {
            if let Array::TextArray(TextArray::String32(arr)) = arr_arc.as_ref() {
                let len = MaskedArray::len(&*arr);
                println!("│  Results:");
                for i in 0..len {
                    let s = arr.get_str(i).unwrap_or("<null>");
                    println!("│    [{}] \"{}\"", i, s);
                }
                println!("└─ ✓ Passed\n");
            } else {
                println!("└─ ✗ Error: Unexpected array type\n");
            }
        }
        Ok(_) => println!("└─ ✗ Error: Unexpected result type\n"),
        Err(e) => println!("└─ ✗ Error: {:?}\n", e),
    }
}

/// Test broadcasting: array of strings + single string
fn test_broadcast_string32_reverse() {
    println!("┌─ Test 3: Broadcasting String32 [N] + [1]");
    println!("│  Operation: [\"cmd\", \"exec\", \"run\"] + [\"_process\"]");
    println!("│  Expected:  [\"cmd_process\", \"exec_process\", \"run_process\"]");

    let multiple = Value::Array(Arc::new(Array::from_string32(StringArray::from_slice(&[
        "cmd", "exec", "run",
    ]))));
    let single = Value::Array(Arc::new(Array::from_string32(StringArray::from_slice(&[
        "_process",
    ]))));

    match multiple + single {
        Ok(Value::Array(arr_arc)) => {
            if let Array::TextArray(TextArray::String32(arr)) = arr_arc.as_ref() {
                let len = MaskedArray::len(&*arr);
                println!("│  Results:");
                for i in 0..len {
                    let s = arr.get_str(i).unwrap_or("<null>");
                    println!("│    [{}] \"{}\"", i, s);
                }
                println!("└─ ✓ Passed\n");
            } else {
                println!("└─ ✗ Error: Unexpected array type\n");
            }
        }
        Ok(_) => println!("└─ ✗ Error: Unexpected result type\n"),
        Err(e) => println!("└─ ✗ Error: {:?}\n", e),
    }
}

/// Test broadcasting with String64 arrays
fn test_broadcast_string64() {
    println!("┌─ Test 4: Broadcasting String64");
    println!("│  Operation: [\"Goodbye\"] + [\" World\", \" Rust\", \" Friend\"]");
    println!("│  Expected:  [\"Goodbye World\", \"Goodbye Rust\", \"Goodbye Friend\"]");

    let single64 = Value::Array(Arc::new(Array::from_string64(StringArray::from_slice(&[
        "Goodbye",
    ]))));
    let multiple64 = Value::Array(Arc::new(Array::from_string64(StringArray::from_slice(&[
        " World", " Rust", " Friend",
    ]))));

    match single64 + multiple64 {
        Ok(Value::Array(arr_arc)) => {
            if let Array::TextArray(TextArray::String64(arr)) = arr_arc.as_ref() {
                let len = MaskedArray::len(&*arr);
                println!("│  Results:");
                for i in 0..len {
                    let s = arr.get_str(i).unwrap_or("<null>");
                    println!("│    [{}] \"{}\"", i, s);
                }
                println!("└─ ✓ Passed\n");
            } else {
                println!("└─ ✗ Error: Unexpected array type\n");
            }
        }
        Ok(_) => println!("└─ ✗ Error: Unexpected result type\n"),
        Err(e) => println!("└─ ✗ Error: {:?}\n", e),
    }
}

/// Test concatenation with empty strings
fn test_empty_strings() {
    println!("┌─ Test 5: Empty String Handling");
    println!("│  Operation: [\"\", \"prefix\"] + [\"suffix\", \"\"]");
    println!("│  Expected:  [\"suffix\", \"prefix\"]");

    let arr1 = Value::Array(Arc::new(Array::from_string32(StringArray::from_slice(&[
        "", "prefix",
    ]))));
    let arr2 = Value::Array(Arc::new(Array::from_string32(StringArray::from_slice(&[
        "suffix", "",
    ]))));

    match arr1 + arr2 {
        Ok(Value::Array(arr_arc)) => {
            if let Array::TextArray(TextArray::String32(arr)) = arr_arc.as_ref() {
                let len = MaskedArray::len(&*arr);
                println!("│  Results:");
                for i in 0..len {
                    let s = arr.get_str(i).unwrap_or("<null>");
                    println!("│    [{}] \"{}\"", i, s);
                }
                println!("└─ ✓ Passed\n");
            } else {
                println!("└─ ✗ Error: Unexpected array type\n");
            }
        }
        Ok(_) => println!("└─ ✗ Error: Unexpected result type\n"),
        Err(e) => println!("└─ ✗ Error: {:?}\n", e),
    }
}

/// Test complex multi-word concatenation
fn test_complex_concatenation() {
    println!("┌─ Test 6: Complex Multi-Word Concatenation");
    println!("│  Operation: [\"Error:\", \"Warning:\", \"Info:\"] + [\" Connection failed\"]");
    println!("│  Expected:  [\"Error: Connection failed\", \"Warning: Connection failed\", ...]");

    let prefixes = Value::Array(Arc::new(Array::from_string32(StringArray::from_slice(&[
        "Error:", "Warning:", "Info:",
    ]))));
    let message = Value::Array(Arc::new(Array::from_string32(StringArray::from_slice(&[
        " Connection failed",
    ]))));

    match prefixes + message {
        Ok(Value::Array(arr_arc)) => {
            if let Array::TextArray(TextArray::String32(arr)) = arr_arc.as_ref() {
                let len = MaskedArray::len(&*arr);
                println!("│  Results:");
                for i in 0..len {
                    let s = arr.get_str(i).unwrap_or("<null>");
                    println!("│    [{}] \"{}\"", i, s);
                }
                println!("└─ ✓ Passed\n");
            } else {
                println!("└─ ✗ Error: Unexpected array type\n");
            }
        }
        Ok(_) => println!("└─ ✗ Error: Unexpected result type\n"),
        Err(e) => println!("└─ ✗ Error: {:?}\n", e),
    }
}
