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

//! # Value Creation Macros Example
//!
//! This example demonstrates the `val_*` macros for creating `Value` instances.
//! These macros wrap the existing `arr_*` macros and `Scalar` constructors, providing
//! a uniform interface for creating Values across all supported types.
//!
//! Value is useful in at least 2 scenarios:
//!     1. Engine routing   - deal with one (still-typed) value, so that you can
//!     match all possibilities, but send data through a uniform path.
//!     2. Broadcasting     - broadcast sum, minus, multiply, division, remainder, from
//!     anything to anything within the Value universe, with automatic broadcasting.
//!
//! Outside of at least these contexts, it is mildly inconvenient as it adds an additional processing
//! match stage, and thus the inner types are preferred.

use minarrow::{val_bool, val_f32, val_f64, val_i32, val_i64, val_str32, val_u32, val_u64, vec64};

#[cfg(feature = "scalar_type")]
use minarrow::{
    val_scalar_bool, val_scalar_f64, val_scalar_i32, val_scalar_null, val_scalar_str32,
};

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Minarrow Value Creation Macros Examples");
    println!("═══════════════════════════════════════════════════════════\n");

    demonstrate_integer_arrays();
    demonstrate_float_arrays();
    demonstrate_boolean_arrays();
    demonstrate_string_arrays();
    demonstrate_scalar_values();
    demonstrate_operations();

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  All value macro tests completed successfully!");
    println!("═══════════════════════════════════════════════════════════");
}

/// Demonstrate integer array value creation
fn demonstrate_integer_arrays() {
    println!("┌─ Integer Array Values");
    println!("│");

    // Create signed integer array values
    let val_a = val_i32![1, 2, 3, 4, 5];
    let val_b = val_i64![10, 20, 30];

    println!("│  val_i32![1, 2, 3, 4, 5] = {:?}", val_a);
    println!("│  val_i64![10, 20, 30]    = {:?}", val_b);
    println!("│");

    // Create unsigned integer array values
    let val_c = val_u32![100, 200, 300];
    let val_d = val_u64![1000, 2000];

    println!("│  val_u32![100, 200, 300] = {:?}", val_c);
    println!("│  val_u64![1000, 2000]    = {:?}", val_d);
    println!("└─ ✓ Passed\n");
}

/// Demonstrate float array value creation
fn demonstrate_float_arrays() {
    println!("┌─ Float Array Values");
    println!("│");

    let val_a = val_f32![1.5, 2.5, 3.5];
    let val_b = val_f64![3.14, 2.71, 1.41];

    println!("│  val_f32![1.5, 2.5, 3.5]      = {:?}", val_a);
    println!("│  val_f64![3.14, 2.71, 1.41]   = {:?}", val_b);
    println!("└─ ✓ Passed\n");
}

/// Demonstrate boolean array value creation
fn demonstrate_boolean_arrays() {
    println!("┌─ Boolean Array Values");
    println!("│");

    let val_a = val_bool![true, false, true, true];

    println!("│  val_bool![true, false, true, true] = {:?}", val_a);
    println!("└─ ✓ Passed\n");
}

/// Demonstrate string array value creation
fn demonstrate_string_arrays() {
    println!("┌─ String Array Values");
    println!("│");

    let val_a = val_str32!["hello", "world", "rust"];

    println!(
        "│  val_str32![\"hello\", \"world\", \"rust\"] = {:?}",
        val_a
    );
    println!("└─ ✓ Passed\n");
}

/// Demonstrate scalar value creation
fn demonstrate_scalar_values() {
    #[cfg(feature = "scalar_type")]
    {
        println!("┌─ Scalar Values");
        println!("│");

        let scalar_int = val_scalar_i32!(42);
        let scalar_float = val_scalar_f64!(3.14159);
        let scalar_bool = val_scalar_bool!(true);
        let scalar_str = val_scalar_str32!("Hello, Minarrow!");
        let scalar_null = val_scalar_null!();

        println!("│  val_scalar_i32!(42)            = {:?}", scalar_int);
        println!("│  val_scalar_f64!(3.14159)       = {:?}", scalar_float);
        println!("│  val_scalar_bool!(true)         = {:?}", scalar_bool);
        println!("│  val_scalar_str32!(\"Hello...\")  = {:?}", scalar_str);
        println!("│  val_scalar_null!()             = {:?}", scalar_null);
        println!("└─ ✓ Passed\n");
    }

    #[cfg(not(feature = "scalar_type"))]
    {
        println!("┌─ Scalar Values");
        println!("└─ ⊘ Skipped (scalar_type feature not enabled)\n");
    }
}

/// Demonstrate operations with macro-created values
fn demonstrate_operations() {
    println!("┌─ Operations with Macro-Created Values");
    println!("│");

    // Create values using macros
    let a = val_i32![1, 2, 3];
    let b = val_i32![10, 20, 30];

    // Perform operations
    match a + b {
        Ok(result) => println!("│  val_i32![1,2,3] + val_i32![10,20,30] = {:?}", result),
        Err(e) => println!("│  Error: {:?}", e),
    }
    println!("│");

    // Broadcasting example
    let single = val_i32![vec64![100]];
    let multi = val_i32![1, 2, 3, 4, 5];

    match single * multi {
        Ok(result) => println!("│  val_i32![100] * val_i32![1,2,3,4,5] = {:?}", result),
        Err(e) => println!("│  Error: {:?}", e),
    }
    println!("│");

    // Float operations
    let f1 = val_f64![1.0, 2.0, 3.0];
    let f2 = val_f64![0.5, 0.5, 0.5];

    match f1 + f2 {
        Ok(result) => println!(
            "│  val_f64![1.0,2.0,3.0] + val_f64![0.5,0.5,0.5] = {:?}",
            result
        ),
        Err(e) => println!("│  Error: {:?}", e),
    }

    println!("└─ ✓ Passed\n");
}
