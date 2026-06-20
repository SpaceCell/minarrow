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

//! # Value Arithmetic Operators
//!
//! Implementation of standard Rust arithmetic operators (Add, Sub, Mul, Div)
//! for the Value enum with automatic broadcasting support.
//!
//! This enables ergonomic arithmetic operations like:
//! ```rust
//! use minarrow::{Value, arr_i32, vec64};
//! use std::sync::Arc;
//! let arr1 = arr_i32![1, 2, 3, 4];
//! let arr2 = arr_i32![5, 6, 7, 8];
//! let a = Value::Array(Arc::new(arr1));
//! let b = Value::Array(Arc::new(arr2));
//! let result = a + b;  // Automatically broadcasts and performs element-wise addition
//! ```

#[cfg(feature = "views")]
use crate::ArrayV;
#[cfg(feature = "cube")]
use crate::Cube;
use crate::kernels::broadcast::{
    value_add, value_divide, value_multiply, value_remainder, value_subtract,
};
use crate::{Array, FieldArray, Table};

use crate::enums::error::MinarrowError;
use crate::enums::value::Value;

#[cfg(feature = "chunked")]
use crate::{SuperArray, SuperTable};
#[cfg(all(feature = "views", feature = "chunked"))]
use crate::{SuperArrayV, SuperTableV};

use std::ops::{Add, Div, Mul, Rem, Sub};
use std::sync::Arc;

impl Add for Value {
    type Output = Result<Value, MinarrowError>;

    fn add(self, rhs: Self) -> Self::Output {
        value_add(self, rhs)
    }
}

impl Sub for Value {
    type Output = Result<Value, MinarrowError>;

    fn sub(self, rhs: Self) -> Self::Output {
        value_subtract(self, rhs)
    }
}

impl Mul for Value {
    type Output = Result<Value, MinarrowError>;

    fn mul(self, rhs: Self) -> Self::Output {
        value_multiply(self, rhs)
    }
}

impl Div for Value {
    type Output = Result<Value, MinarrowError>;

    fn div(self, rhs: Self) -> Self::Output {
        value_divide(self, rhs)
    }
}

impl Rem for Value {
    type Output = Result<Value, MinarrowError>;

    fn rem(self, rhs: Self) -> Self::Output {
        value_remainder(self, rhs)
    }
}

// Reference implementations for convenience
impl Add<&Value> for &Value {
    type Output = Result<Value, MinarrowError>;

    fn add(self, rhs: &Value) -> Self::Output {
        value_add(self.clone(), rhs.clone())
    }
}

impl Sub<&Value> for &Value {
    type Output = Result<Value, MinarrowError>;

    fn sub(self, rhs: &Value) -> Self::Output {
        value_subtract(self.clone(), rhs.clone())
    }
}

impl Mul<&Value> for &Value {
    type Output = Result<Value, MinarrowError>;

    fn mul(self, rhs: &Value) -> Self::Output {
        value_multiply(self.clone(), rhs.clone())
    }
}

impl Div<&Value> for &Value {
    type Output = Result<Value, MinarrowError>;

    fn div(self, rhs: &Value) -> Self::Output {
        value_divide(self.clone(), rhs.clone())
    }
}

impl Rem<&Value> for &Value {
    type Output = Result<Value, MinarrowError>;

    fn rem(self, rhs: &Value) -> Self::Output {
        value_remainder(self.clone(), rhs.clone())
    }
}

// ===== Arithmetic Trait Implementations for Specific Types =====

// Array implementations
impl Add for Array {
    type Output = Result<Array, MinarrowError>;
    fn add(self, rhs: Self) -> Self::Output {
        match value_add(Value::Array(Arc::new(self)), Value::Array(Arc::new(rhs)))? {
            Value::Array(arr) => Ok(Arc::unwrap_or_clone(arr)),
            _ => Err(MinarrowError::TypeError {
                from: "Array",
                to: "Array",
                message: Some("Unexpected result type from addition".to_string()),
            }),
        }
    }
}

impl Sub for Array {
    type Output = Result<Array, MinarrowError>;
    fn sub(self, rhs: Self) -> Self::Output {
        match value_subtract(Value::Array(Arc::new(self)), Value::Array(Arc::new(rhs)))? {
            Value::Array(arr) => Ok(Arc::unwrap_or_clone(arr)),
            _ => Err(MinarrowError::TypeError {
                from: "Array",
                to: "Array",
                message: Some("Unexpected result type from subtraction".to_string()),
            }),
        }
    }
}

impl Mul for Array {
    type Output = Result<Array, MinarrowError>;
    fn mul(self, rhs: Self) -> Self::Output {
        match value_multiply(Value::Array(Arc::new(self)), Value::Array(Arc::new(rhs)))? {
            Value::Array(arr) => Ok(Arc::unwrap_or_clone(arr)),
            _ => Err(MinarrowError::TypeError {
                from: "Array",
                to: "Array",
                message: Some("Unexpected result type from multiplication".to_string()),
            }),
        }
    }
}

impl Div for Array {
    type Output = Result<Array, MinarrowError>;
    fn div(self, rhs: Self) -> Self::Output {
        match value_divide(Value::Array(Arc::new(self)), Value::Array(Arc::new(rhs)))? {
            Value::Array(arr) => Ok(Arc::unwrap_or_clone(arr)),
            _ => Err(MinarrowError::TypeError {
                from: "Array",
                to: "Array",
                message: Some("Unexpected result type from division".to_string()),
            }),
        }
    }
}

impl Rem for Array {
    type Output = Result<Array, MinarrowError>;
    fn rem(self, rhs: Self) -> Self::Output {
        match value_remainder(Value::Array(Arc::new(self)), Value::Array(Arc::new(rhs)))? {
            Value::Array(arr) => Ok(Arc::unwrap_or_clone(arr)),
            _ => Err(MinarrowError::TypeError {
                from: "Array",
                to: "Array",
                message: Some("Unexpected result type from remainder".to_string()),
            }),
        }
    }
}

// View type implementations

// ArrayView implementations
#[cfg(feature = "views")]
impl Add for ArrayV {
    type Output = Result<Array, MinarrowError>;
    fn add(self, rhs: Self) -> Self::Output {
        match value_add(
            Value::ArrayView(Arc::new(self)),
            Value::ArrayView(Arc::new(rhs)),
        )? {
            Value::Array(arr) => Ok(Arc::unwrap_or_clone(arr)),
            _ => Err(MinarrowError::TypeError {
                from: "ArrayView",
                to: "Array",
                message: Some("Unexpected result type from addition".to_string()),
            }),
        }
    }
}

#[cfg(feature = "views")]
impl Sub for ArrayV {
    type Output = Result<Array, MinarrowError>;
    fn sub(self, rhs: Self) -> Self::Output {
        match value_subtract(
            Value::ArrayView(Arc::new(self)),
            Value::ArrayView(Arc::new(rhs)),
        )? {
            Value::Array(arr) => Ok(Arc::unwrap_or_clone(arr)),
            _ => Err(MinarrowError::TypeError {
                from: "ArrayView",
                to: "Array",
                message: Some("Unexpected result type from subtraction".to_string()),
            }),
        }
    }
}

#[cfg(feature = "views")]
impl Mul for ArrayV {
    type Output = Result<Array, MinarrowError>;
    fn mul(self, rhs: Self) -> Self::Output {
        match value_multiply(
            Value::ArrayView(Arc::new(self)),
            Value::ArrayView(Arc::new(rhs)),
        )? {
            Value::Array(arr) => Ok(Arc::unwrap_or_clone(arr)),
            _ => Err(MinarrowError::TypeError {
                from: "ArrayView",
                to: "Array",
                message: Some("Unexpected result type from multiplication".to_string()),
            }),
        }
    }
}

#[cfg(feature = "views")]
impl Div for ArrayV {
    type Output = Result<Array, MinarrowError>;
    fn div(self, rhs: Self) -> Self::Output {
        match value_divide(
            Value::ArrayView(Arc::new(self)),
            Value::ArrayView(Arc::new(rhs)),
        )? {
            Value::Array(arr) => Ok(Arc::unwrap_or_clone(arr)),
            _ => Err(MinarrowError::TypeError {
                from: "ArrayView",
                to: "Array",
                message: Some("Unexpected result type from division".to_string()),
            }),
        }
    }
}

#[cfg(feature = "views")]
impl Rem for ArrayV {
    type Output = Result<Array, MinarrowError>;
    fn rem(self, rhs: Self) -> Self::Output {
        match value_remainder(
            Value::ArrayView(Arc::new(self)),
            Value::ArrayView(Arc::new(rhs)),
        )? {
            Value::Array(arr) => Ok(Arc::unwrap_or_clone(arr)),
            _ => Err(MinarrowError::TypeError {
                from: "ArrayView",
                to: "Array",
                message: Some("Unexpected result type from remainder".to_string()),
            }),
        }
    }
}

// Table implementations
impl Add for Table {
    type Output = Result<Table, MinarrowError>;
    fn add(self, rhs: Self) -> Self::Output {
        match value_add(Value::Table(Arc::new(self)), Value::Table(Arc::new(rhs)))? {
            Value::Table(t) => Ok(Arc::unwrap_or_clone(t)),
            _ => Err(MinarrowError::TypeError {
                from: "Table",
                to: "Table",
                message: Some("Unexpected result type from addition".to_string()),
            }),
        }
    }
}

impl Sub for Table {
    type Output = Result<Table, MinarrowError>;
    fn sub(self, rhs: Self) -> Self::Output {
        match value_subtract(Value::Table(Arc::new(self)), Value::Table(Arc::new(rhs)))? {
            Value::Table(t) => Ok(Arc::unwrap_or_clone(t)),
            _ => Err(MinarrowError::TypeError {
                from: "Table",
                to: "Table",
                message: Some("Unexpected result type from subtraction".to_string()),
            }),
        }
    }
}

impl Mul for Table {
    type Output = Result<Table, MinarrowError>;
    fn mul(self, rhs: Self) -> Self::Output {
        match value_multiply(Value::Table(Arc::new(self)), Value::Table(Arc::new(rhs)))? {
            Value::Table(t) => Ok(Arc::unwrap_or_clone(t)),
            _ => Err(MinarrowError::TypeError {
                from: "Table",
                to: "Table",
                message: Some("Unexpected result type from multiplication".to_string()),
            }),
        }
    }
}

impl Div for Table {
    type Output = Result<Table, MinarrowError>;
    fn div(self, rhs: Self) -> Self::Output {
        match value_divide(Value::Table(Arc::new(self)), Value::Table(Arc::new(rhs)))? {
            Value::Table(t) => Ok(Arc::unwrap_or_clone(t)),
            _ => Err(MinarrowError::TypeError {
                from: "Table",
                to: "Table",
                message: Some("Unexpected result type from division".to_string()),
            }),
        }
    }
}

impl Rem for Table {
    type Output = Result<Table, MinarrowError>;
    fn rem(self, rhs: Self) -> Self::Output {
        match value_remainder(Value::Table(Arc::new(self)), Value::Table(Arc::new(rhs)))? {
            Value::Table(t) => Ok(Arc::unwrap_or_clone(t)),
            _ => Err(MinarrowError::TypeError {
                from: "Table",
                to: "Table",
                message: Some("Unexpected result type from remainder".to_string()),
            }),
        }
    }
}

// FieldArray implementations
impl Add for FieldArray {
    type Output = Result<FieldArray, MinarrowError>;
    fn add(self, rhs: Self) -> Self::Output {
        match value_add(
            Value::FieldArray(Arc::new(self)),
            Value::FieldArray(Arc::new(rhs)),
        )? {
            Value::FieldArray(fa) => Ok(Arc::unwrap_or_clone(fa)),
            _ => Err(MinarrowError::TypeError {
                from: "FieldArray",
                to: "FieldArray",
                message: Some("Unexpected result type from addition".to_string()),
            }),
        }
    }
}

impl Sub for FieldArray {
    type Output = Result<FieldArray, MinarrowError>;
    fn sub(self, rhs: Self) -> Self::Output {
        match value_subtract(
            Value::FieldArray(Arc::new(self)),
            Value::FieldArray(Arc::new(rhs)),
        )? {
            Value::FieldArray(fa) => Ok(Arc::unwrap_or_clone(fa)),
            _ => Err(MinarrowError::TypeError {
                from: "FieldArray",
                to: "FieldArray",
                message: Some("Unexpected result type from subtraction".to_string()),
            }),
        }
    }
}

impl Mul for FieldArray {
    type Output = Result<FieldArray, MinarrowError>;
    fn mul(self, rhs: Self) -> Self::Output {
        match value_multiply(
            Value::FieldArray(Arc::new(self)),
            Value::FieldArray(Arc::new(rhs)),
        )? {
            Value::FieldArray(fa) => Ok(Arc::unwrap_or_clone(fa)),
            _ => Err(MinarrowError::TypeError {
                from: "FieldArray",
                to: "FieldArray",
                message: Some("Unexpected result type from multiplication".to_string()),
            }),
        }
    }
}

impl Div for FieldArray {
    type Output = Result<FieldArray, MinarrowError>;
    fn div(self, rhs: Self) -> Self::Output {
        match value_divide(
            Value::FieldArray(Arc::new(self)),
            Value::FieldArray(Arc::new(rhs)),
        )? {
            Value::FieldArray(fa) => Ok(Arc::unwrap_or_clone(fa)),
            _ => Err(MinarrowError::TypeError {
                from: "FieldArray",
                to: "FieldArray",
                message: Some("Unexpected result type from division".to_string()),
            }),
        }
    }
}

impl Rem for FieldArray {
    type Output = Result<FieldArray, MinarrowError>;
    fn rem(self, rhs: Self) -> Self::Output {
        match value_remainder(
            Value::FieldArray(Arc::new(self)),
            Value::FieldArray(Arc::new(rhs)),
        )? {
            Value::FieldArray(fa) => Ok(Arc::unwrap_or_clone(fa)),
            _ => Err(MinarrowError::TypeError {
                from: "FieldArray",
                to: "FieldArray",
                message: Some("Unexpected result type from remainder".to_string()),
            }),
        }
    }
}

// SuperArray implementations
#[cfg(feature = "chunked")]
impl Add for SuperArray {
    type Output = Result<SuperArray, MinarrowError>;
    fn add(self, rhs: Self) -> Self::Output {
        match value_add(
            Value::SuperArray(Arc::new(self)),
            Value::SuperArray(Arc::new(rhs)),
        )? {
            Value::SuperArray(sa) => Ok(Arc::unwrap_or_clone(sa)),
            _ => Err(MinarrowError::TypeError {
                from: "SuperArray",
                to: "SuperArray",
                message: Some("Unexpected result type from addition".to_string()),
            }),
        }
    }
}

#[cfg(feature = "chunked")]
impl Sub for SuperArray {
    type Output = Result<SuperArray, MinarrowError>;
    fn sub(self, rhs: Self) -> Self::Output {
        match value_subtract(
            Value::SuperArray(Arc::new(self)),
            Value::SuperArray(Arc::new(rhs)),
        )? {
            Value::SuperArray(sa) => Ok(Arc::unwrap_or_clone(sa)),
            _ => Err(MinarrowError::TypeError {
                from: "SuperArray",
                to: "SuperArray",
                message: Some("Unexpected result type from subtraction".to_string()),
            }),
        }
    }
}

#[cfg(feature = "chunked")]
impl Mul for SuperArray {
    type Output = Result<SuperArray, MinarrowError>;
    fn mul(self, rhs: Self) -> Self::Output {
        match value_multiply(
            Value::SuperArray(Arc::new(self)),
            Value::SuperArray(Arc::new(rhs)),
        )? {
            Value::SuperArray(sa) => Ok(Arc::unwrap_or_clone(sa)),
            _ => Err(MinarrowError::TypeError {
                from: "SuperArray",
                to: "SuperArray",
                message: Some("Unexpected result type from multiplication".to_string()),
            }),
        }
    }
}

#[cfg(feature = "chunked")]
impl Div for SuperArray {
    type Output = Result<SuperArray, MinarrowError>;
    fn div(self, rhs: Self) -> Self::Output {
        match value_divide(
            Value::SuperArray(Arc::new(self)),
            Value::SuperArray(Arc::new(rhs)),
        )? {
            Value::SuperArray(sa) => Ok(Arc::unwrap_or_clone(sa)),
            _ => Err(MinarrowError::TypeError {
                from: "SuperArray",
                to: "SuperArray",
                message: Some("Unexpected result type from division".to_string()),
            }),
        }
    }
}

#[cfg(feature = "chunked")]
impl Rem for SuperArray {
    type Output = Result<SuperArray, MinarrowError>;
    fn rem(self, rhs: Self) -> Self::Output {
        match value_remainder(
            Value::SuperArray(Arc::new(self)),
            Value::SuperArray(Arc::new(rhs)),
        )? {
            Value::SuperArray(sa) => Ok(Arc::unwrap_or_clone(sa)),
            _ => Err(MinarrowError::TypeError {
                from: "SuperArray",
                to: "SuperArray",
                message: Some("Unexpected result type from remainder".to_string()),
            }),
        }
    }
}

// SuperArrayView implementations
#[cfg(all(feature = "chunked", feature = "views"))]
impl Add for SuperArrayV {
    type Output = Result<SuperArray, MinarrowError>;
    fn add(self, rhs: Self) -> Self::Output {
        match value_add(
            Value::SuperArrayView(Arc::new(self)),
            Value::SuperArrayView(Arc::new(rhs)),
        )? {
            Value::SuperArray(sa) => Ok(Arc::unwrap_or_clone(sa)),
            _ => Err(MinarrowError::TypeError {
                from: "SuperArrayView",
                to: "SuperArray",
                message: Some("Unexpected result type from addition".to_string()),
            }),
        }
    }
}

#[cfg(all(feature = "chunked", feature = "views"))]
impl Sub for SuperArrayV {
    type Output = Result<SuperArray, MinarrowError>;
    fn sub(self, rhs: Self) -> Self::Output {
        match value_subtract(
            Value::SuperArrayView(Arc::new(self)),
            Value::SuperArrayView(Arc::new(rhs)),
        )? {
            Value::SuperArray(sa) => Ok(Arc::unwrap_or_clone(sa)),
            _ => Err(MinarrowError::TypeError {
                from: "SuperArrayView",
                to: "SuperArray",
                message: Some("Unexpected result type from subtraction".to_string()),
            }),
        }
    }
}

#[cfg(all(feature = "chunked", feature = "views"))]
impl Mul for SuperArrayV {
    type Output = Result<SuperArray, MinarrowError>;
    fn mul(self, rhs: Self) -> Self::Output {
        match value_multiply(
            Value::SuperArrayView(Arc::new(self)),
            Value::SuperArrayView(Arc::new(rhs)),
        )? {
            Value::SuperArray(sa) => Ok(Arc::unwrap_or_clone(sa)),
            _ => Err(MinarrowError::TypeError {
                from: "SuperArrayView",
                to: "SuperArray",
                message: Some("Unexpected result type from multiplication".to_string()),
            }),
        }
    }
}

#[cfg(all(feature = "chunked", feature = "views"))]
impl Div for SuperArrayV {
    type Output = Result<SuperArray, MinarrowError>;
    fn div(self, rhs: Self) -> Self::Output {
        match value_divide(
            Value::SuperArrayView(Arc::new(self)),
            Value::SuperArrayView(Arc::new(rhs)),
        )? {
            Value::SuperArray(sa) => Ok(Arc::unwrap_or_clone(sa)),
            _ => Err(MinarrowError::TypeError {
                from: "SuperArrayView",
                to: "SuperArray",
                message: Some("Unexpected result type from division".to_string()),
            }),
        }
    }
}

#[cfg(all(feature = "chunked", feature = "views"))]
impl Rem for SuperArrayV {
    type Output = Result<SuperArray, MinarrowError>;
    fn rem(self, rhs: Self) -> Self::Output {
        match value_remainder(
            Value::SuperArrayView(Arc::new(self)),
            Value::SuperArrayView(Arc::new(rhs)),
        )? {
            Value::SuperArray(sa) => Ok(Arc::unwrap_or_clone(sa)),
            _ => Err(MinarrowError::TypeError {
                from: "SuperArrayView",
                to: "SuperArray",
                message: Some("Unexpected result type from remainder".to_string()),
            }),
        }
    }
}

// SuperTable implementations
#[cfg(feature = "chunked")]
impl Add for SuperTable {
    type Output = Result<SuperTable, MinarrowError>;
    fn add(self, rhs: Self) -> Self::Output {
        match value_add(
            Value::SuperTable(Arc::new(self)),
            Value::SuperTable(Arc::new(rhs)),
        )? {
            Value::SuperTable(st) => Ok(Arc::unwrap_or_clone(st)),
            _ => Err(MinarrowError::TypeError {
                from: "SuperTable",
                to: "SuperTable",
                message: Some("Unexpected result type from addition".to_string()),
            }),
        }
    }
}

#[cfg(feature = "chunked")]
impl Sub for SuperTable {
    type Output = Result<SuperTable, MinarrowError>;
    fn sub(self, rhs: Self) -> Self::Output {
        match value_subtract(
            Value::SuperTable(Arc::new(self)),
            Value::SuperTable(Arc::new(rhs)),
        )? {
            Value::SuperTable(st) => Ok(Arc::unwrap_or_clone(st)),
            _ => Err(MinarrowError::TypeError {
                from: "SuperTable",
                to: "SuperTable",
                message: Some("Unexpected result type from subtraction".to_string()),
            }),
        }
    }
}

#[cfg(feature = "chunked")]
impl Mul for SuperTable {
    type Output = Result<SuperTable, MinarrowError>;
    fn mul(self, rhs: Self) -> Self::Output {
        match value_multiply(
            Value::SuperTable(Arc::new(self)),
            Value::SuperTable(Arc::new(rhs)),
        )? {
            Value::SuperTable(st) => Ok(Arc::unwrap_or_clone(st)),
            _ => Err(MinarrowError::TypeError {
                from: "SuperTable",
                to: "SuperTable",
                message: Some("Unexpected result type from multiplication".to_string()),
            }),
        }
    }
}

#[cfg(feature = "chunked")]
impl Div for SuperTable {
    type Output = Result<SuperTable, MinarrowError>;
    fn div(self, rhs: Self) -> Self::Output {
        match value_divide(
            Value::SuperTable(Arc::new(self)),
            Value::SuperTable(Arc::new(rhs)),
        )? {
            Value::SuperTable(st) => Ok(Arc::unwrap_or_clone(st)),
            _ => Err(MinarrowError::TypeError {
                from: "SuperTable",
                to: "SuperTable",
                message: Some("Unexpected result type from division".to_string()),
            }),
        }
    }
}

#[cfg(feature = "chunked")]
impl Rem for SuperTable {
    type Output = Result<SuperTable, MinarrowError>;
    fn rem(self, rhs: Self) -> Self::Output {
        match value_remainder(
            Value::SuperTable(Arc::new(self)),
            Value::SuperTable(Arc::new(rhs)),
        )? {
            Value::SuperTable(st) => Ok(Arc::unwrap_or_clone(st)),
            _ => Err(MinarrowError::TypeError {
                from: "SuperTable",
                to: "SuperTable",
                message: Some("Unexpected result type from remainder".to_string()),
            }),
        }
    }
}

// SuperTableView implementations
#[cfg(all(feature = "chunked", feature = "views"))]
impl Add for SuperTableV {
    type Output = Result<SuperTable, MinarrowError>;
    fn add(self, rhs: Self) -> Self::Output {
        match value_add(
            Value::SuperTableView(Arc::new(self)),
            Value::SuperTableView(Arc::new(rhs)),
        )? {
            Value::SuperTable(st) => Ok(Arc::unwrap_or_clone(st)),
            Value::SuperTableView(stv) => Ok(SuperTable::from_views(
                &Arc::unwrap_or_clone(stv).slices,
                "".to_string(),
            )),
            _ => Err(MinarrowError::TypeError {
                from: "SuperTableView",
                to: "SuperTable",
                message: Some("Unexpected result type from addition".to_string()),
            }),
        }
    }
}

#[cfg(all(feature = "chunked", feature = "views"))]
impl Sub for SuperTableV {
    type Output = Result<SuperTable, MinarrowError>;
    fn sub(self, rhs: Self) -> Self::Output {
        match value_subtract(
            Value::SuperTableView(Arc::new(self)),
            Value::SuperTableView(Arc::new(rhs)),
        )? {
            Value::SuperTable(st) => Ok(Arc::unwrap_or_clone(st)),
            Value::SuperTableView(stv) => Ok(SuperTable::from_views(
                &Arc::unwrap_or_clone(stv).slices,
                "".to_string(),
            )),
            _ => Err(MinarrowError::TypeError {
                from: "SuperTableView",
                to: "SuperTable",
                message: Some("Unexpected result type from subtraction".to_string()),
            }),
        }
    }
}

#[cfg(all(feature = "chunked", feature = "views"))]
impl Mul for SuperTableV {
    type Output = Result<SuperTable, MinarrowError>;
    fn mul(self, rhs: Self) -> Self::Output {
        match value_multiply(
            Value::SuperTableView(Arc::new(self)),
            Value::SuperTableView(Arc::new(rhs)),
        )? {
            Value::SuperTable(st) => Ok(Arc::unwrap_or_clone(st)),
            Value::SuperTableView(stv) => Ok(SuperTable::from_views(
                &Arc::unwrap_or_clone(stv).slices,
                "".to_string(),
            )),
            _ => Err(MinarrowError::TypeError {
                from: "SuperTableView",
                to: "SuperTable",
                message: Some("Unexpected result type from multiplication".to_string()),
            }),
        }
    }
}

#[cfg(all(feature = "chunked", feature = "views"))]
impl Div for SuperTableV {
    type Output = Result<SuperTable, MinarrowError>;
    fn div(self, rhs: Self) -> Self::Output {
        match value_divide(
            Value::SuperTableView(Arc::new(self)),
            Value::SuperTableView(Arc::new(rhs)),
        )? {
            Value::SuperTable(st) => Ok(Arc::unwrap_or_clone(st)),
            Value::SuperTableView(stv) => Ok(SuperTable::from_views(
                &Arc::unwrap_or_clone(stv).slices,
                "".to_string(),
            )),
            _ => Err(MinarrowError::TypeError {
                from: "SuperTableView",
                to: "SuperTable",
                message: Some("Unexpected result type from division".to_string()),
            }),
        }
    }
}

#[cfg(all(feature = "chunked", feature = "views"))]
impl Rem for SuperTableV {
    type Output = Result<SuperTable, MinarrowError>;
    fn rem(self, rhs: Self) -> Self::Output {
        match value_remainder(
            Value::SuperTableView(Arc::new(self)),
            Value::SuperTableView(Arc::new(rhs)),
        )? {
            Value::SuperTable(st) => Ok(Arc::unwrap_or_clone(st)),
            Value::SuperTableView(stv) => Ok(SuperTable::from_views(
                &Arc::unwrap_or_clone(stv).slices,
                "".to_string(),
            )),
            _ => Err(MinarrowError::TypeError {
                from: "SuperTableView",
                to: "SuperTable",
                message: Some("Unexpected result type from remainder".to_string()),
            }),
        }
    }
}

// Cube implementations
#[cfg(feature = "cube")]
impl Add for Cube {
    type Output = Result<Cube, MinarrowError>;
    fn add(self, rhs: Self) -> Self::Output {
        match value_add(Value::Cube(Arc::new(self)), Value::Cube(Arc::new(rhs)))? {
            Value::Cube(c) => Ok(Arc::unwrap_or_clone(c)),
            _ => Err(MinarrowError::TypeError {
                from: "Cube",
                to: "Cube",
                message: Some("Unexpected result type from addition".to_string()),
            }),
        }
    }
}

#[cfg(feature = "cube")]
impl Sub for Cube {
    type Output = Result<Cube, MinarrowError>;
    fn sub(self, rhs: Self) -> Self::Output {
        match value_subtract(Value::Cube(Arc::new(self)), Value::Cube(Arc::new(rhs)))? {
            Value::Cube(c) => Ok(Arc::unwrap_or_clone(c)),
            _ => Err(MinarrowError::TypeError {
                from: "Cube",
                to: "Cube",
                message: Some("Unexpected result type from subtraction".to_string()),
            }),
        }
    }
}

#[cfg(feature = "cube")]
impl Mul for Cube {
    type Output = Result<Cube, MinarrowError>;
    fn mul(self, rhs: Self) -> Self::Output {
        match value_multiply(Value::Cube(Arc::new(self)), Value::Cube(Arc::new(rhs)))? {
            Value::Cube(c) => Ok(Arc::unwrap_or_clone(c)),
            _ => Err(MinarrowError::TypeError {
                from: "Cube",
                to: "Cube",
                message: Some("Unexpected result type from multiplication".to_string()),
            }),
        }
    }
}

#[cfg(feature = "cube")]
impl Div for Cube {
    type Output = Result<Cube, MinarrowError>;
    fn div(self, rhs: Self) -> Self::Output {
        match value_divide(Value::Cube(Arc::new(self)), Value::Cube(Arc::new(rhs)))? {
            Value::Cube(c) => Ok(Arc::unwrap_or_clone(c)),
            _ => Err(MinarrowError::TypeError {
                from: "Cube",
                to: "Cube",
                message: Some("Unexpected result type from division".to_string()),
            }),
        }
    }
}

#[cfg(feature = "cube")]
impl Rem for Cube {
    type Output = Result<Cube, MinarrowError>;
    fn rem(self, rhs: Self) -> Self::Output {
        match value_remainder(Value::Cube(Arc::new(self)), Value::Cube(Arc::new(rhs)))? {
            Value::Cube(c) => Ok(Arc::unwrap_or_clone(c)),
            _ => Err(MinarrowError::TypeError {
                from: "Cube",
                to: "Cube",
                message: Some("Unexpected result type from remainder".to_string()),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::selection::ColumnSelection;
    use crate::{Array, IntegerArray, NumericArray, vec64};

    #[test]
    fn test_value_addition() {
        let arr1 =
            Value::Array(Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3])).into());
        let arr2 =
            Value::Array(Array::from_int32(IntegerArray::from_slice(&vec64![4, 5, 6])).into());

        let result = (arr1 + arr2).unwrap();

        if let Value::Array(arr) = result {
            let arr = Arc::unwrap_or_clone(arr);
            if let Array::NumericArray(NumericArray::Int32(result_arr)) = arr {
                assert_eq!(result_arr.data.as_slice(), &[5, 7, 9]);
            } else {
                panic!("Expected Int32 array result");
            }
        } else {
            panic!("Expected Array value");
        }
    }

    #[test]
    fn test_all_arithmetic_operators() {
        let arr1 =
            Value::Array(Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30])).into());
        let arr2 =
            Value::Array(Array::from_int32(IntegerArray::from_slice(&vec64![2, 4, 6])).into());

        // Test addition using operator
        let sum = (&arr1 + &arr2).unwrap();
        if let Value::Array(arr) = sum {
            if let Array::NumericArray(NumericArray::Int32(int_arr)) = arr.as_ref() {
                assert_eq!(int_arr.data.as_slice(), &[12, 24, 36]);
            }
        }

        // Test subtraction using operator
        let diff = (&arr1 - &arr2).unwrap();
        if let Value::Array(arr) = diff {
            if let Array::NumericArray(NumericArray::Int32(int_arr)) = arr.as_ref() {
                assert_eq!(int_arr.data.as_slice(), &[8, 16, 24]);
            }
        }

        // Test multiplication using operator
        let prod = (&arr1 * &arr2).unwrap();
        if let Value::Array(arr) = prod {
            if let Array::NumericArray(NumericArray::Int32(int_arr)) = arr.as_ref() {
                assert_eq!(int_arr.data.as_slice(), &[20, 80, 180]);
            }
        }

        // Test division using operator
        let quot = (&arr1 / &arr2).unwrap();
        if let Value::Array(arr) = quot {
            if let Array::NumericArray(NumericArray::Int32(int_arr)) = arr.as_ref() {
                assert_eq!(int_arr.data.as_slice(), &[5, 5, 5]);
            }
        }

        // Test remainder using operator
        let rem = (&arr1 % &arr2).unwrap();
        if let Value::Array(arr) = rem {
            if let Array::NumericArray(NumericArray::Int32(int_arr)) = arr.as_ref() {
                assert_eq!(int_arr.data.as_slice(), &[0, 0, 0]);
            }
        }
    }

    #[cfg(feature = "scalar_type")]
    #[test]
    fn test_scalar_array_addition() {
        use crate::Scalar;

        let scalar = Value::Scalar(Scalar::Int32(5));
        let array =
            Value::Array(Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3])).into());

        let result = (scalar + array).unwrap();

        if let Value::Array(arr) = result {
            let arr = Arc::unwrap_or_clone(arr);
            if let Array::NumericArray(NumericArray::Int32(result_arr)) = arr {
                assert_eq!(result_arr.data.as_slice(), &[6, 7, 8]);
            } else {
                panic!("Expected Int32 array result");
            }
        } else {
            panic!("Expected Array value");
        }
    }

    #[test]
    fn test_reference_operations() {
        let arr1 =
            Value::Array(Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30])).into());
        let arr2 =
            Value::Array(Array::from_int32(IntegerArray::from_slice(&vec64![2, 4, 5])).into());

        let result = (&arr1 / &arr2).unwrap();

        if let Value::Array(arr) = result {
            let arr = Arc::unwrap_or_clone(arr);
            if let Array::NumericArray(NumericArray::Int32(result_arr)) = arr {
                assert_eq!(result_arr.data.as_slice(), &[5, 5, 6]);
            } else {
                panic!("Expected Int32 array result");
            }
        } else {
            panic!("Expected Array value");
        }
    }

    #[test]
    fn test_broadcasting() {
        let single = Value::Array(Array::from_int32(IntegerArray::from_slice(&vec64![10])).into());
        let array = Value::Array(
            Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3, 4, 5])).into(),
        );

        let result = (single * array).unwrap();

        if let Value::Array(arr) = result {
            let arr = Arc::unwrap_or_clone(arr);
            if let Array::NumericArray(NumericArray::Int32(result_arr)) = arr {
                assert_eq!(result_arr.data.as_slice(), &[10, 20, 30, 40, 50]);
            } else {
                panic!("Expected Int32 array result");
            }
        } else {
            panic!("Expected Array value");
        }
    }

    #[test]
    fn test_value_table_addition() {
        // Create first table: columns [1, 2, 3] and [10, 20, 30]
        let col1_a = FieldArray::from_arr(
            "col1",
            Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3])),
        );
        let col2_a = FieldArray::from_arr(
            "col2",
            Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30])),
        );
        let table_a = Value::Table(Arc::new(Table::new(
            "tableA".to_string(),
            Some(vec![col1_a, col2_a]),
        )));

        // Create second table: columns [4, 5, 6] and [40, 50, 60]
        let col1_b = FieldArray::from_arr(
            "col1",
            Array::from_int32(IntegerArray::from_slice(&vec64![4, 5, 6])),
        );
        let col2_b = FieldArray::from_arr(
            "col2",
            Array::from_int32(IntegerArray::from_slice(&vec64![40, 50, 60])),
        );
        let table_b = Value::Table(Arc::new(Table::new(
            "tableB".to_string(),
            Some(vec![col1_b, col2_b]),
        )));

        // Perform addition
        let result = (table_a + table_b).unwrap();

        // Verify the result
        if let Value::Table(result_table) = result {
            assert_eq!(result_table.n_cols(), 2);
            assert_eq!(result_table.n_rows(), 3);
            assert_eq!(result_table.name, "tableA"); // Takes name from left operand

            // Check first column: [1,2,3] + [4,5,6] = [5,7,9]
            if let Some(col1) = result_table.col_ix(0) {
                if let Array::NumericArray(NumericArray::Int32(arr)) = &col1.array {
                    assert_eq!(arr.data.as_slice(), &[5, 7, 9]);
                } else {
                    panic!("Expected Int32 array in first column");
                }
            } else {
                panic!("Could not get first column");
            }

            // Check second column: [10,20,30] + [40,50,60] = [50,70,90]
            if let Some(col2) = result_table.col_ix(1) {
                if let Array::NumericArray(NumericArray::Int32(arr)) = &col2.array {
                    assert_eq!(arr.data.as_slice(), &[50, 70, 90]);
                } else {
                    panic!("Expected Int32 array in second column");
                }
            } else {
                panic!("Could not get second column");
            }
        } else {
            panic!("Expected Value::Table result");
        }
    }

    #[cfg(feature = "views")]
    #[test]
    fn test_value_table_view_addition() {
        use crate::TableV;

        // Create first table: columns [1, 2, 3] and [10, 20, 30]
        let col1_a = FieldArray::from_arr(
            "col1",
            Array::from_int32(IntegerArray::from_slice(&vec64![1, 2, 3])),
        );
        let col2_a = FieldArray::from_arr(
            "col2",
            Array::from_int32(IntegerArray::from_slice(&vec64![10, 20, 30])),
        );
        let table_a = Table::new("tableA".to_string(), Some(vec![col1_a, col2_a]));
        let table_view_a = Value::TableView(Arc::new(TableV::from(table_a)));

        // Create second table: columns [4, 5, 6] and [40, 50, 60]
        let col1_b = FieldArray::from_arr(
            "col1",
            Array::from_int32(IntegerArray::from_slice(&vec64![4, 5, 6])),
        );
        let col2_b = FieldArray::from_arr(
            "col2",
            Array::from_int32(IntegerArray::from_slice(&vec64![40, 50, 60])),
        );
        let table_b = Table::new("tableB".to_string(), Some(vec![col1_b, col2_b]));
        let table_view_b = Value::TableView(Arc::new(TableV::from(table_b)));

        // Perform addition
        let result = (table_view_a + table_view_b).unwrap();

        // Verify the result (should be a materialised Table)
        if let Value::Table(result_table) = result {
            assert_eq!(result_table.n_cols(), 2);
            assert_eq!(result_table.n_rows(), 3);

            // Check first column: [1,2,3] + [4,5,6] = [5,7,9]
            if let Some(col1) = result_table.col_ix(0) {
                if let Array::NumericArray(NumericArray::Int32(arr)) = &col1.array {
                    assert_eq!(arr.data.as_slice(), &[5, 7, 9]);
                } else {
                    panic!("Expected Int32 array in first column");
                }
            }
        } else {
            panic!("Expected Value::Table result");
        }
    }
}
