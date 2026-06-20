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

//! # Broadcasting Operations Module
//!
//! Provides high-level broadcasting operations for arithmetic operations
//! with automatic scalar expansion and type promotion.
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
//!
pub mod array;
pub mod array_view;
#[cfg(feature = "cube")]
pub mod cube;
pub mod field_array;
#[cfg(feature = "matrix")]
pub mod matrix;
pub mod scalar;
pub mod super_array;
pub mod super_array_view;
pub mod super_table;
pub mod super_table_view;
pub mod table;
pub mod table_view;

#[cfg(feature = "chunked")]
use crate::utils::create_aligned_chunks_from_array;
pub use table::{broadcast_super_table_add, broadcast_table_add};

// Import helper functions from submodules
#[cfg(feature = "scalar_type")]
use crate::kernels::routing::arithmetic::scalar_arithmetic;
#[cfg(all(feature = "chunked", feature = "views"))]
use array::broadcast_array_to_supertableview;
use array::broadcast_array_to_table;
#[cfg(feature = "views")]
use array_view::{
    broadcast_arrayview_to_supertableview, broadcast_arrayview_to_table,
    broadcast_arrayview_to_tableview,
};
#[cfg(all(feature = "scalar_type", feature = "chunked", feature = "views"))]
use scalar::broadcast_scalar_to_supertableview;
#[cfg(feature = "scalar_type")]
use scalar::broadcast_scalar_to_table;
#[cfg(all(feature = "scalar_type", feature = "views"))]
use scalar::broadcast_scalar_to_tableview;
#[cfg(feature = "chunked")]
use super_array::broadcast_superarray_to_table;
use super_array::route_super_array_broadcast;
#[cfg(all(feature = "chunked", feature = "views"))]
use super_array_view::broadcast_superarrayview_to_tableview;
#[cfg(feature = "chunked")]
use super_table::broadcast_super_table_with_operator;
#[cfg(all(feature = "scalar_type", feature = "chunked", feature = "views"))]
use super_table_view::broadcast_supertableview_to_scalar;
#[cfg(all(feature = "chunked", feature = "views"))]
use super_table_view::{
    broadcast_superarrayview_to_table, broadcast_supertableview_to_array,
    broadcast_supertableview_to_arrayview,
};
#[cfg(feature = "views")]
use table::broadcast_table_to_arrayview;
use table::{broadcast_table_to_array, broadcast_table_to_scalar, broadcast_table_with_operator};
#[cfg(feature = "chunked")]
use table::{broadcast_table_to_superarray, broadcast_table_to_superarrayview};
#[cfg(all(feature = "scalar_type", feature = "views"))]
use table_view::broadcast_tableview_to_scalar;
#[cfg(all(feature = "chunked", feature = "views"))]
use table_view::broadcast_tableview_to_superarrayview;
#[cfg(feature = "views")]
use table_view::{broadcast_tableview_to_arrayview, broadcast_tableview_to_tableview};

#[cfg(feature = "cube")]
use crate::Cube;

#[cfg(feature = "views")]
use crate::ArrayV;
use crate::{Array, FloatArray, IntegerArray, StringArray};

#[cfg(feature = "scalar_type")]
use crate::Scalar;
#[cfg(all(feature = "views", feature = "chunked"))]
use crate::SuperTableV;
use crate::enums::error::MinarrowError;
use crate::enums::operators::ArithmeticOperator;
use crate::enums::value::Value;
#[cfg(feature = "chunked")]
use crate::{SuperArray, SuperTable};

use crate::kernels::routing::arithmetic::resolve_binary_arithmetic;

/// Add two Values with automatic broadcasting
pub fn value_add(lhs: Value, rhs: Value) -> Result<Value, MinarrowError> {
    broadcast_value(ArithmeticOperator::Add, lhs, rhs)
}

/// Subtract two Values with automatic broadcasting
pub fn value_subtract(lhs: Value, rhs: Value) -> Result<Value, MinarrowError> {
    broadcast_value(ArithmeticOperator::Subtract, lhs, rhs)
}

/// Multiply two Values with automatic broadcasting
pub fn value_multiply(lhs: Value, rhs: Value) -> Result<Value, MinarrowError> {
    broadcast_value(ArithmeticOperator::Multiply, lhs, rhs)
}

/// Divide two Values with automatic broadcasting
pub fn value_divide(lhs: Value, rhs: Value) -> Result<Value, MinarrowError> {
    broadcast_value(ArithmeticOperator::Divide, lhs, rhs)
}

/// Remainder (modulo) two Values with automatic broadcasting
pub fn value_remainder(lhs: Value, rhs: Value) -> Result<Value, MinarrowError> {
    broadcast_value(ArithmeticOperator::Remainder, lhs, rhs)
}

/// Power/exponentiation of two Values with automatic broadcasting
pub fn value_power(lhs: Value, rhs: Value) -> Result<Value, MinarrowError> {
    broadcast_value(ArithmeticOperator::Power, lhs, rhs)
}

/// Implementation of Add operation for Value enum following the unified pattern
///
/// # Notes:
/// 1.⚠️ Best to keep this out of the binary by disabling value_type unless you
/// require universal broadcasting compatibility.
/// 2.These do not yet implement parallel processing to speed up broadcasting.
#[cfg(all(feature = "scalar_type", feature = "value_type"))]
pub fn broadcast_value(
    op: ArithmeticOperator,
    lhs: Value,
    rhs: Value,
) -> Result<Value, MinarrowError> {
    use std::sync::Arc;
    match (lhs, rhs) {
        // Scalar + Scalar = Scalar
        #[cfg(feature = "scalar_type")]
        (Value::Scalar(l), Value::Scalar(r)) => {
            scalar_arithmetic(l, r, ArithmeticOperator::Add).map(Value::Scalar)
        }

        // Array types - use resolve_binary_arithmetic
        (Value::Array(l), Value::Array(r)) => {
            resolve_binary_arithmetic(op, (*l).clone(), (*r).clone(), None) // no null mask uses the union
                .map(|arr| Value::Array(Arc::new(arr)))
        }

        #[cfg(feature = "views")]
        (Value::ArrayView(l), Value::ArrayView(r)) => resolve_binary_arithmetic(op, (*l).clone(), (*r).clone(), None)
            .map(|arr| Value::Array(Arc::new(arr))),

        // Standard field array
        (Value::FieldArray(l), Value::FieldArray(r)) => resolve_binary_arithmetic(op, (*l).clone(), (*r).clone(), None)
            .map(|arr| Value::Array(Arc::new(arr)))
            ,

        // Mixed Array and FieldArray combinations
        (Value::Array(l), Value::FieldArray(r)) => resolve_binary_arithmetic(op, (*l).clone(), (*r).clone(), None)
            .map(|arr| Value::Array(Arc::new(arr)))
            ,

        (Value::FieldArray(l), Value::Array(r)) => resolve_binary_arithmetic(op, (*l).clone(), (*r).clone(), None)
            .map(|arr| Value::Array(Arc::new(arr)))
            ,

        // Mixed Array and ArrayView combinations
        #[cfg(feature = "views")]
        (Value::Array(l), Value::ArrayView(r)) => resolve_binary_arithmetic(op, (*l).clone(), (*r).clone(), None)
            .map(|arr| Value::Array(Arc::new(arr)))
            ,

        #[cfg(feature = "views")]
        (Value::ArrayView(l), Value::Array(r)) => resolve_binary_arithmetic(op, (*l).clone(), (*r).clone(), None)
            .map(|arr| Value::Array(Arc::new(arr)))
            ,

        // Mixed FieldArray and ArrayView combinations
        #[cfg(feature = "views")]
        (Value::FieldArray(l), Value::ArrayView(r)) => resolve_binary_arithmetic(op, (*l).clone(), (*r).clone(), None)
            .map(|arr| Value::Array(Arc::new(arr)))
            ,

        #[cfg(feature = "views")]
        (Value::ArrayView(l), Value::FieldArray(r)) => resolve_binary_arithmetic(op, (*l).clone(), (*r).clone(), None)
            .map(|arr| Value::Array(Arc::new(arr)))
            ,

        // Scalar broadcasting with Array types
        #[cfg(feature = "scalar_type")]
        (Value::Scalar(l), Value::Array(r)) => {
            scalar::broadcast_scalar_to_array(op, &l, &r).map(|arr| Value::Array(Arc::new(arr)))
        }

        #[cfg(feature = "scalar_type")]
        (Value::Array(l), Value::Scalar(r)) => {
            array::broadcast_array_to_scalar(op, &l, &r).map(|arr| Value::Array(Arc::new(arr)))
        }

        #[cfg(all(feature = "scalar_type"))]
        (Value::Scalar(l), Value::FieldArray(r)) => {
            scalar::broadcast_scalar_to_fieldarray(op, &l, &r).map(|arr| Value::Array(Arc::new(arr)))
        }

        #[cfg(all(feature = "scalar_type"))]
        (Value::FieldArray(l), Value::Scalar(r)) => {
            scalar::broadcast_fieldarray_to_scalar(op, &l, &r).map(|arr| Value::Array(Arc::new(arr)))
        }

        // Scalar with SuperArray types - convert scalar to array then broadcast
        #[cfg(all(feature = "scalar_type", feature = "chunked"))]
        (Value::Scalar(l), Value::SuperArray(r)) => {
            scalar::broadcast_scalar_to_superarray(op, &l, &*r).map(|sa| Value::SuperArray(Arc::new(sa)))
        }

        #[cfg(all(feature = "scalar_type", feature = "chunked"))]
        (Value::SuperArray(l), Value::Scalar(r)) => {
            super_array::broadcast_superarray_to_scalar(op, &*l, &r).map(|sa| Value::SuperArray(Arc::new(sa)))
        }

        #[cfg(all(feature = "scalar_type", feature = "chunked", feature = "views"))]
        (Value::Scalar(l), Value::SuperArrayView(r)) => {
            scalar::broadcast_scalar_to_superarrayview(op, &l, &*r).map(|sa| Value::SuperArray(Arc::new(sa)))
        }

        #[cfg(all(feature = "scalar_type", feature = "chunked", feature = "views"))]
        (Value::SuperArrayView(l), Value::Scalar(r)) => {
            super_array::broadcast_superarrayview_to_scalar(op, &*l, &r).map(|sa| Value::SuperArray(Arc::new(sa)))
        }

        // SuperArray types - use broadcast_super_array_add
        #[cfg(feature = "chunked")]
        (Value::SuperArray(l), Value::SuperArray(r)) => {
            let l_val = Arc::unwrap_or_clone(l);
            let r_val = Arc::unwrap_or_clone(r);
            route_super_array_broadcast(op, l_val, r_val, None)
                .map(|sa| Value::SuperArray(Arc::new(sa)))
        }

        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::SuperArrayView(l), Value::SuperArrayView(r)) => {
            let l_val = Arc::unwrap_or_clone(l);
            let r_val = Arc::unwrap_or_clone(r);
            route_super_array_broadcast(op, l_val, r_val, None)
                .map(|sa| Value::SuperArray(Arc::new(sa)))
        }

        // Mixed SuperArray and SuperArrayView combinations - Convert views to owned
        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::SuperArray(l), Value::SuperArrayView(r)) => {
            let l_val = Arc::unwrap_or_clone(l);
            let r_owned = SuperArray::from_slices(&r.slices, r.field.clone()); // Convert view to owned
            route_super_array_broadcast(op, l_val, r_owned, None)
                .map(|sa| Value::SuperArray(Arc::new(sa)))
        }

        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::SuperArrayView(l), Value::SuperArray(r)) => {
            let r_val = Arc::unwrap_or_clone(r);
            let l_owned = SuperArray::from_slices(&l.slices, l.field.clone()); // Convert view to owned
            route_super_array_broadcast(op, l_owned, r_val, None)
                .map(|sa| Value::SuperArray(Arc::new(sa)))
        }

        // ArcValue cases
        (Value::ArcValue(l), Value::ArcValue(r)) => {
            let l_val = Arc::try_unwrap(l).unwrap_or_else(|arc| (*arc).clone());
            let r_val = Arc::try_unwrap(r).unwrap_or_else(|arc| (*arc).clone());
            broadcast_value(op, l_val, r_val).map(|v| Value::ArcValue(Arc::new(v)))
        }

        (Value::ArcValue(l), r) => {
            let l_val = Arc::try_unwrap(l).unwrap_or_else(|arc| (*arc).clone());
            broadcast_value(op, l_val, r).map(|v| Value::ArcValue(Arc::new(v)))
        }

        (l, Value::ArcValue(r)) => {
            let r_val = Arc::try_unwrap(r).unwrap_or_else(|arc| (*arc).clone());
            broadcast_value(op, l, r_val).map(|v| Value::ArcValue(Arc::new(v)))
        }

        // Tuple cases - apply operation element-wise
        (Value::Tuple2(l_arc), Value::Tuple2(r_arc)) => {
            let (l1, l2) = (l_arc.0.clone(), l_arc.1.clone());
            let (r1, r2) = (r_arc.0.clone(), r_arc.1.clone());
            let res1 = broadcast_value(op, l1, r1)?;
            let res2 = broadcast_value(op, l2, r2)?;
            Ok(Value::Tuple2(Arc::new((res1, res2))))
        }

        (Value::Tuple3(l_arc), Value::Tuple3(r_arc)) => {
            let (l1, l2, l3) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone());
            let (r1, r2, r3) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone());
            let res1 = broadcast_value(op, l1, r1)?;
            let res2 = broadcast_value(op, l2, r2)?;
            let res3 = broadcast_value(op, l3, r3)?;
            Ok(Value::Tuple3(Arc::new((res1, res2, res3))))
        }

        (Value::Tuple4(l_arc), Value::Tuple4(r_arc)) => {
            let (l1, l2, l3, l4) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone());
            let (r1, r2, r3, r4) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone());
            let res1 = broadcast_value(op, l1, r1)?;
            let res2 = broadcast_value(op, l2, r2)?;
            let res3 = broadcast_value(op, l3, r3)?;
            let res4 = broadcast_value(op, l4, r4)?;
            Ok(Value::Tuple4(Arc::new((res1, res2, res3, res4))))
        }

        (Value::Tuple5(l_arc), Value::Tuple5(r_arc)) => {
            let (l1, l2, l3, l4, l5) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone(), l_arc.4.clone());
            let (r1, r2, r3, r4, r5) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone(), r_arc.4.clone());
            let res1 = broadcast_value(op, l1, r1)?;
            let res2 = broadcast_value(op, l2, r2)?;
            let res3 = broadcast_value(op, l3, r3)?;
            let res4 = broadcast_value(op, l4, r4)?;
            let res5 = broadcast_value(op, l5, r5)?;
            Ok(Value::Tuple5(Arc::new((res1, res2, res3, res4, res5))))
        }

        (Value::Tuple6(l_arc), Value::Tuple6(r_arc)) => {
            let (l1, l2, l3, l4, l5, l6) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone(), l_arc.4.clone(), l_arc.5.clone());
            let (r1, r2, r3, r4, r5, r6) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone(), r_arc.4.clone(), r_arc.5.clone());
            let res1 = broadcast_value(op, l1, r1)?;
            let res2 = broadcast_value(op, l2, r2)?;
            let res3 = broadcast_value(op, l3, r3)?;
            let res4 = broadcast_value(op, l4, r4)?;
            let res5 = broadcast_value(op, l5, r5)?;
            let res6 = broadcast_value(op, l6, r6)?;
            Ok(Value::Tuple6(Arc::new((res1, res2, res3, res4, res5, res6))))
        }

        // VecValue case - apply operation element-wise
        (Value::VecValue(l_vec), Value::VecValue(r_vec)) => {
            if l_vec.len() != r_vec.len() {
                return Err(MinarrowError::ColumnLengthMismatch {
                    col: 0,
                    expected: l_vec.len(),
                    found: r_vec.len(),
                });
            }
            let results: Result<Vec<Value>, MinarrowError> = Arc::unwrap_or_clone(l_vec)
                .into_iter()
                .zip(Arc::unwrap_or_clone(r_vec).into_iter())
                .map(|(l, r)| broadcast_value(op, l, r))
                .collect();
            results.map(|v| Value::VecValue(Arc::new(v)))
        }

        // Matrix broadcasting operations
        #[cfg(feature = "matrix")]
        (Value::Matrix(l), Value::Matrix(r)) => {
           matrix::broadcast_matrix_add(l, r)
                .map(|mat| Value::Matrix(Arc::new(mat)))
        }

        // Matrix with scalar broadcasting
        #[cfg(all(feature = "matrix", feature = "scalar_type"))]
        (Value::Matrix(l), Value::Scalar(r)) => {
           matrix::broadcast_matrix_scalar_add(l, r)
                .map(|mat| Value::Matrix(Arc::new(mat)))
        }

        #[cfg(all(feature = "matrix", feature = "scalar_type"))]
        (Value::Scalar(l), Value::Matrix(r)) => {
           matrix::broadcast_scalar_matrix_add(l, r)
                .map(|mat| Value::Matrix(Arc::new(mat)))
        }

        // Matrix with Array broadcasting
        #[cfg(feature = "matrix")]
        (Value::Matrix(l), Value::Array(r)) => {
           matrix::broadcast_matrix_array_add(l, r)
        }

        #[cfg(feature = "matrix")]
        (Value::Array(l), Value::Matrix(r)) => {
           matrix::broadcast_array_matrix_add(l, r)
        }

        // Matrix with other complex types - return specific error
        #[cfg(feature = "matrix")]
        (Value::Matrix(_), _) | (_, Value::Matrix(_)) => Err(MinarrowError::TypeError {
            from: "Matrix",
            to: "compatible broadcasting type",
            message: Some(
                "Matrix can only be broadcast with Matrix, Scalar, or Array types".to_string(),
            ),
        }),

        // Cube cases - use broadcast_cube_add
        #[cfg(feature = "cube")]
        (Value::Cube(l), Value::Cube(r)) => {
            let l_val = Arc::unwrap_or_clone(l);
            let r_val = Arc::unwrap_or_clone(r);
            cube::broadcast_cube_add(l_val, r_val, None)
                .map(|cube| Value::Cube(Arc::new(cube)))
                .map_err(|e| MinarrowError::KernelError(Some(e.to_string())))
        }

        // Table cases - use broadcast_table_add
        (Value::Table(l), Value::Table(r)) => {
            let l_val = Arc::unwrap_or_clone(l);
            let r_val = Arc::unwrap_or_clone(r);
            broadcast_table_with_operator(op, l_val, r_val)
                .map(|tbl| Value::Table(Arc::new(tbl)))
        }

        // Use the optimised TableView broadcasting function
        #[cfg(feature = "views")]
        (Value::TableView(l), Value::TableView(r)) => {
            broadcast_tableview_to_tableview(op, &l, &r)
                .map(|tbl| Value::Table(Arc::new(tbl))) // Result is always a Table (materialised)
        }

        #[cfg(feature = "views")]
        (Value::Table(l), Value::TableView(r)) => {
            let l_val = Arc::unwrap_or_clone(l);
            broadcast_table_with_operator(op, l_val, r.to_table())
                .map(|tbl| Value::Table(Arc::new(tbl)))
        }

        #[cfg(feature = "views")]
        (Value::TableView(l), Value::Table(r)) => {
            let r_val = Arc::unwrap_or_clone(r);
            broadcast_table_with_operator(op, l.to_table(), r_val)
                .map(|tbl| Value::Table(Arc::new(tbl)))
        }

        // SuperTable cases - use broadcast_super_table_add
        #[cfg(feature = "chunked")]
        (Value::SuperTable(l), Value::SuperTable(r)) => {
            let l_val = Arc::unwrap_or_clone(l);
            let r_val = Arc::unwrap_or_clone(r);
            broadcast_super_table_with_operator(op, l_val, r_val)
                .map(|st| Value::SuperTable(Arc::new(st)))
        }

        // SuperTableView cases - use broadcast_super_table_add
        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::SuperTableView(l), Value::SuperTableView(r)) => {
            let l_val = Arc::unwrap_or_clone(l);
            let r_val = Arc::unwrap_or_clone(r);
            broadcast_super_table_with_operator(op, l_val, r_val)
                .map(|st| Value::SuperTable(Arc::new(st))) // Result is always materialised SuperTable
        }

        // Mixed SuperTable and SuperTableView cases
        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::SuperTable(l), Value::SuperTableView(r)) => {
            let l_view: SuperTableV = Arc::unwrap_or_clone(l).into();
            let r_unwrapped = Arc::unwrap_or_clone(r);
            broadcast_super_table_with_operator(op, l_view, r_unwrapped)
                .map(|st| Value::SuperTable(Arc::new(st)))
        }

        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::SuperTableView(l), Value::SuperTable(r)) => {
            let l_unwrapped = Arc::unwrap_or_clone(l);
            let r_view: SuperTableV = Arc::unwrap_or_clone(r).into();
            broadcast_super_table_with_operator(op, l_unwrapped, r_view)
                .map(|st| Value::SuperTable(Arc::new(st)))
        }

        // Scalar with higher-order structures

        // Scalar-Table broadcasting - column-wise application
        #[cfg(feature = "scalar_type")]
        (Value::Scalar(scalar), Value::Table(table)) => {
            broadcast_scalar_to_table(op, &scalar, &table).map(|tbl| Value::Table(Arc::new(tbl)))
        },
        #[cfg(feature = "scalar_type")]
        (Value::Table(table), Value::Scalar(scalar)) => {
            broadcast_table_to_scalar(op, &table, &scalar).map(|tbl| Value::Table(Arc::new(tbl)))
        },

        // Scalar-SuperTable broadcasting - apply to each table in SuperTable
        #[cfg(feature = "scalar_type")]
        (Value::Scalar(scalar), Value::SuperTable(super_table)) => {
            scalar::broadcast_scalar_to_supertable(op, &scalar, &super_table).map(|st| Value::SuperTable(Arc::new(st)))
        },
        #[cfg(feature = "scalar_type")]
        (Value::SuperTable(super_table), Value::Scalar(scalar)) => {
            super_table::broadcast_supertable_to_scalar(op, &super_table, &scalar).map(|st| Value::SuperTable(Arc::new(st)))
        },

        // Scalar-TableView broadcasting - convert views to tables and broadcast
        #[cfg(all(feature = "scalar_type", feature = "views"))]
        (Value::Scalar(scalar), Value::TableView(table_view)) => {
            broadcast_scalar_to_tableview(op, &scalar, &table_view).map(|tbl| Value::Table(Arc::new(tbl)))
        },
        #[cfg(all(feature = "scalar_type", feature = "views"))]
        (Value::TableView(table_view), Value::Scalar(scalar)) => {
            broadcast_tableview_to_scalar(op, &table_view, &scalar).map(|tbl| Value::Table(Arc::new(tbl)))
        },

        // Scalar-SuperTableView broadcasting - convert views to tables and broadcast
        #[cfg(all(feature = "scalar_type", feature = "chunked", feature = "views"))]
        (Value::Scalar(scalar), Value::SuperTableView(super_table_view)) => {
            broadcast_scalar_to_supertableview(op, &scalar, &*super_table_view).map(|st| Value::SuperTableView(Arc::new(st)))
        },
        #[cfg(all(feature = "scalar_type", feature = "chunked", feature = "views"))]
        (Value::SuperTableView(super_table_view), Value::Scalar(scalar)) => {
            broadcast_supertableview_to_scalar(op, &*super_table_view, &scalar).map(|st| Value::SuperTableView(Arc::new(st)))
        },

        // Scalar-Cube broadcasting: apply scalar to each table in the cube
        #[cfg(all(feature = "scalar_type", feature = "cube"))]
        (Value::Scalar(scalar), Value::Cube(cube)) => {
            scalar::broadcast_scalar_to_cube(op, &scalar, &cube).map(|cube| Value::Cube(Arc::new(cube)))
        }

        #[cfg(all(feature = "scalar_type", feature = "cube"))]
        (Value::Cube(cube), Value::Scalar(scalar)) => {
            cube::broadcast_cube_to_scalar(op, &cube, &scalar).map(|cube| Value::Cube(Arc::new(cube)))
        }

        // Custom value combinations - not supported
        (Value::Custom(_), _) | (_, Value::Custom(_)) => {
            panic!("Custom types do not support broadcasting operations./<<")
        }

        // Cross-hierarchy combinations that aren't directly supported
        // These would require explicit conversion or promotion

        // Array-Table broadcasting - column-wise application
        (Value::Array(array), Value::Table(table)) => {
            broadcast_array_to_table(op, &array, &table).map(|tbl| Value::Table(Arc::new(tbl)))
        },
        (Value::Table(table), Value::Array(array)) => {
            broadcast_table_to_array(op, &table, &array).map(|tbl| Value::Table(Arc::new(tbl)))
        },

        // Array-SuperTable broadcasting - apply to each table in SuperTable
        (Value::Array(array), Value::SuperTable(super_table)) => {
            array::broadcast_array_to_supertable(op, &array, &super_table).map(|st| Value::SuperTable(Arc::new(st)))
        },
        (Value::SuperTable(super_table), Value::Array(array)) => {
            super_table::broadcast_supertable_to_array(op, &super_table, &array).map(|st| Value::SuperTable(Arc::new(st)))
        },

        // FieldArray-Table broadcasting - extract array and broadcast
        (Value::FieldArray(field_array), Value::Table(table)) => {
            broadcast_array_to_table(op, &field_array.array, &table).map(|tbl| Value::Table(Arc::new(tbl)))
        },
        (Value::Table(table), Value::FieldArray(field_array)) => {
            broadcast_table_to_array(op, &table, &field_array.array).map(|tbl| Value::Table(Arc::new(tbl)))
        },

        // FieldArray-SuperTable broadcasting - apply to each table in SuperTable
        (Value::FieldArray(field_array), Value::SuperTable(super_table)) => {
            super_table::broadcast_fieldarray_to_supertable(op, &field_array, &super_table).map(|st| Value::SuperTable(Arc::new(st)))
        },
        (Value::SuperTable(super_table), Value::FieldArray(field_array)) => {
            super_table::broadcast_supertable_to_fieldarray(op, &super_table, &field_array).map(|st| Value::SuperTable(Arc::new(st)))
        },

        // ArrayView-Table broadcasting - convert views to arrays and broadcast
        #[cfg(feature = "views")]
        (Value::ArrayView(array_view), Value::Table(table)) => {
            broadcast_arrayview_to_table(op, &array_view, &table).map(|tbl| Value::Table(Arc::new(tbl)))
        },
        #[cfg(feature = "views")]
        (Value::Table(table), Value::ArrayView(array_view)) => {
            broadcast_table_to_arrayview(op, &table, &array_view).map(|tbl| Value::Table(Arc::new(tbl)))
        },

        // ArrayView-SuperTable broadcasting - convert views to arrays and broadcast
        #[cfg(feature = "views")]
        (Value::ArrayView(array_view), Value::SuperTable(super_table)) => {
            super_table::broadcast_arrayview_to_supertable(op, &array_view, &super_table).map(|st| Value::SuperTable(Arc::new(st)))
        },
        #[cfg(feature = "views")]
        (Value::SuperTable(super_table), Value::ArrayView(array_view)) => {
            super_table::broadcast_supertable_to_arrayview(op, &super_table, &array_view).map(|st| Value::SuperTable(Arc::new(st)))
        },

        // ArrayView-TableView broadcasting - work directly with views for zero-copy
        #[cfg(feature = "views")]
        (Value::ArrayView(array_view), Value::TableView(table_view)) => {
            broadcast_arrayview_to_tableview(op, &array_view, &table_view).map(|tbl| Value::Table(Arc::new(tbl)))
        },
        #[cfg(feature = "views")]
        (Value::TableView(table_view), Value::ArrayView(array_view)) => {
            broadcast_tableview_to_arrayview(op, &table_view, &array_view).map(|tv| Value::TableView(Arc::new(tv)))
        },

        // ArrayView-SuperTableView broadcasting - work per chunk, not materialised
        #[cfg(feature = "views")]
        (Value::ArrayView(array_view), Value::SuperTableView(super_table_view)) => {
            broadcast_arrayview_to_supertableview(op, &array_view, &super_table_view).map(|stv| Value::SuperTableView(Arc::new(stv)))
        },
        #[cfg(feature = "views")]
        (Value::SuperTableView(super_table_view), Value::ArrayView(array_view)) => {
            broadcast_supertableview_to_arrayview(op, &super_table_view, &array_view).map(|stv| Value::SuperTableView(Arc::new(stv)))
        },

        // SuperArray-Table broadcasting - broadcast each chunk against table
        #[cfg(feature = "chunked")]
        (Value::SuperArray(super_array), Value::Table(table)) => {
            broadcast_superarray_to_table(op, &super_array, &table).map(|sa| Value::SuperArray(Arc::new(sa)))
        },
        #[cfg(feature = "chunked")]
        (Value::Table(table), Value::SuperArray(super_array)) => {
            broadcast_table_to_superarray(op, &table, &super_array).map(|sa| Value::SuperArray(Arc::new(sa)))
        },

        // SuperArray-TableView broadcasting - convert TableView to Table and broadcast
        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::SuperArray(super_array), Value::TableView(table_view)) => {
            let table = table_view.to_table();
            broadcast_superarray_to_table(op, &super_array, &table).map(|sa| Value::SuperArray(Arc::new(sa)))
        },
        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::TableView(table_view), Value::SuperArray(super_array)) => {
            let table = table_view.to_table();
            broadcast_table_to_superarray(op, &table, &super_array).map(|sa| Value::SuperArray(Arc::new(sa)))
        },

        // SuperArrayView-Table broadcasting - work directly with view structure
        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::SuperArrayView(super_array_view), Value::Table(table)) => {
            broadcast_superarrayview_to_table(op, &super_array_view, &table).map(|stv| Value::SuperTableView(Arc::new(stv)))
        },
        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::Table(table), Value::SuperArrayView(super_array_view)) => {
            broadcast_table_to_superarrayview(op, &table, &super_array_view).map(|stv| Value::SuperTableView(Arc::new(stv)))
        },

        // SuperArrayView-TableView broadcasting - work directly with view structures
        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::SuperArrayView(super_array_view), Value::TableView(table_view)) => {
            broadcast_superarrayview_to_tableview(op, &super_array_view, &table_view).map(|stv| Value::SuperTableView(Arc::new(stv)))
        },
        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::TableView(table_view), Value::SuperArrayView(super_array_view)) => {
            broadcast_tableview_to_superarrayview(op, &table_view, &super_array_view).map(|stv| Value::SuperTableView(Arc::new(stv)))
        },

        // Array-Cube broadcasting - apply array to each table in the cube
        #[cfg(feature = "cube")]
        (Value::Array(array), Value::Cube(cube)) => {
            array::broadcast_array_to_cube(op, &array, &cube).map(|cube| Value::Cube(Arc::new(cube)))
        }


        #[cfg(feature = "cube")]
        (Value::Cube(cube), Value::Array(array)) => {
            cube::broadcast_cube_to_array(op, &cube, &array).map(|cube| Value::Cube(Arc::new(cube)))
        }

        // FieldArray-Cube broadcasting - apply field array to each table in the cube
        #[cfg(feature = "cube")]
        (Value::FieldArray(field), Value::Cube(cube)) => {
            cube::broadcast_fieldarray_to_cube(op, &field, &cube).map(|cube| Value::Cube(Arc::new(cube)))
        }

        #[cfg(feature = "cube")]
        (Value::Cube(cube), Value::FieldArray(field)) => {
            cube::broadcast_cube_to_fieldarray(op, &cube, &field).map(|cube| Value::Cube(Arc::new(cube)))
        }

        // Table-Cube broadcasting - applies table to each table in the cube
        #[cfg(feature = "cube")]
        (Value::Table(table), Value::Cube(cube)) => {
            cube::broadcast_table_to_cube(op, &table, &cube).map(|cube| Value::Cube(Arc::new(cube)))
        }

        #[cfg(feature = "cube")]
        (Value::Cube(cube), Value::Table(table)) => {
            cube::broadcast_cube_to_table(op, &cube, &table).map(|cube| Value::Cube(Arc::new(cube)))
        }

        #[cfg(all(feature = "cube", feature = "chunked"))]
        (Value::SuperArray(super_array), Value::Cube(cube)) => {
            cube::broadcast_superarray_to_cube(op, &super_array, &cube).map(|cube| Value::Cube(Arc::new(cube)))
        }

        #[cfg(all(feature = "cube", feature = "chunked"))]
        (Value::Cube(cube), Value::SuperArray(super_array)) => {
            cube::broadcast_cube_to_superarray(op, &cube, &super_array).map(|cube| Value::Cube(Arc::new(cube)))
        }

        #[cfg(all(feature = "cube", feature = "chunked"))]
        (Value::SuperTable(super_table), Value::Cube(cube)) => {
            cube::broadcast_supertable_to_cube(op, &super_table, &cube).map(|cube| Value::Cube(Arc::new(cube)))
        }

        #[cfg(all(feature = "cube", feature = "chunked"))]
        (Value::Cube(cube), Value::SuperTable(super_table)) => {
            cube::broadcast_cube_to_supertable(op, &cube, &super_table).map(|cube| Value::Cube(Arc::new(cube)))
        }

        #[cfg(all(feature = "cube", feature = "views"))]
        (Value::ArrayView(array_view), Value::Cube(cube)) => {
            cube::broadcast_arrayview_to_cube(op, &array_view, &cube).map(|cube| Value::Cube(Arc::new(cube)))
        }

        #[cfg(all(feature = "cube", feature = "views"))]
        (Value::Cube(cube), Value::ArrayView(array_view)) => {
            cube::broadcast_cube_to_arrayview(op, &cube, &array_view).map(|cube| Value::Cube(Arc::new(cube)))
        }

        #[cfg(all(feature = "cube", feature = "views"))]
        (Value::TableView(table_view), Value::Cube(cube)) => {
            cube::broadcast_tableview_to_cube(op, &table_view, &cube).map(|cube| Value::Cube(Arc::new(cube)))
        }

        #[cfg(all(feature = "cube", feature = "views"))]
        (Value::Cube(cube), Value::TableView(table_view)) => {
            // Note: broadcast_tableview_to_cube actually does the reverse operation
            // This creates a table from view and broadcasts it with cube tables
            let table = table_view.to_table();
            let mut result_tables = Vec::with_capacity(cube.tables.len());
            for cube_table in &cube.tables {
                let broadcasted = broadcast_table_with_operator(op, cube_table.clone(), table.clone())?;
                result_tables.push(broadcasted);
            }
            Ok(Value::Cube(Cube::from_tables(result_tables, cube.name.clone(), cube.third_dim_index.clone()).into()))
        }


        #[cfg(all(feature = "cube", feature = "views", feature = "chunked"))]
        (Value::SuperArrayView(super_array_view), Value::Cube(cube)) => {
            cube::broadcast_superarrayview_to_cube(op, &super_array_view, &cube).map(|cube| Value::Cube(Arc::new(cube)))
        }

        #[cfg(all(feature = "cube", feature = "views", feature = "chunked"))]
        (Value::Cube(cube), Value::SuperArrayView(super_array_view)) => {
            cube::broadcast_cube_to_superarrayview(op, &cube, &super_array_view).map(|cube| Value::Cube(Arc::new(cube)))
        }

        #[cfg(all(feature = "cube", feature = "views", feature = "chunked"))]
        (Value::SuperTableView(super_table_view), Value::Cube(cube)) => {
            cube::broadcast_supertableview_to_cube(op, &super_table_view, &cube).map(|cube| Value::Cube(Arc::new(cube)))
        }

        #[cfg(all(feature = "cube", feature = "views", feature = "chunked"))]
        (Value::Cube(cube), Value::SuperTableView(super_table_view)) => {
            cube::broadcast_cube_to_supertableview(op, &cube, &super_table_view).map(|cube| Value::Cube(Arc::new(cube)))
        }

        // FieldArray-Tuple broadcasting - loop through tuple and do on per value basis
        (Value::FieldArray(fa), Value::Tuple2(r_arc)) => {
            let (r1, r2) = (r_arc.0.clone(), r_arc.1.clone());
            let res1 = broadcast_value(op, Value::FieldArray(fa.clone()), r1)?;
            let res2 = broadcast_value(op, Value::FieldArray(fa), r2)?;
            Ok(Value::Tuple2(Arc::new((res1, res2))))
        }
        (Value::Tuple2(l_arc), Value::FieldArray(fa)) => {
            let (l1, l2) = (l_arc.0.clone(), l_arc.1.clone());
            let res1 = broadcast_value(op, l1, Value::FieldArray(fa.clone()))?;
            let res2 = broadcast_value(op, l2, Value::FieldArray(fa))?;
            Ok(Value::Tuple2(Arc::new((res1, res2))))
        }
        (Value::FieldArray(fa), Value::Tuple3(r_arc)) => {
            let (r1, r2, r3) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone());
            let res1 = broadcast_value(op, Value::FieldArray(fa.clone()), r1)?;
            let res2 = broadcast_value(op, Value::FieldArray(fa.clone()), r2)?;
            let res3 = broadcast_value(op, Value::FieldArray(fa), r3)?;
            Ok(Value::Tuple3(Arc::new((res1, res2, res3))))
        }
        (Value::Tuple3(l_arc), Value::FieldArray(fa)) => {
            let (l1, l2, l3) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone());
            let res1 = broadcast_value(op, l1, Value::FieldArray(fa.clone()))?;
            let res2 = broadcast_value(op, l2, Value::FieldArray(fa.clone()))?;
            let res3 = broadcast_value(op, l3, Value::FieldArray(fa))?;
            Ok(Value::Tuple3(Arc::new((res1, res2, res3))))
        }
        (Value::FieldArray(fa), Value::Tuple4(r_arc)) => {
            let (r1, r2, r3, r4) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone());
            let res1 = broadcast_value(op, Value::FieldArray(fa.clone()), r1)?;
            let res2 = broadcast_value(op, Value::FieldArray(fa.clone()), r2)?;
            let res3 = broadcast_value(op, Value::FieldArray(fa.clone()), r3)?;
            let res4 = broadcast_value(op, Value::FieldArray(fa), r4)?;
            Ok(Value::Tuple4(Arc::new((res1, res2, res3, res4))))
        }
        (Value::Tuple4(l_arc), Value::FieldArray(fa)) => {
            let (l1, l2, l3, l4) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone());
            let res1 = broadcast_value(op, l1, Value::FieldArray(fa.clone()))?;
            let res2 = broadcast_value(op, l2, Value::FieldArray(fa.clone()))?;
            let res3 = broadcast_value(op, l3, Value::FieldArray(fa.clone()))?;
            let res4 = broadcast_value(op, l4, Value::FieldArray(fa))?;
            Ok(Value::Tuple4(Arc::new((res1, res2, res3, res4))))
        }
        (Value::FieldArray(fa), Value::Tuple5(r_arc)) => {
            let (r1, r2, r3, r4, r5) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone(), r_arc.4.clone());
            let res1 = broadcast_value(op, Value::FieldArray(fa.clone()), r1)?;
            let res2 = broadcast_value(op, Value::FieldArray(fa.clone()), r2)?;
            let res3 = broadcast_value(op, Value::FieldArray(fa.clone()), r3)?;
            let res4 = broadcast_value(op, Value::FieldArray(fa.clone()), r4)?;
            let res5 = broadcast_value(op, Value::FieldArray(fa), r5)?;
            Ok(Value::Tuple5(Arc::new((res1, res2, res3, res4, res5))))
        }
        (Value::Tuple5(l_arc), Value::FieldArray(fa)) => {
            let (l1, l2, l3, l4, l5) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone(), l_arc.4.clone());
            let res1 = broadcast_value(op, l1, Value::FieldArray(fa.clone()))?;
            let res2 = broadcast_value(op, l2, Value::FieldArray(fa.clone()))?;
            let res3 = broadcast_value(op, l3, Value::FieldArray(fa.clone()))?;
            let res4 = broadcast_value(op, l4, Value::FieldArray(fa.clone()))?;
            let res5 = broadcast_value(op, l5, Value::FieldArray(fa))?;
            Ok(Value::Tuple5(Arc::new((res1, res2, res3, res4, res5))))
        }
        (Value::FieldArray(fa), Value::Tuple6(r_arc)) => {
            let (r1, r2, r3, r4, r5, r6) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone(), r_arc.4.clone(), r_arc.5.clone());
            let res1 = broadcast_value(op, Value::FieldArray(fa.clone()), r1)?;
            let res2 = broadcast_value(op, Value::FieldArray(fa.clone()), r2)?;
            let res3 = broadcast_value(op, Value::FieldArray(fa.clone()), r3)?;
            let res4 = broadcast_value(op, Value::FieldArray(fa.clone()), r4)?;
            let res5 = broadcast_value(op, Value::FieldArray(fa.clone()), r5)?;
            let res6 = broadcast_value(op, Value::FieldArray(fa), r6)?;
            Ok(Value::Tuple6(Arc::new((res1, res2, res3, res4, res5, res6))))
        }
        (Value::Tuple6(l_arc), Value::FieldArray(fa)) => {
            let (l1, l2, l3, l4, l5, l6) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone(), l_arc.4.clone(), l_arc.5.clone());
            let res1 = broadcast_value(op, l1, Value::FieldArray(fa.clone()))?;
            let res2 = broadcast_value(op, l2, Value::FieldArray(fa.clone()))?;
            let res3 = broadcast_value(op, l3, Value::FieldArray(fa.clone()))?;
            let res4 = broadcast_value(op, l4, Value::FieldArray(fa.clone()))?;
            let res5 = broadcast_value(op, l5, Value::FieldArray(fa.clone()))?;
            let res6 = broadcast_value(op, l6, Value::FieldArray(fa))?;
            Ok(Value::Tuple6(Arc::new((res1, res2, res3, res4, res5, res6))))
        }

        // SuperArray-Tuple broadcasting - loop through tuple and do on per value basis
        #[cfg(feature = "chunked")]
        (Value::SuperArray(sa), Value::Tuple2(r_arc)) => {
            let (r1, r2) = (r_arc.0.clone(), r_arc.1.clone());
            let res1 = broadcast_value(op, Value::SuperArray(sa.clone()), r1)?;
            let res2 = broadcast_value(op, Value::SuperArray(sa), r2)?;
            Ok(Value::Tuple2(Arc::new((res1, res2))))
        }
        #[cfg(feature = "chunked")]
        (Value::Tuple2(l_arc), Value::SuperArray(sa)) => {
            let (l1, l2) = (l_arc.0.clone(), l_arc.1.clone());
            let res1 = broadcast_value(op, l1, Value::SuperArray(sa.clone()))?;
            let res2 = broadcast_value(op, l2, Value::SuperArray(sa))?;
            Ok(Value::Tuple2(Arc::new((res1, res2))))
        }
        #[cfg(feature = "chunked")]
        (Value::SuperArray(sa), Value::Tuple3(r_arc)) => {
            let (r1, r2, r3) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone());
            let res1 = broadcast_value(op, Value::SuperArray(sa.clone()), r1)?;
            let res2 = broadcast_value(op, Value::SuperArray(sa.clone()), r2)?;
            let res3 = broadcast_value(op, Value::SuperArray(sa), r3)?;
            Ok(Value::Tuple3(Arc::new((res1, res2, res3))))
        }
        #[cfg(feature = "chunked")]
        (Value::Tuple3(l_arc), Value::SuperArray(sa)) => {
            let (l1, l2, l3) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone());
            let res1 = broadcast_value(op, l1, Value::SuperArray(sa.clone()))?;
            let res2 = broadcast_value(op, l2, Value::SuperArray(sa.clone()))?;
            let res3 = broadcast_value(op, l3, Value::SuperArray(sa))?;
            Ok(Value::Tuple3(Arc::new((res1, res2, res3))))
        }
        #[cfg(feature = "chunked")]
        (Value::SuperArray(sa), Value::Tuple4(r_arc)) => {
            let (r1, r2, r3, r4) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone());
            let res1 = broadcast_value(op, Value::SuperArray(sa.clone()), r1)?;
            let res2 = broadcast_value(op, Value::SuperArray(sa.clone()), r2)?;
            let res3 = broadcast_value(op, Value::SuperArray(sa.clone()), r3)?;
            let res4 = broadcast_value(op, Value::SuperArray(sa), r4)?;
            Ok(Value::Tuple4(Arc::new((res1, res2, res3, res4))))
        }
        #[cfg(feature = "chunked")]
        (Value::Tuple4(l_arc), Value::SuperArray(sa)) => {
            let (l1, l2, l3, l4) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone());
            let res1 = broadcast_value(op, l1, Value::SuperArray(sa.clone()))?;
            let res2 = broadcast_value(op, l2, Value::SuperArray(sa.clone()))?;
            let res3 = broadcast_value(op, l3, Value::SuperArray(sa.clone()))?;
            let res4 = broadcast_value(op, l4, Value::SuperArray(sa))?;
            Ok(Value::Tuple4(Arc::new((res1, res2, res3, res4))))
        }
        #[cfg(feature = "chunked")]
        (Value::SuperArray(sa), Value::Tuple5(r_arc)) => {
            let (r1, r2, r3, r4, r5) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone(), r_arc.4.clone());
            let res1 = broadcast_value(op, Value::SuperArray(sa.clone()), r1)?;
            let res2 = broadcast_value(op, Value::SuperArray(sa.clone()), r2)?;
            let res3 = broadcast_value(op, Value::SuperArray(sa.clone()), r3)?;
            let res4 = broadcast_value(op, Value::SuperArray(sa.clone()), r4)?;
            let res5 = broadcast_value(op, Value::SuperArray(sa), r5)?;
            Ok(Value::Tuple5(Arc::new((res1, res2, res3, res4, res5))))
        }
        #[cfg(feature = "chunked")]
        (Value::Tuple5(l_arc), Value::SuperArray(sa)) => {
            let (l1, l2, l3, l4, l5) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone(), l_arc.4.clone());
            let res1 = broadcast_value(op, l1, Value::SuperArray(sa.clone()))?;
            let res2 = broadcast_value(op, l2, Value::SuperArray(sa.clone()))?;
            let res3 = broadcast_value(op, l3, Value::SuperArray(sa.clone()))?;
            let res4 = broadcast_value(op, l4, Value::SuperArray(sa.clone()))?;
            let res5 = broadcast_value(op, l5, Value::SuperArray(sa))?;
            Ok(Value::Tuple5(Arc::new((res1, res2, res3, res4, res5))))
        }
        #[cfg(feature = "chunked")]
        (Value::SuperArray(sa), Value::Tuple6(r_arc)) => {
            let (r1, r2, r3, r4, r5, r6) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone(), r_arc.4.clone(), r_arc.5.clone());
            let res1 = broadcast_value(op, Value::SuperArray(sa.clone()), r1)?;
            let res2 = broadcast_value(op, Value::SuperArray(sa.clone()), r2)?;
            let res3 = broadcast_value(op, Value::SuperArray(sa.clone()), r3)?;
            let res4 = broadcast_value(op, Value::SuperArray(sa.clone()), r4)?;
            let res5 = broadcast_value(op, Value::SuperArray(sa.clone()), r5)?;
            let res6 = broadcast_value(op, Value::SuperArray(sa), r6)?;
            Ok(Value::Tuple6(Arc::new((res1, res2, res3, res4, res5, res6))))
        }
        #[cfg(feature = "chunked")]
        (Value::Tuple6(l_arc), Value::SuperArray(sa)) => {
            let (l1, l2, l3, l4, l5, l6) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone(), l_arc.4.clone(), l_arc.5.clone());
            let res1 = broadcast_value(op, l1, Value::SuperArray(sa.clone()))?;
            let res2 = broadcast_value(op, l2, Value::SuperArray(sa.clone()))?;
            let res3 = broadcast_value(op, l3, Value::SuperArray(sa.clone()))?;
            let res4 = broadcast_value(op, l4, Value::SuperArray(sa.clone()))?;
            let res5 = broadcast_value(op, l5, Value::SuperArray(sa.clone()))?;
            let res6 = broadcast_value(op, l6, Value::SuperArray(sa))?;
            Ok(Value::Tuple6(Arc::new((res1, res2, res3, res4, res5, res6))))
        }
        // SuperArrayView-Tuple broadcasting
        #[cfg(all(feature = "views", feature = "chunked"))]
        (Value::SuperArrayView(sav), Value::Tuple2(r_arc)) => {
            let (r1, r2) = (r_arc.0.clone(), r_arc.1.clone());
            let res1 = broadcast_value(op, Value::SuperArrayView(sav.clone()), r1)?;
            let res2 = broadcast_value(op, Value::SuperArrayView(sav), r2)?;
            Ok(Value::Tuple2(Arc::new((res1, res2))))
        }
        #[cfg(all(feature = "views", feature = "chunked"))]
        (Value::Tuple2(l_arc), Value::SuperArrayView(sav)) => {
            let (l1, l2) = (l_arc.0.clone(), l_arc.1.clone());
            let res1 = broadcast_value(op, l1, Value::SuperArrayView(sav.clone()))?;
            let res2 = broadcast_value(op, l2, Value::SuperArrayView(sav))?;
            Ok(Value::Tuple2(Arc::new((res1, res2))))
        }
        #[cfg(all(feature = "views", feature = "chunked"))]
        (Value::SuperArrayView(sav), Value::Tuple3(r_arc)) => {
            let (r1, r2, r3) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone());
            let res1 = broadcast_value(op, Value::SuperArrayView(sav.clone()), r1)?;
            let res2 = broadcast_value(op, Value::SuperArrayView(sav.clone()), r2)?;
            let res3 = broadcast_value(op, Value::SuperArrayView(sav), r3)?;
            Ok(Value::Tuple3(Arc::new((res1, res2, res3))))
        }
        #[cfg(all(feature = "views", feature = "chunked"))]
        (Value::Tuple3(l_arc), Value::SuperArrayView(sav)) => {
            let (l1, l2, l3) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone());
            let res1 = broadcast_value(op, l1, Value::SuperArrayView(sav.clone()))?;
            let res2 = broadcast_value(op, l2, Value::SuperArrayView(sav.clone()))?;
            let res3 = broadcast_value(op, l3, Value::SuperArrayView(sav))?;
            Ok(Value::Tuple3(Arc::new((res1, res2, res3))))
        }
        #[cfg(all(feature = "views", feature = "chunked"))]
        (Value::SuperArrayView(sav), Value::Tuple4(r_arc)) => {
            let (r1, r2, r3, r4) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone());
            let res1 = broadcast_value(op, Value::SuperArrayView(sav.clone()), r1)?;
            let res2 = broadcast_value(op, Value::SuperArrayView(sav.clone()), r2)?;
            let res3 = broadcast_value(op, Value::SuperArrayView(sav.clone()), r3)?;
            let res4 = broadcast_value(op, Value::SuperArrayView(sav), r4)?;
            Ok(Value::Tuple4(Arc::new((res1, res2, res3, res4))))
        }
        #[cfg(all(feature = "views", feature = "chunked"))]
        (Value::Tuple4(l_arc), Value::SuperArrayView(sav)) => {
            let (l1, l2, l3, l4) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone());
            let res1 = broadcast_value(op, l1, Value::SuperArrayView(sav.clone()))?;
            let res2 = broadcast_value(op, l2, Value::SuperArrayView(sav.clone()))?;
            let res3 = broadcast_value(op, l3, Value::SuperArrayView(sav.clone()))?;
            let res4 = broadcast_value(op, l4, Value::SuperArrayView(sav))?;
            Ok(Value::Tuple4(Arc::new((res1, res2, res3, res4))))
        }
        #[cfg(all(feature = "views", feature = "chunked"))]
        (Value::SuperArrayView(sav), Value::Tuple5(r_arc)) => {
            let (r1, r2, r3, r4, r5) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone(), r_arc.4.clone());
            let res1 = broadcast_value(op, Value::SuperArrayView(sav.clone()), r1)?;
            let res2 = broadcast_value(op, Value::SuperArrayView(sav.clone()), r2)?;
            let res3 = broadcast_value(op, Value::SuperArrayView(sav.clone()), r3)?;
            let res4 = broadcast_value(op, Value::SuperArrayView(sav.clone()), r4)?;
            let res5 = broadcast_value(op, Value::SuperArrayView(sav), r5)?;
            Ok(Value::Tuple5(Arc::new((res1, res2, res3, res4, res5))))
        }
        #[cfg(all(feature = "views", feature = "chunked"))]
        (Value::Tuple5(l_arc), Value::SuperArrayView(sav)) => {
            let (l1, l2, l3, l4, l5) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone(), l_arc.4.clone());
            let res1 = broadcast_value(op, l1, Value::SuperArrayView(sav.clone()))?;
            let res2 = broadcast_value(op, l2, Value::SuperArrayView(sav.clone()))?;
            let res3 = broadcast_value(op, l3, Value::SuperArrayView(sav.clone()))?;
            let res4 = broadcast_value(op, l4, Value::SuperArrayView(sav.clone()))?;
            let res5 = broadcast_value(op, l5, Value::SuperArrayView(sav))?;
            Ok(Value::Tuple5(Arc::new((res1, res2, res3, res4, res5))))
        }
        #[cfg(all(feature = "views", feature = "chunked"))]
        (Value::SuperArrayView(sav), Value::Tuple6(r_arc)) => {
            let (r1, r2, r3, r4, r5, r6) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone(), r_arc.4.clone(), r_arc.5.clone());
            let res1 = broadcast_value(op, Value::SuperArrayView(sav.clone()), r1)?;
            let res2 = broadcast_value(op, Value::SuperArrayView(sav.clone()), r2)?;
            let res3 = broadcast_value(op, Value::SuperArrayView(sav.clone()), r3)?;
            let res4 = broadcast_value(op, Value::SuperArrayView(sav.clone()), r4)?;
            let res5 = broadcast_value(op, Value::SuperArrayView(sav.clone()), r5)?;
            let res6 = broadcast_value(op, Value::SuperArrayView(sav), r6)?;
            Ok(Value::Tuple6(Arc::new((res1, res2, res3, res4, res5, res6))))
        }
        #[cfg(all(feature = "views", feature = "chunked"))]
        (Value::Tuple6(l_arc), Value::SuperArrayView(sav)) => {
            let (l1, l2, l3, l4, l5, l6) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone(), l_arc.4.clone(), l_arc.5.clone());
            let res1 = broadcast_value(op, l1, Value::SuperArrayView(sav.clone()))?;
            let res2 = broadcast_value(op, l2, Value::SuperArrayView(sav.clone()))?;
            let res3 = broadcast_value(op, l3, Value::SuperArrayView(sav.clone()))?;
            let res4 = broadcast_value(op, l4, Value::SuperArrayView(sav.clone()))?;
            let res5 = broadcast_value(op, l5, Value::SuperArrayView(sav.clone()))?;
            let res6 = broadcast_value(op, l6, Value::SuperArrayView(sav))?;
            Ok(Value::Tuple6(Arc::new((res1, res2, res3, res4, res5, res6))))
        }

        // ALL remaining SuperTable combinations with complex types
        #[cfg(feature = "chunked")]
        (Value::SuperTable(_), Value::Tuple2(_))
        | (Value::Tuple2(_), Value::SuperTable(_))
        | (Value::SuperTable(_), Value::Tuple3(_))
        | (Value::Tuple3(_), Value::SuperTable(_))
        | (Value::SuperTable(_), Value::Tuple4(_))
        | (Value::Tuple4(_), Value::SuperTable(_))
        | (Value::SuperTable(_), Value::Tuple5(_))
        | (Value::Tuple5(_), Value::SuperTable(_))
        | (Value::SuperTable(_), Value::Tuple6(_))
        | (Value::Tuple6(_), Value::SuperTable(_))
        | (Value::SuperTableView(_), Value::Tuple2(_))
        | (Value::Tuple2(_), Value::SuperTableView(_))
        | (Value::SuperTableView(_), Value::Tuple3(_))
        | (Value::Tuple3(_), Value::SuperTableView(_))
        | (Value::SuperTableView(_), Value::Tuple4(_))
        | (Value::Tuple4(_), Value::SuperTableView(_))
        | (Value::SuperTableView(_), Value::Tuple5(_))
        | (Value::Tuple5(_), Value::SuperTableView(_))
        | (Value::SuperTableView(_), Value::Tuple6(_))
        | (Value::Tuple6(_), Value::SuperTableView(_)) => Err(MinarrowError::TypeError {
            from: "SuperTable and complex/container types",
            to: "compatible broadcasting types",
            message: Some(
                "SuperTable cannot be broadcast with container or metadata types".to_string(),
            ),
        }),

        // Mixed tuple combinations that aren't the same size
        (Value::Tuple2(_), Value::Tuple3(_))
        | (Value::Tuple3(_), Value::Tuple2(_))
        | (Value::Tuple2(_), Value::Tuple4(_))
        | (Value::Tuple4(_), Value::Tuple2(_))
        | (Value::Tuple2(_), Value::Tuple5(_))
        | (Value::Tuple5(_), Value::Tuple2(_))
        | (Value::Tuple2(_), Value::Tuple6(_))
        | (Value::Tuple6(_), Value::Tuple2(_))
        | (Value::Tuple3(_), Value::Tuple4(_))
        | (Value::Tuple4(_), Value::Tuple3(_))
        | (Value::Tuple3(_), Value::Tuple5(_))
        | (Value::Tuple5(_), Value::Tuple3(_))
        | (Value::Tuple3(_), Value::Tuple6(_))
        | (Value::Tuple6(_), Value::Tuple3(_))
        | (Value::Tuple4(_), Value::Tuple5(_))
        | (Value::Tuple5(_), Value::Tuple4(_))
        | (Value::Tuple4(_), Value::Tuple6(_))
        | (Value::Tuple6(_), Value::Tuple4(_))
        | (Value::Tuple5(_), Value::Tuple6(_))
        | (Value::Tuple6(_), Value::Tuple5(_)) => Err(MinarrowError::TypeError {
            from: "Tuples of different sizes",
            to: "compatible broadcasting types",
            message: Some("Cannot broadcast tuples of different sizes".to_string()),
        }),

        // Tuple broadcasting with non-tuple data types - broadcast element-wise
        (Value::Tuple2(l_arc), Value::Array(r)) => {
            let (l1, l2) = (l_arc.0.clone(), l_arc.1.clone());
            let res1 = broadcast_value(op, l1, Value::Array(r.clone()))?;
            let res2 = broadcast_value(op, l2, Value::Array(r))?;
            Ok(Value::Tuple2(Arc::new((res1, res2))))
        }
        (Value::Array(l), Value::Tuple2(r_arc)) => {
            let (r1, r2) = (r_arc.0.clone(), r_arc.1.clone());
            array::broadcast_array_to_tuple2(op, &l, (Arc::new(r1), Arc::new(r2))).map(|(b1, b2)| Value::Tuple2(Arc::new((Arc::unwrap_or_clone(b1), Arc::unwrap_or_clone(b2)))))
        }

        (Value::Tuple3(l_arc), Value::Array(r)) => {
            let (l1, l2, l3) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone());
            let res1 = broadcast_value(op, l1, Value::Array(r.clone()))?;
            let res2 = broadcast_value(op, l2, Value::Array(r.clone()))?;
            let res3 = broadcast_value(op, l3, Value::Array(r))?;
            Ok(Value::Tuple3(Arc::new((res1, res2, res3))))
        }
        (Value::Array(l), Value::Tuple3(r_arc)) => {
            let (r1, r2, r3) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone());
            array::broadcast_array_to_tuple3(op, &l, (Arc::new(r1), Arc::new(r2), Arc::new(r3))).map(|(b1, b2, b3)| Value::Tuple3(Arc::new((Arc::unwrap_or_clone(b1), Arc::unwrap_or_clone(b2), Arc::unwrap_or_clone(b3)))))
        }

        (Value::Tuple4(l_arc), Value::Array(r)) => {
            let (l1, l2, l3, l4) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone());
            let res1 = broadcast_value(op, l1, Value::Array(r.clone()))?;
            let res2 = broadcast_value(op, l2, Value::Array(r.clone()))?;
            let res3 = broadcast_value(op, l3, Value::Array(r.clone()))?;
            let res4 = broadcast_value(op, l4, Value::Array(r))?;
            Ok(Value::Tuple4(Arc::new((res1, res2, res3, res4))))
        }
        (Value::Array(l), Value::Tuple4(r_arc)) => {
            let (r1, r2, r3, r4) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone());
            array::broadcast_array_to_tuple4(op, &l, (Arc::new(r1), Arc::new(r2), Arc::new(r3), Arc::new(r4))).map(|(b1, b2, b3, b4)| Value::Tuple4(Arc::new((Arc::unwrap_or_clone(b1), Arc::unwrap_or_clone(b2), Arc::unwrap_or_clone(b3), Arc::unwrap_or_clone(b4)))))
        }

        (Value::Tuple5(l_arc), Value::Array(r)) => {
            let (l1, l2, l3, l4, l5) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone(), l_arc.4.clone());
            let res1 = broadcast_value(op, l1, Value::Array(r.clone()))?;
            let res2 = broadcast_value(op, l2, Value::Array(r.clone()))?;
            let res3 = broadcast_value(op, l3, Value::Array(r.clone()))?;
            let res4 = broadcast_value(op, l4, Value::Array(r.clone()))?;
            let res5 = broadcast_value(op, l5, Value::Array(r))?;
            Ok(Value::Tuple5(Arc::new((res1, res2, res3, res4, res5))))
        }
        (Value::Array(l), Value::Tuple5(r_arc)) => {
            let (r1, r2, r3, r4, r5) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone(), r_arc.4.clone());
            array::broadcast_array_to_tuple5(op, &l, (Arc::new(r1), Arc::new(r2), Arc::new(r3), Arc::new(r4), Arc::new(r5))).map(|(b1, b2, b3, b4, b5)| Value::Tuple5(Arc::new((Arc::unwrap_or_clone(b1), Arc::unwrap_or_clone(b2), Arc::unwrap_or_clone(b3), Arc::unwrap_or_clone(b4), Arc::unwrap_or_clone(b5)))))
        }

        (Value::Tuple6(l_arc), Value::Array(r)) => {
            let (l1, l2, l3, l4, l5, l6) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone(), l_arc.4.clone(), l_arc.5.clone());
            let res1 = broadcast_value(op, l1, Value::Array(r.clone()))?;
            let res2 = broadcast_value(op, l2, Value::Array(r.clone()))?;
            let res3 = broadcast_value(op, l3, Value::Array(r.clone()))?;
            let res4 = broadcast_value(op, l4, Value::Array(r.clone()))?;
            let res5 = broadcast_value(op, l5, Value::Array(r.clone()))?;
            let res6 = broadcast_value(op, l6, Value::Array(r))?;
            Ok(Value::Tuple6(Arc::new((res1, res2, res3, res4, res5, res6))))
        }
        (Value::Array(l), Value::Tuple6(r_arc)) => {
            let (r1, r2, r3, r4, r5, r6) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone(), r_arc.4.clone(), r_arc.5.clone());
            array::broadcast_array_to_tuple6(op, &l, (Arc::new(r1), Arc::new(r2), Arc::new(r3), Arc::new(r4), Arc::new(r5), Arc::new(r6))).map(|(b1, b2, b3, b4, b5, b6)| Value::Tuple6(Arc::new((Arc::unwrap_or_clone(b1), Arc::unwrap_or_clone(b2), Arc::unwrap_or_clone(b3), Arc::unwrap_or_clone(b4), Arc::unwrap_or_clone(b5), Arc::unwrap_or_clone(b6)))))
        }

        // Tuple broadcasting with scalar types
        #[cfg(feature = "scalar_type")]
        (Value::Tuple2(l_arc), Value::Scalar(r)) => {
            let (l1, l2) = (l_arc.0.clone(), l_arc.1.clone());
            let res1 = broadcast_value(op, l1, Value::Scalar(r.clone()))?;
            let res2 = broadcast_value(op, l2, Value::Scalar(r))?;
            Ok(Value::Tuple2(Arc::new((res1, res2))))
        }
        #[cfg(feature = "scalar_type")]
        (Value::Scalar(l), Value::Tuple2(r_arc)) => {
            let (r1, r2) = (r_arc.0.clone(), r_arc.1.clone());
            scalar::broadcast_scalar_to_tuple2(op, &l, (Arc::new(r1), Arc::new(r2))).map(|(b1, b2)| Value::Tuple2(Arc::new((Arc::unwrap_or_clone(b1), Arc::unwrap_or_clone(b2)))))
        }

        #[cfg(feature = "scalar_type")]
        (Value::Tuple3(l_arc), Value::Scalar(r)) => {
            let (l1, l2, l3) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone());
            let res1 = broadcast_value(op, l1, Value::Scalar(r.clone()))?;
            let res2 = broadcast_value(op, l2, Value::Scalar(r.clone()))?;
            let res3 = broadcast_value(op, l3, Value::Scalar(r))?;
            Ok(Value::Tuple3(Arc::new((res1, res2, res3))))
        }
        #[cfg(feature = "scalar_type")]
        (Value::Scalar(l), Value::Tuple3(r_arc)) => {
            let (r1, r2, r3) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone());
            scalar::broadcast_scalar_to_tuple3(op, &l, (Arc::new(r1), Arc::new(r2), Arc::new(r3))).map(|(b1, b2, b3)| Value::Tuple3(Arc::new((Arc::unwrap_or_clone(b1), Arc::unwrap_or_clone(b2), Arc::unwrap_or_clone(b3)))))
        }

        #[cfg(feature = "scalar_type")]
        (Value::Tuple4(l_arc), Value::Scalar(r)) => {
            let (l1, l2, l3, l4) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone());
            let res1 = broadcast_value(op, l1, Value::Scalar(r.clone()))?;
            let res2 = broadcast_value(op, l2, Value::Scalar(r.clone()))?;
            let res3 = broadcast_value(op, l3, Value::Scalar(r.clone()))?;
            let res4 = broadcast_value(op, l4, Value::Scalar(r))?;
            Ok(Value::Tuple4(Arc::new((res1, res2, res3, res4))))
        }
        #[cfg(feature = "scalar_type")]
        (Value::Scalar(l), Value::Tuple4(r_arc)) => {
            let (r1, r2, r3, r4) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone());
            scalar::broadcast_scalar_to_tuple4(op, &l, (Arc::new(r1), Arc::new(r2), Arc::new(r3), Arc::new(r4))).map(|(b1, b2, b3, b4)| Value::Tuple4(Arc::new((Arc::unwrap_or_clone(b1), Arc::unwrap_or_clone(b2), Arc::unwrap_or_clone(b3), Arc::unwrap_or_clone(b4)))))
        }

        #[cfg(feature = "scalar_type")]
        (Value::Tuple5(l_arc), Value::Scalar(r)) => {
            let (l1, l2, l3, l4, l5) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone(), l_arc.4.clone());
            let res1 = broadcast_value(op, l1, Value::Scalar(r.clone()))?;
            let res2 = broadcast_value(op, l2, Value::Scalar(r.clone()))?;
            let res3 = broadcast_value(op, l3, Value::Scalar(r.clone()))?;
            let res4 = broadcast_value(op, l4, Value::Scalar(r.clone()))?;
            let res5 = broadcast_value(op, l5, Value::Scalar(r))?;
            Ok(Value::Tuple5(Arc::new((res1, res2, res3, res4, res5))))
        }
        #[cfg(feature = "scalar_type")]
        (Value::Scalar(l), Value::Tuple5(r_arc)) => {
            let (r1, r2, r3, r4, r5) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone(), r_arc.4.clone());
            scalar::broadcast_scalar_to_tuple5(op, &l, (Arc::new(r1), Arc::new(r2), Arc::new(r3), Arc::new(r4), Arc::new(r5))).map(|(b1, b2, b3, b4, b5)| Value::Tuple5(Arc::new((Arc::unwrap_or_clone(b1), Arc::unwrap_or_clone(b2), Arc::unwrap_or_clone(b3), Arc::unwrap_or_clone(b4), Arc::unwrap_or_clone(b5)))))
        }

        #[cfg(feature = "scalar_type")]
        (Value::Tuple6(l_arc), Value::Scalar(r)) => {
            let (l1, l2, l3, l4, l5, l6) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone(), l_arc.4.clone(), l_arc.5.clone());
            let res1 = broadcast_value(op, l1, Value::Scalar(r.clone()))?;
            let res2 = broadcast_value(op, l2, Value::Scalar(r.clone()))?;
            let res3 = broadcast_value(op, l3, Value::Scalar(r.clone()))?;
            let res4 = broadcast_value(op, l4, Value::Scalar(r.clone()))?;
            let res5 = broadcast_value(op, l5, Value::Scalar(r.clone()))?;
            let res6 = broadcast_value(op, l6, Value::Scalar(r))?;
            Ok(Value::Tuple6(Arc::new((res1, res2, res3, res4, res5, res6))))
        }
        #[cfg(feature = "scalar_type")]
        (Value::Scalar(l), Value::Tuple6(r_arc)) => {
            let (r1, r2, r3, r4, r5, r6) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone(), r_arc.4.clone(), r_arc.5.clone());
            scalar::broadcast_scalar_to_tuple6(op, &l, (Arc::new(r1), Arc::new(r2), Arc::new(r3), Arc::new(r4), Arc::new(r5), Arc::new(r6))).map(|(b1, b2, b3, b4, b5, b6)| Value::Tuple6(Arc::new((Arc::unwrap_or_clone(b1), Arc::unwrap_or_clone(b2), Arc::unwrap_or_clone(b3), Arc::unwrap_or_clone(b4), Arc::unwrap_or_clone(b5), Arc::unwrap_or_clone(b6)))))
        }

        // Tuple broadcasting with Table types
        (Value::Tuple2(l_arc), Value::Table(r)) => {
            let (l1, l2) = (l_arc.0.clone(), l_arc.1.clone());
            let res1 = broadcast_value(op, l1, Value::Table(r.clone()))?;
            let res2 = broadcast_value(op, l2, Value::Table(r))?;
            Ok(Value::Tuple2(Arc::new((res1, res2))))
        }
        (Value::Table(l), Value::Tuple2(r_arc)) => {
            let (r1, r2) = (r_arc.0.clone(), r_arc.1.clone());
            let res1 = broadcast_value(op, Value::Table(l.clone()), r1)?;
            let res2 = broadcast_value(op, Value::Table(l), r2)?;
            Ok(Value::Tuple2(Arc::new((res1, res2))))
        }

        (Value::Tuple3(l_arc), Value::Table(r)) => {
            let (l1, l2, l3) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone());
            let res1 = broadcast_value(op, l1, Value::Table(r.clone()))?;
            let res2 = broadcast_value(op, l2, Value::Table(r.clone()))?;
            let res3 = broadcast_value(op, l3, Value::Table(r))?;
            Ok(Value::Tuple3(Arc::new((res1, res2, res3))))
        }
        (Value::Table(l), Value::Tuple3(r_arc)) => {
            let (r1, r2, r3) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone());
            let res1 = broadcast_value(op, Value::Table(l.clone()), r1)?;
            let res2 = broadcast_value(op, Value::Table(l.clone()), r2)?;
            let res3 = broadcast_value(op, Value::Table(l), r3)?;
            Ok(Value::Tuple3(Arc::new((res1, res2, res3))))
        }

        (Value::Tuple4(l_arc), Value::Table(r)) => {
            let (l1, l2, l3, l4) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone());
            let res1 = broadcast_value(op, l1, Value::Table(r.clone()))?;
            let res2 = broadcast_value(op, l2, Value::Table(r.clone()))?;
            let res3 = broadcast_value(op, l3, Value::Table(r.clone()))?;
            let res4 = broadcast_value(op, l4, Value::Table(r))?;
            Ok(Value::Tuple4(Arc::new((res1, res2, res3, res4))))
        }
        (Value::Table(l), Value::Tuple4(r_arc)) => {
            let (r1, r2, r3, r4) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone());
            let res1 = broadcast_value(op, Value::Table(l.clone()), r1)?;
            let res2 = broadcast_value(op, Value::Table(l.clone()), r2)?;
            let res3 = broadcast_value(op, Value::Table(l.clone()), r3)?;
            let res4 = broadcast_value(op, Value::Table(l), r4)?;
            Ok(Value::Tuple4(Arc::new((res1, res2, res3, res4))))
        }

        (Value::Tuple5(l_arc), Value::Table(r)) => {
            let (l1, l2, l3, l4, l5) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone(), l_arc.4.clone());
            let res1 = broadcast_value(op, l1, Value::Table(r.clone()))?;
            let res2 = broadcast_value(op, l2, Value::Table(r.clone()))?;
            let res3 = broadcast_value(op, l3, Value::Table(r.clone()))?;
            let res4 = broadcast_value(op, l4, Value::Table(r.clone()))?;
            let res5 = broadcast_value(op, l5, Value::Table(r))?;
            Ok(Value::Tuple5(Arc::new((res1, res2, res3, res4, res5))))
        }
        (Value::Table(l), Value::Tuple5(r_arc)) => {
            let (r1, r2, r3, r4, r5) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone(), r_arc.4.clone());
            let res1 = broadcast_value(op, Value::Table(l.clone()), r1)?;
            let res2 = broadcast_value(op, Value::Table(l.clone()), r2)?;
            let res3 = broadcast_value(op, Value::Table(l.clone()), r3)?;
            let res4 = broadcast_value(op, Value::Table(l.clone()), r4)?;
            let res5 = broadcast_value(op, Value::Table(l), r5)?;
            Ok(Value::Tuple5(Arc::new((res1, res2, res3, res4, res5))))
        }
        (Value::Tuple6(l_arc), Value::Table(r)) => {
            let (l1, l2, l3, l4, l5, l6) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone(), l_arc.4.clone(), l_arc.5.clone());
            let res1 = broadcast_value(op, l1, Value::Table(r.clone()))?;
            let res2 = broadcast_value(op, l2, Value::Table(r.clone()))?;
            let res3 = broadcast_value(op, l3, Value::Table(r.clone()))?;
            let res4 = broadcast_value(op, l4, Value::Table(r.clone()))?;
            let res5 = broadcast_value(op, l5, Value::Table(r.clone()))?;
            let res6 = broadcast_value(op, l6, Value::Table(r))?;
            Ok(Value::Tuple6(Arc::new((res1, res2, res3, res4, res5, res6))))
        }
        (Value::Table(l), Value::Tuple6(r_arc)) => {
            let (r1, r2, r3, r4, r5, r6) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone(), r_arc.4.clone(), r_arc.5.clone());
            let res1 = broadcast_value(op, Value::Table(l.clone()), r1)?;
            let res2 = broadcast_value(op, Value::Table(l.clone()), r2)?;
            let res3 = broadcast_value(op, Value::Table(l.clone()), r3)?;
            let res4 = broadcast_value(op, Value::Table(l.clone()), r4)?;
            let res5 = broadcast_value(op, Value::Table(l.clone()), r5)?;
            let res6 = broadcast_value(op, Value::Table(l), r6)?;
            Ok(Value::Tuple6(Arc::new((res1, res2, res3, res4, res5, res6))))
        }
        // Scalar combinations with generic views - follow existing scalar broadcasting pattern
        #[cfg(all(feature = "scalar_type", feature = "views"))]
        (Value::Scalar(l), Value::ArrayView(r)) => {
            let scalar_array = match l {
                Scalar::Int32(val) => Array::from_int32(IntegerArray::from_slice(&[val])),
                Scalar::Int64(val) => Array::from_int64(IntegerArray::from_slice(&[val])),
                Scalar::Float32(val) => Array::from_float32(FloatArray::from_slice(&[val])),
                Scalar::Float64(val) => Array::from_float64(FloatArray::from_slice(&[val])),
                Scalar::String32(val) => {
                    Array::from_string32(StringArray::from_slice(&[val.as_str()]))
                }
                _ => {
                    return Err(MinarrowError::NotImplemented {
                        feature: "Scalar type not supported for ArrayView broadcasting".to_string(),
                    });
                }
            };
            resolve_binary_arithmetic(op, scalar_array, Arc::unwrap_or_clone(r), None)
                .map(|arr| Value::Array(Arc::new(arr)))
        }
        #[cfg(all(feature = "scalar_type", feature = "views"))]
        (Value::ArrayView(l), Value::Scalar(r)) => {
            let scalar_array = match r {
                Scalar::Int32(val) => Array::from_int32(IntegerArray::from_slice(&[val])),
                Scalar::Int64(val) => Array::from_int64(IntegerArray::from_slice(&[val])),
                Scalar::Float32(val) => Array::from_float32(FloatArray::from_slice(&[val])),
                Scalar::Float64(val) => Array::from_float64(FloatArray::from_slice(&[val])),
                Scalar::String32(val) => {
                    Array::from_string32(StringArray::from_slice(&[val.as_str()]))
                }
                _ => {
                    return Err(MinarrowError::NotImplemented {
                        feature: "Scalar type not supported for ArrayView broadcasting".to_string(),
                    });
                }
            };
            resolve_binary_arithmetic(op, Arc::unwrap_or_clone(l), scalar_array, None)
                .map(|arr| Value::Array(Arc::new(arr)))
        }
        // Array combinations with chunked types - convert Array to SuperArray for broadcasting
        #[cfg(feature = "chunked")]
        (Value::Array(l), Value::SuperArray(r)) => {

            let l_super_array = create_aligned_chunks_from_array(Arc::unwrap_or_clone(l), &r, r.field_ref().name.as_str())?;
            route_super_array_broadcast(op, l_super_array, Arc::unwrap_or_clone(r), None)
                .map(|sa| Value::SuperArray(Arc::new(sa)))
        }
        #[cfg(feature = "chunked")]
        (Value::SuperArray(l), Value::Array(r)) => {
            let r_super_array = create_aligned_chunks_from_array(Arc::unwrap_or_clone(r), &l, l.field_ref().name.as_str())?;
            route_super_array_broadcast(op, Arc::unwrap_or_clone(l), r_super_array, None)
                .map(|sa| Value::SuperArray(Arc::new(sa)))
        }
        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::Array(l), Value::SuperArrayView(r)) => {
            let r_super_array = SuperArray::from_slices(&r.slices, r.field.clone());
            let l_super_array = create_aligned_chunks_from_array(Arc::unwrap_or_clone(l), &r_super_array, &r.field.name)?;
            route_super_array_broadcast(op, l_super_array, r_super_array, None)
                .map(|sa| Value::SuperArray(Arc::new(sa)))
        }
        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::SuperArrayView(l), Value::Array(r)) => {
            let l_super_array = SuperArray::from_slices(&l.slices, l.field.clone());
            let r_super_array = create_aligned_chunks_from_array(Arc::unwrap_or_clone(r), &l_super_array, &l.field.name)?;
            route_super_array_broadcast(op, l_super_array, r_super_array, None)
                .map(|sa| Value::SuperArray(Arc::new(sa)))
        }
        // Array-SuperTableView broadcasting - create aligned array views for each table slice
        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::Array(array), Value::SuperTableView(super_table_view)) => {
            broadcast_array_to_supertableview(op, &Arc::unwrap_or_clone(array), &Arc::unwrap_or_clone(super_table_view)).map(|stv| Value::SuperTableView(Arc::new(stv)))
        },
        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::SuperTableView(super_table_view), Value::Array(array)) => {
            broadcast_supertableview_to_array(op, &Arc::unwrap_or_clone(super_table_view), &Arc::unwrap_or_clone(array)).map(|stv| Value::SuperTableView(Arc::new(stv)))
        },
        // Array-TableView broadcasting - create array view aligned with table view
        #[cfg(feature = "views")]
        (Value::Array(array), Value::TableView(table_view)) => {
            broadcast_arrayview_to_tableview(op, &ArrayV::new(Arc::unwrap_or_clone(array), table_view.offset, table_view.len), &table_view).map(|tbl| Value::Table(Arc::new(tbl)))
        },
        #[cfg(feature = "views")]
        (Value::TableView(table_view), Value::Array(array)) => {
            broadcast_arrayview_to_tableview(op, &ArrayV::new(Arc::unwrap_or_clone(array), table_view.offset, table_view.len), &table_view).map(|tbl| Value::Table(Arc::new(tbl)))
        },

        // FieldArray combinations with chunked types - convert FieldArray to SuperArray
        #[cfg(feature = "chunked")]
        (Value::FieldArray(l), Value::SuperArray(r)) => {
            // Convert FieldArray to SuperArray format for broadcasting
            let l_super_array = SuperArray::from_chunks(vec![Arc::unwrap_or_clone(l)]);
            route_super_array_broadcast(op, l_super_array, Arc::unwrap_or_clone(r), None)
                .map(|sa| Value::SuperArray(Arc::new(sa)))
        }
        #[cfg(feature = "chunked")]
        (Value::SuperArray(l), Value::FieldArray(r)) => {
            // Convert FieldArray to SuperArray format for broadcasting
            let r_super_array = SuperArray::from_chunks(vec![Arc::unwrap_or_clone(r)]);
            route_super_array_broadcast(op, Arc::unwrap_or_clone(l), r_super_array, None)
                .map(|sa| Value::SuperArray(Arc::new(sa)))
        }

        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::FieldArray(l), Value::SuperArrayView(r)) => {
            field_array::broadcast_fieldarray_to_superarrayview(op, &l, &r)
        }

        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::SuperArrayView(l), Value::FieldArray(r)) => {
            field_array::broadcast_superarrayview_to_fieldarray(op, &l, &r)
        }


        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::FieldArray(field_array), Value::SuperTableView(super_table_view)) => {
            field_array::broadcast_fieldarray_to_supertableview(op, &field_array, &super_table_view)
                .map(|stv| Value::SuperTableView(Arc::new(stv)))
        }

        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::SuperTableView(super_table_view), Value::FieldArray(field_array)) => {
            field_array::broadcast_supertableview_to_fieldarray(op, &super_table_view, &field_array)
                .map(|stv| Value::SuperTableView(Arc::new(stv)))
        }

        // FieldArray-TableView broadcasting - extract array and use existing view functions
        #[cfg(feature = "views")]
        (Value::FieldArray(field_array), Value::TableView(table_view)) => {
            // Extract the table from the view and broadcast
            let table = table_view.to_table();
            broadcast_array_to_table(op, &field_array.array, &table).map(|tbl| Value::Table(Arc::new(tbl)))
        },
        // TableView-FieldArray broadcasting - extract array and use existing view functions
        #[cfg(feature = "views")]
        (Value::TableView(table_view), Value::FieldArray(field_array)) => {
            // Extract the table from the view and broadcast
            let table = table_view.to_table();
            broadcast_table_to_array(op, &table, &field_array.array).map(|tbl| Value::Table(Arc::new(tbl)))
        },


        #[cfg(all(feature = "views", feature = "chunked"))]
        (Value::ArrayView(l), Value::SuperArray(r)) => {
            super_array::broadcast_arrayview_to_superarray(op, &l, &r).map(|sa| Value::SuperArray(Arc::new(sa)))
        }

        #[cfg(all(feature = "views", feature = "chunked"))]
        (Value::SuperArray(l), Value::ArrayView(r)) => {
            super_array::broadcast_superarray_to_arrayview(op, &l, &r).map(|sa| Value::SuperArray(Arc::new(sa)))
        }

        #[cfg(all(feature = "views", feature = "chunked"))]
        (Value::ArrayView(l), Value::SuperArrayView(r)) => {
            super_array::broadcast_arrayview_to_superarrayview(op, &l, &r).map(|sa| Value::SuperArray(Arc::new(sa)))
        }

        #[cfg(all(feature = "views", feature = "chunked"))]
        (Value::SuperArrayView(l), Value::ArrayView(r)) => {
            super_array::broadcast_superarrayview_to_arrayview(op, &l, &r).map(|sa| Value::SuperArray(Arc::new(sa)))
        }

        #[cfg(all(feature = "views", feature = "chunked"))]
        (Value::TableView(table_view), Value::SuperTable(super_table)) => {
            super_table::broadcast_tableview_to_supertable(op, &table_view, &super_table)
        },

        #[cfg(all(feature = "views", feature = "chunked"))]
        (Value::SuperTable(super_table), Value::TableView(table_view)) => {
            super_table::broadcast_supertable_to_tableview(op, &super_table, &table_view)
        },

        #[cfg(all(feature = "views", feature = "chunked"))]
        (Value::TableView(table_view), Value::SuperTableView(super_table_view)) => {
            super_table::broadcast_tableview_to_supertableview(op, &table_view, &super_table_view)
        },

        #[cfg(all(feature = "views", feature = "chunked"))]
        (Value::SuperTableView(super_table_view), Value::TableView(table_view)) => {
            super_table::broadcast_supertableview_to_tableview(op, &super_table_view, &table_view)
        },

        (Value::VecValue(vec), Value::Table(table)) => {
            // Iterate through VecValue elements and broadcast with table
            let results: Result<Vec<_>, _> = Arc::unwrap_or_clone(vec).into_iter()
                .map(|elem| broadcast_value(op, elem, Value::Table(table.clone())))
                .collect();
            Ok(Value::VecValue(Arc::new(results?)))
        }
        (Value::Table(table), Value::VecValue(vec)) => {
            // Iterate through VecValue elements and broadcast with table
            let results: Result<Vec<_>, _> = Arc::unwrap_or_clone(vec).into_iter()
                .map(|elem| broadcast_value(op, Value::Table(table.clone()), elem))
                .collect();
            Ok(Value::VecValue(Arc::new(results?)))
        }
        #[cfg(feature = "views")]
        (Value::VecValue(vec), Value::ArrayView(av)) => {
            // Iterate through VecValue elements and broadcast with ArrayView
            let results: Result<Vec<_>, _> = Arc::unwrap_or_clone(vec).into_iter()
                .map(|elem| broadcast_value(op, elem, Value::ArrayView(av.clone())))
                .collect();
            Ok(Value::VecValue(Arc::new(results?)))
        }
        #[cfg(feature = "views")]
        (Value::ArrayView(av), Value::VecValue(vec)) => {
            // Iterate through VecValue elements and broadcast with ArrayView
            let results: Result<Vec<_>, _> = Arc::unwrap_or_clone(vec).into_iter()
                .map(|elem| broadcast_value(op, Value::ArrayView(av.clone()), elem))
                .collect();
            Ok(Value::VecValue(Arc::new(results?)))
        }
        #[cfg(feature = "views")]
        (Value::VecValue(vec), Value::TableView(tv)) => {
            // Iterate through VecValue elements and broadcast with TableView
            let results: Result<Vec<_>, _> = Arc::unwrap_or_clone(vec).into_iter()
                .map(|elem| broadcast_value(op, elem, Value::TableView(tv.clone())))
                .collect();
            Ok(Value::VecValue(Arc::new(results?)))
        }
        #[cfg(feature = "views")]
        (Value::TableView(tv), Value::VecValue(vec)) => {
            // Iterate through VecValue elements and broadcast with TableView
            let results: Result<Vec<_>, _> = Arc::unwrap_or_clone(vec).into_iter()
                .map(|elem| broadcast_value(op, Value::TableView(tv.clone()), elem))
                .collect();
            Ok(Value::VecValue(Arc::new(results?)))
        }
        #[cfg(feature = "chunked")]
        (Value::VecValue(vec), Value::SuperArray(sa)) => {
            let results: Result<Vec<_>, _> = Arc::unwrap_or_clone(vec).into_iter()
                .map(|elem| broadcast_value(op, elem, Value::SuperArray(sa.clone())))
                .collect();
            Ok(Value::VecValue(Arc::new(results?)))
        }
        #[cfg(feature = "chunked")]
        (Value::SuperArray(sa), Value::VecValue(vec)) => {
            let results: Result<Vec<_>, _> = Arc::unwrap_or_clone(vec).into_iter()
                .map(|elem| broadcast_value(op, Value::SuperArray(sa.clone()), elem))
                .collect();
            Ok(Value::VecValue(Arc::new(results?)))
        }
        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::VecValue(vec), Value::SuperArrayView(sav)) => {
            let results: Result<Vec<_>, _> = Arc::unwrap_or_clone(vec).into_iter()
                .map(|elem| broadcast_value(op, elem, Value::SuperArrayView(sav.clone())))
                .collect();
            Ok(Value::VecValue(Arc::new(results?)))
        }
        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::SuperArrayView(sav), Value::VecValue(vec)) => {
            let results: Result<Vec<_>, _> = Arc::unwrap_or_clone(vec).into_iter()
                .map(|elem| broadcast_value(op, Value::SuperArrayView(sav.clone()), elem))
                .collect();
            Ok(Value::VecValue(Arc::new(results?)))
        }
        #[cfg(feature = "chunked")]
        (Value::VecValue(vec), Value::SuperTable(st)) => {
            let results: Result<Vec<_>, _> = Arc::unwrap_or_clone(vec).into_iter()
                .map(|elem| broadcast_value(op, elem, Value::SuperTable(st.clone())))
                .collect();
            Ok(Value::VecValue(Arc::new(results?)))
        }
        #[cfg(feature = "chunked")]
        (Value::SuperTable(st), Value::VecValue(vec)) => {
            let results: Result<Vec<_>, _> = Arc::unwrap_or_clone(vec).into_iter()
                .map(|elem| broadcast_value(op, Value::SuperTable(st.clone()), elem))
                .collect();
            Ok(Value::VecValue(Arc::new(results?)))
        }
        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::VecValue(vec), Value::SuperTableView(stv)) => {
            let results: Result<Vec<_>, _> = Arc::unwrap_or_clone(vec).into_iter()
                .map(|elem| broadcast_value(op, elem, Value::SuperTableView(stv.clone())))
                .collect();
            Ok(Value::VecValue(Arc::new(results?)))
        }
        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::SuperTableView(stv), Value::VecValue(vec)) => {
            let results: Result<Vec<_>, _> = Arc::unwrap_or_clone(vec).into_iter()
                .map(|elem| broadcast_value(op, Value::SuperTableView(stv.clone()), elem))
                .collect();
            Ok(Value::VecValue(Arc::new(results?)))
        }
        #[cfg(feature = "cube")]
        (Value::VecValue(vec), Value::Cube(cube)) => {
            let results: Result<Vec<_>, _> = Arc::unwrap_or_clone(vec).into_iter()
                .map(|elem| broadcast_value(op, elem, Value::Cube(cube.clone())))
                .collect();
            Ok(Value::VecValue(Arc::new(results?)))
        }
        #[cfg(feature = "cube")]
        (Value::Cube(cube), Value::VecValue(vec)) => {
            let results: Result<Vec<_>, _> = Arc::unwrap_or_clone(vec).into_iter()
                .map(|elem| broadcast_value(op, Value::Cube(cube.clone()), elem))
                .collect();
            Ok(Value::VecValue(Arc::new(results?)))
        }

        // Missing tuple combinations with view types - can use tuple element iteration pattern
        #[cfg(feature = "views")]
        (Value::Tuple2(l_arc), Value::ArrayView(r)) => {
            let (l1, l2) = (l_arc.0.clone(), l_arc.1.clone());
            // Recursive broadcasting with each tuple element
            let res1 = broadcast_value(op, l1, Value::ArrayView(r.clone()))?;
            let res2 = broadcast_value(op, l2, Value::ArrayView(r))?;
            Ok(Value::Tuple2(Arc::new((res1, res2))))
        }
        #[cfg(feature = "views")]
        (Value::ArrayView(l), Value::Tuple2(r_arc)) => {
            let (r1, r2) = (r_arc.0.clone(), r_arc.1.clone());
            // Recursive broadcasting with each tuple element
            let res1 = broadcast_value(op, Value::ArrayView(l.clone()), r1)?;
            let res2 = broadcast_value(op, Value::ArrayView(l), r2)?;
            Ok(Value::Tuple2(Arc::new((res1, res2))))
        }
        #[cfg(feature = "views")]
        (Value::Tuple3(l_arc), Value::ArrayView(r)) => {
            let (l1, l2, l3) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone());
            let res1 = broadcast_value(op, l1, Value::ArrayView(r.clone()))?;
            let res2 = broadcast_value(op, l2, Value::ArrayView(r.clone()))?;
            let res3 = broadcast_value(op, l3, Value::ArrayView(r))?;
            Ok(Value::Tuple3(Arc::new((res1, res2, res3))))
        }
        #[cfg(feature = "views")]
        (Value::ArrayView(l), Value::Tuple3(r_arc)) => {
            let (r1, r2, r3) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone());
            let res1 = broadcast_value(op, Value::ArrayView(l.clone()), r1)?;
            let res2 = broadcast_value(op, Value::ArrayView(l.clone()), r2)?;
            let res3 = broadcast_value(op, Value::ArrayView(l), r3)?;
            Ok(Value::Tuple3(Arc::new((res1, res2, res3))))
        }
        #[cfg(feature = "views")]
        (Value::Tuple4(l_arc), Value::ArrayView(r)) => {
            let (l1, l2, l3, l4) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone());
            let res1 = broadcast_value(op, l1, Value::ArrayView(r.clone()))?;
            let res2 = broadcast_value(op, l2, Value::ArrayView(r.clone()))?;
            let res3 = broadcast_value(op, l3, Value::ArrayView(r.clone()))?;
            let res4 = broadcast_value(op, l4, Value::ArrayView(r))?;
            Ok(Value::Tuple4(Arc::new((res1, res2, res3, res4))))
        }
        #[cfg(feature = "views")]
        (Value::ArrayView(l), Value::Tuple4(r_arc)) => {
            let (r1, r2, r3, r4) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone());
            let res1 = broadcast_value(op, Value::ArrayView(l.clone()), r1)?;
            let res2 = broadcast_value(op, Value::ArrayView(l.clone()), r2)?;
            let res3 = broadcast_value(op, Value::ArrayView(l.clone()), r3)?;
            let res4 = broadcast_value(op, Value::ArrayView(l), r4)?;
            Ok(Value::Tuple4(Arc::new((res1, res2, res3, res4))))
        }
        #[cfg(feature = "views")]
        (Value::Tuple5(l_arc), Value::ArrayView(r)) => {
            let (l1, l2, l3, l4, l5) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone(), l_arc.4.clone());
            let res1 = broadcast_value(op, l1, Value::ArrayView(r.clone()))?;
            let res2 = broadcast_value(op, l2, Value::ArrayView(r.clone()))?;
            let res3 = broadcast_value(op, l3, Value::ArrayView(r.clone()))?;
            let res4 = broadcast_value(op, l4, Value::ArrayView(r.clone()))?;
            let res5 = broadcast_value(op, l5, Value::ArrayView(r))?;
            Ok(Value::Tuple5(Arc::new((res1, res2, res3, res4, res5))))
        }
        #[cfg(feature = "views")]
        (Value::ArrayView(l), Value::Tuple5(r_arc)) => {
            let (r1, r2, r3, r4, r5) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone(), r_arc.4.clone());
            let res1 = broadcast_value(op, Value::ArrayView(l.clone()), r1)?;
            let res2 = broadcast_value(op, Value::ArrayView(l.clone()), r2)?;
            let res3 = broadcast_value(op, Value::ArrayView(l.clone()), r3)?;
            let res4 = broadcast_value(op, Value::ArrayView(l.clone()), r4)?;
            let res5 = broadcast_value(op, Value::ArrayView(l), r5)?;
            Ok(Value::Tuple5(Arc::new((res1, res2, res3, res4, res5))))
        }
        #[cfg(feature = "views")]
        (Value::Tuple6(l_arc), Value::ArrayView(r)) => {
            let (l1, l2, l3, l4, l5, l6) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone(), l_arc.4.clone(), l_arc.5.clone());
            let res1 = broadcast_value(op, l1, Value::ArrayView(r.clone()))?;
            let res2 = broadcast_value(op, l2, Value::ArrayView(r.clone()))?;
            let res3 = broadcast_value(op, l3, Value::ArrayView(r.clone()))?;
            let res4 = broadcast_value(op, l4, Value::ArrayView(r.clone()))?;
            let res5 = broadcast_value(op, l5, Value::ArrayView(r.clone()))?;
            let res6 = broadcast_value(op, l6, Value::ArrayView(r))?;
            Ok(Value::Tuple6(Arc::new((res1, res2, res3, res4, res5, res6))))
        }
        #[cfg(feature = "views")]
        (Value::ArrayView(l), Value::Tuple6(r_arc)) => {
            let (r1, r2, r3, r4, r5, r6) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone(), r_arc.4.clone(), r_arc.5.clone());
            let res1 = broadcast_value(op, Value::ArrayView(l.clone()), r1)?;
            let res2 = broadcast_value(op, Value::ArrayView(l.clone()), r2)?;
            let res3 = broadcast_value(op, Value::ArrayView(l.clone()), r3)?;
            let res4 = broadcast_value(op, Value::ArrayView(l.clone()), r4)?;
            let res5 = broadcast_value(op, Value::ArrayView(l.clone()), r5)?;
            let res6 = broadcast_value(op, Value::ArrayView(l), r6)?;
            Ok(Value::Tuple6(Arc::new((res1, res2, res3, res4, res5, res6))))
        }
        #[cfg(feature = "views")]
        (Value::Tuple2(l_arc), Value::TableView(r)) => {
            let (l1, l2) = (l_arc.0.clone(), l_arc.1.clone());
            // Recursive broadcasting with each tuple element
            let res1 = broadcast_value(op, l1, Value::TableView(r.clone()))?;
            let res2 = broadcast_value(op, l2, Value::TableView(r))?;
            Ok(Value::Tuple2(Arc::new((res1, res2))))
        }
        #[cfg(feature = "views")]
        (Value::TableView(l), Value::Tuple2(r_arc)) => {
            let (r1, r2) = (r_arc.0.clone(), r_arc.1.clone());
            // Recursive broadcasting with each tuple element
            let res1 = broadcast_value(op, Value::TableView(l.clone()), r1)?;
            let res2 = broadcast_value(op, Value::TableView(l), r2)?;
            Ok(Value::Tuple2(Arc::new((res1, res2))))
        }

        #[cfg(feature = "chunked")]
        (Value::Table(table), Value::SuperTable(super_table)) => {
            // Promote Table to SuperTable (single batch) and broadcast
            let promoted = SuperTable::from_batches(vec![table], None);
            broadcast_super_table_with_operator(op, promoted, Arc::unwrap_or_clone(super_table))
                .map(|st| Value::SuperTable(Arc::new(st)))
        }
        #[cfg(feature = "chunked")]
        (Value::SuperTable(super_table), Value::Table(table)) => {
            // Promote Table to SuperTable (single batch) and broadcast
            let promoted = SuperTable::from_batches(vec![table], None);
            broadcast_super_table_with_operator(op, Arc::unwrap_or_clone(super_table), promoted)
                .map(|st| Value::SuperTable(Arc::new(st)))
        }
        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::Table(table), Value::SuperTableView(super_table_view)) => {
            super_table_view::broadcast_table_to_supertableview(op, &table, &super_table_view).map(|stv| Value::SuperTableView(Arc::new(stv)))
        }
        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::SuperTableView(super_table_view), Value::Table(table)) => {
            super_table_view::broadcast_supertableview_to_table(op, &super_table_view, &table).map(|stv| Value::SuperTableView(Arc::new(stv)))
        }

        // Complete tuple combinations with remaining types (all remaining Tuple3-6 patterns)
        #[cfg(feature = "views")]
        (Value::Tuple3(l_arc), Value::TableView(r)) => {
            let (l1, l2, l3) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone());
            let res1 = broadcast_value(op, l1, Value::TableView(r.clone()))?;
            let res2 = broadcast_value(op, l2, Value::TableView(r.clone()))?;
            let res3 = broadcast_value(op, l3, Value::TableView(r))?;
            Ok(Value::Tuple3(Arc::new((res1, res2, res3))))
        }
        #[cfg(feature = "views")]
        (Value::TableView(l), Value::Tuple3(r_arc)) => {
            let (r1, r2, r3) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone());
            let res1 = broadcast_value(op, Value::TableView(l.clone()), r1)?;
            let res2 = broadcast_value(op, Value::TableView(l.clone()), r2)?;
            let res3 = broadcast_value(op, Value::TableView(l), r3)?;
            Ok(Value::Tuple3(Arc::new((res1, res2, res3))))
        }
        #[cfg(feature = "views")]
        (Value::Tuple4(l_arc), Value::TableView(r)) => {
            let (l1, l2, l3, l4) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone());
            let res1 = broadcast_value(op, l1, Value::TableView(r.clone()))?;
            let res2 = broadcast_value(op, l2, Value::TableView(r.clone()))?;
            let res3 = broadcast_value(op, l3, Value::TableView(r.clone()))?;
            let res4 = broadcast_value(op, l4, Value::TableView(r))?;
            Ok(Value::Tuple4(Arc::new((res1, res2, res3, res4))))
        }
        #[cfg(feature = "views")]
        (Value::TableView(l), Value::Tuple4(r_arc)) => {
            let (r1, r2, r3, r4) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone());
            let res1 = broadcast_value(op, Value::TableView(l.clone()), r1)?;
            let res2 = broadcast_value(op, Value::TableView(l.clone()), r2)?;
            let res3 = broadcast_value(op, Value::TableView(l.clone()), r3)?;
            let res4 = broadcast_value(op, Value::TableView(l), r4)?;
            Ok(Value::Tuple4(Arc::new((res1, res2, res3, res4))))
        }
        #[cfg(feature = "views")]
        (Value::Tuple5(l_arc), Value::TableView(r)) => {
            let (l1, l2, l3, l4, l5) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone(), l_arc.4.clone());
            let res1 = broadcast_value(op, l1, Value::TableView(r.clone()))?;
            let res2 = broadcast_value(op, l2, Value::TableView(r.clone()))?;
            let res3 = broadcast_value(op, l3, Value::TableView(r.clone()))?;
            let res4 = broadcast_value(op, l4, Value::TableView(r.clone()))?;
            let res5 = broadcast_value(op, l5, Value::TableView(r))?;
            Ok(Value::Tuple5(Arc::new((res1, res2, res3, res4, res5))))
        }
        #[cfg(feature = "views")]
        (Value::TableView(l), Value::Tuple5(r_arc)) => {
            let (r1, r2, r3, r4, r5) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone(), r_arc.4.clone());
            let res1 = broadcast_value(op, Value::TableView(l.clone()), r1)?;
            let res2 = broadcast_value(op, Value::TableView(l.clone()), r2)?;
            let res3 = broadcast_value(op, Value::TableView(l.clone()), r3)?;
            let res4 = broadcast_value(op, Value::TableView(l.clone()), r4)?;
            let res5 = broadcast_value(op, Value::TableView(l), r5)?;
            Ok(Value::Tuple5(Arc::new((res1, res2, res3, res4, res5))))
        }
        #[cfg(feature = "views")]
        (Value::Tuple6(l_arc), Value::TableView(r)) => {
            let (l1, l2, l3, l4, l5, l6) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone(), l_arc.4.clone(), l_arc.5.clone());
            let res1 = broadcast_value(op, l1, Value::TableView(r.clone()))?;
            let res2 = broadcast_value(op, l2, Value::TableView(r.clone()))?;
            let res3 = broadcast_value(op, l3, Value::TableView(r.clone()))?;
            let res4 = broadcast_value(op, l4, Value::TableView(r.clone()))?;
            let res5 = broadcast_value(op, l5, Value::TableView(r.clone()))?;
            let res6 = broadcast_value(op, l6, Value::TableView(r))?;
            Ok(Value::Tuple6(Arc::new((res1, res2, res3, res4, res5, res6))))
        }
        #[cfg(feature = "views")]
        (Value::TableView(l), Value::Tuple6(r_arc)) => {
            let (r1, r2, r3, r4, r5, r6) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone(), r_arc.4.clone(), r_arc.5.clone());
            let res1 = broadcast_value(op, Value::TableView(l.clone()), r1)?;
            let res2 = broadcast_value(op, Value::TableView(l.clone()), r2)?;
            let res3 = broadcast_value(op, Value::TableView(l.clone()), r3)?;
            let res4 = broadcast_value(op, Value::TableView(l.clone()), r4)?;
            let res5 = broadcast_value(op, Value::TableView(l.clone()), r5)?;
            let res6 = broadcast_value(op, Value::TableView(l), r6)?;
            Ok(Value::Tuple6(Arc::new((res1, res2, res3, res4, res5, res6))))
        }

        #[cfg(feature = "cube")]
        (Value::Tuple2(l_arc), Value::Cube(cube)) => {
            let (l1, l2) = (l_arc.0.clone(), l_arc.1.clone());
            let res1 = broadcast_value(op, l1, Value::Cube(cube.clone()))?;
            let res2 = broadcast_value(op, l2, Value::Cube(cube))?;
            Ok(Value::Tuple2(Arc::new((res1, res2))))
        }
        #[cfg(feature = "cube")]
        (Value::Cube(cube), Value::Tuple2(r_arc)) => {
            let (r1, r2) = (r_arc.0.clone(), r_arc.1.clone());
            let res1 = broadcast_value(op, Value::Cube(cube.clone()), r1)?;
            let res2 = broadcast_value(op, Value::Cube(cube), r2)?;
            Ok(Value::Tuple2(Arc::new((res1, res2))))
        }
        #[cfg(feature = "cube")]
        (Value::Tuple3(l_arc), Value::Cube(cube)) => {
            let (l1, l2, l3) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone());
            let res1 = broadcast_value(op, l1, Value::Cube(cube.clone()))?;
            let res2 = broadcast_value(op, l2, Value::Cube(cube.clone()))?;
            let res3 = broadcast_value(op, l3, Value::Cube(cube))?;
            Ok(Value::Tuple3(Arc::new((res1, res2, res3))))
        }
        #[cfg(feature = "cube")]
        (Value::Cube(cube), Value::Tuple3(r_arc)) => {
            let (r1, r2, r3) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone());
            let res1 = broadcast_value(op, Value::Cube(cube.clone()), r1)?;
            let res2 = broadcast_value(op, Value::Cube(cube.clone()), r2)?;
            let res3 = broadcast_value(op, Value::Cube(cube), r3)?;
            Ok(Value::Tuple3(Arc::new((res1, res2, res3))))
        }
        #[cfg(feature = "cube")]
        (Value::Tuple4(l_arc), Value::Cube(cube)) => {
            let (l1, l2, l3, l4) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone());
            let res1 = broadcast_value(op, l1, Value::Cube(cube.clone()))?;
            let res2 = broadcast_value(op, l2, Value::Cube(cube.clone()))?;
            let res3 = broadcast_value(op, l3, Value::Cube(cube.clone()))?;
            let res4 = broadcast_value(op, l4, Value::Cube(cube))?;
            Ok(Value::Tuple4(Arc::new((res1, res2, res3, res4))))
        }
        #[cfg(feature = "cube")]
        (Value::Cube(cube), Value::Tuple4(r_arc)) => {
            let (r1, r2, r3, r4) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone());
            let res1 = broadcast_value(op, Value::Cube(cube.clone()), r1)?;
            let res2 = broadcast_value(op, Value::Cube(cube.clone()), r2)?;
            let res3 = broadcast_value(op, Value::Cube(cube.clone()), r3)?;
            let res4 = broadcast_value(op, Value::Cube(cube), r4)?;
            Ok(Value::Tuple4(Arc::new((res1, res2, res3, res4))))
        }
        #[cfg(feature = "cube")]
        (Value::Tuple5(l_arc), Value::Cube(cube)) => {
            let (l1, l2, l3, l4, l5) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone(), l_arc.4.clone());
            let res1 = broadcast_value(op, l1, Value::Cube(cube.clone()))?;
            let res2 = broadcast_value(op, l2, Value::Cube(cube.clone()))?;
            let res3 = broadcast_value(op, l3, Value::Cube(cube.clone()))?;
            let res4 = broadcast_value(op, l4, Value::Cube(cube.clone()))?;
            let res5 = broadcast_value(op, l5, Value::Cube(cube))?;
            Ok(Value::Tuple5(Arc::new((res1, res2, res3, res4, res5))))
        }
        #[cfg(feature = "cube")]
        (Value::Cube(cube), Value::Tuple5(r_arc)) => {
            let (r1, r2, r3, r4, r5) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone(), r_arc.4.clone());
            let res1 = broadcast_value(op, Value::Cube(cube.clone()), r1)?;
            let res2 = broadcast_value(op, Value::Cube(cube.clone()), r2)?;
            let res3 = broadcast_value(op, Value::Cube(cube.clone()), r3)?;
            let res4 = broadcast_value(op, Value::Cube(cube.clone()), r4)?;
            let res5 = broadcast_value(op, Value::Cube(cube), r5)?;
            Ok(Value::Tuple5(Arc::new((res1, res2, res3, res4, res5))))
        }
        #[cfg(feature = "cube")]
        (Value::Tuple6(l_arc), Value::Cube(cube)) => {
            let (l1, l2, l3, l4, l5, l6) = (l_arc.0.clone(), l_arc.1.clone(), l_arc.2.clone(), l_arc.3.clone(), l_arc.4.clone(), l_arc.5.clone());
            let res1 = broadcast_value(op, l1, Value::Cube(cube.clone()))?;
            let res2 = broadcast_value(op, l2, Value::Cube(cube.clone()))?;
            let res3 = broadcast_value(op, l3, Value::Cube(cube.clone()))?;
            let res4 = broadcast_value(op, l4, Value::Cube(cube.clone()))?;
            let res5 = broadcast_value(op, l5, Value::Cube(cube.clone()))?;
            let res6 = broadcast_value(op, l6, Value::Cube(cube))?;
            Ok(Value::Tuple6(Arc::new((res1, res2, res3, res4, res5, res6))))
        }
        #[cfg(feature = "cube")]
        (Value::Cube(cube), Value::Tuple6(r_arc)) => {
            let (r1, r2, r3, r4, r5, r6) = (r_arc.0.clone(), r_arc.1.clone(), r_arc.2.clone(), r_arc.3.clone(), r_arc.4.clone(), r_arc.5.clone());
            let res1 = broadcast_value(op, Value::Cube(cube.clone()), r1)?;
            let res2 = broadcast_value(op, Value::Cube(cube.clone()), r2)?;
            let res3 = broadcast_value(op, Value::Cube(cube.clone()), r3)?;
            let res4 = broadcast_value(op, Value::Cube(cube.clone()), r4)?;
            let res5 = broadcast_value(op, Value::Cube(cube.clone()), r5)?;
            let res6 = broadcast_value(op, Value::Cube(cube), r6)?;
            Ok(Value::Tuple6(Arc::new((res1, res2, res3, res4, res5, res6))))
        }

        #[cfg(feature = "chunked")]
        (Value::SuperArray(super_array), Value::SuperTable(super_table)) => {
            super_table::broadcast_superarray_to_supertable(op, &super_array, &super_table).map(|st| Value::SuperTable(Arc::new(st)))
        }

        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::SuperArray(super_array), Value::SuperTableView(super_table_view)) => {
            super_table::broadcast_superarray_to_supertableview(op, &super_array, &super_table_view).map(|st| Value::SuperTable(Arc::new(st)))
        }

        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::SuperArrayView(super_array_view), Value::SuperTable(super_table)) => {
            super_table::broadcast_superarrayview_to_supertable(op, &super_array_view, &super_table).map(|st| Value::SuperTable(Arc::new(st)))
        }

        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::SuperArrayView(super_array_view), Value::SuperTableView(super_table_view)) => {
            super_table::broadcast_superarrayview_to_supertableview(op, &super_array_view, &super_table_view).map(|st| Value::SuperTable(Arc::new(st)))
        }

        #[cfg(feature = "chunked")]
        (Value::SuperTable(super_table), Value::SuperArray(super_array)) => {
            super_table::broadcast_supertable_to_superarray(op, &super_table, &super_array).map(|st| Value::SuperTable(Arc::new(st)))
        }

        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::SuperTable(super_table), Value::SuperArrayView(super_array_view)) => {
            super_table::broadcast_supertable_to_superarrayview(op, &super_table, &super_array_view).map(|st| Value::SuperTable(Arc::new(st)))
        }

        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::SuperTableView(super_table_view), Value::SuperArray(super_array)) => {
            super_table::broadcast_supertableview_to_superarray(op, &super_table_view, &super_array).map(|st| Value::SuperTable(Arc::new(st)))
        }

        #[cfg(all(feature = "chunked", feature = "views"))]
        (Value::SuperTableView(super_table_view), Value::SuperArrayView(super_array_view)) => {
            super_table::broadcast_supertableview_to_superarrayview(op, &super_table_view, &super_array_view).map(|st| Value::SuperTable(Arc::new(st)))
        }

        // Recursive cases
        (Value::BoxValue(l), Value::BoxValue(r)) => {
            broadcast_value(op, *l, *r).map(|v| Value::BoxValue(Box::new(v)))
        }

        // BoxValue with other types - recursively unbox and compute
        (Value::BoxValue(l), r) => broadcast_value(op, *l, r).map(|v| Value::BoxValue(Box::new(v))),

        (l, Value::BoxValue(r)) => broadcast_value(op, l, *r).map(|v| Value::BoxValue(Box::new(v))),

        // We choose not to support this. Users can loop through it if required.
        (Value::VecValue(_), _) | (_, Value::VecValue(_)) => Err(MinarrowError::TypeError {
            from: "VecValue and other types",
            to: "compatible broadcasting types",
            message: Some(
                "VecValue arithmetic not supported - use element-wise iteration instead".to_string(),
            ),
        }),
    }
}
