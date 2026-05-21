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

//! Conversion implementations and macros for Value.
//!
//! Contains From/TryFrom implementations and value creation macros.

use super::Value;
use super::impls::value_variant_name;
#[cfg(feature = "cube")]
use crate::Cube;
#[cfg(feature = "matrix")]
use crate::Matrix;
#[cfg(feature = "scalar_type")]
use crate::Scalar;
use crate::traits::concatenate::Concatenate;
use crate::traits::masked_array::MaskedArray;
use crate::{Array, FieldArray, Table, enums::error::MinarrowError};
use std::convert::TryFrom;
use std::sync::Arc;

#[cfg(feature = "chunked")]
use crate::{SuperArray, SuperTable};

#[cfg(feature = "views")]
use crate::{ArrayV, BitmaskV, BooleanArrayV, NumericArrayV, TableV, TextArrayV};

#[cfg(all(feature = "views", feature = "datetime"))]
use crate::TemporalArrayV;

#[cfg(all(feature = "chunked", feature = "views"))]
use crate::{SuperArrayV, SuperTableV};

use crate::traits::custom_value::CustomValue;

// Typed Accessors

impl Value {
    /// Returns the inner `Scalar` if this is a `Value::Scalar`.
    ///
    /// Panics if the value is not a `Scalar` variant.
    #[cfg(feature = "scalar_type")]
    #[inline]
    pub fn scalar(&self) -> &Scalar {
        match self {
            Value::Scalar(s) => s,
            _ => panic!("Expected Value::Scalar, found {}", value_variant_name(self)),
        }
    }

    /// Returns the inner `Scalar` if this is a `Value::Scalar`, or an error otherwise.
    #[cfg(feature = "scalar_type")]
    #[inline]
    pub fn try_scalar(&self) -> Result<&Scalar, MinarrowError> {
        match self {
            Value::Scalar(s) => Ok(s),
            _ => Err(MinarrowError::TypeError {
                from: value_variant_name(self),
                to: "Scalar",
                message: None,
            }),
        }
    }

    /// Returns the inner `Array` if this is a `Value::Array`.
    ///
    /// Panics if the value is not an `Array` variant.
    #[inline]
    pub fn arr(&self) -> &Array {
        match self {
            Value::Array(a) => a,
            _ => panic!("Expected Value::Array, found {}", value_variant_name(self)),
        }
    }

    /// Returns the inner `Array` if this is a `Value::Array`, or an error otherwise.
    #[inline]
    pub fn try_arr(&self) -> Result<&Array, MinarrowError> {
        match self {
            Value::Array(a) => Ok(a),
            _ => Err(MinarrowError::TypeError {
                from: value_variant_name(self),
                to: "Array",
                message: None,
            }),
        }
    }

    /// Returns the inner `ArrayV` if this is a `Value::ArrayView`.
    ///
    /// Panics if the value is not an `ArrayView` variant.
    #[cfg(feature = "views")]
    #[inline]
    pub fn av(&self) -> &ArrayV {
        match self {
            Value::ArrayView(av) => av,
            _ => panic!(
                "Expected Value::ArrayView, found {}",
                value_variant_name(self)
            ),
        }
    }

    /// Returns the inner `ArrayV` if this is a `Value::ArrayView`, or an error otherwise.
    #[cfg(feature = "views")]
    #[inline]
    pub fn try_av(&self) -> Result<&ArrayV, MinarrowError> {
        match self {
            Value::ArrayView(av) => Ok(av),
            _ => Err(MinarrowError::TypeError {
                from: value_variant_name(self),
                to: "ArrayV",
                message: None,
            }),
        }
    }

    /// Returns the inner `FieldArray` if this is a `Value::FieldArray`.
    ///
    /// Panics if the value is not a `FieldArray` variant.
    #[inline]
    pub fn fa(&self) -> &FieldArray {
        match self {
            Value::FieldArray(fa) => fa,
            _ => panic!(
                "Expected Value::FieldArray, found {}",
                value_variant_name(self)
            ),
        }
    }

    /// Returns the inner `FieldArray` if this is a `Value::FieldArray`, or an error otherwise.
    #[inline]
    pub fn try_fa(&self) -> Result<&FieldArray, MinarrowError> {
        match self {
            Value::FieldArray(fa) => Ok(fa),
            _ => Err(MinarrowError::TypeError {
                from: value_variant_name(self),
                to: "FieldArray",
                message: None,
            }),
        }
    }

    /// Returns the inner `Table` if this is a `Value::Table`.
    ///
    /// Panics if the value is not a `Table` variant.
    #[inline]
    pub fn table(&self) -> &Table {
        match self {
            Value::Table(t) => t,
            _ => panic!("Expected Value::Table, found {}", value_variant_name(self)),
        }
    }

    /// Returns the inner `Table` if this is a `Value::Table`, or an error otherwise.
    #[inline]
    pub fn try_table(&self) -> Result<&Table, MinarrowError> {
        match self {
            Value::Table(t) => Ok(t),
            _ => Err(MinarrowError::TypeError {
                from: value_variant_name(self),
                to: "Table",
                message: None,
            }),
        }
    }

    /// Returns the inner `TableV` if this is a `Value::TableView`.
    ///
    /// Panics if the value is not a `TableView` variant.
    #[cfg(feature = "views")]
    #[inline]
    pub fn tv(&self) -> &TableV {
        match self {
            Value::TableView(tv) => tv,
            _ => panic!(
                "Expected Value::TableView, found {}",
                value_variant_name(self)
            ),
        }
    }

    /// Returns the inner `TableV` if this is a `Value::TableView`, or an error otherwise.
    #[cfg(feature = "views")]
    #[inline]
    pub fn try_tv(&self) -> Result<&TableV, MinarrowError> {
        match self {
            Value::TableView(tv) => Ok(tv),
            _ => Err(MinarrowError::TypeError {
                from: value_variant_name(self),
                to: "TableV",
                message: None,
            }),
        }
    }

    /// Returns the inner `SuperArray` if this is a `Value::SuperArray`.
    ///
    /// Panics if the value is not a `SuperArray` variant.
    #[cfg(feature = "chunked")]
    #[inline]
    pub fn sa(&self) -> &SuperArray {
        match self {
            Value::SuperArray(sa) => sa,
            _ => panic!(
                "Expected Value::SuperArray, found {}",
                value_variant_name(self)
            ),
        }
    }

    /// Returns the inner `SuperArray` if this is a `Value::SuperArray`, or an error otherwise.
    #[cfg(feature = "chunked")]
    #[inline]
    pub fn try_sa(&self) -> Result<&SuperArray, MinarrowError> {
        match self {
            Value::SuperArray(sa) => Ok(sa),
            _ => Err(MinarrowError::TypeError {
                from: value_variant_name(self),
                to: "SuperArray",
                message: None,
            }),
        }
    }

    /// Returns the inner `SuperArrayV` if this is a `Value::SuperArrayView`.
    ///
    /// Panics if the value is not a `SuperArrayView` variant.
    #[cfg(all(feature = "chunked", feature = "views"))]
    #[inline]
    pub fn sav(&self) -> &SuperArrayV {
        match self {
            Value::SuperArrayView(sav) => sav,
            _ => panic!(
                "Expected Value::SuperArrayView, found {}",
                value_variant_name(self)
            ),
        }
    }

    /// Returns the inner `SuperArrayV` if this is a `Value::SuperArrayView`, or an error otherwise.
    #[cfg(all(feature = "chunked", feature = "views"))]
    #[inline]
    pub fn try_sav(&self) -> Result<&SuperArrayV, MinarrowError> {
        match self {
            Value::SuperArrayView(sav) => Ok(sav),
            _ => Err(MinarrowError::TypeError {
                from: value_variant_name(self),
                to: "SuperArrayV",
                message: None,
            }),
        }
    }

    /// Returns the inner `SuperTable` if this is a `Value::SuperTable`.
    ///
    /// Panics if the value is not a `SuperTable` variant.
    #[cfg(feature = "chunked")]
    #[inline]
    pub fn st(&self) -> &SuperTable {
        match self {
            Value::SuperTable(st) => st,
            _ => panic!(
                "Expected Value::SuperTable, found {}",
                value_variant_name(self)
            ),
        }
    }

    /// Returns the inner `SuperTable` if this is a `Value::SuperTable`, or an error otherwise.
    #[cfg(feature = "chunked")]
    #[inline]
    pub fn try_st(&self) -> Result<&SuperTable, MinarrowError> {
        match self {
            Value::SuperTable(st) => Ok(st),
            _ => Err(MinarrowError::TypeError {
                from: value_variant_name(self),
                to: "SuperTable",
                message: None,
            }),
        }
    }

    /// Returns the inner `SuperTableV` if this is a `Value::SuperTableView`.
    ///
    /// Panics if the value is not a `SuperTableView` variant.
    #[cfg(all(feature = "chunked", feature = "views"))]
    #[inline]
    pub fn stv(&self) -> &SuperTableV {
        match self {
            Value::SuperTableView(stv) => stv,
            _ => panic!(
                "Expected Value::SuperTableView, found {}",
                value_variant_name(self)
            ),
        }
    }

    /// Returns the inner `SuperTableV` if this is a `Value::SuperTableView`, or an error otherwise.
    #[cfg(all(feature = "chunked", feature = "views"))]
    #[inline]
    pub fn try_stv(&self) -> Result<&SuperTableV, MinarrowError> {
        match self {
            Value::SuperTableView(stv) => Ok(stv),
            _ => Err(MinarrowError::TypeError {
                from: value_variant_name(self),
                to: "SuperTableV",
                message: None,
            }),
        }
    }

    /// Returns the inner `Matrix` if this is a `Value::Matrix`.
    ///
    /// Panics if the value is not a `Matrix` variant.
    #[cfg(feature = "matrix")]
    #[inline]
    pub fn mat(&self) -> &Matrix {
        match self {
            Value::Matrix(m) => m,
            _ => panic!("Expected Value::Matrix, found {}", value_variant_name(self)),
        }
    }

    /// Returns the inner `Matrix` if this is a `Value::Matrix`, or an error otherwise.
    #[cfg(feature = "matrix")]
    #[inline]
    pub fn try_mat(&self) -> Result<&Matrix, MinarrowError> {
        match self {
            Value::Matrix(m) => Ok(m),
            _ => Err(MinarrowError::TypeError {
                from: value_variant_name(self),
                to: "Matrix",
                message: None,
            }),
        }
    }

    /// Returns the inner `Cube` if this is a `Value::Cube`.
    ///
    /// Panics if the value is not a `Cube` variant.
    #[cfg(feature = "cube")]
    #[inline]
    pub fn cube(&self) -> &Cube {
        match self {
            Value::Cube(c) => c,
            _ => panic!("Expected Value::Cube, found {}", value_variant_name(self)),
        }
    }

    /// Returns the inner `Cube` if this is a `Value::Cube`, or an error otherwise.
    #[cfg(feature = "cube")]
    #[inline]
    pub fn try_cube(&self) -> Result<&Cube, MinarrowError> {
        match self {
            Value::Cube(c) => Ok(c),
            _ => Err(MinarrowError::TypeError {
                from: value_variant_name(self),
                to: "Cube",
                message: None,
            }),
        }
    }

    /// Returns the inner `Vec<Value>` if this is a `Value::VecValue`.
    ///
    /// Panics if the value is not a `VecValue` variant.
    #[inline]
    pub fn vec_val(&self) -> &Vec<Value> {
        match self {
            Value::VecValue(v) => v,
            _ => panic!(
                "Expected Value::VecValue, found {}",
                value_variant_name(self)
            ),
        }
    }

    /// Returns the inner `Vec<Value>` if this is a `Value::VecValue`, or an error otherwise.
    #[inline]
    pub fn try_vec_val(&self) -> Result<&Vec<Value>, MinarrowError> {
        match self {
            Value::VecValue(v) => Ok(v),
            _ => Err(MinarrowError::TypeError {
                from: value_variant_name(self),
                to: "Vec<Value>",
                message: None,
            }),
        }
    }

    /// Returns the inner `Value` if this is a `Value::BoxValue`.
    ///
    /// Panics if the value is not a `BoxValue` variant.
    #[inline]
    pub fn box_val(&self) -> &Value {
        match self {
            Value::BoxValue(b) => b,
            _ => panic!(
                "Expected Value::BoxValue, found {}",
                value_variant_name(self)
            ),
        }
    }

    /// Returns the inner `Value` if this is a `Value::BoxValue`, or an error otherwise.
    #[inline]
    pub fn try_box_val(&self) -> Result<&Value, MinarrowError> {
        match self {
            Value::BoxValue(b) => Ok(b),
            _ => Err(MinarrowError::TypeError {
                from: value_variant_name(self),
                to: "Box<Value>",
                message: None,
            }),
        }
    }

    /// Returns the inner `Value` if this is a `Value::ArcValue`.
    ///
    /// Panics if the value is not an `ArcValue` variant.
    #[inline]
    pub fn arc_val(&self) -> &Value {
        match self {
            Value::ArcValue(a) => a,
            _ => panic!(
                "Expected Value::ArcValue, found {}",
                value_variant_name(self)
            ),
        }
    }

    /// Returns the inner `Value` if this is a `Value::ArcValue`, or an error otherwise.
    #[inline]
    pub fn try_arc_val(&self) -> Result<&Value, MinarrowError> {
        match self {
            Value::ArcValue(a) => Ok(a),
            _ => Err(MinarrowError::TypeError {
                from: value_variant_name(self),
                to: "Arc<Value>",
                message: None,
            }),
        }
    }

    /// Returns the inner tuple if this is a `Value::Tuple2`.
    ///
    /// Panics if the value is not a `Tuple2` variant.
    #[inline]
    pub fn t2(&self) -> &(Value, Value) {
        match self {
            Value::Tuple2(t) => t,
            _ => panic!("Expected Value::Tuple2, found {}", value_variant_name(self)),
        }
    }

    /// Returns the inner tuple if this is a `Value::Tuple2`, or an error otherwise.
    #[inline]
    pub fn try_t2(&self) -> Result<&(Value, Value), MinarrowError> {
        match self {
            Value::Tuple2(t) => Ok(t),
            _ => Err(MinarrowError::TypeError {
                from: value_variant_name(self),
                to: "(Value, Value)",
                message: None,
            }),
        }
    }

    /// Returns the inner tuple if this is a `Value::Tuple3`.
    ///
    /// Panics if the value is not a `Tuple3` variant.
    #[inline]
    pub fn t3(&self) -> &(Value, Value, Value) {
        match self {
            Value::Tuple3(t) => t,
            _ => panic!("Expected Value::Tuple3, found {}", value_variant_name(self)),
        }
    }

    /// Returns the inner tuple if this is a `Value::Tuple3`, or an error otherwise.
    #[inline]
    pub fn try_t3(&self) -> Result<&(Value, Value, Value), MinarrowError> {
        match self {
            Value::Tuple3(t) => Ok(t),
            _ => Err(MinarrowError::TypeError {
                from: value_variant_name(self),
                to: "(Value, Value, Value)",
                message: None,
            }),
        }
    }

    /// Returns the inner tuple if this is a `Value::Tuple4`.
    ///
    /// Panics if the value is not a `Tuple4` variant.
    #[inline]
    pub fn t4(&self) -> &(Value, Value, Value, Value) {
        match self {
            Value::Tuple4(t) => t,
            _ => panic!("Expected Value::Tuple4, found {}", value_variant_name(self)),
        }
    }

    /// Returns the inner tuple if this is a `Value::Tuple4`, or an error otherwise.
    #[inline]
    pub fn try_t4(&self) -> Result<&(Value, Value, Value, Value), MinarrowError> {
        match self {
            Value::Tuple4(t) => Ok(t),
            _ => Err(MinarrowError::TypeError {
                from: value_variant_name(self),
                to: "(Value, Value, Value, Value)",
                message: None,
            }),
        }
    }

    /// Returns the inner tuple if this is a `Value::Tuple5`.
    ///
    /// Panics if the value is not a `Tuple5` variant.
    #[inline]
    pub fn t5(&self) -> &(Value, Value, Value, Value, Value) {
        match self {
            Value::Tuple5(t) => t,
            _ => panic!("Expected Value::Tuple5, found {}", value_variant_name(self)),
        }
    }

    /// Returns the inner tuple if this is a `Value::Tuple5`, or an error otherwise.
    #[inline]
    pub fn try_t5(&self) -> Result<&(Value, Value, Value, Value, Value), MinarrowError> {
        match self {
            Value::Tuple5(t) => Ok(t),
            _ => Err(MinarrowError::TypeError {
                from: value_variant_name(self),
                to: "(Value, Value, Value, Value, Value)",
                message: None,
            }),
        }
    }

    /// Returns the inner tuple if this is a `Value::Tuple6`.
    ///
    /// Panics if the value is not a `Tuple6` variant.
    #[inline]
    pub fn t6(&self) -> &(Value, Value, Value, Value, Value, Value) {
        match self {
            Value::Tuple6(t) => t,
            _ => panic!("Expected Value::Tuple6, found {}", value_variant_name(self)),
        }
    }

    /// Returns the inner tuple if this is a `Value::Tuple6`, or an error otherwise.
    #[inline]
    pub fn try_t6(&self) -> Result<&(Value, Value, Value, Value, Value, Value), MinarrowError> {
        match self {
            Value::Tuple6(t) => Ok(t),
            _ => Err(MinarrowError::TypeError {
                from: value_variant_name(self),
                to: "(Value, Value, Value, Value, Value, Value)",
                message: None,
            }),
        }
    }

    /// Returns the inner `CustomValue` trait object if this is a `Value::Custom`.
    ///
    /// Panics if the value is not a `Custom` variant.
    #[inline]
    pub fn custom(&self) -> &dyn CustomValue {
        match self {
            Value::Custom(c) => c.as_ref(),
            _ => panic!("Expected Value::Custom, found {}", value_variant_name(self)),
        }
    }

    /// Returns the inner `CustomValue` trait object if this is a `Value::Custom`, or an error otherwise.
    #[inline]
    pub fn try_custom(&self) -> Result<&dyn CustomValue, MinarrowError> {
        match self {
            Value::Custom(c) => Ok(c.as_ref()),
            _ => Err(MinarrowError::TypeError {
                from: value_variant_name(self),
                to: "Custom",
                message: None,
            }),
        }
    }
}

// Conversion Macros

/// Macro to implement `From` for `Value` variants.
macro_rules! impl_value_from {
    ($variant:ident: $t:ty) => {
        impl From<$t> for Value {
            #[inline]
            fn from(v: $t) -> Self {
                Value::$variant(v)
            }
        }
    };
}

/// Macro to implement `TryFrom<Value>` for `Value` variants.
macro_rules! impl_tryfrom_value {
    ($variant:ident: $t:ty) => {
        impl TryFrom<Value> for $t {
            type Error = MinarrowError;
            fn try_from(v: Value) -> Result<Self, Self::Error> {
                match v {
                    Value::$variant(inner) => Ok(inner),
                    _ => Err(MinarrowError::TypeError {
                        from: "Value",
                        to: stringify!($t),
                        message: Some("Value type mismatch".to_owned()),
                    }),
                }
            }
        }
    };
}

// Scalar Conversions

#[cfg(feature = "scalar_type")]
impl_value_from!(Scalar: Scalar);
#[cfg(feature = "scalar_type")]
impl_tryfrom_value!(Scalar: Scalar);

// Primitive numerics -> Value (wrap as Value::Scalar).

#[cfg(feature = "scalar_type")]
impl From<f64> for Value {
    #[inline]
    fn from(v: f64) -> Self {
        Value::Scalar(Scalar::Float64(v))
    }
}

#[cfg(feature = "scalar_type")]
impl From<f32> for Value {
    #[inline]
    fn from(v: f32) -> Self {
        Value::Scalar(Scalar::Float32(v))
    }
}

#[cfg(feature = "scalar_type")]
impl From<i64> for Value {
    #[inline]
    fn from(v: i64) -> Self {
        Value::Scalar(Scalar::Int64(v))
    }
}

#[cfg(feature = "scalar_type")]
impl From<i32> for Value {
    #[inline]
    fn from(v: i32) -> Self {
        Value::Scalar(Scalar::Int32(v))
    }
}

#[cfg(feature = "scalar_type")]
impl From<u64> for Value {
    #[inline]
    fn from(v: u64) -> Self {
        Value::Scalar(Scalar::UInt64(v))
    }
}

#[cfg(feature = "scalar_type")]
impl From<u32> for Value {
    #[inline]
    fn from(v: u32) -> Self {
        Value::Scalar(Scalar::UInt32(v))
    }
}

#[cfg(feature = "scalar_type")]
impl From<bool> for Value {
    #[inline]
    fn from(v: bool) -> Self {
        Value::Scalar(Scalar::Boolean(v))
    }
}

#[cfg(all(feature = "scalar_type", feature = "extended_numeric_types"))]
impl From<i8> for Value {
    #[inline]
    fn from(v: i8) -> Self {
        Value::Scalar(Scalar::Int8(v))
    }
}

#[cfg(all(feature = "scalar_type", feature = "extended_numeric_types"))]
impl From<i16> for Value {
    #[inline]
    fn from(v: i16) -> Self {
        Value::Scalar(Scalar::Int16(v))
    }
}

#[cfg(all(feature = "scalar_type", feature = "extended_numeric_types"))]
impl From<u8> for Value {
    #[inline]
    fn from(v: u8) -> Self {
        Value::Scalar(Scalar::UInt8(v))
    }
}

#[cfg(all(feature = "scalar_type", feature = "extended_numeric_types"))]
impl From<u16> for Value {
    #[inline]
    fn from(v: u16) -> Self {
        Value::Scalar(Scalar::UInt16(v))
    }
}

// TryFrom<Value> for primitive numeric types
//
// Enables DataStream<T>::collect() -> Result<T, BlockError> for numeric types.
// Extracts Value::Scalar variant and converts to the target numeric type.

#[cfg(feature = "scalar_type")]
macro_rules! impl_tryfrom_value_numeric {
    ($t:ty, $method:ident) => {
        impl TryFrom<Value> for $t {
            type Error = MinarrowError;
            #[inline]
            fn try_from(v: Value) -> Result<Self, Self::Error> {
                match v {
                    Value::Scalar(s) => match s {
                        Scalar::Null => Err(MinarrowError::TypeError {
                            from: "Value::Scalar(Null)",
                            to: stringify!($t),
                            message: Some("Cannot convert Null to numeric type".to_owned()),
                        }),
                        _ => Ok(s.$method()),
                    },
                    _ => Err(MinarrowError::TypeError {
                        from: "Value",
                        to: stringify!($t),
                        message: Some("Expected Value::Scalar variant".to_owned()),
                    }),
                }
            }
        }
    };
}

#[cfg(feature = "scalar_type")]
impl_tryfrom_value_numeric!(f64, f64);
#[cfg(feature = "scalar_type")]
impl_tryfrom_value_numeric!(f32, f32);
#[cfg(feature = "scalar_type")]
impl_tryfrom_value_numeric!(i64, i64);
#[cfg(feature = "scalar_type")]
impl_tryfrom_value_numeric!(i32, i32);
#[cfg(feature = "scalar_type")]
impl_tryfrom_value_numeric!(u64, u64);
#[cfg(feature = "scalar_type")]
impl_tryfrom_value_numeric!(u32, u32);
#[cfg(feature = "scalar_type")]
impl_tryfrom_value_numeric!(bool, bool);

// Option<T> for nullable results - Null becomes None
#[cfg(feature = "scalar_type")]
macro_rules! impl_tryfrom_value_option {
    ($t:ty, $method:ident) => {
        impl TryFrom<Value> for Option<$t> {
            type Error = MinarrowError;
            #[inline]
            fn try_from(v: Value) -> Result<Self, Self::Error> {
                match v {
                    Value::Scalar(s) => match s {
                        Scalar::Null => Ok(None),
                        _ => Ok(Some(s.$method())),
                    },
                    _ => Err(MinarrowError::TypeError {
                        from: "Value",
                        to: concat!("Option<", stringify!($t), ">"),
                        message: Some("Expected Value::Scalar variant".to_owned()),
                    }),
                }
            }
        }
    };
}

#[cfg(feature = "scalar_type")]
impl_tryfrom_value_option!(f64, f64);
#[cfg(feature = "scalar_type")]
impl_tryfrom_value_option!(f32, f32);
#[cfg(feature = "scalar_type")]
impl_tryfrom_value_option!(i64, i64);
#[cfg(feature = "scalar_type")]
impl_tryfrom_value_option!(i32, i32);
#[cfg(feature = "scalar_type")]
impl_tryfrom_value_option!(u64, u64);
#[cfg(feature = "scalar_type")]
impl_tryfrom_value_option!(u32, u32);
#[cfg(feature = "scalar_type")]
impl_tryfrom_value_option!(bool, bool);

// Array-like Conversions

impl From<Array> for Value {
    #[inline]
    fn from(v: Array) -> Self {
        Value::Array(Arc::new(v))
    }
}

#[cfg(feature = "views")]
impl From<ArrayV> for Value {
    #[inline]
    fn from(v: ArrayV) -> Self {
        Value::ArrayView(Arc::new(v))
    }
}

impl From<Table> for Value {
    #[inline]
    fn from(v: Table) -> Self {
        Value::Table(Arc::new(v))
    }
}

#[cfg(feature = "views")]
impl From<TableV> for Value {
    #[inline]
    fn from(v: TableV) -> Self {
        Value::TableView(Arc::new(v))
    }
}

impl From<FieldArray> for Value {
    #[inline]
    fn from(v: FieldArray) -> Self {
        Value::FieldArray(Arc::new(v))
    }
}

#[cfg(feature = "chunked")]
impl From<SuperTable> for Value {
    #[inline]
    fn from(v: SuperTable) -> Self {
        Value::SuperTable(Arc::new(v))
    }
}

#[cfg(feature = "cube")]
impl From<Cube> for Value {
    #[inline]
    fn from(v: Cube) -> Self {
        Value::Cube(Arc::new(v))
    }
}

#[cfg(feature = "chunked")]
impl From<SuperArray> for Value {
    #[inline]
    fn from(v: SuperArray) -> Self {
        Value::SuperArray(Arc::new(v))
    }
}

#[cfg(all(feature = "chunked", feature = "views"))]
impl From<SuperArrayV> for Value {
    #[inline]
    fn from(v: SuperArrayV) -> Self {
        Value::SuperArrayView(Arc::new(v))
    }
}

#[cfg(feature = "matrix")]
impl From<Matrix> for Value {
    #[inline]
    fn from(v: Matrix) -> Self {
        Value::Matrix(Arc::new(v))
    }
}

/// Wrap a `NumericArrayV` as `Value::ArrayView`. The view's offset and length
/// are preserved; the inner `NumericArray` flows through `Array::NumericArray`.
#[cfg(feature = "views")]
impl From<NumericArrayV> for Value {
    #[inline]
    fn from(v: NumericArrayV) -> Self {
        let len = v.len();
        let array = Array::NumericArray(v.array);
        Value::ArrayView(Arc::new(ArrayV::new(array, v.offset, len)))
    }
}

/// Wrap a `TextArrayV` as `Value::ArrayView`. The view's offset and length
/// are preserved; the inner `TextArray` flows through `Array::TextArray`.
#[cfg(feature = "views")]
impl From<TextArrayV> for Value {
    #[inline]
    fn from(v: TextArrayV) -> Self {
        let len = v.len();
        let array = Array::TextArray(v.array);
        Value::ArrayView(Arc::new(ArrayV::new(array, v.offset, len)))
    }
}

/// Wrap a `TemporalArrayV` as `Value::ArrayView`. The view's offset and length
/// are preserved; the inner `TemporalArray` flows through `Array::TemporalArray`.
#[cfg(all(feature = "views", feature = "datetime"))]
impl From<TemporalArrayV> for Value {
    #[inline]
    fn from(v: TemporalArrayV) -> Self {
        let len = v.len();
        let array = Array::TemporalArray(v.array);
        Value::ArrayView(Arc::new(ArrayV::new(array, v.offset, len)))
    }
}

/// Wrap a `BitmaskV` as `Value::ArrayView` of a `BooleanArray`. The bitmask
/// data is taken as the BooleanArray's data; `null_mask` is `None` since
/// `BitmaskV` carries no separate validity.
#[cfg(feature = "views")]
impl From<BitmaskV> for Value {
    #[inline]
    fn from(v: BitmaskV) -> Self {
        let len = v.len();
        let bitmask = Arc::try_unwrap(v.bitmask).unwrap_or_else(|arc| (*arc).clone());
        let bool_arr = crate::BooleanArray::new(bitmask, None);
        let array = Array::BooleanArray(Arc::new(bool_arr));
        Value::ArrayView(Arc::new(ArrayV::new(array, v.offset, len)))
    }
}

/// Wrap a `BooleanArrayV` as `Value::ArrayView`. The view's offset and length
/// are preserved on the inner `ArrayV`. The full null mask carried by the
/// underlying `BooleanArray` is preserved.
#[cfg(feature = "views")]
impl From<BooleanArrayV> for Value {
    #[inline]
    fn from(v: BooleanArrayV) -> Self {
        let len = v.len();
        let bool_arr = Arc::try_unwrap(v.array).unwrap_or_else(|arc| (*arc).clone());
        let array = Array::BooleanArray(Arc::new(bool_arr));
        Value::ArrayView(Arc::new(ArrayV::new(array, v.offset, len)))
    }
}

// TryFrom for Array-like Types

impl TryFrom<Value> for Array {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        let err = || MinarrowError::TypeError {
            from: "Value",
            to: "Array",
            message: Some("Value type mismatch".to_owned()),
        };
        match v {
            Value::Array(inner) => Ok(Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone())),
            #[cfg(feature = "views")]
            Value::ArrayView(inner) => {
                let view = Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone());
                Ok(view.to_array())
            }
            Value::FieldArray(inner) => {
                let fa = Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone());
                Ok(fa.array)
            }
            #[cfg(feature = "scalar_type")]
            Value::Scalar(s) => Ok(Array::from(s)),
            Value::Table(_) => Err(err()),
            #[cfg(feature = "views")]
            Value::TableView(_) => Err(err()),
            #[cfg(feature = "chunked")]
            Value::SuperArray(_) => Err(err()),
            #[cfg(all(feature = "chunked", feature = "views"))]
            Value::SuperArrayView(_) => Err(err()),
            #[cfg(feature = "chunked")]
            Value::SuperTable(_) => Err(err()),
            #[cfg(all(feature = "chunked", feature = "views"))]
            Value::SuperTableView(_) => Err(err()),
            #[cfg(feature = "matrix")]
            Value::Matrix(_) => Err(err()),
            #[cfg(feature = "cube")]
            Value::Cube(_) => Err(err()),
            Value::VecValue(inner) => {
                let values = Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone());
                match values.len() {
                    0 => Ok(Array::default()),
                    1 => Array::try_from(values.into_iter().next().unwrap()),
                    _ => {
                        let mut iter = values.into_iter();
                        let mut acc = Array::try_from(iter.next().unwrap())?;
                        for v in iter {
                            let next = Array::try_from(v)?;
                            acc = acc.concat(next)?;
                        }
                        Ok(acc)
                    }
                }
            }
            Value::BoxValue(_) => Err(err()),
            Value::ArcValue(_) => Err(err()),
            Value::Tuple2(_) => Err(err()),
            Value::Tuple3(_) => Err(err()),
            Value::Tuple4(_) => Err(err()),
            Value::Tuple5(_) => Err(err()),
            Value::Tuple6(_) => Err(err()),
            Value::Custom(_) => Err(err()),
        }
    }
}

#[cfg(feature = "views")]
impl TryFrom<Value> for ArrayV {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::ArrayView(inner) => {
                Ok(Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone()))
            }
            Value::Array(inner) => {
                let array = Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone());
                Ok(ArrayV::from(array))
            }
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "ArrayV",
                message: Some("Value type mismatch".to_owned()),
            }),
        }
    }
}

#[cfg(feature = "views")]
impl TryFrom<Value> for NumericArrayV {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::ArrayView(inner) => {
                let view = Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone());
                let (array, offset, len) = view.as_tuple();
                match array {
                    Array::NumericArray(num_arr) => Ok(NumericArrayV::new(num_arr, offset, len)),
                    _ => Err(MinarrowError::TypeError {
                        from: "Value",
                        to: "NumericArrayV",
                        message: Some("ArrayV is not a NumericArray".to_owned()),
                    }),
                }
            }
            Value::Array(inner) => {
                let arr = Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone());
                match arr {
                    Array::NumericArray(num_arr) => {
                        let len = num_arr.len();
                        Ok(NumericArrayV::new(num_arr, 0, len))
                    }
                    _ => Err(MinarrowError::TypeError {
                        from: "Value",
                        to: "NumericArrayV",
                        message: Some("Array is not a NumericArray".to_owned()),
                    }),
                }
            }
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "NumericArrayV",
                message: Some("Value type mismatch".to_owned()),
            }),
        }
    }
}

#[cfg(feature = "views")]
impl TryFrom<Value> for TextArrayV {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::ArrayView(inner) => {
                let view = Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone());
                let (array, offset, len) = view.as_tuple();
                match array {
                    Array::TextArray(text_arr) => Ok(TextArrayV::new(text_arr, offset, len)),
                    _ => Err(MinarrowError::TypeError {
                        from: "Value",
                        to: "TextArrayV",
                        message: Some("ArrayV is not a TextArray".to_owned()),
                    }),
                }
            }
            Value::Array(inner) => {
                let array = Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone());
                match array {
                    Array::TextArray(text_arr) => {
                        let len = text_arr.len();
                        Ok(TextArrayV::new(text_arr, 0, len))
                    }
                    _ => Err(MinarrowError::TypeError {
                        from: "Value",
                        to: "TextArrayV",
                        message: Some("Array is not a TextArray".to_owned()),
                    }),
                }
            }
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "TextArrayV",
                message: Some("Value type mismatch".to_owned()),
            }),
        }
    }
}

#[cfg(all(feature = "views", feature = "datetime"))]
impl TryFrom<Value> for TemporalArrayV {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::ArrayView(inner) => {
                let view = Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone());
                let (array, offset, len) = view.as_tuple();
                match array {
                    Array::TemporalArray(temp_arr) => {
                        Ok(TemporalArrayV::new(temp_arr, offset, len))
                    }
                    _ => Err(MinarrowError::TypeError {
                        from: "Value",
                        to: "TemporalArrayV",
                        message: Some("ArrayV is not a TemporalArray".to_owned()),
                    }),
                }
            }
            Value::Array(inner) => {
                let array = Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone());
                match array {
                    Array::TemporalArray(temp_arr) => {
                        let len = temp_arr.len();
                        Ok(TemporalArrayV::new(temp_arr, 0, len))
                    }
                    _ => Err(MinarrowError::TypeError {
                        from: "Value",
                        to: "TemporalArrayV",
                        message: Some("Array is not a TemporalArray".to_owned()),
                    }),
                }
            }
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "TemporalArrayV",
                message: Some("Value type mismatch".to_owned()),
            }),
        }
    }
}

impl TryFrom<Value> for BitmaskV {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::ArrayView(inner) => {
                let view = Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone());
                let (array, offset, len) = view.as_tuple();
                match array {
                    Array::BooleanArray(bool_arr) => {
                        Ok(BitmaskV::new(bool_arr.data.clone(), offset, len))
                    }
                    _ => Err(MinarrowError::TypeError {
                        from: "Value",
                        to: "BitmaskV",
                        message: Some("ArrayV is not a BooleanArray".to_owned()),
                    }),
                }
            }
            Value::Array(inner) => {
                let array = Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone());
                match array {
                    Array::BooleanArray(bool_arr) => {
                        let len = bool_arr.data.len();
                        Ok(BitmaskV::new(bool_arr.data.clone(), 0, len))
                    }
                    _ => Err(MinarrowError::TypeError {
                        from: "Value",
                        to: "BitmaskV",
                        message: Some("Array is not a BooleanArray".to_owned()),
                    }),
                }
            }
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "BitmaskV",
                message: Some("Value type mismatch".to_owned()),
            }),
        }
    }
}

#[cfg(feature = "views")]
impl TryFrom<Value> for BooleanArrayV {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::ArrayView(inner) => {
                let view = Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone());
                let (array, offset, len) = view.as_tuple();
                match array {
                    Array::BooleanArray(bool_arr) => {
                        Ok(BooleanArrayV::new(bool_arr, offset, len))
                    }
                    _ => Err(MinarrowError::TypeError {
                        from: "Value",
                        to: "BooleanArrayV",
                        message: Some("ArrayV is not a BooleanArray".to_owned()),
                    }),
                }
            }
            Value::Array(inner) => {
                let array = Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone());
                match array {
                    Array::BooleanArray(bool_arr) => {
                        let len = bool_arr.len();
                        Ok(BooleanArrayV::new(bool_arr, 0, len))
                    }
                    _ => Err(MinarrowError::TypeError {
                        from: "Value",
                        to: "BooleanArrayV",
                        message: Some("Array is not a BooleanArray".to_owned()),
                    }),
                }
            }
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "BooleanArrayV",
                message: Some("Value type mismatch".to_owned()),
            }),
        }
    }
}

impl TryFrom<Value> for Table {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        let err = || MinarrowError::TypeError {
            from: "Value",
            to: "Table",
            message: Some("Value type mismatch".to_owned()),
        };
        match v {
            Value::Table(inner) => Ok(Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone())),
            #[cfg(feature = "views")]
            Value::TableView(inner) => {
                let view = Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone());
                Ok(view.to_table())
            }
            Value::Array(inner) => {
                let array = Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone());
                Ok(Table::new(
                    "array".to_string(),
                    Some(vec![FieldArray::from_arr("column_0", array)]),
                ))
            }
            #[cfg(feature = "views")]
            Value::ArrayView(inner) => {
                let view = Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone());
                Ok(Table::new(
                    "array".to_string(),
                    Some(vec![FieldArray::from_arr("column_0", view.to_array())]),
                ))
            }
            Value::FieldArray(inner) => {
                let fa = Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone());
                Ok(Table::new("array".to_string(), Some(vec![fa])))
            }
            #[cfg(feature = "scalar_type")]
            Value::Scalar(s) => {
                let array = Array::from(s);
                Ok(Table::new(
                    "scalar".to_string(),
                    Some(vec![FieldArray::from_arr("column_0", array)]),
                ))
            }
            #[cfg(feature = "chunked")]
            Value::SuperArray(_) => Err(err()),
            #[cfg(all(feature = "chunked", feature = "views"))]
            Value::SuperArrayView(_) => Err(err()),
            #[cfg(feature = "chunked")]
            Value::SuperTable(_) => Err(err()),
            #[cfg(all(feature = "chunked", feature = "views"))]
            Value::SuperTableView(_) => Err(err()),
            #[cfg(feature = "matrix")]
            Value::Matrix(_) => Err(err()),
            #[cfg(feature = "cube")]
            Value::Cube(_) => Err(err()),
            // VecValue terminal coercion: engine wire produced by `Long` fan-out gather.
            // Empty -> Table::new_empty (typed-empty); single -> recurse on inner;
            // many -> concat-fold across inner Tables.
            Value::VecValue(inner) => {
                let values = Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone());
                match values.len() {
                    0 => Ok(Table::new_empty()),
                    1 => Table::try_from(values.into_iter().next().unwrap()),
                    _ => {
                        let mut iter = values.into_iter();
                        let mut acc = Table::try_from(iter.next().unwrap())?;
                        for v in iter {
                            let next = Table::try_from(v)?;
                            acc = acc.concat(next)?;
                        }
                        Ok(acc)
                    }
                }
            }
            Value::BoxValue(_) => Err(err()),
            Value::ArcValue(_) => Err(err()),
            Value::Tuple2(_) => Err(err()),
            Value::Tuple3(_) => Err(err()),
            Value::Tuple4(_) => Err(err()),
            Value::Tuple5(_) => Err(err()),
            Value::Tuple6(_) => Err(err()),
            Value::Custom(_) => Err(err()),
        }
    }
}

#[cfg(feature = "views")]
impl TryFrom<Value> for TableV {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::TableView(inner) => {
                Ok(Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone()))
            }
            Value::Table(inner) => {
                let table = Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone());
                Ok(TableV::from(table))
            }
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "TableV",
                message: Some("Value type mismatch".to_owned()),
            }),
        }
    }
}

impl TryFrom<Value> for FieldArray {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::FieldArray(inner) => {
                Ok(Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone()))
            }
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "FieldArray",
                message: Some("Value type mismatch".to_owned()),
            }),
        }
    }
}

#[cfg(feature = "chunked")]
impl TryFrom<Value> for SuperTable {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::SuperTable(inner) => {
                Ok(Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone()))
            }
            // VecValue terminal coercion: wrap inner Tables as chunks. Mirrors the
            // `SuperArray::try_from(VecValue)` precedent. No row-data reallocation;
            // each chunk holds an Arc<Table>. `None` lets SuperTable inherit its
            // name from the first batch rather than imposing a fresh one.
            //
            // `SuperTable::from_batches` panics on schema mismatch between
            // chunks; validate up-front and surface mismatches as a typed
            // MinarrowError so this conversion never aborts the caller.
            Value::VecValue(inner) => {
                let values = Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone());
                let chunks: Result<Vec<Arc<Table>>, _> = values
                    .into_iter()
                    .map(|v| Table::try_from(v).map(Arc::new))
                    .collect();
                let chunks = chunks?;
                if let Some(first) = chunks.first() {
                    let n_cols = first.n_cols();
                    for (idx, chunk) in chunks.iter().enumerate().skip(1) {
                        if chunk.n_cols() != n_cols {
                            return Err(MinarrowError::TypeError {
                                from: "Value::VecValue",
                                to: "SuperTable",
                                message: Some(format!(
                                    "batch {idx} column-count mismatch: {} vs {}",
                                    chunk.n_cols(),
                                    n_cols
                                )),
                            });
                        }
                        for (col_idx, fa) in chunk.cols.iter().enumerate() {
                            if fa.field != first.cols[col_idx].field {
                                return Err(MinarrowError::TypeError {
                                    from: "Value::VecValue",
                                    to: "SuperTable",
                                    message: Some(format!(
                                        "batch {idx} col {col_idx} schema mismatch"
                                    )),
                                });
                            }
                        }
                    }
                }
                Ok(SuperTable::from_batches(chunks, None))
            }
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "SuperTable",
                message: Some("Value type mismatch".to_owned()),
            }),
        }
    }
}

#[cfg(feature = "cube")]
impl TryFrom<Value> for Cube {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::Cube(inner) => Ok(Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone())),
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "Cube",
                message: Some("Value type mismatch".to_owned()),
            }),
        }
    }
}

#[cfg(feature = "chunked")]
impl TryFrom<Value> for SuperArray {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::SuperArray(inner) => {
                Ok(Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone()))
            }
            Value::VecValue(inner) => {
                let values = Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone());
                let chunks: Result<Vec<Array>, _> =
                    values.into_iter().map(Array::try_from).collect();
                Ok(SuperArray::from_arrays(chunks?))
            }
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "SuperArray",
                message: Some("Value type mismatch".to_owned()),
            }),
        }
    }
}

#[cfg(all(feature = "chunked", feature = "views"))]
impl TryFrom<Value> for SuperArrayV {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::SuperArrayView(inner) => {
                Ok(Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone()))
            }
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "SuperArrayV",
                message: Some("Value type mismatch".to_owned()),
            }),
        }
    }
}

#[cfg(feature = "matrix")]
impl TryFrom<Value> for Matrix {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::Matrix(inner) => Ok(Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone())),
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "Matrix",
                message: Some("Value type mismatch".to_owned()),
            }),
        }
    }
}

#[cfg(all(feature = "chunked", feature = "views"))]
impl TryFrom<Value> for SuperTableV {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::SuperTableView(inner) => {
                Ok(Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone()))
            }
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "SuperTableV",
                message: Some("Value type mismatch".to_owned()),
            }),
        }
    }
}

// Recursive Container Conversions

impl From<Vec<Value>> for Value {
    fn from(v: Vec<Value>) -> Self {
        Value::VecValue(Arc::new(v))
    }
}

impl From<(Value, Value)> for Value {
    fn from(v: (Value, Value)) -> Self {
        Value::Tuple2(Arc::new(v))
    }
}

impl From<(Value, Value, Value)> for Value {
    fn from(v: (Value, Value, Value)) -> Self {
        Value::Tuple3(Arc::new(v))
    }
}

impl From<(Value, Value, Value, Value)> for Value {
    fn from(v: (Value, Value, Value, Value)) -> Self {
        Value::Tuple4(Arc::new(v))
    }
}

impl From<(Value, Value, Value, Value, Value)> for Value {
    fn from(v: (Value, Value, Value, Value, Value)) -> Self {
        Value::Tuple5(Arc::new(v))
    }
}

impl From<(Value, Value, Value, Value, Value, Value)> for Value {
    fn from(v: (Value, Value, Value, Value, Value, Value)) -> Self {
        Value::Tuple6(Arc::new(v))
    }
}

// TryFrom for recursive containers
impl TryFrom<Value> for Vec<Value> {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::VecValue(inner) => {
                Ok(Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone()))
            }
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "Vec<Value>",
                message: Some("Expected VecValue variant".to_owned()),
            }),
        }
    }
}

impl TryFrom<Value> for (Value, Value) {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::Tuple2(tuple) => Ok(Arc::try_unwrap(tuple).unwrap_or_else(|arc| (*arc).clone())),
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "(Value, Value)",
                message: Some("Expected Tuple2 variant".to_owned()),
            }),
        }
    }
}

impl TryFrom<Value> for (Value, Value, Value) {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::Tuple3(tuple) => Ok(Arc::try_unwrap(tuple).unwrap_or_else(|arc| (*arc).clone())),
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "(Value, Value, Value)",
                message: Some("Expected Tuple3 variant".to_owned()),
            }),
        }
    }
}

impl TryFrom<Value> for (Value, Value, Value, Value) {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::Tuple4(tuple) => Ok(Arc::try_unwrap(tuple).unwrap_or_else(|arc| (*arc).clone())),
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "(Value, Value, Value, Value)",
                message: Some("Expected Tuple4 variant".to_owned()),
            }),
        }
    }
}

impl TryFrom<Value> for (Value, Value, Value, Value, Value) {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::Tuple5(tuple) => Ok(Arc::try_unwrap(tuple).unwrap_or_else(|arc| (*arc).clone())),
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "(Value, Value, Value, Value, Value)",
                message: Some("Expected Tuple5 variant".to_owned()),
            }),
        }
    }
}

impl TryFrom<Value> for (Value, Value, Value, Value, Value, Value) {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::Tuple6(tuple) => Ok(Arc::try_unwrap(tuple).unwrap_or_else(|arc| (*arc).clone())),
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "(Value, Value, Value, Value, Value, Value)",
                message: Some("Expected Tuple6 variant".to_owned()),
            }),
        }
    }
}

// Value Creation Macros

// Macros for creating `Value` instances from arrays and scalars.
// These wrap the existing `arr_*` macros and `Scalar` constructors.

// Signed Integer Array Values

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! val_i8 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_i8![$($x)*])
    };
}

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! val_i16 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_i16![$($x)*])
    };
}

#[macro_export]
macro_rules! val_i32 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_i32![$($x)*])
    };
}

#[macro_export]
macro_rules! val_i64 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_i64![$($x)*])
    };
}

// Unsigned Integer Array Values

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! val_u8 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_u8![$($x)*])
    };
}

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! val_u16 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_u16![$($x)*])
    };
}

#[macro_export]
macro_rules! val_u32 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_u32![$($x)*])
    };
}

#[macro_export]
macro_rules! val_u64 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_u64![$($x)*])
    };
}

// Floating Point Array Values

#[macro_export]
macro_rules! val_f32 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_f32![$($x)*])
    };
}

#[macro_export]
macro_rules! val_f64 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_f64![$($x)*])
    };
}

// Boolean Array Values

#[macro_export]
macro_rules! val_bool {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_bool![$($x)*])
    };
}

// String Array Values

#[macro_export]
macro_rules! val_str32 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_str32![$($x)*])
    };
}

#[cfg(feature = "large_string")]
#[macro_export]
macro_rules! val_str64 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_str64![$($x)*])
    };
}

// Categorical Array Values

#[cfg(feature = "default_categorical_8")]
#[macro_export]
macro_rules! val_cat8 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_cat8![$($x)*])
    };
}

#[cfg(feature = "extended_categorical")]
#[macro_export]
macro_rules! val_cat16 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_cat16![$($x)*])
    };
}

#[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
#[macro_export]
macro_rules! val_cat32 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_cat32![$($x)*])
    };
}

#[cfg(feature = "extended_categorical")]
#[macro_export]
macro_rules! val_cat64 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_cat64![$($x)*])
    };
}

// Scalar Values

#[cfg(feature = "scalar_type")]
#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! val_scalar_i8 {
    ($v:expr) => {
        $crate::Value::from($crate::Scalar::Int8($v))
    };
}

#[cfg(feature = "scalar_type")]
#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! val_scalar_i16 {
    ($v:expr) => {
        $crate::Value::from($crate::Scalar::Int16($v))
    };
}

#[cfg(feature = "scalar_type")]
#[macro_export]
macro_rules! val_scalar_i32 {
    ($v:expr) => {
        $crate::Value::from($crate::Scalar::Int32($v))
    };
}

#[cfg(feature = "scalar_type")]
#[macro_export]
macro_rules! val_scalar_i64 {
    ($v:expr) => {
        $crate::Value::from($crate::Scalar::Int64($v))
    };
}

#[cfg(feature = "scalar_type")]
#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! val_scalar_u8 {
    ($v:expr) => {
        $crate::Value::from($crate::Scalar::UInt8($v))
    };
}

#[cfg(feature = "scalar_type")]
#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! val_scalar_u16 {
    ($v:expr) => {
        $crate::Value::from($crate::Scalar::UInt16($v))
    };
}

#[cfg(feature = "scalar_type")]
#[macro_export]
macro_rules! val_scalar_u32 {
    ($v:expr) => {
        $crate::Value::from($crate::Scalar::UInt32($v))
    };
}

#[cfg(feature = "scalar_type")]
#[macro_export]
macro_rules! val_scalar_u64 {
    ($v:expr) => {
        $crate::Value::from($crate::Scalar::UInt64($v))
    };
}

#[cfg(feature = "scalar_type")]
#[macro_export]
macro_rules! val_scalar_f32 {
    ($v:expr) => {
        $crate::Value::from($crate::Scalar::Float32($v))
    };
}

#[cfg(feature = "scalar_type")]
#[macro_export]
macro_rules! val_scalar_f64 {
    ($v:expr) => {
        $crate::Value::from($crate::Scalar::Float64($v))
    };
}

#[cfg(feature = "scalar_type")]
#[macro_export]
macro_rules! val_scalar_bool {
    ($v:expr) => {
        $crate::Value::from($crate::Scalar::Boolean($v))
    };
}

#[cfg(feature = "scalar_type")]
#[macro_export]
macro_rules! val_scalar_str32 {
    ($v:expr) => {
        $crate::Value::from($crate::Scalar::String32($v.to_string()))
    };
}

#[cfg(feature = "scalar_type")]
#[cfg(feature = "large_string")]
#[macro_export]
macro_rules! val_scalar_str64 {
    ($v:expr) => {
        $crate::Value::from($crate::Scalar::String64($v.to_string()))
    };
}

#[cfg(feature = "scalar_type")]
#[macro_export]
macro_rules! val_scalar_null {
    () => {
        $crate::Value::from($crate::Scalar::Null)
    };
}

#[cfg(test)]
mod accessor_tests {
    use super::*;
    use crate::structs::field_array::field_array;
    use crate::structs::variants::integer::IntegerArray;

    #[test]
    fn test_arr_accessor() {
        let arr = IntegerArray::<i32>::from_slice(&[1, 2, 3]);
        let val = Value::Array(Arc::new(Array::from_int32(arr)));

        // Test panicking accessor
        let array_ref = val.arr();
        assert_eq!(array_ref.len(), 3);

        // Test try accessor
        let result = val.try_arr();
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 3);
    }

    #[test]
    #[should_panic(expected = "Expected Value::Array")]
    fn test_arr_accessor_wrong_type_panics() {
        let val = Value::Table(Arc::new(Table::new_empty()));
        let _ = val.arr();
    }

    #[test]
    fn test_try_arr_accessor_wrong_type_returns_error() {
        let val = Value::Table(Arc::new(Table::new_empty()));
        let result = val.try_arr();
        assert!(result.is_err());
    }

    #[test]
    fn test_table_accessor() {
        let table = Table::new_empty();
        let val = Value::Table(Arc::new(table));

        let table_ref = val.table();
        assert_eq!(table_ref.n_rows, 0);

        let result = val.try_table();
        assert!(result.is_ok());
    }

    #[test]
    fn test_fa_accessor() {
        let arr = IntegerArray::<i32>::from_slice(&[1, 2, 3]);
        let fa = field_array("test", Array::from_int32(arr));
        let val = Value::FieldArray(Arc::new(fa));

        let fa_ref = val.fa();
        assert_eq!(fa_ref.field.name, "test");
        assert_eq!(fa_ref.len(), 3);

        let result = val.try_fa();
        assert!(result.is_ok());
    }

    #[test]
    fn test_vec_val_accessor() {
        let arr = IntegerArray::<i32>::from_slice(&[1, 2]);
        let inner = Value::Array(Arc::new(Array::from_int32(arr)));
        let val = Value::VecValue(Arc::new(vec![inner.clone(), inner]));

        let vec_ref = val.vec_val();
        assert_eq!(vec_ref.len(), 2);

        let result = val.try_vec_val();
        assert!(result.is_ok());
    }

    #[test]
    fn test_box_val_accessor() {
        let arr = IntegerArray::<i32>::from_slice(&[1, 2, 3]);
        let inner = Value::Array(Arc::new(Array::from_int32(arr)));
        let val = Value::BoxValue(Box::new(inner));

        let inner_ref = val.box_val();
        assert_eq!(inner_ref.len(), 3);

        let result = val.try_box_val();
        assert!(result.is_ok());
    }

    #[test]
    fn test_arc_val_accessor() {
        let arr = IntegerArray::<i32>::from_slice(&[1, 2, 3]);
        let inner = Value::Array(Arc::new(Array::from_int32(arr)));
        let val = Value::ArcValue(Arc::new(inner));

        let inner_ref = val.arc_val();
        assert_eq!(inner_ref.len(), 3);

        let result = val.try_arc_val();
        assert!(result.is_ok());
    }

    #[test]
    fn test_tuple_accessors() {
        let arr1 = Value::Array(Arc::new(Array::from_int32(IntegerArray::from_slice(&[1]))));
        let arr2 = Value::Array(Arc::new(Array::from_int32(IntegerArray::from_slice(&[2]))));
        let arr3 = Value::Array(Arc::new(Array::from_int32(IntegerArray::from_slice(&[3]))));

        // Tuple2
        let t2 = Value::Tuple2(Arc::new((arr1.clone(), arr2.clone())));
        let (a, b) = t2.t2();
        assert_eq!(a.len(), 1);
        assert_eq!(b.len(), 1);
        assert!(t2.try_t2().is_ok());

        // Tuple3
        let t3 = Value::Tuple3(Arc::new((arr1.clone(), arr2.clone(), arr3.clone())));
        let (a, b, c) = t3.t3();
        assert_eq!(a.len(), 1);
        assert_eq!(b.len(), 1);
        assert_eq!(c.len(), 1);
        assert!(t3.try_t3().is_ok());
    }

    #[cfg(feature = "scalar_type")]
    #[test]
    fn test_scalar_accessor() {
        let val = Value::Scalar(Scalar::Int32(42));

        let scalar_ref = val.scalar();
        assert_eq!(scalar_ref.i32(), 42);

        let result = val.try_scalar();
        assert!(result.is_ok());
    }

    #[cfg(feature = "views")]
    #[test]
    fn test_av_accessor() {
        let arr = IntegerArray::<i32>::from_slice(&[1, 2, 3]);
        let array = Array::from_int32(arr);
        let view = ArrayV::new(array, 0, 3);
        let val = Value::ArrayView(Arc::new(view));

        let av_ref = val.av();
        assert_eq!(av_ref.len(), 3);

        let result = val.try_av();
        assert!(result.is_ok());
    }

    #[cfg(feature = "views")]
    #[test]
    fn test_tv_accessor() {
        let table = Table::new_empty();
        let view = TableV::from_table(table, 0, 0);
        let val = Value::TableView(Arc::new(view));

        let tv_ref = val.tv();
        assert_eq!(tv_ref.len(), 0);

        let result = val.try_tv();
        assert!(result.is_ok());
    }

    #[cfg(feature = "chunked")]
    #[test]
    fn test_sa_accessor() {
        let sa = SuperArray::new();
        let val = Value::SuperArray(Arc::new(sa));

        let sa_ref = val.sa();
        assert_eq!(sa_ref.len(), 0);

        let result = val.try_sa();
        assert!(result.is_ok());
    }

    #[cfg(feature = "chunked")]
    #[test]
    fn test_st_accessor() {
        let st = SuperTable::new("test".to_string());
        let val = Value::SuperTable(Arc::new(st));

        let st_ref = val.st();
        assert_eq!(st_ref.len(), 0);

        let result = val.try_st();
        assert!(result.is_ok());
    }
}

// VecValue -> typed-T terminal coercion tests. Exercises the engine wire format
// produced by `Long` fan-out gather (`Value::VecValue<Vec<Value>>`) flowing into
// `BlockStack<T>::collect()`'s `T::try_from(value)` step.
#[cfg(test)]
mod vecvalue_terminal_coercion_tests {
    use super::*;
    use crate::structs::field_array::field_array;
    use crate::structs::variants::integer::IntegerArray;

    fn arr_i32(values: &[i32]) -> Array {
        Array::from_int32(IntegerArray::<i32>::from_slice(values))
    }

    fn val_arr_i32(values: &[i32]) -> Value {
        Value::Array(Arc::new(arr_i32(values)))
    }

    fn val_table_one_col(name: &str, values: &[i32]) -> Value {
        let t = Table::new(
            name.to_string(),
            Some(vec![field_array("c", arr_i32(values))]),
        );
        Value::Table(Arc::new(t))
    }

    // ----- Array::try_from -----

    #[test]
    fn try_from_vecvalue_empty_returns_typed_empty_array() {
        let v = Value::VecValue(Arc::new(vec![]));
        let arr = Array::try_from(v).expect("empty VecValue should coerce to Array::Null");
        assert!(matches!(arr, Array::Null));
    }

    #[test]
    fn try_from_vecvalue_single_unwraps_to_array() {
        let inner = val_arr_i32(&[1, 2, 3]);
        let v = Value::VecValue(Arc::new(vec![inner]));
        let arr = Array::try_from(v).expect("single-element VecValue should unwrap");
        assert_eq!(arr.len(), 3);
    }

    #[test]
    fn try_from_vecvalue_multi_consolidates_array_rows() {
        let parts = vec![val_arr_i32(&[1, 2]), val_arr_i32(&[3, 4, 5]), val_arr_i32(&[6])];
        let v = Value::VecValue(Arc::new(parts));
        let arr = Array::try_from(v).expect("multi VecValue<Array> should concat-fold");
        assert_eq!(arr.len(), 6);
        // Row order must be preserved across fan-out gather (invariant I5).
        let na = arr.num().i32().unwrap();
        assert_eq!(&na.data[..], &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn try_from_vecvalue_mismatched_concat_errors() {
        // VecValue<[Array<i32>, Array<i64>]> — both convert to Array, but concat
        // rejects the dtype mismatch.
        let a = val_arr_i32(&[1]);
        let b = Value::Array(Arc::new(Array::from_int64(
            IntegerArray::<i64>::from_slice(&[2]),
        )));
        let v = Value::VecValue(Arc::new(vec![a, b]));
        let res = Array::try_from(v);
        assert!(res.is_err(), "incompatible inner dtypes must error, not silently coerce");
    }

    // ----- Table::try_from -----

    #[test]
    fn try_from_vecvalue_empty_returns_typed_empty_table() {
        let v = Value::VecValue(Arc::new(vec![]));
        let t = Table::try_from(v).expect("empty VecValue should coerce to empty Table");
        assert_eq!(t.n_rows, 0);
    }

    #[test]
    fn try_from_vecvalue_single_unwraps_to_table() {
        let inner = val_table_one_col("t", &[10, 20]);
        let v = Value::VecValue(Arc::new(vec![inner]));
        let t = Table::try_from(v).expect("single-element VecValue should unwrap");
        assert_eq!(t.n_rows, 2);
    }

    #[test]
    fn try_from_vecvalue_multi_consolidates_table_rows() {
        let parts = vec![
            val_table_one_col("t", &[1, 2]),
            val_table_one_col("t", &[3, 4]),
        ];
        let v = Value::VecValue(Arc::new(parts));
        let t = Table::try_from(v).expect("multi VecValue<Table> should concat-fold");
        assert_eq!(t.n_rows, 4);
    }

    // ----- SuperTable::try_from -----

    #[cfg(feature = "chunked")]
    #[test]
    fn try_from_vecvalue_empty_returns_empty_supertable() {
        let v = Value::VecValue(Arc::new(vec![]));
        let st = SuperTable::try_from(v).expect("empty VecValue should coerce to empty SuperTable");
        assert_eq!(st.len(), 0);
    }

    #[cfg(feature = "chunked")]
    #[test]
    fn try_from_vecvalue_supertable_wraps_without_realloc() {
        let parts = vec![
            val_table_one_col("t", &[1, 2]),
            val_table_one_col("t", &[3]),
            val_table_one_col("t", &[4, 5, 6]),
        ];
        let v = Value::VecValue(Arc::new(parts));
        let st = SuperTable::try_from(v).expect("VecValue<Table> should wrap as SuperTable");
        assert_eq!(st.n_batches(), 3, "chunk count preserved");
        assert_eq!(st.n_rows, 6, "row total across chunks");
    }

    // ----- Wrong-typed VecValue paths -----

    #[test]
    fn try_from_vecvalue_array_rejects_unconvertible_inner() {
        // SuperArray inside VecValue cannot become an Array; first conversion errors.
        #[cfg(feature = "chunked")]
        {
            let inner = Value::SuperArray(Arc::new(SuperArray::new()));
            let v = Value::VecValue(Arc::new(vec![inner]));
            let res = Array::try_from(v);
            assert!(res.is_err());
        }
    }
}
