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

use super::Value;
use crate::enums::error::MinarrowError;
use crate::enums::shape_dim::ShapeDim;
use crate::traits::concatenate::Concatenate;
use crate::traits::shape::Shape;
use crate::{BooleanArray, FloatArray, IntegerArray, StringArray};
use std::sync::Arc;

#[cfg(feature = "datetime")]
use crate::DatetimeArray;

/// Implements `PartialEq` for `Value`
///
/// This includes special handling for the `Custom` type.
impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        use super::Value::*;
        match (self, other) {
            #[cfg(feature = "scalar_type")]
            (Scalar(a), Scalar(b)) => a == b,
            (Array(a), Array(b)) => **a == **b,
            #[cfg(feature = "views")]
            (ArrayView(a), ArrayView(b)) => **a == **b,
            (Table(a), Table(b)) => **a == **b,
            #[cfg(feature = "views")]
            (TableView(a), TableView(b)) => **a == **b,
            #[cfg(feature = "chunked")]
            (SuperArray(a), SuperArray(b)) => a == b,
            #[cfg(all(feature = "chunked", feature = "views"))]
            (SuperArrayView(a), SuperArrayView(b)) => a == b,
            #[cfg(feature = "chunked")]
            (SuperTable(a), SuperTable(b)) => **a == **b,
            (FieldArray(a), FieldArray(b)) => **a == **b,
            #[cfg(feature = "matrix")]
            (Matrix(a), Matrix(b)) => a == b,
            #[cfg(feature = "ndarray")]
            (NdArray(a), NdArray(b)) => **a == **b,
            #[cfg(all(feature = "ndarray", feature = "views"))]
            (NdArrayView(a), NdArrayView(b)) => **a == **b,
            #[cfg(all(feature = "ndarray", feature = "chunked"))]
            (SuperNdArray(a), SuperNdArray(b)) => **a == **b,
            #[cfg(all(feature = "ndarray", feature = "chunked", feature = "views"))]
            (SuperNdArrayView(a), SuperNdArrayView(b)) => **a == **b,
            #[cfg(feature = "xarray")]
            (XArray(a), XArray(b)) => **a == **b,
            #[cfg(feature = "cube")]
            (Cube(a), Cube(b)) => **a == **b,
            (Custom(a), Custom(b)) => a.eq_box(&**b),
            (VecValue(a), VecValue(b)) => **a == **b,
            (BoxValue(a), BoxValue(b)) => a == b,
            (ArcValue(a), ArcValue(b)) => a == b,
            (Tuple2(a), Tuple2(b)) => **a == **b,
            (Tuple3(a), Tuple3(b)) => **a == **b,
            (Tuple4(a), Tuple4(b)) => **a == **b,
            (Tuple5(a), Tuple5(b)) => **a == **b,
            (Tuple6(a), Tuple6(b)) => **a == **b,
            _ => false,
        }
    }
}

/// Implements `Eq` for `Value`
///
/// Since PartialEq is reflexive, symmetric, and transitive for Value,
/// we can safely implement Eq.
impl Eq for Value {}

// Shape Implementation

impl Shape for Value {
    fn shape(&self) -> ShapeDim {
        match self {
            #[cfg(feature = "scalar_type")]
            Value::Scalar(_) => ShapeDim::Rank0(1),
            Value::Array(array) => array.shape(),
            #[cfg(feature = "views")]
            Value::ArrayView(array_view) => array_view.shape(),
            Value::Table(table) => table.shape(),
            #[cfg(feature = "views")]
            Value::TableView(table_view) => table_view.shape(),
            #[cfg(feature = "chunked")]
            Value::SuperArray(chunked_array) => ShapeDim::Rank1(chunked_array.len()),
            #[cfg(all(feature = "chunked", feature = "views"))]
            Value::SuperArrayView(chunked_view) => ShapeDim::Rank1(chunked_view.len()),
            #[cfg(feature = "chunked")]
            Value::SuperTable(chunked_table) => ShapeDim::Rank2 {
                rows: chunked_table.n_rows(),
                cols: chunked_table.n_cols(),
            },
            #[cfg(all(feature = "chunked", feature = "views"))]
            Value::SuperTableView(chunked_view) => ShapeDim::Rank2 {
                rows: chunked_view.n_rows(),
                cols: chunked_view.n_cols(),
            },
            Value::FieldArray(field_array) => field_array.shape(),
            #[cfg(feature = "matrix")]
            Value::Matrix(matrix) => matrix.shape(),
            #[cfg(feature = "ndarray")]
            Value::NdArray(nd) => Shape::shape(nd.as_ref()),
            #[cfg(all(feature = "ndarray", feature = "views"))]
            Value::NdArrayView(v) => Shape::shape(v.as_ref()),
            #[cfg(all(feature = "ndarray", feature = "chunked"))]
            Value::SuperNdArray(snd) => Shape::shape(snd.as_ref()),
            #[cfg(all(feature = "ndarray", feature = "chunked", feature = "views"))]
            Value::SuperNdArrayView(v) => Shape::shape(v.as_ref()),
            #[cfg(feature = "xarray")]
            Value::XArray(xa) => Shape::shape(xa.as_ref()),
            #[cfg(feature = "cube")]
            Value::Cube(cube) => cube.shape(),
            Value::VecValue(vec_value) => {
                let shapes: Vec<ShapeDim> = vec_value.iter().map(|v| v.shape()).collect();
                ShapeDim::Collection(shapes)
            }
            Value::BoxValue(boxed_value) => boxed_value.shape(),
            Value::ArcValue(arc_value) => arc_value.shape(),
            Value::Tuple2(tuple) => ShapeDim::Collection(vec![tuple.0.shape(), tuple.1.shape()]),
            Value::Tuple3(tuple) => {
                ShapeDim::Collection(vec![tuple.0.shape(), tuple.1.shape(), tuple.2.shape()])
            }
            Value::Tuple4(tuple) => ShapeDim::Collection(vec![
                tuple.0.shape(),
                tuple.1.shape(),
                tuple.2.shape(),
                tuple.3.shape(),
            ]),
            Value::Tuple5(tuple) => ShapeDim::Collection(vec![
                tuple.0.shape(),
                tuple.1.shape(),
                tuple.2.shape(),
                tuple.3.shape(),
                tuple.4.shape(),
            ]),
            Value::Tuple6(tuple) => ShapeDim::Collection(vec![
                tuple.0.shape(),
                tuple.1.shape(),
                tuple.2.shape(),
                tuple.3.shape(),
                tuple.4.shape(),
                tuple.5.shape(),
            ]),
            Value::Custom(_) => ShapeDim::Unknown,
        }
    }
}

// Concatenate Implementation

impl Concatenate for Value {
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        use super::Value::*;
        match (self, other) {
            // Scalar + Scalar -> Array (length 2)
            #[cfg(feature = "scalar_type")]
            (Scalar(a), Scalar(b)) => {
                use crate::Scalar::*;
                match (a, b) {
                    // Integer types
                    #[cfg(feature = "extended_numeric_types")]
                    (Int8(a), Int8(b)) => {
                        let arr = IntegerArray::from_slice(&[a, b]);
                        Ok(Value::Array(Arc::new(crate::Array::from_int8(arr))))
                    }
                    #[cfg(feature = "extended_numeric_types")]
                    (Int16(a), Int16(b)) => {
                        let arr = IntegerArray::from_slice(&[a, b]);
                        Ok(Value::Array(Arc::new(crate::Array::from_int16(arr))))
                    }
                    (Int32(a), Int32(b)) => {
                        let arr = IntegerArray::from_slice(&[a, b]);
                        Ok(Value::Array(Arc::new(crate::Array::from_int32(arr))))
                    }
                    (Int64(a), Int64(b)) => {
                        let arr = IntegerArray::from_slice(&[a, b]);
                        Ok(Value::Array(Arc::new(crate::Array::from_int64(arr))))
                    }
                    #[cfg(feature = "extended_numeric_types")]
                    (UInt8(a), UInt8(b)) => {
                        let arr = IntegerArray::from_slice(&[a, b]);
                        Ok(Value::Array(Arc::new(crate::Array::from_uint8(arr))))
                    }
                    #[cfg(feature = "extended_numeric_types")]
                    (UInt16(a), UInt16(b)) => {
                        let arr = IntegerArray::from_slice(&[a, b]);
                        Ok(Value::Array(Arc::new(crate::Array::from_uint16(arr))))
                    }
                    (UInt32(a), UInt32(b)) => {
                        let arr = IntegerArray::from_slice(&[a, b]);
                        Ok(Value::Array(Arc::new(crate::Array::from_uint32(arr))))
                    }
                    (UInt64(a), UInt64(b)) => {
                        let arr = IntegerArray::from_slice(&[a, b]);
                        Ok(Value::Array(Arc::new(crate::Array::from_uint64(arr))))
                    }
                    // Float types
                    (Float32(a), Float32(b)) => {
                        let arr = FloatArray::from_slice(&[a, b]);
                        Ok(Value::Array(Arc::new(crate::Array::from_float32(arr))))
                    }
                    (Float64(a), Float64(b)) => {
                        let arr = FloatArray::from_slice(&[a, b]);
                        Ok(Value::Array(Arc::new(crate::Array::from_float64(arr))))
                    }
                    // Boolean
                    (Boolean(a), Boolean(b)) => {
                        let arr = BooleanArray::from_slice(&[a, b]);
                        Ok(Value::Array(Arc::new(crate::Array::from_bool(arr))))
                    }
                    // String types
                    (String32(a), String32(b)) => {
                        let arr = StringArray::from_slice(&[a.as_str(), b.as_str()]);
                        Ok(Value::Array(Arc::new(crate::Array::from_string32(arr))))
                    }
                    #[cfg(feature = "large_string")]
                    (String64(a), String64(b)) => {
                        let arr = StringArray::from_slice(&[a.as_str(), b.as_str()]);
                        Ok(Value::Array(Arc::new(crate::Array::from_string64(arr))))
                    }
                    // Datetime types
                    #[cfg(feature = "datetime")]
                    (Datetime32(a), Datetime32(b)) => {
                        let arr = DatetimeArray::from_slice(&[a, b], None);
                        Ok(Value::Array(Arc::new(crate::Array::from_datetime_i32(arr))))
                    }
                    #[cfg(feature = "datetime")]
                    (Datetime64(a), Datetime64(b)) => {
                        let arr = DatetimeArray::from_slice(&[a, b], None);
                        Ok(Value::Array(Arc::new(crate::Array::from_datetime_i64(arr))))
                    }
                    // Null + Null
                    (Null, Null) => Ok(Value::Array(Arc::new(crate::Array::Null))),
                    // Mismatched scalar types
                    (lhs, rhs) => Err(MinarrowError::IncompatibleTypeError {
                        from: "Scalar",
                        to: "Array",
                        message: Some(format!(
                            "Cannot concatenate mismatched Scalar types: {:?} and {:?}",
                            scalar_variant_name(&lhs),
                            scalar_variant_name(&rhs)
                        )),
                    }),
                }
            }

            // Array + Array -> Array
            (Array(a), Array(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(Value::Array(Arc::new(a.concat(b)?)))
            }

            // Table + Table -> Table
            (Table(a), Table(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(Value::Table(Arc::new(a.concat(b)?)))
            }

            // Matrix + Matrix -> Matrix
            #[cfg(feature = "matrix")]
            (Matrix(a), Matrix(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(Value::Matrix(Arc::new(a.concat(b)?)))
            }

            // Cube + Cube -> Cube
            #[cfg(feature = "cube")]
            (Cube(a), Cube(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(Value::Cube(Arc::new(a.concat(b)?)))
            }

            // NdArray + NdArray -> NdArray
            #[cfg(feature = "ndarray")]
            (NdArray(a), NdArray(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(Value::NdArray(Arc::new(a.concat(b)?)))
            }

            // NdArrayView + NdArrayView -> NdArray, materialising both
            // windows, mirroring the broadcast semantics for view pairs.
            #[cfg(all(feature = "ndarray", feature = "views"))]
            (NdArrayView(a), NdArrayView(b)) => {
                let a = a.to_ndarray();
                let b = b.to_ndarray();
                Ok(Value::NdArray(Arc::new(a.concat(b)?)))
            }

            // SuperNdArray + SuperNdArray -> SuperNdArray
            #[cfg(all(feature = "ndarray", feature = "chunked"))]
            (SuperNdArray(a), SuperNdArray(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(Value::SuperNdArray(Arc::new(a.concat(b)?)))
            }

            // SuperNdArrayView + SuperNdArrayView -> SuperNdArrayView, zero-copy
            #[cfg(all(feature = "ndarray", feature = "chunked", feature = "views"))]
            (SuperNdArrayView(a), SuperNdArrayView(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(Value::SuperNdArrayView(Arc::new(a.concat(b)?)))
            }

            // XArray + XArray -> XArray
            #[cfg(feature = "xarray")]
            (XArray(a), XArray(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(Value::XArray(Arc::new(a.concat(b)?)))
            }

            // Chunked types
            #[cfg(feature = "chunked")]
            (SuperArray(a), SuperArray(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(Value::SuperArray(Arc::new(a.concat(b)?)))
            }

            #[cfg(feature = "chunked")]
            (SuperTable(a), SuperTable(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(Value::SuperTable(Arc::new(a.concat(b)?)))
            }

            // Tuples (element-wise concatenation, recursive)
            (Tuple2(a_arc), Tuple2(b_arc)) => {
                let (a1, a2) = Arc::try_unwrap(a_arc).unwrap_or_else(|arc| (*arc).clone());
                let (b1, b2) = Arc::try_unwrap(b_arc).unwrap_or_else(|arc| (*arc).clone());
                let c1 = a1.concat(b1)?;
                let c2 = a2.concat(b2)?;
                Ok(Value::Tuple2(Arc::new((c1, c2))))
            }

            (Tuple3(a_arc), Tuple3(b_arc)) => {
                let (a1, a2, a3) = Arc::try_unwrap(a_arc).unwrap_or_else(|arc| (*arc).clone());
                let (b1, b2, b3) = Arc::try_unwrap(b_arc).unwrap_or_else(|arc| (*arc).clone());
                let c1 = a1.concat(b1)?;
                let c2 = a2.concat(b2)?;
                let c3 = a3.concat(b3)?;
                Ok(Value::Tuple3(Arc::new((c1, c2, c3))))
            }

            (Tuple4(a_arc), Tuple4(b_arc)) => {
                let (a1, a2, a3, a4) = Arc::try_unwrap(a_arc).unwrap_or_else(|arc| (*arc).clone());
                let (b1, b2, b3, b4) = Arc::try_unwrap(b_arc).unwrap_or_else(|arc| (*arc).clone());
                let c1 = a1.concat(b1)?;
                let c2 = a2.concat(b2)?;
                let c3 = a3.concat(b3)?;
                let c4 = a4.concat(b4)?;
                Ok(Value::Tuple4(Arc::new((c1, c2, c3, c4))))
            }

            (Tuple5(a_arc), Tuple5(b_arc)) => {
                let (a1, a2, a3, a4, a5) =
                    Arc::try_unwrap(a_arc).unwrap_or_else(|arc| (*arc).clone());
                let (b1, b2, b3, b4, b5) =
                    Arc::try_unwrap(b_arc).unwrap_or_else(|arc| (*arc).clone());
                let c1 = a1.concat(b1)?;
                let c2 = a2.concat(b2)?;
                let c3 = a3.concat(b3)?;
                let c4 = a4.concat(b4)?;
                let c5 = a5.concat(b5)?;
                Ok(Value::Tuple5(Arc::new((c1, c2, c3, c4, c5))))
            }

            (Tuple6(a_arc), Tuple6(b_arc)) => {
                let (a1, a2, a3, a4, a5, a6) =
                    Arc::try_unwrap(a_arc).unwrap_or_else(|arc| (*arc).clone());
                let (b1, b2, b3, b4, b5, b6) =
                    Arc::try_unwrap(b_arc).unwrap_or_else(|arc| (*arc).clone());
                let c1 = a1.concat(b1)?;
                let c2 = a2.concat(b2)?;
                let c3 = a3.concat(b3)?;
                let c4 = a4.concat(b4)?;
                let c5 = a5.concat(b5)?;
                let c6 = a6.concat(b6)?;
                Ok(Value::Tuple6(Arc::new((c1, c2, c3, c4, c5, c6))))
            }

            // Recursive containers (Box, Arc)
            (BoxValue(a), BoxValue(b)) => {
                let result = (*a).concat(*b)?;
                Ok(Value::BoxValue(Box::new(result)))
            }

            (ArcValue(a), ArcValue(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                let result = a.concat(b)?;
                Ok(Value::ArcValue(Arc::new(result)))
            }

            // Views (materialise to owned, concat, wrap back in view)
            #[cfg(feature = "views")]
            (ArrayView(a), ArrayView(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(Value::ArrayView(Arc::new(a.concat(b)?)))
            }

            #[cfg(feature = "views")]
            (TableView(a), TableView(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(Value::TableView(Arc::new(a.concat(b)?)))
            }

            #[cfg(all(feature = "chunked", feature = "views"))]
            (SuperArrayView(a), SuperArrayView(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(Value::SuperArrayView(Arc::new(a.concat(b)?)))
            }

            #[cfg(all(feature = "chunked", feature = "views"))]
            (SuperTableView(a), SuperTableView(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(Value::SuperTableView(Arc::new(a.concat(b)?)))
            }

            // FieldArray + FieldArray => FieldArray
            (FieldArray(a), FieldArray(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(Value::FieldArray(Arc::new(a.concat(b)?)))
            }

            // VecValue - element-wise concatenation (recursive)
            (VecValue(a), VecValue(b)) => {
                // Unwrap Arcs
                let a_vec = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b_vec = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());

                // Validate same length
                if a_vec.len() != b_vec.len() {
                    return Err(MinarrowError::IncompatibleTypeError {
                        from: "VecValue",
                        to: "VecValue",
                        message: Some(format!(
                            "Cannot concatenate VecValue of different lengths: {} vs {}",
                            a_vec.len(),
                            b_vec.len()
                        )),
                    });
                }

                // Element-wise concatenation
                let mut result = Vec::with_capacity(a_vec.len());
                for (val_a, val_b) in a_vec.into_iter().zip(b_vec.into_iter()) {
                    result.push(val_a.concat(val_b)?);
                }

                Ok(Value::VecValue(Arc::new(result)))
            }

            // Custom values cannot be concatenated (no generic way to do it)
            (Custom(_), Custom(_)) => Err(MinarrowError::IncompatibleTypeError {
                from: "Custom",
                to: "Custom",
                message: Some("Cannot concatenate Custom values".to_string()),
            }),

            // Mismatched types
            (lhs, rhs) => Err(MinarrowError::IncompatibleTypeError {
                from: "Value",
                to: "Value",
                message: Some(format!(
                    "Cannot concatenate mismatched Value types: {} and {}",
                    value_variant_name(&lhs),
                    value_variant_name(&rhs)
                )),
            }),
        }
    }
}

// Helper Functions

/// Helper function to get scalar variant name for error messages
#[cfg(feature = "scalar_type")]
fn scalar_variant_name(scalar: &crate::Scalar) -> &'static str {
    use crate::Scalar::*;
    match scalar {
        Null => "Null",
        Boolean(_) => "Boolean",
        #[cfg(feature = "extended_numeric_types")]
        Int8(_) => "Int8",
        #[cfg(feature = "extended_numeric_types")]
        Int16(_) => "Int16",
        Int32(_) => "Int32",
        Int64(_) => "Int64",
        #[cfg(feature = "extended_numeric_types")]
        UInt8(_) => "UInt8",
        #[cfg(feature = "extended_numeric_types")]
        UInt16(_) => "UInt16",
        UInt32(_) => "UInt32",
        UInt64(_) => "UInt64",
        Float32(_) => "Float32",
        Float64(_) => "Float64",
        String32(_) => "String32",
        #[cfg(feature = "large_string")]
        String64(_) => "String64",
        #[cfg(feature = "datetime")]
        Datetime32(_) => "Datetime32",
        #[cfg(feature = "datetime")]
        Datetime64(_) => "Datetime64",
        #[cfg(feature = "datetime")]
        Interval => "Interval",
    }
}

/// Helper function to get value variant name for error messages
pub(crate) fn value_variant_name(value: &Value) -> &'static str {
    match value {
        #[cfg(feature = "scalar_type")]
        Value::Scalar(_) => "Scalar",
        Value::Array(_) => "Array",
        #[cfg(feature = "views")]
        Value::ArrayView(_) => "ArrayView",
        Value::Table(_) => "Table",
        #[cfg(feature = "views")]
        Value::TableView(_) => "TableView",
        #[cfg(feature = "chunked")]
        Value::SuperArray(_) => "SuperArray",
        #[cfg(all(feature = "chunked", feature = "views"))]
        Value::SuperArrayView(_) => "SuperArrayView",
        #[cfg(feature = "chunked")]
        Value::SuperTable(_) => "SuperTable",
        #[cfg(all(feature = "chunked", feature = "views"))]
        Value::SuperTableView(_) => "SuperTableView",
        Value::FieldArray(_) => "FieldArray",
        #[cfg(feature = "matrix")]
        Value::Matrix(_) => "Matrix",
        #[cfg(feature = "ndarray")]
        Value::NdArray(_) => "NdArray",
        #[cfg(all(feature = "ndarray", feature = "views"))]
        Value::NdArrayView(_) => "NdArrayView",
        #[cfg(all(feature = "ndarray", feature = "chunked"))]
        Value::SuperNdArray(_) => "SuperNdArray",
        #[cfg(all(feature = "ndarray", feature = "chunked", feature = "views"))]
        Value::SuperNdArrayView(_) => "SuperNdArrayView",
        #[cfg(feature = "xarray")]
        Value::XArray(_) => "XArray",
        #[cfg(feature = "cube")]
        Value::Cube(_) => "Cube",
        Value::VecValue(_) => "VecValue",
        Value::BoxValue(_) => "BoxValue",
        Value::ArcValue(_) => "ArcValue",
        Value::Tuple2(_) => "Tuple2",
        Value::Tuple3(_) => "Tuple3",
        Value::Tuple4(_) => "Tuple4",
        Value::Tuple5(_) => "Tuple5",
        Value::Tuple6(_) => "Tuple6",
        Value::Custom(_) => "Custom",
    }
}

// Consolidate for Vec<Value>

#[cfg(feature = "chunked")]
use crate::traits::consolidate::Consolidate;

#[cfg(feature = "chunked")]
impl Consolidate for Vec<Value> {
    type Output = Value;

    /// Consolidate a vector of Values into a single Value.
    ///
    /// Uses `Concatenate` to fold matching-typed Values together.
    /// A single-element vector returns the element directly.
    /// An empty vector returns an empty VecValue.
    ///
    /// All Values should be the same variant. Panics if concatenation
    /// fails due to type mismatch, as this indicates a programming error
    /// in the parallel execution that produced the chunks.
    fn consolidate(self) -> Value {
        match self.len() {
            0 => Value::VecValue(Arc::new(vec![])),
            1 => self.into_iter().next().unwrap(),
            _ => {
                let mut iter = self.into_iter();
                let first = iter.next().unwrap();
                iter.fold(first, |acc, val| {
                    // Should only produce VecValue containing homogeneous Value
                    // variants. The user-facing `TryFrom<Value::VecValue>` impls
                    // surface this as a typed error instead of a panic.
                    acc.concat(val)
                        .expect("consolidate: all Values must be the same variant")
                })
            }
        }
    }
}

// IntoIterator / FromIterator for streaming filter support

/// Iterate over the elements of a Value.
///
/// VecValue yields its inner items. All other variants yield themselves
/// as a single element, supporting keep-or-drop filter semantics.
impl IntoIterator for Value {
    type Item = Value;
    type IntoIter = std::vec::IntoIter<Value>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            Value::VecValue(items) => Arc::try_unwrap(items)
                .unwrap_or_else(|arc| (*arc).clone())
                .into_iter(),
            other => vec![other].into_iter(),
        }
    }
}

/// Collect Values back from an iterator.
///
/// Zero items produces an empty VecValue. A single item is returned
/// directly. Multiple items are wrapped in VecValue.
impl FromIterator<Value> for Value {
    fn from_iter<I: IntoIterator<Item = Value>>(iter: I) -> Self {
        let items: Vec<Value> = iter.into_iter().collect();
        match items.len() {
            0 => Value::VecValue(Arc::new(vec![])),
            1 => items.into_iter().next().unwrap(),
            _ => Value::VecValue(Arc::new(items)),
        }
    }
}

// Tests

#[cfg(test)]
mod concat_tests {
    use super::*;
    use crate::Array;
    use crate::MaskedArray;
    use crate::structs::field_array::field_array;
    use crate::structs::variants::integer::IntegerArray;

    #[test]
    fn test_value_size() {
        use std::mem::size_of;
        println!("\n=== Value Enum Size Analysis ===");
        println!("Total Value enum size: {} bytes", size_of::<Value>());
        println!("\nIndividual type sizes:");
        println!("  Array: {} bytes", size_of::<crate::Array>());
        println!("  Table: {} bytes", size_of::<crate::Table>());
        println!("  Bitmask: {} bytes", size_of::<crate::Bitmask>());
        println!("  FieldArray: {} bytes", size_of::<crate::FieldArray>());
        println!("  Field: {} bytes", size_of::<crate::Field>());
        #[cfg(feature = "matrix")]
        println!("  Matrix: {} bytes", size_of::<crate::Matrix>());
        #[cfg(feature = "cube")]
        println!("  Cube: {} bytes", size_of::<crate::Cube>());
        #[cfg(feature = "chunked")]
        println!("  SuperArray: {} bytes", size_of::<crate::SuperArray>());
        #[cfg(feature = "chunked")]
        println!("  SuperTable: {} bytes", size_of::<crate::SuperTable>());
        #[cfg(feature = "views")]
        println!("  ArrayView: {} bytes", size_of::<crate::ArrayV>());
        #[cfg(feature = "views")]
        println!("  TableView: {} bytes", size_of::<crate::TableV>());
        #[cfg(all(feature = "views", feature = "chunked"))]
        println!(
            "  SuperArrayView: {} bytes",
            size_of::<crate::SuperArrayV>()
        );
        #[cfg(all(feature = "views", feature = "chunked"))]
        println!(
            "  SuperTableView: {} bytes",
            size_of::<crate::SuperTableV>()
        );
        println!("  Box<Vec<Value>>: {} bytes", size_of::<Box<Vec<Value>>>());
        println!("  Vec<Value>: {} bytes", size_of::<Vec<Value>>());
    }

    #[test]
    fn test_value_concat_field_array() {
        // Create two FieldArrays with matching metadata
        let arr1 = IntegerArray::<i32>::from_slice(&[1, 2, 3]);
        let fa1 = field_array("data", Array::from_int32(arr1));
        let val1 = Value::FieldArray(Arc::new(fa1));

        let arr2 = IntegerArray::<i32>::from_slice(&[4, 5, 6]);
        let fa2 = field_array("data", Array::from_int32(arr2));
        let val2 = Value::FieldArray(Arc::new(fa2));

        let result = val1.concat(val2).unwrap();

        if let Value::FieldArray(fa_arc) = result {
            let fa = Arc::unwrap_or_clone(fa_arc);
            assert_eq!(fa.len(), 6);
            assert_eq!(fa.field.name, "data");
            if let Array::NumericArray(crate::NumericArray::Int32(arr)) = fa.array {
                assert_eq!(arr.get(0), Some(1));
                assert_eq!(arr.get(5), Some(6));
            } else {
                panic!("Expected Int32 array");
            }
        } else {
            panic!("Expected FieldArray value");
        }
    }

    #[test]
    fn test_value_concat_vec_value() {
        // Create two VecValues with same length and matching types
        let arr1_1 = IntegerArray::<i32>::from_slice(&[1, 2]);
        let arr1_2 = IntegerArray::<i32>::from_slice(&[10, 20]);
        let val1 = Value::VecValue(Arc::new(vec![
            Value::Array(Arc::new(Array::from_int32(arr1_1))),
            Value::Array(Arc::new(Array::from_int32(arr1_2))),
        ]));

        let arr2_1 = IntegerArray::<i32>::from_slice(&[3, 4]);
        let arr2_2 = IntegerArray::<i32>::from_slice(&[30, 40]);
        let val2 = Value::VecValue(Arc::new(vec![
            Value::Array(Arc::new(Array::from_int32(arr2_1))),
            Value::Array(Arc::new(Array::from_int32(arr2_2))),
        ]));

        let result = val1.concat(val2).unwrap();

        if let Value::VecValue(vec) = result {
            assert_eq!(vec.len(), 2);

            // Check first element
            if let Value::Array(arc) = &vec[0] {
                if let Array::NumericArray(crate::NumericArray::Int32(arr)) = arc.as_ref() {
                    assert_eq!(arr.len(), 4);
                    assert_eq!(arr.get(0), Some(1));
                    assert_eq!(arr.get(1), Some(2));
                    assert_eq!(arr.get(2), Some(3));
                    assert_eq!(arr.get(3), Some(4));
                } else {
                    panic!("Expected Int32 array in first element");
                }
            } else {
                panic!("Expected Array value in first element");
            }

            // Check second element
            if let Value::Array(arc) = &vec[1] {
                if let Array::NumericArray(crate::NumericArray::Int32(arr)) = arc.as_ref() {
                    assert_eq!(arr.len(), 4);
                    assert_eq!(arr.get(0), Some(10));
                    assert_eq!(arr.get(1), Some(20));
                    assert_eq!(arr.get(2), Some(30));
                    assert_eq!(arr.get(3), Some(40));
                } else {
                    panic!("Expected Int32 array in second element");
                }
            } else {
                panic!("Expected Array value in second element");
            }
        } else {
            panic!("Expected VecValue");
        }
    }

    #[test]
    fn test_value_concat_vec_value_length_mismatch() {
        let arr1 = IntegerArray::<i32>::from_slice(&[1, 2]);
        let val1 = Value::VecValue(Arc::new(vec![Value::Array(Arc::new(Array::from_int32(
            arr1,
        )))]));

        let arr2_1 = IntegerArray::<i32>::from_slice(&[3, 4]);
        let arr2_2 = IntegerArray::<i32>::from_slice(&[5, 6]);
        let val2 = Value::VecValue(Arc::new(vec![
            Value::Array(Arc::new(Array::from_int32(arr2_1))),
            Value::Array(Arc::new(Array::from_int32(arr2_2))),
        ]));

        let result = val1.concat(val2);
        assert!(result.is_err());

        if let Err(MinarrowError::IncompatibleTypeError { message, .. }) = result {
            assert!(message.unwrap().contains("different lengths"));
        } else {
            panic!("Expected IncompatibleTypeError");
        }
    }

    #[test]
    fn test_value_concat_vec_value_type_mismatch() {
        // Element types don't match - first element is Int32, second is Float64
        let arr1_1 = IntegerArray::<i32>::from_slice(&[1, 2]);
        let arr1_2 = IntegerArray::<i32>::from_slice(&[10, 20]);
        let val1 = Value::VecValue(Arc::new(vec![
            Value::Array(Arc::new(Array::from_int32(arr1_1))),
            Value::Array(Arc::new(Array::from_int32(arr1_2))),
        ]));

        let arr2_1 = IntegerArray::<i32>::from_slice(&[3, 4]);
        let arr2_2 = crate::FloatArray::<f64>::from_slice(&[30.0, 40.0]);
        let val2 = Value::VecValue(Arc::new(vec![
            Value::Array(Arc::new(Array::from_int32(arr2_1))),
            Value::Array(Arc::new(Array::from_float64(arr2_2))),
        ]));

        let result = val1.concat(val2);
        assert!(result.is_err());

        // Should fail when trying to concat the second elements
        if let Err(MinarrowError::IncompatibleTypeError { .. }) = result {
            // Expected
        } else {
            panic!("Expected IncompatibleTypeError");
        }
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_value_len_ndarray_types() {
        use crate::NdArray;

        // Column-major [3, 2] with 3 axis-0 observations
        let nd = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        assert_eq!(Value::NdArray(Arc::new(nd.clone())).len(), 3);

        #[cfg(feature = "views")]
        assert_eq!(Value::NdArrayView(Arc::new(nd.as_view())).len(), 3);

        #[cfg(feature = "chunked")]
        {
            use crate::SuperNdArray;
            let snd = SuperNdArray::from_batches(
                vec![
                    NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
                    NdArray::from_slice(&[5.0, 6.0], &[1, 2]),
                ],
                "s",
            );
            assert_eq!(Value::SuperNdArray(Arc::new(snd.clone())).len(), 3);
            #[cfg(feature = "views")]
            assert_eq!(Value::SuperNdArrayView(Arc::new(snd.slice(0, 2))).len(), 2);
        }

        #[cfg(feature = "xarray")]
        {
            use crate::XArray;
            let xa = XArray::new(nd, &["obs", "feat"]);
            assert_eq!(Value::XArray(Arc::new(xa)).len(), 3);
        }
    }

    #[cfg(all(feature = "ndarray", feature = "views"))]
    #[test]
    fn test_value_slice_ndarray() {
        use crate::NdArray;

        let nd = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let v = Value::NdArray(Arc::new(nd));

        // The full window keeps the shape.
        let full = v.slice(0, v.len());
        let Value::NdArrayView(view) = full else {
            panic!("Expected Value::NdArrayView");
        };
        assert_eq!(view.shape(), &[3, 2]);
        assert_eq!(view.get(&[0, 0]), 1.0);
        assert_eq!(view.get(&[2, 1]), 6.0);

        // A partial window covers rows 1..3.
        let window = v.slice(1, 2);
        let Value::NdArrayView(view) = window else {
            panic!("Expected Value::NdArrayView");
        };
        assert_eq!(view.shape(), &[2, 2]);
        assert_eq!(view.get(&[0, 0]), 2.0);
        assert_eq!(view.get(&[1, 0]), 3.0);
        assert_eq!(view.get(&[0, 1]), 5.0);
        assert_eq!(view.get(&[1, 1]), 6.0);
    }

    #[cfg(all(feature = "ndarray", feature = "views"))]
    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_value_slice_ndarray_out_of_bounds_panics() {
        use crate::NdArray;

        let nd = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let v = Value::NdArray(Arc::new(nd));
        let _ = v.slice(2, 2);
    }

    #[cfg(all(feature = "xarray", feature = "views", feature = "select"))]
    #[test]
    fn test_value_slice_xarray_narrows_coords() {
        use crate::{FloatArray, NdArray, NumericArray, XArray};

        let mut xa = XArray::new(
            NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]),
            &["obs", "feat"],
        );
        xa.assign_coords(
            "obs",
            Array::NumericArray(NumericArray::Float64(Arc::new(
                FloatArray::from_slice(&[10.0, 20.0, 30.0]),
            ))),
        );
        let v = Value::XArray(Arc::new(xa));

        let sliced = v.slice(1, 2);
        let Value::XArray(out) = sliced else {
            panic!("Expected Value::XArray");
        };
        assert_eq!(out.shape(), vec![2, 2]);
        // The axis-0 coords narrow alongside the data window.
        let coords = out.ax("obs").coords.as_ref().unwrap();
        let expected = Array::NumericArray(NumericArray::Float64(Arc::new(
            FloatArray::from_slice(&[20.0, 30.0]),
        )));
        assert_eq!(*coords, expected);
    }

    #[cfg(all(feature = "ndarray", feature = "views"))]
    #[test]
    fn test_value_concat_ndarray_views() {
        use crate::NdArray;

        let a = NdArray::from_slice(&[1.0, 2.0], &[2]);
        let b = NdArray::from_slice(&[3.0], &[1]);
        let va = Value::NdArrayView(Arc::new(a.as_view()));
        let vb = Value::NdArrayView(Arc::new(b.as_view()));

        let out = va.concat(vb).unwrap();
        let Value::NdArray(nd) = out else {
            panic!("Expected Value::NdArray");
        };
        assert_eq!(nd.shape(), &[3]);
        assert_eq!(nd.get(&[0]), 1.0);
        assert_eq!(nd.get(&[1]), 2.0);
        assert_eq!(nd.get(&[2]), 3.0);
    }
}
