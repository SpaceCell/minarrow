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

//! Roundtrip tests for every `PyInput` implementation.
//!
//! Each test sends a Minarrow value to Python through `to_python`, then
//! returns it immediately. `classify` on the Python side maps the result
//! back to a Minarrow `Value`, and the test verifies the data survived
//! the round trip.
//!
//! ## Running
//! ```bash
//! cd minarrow-py
//! PYO3_PYTHON=$PWD/../pyo3/.venv/bin/python \
//!   PYTHONHOME=/usr \
//!   PYTHONPATH=$PWD/../pyo3/.venv/lib/python3.12/site-packages \
//!   LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
//!   cargo run --example roundtrip_to_python --features embed
//! ```

use std::sync::Arc;

use minarrow::{
    arr_f64, fa_f64, fa_str32, Array, ArrayV, FieldArray, Scalar, SuperArray,
    SuperTable, Table, TableV, Value,
};
use minarrow_py::{PyInput, PyLiquid};
use pyo3::prelude::*;

fn main() -> PyResult<()> {
    let rt = PyLiquid::start();

    roundtrip_table(&rt);
    roundtrip_array(&rt);
    roundtrip_field_array(&rt);
    roundtrip_table_view(&rt);
    roundtrip_array_view(&rt);
    roundtrip_super_table(&rt);
    roundtrip_super_array(&rt);
    roundtrip_scalar_float(&rt);
    roundtrip_scalar_int(&rt);
    roundtrip_scalar_bool(&rt);
    roundtrip_scalar_string(&rt);
    roundtrip_scalar_null(&rt);
    roundtrip_value_table(&rt);
    roundtrip_value_array(&rt);
    roundtrip_value_scalar(&rt);

    println!("All PyInput roundtrip tests passed.");
    Ok(())
}

fn roundtrip_table(rt: &PyLiquid) {
    let table = Table::new("test", Some(vec![
        fa_f64!("x", 1.0, 2.0, 3.0),
        fa_str32!("y", "a", "b", "c"),
    ]));
    let result = rt.with_python(&table, |_py, obj| Ok(obj)).unwrap();
    let Value::Table(t) = result else { panic!("expected Table, got {result:?}") };
    assert_eq!(t.n_rows, 3);
    assert_eq!(t.n_cols(), 2);
    println!("  Table roundtrip: ok");
}

fn roundtrip_array(rt: &PyLiquid) {
    let array: Array = arr_f64![10.0, 20.0, 30.0].into();
    let result = rt.with_python(&array, |_py, obj| Ok(obj)).unwrap();
    let Value::Array(a) = result else { panic!("expected Array, got {result:?}") };
    assert_eq!(a.len(), 3);
    println!("  Array roundtrip: ok");
}

fn roundtrip_field_array(rt: &PyLiquid) {
    let fa = fa_f64!("amount", 1.5, 2.5);
    let empty = Table::new_empty();
    let result = rt.with_python(&empty, |py, _| fa.to_python(py)).unwrap();
    let Value::Array(a) = result else { panic!("expected Array, got {result:?}") };
    assert_eq!(a.len(), 2);
    println!("  FieldArray roundtrip: ok");
}

fn roundtrip_table_view(rt: &PyLiquid) {
    let table = Table::new("test", Some(vec![fa_f64!("x", 1.0, 2.0, 3.0, 4.0)]));
    let view = TableV::from(table);
    let empty = Table::new_empty();
    let result = rt.with_python(&empty, |py, _| view.to_python(py)).unwrap();
    let Value::Table(t) = result else { panic!("expected Table, got {result:?}") };
    assert_eq!(t.n_rows, 4);
    println!("  TableV roundtrip: ok");
}

fn roundtrip_array_view(rt: &PyLiquid) {
    let array: Array = arr_f64![5.0, 6.0, 7.0].into();
    let view = ArrayV::from(array);
    let empty = Table::new_empty();
    let result = rt.with_python(&empty, |py, _| view.to_python(py)).unwrap();
    let Value::Array(a) = result else { panic!("expected Array, got {result:?}") };
    assert_eq!(a.len(), 3);
    println!("  ArrayV roundtrip: ok");
}

fn roundtrip_super_table(rt: &PyLiquid) {
    let t1 = Table::new("t", Some(vec![fa_f64!("x", 1.0, 2.0)]));
    let t2 = Table::new("t", Some(vec![fa_f64!("x", 3.0, 4.0)]));
    let st = SuperTable::from_batches(vec![Arc::new(t1), Arc::new(t2)], None);
    let empty = Table::new_empty();
    let result = rt.with_python(&empty, |py, _| st.to_python(py)).unwrap();
    let Value::SuperTable(s) = result else { panic!("expected SuperTable, got {result:?}") };
    assert_eq!(s.n_rows(), 4);
    println!("  SuperTable roundtrip: ok");
}

fn roundtrip_super_array(rt: &PyLiquid) {
    let a1: Array = arr_f64![1.0, 2.0].into();
    let a2: Array = arr_f64![3.0, 4.0].into();
    let sa = SuperArray::from_arrays(vec![a1, a2]);
    let empty = Table::new_empty();
    let result = rt.with_python(&empty, |py, _| sa.to_python(py)).unwrap();
    let Value::SuperArray(s) = result else { panic!("expected SuperArray, got {result:?}") };
    assert_eq!(s.len(), 4);
    println!("  SuperArray roundtrip: ok");
}

fn roundtrip_scalar_float(rt: &PyLiquid) {
    let s = Scalar::Float64(3.14);
    let empty = Table::new_empty();
    let result = rt.with_python(&empty, |py, _| s.to_python(py)).unwrap();
    let Value::Scalar(Scalar::Float64(v)) = result else { panic!("expected Float64, got {result:?}") };
    assert!((v - 3.14).abs() < 1e-10);
    println!("  Scalar::Float64 roundtrip: ok");
}

fn roundtrip_scalar_int(rt: &PyLiquid) {
    let s = Scalar::Int64(42);
    let empty = Table::new_empty();
    let result = rt.with_python(&empty, |py, _| s.to_python(py)).unwrap();
    let Value::Scalar(Scalar::Int64(v)) = result else { panic!("expected Int64, got {result:?}") };
    assert_eq!(v, 42);
    println!("  Scalar::Int64 roundtrip: ok");
}

fn roundtrip_scalar_bool(rt: &PyLiquid) {
    let s = Scalar::Boolean(true);
    let empty = Table::new_empty();
    let result = rt.with_python(&empty, |py, _| s.to_python(py)).unwrap();
    let Value::Scalar(Scalar::Boolean(v)) = result else { panic!("expected Boolean, got {result:?}") };
    assert!(v);
    println!("  Scalar::Boolean roundtrip: ok");
}

fn roundtrip_scalar_string(rt: &PyLiquid) {
    let s = Scalar::String32("hello".to_string());
    let empty = Table::new_empty();
    let result = rt.with_python(&empty, |py, _| s.to_python(py)).unwrap();
    let Value::Scalar(Scalar::String32(v)) = result else { panic!("expected String32, got {result:?}") };
    assert_eq!(v, "hello");
    println!("  Scalar::String32 roundtrip: ok");
}

fn roundtrip_scalar_null(rt: &PyLiquid) {
    let s = Scalar::Null;
    let empty = Table::new_empty();
    let result = rt.with_python(&empty, |py, _| s.to_python(py)).unwrap();
    let Value::Scalar(Scalar::Null) = result else { panic!("expected Null, got {result:?}") };
    println!("  Scalar::Null roundtrip: ok");
}

fn roundtrip_value_table(rt: &PyLiquid) {
    let table = Table::new("v", Some(vec![fa_f64!("a", 1.0)]));
    let value = Value::Table(Arc::new(table));
    let empty = Table::new_empty();
    let result = rt.with_python(&empty, |py, _| value.to_python(py)).unwrap();
    let Value::Table(t) = result else { panic!("expected Table, got {result:?}") };
    assert_eq!(t.n_rows, 1);
    println!("  Value::Table roundtrip: ok");
}

fn roundtrip_value_array(rt: &PyLiquid) {
    let array: Array = arr_f64![9.0, 8.0].into();
    let value = Value::Array(Arc::new(array));
    let empty = Table::new_empty();
    let result = rt.with_python(&empty, |py, _| value.to_python(py)).unwrap();
    let Value::Array(a) = result else { panic!("expected Array, got {result:?}") };
    assert_eq!(a.len(), 2);
    println!("  Value::Array roundtrip: ok");
}

fn roundtrip_value_scalar(rt: &PyLiquid) {
    let value = Value::Scalar(Scalar::Int64(99));
    let empty = Table::new_empty();
    let result = rt.with_python(&empty, |py, _| value.to_python(py)).unwrap();
    let Value::Scalar(Scalar::Int64(v)) = result else { panic!("expected Int64, got {result:?}") };
    assert_eq!(v, 99);
    println!("  Value::Scalar roundtrip: ok");
}
