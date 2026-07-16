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

//! Demonstrates lifetime soundness across Python interpreter shutdown.
//!
//! A Minarrow table is built in Rust and passed into Python. Polars multiplies a
//! column through its own engine and returns the resulting frame, which is the
//! object Minarrow imports back over the Arrow PyCapsule interface. CPython is
//! then finalised, and the result is read and dropped with no interpreter
//! present.
//!
//! The example leverages the minarrow-pyo3 crate, which manages lifetime transfers
//! of the foreign-held buffer back to Rust, ensuring that even large objects
//! pass over the Python -> Rust boundary as a cheap pointer, with Rust then reclaiming
//! ownership to avoid double frees.

use std::thread::sleep;
use std::time::Duration;

use minarrow::{fa_i64, Table, Value, Vec64};
use minarrow_py::PyLiquid;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn main() -> PyResult<()> {
    const ROWS: i64 = 1_000_000;
    let rt = PyLiquid::start().preimport(["polars"])?;

    // Build the input table in Rust.
    let ids: Vec64<i64> = (0..ROWS).collect();
    let input = Table::new("input".to_string(), Some(vec![fa_i64!("x", @vec64 ids)]));
    let input_address = input.cols[0].array.num().i64().data.as_ptr() as usize;

    // Pass the table into Python, multiply the column in Polars, and return the
    // resulting frame. That dataframe is the object imported back into Minarrow.
    let value = rt.with_python(&input, |py, obj| {
        let scope = PyDict::new(py);
        scope.set_item("table", obj)?;
        py.run(
                        cr#"
import polars as pl

df = table.to_polars()
result = df.with_columns((pl.col("x") * 7).alias("x"))
"#,
            Some(&scope),
            Some(&scope),
        )?;
        scope
            .get_item("result")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("result not set"))
    })?;

    let result = match value {
        Value::Table(table) => (*table).clone(),
        other => panic!("expected a table, got {other:?}"),
    };
    let result_address = result.cols[0].array.num().i64().data.as_ptr() as usize;
    println!("Received the Polars result into Rust.");
    println!("  Rust input buffer    : {input_address:#x}");
    println!("  Polars result buffer : {result_address:#x}");
    println!(
        "  Result is the Polars-produced buffer: {}",
        result_address != input_address
    );

    // Shut CPython down completely. Acquire the GIL through the raw API and
    // finalise without the pyo3 guard. No Python call is made after this point.
    unsafe {
        pyo3::ffi::PyGILState_Ensure();
        pyo3::ffi::Py_FinalizeEx();
    }
    println!("CPython finalised.");
    sleep(Duration::from_secs(1));

    // Use the Polars result after Python is gone.
    let sum: i64 = result.cols[0].array.clone().num().i64().data.iter().sum();
    let expected: i64 = (0..ROWS).map(|value| value * 7).sum();
    println!(
        "After shutdown: rows = {}, sum = {sum} (expected {expected})",
        result.cols[0].len()
    );
    assert_eq!(sum, expected, "result was wrong after Python shut down");

    // Free the result after shutdown. This runs the Arrow release callback that
    // frees the Polars-produced buffer, with no interpreter present.
    drop(result);
    println!("Python result used from Rust and then freed after Python was shut down.");

    Ok(())
}
