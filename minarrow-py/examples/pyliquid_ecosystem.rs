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

//! Demonstrates calling Python data libraries from a Rust application.
//!
//! Minarrow ensures that ownership fully passes between the two runtimes,
//! so that any Arrow-shaped data that comes back from a library implementing
//! `PyCapsule` is trivially cheap, meaning that even 100m rows of data can 
//! cross the boundary from either direction within single digit microseconds.
//! 
//! The example creates a Minarrow table in Rust and passes it to an embedded
//! Python interpreter. It runs operations with Polars, NumPy and pandas, then
//! converts the results back to Minarrow [`Value`] variants, which leverages
//! enum-based routing for any custom-behaviour.
//!
//! The examples cover:
//!
//! - A Polars aggregation returning [`Value::Table`]
//! - A NumPy correlation returning [`Value::Scalar`]
//! - A pandas aggregation returned through PyArrow as [`Value::Table`]
//! - A tuple containing a table, scalar and list
//! - Basic measurements of interpreter, bridge and Python-operation costs
//!
//! Python receives the input as a native `minarrow.Table`. The inline Python
//! code can therefore use in-built methods such as `to_polars()` and `to_pandas()`,
//! provided the virtual environment already has these installed (see below).
//!
//! ## Running
//!
//! Build with the `embed` feature and use a Python environment containing
//! `polars`, `numpy`, `pandas` and `pyarrow`. `PYO3_PYTHON` must identify the
//! interpreter used for the build.
//!
//! ```bash
//! cd minarrow-py
//! PYO3_PYTHON=$PWD/../pyo3/.venv/bin/python \
//! PYTHONHOME=/usr \
//! PYTHONPATH=$PWD/../pyo3/.venv/lib/python3.12/site-packages \
//! LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
//! cargo run --example pyliquid_ecosystem --features embed
//! ```

use std::ffi::CStr;
use std::hint::black_box;
use std::time::Instant;

use minarrow::{fa_f64, fa_i64, fa_str32, Array, NumericArray, Print, Scalar, Table, Value, Vec64};
use minarrow_py::{PyInput, PyLiquid};
use minarrow_pyo3::ffi::to_rust;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn main() -> PyResult<()> {
    let total = Instant::now();

    // Initialise the interpreter and load the libraries used by the example.
    let t = Instant::now();
    let rt = PyLiquid::start().preimport(["polars", "numpy", "pandas", "pyarrow"])?;
    let startup = t.elapsed();

    let trades = build_trades();
    banner("Source table built in Rust");
    trades.print();

    let t = Instant::now();
    polars_group_by(&rt, &trades)?;
    let polars = t.elapsed();

    let t = Instant::now();
    numpy_correlation(&rt, &trades)?;
    let numpy = t.elapsed();

    let t = Instant::now();
    pandas_summary(&rt, &trades)?;
    let pandas = t.elapsed();

    let t = Instant::now();
    composite_result(&rt, &trades)?;
    let composite = t.elapsed();

    bridge_costs(&trades)?;

    banner("Timing");
    if cfg!(debug_assertions) {
        println!("  Debug build. Use --release for representative timings.\n");
    }
    println!("  Startup and preimport : {startup:>10.2?}");
    println!("  1. Polars group-by    : {polars:>10.2?}");
    println!("  2. NumPy correlation  : {numpy:>10.2?}");
    println!("  3. pandas summary     : {pandas:>10.2?}");
    println!("  4. Composite result   : {composite:>10.2?}");
    println!("  {}", "-".repeat(34));
    println!("  Total                 : {:>10.2?}", total.elapsed());

    Ok(())
}

// *************************************************************************
// Polars aggregation
// *************************************************************************

fn polars_group_by(rt: &PyLiquid, trades: &Table) -> PyResult<()> {
    banner("1. Notional by symbol with Polars");

    let code = cr#"
import polars as pl

df = trades.to_polars()

result = (
    df.with_columns((pl.col("quantity") * pl.col("price")).alias("notional"))
      .group_by("symbol")
      .agg(
          pl.col("notional").sum().round(2).alias("notional"),
          pl.col("price").mean().round(2).alias("avg_price"),
          pl.len().alias("n_trades"),
      )
      .sort("notional", descending=True)
)
"#;

    let result = rt.with_python(trades, |py, obj| run_block(py, obj, code))?;
    match result {
        Value::Table(table) => {
            table.print();
            report_alignment(&table);
        }
        other => println!("Unexpected result variant {other:?}"),
    }

    Ok(())
}

// *************************************************************************
// NumPy correlation
// *************************************************************************

fn numpy_correlation(rt: &PyLiquid, trades: &Table) -> PyResult<()> {
    banner("2. Quantity and price correlation with NumPy");

    let code = cr#"
import numpy as np

df = trades.to_polars()
quantity = df["quantity"].to_numpy()
price = df["price"].to_numpy()

result = float(np.corrcoef(quantity, price)[0, 1])
"#;

    let result = rt.with_python(trades, |py, obj| run_block(py, obj, code))?;
    match result {
        Value::Scalar(Scalar::Float64(r)) => println!("Pearson r = {r:.4}"),
        other => println!("Unexpected result variant {other:?}"),
    }

    Ok(())
}

// *************************************************************************
// pandas aggregation
// *************************************************************************

fn pandas_summary(rt: &PyLiquid, trades: &Table) -> PyResult<()> {
    banner("3. Price statistics by symbol with pandas");

    let code = cr#"
import pandas as pd
import pyarrow as pa

frame = trades.to_pandas()
summary = (
    frame.groupby("symbol")["price"]
         .agg(["mean", "std", "min", "max"])
         .round(2)
         .reset_index()
)

result = pa.Table.from_pandas(summary, preserve_index=False)
"#;

    let result = rt.with_python(trades, |py, obj| run_block(py, obj, code))?;
    match result {
        Value::Table(table) => table.print(),
        other => println!("Unexpected result variant {other:?}"),
    }

    Ok(())
}

// *************************************************************************
// Composite return value
// *************************************************************************

fn composite_result(rt: &PyLiquid, trades: &Table) -> PyResult<()> {
    banner("4. A table, scalar and list in one call");

    let code = cr#"
import polars as pl

df = trades.to_polars().with_columns(
    (pl.col("quantity") * pl.col("price")).alias("notional")
)

by_symbol = (
    df.group_by("symbol")
      .agg(pl.col("notional").sum().round(2).alias("notional"))
      .sort("notional", descending=True)
)

total_notional = float(df["notional"].sum())
ranking = by_symbol["symbol"].to_list()

result = (by_symbol, total_notional, ranking)
"#;

    let result = rt.with_python(trades, |py, obj| run_block(py, obj, code))?;
    let Value::Tuple3(parts) = result else {
        println!("Unexpected result variant {result:?}");
        return Ok(());
    };

    let (ranked, total, ranking) = &*parts;

    if let Value::Table(table) = ranked {
        println!("Symbols ranked by notional");
        table.print();
    }

    if let Value::Scalar(Scalar::Float64(total)) = total {
        println!("Total notional = {total:.2}");
    }

    if let Value::VecValue(items) = ranking {
        let names: Vec<&str> = items
            .iter()
            .filter_map(|value| match value {
                Value::Scalar(Scalar::String32(name)) => Some(name.as_str()),
                _ => None,
            })
            .collect();

        println!("Symbol order = {names:?}");
    }

    Ok(())
}

// *************************************************************************
// Bridge measurements
// *************************************************************************

fn bridge_costs(small: &Table) -> PyResult<()> {
    banner("Bridge cost");

    // Measure uncontended GIL acquisition and release on the current thread.
    const G: u32 = 100_000;
    let t = Instant::now();

    for _ in 0..G {
        Python::with_gil(|_py| {});
    }

    println!("  GIL acquire and release                : {:>11.3?}", t.elapsed() / G);

    let large = build_large(1_000_000);

    Python::with_gil(|py| -> PyResult<()> {
        const N: u32 = 50;

        println!();
        println!("  Numeric buffers are shared, not copied, so import time does not grow with row count.");

        // Export a Minarrow table to a native Python object.
        let t = Instant::now();
        for _ in 0..N {
            let object = large.to_python(py)?;
            black_box(&object);
        }
        println!("  Rust to Python, 1,000,000 rows         : {:>11.3?}", t.elapsed() / N);

        // Import numeric tables at several row counts. The flat time confirms no memory copy.
        for (rows, label) in [
            (100_000usize, "100,000"),
            (1_000_000, "1,000,000"),
            (10_000_000, "10,000,000"),
        ] {
            let source = build_large(rows).to_python(py)?;
            let t = Instant::now();
            for _ in 0..N {
                let back: Table = to_rust::record_batch_to_rust(&source)?;
                black_box(&back);
            }
            println!("  Python to Rust, {label:>10} rows        : {:>11.3?}", t.elapsed() / N);
        }

        println!();
        println!("  String offsets are currently copied Minarrow, while the text is shared, so import time tracks row count rather than text length. This will change in a future version.");

        for width in [9usize, 64] {
            let source = build_strings(1_000_000, width).to_python(py)?;
            let t = Instant::now();
            for _ in 0..N {
                let back: Table = to_rust::record_batch_to_rust(&source)?;
                black_box(&back);
            }
            println!("  Python to Rust, 1,000,000 rows of {width:>2}-byte strings : {:>11.3?}", t.elapsed() / N);
        }

        println!();

        // A complete round trip using the small example table.
        const M: u32 = 2_000;
        let t = Instant::now();
        for _ in 0..M {
            let object = small.to_python(py)?;
            let back: Table = to_rust::record_batch_to_rust(&object)?;
            black_box(&back);
        }
        println!("  Round trip on the 8-row table          : {:>11.3?}", t.elapsed() / M);

        // The repeated Polars operation, measured once the engine has started.
        let code = cr#"
import polars as pl
result = trades.to_polars().group_by("symbol").agg(pl.col("price").mean())
"#;
        let scope = PyDict::new(py);
        scope.set_item("trades", small.to_python(py)?)?;

        const P: u32 = 200;
        let t = Instant::now();
        for _ in 0..P {
            py.run(code, Some(&scope), Some(&scope))?;
        }
        println!("  Polars group-by, repeated              : {:>11.3?}", t.elapsed() / P);

        Ok(())
    })?;

    Ok(())
}

// *************************************************************************
// Python execution and test data
// *************************************************************************

/// Executes a Python block with `obj` bound as `trades`.
///
/// The block must assign its return value to `result`.
fn run_block<'py>(
    py: Python<'py>,
    obj: Bound<'py, PyAny>,
    code: &CStr,
) -> PyResult<Bound<'py, PyAny>> {
    let scope = PyDict::new(py);
    scope.set_item("trades", obj)?;
    py.run(code, Some(&scope), Some(&scope))?;

    scope
        .get_item("result")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("python block did not set `result`"))
}

/// Builds the example trades table.
fn build_trades() -> Table {
    Table::new(
        "trades".to_string(),
        Some(vec![
            fa_str32!("symbol", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT", "GOOG", "AAPL", "GOOG"),
            fa_i64!("quantity", 100, 50, 200, 120, 75, 300, 80, 150),
            fa_f64!("price", 191.2, 192.0, 410.5, 408.2, 411.0, 138.7, 190.5, 139.4),
        ]),
    )
}

/// Builds a numeric table for bridge measurements.
fn build_large(rows: usize) -> Table {
    let ids: Vec64<i64> = (0..rows as i64).collect();
    let pxs: Vec64<f64> = (0..rows).map(|index| index as f64 * 1.5).collect();

    Table::new(
        "bench".to_string(),
        Some(vec![fa_i64!("id", @vec64 ids), fa_f64!("px", @vec64 pxs)]),
    )
}

/// Builds a fixed-width string column for import measurements.
fn build_strings(rows: usize, width: usize) -> Table {
    let owned: Vec<String> = (0..rows)
        .map(|index| format!("{:0width$}", index % 1000))
        .collect();
    let refs: Vec<&str> = owned.iter().map(String::as_str).collect();

    Table::new(
        "strings".to_string(),
        Some(vec![fa_str32!("key", @slice refs)]),
    )
}

/// Prints the alignment of returned `f64` column buffers.
fn report_alignment(table: &Table) {
    for column in &table.cols {
        if let Array::NumericArray(NumericArray::Float64(array)) = &column.array {
            let address = array.data.as_ptr() as usize;

            let status = if address % 64 == 0 {
                "64-byte aligned"
            } else {
                "not 64-byte aligned"
            };
            println!(
                "  Column {} float buffer at {:#x} is {}.",
                column.field.name, address, status
            );
        }
    }
}

fn banner(title: &str) {
    println!("\n{0}\n {title}\n{0}", "=".repeat(60));
}
