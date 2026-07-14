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

//! Round-trips a dataset through a scikit-learn model.
//!
//! Features and labels are built in Rust and passed into Python. scikit-learn is
//! numpy-native rather than Arrow-native, so the closure reads the table into
//! numpy through Polars, trains a classifier, predicts on a held-out split, and
//! returns the actual and predicted labels as a Polars frame. That frame crosses
//! back over the Arrow PyCapsule interface and Minarrow scores the accuracy.
//!
//! The labels are a clear linear function of three of the four features plus mild
//! noise, so the model converges immediately without real data.
//!
//! ## Running
//! Needs a venv with `polars`, `numpy`, `scikit-learn`. `PYO3_PYTHON` must be an
//! absolute path to it.
//! ```bash
//! cd minarrow-py
//! PYO3_PYTHON=$PWD/../pyo3/.venv/bin/python \
//!   PYTHONHOME=/usr \
//!   PYTHONPATH=$PWD/../pyo3/.venv/lib/python3.12/site-packages \
//!   LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
//!   cargo run --release --example roundtrip_ml --features embed
//! ```

use minarrow::{fa_f64, fa_i64, Table, Value, Vec64};
use minarrow_py::PyLiquid;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn main() -> PyResult<()> {
    let rt = PyLiquid::start().preimport(["numpy", "polars"])?;

    let dataset = build_dataset(4000);
    println!("Built {} rows x {} cols in Rust.", dataset.n_rows, dataset.cols.len());

    // Hand the table to scikit-learn, train, predict, and return actual vs
    // predicted labels for the held-out split.
    let value = rt.with_python(&dataset, |py, obj| {
        let scope = PyDict::new(py);
        scope.set_item("table", obj)?;
        py.run(
            cr#"
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = table.to_polars()
features = ["x0", "x1", "x2", "x3"]
X = df.select(features).to_numpy()
y = df["label"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)
predicted = model.predict(X_test)

result = pl.DataFrame(
    {
        "actual": y_test.astype("int64"),
        "predicted": predicted.astype("int64"),
    }
)
"#,
            Some(&scope),
            Some(&scope),
        )?;
        scope
            .get_item("result")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("result not set"))
    })?;

    let Value::Table(table) = value else {
        panic!("expected a table, got {value:?}");
    };
    let actual = table.cols[0].array.num().i64();
    let predicted = table.cols[1].array.num().i64();

    let rows = actual.len();
    let correct = (0..rows).filter(|&i| actual.data[i] == predicted.data[i]).count();
    println!("Scored the predictions back in Rust:");
    println!("  test rows : {rows}");
    println!("  correct   : {correct}");
    println!("  accuracy  : {:.1}%", correct as f64 / rows as f64 * 100.0);

    Ok(())
}

/// Builds a labelled dataset whose label is a clear linear function of three
/// features plus mild noise, using a deterministic xorshift generator.
fn build_dataset(rows: usize) -> Table {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let draw = |state: &mut u64| -> f64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };

    let mut x0 = Vec64::with_capacity(rows);
    let mut x1 = Vec64::with_capacity(rows);
    let mut x2 = Vec64::with_capacity(rows);
    let mut x3 = Vec64::with_capacity(rows);
    let mut label = Vec64::with_capacity(rows);

    for _ in 0..rows {
        let a = draw(&mut state);
        let b = draw(&mut state);
        let c = draw(&mut state);
        let d = draw(&mut state);
        let noise = draw(&mut state) * 0.5;
        let score = 2.0 * a - 1.5 * b + c + noise;
        x0.push(a);
        x1.push(b);
        x2.push(c);
        x3.push(d);
        label.push(if score > 0.0 { 1i64 } else { 0 });
    }

    Table::new(
        "dataset".to_string(),
        Some(vec![
            fa_f64!("x0", @vec64 x0),
            fa_f64!("x1", @vec64 x1),
            fa_f64!("x2", @vec64 x2),
            fa_f64!("x3", @vec64 x3),
            fa_i64!("label", @vec64 label),
        ]),
    )
}
