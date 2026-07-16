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

//! Round-trips a tensor through PyTorch over DLPack.
//!
//! An `NdArray` of features is built in Rust and passed into Python, where
//! `torch.from_dlpack` picks up the buffer zero-copy. The closure
//! standardises each feature column, scores every row against a weight
//! vector, and returns the score tensor. That tensor crosses back over the
//! DLPack capsule protocol and Rust reads the results from Minarrow's own
//! 64-byte aligned memory.
//!
//! ## Running
//! Needs a venv with `torch`. `PYO3_PYTHON` must be an absolute path to it.
//! ```bash
//! cd minarrow-py
//! PYO3_PYTHON=$PWD/../pyo3/.venv/bin/python \
//!   PYTHONHOME=/usr \
//!   PYTHONPATH=$PWD/../pyo3/.venv/lib/python3.12/site-packages \
//!   LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
//!   cargo run --release --example roundtrip_tensor --features embed,ndarray
//! ```

use minarrow::{NdArray, Value, Vec64};
use minarrow_py::PyLiquid;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn main() -> PyResult<()> {
    let rt = PyLiquid::start().preimport(["torch"])?;

    let features = build_features(1024, 8);
    println!(
        "Built a {:?} tensor in Rust at {:p}.",
        features.shape(),
        features.as_slice().as_ptr()
    );

    // Hand the tensor to PyTorch, standardise the feature columns, and
    // score every row against a fixed weight vector.
    let value = rt.with_python(&features, |py, obj| {
        let scope = PyDict::new(py);
        scope.set_item("tensor", obj)?;
        py.run(
            cr#"
import torch

x = torch.from_dlpack(tensor)
mean = x.mean(dim=0, keepdim=True)
std = x.std(dim=0, keepdim=True)
z = (x - mean) / std
weights = torch.linspace(1.0, 2.0, x.shape[1], dtype=torch.float64)
result = (z * weights).sum(dim=1)
"#,
            Some(&scope),
            Some(&scope),
        )?;
        scope
            .get_item("result")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("result not set"))
    })?;

    let Value::NdArray(scores) = value else {
        panic!("expected a tensor, got {value:?}");
    };
    println!("Scored the rows in PyTorch and read them back in Rust:");
    println!("  shape       : {:?}", scores.shape());
    println!("  first three : {:?}", &scores.as_slice()[..3]);
    println!(
        "  mean score  : {:.4}",
        scores.as_slice().iter().sum::<f64>() / scores.len() as f64
    );

    Ok(())
}

/// Builds a feature tensor from a deterministic xorshift generator, laid
/// out column-major as `[rows, cols]`.
fn build_features(rows: usize, cols: usize) -> NdArray<f64> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut draw = move || -> f64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };

    let mut flat = Vec64::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        flat.push(draw());
    }
    NdArray::from_vec64(flat, &[rows, cols])
}
