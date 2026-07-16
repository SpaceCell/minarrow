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

use minarrow::traits::selection::RowSelection;
use minarrow::{Concatenate, NdArray, nd};

fn main() {
    // Flat input follows NdArray's column-major layout.
    let a = NdArray::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        &[4, 3],
    );

    println!("=== Shape and access ===\n");
    println!("shape: {:?}", a.shape());
    println!("strides: {:?}", a.strides());
    println!("n_obs: {}", a.n_obs());
    println!("a.get(&[2, 1]) = {}\n", a.get(&[2, 1]));

    // Convenience constructors create ordinary NdArray containers.
    println!("=== Constructors ===\n");
    let steps = NdArray::<f64>::linspace(0.0, 1.0, 5);
    println!("linspace(0.0, 1.0, 5): {:?}", steps.as_slice());
    let identity = NdArray::<f64>::eye(3);
    println!("eye(3) diagonal: {:?}\n", (0..3).map(|i| identity.get(&[i, i])).collect::<Vec<_>>());

    // Views change shape, offsets, or strides without copying the data.
    println!("=== Views ===\n");
    println!("a.obs(1) - one observation as a 1D view");
    let row = a.obs(1);
    println!("  shape {:?}, values {:?}\n", row.shape(), (0..3).map(|j| row.get(&[j])).collect::<Vec<_>>());

    println!("a.slice(nd![1..3, 0..2]) - range on both axes");
    let window = a.slice(nd![1..3, 0..2]);
    println!("  shape {:?}, window[0, 0] = {}\n", window.shape(), window.get(&[0, 0]));

    println!("a.r(0..2) - axis-0 selection");
    let head = a.r(0..2);
    println!("  shape {:?}\n", head.shape());

    println!("view.transpose() - zero-copy stride swap");
    let t = a.as_view().transpose();
    println!("  shape {:?}, t.get(&[1, 2]) = {}\n", t.shape(), t.get(&[1, 2]));

    // The caller supplies the operation; NdArray supplies traversal.
    println!("=== Apply ===\n");
    let doubled = a.apply(|v| v * 2.0);
    println!("a.apply(|v| v * 2.0): doubled[3, 2] = {}\n", doubled.get(&[3, 2]));

    // Concatenation extends axis 0 while retaining the trailing shape.
    println!("=== Concatenate ===\n");
    let b = NdArray::from_slice(&[13.0, 14.0, 15.0], &[1, 3]);
    let joined = a.clone().concat(b).unwrap();
    println!("a.concat(b): shape {:?}, last row starts with {}\n", joined.shape(), joined.get(&[4, 0]));

    // A 2D conversion produces one Table column per axis-1 entry.
    println!("=== To Table ===\n");
    let table = a.clone().to_table(None).unwrap();
    println!("{}", table);
}
