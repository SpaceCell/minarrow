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

use minarrow::{NdArray, XArray, arr_f64, arr_str32};

fn main() {
    // Hourly readings for three instruments: 4 observations x 3 features
    let data = NdArray::from_slice(
        &[
            20.1, 20.4, 20.9, 21.3, // temp
            1.01, 1.02, 1.00, 0.99, // pressure
            55.0, 54.0, 52.0, 51.0, // humidity
        ],
        &[4, 3],
    );
    let mut xa = XArray::new(data, &["hour", "instrument"]);
    xa.assign_coords("hour", arr_f64![0.0, 1.0, 2.0, 3.0]);

    println!("=== Dims and coords ===\n");
    println!("{:?}", xa);
    println!("dim_names: {:?}", xa.dim_names());
    println!("shape: {:?}\n", xa.shape());

    // Coordinate-value selection
    println!("=== Coordinate selection ===\n");

    println!("xa.at(\"hour\", 2.0) - collapse to one observation");
    let at_two = xa.at("hour", 2.0);
    println!("  shape {:?}, temp = {}\n", at_two.shape(), at_two.get(&[0]));

    println!("xa.between(\"hour\", 1.0, 3.0) - inclusive window");
    let window = xa.between("hour", 1.0, 3.0);
    println!("  shape {:?}, first temp = {}\n", window.shape(), window.get(&[0, 0]));

    println!("xa.nearest(\"hour\", 1.8) - closest coordinate");
    let near = xa.nearest("hour", 1.8);
    println!("  shape {:?}, temp = {}\n", near.shape(), near.get(&[0]));

    // Positional selection by axis name
    println!("=== Named positional selection ===\n");
    let first_two = xa.select(&[("hour", &(0..2))]);
    println!("xa.select(&[(\"hour\", &(0..2))]): shape {:?}\n", first_two.shape());

    // Transpose reorders data and labels together
    println!("=== Transpose ===\n");
    let t = xa.transpose(&["instrument", "hour"]).unwrap();
    println!("dims after transpose: {:?}, shape {:?}\n", t.dim_names(), t.shape());

    // Convert to a Table, with axis-1 labels becoming column names
    println!("=== To Table ===\n");
    let mut named = xa.clone();
    named.assign_coords("instrument", arr_str32!(&["temp", "pressure", "humidity"]));
    let table = named.to_table().unwrap();
    println!("{}", table);
}
