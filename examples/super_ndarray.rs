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

use minarrow::structs::chunked::super_array::RechunkStrategy;
use minarrow::{Consolidate, NdArray, SuperNdArray};

fn main() {
    // Simulate three sensor batches arriving with different observation
    // counts. Every batch has two measurements per observation.
    let mut snd = SuperNdArray::new("sensor_frames");
    snd.push(NdArray::from_slice(&[1.0, 2.0, 10.0, 20.0], &[2, 2]));
    snd.push(NdArray::from_slice(&[3.0, 4.0, 5.0, 30.0, 40.0, 50.0], &[3, 2]));
    snd.push(NdArray::from_slice(&[6.0, 60.0], &[1, 2]));

    println!("=== Shape across batches ===\n");
    println!("n_batches: {}", snd.n_batches());
    println!("n_obs: {}", snd.n_obs());
    println!("shape: {:?}\n", snd.shape());

    // Global indices address the combined observation range across batches.
    println!("=== Global access ===\n");
    println!("snd.get(&[0, 0]) = {} (batch 0)", snd.get(&[0, 0]));
    println!("snd.get(&[3, 1]) = {} (batch 1)", snd.get(&[3, 1]));
    println!("snd.get(&[5, 1]) = {} (batch 2)\n", snd.get(&[5, 1]));

    // The requested observation window starts in one batch and ends in the next.
    println!("=== Batch-spanning window ===\n");
    let window = snd.slice(1, 3);
    println!("snd.slice(1, 3): n_obs {}, spans {} slices", window.n_obs(), window.n_slices());
    println!("window.get(&[1, 0]) = {}\n", window.get(&[1, 0]));

    // Consolidate only when a consumer requires one contiguous allocation.
    println!("=== Consolidate ===\n");
    let flat = snd.clone().consolidate();
    println!("consolidated shape: {:?}", flat.shape());
    println!("flat.get(&[5, 1]) = {}\n", flat.get(&[5, 1]));

    // Change the batch boundaries without changing the logical values.
    println!("=== Rechunk ===\n");
    let mut even = snd.clone();
    even.rechunk(RechunkStrategy::Count(2)).unwrap();
    println!("rechunk(Count(2)): n_batches {}, first batch obs {}", even.n_batches(), even.batch(0).unwrap().shape()[0]);

    // Batch boundaries do not affect logical equality.
    println!("\n=== Logical equality ===\n");
    let single = SuperNdArray::from_batches(vec![flat], "sensor_frames");
    println!("snd == consolidated single batch: {}", snd == single);
}
