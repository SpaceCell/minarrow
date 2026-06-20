// Copyright 2025-2026 Peter Garfield Bower. All Rights Reserved.
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

//! ---------------------------------------------------------
//! Runs sum benchmark on Minarrow with Rayon (Multi-core processing) and SIMD
//!
//! Run with:
//!     RUSTFLAGS="-C target-cpu=native" cargo bench --bench benchmark_parallel_simd --features parallel_proc
//!
//! The *RUSTFLAGS* argument ensures it compiles to your host instruction-set.
//!
//! Use 2, 4, 8, or 16 SIMD_LANES as per your processor's SIMD support.
//! ---------------------------------------------------------

#![feature(portable_simd)]

#[cfg(feature = "parallel_proc")]
mod parallel_simd {
    use std::hint::black_box;
    use std::simd::Simd;
    use std::simd::num::{SimdFloat, SimdInt};
    use std::time::Instant;

    use minarrow::{Buffer, Vec64};
    use rayon::iter::ParallelIterator;
    use rayon::slice::ParallelSlice;

    const N: usize = 1_000_000_000;
    const SIMD_LANES: usize = 4;

    // SIMD chunk sum for i64
    #[inline(always)]
    fn simd_sum_i64<const LANES: usize>(data: &[i64]) -> i64 {
        let n = data.len();
        let simd_width = LANES;
        let simd_chunks = n / simd_width;

        let mut acc_simd = Simd::<i64, LANES>::splat(0);
        for i in 0..simd_chunks {
            let v = Simd::<i64, LANES>::from_slice(&data[i * simd_width..][..simd_width]);
            acc_simd += v;
        }
        let mut result = acc_simd.reduce_sum();
        for i in (simd_chunks * simd_width)..n {
            result += data[i];
        }
        result
    }

    // SIMD chunk sum for f64
    #[inline(always)]
    fn simd_sum_f64<const LANES: usize>(data: &[f64]) -> f64 {
        let n = data.len();
        let simd_width = LANES;
        let simd_chunks = n / simd_width;

        let mut acc_simd = Simd::<f64, LANES>::splat(0.0);
        for i in 0..simd_chunks {
            let v = Simd::<f64, LANES>::from_slice(&data[i * simd_width..][..simd_width]);
            acc_simd += v;
        }
        let mut result = acc_simd.reduce_sum();
        for i in (simd_chunks * simd_width)..n {
            result += data[i];
        }
        result
    }

    // Rayon + SIMD for i64
    fn rayon_simd_sum_i64(buffer: &Buffer<i64>) -> i64 {
        let slice = buffer.as_slice();
        let chunk_size = 1 << 20; // 1M per chunk, tune if desired
        slice
            .par_chunks(chunk_size)
            .map(|chunk| simd_sum_i64::<SIMD_LANES>(chunk))
            .sum()
    }

    // Rayon + SIMD for f64
    fn rayon_simd_sum_f64(buffer: &Buffer<f64>) -> f64 {
        let slice = buffer.as_slice();
        let chunk_size = 1 << 20; // 1M per chunk, tune if desired
        slice
            .par_chunks(chunk_size)
            .map(|chunk| simd_sum_f64::<SIMD_LANES>(chunk))
            .sum()
    }
    pub fn run_benchmark() {
        println!("--- SIMD + Rayon Benchmark, N = {} ---", N);

        // IntegerArray<i64>
        let data: Vec64<i64> = (0..N as i64).collect();
        let buffer = Buffer::from(data);

        let start = Instant::now();
        let sum = black_box(rayon_simd_sum_i64(&buffer));
        let dur = start.elapsed();
        println!(
            "SIMD + Rayon IntegerArray<i64>: sum = {}, time = {:?}",
            sum, dur
        );

        // FloatArray<f64>
        let data: Vec64<f64> = (0..N as i64).map(|x| x as f64).collect();
        let buffer = Buffer::from(data);

        let start = Instant::now();
        let sum = black_box(rayon_simd_sum_f64(&buffer));
        let dur = start.elapsed();
        println!(
            "SIMD + Rayon FloatArray<f64>: sum = {}, time = {:?}",
            sum, dur
        );
    }
}

fn main() {
    if cfg!(feature = "parallel_proc") {
        #[cfg(feature = "parallel_proc")]
        parallel_simd::run_benchmark()
    } else {
        println!("The parallel_simd example requires enabling the `parallel_proc` feature.")
    }
}
