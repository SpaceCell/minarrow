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

//! ---------------------------------------------------------
//! Runs sum benchmark comparisons on `Minarrow` and `Arrow-Rs`,
//! at various layers of library abstraction, using SIMD:
//!
//!     1. Raw Vec / Vec64
//!     2. Typed "inner" arrays
//!     3. Top-level unified `Array` type
//!
//! Run with:
//!     RUSTFLAGS="-C target-cpu=native" cargo bench --bench hotloop_benchmark_simd --features cast_arrow
//!
//! The *RUSTFLAGS* argument ensures it compiles to your host instruction-set.
//!
//! Use 2, 4, 8, or 16 LANES as per your processor's SIMD support.
//! ---------------------------------------------------------

#![feature(portable_simd)]

#[cfg(feature = "cast_arrow")]
use crate::benchmarks_simd::run_benchmark;

pub(crate) const N: usize = 1_000;
pub(crate) const SIMD_LANES: usize = 4;

#[cfg(feature = "cast_arrow")]
mod benchmarks_simd {

    use std::hint::black_box;
    use std::simd::Simd;
    use std::sync::Arc;
    use std::time::Instant;

    use crate::SIMD_LANES;

    use arrow::array::{
        Array as ArrowArrayTrait, ArrayRef, Float64Array as ArrowF64Array,
        Int64Array as ArrowI64Array,
    };
    use minarrow::{Array, Buffer, FloatArray, IntegerArray, NumericArray, Vec64};

    #[inline(always)]
    fn simd_sum_i64<const LANES: usize>(data: &[i64]) -> i64 {
        let n = data.len();
        let simd_width = LANES;
        let simd_chunks = n / simd_width;

        let mut acc_simd: Simd<i64, LANES>;

        unsafe {
            let data_ptr = data.as_ptr();
            let mut acc1 = Simd::<i64, LANES>::splat(0);
            let mut acc2 = Simd::<i64, LANES>::splat(0);
            let mut acc3 = Simd::<i64, LANES>::splat(0);
            let mut acc4 = Simd::<i64, LANES>::splat(0);

            let unroll_factor = 4;
            let unrolled_chunks = simd_chunks / unroll_factor;

            for i in 0..unrolled_chunks {
                let base_offset = i * unroll_factor * simd_width;
                let v1 =
                    std::ptr::read_unaligned(data_ptr.add(base_offset) as *const Simd<i64, LANES>);
                let v2 = std::ptr::read_unaligned(
                    data_ptr.add(base_offset + simd_width) as *const Simd<i64, LANES>
                );
                let v3 = std::ptr::read_unaligned(
                    data_ptr.add(base_offset + 2 * simd_width) as *const Simd<i64, LANES>
                );
                let v4 = std::ptr::read_unaligned(
                    data_ptr.add(base_offset + 3 * simd_width) as *const Simd<i64, LANES>
                );
                acc1 += v1;
                acc2 += v2;
                acc3 += v3;
                acc4 += v4;
            }

            acc_simd = acc1 + acc2 + acc3 + acc4;

            let processed = unrolled_chunks * unroll_factor;
            for i in processed..simd_chunks {
                let offset = i * simd_width;
                let v = std::ptr::read_unaligned(data_ptr.add(offset) as *const Simd<i64, LANES>);
                acc_simd += v;
            }
        }

        // Horizontal sum
        let mut result = 0i64;
        for i in 0..LANES {
            result += acc_simd[i];
        }

        let remainder_start = simd_chunks * simd_width;
        for i in remainder_start..n {
            result += data[i];
        }

        result
    }

    #[inline(always)]
    fn simd_sum_f64<const LANES: usize>(data: &[f64]) -> f64 {
        let n = data.len();
        let simd_width = LANES;
        let simd_chunks = n / simd_width;

        let mut acc_simd: Simd<f64, LANES>;

        unsafe {
            let data_ptr = data.as_ptr();
            let mut acc1 = Simd::<f64, LANES>::splat(0.0);
            let mut acc2 = Simd::<f64, LANES>::splat(0.0);
            let mut acc3 = Simd::<f64, LANES>::splat(0.0);
            let mut acc4 = Simd::<f64, LANES>::splat(0.0);

            let unroll_factor = 4;
            let unrolled_chunks = simd_chunks / unroll_factor;

            for i in 0..unrolled_chunks {
                let base_offset = i * unroll_factor * simd_width;
                let v1 =
                    std::ptr::read_unaligned(data_ptr.add(base_offset) as *const Simd<f64, LANES>);
                let v2 = std::ptr::read_unaligned(
                    data_ptr.add(base_offset + simd_width) as *const Simd<f64, LANES>
                );
                let v3 = std::ptr::read_unaligned(
                    data_ptr.add(base_offset + 2 * simd_width) as *const Simd<f64, LANES>
                );
                let v4 = std::ptr::read_unaligned(
                    data_ptr.add(base_offset + 3 * simd_width) as *const Simd<f64, LANES>
                );
                acc1 += v1;
                acc2 += v2;
                acc3 += v3;
                acc4 += v4;
            }

            acc_simd = acc1 + acc2 + acc3 + acc4;

            let processed = unrolled_chunks * unroll_factor;
            for i in processed..simd_chunks {
                let offset = i * simd_width;
                let v = std::ptr::read_unaligned(data_ptr.add(offset) as *const Simd<f64, LANES>);
                acc_simd += v;
            }
        }

        let mut result = 0.0;
        for i in 0..LANES {
            result += acc_simd[i];
        }

        let remainder_start = simd_chunks * simd_width;
        for i in remainder_start..n {
            result += data[i];
        }

        result
    }

    fn simd_sum_f64_runtime(data: &[f64], lanes: usize) -> f64 {
        match lanes {
            2 => simd_sum_f64::<2>(data),
            4 => simd_sum_f64::<4>(data),
            8 => simd_sum_f64::<8>(data),
            16 => simd_sum_f64::<16>(data),
            _ => panic!("Unsupported SIMD lanes. Only 2, 4, 8, 16 supported."),
        }
    }

    fn simd_sum_i64_runtime(data: &[i64], lanes: usize) -> i64 {
        match lanes {
            2 => simd_sum_i64::<2>(data),
            4 => simd_sum_i64::<4>(data),
            8 => simd_sum_i64::<8>(data),
            16 => simd_sum_i64::<16>(data),
            _ => panic!("Unsupported SIMD lanes. Only 2, 4, 8, 16 supported."),
        }
    }

    pub fn run_benchmark(n: usize, simd_lanes: usize) {
        // ----------- Integer (i64) tests -----------

        let data: Vec<i64> = (0..n as i64).collect();
        black_box(simd_sum_i64_runtime(&data[..], simd_lanes)); // warmup, ignore result

        println!("|------------ Integer Tests ------------ |\n");
        // Raw Vec<i64>
        // Sometimes this will randomly align, other times it will not.
        let data: Vec<i64> = (0..n as i64).collect();
        let start = Instant::now();
        let slice = &data[..];
        let sum = simd_sum_i64_runtime(slice, simd_lanes);
        let dur_vec = start.elapsed();
        println!("raw vec: Vec<i64> sum = {}, {:?}", sum, dur_vec);
        let v_aligned =
            (&data[0] as *const i64 as usize) % std::mem::align_of::<Simd<i64, SIMD_LANES>>() == 0;
        black_box(sum);

        // Raw Vec64<i64>
        let data: Vec64<i64> = (0..n as i64).collect();
        let start = Instant::now();
        let slice = &data[..];
        let sum = simd_sum_i64_runtime(slice, simd_lanes);
        let dur_vec = start.elapsed();
        println!("raw vec64: Vec64<i64> sum = {}, {:?}", sum, dur_vec);
        let v64_aligned =
            (&data[0] as *const i64 as usize) % std::mem::align_of::<Simd<i64, SIMD_LANES>>() == 0;
        black_box(sum);

        // Minarrow i64 (direct struct)
        let data: Vec64<i64> = (0..n as i64).collect();
        let data_copy = data.clone();

        let start = Instant::now();
        let int_arr = IntegerArray {
            data: Buffer::from(data),
            null_mask: None,
        };
        let slice = int_arr.as_ref();
        let sum = simd_sum_i64_runtime(slice, simd_lanes);
        let dur_minarrow_direct = start.elapsed();
        println!(
            "minarrow direct: IntegerArray sum = {}, {:?}",
            sum, dur_minarrow_direct
        );
        let int_array_aligned = (&data_copy[0] as *const i64 as usize)
            % std::mem::align_of::<Simd<i64, SIMD_LANES>>()
            == 0;
        black_box(sum);

        // Arrow i64 (struct direct)
        let data: Vec<i64> = (0..n as i64).collect();
        let data_copy = data.clone();

        let start = Instant::now();
        let arr = ArrowI64Array::from(data);
        let slice = arr.values();
        let sum = simd_sum_i64_runtime(slice, simd_lanes);
        let dur_arrow_struct = start.elapsed();
        println!(
            "arrow-rs struct: Int64Array sum = {}, {:?}",
            sum, dur_arrow_struct
        );
        let i64_arrow_aligned = (&data_copy[0] as *const i64 as usize)
            % std::mem::align_of::<Simd<i64, SIMD_LANES>>()
            == 0;
        black_box(sum);

        // Minarrow i64 (enum)
        let data: Vec64<i64> = (0..n as i64).collect();
        let data_copy = data.clone();

        let start = Instant::now();
        let array = Array::NumericArray(NumericArray::Int64(Arc::new(IntegerArray {
            data: Buffer::from(data),
            null_mask: None,
        })));
        let int_arr = array.num().i64();
        let slice = int_arr.as_ref();
        let sum = simd_sum_i64_runtime(slice, simd_lanes);
        let dur_minarrow_enum = start.elapsed();
        println!(
            "minarrow enum: IntegerArray sum = {}, {:?}",
            sum, dur_minarrow_enum
        );
        let arr_int_enum_aligned = (&data_copy[0] as *const i64 as usize)
            % std::mem::align_of::<Simd<i64, SIMD_LANES>>()
            == 0;
        black_box(sum);

        // Arrow i64 (dynamic)
        let data: Vec<i64> = (0..n as i64).collect();
        let data_copy = data.clone();

        let start = Instant::now();
        let arr: ArrayRef = Arc::new(ArrowI64Array::from(data));
        let slice = if let Some(f) = arr.as_any().downcast_ref::<ArrowI64Array>() {
            f.values()
        } else {
            panic!("downcast failed")
        };
        let sum = simd_sum_i64_runtime(slice, simd_lanes);
        let dur_arrow_dyn_i64 = start.elapsed();
        println!(
            "arrow-rs dyn: Int64Array sum = {}, {:?}",
            sum, dur_arrow_dyn_i64
        );
        let array_ref_int_aligned = (&data_copy[0] as *const i64 as usize)
            % std::mem::align_of::<Simd<i64, SIMD_LANES>>()
            == 0;
        black_box(sum);
        println!("\n");

        // ----------- Float (f64) tests ----------------

        println!("|------------ Float Tests ------------ |\n");

        // Raw Vec<f64>
        // Sometimes this will randomly align, other times it will not.
        let data: Vec<f64> = (0..n as i64).map(|x| x as f64).collect();
        let start = Instant::now();
        let sum = simd_sum_f64_runtime(&data[..], simd_lanes);
        let dur_vec_f64 = start.elapsed();
        println!("raw vec: Vec<f64> sum = {}, {:?}", sum, dur_vec_f64);
        let v_float_aligned =
            (&data[0] as *const f64 as usize) % std::mem::align_of::<Simd<f64, SIMD_LANES>>() == 0;

        black_box(sum);

        // Raw Vec64<f64>
        let data: Vec64<f64> = (0..n as i64).map(|x| x as f64).collect();
        let start = Instant::now();
        let sum = simd_sum_f64_runtime(&data[..], simd_lanes);
        let dur_vec_f64 = start.elapsed();
        println!("raw vec64: Vec64<f64> sum = {}, {:?}", sum, dur_vec_f64);
        let v64_float_aligned =
            (&data[0] as *const f64 as usize) % std::mem::align_of::<Simd<f64, SIMD_LANES>>() == 0;

        black_box(sum);

        // Minarrow f64 (direct struct, SIMD)
        let data: Vec64<f64> = (0..n as i64).map(|x| x as f64).collect();
        let data_copy = data.clone();

        let start = Instant::now();
        let float_arr = FloatArray {
            data: Buffer::from(data),
            null_mask: None,
        };
        let sum = simd_sum_f64_runtime(float_arr.as_ref(), simd_lanes);
        let dur_minarrow_direct_f64 = start.elapsed();
        println!(
            "minarrow direct: FloatArray sum = {}, {:?}",
            sum, dur_minarrow_direct_f64
        );
        let float_arr_aligned = (&data_copy[0] as *const f64 as usize)
            % std::mem::align_of::<Simd<f64, SIMD_LANES>>()
            == 0;
        black_box(sum);

        // Arrow f64 (struct direct)
        let data: Vec<f64> = (0..n as i64).map(|x| x as f64).collect();
        let data_copy = data.clone();

        let start = Instant::now();
        let arr = ArrowF64Array::from(data);
        let sum = simd_sum_f64_runtime(arr.values(), simd_lanes);
        let dur_arrow_struct_f64 = start.elapsed();
        println!(
            "arrow-rs struct: Float64Array sum = {}, {:?}",
            sum, dur_arrow_struct_f64
        );
        let arrow_f64_aligned = (&data_copy[0] as *const f64 as usize)
            % std::mem::align_of::<Simd<f64, SIMD_LANES>>()
            == 0;
        black_box(sum);

        // Minarrow f64 (enum)
        let data: Vec64<f64> = (0..n as i64).map(|x| x as f64).collect();
        let data_copy = data.clone();

        let start = Instant::now();
        let array = Array::NumericArray(NumericArray::Float64(Arc::new(FloatArray {
            data: Buffer::from(data),
            null_mask: None,
        })));
        let float_arr = array.num().f64();
        let sum = simd_sum_f64_runtime(float_arr.as_ref(), simd_lanes);
        let dur_minarrow_enum_f64 = start.elapsed();
        println!(
            "minarrow enum: FloatArray sum = {}, {:?}",
            sum, dur_minarrow_enum_f64
        );
        let float_enum_aligned = (&data_copy[0] as *const f64 as usize)
            % std::mem::align_of::<Simd<f64, SIMD_LANES>>()
            == 0;
        black_box(sum);

        // Arrow f64 (dynamic)
        let data: Vec<f64> = (0..n as i64).map(|x| x as f64).collect();
        let data_copy = data.clone();

        let start = Instant::now();
        let arr: ArrayRef = Arc::new(ArrowF64Array::from(data));
        let slice = if let Some(f) = arr.as_any().downcast_ref::<ArrowF64Array>() {
            f.values()
        } else {
            panic!("downcast failed")
        };
        let sum = simd_sum_f64_runtime(slice, simd_lanes);
        let dur_arrow_dyn_f64 = start.elapsed();
        println!(
            "arrow-rs dyn: Float64Array sum = {}, {:?}",
            sum, dur_arrow_dyn_f64
        );
        let arrow_f64_arr_aligned = (&data_copy[0] as *const f64 as usize)
            % std::mem::align_of::<Simd<f64, SIMD_LANES>>()
            == 0;
        black_box(sum);
        println!("\n");
        println!("Verify SIMD pointer alignment for Integer calculations (based on lane width):");
        println!("Vec<i64> is aligned: {}", v_aligned);
        println!("Minarrow Vec64<i64> is aligned: {}", v64_aligned);
        println!(
            "Minarrow IntegerArray<i64> is aligned: {}",
            int_array_aligned
        );
        println!("Arrow ArrowI64Array is aligned: {}", i64_arrow_aligned);
        println!(
            "Minarrow Array::NumericArray<i64> is aligned: {}",
            arr_int_enum_aligned
        );
        println!("Arrow ArrayRef<int> is aligned: {}", array_ref_int_aligned);
        println!("\n");
        println!("Verify SIMD pointer alignment for Float calculations (based on lane width):");
        println!("Vec<f64> is aligned: {}", v_float_aligned);
        println!("Vec64<f64> is aligned: {}", v64_float_aligned);
        println!("FloatArray<f64> is aligned: {}", float_arr_aligned);
        println!("ArrowF64Array is aligned: {}", arrow_f64_aligned);
        println!(
            "Array::NumericArray<f64> is aligned: {}",
            float_enum_aligned
        );
        println!("ArrayRef is aligned: {}", arrow_f64_arr_aligned);
        println!("\n");

        println!("---------------------- END OF SIMD BENCHMARKS ---------------------------");
    }
}

fn main() {
    if cfg!(feature = "cast_arrow") {
        use crate::N;

        println!(
            "Running SIMD/Arrow/minarrow parity benchmarks (n={}, lanes={})",
            N, SIMD_LANES
        );
        #[cfg(feature = "cast_arrow")]
        run_benchmark(N, SIMD_LANES);
    } else {
        println!("The apache-FFI example requires enabling the `cast_arrow` feature.")
    }
}
