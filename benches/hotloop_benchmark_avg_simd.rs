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
//! Runs averaged sum benchmark comparisons on `Minarrow` and `Arrow-Rs`,
//! at various layers of library abstraction:
//!
//!     1. Raw Vec / Vec64
//!     2. Typed "inner" arrays
//!     3. Top-level unified `Array` type
//!
//! Run with:
//!     RUSTFLAGS="-C target-cpu=native" cargo bench --bench hotloop_benchmark_avg_simd --features cast_arrow
//! ---------------------------------------------------------

#![feature(portable_simd)]

#[cfg(feature = "cast_arrow")]
use crate::avg_simd::run_benchmark;

pub(crate) const N: usize = 1000000;
pub(crate) const SIMD_LANES: usize = 4;
pub(crate) const ITERATIONS: usize = 1000;

#[cfg(feature = "cast_arrow")]
mod avg_simd {
    use std::hint::black_box;
    use std::simd::Simd;
    use std::sync::Arc;
    use std::time::Instant;

    use crate::ITERATIONS;
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
        let mut total_vec = std::time::Duration::ZERO;
        let mut total_vec64 = std::time::Duration::ZERO;
        let mut total_minarrow_direct = std::time::Duration::ZERO;
        let mut total_arrow_struct = std::time::Duration::ZERO;
        let mut total_minarrow_enum = std::time::Duration::ZERO;
        let mut total_arrow_dyn = std::time::Duration::ZERO;

        let mut total_vec_f64 = std::time::Duration::ZERO;
        let mut total_vec64_f64 = std::time::Duration::ZERO;
        let mut total_minarrow_direct_f64 = std::time::Duration::ZERO;
        let mut total_arrow_struct_f64 = std::time::Duration::ZERO;
        let mut total_minarrow_enum_f64 = std::time::Duration::ZERO;
        let mut total_arrow_dyn_f64 = std::time::Duration::ZERO;

        // Data construction - This is the only part we
        // exclude from the overall benchmark, however, we time Vec
        // vs. Vec64 here as an indicative profile, given this is the
        // starting setup of all other reference points.
        let mut sum_vec_i64 = 0u128;
        let mut sum_vec64_i64 = 0u128;

        // for keeping scope alive
        // after the Vec benchmarks, we keep the last one each
        let mut v_int_data = Vec::with_capacity(n);
        let mut v64_int_data = Vec64::with_capacity(n);

        for _ in 0..ITERATIONS {
            let t0 = Instant::now();
            v_int_data = (0..n as i64).collect();
            let dur_vec_i64 = t0.elapsed();

            let t1 = Instant::now();
            v64_int_data = (0..n as i64).collect();
            let dur_vec64_i64 = t1.elapsed();

            sum_vec_i64 += dur_vec_i64.as_nanos();
            sum_vec64_i64 += dur_vec64_i64.as_nanos();
        }

        let avg_vec_i64 = sum_vec_i64 as f64 / ITERATIONS as f64;
        let avg_vec64_i64 = sum_vec64_i64 as f64 / ITERATIONS as f64;

        println!(
            "Vec<i64> construction (avg):    {}",
            fmt_duration_ns(avg_vec_i64)
        );
        println!(
            "Vec64<i64> construction (avg):  {}",
            fmt_duration_ns(avg_vec64_i64)
        );
        println!("\n=> Keep the above Vec construction delta in mind when interpreting the below results,
    as it is not included in the benchmarks that follow.\n");

        // Alignment checks - once, outside timing

        let v_aligned = {
            (&v_int_data[0] as *const i64 as usize) % std::mem::align_of::<Simd<i64, SIMD_LANES>>()
                == 0
        };

        let v64_aligned = {
            (&v64_int_data[0] as *const i64 as usize)
                % std::mem::align_of::<Simd<i64, SIMD_LANES>>()
                == 0
        };

        let int_array_aligned = {
            let int_arr = IntegerArray {
                data: Buffer::from(v64_int_data.clone()),
                null_mask: None,
            };
            let slice = int_arr.as_ref();
            (slice.as_ptr() as usize) % std::mem::align_of::<Simd<i64, SIMD_LANES>>() == 0
        };

        let i64_arrow_aligned = {
            let arr = ArrowI64Array::from(v_int_data.clone());
            (arr.values().as_ptr() as usize) % std::mem::align_of::<Simd<i64, SIMD_LANES>>() == 0
        };

        let arr_int_enum_aligned = {
            let array = Array::NumericArray(NumericArray::Int64(Arc::new(IntegerArray {
                data: Buffer::from(v64_int_data.clone()),
                null_mask: None,
            })));
            let int_arr = array.num().i64();
            (int_arr.data.as_slice().as_ptr() as usize)
                % std::mem::align_of::<Simd<i64, SIMD_LANES>>()
                == 0
        };

        let array_ref_int_aligned = {
            let arr: ArrayRef = Arc::new(ArrowI64Array::from(v_int_data.clone()));
            let int_arr = arr.as_any().downcast_ref::<ArrowI64Array>().unwrap();
            (int_arr.values().as_ptr() as usize) % std::mem::align_of::<Simd<i64, SIMD_LANES>>()
                == 0
        };

        let v_float_data: Vec<f64> = (0..n as i64).map(|x| x as f64).collect();
        let v64_float_data: Vec64<f64> = (0..n as i64).map(|x| x as f64).collect();

        let v_float_aligned = {
            (&v_float_data[0] as *const f64 as usize)
                % std::mem::align_of::<Simd<f64, SIMD_LANES>>()
                == 0
        };

        let v64_float_aligned = {
            (&v64_float_data[0] as *const f64 as usize)
                % std::mem::align_of::<Simd<f64, SIMD_LANES>>()
                == 0
        };

        let float_arr_aligned = {
            let float_arr = FloatArray {
                data: Buffer::from(v64_float_data.clone()),
                null_mask: None,
            };
            (&float_arr.data.as_slice()[0] as *const f64 as usize)
                % std::mem::align_of::<Simd<f64, SIMD_LANES>>()
                == 0
        };

        let arrow_f64_aligned = {
            let arr = ArrowF64Array::from(v_float_data.clone());
            (arr.values().as_ptr() as usize) % std::mem::align_of::<Simd<f64, SIMD_LANES>>() == 0
        };

        let float_enum_aligned = {
            let array = Array::NumericArray(NumericArray::Float64(Arc::new(FloatArray {
                data: Buffer::from(v64_float_data.clone()),
                null_mask: None,
            })));
            let float_arr = array.num().f64();
            (float_arr.data.as_slice().as_ptr() as usize)
                % std::mem::align_of::<Simd<f64, SIMD_LANES>>()
                == 0
        };

        let arrow_f64_arr_aligned = {
            let arr: ArrayRef = Arc::new(ArrowF64Array::from(v_float_data.clone()));
            let float_arr = arr.as_any().downcast_ref::<ArrowF64Array>().unwrap();
            (float_arr.values().as_ptr() as usize) % std::mem::align_of::<Simd<f64, SIMD_LANES>>()
                == 0
        };

        for _ in 0..ITERATIONS {
            // --- Integer (i64) tests ---
            // Raw Vec<i64>
            let data = v_int_data.clone();
            let start = Instant::now();
            let sum = simd_sum_i64_runtime(&data[..], simd_lanes);
            let dur = start.elapsed();
            total_vec += dur;
            black_box(sum);

            // Raw Vec64<i64>
            let data: Vec64<i64> = v64_int_data.clone();
            let start = Instant::now();
            let sum = simd_sum_i64_runtime(&data[..], simd_lanes);
            let dur = start.elapsed();
            total_vec64 += dur;
            black_box(sum);

            // Minarrow i64 (direct struct)
            let data: Vec64<i64> = v64_int_data.clone();
            let start = Instant::now();
            let int_arr = IntegerArray {
                data: Buffer::from(data),
                null_mask: None,
            };
            let sum = simd_sum_i64_runtime(int_arr.as_ref(), simd_lanes);
            let dur = start.elapsed();
            total_minarrow_direct += dur;
            black_box(sum);

            // Arrow i64 (struct direct)
            let data: Vec<i64> = v_int_data.clone();
            let start = Instant::now();
            let arr = ArrowI64Array::from(data);
            let sum = simd_sum_i64_runtime(arr.values(), simd_lanes);
            let dur = start.elapsed();
            total_arrow_struct += dur;
            black_box(sum);

            // Minarrow i64 (enum)
            let data: Vec64<i64> = v64_int_data.clone();
            let start = Instant::now();
            let array = Array::NumericArray(NumericArray::Int64(Arc::new(IntegerArray {
                data: Buffer::from(data),
                null_mask: None,
            })));
            let int_arr = array.num().i64();
            let sum = simd_sum_i64_runtime(int_arr.as_ref(), simd_lanes);
            let dur = start.elapsed();
            total_minarrow_enum += dur;
            black_box(sum);

            // Arrow i64 (dynamic)
            let data: Vec<i64> = v_int_data.clone();
            let start = Instant::now();
            let arr: ArrayRef = Arc::new(ArrowI64Array::from(data));
            let int_arr = arr.as_any().downcast_ref::<ArrowI64Array>().unwrap();
            let sum = simd_sum_i64_runtime(int_arr.values(), simd_lanes);
            let dur = start.elapsed();
            total_arrow_dyn += dur;
            black_box(sum);

            // --- Float (f64) tests ---

            // Raw Vec<f64>
            let data: Vec<f64> = v_float_data.clone();
            let start = Instant::now();
            let sum = simd_sum_f64_runtime(&data[..], simd_lanes);
            let dur = start.elapsed();
            total_vec_f64 += dur;
            black_box(sum);

            // Raw Vec64<f64>
            let data: Vec64<f64> = v64_float_data.clone();
            let start = Instant::now();
            let sum = simd_sum_f64_runtime(&data[..], simd_lanes);
            let dur = start.elapsed();
            total_vec64_f64 += dur;
            black_box(sum);

            // Minarrow f64 (direct struct)
            let data: Vec64<f64> = v64_float_data.clone();
            let start = Instant::now();
            let float_arr = FloatArray {
                data: Buffer::from(data),
                null_mask: None,
            };
            let sum = simd_sum_f64_runtime(float_arr.as_ref(), simd_lanes);
            let dur = start.elapsed();
            total_minarrow_direct_f64 += dur;
            black_box(sum);

            // Arrow f64 (struct direct)
            let data: Vec<f64> = v_float_data.clone();
            let start = Instant::now();
            let arr = ArrowF64Array::from(data);
            let sum = simd_sum_f64_runtime(arr.values(), simd_lanes);
            let dur = start.elapsed();
            total_arrow_struct_f64 += dur;
            black_box(sum);

            // Minarrow f64 (enum)
            let data: Vec64<f64> = v64_float_data.clone();
            let start = Instant::now();
            let array = Array::NumericArray(NumericArray::Float64(Arc::new(FloatArray {
                data: Buffer::from(data),
                null_mask: None,
            })));
            let float_arr = array.num().f64();
            let sum = simd_sum_f64_runtime(float_arr.as_ref(), simd_lanes);
            let dur = start.elapsed();
            total_minarrow_enum_f64 += dur;
            black_box(sum);

            // Arrow f64 (dynamic)
            let data: Vec<f64> = v_float_data.clone();
            let start = Instant::now();
            let arr: ArrayRef = Arc::new(ArrowF64Array::from(data));
            let float_arr = arr.as_any().downcast_ref::<ArrowF64Array>().unwrap();
            let sum = simd_sum_f64_runtime(float_arr.values(), simd_lanes);
            let dur = start.elapsed();
            total_arrow_dyn_f64 += dur;
            black_box(sum);
        }

        println!("Averaged Results from {} runs:", ITERATIONS);
        println!("---------------------------------");

        let avg_vec = total_vec.as_nanos() as f64 / ITERATIONS as f64;
        let avg_vec64 = total_vec64.as_nanos() as f64 / ITERATIONS as f64;
        let avg_minarrow_direct = total_minarrow_direct.as_nanos() as f64 / ITERATIONS as f64;
        let avg_arrow_struct = total_arrow_struct.as_nanos() as f64 / ITERATIONS as f64;
        let avg_minarrow_enum = total_minarrow_enum.as_nanos() as f64 / ITERATIONS as f64;
        let avg_arrow_dyn = total_arrow_dyn.as_nanos() as f64 / ITERATIONS as f64;

        let avg_vec_f64 = total_vec_f64.as_nanos() as f64 / ITERATIONS as f64;
        let avg_vec64_f64 = total_vec64_f64.as_nanos() as f64 / ITERATIONS as f64;
        let avg_minarrow_direct_f64 =
            total_minarrow_direct_f64.as_nanos() as f64 / ITERATIONS as f64;
        let avg_arrow_struct_f64 = total_arrow_struct_f64.as_nanos() as f64 / ITERATIONS as f64;
        let avg_minarrow_enum_f64 = total_minarrow_enum_f64.as_nanos() as f64 / ITERATIONS as f64;
        let avg_arrow_dyn_f64 = total_arrow_dyn_f64.as_nanos() as f64 / ITERATIONS as f64;

        println!("|------------ Integer Tests (SIMD) ------------|");
        println!(
            "raw vec: Vec<i64>                             avg = {} (n={})",
            fmt_duration_ns(avg_vec),
            ITERATIONS
        );
        println!(
            "raw vec64: Vec64<i64>                         avg = {} (n={})",
            fmt_duration_ns(avg_vec64),
            ITERATIONS
        );
        println!(
            "minarrow direct: IntegerArray                  avg = {} (n={})",
            fmt_duration_ns(avg_minarrow_direct),
            ITERATIONS
        );
        println!(
            "arrow-rs struct: Int64Array                   avg = {} (n={})",
            fmt_duration_ns(avg_arrow_struct),
            ITERATIONS
        );
        println!(
            "minarrow enum: IntegerArray                   avg = {} (n={})",
            fmt_duration_ns(avg_minarrow_enum),
            ITERATIONS
        );
        println!(
            "arrow-rs dyn: Int64Array                      avg = {} (n={})",
            fmt_duration_ns(avg_arrow_dyn),
            ITERATIONS
        );

        println!();
        println!("|------------ Float Tests (SIMD) --------------|");
        println!(
            "raw vec: Vec<f64>                             avg = {} (n={})",
            fmt_duration_ns(avg_vec_f64),
            ITERATIONS
        );
        println!(
            "raw vec64: Vec64<f64>                         avg = {} (n={})",
            fmt_duration_ns(avg_vec64_f64),
            ITERATIONS
        );
        println!(
            "minarrow direct: FloatArray                   avg = {} (n={})",
            fmt_duration_ns(avg_minarrow_direct_f64),
            ITERATIONS
        );
        println!(
            "arrow-rs struct: Float64Array                 avg = {} (n={})",
            fmt_duration_ns(avg_arrow_struct_f64),
            ITERATIONS
        );
        println!(
            "minarrow enum: FloatArray                     avg = {} (n={})",
            fmt_duration_ns(avg_minarrow_enum_f64),
            ITERATIONS
        );
        println!(
            "arrow-rs dyn: Float64Array                    avg = {} (n={})",
            fmt_duration_ns(avg_arrow_dyn_f64),
            ITERATIONS
        );

        println!("\n=> Vec64 backs the above `Minarrow` types and `Vec` backs Arrow_Rs.");

        println!("\nVerify SIMD pointer alignment for Integer calculations (based on lane width):");
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

        println!("\nVerify SIMD pointer alignment for Float calculations (based on lane width):");
        println!("Vec<f64> is aligned: {}", v_float_aligned);
        println!("Vec64<f64> is aligned: {}", v64_float_aligned);
        println!("FloatArray<f64> is aligned: {}", float_arr_aligned);
        println!("ArrowF64Array is aligned: {}", arrow_f64_aligned);
        println!(
            "Array::NumericArray<f64> is aligned: {}",
            float_enum_aligned
        );
        println!("ArrayRef is aligned: {}", arrow_f64_arr_aligned);

        println!("\n---------------------- END OF SIMD AVG BENCHMARKS ---------------------------");
    }

    fn fmt_duration_ns(avg_ns: f64) -> String {
        if avg_ns < 1000.0 {
            format!("{:.0} ns", avg_ns)
        } else if avg_ns < 1_000_000.0 {
            format!("{:.3} µs", avg_ns / 1000.0)
        } else {
            format!("{:.3} ms", avg_ns / 1_000_000.0)
        }
    }
}

fn main() {
    if cfg!(feature = "cast_arrow") {
        use crate::N;
        println!(
            "Running SIMD/Arrow/minarrow parity benchmarks (n={}, lanes={}, iters={})",
            N, SIMD_LANES, ITERATIONS
        );
        #[cfg(feature = "cast_arrow")]
        run_benchmark(N, SIMD_LANES);
    } else {
        println!(
            "The hotloop_benchmark_avg_simd example requires enabling the `cast_arrow` feature."
        )
    }
}
