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
//! at various layers of library abstraction:
//!
//!     1. Raw Vec / Vec64
//!     2. Typed "inner" arrays
//!     3. Top-level unified `Array` type
//!
//! Run with:
//!     cargo bench --bench hotloop_benchmark_std --features cast_arrow
//!
//! Use ./benchmark_avg.sh for a realistic sample, that
//! avoids compiler optimisations that otherwise distort
//! the results, or the `hotloop_benchmark_avg` example.
//! ---------------------------------------------------------

#[cfg(feature = "cast_arrow")]
use crate::benchmarks_std::run_benchmark;
#[cfg(feature = "cast_arrow")]
mod benchmarks_std {
    use std::hint::black_box;
    use std::sync::Arc;
    use std::time::Instant;

    use arrow::array::{
        Array as ArrowArrayTrait, ArrayRef, Float64Array as ArrowF64Array,
        Int64Array as ArrowI64Array,
    };
    use minarrow::{Array, Buffer, FloatArray, IntegerArray, NumericArray, Vec64};

    const N: usize = 1_000;

    pub(crate) fn run_benchmark() {
        // ----------- Raw Vec<i64> -----------
        let raw_vec: Vec<i64> = (0..N as i64).collect();
        let start = Instant::now();
        let mut acc = 0i64;
        for &v in &raw_vec {
            acc += v;
        }
        let dur_vec_i64 = start.elapsed();
        println!("raw vec: Vec<i64> sum = {}, {:?}", acc, dur_vec_i64);
        black_box(acc);
        std::mem::drop(raw_vec);

        // ----------- Raw Vec64<i64> -----------
        let raw_vec: Vec64<i64> = (0..N as i64).collect();
        let start = Instant::now();
        let mut acc = 0i64;
        for &v in &raw_vec {
            acc += v;
        }
        let dur_vec_i64 = start.elapsed();
        println!("raw vec: Vec64<i64> sum = {}, {:?}", acc, dur_vec_i64);
        black_box(acc);
        std::mem::drop(raw_vec);

        // ----------- Minarrow i64 (direct struct, no enum) -----------
        let min_data: Vec64<i64> = (0..N as i64).collect();
        let start = Instant::now();
        let int_arr = IntegerArray {
            data: Buffer::from(min_data),
            null_mask: None,
        };
        let mut acc = 0i64;
        let slice = int_arr.data.as_slice();
        for &v in slice {
            acc += v;
        }
        let dur_minarrow_direct_i64 = start.elapsed();
        println!(
            "minarrow direct: IntegerArray sum = {}, {:?}",
            acc, dur_minarrow_direct_i64
        );
        black_box(acc);
        std::mem::drop(int_arr);

        // ----------- Arrow i64 (struct direct) -----------
        let data: Vec<i64> = (0..N as i64).collect();
        let start = Instant::now();
        let arr = ArrowI64Array::from(data);
        let mut acc = 0i64;
        for i in 0..arr.len() {
            acc += arr.value(i);
        }
        let dur_arrow_struct_i64 = start.elapsed();
        println!(
            "arrow-rs struct: Int64Array sum = {}, {:?}",
            acc, dur_arrow_struct_i64
        );
        black_box(acc);
        std::mem::drop(arr);

        // ----------- Minarrow i64 (enum) -----------
        let min_data: Vec64<i64> = (0..N as i64).collect();
        let start = Instant::now();
        let array = Array::NumericArray(NumericArray::Int64(Arc::new(IntegerArray {
            data: Buffer::from(min_data),
            null_mask: None,
        })));
        let mut acc = 0i64;
        let int_arr = array.num().i64();
        let slice = int_arr.data.as_slice();
        for &v in slice {
            acc += v;
        }
        let dur_minarrow_enum_i64 = start.elapsed();
        println!(
            "minarrow enum: IntegerArray sum = {}, {:?}",
            acc, dur_minarrow_enum_i64
        );
        black_box(acc);
        std::mem::drop(int_arr);

        // ----------- Arrow i64 (dynamic) -----------
        let data_dyn: Vec<i64> = (0..N as i64).collect();
        let start = Instant::now();
        let arr_dyn: ArrayRef = Arc::new(ArrowI64Array::from(data_dyn));
        let mut acc = 0i64;
        if let Some(int) = arr_dyn.as_any().downcast_ref::<ArrowI64Array>() {
            for i in 0..int.len() {
                acc += int.value(i);
            }
        }
        let dur_arrow_dyn_i64 = start.elapsed();
        println!(
            "arrow-rs dyn: ArrayRef Int64Array sum = {}, {:?}",
            acc, dur_arrow_dyn_i64
        );
        black_box(acc);
        std::mem::drop(arr_dyn);

        // ----------- Raw Vec<f64> -----------
        let raw_vec: Vec<f64> = (0..N as i64).map(|x| x as f64).collect();
        let start = Instant::now();
        let mut acc = 0.0f64;
        for &v in &raw_vec {
            acc += v;
        }
        let dur_vec_f64 = start.elapsed();
        println!("raw vec: Vec<f64> sum = {}, {:?}", acc, dur_vec_f64);
        black_box(acc);
        std::mem::drop(raw_vec);

        // ----------- Raw Vec64<f64> -----------
        let raw_vec: Vec64<f64> = (0..N as i64).map(|x| x as f64).collect();
        let start = Instant::now();
        let mut acc = 0.0f64;
        for &v in &raw_vec {
            acc += v;
        }
        let dur_vec_f64 = start.elapsed();
        println!("raw vec: Vec<f64> sum = {}, {:?}", acc, dur_vec_f64);
        black_box(acc);
        std::mem::drop(raw_vec);

        // ----------- Minarrow f64 (direct struct, no enum) -----------
        let min_data_f64: Vec64<f64> = (0..N as i64).map(|x| x as f64).collect();
        let start = Instant::now();
        let float_arr = FloatArray {
            data: Buffer::from(min_data_f64),
            null_mask: None,
        };
        let mut acc = 0.0f64;
        let slice = float_arr.data.as_slice();
        for &v in slice {
            acc += v;
        }
        let dur_minarrow_direct_f64 = start.elapsed();
        println!(
            "minarrow direct: FloatArray sum = {}, {:?}",
            acc, dur_minarrow_direct_f64
        );
        black_box(acc);
        std::mem::drop(float_arr);

        // ----------- Arrow f64 (struct direct) -----------
        let data_f64: Vec<f64> = (0..N as i64).map(|x| x as f64).collect();
        let start = Instant::now();
        let arr = ArrowF64Array::from(data_f64);
        let mut acc = 0.0f64;
        for i in 0..arr.len() {
            acc += arr.value(i);
        }
        let dur_arrow_struct_f64 = start.elapsed();
        println!(
            "arrow-rs struct: Float64Array sum = {}, {:?}",
            acc, dur_arrow_struct_f64
        );
        black_box(acc);
        std::mem::drop(arr);

        // ----------- Minarrow f64 (enum) -----------
        let min_data_f64: Vec64<f64> = (0..N as i64).map(|x| x as f64).collect();
        let start = Instant::now();
        let array = Array::NumericArray(NumericArray::Float64(Arc::new(FloatArray {
            data: Buffer::from(min_data_f64),
            null_mask: None,
        })));
        let mut acc = 0.0f64;
        let float_arr = array.num().f64();
        let slice = float_arr.data.as_slice();
        for &v in slice {
            acc += v;
        }
        let dur_minarrow_enum_f64 = start.elapsed();
        println!(
            "minarrow enum: FloatArray sum = {}, {:?}",
            acc, dur_minarrow_enum_f64
        );
        black_box(acc);
        std::mem::drop(float_arr);

        // ----------- Arrow f64 (dynamic) -----------
        let data_f64: Vec<f64> = (0..N as i64).map(|x| x as f64).collect();
        let start = Instant::now();
        let arr: ArrayRef = Arc::new(ArrowF64Array::from(data_f64));
        let mut acc = 0.0f64;
        if let Some(f) = arr.as_any().downcast_ref::<ArrowF64Array>() {
            for i in 0..f.len() {
                acc += f.value(i);
            }
        }
        let dur_arrow_dyn_f64 = start.elapsed();
        println!(
            "arrow-rs dyn: Float64Array sum = {}, {:?}",
            acc, dur_arrow_dyn_f64
        );
        black_box(acc);
        std::mem::drop(arr);
    }
}

fn main() {
    if cfg!(feature = "cast_arrow") {
        #[cfg(feature = "cast_arrow")]
        run_benchmark()
    } else {
        println!("The apache-FFI example requires enabling the `cast_arrow` feature.")
    }
}
