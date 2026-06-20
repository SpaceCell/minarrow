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

//! # **Minarrow** – High-Performance Rust with Apache Arrow Compatibility
//!
//! Modern Rust implementation of the Apache Arrow zero-copy memory layout,
//! for high-performance computing, streaming, and embedded systems.
//! Built for those who like it fast and simple.
//!
//! ## Key Features
//! - **Fast compile times** – typically <2s for standard builds, <0.15s for rebuilds.
//! - **64-byte SIMD alignment** for optimal CPU utilisation.
//! - **High runtime performance** – see benchmarks below.
//! - **Typed enum accessors** - `arr.num().f64()` returns a shared handle,
//!   an Arc bump when the variant matches. `try_` variants return `Result`.
//! - Cohesive, well-documented API with extensive coverage.
//! - Built-in FFI with simple `to_apache_arrow()` and `to_polars()` conversions.
//! - Apache-2.0 Licensed.
//!
//! ## Quick Start
//! ```rust
//! use minarrow::{MaskedArray, Print, arr_f64, fa_i32, fa_str32, tbl};
//!
//! // Create arrays with macros
//! let prices = arr_f64![10.5, 20.0, 15.75];
//!
//! // Direct typed access - no downcasting
//! assert_eq!(prices.num().f64().get(0), Some(10.5));
//!
//! // Build tables via FieldArrays with constructor macros
//! let table = tbl!(
//!     "users",
//!     fa_i32!("id", 1, 2, 3),
//!     fa_str32!("name", "alice", "bob", "charlie"),
//! );
//! table.print();
//! ```
//!
//! ## Ecosystem
//! - [`lightstream`](https://crates.io/crates/lightstream) – zero-copy Arrow IPC streaming
//!   with SIMD-aligned I/O over Tokio, TCP, QUIC, WebSocket, Unix sockets, and Stdio.
//! - `minarrow-pyo3` – zero-copy Python interop via PyArrow.
//! - [`vec64`](https://crates.io/crates/vec64) – the 64-byte aligned Vec backing every buffer.
//! - [Lightning Analytics Engine](https://spacecell.com) – sub-millisecond, zero-config
//!   live streaming engine with statistical modelling and data processing, built on Minarrow.
//!
//! ## Compatibility
//! Implements Apache Arrow’s documented memory layouts while simplifying some APIs.
//! Additional logical types are provided where they add practical value.
//! Learn more about Apache Arrow at: <https://arrow.apache.org/overview/>.
//!
//! Minarrow is not affiliated with Apache Arrow or the Apache Software Foundation.
//! *Apache Arrow* is a registered trademark of the ASF, referenced under fair use.
//!
//! ## Acknowledgements
//! Thanks to the Apache Arrow community and contributors, with inspiration
//! from `Arrow2` and `Polars`.
//!
//! ## Requirements
//! Requires Rust nightly for features such as `allocator_api`.
//!
//! ## Benchmarks
//!
//! **Intel(R) Core(TM) Ultra 7 155H | x86_64 | 22 CPUs**
//!
//! ### No SIMD
//! ***(n=1000, lanes=4, iters=1000)***
//!
//! | Case                            | Avg time |
//! |---------------------------------|----------|
//! | Vec<i64>                        | 85 ns    |
//! | Minarrow direct IntegerArray    | 88 ns    |
//! | arrow-rs struct Int64Array      | 147 ns   |
//! | Minarrow enum IntegerArray      | 124 ns   |
//! | arrow-rs dyn Int64Array         | 181 ns   |
//! | Vec<f64>                        | 475 ns   |
//! | Minarrow direct FloatArray      | 476 ns   |
//! | arrow-rs struct Float64Array    | 527 ns   |
//! | Minarrow enum FloatArray        | 507 ns   |
//! | arrow-rs dyn Float64Array       | 1.952 µs |
//!
//! ### SIMD
//! ***(n=1000, lanes=4, iters=1000)***
//!
//! | Case                            | Avg time |
//! |---------------------------------|----------|
//! | Vec<i64>                        | 64 ns    |
//! | Vec64<i64>                      | 55 ns    |
//! | Minarrow direct IntegerArray    | 88 ns    |
//! | arrow-rs struct Int64Array      | 162 ns   |
//! | Minarrow enum IntegerArray      | 170 ns   |
//! | arrow-rs dyn Int64Array         | 173 ns   |
//! | Vec<f64>                        | 57 ns    |
//! | Vec64<f64>                      | 58 ns    |
//! | Minarrow direct FloatArray      | 91 ns    |
//! | arrow-rs struct Float64Array    | 181 ns   |
//! | Minarrow enum FloatArray        | 180 ns   |
//! | arrow-rs dyn Float64Array       | 196 ns   |
//!
//! ### SIMD + Rayon
//! ***(n=1,000,000,000, lanes=4)***
//!
//! | Case                              | Time (ms) |
//! |-----------------------------------|-----------|
//! | SIMD + Rayon IntegerArray<i64>    | 113.874   |
//! | SIMD + Rayon FloatArray<f64>      | 114.095   |
//!
//! _Construction time for Vec<i64> (87 ns) and Vec64<i64> (84 ns) excluded from benchmarks._

#![feature(allocator_api)]
#![feature(portable_simd)]

pub use ::vec64::{Vec64, Vec64Alloc, vec64};

/// **Array**, **TextArray**, **NumericArray**...- *All the *High-Level Array containers* are here.*
pub mod enums {
    pub mod array;
    pub mod error;
    #[cfg(feature = "scalar_type")]
    pub mod scalar;
    pub mod time_units;
    #[cfg(feature = "value_type")]
    pub mod value;
    pub mod collections {
        pub mod numeric_array;
        #[cfg(feature = "datetime")]
        pub mod temporal_array;
        pub mod text_array;
    }
    pub mod operators;
    pub mod shape_dim;
}

/// Contains SIMD-accelerated kernels for standard arithmetic, string, bitmask and logical operations.
pub mod kernels {
    pub mod arithmetic;
    pub mod bitmask;
    #[cfg(feature = "broadcast")]
    pub mod broadcast;
    #[cfg(feature = "datetime_ops")]
    pub mod datetime;
    #[cfg(feature = "views")]
    pub mod routing;
    pub mod string;
}

/// **Table**, **IntegerArray**, **FloatArray**, **Vec64** - *All the **Low-Level Control**, **Tables** and **Views***.
pub mod structs {
    #[cfg(feature = "arena")]
    pub mod arena;

    #[cfg(feature = "chunked")]
    pub mod chunked {
        pub mod super_array;
        pub mod super_table;
    }

    pub mod variants {
        pub mod boolean;
        pub mod categorical;
        #[cfg(feature = "datetime")]
        pub mod datetime;
        pub mod float;
        pub mod integer;
        pub mod string;
    }
    pub mod views {
        #[cfg(feature = "views")]
        #[cfg(feature = "chunked")]
        pub mod chunked {
            pub mod super_array_view;
            pub mod super_table_view;
        }
        #[cfg(feature = "views")]
        pub mod collections {
            pub mod boolean_array_view;
            pub mod numeric_array_view;
            #[cfg(feature = "datetime")]
            pub mod temporal_array_view;
            pub mod text_array_view;
        }
        #[cfg(feature = "views")]
        pub mod array_view;
        pub mod bitmask_view;

        #[cfg(feature = "views")]
        pub mod table_view;
    }
    pub mod bitmask;
    pub mod buffer;
    #[cfg(feature = "lbuffer")]
    pub mod lbuffer;
    pub mod column;
    #[cfg(feature = "shared_dict")]
    pub mod dictionary;
    #[cfg(feature = "cube")]
    pub mod cube;
    pub mod field;
    pub mod field_array;
    #[cfg(feature = "matrix")]
    pub mod matrix;
    pub mod shared_buffer;
    pub mod table;
}

/// **Shared Memory** - *For sending data to other runtime(s) over FFI.*
pub mod ffi {
    pub mod arrow_c_ffi;
    pub mod arrow_dtype;
    pub mod schema;
    #[cfg(feature = "cast_arrow")]
    pub mod arrow_rs;
    #[cfg(feature = "cast_polars")]
    pub mod polars;
}

/// **Type Standardisation** - `MaskedArray`, `View`, `Print` traits + more,
pub mod traits {
    #[cfg(feature = "size")]
    pub mod byte_size;
    pub mod combine;
    pub mod concatenate;
    #[cfg(feature = "chunked")]
    pub mod consolidate;
    pub mod custom_value;
    #[cfg(feature = "datetime_ops")]
    pub mod datetime_ops;
    pub mod masked_array;
    pub mod print;
    #[cfg(feature = "select")]
    pub mod selection;
    pub mod shape;
    pub mod type_unions;
    #[cfg(feature = "views")]
    pub mod view;
}

pub mod aliases;
pub mod conversions;
pub mod macros;
pub mod utils;

pub use aliases::{
    ArrayVT, BitmaskVT, BooleanAVT, BytesLength, CategoricalAVT, CategoricalAVTExt, DictLength,
    FloatAVT, IntegerAVT, Length, Offset, StringAVT, StringAVTExt,
};

#[cfg(feature = "datetime")]
pub use aliases::DatetimeAVT;
pub use enums::array::Array;
pub use enums::collections::numeric_array::NumericArray;
#[cfg(feature = "datetime")]
pub use enums::collections::temporal_array::TemporalArray;
pub use enums::collections::text_array::TextArray;
#[cfg(feature = "scalar_type")]
pub use enums::scalar::Scalar;
#[cfg(feature = "datetime")]
pub use enums::time_units::{IntervalUnit, TimePeriod, TimeUnit};
#[cfg(feature = "value_type")]
pub use enums::value::Value;

#[cfg(feature = "arena")]
pub use structs::arena::{AAMaker, Arena, ArenaRegion};
pub use structs::bitmask::Bitmask;
pub use structs::buffer::Buffer;
#[cfg(feature = "lbuffer")]
pub use structs::lbuffer::{LBuffer, LBufferV};
#[cfg(feature = "chunked")]
pub use structs::chunked::{
    super_array::{RechunkStrategy, SuperArray},
    super_table::SuperTable,
};
#[cfg(feature = "views")]
pub use structs::views::array_view::ArrayV;
pub use structs::views::bitmask_view::BitmaskV;
#[cfg(feature = "views")]
#[cfg(feature = "chunked")]
pub use structs::views::chunked::{super_array_view::SuperArrayV, super_table_view::SuperTableV};
#[cfg(feature = "views")]
pub use structs::views::collections::boolean_array_view::BooleanArrayV;
#[cfg(feature = "views")]
pub use structs::views::collections::numeric_array_view::NumericArrayV;
#[cfg(feature = "views")]
#[cfg(feature = "datetime")]
pub use structs::views::collections::temporal_array_view::TemporalArrayV;
#[cfg(feature = "views")]
pub use structs::views::collections::text_array_view::TextArrayV;

pub use ffi::arrow_dtype::ArrowType;
pub use structs::column::{Column, column};
#[cfg(feature = "shared_dict")]
pub use structs::dictionary::Dictionary;
#[cfg(feature = "cube")]
pub use structs::cube::Cube;
pub use structs::field::Field;
pub use structs::field_array::{FieldArray, field_array};
#[cfg(feature = "matrix")]
pub use structs::matrix::Matrix;
pub use structs::shared_buffer::SharedBuffer;
pub use structs::table::Table;
pub use structs::variants::boolean::BooleanArray;
pub use structs::variants::categorical::CategoricalArray;
#[cfg(feature = "datetime")]
pub use structs::variants::datetime::DatetimeArray;
#[cfg(feature = "datetime")]
pub use structs::variants::datetime::parse::{parse_iso8601_utc, parse_iso8601_utc_ns};
pub use structs::variants::float::FloatArray;
pub use structs::variants::integer::IntegerArray;
pub use structs::variants::string::StringArray;
#[cfg(feature = "views")]
pub use structs::views::table_view::TableV;
#[cfg(feature = "size")]
pub use traits::byte_size::ByteSize;
pub use traits::combine::Combine;
pub use traits::concatenate::Concatenate;
#[cfg(feature = "chunked")]
pub use traits::consolidate::Consolidate;
#[cfg(feature = "datetime_ops")]
pub use traits::datetime_ops::DatetimeOps;
pub use traits::masked_array::MaskedArray;
pub use traits::print::Print;
#[cfg(feature = "select")]
pub use traits::selection::{ColumnSelection, RowSelection, Selection2D};
pub use traits::type_unions::{Float, Integer, Numeric, Primitive};
