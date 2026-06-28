# Minarrow

**A fast, minimal columnar data library for Rust with Arrow compatibility.**

[![Crates.io](https://img.shields.io/crates/v/minarrow.svg)](https://crates.io/crates/minarrow)
[![Documentation](https://docs.rs/minarrow/badge.svg)](https://docs.rs/minarrow)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Minarrow gives you typed columnar arrays that compile in ~2 seconds, run with SIMD alignment, and convert to Arrow when you need interop. It keeps the common path concrete and lightweight, so iteration stays fast.

## Why Minarrow?

Minarrow buys you productivity.

**Problems:**

- **Compile times**: Other Arrow Rust libraries are powerful but heavy, and build times can take minutes. Lean base dependencies keep yours fast.
- **Type erasure**: Other libraries use `dyn Array`, where the concrete backing vector is hidden behind the trait object. The impact:
  - **Reliability**: A class of errors moves from compile-time to run-time, as the compiler can't see through the downcasting boundary.
  - **Ergonomics**: You need to downcast to recover the types.
  - **Productivity**: If you are using AI, it needs to iterate types against the compiler instead of leaving run-time downcast bombs. Let it work for you.

**The solution:** Minarrow keeps concrete types throughout. An `IntegerArray<i64>` stays fully typed through composable abstractions. You get direct access, ergonomics, IDE autocomplete, and fast compilation. When you need to talk to Arrow, Polars, or PyArrow, zero-copy conversion is one method call away.

## Installation

```bash
cargo add minarrow
```

Minarrow uses the nightly toolchain for `allocator_api` and `portable_simd`:

```bash
rustup override set nightly
```

## Quick Start

```rust
use minarrow::{arr_i32, arr_f64, arr_str32, arr_bool, MaskedArray};

// Create arrays with macros
let ids = arr_i32![1, 2, 3, 4];
let prices = arr_f64![10.5, 20.0, 15.75];
let names = arr_str32!["alice", "bob", "charlie"];
let flags = arr_bool![true, false, true];

// Direct typed access - no downcasting
assert_eq!(ids.len(), 4);
assert_eq!(prices.num().f64().get(0), Some(10.5));
```

```rust
use minarrow::{tbl, fa_i32, fa_str32, Print};

// Build tables via FieldArrays with constructor macros
let table = tbl!(
    "users",
    fa_i32!("id", 1, 2, 3),
    fa_str32!("name", "alice", "bob", "charlie"),
);
table.print();
```

## Core Features

### Typed Arrays

Six typed arrays back standard workloads:

| Type | Description |
|------|-------------|
| `IntegerArray<T>` | i8 through u64 |
| `FloatArray<T>` | f32, f64 |
| `StringArray<T>` | UTF-8 with u32 or u64 offsets |
| `BooleanArray` | Bit-packed with validity mask |
| `CategoricalArray<T>` | Dictionary-encoded |
| `DatetimeArray<T>` | Timestamps, dates, durations |

Semantic groupings (`NumericArray`, `TextArray`, `TemporalArray`) let you write generic functions while keeping static dispatch.

`Array` and `Table` complete the picture, with chunked `Super` versions for streaming.

Bonus LAPACK-compatible `Matrix` and `Cube` types.

### Fast Compilation

| Metric | Time |
|--------|------|
| Clean build | < 2s |
| Incremental rebuild | < 0.15s |

Achieved through minimal dependencies: `num-traits`, `vec64`, and `log`, with optional `rayon` for parallelism.

### SIMD Alignment

All buffers use 64-byte alignment via `Vec64`. There is no reallocation step to fix alignment: data is ready for vectorised operations from the moment it's created.

### Zero-Copy Views

Select columns and rows without copying data:

```rust
use minarrow::*;

let table = create_table();

// Pandas-style selection
let view = table.c(&["name", "value"]);    // columns
let view = table.r(10..20);                // rows
let view = table.c(&["A", "B"]).r(0..100); // both

// Materialise only when needed
let owned = view.to_table();
```

### Streaming with SuperArrays

For streaming workloads, `SuperArray` and `SuperTable` hold multiple chunks with consistent schema:

```rust
use std::sync::Arc;
use minarrow::{st, Consolidate, SuperTable};

// Append batches as they arrive
let mut stream = SuperTable::new("trades".into());
stream.push(Arc::new(batch1));
stream.push(Arc::new(batch2));

// Or assemble from existing batches in one line
let stream = st!("trades", batch1, batch2);

// Consolidate to a single table when ready
let table = stream.consolidate();
```

### Arrow Interop

Convert at the boundary, stay native internally:

```rust
// To Arrow (feature: cast_arrow)
let batch = table.to_apache_arrow();          // RecordBatch
let column = table.cols[0].to_apache_arrow(); // ArrayRef

// To Polars (feature: cast_polars)
let df = table.to_polars();                   // DataFrame

// FFI via the Arrow C Data Interface
use minarrow::ffi::arrow_c_ffi::export_to_c;
let (array_ptr, schema_ptr) = export_to_c(array, schema);
```

## Architecture

Minarrow uses enums for type dispatch instead of trait objects:

```rust
// Static dispatch, full inlining
match array {
    Array::NumericArray(num) => match num {
        NumericArray::Int64(arr) => process(arr),
        NumericArray::Float64(arr) => process(arr),
        // ...
    },
    // ...
}
```

This gives you:
- **Performance** - Compiler inlines through the dispatch
- **Type safety** - No `Any`, no runtime downcasts
- **Ergonomics** - Direct accessors like `array.num().i64()`, an Arc bump when the variant matches

## Benchmarks

Sum of 1,000 integers, averaged over 1,000 runs (Intel Ultra 7 155H):

| Implementation | Time |
|----------------|------|
| Raw `Vec<i64>` | 85 ns |
| Minarrow `IntegerArray` (direct) | 88 ns |
| Minarrow `IntegerArray` (via enum) | 124 ns |
| Arrow-rs `Int64Array` (struct) | 147 ns |
| Arrow-rs `Int64Array` (dyn) | 181 ns |

Minarrow's direct access matches raw Vec performance. Even through enum dispatch, it outperforms arrow-rs.

With SIMD + Rayon, summing 1 billion integers takes ~114ms.

## Feature Flags

Default features: `views`, `chunked`, `large_string`, `simd`, `select`.

| Feature | Description |
|---------|-------------|
| `views` | Zero-copy windowed access (default) |
| `chunked` | SuperArray/SuperTable for streaming (default) |
| `large_string` | String arrays with 64-bit offsets (default) |
| `simd` | SIMD kernels for Bitmask and arithmetic (default) |
| `select` | Pandas-style `.c()` / `.r()` selection (default) |

Interop:

| Feature | Description |
|---------|-------------|
| `cast_arrow` | Arrow-rs conversion via `to_apache_arrow()` |
| `cast_polars` | Polars conversion via `to_polars()` / `from_polars()` |
| `memfd` | Memfd-backed buffers for zero-copy cross-process sharing (Linux) |

Additional types:

| Feature | Description |
|---------|-------------|
| `datetime` | Temporal array types as raw integer offsets |
| `datetime_ops` | Full datetime functionality: ISO 8601 parsing, timezone-aware operations, arithmetic, component extraction |
| `extended_numeric_types` | i8, i16, u8, u16 variants |
| `extended_categorical` | Categorical8/16/64 dictionary index widths |
| `scalar_type` | Unified `Scalar` for aggregation results |
| `value_type` | Catch-all `Value` enum for engine-level orchestration |
| `matrix` | 2D matrix with BLAS/LAPACK-compatible layout |
| `cube` | Stacks tables along an extra axis for time series and group analytics |

Performance:

| Feature | Description |
|---------|-------------|
| `parallel_proc` | Rayon parallel iterators |
| `shared_dict` | Cross-batch shared categorical dictionaries |
| `fast_dict` | ~30% faster `shared_dict`, at the cost of 3 dependencies |
| `fast_hash` | Swaps hashing to `ahash` |
| `arena` | Bump allocator for bulk array and Table construction |
| `vmap64` | Mmap-backed `Vec64` on Linux |

Extras:

| Feature | Description |
|---------|-------------|
| `broadcast` | Typed arithmetic broadcasting |
| `str_arithmetic` | String arithmetic kernels, e.g. `+` concatenation |
| `hash` | Hash and Eq for `Scalar` |
| `size` | Byte size estimation |
| `table_metadata` | Schema-level metadata map on `Table` |

See [Cargo.toml](Cargo.toml) for the full list with detailed notes on each.

## Ecosystem

| Project | Purpose |
|---------|---------|
| [`minarrow-pyo3`](pyo3/README.md) | Zero-copy Python interop via PyArrow |
| [`lightstream`](https://crates.io/crates/lightstream) | Zero-copy Arrow streaming over Tokio, TCP, QUIC, WebSocket, Unix sockets, and Stdio |
| [`vec64`](https://crates.io/crates/vec64) | 64-byte aligned Vec for optimal SIMD |
| [Lightning Analytics Engine](https://spacecell.com) | Sub-millisecond, zero-config live streaming engine with statistical modelling and data processing. Built on Minarrow |

## Limitations

Minarrow focuses on flat columnar data, covering the common 80% of analytical workloads. Nested types (List, Struct) are not currently supported. If you need deeply nested schemas, arrow-rs is the better choice.

## Contributing

Contributions are welcome, particularly in the following areas:

1. **Connectors** – Data source and sink integrations
2. **Optimisations** – Performance improvements
3. **Nested types** – List and Struct support
4. **Bug fixes**

All contributions are subject to the Contributor Licence Agreement (CLA).
See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

Copyright © 2025–2026 Peter Garfield Bower.

Released under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

## Acknowledgements

Minarrow is a from-scratch implementation of the Apache Arrow memory layout inspired by the standards pioneered by Apache Arrow, Arrow2, and Polars.

Minarrow is not affiliated with Apache Arrow.

## SpaceCell

Minarrow is maintained by [SpaceCell](https://spacecell.com) and forms part of its open-source foundation for high-performance data computing.
