<p align="center">
  <img src="https://raw.githubusercontent.com/SpaceCell/minarrow/main/minarrow-py/docs/assets/logo-trim.png" alt="Minarrow" width="320">
</p>

<p align="center">
  <a href="https://pypi.org/project/minarrow/"><img src="https://img.shields.io/pypi/v/minarrow.svg?color=001741" alt="PyPI"></a>
  <a href="https://crates.io/crates/minarrow"><img src="https://img.shields.io/crates/v/minarrow.svg?color=d1980b&labelColor=001741" alt="Crates.io"></a>
  <a href="https://docs.rs/minarrow"><img src="https://img.shields.io/docsrs/minarrow?labelColor=001741" alt="docs.rs"></a>
  <a href="https://minarrow.com"><img src="https://img.shields.io/badge/docs-minarrow.com-d1980b?labelColor=001741" alt="Docs"></a>
  <a href="https://github.com/SpaceCell/minarrow/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-001741.svg" alt="License"></a>
</p>

**Fast, minimal, and ergonomic Apache Arrow implementation in Rust, with PyO3 and Python bindings.**

**Highlights**:
- < 2s compilation times (_0.15s rebuilds_)
- Rust to Python for 1m rows in under 220ns¹
- Guaranteed 64-byte SIMD alignment
- Fully-typed throughout
- Embed Python ML in Rust
- Plug in Polars, Arrow, and Python with Zero-copy FFI, PyCapsule and PYO3

**Limitations**:
- Tabular data only. No Arrow lists or structs.

## Why Minarrow?

- **Productive**: Lean base dependencies help keep flow and library fast.
- **Fully typed**: Enum dispatch ensures constant feedback through a strong compiler + IDE feedback loop
- **Fail fast**: Errors that might slip through a dynamic dispatch boundary into run-time errors, stay compile-time.
- **Ergonomics**: Convenient dot syntax, composable opt-up abstractions, and flexible signatures via liberal `From` trait implementations.
- **Speed**: builds on the `Vec64` crate so data is fully SIMD compatible for an extra layer of low-latency parallelism, with out of the box numeric, bitmask and string SIMD kernels to help get you started.
- **Minimal Dependencies**: Small security surface - built from foundations. Only `log`, `num-traits` _(and custom-built `Vec64`)_ in the base build.

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

Semantic groupings (`NumericArray`, `TextArray`, `TemporalArray`) support flexibility call-site signatures and static dispatch.

`Array` and `Table` (`RecordBatch`) sit on top, with chunked `Super` (`Vec<Table>`) versions for streaming.

Bonus LAPACK-compatible `Matrix` and `Cube` types support analytical workload variations.

`NdArray` lands n-dimensional numeric data, with chunked (`SuperNdArray`) and labelled (`XArray`) forms, and hands tensors to PyTorch, JAX, and NumPy zero-copy over DLPack.

### Zero-Copy Views

Ergonomic zero-copy row and column selection.

```rust
use minarrow::*;

let table = create_table();

// Pandas-style selection
let view = table.c(&["name", "value"]);    // columns
let view = table.r(10..20);                // rows
let view = table.c(&["A", "B"]).r(0..100); // both

// Materialise when needed
let owned = view.to_table();
```

### Off-the-Wire pattern

```rust
use std::sync::Arc;
use minarrow::{st, Consolidate, SuperTable};

// Append batches as they arrive
let mut stream = SuperTable::new("trades".into());
stream.push(Arc::new(batch1));
stream.push(Arc::new(batch2));

// Assemble from existing batches
let stream = st!("trades", batch1, batch2);

// Consolidate to a single table when ready
let table = stream.consolidate();
```

### Arrow Interop

Embed Python in Rust.
Bench in /examples.

```rust
// Run a Random Forest Classifier using Python in Rust
let value = rt.with_python(&dataset, |py, obj| {
    let scope = PyDict::new(py);
    scope.set_item("table", obj)?;
    py.run(
        cr#"
        import polars as pl
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        df = table.to_polars()
        features = ["x0", "x1", "x2", "x3"]
        X = df.select(features).to_numpy()
        y = df["label"].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        model = RandomForestClassifier(n_estimators=100, random_state=0)
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)

        result = pl.DataFrame(
            {
                "actual": y_test.astype("int64"),
                "predicted": predicted.astype("int64"),
            }
        )
        "#,
        Some(&scope),
        Some(&scope),
    )?;
    scope
        .get_item("result")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("result not set"))
})?;
```

Normal Rust - use Polars, Arrow, convert at the boundary, stay native internally:

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

Minarrow uses enums for type dispatch instead of trait objects.

```rust
// Static dispatch routing, full compiler inlining
match array {
    Array::NumericArray(num) => match num {
        NumericArray::Int64(arr) => process(arr),
        NumericArray::Float64(arr) => process(arr),
        // ...
    },
    // ...
}

// Get fully typed IntegerArray<i64> instead of Array:
let array = array.num().i64();
```

## Benchmarks

### Array performance
Sum of 1,000 integers, averaged over 1,000 runs¹:

| Implementation | Time |
|----------------|------|
| Raw `Vec<i64>` | 85 ns |
| Minarrow `IntegerArray` (direct) | 88 ns |
| Minarrow `IntegerArray` (via enum) | 124 ns |
| Arrow-rs `Int64Array` (struct) | 147 ns |
| Arrow-rs `Int64Array` (dyn) | 181 ns |

Minarrow's direct access is within the noise threshold of raw Vec performance whilst maintaining SIMD-compatible alignment.

### Python Roundtrip
- GIL acquisition (uncontended): **53ns**
- **Rust to Python¹** 1m rows, 2 columns : **165ns**
- **Python to Rust¹** 1m rows, 2 columns:  **2.5μs**

See `minarrow-pyo3/examples` 

### Test machine
▎ ¹ = Intel Core Ultra 7 155H · 32 GB · Ubuntu 24.04 · 1.97-nightly release build.

## Feature Flags

Default features: `views`, `chunked`, `large_string`, `simd`, `select`.

| Feature | Description |
|---------|-------------|
| `views` | Zero-copy windowed access |
| `chunked` | SuperArray/SuperTable for streaming |
| `large_string` | String arrays with 64-bit offsets |
| `simd` | SIMD kernels for Bitmask and arithmetic |
| `select` | Pandas-esque `.c()` / `.r()` selection |

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
| `datetime_ops` | Datetime library functionality: ISO 8601 parsing, timezone-aware operations, arithmetic, component extraction |
| `extended_numeric_types` | i8, i16, u8, u16 variants |
| `extended_categorical` | Categorical8/16/64 dictionary index widths |
| `scalar_type` | Unified `Scalar` for aggregation results |
| `value_type` | Catch-all `Value` enum for unified typing |
| `matrix` | 2D matrix with BLAS/LAPACK-compatible layout |
| `cube` | Stacks tables along an extra axis |
| `ndarray` | N-dimensional dense array with zero-copy views and chunked form |
| `xarray` | Labelled n-dimensional array with named dims and coordinates |
| `dlpack` | DLPack tensor interchange with PyTorch, JAX, TensorFlow |
| `shared_dict` | Shared source of truth for categorical dictionaries. |

Performance:

| Feature | Description |
|---------|-------------|
| `parallel_proc` | Rayon parallel iterators |
| `fast_dict` | ~30% faster `shared_dict`, at cost of 3 dependencies |
| `fast_hash` | Swaps hashing to `ahash` |
| `arena` | Bump allocator for bulk array and Table construction |
| `vmap64` | Mmap-backed `Vec64` on Linux |
| `lbuffer` | Atomically updated array source |

Extras:

| Feature | Description |
|---------|-------------|
| `broadcast` | Typed arithmetic broadcasting |
| `str_arithmetic` | String arithmetic kernels for outlandish concatenation |
| `hash` | Hash and Eq for `Scalar` |
| `size` | Byte size estimation |
| `table_metadata` | Schema-level metadata map on `Table` |

See [Cargo.toml](Cargo.toml) for the full list with detailed notes on each.

## Ecosystem

| Project | Purpose |
|---------|---------|
| [`minarrow-py`](py/README.md) | Minarrow Python bindings  |
| [`minarrow-pyo3`](pyo3/README.md) | Zero-copy Python interop via PyArrow |
| [`vec64`](https://crates.io/crates/vec64) | Custom 64-byte aligned Vec for SIMD compatible workloads |
| [`lightstream`](https://crates.io/crates/lightstream) | Zero-copy Arrow streaming over Tokio, TCP, QUIC, WebSocket, Unix sockets, and Stdio |
| [Lightning Analytics Engine](https://spacecell.com) | Sub-millisecond, zero-config live streaming engine with statistical modelling and data processing. |

## Contributing

Contributions are welcome, particularly in the following areas:

1. **Nested types** - List and Struct support
2. **Bug fixes**

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
