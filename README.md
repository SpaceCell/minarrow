# Minarrow

**A fast, minimal columnar data library for Rust with Arrow compatibility.**

Minarrow gives you typed columnar arrays that compile in ~1.5 seconds, run with SIMD alignment, and convert to Arrow when you need interop. It keeps the common path concrete and lightweight, so iteration stays fast.

## Why Minarrow?

**The gap:** Arrow-rs is powerful but heavy. Build times can stretch to minutes, which matters when it becomes a base dependency for other systems. Working with arrays often means handling them through `dyn Array`, where the concrete backing vector is hidden behind the trait object. You downcast to recover the type before the data becomes ergonomic again. For real-time systems, embedded devices, or rapid iteration, that friction adds up.

**The solution:** Minarrow keeps concrete types throughout. An `IntegerArray<i64>` stays fully typed through composable abstractions. You get direct access, ergonomics, IDE autocomplete, and fast compilation. When you need to talk to Arrow, Polars, or PyArrow, zero-copy conversion is one method call away.

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
use minarrow::{FieldArray, Table, arr_i32, arr_str32, Print};

// Build tables from columns
let table = Table::new(
    "users".into(),
    vec![
        FieldArray::from_arr("id", arr_i32![1, 2, 3]),
        FieldArray::from_arr("name", arr_str32!["alice", "bob", "charlie"]),
    ].into(),
);
table.print();
```

## Core Features

### Typed Arrays

Six array types cover standard workloads:

| Type | Description |
|------|-------------|
| `IntegerArray<T>` | i8 through u64 |
| `FloatArray<T>` | f32, f64 |
| `StringArray<T>` | UTF-8 with u32 or u64 offsets |
| `BooleanArray` | Bit-packed with validity mask |
| `CategoricalArray<T>` | Dictionary-encoded |
| `DatetimeArray<T>` | Timestamps, dates, durations |

Semantic groupings (`NumericArray`, `TextArray`, `TemporalArray`) let you write generic functions while keeping static dispatch.

`Array` and `Table` complete the story, with chunked `Super` versions for streaming.

### Fast Compilation

| Metric | Time |
|--------|------|
| Clean build | < 1.5s |
| Incremental rebuild | < 0.15s |

Achieved through minimal dependencies: primarily `num-traits`, with optional `rayon` for parallelism.

### SIMD Alignment

All buffers use 64-byte alignment via `Vec64`. No reallocation step to fix alignment-data is ready for vectorised operations from the moment it's created.

### Zero-Copy Views

Select columns and rows without copying data:

```rust
use minarrow::*;

let table = create_table();

// Pandas-style selection
let view = table.c(&["name", "value"]);  // columns
let view = table.r(10..20);               // rows
let view = table.c(&["A", "B"]).r(0..100); // both

// Materialise only when needed
let owned = view.to_table();
```

### Streaming with SuperArrays

For streaming workloads, `SuperArray` and `SuperTable` hold multiple chunks with consistent schema:

```rust
// Append chunks as they arrive
let mut super_table = SuperTable::new();
super_table.push_table(batch1);
super_table.push_table(batch2);

// Consolidate to single table when ready
let table = super_table.consolidate();
```

### Arrow Interop

Convert at the boundary, stay native internally:

```rust
// To Arrow (feature: cast_arrow)
let arrow_array = minarrow_array.to_apache_arrow();

// To Polars (feature: cast_polars)
let series = minarrow_array.to_polars();

// FFI via Arrow C Data Interface
let (array_ptr, schema_ptr) = minarrow_array.export_to_c();
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
- **Ergonomics** - Direct accessors like `array.num().i64()`

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

Enable what you need:

| Feature | Description |
|---------|-------------|
| `views` | Zero-copy windowed access (default) |
| `chunked` | SuperArray/SuperTable for streaming (default) |
| `datetime` | Temporal types |
| `cast_arrow` | Arrow-rs conversion |
| `cast_polars` | Polars conversion |
| `parallel_proc` | Rayon parallel iterators |
| `select` | Pandas-style `.c()` / `.r()` selection |
| `broadcast` | Arithmetic broadcasting |
| `shared_dict` | Cross-batch shared categorical dictionaries. |
| `fast_dict` | ~30% faster variant of `shared_dict`, at the cost of 3 dependencies.  |

## Ecosystem

| Crate | Purpose |
|-------|---------|
| `minarrow-pyo3` | Zero-copy Python interop via PyArrow. See [pyo3/README.md](pyo3/README.md) |
| `lightstream` | Zero-copy Arrow streaming over Tokio, TCP, QUIC, WebSocket, Unix sockets, and Stdio |
| `simd-kernels` | 60+ SIMD kernels including statistical distributions |
| `vec64` | 64-byte aligned Vec for optimal SIMD |

## Limitations

Minarrow focuses on flat columnar data and 80/20. Nested types (List, Struct) are not currently supported. If you need deeply nested schemas, arrow-rs is the better choice.

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

## HPC
Minarrow is backed by SpaceCell, ensuring ongoing support for the project.

For an edge in high-performance data computing, visit [spacecell.com](http://spacecell.com)
