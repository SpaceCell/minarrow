# Minarrow

**Arrow-compatible data that moves cleanly between Rust and Python, stays ready for SIMD, and plugs into the rest of the Python data ecosystem.**

[![PyPI](https://img.shields.io/pypi/v/minarrow.svg?color=001741)](https://pypi.org/project/minarrow/)
[![License](https://img.shields.io/badge/license-Apache--2.0-001741.svg)](https://github.com/SpaceCell/minarrow/blob/main/LICENSE)
[![Minarrow Rust crate](https://img.shields.io/badge/rust-Minarrow%20-d1980b?labelColor=001741)](https://crates.io/crates/minarrow)

**Minarrow** is a compact *Python* interface over **Minarrow Rust**. It gives *Rust* and *Python* the same columnar data model, with *Apache Arrow*-compatible memory layouts and 64-byte-aligned buffers. Data produced in *Rust* can be exposed directly to *Python*, operated on by native SIMD kernels, and passed to *PyArrow*, *Polars*, *DuckDB* and other *Arrow*-aware libraries through the *Arrow* *PyCapsule* interface.

<div class="grid cards" markdown>

-   :material-flash:{ .lg .middle } __Fast__

    ---

    Data crosses between *Rust* and *Python* zero-copy, with no serialisation step.

-   :material-memory:{ .lg .middle } __Compatible__

    ---

    Buffers sit on SIMD 64-byte boundaries, ready for AVX2 and AVX-512 kernels.

-   :material-puzzle:{ .lg .middle } __Pluggable__

    ---

    Hand data to *Polars*, *DuckDB*, *pandas* and *PyArrow* Ecosystem through the *Arrow* *PyCapsule* interface.

-   :material-feather:{ .lg .middle } __Simple__

    ---

    Focused containers cover flat columnar and N-dimensional data.

</div>

The same data model exists on both sides of the boundary.

=== "Python"

    ```python
    import minarrow as ma

    table = ma.Table(
        {"id": [1, 2, 3], "price": [9.5, 10.0, 11.2]},
        name="prices",
    )

    series = table["price"].to_polars()
    relation = table.to_duckdb()
    ```

=== "Rust"

    ```rust
    use minarrow::{Table, fa_i64, fa_f64};

    let table = Table::new(
        "prices",
        Some(vec![
            fa_i64!("id", 1, 2, 3),
            fa_f64!("price", 9.5, 10.0, 11.2),
        ]),
    );

    let series = table.c[1].to_polars();
    let batch = table.to_apache_arrow();
    ```

## Overview

Minarrow provides four primary columnar *Python* containers:

* `Array` for typed columnar data
* `Table` for named collections of equal-length arrays
* `ChunkedArray` for one logical column split across multiple chunks
* `ChunkedTable` for a sequence of table batches

For f32/f64 N-dimensional data it also provides:

* `NdArray` for contiguous data and zero-copy selections
* `ChunkedNdArray` for compatible pieces backed by Rust `SuperNdArray`
* `XArray` for named axes and coordinate selection

These are data containers and interchange objects, not a replacement for a
numerical or statistical runtime. Use their DLPack support to pass data to
NumPy, PyTorch, JAX, or another compute library.

The *Python* package is backed by the [Minarrow Rust core](https://github.com/pbower/minarrow), which provides the underlying array types, schemas and aligned buffers.

Minarrow is intended for:

* Passing columnar data between *Rust* and *Python*
* Building *Python* extensions that operate on *Arrow*-compatible data
* Feeding *Polars*, *DuckDB*, *PyArrow* and other *Arrow*-aware libraries
* Running SIMD-oriented native kernels over guaranteed aligned buffers
* Keeping binary size and dependency footprint small

## Installation

```bash
pip install minarrow
```

## Basic usage

```python
import minarrow as ma

table = ma.Table(
    {
        "id": [1, 2, 3],
        "price": [9.5, 10.0, 11.2],
    },
    name="prices",
)

print(table.columns)
# ['id', 'price']

print(table.dtypes)
# {'id': DType.Integer, 'price': DType.Float}
```

Create an individual array:

```python
ids = ma.Array([1, 2, 3], name="id")
```

Pass data to another *Arrow*-aware library:

```python
polars_series = ids.to_polars()
duckdb_relation = table.to_duckdb()
```

See the [Quickstart](quickstart.md) for arrays, tables, indexing and schema operations.

## Why Minarrow

### Small typed API

Minarrow exposes a compact object model rather than reproducing the full *Apache Arrow* surface area.

Arrays retain a concrete data type, exposed through `.dtype`, while tables provide named access to their constituent arrays.

### Arrow ecosystem integration

`Array` and `Table` implement the *Arrow* *PyCapsule* interface. Compatible libraries can import their schema and buffers without requiring an intermediate serialisation format.

This provides an interoperability path to:

* *Polars*
* *DuckDB*
* *PyArrow*
* *pandas* through an *Arrow*-compatible backend
* Native *Python* extensions that consume *Arrow* capsules

The *PyCapsule* integration is zero-copy.

### 64-byte-aligned buffers

The *Rust* core stores supported data buffers with 64-byte alignment.

This is useful for native kernels using SIMD instruction sets such as *AVX2* or *AVX-512*, as suitably aligned buffers can be processed without first copying them into a separate aligned allocation.

Alignment does not make an operation SIMD-accelerated by itself. It provides a predictable memory layout for native code that implements vectorised kernels.

### Rust-backed implementation

The core data structures are implemented in *Rust* using concrete array types and enum-based dispatch. This allows type-specific paths to be compiled and optimised without requiring dynamic *Python* dispatch inside inner loops.

*Python* remains the orchestration layer, while storage and native operations are handled by the *Rust* implementation.

## Interoperability

Minarrow is a columnar data container/bridge as opposed to a full dataframe execution engine.

Use it to construct or receive data, operate on it through native extensions, and pass it to an appropriate query or dataframe system zero-copy.

| Requirement               | Example integration                 |
| ------------------------- | ------------------------------------- |
| Dataframe expressions     | *Polars*                                |
| SQL queries               | *DuckDB*                                |
| General *Arrow* interchange | *PyArrow*                               |
| *pandas* workflows          | *pandas* with an *Arrow*-compatible path  |
| Custom native computation | *Rust*, C or *C++* through *Arrow* capsules |

The [Rust crate](https://docs.rs/minarrow/latest/minarrow/) does provide a minimal set of tabular data-processing capabilities but without full parallelism, which serves as a useful basis for foundational operations in that environment. It compiles in less than 2 seconds with the standard feature set to help you stay productive, and your project lightweight. 

See [Ecosystem interoperability](interop.md) for supported conversion paths and ownership behaviour.

## Documentation

* [Quickstart](quickstart.md) - arrays, tables, indexing, fields and schemas
* [Types and schemas](types.md) - data types and *Arrow* schema mapping
* [Ecosystem interoperability](interop.md) - *PyCapsule* and library integration
* [API reference](api.md) - public types, methods and properties

## Support

Report issues through the [Minarrow GitHub repository](https://github.com/spacecell/minarrow).

Minarrow is built and maintained by [SpaceCell](https://spacecell.com).
