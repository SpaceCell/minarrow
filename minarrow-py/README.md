# minarrow

**Arrow-compatible data that moves cleanly between Rust and Python, stays ready for SIMD, and plugs into the rest of the Python data ecosystem.**

[![Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Documentation: [minarrow.com](https://minarrow.com)

`minarrow` is a compact Python interface over [Minarrow Rust](https://github.com/pbower/minarrow).

It gives Rust and Python the same columnar data model, with Arrow-compatible memory layouts and 64-byte-aligned buffers. Data produced in Rust can be exposed directly to Python, operated on by native SIMD kernels, and passed to PyArrow, Polars, DuckDB and other Arrow-aware libraries through the Arrow PyCapsule interface.

```python
import minarrow as ma

table = ma.Table(
    {
        "id": [1, 2, 3],
        "price": [9.5, 10.0, 11.2],
    },
    name="prices",
)

series = table["price"].to_polars()
relation = table.to_duckdb()
```

Minarrow is useful when:

* Data is produced or processed in Rust and consumed in Python
* Downstream libraries such as Polars benefit from SIMD-compatible, 64-byte-aligned memory
* Arrow interoperability is required without taking an internal dependency on PyArrow
* A small set of typed containers is preferable to a large object hierarchy
* Higher-level computation is delegated to Polars, DuckDB, PyArrow or another execution engine

The Python API centres on four containers:

* `Array` for a single typed column
* `Table` for a fixed set of equal-length columns
* `ChunkedArray` for one logical column split across multiple chunks
* `ChunkedTable` for a sequence of table batches

Together, they cover the common Arrow tabular data model. Nested `Struct` and `List` types are not currently supported.

## Why not just use PyArrow?

PyArrow is an excellent Arrow implementation, but it solves a different problem. It brings the full Arrow C++ runtime into Python, together with a large type system, compute engine, file formats and dataset APIs.

Minarrow is for applications where the important path is **Rust -> Python -> native analytics**.

It uses the same Minarrow data model on both sides of the Rust–Python boundary, allocates supported buffers at 64-byte alignment (PyArrow only guarantees 8-byte), and exposes them through the Arrow PyCapsule interface. Rust can produce the data, Python can inspect and compose it, and Polars, DuckDB, PyArrow or a custom native kernel can consume it without an intermediate serialisation step.

For example, a Rust service can use Lightstream to receive a network feed directly into 64-byte-aligned Arrow buffers, expose those buffers to Python through Minarrow, and continue processing them with Python, native SIMD kernels, or machine-learning libraries without serialising or rebuilding the dataset.


|                        | Minarrow                                            | PyArrow                                               |
| ---------------------- | --------------------------------------------------- | ----------------------------------------------------- |
| Rust–Python data model | Native Minarrow types across both runtimes          | Rust integration through Arrow interchange interfaces |
| Host runtime           | Rust, embedded in the application service           | Arrow C++ runtime loaded into Python                  |
| Buffer alignment       | 64-byte aligned for Minarrow-backed buffers      | 8-byte Arrow alignment guarantee                      |
| Python API             | Four core containers for flat columnar data         | Full Arrow type and container hierarchy               |
| Native SIMD use        | Buffers are ready for aligned SIMD kernels          | Consumers must inspect alignment or realign the data  |
| Runtime role           | Application data model, interchange and composition | Arrow compute, storage and dataset platform           |
| Ecosystem integration  | Arrow PyCapsule and direct runtime bridges          | PyArrow APIs and Arrow interoperability               |
| Dependency footprint   | Focused Rust extension                              | Full Arrow C++ and Python distribution                |


Minarrow avoids making PyArrow a required part of the data path. Applications can depend only on Minarrow, hand its objects directly to capsule-aware libraries, and introduce PyArrow only where its broader functionality is useful.

```python
import minarrow as ma

table = ma.Table(
    {
        "id": [1, 2, 3],
        "price": [9.5, 10.0, 11.2],
    }
)

polars_frame = table.to_polars()
duckdb_relation = table.to_duckdb()
```

PyArrow remains fully interoperable:

```python
import pyarrow as pa

arrow_table = pa.table(table)
minarrow_table = ma.Table.from_arrow(arrow_table)
```

## Rust and Python share the same data model

Minarrow arrays and tables exist on both sides of the Rust–Python boundary.

Rust code can construct the data, perform native processing, and expose it to Python without first serialising it into another format. Python can then pass the same Arrow-compatible buffers into Polars, DuckDB, PyArrow or another capsule-aware library zero-copy.

This suits:

* Rust-based parsers and decoders
* Python extensions with columnar outputs
* Low-latency ingestion systems
* Scientific and simulation code
* Feature-generation pipelines
* Native analytics kernels

This pattern suits systems where Rust provides the production application, transport and data services, while Python delivers the intelligence layer.

## SIMD-ready buffers

Minarrow allocates supported buffers on 64-byte boundaries, matching the alignment required by wide SIMD paths such as AVX2 and AVX-512.

For compatible workloads, this adds a low-latency layer of parallelism alongside standard multi-core execution:

* Native kernels can operate directly on the buffers without copying or realigning the data.
* The buffers remain in host memory and can be processed immediately by the CPU.
* GPUs offer greater peak throughput for sufficiently large, highly parallel workloads, but first require the data to be transferred into device memory.
* For smaller batches and latency-sensitive workloads, the transfer can take longer than the computation itself, making CPU SIMD the faster end-to-end path for many tabular workloads that require a rapid decision.

The alignment is preserved when Minarrow-owned data moves from Rust into Python. Python extensions and Arrow PyCapsule consumers can access the same underlying buffers and pass them directly to compatible native kernels.

By contrast, PyArrow guarantees 8-byte alignment rather than 64-byte alignment. A native kernel that requires 64-byte-aligned input may therefore need to copy the data into a new allocation before processing it. For small and latency-sensitive workloads, that copy can take longer than the SIMD calculation itself.

## Small Python surface

Minarrow exposes one container per structural role rather than one Python class for every Arrow type.

```python
array.dtype
array.dtype.group
array.bit_width
array.arrow_type
```

`.dtype` provides a compact application-level type, while `.arrow_type` retains the complete Arrow logical type.

## Pluggable execution

Minarrow stores and exchanges columnar data. It does not attempt to replace a dataframe or query engine.

Use Minarrow for the Rust–Python boundary, then choose the execution engine appropriate to the workload:

```python
polars_frame = table.to_polars()
duckdb_relation = table.to_duckdb()
arrow_table = table.to_arrow()
```

This keeps the data layer small without restricting the rest of the application to a Minarrow-specific API.
