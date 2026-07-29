# minarrow-pyo3

PyO3 bindings for MinArrow - zero-copy Arrow interop with Python via the Arrow PyCapsule and C Data Interfaces.

## Overview

This crate provides transparent wrapper types that enable straightforward conversion between MinArrow's Rust types and Python Arrow-compatible types. Any library supporting the Arrow PyCapsule protocol works out of the box: PyArrow, Polars, DuckDB, nanoarrow, pandas with ArrowDtype, etc.

## Container Type Mappings

MinArrow calls an object with a header, rows and columns a "Table" favouring broader matter-of-factness. Apache Arrow calls it a "RecordBatch" in line with the Apache Arrow standard, whereby a "Table" (at least in PyArrow) is considered a chunked composition of those RecordBatches, for a more highly engineered approach. Below is how they map to one another for the equivalent memory and object layout.

| MinArrow | PyArrow | Wrapper Type | Protocol |
|----------|---------|--------------|----------|
| `Array` | `pa.Array` | `PyArray` | `__arrow_c_array__` |
| `Table` | `pa.RecordBatch` | `PyRecordBatch` | `__arrow_c_stream__` (one batch) |
| `SuperTable` | `pa.Table` | `PyTable` | `__arrow_c_stream__` (multiple batches) |
| `SuperArray` | `pa.ChunkedArray` | `PyChunkedArray` | `__arrow_c_stream__` (one array per chunk) |

## Array Data Type Mappings

Each inner MinArrow array type maps 1:1 to a specific PyArrow typed array. The Arrow C Data Interface preserves schema metadata so PyArrow reconstructs the correct type on import.

### Numeric types

| MinArrow inner type | `Array` enum path | Arrow format | PyArrow type |
|---------------------|-------------------|--------------|--------------|
| `IntegerArray<i32>` | `NumericArray::Int32` | `i` | `pa.Int32Array` |
| `IntegerArray<i64>` | `NumericArray::Int64` | `l` | `pa.Int64Array` |
| `IntegerArray<u32>` | `NumericArray::UInt32` | `I` | `pa.UInt32Array` |
| `IntegerArray<u64>` | `NumericArray::UInt64` | `L` | `pa.UInt64Array` |
| `FloatArray<f32>` | `NumericArray::Float32` | `f` | `pa.FloatArray` |
| `FloatArray<f64>` | `NumericArray::Float64` | `g` | `pa.DoubleArray` |

### Extended numeric types (feature `extended_numeric_types`)

| MinArrow inner type | `Array` enum path | Arrow format | PyArrow type |
|---------------------|-------------------|--------------|--------------|
| `IntegerArray<i8>` | `NumericArray::Int8` | `c` | `pa.Int8Array` |
| `IntegerArray<i16>` | `NumericArray::Int16` | `s` | `pa.Int16Array` |
| `IntegerArray<u8>` | `NumericArray::UInt8` | `C` | `pa.UInt8Array` |
| `IntegerArray<u16>` | `NumericArray::UInt16` | `S` | `pa.UInt16Array` |

### Boolean

| MinArrow inner type | `Array` enum path | Arrow format | PyArrow type |
|---------------------|-------------------|--------------|--------------|
| `BooleanArray` | `Array::BooleanArray` | `b` | `pa.BooleanArray` |

### Text types

| MinArrow inner type | `Array` enum path | Arrow format | PyArrow type |
|---------------------|-------------------|--------------|--------------|
| `StringArray<u32>` | `TextArray::String32` | `u` | `pa.StringArray` |
| `StringArray<u64>` | `TextArray::String64` | `U` | `pa.LargeStringArray` |

#### Utf8View import

Arrow and Polars define two distinct string layouts: the original **Utf8** format stores strings as a contiguous data buffer indexed by an offsets array, while the newer **Utf8View** format stores an array of 16-byte view structs that either inline short strings or reference slices of separate variadic data buffers. Utf8View avoids offset recomputation during operations like filtering and slicing, which benefits query engines that chain many such operations before materialising a result. The trade-off is a more complex physical layout and larger per-element overhead for short strings.

MinArrow uses a single `StringArray<u32>` representation with contiguous offsets and data, and does not maintain a separate Utf8View layout internally. Utf8View's optimisation is most effective where the engine controls the full pipeline and can keep data in view form across many intermediate operations, deferring materialisation until the end. When a Utf8View array is received over the Arrow C Data Interface, e.g. from Polars `pl.Enum` which exports its dictionary values in Utf8View format, MinArrow reads the view structs and reassembles the string data into its standard offsets+data layout. The resulting array is a regular `StringArray<u32>` and is re-exported as Arrow `Utf8` on the way back out. This means Utf8View import is always a copy rather than zero-copy.

### Temporal types (feature `datetime`)

MinArrow stores temporal data in `DatetimeArray<i32>` or `DatetimeArray<i64>` with a `TimeUnit` discriminator. The Arrow type is determined by the `ArrowType` in the `Field`, not the storage type alone.

| MinArrow inner type | `ArrowType` | Arrow format | PyArrow type |
|---------------------|-------------|--------------|--------------|
| `DatetimeArray<i32>` | `Date32` | `tdD` | `pa.Date32Array` |
| `DatetimeArray<i64>` | `Date64` | `tdm` | `pa.Date64Array` |
| `DatetimeArray<i32>` | `Time32(Seconds)` | `tts` | `pa.Time32Array` |
| `DatetimeArray<i32>` | `Time32(Milliseconds)` | `ttm` | `pa.Time32Array` |
| `DatetimeArray<i64>` | `Time64(Microseconds)` | `ttu` | `pa.Time64Array` |
| `DatetimeArray<i64>` | `Time64(Nanoseconds)` | `ttn` | `pa.Time64Array` |
| `DatetimeArray<i64>` | `Timestamp(Seconds, tz)` | `tss:tz` | `pa.TimestampArray` |
| `DatetimeArray<i64>` | `Timestamp(Milliseconds, tz)` | `tsm:tz` | `pa.TimestampArray` |
| `DatetimeArray<i64>` | `Timestamp(Microseconds, tz)` | `tsu:tz` | `pa.TimestampArray` |
| `DatetimeArray<i64>` | `Timestamp(Nanoseconds, tz)` | `tsn:tz` | `pa.TimestampArray` |
| `DatetimeArray<i32>` | `Duration32(Seconds)` | `tDs` | `pa.DurationArray` |
| `DatetimeArray<i32>` | `Duration32(Milliseconds)` | `tDm` | `pa.DurationArray` |
| `DatetimeArray<i64>` | `Duration64(Microseconds)` | `tDu` | `pa.DurationArray` |
| `DatetimeArray<i64>` | `Duration64(Nanoseconds)` | `tDn` | `pa.DurationArray` |

Timezone metadata for timestamps is preserved via the Arrow schema format string.

### Categorical / dictionary types

| MinArrow inner type | `Array` enum path | Arrow format | PyArrow type |
|---------------------|-------------------|--------------|--------------|
| `CategoricalArray<u32>` | `TextArray::Categorical32` | dictionary(int32, utf8) | `pa.DictionaryArray` |

With feature `extended_categorical` + `extended_numeric_types`:

| MinArrow inner type | `Array` enum path | Arrow format | PyArrow type |
|---------------------|-------------------|--------------|--------------|
| `CategoricalArray<u8>` | `TextArray::Categorical8` | dictionary(int8, utf8) | `pa.DictionaryArray` |
| `CategoricalArray<u16>` | `TextArray::Categorical16` | dictionary(int16, utf8) | `pa.DictionaryArray` |

With feature `extended_categorical`:

| MinArrow inner type | `Array` enum path | Arrow format | PyArrow type |
|---------------------|-------------------|--------------|--------------|
| `CategoricalArray<u64>` | `TextArray::Categorical64` | dictionary(int64, utf8) | `pa.DictionaryArray` |

For categorical types, the integer buffer is zero-copy but we clone the (finite) dictionary categories.
Unless you have a very large unique category count, this should not cause performance issues.

### Nullability

All array types support null values via MinArrow's `MaskedArray` wrapper. The null mask is transferred through the Arrow C Data Interface and PyArrow reconstructs the same null positions on import.

## Conversion Path

Inner array types like `IntegerArray<T>` are not exported directly. They must first be wrapped in the `Array` enum via a factory method, then paired with a `Field` to form a `FieldArray`:

```text
IntegerArray<i64>
  -> Array::from_int64(arr)          // wraps in NumericArray::Int64(Arc<IntegerArray<i64>>)
  -> FieldArray::new(field, array)   // pairs with schema metadata
  -> PyArray                         // wrapper implementing PyO3 traits
  -> export_to_c / PyCapsule        // Arrow C Data Interface export
```

The `Array::from_*` factory methods wrap inner arrays in `Arc`, so subsequent clones of the `Array` enum only increment reference counts - the underlying buffer is never copied. The FFI `Holder` struct stores this `Arc<Array>` in its `private_data` field, keeping the buffer alive until the consumer calls `release()`.

## Installation

### Prerequisites

- Python 3.9+
- PyArrow 14+
- Rust nightly (for MinArrow)
- maturin

```bash
pip install maturin pyarrow
```

### Building

```bash
cd pyo3
maturin develop --all-features
```

For a release build:
```bash
maturin build --release --all-features
```

## Usage

### Rust Side

Create PyO3 functions that accept and return PyArrow types:

```rust
use minarrow_pyo3::{PyArray, PyRecordBatch};
use minarrow::{Array, Table, IntegerArray, MaskedArray};
use pyo3::prelude::*;

#[pyfunction]
fn double_values(input: PyArray) -> PyResult<PyArray> {
    let array = input.inner();
    // Process... (example: clone and return)
    Ok(PyArray::from(array.clone()))
}

#[pyfunction]
fn process_batch(input: PyRecordBatch) -> PyResult<PyRecordBatch> {
    let table: Table = input.into();
    // Process the table...
    Ok(PyRecordBatch::from(table))
}

#[pymodule]
fn my_extension(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(double_values, m)?)?;
    m.add_function(wrap_pyfunction!(process_batch, m)?)?;
    Ok(())
}
```

### Python Side

```python
import pyarrow as pa
import my_extension

# Array roundtrip
arr = pa.array([1, 2, 3, 4, 5], type=pa.int32())
result = my_extension.double_values(arr)
print(result)  # PyArrow array

# RecordBatch roundtrip
batch = pa.RecordBatch.from_pydict({
    "id": [1, 2, 3],
    "name": ["alpha", "beta", "gamma"]
})
result = my_extension.process_batch(batch)
print(result)  # PyArrow RecordBatch
```

## Features

- `datetime` - Enable datetime/temporal type support (Date32, Date64, Timestamp, Duration, Time32, Time64)
- `extended_numeric_types` - Enable i8, i16, u8, u16 types
- `extended_categorical` - Enable Categorical8, Categorical16, Categorical64

All features are enabled by default.

## Testing

### Python tests (20 tests, both directions)

Tests all types as Rust -> Python -> Rust roundtrips via PyArrow:

```bash
cd pyo3
maturin develop --all-features
.venv/bin/python tests/test_roundtrip.py
```

### Rust tests (48 tests, both directions)

Tests all types atomically in both directions - Rust -> Python -> Rust roundtrips, and separate Python -> Rust imports via the PyCapsule protocol:

```bash
cd pyo3

# PYO3_PYTHON MUST be an absolute path - relative paths fail in cargo build scripts.
# Run cargo clean first if you previously built against a different Python or venv,
# because PyO3 caches the Python path in build artefacts.

PYO3_PYTHON=$(pwd)/.venv/bin/python \
  PYTHONHOME=/usr \
  PYTHONPATH=$(pwd)/.venv/lib/python3.12/site-packages \
  LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
  cargo run --example atomic_tests \
    --no-default-features \
    --features "datetime,extended_numeric_types,extended_categorical"
```

The `--no-default-features` disables `extension-module`, allowing the binary to link against libpython for standalone execution. `PYO3_PYTHON` must be an absolute path to the venv Python. `PYTHONHOME` must match your system Python prefix. `PYTHONPATH` must include the venv site-packages.

### PyCapsule examples

Python demo showing PyCapsule consumption by PyArrow, nanoarrow, and Polars:

```bash
cd pyo3
maturin develop --all-features
.venv/bin/python examples/pycapsule_demo.py
```

Rust demo showing PyCapsule export and import with an embedded Python interpreter:

```bash
cd pyo3

PYO3_PYTHON=$(pwd)/.venv/bin/python \
  PYTHONHOME=/usr \
  PYTHONPATH=$(pwd)/.venv/lib/python3.12/site-packages \
  LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
  cargo run --example pycapsule_exchange \
    --no-default-features \
    --features "datetime,extended_numeric_types,extended_categorical"
```

## Architecture

The bindings use two exchange protocols:

1. **Arrow PyCapsule Interface** - the standard `__arrow_c_array__` / `__arrow_c_stream__` protocol. Import functions try this first. Works with any Arrow-compatible Python library.

2. **`_export_to_c`** - PyArrow-specific fallback using raw pointer integers for older PyArrow versions.

Memory is managed through Arc reference counting. The Arrow release callbacks ensure the Rust-side buffers remain alive until the consumer is done with them.

## Copy Semantics

### Zero-copy

All primary data buffers are transferred without copying in both directions. This applies to all export paths, single array imports, ChunkedArray chunk imports, and RecordBatch/Table column imports via both the PyCapsule stream and legacy `_import_from_c` paths.

### Copied

The following are copied during import because they require structural transformation between MinArrow and Arrow representations:

- **Null bitmasks** — reconstructed into MinArrow's `Bitmask` type on import. These are small: ceil(N/8) bytes for N elements.
- **String offsets** — Minarrow currently uses `Vec64<T>` rather than `Buffer<T>` for storing offsets. This will be rectified in a future upgrade to support zero-copy, and is a temporary hangover from an earlier data model.
- **Utf8View strings** — Arrow's Utf8View layout uses 16-byte view structs and variadic data buffers, which is structurally incompatible with MinArrow's contiguous offsets+data string layout. The entire string content is copied and reassembled into a standard `StringArray<u32>`. See [Utf8View import](#utf8view-import) above for details.
- **Categorical dictionary strings** — Arrow stores dictionaries as contiguous offsets+data; MinArrow stores them as `Vec64<String>` with individual heap allocations (as for a categorical data use case, a relatively small number of categories is the norm). The integer codes buffer is zero-copy, which is the instances within the (potentially large) dataset.

- **Field metadata** — names, types, and flags are lightweight and always copied.

## License

Apache-2.0
