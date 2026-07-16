# Ecosystem interoperability

Minarrow exchanges columnar data with other Python libraries through the Arrow PyCapsule interface.

A runnable example is available at [`examples/ecosystem.py`](https://github.com/pbower/minarrow/blob/main/minarrow-py/examples/ecosystem.py). It detects the installed libraries and skips integrations that are unavailable.

## Arrow PyCapsule interface

The Arrow PyCapsule interface allows compatible libraries to exchange Arrow schemas, arrays and streams through the Arrow C Data and C Stream interfaces.

Objects expose one or more of the following methods:

* `__arrow_c_schema__` — exports an Arrow data type or schema
* `__arrow_c_array__` — exports an array or record batch
* `__arrow_c_stream__` — exports a stream of record batches

Minarrow `Array` and `Table` objects implement the relevant interfaces. This allows an Arrow-aware consumer to import their buffers without serialising the data through an intermediate format.

```python
import duckdb
import minarrow as ma
import polars as pl

array = ma.Array([1, 2, 3], name="id")
table = ma.Table(
    {
        "id": [1, 2, 3],
        "price": [9.5, 10.0, 11.2],
    },
    name="prices",
)

series = pl.from_arrow(array)
relation = duckdb.sql("SELECT * FROM table")
```

The Minarrow Rust core retains ownership of exported memory for the lifetime required by the receiving object.

Whether an integration remains zero-copy depends on the receiving library, data type and requested operation. A consumer may copy data when converting types, changing alignment, combining chunks or moving data to another device.

The PyCapsule interface is part of the Apache Arrow interoperability specifications. Minarrow implements those interfaces independently and does not require PyArrow to own or manage its buffers.

## DLPack tensor interchange

`NdArray` and `XArray` implement `__dlpack__` and `__dlpack_device__` for CPU
tensor interchange. Their owned arrays and zero-copy selections retain shape
and element strides, including Minarrow's column-major layout. Versioned
DLPack exports mark shared storage read-only; legacy exports copy shared data
because that protocol cannot express read-only memory.

`ChunkedNdArray` does not pretend its multiple allocations are one DLPack
tensor. Its `chunks` are `NdArray` objects, and each chunk implements DLPack
with its own shape, strides, and data pointer. `to_numpy()` consequently
returns one NumPy array per chunk. Call `to_ndarray()` explicitly when one
consolidated allocation is required.

## Supported integrations

Minarrow provides named conversion methods for commonly used libraries.

| Library    | `Array`                           | `Table`                             |
| ---------- | --------------------------------- | ----------------------------------- |
| Polars     | `to_polars` / `from_polars`       | `to_polars` / `from_polars`         |
| DuckDB     | —                                 | `to_duckdb` / `from_duckdb`         |
| DataFusion | —                                 | `to_datafusion` / `from_datafusion` |
| Daft       | —                                 | `to_daft` / `from_daft`             |
| nanoarrow  | `to_nanoarrow` / `from_nanoarrow` | `to_nanoarrow` / `from_nanoarrow`   |
| pandas     | `to_pandas` / `from_pandas`       | `to_pandas` / `from_pandas`         |
| cuDF       | `to_cudf` / `from_cudf`           | `to_cudf` / `from_cudf`             |
| Ibis       | —                                 | `to_ibis` / `from_ibis`             |
| Narwhals   | —                                 | `to_narwhals` / `from_narwhals`     |

```python
polars_series = array.to_polars()
polars_frame = table.to_polars()
duckdb_relation = table.to_duckdb()

restored = ma.Table.from_polars(polars_frame)
```

These methods provide explicit, readable integration points. Libraries that implement the Arrow PyCapsule interface may also work through the generic Arrow import and export methods even when they are not listed above.

## Generic Arrow conversion

Use the generic conversion methods when working with a compatible library that does not have a named Minarrow adapter:

```python
exported = table.to_arrow()
restored = ma.Table.from_arrow(exported)
```

The exact accepted object types depend on the interfaces exposed by the other library.

## Databases through ADBC

[Arrow Database Connectivity](https://arrow.apache.org/adbc/) provides an Arrow-oriented interface for database drivers.

Minarrow tables can be written through an ADBC cursor and query results can be read back into a `Table`:

```python
import adbc_driver_sqlite.dbapi as dbapi
import minarrow as ma

table = ma.Table(
    {
        "id": [1, 2, 3],
        "price": [9.5, 10.0, 11.2],
    },
    name="prices",
)

connection = dbapi.connect()
cursor = connection.cursor()

table.write_adbc(cursor, "prices", mode="create")

cursor.execute("SELECT * FROM prices")
result = ma.Table.read_adbc(cursor)
```

`write_adbc` supports the following modes:

| Mode            | Behaviour                                           |
| --------------- | --------------------------------------------------- |
| `create`        | Creates a new table and fails if it already exists. |
| `append`        | Appends to an existing table.                       |
| `replace`       | Replaces the existing table.                        |
| `create_append` | Creates the table when absent, otherwise appends.   |

The same Minarrow integration can be used with any compatible ADBC driver supported by the installed Python environment.
