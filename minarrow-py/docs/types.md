# Types and schemas

Minarrow separates broad type categories from the complete Arrow logical type.

Every `Array` exposes:

* `.dtype` — the Minarrow type family
* `.dtype.group` — the broader `TypeClass`
* `.bit_width` — the physical width of the stored values or indices
* `.arrow_type` — the complete Arrow logical type

```python
import minarrow as ma

array = ma.Array([1, 2, 3])

array.dtype        # DType.Integer
array.dtype.group  # TypeClass.Numeric
array.bit_width    # 64
array.arrow_type   # Int64
```

## Type taxonomy

| `.dtype`      | Values                                      |
| ------------- | ------------------------------------------- |
| `Integer`     | Signed and unsigned integers                |
| `Float`       | 32-bit and 64-bit floating-point values     |
| `String`      | UTF-8 strings with 32-bit or 64-bit offsets |
| `Categorical` | Dictionary-encoded strings                  |
| `Datetime`    | Dates, times, timestamps and durations      |
| `Boolean`     | Bit-packed Boolean values                   |
| `Null`        | All-null arrays                             |

The corresponding `TypeClass` groups are:

| `TypeClass` | Included dtypes         |
| ----------- | ----------------------- |
| `Numeric`   | `Integer`, `Float`      |
| `Text`      | `String`, `Categorical` |
| `Temporal`  | `Datetime`              |
| `Boolean`   | `Boolean`               |
| `Null`      | `Null`                  |

`TypeClass` is useful when code needs to handle related types together:

```python
if array.dtype.group is ma.TypeClass.Numeric:
    process_numeric(array)
```

Use `.arrow_type` when the exact logical type matters, such as distinguishing `Int32` from `Int64` or inspecting timestamp units and time zones.

## Supported types

Minarrow exposes one Python `Array` class. Its `.arrow_type` identifies the logical Arrow type and determines the physical representation.

Some types depend on build features.

| `.dtype`      | `.arrow_type`                                     | Physical representation | PyArrow equivalent                      |
| ------------- | ------------------------------------------------- | ----------------------- | --------------------------------------- |
| `Integer`     | `Int8()` ¹                                        | `i8`, 8-bit             | `pa.int8()`                             |
| `Integer`     | `Int16()` ¹                                       | `i16`, 16-bit           | `pa.int16()`                            |
| `Integer`     | `Int32()`                                         | `i32`, 32-bit           | `pa.int32()`                            |
| `Integer`     | `Int64()`                                         | `i64`, 64-bit           | `pa.int64()`                            |
| `Integer`     | `UInt8()` ¹                                       | `u8`, 8-bit             | `pa.uint8()`                            |
| `Integer`     | `UInt16()` ¹                                      | `u16`, 16-bit           | `pa.uint16()`                           |
| `Integer`     | `UInt32()`                                        | `u32`, 32-bit           | `pa.uint32()`                           |
| `Integer`     | `UInt64()`                                        | `u64`, 64-bit           | `pa.uint64()`                           |
| `Float`       | `Float32()`                                       | `f32`, 32-bit           | `pa.float32()`                          |
| `Float`       | `Float64()`                                       | `f64`, 64-bit           | `pa.float64()`                          |
| `Boolean`     | `Boolean()`                                       | Bit-packed              | `pa.bool_()`                            |
| `String`      | `String()`                                        | 32-bit offsets          | `pa.string()`                           |
| `String`      | `LargeString()` ²                                 | 64-bit offsets          | `pa.large_string()`                     |
| `Categorical` | `Dictionary(index=CategoricalIndexType.UInt8)` ³  | `u8` indices            | `pa.dictionary(pa.uint8(), pa.utf8())`  |
| `Categorical` | `Dictionary(index=CategoricalIndexType.UInt16)` ⁴ | `u16` indices           | `pa.dictionary(pa.uint16(), pa.utf8())` |
| `Categorical` | `Dictionary(index=CategoricalIndexType.UInt32)`   | `u32` indices           | `pa.dictionary(pa.uint32(), pa.utf8())` |
| `Categorical` | `Dictionary(index=CategoricalIndexType.UInt64)` ⁴ | `u64` indices           | `pa.dictionary(pa.uint64(), pa.utf8())` |
| `Datetime`    | `Date32()` ⁵                                      | `i32` days              | `pa.date32()`                           |
| `Datetime`    | `Date64()` ⁵                                      | `i64` milliseconds      | `pa.date64()`                           |
| `Datetime`    | `Time32(unit)` ⁵                                  | `i32`                   | `pa.time32(unit)`                       |
| `Datetime`    | `Time64(unit)` ⁵                                  | `i64`                   | `pa.time64(unit)`                       |
| `Datetime`    | `Timestamp(unit, tz)` ⁵                           | `i64`                   | `pa.timestamp(unit, tz)`                |
| `Datetime`    | `Duration64(unit)` ⁵                              | `i64`                   | `pa.duration(unit)`                     |
| `Null`        | `Null()`                                          | No value buffer         | `pa.null()`                             |

If you are compiling from source, the following Rust feature-flags are required:
¹ = Requires `extended_numeric_types`.
² = Requires `large_string`, enabled by default.
³ = Requires `default_categorical_8`.
⁴ = Requires `extended_categorical`. `cat32` is available under `default_categorical_8`.
⁵ = Requires `datetime`, enabled by default.

The default Python package on Pypi.org includes these by default.

The Arrow `Interval` type is represented in the type system but does not have a dedicated Minarrow array implementation.

## Container types

Minarrow provides four columnar containers:

| Minarrow type  | Arrow equivalent  | Description                                  |
| -------------- | ----------------- | -------------------------------------------- |
| `Array`        | `pa.Array`        | One contiguous logical column                |
| `Table`        | `pa.RecordBatch`  | One batch of equal-length columns            |
| `ChunkedArray` | `pa.ChunkedArray` | One logical column stored as multiple arrays |
| `ChunkedTable` | `pa.Table`        | One logical table stored as multiple batches |

A `Table` represents one record batch rather than an arbitrarily chunked dataset. Use `ChunkedTable` when data arrives or is stored as multiple batches.

## Why use chunked containers

Chunked containers preserve existing allocation boundaries.

This avoids combining multiple arrays or record batches into one contiguous allocation when consolidation is unnecessary or expensive. The trade-off is additional bookkeeping and potentially less contiguous memory access.

Chunking is useful when:

* Data arrives incrementally from a stream or transport
* Batches correspond to processing windows
* Files are read in row groups or record batches
* A dataset is larger than available memory
* Another library already uses a chunked internal representation

For example, a dataframe library may expose one logical column while retaining several Arrow arrays underneath it.

Chunking does not itself provide out-of-core execution. It allows consumers that support incremental or out-of-core processing to operate on one portion of the dataset at a time.

## ArrowType

`ArrowType` mirrors the `minarrow::ArrowType` enum from the Rust core.

Because many enum variants carry values, they are constructed as callable Python variants:

```python
ma.ArrowType.Int64()
ma.ArrowType.Float64()
ma.ArrowType.String()

ma.ArrowType.Timestamp(
    ma.TimeUnit.Microseconds,
    "UTC",
)

ma.ArrowType.Time32(
    ma.TimeUnit.Milliseconds,
)

ma.ArrowType.Dictionary(
    index=ma.CategoricalIndexType.UInt32,
)
```

The available variants follow the features enabled when the Python extension was built.

## Fields

A `Field` describes one column. It contains:

* A name
* An Arrow logical type
* A nullability flag
* Optional metadata

```python
amount = ma.Field(
    "amount",
    ma.ArrowType.Float64(),
    nullable=False,
    metadata={"unit": "USD"},
)

amount.name        # "amount"
amount.arrow_type  # Float64
amount.dtype       # DType.Float
amount.nullable    # False
amount.metadata    # {"unit": "USD"}
```

Field metadata is stored as string key-value pairs and is preserved through compatible Arrow interchange paths.

## Schemas

A `Schema` is an ordered collection of fields with optional schema-level metadata.

```python
schema = ma.Schema(
    [
        ma.Field(
            "id",
            ma.ArrowType.Int64(),
            nullable=False,
        ),
        amount,
        ma.Field(
            "symbol",
            ma.ArrowType.String(),
        ),
    ],
    metadata={"dataset": "prices"},
)

schema.names        # ["id", "amount", "symbol"]
len(schema)         # 3
schema.metadata     # {"dataset": "prices"}

schema["amount"]    # Lookup by name
schema[0]           # Lookup by position
```

Every `Table` exposes its schema:

```python
table = ma.Table(
    {
        "id": [1, 2],
        "price": [1.5, 2.5],
    }
)

table.schema
# Schema([id: Int64, price: Float64])
```

For a `ChunkedTable`, the schema applies to every batch. A user-provided schema may also carry metadata that is not present on the individual batches.
