#!/usr/bin/env python3
"""Test roundtrip conversions between PyArrow and MinArrow.
These test Python -> Rust -> Python.
"""

import pyarrow as pa
import minarrow_pyo3 as ma

print("=" * 50)
print("MinArrow <-> PyArrow Roundtrip Tests")
print("=" * 50)

# Test 1: Integer Array roundtrip
print("\nTest 1: Integer Array Roundtrip")
print("-" * 40)
arr = pa.array([1, 2, 3, 4, 5], type=pa.int32())
print(f"  Input:  {arr.to_pylist()}")
result = ma.echo_array(arr)
print(f"  Output: {result.to_pylist()}")
print(f"  Info:   {ma.array_info(arr)}")
assert arr.to_pylist() == result.to_pylist(), "Integer array mismatch!"
print("  ✓ PASSED")

# Test 2: Float Array roundtrip
print("\nTest 2: Float Array Roundtrip")
print("-" * 40)
arr = pa.array([1.1, 2.2, 3.3, 4.4], type=pa.float64())
print(f"  Input:  {arr.to_pylist()}")
result = ma.echo_array(arr)
print(f"  Output: {result.to_pylist()}")
assert arr.to_pylist() == result.to_pylist(), "Float array mismatch!"
print("  ✓ PASSED")

# Test 3: String Array roundtrip
print("\nTest 3: String Array Roundtrip")
print("-" * 40)
arr = pa.array(["hello", "world", "from", "minarrow"], type=pa.string())
print(f"  Input:  {arr.to_pylist()}")
result = ma.echo_array(arr)
print(f"  Output: {result.to_pylist()}")
assert arr.to_pylist() == result.to_pylist(), "String array mismatch!"
print("  ✓ PASSED")

# Test 4: Boolean Array roundtrip
print("\nTest 4: Boolean Array Roundtrip")
print("-" * 40)
arr = pa.array([True, False, True, False], type=pa.bool_())
print(f"  Input:  {arr.to_pylist()}")
result = ma.echo_array(arr)
print(f"  Output: {result.to_pylist()}")
assert arr.to_pylist() == result.to_pylist(), "Boolean array mismatch!"
print("  ✓ PASSED")

# Test 5: RecordBatch roundtrip
print("\nTest 5: RecordBatch Roundtrip")
print("-" * 40)
batch = pa.RecordBatch.from_pydict({
    "id": pa.array([100, 200, 300], type=pa.int64()),
    "name": pa.array(["alpha", "beta", "gamma"], type=pa.string()),
})
print(f"  Input:  {batch.num_rows} rows, {batch.num_columns} cols")
print(f"  Info:   {ma.batch_info(batch)}")
result = ma.echo_batch(batch)
print(f"  Output: {result.num_rows} rows, {result.num_columns} cols")
assert batch.num_rows == result.num_rows, "Row count mismatch!"
assert batch.num_columns == result.num_columns, "Column count mismatch!"
print("  ✓ PASSED")

# Test 6: Array with nulls
print("\nTest 6: Array with Nulls")
print("-" * 40)
arr = pa.array([1, None, 3, None, 5], type=pa.int32())
print(f"  Input:  {arr.to_pylist()}")
print(f"  Info:   {ma.array_info(arr)}")
result = ma.echo_array(arr)
print(f"  Output: {result.to_pylist()}")
assert arr.to_pylist() == result.to_pylist(), "Nullable array mismatch!"
print("  ✓ PASSED")

# Test 7: Int8 Array roundtrip
print("\nTest 7: Int8 Array Roundtrip")
print("-" * 40)
arr = pa.array([1, -128, 127, 0], type=pa.int8())
print(f"  Input:  {arr.to_pylist()}")
result = ma.echo_array(arr)
print(f"  Output: {result.to_pylist()}")
assert arr.to_pylist() == result.to_pylist(), "Int8 array mismatch!"
print("  ✓ PASSED")

# Test 8: UInt8 Array roundtrip
print("\nTest 8: UInt8 Array Roundtrip")
print("-" * 40)
arr = pa.array([0, 128, 255], type=pa.uint8())
print(f"  Input:  {arr.to_pylist()}")
result = ma.echo_array(arr)
print(f"  Output: {result.to_pylist()}")
assert arr.to_pylist() == result.to_pylist(), "UInt8 array mismatch!"
print("  ✓ PASSED")

# Test 9: Int16 Array roundtrip
print("\nTest 9: Int16 Array Roundtrip")
print("-" * 40)
arr = pa.array([1, -32768, 32767], type=pa.int16())
print(f"  Input:  {arr.to_pylist()}")
result = ma.echo_array(arr)
print(f"  Output: {result.to_pylist()}")
assert arr.to_pylist() == result.to_pylist(), "Int16 array mismatch!"
print("  ✓ PASSED")

# Test 10: UInt16 Array roundtrip
print("\nTest 10: UInt16 Array Roundtrip")
print("-" * 40)
arr = pa.array([0, 32768, 65535], type=pa.uint16())
print(f"  Input:  {arr.to_pylist()}")
result = ma.echo_array(arr)
print(f"  Output: {result.to_pylist()}")
assert arr.to_pylist() == result.to_pylist(), "UInt16 array mismatch!"
print("  ✓ PASSED")

# Test 11: Large String Array roundtrip
print("\nTest 11: Large String Array Roundtrip")
print("-" * 40)
arr = pa.array(["hello", "large", "strings"], type=pa.large_string())
print(f"  Input:  {arr.to_pylist()}")
result = ma.echo_array(arr)
print(f"  Output: {result.to_pylist()}")
assert arr.to_pylist() == result.to_pylist(), "Large string array mismatch!"
print("  ✓ PASSED")

# Test 12: Dictionary/Categorical Array roundtrip
print("\nTest 12: Dictionary Array Roundtrip")
print("-" * 40)
arr = pa.array(["cat", "dog", "cat", "bird", "dog"]).dictionary_encode()
print(f"  Input:  {arr.to_pylist()}")
result = ma.echo_array(arr)
print(f"  Output: {result.to_pylist()}")
assert arr.to_pylist() == result.to_pylist(), "Dictionary array mismatch!"
print("  ✓ PASSED")

# Test 13: Date32 Array roundtrip
print("\nTest 13: Date32 Array Roundtrip")
print("-" * 40)
arr = pa.array([0, 1, 100, 19000], type=pa.date32())  # days since epoch
print(f"  Input:  {arr.to_pylist()}")
result = ma.echo_array(arr)
print(f"  Output: {result.to_pylist()}")
assert arr.to_pylist() == result.to_pylist(), "Date32 array mismatch!"
print("  ✓ PASSED")

# Test 14: Date64 Array roundtrip
print("\nTest 14: Date64 Array Roundtrip")
print("-" * 40)
arr = pa.array([0, 86400000, 172800000], type=pa.date64())  # ms since epoch
print(f"  Input:  {arr.to_pylist()}")
result = ma.echo_array(arr)
print(f"  Output: {result.to_pylist()}")
assert arr.to_pylist() == result.to_pylist(), "Date64 array mismatch!"
print("  ✓ PASSED")

# Test 15: Timestamp Array roundtrip
print("\nTest 15: Timestamp Array Roundtrip")
print("-" * 40)
arr = pa.array([1000000, 2000000, 3000000], type=pa.timestamp('us'))
print(f"  Input:  {arr.to_pylist()}")
result = ma.echo_array(arr)
print(f"  Output: {result.to_pylist()}")
assert arr.to_pylist() == result.to_pylist(), "Timestamp array mismatch!"
assert result.type == arr.type, f"Timestamp type mismatch: expected {arr.type}, got {result.type}"
print("  ✓ PASSED")

# Test 16: Duration Array roundtrip
print("\nTest 16: Duration Array Roundtrip")
print("-" * 40)
arr = pa.array([1000000, 2000000, 3000000], type=pa.duration('us'))
print(f"  Input:  {arr.to_pylist()}")
result = ma.echo_array(arr)
print(f"  Output: {result.to_pylist()}")
assert arr.to_pylist() == result.to_pylist(), "Duration array mismatch!"
assert result.type == arr.type, f"Duration type mismatch: expected {arr.type}, got {result.type}"
print("  ✓ PASSED")

# Test 17: Timestamp with timezone roundtrip
print("\nTest 17: Timestamp with Timezone Roundtrip")
print("-" * 40)
arr = pa.array([1000000, 2000000, 3000000], type=pa.timestamp('us', tz='UTC'))
print(f"  Input:  {arr.to_pylist()}")
result = ma.echo_array(arr)
print(f"  Output: {result.to_pylist()}")
assert arr.to_pylist() == result.to_pylist(), "Timestamp with tz array mismatch!"
assert result.type == arr.type, f"Timestamp with tz type mismatch: expected {arr.type}, got {result.type}"
print("  ✓ PASSED")

# Test 18: PyArrow Table roundtrip (multiple batches)
print("\nTest 18: Table Roundtrip")
print("-" * 40)
batch1 = pa.RecordBatch.from_pydict({
    "id": pa.array([1, 2, 3], type=pa.int64()),
    "name": pa.array(["a", "b", "c"], type=pa.string()),
})
batch2 = pa.RecordBatch.from_pydict({
    "id": pa.array([4, 5], type=pa.int64()),
    "name": pa.array(["d", "e"], type=pa.string()),
})
table = pa.Table.from_batches([batch1, batch2])
print(f"  Input:  {table.num_rows} rows, {table.num_columns} cols")
print(f"  Info:   {ma.table_info(table)}")
result = ma.echo_table(table)
print(f"  Output: {result.num_rows} rows, {result.num_columns} cols")
assert table.num_rows == result.num_rows, "Table row count mismatch!"
assert table.num_columns == result.num_columns, "Table column count mismatch!"
# Verify data content
assert table.to_pydict() == result.to_pydict(), "Table data mismatch!"
print("  ✓ PASSED")

# Test 19: ChunkedArray roundtrip
print("\nTest 19: ChunkedArray Roundtrip")
print("-" * 40)
arr1 = pa.array([1, 2, 3], type=pa.int32())
arr2 = pa.array([4, 5, 6, 7], type=pa.int32())
chunked = pa.chunked_array([arr1, arr2])
print(f"  Input:  {len(chunked)} elements, {chunked.num_chunks} chunks")
print(f"  Info:   {ma.chunked_info(chunked)}")
result = ma.echo_chunked(chunked)
print(f"  Output: {len(result)} elements, {result.num_chunks} chunks")
assert len(chunked) == len(result), "ChunkedArray length mismatch!"
assert chunked.to_pylist() == result.to_pylist(), "ChunkedArray data mismatch!"
print("  ✓ PASSED")

# Test 20: ChunkedArray with strings
print("\nTest 20: ChunkedArray with Strings Roundtrip")
print("-" * 40)
arr1 = pa.array(["hello", "world"], type=pa.string())
arr2 = pa.array(["foo", "bar", "baz"], type=pa.string())
chunked = pa.chunked_array([arr1, arr2])
print(f"  Input:  {len(chunked)} elements, {chunked.num_chunks} chunks")
result = ma.echo_chunked(chunked)
print(f"  Output: {len(result)} elements, {result.num_chunks} chunks")
assert chunked.to_pylist() == result.to_pylist(), "String ChunkedArray mismatch!"
print("  ✓ PASSED")

# Test 21: Decimal128 Array roundtrip
print("\nTest 21: Decimal128 Array Roundtrip")
print("-" * 40)
import decimal
arr = pa.array([decimal.Decimal("123.45"), decimal.Decimal("678.90"), None], type=pa.decimal128(10, 2))
print(f"  Input:  {arr.to_pylist()}")
result = ma.echo_array(arr)
print(f"  Output: {result.to_pylist()}")
assert arr.to_pylist() == result.to_pylist(), "Decimal128 array mismatch!"
assert result.type == arr.type, f"Decimal128 type mismatch: expected {arr.type}, got {result.type}"
print("  ✓ PASSED")

# Test 22: Decimal128 Array roundtrip with high precision
print("\nTest 22: Decimal128 High Precision Roundtrip")
print("-" * 40)
arr = pa.array(
    [decimal.Decimal("12345678901234567890.1234567890"), None, decimal.Decimal("-9876543210.0987654321")],
    type=pa.decimal128(38, 10),
)
print(f"  Input:  {arr.to_pylist()}")
result = ma.echo_array(arr)
print(f"  Output: {result.to_pylist()}")
assert arr.to_pylist() == result.to_pylist(), "Decimal128 high precision array mismatch!"
assert result.type == arr.type, f"Decimal128 type mismatch: expected {arr.type}, got {result.type}"
print("  ✓ PASSED")

# Test 23: Decimal128 in RecordBatch roundtrip
print("\nTest 23: Decimal128 in RecordBatch Roundtrip")
print("-" * 40)
batch = pa.RecordBatch.from_pydict({
    "id": pa.array([1, 2, 3], type=pa.int64()),
    "price": pa.array(
        [decimal.Decimal("19.99"), decimal.Decimal("29.99"), decimal.Decimal("39.99")],
        type=pa.decimal128(10, 2),
    ),
})
print(f"  Input:  {batch.num_rows} rows, {batch.num_columns} cols")
result = ma.echo_batch(batch)
print(f"  Output: {result.num_rows} rows, {result.num_columns} cols")
assert batch.column("id").to_pylist() == result.column("id").to_pylist(), "ID column mismatch!"
assert batch.column("price").to_pylist() == result.column("price").to_pylist(), "Price column mismatch!"
assert result.schema.field("price").type == pa.decimal128(10, 2), "Decimal type not preserved in RecordBatch!"
print("  ✓ PASSED")

print("\n" + "=" * 50)
print("All tests PASSED!")
print("=" * 50)
