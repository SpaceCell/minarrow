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

//! # FFI Module for minarrow-pyo3
//!
//! Provides conversion functions between MinArrow and PyArrow via the Arrow C Data Interface
//! and the Arrow PyCapsule Interface.
//!
//! ## Conversion Protocols
//!
//! Two protocols are supported, tried in order:
//!
//! 1. **Arrow PyCapsule Interface** - the standard `__arrow_c_array__` / `__arrow_c_stream__`
//!    protocol. Works with any Arrow-compatible Python library: PyArrow, Polars, DuckDB,
//!    nanoarrow, pandas with ArrowDtype, etc.
//!
//! 2. **Legacy `_export_to_c`** - PyArrow-specific private API using raw pointer integers.
//!    Used as a fallback when PyCapsule is unavailable.
//! 
//! Minarrow additionally supports the 'export_to_c' API natively from the base crate, as it 
//! allows moving data zero copy between any language that implements Apache Arrow with FFI.
//!
//! ## Conversion Path
//!
//! Inner array types such as `IntegerArray<T>` or `FloatArray<T>` are not exported directly.
//! They must first be wrapped in the `Array` enum via a factory method, then paired with
//! a `Field` to form a `FieldArray`:
//!
//! ```text
//! IntegerArray<i64>
//!   -> Array::from_int64(arr)          // wraps in NumericArray::Int64(Arc<IntegerArray<i64>>)
//!   -> FieldArray::new(field, array)   // pairs with schema metadata
//!   -> PyArray                         // wrapper implementing PyO3 traits
//!   -> export_to_c / PyCapsule        // Arrow C Data Interface export
//! ```
//! This is because *Minarrow* keeps these types light, but the `Field` dresses them up with additional
//! metadata required to conform to the *Apache Arrow* specification.
//! 
//! 
//! The `Array::from_*` factory methods wrap inner arrays in `Arc`, so subsequent clones
//! of the `Array` enum only increment reference counts - the underlying buffer is never
//! copied. The FFI `Holder` struct stores this `Arc<Array>` in its `private_data` field,
//! keeping the buffer alive until the consumer calls `release()`.
//!
//! ## Container Type Mappings
//!
//! | MinArrow | PyArrow | Wrapper | Protocol |
//! |----------|---------|---------|----------|
//! | [`Array`] | `pa.Array` | [`PyArray`] | `__arrow_c_array__` |
//! | [`Table`] | `pa.RecordBatch` | [`PyRecordBatch`] | `__arrow_c_stream__` (one batch) |
//! | [`SuperTable`] | `pa.Table` | [`PyTable`] | `__arrow_c_stream__` (multiple batches) |
//! | [`SuperArray`] | `pa.ChunkedArray` | [`PyChunkedArray`] | `__arrow_c_stream__` (one array per chunk) |
//!
//! [`Array`]: minarrow::Array
//! [`Table`]: minarrow::Table
//! [`SuperTable`]: minarrow::SuperTable
//! [`SuperArray`]: minarrow::SuperArray
//! [`PyArray`]: crate::types::PyArray
//! [`PyRecordBatch`]: crate::types::PyRecordBatch
//! [`PyTable`]: crate::types::PyTable
//! [`PyChunkedArray`]: crate::types::PyChunkedArray
//!
//! ## Array Data Type Mappings
//!
//! Each inner array type maps 1:1 to a specific PyArrow typed array. The Arrow C Data
//! Interface preserves the schema metadata so PyArrow reconstructs the correct type
//! on import.
//!
//! ### Numeric types
//!
//! | MinArrow inner type | `Array` enum path | Arrow format | PyArrow type |
//! |---------------------|-------------------|--------------|--------------|
//! | `IntegerArray<i32>` | `NumericArray::Int32` | `i` | `pa.Int32Array` |
//! | `IntegerArray<i64>` | `NumericArray::Int64` | `l` | `pa.Int64Array` |
//! | `IntegerArray<u32>` | `NumericArray::UInt32` | `I` | `pa.UInt32Array` |
//! | `IntegerArray<u64>` | `NumericArray::UInt64` | `L` | `pa.UInt64Array` |
//! | `FloatArray<f32>` | `NumericArray::Float32` | `f` | `pa.FloatArray` |
//! | `FloatArray<f64>` | `NumericArray::Float64` | `g` | `pa.DoubleArray` |
//!
//! ### Extended numeric types (feature `extended_numeric_types`)
//!
//! | MinArrow inner type | `Array` enum path | Arrow format | PyArrow type |
//! |---------------------|-------------------|--------------|--------------|
//! | `IntegerArray<i8>` | `NumericArray::Int8` | `c` | `pa.Int8Array` |
//! | `IntegerArray<i16>` | `NumericArray::Int16` | `s` | `pa.Int16Array` |
//! | `IntegerArray<u8>` | `NumericArray::UInt8` | `C` | `pa.UInt8Array` |
//! | `IntegerArray<u16>` | `NumericArray::UInt16` | `S` | `pa.UInt16Array` |
//!
//! ### Boolean
//!
//! | MinArrow inner type | `Array` enum path | Arrow format | PyArrow type |
//! |---------------------|-------------------|--------------|--------------|
//! | `BooleanArray` | `Array::BooleanArray` | `b` | `pa.BooleanArray` |
//!
//! ### Text types
//!
//! | MinArrow inner type | `Array` enum path | Arrow format | PyArrow type |
//! |---------------------|-------------------|--------------|--------------|
//! | `StringArray<u32>` | `TextArray::String32` | `u` | `pa.StringArray` |
//! | `StringArray<u64>` | `TextArray::String64` | `U` | `pa.LargeStringArray` |
//!
//! `StringArray<u64>` is always available in minarrow-pyo3 via the `large_string` dependency feature.
//!
//! ### Temporal types (feature `datetime`)
//!
//! MinArrow stores temporal data in `DatetimeArray<i32>` or `DatetimeArray<i64>` with a
//! `TimeUnit` discriminator. The Arrow type is determined by the `ArrowType` in the `Field`,
//! not the storage type alone.
//!
//! | MinArrow inner type | `ArrowType` | Arrow format | PyArrow type |
//! |---------------------|-------------|--------------|--------------|
//! | `DatetimeArray<i32>` | `Date32` | `tdD` | `pa.Date32Array` |
//! | `DatetimeArray<i64>` | `Date64` | `tdm` | `pa.Date64Array` |
//! | `DatetimeArray<i32>` | `Time32(Seconds)` | `tts` | `pa.Time32Array` |
//! | `DatetimeArray<i32>` | `Time32(Milliseconds)` | `ttm` | `pa.Time32Array` |
//! | `DatetimeArray<i64>` | `Time64(Microseconds)` | `ttu` | `pa.Time64Array` |
//! | `DatetimeArray<i64>` | `Time64(Nanoseconds)` | `ttn` | `pa.Time64Array` |
//! | `DatetimeArray<i64>` | `Timestamp(Seconds, tz)` | `tss:tz` | `pa.TimestampArray` |
//! | `DatetimeArray<i64>` | `Timestamp(Milliseconds, tz)` | `tsm:tz` | `pa.TimestampArray` |
//! | `DatetimeArray<i64>` | `Timestamp(Microseconds, tz)` | `tsu:tz` | `pa.TimestampArray` |
//! | `DatetimeArray<i64>` | `Timestamp(Nanoseconds, tz)` | `tsn:tz` | `pa.TimestampArray` |
//! | `DatetimeArray<i32>` | `Duration32(Seconds)` | `tDs` | `pa.DurationArray` |
//! | `DatetimeArray<i32>` | `Duration32(Milliseconds)` | `tDm` | `pa.DurationArray` |
//! | `DatetimeArray<i64>` | `Duration64(Microseconds)` | `tDu` | `pa.DurationArray` |
//! | `DatetimeArray<i64>` | `Duration64(Nanoseconds)` | `tDn` | `pa.DurationArray` |
//!
//! Timezone metadata for timestamps is preserved via the Arrow schema format string.
//! When a timezone is present, PyArrow reconstructs it as `pyarrow.timestamp('us', tz=...)`.
//!
//! ### Categorical / dictionary types
//!
//! | MinArrow inner type | `Array` enum path | Arrow format | PyArrow type |
//! |---------------------|-------------------|--------------|--------------|
//! | `CategoricalArray<u32>` | `TextArray::Categorical32` | dictionary(int32, utf8) | `pa.DictionaryArray` |
//!
//! With feature `extended_categorical` + `extended_numeric_types`:
//!
//! | MinArrow inner type | `Array` enum path | Arrow format | PyArrow type |
//! |---------------------|-------------------|--------------|--------------|
//! | `CategoricalArray<u8>` | `TextArray::Categorical8` | dictionary(int8, utf8) | `pa.DictionaryArray` |
//! | `CategoricalArray<u16>` | `TextArray::Categorical16` | dictionary(int16, utf8) | `pa.DictionaryArray` |
//!
//! With feature `extended_categorical`:
//!
//! | MinArrow inner type | `Array` enum path | Arrow format | PyArrow type |
//! |---------------------|-------------------|--------------|--------------|
//! | `CategoricalArray<u64>` | `TextArray::Categorical64` | dictionary(int64, utf8) | `pa.DictionaryArray` |
//!
//! Dictionary-encoded arrays are exported as Arrow dictionary arrays where the indices
//! correspond to the categorical key size and the values are utf8 strings.
//!
//! For categorical types, the integer buffer is zero-copy but we clone the (finite) dictionary categories.
//! Unless you have a very large unique category count, this should not cause performance issues.
//! 
//! ## Nullability
//!
//! All array types support null values via MinArrow's `MaskedArray` wrapper. When an
//! array contains nulls, the validity bitmap is transferred through the Arrow C Data
//! Interface. PyArrow reconstructs the same null positions on import.
//!
//! ## Ownership Model
//!
//! The FFI layer is zero-copy for buffer data. Inner arrays like `IntegerArray<T>` are
//! stored behind `Arc` inside the `Array` enum variants, e.g.:
//!
//! ```text
//! NumericArray::Int64(Arc<IntegerArray<i64>>)
//! ```
//!
//! When exporting, `Array::clone()` increments the inner Arc refcount without copying
//! the buffer. The `export_to_c` function stores this `Arc<Array>` in a `Holder` struct
//! behind the `ArrowArray.private_data` pointer. The buffer pointers in `ArrowArray.buffers`
//! point directly into the original `Vec64<T>` allocation.
//!
//! The data remains alive until the consumer calls `release()`, which drops the `Holder`
//! and decrements the Arc refcount. For PyCapsule exports, an additional destructor on the
//! capsule calls `release()` if the capsule is garbage collected without being consumed.
//!
//! ## Modules
//!
//! - [`to_py`] - MinArrow to Python conversion (export)
//! - [`to_rust`] - Python to MinArrow conversion (import)

pub mod to_py;
pub mod to_rust;
