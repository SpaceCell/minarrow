# Changelog

All notable changes to **minarrow** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# Minarrow Rust

## 0.17.0 - 2026-08-15

This release:
    - Corrects integer arithmetic rounding semantics for scalar
    - Adds `Display` for all public types
    - Includes a broad set of API additions and performance improvements.

### Numerical result changes

Several arithmetic corrections in this release produce different results for
integer inputs compared to 0.16.x. The affected operations are listed below so
that users can validate downstream expectations.

**Please consider the below carefully before upgrading:**

- **Integer array division now floors toward negative infinity.** Previously the
SIMD and scalar kernel paths used Rust's truncating `/` (toward zero). Both
`Divide` and `FloorDiv` now floor consistently, matching Python and NumPy
semantics. For example, `-7 / 2` at the array kernel level was `-3` and is now
`-4`. Unsigned integer division is unaffected.

- **Scalar integer division returns `Scalar::Float64`.** `Scalar::Int64(-7) /
Scalar::Int64(2)` was `Scalar::Int64(-3)` and is now `Scalar::Float64(-3.5)`.
Division by zero follows IEEE 754 (Inf/NaN) instead of panicking.

- **Integer power with an out-of-range exponent.** Previously an exponent that
did not fit `u32` was silently treated as 0, returning 1. The kernel now panics
for dense arrays and nullifies the lane for masked arrays. Integer power
overflow wraps consistently with add/subtract/multiply.

### Added
- `Display` implementation for `Value` and every constituent type that
  previously lacked one: `Matrix`, `NdArray`, `NdArrayV`, `SuperNdArray`,
  `SuperNdArrayV`, `SuperArrayV`, `SuperTableV`, `XArray` and `Cube`. Each
  type's output matches the register of the existing `Table` display, with
  `MAX_PREVIEW` truncation for large data.
- `Bitmask` iteration kernels (`iter_window_bits`) with a SIMD fast path on
  x86-64. `Bitmask::iter_set()` and `iter_cleared()` now run in
  O(matching_bits) rather than O(total_bits).
- `BitmaskV` gains `get_unchecked` and now derives `Copy`.
- `as_tuple_ref()` on `BooleanArrayV`, `TextArrayV` and `TemporalArrayV` for
  borrow-only access without `Arc::clone`.
- `TemporalArray::time_unit()` and `TemporalArrayV::time_unit()` returning the
  backing `TimeUnit`.
- `TryFrom<Vec<Array>> for Array` for end-to-end concatenation.
- `TryFrom<Array> for Scalar` for single-element extraction.
- `Array::set_range(range, scalar)` and `Array::set(idx, Scalar::Null)` for
  in-place range fills and null masking.
- Broadened `TryFrom<Value>` and `From` impls across `BoxValue`, `ArcValue`,
  `SuperArray`, `SuperNdArray`, `SuperTable`, `NdArrayV`, `Cube` and their view
  counterparts. Wrapped values now unwrap transparently.
- `ArrowType::minarrow_category_label()` returning high-level type categories.
- `CategoricalIndexType::arrow_index_name()` for the Arrow index type string.
- `Cube::from_table(table, col, name)` for splitting a table into a cube along
  a column's distinct values.

### Changed
- `ArrayV::gather_indices` rewritten for throughput: single variant match per
  gather, bulk null mask transfer, x86-64 prefetch hints, and `TableV` gather
  now delegates per-column to avoid duplicated per-type logic.
- `Cube::from_table` grouping uses hash-based value equality via
  `hash_element_at` and `value_eq` instead of per-row string allocation.
- `cube` feature now implies `views`, `select` and `hash`.
- `Cube` has had various semantic improvements to its operation.
- `libc` dependency is now optional, gated behind the `memfd` feature.
- `log` dependency is now optional, included in default features. When disabled,
  `warn!` falls back to `eprintln!`.
- `vec64` dependency bumped to 0.5.0 for Rust nightly compatibility.
- `cast_polars` feature now depends on `polars-core` instead of the umbrella
  `polars` crate. This improves compile times when the polars (FFI bridge)
  feature is on and avoids `quick-xml` security advisories (RUSTSEC-2026-0194, RUSTSEC-2026-0195)
  that were getting pulled in transitively.
- Bumped `arrow` / `arrow-schema` from 58.x to 59.2.0 and `polars-core` /
  `polars-arrow` from 0.53.0 to 0.55.2. Trimmed back Arrow feature flags.
- Full `cargo update` across all transitive dependencies.

### Fixed
- `Value::len()` for `ArrayView` variants now returns the view length instead
  of the backing array length.
- `NumericArrayV::guarantee_f64()` returns a windowed `BitmaskV` aligned with
  the data slice, fixing misaligned null masks on non-zero-offset views.
- Schema-level metadata preserved during arena-based table consolidation when
  the `table_metadata` feature is enabled.

# Minarrow-py / Minarrow-pyo3

## 0.17.0 - 2026-08-15

### Added
- `PyMatrix` class with construction from rows, columns or a `PyTable`,
  indexing, transpose, row/column extraction and `to_table()`.
- `PyCube` class with construction from a list of tables, positional and named
  indexing, iteration, and `from_table` for column-based splitting.
- Temporal dtype support in `build_array`: `Date32`, `Date64`, `Time32`,
  `Time64`, `Duration32`, `Duration64` and `Timestamp` with unit specifiers.
- `py_to_scalar` publicly re-exported from minarrow-py.
- `PyArray` and `PyTable` publicly re-exported from minarrow-py for downstream
  pyo3 crates.
- Efficient `read_sequence<T>` extraction into `Vec64<T>`, avoiding double
  allocation through pyo3's `Vec` path.

### Changed
- `parse_dtype` categorical alias respects `default_categorical_8`, returning
  `UInt8` when that feature is active.
- `default_categorical_8` feature propagates from minarrow-py to minarrow-pyo3.
- Bumped minarrow dependency to 0.17.0.

---

## 0.16.2 - 2026-07-23

### Added
- Added `ByteSize::logical_bytes` for logical byte size accounting.
- Table::get_views_for_target_batch_size for easier Table -> wire batching.
- Producer-driven record batch stream export for FFI.

### Bug fix
- `TextArrayV` conversions from `Vec64<String>`, `Vec<String>` and `&[&str]` now build a `String32` variant when the `large_string` feature is off.

## 0.16.0 - 2026-07-16

This version completes Python ecosystem compatibility with DLPack support for landing data (or `ChunkedNdArray`) data in Rust
and bridging to Python for AI/ML and back.

Moving forward we will aim to minimise breaking changes and stabilise to a 1.0 version in the coming month(s). Additionally, include a `Stable` Rust build.

The new `NdArray`, `XArray` and `SuperNdArray` may however see breaking changes before then until they have had time for production use.

Please report any known bugs and/or feature requests through GitHub Issues, or if necessary discreetly through the contact for on `SpaceCell.com`,
where any bugs or reasonable feature requests may be requested anonymously / treated confidentially.
`
### Added

**NdArray, XArray and DLPack Feature**:
- `NdArray` n-dimensional contiguous array (`ndarray` feature) over `Buffer<T>`,
  generic over f32/f64, with NaN null semantics, `NdArrayV` zero-copy views,
  transpose and axis permutation, and Table/Matrix/Array interop.
- `SuperNdArray` chunked n-dimensional array with `SuperNdArrayV`
  chunk-spanning views and rechunking.
- `XArray` labelled n-dimensional array (`xarray` feature) with named
  dimensions, coordinate selection via `at`/`between`/`nearest`, and owned,
  view, or chunked storage behind one type.
- DLPack tensor interchange (`dlpack` feature) over the legacy and 1.x
  versioned ABIs for zero-copy sharing with PyTorch, JAX, and TensorFlow.
- `AxisSelection` trait for `.s()` axis slicing, with `NdArrayVT`,
  `SuperNdArrayVT`, and `XArrayVT` view tuple aliases.
- Broadcast kernels for `NdArray`, `SuperNdArray`, and `XArray`, plus
  `Value` variants, conversions, and concatenation for the new types.
- `NdArray` in minarrow-py with the DLPack protocol - `__dlpack__`,
  `from_dlpack`, and named bridges to NumPy, PyTorch, JAX, TensorFlow,
  and CuPy.
- `PyNdArray` in minarrow-pyo3 (`ndarray` feature) for Rust extensions
  bridging tensors to Python, with the DLPack capsule glue hosted in
  its `ffi::dlpack` and shared by minarrow-py.

**Minor Feature Additions**
- At `value_at` to `Array` and `ArrayV` for (opt-in) normalised equality checking.
- Added bitmask-driven gather to Array, Table and their view variants.
- Added default SIMD chipset behaviour.
- Added bitwise kernels.
- Added default display trait impl to `Scalar`.
- Added `LBuffer::freeze()`.
- Added `NdArray` n-dimensional array with views and chunked variants.
- Added `XArray` labelled n-dimensional array.
- Added axis selection, broadcasting, and `Value` support for the n-dimensional types.
- Added DLPack FFI for zero-copy PyTorch, JAX, and TensorFlow interchange, with `NdArray` in minarrow-py and minarrow-pyo3.
- Added `get`, `get_unchecked`, `get_str` and `get_str_unchecked` methods to `Array`
- Added bitmask gather to Array, Table and their view variants.

### Bugfixes
- Fixed a kernel branching issue in the arithmetic SIMD paths.
- Fixed out-of-range categorical codes to resolve to the empty string.
- Fixed `StringArray` length reporting the byte length when `MaskedArray` was not imported.
- Fixed null handling bug on String and Categorical set_str method.

### Modified
- Bumped pyo3 to 0.29.

- Normalised `hash_element_at` so NaN maps to a dummy/sentinel value
- Added `value_eq` method to `Array` and `ArrayV` where NaN == NaN, -0.0 == 0.0, Null == Null etc.
- Normalised equality checking for `Scalar` -0.0 == 0.0 and NaN == NaN.
Note: the above three only affect users opting into the methods and/or `Scalar` abstraction.
For users wanting IEEE compatibility for floats using `myarr.clone().num().f64()` gets to a `FloatArray` zero-copy.

## [0.15.0] - 2026-06-30

### Added
- `_into` Datetime kernel variants for efficient non-allocations.
- `_into` Bitmask kernel variants for efficient non-allocations.
- `UnsafeMut` utility `Buffer` for handling cross-thread bitpacked writes.

### Changed
- Improved `name` construction ergonomics.

## [0.14.0] - 2026-06-20

### Added
- `_into` mutable output buffer datetime variants supporting efficient parallelism.
- `TimePeriod` enum consistency for datetime periods with automatic deref on
  string constituents i.e., `week` becomes `TimePeriod::Week` at the call-site.

### Fixed
- `BitmaskV` lifetime.
- `BooleanArray` length visibility.

## [0.13.0] - 2026-06-13

### Added
- Fixed-format ISO 8601 / RFC 3339 timestamp parsing for `DatetimeArray`.
- `ArrowType::upcast` for binary operation type promotion.
- `_into` kernel variants for buffer mutation support.
- `LBuffer` as a backing buffer type.

### Fixed
- The enum accessors. `num()`, `str()`, `bool()`, `dt()` and the typed
  accessors (`i32()` through `f64()`, `str32()`, `cat32()`, `dt32()`, etc.) now
  borrow `&self` and return shared `Arc` handles. This addresses an earlier
  regression where some APIs duplicated, impacting architectural clarity.

## [0.12.1] - 2026-05-26

### Added
- Ergonomic constructor macros for `Table` `tbl!`, `Matrix` `mat!`, and `SuperTable` `st!`.
- Add null mask support for existing `Array` `arr!` and `FieldArray` `fa!` constructors.
- Zero-Copy FFI for the Minarrow `View` types.

## [0.12.0] - 2026-05-25

### Added
- Shared categorical dictionaries behind the new `shared_dict` feature.
  With the feature on, categorical arrays in the same group share one
  dictionary, so codes mean the same string across batches and group-by /
  joins / filters can work on codes directly (#100).
- `fast_dict` feature for faster dictionary interning under heavy
  multi-thread workloads (#100).
- `Consolidate` is now implemented on each typed array via the AVT view
  tuples in `aliases.rs`. The top-level `Vec<ArrayVT<'a>>::consolidate`
  matches on the first chunk's variant and routes to the typed impl,
  replacing the previous per-variant macros (#100).
- `Bitmask::set_range(start, end, value)` for writing a contiguous run
  of bits in one call (#99).
- `Default` for `Scalar` returning `Scalar::Null` (#101).
- `Vec<Value>` terminal coercion into `Array` / `Table` / `SuperTable`
  via the corresponding `TryFrom` impls (#101).

### Changed
- Renamed `CategoricalArray::values()` to `unique_values()`.
- Typed-array `with_capacity(N, true)` constructors now build the null
  mask as all-valid, so a freshly reserved array reports
  `null_count() == 0` rather than `N` (#99).
- `Bitmask::resize` correctly handles extension when `set = true` across
  a partial old-byte boundary (#99).
- `SuperTable::from_batches` validates schema consistency across all
  batches up front, surfacing mismatches as an error rather than
  panicking on a downstream read (#101).
- `SuperArrayV::consolidate` now collects view tuples and routes through
  the new typed `Consolidate` impls.

### Fixed
- `BooleanArray` consolidate path with mixed-null chunks: result mask
  was pre-sized to `total_len` and then extended in place, ending up at
  `2 * total_len` and tripping the validator in `BooleanArray::new`.
  Result mask is now overwritten in place for chunks that carry a null mask (#100).
- `push_nulls` on typed arrays now correctly marks the new positions as null via `Bitmask::set_range`.
- `MinarrowError::DictionaryOverflow` is always present. The variant
  was previously cfg-gated, which forced downstream matches to also
  feature-gate the arm (#100).

## [0.11.0] - 2026-05-15

### Added
- **arrow-rs / polars import bridges** (#70): symmetric `from_apache_arrow` /
  `from_polars` across `Array`, `FieldArray`, `Table`, `SuperArray`, and
  `SuperTable`, each with a `try_*` sibling returning `Result<_, MinarrowError>`.
  Adds `From<&Series>`, `From<&RecordBatch>`, `From<&DataFrame>`, and
  `From<&[RecordBatch]>` for `.into()` ergonomics.
- New feature-gated bridge modules `src/ffi/arrow_rs.rs` (`cast_arrow`) and
  `src/ffi/polars.rs` (`cast_polars`) centralising the export / import pairs (#70).
- `MinarrowError::BridgeError` variant carrying arrow-rs / polars FFI failures,
  with feature-gated `From` impls for `ArrowError` and `PolarsError` (#70).
- Matrix: strided LAPACK matrix getters, improved interoperability, and
  improved docs (#59, #62).
- Cross-tabulate string kernel.
- `has_nulls` accessor (#63).
- `with_capacity` for `CategoricalArray<T>` (#67).
- Filled missing trait impls: `SuperTable` `From`, `TryFrom<Value>` gaps,
  `From<BooleanArrayV>` for `Value`, plus other `From` impls (#67).
- `Vec64` arm for the `fa!` macro.
- Zero-copy `to_array` escape path.
- Minimal `log` dependency.

### Changed
- View -> owned conversions take a fast path when the view spans its full
  backing storage (offset 0, length matches the backing array): skips
  `slice_clone` and returns an `Arc` clone of the underlying allocation (#80).
- Polars `Array` / `FieldArray` / `Table` `from_polars` now route through
  `SuperArray::from_polars` / `SuperTable::from_polars`, correctly handling
  multi-chunked `Series` / `DataFrame` inputs (#76, #77).
- Bumped all dependencies to latest semver (#68).
- **Breaking:** renamed `SuperArray::to_apache_arrow_chunks` and
  `SuperTable::to_apache_arrow_batches` to `to_apache_arrow` (#70).

### Removed
- **Breaking:** removed `Array::to_apache_arrow_with_field` and
  `Array::to_polars_with_field`. Wrap in a `FieldArray` and call its `to_*()`
  method when an explicit `Field` is needed (#70).

### Fixed
- Per-call `Box` leak in the `to_*` export paths — `ArrowArray` / `ArrowSchema`
  heap wrappers were not freed after `read()` (#70).
- Categorical pandas `-1` null sentinel handling (#69).
- Zero-sized shared buffer checks: round-trips through `SharedBuffer` for
  zero-row columns no longer panic on alignment assertions (#73).
- Polars default features (#77).
- Default features and the test harness; stale ref in benches.
- Pinned the toolchain via `rust-toolchain.toml` to work around an upstream
  issue.

## [0.10.1] - 2026-04-08

### Added
- Selection variants and edge-case handling.
- `fa!` macro for `FieldArray` construction; migrated existing `FieldArray`
  construction sites to it.

### Changed
- Refined `Selections` trait and `Cube` behaviour.

### Fixed
- Popcount edge cases.

### Documentation
- Documented `ArrayView`.

## [0.10.0] - 2026-04-04

### Added
- `TableView` column selections.
- `Value` arity.
- `Eq` / `Hash` impls across `Field` relations.
- `SharedBuffer` `Arc` interoperability tool.
- `default_categorical_8` feature.
- `append_range` on arrays.

### Changed
- Kernel performance improvements via better memory management.

### Fixed
- `SharedBuffer` drop bug.

## [0.9.3] - 2026-03-22

### Added
- Feature-flagged table metadata, with conditional broadcast handling.
- Matrix upgraded to an arena-style strided buffer.
- Minor table numeric and view improvements for floats.
- Extended string kernels.

### Fixed
- `extended_categorical` feature flag.
- Misc kernel improvements.

## [0.9.2] - 2026-03-15

### Added
- `apply` helpers and typed column getters.

### Changed
- Improved categorical type handling for the arrow-stream FFI.
- Improved Pandas FFI compatibility.

## [0.9.1] - 2026-03-08

### Changed
- Dependency maintenance: bumped package versions.

## [0.9.0] - 2026-03-08

### Added
- Conversion checks.
- Consolidate bench.

### Changed
- Bumped `Vec64` for improved performance; reorganised benches.
- Updated SIMD API for lane count to track upstream changes.
