# Changelog

All notable changes to **minarrow** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased
- Added `_into` mutable output buffer datetime variants supporting efficient parallelism.
- Added `TimePeriod` enum consistency for datetime periods with automatic deref on string
constituents i.e., `week` becomes `TimePeriod::Week` at the call-site.
- Fixed BitmaskV lifetime.

## 0.13.0

- Added fixed-format ISO 8601 / RFC 3339 timestamp parsing for DatetimeArray
- Add ArrowType::upcast for binary operation type promotion
- Added `_into` kernel variants for buffer mutation support.
- Added LBuffer as a backing buffer type
- Fixed the enum accessors. `num()`, `str()`, `bool()`, `dt()` and the typed
  accessors (`i32()` through `f64()`, `str32()`, `cat32()`, `dt32()`, etc.) now
  borrow `&self` and return shared `Arc` handles. This addresses an earlier
  regression where some APIs unfortunately duplicated impacting architectural clarity.

## 0.12.1

### Added
- Ergonomic constructor macros for `Table` `tbl!`, `Matrix` `mat!`, and `SuperTable` `st!`.
- Add null mask support for existing `Array` `arr!` and `FieldArray` `fa!` constructors.
- Zero-Copy FFI for the Minarrow `View` types.

## 0.12.0

**IMPORTANT**: This change includes important Null Mask initialisation corrections.
These alter (correct) the behaviour of initialisation code paths for `with_capacity`
typed array constructors, and Bitmask `resize`. It is recommended that all users upgrade.

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
