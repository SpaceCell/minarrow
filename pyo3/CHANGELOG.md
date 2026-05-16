# Changelog

All notable changes to **minarrow-pyo3** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-05-17

### Changed
- Bumped `minarrow` dependency from `0.10.1` to `0.11.0`.

### Removed
- Unused nightly feature gates (`allocator_api`, `slice_ptr_get`,
  `portable_simd`) from the crate root. The underlying allocator-aware types
  are re-exported from `minarrow`, so the gates are not required at this
  crate's call sites.
