// Copyright 2025 Peter Garfield Bower
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

//! # **Arena** - Bump allocator for bulk array construction
//!
//! Allocates all column buffers from a single 64-byte aligned `Vec64<u8>`,
//! reducing per-batch allocation count from O(columns) to O(1). This improves
//! cache locality across columns, reduces TLB pressure from scattered VMAs,
//! and eliminates allocator contention in multi-threaded ingestion.
//!
//! This is similar to the *MemoryPool* concept in Apache Arrow, except that is
//! a per-buffer allocator facade with accounting - each column still gets its
//! own independent allocation and refcount. This arena is batch-scoped: it
//! knows upfront which columns belong together, allocates them as one
//! contiguous region, and shares a single refcount across all of them.
//!
//! ## Benefits
//!
//! - **Guaranteed cross-column locality**: all buffers sit in one contiguous
//!   region, so scanning across columns hits adjacent cache lines.
//! - **Single refcount per batch**: one `SharedBuffer` owns the entire
//!   allocation. Cloning a column is an `Arc` bump, not a deep copy.
//! - **One free per batch**: when the last reference drops, the entire
//!   batch is deallocated in a single operation.
//!
//! ## Trade-offs
//!
//! - **No independent column lifetime**: dropping one column does not reclaim
//!   its memory - the backing allocation lives until all columns are dropped.
//!   In practice this is rarely a concern since, assuming the immutability contraint
//!   is upheld, columns in a batch are almost always consumed and released together.
//! - **Sizes must be known upfront**: the arena is pre-sized from column
//!   lengths. For incremental `push()`-based building where sizes are unknown,
//!   the standard `with_capacity()` constructors remain the better choice.
//! - **Copy-on-write on mutation**: since buffers are `SharedBuffer`-backed,
//!   the first mutation triggers a copy into an owned `Vec64`.
//!
//! ## Usage
//!
//! ```rust
//! use minarrow::{Arena, ArenaRegion, Buffer, Bitmask, IntegerArray, MaskedArray};
//!
//! // 1. Write phase - push data into a single allocation
//! let values: Vec<i64> = vec![1, 2, 3, 4, 5];
//! let mut arena = Arena::with_capacity(1024);
//! let r_data = arena.push_slice(&values);
//!
//! // 2. Freeze - convert to shared, immutable buffer
//! let shared = arena.freeze();
//!
//! // 3. Extract typed views - zero-copy slices into the frozen arena
//! let buffer: Buffer<i64> = r_data.to_buffer(&shared);
//! let arr = IntegerArray::new(buffer, None);
//! assert_eq!(arr.len(), 5);
//! assert_eq!(arr.get(0), Some(1));
//! ```
//!
//! ## Design
//!
//! `Arena` is a write-then-freeze allocator:
//! - **Write phase**: backed by a mutable `Vec64<u8>` with a bump cursor.
//!   Each sub-allocation is 64-byte aligned for SIMD compatibility.
//! - **Freeze phase**: `.freeze()` consumes the arena, producing a `SharedBuffer`.
//!   `ArenaRegion` handles are then used to slice typed views from it.
//!
//! The resulting `Buffer<T>` instances are `SharedBuffer`-backed and read-only.
//! Mutations trigger copy-on-write via `Buffer::make_owned_mut()`, which is the
//! correct semantic for ingestion, streaming, and consolidation paths.
//!
//! ## When to use
//!
//! - IPC/streaming ingestion where batch sizes are known from message headers
//! - `SuperTable` consolidation where all chunk sizes are already materialised
//! - Any construction path where column sizes are known upfront
//!
//! For incremental `push()`-based building where sizes are unknown, the standard
//! `with_capacity()` constructors remain the better choice.

use std::mem;
use std::ptr;
use std::sync::Arc;

use crate::enums::array::Array;
use crate::enums::collections::numeric_array::NumericArray;
#[cfg(feature = "datetime")]
use crate::enums::collections::temporal_array::TemporalArray;
use crate::enums::collections::text_array::TextArray;
#[cfg(feature = "datetime")]
use crate::enums::time_units::TimeUnit;
use crate::ffi::arrow_dtype::{ArrowType, CategoricalIndexType};
use crate::structs::shared_buffer::SharedBuffer;
use crate::utils::align64;
use crate::{Bitmask, Buffer};
use vec64::Vec64;

/// Bump allocator for bulk array construction.
///
/// Allocates all column buffers from a single 64-byte aligned `Vec64<u8>`,
/// reducing per-batch allocation count from O(columns) to O(1).
///
/// See the [module-level documentation](self) for usage examples.
pub struct Arena {
    buffer: Vec64<u8>,
    cursor: usize,
}

impl Arena {
    /// Create an arena with the given byte capacity.
    ///
    /// The backing buffer is allocated once as a single `Vec64<u8>`.
    /// All subsequent `push_*` and `reserve_*` calls bump a cursor
    /// within this allocation.
    #[inline]
    pub fn with_capacity(bytes: usize) -> Self {
        let mut buffer = Vec64::with_capacity(bytes);
        // Pre-fill to capacity so we can write into it via ptr::copy
        buffer.resize(bytes, 0);
        Self { buffer, cursor: 0 }
    }

    /// Number of bytes used so far, including alignment padding.
    #[inline]
    pub fn used(&self) -> usize {
        self.cursor
    }

    /// Remaining capacity in bytes.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.buffer.len().saturating_sub(self.cursor)
    }

    /// Total capacity in bytes.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Align the cursor to a 64-byte boundary.
    #[inline]
    fn align_cursor(&mut self) {
        self.cursor = (self.cursor + 63) & !63;
    }

    /// Copy a typed slice into the arena, returning a region handle.
    ///
    /// The region is 64-byte aligned for SIMD compatibility.
    ///
    /// # Panics
    /// Panics if the arena does not have sufficient remaining capacity.
    #[inline]
    pub fn push_slice<T: Copy>(&mut self, data: &[T]) -> ArenaRegion {
        self.align_cursor();
        let byte_len = data.len() * mem::size_of::<T>();
        assert!(
            self.cursor + byte_len <= self.buffer.len(),
            "Arena overflow: need {} bytes at offset {}, but capacity is {}",
            byte_len,
            self.cursor,
            self.buffer.len()
        );
        unsafe {
            ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                self.buffer.as_mut_ptr().add(self.cursor),
                byte_len,
            );
        }
        let region = ArenaRegion {
            byte_offset: self.cursor,
            byte_len,
        };
        self.cursor += byte_len;
        region
    }

    /// Copy a `Bitmask`'s raw bytes into the arena, returning a region handle.
    ///
    /// The region is 64-byte aligned.
    ///
    /// # Panics
    /// Panics if the arena does not have sufficient remaining capacity.
    #[inline]
    pub fn push_bitmask(&mut self, mask: &Bitmask) -> ArenaRegion {
        self.push_slice(mask.bits.as_slice())
    }

    /// Reserve uninitialised space for `count` elements of type `T`.
    ///
    /// The region is 64-byte aligned. Use `region_as_mut_slice()` to
    /// write data into the reserved region before freezing.
    ///
    /// # Panics
    /// Panics if the arena does not have sufficient remaining capacity.
    #[inline]
    pub fn reserve_slice<T>(&mut self, count: usize) -> ArenaRegion {
        self.align_cursor();
        let byte_len = count
            .checked_mul(mem::size_of::<T>())
            .expect("Arena::reserve_slice: count * size_of::<T> overflow");
        let buf_len = self.buffer.len();
        assert!(
            byte_len <= buf_len && self.cursor <= buf_len - byte_len,
            "Arena overflow: need {byte_len} bytes at offset {}, but capacity is {buf_len}",
            self.cursor
        );
        let region = ArenaRegion {
            byte_offset: self.cursor,
            byte_len,
        };
        self.cursor += byte_len;
        region
    }

    /// Get a mutable typed slice for a previously reserved region.
    ///
    /// Use this to write data into a region obtained from `reserve_slice()`
    /// before calling `freeze()`.
    ///
    /// # Panics
    /// Panics if the region is out of bounds or its byte length is not a
    /// multiple of `size_of::<T>()`.
    #[inline]
    pub fn region_as_mut_slice<T>(&mut self, region: &ArenaRegion) -> &mut [T] {
        let size_of_t = mem::size_of::<T>();
        assert_eq!(
            region.byte_len % size_of_t,
            0,
            "Region byte_len {} is not a multiple of size_of::<T>() = {}",
            region.byte_len,
            size_of_t
        );
        assert!(
            region.byte_offset + region.byte_len <= self.buffer.len(),
            "Region exceeds arena bounds"
        );
        let count = region.byte_len / size_of_t;
        unsafe {
            std::slice::from_raw_parts_mut(
                self.buffer.as_mut_ptr().add(region.byte_offset) as *mut T,
                count,
            )
        }
    }

    /// Write multiple typed slices into a single contiguous region.
    ///
    /// Reserves space for `total_count` elements, then copies each slice
    /// sequentially into the region.
    ///
    /// Optionally builds a null mask from per-slice masks. When `has_nulls`
    /// is true, a mask region is reserved and populated from the provided
    /// masks. Slices without a mask contribute all-valid bits.
    pub fn write_slices<T: Copy>(
        &mut self,
        slices: &[(&[T], Option<&Bitmask>)],
        total_count: usize,
        has_nulls: bool,
    ) -> AAMaker {
        let data_region = self.reserve_slice::<T>(total_count);
        let mask_region = if has_nulls {
            Some(self.reserve_slice::<u8>((total_count + 7) / 8))
        } else {
            None
        };

        // Write data
        {
            let dest = self.region_as_mut_slice::<T>(&data_region);
            let mut pos = 0usize;
            for (src, _) in slices {
                dest[pos..pos + src.len()].copy_from_slice(src);
                pos += src.len();
            }
        }

        // Write mask
        if mask_region.is_some() {
            let mut mask = Bitmask::default();
            for (src, null_mask) in slices {
                match null_mask {
                    Some(src_mask) => mask.extend_from_bitmask(src_mask),
                    None => mask.resize(mask.len + src.len(), true),
                }
            }
            self.region_as_mut_slice::<u8>(mask_region.as_ref().unwrap())
                .copy_from_slice(mask.bits.as_slice());
        }

        AAMaker::Primitive {
            data: data_region,
            mask: mask_region,
        }
    }

    /// Write multiple string arrays into a single contiguous region.
    ///
    /// Each entry is `(offsets, data_bytes, null_mask)` from one batch.
    /// Handles offset adjustment so the combined offsets reference the
    /// concatenated data buffer.
    pub fn write_string_slices<T>(
        &mut self,
        slices: &[(&[T], &[u8], Option<&Bitmask>)],
        total_rows: usize,
        total_data_bytes: usize,
        has_nulls: bool,
    ) -> AAMaker
    where
        T: Copy
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + num_traits::ToPrimitive
            + num_traits::NumCast,
    {
        let offsets_region = self.reserve_slice::<T>(total_rows + 1);
        let data_region = self.reserve_slice::<u8>(total_data_bytes);
        let mask_region = if has_nulls {
            Some(self.reserve_slice::<u8>((total_rows + 7) / 8))
        } else {
            None
        };

        // Write offsets and data via raw pointers to avoid overlapping &mut self borrows.
        // Safety: the two regions are non-overlapping within the arena buffer, and both
        // fall within the pre-allocated capacity asserted by reserve_slice.
        unsafe {
            let off_ptr = self.buffer.as_mut_ptr().add(offsets_region.byte_offset) as *mut T;
            let dat_ptr = self.buffer.as_mut_ptr().add(data_region.byte_offset);

            *off_ptr = T::default(); // first offset = 0

            let mut off_pos = 1usize;
            let mut data_pos = 0usize;

            for (offsets, data, _) in slices {
                let batch_len = offsets.len() - 1;
                let byte_start = offsets[0].to_usize().unwrap();
                let byte_end = offsets[batch_len].to_usize().unwrap();
                let batch_data = &data[byte_start..byte_end];

                ptr::copy_nonoverlapping(
                    batch_data.as_ptr(),
                    dat_ptr.add(data_pos),
                    batch_data.len(),
                );

                let base = offsets[0];
                let shift: T = num_traits::cast(data_pos).unwrap();
                for i in 1..=batch_len {
                    *off_ptr.add(off_pos) = shift + (offsets[i] - base);
                    off_pos += 1;
                }

                data_pos += batch_data.len();
            }
        }

        // Write mask
        if mask_region.is_some() {
            let mut mask = Bitmask::default();
            for (offsets, _, null_mask) in slices {
                let batch_len = offsets.len() - 1;
                match null_mask {
                    Some(src_mask) => mask.extend_from_bitmask(src_mask),
                    None => mask.resize(mask.len + batch_len, true),
                }
            }
            self.region_as_mut_slice::<u8>(mask_region.as_ref().unwrap())
                .copy_from_slice(mask.bits.as_slice());
        }

        AAMaker::String {
            offsets: offsets_region,
            data: data_region,
            mask: mask_region,
        }
    }

    /// Write multiple bitmasks into a single contiguous boolean region.
    ///
    /// Each entry is `(data_bitmask, null_mask)` from one batch.
    pub fn write_boolean_slices(
        &mut self,
        slices: &[(&Bitmask, Option<&Bitmask>)],
        total_rows: usize,
        has_nulls: bool,
    ) -> AAMaker {
        let data_region = self.reserve_slice::<u8>((total_rows + 7) / 8);
        let mask_region = if has_nulls {
            Some(self.reserve_slice::<u8>((total_rows + 7) / 8))
        } else {
            None
        };

        // Build combined data bitmask and optional null mask
        let mut data_builder = Bitmask::default();
        let mut mask_builder: Option<Bitmask> = if has_nulls {
            Some(Bitmask::default())
        } else {
            None
        };

        for (src_data, null_mask) in slices {
            data_builder.extend_from_bitmask(src_data);
            if let Some(ref mut mask) = mask_builder {
                match null_mask {
                    Some(src_mask) => mask.extend_from_bitmask(src_mask),
                    None => mask.resize(mask.len + src_data.len(), true),
                }
            }
        }

        self.region_as_mut_slice::<u8>(&data_region)
            .copy_from_slice(data_builder.bits.as_slice());

        if let (Some(mask), Some(mr)) = (mask_builder, mask_region.as_ref()) {
            self.region_as_mut_slice::<u8>(mr)
                .copy_from_slice(mask.bits.as_slice());
        }

        AAMaker::Boolean {
            data: data_region,
            mask: mask_region,
        }
    }

    /// Calculate total arena bytes needed for a set of regions.
    ///
    /// Each entry is `(len, element_size_in_bytes)`. Each region is
    /// rounded up to 64-byte alignment, matching `reserve_slice` padding.
    pub fn capacity_for_regions(entries: &[(usize, usize)]) -> usize {
        entries
            .iter()
            .map(|&(len, elem_size)| align64(len * elem_size))
            .sum()
    }

    /// Freeze the arena, returning the backing `SharedBuffer`.
    ///
    /// The arena is consumed and can no longer be written to.
    /// Use `ArenaRegion::to_buffer()` and `ArenaRegion::to_bitmask()` to
    /// extract typed views from the returned `SharedBuffer`.
    #[inline]
    pub fn freeze(mut self) -> SharedBuffer {
        // Truncate to used portion to avoid wasting memory
        self.buffer.truncate(self.cursor);
        SharedBuffer::from_vec64(self.buffer)
    }
}

/// Handle to a region within an Arena allocation.
///
/// Returned by `Arena` allocation methods. Use with a frozen `SharedBuffer`
/// to extract typed `Buffer<T>` or `Bitmask` views.
///
/// This is a lightweight value type with no lifetime coupling to the arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaRegion {
    byte_offset: usize,
    byte_len: usize,
}

impl ArenaRegion {
    /// Zero-length sentinel for null/empty array placeholders.
    pub const EMPTY: Self = Self {
        byte_offset: 0,
        byte_len: 0,
    };

    /// Byte offset of this region within the arena.
    #[inline]
    pub fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    /// Byte length of this region.
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    /// Create a typed `Buffer<T>` by slicing into a frozen `SharedBuffer`.
    ///
    /// The returned buffer is shared and read-only. Mutations trigger
    /// copy-on-write into an owned `Vec64<T>`.
    ///
    /// # Panics
    /// Panics if the region's byte length is not a multiple of `size_of::<T>()`,
    /// or if the region exceeds the `SharedBuffer` bounds.
    #[inline]
    pub fn to_buffer<T>(&self, shared: &SharedBuffer) -> Buffer<T> {
        if self.byte_len == 0 {
            return Buffer::default();
        }
        let slice = shared.slice(self.byte_offset..self.byte_offset + self.byte_len);
        Buffer::from_shared(slice)
    }

    /// Create a `Bitmask` by slicing into a frozen `SharedBuffer`.
    ///
    /// # Arguments
    /// * `shared` - the frozen arena buffer
    /// * `num_bits` - logical bit count for the bitmask
    ///
    /// # Panics
    /// Panics if the region exceeds the `SharedBuffer` bounds.
    #[inline]
    pub fn to_bitmask(&self, shared: &SharedBuffer, num_bits: usize) -> Bitmask {
        if self.byte_len == 0 {
            return Bitmask::default();
        }
        let slice = shared.slice(self.byte_offset..self.byte_offset + self.byte_len);
        let buffer: Buffer<u8> = Buffer::from_shared(slice);
        Bitmask::new(buffer, num_bits)
    }
}

/// **Arena-Array Maker**
///
/// Arena region set for a single array.
///
/// Captures all the `ArenaRegion` handles needed to reconstruct one array
/// from a frozen `SharedBuffer`. Each variant matches an array layout:
/// fixed-width data, variable-length strings, bit-packed booleans, or
/// dictionary-encoded categoricals.
pub enum AAMaker {
    /// Fixed-width array: integer, float.
    Primitive {
        data: ArenaRegion,
        mask: Option<ArenaRegion>,
    },
    /// Variable-length string array with offsets + byte data.
    String {
        offsets: ArenaRegion,
        data: ArenaRegion,
        mask: Option<ArenaRegion>,
    },
    /// Bit-packed boolean array.
    Boolean {
        data: ArenaRegion,
        mask: Option<ArenaRegion>,
    },
    /// Dictionary-encoded categorical array.
    Categorical {
        indices: ArenaRegion,
        mask: Option<ArenaRegion>,
        unique_values: Vec64<String>,
    },
    /// Datetime array with time unit metadata.
    #[cfg(feature = "datetime")]
    Temporal {
        data: ArenaRegion,
        mask: Option<ArenaRegion>,
        time_unit: TimeUnit,
    },
}

impl AAMaker {
    /// Reconstruct an `Array` from the frozen `SharedBuffer`.
    ///
    /// Uses `dtype` to determine which concrete array variant to build.
    pub fn to_array(self, dtype: &ArrowType, shared: &SharedBuffer, n_rows: usize) -> Array {
        match (dtype, self) {
            // --- Numeric: integer ---
            (ArrowType::Int32, AAMaker::Primitive { data, mask }) => {
                let m = mask.map(|r| r.to_bitmask(shared, n_rows));
                Array::NumericArray(NumericArray::Int32(Arc::new(crate::IntegerArray::new(
                    data.to_buffer::<i32>(shared),
                    m,
                ))))
            }
            (ArrowType::Int64, AAMaker::Primitive { data, mask }) => {
                let m = mask.map(|r| r.to_bitmask(shared, n_rows));
                Array::NumericArray(NumericArray::Int64(Arc::new(crate::IntegerArray::new(
                    data.to_buffer::<i64>(shared),
                    m,
                ))))
            }
            (ArrowType::UInt32, AAMaker::Primitive { data, mask }) => {
                let m = mask.map(|r| r.to_bitmask(shared, n_rows));
                Array::NumericArray(NumericArray::UInt32(Arc::new(crate::IntegerArray::new(
                    data.to_buffer::<u32>(shared),
                    m,
                ))))
            }
            (ArrowType::UInt64, AAMaker::Primitive { data, mask }) => {
                let m = mask.map(|r| r.to_bitmask(shared, n_rows));
                Array::NumericArray(NumericArray::UInt64(Arc::new(crate::IntegerArray::new(
                    data.to_buffer::<u64>(shared),
                    m,
                ))))
            }
            #[cfg(feature = "extended_numeric_types")]
            (ArrowType::Int8, AAMaker::Primitive { data, mask }) => {
                let m = mask.map(|r| r.to_bitmask(shared, n_rows));
                Array::NumericArray(NumericArray::Int8(Arc::new(crate::IntegerArray::new(
                    data.to_buffer::<i8>(shared),
                    m,
                ))))
            }
            #[cfg(feature = "extended_numeric_types")]
            (ArrowType::Int16, AAMaker::Primitive { data, mask }) => {
                let m = mask.map(|r| r.to_bitmask(shared, n_rows));
                Array::NumericArray(NumericArray::Int16(Arc::new(crate::IntegerArray::new(
                    data.to_buffer::<i16>(shared),
                    m,
                ))))
            }
            #[cfg(feature = "extended_numeric_types")]
            (ArrowType::UInt8, AAMaker::Primitive { data, mask }) => {
                let m = mask.map(|r| r.to_bitmask(shared, n_rows));
                Array::NumericArray(NumericArray::UInt8(Arc::new(crate::IntegerArray::new(
                    data.to_buffer::<u8>(shared),
                    m,
                ))))
            }
            #[cfg(feature = "extended_numeric_types")]
            (ArrowType::UInt16, AAMaker::Primitive { data, mask }) => {
                let m = mask.map(|r| r.to_bitmask(shared, n_rows));
                Array::NumericArray(NumericArray::UInt16(Arc::new(crate::IntegerArray::new(
                    data.to_buffer::<u16>(shared),
                    m,
                ))))
            }

            // --- Numeric: float ---
            (ArrowType::Float32, AAMaker::Primitive { data, mask }) => {
                let m = mask.map(|r| r.to_bitmask(shared, n_rows));
                Array::NumericArray(NumericArray::Float32(Arc::new(crate::FloatArray::new(
                    data.to_buffer::<f32>(shared),
                    m,
                ))))
            }
            (ArrowType::Float64, AAMaker::Primitive { data, mask }) => {
                let m = mask.map(|r| r.to_bitmask(shared, n_rows));
                Array::NumericArray(NumericArray::Float64(Arc::new(crate::FloatArray::new(
                    data.to_buffer::<f64>(shared),
                    m,
                ))))
            }

            // --- String ---
            (
                ArrowType::String,
                AAMaker::String {
                    offsets,
                    data,
                    mask,
                },
            ) => {
                let m = mask.map(|r| r.to_bitmask(shared, n_rows));
                Array::TextArray(TextArray::String32(Arc::new(crate::StringArray::new(
                    data.to_buffer::<u8>(shared),
                    m,
                    offsets.to_buffer::<u32>(shared),
                ))))
            }
            #[cfg(feature = "large_string")]
            (
                ArrowType::LargeString,
                AAMaker::String {
                    offsets,
                    data,
                    mask,
                },
            ) => {
                let m = mask.map(|r| r.to_bitmask(shared, n_rows));
                Array::TextArray(TextArray::String64(Arc::new(crate::StringArray::new(
                    data.to_buffer::<u8>(shared),
                    m,
                    offsets.to_buffer::<u64>(shared),
                ))))
            }

            // --- Categorical ---
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            (
                ArrowType::Dictionary(CategoricalIndexType::UInt32),
                AAMaker::Categorical {
                    indices,
                    mask,
                    unique_values,
                },
            ) => {
                let m = mask.map(|r| r.to_bitmask(shared, n_rows));
                Array::TextArray(TextArray::Categorical32(Arc::new(
                    crate::CategoricalArray::new(
                        indices.to_buffer::<u32>(shared),
                        unique_values,
                        m,
                    ),
                )))
            }
            #[cfg(feature = "default_categorical_8")]
            (
                ArrowType::Dictionary(CategoricalIndexType::UInt8),
                AAMaker::Categorical {
                    indices,
                    mask,
                    unique_values,
                },
            ) => {
                let m = mask.map(|r| r.to_bitmask(shared, n_rows));
                Array::TextArray(TextArray::Categorical8(Arc::new(
                    crate::CategoricalArray::new(indices.to_buffer::<u8>(shared), unique_values, m),
                )))
            }
            #[cfg(feature = "extended_categorical")]
            (
                ArrowType::Dictionary(CategoricalIndexType::UInt16),
                AAMaker::Categorical {
                    indices,
                    mask,
                    unique_values,
                },
            ) => {
                let m = mask.map(|r| r.to_bitmask(shared, n_rows));
                Array::TextArray(TextArray::Categorical16(Arc::new(
                    crate::CategoricalArray::new(
                        indices.to_buffer::<u16>(shared),
                        unique_values,
                        m,
                    ),
                )))
            }
            #[cfg(feature = "extended_categorical")]
            (
                ArrowType::Dictionary(CategoricalIndexType::UInt64),
                AAMaker::Categorical {
                    indices,
                    mask,
                    unique_values,
                },
            ) => {
                let m = mask.map(|r| r.to_bitmask(shared, n_rows));
                Array::TextArray(TextArray::Categorical64(Arc::new(
                    crate::CategoricalArray::new(
                        indices.to_buffer::<u64>(shared),
                        unique_values,
                        m,
                    ),
                )))
            }

            // --- Boolean ---
            (ArrowType::Boolean, AAMaker::Boolean { data, mask }) => {
                let m = mask.map(|r| r.to_bitmask(shared, n_rows));
                let data_bitmask = data.to_bitmask(shared, n_rows);
                Array::BooleanArray(Arc::new(crate::BooleanArray::new(data_bitmask, m)))
            }

            // --- Datetime ---
            #[cfg(feature = "datetime")]
            (
                ArrowType::Date32 | ArrowType::Time32(_) | ArrowType::Duration32(_),
                AAMaker::Temporal {
                    data,
                    mask,
                    time_unit,
                },
            ) => {
                let m = mask.map(|r| r.to_bitmask(shared, n_rows));
                Array::TemporalArray(TemporalArray::Datetime32(Arc::new(
                    crate::DatetimeArray::new(data.to_buffer::<i32>(shared), m, Some(time_unit)),
                )))
            }
            #[cfg(feature = "datetime")]
            (
                ArrowType::Date64
                | ArrowType::Time64(_)
                | ArrowType::Duration64(_)
                | ArrowType::Timestamp(_, _),
                AAMaker::Temporal {
                    data,
                    mask,
                    time_unit,
                },
            ) => {
                let m = mask.map(|r| r.to_bitmask(shared, n_rows));
                Array::TemporalArray(TemporalArray::Datetime64(Arc::new(
                    crate::DatetimeArray::new(data.to_buffer::<i64>(shared), m, Some(time_unit)),
                )))
            }

            (ArrowType::Null, _) => Array::Null,
            _ => unreachable!("Mismatched ArrowType and AAMaker variant"),
        }
    }
}

/// Consolidate multiple array chunks into one using a single arena allocation.
///
/// All buffers are written into a single contiguous allocation, then
/// sliced into typed views. Both `SuperArray::consolidate` and any
/// future single-column consolidation paths delegate here when the
/// `arena` feature is enabled.
///
/// # Panics
/// Panics if `chunks` is empty.
#[cfg(feature = "chunked")]
pub(crate) fn consolidate_array_arena(chunks: &[&Array], dtype: &ArrowType) -> Array {
    assert!(!chunks.is_empty(), "consolidate called on empty chunk set");

    let n_rows: usize = chunks.iter().map(|a| a.len()).sum();
    let mask_bytes = (n_rows + 7) / 8;
    let has_nulls = chunks.iter().any(|a| a.null_mask().is_some());
    let first = chunks[0];

    // --- Step 1: Calculate total arena capacity ---
    let mut total_bytes = 0usize;

    match first {
        Array::NumericArray(num) => {
            let elem = match num {
                NumericArray::Int32(_) => 4,
                NumericArray::Int64(_) => 8,
                NumericArray::UInt32(_) => 4,
                NumericArray::UInt64(_) => 8,
                NumericArray::Float32(_) => 4,
                NumericArray::Float64(_) => 8,
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(_) => 1,
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(_) => 2,
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(_) => 1,
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(_) => 2,
                NumericArray::Null => 0,
            };
            total_bytes += align64(n_rows * elem);
        }
        Array::TextArray(text) => match text {
            TextArray::String32(_) => {
                total_bytes += align64((n_rows + 1) * 4);
                let data_bytes: usize = chunks
                    .iter()
                    .map(|a| {
                        if let Array::TextArray(TextArray::String32(s)) = a {
                            s.data.len()
                        } else {
                            0
                        }
                    })
                    .sum();
                total_bytes += align64(data_bytes);
            }
            #[cfg(feature = "large_string")]
            TextArray::String64(_) => {
                total_bytes += align64((n_rows + 1) * 8);
                let data_bytes: usize = chunks
                    .iter()
                    .map(|a| {
                        if let Array::TextArray(TextArray::String64(s)) = a {
                            s.data.len()
                        } else {
                            0
                        }
                    })
                    .sum();
                total_bytes += align64(data_bytes);
            }
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            TextArray::Categorical32(_) => {
                total_bytes += align64(n_rows * 4);
            }
            #[cfg(feature = "default_categorical_8")]
            TextArray::Categorical8(_) => {
                total_bytes += align64(n_rows);
            }
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical16(_) => {
                total_bytes += align64(n_rows * 2);
            }
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical64(_) => {
                total_bytes += align64(n_rows * 8);
            }
            TextArray::Null => {}
        },
        Array::BooleanArray(_) => {
            total_bytes += align64(mask_bytes);
        }
        #[cfg(feature = "datetime")]
        Array::TemporalArray(temp) => {
            let elem = match temp {
                TemporalArray::Datetime32(_) => 4,
                TemporalArray::Datetime64(_) => 8,
                TemporalArray::Null => 0,
            };
            total_bytes += align64(n_rows * elem);
        }
        Array::Null => {}
    }

    if has_nulls {
        total_bytes += align64(mask_bytes);
    }

    // --- Step 2: Allocate arena and write data ---
    let mut arena = Arena::with_capacity(total_bytes);

    let aa = match first {
        Array::NumericArray(num) => {
            macro_rules! write_numeric {
                ($variant:ident, $ty:ty) => {{
                    let slices: Vec<_> = chunks
                        .iter()
                        .map(|a| {
                            if let Array::NumericArray(NumericArray::$variant(inner)) = a {
                                (inner.data.as_slice() as &[$ty], inner.null_mask.as_ref())
                            } else {
                                unreachable!()
                            }
                        })
                        .collect();
                    arena.write_slices::<$ty>(&slices, n_rows, has_nulls)
                }};
            }
            match num {
                NumericArray::Int32(_) => write_numeric!(Int32, i32),
                NumericArray::Int64(_) => write_numeric!(Int64, i64),
                NumericArray::UInt32(_) => write_numeric!(UInt32, u32),
                NumericArray::UInt64(_) => write_numeric!(UInt64, u64),
                NumericArray::Float32(_) => write_numeric!(Float32, f32),
                NumericArray::Float64(_) => write_numeric!(Float64, f64),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(_) => write_numeric!(Int8, i8),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(_) => write_numeric!(Int16, i16),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(_) => write_numeric!(UInt8, u8),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(_) => write_numeric!(UInt16, u16),
                NumericArray::Null => AAMaker::Primitive {
                    data: ArenaRegion::EMPTY,
                    mask: None,
                },
            }
        }

        Array::TextArray(text) => match text {
            TextArray::String32(_) => {
                let slices: Vec<_> = chunks
                    .iter()
                    .map(|a| {
                        if let Array::TextArray(TextArray::String32(s)) = a {
                            (
                                s.offsets.as_slice() as &[u32],
                                s.data.as_slice() as &[u8],
                                s.null_mask.as_ref(),
                            )
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();
                let total_data: usize = slices.iter().map(|(_, d, _)| d.len()).sum();
                arena.write_string_slices(&slices, n_rows, total_data, has_nulls)
            }
            #[cfg(feature = "large_string")]
            TextArray::String64(_) => {
                let slices: Vec<_> = chunks
                    .iter()
                    .map(|a| {
                        if let Array::TextArray(TextArray::String64(s)) = a {
                            (
                                s.offsets.as_slice() as &[u64],
                                s.data.as_slice() as &[u8],
                                s.null_mask.as_ref(),
                            )
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();
                let total_data: usize = slices.iter().map(|(_, d, _)| d.len()).sum();
                arena.write_string_slices(&slices, n_rows, total_data, has_nulls)
            }
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            TextArray::Categorical32(_) => {
                let slices: Vec<_> = chunks
                    .iter()
                    .map(|a| {
                        if let Array::TextArray(TextArray::Categorical32(c)) = a {
                            (c.data.as_slice() as &[u32], c.null_mask.as_ref())
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();
                let aa = arena.write_slices::<u32>(&slices, n_rows, has_nulls);
                let dict = if let Array::TextArray(TextArray::Categorical32(c)) = first {
                    c.unique_values.clone()
                } else {
                    unreachable!()
                };
                if let AAMaker::Primitive { data, mask } = aa {
                    AAMaker::Categorical {
                        indices: data,
                        mask,
                        unique_values: dict,
                    }
                } else {
                    unreachable!()
                }
            }
            #[cfg(feature = "default_categorical_8")]
            TextArray::Categorical8(_) => {
                let slices: Vec<_> = chunks
                    .iter()
                    .map(|a| {
                        if let Array::TextArray(TextArray::Categorical8(c)) = a {
                            (c.data.as_slice() as &[u8], c.null_mask.as_ref())
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();
                let aa = arena.write_slices::<u8>(&slices, n_rows, has_nulls);
                let dict = if let Array::TextArray(TextArray::Categorical8(c)) = first {
                    c.unique_values.clone()
                } else {
                    unreachable!()
                };
                if let AAMaker::Primitive { data, mask } = aa {
                    AAMaker::Categorical {
                        indices: data,
                        mask,
                        unique_values: dict,
                    }
                } else {
                    unreachable!()
                }
            }
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical16(_) => {
                let slices: Vec<_> = chunks
                    .iter()
                    .map(|a| {
                        if let Array::TextArray(TextArray::Categorical16(c)) = a {
                            (c.data.as_slice() as &[u16], c.null_mask.as_ref())
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();
                let aa = arena.write_slices::<u16>(&slices, n_rows, has_nulls);
                let dict = if let Array::TextArray(TextArray::Categorical16(c)) = first {
                    c.unique_values.clone()
                } else {
                    unreachable!()
                };
                if let AAMaker::Primitive { data, mask } = aa {
                    AAMaker::Categorical {
                        indices: data,
                        mask,
                        unique_values: dict,
                    }
                } else {
                    unreachable!()
                }
            }
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical64(_) => {
                let slices: Vec<_> = chunks
                    .iter()
                    .map(|a| {
                        if let Array::TextArray(TextArray::Categorical64(c)) = a {
                            (c.data.as_slice() as &[u64], c.null_mask.as_ref())
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();
                let aa = arena.write_slices::<u64>(&slices, n_rows, has_nulls);
                let dict = if let Array::TextArray(TextArray::Categorical64(c)) = first {
                    c.unique_values.clone()
                } else {
                    unreachable!()
                };
                if let AAMaker::Primitive { data, mask } = aa {
                    AAMaker::Categorical {
                        indices: data,
                        mask,
                        unique_values: dict,
                    }
                } else {
                    unreachable!()
                }
            }
            TextArray::Null => AAMaker::Primitive {
                data: ArenaRegion::EMPTY,
                mask: None,
            },
        },

        Array::BooleanArray(_) => {
            let slices: Vec<_> = chunks
                .iter()
                .map(|a| {
                    if let Array::BooleanArray(b) = a {
                        (&b.data as &Bitmask, b.null_mask.as_ref())
                    } else {
                        unreachable!()
                    }
                })
                .collect();
            arena.write_boolean_slices(&slices, n_rows, has_nulls)
        }

        #[cfg(feature = "datetime")]
        Array::TemporalArray(temp) => {
            macro_rules! write_temporal {
                ($variant:ident, $ty:ty) => {{
                    let slices: Vec<_> = chunks
                        .iter()
                        .map(|a| {
                            if let Array::TemporalArray(TemporalArray::$variant(inner)) = a {
                                (inner.data.as_slice() as &[$ty], inner.null_mask.as_ref())
                            } else {
                                unreachable!()
                            }
                        })
                        .collect();
                    let aa = arena.write_slices::<$ty>(&slices, n_rows, has_nulls);
                    let tu = if let Array::TemporalArray(TemporalArray::$variant(inner)) = first {
                        inner.time_unit.clone()
                    } else {
                        unreachable!()
                    };
                    if let AAMaker::Primitive { data, mask } = aa {
                        AAMaker::Temporal {
                            data,
                            mask,
                            time_unit: tu,
                        }
                    } else {
                        unreachable!()
                    }
                }};
            }
            match temp {
                TemporalArray::Datetime32(_) => write_temporal!(Datetime32, i32),
                TemporalArray::Datetime64(_) => write_temporal!(Datetime64, i64),
                TemporalArray::Null => AAMaker::Primitive {
                    data: ArenaRegion::EMPTY,
                    mask: None,
                },
            }
        }

        Array::Null => AAMaker::Primitive {
            data: ArenaRegion::EMPTY,
            mask: None,
        },
    };

    // --- Step 3: Freeze and reconstruct ---
    let shared = arena.freeze();
    aa.to_array(dtype, &shared, n_rows)
}

/// Consolidate multiple tables into one using a single arena allocation.
///
/// All column buffers, offset buffers, and null masks are written into a
/// single contiguous allocation, then sliced into typed views via
/// `Table::from_arena`. Reduces allocation count from O(columns) to O(1).
///
/// Both `SuperTable::consolidate` and `Vec<Table>::consolidate` delegate
/// here when the `arena` feature is enabled.
///
/// # Panics
/// Panics if `tables` is empty.
#[cfg(feature = "chunked")]
pub(crate) fn consolidate_tables_arena(
    tables: &[&crate::structs::table::Table],
    name: String,
) -> crate::structs::table::Table {
    use crate::structs::table::Table;

    assert!(!tables.is_empty(), "consolidate called on empty table set");

    let n_cols = tables[0].cols.len();
    let n_rows: usize = tables.iter().map(|t| t.n_rows).sum();
    let mask_bytes = (n_rows + 7) / 8;

    let schema: Vec<Arc<crate::Field>> = tables[0].cols.iter().map(|c| c.field.clone()).collect();

    // --- Step 1: Calculate total arena capacity ---
    let mut total_bytes = 0usize;
    for col_idx in 0..n_cols {
        let first = &tables[0].cols[col_idx].array;
        let has_nulls = tables
            .iter()
            .any(|t| t.cols[col_idx].array.null_mask().is_some());

        match first {
            Array::NumericArray(num) => {
                let elem = match num {
                    NumericArray::Int32(_) => 4,
                    NumericArray::Int64(_) => 8,
                    NumericArray::UInt32(_) => 4,
                    NumericArray::UInt64(_) => 8,
                    NumericArray::Float32(_) => 4,
                    NumericArray::Float64(_) => 8,
                    #[cfg(feature = "extended_numeric_types")]
                    NumericArray::Int8(_) => 1,
                    #[cfg(feature = "extended_numeric_types")]
                    NumericArray::Int16(_) => 2,
                    #[cfg(feature = "extended_numeric_types")]
                    NumericArray::UInt8(_) => 1,
                    #[cfg(feature = "extended_numeric_types")]
                    NumericArray::UInt16(_) => 2,
                    NumericArray::Null => 0,
                };
                total_bytes += align64(n_rows * elem);
            }
            Array::TextArray(text) => match text {
                TextArray::String32(_) => {
                    total_bytes += align64((n_rows + 1) * 4);
                    let data_bytes: usize = tables
                        .iter()
                        .map(|t| {
                            if let Array::TextArray(TextArray::String32(a)) = &t.cols[col_idx].array
                            {
                                a.data.len()
                            } else {
                                0
                            }
                        })
                        .sum();
                    total_bytes += align64(data_bytes);
                }
                #[cfg(feature = "large_string")]
                TextArray::String64(_) => {
                    total_bytes += align64((n_rows + 1) * 8);
                    let data_bytes: usize = tables
                        .iter()
                        .map(|t| {
                            if let Array::TextArray(TextArray::String64(a)) = &t.cols[col_idx].array
                            {
                                a.data.len()
                            } else {
                                0
                            }
                        })
                        .sum();
                    total_bytes += align64(data_bytes);
                }
                #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
                TextArray::Categorical32(_) => {
                    total_bytes += align64(n_rows * 4);
                }
                #[cfg(feature = "default_categorical_8")]
                TextArray::Categorical8(_) => {
                    total_bytes += align64(n_rows);
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(_) => {
                    total_bytes += align64(n_rows * 2);
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(_) => {
                    total_bytes += align64(n_rows * 8);
                }
                TextArray::Null => {}
            },
            Array::BooleanArray(_) => {
                total_bytes += align64(mask_bytes);
            }
            #[cfg(feature = "datetime")]
            Array::TemporalArray(temp) => {
                let elem = match temp {
                    TemporalArray::Datetime32(_) => 4,
                    TemporalArray::Datetime64(_) => 8,
                    TemporalArray::Null => 0,
                };
                total_bytes += align64(n_rows * elem);
            }
            Array::Null => {}
        }

        if has_nulls {
            total_bytes += align64(mask_bytes);
        }
    }

    // --- Step 2: Allocate arena and write all data ---
    let mut arena = Arena::with_capacity(total_bytes);
    let mut regions: Vec<AAMaker> = Vec::with_capacity(n_cols);

    for col_idx in 0..n_cols {
        let first = &tables[0].cols[col_idx].array;
        let has_nulls = tables
            .iter()
            .any(|t| t.cols[col_idx].array.null_mask().is_some());

        let aa = match first {
            Array::NumericArray(num) => {
                macro_rules! write_numeric {
                    ($variant:ident, $ty:ty) => {{
                        let slices: Vec<_> = tables
                            .iter()
                            .map(|t| {
                                if let Array::NumericArray(NumericArray::$variant(a)) =
                                    &t.cols[col_idx].array
                                {
                                    (a.data.as_slice() as &[$ty], a.null_mask.as_ref())
                                } else {
                                    unreachable!()
                                }
                            })
                            .collect();
                        arena.write_slices::<$ty>(&slices, n_rows, has_nulls)
                    }};
                }
                match num {
                    NumericArray::Int32(_) => write_numeric!(Int32, i32),
                    NumericArray::Int64(_) => write_numeric!(Int64, i64),
                    NumericArray::UInt32(_) => write_numeric!(UInt32, u32),
                    NumericArray::UInt64(_) => write_numeric!(UInt64, u64),
                    NumericArray::Float32(_) => write_numeric!(Float32, f32),
                    NumericArray::Float64(_) => write_numeric!(Float64, f64),
                    #[cfg(feature = "extended_numeric_types")]
                    NumericArray::Int8(_) => write_numeric!(Int8, i8),
                    #[cfg(feature = "extended_numeric_types")]
                    NumericArray::Int16(_) => write_numeric!(Int16, i16),
                    #[cfg(feature = "extended_numeric_types")]
                    NumericArray::UInt8(_) => write_numeric!(UInt8, u8),
                    #[cfg(feature = "extended_numeric_types")]
                    NumericArray::UInt16(_) => write_numeric!(UInt16, u16),
                    NumericArray::Null => AAMaker::Primitive {
                        data: ArenaRegion::EMPTY,
                        mask: None,
                    },
                }
            }

            Array::TextArray(text) => match text {
                TextArray::String32(_) => {
                    let slices: Vec<_> = tables
                        .iter()
                        .map(|t| {
                            if let Array::TextArray(TextArray::String32(a)) = &t.cols[col_idx].array
                            {
                                (
                                    a.offsets.as_slice() as &[u32],
                                    a.data.as_slice() as &[u8],
                                    a.null_mask.as_ref(),
                                )
                            } else {
                                unreachable!()
                            }
                        })
                        .collect();
                    let total_data: usize = slices.iter().map(|(_, d, _)| d.len()).sum();
                    arena.write_string_slices(&slices, n_rows, total_data, has_nulls)
                }
                #[cfg(feature = "large_string")]
                TextArray::String64(_) => {
                    let slices: Vec<_> = tables
                        .iter()
                        .map(|t| {
                            if let Array::TextArray(TextArray::String64(a)) = &t.cols[col_idx].array
                            {
                                (
                                    a.offsets.as_slice() as &[u64],
                                    a.data.as_slice() as &[u8],
                                    a.null_mask.as_ref(),
                                )
                            } else {
                                unreachable!()
                            }
                        })
                        .collect();
                    let total_data: usize = slices.iter().map(|(_, d, _)| d.len()).sum();
                    arena.write_string_slices(&slices, n_rows, total_data, has_nulls)
                }
                #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
                TextArray::Categorical32(_) => {
                    let slices: Vec<_> = tables
                        .iter()
                        .map(|t| {
                            if let Array::TextArray(TextArray::Categorical32(a)) =
                                &t.cols[col_idx].array
                            {
                                (a.data.as_slice() as &[u32], a.null_mask.as_ref())
                            } else {
                                unreachable!()
                            }
                        })
                        .collect();
                    let aa = arena.write_slices::<u32>(&slices, n_rows, has_nulls);
                    let dict = if let Array::TextArray(TextArray::Categorical32(a)) =
                        &tables[0].cols[col_idx].array
                    {
                        a.unique_values.clone()
                    } else {
                        unreachable!()
                    };
                    if let AAMaker::Primitive { data, mask } = aa {
                        AAMaker::Categorical {
                            indices: data,
                            mask,
                            unique_values: dict,
                        }
                    } else {
                        unreachable!()
                    }
                }
                #[cfg(feature = "default_categorical_8")]
                TextArray::Categorical8(_) => {
                    let slices: Vec<_> = tables
                        .iter()
                        .map(|t| {
                            if let Array::TextArray(TextArray::Categorical8(a)) =
                                &t.cols[col_idx].array
                            {
                                (a.data.as_slice() as &[u8], a.null_mask.as_ref())
                            } else {
                                unreachable!()
                            }
                        })
                        .collect();
                    let aa = arena.write_slices::<u8>(&slices, n_rows, has_nulls);
                    let dict = if let Array::TextArray(TextArray::Categorical8(a)) =
                        &tables[0].cols[col_idx].array
                    {
                        a.unique_values.clone()
                    } else {
                        unreachable!()
                    };
                    if let AAMaker::Primitive { data, mask } = aa {
                        AAMaker::Categorical {
                            indices: data,
                            mask,
                            unique_values: dict,
                        }
                    } else {
                        unreachable!()
                    }
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(_) => {
                    let slices: Vec<_> = tables
                        .iter()
                        .map(|t| {
                            if let Array::TextArray(TextArray::Categorical16(a)) =
                                &t.cols[col_idx].array
                            {
                                (a.data.as_slice() as &[u16], a.null_mask.as_ref())
                            } else {
                                unreachable!()
                            }
                        })
                        .collect();
                    let aa = arena.write_slices::<u16>(&slices, n_rows, has_nulls);
                    let dict = if let Array::TextArray(TextArray::Categorical16(a)) =
                        &tables[0].cols[col_idx].array
                    {
                        a.unique_values.clone()
                    } else {
                        unreachable!()
                    };
                    if let AAMaker::Primitive { data, mask } = aa {
                        AAMaker::Categorical {
                            indices: data,
                            mask,
                            unique_values: dict,
                        }
                    } else {
                        unreachable!()
                    }
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(_) => {
                    let slices: Vec<_> = tables
                        .iter()
                        .map(|t| {
                            if let Array::TextArray(TextArray::Categorical64(a)) =
                                &t.cols[col_idx].array
                            {
                                (a.data.as_slice() as &[u64], a.null_mask.as_ref())
                            } else {
                                unreachable!()
                            }
                        })
                        .collect();
                    let aa = arena.write_slices::<u64>(&slices, n_rows, has_nulls);
                    let dict = if let Array::TextArray(TextArray::Categorical64(a)) =
                        &tables[0].cols[col_idx].array
                    {
                        a.unique_values.clone()
                    } else {
                        unreachable!()
                    };
                    if let AAMaker::Primitive { data, mask } = aa {
                        AAMaker::Categorical {
                            indices: data,
                            mask,
                            unique_values: dict,
                        }
                    } else {
                        unreachable!()
                    }
                }
                TextArray::Null => AAMaker::Primitive {
                    data: ArenaRegion::EMPTY,
                    mask: None,
                },
            },

            Array::BooleanArray(_) => {
                let slices: Vec<_> = tables
                    .iter()
                    .map(|t| {
                        if let Array::BooleanArray(a) = &t.cols[col_idx].array {
                            (&a.data as &Bitmask, a.null_mask.as_ref())
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();
                arena.write_boolean_slices(&slices, n_rows, has_nulls)
            }

            #[cfg(feature = "datetime")]
            Array::TemporalArray(temp) => {
                macro_rules! write_temporal {
                    ($variant:ident, $ty:ty) => {{
                        let slices: Vec<_> = tables
                            .iter()
                            .map(|t| {
                                if let Array::TemporalArray(TemporalArray::$variant(a)) =
                                    &t.cols[col_idx].array
                                {
                                    (a.data.as_slice() as &[$ty], a.null_mask.as_ref())
                                } else {
                                    unreachable!()
                                }
                            })
                            .collect();
                        let aa = arena.write_slices::<$ty>(&slices, n_rows, has_nulls);
                        let tu = if let Array::TemporalArray(TemporalArray::$variant(a)) =
                            &tables[0].cols[col_idx].array
                        {
                            a.time_unit.clone()
                        } else {
                            unreachable!()
                        };
                        if let AAMaker::Primitive { data, mask } = aa {
                            AAMaker::Temporal {
                                data,
                                mask,
                                time_unit: tu,
                            }
                        } else {
                            unreachable!()
                        }
                    }};
                }
                match temp {
                    TemporalArray::Datetime32(_) => write_temporal!(Datetime32, i32),
                    TemporalArray::Datetime64(_) => write_temporal!(Datetime64, i64),
                    TemporalArray::Null => AAMaker::Primitive {
                        data: ArenaRegion::EMPTY,
                        mask: None,
                    },
                }
            }

            Array::Null => AAMaker::Primitive {
                data: ArenaRegion::EMPTY,
                mask: None,
            },
        };

        regions.push(aa);
    }

    // --- Step 3: Freeze and reconstruct via Table::from_arena ---
    Table::from_arena(name, &schema, arena, regions, n_rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FloatArray, IntegerArray, MaskedArray, StringArray};

    #[test]
    fn test_basic_i64_roundtrip() {
        let values: Vec<i64> = vec![10, 20, 30, 40, 50];
        let mut arena = Arena::with_capacity(1024);
        let region = arena.push_slice(&values);
        let shared = arena.freeze();
        let buffer: Buffer<i64> = region.to_buffer(&shared);
        assert_eq!(buffer.as_slice(), &[10, 20, 30, 40, 50]);
    }

    #[test]
    fn test_multiple_types_in_one_arena() {
        let ints: Vec<i64> = vec![1, 2, 3];
        let floats: Vec<f64> = vec![1.5, 2.5, 3.5];
        let bytes: Vec<u8> = vec![0xFF, 0x00, 0xAB];

        let mut arena = Arena::with_capacity(4096);
        let r_ints = arena.push_slice(&ints);
        let r_floats = arena.push_slice(&floats);
        let r_bytes = arena.push_slice(&bytes);

        let shared = arena.freeze();

        let buf_ints: Buffer<i64> = r_ints.to_buffer(&shared);
        let buf_floats: Buffer<f64> = r_floats.to_buffer(&shared);
        let buf_bytes: Buffer<u8> = r_bytes.to_buffer(&shared);

        assert_eq!(buf_ints.as_slice(), &[1i64, 2, 3]);
        assert_eq!(buf_floats.as_slice(), &[1.5f64, 2.5, 3.5]);
        assert_eq!(buf_bytes.as_slice(), &[0xFFu8, 0x00, 0xAB]);
    }

    #[test]
    fn test_bitmask_roundtrip() {
        let mask = Bitmask::new_set_all(10, true);
        let mut arena = Arena::with_capacity(1024);
        let region = arena.push_bitmask(&mask);
        let shared = arena.freeze();
        let recovered = region.to_bitmask(&shared, 10);
        assert_eq!(recovered.len, 10);
        for i in 0..10 {
            assert!(recovered.get(i), "Bit {} should be set", i);
        }
    }

    #[test]
    fn test_bitmask_with_nulls() {
        let mut mask = Bitmask::new_set_all(8, true);
        mask.set(2, false);
        mask.set(5, false);

        let mut arena = Arena::with_capacity(1024);
        let region = arena.push_bitmask(&mask);
        let shared = arena.freeze();
        let recovered = region.to_bitmask(&shared, 8);

        assert!(recovered.get(0));
        assert!(recovered.get(1));
        assert!(!recovered.get(2));
        assert!(recovered.get(3));
        assert!(recovered.get(4));
        assert!(!recovered.get(5));
        assert!(recovered.get(6));
        assert!(recovered.get(7));
    }

    #[test]
    fn test_alignment() {
        let a: Vec<u8> = vec![1, 2, 3]; // 3 bytes
        let b: Vec<i64> = vec![100, 200]; // 16 bytes

        let mut arena = Arena::with_capacity(4096);
        let r_a = arena.push_slice(&a);
        let r_b = arena.push_slice(&b);

        // First region starts at 0 (already aligned)
        assert_eq!(r_a.byte_offset() % 64, 0);
        // Second region must also be 64-byte aligned
        assert_eq!(r_b.byte_offset() % 64, 0);
        // Second region starts at 64 (after padding the 3-byte first region)
        assert_eq!(r_b.byte_offset(), 64);

        let shared = arena.freeze();
        let buf_a: Buffer<u8> = r_a.to_buffer(&shared);
        let buf_b: Buffer<i64> = r_b.to_buffer(&shared);
        assert_eq!(buf_a.as_slice(), &[1u8, 2, 3]);
        assert_eq!(buf_b.as_slice(), &[100i64, 200]);
    }

    #[test]
    fn test_reserve_and_write() {
        let mut arena = Arena::with_capacity(1024);
        let region = arena.reserve_slice::<i32>(4);

        // Write data into reserved region
        let slice = arena.region_as_mut_slice::<i32>(&region);
        slice[0] = 10;
        slice[1] = 20;
        slice[2] = 30;
        slice[3] = 40;

        let shared = arena.freeze();
        let buffer: Buffer<i32> = region.to_buffer(&shared);
        assert_eq!(buffer.as_slice(), &[10i32, 20, 30, 40]);
    }

    #[test]
    #[should_panic(expected = "Arena overflow")]
    fn test_capacity_overflow() {
        let mut arena = Arena::with_capacity(16);
        let data: Vec<i64> = vec![1, 2, 3]; // 24 bytes, exceeds 16
        arena.push_slice(&data);
    }

    #[test]
    fn test_empty_arena_freeze() {
        let arena = Arena::with_capacity(1024);
        let shared = arena.freeze();
        assert!(shared.is_empty());
    }

    #[test]
    fn test_used_and_remaining() {
        let mut arena = Arena::with_capacity(1024);
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.remaining(), 1024);

        arena.push_slice(&[1u8, 2, 3]);
        assert_eq!(arena.used(), 3);
        assert_eq!(arena.remaining(), 1024 - 3);
    }

    #[test]
    fn test_full_table_construction() {
        // Simulate building a 3-column Table from a single arena
        let ids: Vec<i64> = vec![1, 2, 3, 4, 5];
        let prices: Vec<f64> = vec![10.5, 20.0, 15.75, 8.25, 99.99];
        let mut null_mask = Bitmask::new_set_all(5, true);
        null_mask.set(2, false); // third price is null

        let mut arena = Arena::with_capacity(4096);
        let r_ids = arena.push_slice(&ids);
        let r_prices = arena.push_slice(&prices);
        let r_mask = arena.push_bitmask(&null_mask);

        let shared = arena.freeze();

        let id_buf: Buffer<i64> = r_ids.to_buffer(&shared);
        let price_buf: Buffer<f64> = r_prices.to_buffer(&shared);
        let mask = r_mask.to_bitmask(&shared, 5);

        let id_arr = IntegerArray::new(id_buf, None);
        let price_arr = FloatArray::new(price_buf, Some(mask));

        assert_eq!(id_arr.len(), 5);
        assert_eq!(id_arr.get(0), Some(1));
        assert_eq!(id_arr.get(4), Some(5));

        assert_eq!(price_arr.len(), 5);
        assert_eq!(price_arr.get(0), Some(10.5));
        assert_eq!(price_arr.get(2), None); // null
        assert_eq!(price_arr.get(4), Some(99.99));
    }

    #[test]
    fn test_string_array_from_arena() {
        // Build a StringArray's offsets + data from an arena
        let strings = ["hello", "world", "foo"];
        let mut offsets: Vec<u32> = Vec::with_capacity(strings.len() + 1);
        let mut data: Vec<u8> = Vec::new();
        offsets.push(0);
        for s in &strings {
            data.extend_from_slice(s.as_bytes());
            offsets.push(data.len() as u32);
        }

        let mut arena = Arena::with_capacity(4096);
        let r_offsets = arena.push_slice(&offsets);
        let r_data = arena.push_slice(&data);

        let shared = arena.freeze();

        let off_buf: Buffer<u32> = r_offsets.to_buffer(&shared);
        let data_buf: Buffer<u8> = r_data.to_buffer(&shared);

        let str_arr = StringArray::<u32>::new(data_buf, None, off_buf);
        assert_eq!(str_arr.len(), 3);
        assert_eq!(str_arr.get_str(0), Some("hello"));
        assert_eq!(str_arr.get_str(1), Some("world"));
        assert_eq!(str_arr.get_str(2), Some("foo"));
    }

    #[test]
    fn test_shared_buffer_sharing() {
        let a: Vec<i64> = vec![1, 2, 3];
        let b: Vec<f64> = vec![4.0, 5.0];

        let mut arena = Arena::with_capacity(4096);
        let r_a = arena.push_slice(&a);
        let r_b = arena.push_slice(&b);

        let shared = arena.freeze();

        let buf_a: Buffer<i64> = r_a.to_buffer(&shared);
        let buf_b: Buffer<f64> = r_b.to_buffer(&shared);

        // Both buffers are shared views into the same allocation
        assert!(buf_a.is_shared());
        assert!(buf_b.is_shared());
    }

    #[test]
    fn test_clone_is_cheap() {
        let values: Vec<i64> = vec![1, 2, 3, 4, 5];
        let mut arena = Arena::with_capacity(1024);
        let region = arena.push_slice(&values);
        let shared = arena.freeze();

        let buffer: Buffer<i64> = region.to_buffer(&shared);
        let cloned = buffer.clone();

        // Clone is shared (O(1) refcount bump, not a data copy)
        assert!(cloned.is_shared());
        assert_eq!(cloned.as_slice(), buffer.as_slice());
    }

    #[test]
    fn test_cow_on_mutation() {
        let values: Vec<i64> = vec![1, 2, 3];
        let mut arena = Arena::with_capacity(1024);
        let region = arena.push_slice(&values);
        let shared = arena.freeze();

        let mut buffer: Buffer<i64> = region.to_buffer(&shared);
        assert!(buffer.is_shared());

        // Mutation triggers copy-on-write
        buffer.push(4);
        assert!(!buffer.is_shared());
        assert_eq!(buffer.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_zero_length_slice() {
        let empty: Vec<i64> = vec![];
        let mut arena = Arena::with_capacity(1024);
        let region = arena.push_slice(&empty);

        assert_eq!(region.byte_len(), 0);

        let shared = arena.freeze();
        let buffer: Buffer<i64> = region.to_buffer(&shared);
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_many_small_allocations() {
        // Simulate 20 columns (10 data + 10 null masks) in one arena
        let mut arena = Arena::with_capacity(64 * 1024); // 64 KB
        let mut regions = Vec::new();

        for i in 0..10 {
            let data: Vec<i64> = (0..100).map(|x| x + i * 100).collect();
            let mask = Bitmask::new_set_all(100, true);
            regions.push((arena.push_slice(&data), arena.push_bitmask(&mask)));
        }

        let shared = arena.freeze();

        for (i, (r_data, r_mask)) in regions.iter().enumerate() {
            let buf: Buffer<i64> = r_data.to_buffer(&shared);
            let mask = r_mask.to_bitmask(&shared, 100);
            let arr = IntegerArray::new(buf, Some(mask));

            assert_eq!(arr.len(), 100);
            assert_eq!(arr.get(0), Some((i as i64) * 100));
        }
    }
}
