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

//! # **Arrow-C-FFI Module** - *Share data to another language and/or run-time**
//!
//! Implements the *Apache Arrow* **C Data Interface** for Minarrow, enabling zero-copy
//! data exchange across language boundaries.  
//! Compatible with any runtime implementing the Arrow C interface, including Python, C++,
//! Java, and others.
//!
//! ## Features
//! - **Export**: Convert Minarrow arrays into Arrow C `ArrowArray` and `ArrowSchema` pointers.
//! - **Import**: Construct Minarrow arrays directly from Arrow C pointers.
//! - **Type fidelity**: Preserves physical layout, logical type metadata, and temporal units.
//! - **FFI safety**: Backing buffers are reference-counted to ensure lifetime validity
//!   across the boundary, with explicit release functions for deallocation.
//!
//! ## Usage
//! 1. Ensure the array has an associated `Field` or is wrapped in a `FieldArray`.
//! 2. For temporal arrays, set the correct `ArrowType` with appropriate `TimeUnit` or `IntervalUnit`.
//! 3. Call [`export_to_c`] to obtain Arrow-compatible pointers for FFI transmission.
//! 4. On the receiving side, reconstruct a Minarrow array with [`import_from_c`].
//!
//! ## Examples
//! - Rust -> Arrow: `cargo run --example apache-ffi`
//! - Planned: Python, C++, and other language bindings.
//!
//! ## Notes
//! - Dictionary-encoded (categorical) arrays are supported, including index type mapping.
//! - UTF-8 and large UTF-8 string arrays preserve offset and value buffer ordering.
//! - Temporal arrays validate logical type <-> physical storage alignment prior to export.
//!- `pyo3` normally abstracts pointer handling and lifetime management when integrating
//!   with Python; we do not yet use it, but once integrated, instead of manual `Arc` reference
//!  count handling and explicit clean-up, one will be able to instead leverage automatic,
//!  Python-owned lifetimes.
//!
//! ## Trademark Notice
//! *Apache Arrow* is a trademark of the Apache Software Foundation, used here under
//! fair-use to implement its published interoperability standard as per
//! https://www.apache.org/foundation/marks/ .

use std::ffi::{CString, c_char, c_void};
use std::sync::Arc;
use std::{ptr, slice};

use crate::ffi::arrow_dtype::ArrowType;
use crate::ffi::arrow_dtype::CategoricalIndexType;
use crate::ffi::schema::Schema;
use crate::structs::buffer::Buffer;
use crate::structs::shared_buffer::SharedBuffer;
use crate::{
    Array, Bitmask, BooleanArray, CategoricalArray, Field, Float, FloatArray, Integer,
    IntegerArray, MaskedArray, StringArray, TextArray, Vec64, vec64,
};
#[cfg(feature = "datetime")]
use crate::{DatetimeArray, IntervalUnit, TemporalArray, TimeUnit};

// Provides compatibility with the cross-platform `Apache Arrow` standard
// via the `C Data Interface` specification:
// https://arrow.apache.org/docs/format/CDataInterface.html

// The `C Data Interface` supports only individual arrays. To transmit tables,
// pass each column separately along with its corresponding `Schema`, constructed
// using the relevant `Field` definitions and metadata.
//
// On the receiving side, these arrays and the schema can combine to reconstruct
// an Arrow `RecordBatch`. This process can repeat to transmit multiple `RecordBatch`
// instances or once to transmit a single array-depending on the use case.

/// ArrowArray as per the Arrow C spec
///
/// 1. Box::new(ArrowArray::empty()) - allocates ~80 bytes for the metadata struct
/// 2. PyArrow's _export_to_c fills in the struct, setting buffers to point to PyArrow's memory
/// 3. ForeignBuffer stores this Box and uses the buffers pointer for zero-copy access
/// 4. The data never moves - we just hold a pointer to it
#[repr(C)]
pub struct ArrowArray {
    pub length: i64,
    pub null_count: i64,
    pub offset: i64,
    pub n_buffers: i64,
    pub n_children: i64,
    pub buffers: *mut *const u8,
    pub children: *mut *mut ArrowArray,
    pub dictionary: *mut ArrowArray,
    pub release: Option<unsafe extern "C" fn(*mut ArrowArray)>,
    pub private_data: *mut c_void,
}

impl ArrowArray {
    /// Creates an empty ArrowArray for receiving FFI data.
    /// Used when importing from external sources (e.g., PyArrow).
    pub fn empty() -> Self {
        Self {
            length: 0,
            null_count: 0,
            offset: 0,
            n_buffers: 0,
            n_children: 0,
            buffers: ptr::null_mut(),
            children: ptr::null_mut(),
            dictionary: ptr::null_mut(),
            release: None,
            private_data: ptr::null_mut(),
        }
    }
}

/// ArrowSchema as per the Arrow C spec
#[repr(C)]
#[derive(Clone)]
pub struct ArrowSchema {
    pub format: *const c_char,
    pub name: *const c_char,
    pub metadata: *const c_char,
    pub flags: i64,
    pub n_children: i64,
    pub children: *mut *mut ArrowSchema,
    pub dictionary: *mut ArrowSchema,
    pub release: Option<unsafe extern "C" fn(*mut ArrowSchema)>,
    pub private_data: *mut c_void,
}

impl ArrowSchema {
    /// Creates an empty ArrowSchema for receiving FFI data.
    /// Used when importing from external sources (e.g., PyArrow).
    pub fn empty() -> Self {
        Self {
            format: ptr::null(),
            name: ptr::null(),
            metadata: ptr::null(),
            flags: 0,
            n_children: 0,
            children: ptr::null_mut(),
            dictionary: ptr::null_mut(),
            release: None,
            private_data: ptr::null_mut(),
        }
    }
}

/// ArrowArrayStream as per the Arrow C Stream Interface spec.
///
/// Provides a streaming interface for consuming a sequence of ArrowArrays
/// with a shared schema. Used by the PyCapsule protocol for chunked data
/// exchange between different runtimes and languages.
///
/// See: https://arrow.apache.org/docs/format/CStreamInterface.html
#[repr(C)]
pub struct ArrowArrayStream {
    pub get_schema:
        Option<unsafe extern "C" fn(stream: *mut ArrowArrayStream, out: *mut ArrowSchema) -> i32>,
    pub get_next:
        Option<unsafe extern "C" fn(stream: *mut ArrowArrayStream, out: *mut ArrowArray) -> i32>,
    pub get_last_error: Option<unsafe extern "C" fn(stream: *mut ArrowArrayStream) -> *const c_char>,
    pub release: Option<unsafe extern "C" fn(stream: *mut ArrowArrayStream)>,
    pub private_data: *mut c_void,
}

impl ArrowArrayStream {
    /// Creates an empty ArrowArrayStream for receiving FFI data.
    pub fn empty() -> Self {
        Self {
            get_schema: None,
            get_next: None,
            get_last_error: None,
            release: None,
            private_data: ptr::null_mut(),
        }
    }
}

unsafe impl Send for ArrowArrayStream {}
unsafe impl Sync for ArrowArrayStream {}

/// Keep buffers and data alive for the C Data Interface.
/// Ensures backing data and all pointers while referenced by ArrowArray.private_data.
/// Keeps export buffers and metadata alive for the lifetime of an ArrowArray/ArrowSchema pair.
/// Fields are never read directly but must remain allocated so that raw pointers
/// in the exported ArrowArray and ArrowSchema stay valid.
#[allow(dead_code)]
struct Holder {
    array: Arc<Array>,
    _schema: Box<ArrowSchema>,
    buf_ptrs: Vec64<*const u8>,
    name_cstr: CString,
    format_cstr: CString,
    /// Encoded Arrow metadata bytes kept alive for the pointer in ArrowSchema.metadata.
    metadata_bytes: Option<Vec<u8>>,
}

/// Wrapper that owns a foreign ArrowArray's buffer memory and calls release on drop.
/// Implements `AsRef<[u8]>` for use with `SharedBuffer::from_owner()`.
///
/// This enables zero-copy FFI by:
/// 1. Holding the ArrowArray (with its release callback) alive
/// 2. Providing access to the buffer data as a byte slice
/// 3. Calling the release callback when dropped, freeing the foreign memory
struct ForeignBuffer {
    /// Raw pointer to the buffer data
    ptr: *const u8,
    /// Length in bytes
    len: usize,
    /// The ArrowArray we need to release when done.
    /// Only one ForeignBuffer per ArrowArray should hold this.
    array: Option<Box<ArrowArray>>,
}

impl AsRef<[u8]> for ForeignBuffer {
    fn as_ref(&self) -> &[u8] {
        if self.len == 0 || self.ptr.is_null() {
            return &[];
        }
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for ForeignBuffer {
    fn drop(&mut self) {
        if let Some(mut arr_box) = self.array.take() {
            if let Some(release) = arr_box.release {
                unsafe { release(arr_box.as_mut() as *mut ArrowArray) };
            }
        }
    }
}

// Required for SharedBuffer::from_owner()
unsafe impl Send for ForeignBuffer {}
unsafe impl Sync for ForeignBuffer {}

/// Releases memory for ArrowArray by deallocating Holder and zeroing the structure.
/// # Safety
/// Caller must ensure this is only called once per ArrowArray.
unsafe extern "C" fn release_arrow_array(arr: *mut ArrowArray) {
    if arr.is_null() || (unsafe { &*arr }).release.is_none() {
        return;
    }
    let _: Box<Holder> = unsafe { Box::from_raw((*arr).private_data as *mut Holder) };
    unsafe { ptr::write_bytes(arr, 0, 1) };
}

/// Releases memory for ArrowSchema by zeroing the structure.
/// # Safety
/// Caller must ensure this is only called once per ArrowSchema.
unsafe extern "C" fn release_arrow_schema(s: *mut ArrowSchema) {
    if s.is_null() || (unsafe { &*s }).release.is_none() {
        return;
    }
    unsafe { ptr::write_bytes(s, 0, 1) };
}

/// Constructs the Arrow C FFI format string for the given ArrowType.
pub fn fmt_c(dtype: ArrowType) -> CString {
    #[cfg(feature = "datetime")]
    if let ArrowType::Timestamp(u, tz) = &dtype {
        let unit_str = match u {
            TimeUnit::Seconds => "tss:",
            TimeUnit::Milliseconds => "tsm:",
            TimeUnit::Microseconds => "tsu:",
            TimeUnit::Nanoseconds => "tsn:",
            TimeUnit::Days => panic!("Timestamp(Days) is invalid in Arrow C format"),
        };
        let tz_str = tz.as_deref().unwrap_or("");
        let format_str = format!("{}{}", unit_str, tz_str);
        return CString::new(format_str).expect("CString formatting failed: invalid bytes");
    }

    let bytes: &'static [u8] = match dtype {
        ArrowType::Null => b"n",
        ArrowType::Boolean => b"b",

        #[cfg(feature = "extended_numeric_types")]
        ArrowType::Int8 => b"c",
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::UInt8 => b"C",
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::Int16 => b"s",
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::UInt16 => b"S",

        ArrowType::Int32 => b"i",
        ArrowType::UInt32 => b"I",
        ArrowType::Int64 => b"l",
        ArrowType::UInt64 => b"L",
        ArrowType::Float32 => b"f",
        ArrowType::Float64 => b"g",

        ArrowType::String => b"u",
        #[cfg(feature = "large_string")]
        ArrowType::LargeString => b"U",
        ArrowType::Utf8View => b"vu",

        // ---- datetime ----
        #[cfg(feature = "datetime")]
        ArrowType::Date32 => b"tdD",
        #[cfg(feature = "datetime")]
        ArrowType::Date64 => b"tdm",

        #[cfg(feature = "datetime")]
        ArrowType::Time32(u) => match u {
            TimeUnit::Seconds => b"tts",
            TimeUnit::Milliseconds => b"ttm",
            _ => panic!("Time32 supports Seconds or Milliseconds only"),
        },
        #[cfg(feature = "datetime")]
        ArrowType::Time64(u) => match u {
            TimeUnit::Microseconds => b"ttu",
            TimeUnit::Nanoseconds => b"ttn",
            _ => panic!("Time64 supports Microseconds or Nanoseconds only"),
        },

        #[cfg(feature = "datetime")]
        ArrowType::Duration32(u) => match u {
            TimeUnit::Seconds => b"tDs",
            TimeUnit::Milliseconds => b"tDm",
            _ => panic!("Duration32 supports Seconds or Milliseconds only"),
        },
        #[cfg(feature = "datetime")]
        ArrowType::Duration64(u) => match u {
            TimeUnit::Microseconds => b"tDu",
            TimeUnit::Nanoseconds => b"tDn",
            _ => panic!("Duration64 supports Microseconds or Nanoseconds only"),
        },

        #[cfg(feature = "datetime")]
        ArrowType::Timestamp(_, _) => {
            unreachable!("Timestamp case handled above")
        }

        #[cfg(feature = "datetime")]
        ArrowType::Interval(u) => match u {
            IntervalUnit::YearMonth => b"tiM",
            IntervalUnit::DaysTime => b"tiD",
            IntervalUnit::MonthDaysNs => b"tin",
        },

        // ---- dictionary (categorical) ----
        ArrowType::Dictionary(idx) => match idx {
            #[cfg(feature = "default_categorical_8")]
            CategoricalIndexType::UInt8 => b"C",
            #[cfg(feature = "extended_categorical")]
            CategoricalIndexType::UInt16 => b"S",
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            CategoricalIndexType::UInt32 => b"I",
            #[cfg(feature = "extended_categorical")]
            CategoricalIndexType::UInt64 => b"L",
        },
    };

    CString::new(bytes).expect("CString formatting failed: invalid bytes")
}

#[cfg(feature = "datetime")]
fn validate_temporal_field(array: &Array, dtype: &ArrowType) {
    use crate::{TemporalArray, TimeUnit};

    match (array, dtype) {
        // Date32 requires days since epoch in i32
        (Array::TemporalArray(TemporalArray::Datetime32(arr)), ArrowType::Date32) => {
            assert!(
                arr.time_unit == TimeUnit::Days,
                "FFI export: Field=Date32 requires Datetime32(Days); got {:?}",
                arr.time_unit
            );
        }
        // Date64 requires milliseconds since epoch in i64
        (Array::TemporalArray(TemporalArray::Datetime64(arr)), ArrowType::Date64) => {
            assert!(
                arr.time_unit == TimeUnit::Milliseconds,
                "FFI export: Field=Date64 requires Datetime64(Milliseconds); got {:?}",
                arr.time_unit
            );
        }
        // Time32 requires i32 with the exact unit
        (Array::TemporalArray(TemporalArray::Datetime32(arr)), ArrowType::Time32(u)) => {
            assert!(
                arr.time_unit == *u,
                "FFI export: Field=Time32({u:?}) requires Datetime32({u:?}); got {:?}",
                arr.time_unit
            );
        }
        // Time64 requires i64 with the exact unit
        (Array::TemporalArray(TemporalArray::Datetime64(arr)), ArrowType::Time64(u)) => {
            assert!(
                arr.time_unit == *u,
                "FFI export: Field=Time64({u:?}) requires Datetime64({u:?}); got {:?}",
                arr.time_unit
            );
        }
        // Timestamp requires i64 with the exact unit
        (Array::TemporalArray(TemporalArray::Datetime64(arr)), ArrowType::Timestamp(u, _tz)) => {
            assert!(
                arr.time_unit == *u,
                "FFI export: Field=Timestamp({u:?}) requires Datetime64({u:?}); got {:?}",
                arr.time_unit
            );
        }
        // Duration32 requires i32 with the exact unit
        (Array::TemporalArray(TemporalArray::Datetime32(arr)), ArrowType::Duration32(u)) => {
            assert!(
                arr.time_unit == *u,
                "FFI export: Field=Duration32({u:?}) requires Datetime32({u:?}); got {:?}",
                arr.time_unit
            );
        }
        // Duration64 requires i64 with the exact unit
        (Array::TemporalArray(TemporalArray::Datetime64(arr)), ArrowType::Duration64(u)) => {
            assert!(
                arr.time_unit == *u,
                "FFI export: Field=Duration64({u:?}) requires Datetime64({u:?}); got {:?}",
                arr.time_unit
            );
        }
        // Interval mapping enforced at type-selection time; nothing to assert here yet.
        _ => {}
    }
}

/// Exports a Minarrow array to Arrow C Data Interface pointers.
///
/// Equivalent to [`export_view_to_c`] with offset = 0 and length = `array.len()`.
pub fn export_to_c(array: Arc<Array>, schema: Schema) -> (*mut ArrowArray, *mut ArrowSchema) {
    let len = array.len() as i64;
    export_view_to_c(array, schema, 0, len)
}

/// Exports a windowed view of a Minarrow array to Arrow C Data Interface pointers.
///
/// `offset` and `length` are applied at the Arrow C layer through
/// `ArrowArray.offset` and `ArrowArray.length`. Buffer pointers always
/// point at the start of the backing buffers; consumers read
/// `length` elements starting at `offset` from each buffer
/// (validity bitmap, primitive data or codes, string offsets/values).
///
/// The full backing [`Array`] is kept alive in the FFI holder, so the
/// pointers remain valid for the lifetime of the exported `ArrowArray`.
pub fn export_view_to_c(
    array: Arc<Array>,
    schema: Schema,
    offset: i64,
    length: i64,
) -> (*mut ArrowArray, *mut ArrowSchema) {
    #[cfg(feature = "datetime")]
    {
        let field_ty = &schema.fields[0].dtype;
        // Validate temporal logical type <-> physical unit before export.
        validate_temporal_field(&*array, field_ty);
    }

    match &*array {
        Array::TextArray(TextArray::String32(_)) => {
            export_string_array_to_c(&array, schema, offset, length)
        }
        #[cfg(feature = "large_string")]
        Array::TextArray(TextArray::String64(_)) => {
            export_string_array_to_c(&array, schema, offset, length)
        }
        #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
        Array::TextArray(TextArray::Categorical32(cat)) => {
            let uniques = cat.unique_values().to_vec();
            export_categorical_array_to_c(&array, schema, offset, length, &uniques, 32)
        }
        #[cfg(feature = "default_categorical_8")]
        Array::TextArray(TextArray::Categorical8(cat)) => {
            let uniques = cat.unique_values().to_vec();
            export_categorical_array_to_c(&array, schema, offset, length, &uniques, 8)
        }
        #[cfg(feature = "extended_categorical")]
        Array::TextArray(TextArray::Categorical16(cat)) => {
            let uniques = cat.unique_values().to_vec();
            export_categorical_array_to_c(&array, schema, offset, length, &uniques, 16)
        }
        #[cfg(feature = "extended_categorical")]
        Array::TextArray(TextArray::Categorical64(cat)) => {
            let uniques = cat.unique_values().to_vec();
            export_categorical_array_to_c(&array, schema, offset, length, &uniques, 64)
        }

        _ => {
            let (data_ptr, _, _) = array.data_ptr_and_byte_len();
            let (mask_ptr, _) = array
                .null_mask_ptr_and_byte_len()
                .unwrap_or((ptr::null(), 0));
            let mut buf_ptrs = vec64![mask_ptr, data_ptr];
            let name_cstr = CString::new(schema.fields[0].name.clone()).unwrap();
            check_alignment(&mut buf_ptrs, length);
            create_arrow_export(array, schema, buf_ptrs, 2, length, offset, name_cstr)
        }
    }
}

/// Exports a Utf8 or LargeUtf8 string array to Arrow C format.
///
/// Buffer pointers point at the start of the backing offsets/values buffers;
/// the consumer applies `offset` via `ArrowArray.offset` to read
/// `offsets[offset..offset+length+1]`.
fn export_string_array_to_c(
    array: &Arc<Array>,
    schema: Schema,
    offset: i64,
    length: i64,
) -> (*mut ArrowArray, *mut ArrowSchema) {
    let (offsets_ptr, _) = array.offsets_ptr_and_len().unwrap();
    let (values_ptr, values_len, _) = array.data_ptr_and_byte_len();
    let (null_ptr, _) = array
        .null_mask_ptr_and_byte_len()
        .unwrap_or((ptr::null(), 0));
    // Arrow expects: [null, offsets, values]
    // For values buffer, only use it if non-empty; otherwise use null to avoid sentinel pointers
    let values_buf_ptr = if values_len > 0 {
        values_ptr
    } else {
        ptr::null()
    };
    let mut buf_ptrs = vec64![null_ptr, offsets_ptr, values_buf_ptr];
    let name_cstr = CString::new(schema.fields[0].name.clone()).unwrap();
    check_alignment(&mut buf_ptrs, length);
    create_arrow_export(array.clone(), schema, buf_ptrs, 3, length, offset, name_cstr)
}

/// Exports a categorical array and its dictionary in Arrow C format.
///
/// `offset` and `length` are applied to the codes buffer through
/// `ArrowArray.offset`/`ArrowArray.length`. The dictionary values are
/// exported in full and shared via Arrow's `ArrowArray.dictionary` field.
fn export_categorical_array_to_c(
    array: &Arc<Array>,
    schema: Schema,
    offset: i64,
    length: i64,
    unique_values: &[String],
    index_bits: usize,
) -> (*mut ArrowArray, *mut ArrowSchema) {
    let codes_ptr = array.data_ptr_and_byte_len().0;
    let null_ptr = array
        .null_mask_ptr_and_byte_len()
        .map_or(ptr::null(), |(p, _)| p);

    let mut buf_ptrs = vec64![null_ptr, codes_ptr];
    check_alignment(&mut buf_ptrs, length);

    // Export dictionary as string array with correct buffer order
    let dict_offsets: Vec64<u32> = {
        let mut offsets = Vec64::with_capacity(unique_values.len() + 1);
        let mut total = 0u32;
        offsets.push(0);
        for s in unique_values {
            total = total
                .checked_add(s.len() as u32)
                .expect("String data too large for u32 offset");
            offsets.push(total);
        }
        offsets
    };
    let dict_data: Vec64<u8> = unique_values
        .iter()
        .flat_map(|s| s.as_bytes())
        .copied()
        .collect();

    let dict_array = StringArray {
        offsets: dict_offsets.into(),
        data: dict_data.into(),
        null_mask: None,
    };

    let dict_schema = Schema::from(vec![Field::new(
        "dictionary",
        ArrowType::String,
        false,
        None,
    )]);
    let dict_array_arc = Arc::new(Array::TextArray(TextArray::String32(Arc::new(dict_array))));
    let (dict_arr_ptr, dict_schema_ptr) = export_to_c(dict_array_arc, dict_schema);

    let name_cstr = CString::new(schema.fields[0].name.clone()).unwrap();

    let mut field = schema.fields[0].clone();
    field.dtype = match index_bits {
        #[cfg(feature = "default_categorical_8")]
        8 => ArrowType::Dictionary(crate::ffi::arrow_dtype::CategoricalIndexType::UInt8),
        #[cfg(feature = "extended_categorical")]
        16 => ArrowType::Dictionary(crate::ffi::arrow_dtype::CategoricalIndexType::UInt16),
        #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
        32 => ArrowType::Dictionary(crate::ffi::arrow_dtype::CategoricalIndexType::UInt32),
        #[cfg(feature = "extended_categorical")]
        64 => ArrowType::Dictionary(crate::ffi::arrow_dtype::CategoricalIndexType::UInt64),
        _ => panic!("Invalid index bits for categorical array"),
    };
    let format_cstr = fmt_c(field.dtype.clone());
    let format_ptr = format_cstr.as_ptr();

    let metadata_bytes = if field.metadata.is_empty() {
        None
    } else {
        Some(encode_arrow_metadata(&field.metadata))
    };
    let metadata_ptr = metadata_bytes
        .as_ref()
        .map(|b| b.as_ptr() as *const c_char)
        .unwrap_or(ptr::null());

    let arr = Box::new(ArrowArray {
        length,
        null_count: if buf_ptrs[0].is_null() { 0 } else { -1 },
        offset,
        n_buffers: 2,
        n_children: 0,
        buffers: buf_ptrs.as_mut_ptr(),
        children: ptr::null_mut(),
        dictionary: dict_arr_ptr,
        release: Some(release_arrow_array),
        private_data: ptr::null_mut(),
    });

    let flags = if field.nullable { 1 } else { 0 };
    let schema_box = Box::new(ArrowSchema {
        format: format_ptr,
        name: name_cstr.as_ptr(),
        metadata: metadata_ptr,
        flags,
        n_children: 0,
        children: ptr::null_mut(),
        dictionary: dict_schema_ptr,
        release: Some(release_arrow_schema),
        private_data: ptr::null_mut(),
    });

    let holder = Box::new(Holder {
        array: array.clone(),
        _schema: schema_box.clone(),
        buf_ptrs,
        name_cstr,
        format_cstr,
        metadata_bytes,
    });

    let arr_ptr = Box::into_raw(arr);
    unsafe {
        (*arr_ptr).private_data = Box::into_raw(holder) as *mut c_void;
    }
    (arr_ptr, Box::into_raw(schema_box))
}

/// Imports a Minarrow array from ArrowArray and ArrowSchema C pointers.
/// # Safety
/// Both pointers must be valid and follow the Arrow C Data Interface specification.
pub unsafe fn import_from_c(arr_ptr: *const ArrowArray, sch_ptr: *const ArrowSchema) -> Arc<Array> {
    if arr_ptr.is_null() || sch_ptr.is_null() {
        panic!("FFI import_from_c: null pointer");
    }
    let arr = unsafe { &*arr_ptr };
    let sch = unsafe { &*sch_ptr };
    let fmt = unsafe { std::ffi::CStr::from_ptr(sch.format).to_bytes() };
    let is_dict = !arr.dictionary.is_null() || !sch.dictionary.is_null();

    let dtype = match fmt {
        b"n" => ArrowType::Null,
        b"b" => ArrowType::Boolean,
        #[cfg(feature = "extended_numeric_types")]
        b"c" => ArrowType::Int8,
        #[cfg(feature = "extended_numeric_types")]
        b"C" => ArrowType::UInt8,
        #[cfg(feature = "extended_numeric_types")]
        b"s" => ArrowType::Int16,
        #[cfg(feature = "extended_numeric_types")]
        b"S" => ArrowType::UInt16,
        b"i" => ArrowType::Int32,
        b"I" => ArrowType::UInt32,
        b"l" => ArrowType::Int64,
        b"L" => ArrowType::UInt64,
        b"f" => ArrowType::Float32,
        b"g" => ArrowType::Float64,
        b"u" => ArrowType::String,
        #[cfg(feature = "large_string")]
        b"U" => ArrowType::LargeString,
        b"vu" => ArrowType::Utf8View,
        #[cfg(feature = "datetime")]
        b"tdD" => ArrowType::Date32,
        #[cfg(feature = "datetime")]
        b"tdm" => ArrowType::Date64,
        #[cfg(feature = "datetime")]
        b"tts" => ArrowType::Time32(crate::TimeUnit::Seconds),
        #[cfg(feature = "datetime")]
        b"ttm" => ArrowType::Time32(crate::TimeUnit::Milliseconds),
        #[cfg(feature = "datetime")]
        b"ttu" => ArrowType::Time64(crate::TimeUnit::Microseconds),
        #[cfg(feature = "datetime")]
        b"ttn" => ArrowType::Time64(crate::TimeUnit::Nanoseconds),
        #[cfg(feature = "datetime")]
        b"tDs" => ArrowType::Duration32(crate::TimeUnit::Seconds),
        #[cfg(feature = "datetime")]
        b"tDm" => ArrowType::Duration32(crate::TimeUnit::Milliseconds),
        #[cfg(feature = "datetime")]
        b"tDu" => ArrowType::Duration64(crate::TimeUnit::Microseconds),
        #[cfg(feature = "datetime")]
        b"tDn" => ArrowType::Duration64(crate::TimeUnit::Nanoseconds),
        #[cfg(feature = "datetime")]
        b"tiM" => ArrowType::Interval(IntervalUnit::YearMonth),
        #[cfg(feature = "datetime")]
        b"tiD" => ArrowType::Interval(IntervalUnit::DaysTime),
        #[cfg(feature = "datetime")]
        b"tin" => ArrowType::Interval(IntervalUnit::MonthDaysNs),
        #[cfg(feature = "datetime")]
        _ if fmt.starts_with(b"tss")
            || fmt.starts_with(b"tsm")
            || fmt.starts_with(b"tsu")
            || fmt.starts_with(b"tsn") =>
        {
            let unit = match &fmt[..3] {
                b"tss" => crate::TimeUnit::Seconds,
                b"tsm" => crate::TimeUnit::Milliseconds,
                b"tsu" => crate::TimeUnit::Microseconds,
                b"tsn" => crate::TimeUnit::Nanoseconds,
                _ => unreachable!(),
            };
            let tz = if fmt.len() > 4 {
                let tz_bytes = &fmt[4..];
                let tz_str = String::from_utf8_lossy(tz_bytes).into_owned();
                if tz_str.is_empty() {
                    None
                } else {
                    Some(tz_str)
                }
            } else {
                None
            };
            ArrowType::Timestamp(unit, tz)
        }
        o => panic!("unsupported format {:?}", o),
    };

    // When default_categorical_8 is on without extended_categorical and Arrow sends
    // i32 dictionary indices, narrow them to u8 before building CategoricalArray<u8>
    #[cfg(all(feature = "default_categorical_8", not(feature = "extended_categorical")))]
    if is_dict && matches!(dtype, ArrowType::Int32 | ArrowType::UInt32) {
        return unsafe { import_categorical_narrow_to_u8(arr, sch, None) };
    }

    // if the array owns a dictionary, map the physical index dtype ➜ CategoricalIndexType
    #[allow(unreachable_code)]
    let maybe_cat_index = if is_dict {
        Some(match dtype {
            #[cfg(feature = "extended_numeric_types")]
            #[cfg(feature = "default_categorical_8")]
            ArrowType::Int8 | ArrowType::UInt8 => CategoricalIndexType::UInt8,
            #[cfg(feature = "extended_numeric_types")]
            #[cfg(feature = "extended_categorical")]
            ArrowType::Int16 | ArrowType::UInt16 => CategoricalIndexType::UInt16,
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            ArrowType::Int32 | ArrowType::UInt32 => CategoricalIndexType::UInt32,
            #[cfg(feature = "extended_numeric_types")]
            #[cfg(feature = "extended_categorical")]
            ArrowType::Int64 | ArrowType::UInt64 => CategoricalIndexType::UInt64,
            _ => panic!(
                "FFI import_from_c: unsupported dictionary index type {:?}",
                dtype
            ),
        })
    } else {
        None
    };

    if let Some(idx_ty) = maybe_cat_index {
        // SAFETY: we just verified pointers, types and dictionary presence
        return unsafe { import_categorical(arr, sch, idx_ty, None) };
    }

    if is_dict {
        unsafe {
            import_categorical(
                arr,
                sch,
                match dtype {
                    ArrowType::Dictionary(i) => i,
                    _ => panic!("Expected Dictionary type"),
                },
                None,
            )
        }
    } else {
        // Pass None for ownership - this function copies data because it's used for:
        // 1. Recursive imports of dictionary arrays inside categoricals (the parent
        //    ArrowArray's release callback frees the nested dictionary's memory)
        // 2. Legacy callers that don't transfer ownership
        // For zero-copy top-level imports, use import_from_c_owned instead.
        match dtype {
            ArrowType::Boolean => unsafe { import_boolean(arr, None) },
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::Int8 => unsafe { import_integer::<i8>(arr, None, Array::from_int8) },
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::UInt8 => unsafe { import_integer::<u8>(arr, None, Array::from_uint8) },
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::Int16 => unsafe { import_integer::<i16>(arr, None, Array::from_int16) },
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::UInt16 => unsafe { import_integer::<u16>(arr, None, Array::from_uint16) },
            ArrowType::Int32 => unsafe { import_integer::<i32>(arr, None, Array::from_int32) },
            ArrowType::UInt32 => unsafe { import_integer::<u32>(arr, None, Array::from_uint32) },
            ArrowType::Int64 => unsafe { import_integer::<i64>(arr, None, Array::from_int64) },
            ArrowType::UInt64 => unsafe { import_integer::<u64>(arr, None, Array::from_uint64) },
            ArrowType::Float32 => unsafe { import_float::<f32>(arr, None, Array::from_float32) },
            ArrowType::Float64 => unsafe { import_float::<f64>(arr, None, Array::from_float64) },
            ArrowType::String => unsafe { import_utf8::<u32>(arr, None) },
            #[cfg(feature = "large_string")]
            ArrowType::LargeString => unsafe { import_utf8::<u64>(arr, None) },
            ArrowType::Utf8View => unsafe { import_utf8_view(arr, None) },
            #[cfg(feature = "datetime")]
            ArrowType::Date32 => unsafe {
                import_datetime::<i32>(arr, None, crate::TimeUnit::Days)
            },
            #[cfg(feature = "datetime")]
            ArrowType::Date64 => unsafe {
                import_datetime::<i64>(arr, None, crate::TimeUnit::Milliseconds)
            },
            #[cfg(feature = "datetime")]
            ArrowType::Time32(u) => unsafe { import_datetime::<i32>(arr, None, u) },
            #[cfg(feature = "datetime")]
            ArrowType::Time64(u) => unsafe { import_datetime::<i64>(arr, None, u) },
            #[cfg(feature = "datetime")]
            ArrowType::Timestamp(u, _tz) => unsafe { import_datetime::<i64>(arr, None, u) },
            #[cfg(feature = "datetime")]
            ArrowType::Duration32(u) => unsafe { import_datetime::<i32>(arr, None, u) },
            #[cfg(feature = "datetime")]
            ArrowType::Duration64(u) => unsafe { import_datetime::<i64>(arr, None, u) },
            #[cfg(feature = "datetime")]
            ArrowType::Interval(_u) => {
                panic!("FFI import_from_c: Arrow Interval types are not yet supported");
            }
            ArrowType::Null => {
                panic!("FFI import_from_c: Arrow Null arrays types are not yet supported")
            }
            ArrowType::Dictionary(idx) => {
                if arr.dictionary.is_null() || sch.dictionary.is_null() {
                    panic!(
                        "FFI import_from_c: dictionary pointers missing for dictionary-encoded array"
                    );
                }
                unsafe { import_categorical(arr, sch, idx, None) }
            }
        }
    }
}

/// Imports a Minarrow array from owned ArrowArray and ArrowSchema C pointers.
///
/// This is the zero-copy version that takes ownership of the ArrowArray.
/// The release callback will be called when the imported array is dropped.
///
/// Returns both the Array and its Field metadata, preserving the exact Arrow type
/// (e.g., Timestamp vs Date64) from the schema format string.
///
/// # Safety
/// Both boxes must contain valid Arrow C Data Interface structures.
/// After this call, ownership is transferred - do not access the boxes again.
pub unsafe fn import_from_c_owned(
    arr_box: Box<ArrowArray>,
    sch_box: Box<ArrowSchema>,
) -> (Arc<Array>, crate::Field) {
    // Get raw pointers for reading metadata.
    // This is safe because:
    // 1. For non-dict types: arr_box is passed to import functions which store it in ForeignBuffer
    // 2. The ForeignBuffer keeps the ArrowArray alive, so the pointer remains valid
    // 3. For dict types: we explicitly release after import_categorical completes
    let arr_ptr = &*arr_box as *const ArrowArray;
    let sch_ptr = &*sch_box as *const ArrowSchema;
    let arr = unsafe { &*arr_ptr };
    let sch = unsafe { &*sch_ptr };

    if arr.release.is_none() {
        panic!("FFI import_from_c_owned: ArrowArray has no release callback");
    }

    // Extract complete field including metadata before dropping sch_box
    let mut field = unsafe { field_from_c_schema(sch) };
    let dtype = field.dtype.clone();
    let is_dict = !arr.dictionary.is_null() || !sch.dictionary.is_null();

    // For categorical (dictionary-encoded) types, the codes buffer is zero-copy
    // via ForeignBuffer. Dictionary strings are currently copied into Vec64<String>,
    // however this will be addressed in a future enhancement.
    if is_dict {
        drop(sch_box);
        let result = unsafe {
            import_categorical(
                arr,
                sch,
                match &dtype {
                    ArrowType::Dictionary(i) => i.clone(),
                    #[cfg(feature = "extended_numeric_types")]
                    ArrowType::Int8 | ArrowType::UInt8 => {
                        #[cfg(feature = "default_categorical_8")]
                        {
                            CategoricalIndexType::UInt8
                        }
                        #[cfg(not(feature = "default_categorical_8"))]
                        panic!("default_categorical_8 not enabled")
                    }
                    #[cfg(feature = "extended_numeric_types")]
                    ArrowType::Int16 | ArrowType::UInt16 => {
                        #[cfg(feature = "extended_categorical")]
                        {
                            CategoricalIndexType::UInt16
                        }
                        #[cfg(not(feature = "extended_categorical"))]
                        panic!("Extended categorical not enabled")
                    }
                    #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
                    ArrowType::Int32 | ArrowType::UInt32 => CategoricalIndexType::UInt32,
                    #[cfg(feature = "extended_numeric_types")]
                    ArrowType::Int64 | ArrowType::UInt64 => {
                        #[cfg(feature = "extended_categorical")]
                        {
                            CategoricalIndexType::UInt64
                        }
                        #[cfg(not(feature = "extended_categorical"))]
                        panic!("Extended categorical not enabled")
                    }
                    _ => panic!("FFI: unsupported dictionary index type {:?}", dtype),
                },
                Some(arr_box),
            )
        };
        return (result, field);
    }

    drop(sch_box);

    // Pass ownership for zero-copy import
    let array = unsafe {
        match dtype {
            ArrowType::Boolean => import_boolean(arr, Some(arr_box)),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::Int8 => import_integer::<i8>(arr, Some(arr_box), Array::from_int8),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::UInt8 => import_integer::<u8>(arr, Some(arr_box), Array::from_uint8),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::Int16 => import_integer::<i16>(arr, Some(arr_box), Array::from_int16),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::UInt16 => import_integer::<u16>(arr, Some(arr_box), Array::from_uint16),
            ArrowType::Int32 => import_integer::<i32>(arr, Some(arr_box), Array::from_int32),
            ArrowType::UInt32 => import_integer::<u32>(arr, Some(arr_box), Array::from_uint32),
            ArrowType::Int64 => import_integer::<i64>(arr, Some(arr_box), Array::from_int64),
            ArrowType::UInt64 => import_integer::<u64>(arr, Some(arr_box), Array::from_uint64),
            ArrowType::Float32 => import_float::<f32>(arr, Some(arr_box), Array::from_float32),
            ArrowType::Float64 => import_float::<f64>(arr, Some(arr_box), Array::from_float64),
            ArrowType::String => import_utf8::<u32>(arr, Some(arr_box)),
            #[cfg(feature = "large_string")]
            ArrowType::LargeString => import_utf8::<u64>(arr, Some(arr_box)),
            ArrowType::Utf8View => import_utf8_view(arr, Some(arr_box)),
            #[cfg(feature = "datetime")]
            ArrowType::Date32 => import_datetime::<i32>(arr, Some(arr_box), crate::TimeUnit::Days),
            #[cfg(feature = "datetime")]
            ArrowType::Date64 => {
                import_datetime::<i64>(arr, Some(arr_box), crate::TimeUnit::Milliseconds)
            }
            #[cfg(feature = "datetime")]
            ArrowType::Time32(u) => import_datetime::<i32>(arr, Some(arr_box), u),
            #[cfg(feature = "datetime")]
            ArrowType::Time64(u) => import_datetime::<i64>(arr, Some(arr_box), u),
            #[cfg(feature = "datetime")]
            ArrowType::Timestamp(u, ref _tz) => import_datetime::<i64>(arr, Some(arr_box), u),
            #[cfg(feature = "datetime")]
            ArrowType::Duration32(u) => import_datetime::<i32>(arr, Some(arr_box), u),
            #[cfg(feature = "datetime")]
            ArrowType::Duration64(u) => import_datetime::<i64>(arr, Some(arr_box), u),
            #[cfg(feature = "datetime")]
            ArrowType::Interval(_u) => {
                panic!("FFI import_from_c_owned: Arrow Interval types are not yet supported");
            }
            ArrowType::Null => {
                panic!("FFI import_from_c_owned: Arrow Null arrays types are not yet supported")
            }
            ArrowType::Dictionary(_) => unreachable!("Dictionary handled above"),
        }
    };

    // Utf8View is stored internally as String since the data is converted during import
    if field.dtype == ArrowType::Utf8View {
        field.dtype = ArrowType::String;
    }
    (array, field)
}

/// Imports an owned ArrowArray zero-copy using the given dtype.
///
/// This is used by the stream import path to take ownership of individual children
/// stolen from a struct array. For dictionary/categorical types, falls back to the
/// copying `import_from_c` path since dictionary children have nested ownership.
///
/// # Safety
/// `arr_box` must contain a valid ArrowArray with valid buffers.
/// `sch_ptr` must point to a valid ArrowSchema (borrowed, not consumed).
unsafe fn import_array_zero_copy(
    arr_box: Box<ArrowArray>,
    dtype: ArrowType,
    sch_ptr: *const ArrowSchema,
) -> Arc<Array> {
    let arr = unsafe { &*(&*arr_box as *const ArrowArray) };

    // Dictionary types use the copying path since dictionary children
    // have nested ownership that can't be split from the parent.
    // Dictionary types: codes are zero-copy, dictionary strings are copied.
    if !arr.dictionary.is_null() {
        let sch = unsafe { &*sch_ptr };

        // Narrow i32 Arrow dictionary indices to u8 when only CategoricalArray<u8> is available
        #[cfg(all(feature = "default_categorical_8", not(feature = "extended_categorical")))]
        if matches!(dtype, ArrowType::Int32 | ArrowType::UInt32 | ArrowType::Dictionary(_)) {
            return unsafe { import_categorical_narrow_to_u8(arr, sch, Some(arr_box)) };
        }

        let idx_type = match dtype.clone() {
            ArrowType::Dictionary(idx) => idx,
            _ => {
                #[allow(unused_imports)]
                use crate::ffi::arrow_dtype::CategoricalIndexType;
                match dtype {
                    #[cfg(all(
                        feature = "extended_numeric_types",
                        feature = "default_categorical_8"
                    ))]
                    ArrowType::Int8 | ArrowType::UInt8 => CategoricalIndexType::UInt8,
                    #[cfg(all(
                        feature = "extended_numeric_types",
                        feature = "extended_categorical"
                    ))]
                    ArrowType::Int16 | ArrowType::UInt16 => CategoricalIndexType::UInt16,
                    #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
                    ArrowType::Int32 | ArrowType::UInt32 => CategoricalIndexType::UInt32,
                    #[cfg(all(
                        feature = "extended_numeric_types",
                        feature = "extended_categorical"
                    ))]
                    ArrowType::Int64 | ArrowType::UInt64 => CategoricalIndexType::UInt64,
                    _ => panic!(
                        "import_array_zero_copy: unsupported dictionary index type {:?}",
                        dtype
                    ),
                }
            }
        };
        return unsafe { import_categorical(arr, sch, idx_type, Some(arr_box)) };
    }

    // Non-dict types: zero-copy via ForeignBuffer
    unsafe {
        match dtype {
            ArrowType::Boolean => import_boolean(arr, Some(arr_box)),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::Int8 => import_integer::<i8>(arr, Some(arr_box), Array::from_int8),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::UInt8 => import_integer::<u8>(arr, Some(arr_box), Array::from_uint8),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::Int16 => import_integer::<i16>(arr, Some(arr_box), Array::from_int16),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::UInt16 => import_integer::<u16>(arr, Some(arr_box), Array::from_uint16),
            ArrowType::Int32 => import_integer::<i32>(arr, Some(arr_box), Array::from_int32),
            ArrowType::UInt32 => import_integer::<u32>(arr, Some(arr_box), Array::from_uint32),
            ArrowType::Int64 => import_integer::<i64>(arr, Some(arr_box), Array::from_int64),
            ArrowType::UInt64 => import_integer::<u64>(arr, Some(arr_box), Array::from_uint64),
            ArrowType::Float32 => import_float::<f32>(arr, Some(arr_box), Array::from_float32),
            ArrowType::Float64 => import_float::<f64>(arr, Some(arr_box), Array::from_float64),
            ArrowType::String => import_utf8::<u32>(arr, Some(arr_box)),
            #[cfg(feature = "large_string")]
            ArrowType::LargeString => import_utf8::<u64>(arr, Some(arr_box)),
            ArrowType::Utf8View => import_utf8_view(arr, Some(arr_box)),
            #[cfg(feature = "datetime")]
            ArrowType::Date32 => import_datetime::<i32>(arr, Some(arr_box), crate::TimeUnit::Days),
            #[cfg(feature = "datetime")]
            ArrowType::Date64 => {
                import_datetime::<i64>(arr, Some(arr_box), crate::TimeUnit::Milliseconds)
            }
            #[cfg(feature = "datetime")]
            ArrowType::Time32(u) => import_datetime::<i32>(arr, Some(arr_box), u),
            #[cfg(feature = "datetime")]
            ArrowType::Time64(u) => import_datetime::<i64>(arr, Some(arr_box), u),
            #[cfg(feature = "datetime")]
            ArrowType::Timestamp(u, ref _tz) => import_datetime::<i64>(arr, Some(arr_box), u),
            #[cfg(feature = "datetime")]
            ArrowType::Duration32(u) => import_datetime::<i32>(arr, Some(arr_box), u),
            #[cfg(feature = "datetime")]
            ArrowType::Duration64(u) => import_datetime::<i64>(arr, Some(arr_box), u),
            #[cfg(feature = "datetime")]
            ArrowType::Interval(_u) => {
                panic!("import_array_zero_copy: Interval types are not yet supported");
            }
            ArrowType::Null => {
                panic!("import_array_zero_copy: Null array types are not yet supported");
            }
            ArrowType::Dictionary(_) => unreachable!("Dictionary handled above"),
        }
    }
}

/// Imports a null bitmap honouring an `ArrowArray.offset` bit window.
///
/// The Arrow C Data Interface stores the validity bitmap at the start of the
/// parent buffer; consumers read `length` bits starting at bit `offset`.
/// When `offset == 0` this is a straight copy; otherwise the source bits are
/// realigned into a fresh `Bitmask`.
///
/// # Safety
/// `ptr` must point to a bit-packed buffer of at least `(offset + len + 7) / 8` bytes.
unsafe fn import_null_mask_offset(
    ptr: *const u8,
    offset: usize,
    len: usize,
) -> Bitmask {
    if offset == 0 {
        unsafe { Bitmask::from_raw_slice(ptr, len) }
    } else {
        let total_bits = offset + len;
        let full = unsafe { Bitmask::from_raw_slice(ptr, total_bits) };
        full.slice_clone(offset, len)
    }
}

/// Imports an integer array from Arrow C format using the given constructor.
///
/// Honours `ArrowArray.offset`: buffer pointers are advanced by `offset`
/// elements and the null bitmap is sliced at the bit level.
///
/// # Arguments
/// * `arr` - Reference to the ArrowArray containing the data
/// * `ownership` - If `Some(box)`, takes ownership and does zero-copy via ForeignBuffer.
///                 If `None`, copies the data. This is needed for dictionary arrays inside
///                 categoricals, where the parent's release callback owns the child memory.
/// * `tag` - Constructor function to wrap the array
///
/// # Safety
/// `arr` must contain valid buffers of expected length and type.
unsafe fn import_integer<T: Integer>(
    arr: &ArrowArray,
    ownership: Option<Box<ArrowArray>>,
    tag: fn(IntegerArray<T>) -> Array,
) -> Arc<Array> {
    let len = arr.length as usize;
    let offset = arr.offset as usize;
    let buffers = unsafe { slice::from_raw_parts(arr.buffers, 2) };
    let data_ptr = unsafe { (buffers[1] as *const T).add(offset) };
    let data_len_bytes = len * std::mem::size_of::<T>();

    // Null mask is always copied (small overhead). Honours bit-level offset.
    let null_mask = if !buffers[0].is_null() {
        Some(unsafe { import_null_mask_offset(buffers[0], offset, len) })
    } else {
        None
    };

    // For empty arrays, create an empty buffer directly rather than using sentinel pointers
    let buffer: Buffer<T> = if len == 0 {
        Buffer::default()
    } else if let Some(arr_box) = ownership {
        // Zero-copy: wrap foreign buffer
        let foreign = ForeignBuffer {
            ptr: data_ptr as *const u8,
            len: data_len_bytes,
            array: Some(arr_box),
        };
        let shared = SharedBuffer::from_owner(foreign);
        Buffer::from_shared(shared)
    } else {
        // Copy: dictionary arrays inside categoricals have their memory owned by the
        // parent ArrowArray's release callback, so we can't take ownership of them
        let data = unsafe { slice::from_raw_parts(data_ptr, len) };
        Vec64::from(data).into()
    };

    let int_arr = IntegerArray::<T>::new(buffer, null_mask);
    Arc::new(tag(int_arr))
}

/// Imports a floating-point array from Arrow C format using the given constructor.
///
/// # Arguments
/// * `arr` - Reference to the ArrowArray containing the data
/// * `ownership` - If `Some(box)`, takes ownership and does zero-copy via ForeignBuffer.
///                 If `None`, copies the data. This is needed for dictionary arrays inside
///                 categoricals, where the parent's release callback owns the child memory.
/// * `tag` - Constructor function to wrap the array
///
/// # Safety
/// `arr` must contain valid buffers of expected length and type.
unsafe fn import_float<T>(
    arr: &ArrowArray,
    ownership: Option<Box<ArrowArray>>,
    tag: fn(FloatArray<T>) -> Array,
) -> Arc<Array>
where
    T: Float,
    FloatArray<T>: 'static,
{
    let len = arr.length as usize;
    let offset = arr.offset as usize;
    let buffers = unsafe { slice::from_raw_parts(arr.buffers, 2) };
    let data_ptr = unsafe { (buffers[1] as *const T).add(offset) };
    let data_len_bytes = len * std::mem::size_of::<T>();

    // Null mask is always copied (small overhead). Honours bit-level offset.
    let null_mask = if !buffers[0].is_null() {
        Some(unsafe { import_null_mask_offset(buffers[0], offset, len) })
    } else {
        None
    };

    // For empty arrays, create an empty buffer directly rather than using sentinel pointers
    let buffer: Buffer<T> = if len == 0 {
        Buffer::default()
    } else if let Some(arr_box) = ownership {
        // Zero-copy: wrap foreign buffer
        let foreign = ForeignBuffer {
            ptr: data_ptr as *const u8,
            len: data_len_bytes,
            array: Some(arr_box),
        };
        let shared = SharedBuffer::from_owner(foreign);
        Buffer::from_shared(shared)
    } else {
        // Copy: dictionary arrays inside categoricals have their memory owned by the
        // parent ArrowArray's release callback, so we can't take ownership of them
        let data = unsafe { slice::from_raw_parts(data_ptr, len) };
        Vec64::from(data).into()
    };

    let float_arr = FloatArray::<T>::new(buffer, null_mask);
    Arc::new(tag(float_arr))
}

/// Imports a boolean array from Arrow C format.
///
/// # Arguments
/// * `arr` - Reference to the ArrowArray containing the data
/// * `ownership` - If `Some(box)`, takes ownership and does zero-copy via ForeignBuffer.
///                 If `None`, copies the data. This is needed for dictionary arrays inside
///                 categoricals, where the parent's release callback owns the child memory.
///
/// # Safety
/// Buffers must be correctly aligned and sized for the declared length.
unsafe fn import_boolean(arr: &ArrowArray, ownership: Option<Box<ArrowArray>>) -> Arc<Array> {
    let len = arr.length as usize;
    let offset = arr.offset as usize;
    let buffers = unsafe { slice::from_raw_parts(arr.buffers, 2) };
    let data_ptr = buffers[1];
    let data_len = (len + 7) / 8; // bytes needed for bit-packed data

    // Null mask is always copied (small overhead). Honours bit-level offset.
    let null_mask = if !buffers[0].is_null() {
        Some(unsafe { import_null_mask_offset(buffers[0], offset, len) })
    } else {
        None
    };

    // For empty arrays, create an empty buffer directly rather than using sentinel pointers.
    // When `offset == 0`, the buffer is byte-aligned with bit index 0, so the
    // zero-copy path via ForeignBuffer applies. With a non-zero bit offset the
    // bits must be realigned into a fresh Bitmask, dropping zero-copy on bool data.
    let bool_mask = if len == 0 {
        Bitmask::new(Buffer::default(), 0)
    } else if offset == 0 {
        let buffer: Buffer<u8> = if let Some(arr_box) = ownership {
            let foreign = ForeignBuffer {
                ptr: data_ptr as *const u8,
                len: data_len,
                array: Some(arr_box),
            };
            let shared = SharedBuffer::from_owner(foreign);
            Buffer::from_shared(shared)
        } else {
            let data = unsafe { slice::from_raw_parts(data_ptr, data_len) };
            Vec64::from(data).into()
        };
        Bitmask::new(buffer, len)
    } else {
        // Bit-level realignment: copy the windowed bits into a new Bitmask.
        let total_bits = offset + len;
        let full = unsafe { Bitmask::from_raw_slice(data_ptr, total_bits) };
        let sliced = full.slice_clone(offset, len);
        // Once realigned, drop ownership: the foreign buffer is no longer
        // referenced from the imported Bitmask, so release immediately.
        if let Some(mut arr_box) = ownership {
            if let Some(release) = arr_box.release {
                unsafe { release(&mut *arr_box as *mut ArrowArray) };
            }
        }
        sliced
    };

    let bool_arr = BooleanArray::new(bool_mask, null_mask);
    Arc::new(Array::BooleanArray(bool_arr.into()))
}

/// Imports a Utf8 or LargeUtf8 string array from Arrow C format.
///
/// # Arguments
/// * `arr` - Reference to the ArrowArray containing the data
/// * `ownership` - If `Some(box)`, takes ownership and does zero-copy for values buffer.
///                 Offsets are always copied (small overhead).
///                 If `None`, copies all data. This is needed for dictionary arrays inside
///                 categoricals, where the parent's release callback owns the child memory.
///
/// # Safety
/// Expects three buffers: [nulls, offsets, values].
unsafe fn import_utf8<T: Integer>(
    arr: &ArrowArray,
    ownership: Option<Box<ArrowArray>>,
) -> Arc<Array> {
    let len = arr.length as usize;
    let offset = arr.offset as usize;

    // For empty arrays, return an empty StringArray directly without touching sentinel pointers
    if len == 0 {
        return Arc::new(if std::mem::size_of::<T>() == 4 {
            Array::from_string32(StringArray::<u32>::default())
        } else {
            #[cfg(feature = "large_string")]
            {
                Array::from_string64(StringArray::<u64>::default())
            }
            #[cfg(not(feature = "large_string"))]
            {
                panic!("LargeUtf8 (u64 offsets) requires the 'large_string' feature")
            }
        });
    }

    let buffers = unsafe { std::slice::from_raw_parts(arr.buffers, 3) };
    let null_ptr = buffers[0];
    let offsets_ptr = buffers[1];
    let values_ptr = buffers[2];

    // Read offsets[offset..offset+len+1]; with offset=0 this matches the
    // pre-existing behaviour. Validation walks the windowed range.
    let offsets_slice = unsafe {
        std::slice::from_raw_parts((offsets_ptr as *const T).add(offset), len + 1)
    };

    assert_eq!(
        offsets_slice.len(),
        len + 1,
        "UTF8: offsets length must be len+1"
    );
    let base: usize = Integer::to_usize(offsets_slice[0]);
    let mut prev = base;
    for (i, off) in offsets_slice.iter().enumerate().take(len + 1) {
        let cur: usize = Integer::to_usize(*off);
        assert!(
            cur >= prev,
            "UTF8: offsets not monotonically non-decreasing at {i}: {cur} < {prev}"
        );
        prev = cur;
    }
    let values_end: usize = Integer::to_usize(offsets_slice[len]);
    let data_len = values_end - base;

    // Null mask - always copied. Honours bit-level offset.
    let null_mask = if !null_ptr.is_null() {
        Some(unsafe { import_null_mask_offset(null_ptr, offset, len) })
    } else {
        None
    };

    // Rebase offsets so the imported StringArray starts at 0.
    // We currently do a full copy on the offsets buffer, at the cost of
    // 4-8 bytes per string. A future revision could keep the offsets
    // buffer foreign and carry the rebase as a logical adjustment.
    let offsets: Vec64<T> = if offset == 0 && base == 0 {
        Vec64::from(offsets_slice)
    } else {
        let mut out = Vec64::<T>::with_capacity(len + 1);
        for off in offsets_slice {
            let cur = Integer::to_usize(*off) - base;
            out.push(T::from_usize(cur));
        }
        out
    };

    // Values: zero-copy if we own the ArrowArray. The buffer pointer is
    // advanced by `base` so it aligns with the rebased offsets.
    let values_buffer: Buffer<u8> = if data_len == 0 {
        // Empty values buffer - don't use sentinel pointer
        Buffer::default()
    } else if let Some(arr_box) = ownership {
        let foreign = ForeignBuffer {
            ptr: unsafe { (values_ptr as *const u8).add(base) },
            len: data_len,
            array: Some(arr_box),
        };
        let shared = SharedBuffer::from_owner(foreign);
        Buffer::from_shared(shared)
    } else {
        let data = unsafe {
            std::slice::from_raw_parts((values_ptr as *const u8).add(base), data_len)
        };
        Vec64::from(data).into()
    };

    let str_arr = StringArray::<T>::new(values_buffer, null_mask, offsets);

    #[cfg(feature = "large_string")]
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u64>() {
        return Arc::new(Array::TextArray(TextArray::String64(Arc::new(unsafe {
            std::mem::transmute::<StringArray<T>, StringArray<u64>>(str_arr)
        }))));
    }
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u32>() {
        Arc::new(Array::TextArray(TextArray::String32(Arc::new(unsafe {
            std::mem::transmute::<StringArray<T>, StringArray<u32>>(str_arr)
        }))))
    } else {
        panic!("Unsupported offset type for StringArray (expected u32 or u64)");
    }
}

/// Imports a Utf8View array from Arrow C format into a MinArrow StringArray.
///
/// Arrow's Utf8View layout uses 16-byte view structs per element plus variadic
/// data buffers, which is structurally different from MinArrow's offsets+data
/// representation. Data is always copied during this conversion.
///
/// Each view struct is either:
/// - Short string (length <= 12): `[i32 length][12 bytes inline data]`
/// - Long string (length > 12): `[i32 length][4 bytes prefix][i32 buf_index][i32 offset]`
///
/// In the Arrow C Data Interface, Utf8View arrays have buffers:
/// `[validity, views, variadic_buf_0, ..., variadic_buf_n, variadic_sizes]`
///
/// # Safety
/// `arr` must contain a valid Utf8View ArrowArray.
unsafe fn import_utf8_view(arr: &ArrowArray, ownership: Option<Box<ArrowArray>>) -> Arc<Array> {
    let len = arr.length as usize;

    if len == 0 {
        // Clean up ownership if provided
        if let Some(mut arr_box) = ownership {
            if let Some(release) = arr_box.release {
                unsafe { release(&mut *arr_box as *mut ArrowArray) };
            }
        }
        return Arc::new(Array::from_string32(StringArray::<u32>::default()));
    }

    let n_buffers = arr.n_buffers as usize;
    let buffers = unsafe { slice::from_raw_parts(arr.buffers, n_buffers) };
    let null_ptr = buffers[0];
    let views_ptr = buffers[1] as *const u8;

    // Variadic data buffers are at indices 2..n_buffers-1
    // The last buffer stores variadic buffer sizes as int64
    let n_variadic = if n_buffers > 3 { n_buffers - 3 } else { 0 };

    // Pre-read null bitmap so we can skip undefined views at null positions
    let null_bitmap: Option<&[u8]> = if !null_ptr.is_null() {
        let bitmap_bytes = (len + 7) / 8;
        Some(unsafe { slice::from_raw_parts(null_ptr, bitmap_bytes) })
    } else {
        None
    };

    // Build offsets and contiguous data from views
    let mut offsets = Vec64::<u32>::with_capacity(len + 1);
    let mut data = Vec64::<u8>::new();
    offsets.push(0u32);

    for i in 0..len {
        // Check null bitmap — view bytes are undefined for null elements
        if let Some(bitmap) = null_bitmap {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            if (bitmap[byte_idx] & (1 << bit_idx)) == 0 {
                // Null element: skip view, push same offset (empty string)
                offsets.push(data.len() as u32);
                continue;
            }
        }

        let view = unsafe { views_ptr.add(i * 16) };
        let str_len = unsafe { *(view as *const i32) } as usize;

        if str_len <= 12 {
            // Short string: inline data at bytes 4..4+str_len
            let inline_data = unsafe { slice::from_raw_parts(view.add(4), str_len) };
            data.extend_from_slice(inline_data);
        } else {
            // Long string: buf_index at bytes 8..12, offset at bytes 12..16
            let buf_index = unsafe { *(view.add(8) as *const i32) } as usize;
            let buf_offset = unsafe { *(view.add(12) as *const i32) } as usize;
            assert!(
                buf_index < n_variadic,
                "Utf8View: buf_index {} out of range (have {} variadic buffers)",
                buf_index,
                n_variadic
            );
            let data_buf = buffers[2 + buf_index] as *const u8;
            let str_data = unsafe { slice::from_raw_parts(data_buf.add(buf_offset), str_len) };
            data.extend_from_slice(str_data);
        }

        offsets.push(data.len() as u32);
    }

    // Null mask
    let null_mask = if !null_ptr.is_null() {
        Some(unsafe { Bitmask::from_raw_slice(null_ptr, len) })
    } else {
        None
    };

    let str_arr = StringArray::<u32>::new(data, null_mask, offsets);

    // Clean up ownership if provided — data was copied, so we can release now
    if let Some(mut arr_box) = ownership {
        if let Some(release) = arr_box.release {
            unsafe { release(&mut *arr_box as *mut ArrowArray) };
        }
    }

    Arc::new(Array::TextArray(TextArray::String32(Arc::new(str_arr))))
}

/// Imports a categorical array and dictionary from Arrow C format.
///
/// When `ownership` is `Some`, the codes buffer is zero-copy: a `ForeignBuffer`
/// wraps the raw pointer and keeps the ArrowArray alive via its release callback.
/// When `ownership` is `None`, the codes are copied.
///
/// Dictionary strings are always copied because Arrow stores them as contiguous
/// offsets+data while MinArrow stores them as `Vec64<String>` with individual
/// heap allocations.
///
/// # Safety
/// Caller must ensure dictionary pointers are valid and formatted correctly.
unsafe fn import_categorical(
    arr: &ArrowArray,
    sch: &ArrowSchema,
    index_type: CategoricalIndexType,
    ownership: Option<Box<ArrowArray>>,
) -> Arc<Array> {
    // buffers: [null, codes]
    let len = arr.length as usize;
    let offset = arr.offset as usize;
    let buffers = unsafe { slice::from_raw_parts(arr.buffers, 2) };
    let null_ptr = buffers[0];
    let codes_ptr = buffers[1];

    // Import dictionary strings into Vec64<String>.
    //
    // When the schema carries a dictionary pointer we use it directly.
    // Some producers (e.g. pandas Series via __arrow_c_stream__) leave
    // sch.dictionary null while still populating arr.dictionary. In that
    // case we construct a synthetic UTF-8 schema so import_from_c can
    // interpret the dictionary values array.
    let synthetic_schema;
    let dict_sch_ptr: *const ArrowSchema = if !sch.dictionary.is_null() {
        sch.dictionary as *const _
    } else {
        synthetic_schema = ArrowSchema {
            format: b"u\0".as_ptr() as *const c_char,
            name: b"\0".as_ptr() as *const c_char,
            metadata: ptr::null(),
            flags: 0,
            n_children: 0,
            children: ptr::null_mut(),
            dictionary: ptr::null_mut(),
            release: None,
            private_data: ptr::null_mut(),
        };
        &synthetic_schema as *const _
    };
    let dict = unsafe { import_from_c(arr.dictionary as *const _, dict_sch_ptr) };
    let dict_strings = match dict.as_ref() {
        Array::TextArray(TextArray::String32(s)) => (0..s.len())
            .map(|i| s.get(i).unwrap_or_default().to_string())
            .collect(),
        #[cfg(feature = "large_string")]
        Array::TextArray(TextArray::String64(s)) => (0..s.len())
            .map(|i| s.get(i).unwrap_or_default().to_string())
            .collect(),
        _ => panic!("Expected String32 dictionary"),
    };
    // Honours bit-level offset for the codes' validity bitmap.
    let null_mask = if !null_ptr.is_null() {
        Some(unsafe { import_null_mask_offset(null_ptr, offset, len) })
    } else {
        None
    };

    /// Builds a zero-copy or copied `Buffer<T>` for the codes, applying `offset`
    /// to advance the pointer / source slice into the windowed range.
    unsafe fn build_codes<T: Integer>(
        codes_ptr: *const u8,
        offset: usize,
        len: usize,
        ownership: Option<Box<ArrowArray>>,
    ) -> Buffer<T> {
        let typed_base = codes_ptr as *const T;
        let typed_start = unsafe { typed_base.add(offset) };
        if let Some(arr_box) = ownership {
            let data_len_bytes = len * std::mem::size_of::<T>();
            let foreign = ForeignBuffer {
                ptr: typed_start as *const u8,
                len: data_len_bytes,
                array: Some(arr_box),
            };
            let shared = SharedBuffer::from_owner(foreign);
            Buffer::from_shared(shared)
        } else {
            let data = unsafe { slice::from_raw_parts(typed_start, len) };
            Vec64::from(data).into()
        }
    }

    // Build codes & wrap
    match index_type {
        #[cfg(feature = "default_categorical_8")]
        CategoricalIndexType::UInt8 => {
            let codes_buf = unsafe { build_codes::<u8>(codes_ptr, offset, len, ownership) };
            let arr = CategoricalArray::<u8>::new(codes_buf, dict_strings, null_mask);
            Arc::new(Array::TextArray(TextArray::Categorical8(Arc::new(arr))))
        }
        #[cfg(feature = "extended_categorical")]
        CategoricalIndexType::UInt16 => {
            let codes_buf = unsafe { build_codes::<u16>(codes_ptr, offset, len, ownership) };
            let arr = CategoricalArray::<u16>::new(codes_buf, dict_strings, null_mask);
            Arc::new(Array::TextArray(TextArray::Categorical16(Arc::new(arr))))
        }
        #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
        CategoricalIndexType::UInt32 => {
            let codes_buf = unsafe { build_codes::<u32>(codes_ptr, offset, len, ownership) };
            let arr = CategoricalArray::<u32>::new(codes_buf, dict_strings, null_mask);
            Arc::new(Array::TextArray(TextArray::Categorical32(Arc::new(arr))))
        }
        #[cfg(feature = "extended_categorical")]
        CategoricalIndexType::UInt64 => {
            let codes_buf = unsafe { build_codes::<u64>(codes_ptr, offset, len, ownership) };
            let arr = CategoricalArray::<u64>::new(codes_buf, dict_strings, null_mask);
            Arc::new(Array::TextArray(TextArray::Categorical64(Arc::new(arr))))
        }
    }
}

/// Imports a categorical array from Arrow C format, narrowing i32 indices to u8.
///
/// Used when `default_categorical_8` is on and Arrow sends the standard i32 dictionary
/// indices. Reads each i32 code and casts to u8. Panics if any index exceeds 255.
///
/// Dictionary strings are always copied, same as `import_categorical`.
///
/// # Safety
/// Caller must ensure the ArrowArray and ArrowSchema pointers are valid.
#[cfg(all(feature = "default_categorical_8", not(feature = "extended_categorical")))]
unsafe fn import_categorical_narrow_to_u8(
    arr: &ArrowArray,
    sch: &ArrowSchema,
    ownership: Option<Box<ArrowArray>>,
) -> Arc<Array> {
    let len = arr.length as usize;
    let offset = arr.offset as usize;
    let buffers = unsafe { slice::from_raw_parts(arr.buffers, 2) };
    let null_ptr = buffers[0];
    let codes_ptr = unsafe { (buffers[1] as *const i32).add(offset) };

    // Import dictionary strings into Vec64<String>, same as import_categorical
    let synthetic_schema;
    let dict_sch_ptr: *const ArrowSchema = if !sch.dictionary.is_null() {
        sch.dictionary as *const _
    } else {
        synthetic_schema = ArrowSchema {
            format: b"u\0".as_ptr() as *const c_char,
            name: b"\0".as_ptr() as *const c_char,
            metadata: ptr::null(),
            flags: 0,
            n_children: 0,
            children: ptr::null_mut(),
            dictionary: ptr::null_mut(),
            release: None,
            private_data: ptr::null_mut(),
        };
        &synthetic_schema as *const _
    };
    let dict = unsafe { import_from_c(arr.dictionary as *const _, dict_sch_ptr) };
    let dict_strings = match dict.as_ref() {
        Array::TextArray(TextArray::String32(s)) => (0..s.len())
            .map(|i| s.get(i).unwrap_or_default().to_string())
            .collect(),
        #[cfg(feature = "large_string")]
        Array::TextArray(TextArray::String64(s)) => (0..s.len())
            .map(|i| s.get(i).unwrap_or_default().to_string())
            .collect(),
        _ => panic!("Expected String32 dictionary"),
    };

    let null_mask = if !null_ptr.is_null() {
        Some(unsafe { import_null_mask_offset(null_ptr, offset, len) })
    } else {
        None
    };

    // Read i32 codes and narrow to u8 (codes_ptr is already offset-adjusted).
    let i32_codes = unsafe { slice::from_raw_parts(codes_ptr, len) };
    let mut u8_codes = Vec64::<u8>::with_capacity(len);
    for &code in i32_codes {
        assert!(
            code >= 0 && code <= u8::MAX as i32,
            "Arrow dictionary index {} exceeds u8 range (0..255)",
            code
        );
        u8_codes.push(code as u8);
    }

    // Release the source ArrowArray if we own it
    if let Some(mut arr_box) = ownership {
        if let Some(release) = arr_box.release {
            unsafe { release(&mut *arr_box as *mut ArrowArray) };
        }
    }

    let codes_buf: Buffer<u8> = u8_codes.into();
    let cat = CategoricalArray::<u8>::new(codes_buf, dict_strings, null_mask);
    Arc::new(Array::TextArray(TextArray::Categorical8(Arc::new(cat))))
}

/// Imports a datetime array from Arrow C format.
///
/// # Arguments
/// * `arr` - Reference to the ArrowArray containing the data
/// * `ownership` - If `Some(box)`, takes ownership and does zero-copy via ForeignBuffer.
///                 If `None`, copies the data. This is needed for dictionary arrays inside
///                 categoricals, where the parent's release callback owns the child memory.
/// * `unit` - The time unit for the datetime values
///
/// # Safety
/// `arr` must contain valid time values and optional null mask.
#[cfg(feature = "datetime")]
unsafe fn import_datetime<T: Integer>(
    arr: &ArrowArray,
    ownership: Option<Box<ArrowArray>>,
    unit: crate::TimeUnit,
) -> Arc<Array> {
    let len = arr.length as usize;
    let offset = arr.offset as usize;
    let buffers = unsafe { std::slice::from_raw_parts(arr.buffers, 2) };
    let data_ptr = unsafe { (buffers[1] as *const T).add(offset) };
    let data_len_bytes = len * std::mem::size_of::<T>();

    // Null mask is always copied (small overhead). Honours bit-level offset.
    let null_mask = if !buffers[0].is_null() {
        Some(unsafe { import_null_mask_offset(buffers[0], offset, len) })
    } else {
        None
    };

    let buffer: Buffer<T> = if let Some(arr_box) = ownership {
        // Zero-copy: wrap foreign buffer
        let foreign = ForeignBuffer {
            ptr: data_ptr as *const u8,
            len: data_len_bytes,
            array: Some(arr_box),
        };
        let shared = SharedBuffer::from_owner(foreign);
        Buffer::from_shared(shared)
    } else {
        // Copy: dictionary arrays inside categoricals have their memory owned by the
        // parent ArrowArray's release callback, so we can't take ownership of them
        let data = unsafe { std::slice::from_raw_parts(data_ptr, len) };
        Vec64::from(data).into()
    };

    let dt_arr = DatetimeArray::<T> {
        data: buffer,
        null_mask,
        time_unit: unit,
    };

    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i64>() {
        Arc::new(Array::TemporalArray(TemporalArray::Datetime64(Arc::new(
            unsafe { std::mem::transmute::<DatetimeArray<T>, DatetimeArray<i64>>(dt_arr) },
        ))))
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i32>() {
        Arc::new(Array::TemporalArray(TemporalArray::Datetime32(Arc::new(
            unsafe { std::mem::transmute::<DatetimeArray<T>, DatetimeArray<i32>>(dt_arr) },
        ))))
    } else {
        panic!("Unsupported DatetimeArray type (expected i32 or i64)");
    }
}

/// Verifies that all buffer pointers are 64-byte aligned.
/// This happens automatically when creating `Minarrow` buffers
/// so shouldn't be an issue.
///
/// Note: For empty arrays (length=0), we skip alignment checks because PyArrow
/// may return sentinel pointers (e.g., 0x4) for empty buffers that aren't
/// 64-byte aligned but also aren't actually dereferenced.
fn check_alignment(buf_ptrs: &mut Vec64<*const u8>, length: i64) {
    // Skip alignment check for empty arrays - buffer pointers may be invalid/sentinel values
    if length == 0 {
        return;
    }
    for &p in buf_ptrs.iter().take(3) {
        if !p.is_null() {
            assert_eq!(
                (p as usize) % 64,
                0,
                "FFI: Array buffer pointer {:p} is not 64-byte aligned",
                p
            );
        }
    }
}

/// Builds and returns ArrowArray and ArrowSchema for export.
///
/// `offset` is written into `ArrowArray.offset`; the consumer slices each
/// buffer via this offset to obtain the windowed range. Pass `0` for a
/// full-array export.
///
/// # Safety
/// Caller must ensure buffer pointers remain valid for the lifetime of the consumer.
fn create_arrow_export(
    array: Arc<Array>,
    schema: Schema,
    mut buf_ptrs: Vec64<*const u8>,
    n_buffers: i64,
    length: i64,
    offset: i64,
    name_cstr: CString,
) -> (*mut ArrowArray, *mut ArrowSchema) {
    let null_count = if buf_ptrs[0].is_null() { 0 } else { -1 };

    let field = &schema.fields[0];

    // Format string as CString
    let format_cstr = fmt_c(field.dtype.clone());
    let format_ptr = format_cstr.as_ptr();

    // Metadata
    let metadata_bytes = if field.metadata.is_empty() {
        None
    } else {
        Some(encode_arrow_metadata(&field.metadata))
    };
    let metadata_ptr = metadata_bytes
        .as_ref()
        .map(|b| b.as_ptr() as *const c_char)
        .unwrap_or(ptr::null());

    // ArrowArray
    let arr = Box::new(ArrowArray {
        length,
        null_count,
        offset,
        n_buffers,
        n_children: 0,
        buffers: buf_ptrs.as_mut_ptr(),
        children: ptr::null_mut(),
        dictionary: ptr::null_mut(),
        release: Some(release_arrow_array),
        private_data: ptr::null_mut(),
    });

    // ArrowSchema
    let flags = if field.nullable { 1 } else { 0 };
    let schema_box = Box::new(ArrowSchema {
        format: format_ptr,
        name: name_cstr.as_ptr(),
        metadata: metadata_ptr,
        flags,
        n_children: 0,
        children: ptr::null_mut(),
        dictionary: ptr::null_mut(),
        release: Some(release_arrow_schema),
        private_data: ptr::null_mut(),
    });

    let holder = Box::new(Holder {
        array,
        _schema: schema_box.clone(),
        buf_ptrs,
        name_cstr,
        format_cstr,
        metadata_bytes,
    });

    let arr_ptr = Box::into_raw(arr);
    unsafe {
        (*arr_ptr).private_data = Box::into_raw(holder) as *mut c_void;
    }

    (arr_ptr, Box::into_raw(schema_box))
}

// ─── Arrow C Stream Interface ─────────────────────────────────────────
//
// Implements the Arrow C Stream Interface for streaming sequences of arrays.
// Used by the PyCapsule protocol: `__arrow_c_stream__`.
//
// Two stream flavours:
// 1. **Record batch stream** - yields struct arrays (one per batch), used for
//    Table / SuperTable / DataFrame exchange.
// 2. **Array stream** - yields plain arrays (one per chunk), used for
//    ChunkedArray / SuperArray exchange.

/// Private holder for record-batch stream state.
struct RecordBatchStreamHolder {
    /// Schema fields for the struct array (one per column).
    fields: Vec<crate::Field>,
    /// Batches to yield: each batch is a Vec of column (Arc<Array>, Schema) pairs.
    batches: Vec<Vec<(Arc<Array>, Schema)>>,
    /// Index of the next batch to return from get_next.
    cursor: usize,
    /// Last error message.
    last_error: Option<CString>,
    /// Optional schema-level metadata, encoded in Arrow binary format.
    /// Kept alive so the pointer in ArrowSchema.metadata remains valid.
    schema_metadata: Option<Vec<u8>>,
}

/// Private holder for plain-array stream state.
struct ArrayStreamHolder {
    /// Field describing the array type.
    field: crate::Field,
    /// Chunks to yield.
    chunks: Vec<Arc<Array>>,
    /// Index of the next chunk to return from get_next.
    cursor: usize,
    /// Last error message.
    last_error: Option<CString>,
}

// ── Struct array helpers ────────────────────────────────────────────────

/// Exports a single record batch (a set of column arrays with a shared schema)
/// as an Arrow struct array + struct schema pair.
///
/// The resulting `ArrowArray` has format `"+s"` with one child per column.
/// Callers must eventually call the release callback on the returned pointers.
///
/// When `metadata` is provided, it is encoded and attached to the top-level
/// struct ArrowSchema. The `table_metadata` feature controls whether metadata
/// is preserved on the minarrow rust side during import.
pub fn export_struct_to_c(
    columns: Vec<(Arc<Array>, Schema)>,
    metadata: Option<std::collections::BTreeMap<String, String>>,
) -> (*mut ArrowArray, *mut ArrowSchema) {
    export_struct_to_c_inner(columns, metadata)
}

fn export_struct_to_c_inner(
    columns: Vec<(Arc<Array>, Schema)>,
    metadata: Option<std::collections::BTreeMap<String, String>>,
) -> (*mut ArrowArray, *mut ArrowSchema) {
    let n_cols = columns.len();
    let n_rows = if n_cols > 0 {
        columns[0].0.len() as i64
    } else {
        0
    };

    // Export each column individually
    let mut child_array_ptrs: Vec<*mut ArrowArray> = Vec::with_capacity(n_cols);
    let mut child_schema_ptrs: Vec<*mut ArrowSchema> = Vec::with_capacity(n_cols);

    for (array, schema) in &columns {
        let (arr_ptr, sch_ptr) = export_to_c(array.clone(), schema.clone());
        child_array_ptrs.push(arr_ptr);
        child_schema_ptrs.push(sch_ptr);
    }

    // Build the parent struct ArrowArray
    let children_arr_box = child_array_ptrs.into_boxed_slice();
    let children_arr_ptr = Box::into_raw(children_arr_box) as *mut *mut ArrowArray;

    let struct_arr = Box::new(ArrowArray {
        length: n_rows,
        null_count: 0,
        offset: 0,
        n_buffers: 1, // struct arrays have a single null bitmap buffer
        n_children: n_cols as i64,
        buffers: Box::into_raw(Box::new(ptr::null::<u8>())) as *mut *const u8,
        children: children_arr_ptr,
        dictionary: ptr::null_mut(),
        release: Some(release_struct_array),
        private_data: ptr::null_mut(),
    });

    // Build the parent struct ArrowSchema
    let children_sch_box = child_schema_ptrs.into_boxed_slice();
    let children_sch_ptr = Box::into_raw(children_sch_box) as *mut *mut ArrowSchema;

    let format_cstr = CString::new("+s").unwrap();
    let name_cstr = CString::new("").unwrap();

    let metadata_bytes = metadata.map(|m| encode_arrow_metadata(&m));
    let metadata_ptr = metadata_bytes
        .as_ref()
        .map(|b| b.as_ptr() as *const c_char)
        .unwrap_or(ptr::null());

    let struct_holder = Box::new(StructSchemaHolder {
        format_cstr,
        name_cstr,
        metadata_bytes,
    });

    let struct_sch = Box::new(ArrowSchema {
        format: struct_holder.format_cstr.as_ptr(),
        name: struct_holder.name_cstr.as_ptr(),
        metadata: metadata_ptr,
        flags: 0,
        n_children: n_cols as i64,
        children: children_sch_ptr,
        dictionary: ptr::null_mut(),
        release: Some(release_struct_schema),
        private_data: Box::into_raw(struct_holder) as *mut c_void,
    });

    (Box::into_raw(struct_arr), Box::into_raw(struct_sch))
}

/// Exports a [`crate::TableV`] as an Arrow struct array + struct schema pair.
///
/// Each column is exported through [`export_view_to_c`] so the windowed
/// `(offset, len)` is carried at the Arrow C layer without copying buffers.
/// The parent struct ArrowArray uses `offset = 0` and `length = view.len()`,
/// while each child carries its own `offset` matching the view.
#[cfg(feature = "views")]
pub fn export_table_view_to_c(
    view: &crate::TableV,
    metadata: Option<std::collections::BTreeMap<String, String>>,
) -> (*mut ArrowArray, *mut ArrowSchema) {
    let n_cols = view.cols.len();
    let n_rows = view.len as i64;

    let mut child_array_ptrs: Vec<*mut ArrowArray> = Vec::with_capacity(n_cols);
    let mut child_schema_ptrs: Vec<*mut ArrowSchema> = Vec::with_capacity(n_cols);

    for (field, col) in view.fields.iter().zip(view.cols.iter()) {
        let col_schema = Schema::from(vec![(**field).clone()]);
        let arr_arc = Arc::new(col.array.clone());
        let (arr_ptr, sch_ptr) = export_view_to_c(
            arr_arc,
            col_schema,
            col.offset as i64,
            col.len() as i64,
        );
        child_array_ptrs.push(arr_ptr);
        child_schema_ptrs.push(sch_ptr);
    }

    let children_arr_box = child_array_ptrs.into_boxed_slice();
    let children_arr_ptr = Box::into_raw(children_arr_box) as *mut *mut ArrowArray;

    let struct_arr = Box::new(ArrowArray {
        length: n_rows,
        null_count: 0,
        offset: 0,
        n_buffers: 1, // struct arrays have a single null bitmap buffer
        n_children: n_cols as i64,
        buffers: Box::into_raw(Box::new(ptr::null::<u8>())) as *mut *const u8,
        children: children_arr_ptr,
        dictionary: ptr::null_mut(),
        release: Some(release_struct_array),
        private_data: ptr::null_mut(),
    });

    let children_sch_box = child_schema_ptrs.into_boxed_slice();
    let children_sch_ptr = Box::into_raw(children_sch_box) as *mut *mut ArrowSchema;

    let format_cstr = CString::new("+s").unwrap();
    let name_cstr = CString::new(view.name.clone()).unwrap_or_default();

    let metadata_bytes = metadata.map(|m| encode_arrow_metadata(&m));
    let metadata_ptr = metadata_bytes
        .as_ref()
        .map(|b| b.as_ptr() as *const c_char)
        .unwrap_or(ptr::null());

    let struct_holder = Box::new(StructSchemaHolder {
        format_cstr,
        name_cstr,
        metadata_bytes,
    });

    let struct_sch = Box::new(ArrowSchema {
        format: struct_holder.format_cstr.as_ptr(),
        name: struct_holder.name_cstr.as_ptr(),
        metadata: metadata_ptr,
        flags: 0,
        n_children: n_cols as i64,
        children: children_sch_ptr,
        dictionary: ptr::null_mut(),
        release: Some(release_struct_schema),
        private_data: Box::into_raw(struct_holder) as *mut c_void,
    });

    (Box::into_raw(struct_arr), Box::into_raw(struct_sch))
}

/// Keeps struct-level schema strings and metadata alive for ArrowSchema pointers.
#[allow(dead_code)]
struct StructSchemaHolder {
    format_cstr: CString,
    name_cstr: CString,
    /// Encoded Arrow metadata bytes kept alive for the pointer in ArrowSchema.metadata.
    metadata_bytes: Option<Vec<u8>>,
}

/// Encodes key-value pairs into the Arrow C Data Interface metadata binary format.
///
/// Format: int32 num_pairs, then for each pair: int32 key_len, key bytes, int32 value_len, value bytes.
/// All integers are little-endian as per the Arrow spec.
fn encode_arrow_metadata(pairs: &std::collections::BTreeMap<String, String>) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(pairs.len() as i32).to_le_bytes());
    for (k, v) in pairs {
        buf.extend_from_slice(&(k.len() as i32).to_le_bytes());
        buf.extend_from_slice(k.as_bytes());
        buf.extend_from_slice(&(v.len() as i32).to_le_bytes());
        buf.extend_from_slice(v.as_bytes());
    }
    buf
}

/// Decodes Arrow C Data Interface metadata from a raw pointer into key-value pairs.
///
/// Returns `None` if the pointer is null. The binary format is:
/// int32 num_pairs, then for each pair: int32 key_len, key bytes, int32 value_len, value bytes.
/// All integers are little-endian.
///
/// # Safety
/// The pointer must be null or point to a valid Arrow metadata buffer.
pub unsafe fn decode_arrow_metadata(
    ptr: *const c_char,
) -> Option<std::collections::BTreeMap<String, String>> {
    if ptr.is_null() {
        return None;
    }
    let mut cursor = ptr as *const u8;

    let num_pairs = i32::from_le_bytes(unsafe {
        let bytes = slice::from_raw_parts(cursor, 4);
        cursor = cursor.add(4);
        bytes.try_into().unwrap()
    });

    let mut map = std::collections::BTreeMap::new();
    for _ in 0..num_pairs {
        let key_len = i32::from_le_bytes(unsafe {
            let bytes = slice::from_raw_parts(cursor, 4);
            cursor = cursor.add(4);
            bytes.try_into().unwrap()
        }) as usize;
        let key = unsafe {
            let bytes = slice::from_raw_parts(cursor, key_len);
            cursor = cursor.add(key_len);
            String::from_utf8_lossy(bytes).into_owned()
        };

        let val_len = i32::from_le_bytes(unsafe {
            let bytes = slice::from_raw_parts(cursor, 4);
            cursor = cursor.add(4);
            bytes.try_into().unwrap()
        }) as usize;
        let val = unsafe {
            let bytes = slice::from_raw_parts(cursor, val_len);
            cursor = cursor.add(val_len);
            String::from_utf8_lossy(bytes).into_owned()
        };

        map.insert(key, val);
    }
    Some(map)
}

/// Release callback for struct ArrowArrays.
/// Frees each child's ArrowArray via their own release callbacks, then frees
/// the parent's buffers and children array.
unsafe extern "C" fn release_struct_array(arr: *mut ArrowArray) {
    if arr.is_null() || (unsafe { &*arr }).release.is_none() {
        return;
    }
    let a = unsafe { &*arr };
    let n_children = a.n_children as usize;

    // Release each child array
    if !a.children.is_null() {
        let children = unsafe { slice::from_raw_parts_mut(a.children, n_children) };
        for child_ptr in children.iter_mut() {
            if !child_ptr.is_null() {
                let child = unsafe { &mut **child_ptr };
                if let Some(release) = child.release {
                    unsafe { release(*child_ptr) };
                }
                let _ = unsafe { Box::from_raw(*child_ptr) };
            }
        }
        // Free the children pointer array
        let _ = unsafe {
            Box::from_raw(
                slice::from_raw_parts_mut(a.children, n_children) as *mut [*mut ArrowArray]
            )
        };
    }

    // Free the buffers pointer (single null bitmap)
    if !a.buffers.is_null() {
        let _ = unsafe { Box::from_raw(a.buffers as *mut *const u8) };
    }

    unsafe { ptr::write_bytes(arr, 0, 1) };
}

/// Release callback for struct ArrowSchemas.
unsafe extern "C" fn release_struct_schema(sch: *mut ArrowSchema) {
    if sch.is_null() || (unsafe { &*sch }).release.is_none() {
        return;
    }
    let s = unsafe { &*sch };
    let n_children = s.n_children as usize;

    // Release each child schema
    if !s.children.is_null() {
        let children = unsafe { slice::from_raw_parts_mut(s.children, n_children) };
        for child_ptr in children.iter_mut() {
            if !child_ptr.is_null() {
                let child = unsafe { &mut **child_ptr };
                if let Some(release) = child.release {
                    unsafe { release(*child_ptr) };
                }
                let _ = unsafe { Box::from_raw(*child_ptr) };
            }
        }
        let _ = unsafe {
            Box::from_raw(
                slice::from_raw_parts_mut(s.children, n_children) as *mut [*mut ArrowSchema]
            )
        };
    }

    // Free the private data
    if !s.private_data.is_null() {
        let _ = unsafe { Box::from_raw(s.private_data as *mut StructSchemaHolder) };
    }

    unsafe { ptr::write_bytes(sch, 0, 1) };
}

// ── Stream export: record batches ───────────────────────────────────────

/// Creates an ArrowArrayStream that yields record batches as struct arrays.
///
/// Each batch is a vector of column (Array, Schema) pairs. The stream schema
/// is a struct type with one child field per column, derived from the first batch.
///
/// # Arguments
/// * `batches` - one or more batches of column arrays with their schemas
///
/// # Returns
/// A heap-allocated ArrowArrayStream. The caller must eventually call its
/// release callback or pass it to a consumer that will.
pub fn export_record_batch_stream(
    batches: Vec<Vec<(Arc<Array>, Schema)>>,
    fields: Vec<crate::Field>,
) -> Box<ArrowArrayStream> {
    export_record_batch_stream_with_metadata(batches, fields, None)
}

/// Like [`export_record_batch_stream`], but attaches key-value metadata to the
/// struct-level schema. Use this to preserve a table name or other metadata
/// across the FFI boundary.
pub fn export_record_batch_stream_with_metadata(
    batches: Vec<Vec<(Arc<Array>, Schema)>>,
    fields: Vec<crate::Field>,
    metadata: Option<std::collections::BTreeMap<String, String>>,
) -> Box<ArrowArrayStream> {
    let schema_metadata = metadata.map(|m| encode_arrow_metadata(&m));
    let holder = Box::new(RecordBatchStreamHolder {
        fields,
        batches,
        cursor: 0,
        last_error: None,
        schema_metadata,
    });

    let stream = Box::new(ArrowArrayStream {
        get_schema: Some(rb_stream_get_schema),
        get_next: Some(rb_stream_get_next),
        get_last_error: Some(stream_get_last_error::<RecordBatchStreamHolder>),
        release: Some(rb_stream_release),
        private_data: Box::into_raw(holder) as *mut c_void,
    });

    stream
}

unsafe extern "C" fn rb_stream_get_schema(
    stream: *mut ArrowArrayStream,
    out: *mut ArrowSchema,
) -> i32 {
    let holder = unsafe { &*((*stream).private_data as *const RecordBatchStreamHolder) };

    // Build a struct schema from the fields
    let n_fields = holder.fields.len();
    let mut child_schemas: Vec<*mut ArrowSchema> = Vec::with_capacity(n_fields);

    for field in &holder.fields {
        let format_cstr = fmt_c(field.dtype.clone());
        let name_cstr = CString::new(field.name.clone()).unwrap_or_default();
        let flags = if field.nullable { 2 } else { 0 };

        let metadata_bytes = if field.metadata.is_empty() {
            None
        } else {
            Some(encode_arrow_metadata(&field.metadata))
        };
        let metadata_ptr = metadata_bytes
            .as_ref()
            .map(|b| b.as_ptr() as *const c_char)
            .unwrap_or(ptr::null());

        let child_holder = Box::new(StructSchemaHolder {
            format_cstr,
            name_cstr,
            metadata_bytes,
        });

        let child = Box::new(ArrowSchema {
            format: child_holder.format_cstr.as_ptr(),
            name: child_holder.name_cstr.as_ptr(),
            metadata: metadata_ptr,
            flags,
            n_children: 0,
            children: ptr::null_mut(),
            dictionary: ptr::null_mut(),
            release: Some(release_struct_schema),
            private_data: Box::into_raw(child_holder) as *mut c_void,
        });

        child_schemas.push(Box::into_raw(child));
    }

    let children_box = child_schemas.into_boxed_slice();
    let children_ptr = Box::into_raw(children_box) as *mut *mut ArrowSchema;

    let format_cstr = CString::new("+s").unwrap();
    let name_cstr = CString::new("").unwrap();

    let metadata_bytes = holder.schema_metadata.clone();
    let metadata_ptr = metadata_bytes
        .as_ref()
        .map(|b| b.as_ptr() as *const c_char)
        .unwrap_or(ptr::null());

    let schema_holder = Box::new(StructSchemaHolder {
        format_cstr,
        name_cstr,
        metadata_bytes,
    });

    let struct_schema = ArrowSchema {
        format: schema_holder.format_cstr.as_ptr(),
        name: schema_holder.name_cstr.as_ptr(),
        metadata: metadata_ptr,
        flags: 0,
        n_children: n_fields as i64,
        children: children_ptr,
        dictionary: ptr::null_mut(),
        release: Some(release_struct_schema),
        private_data: Box::into_raw(schema_holder) as *mut c_void,
    };

    unsafe { ptr::write(out, struct_schema) };
    0
}

unsafe extern "C" fn rb_stream_get_next(
    stream: *mut ArrowArrayStream,
    out: *mut ArrowArray,
) -> i32 {
    let holder = unsafe { &mut *((*stream).private_data as *mut RecordBatchStreamHolder) };

    if holder.cursor >= holder.batches.len() {
        // Signal end of stream: write empty array with release = None
        unsafe { ptr::write(out, ArrowArray::empty()) };
        return 0;
    }

    let batch = holder.batches[holder.cursor].clone();
    holder.cursor += 1;

    let (arr_ptr, _sch_ptr) = export_struct_to_c_inner(batch, None);

    // Copy the exported struct array into the caller's out pointer
    unsafe {
        ptr::write(out, ptr::read(arr_ptr));
        // Free the box wrapper without running drop on the ArrowArray itself
        // (caller now owns the data through the out pointer)
        std::alloc::dealloc(arr_ptr as *mut u8, std::alloc::Layout::for_value(&*arr_ptr));
    }

    // Release the schema (get_next only returns arrays, schema came from get_schema)
    unsafe {
        if let Some(release) = (*_sch_ptr).release {
            release(_sch_ptr);
        }
        let _ = Box::from_raw(_sch_ptr);
    }

    0
}

unsafe extern "C" fn rb_stream_release(stream: *mut ArrowArrayStream) {
    if stream.is_null() || (unsafe { &*stream }).release.is_none() {
        return;
    }
    let _ = unsafe { Box::from_raw((*stream).private_data as *mut RecordBatchStreamHolder) };
    unsafe { ptr::write_bytes(stream, 0, 1) };
}

// ── Stream export: plain arrays ─────────────────────────────────────────

/// Creates an ArrowArrayStream that yields plain arrays (one per chunk).
///
/// Used for SuperArray / ChunkedArray exchange.
///
/// # Arguments
/// * `chunks` - the arrays to yield
/// * `field` - the field describing the array type
///
/// # Returns
/// A heap-allocated ArrowArrayStream.
pub fn export_array_stream(chunks: Vec<Arc<Array>>, field: crate::Field) -> Box<ArrowArrayStream> {
    let holder = Box::new(ArrayStreamHolder {
        field,
        chunks,
        cursor: 0,
        last_error: None,
    });

    let stream = Box::new(ArrowArrayStream {
        get_schema: Some(arr_stream_get_schema),
        get_next: Some(arr_stream_get_next),
        get_last_error: Some(stream_get_last_error::<ArrayStreamHolder>),
        release: Some(arr_stream_release),
        private_data: Box::into_raw(holder) as *mut c_void,
    });

    stream
}

unsafe extern "C" fn arr_stream_get_schema(
    stream: *mut ArrowArrayStream,
    out: *mut ArrowSchema,
) -> i32 {
    let holder = unsafe { &*((*stream).private_data as *const ArrayStreamHolder) };
    let field = &holder.field;

    let format_cstr = fmt_c(field.dtype.clone());
    let name_cstr = CString::new(field.name.clone()).unwrap_or_default();
    let flags = if field.nullable { 2 } else { 0 };

    let schema_holder = Box::new(StructSchemaHolder {
        format_cstr,
        name_cstr,
        metadata_bytes: None,
    });

    let schema = ArrowSchema {
        format: schema_holder.format_cstr.as_ptr(),
        name: schema_holder.name_cstr.as_ptr(),
        metadata: ptr::null(),
        flags,
        n_children: 0,
        children: ptr::null_mut(),
        dictionary: ptr::null_mut(),
        release: Some(release_struct_schema),
        private_data: Box::into_raw(schema_holder) as *mut c_void,
    };

    unsafe { ptr::write(out, schema) };
    0
}

unsafe extern "C" fn arr_stream_get_next(
    stream: *mut ArrowArrayStream,
    out: *mut ArrowArray,
) -> i32 {
    let holder = unsafe { &mut *((*stream).private_data as *mut ArrayStreamHolder) };

    if holder.cursor >= holder.chunks.len() {
        unsafe { ptr::write(out, ArrowArray::empty()) };
        return 0;
    }

    let array = holder.chunks[holder.cursor].clone();
    holder.cursor += 1;

    let schema = Schema::from(vec![holder.field.clone()]);
    let (arr_ptr, sch_ptr) = export_to_c(array, schema);

    // Move the exported array into the caller's out pointer
    unsafe {
        ptr::write(out, ptr::read(arr_ptr));
        std::alloc::dealloc(arr_ptr as *mut u8, std::alloc::Layout::for_value(&*arr_ptr));
    }

    // Release the schema (not needed for get_next)
    unsafe {
        if let Some(release) = (*sch_ptr).release {
            release(sch_ptr);
        }
        let _ = Box::from_raw(sch_ptr);
    }

    0
}

unsafe extern "C" fn arr_stream_release(stream: *mut ArrowArrayStream) {
    if stream.is_null() || (unsafe { &*stream }).release.is_none() {
        return;
    }
    let _ = unsafe { Box::from_raw((*stream).private_data as *mut ArrayStreamHolder) };
    unsafe { ptr::write_bytes(stream, 0, 1) };
}

// ── Stream export: view variants ────────────────────────────────────────
//
// Mirror of the owned `ArrayStreamHolder` / `RecordBatchStreamHolder` flow,
// but each chunk/batch carries a windowed `(offset, len)` that is propagated
// to the per-chunk ArrowArray through `export_view_to_c`. No buffer copies
// occur for primitive, string, datetime, or boolean data.

/// Private holder for plain-array view streams (one [`crate::ArrayV`] per chunk).
#[cfg(feature = "views")]
struct ArrayViewStreamHolder {
    field: crate::Field,
    slices: Vec<crate::ArrayV>,
    cursor: usize,
    last_error: Option<CString>,
}

#[cfg(feature = "views")]
impl HasLastError for ArrayViewStreamHolder {
    fn last_error(&self) -> Option<&CString> {
        self.last_error.as_ref()
    }
}

/// Creates an ArrowArrayStream that yields one Arrow array per [`crate::ArrayV`] slice.
///
/// Used to export a windowed [`crate::SuperArrayV`] or any sequence of
/// `ArrayV` chunks without materialising them. Each slice's `(offset, len)`
/// is conveyed at the Arrow C layer.
#[cfg(feature = "views")]
pub fn export_array_view_stream(
    slices: Vec<crate::ArrayV>,
    field: crate::Field,
) -> Box<ArrowArrayStream> {
    let holder = Box::new(ArrayViewStreamHolder {
        field,
        slices,
        cursor: 0,
        last_error: None,
    });

    Box::new(ArrowArrayStream {
        get_schema: Some(arr_view_stream_get_schema),
        get_next: Some(arr_view_stream_get_next),
        get_last_error: Some(stream_get_last_error::<ArrayViewStreamHolder>),
        release: Some(arr_view_stream_release),
        private_data: Box::into_raw(holder) as *mut c_void,
    })
}

#[cfg(feature = "views")]
unsafe extern "C" fn arr_view_stream_get_schema(
    stream: *mut ArrowArrayStream,
    out: *mut ArrowSchema,
) -> i32 {
    let holder = unsafe { &*((*stream).private_data as *const ArrayViewStreamHolder) };
    let field = &holder.field;

    let format_cstr = fmt_c(field.dtype.clone());
    let name_cstr = CString::new(field.name.clone()).unwrap_or_default();
    let flags = if field.nullable { 2 } else { 0 };

    let schema_holder = Box::new(StructSchemaHolder {
        format_cstr,
        name_cstr,
        metadata_bytes: None,
    });

    let schema = ArrowSchema {
        format: schema_holder.format_cstr.as_ptr(),
        name: schema_holder.name_cstr.as_ptr(),
        metadata: ptr::null(),
        flags,
        n_children: 0,
        children: ptr::null_mut(),
        dictionary: ptr::null_mut(),
        release: Some(release_struct_schema),
        private_data: Box::into_raw(schema_holder) as *mut c_void,
    };

    unsafe { ptr::write(out, schema) };
    0
}

#[cfg(feature = "views")]
unsafe extern "C" fn arr_view_stream_get_next(
    stream: *mut ArrowArrayStream,
    out: *mut ArrowArray,
) -> i32 {
    let holder = unsafe { &mut *((*stream).private_data as *mut ArrayViewStreamHolder) };

    if holder.cursor >= holder.slices.len() {
        unsafe { ptr::write(out, ArrowArray::empty()) };
        return 0;
    }

    let slice = holder.slices[holder.cursor].clone();
    holder.cursor += 1;

    let schema = Schema::from(vec![holder.field.clone()]);
    let arr_arc = Arc::new(slice.array.clone());
    let (arr_ptr, sch_ptr) =
        export_view_to_c(arr_arc, schema, slice.offset as i64, slice.len() as i64);

    unsafe {
        ptr::write(out, ptr::read(arr_ptr));
        std::alloc::dealloc(arr_ptr as *mut u8, std::alloc::Layout::for_value(&*arr_ptr));
    }

    unsafe {
        if let Some(release) = (*sch_ptr).release {
            release(sch_ptr);
        }
        let _ = Box::from_raw(sch_ptr);
    }

    0
}

#[cfg(feature = "views")]
unsafe extern "C" fn arr_view_stream_release(stream: *mut ArrowArrayStream) {
    if stream.is_null() || (unsafe { &*stream }).release.is_none() {
        return;
    }
    let _ = unsafe { Box::from_raw((*stream).private_data as *mut ArrayViewStreamHolder) };
    unsafe { ptr::write_bytes(stream, 0, 1) };
}

/// Private holder for record-batch view streams (one [`crate::TableV`] per batch).
#[cfg(feature = "views")]
struct RecordBatchViewStreamHolder {
    fields: Vec<crate::Field>,
    batches: Vec<crate::TableV>,
    cursor: usize,
    last_error: Option<CString>,
    schema_metadata: Option<Vec<u8>>,
}

#[cfg(feature = "views")]
impl HasLastError for RecordBatchViewStreamHolder {
    fn last_error(&self) -> Option<&CString> {
        self.last_error.as_ref()
    }
}

/// Creates an ArrowArrayStream that yields one struct array per [`crate::TableV`] batch.
///
/// Each batch is exported through [`export_table_view_to_c`], so the
/// per-column `(offset, len)` propagates to each child ArrowArray without
/// copying buffers. The top-level struct schema mirrors `fields`.
#[cfg(feature = "views")]
pub fn export_record_batch_view_stream(
    batches: Vec<crate::TableV>,
    fields: Vec<crate::Field>,
    metadata: Option<std::collections::BTreeMap<String, String>>,
) -> Box<ArrowArrayStream> {
    let schema_metadata = metadata.map(|m| encode_arrow_metadata(&m));
    let holder = Box::new(RecordBatchViewStreamHolder {
        fields,
        batches,
        cursor: 0,
        last_error: None,
        schema_metadata,
    });

    Box::new(ArrowArrayStream {
        get_schema: Some(rb_view_stream_get_schema),
        get_next: Some(rb_view_stream_get_next),
        get_last_error: Some(stream_get_last_error::<RecordBatchViewStreamHolder>),
        release: Some(rb_view_stream_release),
        private_data: Box::into_raw(holder) as *mut c_void,
    })
}

#[cfg(feature = "views")]
unsafe extern "C" fn rb_view_stream_get_schema(
    stream: *mut ArrowArrayStream,
    out: *mut ArrowSchema,
) -> i32 {
    let holder = unsafe { &*((*stream).private_data as *const RecordBatchViewStreamHolder) };

    let n_fields = holder.fields.len();
    let mut child_schemas: Vec<*mut ArrowSchema> = Vec::with_capacity(n_fields);

    for field in &holder.fields {
        let format_cstr = fmt_c(field.dtype.clone());
        let name_cstr = CString::new(field.name.clone()).unwrap_or_default();
        let flags = if field.nullable { 2 } else { 0 };

        let metadata_bytes = if field.metadata.is_empty() {
            None
        } else {
            Some(encode_arrow_metadata(&field.metadata))
        };
        let metadata_ptr = metadata_bytes
            .as_ref()
            .map(|b| b.as_ptr() as *const c_char)
            .unwrap_or(ptr::null());

        let child_holder = Box::new(StructSchemaHolder {
            format_cstr,
            name_cstr,
            metadata_bytes,
        });

        let child = Box::new(ArrowSchema {
            format: child_holder.format_cstr.as_ptr(),
            name: child_holder.name_cstr.as_ptr(),
            metadata: metadata_ptr,
            flags,
            n_children: 0,
            children: ptr::null_mut(),
            dictionary: ptr::null_mut(),
            release: Some(release_struct_schema),
            private_data: Box::into_raw(child_holder) as *mut c_void,
        });

        child_schemas.push(Box::into_raw(child));
    }

    let children_box = child_schemas.into_boxed_slice();
    let children_ptr = Box::into_raw(children_box) as *mut *mut ArrowSchema;

    let format_cstr = CString::new("+s").unwrap();
    let name_cstr = CString::new("").unwrap();

    let metadata_bytes = holder.schema_metadata.clone();
    let metadata_ptr = metadata_bytes
        .as_ref()
        .map(|b| b.as_ptr() as *const c_char)
        .unwrap_or(ptr::null());

    let schema_holder = Box::new(StructSchemaHolder {
        format_cstr,
        name_cstr,
        metadata_bytes,
    });

    let struct_schema = ArrowSchema {
        format: schema_holder.format_cstr.as_ptr(),
        name: schema_holder.name_cstr.as_ptr(),
        metadata: metadata_ptr,
        flags: 0,
        n_children: n_fields as i64,
        children: children_ptr,
        dictionary: ptr::null_mut(),
        release: Some(release_struct_schema),
        private_data: Box::into_raw(schema_holder) as *mut c_void,
    };

    unsafe { ptr::write(out, struct_schema) };
    0
}

#[cfg(feature = "views")]
unsafe extern "C" fn rb_view_stream_get_next(
    stream: *mut ArrowArrayStream,
    out: *mut ArrowArray,
) -> i32 {
    let holder = unsafe { &mut *((*stream).private_data as *mut RecordBatchViewStreamHolder) };

    if holder.cursor >= holder.batches.len() {
        unsafe { ptr::write(out, ArrowArray::empty()) };
        return 0;
    }

    let batch = holder.batches[holder.cursor].clone();
    holder.cursor += 1;

    let (arr_ptr, sch_ptr) = export_table_view_to_c(&batch, None);

    unsafe {
        ptr::write(out, ptr::read(arr_ptr));
        std::alloc::dealloc(arr_ptr as *mut u8, std::alloc::Layout::for_value(&*arr_ptr));
    }

    unsafe {
        if let Some(release) = (*sch_ptr).release {
            release(sch_ptr);
        }
        let _ = Box::from_raw(sch_ptr);
    }

    0
}

#[cfg(feature = "views")]
unsafe extern "C" fn rb_view_stream_release(stream: *mut ArrowArrayStream) {
    if stream.is_null() || (unsafe { &*stream }).release.is_none() {
        return;
    }
    let _ = unsafe { Box::from_raw((*stream).private_data as *mut RecordBatchViewStreamHolder) };
    unsafe { ptr::write_bytes(stream, 0, 1) };
}

/// Exports a [`crate::SuperArrayV`] as an `ArrowArrayStream` capsule-ready box.
///
/// One Arrow array is yielded per underlying [`crate::ArrayV`] slice. Each
/// slice's window is conveyed through `ArrowArray.offset`/`length` rather
/// than materialised.
#[cfg(all(feature = "views", feature = "chunked"))]
pub fn export_super_array_view_stream(view: &crate::SuperArrayV) -> Box<ArrowArrayStream> {
    let field = (*view.field).clone();
    export_array_view_stream(view.slices.clone(), field)
}

/// Exports a [`crate::SuperTableV`] as an `ArrowArrayStream` capsule-ready box.
///
/// One struct array is yielded per underlying [`crate::TableV`] batch. Per-column
/// windowed `(offset, len)` is propagated zero-copy.
#[cfg(all(feature = "views", feature = "chunked"))]
pub fn export_super_table_view_stream(
    view: &crate::SuperTableV,
    metadata: Option<std::collections::BTreeMap<String, String>>,
) -> Box<ArrowArrayStream> {
    let fields: Vec<crate::Field> = if view.slices.is_empty() {
        Vec::new()
    } else {
        view.slices[0]
            .fields
            .iter()
            .map(|f| (**f).clone())
            .collect()
    };
    export_record_batch_view_stream(view.slices.clone(), fields, metadata)
}

/// Generic get_last_error implementation for stream holders.
///
/// Expects the holder's first two fields to be the type-specific data, followed
/// by `last_error: Option<CString>`. We use a trait to abstract over holder types.
trait HasLastError {
    fn last_error(&self) -> Option<&CString>;
}

impl HasLastError for RecordBatchStreamHolder {
    fn last_error(&self) -> Option<&CString> {
        self.last_error.as_ref()
    }
}

impl HasLastError for ArrayStreamHolder {
    fn last_error(&self) -> Option<&CString> {
        self.last_error.as_ref()
    }
}

unsafe extern "C" fn stream_get_last_error<T: HasLastError>(
    stream: *mut ArrowArrayStream,
) -> *const c_char {
    unsafe {
        if stream.is_null() || (*stream).private_data.is_null() {
            return ptr::null();
        }
        let holder = &*((*stream).private_data as *const T);
        match holder.last_error() {
            Some(err) => err.as_ptr(),
            None => ptr::null(),
        }
    }
}

// ── Stream import ───────────────────────────────────────────────────────

/// Consumes an ArrowArrayStream that yields struct arrays (record batches)
/// and returns a list of Tables.
///
/// Each yielded struct array is decomposed into individual column arrays.
/// Children are stolen from the parent struct and imported zero-copy via
/// `import_array_zero_copy`, which takes ownership of each child's ArrowArray.
///
/// # Safety
/// `stream` must be a valid, non-null pointer to an initialised ArrowArrayStream.
pub unsafe fn import_record_batch_stream(
    stream: *mut ArrowArrayStream,
) -> Vec<Vec<(Arc<Array>, crate::Field)>> {
    let (batches, _metadata) = unsafe { import_record_batch_stream_with_metadata(stream) };
    batches
}

/// Like [`import_record_batch_stream`], but also returns the schema-level
/// metadata as key-value pairs. Returns `None` for metadata when the schema
/// has no metadata attached.
///
/// Column data buffers are imported zero-copy by stealing each child ArrowArray
/// from the struct array and transferring ownership to the imported array via
/// `ForeignBuffer`. Null bitmasks and string offsets are copied (small metadata).
/// Dictionary/categorical columns use the copying import path due to the
/// structural difference between Arrow's contiguous dictionary encoding and
/// MinArrow's `Vec64<String>` dictionary storage.
pub unsafe fn import_record_batch_stream_with_metadata(
    stream: *mut ArrowArrayStream,
) -> (
    Vec<Vec<(Arc<Array>, crate::Field)>>,
    Option<std::collections::BTreeMap<String, String>>,
) {
    unsafe {
        // 1. Get schema
        let mut schema = ArrowSchema::empty();
        let get_schema = ((*stream).get_schema).expect("stream has no get_schema callback");
        let rc = get_schema(stream, &mut schema);
        assert_eq!(
            rc, 0,
            "ArrowArrayStream get_schema returned error code {}",
            rc
        );

        // Extract metadata before parsing children
        let metadata = decode_arrow_metadata(schema.metadata);

        // Parse child field info from the struct schema
        let n_fields = schema.n_children as usize;
        let child_schemas: Vec<&ArrowSchema> = if n_fields > 0 && !schema.children.is_null() {
            (0..n_fields).map(|i| &**schema.children.add(i)).collect()
        } else {
            Vec::new()
        };

        // 2. Consume batches
        let mut batches = Vec::new();
        let get_next = ((*stream).get_next).expect("stream has no get_next callback");

        loop {
            let mut arr = ArrowArray::empty();
            let rc = get_next(stream, &mut arr);
            assert_eq!(
                rc, 0,
                "ArrowArrayStream get_next returned error code {}",
                rc
            );

            // End of stream: release is None
            if arr.release.is_none() {
                break;
            }

            // Decompose struct array into columns via zero-copy child stealing.
            // Each child has its own release callback, so we move each child's
            // ArrowArray out of the parent and replace it with an empty sentinel.
            // The parent's release then skips the empty children and just frees
            // its own allocations.
            let n_children = arr.n_children as usize;
            assert_eq!(
                n_children, n_fields,
                "Struct array child count ({}) does not match schema ({})",
                n_children, n_fields
            );

            let mut columns = Vec::with_capacity(n_children);
            for i in 0..n_children {
                let child_sch = child_schemas[i];

                // Extract complete field including metadata from schema
                let mut field = field_from_c_schema(child_sch);
                let dtype = field.dtype.clone();

                // Steal the child: move its ArrowArray out and replace with empty
                let child_raw: *mut ArrowArray = *arr.children.add(i);
                let child_content = ptr::read(child_raw);
                ptr::write(child_raw, ArrowArray::empty());

                // Import zero-copy with owned ArrowArray
                let child_box = Box::new(child_content);
                let imported =
                    import_array_zero_copy(child_box, dtype, child_sch as *const ArrowSchema);

                // Utf8View is stored internally as String since the data is
                // restructured during import from views to offsets+data.
                if field.dtype == ArrowType::Utf8View {
                    field.dtype = ArrowType::String;
                }
                columns.push((imported, field));
            }

            // Release the parent struct array. Children are now empty (release=None),
            // so release_struct_array skips their release callbacks but still frees
            // their Box allocations and the children pointer array.
            if let Some(release) = arr.release {
                release(&mut arr as *mut ArrowArray);
            }

            batches.push(columns);
        }

        // 3. Release the struct schema
        if let Some(release) = schema.release {
            release(&mut schema as *mut ArrowSchema);
        }

        // 4. Release the stream
        if let Some(release) = (*stream).release {
            release(stream);
        }

        (batches, metadata)
    }
}

/// Consumes an ArrowArrayStream that yields plain arrays and returns the
/// imported arrays along with the field metadata.
///
/// Uses the zero-copy import path since each yielded array is independently
/// owned. The stream schema is kept alive for the duration of the import so
/// that dictionary-encoded arrays can reference it directly, avoiding the need
/// to reconstruct synthetic schemas per chunk.
///
/// # Safety
/// `stream` must be a valid, non-null pointer to an initialised ArrowArrayStream.
pub unsafe fn import_array_stream(
    stream: *mut ArrowArrayStream,
) -> (Vec<Arc<Array>>, crate::Field) {
    unsafe {
        // 1. Get schema
        let mut schema_c = ArrowSchema::empty();
        let get_schema = ((*stream).get_schema).expect("stream has no get_schema callback");
        let rc = get_schema(stream, &mut schema_c);
        assert_eq!(
            rc, 0,
            "ArrowArrayStream get_schema returned error code {}",
            rc
        );

        // Extract complete field including metadata
        let mut field = field_from_c_schema(&schema_c);

        // 2. Consume arrays using the original stream schema directly.
        // This is essential for dictionary-encoded arrays where import_categorical
        // needs schema_c.dictionary to describe the dictionary value type.
        let mut arrays = Vec::new();
        let get_next = ((*stream).get_next).expect("stream has no get_next callback");

        loop {
            let mut arr = ArrowArray::empty();
            let rc = get_next(stream, &mut arr);
            assert_eq!(
                rc, 0,
                "ArrowArrayStream get_next returned error code {}",
                rc
            );

            if arr.release.is_none() {
                break;
            }

            let arr_box = Box::new(arr);
            let imported = import_array_zero_copy(
                arr_box,
                field.dtype.clone(),
                &schema_c as *const ArrowSchema,
            );
            arrays.push(imported);
        }

        // Utf8View is stored internally as String since the data is
        // restructured during import from views to offsets+data.
        if field.dtype == ArrowType::Utf8View {
            field.dtype = ArrowType::String;
        }

        // 3. Release the schema now that all chunks have been imported
        if let Some(release) = schema_c.release {
            release(&mut schema_c as *mut ArrowSchema);
        }

        // 4. Release the stream
        if let Some(release) = (*stream).release {
            release(stream);
        }

        (arrays, field)
    }
}

/// Extracts a complete Field from an ArrowSchema, including metadata.
///
/// When the schema has a non-null `dictionary` pointer, the field is recognised
/// as dictionary-encoded and the returned ArrowType is `Dictionary(...)` rather
/// than the raw index type.
///
/// # Safety
/// `schema` must point to a valid ArrowSchema with valid name and format pointers.
unsafe fn field_from_c_schema(schema: &ArrowSchema) -> crate::Field {
    let name = if schema.name.is_null() {
        String::new()
    } else {
        unsafe { std::ffi::CStr::from_ptr(schema.name) }
            .to_string_lossy()
            .into_owned()
    };
    let nullable = (schema.flags & 2) != 0;
    let fmt = unsafe { std::ffi::CStr::from_ptr(schema.format).to_bytes() };

    let dtype = if !schema.dictionary.is_null() {
        // Dictionary-encoded: format string describes the index type, the
        // dictionary field describes the value type.
        use crate::ffi::arrow_dtype::CategoricalIndexType;
        let index_type = match fmt {
            #[cfg(feature = "default_categorical_8")]
            b"c" | b"C" => CategoricalIndexType::UInt8,
            #[cfg(all(feature = "extended_numeric_types", feature = "extended_categorical"))]
            b"s" | b"S" => CategoricalIndexType::UInt16,
            #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
            b"i" | b"I" => CategoricalIndexType::UInt32,
            // When default_categorical_8 is on without extended_categorical,
            // record as UInt8 for schema purposes - actual narrowing happens at import time
            #[cfg(all(feature = "default_categorical_8", not(feature = "extended_categorical")))]
            b"i" | b"I" => CategoricalIndexType::UInt8,
            #[cfg(all(feature = "extended_numeric_types", feature = "extended_categorical"))]
            b"l" | b"L" => CategoricalIndexType::UInt64,
            _ => panic!(
                "Unsupported dictionary index format: {:?}",
                std::str::from_utf8(fmt).unwrap_or("??")
            ),
        };
        ArrowType::Dictionary(index_type)
    } else {
        parse_arrow_format(fmt)
    };

    let metadata = unsafe { decode_arrow_metadata(schema.metadata) };
    crate::Field::new(name, dtype, nullable, metadata)
}

/// Parses an Arrow C format string into an ArrowType.
/// Shared between import_from_c, import_from_c_owned, and stream import.
fn parse_arrow_format(fmt: &[u8]) -> ArrowType {
    match fmt {
        b"n" => ArrowType::Null,
        b"b" => ArrowType::Boolean,
        #[cfg(feature = "extended_numeric_types")]
        b"c" => ArrowType::Int8,
        #[cfg(feature = "extended_numeric_types")]
        b"C" => ArrowType::UInt8,
        #[cfg(feature = "extended_numeric_types")]
        b"s" => ArrowType::Int16,
        #[cfg(feature = "extended_numeric_types")]
        b"S" => ArrowType::UInt16,
        b"i" => ArrowType::Int32,
        b"I" => ArrowType::UInt32,
        b"l" => ArrowType::Int64,
        b"L" => ArrowType::UInt64,
        b"f" => ArrowType::Float32,
        b"g" => ArrowType::Float64,
        b"u" => ArrowType::String,
        #[cfg(feature = "large_string")]
        b"U" => ArrowType::LargeString,
        b"vu" => ArrowType::Utf8View,
        #[cfg(feature = "datetime")]
        b"tdD" => ArrowType::Date32,
        #[cfg(feature = "datetime")]
        b"tdm" => ArrowType::Date64,
        #[cfg(feature = "datetime")]
        b"tts" => ArrowType::Time32(crate::TimeUnit::Seconds),
        #[cfg(feature = "datetime")]
        b"ttm" => ArrowType::Time32(crate::TimeUnit::Milliseconds),
        #[cfg(feature = "datetime")]
        b"ttu" => ArrowType::Time64(crate::TimeUnit::Microseconds),
        #[cfg(feature = "datetime")]
        b"ttn" => ArrowType::Time64(crate::TimeUnit::Nanoseconds),
        #[cfg(feature = "datetime")]
        b"tDs" => ArrowType::Duration32(crate::TimeUnit::Seconds),
        #[cfg(feature = "datetime")]
        b"tDm" => ArrowType::Duration32(crate::TimeUnit::Milliseconds),
        #[cfg(feature = "datetime")]
        b"tDu" => ArrowType::Duration64(crate::TimeUnit::Microseconds),
        #[cfg(feature = "datetime")]
        b"tDn" => ArrowType::Duration64(crate::TimeUnit::Nanoseconds),
        #[cfg(feature = "datetime")]
        b"tiM" => ArrowType::Interval(crate::IntervalUnit::YearMonth),
        #[cfg(feature = "datetime")]
        b"tiD" => ArrowType::Interval(crate::IntervalUnit::DaysTime),
        #[cfg(feature = "datetime")]
        b"tin" => ArrowType::Interval(crate::IntervalUnit::MonthDaysNs),
        b"+s" => ArrowType::Null, // struct marker (handled at stream level)
        #[cfg(feature = "datetime")]
        _ if fmt.starts_with(b"tss")
            || fmt.starts_with(b"tsm")
            || fmt.starts_with(b"tsu")
            || fmt.starts_with(b"tsn") =>
        {
            let unit = match &fmt[..3] {
                b"tss" => crate::TimeUnit::Seconds,
                b"tsm" => crate::TimeUnit::Milliseconds,
                b"tsu" => crate::TimeUnit::Microseconds,
                b"tsn" => crate::TimeUnit::Nanoseconds,
                _ => unreachable!(),
            };
            let tz = if fmt.len() > 4 {
                let tz_bytes = &fmt[4..];
                let tz_str = String::from_utf8_lossy(tz_bytes).into_owned();
                if tz_str.is_empty() {
                    None
                } else {
                    Some(tz_str)
                }
            } else {
                None
            };
            ArrowType::Timestamp(unit, tz)
        }
        o => panic!(
            "unsupported Arrow format {:?}",
            std::str::from_utf8(o).unwrap_or("??")
        ),
    }
}

// Arrow C Data Interface basic tests
// E2E Tests with 'C' are under `tests/`.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    #[cfg(feature = "datetime")]
    use crate::DatetimeArray;
    use crate::ffi::arrow_c_ffi::export_to_c;
    #[cfg(feature = "datetime")]
    use crate::ffi::arrow_c_ffi::import_from_c;
    use crate::ffi::arrow_dtype::ArrowType;
    use crate::ffi::schema::Schema;
    use crate::{Array, BooleanArray, Field, FloatArray, IntegerArray, MaskedArray, StringArray};

    // Helper for constructing a one-field schema for the given type
    fn schema_for(name: &str, ty: ArrowType, nullable: bool) -> Schema {
        Schema {
            fields: vec![Field::new(name, ty, nullable, None)],
            metadata: Default::default(),
        }
    }

    #[test]
    fn test_arrow_c_export_int32() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(1);
        arr.push(2);
        arr.push(3);

        let array = Arc::new(Array::from_int32(arr));
        let schema = schema_for("ints", ArrowType::Int32, false);

        let (arr_ptr, sch_ptr) = export_to_c(array, schema);

        unsafe {
            assert_eq!((*arr_ptr).length, 3);
            let bufs = std::slice::from_raw_parts((*arr_ptr).buffers, 2);
            let vals = std::slice::from_raw_parts(bufs[1] as *const i32, 3);
            assert_eq!(vals, &[1, 2, 3]);
            // Idempotent release
            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[test]
    fn test_arrow_c_export_int64() {
        let mut arr = IntegerArray::<i64>::default();
        arr.push(-42);
        arr.push(99);
        arr.push(1001);

        let array = Arc::new(Array::from_int64(arr));
        let schema = schema_for("big", ArrowType::Int64, false);

        let (arr_ptr, sch_ptr) = export_to_c(array, schema);

        unsafe {
            assert_eq!((*arr_ptr).length, 3);
            let bufs = std::slice::from_raw_parts((*arr_ptr).buffers, 2);
            let vals = std::slice::from_raw_parts(bufs[1] as *const i64, 3);
            assert_eq!(vals, &[-42, 99, 1001]);
            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[test]
    fn test_arrow_c_export_u32() {
        let mut arr = IntegerArray::<u32>::default();
        arr.push(100);
        arr.push(200);
        arr.push(300);

        let array = Arc::new(Array::from_uint32(arr));
        let schema = schema_for("uints", ArrowType::UInt32, false);

        let (arr_ptr, sch_ptr) = export_to_c(array, schema);

        unsafe {
            assert_eq!((*arr_ptr).length, 3);
            let bufs = std::slice::from_raw_parts((*arr_ptr).buffers, 2);
            let vals = std::slice::from_raw_parts(bufs[1] as *const u32, 3);
            assert_eq!(vals, &[100, 200, 300]);
            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[test]
    fn test_arrow_c_export_f32() {
        let mut arr = FloatArray::<f32>::default();
        arr.push(1.5);
        arr.push(-2.0);
        arr.push(3.25);

        let array = Arc::new(Array::from_float32(arr));
        let schema = schema_for("floats", ArrowType::Float32, false);

        let (arr_ptr, sch_ptr) = export_to_c(array, schema);

        unsafe {
            assert_eq!((*arr_ptr).length, 3);
            let bufs = std::slice::from_raw_parts((*arr_ptr).buffers, 2);
            let vals = std::slice::from_raw_parts(bufs[1] as *const f32, 3);
            assert_eq!(vals, &[1.5, -2.0, 3.25]);
            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[test]
    fn test_arrow_c_export_f64() {
        let mut arr = FloatArray::<f64>::default();
        arr.push(0.1);
        arr.push(0.2);
        arr.push(0.3);

        let array = Arc::new(Array::from_float64(arr));
        let schema = schema_for("doubles", ArrowType::Float64, false);

        let (arr_ptr, sch_ptr) = export_to_c(array, schema);

        unsafe {
            assert_eq!((*arr_ptr).length, 3);
            let bufs = std::slice::from_raw_parts((*arr_ptr).buffers, 2);
            let vals = std::slice::from_raw_parts(bufs[1] as *const f64, 3);
            assert_eq!(vals, &[0.1, 0.2, 0.3]);
            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[test]
    fn test_arrow_c_export_bool() {
        let mut arr = BooleanArray::default();
        arr.push(true);
        arr.push(false);
        arr.push(true);

        let array = Arc::new(Array::BooleanArray(arr.into()));
        let schema = schema_for("b", ArrowType::Boolean, false);

        let (arr_ptr, sch_ptr) = export_to_c(array, schema);

        unsafe {
            assert_eq!((*arr_ptr).length, 3);
            let bufs = std::slice::from_raw_parts((*arr_ptr).buffers, 2);
            assert!(!bufs[1].is_null());
            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[test]
    fn test_arrow_c_export_str() {
        let mut utf = StringArray::default();
        utf.push_str("foo");
        utf.push_str("bar");

        let array = Arc::new(Array::from_string32(utf));
        let schema = schema_for("txt", ArrowType::String, false);

        let (arr_ptr, sch_ptr) = export_to_c(array, schema);

        unsafe {
            assert_eq!((*arr_ptr).length, 2);
            let bufs = std::slice::from_raw_parts((*arr_ptr).buffers, 3);
            // Arrow buffer order for UTF8: [nulls, offsets, values]
            assert!(!bufs[1].is_null(), "offsets buffer must be non-null");
            assert!(!bufs[2].is_null(), "values buffer must be non-null");
            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[test]
    fn test_arrow_c_export_str_offsets() {
        let mut utf = StringArray::default();
        utf.push_str("foo");
        utf.push_str("bar");
        utf.push_str("baz");

        let array = Arc::new(Array::from_string32(utf));
        let schema = schema_for("txt", ArrowType::String, false);

        let (arr_ptr, sch_ptr) = export_to_c(array, schema);

        unsafe {
            assert_eq!((*arr_ptr).length, 3);
            let bufs = std::slice::from_raw_parts((*arr_ptr).buffers, 3);
            // Offsets are buffer[1], values are buffer[2]
            let offsets = std::slice::from_raw_parts(bufs[1] as *const u32, 4);
            assert_eq!(
                offsets,
                &[0, 3, 6, 9],
                "UTF8 offsets must be monotonically increasing starting at 0"
            );
            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[test]
    fn test_arrow_c_export_with_null_mask() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(42);
        arr.push_null();
        arr.push(88);

        let array = Arc::new(Array::from_int32(arr));
        let schema = schema_for("ints", ArrowType::Int32, true);

        let (arr_ptr, sch_ptr) = export_to_c(array, schema);

        unsafe {
            assert_eq!((*arr_ptr).length, 3);
            let bufs = std::slice::from_raw_parts((*arr_ptr).buffers, 2);
            // null bitmap is buffer[0]
            assert!(!bufs[0].is_null());
            let bitmap = std::slice::from_raw_parts(bufs[0], 1);
            // Should be 0b00000101
            assert_eq!(bitmap[0] & 0b111, 0b101);
            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn test_arrow_c_export_datetime() {
        use crate::TimeUnit;
        let mut dt = DatetimeArray::<i64>::default();
        dt.push(1);
        dt.push(2);
        dt.time_unit = TimeUnit::Milliseconds;

        let array = Arc::new(Array::from_datetime_i64(dt));
        let schema = schema_for("dt", ArrowType::Date64, false);

        let (arr_ptr, sch_ptr) = export_to_c(array, schema);

        unsafe {
            assert_eq!((*arr_ptr).length, 2);
            let bufs = std::slice::from_raw_parts((*arr_ptr).buffers, 2);
            assert!(!bufs[1].is_null());
            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn test_arrow_c_timezone_roundtrip_utc() {
        use crate::TimeUnit;
        let mut dt = DatetimeArray::<i64>::default();
        dt.push(1609459200);
        dt.push(1640995200);
        dt.time_unit = TimeUnit::Seconds;

        let array = Arc::new(Array::from_datetime_i64(dt));
        let schema = schema_for(
            "ts",
            ArrowType::Timestamp(TimeUnit::Seconds, Some("UTC".to_string())),
            false,
        );

        let (arr_ptr, sch_ptr) = export_to_c(array.clone(), schema);

        unsafe {
            let fmt = std::ffi::CStr::from_ptr((*sch_ptr).format).to_bytes();
            assert_eq!(fmt, b"tss:UTC", "Format string should include timezone");

            let imported = import_from_c(arr_ptr as *const _, sch_ptr as *const _);

            if let Array::TemporalArray(crate::TemporalArray::Datetime64(imported_dt)) =
                imported.as_ref()
            {
                assert_eq!(imported_dt.len(), 2);
                assert_eq!(imported_dt.time_unit, TimeUnit::Seconds);
            } else {
                panic!("Expected Datetime64 array");
            }

            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn test_arrow_c_timezone_roundtrip_iana() {
        use crate::TimeUnit;
        let mut dt = DatetimeArray::<i64>::default();
        dt.push(1609459200000);
        dt.push(1640995200000);
        dt.time_unit = TimeUnit::Milliseconds;

        let array = Arc::new(Array::from_datetime_i64(dt));
        let tz = "America/New_York".to_string();
        let schema = schema_for(
            "ts",
            ArrowType::Timestamp(TimeUnit::Milliseconds, Some(tz.clone())),
            false,
        );

        let (arr_ptr, sch_ptr) = export_to_c(array.clone(), schema);

        unsafe {
            let fmt = std::ffi::CStr::from_ptr((*sch_ptr).format).to_bytes();
            assert_eq!(
                fmt, b"tsm:America/New_York",
                "Format string should include IANA timezone"
            );

            let imported = import_from_c(arr_ptr as *const _, sch_ptr as *const _);

            if let Array::TemporalArray(crate::TemporalArray::Datetime64(imported_dt)) =
                imported.as_ref()
            {
                assert_eq!(imported_dt.len(), 2);
                assert_eq!(imported_dt.time_unit, TimeUnit::Milliseconds);
            } else {
                panic!("Expected Datetime64 array");
            }

            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn test_arrow_c_timezone_roundtrip_offset() {
        use crate::TimeUnit;
        let mut dt = DatetimeArray::<i64>::default();
        dt.push(1609459200000000);
        dt.time_unit = TimeUnit::Microseconds;

        let array = Arc::new(Array::from_datetime_i64(dt));
        let tz = "+05:30".to_string();
        let schema = schema_for(
            "ts",
            ArrowType::Timestamp(TimeUnit::Microseconds, Some(tz.clone())),
            false,
        );

        let (arr_ptr, sch_ptr) = export_to_c(array.clone(), schema);

        unsafe {
            let fmt = std::ffi::CStr::from_ptr((*sch_ptr).format).to_bytes();
            assert_eq!(
                fmt, b"tsu:+05:30",
                "Format string should include offset timezone"
            );

            let imported = import_from_c(arr_ptr as *const _, sch_ptr as *const _);

            if let Array::TemporalArray(crate::TemporalArray::Datetime64(imported_dt)) =
                imported.as_ref()
            {
                assert_eq!(imported_dt.len(), 1);
                assert_eq!(imported_dt.time_unit, TimeUnit::Microseconds);
            } else {
                panic!("Expected Datetime64 array");
            }

            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn test_arrow_c_timezone_none() {
        use crate::TimeUnit;
        let mut dt = DatetimeArray::<i64>::default();
        dt.push(1609459200);
        dt.time_unit = TimeUnit::Seconds;

        let array = Arc::new(Array::from_datetime_i64(dt));
        let schema = schema_for("ts", ArrowType::Timestamp(TimeUnit::Seconds, None), false);

        let (arr_ptr, sch_ptr) = export_to_c(array.clone(), schema);

        unsafe {
            let fmt = std::ffi::CStr::from_ptr((*sch_ptr).format).to_bytes();
            assert_eq!(
                fmt, b"tss:",
                "Format string should have colon but no timezone"
            );

            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[test]
    fn test_field_metadata_round_trip_via_export_import() {
        use super::import_from_c_owned;
        use std::collections::BTreeMap;

        let mut meta = BTreeMap::new();
        meta.insert("source".to_string(), "test_db".to_string());
        meta.insert("units".to_string(), "kg".to_string());

        let mut arr = IntegerArray::<i32>::default();
        arr.push(1);
        arr.push(2);

        let array = Arc::new(Array::from_int32(arr));
        let schema = Schema {
            fields: vec![Field::new(
                "col",
                ArrowType::Int32,
                false,
                Some(meta.clone()),
            )],
            metadata: Default::default(),
        };

        let (arr_ptr, sch_ptr) = export_to_c(array, schema);

        unsafe {
            let arr_box = Box::from_raw(arr_ptr);
            let sch_box = Box::from_raw(sch_ptr);
            let (_, field) = import_from_c_owned(arr_box, sch_box);
            assert_eq!(
                field.metadata, meta,
                "Field metadata should survive round-trip"
            );
        }
    }

    #[test]
    fn test_field_metadata_round_trip_record_batch_stream() {
        use super::{
            export_record_batch_stream_with_metadata, import_record_batch_stream_with_metadata,
        };
        use std::collections::BTreeMap;

        let mut field_meta = BTreeMap::new();
        field_meta.insert("origin".to_string(), "sensor_1".to_string());

        let field = Field::new("vals", ArrowType::Int32, false, Some(field_meta.clone()));

        let mut arr = IntegerArray::<i32>::default();
        arr.push(10);
        arr.push(20);
        let array = Arc::new(Array::from_int32(arr));
        let col_schema = Schema {
            fields: vec![field.clone()],
            metadata: Default::default(),
        };

        let batches = vec![vec![(array, col_schema)]];
        let fields = vec![field];

        let mut table_meta = BTreeMap::new();
        table_meta.insert("table_name".to_string(), "test_tbl".to_string());

        let stream =
            export_record_batch_stream_with_metadata(batches, fields, Some(table_meta.clone()));
        let stream_ptr = Box::into_raw(stream);

        unsafe {
            let (columns_batches, schema_meta) =
                import_record_batch_stream_with_metadata(stream_ptr);

            assert_eq!(
                schema_meta,
                Some(table_meta),
                "Schema metadata should survive round-trip"
            );
            assert_eq!(columns_batches.len(), 1);
            let batch = &columns_batches[0];
            assert_eq!(batch.len(), 1);
            let (_, imported_field) = &batch[0];
            assert_eq!(
                imported_field.metadata, field_meta,
                "Field metadata should survive record batch stream round-trip"
            );
        }
    }

    #[test]
    #[cfg(feature = "table_metadata")]
    fn test_table_metadata_round_trip() {
        use super::{
            export_record_batch_stream_with_metadata, import_record_batch_stream_with_metadata,
        };
        use std::collections::BTreeMap;

        // Build a simple column
        let mut arr = IntegerArray::<i32>::default();
        arr.push(1);
        arr.push(2);
        arr.push(3);
        let array = Arc::new(Array::from_int32(arr));
        let field = Field::new("col1", ArrowType::Int32, false, None);
        let col_schema = Schema {
            fields: vec![field.clone()],
            metadata: Default::default(),
        };

        // Schema-level metadata simulating pandas categorical info
        let mut table_meta = BTreeMap::new();
        table_meta.insert(
            "pandas".to_string(),
            r#"{"columns":[{"name":"col1","pandas_type":"categorical","metadata":{"ordered":true}}]}"#.to_string(),
        );

        // Export as stream with metadata
        let stream = export_record_batch_stream_with_metadata(
            vec![vec![(array, col_schema)]],
            vec![field],
            Some(table_meta.clone()),
        );
        let stream_ptr = Box::into_raw(stream);

        // Import with metadata
        let (batches, schema_meta) =
            unsafe { import_record_batch_stream_with_metadata(stream_ptr) };

        // Verify metadata survived the roundtrip
        assert_eq!(schema_meta, Some(table_meta.clone()));

        // Verify we can construct a Table with the imported metadata
        let imported_cols: Vec<_> = batches[0]
            .iter()
            .map(|(arr, f)| crate::FieldArray::from_arr(f.name.as_str(), (**arr).clone()))
            .collect();
        let table = crate::Table::new_with_metadata(
            "test".to_string(),
            Some(imported_cols),
            table_meta.clone(),
        );
        assert_eq!(table.metadata(), &table_meta);
        assert_eq!(table.n_rows(), 3);
    }

    // ─── View export round-trip tests ──────────────────────────────────

    #[cfg(feature = "views")]
    #[test]
    fn test_view_export_int32_window() {
        use super::{export_view_to_c, import_from_c};
        use crate::ArrayV;

        let mut arr = IntegerArray::<i32>::default();
        for v in [10, 20, 30, 40, 50, 60, 70, 80] {
            arr.push(v);
        }
        let array = Arc::new(Array::from_int32(arr));
        let view = ArrayV::new((*array).clone(), 2, 4); // 30,40,50,60

        let schema = schema_for("x", ArrowType::Int32, false);
        let (arr_ptr, sch_ptr) =
            export_view_to_c(array.clone(), schema, view.offset as i64, view.len() as i64);

        unsafe {
            assert_eq!((*arr_ptr).length, 4);
            assert_eq!((*arr_ptr).offset, 2);

            // Importer applies offset; result must equal the windowed values.
            let imported = import_from_c(arr_ptr as *const _, sch_ptr as *const _);
            let inner = imported.num().i32();
            assert_eq!(inner.data.as_slice(), &[30, 40, 50, 60]);

            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[cfg(feature = "views")]
    #[test]
    fn test_view_export_string_window() {
        use super::{export_view_to_c, import_from_c};
        use crate::ArrayV;

        let mut utf = StringArray::<u32>::default();
        utf.push_str("alpha");
        utf.push_str("beta");
        utf.push_str("gamma");
        utf.push_str("delta");

        let array = Arc::new(Array::from_string32(utf));
        let view = ArrayV::new((*array).clone(), 1, 2); // beta, gamma

        let schema = schema_for("s", ArrowType::String, false);
        let (arr_ptr, sch_ptr) =
            export_view_to_c(array.clone(), schema, view.offset as i64, view.len() as i64);

        unsafe {
            assert_eq!((*arr_ptr).length, 2);
            assert_eq!((*arr_ptr).offset, 1);

            let imported = import_from_c(arr_ptr as *const _, sch_ptr as *const _);
            if let Array::TextArray(crate::TextArray::String32(s)) = imported.as_ref() {
                assert_eq!(s.len(), 2);
                assert_eq!(s.get_str(0), Some("beta"));
                assert_eq!(s.get_str(1), Some("gamma"));
            } else {
                panic!("Expected String32 array");
            }

            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[cfg(feature = "views")]
    #[test]
    fn test_view_export_full_array_matches_owned() {
        use super::{export_to_c, export_view_to_c};

        let mut arr = IntegerArray::<i32>::default();
        for v in 0..5 {
            arr.push(v);
        }
        let array = Arc::new(Array::from_int32(arr));

        let schema_a = schema_for("x", ArrowType::Int32, false);
        let schema_b = schema_for("x", ArrowType::Int32, false);

        let (owned_ptr, owned_sch) = export_to_c(array.clone(), schema_a);
        let (view_ptr, view_sch) = export_view_to_c(array.clone(), schema_b, 0, 5);

        unsafe {
            assert_eq!((*owned_ptr).length, (*view_ptr).length);
            assert_eq!((*owned_ptr).offset, (*view_ptr).offset);
            ((*owned_ptr).release.unwrap())(owned_ptr);
            ((*owned_sch).release.unwrap())(owned_sch);
            ((*view_ptr).release.unwrap())(view_ptr);
            ((*view_sch).release.unwrap())(view_sch);
        }
    }

    #[cfg(feature = "views")]
    #[test]
    fn test_table_view_export_round_trip() {
        use super::{export_table_view_to_c, import_record_batch_stream_with_metadata,
                    export_record_batch_view_stream};
        use crate::{Table, TableV};

        let mut a = IntegerArray::<i32>::default();
        for v in 0..6 {
            a.push(v);
        }
        let mut b = FloatArray::<f64>::default();
        for v in 0..6 {
            b.push(v as f64 * 0.5);
        }

        let fa_a = crate::FieldArray::new(
            Field::new("a", ArrowType::Int32, false, None),
            Array::from_int32(a),
        );
        let fa_b = crate::FieldArray::new(
            Field::new("b", ArrowType::Float64, false, None),
            Array::from_float64(b),
        );
        let table = Table::new("t".to_string(), Some(vec![fa_a, fa_b]));

        // Window rows 2..5
        let view = TableV::from_table(table, 2, 3);

        // Export through stream and re-import
        let fields: Vec<Field> = view.fields.iter().map(|f| (**f).clone()).collect();
        let stream = export_record_batch_view_stream(vec![view.clone()], fields, None);
        let stream_ptr = Box::into_raw(stream);
        let (batches, _meta) =
            unsafe { import_record_batch_stream_with_metadata(stream_ptr) };

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 2);

        let (col_a, _) = &batches[0][0];
        let (col_b, _) = &batches[0][1];

        let inner_a = col_a.num().i32();
        let inner_b = col_b.num().f64();
        assert_eq!(inner_a.data.as_slice(), &[2, 3, 4]);
        assert_eq!(inner_b.data.as_slice(), &[1.0, 1.5, 2.0]);

        // Direct struct export path
        let (arr_ptr, sch_ptr) = export_table_view_to_c(&view, None);
        unsafe {
            assert_eq!((*arr_ptr).length, 3);
            assert_eq!((*arr_ptr).n_children, 2);
            // Each child carries its own offset
            let child0 = &**((*arr_ptr).children.add(0));
            assert_eq!(child0.offset, 2);
            assert_eq!(child0.length, 3);
            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[cfg(all(feature = "views", feature = "chunked"))]
    #[test]
    fn test_super_array_view_stream_round_trip() {
        use super::{export_super_array_view_stream, import_array_stream};
        use crate::{ArrayV, SuperArrayV};

        // Two backing arrays, each contributing a window
        let mut a1 = IntegerArray::<i32>::default();
        for v in [1, 2, 3, 4, 5] {
            a1.push(v);
        }
        let mut a2 = IntegerArray::<i32>::default();
        for v in [10, 20, 30, 40, 50] {
            a2.push(v);
        }
        let arr1 = Array::from_int32(a1);
        let arr2 = Array::from_int32(a2);

        // Window: take 3..5 of arr1 (4,5), then 0..2 of arr2 (10,20)
        let v1 = ArrayV::new(arr1, 3, 2);
        let v2 = ArrayV::new(arr2, 0, 2);
        let field = Arc::new(Field::new("x", ArrowType::Int32, false, None));
        let super_view = SuperArrayV {
            slices: vec![v1, v2],
            len: 4,
            field: field.clone(),
        };

        let stream = export_super_array_view_stream(&super_view);
        let stream_ptr = Box::into_raw(stream);

        let (arrays, _field) = unsafe { import_array_stream(stream_ptr) };
        assert_eq!(arrays.len(), 2);

        let inner0 = arrays[0].num().i32();
        let inner1 = arrays[1].num().i32();
        assert_eq!(inner0.data.as_slice(), &[4, 5]);
        assert_eq!(inner1.data.as_slice(), &[10, 20]);
    }

    #[cfg(all(feature = "views", feature = "chunked"))]
    #[test]
    fn test_super_table_view_stream_round_trip() {
        use super::{export_super_table_view_stream, import_record_batch_stream_with_metadata};
        use crate::{SuperTableV, Table, TableV};

        // Two backing tables. Each TableV picks a different row window.
        // Layout: schema = (a: i32, b: f64)
        let mut a1 = IntegerArray::<i32>::default();
        for v in 0..6 {
            a1.push(v);
        }
        let mut b1 = FloatArray::<f64>::default();
        for v in 0..6 {
            b1.push(v as f64);
        }
        let t1 = Table::new(
            "t1".to_string(),
            Some(vec![
                crate::FieldArray::new(
                    Field::new("a", ArrowType::Int32, false, None),
                    Array::from_int32(a1),
                ),
                crate::FieldArray::new(
                    Field::new("b", ArrowType::Float64, false, None),
                    Array::from_float64(b1),
                ),
            ]),
        );

        let mut a2 = IntegerArray::<i32>::default();
        for v in 100..108 {
            a2.push(v);
        }
        let mut b2 = FloatArray::<f64>::default();
        for v in 100..108 {
            b2.push(v as f64);
        }
        let t2 = Table::new(
            "t2".to_string(),
            Some(vec![
                crate::FieldArray::new(
                    Field::new("a", ArrowType::Int32, false, None),
                    Array::from_int32(a2),
                ),
                crate::FieldArray::new(
                    Field::new("b", ArrowType::Float64, false, None),
                    Array::from_float64(b2),
                ),
            ]),
        );

        // Windows: rows 1..4 of t1 (3 rows), rows 5..7 of t2 (2 rows)
        let v1 = TableV::from_table(t1, 1, 3);
        let v2 = TableV::from_table(t2, 5, 2);
        let super_view = SuperTableV {
            slices: vec![v1, v2],
            len: 5,
        };

        let stream = export_super_table_view_stream(&super_view, None);
        let stream_ptr = Box::into_raw(stream);

        let (batches, _meta) =
            unsafe { import_record_batch_stream_with_metadata(stream_ptr) };
        assert_eq!(batches.len(), 2);

        // Batch 0: a=[1,2,3], b=[1.0,2.0,3.0]
        let (col_a0, _) = &batches[0][0];
        let (col_b0, _) = &batches[0][1];
        assert_eq!(
            col_a0.num().i32().data.as_slice(),
            &[1, 2, 3]
        );
        assert_eq!(
            col_b0.num().f64().data.as_slice(),
            &[1.0, 2.0, 3.0]
        );

        // Batch 1: a=[105,106], b=[105.0,106.0]
        let (col_a1, _) = &batches[1][0];
        let (col_b1, _) = &batches[1][1];
        assert_eq!(
            col_a1.num().i32().data.as_slice(),
            &[105, 106]
        );
        assert_eq!(
            col_b1.num().f64().data.as_slice(),
            &[105.0, 106.0]
        );
    }

    /// Exercises `import_null_mask_offset`: a windowed view that spans
    /// across byte boundaries in the validity bitmap (offset=3, len=7),
    /// with nulls at non-aligned positions.
    #[cfg(feature = "views")]
    #[test]
    fn test_view_export_with_nulls_offset() {
        use super::{export_view_to_c, import_from_c};
        use crate::ArrayV;

        // 10 elements. Nulls at indices 1, 4, 7 in the parent array.
        let mut arr = IntegerArray::<i32>::default();
        for v in 0..10 {
            arr.push(v);
        }
        arr.set_null(1);
        arr.set_null(4);
        arr.set_null(7);

        let array = Arc::new(Array::from_int32(arr));
        // Window indices 3..10. Parent nulls at 4 and 7 fall inside it,
        // which become positions 1 and 4 of the view.
        let view = ArrayV::new((*array).clone(), 3, 7);

        let schema = schema_for("x", ArrowType::Int32, true);
        let (arr_ptr, sch_ptr) =
            export_view_to_c(array.clone(), schema, view.offset as i64, view.len() as i64);

        unsafe {
            let imported = import_from_c(arr_ptr as *const _, sch_ptr as *const _);
            let inner = imported.num().i32();
            assert_eq!(inner.data.as_slice(), &[3, 4, 5, 6, 7, 8, 9]);

            let mask = inner.null_mask.clone().expect("null mask must survive offset");
            // Indices in the imported array: 1 (was parent index 4) and
            // 4 (was parent index 7) should read as null; everything else valid.
            for i in 0..7 {
                let is_set = mask.get(i);
                let expected_valid = !(i == 1 || i == 4);
                assert_eq!(
                    is_set, expected_valid,
                    "mask bit {i} expected valid={expected_valid} got set={is_set}"
                );
            }

            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[cfg(feature = "views")]
    #[test]
    fn test_view_export_boolean_window() {
        use super::{export_view_to_c, import_from_c};
        use crate::ArrayV;

        // 11 bools across two bytes. Window crosses the byte boundary
        // so the bit-level realignment path in `import_boolean` runs.
        let mut barr = BooleanArray::default();
        let values = [
            true, false, true, true, false, true, false, false, true, false, true,
        ];
        for v in values {
            barr.push(v);
        }
        let array = Arc::new(Array::BooleanArray(barr.into()));
        let view = ArrayV::new((*array).clone(), 3, 6); // bits 3..9 -> crosses byte 0/1

        let schema = schema_for("b", ArrowType::Boolean, false);
        let (arr_ptr, sch_ptr) =
            export_view_to_c(array.clone(), schema, view.offset as i64, view.len() as i64);

        unsafe {
            let imported = import_from_c(arr_ptr as *const _, sch_ptr as *const _);
            if let Array::BooleanArray(b) = imported.as_ref() {
                assert_eq!(b.len(), 6);
                for (i, expected) in values[3..9].iter().enumerate() {
                    assert_eq!(b.get(i), Some(*expected), "bool index {i}");
                }
            } else {
                panic!("Expected BooleanArray");
            }

            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[cfg(all(feature = "views", feature = "datetime"))]
    #[test]
    fn test_view_export_datetime_window() {
        use super::{export_view_to_c, import_from_c};
        use crate::{ArrayV, TimeUnit};

        let mut dt = DatetimeArray::<i64>::default();
        for v in [1_000, 2_000, 3_000, 4_000, 5_000, 6_000] {
            dt.push(v);
        }
        dt.time_unit = TimeUnit::Milliseconds;

        let array = Arc::new(Array::from_datetime_i64(dt));
        let view = ArrayV::new((*array).clone(), 2, 3); // [3_000, 4_000, 5_000]

        let schema = schema_for("ts", ArrowType::Date64, false);
        let (arr_ptr, sch_ptr) =
            export_view_to_c(array.clone(), schema, view.offset as i64, view.len() as i64);

        unsafe {
            let imported = import_from_c(arr_ptr as *const _, sch_ptr as *const _);
            if let Array::TemporalArray(crate::TemporalArray::Datetime64(d)) = imported.as_ref() {
                assert_eq!(d.data.as_slice(), &[3_000, 4_000, 5_000]);
                assert_eq!(d.time_unit, TimeUnit::Milliseconds);
            } else {
                panic!("Expected Datetime64 array");
            }

            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    /// Exercises offset propagation through `import_categorical`: codes
    /// buffer is windowed via `ArrowArray.offset`, dictionary stays full.
    #[cfg(all(
        feature = "views",
        any(not(feature = "default_categorical_8"), feature = "extended_categorical")
    ))]
    #[test]
    fn test_view_export_categorical_window() {
        use super::{export_view_to_c, import_from_c};
        use crate::{ArrayV, CategoricalArray};

        // Dictionary: ["apple", "banana", "cherry"]. Codes drawn from those.
        let mut cat = CategoricalArray::<u32>::default();
        for s in ["apple", "banana", "apple", "cherry", "banana", "cherry"] {
            cat.push_str(s);
        }
        let array = Arc::new(Array::from_categorical32(cat));
        let view = ArrayV::new((*array).clone(), 2, 3); // apple, cherry, banana

        let schema = schema_for(
            "c",
            ArrowType::Dictionary(crate::ffi::arrow_dtype::CategoricalIndexType::UInt32),
            false,
        );
        let (arr_ptr, sch_ptr) =
            export_view_to_c(array.clone(), schema, view.offset as i64, view.len() as i64);

        unsafe {
            let imported = import_from_c(arr_ptr as *const _, sch_ptr as *const _);
            if let Array::TextArray(crate::TextArray::Categorical32(c)) = imported.as_ref() {
                assert_eq!(c.data.len(), 3);
                assert_eq!(c.get_str(0), Some("apple"));
                assert_eq!(c.get_str(1), Some("cherry"));
                assert_eq!(c.get_str(2), Some("banana"));
            } else {
                panic!("Expected Categorical32 array");
            }

            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }
}
