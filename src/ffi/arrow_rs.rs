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

//! # **arrow-rs Bridge** - *Adapter between Minarrow's C Data Interface and the `arrow` crate*
//!
//! Thin reinterpret layer over [`crate::ffi::arrow_c_ffi::export_to_c`] and
//! [`crate::ffi::arrow_c_ffi::import_from_c_owned`]. Both libraries follow the
//! Apache Arrow C Data Interface spec, so the C structs are layout-compatible
//! and the conversion is just a pointer cast plus a hand-off.
//!
//! Used by `Array::to_apache_arrow`, `FieldArray::to_apache_arrow`,
//! `Table::to_apache_arrow` and their `from_*` siblings (and the `Super*`
//! chunked equivalents).
//!
//! Gated by the `cast_arrow` feature.

use std::sync::Arc;

use arrow::array::ArrayRef;

use crate::Field;
use crate::enums::error::MinarrowError;
use crate::ffi::arrow_c_ffi::{ArrowArray, ArrowSchema, export_to_c, import_from_c_owned};
use crate::ffi::schema::Schema;
use crate::Array;

/// Export a Minarrow array to an arrow-rs `ArrayRef`.
///
/// `schema.fields[0]` supplies the logical type for the export (preserves
/// Timestamp/Time/Duration/Interval semantics).
pub fn export(array: Arc<Array>, schema: Schema) -> Result<ArrayRef, MinarrowError> {
    let (c_arr, c_schema) = export_to_c(array, schema);

    // Reinterpret as arrow-rs's layout-identical FFI structs, move contents
    // out (arrow-rs takes ownership of the release callback), then free the
    // now-empty heap wrappers allocated by `export_to_c`. The data Holder
    // lives at `private_data` which moved into arrow-rs's copy, so release
    // still fires when arrow-rs drops its copy.
    let arr_ptr = c_arr as *mut arrow::array::ffi::FFI_ArrowArray;
    let sch_ptr = c_schema as *mut arrow::array::ffi::FFI_ArrowSchema;
    let ffi_arr = unsafe { arr_ptr.read() };
    let ffi_sch = unsafe { sch_ptr.read() };
    unsafe {
        drop(Box::from_raw(c_arr));
        drop(Box::from_raw(c_schema));
    }

    let array_data = unsafe { arrow::array::ffi::from_ffi(ffi_arr, &ffi_sch) }?;
    Ok(arrow::array::make_array(array_data))
}

/// Import an arrow-rs `ArrayRef` into a Minarrow `(Arc<Array>, Field)`.
///
/// arrow-rs `ArrayRef` does not carry a column name; the returned `Field`
/// has an empty `name` slot. Callers wanting a name should assign one.
pub fn import(arr: &ArrayRef) -> Result<(Arc<Array>, Field), MinarrowError> {
    let array_data = arr.to_data();
    let (ffi_arr, ffi_sch) = arrow::array::ffi::to_ffi(&array_data)?;

    // arrow-rs's FFI structs are layout-identical to minarrow's; transfer
    // ownership via a Box pointer cast so `import_from_c_owned` can take
    // it zero-copy.
    let arr_ptr = Box::into_raw(Box::new(ffi_arr)) as *mut ArrowArray;
    let sch_ptr = Box::into_raw(Box::new(ffi_sch)) as *mut ArrowSchema;
    let arr_box = unsafe { Box::from_raw(arr_ptr) };
    let sch_box = unsafe { Box::from_raw(sch_ptr) };

    let (array, field) = unsafe { import_from_c_owned(arr_box, sch_box) };
    Ok((array, field))
}
