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

//! ---------------------------------------------------------
//! Runs a roundtrip to and from Apache Arrow
//!
//! Run with:
//!    cargo run --example apache_arrow_ffi --features cast_arrow
//! ---------------------------------------------------------

#[cfg(feature = "cast_arrow")]
use crate::apache_arrow_test::run_example;

// examples/ffi_roundtrip.rs
#[cfg(feature = "cast_arrow")]
mod apache_arrow_test {
    use std::sync::Arc;

    use arrow::array::ffi::{
        FFI_ArrowArray, FFI_ArrowSchema, from_ffi as arrow_from_ffi, to_ffi as arrow_to_ffi,
    };
    use arrow::array::{ArrayRef, RecordBatch, make_array};
    use minarrow::ffi::arrow_c_ffi::{export_to_c, import_from_c};
    #[cfg(any(
        not(feature = "default_categorical_8"),
        feature = "extended_categorical"
    ))]
    use minarrow::ffi::arrow_dtype::CategoricalIndexType;
    use minarrow::ffi::schema::Schema;
    use minarrow::{Array, ArrowType, Field, FieldArray, NumericArray, Table, TextArray};
    #[cfg(feature = "datetime")]
    use minarrow::{TemporalArray, TimeUnit};

    pub(crate) fn run_example() {
        // ---- 1. Build a Minarrow Table with all types ----

        #[cfg(feature = "extended_numeric_types")]
        let arr_int8 = Arc::new(minarrow::IntegerArray::<i8>::from_slice(&[1, 2, -1])) as Arc<_>;
        #[cfg(feature = "extended_numeric_types")]
        let arr_int16 =
            Arc::new(minarrow::IntegerArray::<i16>::from_slice(&[10, 20, -10])) as Arc<_>;
        let arr_int32 =
            Arc::new(minarrow::IntegerArray::<i32>::from_slice(&[100, 200, -100])) as Arc<_>;
        let arr_int64 = Arc::new(minarrow::IntegerArray::<i64>::from_slice(&[
            1000, 2000, -1000,
        ])) as Arc<_>;

        #[cfg(feature = "extended_numeric_types")]
        let arr_uint8 = Arc::new(minarrow::IntegerArray::<u8>::from_slice(&[1, 2, 255]))
            as Arc<minarrow::IntegerArray<u8>>;
        #[cfg(feature = "extended_numeric_types")]
        let arr_uint16 = Arc::new(minarrow::IntegerArray::<u16>::from_slice(&[1, 2, 65535]))
            as Arc<minarrow::IntegerArray<u16>>;
        let arr_uint32 = Arc::new(minarrow::IntegerArray::<u32>::from_slice(&[
            1, 2, 4294967295,
        ])) as Arc<minarrow::IntegerArray<u32>>;
        let arr_uint64 = Arc::new(minarrow::IntegerArray::<u64>::from_slice(&[
            1,
            2,
            18446744073709551615,
        ])) as Arc<minarrow::IntegerArray<u64>>;

        let arr_float32 = Arc::new(minarrow::FloatArray::<f32>::from_slice(&[1.5, -0.5, 0.0]))
            as Arc<minarrow::FloatArray<f32>>;
        let arr_float64 = Arc::new(minarrow::FloatArray::<f64>::from_slice(&[1.0, -2.0, 0.0]))
            as Arc<minarrow::FloatArray<f64>>;

        let arr_bool = Arc::new(minarrow::BooleanArray::<()>::from_slice(&[
            true, false, true,
        ])) as Arc<minarrow::BooleanArray<()>>;

        let arr_string32 = Arc::new(minarrow::StringArray::<u32>::from_slice(&[
            "abc", "def", "",
        ])) as Arc<minarrow::StringArray<u32>>;
        #[cfg(any(
            not(feature = "default_categorical_8"),
            feature = "extended_categorical"
        ))]
        let arr_categorical32 = Arc::new(minarrow::CategoricalArray::<u32>::from_slices(
            &[0, 1, 2],
            &["A".to_string(), "B".to_string(), "C".to_string()],
        )) as Arc<minarrow::CategoricalArray<u32>>;

        #[cfg(feature = "datetime")]
        let arr_datetime32 = Arc::new(minarrow::DatetimeArray::<i32> {
            data: minarrow::Buffer::<i32>::from_slice(&[
                1_600_000_000 / 86_400,
                1_600_000_001 / 86_400,
                1_600_000_002 / 86_400,
            ]),
            null_mask: None,
            time_unit: TimeUnit::Days,
        });
        #[cfg(feature = "datetime")]
        let arr_datetime64 = Arc::new(minarrow::DatetimeArray::<i64> {
            data: minarrow::Buffer::<i64>::from_slice(&[
                1_600_000_000_000,
                1_600_000_000_001,
                1_600_000_000_002,
            ]),
            null_mask: None,
            time_unit: TimeUnit::Milliseconds,
        }) as Arc<_>;

        // ---- 2. Wrap into Array enums ----
        #[cfg(feature = "extended_numeric_types")]
        let minarr_int8 = Array::NumericArray(NumericArray::Int8(arr_int8));
        #[cfg(feature = "extended_numeric_types")]
        let minarr_int16 = Array::NumericArray(NumericArray::Int16(arr_int16));
        let minarr_int32 = Array::NumericArray(NumericArray::Int32(arr_int32));
        let minarr_int64 = Array::NumericArray(NumericArray::Int64(arr_int64));
        #[cfg(feature = "extended_numeric_types")]
        let minarr_uint8 = Array::NumericArray(NumericArray::UInt8(arr_uint8));
        #[cfg(feature = "extended_numeric_types")]
        let minarr_uint16 = Array::NumericArray(NumericArray::UInt16(arr_uint16));
        let minarr_uint32 = Array::NumericArray(NumericArray::UInt32(arr_uint32));
        let minarr_uint64 = Array::NumericArray(NumericArray::UInt64(arr_uint64));
        let minarr_float32 = Array::NumericArray(NumericArray::Float32(arr_float32));
        let minarr_float64 = Array::NumericArray(NumericArray::Float64(arr_float64));
        let minarr_bool = Array::BooleanArray(arr_bool);
        let minarr_string32 = Array::TextArray(TextArray::String32(arr_string32));
        #[cfg(any(
            not(feature = "default_categorical_8"),
            feature = "extended_categorical"
        ))]
        let minarr_categorical32 = Array::TextArray(TextArray::Categorical32(arr_categorical32));
        #[cfg(feature = "datetime")]
        let minarr_datetime32 = Array::TemporalArray(TemporalArray::Datetime32(arr_datetime32));
        #[cfg(feature = "datetime")]
        let minarr_datetime64 = Array::TemporalArray(TemporalArray::Datetime64(arr_datetime64));

        // ---- 3. Build Fields with correct logical types ----
        #[cfg(feature = "extended_numeric_types")]
        let field_int8 = Field::new("int8", ArrowType::Int8, false, None);
        #[cfg(feature = "extended_numeric_types")]
        let field_int16 = Field::new("int16", ArrowType::Int16, false, None);
        let field_int32 = Field::new("int32", ArrowType::Int32, false, None);
        let field_int64 = Field::new("int64", ArrowType::Int64, false, None);
        #[cfg(feature = "extended_numeric_types")]
        let field_uint8 = Field::new("uint8", ArrowType::UInt8, false, None);
        #[cfg(feature = "extended_numeric_types")]
        let field_uint16 = Field::new("uint16", ArrowType::UInt16, false, None);
        let field_uint32 = Field::new("uint32", ArrowType::UInt32, false, None);
        let field_uint64 = Field::new("uint64", ArrowType::UInt64, false, None);
        let field_float32 = Field::new("float32", ArrowType::Float32, false, None);
        let field_float64 = Field::new("float64", ArrowType::Float64, false, None);
        let field_bool = Field::new("bool", ArrowType::Boolean, false, None);
        let field_string32 = Field::new("string32", ArrowType::String, false, None);
        #[cfg(any(
            not(feature = "default_categorical_8"),
            feature = "extended_categorical"
        ))]
        let field_categorical32 = Field::new(
            "categorical32",
            ArrowType::Dictionary(CategoricalIndexType::UInt32),
            false,
            None,
        );

        #[cfg(feature = "datetime")]
        let field_datetime32 = Field::new("dt32", ArrowType::Date32, false, None);
        #[cfg(feature = "datetime")]
        let field_datetime64 = Field::new("dt64", ArrowType::Date64, false, None);

        // ---- 4. Build FieldArrays ----
        #[cfg(feature = "extended_numeric_types")]
        let fa_int8 = FieldArray::new(field_int8, minarr_int8);
        #[cfg(feature = "extended_numeric_types")]
        let fa_int16 = FieldArray::new(field_int16, minarr_int16);
        let fa_int32 = FieldArray::new(field_int32, minarr_int32);
        let fa_int64 = FieldArray::new(field_int64, minarr_int64);
        #[cfg(feature = "extended_numeric_types")]
        let fa_uint8 = FieldArray::new(field_uint8, minarr_uint8);
        #[cfg(feature = "extended_numeric_types")]
        let fa_uint16 = FieldArray::new(field_uint16, minarr_uint16);
        let fa_uint32 = FieldArray::new(field_uint32, minarr_uint32);
        let fa_uint64 = FieldArray::new(field_uint64, minarr_uint64);
        let fa_float32 = FieldArray::new(field_float32, minarr_float32);
        let fa_float64 = FieldArray::new(field_float64, minarr_float64);
        let fa_bool = FieldArray::new(field_bool, minarr_bool);
        let fa_string32 = FieldArray::new(field_string32, minarr_string32);
        #[cfg(any(
            not(feature = "default_categorical_8"),
            feature = "extended_categorical"
        ))]
        let fa_categorical32 = FieldArray::new(field_categorical32, minarr_categorical32);
        #[cfg(feature = "datetime")]
        let fa_datetime32 = FieldArray::new(field_datetime32, minarr_datetime32);
        #[cfg(feature = "datetime")]
        let fa_datetime64 = FieldArray::new(field_datetime64, minarr_datetime64);

        // ---- 5. Build Table ----
        let mut cols = Vec::new();
        #[cfg(feature = "extended_numeric_types")]
        {
            cols.push(fa_int8);
            cols.push(fa_int16);
        }
        cols.push(fa_int32);
        cols.push(fa_int64);
        #[cfg(feature = "extended_numeric_types")]
        {
            cols.push(fa_uint8);
            cols.push(fa_uint16);
        }
        cols.push(fa_uint32);
        cols.push(fa_uint64);
        cols.push(fa_float32);
        cols.push(fa_float64);
        cols.push(fa_bool);
        cols.push(fa_string32);
        #[cfg(any(
            not(feature = "default_categorical_8"),
            feature = "extended_categorical"
        ))]
        cols.push(fa_categorical32);
        #[cfg(feature = "datetime")]
        {
            cols.push(fa_datetime32);
            cols.push(fa_datetime64);
        }
        let minarrow_table = Table::new("ffi_test".to_string(), Some(cols));

        // ---- 6. Export each column over FFI, import into Arrow-RS, and roundtrip back to Minarrow ----
        for (_, col) in minarrow_table.cols.iter().enumerate() {
            let array_arc = Arc::new(col.array.clone());
            let schema = Schema::from(vec![(*col.field).clone()]);

            // println!("Minarrow Pre-roundtrip for '{:?}':\n{:#?}", *col.field, array_arc);

            let (c_arr, c_schema) = export_to_c(array_arc.clone(), schema);

            // SAFETY: Arrow-RS expects raw pointers to FFI_ArrowArray/Schema
            let arr_ptr = c_arr as *mut FFI_ArrowArray;
            let schema_ptr = c_schema as *mut FFI_ArrowSchema;
            let arrow_array = unsafe { arr_ptr.read() };
            let arrow_schema = unsafe { schema_ptr.read() };
            let array_data = unsafe { arrow_from_ffi(arrow_array, &arrow_schema) }
                .expect("Arrow FFI import failed");
            let field_name = &col.field.name;
            println!(
                "Imported field '{}' as Arrow type {:?}",
                field_name,
                array_data.data_type()
            );
            println!("Arrow-RS values for '{}':", field_name);
            println!("  {:?}", array_data);

            // Convert ArrayData to ArrayRef
            let array_ref: ArrayRef = make_array(array_data.clone());

            // Pretty print as a table
            let arrow_schema = Arc::new(arrow::datatypes::Schema::new(vec![
                arrow::datatypes::Field::new(field_name, array_ref.data_type().clone(), false),
            ]));
            let batch = RecordBatch::try_new(arrow_schema, vec![array_ref.clone()]).unwrap();
            println!("Arrow-RS pretty-print for '{}':", field_name);
            arrow::util::pretty::print_batches(&[batch]).unwrap();

            // ---- 7. Export Arrow-RS back to Minarrow FFI, roundtrip ----
            let (ffi_out_arr, ffi_out_schema) =
                arrow_to_ffi(&array_data).expect("Arrow to FFI failed");

            // Correctly allocate Arrow-RS FFI structs on the heap and cast as raw pointers to your C ABI structs
            let ffi_out_arr_box = Box::new(ffi_out_arr);
            let ffi_out_schema_box = Box::new(ffi_out_schema);

            let arr_ptr =
                Box::into_raw(ffi_out_arr_box) as *const minarrow::ffi::arrow_c_ffi::ArrowArray;
            let schema_ptr =
                Box::into_raw(ffi_out_schema_box) as *const minarrow::ffi::arrow_c_ffi::ArrowSchema;

            // Now import back into minarrow using your real FFI import
            let minarr_back_array: Arc<Array> = unsafe { import_from_c(arr_ptr, schema_ptr) };

            println!(
                "Minarrow array (roundtrip) for '{}':\n{:#?}",
                field_name, minarr_back_array
            );

            // ---- 8. Validate roundtrip equality ----
            assert_eq!(
                &col.array,
                minarr_back_array.as_ref(),
                "Roundtrip array does not match for field {}",
                field_name
            );
        }

        println!("FFI roundtrip test completed for all supported types.");
    }
}

fn main() {
    if cfg!(feature = "cast_arrow") {
        #[cfg(feature = "cast_arrow")]
        run_example()
    } else {
        println!("The apache-FFI example requires enabling the `cast_arrow` feature.")
    }
}
