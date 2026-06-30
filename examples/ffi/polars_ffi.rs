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
//! Minarrow <->️ Polars (via polars_arrow/arrow2) FFI roundtrip
//!
//! Run with:
//!    cargo run --example polars_ffi --features cast_polars
//!
//! This is for custom FFI - you can instead also directly go to polars
//! via `to_polars()` from the `Array`, `FieldArray` or `Table`
//! types when the *cast_polars* feature is activated.
//! ---------------------------------------------------------

#[cfg(feature = "cast_polars")]
use crate::polars_roundtrip::run_example;

#[cfg(feature = "cast_polars")]
mod polars_roundtrip {
    use std::sync::Arc;

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
    use polars::prelude::*;
    use polars_arrow as pa;

    // -------------------------------------------------------------------------
    // Build test table with full type coverage
    // -------------------------------------------------------------------------
    fn build_minarrow_table() -> Table {
        // Arrays
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

        // Wrap in Array enums
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

        // Fields
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

        // FieldArrays
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

        // Build table
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
        Table::new("polars_ffi_test".to_string(), Some(cols))
    }

    // Minarrow -> C -> arrow2
    fn minarrow_col_to_arrow2(
        array: &Array,
        field: &Field,
    ) -> (Box<dyn pa::array::Array>, pa::datatypes::Field) {
        let schema = Schema::from(vec![field.clone()]);
        let (c_arr, c_schema) = export_to_c(Arc::new(array.clone()), schema);
        let arr_ptr = c_arr as *mut pa::ffi::ArrowArray;
        let sch_ptr = c_schema as *mut pa::ffi::ArrowSchema;
        unsafe {
            let arr_val = std::ptr::read(arr_ptr);
            let sch_val = std::ptr::read(sch_ptr);
            let fld = pa::ffi::import_field_from_c(&sch_val)
                .expect("polars_arrow import_field_from_c failed");
            let dtype = fld.dtype().clone();
            let arr = pa::ffi::import_array_from_c(arr_val, dtype)
                .expect("polars_arrow import_array_from_c failed");
            (arr, fld)
        }
    }

    // arrow2 -> Polars
    fn series_from_arrow(name: &str, a: Box<dyn pa::array::Array>) -> Series {
        Series::from_arrow(name.into(), a).expect("Polars Series::from_arrow failed")
    }

    // Polars -> C
    fn export_series_to_c(name: &str, s: &Series) -> (pa::ffi::ArrowArray, pa::ffi::ArrowSchema) {
        let arr2 = s.to_arrow(0, CompatLevel::oldest());
        let out_arr: pa::ffi::ArrowArray = pa::ffi::export_array_to_c(arr2.clone());
        let fld = pa::datatypes::Field::new(name.into(), arr2.dtype().clone(), false);
        let out_sch: pa::ffi::ArrowSchema = pa::ffi::export_field_to_c(&fld);
        (out_arr, out_sch)
    }

    // C -> Minarrow
    fn import_back_minarrow(
        out_arr: pa::ffi::ArrowArray,
        out_sch: pa::ffi::ArrowSchema,
    ) -> Arc<Array> {
        let back_arr_ptr =
            Box::into_raw(Box::new(out_arr)) as *const minarrow::ffi::arrow_c_ffi::ArrowArray;
        let back_sch_ptr =
            Box::into_raw(Box::new(out_sch)) as *const minarrow::ffi::arrow_c_ffi::ArrowSchema;
        unsafe { import_from_c(back_arr_ptr, back_sch_ptr) }
    }

    // Equality with String32 <-> String64 relaxed match
    fn arrays_equal_allow_utf8_width(left: &Array, right: &Array) -> bool {
        if left == right {
            return true;
        }
        match (left, right) {
            (
                Array::TextArray(TextArray::String32(a)),
                Array::TextArray(TextArray::String64(b)),
            )
            | (
                Array::TextArray(TextArray::String64(b)),
                Array::TextArray(TextArray::String32(a)),
            ) => {
                let a = a.as_ref();
                let b = b.as_ref();
                a.len() == b.len()
                    && a.null_mask == b.null_mask
                    && (0..a.len()).all(|i| a.get(i) == b.get(i))
            }
            _ => false,
        }
    }

    pub(crate) fn run_example() {
        let minarrow_table = build_minarrow_table();
        for col in &minarrow_table.cols {
            let field_name = &col.field.name;

            let (arrow2_array, _) = minarrow_col_to_arrow2(&col.array, &col.field);
            let s = series_from_arrow(field_name, arrow2_array);

            let c = Column::new("TestCol".into(), s.clone());
            let df = DataFrame::new(s.len(), vec![c]).expect("build DataFrame");
            println!("{df}");

            let (out_arr, out_sch) = export_series_to_c(field_name, &s);
            let minarr_back = import_back_minarrow(out_arr, out_sch);

            assert!(
                arrays_equal_allow_utf8_width(&col.array, minarr_back.as_ref()),
                "Roundtrip mismatch for field {field_name}"
            );
        }
    }
}

fn main() {
    if cfg!(feature = "cast_polars") {
        #[cfg(feature = "cast_polars")]
        run_example()
    } else {
        println!("The polars-FFI example requires enabling the `cast_polars` feature.")
    }
}
