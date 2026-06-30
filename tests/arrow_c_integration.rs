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

//! End-to-end tests for  Rust -> Arrow C FFI -> C inspectors
//! Run with:
//!     cargo test arrow_c_integration --features c_ffi_tests
#[cfg(all(feature = "c_ffi_tests", test))]
mod arrow_c_integration {
    use std::ffi::CString;
    use std::os::raw::{c_char, c_int};
    use std::sync::Arc;

    #[cfg(feature = "datetime")]
    use minarrow::TimeUnit;
    use minarrow::ffi::arrow_c_ffi::{ArrowArray, ArrowSchema, export_to_c};
    use minarrow::ffi::schema::Schema;
    use minarrow::{
        Array, ArrowType, BooleanArray, Field, FloatArray, IntegerArray, MaskedArray, StringArray,
        TextArray,
    };

    // ---- C inspectors ----------------------------------------------------
    #[link(name = "cinspect_arrow", kind = "static")]
    unsafe extern "C" {
        fn c_arrow_check_i32(arr: *const ArrowArray) -> c_int;
        fn c_arrow_check_i64(arr: *const ArrowArray) -> c_int;
        fn c_arrow_check_u32(arr: *const ArrowArray) -> c_int;
        fn c_arrow_check_f32(arr: *const ArrowArray) -> c_int;
        fn c_arrow_check_f64(arr: *const ArrowArray) -> c_int;
        fn c_arrow_check_bool(arr: *const ArrowArray) -> c_int;
        fn c_arrow_check_str(arr: *const ArrowArray) -> c_int;
        fn c_arrow_check_i32_null(arr: *const ArrowArray) -> c_int;
        #[cfg(feature = "datetime")]
        fn c_arrow_check_dt64(arr: *const ArrowArray) -> c_int;
        fn c_arrow_check_dict32(arr: *const ArrowArray) -> c_int;
        fn c_arrow_check_schema(
            schema: *const ArrowSchema,
            expected_name: *const c_char,
            expected_format: *const c_char,
        ) -> c_int;
    }

    fn schema_for(name: &str, ty: ArrowType, nullable: bool) -> Schema {
        Schema {
            fields: vec![Field::new(name, ty, nullable, None)],
            ..Default::default()
        }
    }

    fn expect_format_bytes(ty: &ArrowType) -> &'static [u8] {
        match ty {
            ArrowType::Int32 => b"i",
            ArrowType::Int64 => b"l",
            ArrowType::UInt32 => b"I",
            ArrowType::Float32 => b"f",
            ArrowType::Float64 => b"g",
            ArrowType::Boolean => b"b",
            ArrowType::String => b"u",
            #[cfg(feature = "datetime")]
            ArrowType::Date32 => b"tdD",
            #[cfg(feature = "datetime")]
            ArrowType::Date64 => b"tdm",
            // For dictionary, Arrow C format is the *index* physical code.
            ArrowType::Dictionary(_k) => b"I",
            _ => panic!("format mapping not covered for {:?}", ty),
        }
    }

    macro_rules! roundtrip {
        ($arr_enum:expr, $arrow_ty:expr, $nullable:expr, $checker:path) => {{
            let name = "x";
            let schema = schema_for(name, $arrow_ty.clone(), $nullable);
            let (arrow_ptr, schema_ptr) = export_to_c(Arc::new($arr_enum), schema);

            let ok = unsafe { $checker(arrow_ptr) };
            assert_eq!(ok, 1, "C inspector failed for {:?}", $arrow_ty);

            // Also validate schema fields match (name + format)
            let cname = CString::new(name).unwrap();
            let cfmt = CString::new(expect_format_bytes(&$arrow_ty)).unwrap();
            let sch_ok = unsafe { c_arrow_check_schema(schema_ptr, cname.as_ptr(), cfmt.as_ptr()) };
            assert_eq!(sch_ok, 1, "schema check failed for {:?}", $arrow_ty);

            unsafe {
                ((*arrow_ptr).release.unwrap())(arrow_ptr);
                ((*schema_ptr).release.unwrap())(schema_ptr);
            }
        }};
    }

    #[test]
    fn rt_i32() {
        let arr = IntegerArray::<i32>::from_slice(&[11, 22, 33]);
        roundtrip!(
            Array::from_int32(arr),
            ArrowType::Int32,
            false,
            c_arrow_check_i32
        );
    }

    #[test]
    fn rt_i64() {
        let arr = IntegerArray::<i64>::from_slice(&[1001, -42, 777]);
        roundtrip!(
            Array::from_int64(arr),
            ArrowType::Int64,
            false,
            c_arrow_check_i64
        );
    }

    #[test]
    fn rt_u32() {
        let arr = IntegerArray::<u32>::from_slice(&[1, 2, 3]);
        roundtrip!(
            Array::from_uint32(arr),
            ArrowType::UInt32,
            false,
            c_arrow_check_u32
        );
    }

    #[test]
    fn rt_f32() {
        let arr = FloatArray::<f32>::from_slice(&[1.5, -2.0, 3.25]);
        roundtrip!(
            Array::from_float32(arr),
            ArrowType::Float32,
            false,
            c_arrow_check_f32
        );
    }

    #[test]
    fn rt_f64() {
        let arr = FloatArray::<f64>::from_slice(&[0.1, 0.2, 0.3]);
        roundtrip!(
            Array::from_float64(arr),
            ArrowType::Float64,
            false,
            c_arrow_check_f64
        );
    }

    #[test]
    fn rt_bool() {
        let arr = BooleanArray::<()>::from_slice(&[true, false, true]);
        roundtrip!(
            Array::BooleanArray(arr.into()),
            ArrowType::Boolean,
            false,
            c_arrow_check_bool
        );
    }

    #[test]
    fn rt_utf8() {
        let arr = StringArray::<u32>::from_slice(&["foo", "bar"]);
        roundtrip!(
            Array::TextArray(TextArray::String32(Arc::new(arr))),
            ArrowType::String,
            false,
            c_arrow_check_str
        );
    }

    #[test]
    fn rt_i32_with_nulls() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(42);
        arr.push_null();
        arr.push(88);
        roundtrip!(
            Array::from_int32(arr),
            ArrowType::Int32,
            true,
            c_arrow_check_i32_null
        );
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn rt_date64_ms() {
        let mut dt = minarrow::DatetimeArray::<i64>::default();
        dt.push(1);
        dt.push(2);
        dt.time_unit = TimeUnit::Milliseconds; // Date64 == ms since epoch
        roundtrip!(
            Array::from_datetime_i64(dt),
            ArrowType::Date64,
            false,
            c_arrow_check_dt64
        );
    }

    #[cfg(any(
        not(feature = "default_categorical_8"),
        feature = "extended_categorical"
    ))]
    #[test]
    fn rt_dict32() {
        let cat = minarrow::CategoricalArray::<u32>::from_slices(
            &[0, 1, 0],
            &["A".to_string(), "B".to_string()],
        );
        let arr = Array::TextArray(TextArray::Categorical32(Arc::new(cat)));
        roundtrip!(
            arr,
            ArrowType::Dictionary(minarrow::ffi::arrow_dtype::CategoricalIndexType::UInt32),
            false,
            c_arrow_check_dict32
        );
    }
}
