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

use crate::enums::error::KernelError;
use crate::{
    Array, ArrayV, Bitmask, BooleanArray, FloatArray, IntegerArray, MaskedArray, NumericArray,
    StringArray, TextArray, Vec64, vec64,
};

/// Repeat a length-1 `Array` to `len`.  
/// Errors if the input length is *not* 1, or the variant is unsupported.  
pub fn broadcast_length_1_array(av: ArrayV, len: usize) -> Result<Array, KernelError> {
    debug_assert_eq!(av.len(), 1, "caller guarantees scalar input");

    match av.array {
        Array::NumericArray(NumericArray::Int32(a)) => Ok(Array::from_int32(
            IntegerArray::<i32>::from_vec64(vec64![a.data[0]; len], None),
        )),
        Array::NumericArray(NumericArray::Int64(a)) => Ok(Array::from_int64(
            IntegerArray::<i64>::from_vec64(vec64![a.data[0]; len], None),
        )),
        Array::NumericArray(NumericArray::UInt32(a)) => Ok(Array::from_uint32(
            IntegerArray::<u32>::from_vec64(vec64![a.data[0]; len], None),
        )),
        Array::NumericArray(NumericArray::UInt64(a)) => Ok(Array::from_uint64(
            IntegerArray::<u64>::from_vec64(vec64![a.data[0]; len], None),
        )),
        Array::NumericArray(NumericArray::Float32(a)) => Ok(Array::from_float32(
            FloatArray::<f32>::from_vec64(vec64![a.data[0]; len], None),
        )),
        Array::NumericArray(NumericArray::Float64(a)) => Ok(Array::from_float64(
            FloatArray::<f64>::from_vec64(vec64![a.data[0]; len], None),
        )),
        Array::BooleanArray(a) => match a.get(0) {
            Some(v) => {
                let bitmask = Bitmask::new_set_all(len, v);
                Ok(Array::BooleanArray(BooleanArray::new(bitmask, None).into()))
            }
            None => Err(KernelError::UnsupportedType(
                "broadcasting null boolean values not supported in dense mode".into(),
            )),
        },
        Array::TextArray(TextArray::String32(a)) => {
            // Get the first string from the array, which should have exactly 1 string
            let s = a.get_str(av.offset).unwrap_or("");
            let strs: Vec64<&str> = std::iter::repeat(s).take(len).collect();
            Ok(Array::from_string32(StringArray::from_vec64(strs, None)))
        }
        #[cfg(feature = "large_string")]
        Array::TextArray(TextArray::String64(a)) => {
            // Get the first string from the array, which should have exactly 1 string
            let s = a.get_str(av.offset).unwrap_or("");
            let strs: Vec64<&str> = std::iter::repeat(s).take(len).collect();
            Ok(Array::from_string64(StringArray::from_vec64(strs, None)))
        }
        _ => {
            return Err(KernelError::UnsupportedType(
                "broadcast not yet implemented for this array variant".into(),
            ));
        }
    }
}

/// Ensure `lhs` and `rhs` have identical length, broadcasting the scalar
/// side if exactly one of them has length 1.
pub fn maybe_broadcast_scalar_array<'a>(
    lhs: ArrayV,
    rhs: ArrayV,
) -> Result<(ArrayV, ArrayV), KernelError> {
    let (l, r) = (lhs.len(), rhs.len());

    if l == r {
        return Ok((lhs.clone(), rhs.clone()));
    }
    if l == 1 {
        return Ok((
            ArrayV::new(broadcast_length_1_array(lhs, r)?, 0, rhs.len()),
            rhs.clone(),
        ));
    }
    if r == 1 {
        return Ok((
            lhs.clone(),
            ArrayV::new(broadcast_length_1_array(rhs, l)?, 0, lhs.len()),
        ));
    }

    Err(KernelError::LengthMismatch(format!(
        "cannot broadcast arrays of length {l} and {r}"
    )))
}
