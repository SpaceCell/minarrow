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

use crate::enums::error::KernelError;
#[cfg(feature = "decimal")]
use crate::DecimalArray;
use crate::{
    Array, ArrayV, Bitmask, BooleanArray, FloatArray, IntegerArray, NumericArray,
    StringArray, TextArray, Vec64, vec64,
};

/// Repeat a length-1 `Array` to `len`.
/// Errors if the input length is *not* 1, or the variant is unsupported.  
pub fn broadcast_length_1_array(av: ArrayV, len: usize) -> Result<Array, KernelError> {
    debug_assert_eq!(av.len(), 1, "caller guarantees scalar input");

    let null_mask = match av.array.null_mask() {
        Some(m) if !m.get(0) => Some(Bitmask::new_set_all(len, false)),
        _ => None,
    };

    match av.array {
        Array::NumericArray(NumericArray::Int32(a)) => Ok(Array::from_int32(
            IntegerArray::<i32>::from_vec64(vec64![a.data[0]; len], null_mask),
        )),
        Array::NumericArray(NumericArray::Int64(a)) => Ok(Array::from_int64(
            IntegerArray::<i64>::from_vec64(vec64![a.data[0]; len], null_mask),
        )),
        Array::NumericArray(NumericArray::UInt32(a)) => Ok(Array::from_uint32(
            IntegerArray::<u32>::from_vec64(vec64![a.data[0]; len], null_mask),
        )),
        Array::NumericArray(NumericArray::UInt64(a)) => Ok(Array::from_uint64(
            IntegerArray::<u64>::from_vec64(vec64![a.data[0]; len], null_mask),
        )),
        Array::NumericArray(NumericArray::Float32(a)) => Ok(Array::from_float32(
            FloatArray::<f32>::from_vec64(vec64![a.data[0]; len], null_mask),
        )),
        Array::NumericArray(NumericArray::Float64(a)) => Ok(Array::from_float64(
            FloatArray::<f64>::from_vec64(vec64![a.data[0]; len], null_mask),
        )),
        Array::BooleanArray(a) => {
            let v = a.data.get(0);
            let bitmask = Bitmask::new_set_all(len, v);
            Ok(Array::BooleanArray(BooleanArray::new(bitmask, null_mask).into()))
        }
        Array::TextArray(TextArray::String32(a)) => {
            let s = a.get_str(av.offset).unwrap_or("");
            let strs: Vec64<&str> = std::iter::repeat(s).take(len).collect();
            Ok(Array::from_string32(StringArray::from_vec64(strs, null_mask)))
        }
        #[cfg(feature = "large_string")]
        Array::TextArray(TextArray::String64(a)) => {
            let s = a.get_str(av.offset).unwrap_or("");
            let strs: Vec64<&str> = std::iter::repeat(s).take(len).collect();
            Ok(Array::from_string64(StringArray::from_vec64(strs, null_mask)))
        }
        #[cfg(feature = "decimal")]
        Array::NumericArray(NumericArray::Decimal32(a)) => Ok(Array::from_decimal32(
            DecimalArray::from_vec64(vec64![a.data[0]; len], null_mask, a.precision, a.scale),
        )),
        #[cfg(feature = "decimal")]
        Array::NumericArray(NumericArray::Decimal64(a)) => Ok(Array::from_decimal64(
            DecimalArray::from_vec64(vec64![a.data[0]; len], null_mask, a.precision, a.scale),
        )),
        #[cfg(feature = "decimal")]
        Array::NumericArray(NumericArray::Decimal128(a)) => Ok(Array::from_decimal128(
            DecimalArray::from_vec64(vec64![a.data[0]; len], null_mask, a.precision, a.scale),
        )),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "decimal")]
    #[test]
    fn test_broadcast_length_1_decimal32() {
        let arr = Array::from_decimal32(DecimalArray::from_slice(&[12345i32], 10, 2));
        let av = ArrayV::new(arr, 0, 1);
        let result = broadcast_length_1_array(av, 4).unwrap();
        if let Array::NumericArray(NumericArray::Decimal32(a)) = result {
            assert_eq!(a.data.as_slice(), &[12345, 12345, 12345, 12345]);
            assert_eq!(a.precision, 10);
            assert_eq!(a.scale, 2);
            assert!(a.null_mask.is_none());
        } else {
            panic!("Expected Decimal32 array");
        }
    }

    #[cfg(feature = "decimal")]
    #[test]
    fn test_broadcast_length_1_decimal64() {
        let arr = Array::from_decimal64(DecimalArray::from_slice(&[99999i64], 18, 4));
        let av = ArrayV::new(arr, 0, 1);
        let result = broadcast_length_1_array(av, 3).unwrap();
        if let Array::NumericArray(NumericArray::Decimal64(a)) = result {
            assert_eq!(a.data.as_slice(), &[99999, 99999, 99999]);
            assert_eq!(a.precision, 18);
            assert_eq!(a.scale, 4);
        } else {
            panic!("Expected Decimal64 array");
        }
    }

    #[cfg(feature = "decimal")]
    #[test]
    fn test_broadcast_length_1_decimal128() {
        let arr = Array::from_decimal128(DecimalArray::from_slice(&[500i128], 38, 0));
        let av = ArrayV::new(arr, 0, 1);
        let result = broadcast_length_1_array(av, 2).unwrap();
        if let Array::NumericArray(NumericArray::Decimal128(a)) = result {
            assert_eq!(a.data.as_slice(), &[500, 500]);
            assert_eq!(a.precision, 38);
            assert_eq!(a.scale, 0);
        } else {
            panic!("Expected Decimal128 array");
        }
    }

    #[cfg(feature = "decimal")]
    #[test]
    fn test_broadcast_length_1_decimal_null_propagation() {
        let mut arr = DecimalArray::from_slice(&[0i64], 10, 2);
        arr.null_mask = Some(Bitmask::new_set_all(1, false));
        let av = ArrayV::new(Array::from_decimal64(arr), 0, 1);
        let result = broadcast_length_1_array(av, 3).unwrap();
        if let Array::NumericArray(NumericArray::Decimal64(a)) = result {
            assert_eq!(a.data.len(), 3);
            let mask = a.null_mask.as_ref().unwrap();
            for i in 0..3 {
                assert!(!mask.get(i), "element {i} should be null");
            }
        } else {
            panic!("Expected Decimal64 array");
        }
    }

    #[cfg(feature = "decimal")]
    #[test]
    fn test_maybe_broadcast_decimal_scalar_array() {
        let scalar = Array::from_decimal32(DecimalArray::from_slice(&[100i32], 5, 1));
        let target = Array::from_decimal32(DecimalArray::from_slice(&[1, 2, 3], 5, 1));
        let (lhs, rhs) = maybe_broadcast_scalar_array(
            ArrayV::new(scalar, 0, 1),
            ArrayV::new(target, 0, 3),
        )
        .unwrap();
        assert_eq!(lhs.len(), 3);
        assert_eq!(rhs.len(), 3);
        if let Array::NumericArray(NumericArray::Decimal32(a)) = &lhs.array {
            assert_eq!(a.data.as_slice(), &[100, 100, 100]);
            assert_eq!(a.precision, 5);
            assert_eq!(a.scale, 1);
        } else {
            panic!("Expected broadcast Decimal32 array");
        }
    }
}
