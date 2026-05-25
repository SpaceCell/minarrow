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

//! # **Conversions & Views** - *Most To/From Boilerplate Implements Here*
//!
//! Implementations that convert between Minarrow array types and wire them into the
//! unified [`Array`] enum, plus `View` impls where the `views` feature is enabled.
//!
//! ## What’s included
//! - **Numeric <-> Numeric**
//!   - *Widening* `From<&IntegerArray<S>> for IntegerArray<D>` and `From<&IntegerArray<S>> for FloatArray<D>`.
//!   - *Narrowing / signedness changes* via `TryFrom<&IntegerArray<S>> for IntegerArray<D>` with
//!     [`MinarrowError::Overflow`] on out-of-range values.
//! - **Float <-> Integer**
//!   - `TryFrom<&FloatArray<F>> for IntegerArray<I>` with strict checks (finite + exact truncation);
////!     returns [`MinarrowError::LossyCast`] if fidelity is lost.
//! - **Booleans <-> Primitives**
//!   - `From<&BooleanArray<u8>>` to integer/float (true->1/1.0, false->0/0.0).
//!   - `From<&IntegerArray<T>>` / `From<&FloatArray<T>>` to `BooleanArray<u8>` (non-zero -> true).
//! - **Numerics/Booleans -> Strings**
//!   - `From<&IntegerArray<T>>`, `From<&FloatArray<T>>`, and `From<&BooleanArray<u8>>` to
//!     `StringArray<u32>` (UTF-8), preserving null masks.
//! - **Strings <-> Categoricals**
//!   - `TryFrom<&StringArray<Off>> for CategoricalArray<Idx>` builds a dictionary with stable codes.
//!   - `TryFrom<&CategoricalArray<Idx>> for StringArray<Off>` materialises codes back to UTF-8.
//!   - Widening/narrowing categorical index conversions (`From`/`TryFrom`) with overflow checks.
//! - **String offset width changes**
//!   - `From<&StringArray<u32>> for StringArray<u64>` and `TryFrom<&StringArray<u64>> for StringArray<u32>`.
//! - **Datetime conversions** *(feature `datetime`)*
//!   - Integer view of datetimes and width changes between `DatetimeArray<i32>` and `DatetimeArray<i64>`.
//! - **Into [`Array`] enum**
//!   - `From<Arc<...>> for Array` and `From<...> for Array` for all core variants,
//!     using cheap `Arc` clones for zero-copy wrapping.
//! - **`View` trait impls** *(feature `views`)*
//!   - Provides `BufferT` for owned and `Arc` array variants so you can call `.view(...)` / `.view_tuple(...)`.
//!
//! ## Null masks & semantics
//! Unless noted, conversions **preserve the source null mask**. Errors are explicit:
//! overflows use [`MinarrowError::Overflow`]; inexact float->int casts use [`MinarrowError::LossyCast`].
//!
//! ## Feature gates
//! Some conversions are available only with `extended_numeric_types`, `extended_categorical`,
//! `large_string`, `datetime`, or `views`. Enable the features you need in `Cargo.toml`.

use std::collections::HashMap;
use std::convert::{From, TryFrom};
use std::marker::PhantomData;
use std::sync::Arc;

use crate::enums::error::MinarrowError;
#[cfg(feature = "views")]
use crate::traits::view::View;
use crate::{
    Array, Bitmask, BooleanArray, CategoricalArray, FloatArray, Integer, IntegerArray,
    NumericArray, StringArray, TextArray, Vec64,
};
use num_traits::FromPrimitive;

#[cfg(feature = "datetime")]
use crate::{DatetimeArray, TemporalArray};

#[cfg(feature = "scalar_type")]
use crate::Scalar;

// Integer <-> Float

macro_rules! int_to_float_from {
    ($src:ty, $dst:ty) => {
        impl From<&IntegerArray<$src>> for FloatArray<$dst> {
            fn from(src: &IntegerArray<$src>) -> Self {
                let data = src.data.iter().map(|&x| x as $dst).collect();
                FloatArray {
                    data,
                    null_mask: src.null_mask.clone(),
                }
            }
        }
    };
}

#[cfg(feature = "extended_numeric_types")]
int_to_float_from!(i8, f32);
#[cfg(feature = "extended_numeric_types")]
int_to_float_from!(i8, f64);
#[cfg(feature = "extended_numeric_types")]
int_to_float_from!(i16, f32);
#[cfg(feature = "extended_numeric_types")]
int_to_float_from!(i16, f64);
int_to_float_from!(i32, f32);
int_to_float_from!(i32, f64);
int_to_float_from!(i64, f32);
int_to_float_from!(i64, f64);
#[cfg(feature = "extended_numeric_types")]
int_to_float_from!(u8, f32);
#[cfg(feature = "extended_numeric_types")]
int_to_float_from!(u8, f64);
#[cfg(feature = "extended_numeric_types")]
int_to_float_from!(u16, f32);
#[cfg(feature = "extended_numeric_types")]
int_to_float_from!(u16, f64);
int_to_float_from!(u32, f32);
int_to_float_from!(u32, f64);
int_to_float_from!(u64, f32);
int_to_float_from!(u64, f64);

macro_rules! int_to_int_from {
    ($src:ty, $dst:ty) => {
        impl From<&IntegerArray<$src>> for IntegerArray<$dst> {
            fn from(src: &IntegerArray<$src>) -> Self {
                let data = src.data.iter().map(|&x| x as $dst).collect();
                IntegerArray {
                    data,
                    null_mask: src.null_mask.clone(),
                }
            }
        }
    };
}

int_to_int_from!(i32, i64);
int_to_int_from!(i32, u64);
int_to_int_from!(u32, u64);
int_to_int_from!(u32, i64);

#[cfg(feature = "extended_numeric_types")]
int_to_int_from!(i8, i16);
#[cfg(feature = "extended_numeric_types")]
int_to_int_from!(i8, i32);
#[cfg(feature = "extended_numeric_types")]
int_to_int_from!(i8, i64);
#[cfg(feature = "extended_numeric_types")]
int_to_int_from!(i8, u16);
#[cfg(feature = "extended_numeric_types")]
int_to_int_from!(i8, u32);
#[cfg(feature = "extended_numeric_types")]
int_to_int_from!(i8, u64);
#[cfg(feature = "extended_numeric_types")]
int_to_int_from!(i16, i32);
#[cfg(feature = "extended_numeric_types")]
int_to_int_from!(i16, i64);
#[cfg(feature = "extended_numeric_types")]
int_to_int_from!(i16, u32);
#[cfg(feature = "extended_numeric_types")]
int_to_int_from!(i16, u64);

#[cfg(feature = "extended_numeric_types")]
int_to_int_from!(u8, u16);
#[cfg(feature = "extended_numeric_types")]
int_to_int_from!(u8, u32);
#[cfg(feature = "extended_numeric_types")]
int_to_int_from!(u8, u64);
#[cfg(feature = "extended_numeric_types")]
int_to_int_from!(u8, i16);
#[cfg(feature = "extended_numeric_types")]
int_to_int_from!(u8, i32);
#[cfg(feature = "extended_numeric_types")]
int_to_int_from!(u8, i64);
#[cfg(feature = "extended_numeric_types")]
int_to_int_from!(u16, u32);
#[cfg(feature = "extended_numeric_types")]
int_to_int_from!(u16, u64);
#[cfg(feature = "extended_numeric_types")]
int_to_int_from!(u16, i32);
#[cfg(feature = "extended_numeric_types")]
int_to_int_from!(u16, i64);

// TryFrom impls for IntegerArray<T> -> IntegerArray<U> (narrowing/signedness)

macro_rules! int_to_int_tryfrom {
    ($src:ty, $dst:ty) => {
        impl TryFrom<&IntegerArray<$src>> for IntegerArray<$dst> {
            type Error = MinarrowError;
            fn try_from(src: &IntegerArray<$src>) -> Result<Self, Self::Error> {
                let mut data = Vec64::with_capacity(src.data.len());
                for &x in &src.data {
                    let v = <$dst>::try_from(x).map_err(|_| MinarrowError::Overflow {
                        value: x.to_string(),
                        target: stringify!($dst),
                    })?;
                    data.push(v);
                }
                Ok(IntegerArray {
                    data: data.into(),
                    null_mask: src.null_mask.clone(),
                })
            }
        }
    };
}

// All lossily/narrowing/signedness-changing combinations
int_to_int_tryfrom!(i64, u32);
int_to_int_tryfrom!(u64, i32);
int_to_int_tryfrom!(u64, u32);
int_to_int_tryfrom!(i64, i32);
int_to_int_tryfrom!(i64, u64);
int_to_int_tryfrom!(u32, i32);
int_to_int_tryfrom!(u64, i64);
int_to_int_tryfrom!(i32, u32);

#[cfg(feature = "extended_numeric_types")]
int_to_int_tryfrom!(i16, i8);
#[cfg(feature = "extended_numeric_types")]
int_to_int_tryfrom!(i32, i8);
#[cfg(feature = "extended_numeric_types")]
int_to_int_tryfrom!(i32, i16);
#[cfg(feature = "extended_numeric_types")]
int_to_int_tryfrom!(i64, i8);
#[cfg(feature = "extended_numeric_types")]
int_to_int_tryfrom!(i64, i16);
#[cfg(feature = "extended_numeric_types")]
int_to_int_tryfrom!(u16, u8);
#[cfg(feature = "extended_numeric_types")]
int_to_int_tryfrom!(u32, u8);
#[cfg(feature = "extended_numeric_types")]
int_to_int_tryfrom!(u32, u16);
#[cfg(feature = "extended_numeric_types")]
int_to_int_tryfrom!(u64, u8);
#[cfg(feature = "extended_numeric_types")]
int_to_int_tryfrom!(u64, u16);
#[cfg(feature = "extended_numeric_types")]
int_to_int_tryfrom!(i8, u8);
#[cfg(feature = "extended_numeric_types")]
int_to_int_tryfrom!(i16, u8);
#[cfg(feature = "extended_numeric_types")]
int_to_int_tryfrom!(i16, u16);
#[cfg(feature = "extended_numeric_types")]
int_to_int_tryfrom!(i32, u8);
#[cfg(feature = "extended_numeric_types")]
int_to_int_tryfrom!(i32, u16);
#[cfg(feature = "extended_numeric_types")]
int_to_int_tryfrom!(i64, u8);
#[cfg(feature = "extended_numeric_types")]
int_to_int_tryfrom!(i64, u16);
#[cfg(feature = "extended_numeric_types")]
int_to_int_tryfrom!(u8, i8);
#[cfg(feature = "extended_numeric_types")]
int_to_int_tryfrom!(u16, i8);
#[cfg(feature = "extended_numeric_types")]
int_to_int_tryfrom!(u16, i16);
#[cfg(feature = "extended_numeric_types")]
int_to_int_tryfrom!(u32, i8);
#[cfg(feature = "extended_numeric_types")]
int_to_int_tryfrom!(u32, i16);
#[cfg(feature = "extended_numeric_types")]
int_to_int_tryfrom!(u64, i8);
#[cfg(feature = "extended_numeric_types")]
int_to_int_tryfrom!(u64, i16);

macro_rules! float_to_float_from {
    ($src:ty, $dst:ty) => {
        impl From<&FloatArray<$src>> for FloatArray<$dst> {
            fn from(src: &FloatArray<$src>) -> Self {
                let data = src.data.iter().map(|&x| x as $dst).collect();
                FloatArray {
                    data,
                    null_mask: src.null_mask.clone(),
                }
            }
        }
    };
}
float_to_float_from!(f32, f32);
float_to_float_from!(f32, f64);
float_to_float_from!(f64, f32);
float_to_float_from!(f64, f64);

macro_rules! float_to_int_tryfrom {
    ($src:ty, $dst:ty) => {
        impl TryFrom<&FloatArray<$src>> for IntegerArray<$dst> {
            type Error = MinarrowError;
            fn try_from(src: &FloatArray<$src>) -> Result<Self, Self::Error> {
                let mut data = Vec64::with_capacity(src.data.len());
                for &v in src.data.iter() {
                    // Accept only values that are finite and within target bounds
                    if !v.is_finite() {
                        return Err(MinarrowError::LossyCast {
                            value: v.to_string(),
                            target: stringify!($dst),
                        });
                    }
                    let cast = v as $dst;
                    // Reverse cast and check for fidelity (handles over/underflow and truncation)
                    if cast as $src != v.trunc() {
                        return Err(MinarrowError::LossyCast {
                            value: v.to_string(),
                            target: stringify!($dst),
                        });
                    }
                    data.push(cast);
                }
                Ok(IntegerArray {
                    data: data.into(),
                    null_mask: src.null_mask.clone(),
                })
            }
        }
    };
}

float_to_int_tryfrom!(f32, i64);
float_to_int_tryfrom!(f64, i64);
float_to_int_tryfrom!(f32, i32);
float_to_int_tryfrom!(f64, i32);
float_to_int_tryfrom!(f32, u32);
float_to_int_tryfrom!(f64, u32);
float_to_int_tryfrom!(f32, u64);
float_to_int_tryfrom!(f64, u64);

// Macro to implement From<&BooleanArray<u8>> for IntegerArray<T> and FloatArray<T>
macro_rules! bool_to_primitive_from {
    ($($ity:ty => $one:expr, $zero:expr),* ; $($fty:ty => $fone:expr, $fzero:expr),*) => {
        $(
            impl From<&BooleanArray<u8>> for IntegerArray<$ity> {
                fn from(src: &BooleanArray<u8>) -> Self {
                    let mut data = Vec64::with_capacity(src.len);
                    for i in 0..src.len {
                        data.push(if unsafe { src.data.get_unchecked(i) } { $one } else { $zero });
                    }
                    IntegerArray { data: data.into(), null_mask: src.null_mask.clone() }
                }
            }
        )*
        $(
            impl From<&BooleanArray<u8>> for FloatArray<$fty> {
                fn from(src: &BooleanArray<u8>) -> Self {
                    let mut data = Vec64::with_capacity(src.len);
                    for i in 0..src.len {
                        data.push(if unsafe { src.data.get_unchecked(i) } { $fone } else { $fzero });
                    }
                    FloatArray { data: data.into(), null_mask: src.null_mask.clone() }
                }
            }
        )*
    };
}

// Usage
bool_to_primitive_from!(
    i32 => 1, 0,
    i64 => 1, 0,
    u32 => 1, 0,
    u64 => 1, 0;
    f32 => 1.0, 0.0,
    f64 => 1.0, 0.0
);

// IntegerArray<T> -> BooleanArray<u8>
macro_rules! int_to_bool_from {
    ($src:ty) => {
        impl From<&IntegerArray<$src>> for BooleanArray<u8> {
            fn from(src: &IntegerArray<$src>) -> Self {
                let mut data = Bitmask::with_capacity(src.data.len());
                for (i, &v) in src.data.iter().enumerate() {
                    // Any non-zero value is true
                    data.set(i, v != 0);
                }
                BooleanArray {
                    data,
                    // Null mask remains identical
                    null_mask: src.null_mask.clone(),
                    len: src.data.len(),
                    _phantom: PhantomData,
                }
            }
        }
    };
}

int_to_bool_from!(i32);
int_to_bool_from!(u32);
int_to_bool_from!(i64);
int_to_bool_from!(u64);

#[cfg(feature = "extended_numeric_types")]
int_to_bool_from!(i8);
#[cfg(feature = "extended_numeric_types")]
int_to_bool_from!(u8);
#[cfg(feature = "extended_numeric_types")]
int_to_bool_from!(i16);
#[cfg(feature = "extended_numeric_types")]
int_to_bool_from!(u16);

// FloatArray<T> -> BooleanArray<u8>
macro_rules! float_to_bool_from {
    ($src:ty) => {
        impl From<&FloatArray<$src>> for BooleanArray<u8> {
            fn from(src: &FloatArray<$src>) -> Self {
                let mut data = Bitmask::with_capacity(src.data.len());
                for (i, &v) in src.data.iter().enumerate() {
                    data.set(i, v != 0.0);
                }
                BooleanArray {
                    data,
                    null_mask: src.null_mask.clone(),
                    len: src.data.len(),
                    _phantom: PhantomData,
                }
            }
        }
    };
}

float_to_bool_from!(f32);
float_to_bool_from!(f64);

// Primitive to string
macro_rules! numeric_to_string {
    ($src:ty) => {
        impl From<&$src> for StringArray<u32> {
            fn from(src: &$src) -> Self {
                let mut data = Vec64::new();
                let mut offsets = Vec64::with_capacity(src.data.len() + 1);
                let mut offset = 0u32;
                offsets.push(offset);
                for v in src.data.iter() {
                    let s = v.to_string();
                    let bytes = s.as_bytes();
                    data.extend_from_slice(bytes);
                    offset += bytes.len() as u32;
                    offsets.push(offset);
                }
                StringArray {
                    offsets: offsets.into(),
                    data: data.into(),
                    null_mask: src.null_mask.clone(),
                }
            }
        }
    };
}

#[cfg(feature = "extended_numeric_types")]
numeric_to_string!(IntegerArray<i8>);
#[cfg(feature = "extended_numeric_types")]
numeric_to_string!(IntegerArray<u8>);
#[cfg(feature = "extended_numeric_types")]
numeric_to_string!(IntegerArray<i16>);
#[cfg(feature = "extended_numeric_types")]
numeric_to_string!(IntegerArray<u16>);
numeric_to_string!(IntegerArray<i32>);
numeric_to_string!(IntegerArray<u32>);
numeric_to_string!(IntegerArray<i64>);
numeric_to_string!(IntegerArray<u64>);
numeric_to_string!(FloatArray<f32>);
numeric_to_string!(FloatArray<f64>);

impl From<&BooleanArray<u8>> for StringArray<u32> {
    fn from(src: &BooleanArray<u8>) -> Self {
        let mut data = Vec64::new();
        let mut offsets = Vec64::with_capacity(src.len + 1);
        let mut offset = 0u32;
        offsets.push(offset);
        for i in 0..src.len {
            let s = if unsafe { src.data.get_unchecked(i) } {
                "1"
            } else {
                "0"
            };
            let bytes = s.as_bytes();
            data.extend_from_slice(bytes);
            offset += bytes.len() as u32;
            offsets.push(offset);
        }
        StringArray {
            offsets: offsets.into(),
            data: data.into(),
            null_mask: src.null_mask.clone(),
        }
    }
}

// Categorical <-> String

// ---------- String<Idx>  ->  Categorical<Idx> ----------
macro_rules! string_to_cat {
    ($off:ty, $idx:ty) => {
        impl TryFrom<&StringArray<$off>> for CategoricalArray<$idx> {
            type Error = MinarrowError;

            fn try_from(src: &StringArray<$off>) -> Result<Self, Self::Error> {
                let mut dict = HashMap::<&str, $idx>::new();
                let mut uniq = Vec64::new();
                let mut codes = Vec64::with_capacity(src.offsets.len().saturating_sub(1));

                for win in src.offsets.windows(2) {
                    let (start, end) = (win[0].to_usize(), win[1].to_usize());
                    let slice = &src.data[start..end];
                    let s = std::str::from_utf8(slice).map_err(|e| MinarrowError::TypeError {
                        from: "String",
                        to: "Categorical",
                        message: Some(e.to_string()),
                    })?;

                    let code = *dict.entry(s).or_insert_with(|| {
                        let next = uniq.len();
                        let idx_val: $idx = FromPrimitive::from_usize(next)
                            .ok_or_else(|| MinarrowError::Overflow {
                                value: next.to_string(),
                                target: stringify!($idx),
                            })
                            .unwrap(); // checked above
                        uniq.push(s.to_owned());
                        idx_val
                    });
                    codes.push(code);
                }

                Ok(CategoricalArray {
                    data: codes.into(),
                    #[cfg(not(feature = "shared_dict"))]
                    unique_values: uniq,
                    #[cfg(feature = "shared_dict")]
                    dictionary: $crate::Dictionary::from(uniq),
                    null_mask: src.null_mask.clone(),
                })
            }
        }
    };
}

#[cfg(feature = "default_categorical_8")]
string_to_cat!(u32, u8);
#[cfg(feature = "extended_categorical")]
string_to_cat!(u32, u16);
#[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
string_to_cat!(u32, u32);
#[cfg(feature = "extended_categorical")]
string_to_cat!(u32, u64);
#[cfg(feature = "default_categorical_8")]
#[cfg(feature = "large_string")]
string_to_cat!(u64, u8);
#[cfg(feature = "extended_categorical")]
#[cfg(feature = "large_string")]
string_to_cat!(u64, u16);
#[cfg(feature = "large_string")]
#[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
string_to_cat!(u64, u32);
#[cfg(feature = "extended_categorical")]
#[cfg(feature = "large_string")]
string_to_cat!(u64, u64);

macro_rules! cat_to_string {
    ($idx:ty, $off:ty) => {
        impl TryFrom<&CategoricalArray<$idx>> for StringArray<$off> {
            type Error = MinarrowError;

            fn try_from(src: &CategoricalArray<$idx>) -> Result<Self, Self::Error> {
                let mut data = Vec64::new();
                let mut offsets = Vec64::with_capacity(src.data.len() + 1);
                let mut pos: $off = <$off>::from(0u8); // starting offset = 0
                offsets.push(pos);

                for &code in &src.data {
                    let idx = code.to_usize();
                    let s = &src.unique_values()[idx];
                    let bytes = s.as_bytes();
                    data.extend_from_slice(bytes);

                    // checked add in native width
                    let added =
                        <$off>::try_from(bytes.len()).map_err(|_| MinarrowError::Overflow {
                            value: bytes.len().to_string(),
                            target: stringify!($off),
                        })?;
                    pos = pos.checked_add(added).ok_or(MinarrowError::Overflow {
                        value: added.to_string(),
                        target: stringify!($off),
                    })?;
                    offsets.push(pos);
                }

                Ok(StringArray {
                    offsets: offsets.into(),
                    data: data.into(),
                    null_mask: src.null_mask.clone(),
                })
            }
        }
    };
}

#[cfg(feature = "default_categorical_8")]
cat_to_string!(u8, u32);
#[cfg(feature = "extended_categorical")]
cat_to_string!(u16, u32);
#[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
cat_to_string!(u32, u32);
#[cfg(feature = "extended_categorical")]
cat_to_string!(u64, u32);
#[cfg(feature = "default_categorical_8")]
#[cfg(feature = "large_string")]
cat_to_string!(u8, u64);
#[cfg(feature = "large_string")]
#[cfg(feature = "extended_categorical")]
cat_to_string!(u16, u64);
#[cfg(feature = "large_string")]
#[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
cat_to_string!(u32, u64);
#[cfg(feature = "large_string")]
#[cfg(feature = "extended_categorical")]
cat_to_string!(u64, u64);

// =
// StringArray<T>  ⇄  StringArray<U>
// =

#[cfg(feature = "large_string")]
impl From<&StringArray<u32>> for StringArray<u64> {
    fn from(src: &StringArray<u32>) -> Self {
        let offsets = src
            .offsets
            .iter()
            .map(|&o| o as u64)
            .collect::<Vec64<u64>>();
        Self {
            offsets: offsets.into(),
            data: src.data.clone(),
            null_mask: src.null_mask.clone(),
        }
    }
}

#[cfg(feature = "large_string")]
impl TryFrom<&StringArray<u64>> for StringArray<u32> {
    type Error = MinarrowError;
    fn try_from(src: &StringArray<u64>) -> Result<Self, Self::Error> {
        let mut offsets = Vec64::with_capacity(src.offsets.len());
        for &o in &src.offsets {
            offsets.push(u32::try_from(o).map_err(|_| MinarrowError::Overflow {
                value: o.to_string(),
                target: "u32",
            })?);
        }
        Ok(Self {
            offsets: offsets.into(),
            data: src.data.clone(),
            null_mask: src.null_mask.clone(),
        })
    }
}

#[cfg(any(feature = "default_categorical_8", feature = "extended_categorical"))]
macro_rules! cat_to_cat_widen {
    ($src:ty, $dst:ty) => {
        impl From<&CategoricalArray<$src>> for CategoricalArray<$dst> {
            fn from(src: &CategoricalArray<$src>) -> Self {
                let data = src.data.iter().map(|&x| x as $dst).collect();
                let values: Vec<String> = src.unique_values().to_vec();
                CategoricalArray {
                    data,
                    #[cfg(not(feature = "shared_dict"))]
                    unique_values: $crate::Vec64::from(values),
                    #[cfg(feature = "shared_dict")]
                    dictionary: $crate::Dictionary::<$dst>::from(values),
                    null_mask: src.null_mask.clone(),
                }
            }
        }
    };
}

#[cfg(any(feature = "default_categorical_8", feature = "extended_categorical"))]
macro_rules! cat_to_cat_narrow {
    ($src:ty, $dst:ty) => {
        impl TryFrom<&CategoricalArray<$src>> for CategoricalArray<$dst> {
            type Error = MinarrowError;
            fn try_from(src: &CategoricalArray<$src>) -> Result<Self, Self::Error> {
                let mut data = Vec64::with_capacity(src.data.len());
                for &v in &src.data {
                    data.push(<$dst>::try_from(v).map_err(|_| MinarrowError::Overflow {
                        value: v.to_string(),
                        target: stringify!($dst),
                    })?);
                }
                let values: Vec<String> = src.unique_values().to_vec();
                Ok(CategoricalArray {
                    data: data.into(),
                    #[cfg(not(feature = "shared_dict"))]
                    unique_values: $crate::Vec64::from(values),
                    #[cfg(feature = "shared_dict")]
                    dictionary: $crate::Dictionary::<$dst>::from(values),
                    null_mask: src.null_mask.clone(),
                })
            }
        }
    };
}

#[cfg(feature = "default_categorical_8")]
cat_to_cat_widen!(u8, u16);
#[cfg(feature = "default_categorical_8")]
cat_to_cat_widen!(u8, u32);
#[cfg(feature = "default_categorical_8")]
cat_to_cat_widen!(u8, u64);
#[cfg(feature = "extended_categorical")]
cat_to_cat_widen!(u16, u32);
#[cfg(feature = "extended_categorical")]
cat_to_cat_widen!(u16, u64);
#[cfg(feature = "extended_categorical")]
cat_to_cat_widen!(u32, u64);
#[cfg(feature = "default_categorical_8")]
cat_to_cat_narrow!(u16, u8);
#[cfg(feature = "default_categorical_8")]
cat_to_cat_narrow!(u32, u8);
#[cfg(feature = "default_categorical_8")]
cat_to_cat_narrow!(u64, u8);
#[cfg(feature = "extended_categorical")]
cat_to_cat_narrow!(u32, u16);
#[cfg(feature = "extended_categorical")]
cat_to_cat_narrow!(u64, u16);
#[cfg(feature = "extended_categorical")]
cat_to_cat_narrow!(u64, u32);

// identity conversions (Arc-clone) for completeness
#[cfg(feature = "default_categorical_8")]
impl From<&CategoricalArray<u8>> for CategoricalArray<u8> {
    fn from(c: &CategoricalArray<u8>) -> Self {
        c.clone()
    }
}
#[cfg(feature = "extended_categorical")]
impl From<&CategoricalArray<u16>> for CategoricalArray<u16> {
    fn from(c: &CategoricalArray<u16>) -> Self {
        c.clone()
    }
}
#[cfg(feature = "extended_categorical")]
impl From<&CategoricalArray<u64>> for CategoricalArray<u64> {
    fn from(c: &CategoricalArray<u64>) -> Self {
        c.clone()
    }
}

// Datetime -> Integer
#[cfg(feature = "datetime")]
impl<T: Copy> From<&DatetimeArray<T>> for IntegerArray<T> {
    fn from(src: &DatetimeArray<T>) -> Self {
        IntegerArray {
            data: src.data.clone(),
            null_mask: src.null_mask.clone(),
        }
    }
}

// Datetime <-> Datetime
#[cfg(feature = "datetime")]
impl TryFrom<&DatetimeArray<i64>> for DatetimeArray<i32> {
    type Error = MinarrowError;
    fn try_from(src: &DatetimeArray<i64>) -> Result<Self, Self::Error> {
        let mut data = Vec64::with_capacity(src.data.len());
        for &v in &src.data {
            if v < i32::MIN as i64 || v > i32::MAX as i64 {
                return Err(MinarrowError::Overflow {
                    value: v.to_string(),
                    target: "i32",
                });
            }
            data.push(v as i32);
        }
        Ok(DatetimeArray {
            data: data.into(),
            null_mask: src.null_mask.clone(),
            time_unit: src.time_unit.clone(),
        })
    }
}

#[cfg(feature = "datetime")]
impl From<&DatetimeArray<i32>> for DatetimeArray<i64> {
    fn from(src: &DatetimeArray<i32>) -> Self {
        let data = src.data.iter().map(|&v| v as i64).collect();
        DatetimeArray {
            data,
            null_mask: src.null_mask.clone(),
            time_unit: src.time_unit.clone(),
        }
    }
}

// --------------------------------
//      From Arc<T> for Array
// --------------------------------

// --------- NumericArray Variants ---------

#[cfg(feature = "extended_numeric_types")]
impl From<Arc<IntegerArray<i8>>> for Array {
    fn from(a: Arc<IntegerArray<i8>>) -> Self {
        Array::NumericArray(NumericArray::Int8(a))
    }
}

#[cfg(feature = "views")]
#[cfg(feature = "extended_numeric_types")]
impl View for Arc<IntegerArray<i8>> {
    type BufferT = i8;
}

#[cfg(feature = "extended_numeric_types")]
impl From<Arc<IntegerArray<i16>>> for Array {
    fn from(a: Arc<IntegerArray<i16>>) -> Self {
        Array::NumericArray(NumericArray::Int16(a))
    }
}

#[cfg(feature = "views")]
#[cfg(feature = "extended_numeric_types")]
impl View for Arc<IntegerArray<i16>> {
    type BufferT = i16;
}

impl From<Arc<IntegerArray<i32>>> for Array {
    fn from(a: Arc<IntegerArray<i32>>) -> Self {
        Array::NumericArray(NumericArray::Int32(a))
    }
}

#[cfg(feature = "views")]
impl View for Arc<IntegerArray<i32>> {
    type BufferT = i32;
}

impl From<Arc<IntegerArray<i64>>> for Array {
    fn from(a: Arc<IntegerArray<i64>>) -> Self {
        Array::NumericArray(NumericArray::Int64(a))
    }
}

#[cfg(feature = "views")]
impl View for Arc<IntegerArray<i64>> {
    type BufferT = i64;
}

#[cfg(feature = "extended_numeric_types")]
impl From<Arc<IntegerArray<u8>>> for Array {
    fn from(a: Arc<IntegerArray<u8>>) -> Self {
        Array::NumericArray(NumericArray::UInt8(a))
    }
}

#[cfg(feature = "views")]
#[cfg(feature = "extended_numeric_types")]
impl View for Arc<IntegerArray<u8>> {
    type BufferT = u8;
}

#[cfg(feature = "extended_numeric_types")]
impl From<Arc<IntegerArray<u16>>> for Array {
    fn from(a: Arc<IntegerArray<u16>>) -> Self {
        Array::NumericArray(NumericArray::UInt16(a))
    }
}

#[cfg(feature = "views")]
#[cfg(feature = "extended_numeric_types")]
impl View for Arc<IntegerArray<u16>> {
    type BufferT = u16;
}

impl From<Arc<IntegerArray<u32>>> for Array {
    fn from(a: Arc<IntegerArray<u32>>) -> Self {
        Array::NumericArray(NumericArray::UInt32(a))
    }
}

#[cfg(feature = "views")]
impl View for Arc<IntegerArray<u32>> {
    type BufferT = u32;
}

impl From<Arc<IntegerArray<u64>>> for Array {
    fn from(a: Arc<IntegerArray<u64>>) -> Self {
        Array::NumericArray(NumericArray::UInt64(a))
    }
}

#[cfg(feature = "views")]
impl View for Arc<IntegerArray<u64>> {
    type BufferT = u64;
}

impl From<Arc<FloatArray<f32>>> for Array {
    fn from(a: Arc<FloatArray<f32>>) -> Self {
        Array::NumericArray(NumericArray::Float32(a))
    }
}

#[cfg(feature = "views")]
impl View for Arc<FloatArray<f32>> {
    type BufferT = f32;
}

impl From<Arc<FloatArray<f64>>> for Array {
    fn from(a: Arc<FloatArray<f64>>) -> Self {
        Array::NumericArray(NumericArray::Float64(a))
    }
}

#[cfg(feature = "views")]
impl View for Arc<FloatArray<f64>> {
    type BufferT = f64;
}

// --------- TemporalArray Variants ---------

#[cfg(feature = "datetime")]
impl From<Arc<DatetimeArray<i32>>> for Array {
    fn from(a: Arc<DatetimeArray<i32>>) -> Self {
        Array::TemporalArray(TemporalArray::Datetime32(a))
    }
}

#[cfg(feature = "views")]
#[cfg(feature = "datetime")]
impl View for Arc<DatetimeArray<i32>> {
    type BufferT = i32;
}

#[cfg(feature = "datetime")]
impl From<Arc<DatetimeArray<i64>>> for Array {
    fn from(a: Arc<DatetimeArray<i64>>) -> Self {
        Array::TemporalArray(TemporalArray::Datetime64(a))
    }
}

#[cfg(feature = "views")]
#[cfg(feature = "datetime")]
impl View for Arc<DatetimeArray<i64>> {
    type BufferT = i64;
}

// --------- TextArray Variants ---------

impl From<Arc<StringArray<u32>>> for Array {
    fn from(a: Arc<StringArray<u32>>) -> Self {
        Array::TextArray(TextArray::String32(a))
    }
}

#[cfg(feature = "views")]
impl View for Arc<StringArray<u32>> {
    type BufferT = u8;
}

#[cfg(feature = "large_string")]
impl From<Arc<StringArray<u64>>> for Array {
    fn from(a: Arc<StringArray<u64>>) -> Self {
        Array::TextArray(TextArray::String64(a))
    }
}

#[cfg(feature = "views")]
#[cfg(feature = "large_string")]
impl View for Arc<StringArray<u64>> {
    type BufferT = u8;
}

#[cfg(feature = "default_categorical_8")]
impl From<Arc<CategoricalArray<u8>>> for Array {
    fn from(a: Arc<CategoricalArray<u8>>) -> Self {
        Array::TextArray(TextArray::Categorical8(a))
    }
}

#[cfg(feature = "views")]
#[cfg(feature = "default_categorical_8")]
impl View for Arc<CategoricalArray<u8>> {
    type BufferT = u8;
}

#[cfg(feature = "extended_categorical")]
impl From<Arc<CategoricalArray<u16>>> for Array {
    fn from(a: Arc<CategoricalArray<u16>>) -> Self {
        Array::TextArray(TextArray::Categorical16(a))
    }
}

#[cfg(feature = "views")]
#[cfg(feature = "extended_categorical")]
impl View for Arc<CategoricalArray<u16>> {
    type BufferT = u16;
}

#[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
impl From<Arc<CategoricalArray<u32>>> for Array {
    fn from(a: Arc<CategoricalArray<u32>>) -> Self {
        Array::TextArray(TextArray::Categorical32(a))
    }
}

#[cfg(feature = "views")]
#[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
impl View for Arc<CategoricalArray<u32>> {
    type BufferT = u32;
}

#[cfg(feature = "extended_categorical")]
impl From<Arc<CategoricalArray<u64>>> for Array {
    fn from(a: Arc<CategoricalArray<u64>>) -> Self {
        Array::TextArray(TextArray::Categorical64(a))
    }
}

#[cfg(feature = "views")]
#[cfg(feature = "extended_categorical")]
impl View for Arc<CategoricalArray<u64>> {
    type BufferT = u64;
}

// --------- BooleanArray ---------

impl From<Arc<BooleanArray<()>>> for Array {
    fn from(a: Arc<BooleanArray<()>>) -> Self {
        Array::BooleanArray(a)
    }
}

#[cfg(feature = "views")]
impl View for Arc<BooleanArray<()>> {
    type BufferT = u8;
}

// --------------------------------
//      From T for Array
// --------------------------------

#[cfg(feature = "extended_numeric_types")]
impl From<IntegerArray<i8>> for Array {
    fn from(a: IntegerArray<i8>) -> Self {
        Array::NumericArray(NumericArray::Int8(a.into()))
    }
}

#[cfg(feature = "views")]
#[cfg(feature = "extended_numeric_types")]
impl View for IntegerArray<i8> {
    type BufferT = i8;
}

#[cfg(feature = "extended_numeric_types")]
impl From<IntegerArray<i16>> for Array {
    fn from(a: IntegerArray<i16>) -> Self {
        Array::NumericArray(NumericArray::Int16(a.into()))
    }
}

#[cfg(feature = "views")]
#[cfg(feature = "extended_numeric_types")]
impl View for IntegerArray<i16> {
    type BufferT = i16;
}

impl From<IntegerArray<i32>> for Array {
    fn from(a: IntegerArray<i32>) -> Self {
        Array::NumericArray(NumericArray::Int32(a.into()))
    }
}

#[cfg(feature = "views")]
impl View for IntegerArray<i32> {
    type BufferT = i32;
}

impl From<IntegerArray<i64>> for Array {
    fn from(a: IntegerArray<i64>) -> Self {
        Array::NumericArray(NumericArray::Int64(a.into()))
    }
}

#[cfg(feature = "views")]
impl View for IntegerArray<i64> {
    type BufferT = i64;
}

#[cfg(feature = "extended_numeric_types")]
impl From<IntegerArray<u8>> for Array {
    fn from(a: IntegerArray<u8>) -> Self {
        Array::NumericArray(NumericArray::UInt8(a.into()))
    }
}

#[cfg(feature = "views")]
#[cfg(feature = "extended_numeric_types")]
impl View for IntegerArray<u8> {
    type BufferT = u8;
}

#[cfg(feature = "extended_numeric_types")]
impl From<IntegerArray<u16>> for Array {
    fn from(a: IntegerArray<u16>) -> Self {
        Array::NumericArray(NumericArray::UInt16(a.into()))
    }
}

#[cfg(feature = "views")]
#[cfg(feature = "extended_numeric_types")]
impl View for IntegerArray<u16> {
    type BufferT = u16;
}

impl From<IntegerArray<u32>> for Array {
    fn from(a: IntegerArray<u32>) -> Self {
        Array::NumericArray(NumericArray::UInt32(a.into()))
    }
}

#[cfg(feature = "views")]
impl View for IntegerArray<u32> {
    type BufferT = u32;
}

impl From<IntegerArray<u64>> for Array {
    fn from(a: IntegerArray<u64>) -> Self {
        Array::NumericArray(NumericArray::UInt64(a.into()))
    }
}

#[cfg(feature = "views")]
impl View for IntegerArray<u64> {
    type BufferT = u64;
}

impl From<FloatArray<f32>> for Array {
    fn from(a: FloatArray<f32>) -> Self {
        Array::NumericArray(NumericArray::Float32(a.into()))
    }
}

#[cfg(feature = "views")]
impl View for FloatArray<f32> {
    type BufferT = f32;
}

impl From<FloatArray<f64>> for Array {
    fn from(a: FloatArray<f64>) -> Self {
        Array::NumericArray(NumericArray::Float64(a.into()))
    }
}

#[cfg(feature = "views")]
impl View for FloatArray<f64> {
    type BufferT = f64;
}

// --------- TemporalArray Variants ---------

#[cfg(feature = "datetime")]
impl From<DatetimeArray<i32>> for Array {
    fn from(a: DatetimeArray<i32>) -> Self {
        Array::TemporalArray(TemporalArray::Datetime32(a.into()))
    }
}

#[cfg(feature = "views")]
#[cfg(feature = "datetime")]
impl View for DatetimeArray<i32> {
    type BufferT = i32;
}

#[cfg(feature = "datetime")]
impl From<DatetimeArray<i64>> for Array {
    fn from(a: DatetimeArray<i64>) -> Self {
        Array::TemporalArray(TemporalArray::Datetime64(a.into()))
    }
}

#[cfg(feature = "views")]
#[cfg(feature = "datetime")]
impl View for DatetimeArray<i64> {
    type BufferT = i64;
}

// --------- TextArray Variants ---------

impl From<StringArray<u32>> for Array {
    fn from(a: StringArray<u32>) -> Self {
        Array::TextArray(TextArray::String32(a.into()))
    }
}

#[cfg(feature = "views")]
impl View for StringArray<u32> {
    type BufferT = u8;
}

#[cfg(feature = "large_string")]
impl From<StringArray<u64>> for Array {
    fn from(a: StringArray<u64>) -> Self {
        Array::TextArray(TextArray::String64(a.into()))
    }
}

#[cfg(feature = "views")]
#[cfg(feature = "large_string")]
impl View for StringArray<u64> {
    type BufferT = u8;
}

#[cfg(feature = "default_categorical_8")]
impl From<CategoricalArray<u8>> for Array {
    fn from(a: CategoricalArray<u8>) -> Self {
        Array::TextArray(TextArray::Categorical8(a.into()))
    }
}

#[cfg(feature = "views")]
#[cfg(feature = "default_categorical_8")]
impl View for CategoricalArray<u8> {
    type BufferT = u8;
}

#[cfg(feature = "extended_categorical")]
impl From<CategoricalArray<u16>> for Array {
    fn from(a: CategoricalArray<u16>) -> Self {
        Array::TextArray(TextArray::Categorical16(a.into()))
    }
}

#[cfg(feature = "views")]
#[cfg(feature = "extended_categorical")]
impl View for CategoricalArray<u16> {
    type BufferT = u16;
}

#[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
impl From<CategoricalArray<u32>> for Array {
    fn from(a: CategoricalArray<u32>) -> Self {
        Array::TextArray(TextArray::Categorical32(a.into()))
    }
}

#[cfg(feature = "views")]
#[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
impl View for CategoricalArray<u32> {
    type BufferT = u32;
}

#[cfg(feature = "extended_categorical")]
impl From<CategoricalArray<u64>> for Array {
    fn from(a: CategoricalArray<u64>) -> Self {
        Array::TextArray(TextArray::Categorical64(a.into()))
    }
}

#[cfg(feature = "views")]
#[cfg(feature = "extended_categorical")]
impl View for CategoricalArray<u64> {
    type BufferT = u64;
}

// --------- BooleanArray ---------

impl From<BooleanArray<()>> for Array {
    fn from(a: BooleanArray<()>) -> Self {
        Array::BooleanArray(a.into())
    }
}

#[cfg(feature = "views")]
impl View for BooleanArray<()> {
    type BufferT = u8;
}

// --------------------------------
//      From Scalar for Array
// --------------------------------

#[cfg(feature = "scalar_type")]
impl From<Scalar> for Array {
    fn from(scalar: Scalar) -> Self {
        use crate::Scalar::*;
        match scalar {
            Null => Array::from_int32(IntegerArray::from_slice(&[0i32])),
            Boolean(v) => Array::from_bool(BooleanArray::from_slice(&[v])),
            #[cfg(feature = "extended_numeric_types")]
            Int8(v) => Array::from_int8(IntegerArray::from_slice(&[v])),
            #[cfg(feature = "extended_numeric_types")]
            Int16(v) => Array::from_int16(IntegerArray::from_slice(&[v])),
            Int32(v) => Array::from_int32(IntegerArray::from_slice(&[v])),
            Int64(v) => Array::from_int64(IntegerArray::from_slice(&[v])),
            #[cfg(feature = "extended_numeric_types")]
            UInt8(v) => Array::from_uint8(IntegerArray::from_slice(&[v])),
            #[cfg(feature = "extended_numeric_types")]
            UInt16(v) => Array::from_uint16(IntegerArray::from_slice(&[v])),
            UInt32(v) => Array::from_uint32(IntegerArray::from_slice(&[v])),
            UInt64(v) => Array::from_uint64(IntegerArray::from_slice(&[v])),
            Float32(v) => Array::from_float32(FloatArray::from_slice(&[v])),
            Float64(v) => Array::from_float64(FloatArray::from_slice(&[v])),
            String32(v) => Array::from_string32(StringArray::from_slice(&[v.as_str()])),
            #[cfg(feature = "large_string")]
            String64(v) => Array::from_string64(StringArray::from_slice(&[v.as_str()])),
            #[cfg(feature = "datetime")]
            Datetime32(v) => Array::from_datetime_i32(DatetimeArray::from_slice(&[v], None)),
            #[cfg(feature = "datetime")]
            Datetime64(v) => Array::from_datetime_i64(DatetimeArray::from_slice(&[v], None)),
            #[cfg(feature = "datetime")]
            Interval => Array::from_int32(IntegerArray::from_slice(&[0i32])),
        }
    }
}
