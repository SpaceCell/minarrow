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

//! # **ArrowDType Module** - *Arrow type tagging for self-documenting data*
//!
//! Unified Minarrow representations of supported *Apache Arrow* data types.
//!
//! ## Overview
//! - Covers integer, floating-point, boolean, string, dictionary-encoded, and optional temporal types  
//!   (date, time, duration, timestamp, interval).
//! - Each Minarrow array type implements `arrow_type()` to return its matching `ArrowType`.
//! - Enables consistent Arrow FFI compatibility without requiring the full Arrow type system.
//!
//! ## CategoricalIndexType
//! - Specifies the integer size of dictionary keys for categorical arrays.
//! - Supports multiple unsigned integer widths depending on feature flags.
//!
//! ## Display
//! - Human-readable type names are produced for all variants.
//! - Temporal types include their units in the rendered output.
//!
//! ## Interoperability
//! - Implements a focused subset of the public Arrow specification.
//! - Maintains compatibility while keeping Minarrow minimal.
//!
//! ## Copyright Notice
//! - The `Minarrow` crate is not affiliated with the `Apache Arrow` project.
//! - The term `Apache Arrow` is a trademark of the *Apache Software Foundation*.
//! - The term `Arrow` is used here under fair use to implement the public FFI compatibility standard,  
//!   in accordance with the official guidance: <https://www.apache.org/foundation/marks/>.
//!
//! See `./LICENSE` for more information.

use std::any::TypeId;
use std::fmt::{Display, Formatter, Result as FmtResult};

#[cfg(feature = "datetime")]
use crate::DatetimeArray;
#[cfg(feature = "datetime")]
use crate::enums::time_units::{IntervalUnit, TimeUnit};
use crate::{BooleanArray, CategoricalArray, Float, FloatArray, Integer, StringArray};

/// # ArrowType
///
/// Unified representation of supported *Apache Arrow* data types in Minarrow.
///
/// ## Purpose
/// - Encodes the physical type and, for temporal variants, associated unit information for all supported Minarrow arrays.
/// - Provides a single discriminant used across the crate for schema definitions, type matching, and Arrow FFI export.
/// - Implements a focused subset of the official Arrow type specification:  
///   <https://arrow.apache.org/docs/python/api/datatypes.html>.
///
/// ## Coverage
/// - **Core primitives**: integer, floating-point, boolean.
/// - **Strings**: UTF-8 (`String`) and optionally large UTF-8 (`LargeString`).
/// - **Dictionary-encoded strings**: via `Dictionary(CategoricalIndexType)`.
/// - **Optional temporal types**: `date`, `time`, `duration`, `timestamp`, and `interval` with explicit units.
/// - **`Null`**: placeholder or metadata-only fields.
///
/// ## Interoperability
/// - Directly compatible with the Apache Arrow C Data Interface type descriptors.
/// - Preserves type and temporal unit information when arrays are transmitted over FFI.
/// - Simplifies Minarrow’s type system *(e.g., one `DatetimeArray` type)* while tagging `ArrowType` on `Field` for ecosystem compatibility.
///
/// ## Notes
/// - For `DatetimeArray` types, `ArrowType` reflects only the physical encoding.  
///   Logical distinctions (e.g., interpreting a `Date64` as a timestamp vs. a duration) are stored in `Field` metadata.
/// - Dictionary key widths are defined by the associated `CategoricalIndexType`.
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum ArrowType {
    Null,
    Boolean,
    #[cfg(feature = "extended_numeric_types")]
    Int8,
    #[cfg(feature = "extended_numeric_types")]
    Int16,
    Int32,
    Int64,
    #[cfg(feature = "extended_numeric_types")]
    UInt8,
    #[cfg(feature = "extended_numeric_types")]
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    #[cfg(feature = "datetime")]
    Date32,
    #[cfg(feature = "datetime")]
    Date64,
    #[cfg(feature = "datetime")]
    Time32(TimeUnit),
    #[cfg(feature = "datetime")]
    Time64(TimeUnit),
    #[cfg(feature = "datetime")]
    Duration32(TimeUnit),
    #[cfg(feature = "datetime")]
    Duration64(TimeUnit),
    #[cfg(feature = "datetime")]
    Timestamp(TimeUnit, Option<String>), // TimeUnit + optional timezone string (e.g., "UTC", "America/New_York")
    #[cfg(feature = "datetime")]
    Interval(IntervalUnit),
    String,
    #[cfg(feature = "large_string")]
    LargeString,
    Utf8View,

    // Integer size for the categorical dictionary key,
    // and therefore how much storage space for each entry there is,
    // on top of the base string collection.
    Dictionary(CategoricalIndexType),
}

impl ArrowType {
    /// Upcast target for a binary operation over a pair of input types.
    ///
    /// Returns the element type that carries the result of a binary
    /// operation across the two inputs, choosing the narrowest type that
    /// preserves the values of both. Symmetric in its arguments. Returns
    /// `None` for pairs with no defined upcast, i.e. unrelated families
    /// such as a date with a time-of-day, or non-numeric types.
    ///
    /// | Pair | Upcast |
    /// |---|---|
    /// | `T` x `T` | `T` |
    /// | signed x signed, unsigned x unsigned | the wider type |
    /// | signed x unsigned | narrowest signed type holding both ranges |
    /// | integer x float | the float type that preserves the integer range |
    /// | `Float32` x `Float64` | `Float64` |
    /// | `Date32` x `Date64` | `Date64` |
    /// | `Time32` / `Time64` | wider storage, finer `TimeUnit` |
    /// | `Duration32` / `Duration64` | wider storage, finer `TimeUnit` |
    /// | `Timestamp` x `Timestamp` (same timezone) | finer `TimeUnit` |
    /// | `Interval` x `Interval` | `MonthDaysNs` when units differ |
    ///
    /// `UInt64` mixed with a signed integer upcasts to `Int64`, which
    /// narrows the unsigned range: converting implementations validate that
    /// each value fits `Int64` and surface an error when one does not.
    /// 64-bit integers upcast into `Float64` follow the universal convention
    /// that values beyond 2^53 lose integer precision. Timestamps with
    /// differing timezones carry no common frame and return `None`.
    pub fn upcast(&self, other: &ArrowType) -> Option<ArrowType> {
        use ArrowType::*;

        /// Promotion axis of a numeric type: family plus bit width.
        /// Numeric pairs promote on these two properties alone, so the
        /// ten numeric types resolve through one set of rules.
        enum PromotionVariant {
            Signed(u32),
            Unsigned(u32),
            Float(u32),
        }
        /// Classifies a type onto its promotion axis. Exhaustive over
        /// every `ArrowType` variant so a new variant does not compile
        /// until it is classified.
        fn promotion_variant(t: &ArrowType) -> Option<PromotionVariant> {
            match t {
                #[cfg(feature = "extended_numeric_types")]
                ArrowType::Int8 => Some(PromotionVariant::Signed(8)),
                #[cfg(feature = "extended_numeric_types")]
                ArrowType::Int16 => Some(PromotionVariant::Signed(16)),
                ArrowType::Int32 => Some(PromotionVariant::Signed(32)),
                ArrowType::Int64 => Some(PromotionVariant::Signed(64)),
                #[cfg(feature = "extended_numeric_types")]
                ArrowType::UInt8 => Some(PromotionVariant::Unsigned(8)),
                #[cfg(feature = "extended_numeric_types")]
                ArrowType::UInt16 => Some(PromotionVariant::Unsigned(16)),
                ArrowType::UInt32 => Some(PromotionVariant::Unsigned(32)),
                ArrowType::UInt64 => Some(PromotionVariant::Unsigned(64)),
                ArrowType::Float32 => Some(PromotionVariant::Float(32)),
                ArrowType::Float64 => Some(PromotionVariant::Float(64)),
                ArrowType::Null | ArrowType::Boolean => None,
                ArrowType::String | ArrowType::Utf8View | ArrowType::Dictionary(_) => None,
                #[cfg(feature = "large_string")]
                ArrowType::LargeString => None,
                #[cfg(feature = "datetime")]
                ArrowType::Date32
                | ArrowType::Date64
                | ArrowType::Time32(_)
                | ArrowType::Time64(_)
                | ArrowType::Duration32(_)
                | ArrowType::Duration64(_)
                | ArrowType::Timestamp(_, _)
                | ArrowType::Interval(_) => None,
            }
        }
        fn signed(bits: u32) -> ArrowType {
            match bits {
                #[cfg(feature = "extended_numeric_types")]
                8 => ArrowType::Int8,
                #[cfg(feature = "extended_numeric_types")]
                16 => ArrowType::Int16,
                32 => ArrowType::Int32,
                _ => ArrowType::Int64,
            }
        }
        fn unsigned(bits: u32) -> ArrowType {
            match bits {
                #[cfg(feature = "extended_numeric_types")]
                8 => ArrowType::UInt8,
                #[cfg(feature = "extended_numeric_types")]
                16 => ArrowType::UInt16,
                32 => ArrowType::UInt32,
                _ => ArrowType::UInt64,
            }
        }

        if let (Some(a), Some(b)) = (promotion_variant(self), promotion_variant(other)) {
            use PromotionVariant::*;
            return Some(match (a, b) {
                (Float(x), Float(y)) => {
                    if x.max(y) == 32 {
                        Float32
                    } else {
                        Float64
                    }
                }
                // A float preserves an integer when the integer's width
                // fits the float's mantissa: 24 bits for Float32, with
                // 64-bit integers into Float64 by the 2^53 convention.
                (Float(f), Signed(w) | Unsigned(w)) | (Signed(w) | Unsigned(w), Float(f)) => {
                    if f == 64 || w >= 32 { Float64 } else { Float32 }
                }
                (Signed(x), Signed(y)) => signed(x.max(y)),
                (Unsigned(x), Unsigned(y)) => unsigned(x.max(y)),
                // The narrowest signed type holding both ranges: an
                // unsigned width doubles to cover its full range, capped
                // at 64 bits where conversion validates the values.
                (Signed(s), Unsigned(u)) | (Unsigned(u), Signed(s)) => {
                    signed(s.max((u * 2).min(64)))
                }
            });
        }

        #[cfg(feature = "datetime")]
        {
            use crate::enums::time_units::TimeUnit;
            fn finer(a: &TimeUnit, b: &TimeUnit) -> TimeUnit {
                fn rank(u: &TimeUnit) -> u8 {
                    match u {
                        TimeUnit::Days => 0,
                        TimeUnit::Seconds => 1,
                        TimeUnit::Milliseconds => 2,
                        TimeUnit::Microseconds => 3,
                        TimeUnit::Nanoseconds => 4,
                    }
                }
                if rank(a) >= rank(b) {
                    a.clone()
                } else {
                    b.clone()
                }
            }

            match (self, other) {
                (Date32, Date32) => return Some(Date32),
                (Date32, Date64) | (Date64, Date32) | (Date64, Date64) => return Some(Date64),
                (Time32(a), Time32(b)) => return Some(Time32(finer(a, b))),
                (Time64(a), Time64(b)) | (Time32(a), Time64(b)) | (Time64(a), Time32(b)) => {
                    return Some(Time64(finer(a, b)));
                }
                (Duration32(a), Duration32(b)) => return Some(Duration32(finer(a, b))),
                (Duration64(a), Duration64(b))
                | (Duration32(a), Duration64(b))
                | (Duration64(a), Duration32(b)) => return Some(Duration64(finer(a, b))),
                (Timestamp(a, tz_a), Timestamp(b, tz_b)) => {
                    return if tz_a == tz_b {
                        Some(Timestamp(finer(a, b), tz_a.clone()))
                    } else {
                        None
                    };
                }
                (Interval(a), Interval(b)) => {
                    return Some(if a == b {
                        Interval(a.clone())
                    } else {
                        Interval(IntervalUnit::MonthDaysNs)
                    });
                }
                _ => {}
            }
        }

        // Remaining pairs span unrelated families: temporal types crossed
        // with anything outside their own family, non-numeric types, and
        // numeric crossed with non-numeric. Exhaustive so a new variant
        // does not compile until it is placed.
        match (self, other) {
            #[cfg(feature = "datetime")]
            (
                Date32
                | Date64
                | Time32(_)
                | Time64(_)
                | Duration32(_)
                | Duration64(_)
                | Timestamp(_, _)
                | Interval(_),
                _,
            )
            | (
                _,
                Date32
                | Date64
                | Time32(_)
                | Time64(_)
                | Duration32(_)
                | Duration64(_)
                | Timestamp(_, _)
                | Interval(_),
            ) => None,
            (Null | Boolean | String | Utf8View | Dictionary(_), _)
            | (_, Null | Boolean | String | Utf8View | Dictionary(_)) => None,
            #[cfg(feature = "large_string")]
            (LargeString, _) | (_, LargeString) => None,
            #[cfg(feature = "extended_numeric_types")]
            (
                Int8 | Int16 | UInt8 | UInt16 | Int32 | Int64 | UInt32 | UInt64 | Float32 | Float64,
                Int8 | Int16 | UInt8 | UInt16 | Int32 | Int64 | UInt32 | UInt64 | Float32 | Float64,
            ) => unreachable!("numeric pairs resolve in the numeric rules above"),
            #[cfg(not(feature = "extended_numeric_types"))]
            (
                Int32 | Int64 | UInt32 | UInt64 | Float32 | Float64,
                Int32 | Int64 | UInt32 | UInt64 | Float32 | Float64,
            ) => unreachable!("numeric pairs resolve in the numeric rules above"),
        }
    }
}

/// # CategoricalIndexType
///
/// Specifies the unsigned integer width used for dictionary keys in categorical arrays.
///
/// ## Overview
/// - Determines the storage size of the key column that indexes into the categorical dictionary.
/// - Smaller widths reduce memory footprint for low-cardinality data.
/// - Larger widths enable more distinct categories without overflow.
/// - Variant availability depends on feature flags:
///   - `UInt8` requires `default_categorical_8` or `extended_categorical`.
///   - `UInt16` and `UInt64` require `extended_categorical`.
///   - `UInt32` is available unless `default_categorical_8` is enabled without `extended_categorical`.
///
/// ## Interoperability
/// - Maps directly to the integer index type in Apache Arrow's `DictionaryType`.
/// - Preserved when sending categorical arrays over the Arrow C Data Interface.

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum CategoricalIndexType {
    #[cfg(feature = "default_categorical_8")]
    UInt8,
    #[cfg(feature = "extended_categorical")]
    UInt16,
    #[cfg(any(
        not(feature = "default_categorical_8"),
        feature = "extended_categorical"
    ))]
    UInt32,
    #[cfg(feature = "extended_categorical")]
    UInt64,
}

// Design documentation: arrow_type()
//
// Whilst `arrow_type()` could be on a trait, the ergonomics of using one aren't great
// due to then needing to import the trait at every usage point, for one method.
// Additionally, for cases like `DateTime`, the user is required to select a type when
// preparing `Field` metadata, and thus it is misleading. For this reason, they are
// here on the main objects as regular methods, so that they are available for most
// individual cases, but uniform dispatch methods that then don't work for datetime
// exceptions are implicitly discouraged. Adding it to MaskedArray as a method is also not a great
// option as the above still applies but then customising it per type and variant would
// require extra type storage on the MaskedArray trait which is too much for this.
// The other option is to use our types, rather than Arrow's for `Field`s, but that complicates
// FFI, as it's much better that once that's written it's compatible, so we settle on the below,
// which means the experience is:
// - "Field::new("myfield", existing_arr.arrow_type(), false, None)",
// - "Field::new("key", ArrowType::Date32, false, None)" when working with dates.

impl BooleanArray<()> {
    /// The arrow type that backs this array
    pub fn arrow_type() -> ArrowType {
        ArrowType::Boolean
    }
}

impl<T: Integer> CategoricalArray<T> {
    /// The arrow type that backs this array
    pub fn arrow_type() -> ArrowType {
        let t = TypeId::of::<T>();
        #[cfg(feature = "default_categorical_8")]
        if t == TypeId::of::<u8>() {
            return ArrowType::Dictionary(CategoricalIndexType::UInt8);
        }
        #[cfg(feature = "extended_categorical")]
        if t == TypeId::of::<u16>() {
            return ArrowType::Dictionary(CategoricalIndexType::UInt16);
        }
        #[cfg(any(
            not(feature = "default_categorical_8"),
            feature = "extended_categorical"
        ))]
        if t == TypeId::of::<u32>() {
            return ArrowType::Dictionary(CategoricalIndexType::UInt32);
        }
        #[cfg(feature = "extended_categorical")]
        if t == TypeId::of::<u64>() {
            return ArrowType::Dictionary(CategoricalIndexType::UInt64);
        }
        unsafe { std::hint::unreachable_unchecked() }
    }
}

impl<T: Float> FloatArray<T> {
    /// The arrow type that backs this array
    pub fn arrow_type() -> ArrowType {
        let t = TypeId::of::<T>();
        if t == TypeId::of::<f32>() {
            ArrowType::Float32
        } else if t == TypeId::of::<f64>() {
            ArrowType::Float64
        } else {
            unsafe { std::hint::unreachable_unchecked() }
        }
    }
}

impl<T: Integer> StringArray<T> {
    /// The arrow type that backs this array
    pub fn arrow_type() -> ArrowType {
        let t = TypeId::of::<T>();
        if t == TypeId::of::<u32>() {
            return ArrowType::String;
        }
        #[cfg(feature = "large_string")]
        if t == TypeId::of::<u64>() {
            return ArrowType::LargeString;
        }
        unsafe { std::hint::unreachable_unchecked() }
    }
}

#[cfg(feature = "datetime")]
impl<T: Integer> DatetimeArray<T> {
    /// For DateTime, the logical type is undocumented until attached to the type with a `Field` via `Field::new`.
    /// At this stage, one can convert the array into a `FieldArray` which makes it immutable and hooks it into Arrow FFI-ready
    /// format. This helps enable reducing 8 separate logical *Arrow* types down to 1 `DateTimeArray` data structure,
    /// keeping *MinArrow* minimal whilst retaining a compatibility path.
    pub fn arrow_type() -> ArrowType {
        ArrowType::Null
    }
}

impl Display for ArrowType {
    /// Render the ArrowType as its variant name, including associated units where applicable.
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            ArrowType::Null => f.write_str("Null"),
            ArrowType::Boolean => f.write_str("Boolean"),

            #[cfg(feature = "extended_numeric_types")]
            ArrowType::Int8 => f.write_str("Int8"),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::Int16 => f.write_str("Int16"),
            ArrowType::Int32 => f.write_str("Int32"),
            ArrowType::Int64 => f.write_str("Int64"),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::UInt8 => f.write_str("UInt8"),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::UInt16 => f.write_str("UInt16"),
            ArrowType::UInt32 => f.write_str("UInt32"),
            ArrowType::UInt64 => f.write_str("UInt64"),

            ArrowType::Float32 => f.write_str("Float32"),
            ArrowType::Float64 => f.write_str("Float64"),

            #[cfg(feature = "datetime")]
            ArrowType::Date32 => f.write_str("Date32"),
            #[cfg(feature = "datetime")]
            ArrowType::Date64 => f.write_str("Date64"),

            #[cfg(feature = "datetime")]
            ArrowType::Time32(unit) => write!(f, "Time32({unit})"),
            #[cfg(feature = "datetime")]
            ArrowType::Time64(unit) => write!(f, "Time64({unit})"),
            #[cfg(feature = "datetime")]
            ArrowType::Duration32(unit) => write!(f, "Duration32({unit})"),
            #[cfg(feature = "datetime")]
            ArrowType::Duration64(unit) => write!(f, "Duration64({unit})"),
            #[cfg(feature = "datetime")]
            ArrowType::Timestamp(unit, tz) => {
                if let Some(tz_str) = tz {
                    write!(f, "Timestamp({unit}, {})", tz_str)
                } else {
                    write!(f, "Timestamp({unit})")
                }
            }
            #[cfg(feature = "datetime")]
            ArrowType::Interval(interval) => write!(f, "Interval({interval})"),

            ArrowType::String => f.write_str("String"),
            #[cfg(feature = "large_string")]
            ArrowType::LargeString => f.write_str("LargeString"),
            ArrowType::Utf8View => f.write_str("Utf8View"),

            ArrowType::Dictionary(key_type) => write!(f, "Dictionary({key_type})"),
        }
    }
}

impl Display for CategoricalIndexType {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            #[cfg(feature = "default_categorical_8")]
            CategoricalIndexType::UInt8 => f.write_str("UInt8"),
            #[cfg(feature = "extended_categorical")]
            CategoricalIndexType::UInt16 => f.write_str("UInt16"),
            #[cfg(any(
                not(feature = "default_categorical_8"),
                feature = "extended_categorical"
            ))]
            CategoricalIndexType::UInt32 => f.write_str("UInt32"),
            #[cfg(feature = "extended_categorical")]
            CategoricalIndexType::UInt64 => f.write_str("UInt64"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every numeric type available under the active feature set.
    fn numeric_types() -> Vec<ArrowType> {
        vec![
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::Int8,
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::Int16,
            ArrowType::Int32,
            ArrowType::Int64,
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::UInt8,
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::UInt16,
            ArrowType::UInt32,
            ArrowType::UInt64,
            ArrowType::Float32,
            ArrowType::Float64,
        ]
    }

    #[test]
    fn upcast_identity_over_numerics() {
        for t in numeric_types() {
            assert_eq!(t.upcast(&t), Some(t.clone()), "identity for {t:?}");
        }
    }

    #[test]
    fn upcast_is_symmetric() {
        for a in numeric_types() {
            for b in numeric_types() {
                assert_eq!(a.upcast(&b), b.upcast(&a), "symmetry for {a:?} x {b:?}");
            }
        }
    }

    #[test]
    fn upcast_integer_widening() {
        use ArrowType::*;
        assert_eq!(Int32.upcast(&Int64), Some(Int64));
        assert_eq!(UInt32.upcast(&UInt64), Some(UInt64));
    }

    #[test]
    fn upcast_signed_unsigned_takes_narrowest_signed() {
        use ArrowType::*;
        assert_eq!(Int32.upcast(&UInt32), Some(Int64));
        assert_eq!(Int64.upcast(&UInt32), Some(Int64));
        assert_eq!(Int32.upcast(&UInt64), Some(Int64));
        assert_eq!(Int64.upcast(&UInt64), Some(Int64));
    }

    #[test]
    fn upcast_integer_float_preserves_integer_range() {
        use ArrowType::*;
        assert_eq!(Int32.upcast(&Float32), Some(Float64));
        assert_eq!(Int64.upcast(&Float32), Some(Float64));
        assert_eq!(UInt64.upcast(&Float32), Some(Float64));
        assert_eq!(Int64.upcast(&Float64), Some(Float64));
        assert_eq!(Float32.upcast(&Float64), Some(Float64));
    }

    #[cfg(feature = "extended_numeric_types")]
    #[test]
    fn upcast_extended_integers() {
        use ArrowType::*;
        assert_eq!(Int8.upcast(&Int16), Some(Int16));
        assert_eq!(UInt8.upcast(&UInt16), Some(UInt16));
        assert_eq!(Int8.upcast(&UInt8), Some(Int16));
        assert_eq!(Int16.upcast(&UInt16), Some(Int32));
        assert_eq!(Int8.upcast(&Int64), Some(Int64));
        assert_eq!(UInt16.upcast(&Int32), Some(Int32));
        assert_eq!(UInt8.upcast(&Float32), Some(Float32));
        assert_eq!(Int16.upcast(&Float32), Some(Float32));
        assert_eq!(Int16.upcast(&Float64), Some(Float64));
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn upcast_temporal_families() {
        use crate::enums::time_units::TimeUnit;
        use ArrowType::*;
        assert_eq!(Date32.upcast(&Date64), Some(Date64));
        assert_eq!(Date32.upcast(&Date32), Some(Date32));
        assert_eq!(
            Time32(TimeUnit::Seconds).upcast(&Time64(TimeUnit::Nanoseconds)),
            Some(Time64(TimeUnit::Nanoseconds))
        );
        assert_eq!(
            Time32(TimeUnit::Seconds).upcast(&Time32(TimeUnit::Milliseconds)),
            Some(Time32(TimeUnit::Milliseconds))
        );
        assert_eq!(
            Duration32(TimeUnit::Seconds).upcast(&Duration64(TimeUnit::Microseconds)),
            Some(Duration64(TimeUnit::Microseconds))
        );
        assert_eq!(
            Timestamp(TimeUnit::Milliseconds, Some("UTC".into()))
                .upcast(&Timestamp(TimeUnit::Nanoseconds, Some("UTC".into()))),
            Some(Timestamp(TimeUnit::Nanoseconds, Some("UTC".into())))
        );
        assert_eq!(
            Timestamp(TimeUnit::Milliseconds, Some("UTC".into())).upcast(&Timestamp(
                TimeUnit::Milliseconds,
                Some("America/New_York".into())
            )),
            None
        );
        assert_eq!(
            Interval(IntervalUnit::YearMonth).upcast(&Interval(IntervalUnit::DaysTime)),
            Some(Interval(IntervalUnit::MonthDaysNs))
        );
        assert_eq!(Date32.upcast(&Time32(TimeUnit::Seconds)), None);
        assert_eq!(Date64.upcast(&Int64), None);
    }

    #[test]
    fn upcast_undefined_pairs() {
        use ArrowType::*;
        assert_eq!(Boolean.upcast(&Int32), None);
        assert_eq!(String.upcast(&Float64), None);
        assert_eq!(Null.upcast(&Null), None);
        assert_eq!(Boolean.upcast(&Boolean), None);
        assert_eq!(String.upcast(&String), None);
    }
}
