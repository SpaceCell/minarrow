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

//! # **TimeUnits Module** - *Arrow Datetime Units*
//!
//! Defines time and interval units used by temporal arrays in Minarrow.
//!  
//! `TimeUnit` standardises second, millisecond, microsecond, nanosecond, and day resolution
//! across `DatetimeArray32` and `DatetimeArray64`.  
//! `IntervalUnit` specifies year–month, day–time, or month–day–nanosecond intervals
//! for representing durations or periods in `DatetimeArray` values.
//!  
//! Both integrate with Apache Arrow’s native types for FFI compatibility.

use std::fmt::{Display, Formatter, Result as FmtResult};
use std::str::FromStr;

use crate::enums::error::MinarrowError;

/// # TimeUnit
///
/// Unified time unit enumeration.
///
/// ## Purpose
/// - Combines time units for both `DatetimeArray32` and `DatetimeArray64`.
/// - Confirm time since epoch units, or a raw duration value *(depending on the `ArrowType`
/// that's attached to `Field` during `FieldArray` construction)*.
/// - Avoids proliferating variants that require explicit handling throughout match statements.
///
/// ## Behaviour
/// - Unit values are stored on the `DatetimeArray`, enabling variant-specific logic.
/// - When transmitted over FFI, an `Apache Arrow`- produces compatible native format.
#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug, Default)]
pub enum TimeUnit {
    /// Seconds for Apache Arrow `Time32` and `Time64` units.
    Seconds,
    /// Milliseconds for Apache Arrow `Time32` and `Time64` units.
    Milliseconds,
    /// Microseconds for Apache Arrow `Time32` and `Time64` units.
    Microseconds,
    /// Nanoseconds for Apache Arrow `Time32` and `Time64` units.
    Nanoseconds,
    /// Default = days unspecified
    ///
    /// Apache Arrow's `Date32` and `Date64` types use days implicitly.
    #[default]
    Days,
}

/// # IntervalUnit
///
/// Inner Arrow discriminant for representing interval types
///
/// ## Usage
/// Attach via `ArrowType` to `Field` when your `DatetimeArray<T>`
/// T-integer represents an interval, rather than an epoch value.
/// Then, it will materialise as an `Interval` *Apache Arrow* type
/// when sent over FFI.
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum IntervalUnit {
    YearMonth,
    DaysTime,
    MonthDaysNs,
}

impl Display for TimeUnit {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            TimeUnit::Seconds => f.write_str("Seconds"),
            TimeUnit::Milliseconds => f.write_str("Milliseconds"),
            TimeUnit::Microseconds => f.write_str("Microseconds"),
            TimeUnit::Nanoseconds => f.write_str("Nanoseconds"),
            TimeUnit::Days => f.write_str("Days"),
        }
    }
}

impl Display for IntervalUnit {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            IntervalUnit::YearMonth => f.write_str("YearMonth"),
            IntervalUnit::DaysTime => f.write_str("DaysTime"),
            IntervalUnit::MonthDaysNs => f.write_str("MonthDaysNs"),
        }
    }
}

/// # TimePeriod
///
/// Calendar granularity for internal datetime mechanics such as truncation and
/// flooring.
///
/// ## Role
/// `TimePeriod` names the granularity a datetime is floored to, for example flooring
/// to the start of the month. It drives the internal date arithmetic in
/// `DatetimeArray` and is not part of the Apache Arrow type system.
///
/// For stored resolution and FFI, use [`TimeUnit`] instead - that is the Apache Arrow
/// and FFI-compatible unit carried on `Field` and `ArrowType`. `TimePeriod` is a
/// distinct concept: it names a calendar step, not the resolution of a stored value.
///
/// ## String form
/// Each variant has a canonical lower-case label (`year`, `month`, `week`, `day`,
/// `hour`, `minute`, `second`, `millisecond`, `microsecond`). A `TimePeriod` converts
/// to and from that label, so a call site can pass either the variant or its string.
/// Parse an untrusted string with `str::parse` (via [`FromStr`]) to get a `Result` -
/// the `From<&str>` path panics on an unknown label and is for known-good literals.
#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub enum TimePeriod {
    Year,
    Month,
    Week,
    Day,
    Hour,
    Minute,
    Second,
    Millisecond,
    Microsecond,
}

impl TimePeriod {
    /// The canonical lower-case label for this period.
    pub const fn as_str(&self) -> &'static str {
        match self {
            TimePeriod::Year => "year",
            TimePeriod::Month => "month",
            TimePeriod::Week => "week",
            TimePeriod::Day => "day",
            TimePeriod::Hour => "hour",
            TimePeriod::Minute => "minute",
            TimePeriod::Second => "second",
            TimePeriod::Millisecond => "millisecond",
            TimePeriod::Microsecond => "microsecond",
        }
    }
}

impl Display for TimePeriod {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str(self.as_str())
    }
}

impl FromStr for TimePeriod {
    type Err = MinarrowError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "year" => Ok(TimePeriod::Year),
            "month" => Ok(TimePeriod::Month),
            "week" => Ok(TimePeriod::Week),
            "day" => Ok(TimePeriod::Day),
            "hour" => Ok(TimePeriod::Hour),
            "minute" => Ok(TimePeriod::Minute),
            "second" => Ok(TimePeriod::Second),
            "millisecond" => Ok(TimePeriod::Millisecond),
            "microsecond" => Ok(TimePeriod::Microsecond),
            other => Err(MinarrowError::TypeError {
                from: "String",
                to: "TimePeriod",
                message: Some(format!("Unknown time period: {}", other)),
            }),
        }
    }
}

impl From<&str> for TimePeriod {
    fn from(value: &str) -> Self {
        value.parse().unwrap_or_else(|e| panic!("{}", e))
    }
}

#[cfg(test)]
mod time_period_tests {
    use super::TimePeriod;

    #[test]
    fn label_round_trips_through_string() {
        for period in [
            TimePeriod::Year,
            TimePeriod::Month,
            TimePeriod::Week,
            TimePeriod::Day,
            TimePeriod::Hour,
            TimePeriod::Minute,
            TimePeriod::Second,
            TimePeriod::Millisecond,
            TimePeriod::Microsecond,
        ] {
            assert_eq!(period.as_str().parse::<TimePeriod>().unwrap(), period);
            assert_eq!(period.to_string(), period.as_str());
        }
    }

    #[test]
    fn from_str_accepts_known_label() {
        assert_eq!("day".parse::<TimePeriod>().unwrap(), TimePeriod::Day);
        assert_eq!(TimePeriod::from("week"), TimePeriod::Week);
    }

    #[test]
    fn parse_rejects_unknown_label() {
        assert!("fortnight".parse::<TimePeriod>().is_err());
    }

    #[test]
    #[should_panic]
    fn from_panics_on_unknown_label() {
        let _ = TimePeriod::from("fortnight");
    }
}
