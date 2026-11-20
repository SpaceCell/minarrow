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

//! # **Time Zone Abbreviations** - *Fixed UTC Offsets*
//!
//! Maps time zone abbreviations, such as `AEST` and `EDT`, to their fixed UTC
//! offsets.
//!
//! IANA identifiers, such as `Australia/Sydney`, are resolved through the IANA
//! Time Zone Database, which includes daylight saving and historical
//! transitions. Abbreviations refer to one offset throughout the year, and are
//! listed in this module.
//!
//! Some abbreviations are ambiguous. For example, `CST` refers to Central
//! Standard Time, China Standard Time and Cuba Standard Time. The most common
//! reading is used. Consider IANA identifiers or offsets such as `+10:00`
//! where an exact reading is required.

/// Returns the fixed UTC offset of a time zone abbreviation in `±HH:MM` form.
///
/// Returns `None` for text that is not a listed abbreviation.
///
/// # Example
/// ```
/// # #[cfg(feature = "datetime_ops")]
/// # {
/// use minarrow::structs::variants::datetime::tz::abbreviation_offset;
///
/// assert_eq!(abbreviation_offset("AEST"), Some("+10:00"));
/// assert_eq!(abbreviation_offset("AEDT"), Some("+11:00"));
/// assert_eq!(abbreviation_offset("Australia/Sydney"), None);
/// # }
/// ```
#[cfg(feature = "datetime_ops")]
pub fn abbreviation_offset(abbreviation: &str) -> Option<&'static str> {
    let offset = match abbreviation {
        // Common Standard Time Zones
        "GMT" => "+00:00",
        "UTC" => "+00:00",
        "WET" => "+00:00",
        "WEST" => "+01:00",
        "CET" => "+01:00",
        "CEST" => "+02:00",
        "EET" => "+02:00",
        "EEST" => "+03:00",
        "MSK" => "+03:00",

        // Africa
        "WAT" => "+01:00",   // West Africa Time
        "CAT" => "+02:00",   // Central Africa Time
        "EAT" => "+03:00",   // East Africa Time
        "SAST" => "+02:00",  // South Africa Standard Time

        // Americas
        "NST" => "-03:30",   // Newfoundland Standard Time
        "NDT" => "-02:30",   // Newfoundland Daylight Time
        "AST" => "-04:00",   // Atlantic Standard Time
        "ADT" => "-03:00",   // Atlantic Daylight Time
        "EST" => "-05:00",   // Eastern Standard Time (North America)
        "EDT" => "-04:00",   // Eastern Daylight Time
        "CST" => "-06:00",   // Central Standard Time (North America, most common)
        "CDT" => "-05:00",   // Central Daylight Time
        "MST" => "-07:00",   // Mountain Standard Time
        "MDT" => "-06:00",   // Mountain Daylight Time
        "PST" => "-08:00",   // Pacific Standard Time (North America, most common)
        "PDT" => "-07:00",   // Pacific Daylight Time
        "AKST" => "-09:00",  // Alaska Standard Time
        "AKDT" => "-08:00",  // Alaska Daylight Time
        "HST" => "-10:00",   // Hawaii Standard Time
        "HDT" => "-09:00",   // Hawaii Daylight Time
        "SST" => "-11:00",   // Samoa Standard Time

        // Asia
        "IST" => "+05:30",   // India Standard Time (most common)
        "IDT" => "+03:00",   // Israel Daylight Time
        "PKT" => "+05:00",   // Pakistan Standard Time
        "WIB" => "+07:00",   // Western Indonesian Time
        "WITA" => "+08:00",  // Central Indonesian Time
        "WIT" => "+09:00",   // Eastern Indonesian Time
        "HKT" => "+08:00",   // Hong Kong Time
        "JST" => "+09:00",   // Japan Standard Time
        "KST" => "+09:00",   // Korea Standard Time

        // Australia/Pacific
        "AWST" => "+08:00",  // Australian Western Standard Time
        "ACST" => "+09:30",  // Australian Central Standard Time
        "ACDT" => "+10:30",  // Australian Central Daylight Time
        "AEST" => "+10:00",  // Australian Eastern Standard Time
        "AEDT" => "+11:00",  // Australian Eastern Daylight Time
        "NZST" => "+12:00",  // New Zealand Standard Time
        "NZDT" => "+13:00",  // New Zealand Daylight Time
        "ChST" => "+10:00",  // Chamorro Standard Time

        // Europe
        "BST" => "+01:00",   // British Summer Time
        _ => return None,
    };
    Some(offset)
}

#[cfg(all(feature = "datetime", feature = "datetime_ops"))]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abbreviation_offsets() {
        assert_eq!(abbreviation_offset("AEST"), Some("+10:00"));
        assert_eq!(abbreviation_offset("EST"), Some("-05:00"));
        assert_eq!(abbreviation_offset("GMT"), Some("+00:00"));
        assert_eq!(abbreviation_offset("UTC"), Some("+00:00"));
        assert_eq!(abbreviation_offset("PST"), Some("-08:00"));
        assert_eq!(abbreviation_offset("JST"), Some("+09:00"));
    }

    #[test]
    fn test_text_other_than_abbreviations_has_no_offset() {
        assert_eq!(abbreviation_offset("Australia/Sydney"), None);
        assert_eq!(abbreviation_offset("+10:00"), None);
        assert_eq!(abbreviation_offset("Invalid/Timezone"), None);
        assert_eq!(abbreviation_offset("INVALID"), None);
        assert_eq!(abbreviation_offset("25:00"), None);
    }
}
