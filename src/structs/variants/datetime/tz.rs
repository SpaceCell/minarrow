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
//! IANA identifiers, such as `Australia/Sydney`, use the IANA Time Zone
//! Database, including daylight saving and historical transitions.
//! Abbreviations refer to one offset throughout the year, and are listed in
//! this module.
//!
//! Some abbreviations are ambiguous. For example, `CST` refers to Central
//! Standard Time, China Standard Time and Cuba Standard Time. The most common
//! reading is used. Consider IANA identifiers or offsets such as `+10:00`
//! where an exact reading is required.
//!
//! This module is part of the `time_zones` feature, which compiles the IANA
//! Time Zone Database into the binary.

use std::cell::Cell;

use tzdb_data::find_tz;

const SECONDS_PER_DAY: i64 = 86_400;

/// Returns the fixed UTC offset of a time zone abbreviation in `±HH:MM` form.
///
/// Returns `None` for text that is not a listed abbreviation.
///
/// # Example
/// ```
/// use minarrow::structs::variants::datetime::tz::abbreviation_offset;
///
/// assert_eq!(abbreviation_offset("AEST"), Some("+10:00"));
/// assert_eq!(abbreviation_offset("AEDT"), Some("+11:00"));
/// assert_eq!(abbreviation_offset("Australia/Sydney"), None);
/// ```
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

/// Resolves the time zone text `tz` and returns its UTC offset lookup.
///
/// The lookup receives an instant in seconds since the Unix epoch and returns
/// the UTC offset in seconds at that instant. The time zone is resolved once
/// here, and the kernels call the lookup for each row.
///
/// ## Behaviour
/// - Abbreviations listed in `abbreviation_offset` use their fixed offset.
/// - IANA identifiers use the IANA Time Zone Database, including daylight
///   saving and historical transitions.
/// - `UTC`, `Z` and offsets such as `+05:00` and `+0530` use that offset.
/// - Any other text returns `None`.
///
/// ## Details
/// - The lookup keeps two ranges of UTC days that each have a single offset.
///   Instants inside those ranges return the offset without a database search.
/// - Days that contain a transition are searched for every instant.
pub(crate) fn utc_offset_seconds(tz: &str) -> Option<impl Fn(i64) -> i64> {
    let zone = match abbreviation_offset(tz) {
        Some(_) => None,
        None => find_tz(tz.as_bytes()),
    };
    let fixed = match zone {
        Some(_) => 0,
        None => offset_text_seconds(abbreviation_offset(tz).unwrap_or(tz))?,
    };
    // Cache of two ranges of UTC days that each have a single offset, stored as
    // (first day, last day, offset). An empty range has its first day after its
    // last day. Two ranges are kept because truncation and month arithmetic look
    // up offsets near the input and near the result.
    let ranges = Cell::new([(1i64, 0i64, 0i64); 2]);
    Some(move |unix_seconds: i64| match zone {
        Some(zone) => {
            let offset_at = |seconds: i64| {
                zone.find_local_time_type(seconds)
                    .map_or(0, |local| i64::from(local.ut_offset()))
            };
            let day = unix_seconds.div_euclid(SECONDS_PER_DAY);
            let mut found = ranges.get();
            for (first_day, last_day, offset) in found {
                if first_day <= day && day <= last_day {
                    return offset;
                }
            }
            // Transitions are more than a day apart, so equal offsets at the first
            // and last second of a day hold for the whole day.
            let day_start = day.saturating_mul(SECONDS_PER_DAY);
            let first = offset_at(day_start);
            let last = offset_at(day_start.saturating_add(SECONDS_PER_DAY - 1));
            if first != last {
                return offset_at(unix_seconds);
            }
            // Extend an adjacent range with the same offset. Otherwise replace
            // the older range.
            let adjoined = found.iter_mut().find(|(first_day, last_day, offset)| {
                first_day <= last_day
                    && *offset == first
                    && (day == last_day.saturating_add(1) || day.saturating_add(1) == *first_day)
            });
            match adjoined {
                Some(range) => {
                    range.0 = range.0.min(day);
                    range.1 = range.1.max(day);
                }
                None => found = [(day, day, first), found[0]],
            }
            ranges.set(found);
            first
        }
        None => fixed,
    })
}

/// Parses offset text such as `+05:00`, `-0800`, `UTC` and `Z` into seconds.
///
/// Returns `None` for any other text.
fn offset_text_seconds(text: &str) -> Option<i64> {
    let text = text.trim();

    // Handle UTC specially
    if text.eq_ignore_ascii_case("UTC") || text.eq_ignore_ascii_case("Z") {
        return Some(0);
    }

    // Parse offset strings like "+05:00", "-08:00", "+0530"
    if !text.starts_with('+') && !text.starts_with('-') {
        return None;
    }

    let (sign, rest) = text.split_at(1);
    let sign = if sign == "+" { 1 } else { -1 };

    // Try parsing HH:MM format
    if let Some((hours_str, mins_str)) = rest.split_once(':') {
        let hours: i8 = hours_str.parse().ok()?;
        let mins: i8 = mins_str.parse().ok()?;
        return Some(sign * (hours as i64 * 3600 + mins as i64 * 60));
    }

    // Try parsing HHMM format
    if rest.len() == 4 {
        let hours: i8 = rest[0..2].parse().ok()?;
        let mins: i8 = rest[2..4].parse().ok()?;
        return Some(sign * (hours as i64 * 3600 + mins as i64 * 60));
    }

    None
}

/// Search distance in seconds around a wall time when resolving its instants.
///
/// Every UTC offset in the IANA Time Zone Database is smaller than 16 hours,
/// and consecutive transitions of a time zone are more than 32 hours apart.
/// The search range around a wall time therefore holds at most one transition.
const WALL_TIME_WINDOW: i64 = 16 * 3_600;

/// Resolves the wall time `wall_seconds` of a time zone to instants.
///
/// `offset_seconds` is the lookup from `utc_offset_seconds`. `wall_seconds` is
/// the wall clock reading in seconds since the Unix epoch. Both returned
/// instants are in seconds since the Unix epoch.
///
/// Most wall times map to one instant, and both values are that instant. The
/// two values differ only for wall times that a transition skips.
///
/// ## Behaviour
/// - Wall times that occur once: both values are that instant.
/// - Wall times that occur twice, when clocks move back: both values are the
///   earlier instant.
/// - Wall times that are skipped, when clocks move forward: the first value is
///   the instant of the transition. The second value applies the offset from
///   before the transition, and is later than the transition.
pub(crate) fn wall_time_instants(
    offset_seconds: &impl Fn(i64) -> i64,
    wall_seconds: i64,
) -> (i64, i64) {
    let before = offset_seconds(wall_seconds.saturating_sub(WALL_TIME_WINDOW));
    let after = offset_seconds(wall_seconds.saturating_add(WALL_TIME_WINDOW));
    let with_before = wall_seconds.saturating_sub(before);
    let with_after = wall_seconds.saturating_sub(after);
    if before == after {
        return (with_before, with_before);
    }
    let before_holds = offset_seconds(with_before) == before;
    let after_holds = offset_seconds(with_after) == after;
    match (before_holds, after_holds) {
        (true, true) => {
            let first = with_before.min(with_after);
            (first, first)
        }
        (true, false) => (with_before, with_before),
        (false, true) => (with_after, with_after),
        (false, false) => {
            // The wall time is skipped. Find the transition, the first instant
            // in `(with_after, with_before]` that has the later offset.
            let (mut low, mut high) = (with_after, with_before);
            while high - low > 1 {
                let middle = low + (high - low) / 2;
                if offset_seconds(middle) == after {
                    high = middle;
                } else {
                    low = middle;
                }
            }
            (high, with_before)
        }
    }
}

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

    // `wall_time_instants` reads the offsets 16 hours either side of a wall time.
    #[test]
    fn test_database_offsets_and_transitions_fit_the_wall_time_window() {
        for name in tzdb_data::TZ_NAMES {
            let zone = find_tz(name.as_bytes()).unwrap();
            for local in zone.local_time_types() {
                assert!(
                    i64::from(local.ut_offset()).abs() < WALL_TIME_WINDOW,
                    "{name} has the offset {}",
                    local.ut_offset()
                );
            }
            for pair in zone.transitions().windows(2) {
                let apart = pair[1].unix_leap_time() - pair[0].unix_leap_time();
                assert!(apart > 2 * WALL_TIME_WINDOW, "{name} has transitions {apart} seconds apart");
            }
        }
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
