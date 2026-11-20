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

//! # **Timezone Database** - *IANA Timezone Mappings*
//!
//! Provides compile-time static maps for looking up timezone offsets by:
//! - IANA timezone identifier (e.g., "Australia/Sydney")
//! - Timezone abbreviation (e.g., "AEST", "AEDT")
//! - Direct offset string (e.g., "+10:00")
//!
//! All 341 canonical timezones from the IANA timezone database are included.
//! However, these are based on static entries. If you have any specific
//! Datetime timezone requirements, you are responsible for checking and verifying
//! them.
//!
//! If you would like to update a timezone, please file a PR or issue.

#[cfg(feature = "datetime_ops")]
use phf::{Map, phf_map};

/// Timezone information including standard and DST offsets with abbreviations
#[cfg(feature = "datetime_ops")]
#[derive(Debug, Clone, Copy)]
pub struct TimezoneInfo {
    /// Standard time offset (e.g., "+10:00")
    pub std_offset: &'static str,
    /// Daylight saving time offset (e.g., "+11:00")
    pub dst_offset: &'static str,
    /// Standard time abbreviation (e.g., "AEST")
    pub std_abbr: &'static str,
    /// Daylight saving time abbreviation (e.g., "AEDT")
    pub dst_abbr: &'static str,
}

/// Static map of IANA timezone identifiers to TimezoneInfo
#[cfg(feature = "datetime_ops")]
pub static TZ_DATABASE: Map<&'static str, TimezoneInfo> = phf_map! {
    // Africa
    "Africa/Abidjan" => TimezoneInfo { std_offset: "+00:00", dst_offset: "+00:00", std_abbr: "GMT", dst_abbr: "" },
    "Africa/Algiers" => TimezoneInfo { std_offset: "+01:00", dst_offset: "+01:00", std_abbr: "CET", dst_abbr: "" },
    "Africa/Bissau" => TimezoneInfo { std_offset: "+00:00", dst_offset: "+00:00", std_abbr: "GMT", dst_abbr: "" },
    "Africa/Cairo" => TimezoneInfo { std_offset: "+02:00", dst_offset: "+03:00", std_abbr: "EET", dst_abbr: "EEST" },
    "Africa/Casablanca" => TimezoneInfo { std_offset: "+01:00", dst_offset: "+01:00", std_abbr: "+01", dst_abbr: "+01" },
    "Africa/Ceuta" => TimezoneInfo { std_offset: "+01:00", dst_offset: "+02:00", std_abbr: "CET", dst_abbr: "CEST" },
    "Africa/El_Aaiun" => TimezoneInfo { std_offset: "+01:00", dst_offset: "+01:00", std_abbr: "+01", dst_abbr: "+01" },
    "Africa/Johannesburg" => TimezoneInfo { std_offset: "+02:00", dst_offset: "+02:00", std_abbr: "SAST", dst_abbr: "" },
    "Africa/Juba" => TimezoneInfo { std_offset: "+02:00", dst_offset: "+02:00", std_abbr: "CAT", dst_abbr: "" },
    "Africa/Khartoum" => TimezoneInfo { std_offset: "+02:00", dst_offset: "+02:00", std_abbr: "CAT", dst_abbr: "" },
    "Africa/Lagos" => TimezoneInfo { std_offset: "+01:00", dst_offset: "+01:00", std_abbr: "WAT", dst_abbr: "" },
    "Africa/Maputo" => TimezoneInfo { std_offset: "+02:00", dst_offset: "+02:00", std_abbr: "CAT", dst_abbr: "" },
    "Africa/Monrovia" => TimezoneInfo { std_offset: "+00:00", dst_offset: "+00:00", std_abbr: "GMT", dst_abbr: "" },
    "Africa/Nairobi" => TimezoneInfo { std_offset: "+03:00", dst_offset: "+03:00", std_abbr: "EAT", dst_abbr: "" },
    "Africa/Ndjamena" => TimezoneInfo { std_offset: "+01:00", dst_offset: "+01:00", std_abbr: "WAT", dst_abbr: "" },
    "Africa/Sao_Tome" => TimezoneInfo { std_offset: "+00:00", dst_offset: "+00:00", std_abbr: "GMT", dst_abbr: "" },
    "Africa/Tripoli" => TimezoneInfo { std_offset: "+02:00", dst_offset: "+02:00", std_abbr: "EET", dst_abbr: "" },
    "Africa/Tunis" => TimezoneInfo { std_offset: "+01:00", dst_offset: "+01:00", std_abbr: "CET", dst_abbr: "" },
    "Africa/Windhoek" => TimezoneInfo { std_offset: "+02:00", dst_offset: "+02:00", std_abbr: "CAT", dst_abbr: "" },

    // America
    "America/Adak" => TimezoneInfo { std_offset: "-10:00", dst_offset: "-09:00", std_abbr: "HST", dst_abbr: "HDT" },
    "America/Anchorage" => TimezoneInfo { std_offset: "-09:00", dst_offset: "-08:00", std_abbr: "AKST", dst_abbr: "AKDT" },
    "America/Araguaina" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-03:00", std_abbr: "-03", dst_abbr: "" },
    "America/Argentina/Buenos_Aires" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-03:00", std_abbr: "-03", dst_abbr: "" },
    "America/Argentina/Catamarca" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-03:00", std_abbr: "-03", dst_abbr: "" },
    "America/Argentina/Cordoba" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-03:00", std_abbr: "-03", dst_abbr: "" },
    "America/Argentina/Jujuy" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-03:00", std_abbr: "-03", dst_abbr: "" },
    "America/Argentina/La_Rioja" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-03:00", std_abbr: "-03", dst_abbr: "" },
    "America/Argentina/Mendoza" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-03:00", std_abbr: "-03", dst_abbr: "" },
    "America/Argentina/Rio_Gallegos" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-03:00", std_abbr: "-03", dst_abbr: "" },
    "America/Argentina/Salta" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-03:00", std_abbr: "-03", dst_abbr: "" },
    "America/Argentina/San_Juan" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-03:00", std_abbr: "-03", dst_abbr: "" },
    "America/Argentina/San_Luis" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-03:00", std_abbr: "-03", dst_abbr: "" },
    "America/Argentina/Tucuman" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-03:00", std_abbr: "-03", dst_abbr: "" },
    "America/Argentina/Ushuaia" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-03:00", std_abbr: "-03", dst_abbr: "" },
    "America/Asuncion" => TimezoneInfo { std_offset: "-04:00", dst_offset: "-04:00", std_abbr: "-04", dst_abbr: "-04" },
    "America/Bahia" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-03:00", std_abbr: "-03", dst_abbr: "" },
    "America/Bahia_Banderas" => TimezoneInfo { std_offset: "-06:00", dst_offset: "-06:00", std_abbr: "CST", dst_abbr: "" },
    "America/Barbados" => TimezoneInfo { std_offset: "-04:00", dst_offset: "-04:00", std_abbr: "AST", dst_abbr: "" },
    "America/Belem" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-03:00", std_abbr: "-03", dst_abbr: "" },
    "America/Belize" => TimezoneInfo { std_offset: "-06:00", dst_offset: "-06:00", std_abbr: "CST", dst_abbr: "" },
    "America/Boa_Vista" => TimezoneInfo { std_offset: "-04:00", dst_offset: "-04:00", std_abbr: "-04", dst_abbr: "" },
    "America/Bogota" => TimezoneInfo { std_offset: "-05:00", dst_offset: "-05:00", std_abbr: "-05", dst_abbr: "" },
    "America/Boise" => TimezoneInfo { std_offset: "-07:00", dst_offset: "-06:00", std_abbr: "MST", dst_abbr: "MDT" },
    "America/Cambridge_Bay" => TimezoneInfo { std_offset: "-07:00", dst_offset: "-06:00", std_abbr: "MST", dst_abbr: "MDT" },
    "America/Campo_Grande" => TimezoneInfo { std_offset: "-04:00", dst_offset: "-04:00", std_abbr: "-04", dst_abbr: "" },
    "America/Cancun" => TimezoneInfo { std_offset: "-05:00", dst_offset: "-05:00", std_abbr: "EST", dst_abbr: "" },
    "America/Caracas" => TimezoneInfo { std_offset: "-04:00", dst_offset: "-04:00", std_abbr: "-04", dst_abbr: "" },
    "America/Cayenne" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-03:00", std_abbr: "-03", dst_abbr: "" },
    "America/Chicago" => TimezoneInfo { std_offset: "-06:00", dst_offset: "-05:00", std_abbr: "CST", dst_abbr: "CDT" },
    "America/Chihuahua" => TimezoneInfo { std_offset: "-06:00", dst_offset: "-06:00", std_abbr: "CST", dst_abbr: "" },
    "America/Ciudad_Juarez" => TimezoneInfo { std_offset: "-07:00", dst_offset: "-06:00", std_abbr: "MST", dst_abbr: "MDT" },
    "America/Costa_Rica" => TimezoneInfo { std_offset: "-06:00", dst_offset: "-06:00", std_abbr: "CST", dst_abbr: "" },
    "America/Coyhaique" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-03:00", std_abbr: "-03", dst_abbr: "" },
    "America/Cuiaba" => TimezoneInfo { std_offset: "-04:00", dst_offset: "-04:00", std_abbr: "-04", dst_abbr: "" },
    "America/Danmarkshavn" => TimezoneInfo { std_offset: "+00:00", dst_offset: "+00:00", std_abbr: "GMT", dst_abbr: "" },
    "America/Dawson" => TimezoneInfo { std_offset: "-07:00", dst_offset: "-07:00", std_abbr: "MST", dst_abbr: "" },
    "America/Dawson_Creek" => TimezoneInfo { std_offset: "-07:00", dst_offset: "-07:00", std_abbr: "MST", dst_abbr: "" },
    "America/Denver" => TimezoneInfo { std_offset: "-07:00", dst_offset: "-06:00", std_abbr: "MST", dst_abbr: "MDT" },
    "America/Detroit" => TimezoneInfo { std_offset: "-05:00", dst_offset: "-04:00", std_abbr: "EST", dst_abbr: "EDT" },
    "America/Edmonton" => TimezoneInfo { std_offset: "-07:00", dst_offset: "-06:00", std_abbr: "MST", dst_abbr: "MDT" },
    "America/Eirunepe" => TimezoneInfo { std_offset: "-05:00", dst_offset: "-05:00", std_abbr: "-05", dst_abbr: "" },
    "America/El_Salvador" => TimezoneInfo { std_offset: "-06:00", dst_offset: "-06:00", std_abbr: "CST", dst_abbr: "" },
    "America/Fort_Nelson" => TimezoneInfo { std_offset: "-07:00", dst_offset: "-07:00", std_abbr: "MST", dst_abbr: "" },
    "America/Fortaleza" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-03:00", std_abbr: "-03", dst_abbr: "" },
    "America/Glace_Bay" => TimezoneInfo { std_offset: "-04:00", dst_offset: "-03:00", std_abbr: "AST", dst_abbr: "ADT" },
    "America/Goose_Bay" => TimezoneInfo { std_offset: "-04:00", dst_offset: "-03:00", std_abbr: "AST", dst_abbr: "ADT" },
    "America/Grand_Turk" => TimezoneInfo { std_offset: "-05:00", dst_offset: "-04:00", std_abbr: "EST", dst_abbr: "EDT" },
    "America/Guatemala" => TimezoneInfo { std_offset: "-06:00", dst_offset: "-06:00", std_abbr: "CST", dst_abbr: "" },
    "America/Guayaquil" => TimezoneInfo { std_offset: "-05:00", dst_offset: "-05:00", std_abbr: "-05", dst_abbr: "" },
    "America/Guyana" => TimezoneInfo { std_offset: "-04:00", dst_offset: "-04:00", std_abbr: "-04", dst_abbr: "" },
    "America/Halifax" => TimezoneInfo { std_offset: "-04:00", dst_offset: "-03:00", std_abbr: "AST", dst_abbr: "ADT" },
    "America/Havana" => TimezoneInfo { std_offset: "-05:00", dst_offset: "-04:00", std_abbr: "CST", dst_abbr: "CDT" },
    "America/Hermosillo" => TimezoneInfo { std_offset: "-07:00", dst_offset: "-07:00", std_abbr: "MST", dst_abbr: "" },
    "America/Indiana/Indianapolis" => TimezoneInfo { std_offset: "-05:00", dst_offset: "-04:00", std_abbr: "EST", dst_abbr: "EDT" },
    "America/Indiana/Knox" => TimezoneInfo { std_offset: "-06:00", dst_offset: "-05:00", std_abbr: "CST", dst_abbr: "CDT" },
    "America/Indiana/Marengo" => TimezoneInfo { std_offset: "-05:00", dst_offset: "-04:00", std_abbr: "EST", dst_abbr: "EDT" },
    "America/Indiana/Petersburg" => TimezoneInfo { std_offset: "-05:00", dst_offset: "-04:00", std_abbr: "EST", dst_abbr: "EDT" },
    "America/Indiana/Tell_City" => TimezoneInfo { std_offset: "-06:00", dst_offset: "-05:00", std_abbr: "CST", dst_abbr: "CDT" },
    "America/Indiana/Vevay" => TimezoneInfo { std_offset: "-05:00", dst_offset: "-04:00", std_abbr: "EST", dst_abbr: "EDT" },
    "America/Indiana/Vincennes" => TimezoneInfo { std_offset: "-05:00", dst_offset: "-04:00", std_abbr: "EST", dst_abbr: "EDT" },
    "America/Indiana/Winamac" => TimezoneInfo { std_offset: "-05:00", dst_offset: "-04:00", std_abbr: "EST", dst_abbr: "EDT" },
    "America/Inuvik" => TimezoneInfo { std_offset: "-07:00", dst_offset: "-06:00", std_abbr: "MST", dst_abbr: "MDT" },
    "America/Iqaluit" => TimezoneInfo { std_offset: "-05:00", dst_offset: "-04:00", std_abbr: "EST", dst_abbr: "EDT" },
    "America/Jamaica" => TimezoneInfo { std_offset: "-05:00", dst_offset: "-05:00", std_abbr: "EST", dst_abbr: "" },
    "America/Juneau" => TimezoneInfo { std_offset: "-09:00", dst_offset: "-08:00", std_abbr: "AKST", dst_abbr: "AKDT" },
    "America/Kentucky/Louisville" => TimezoneInfo { std_offset: "-05:00", dst_offset: "-04:00", std_abbr: "EST", dst_abbr: "EDT" },
    "America/Kentucky/Monticello" => TimezoneInfo { std_offset: "-05:00", dst_offset: "-04:00", std_abbr: "EST", dst_abbr: "EDT" },
    "America/La_Paz" => TimezoneInfo { std_offset: "-04:00", dst_offset: "-04:00", std_abbr: "-04", dst_abbr: "" },
    "America/Lima" => TimezoneInfo { std_offset: "-05:00", dst_offset: "-05:00", std_abbr: "-05", dst_abbr: "" },
    "America/Los_Angeles" => TimezoneInfo { std_offset: "-08:00", dst_offset: "-07:00", std_abbr: "PST", dst_abbr: "PDT" },
    "America/Maceio" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-03:00", std_abbr: "-03", dst_abbr: "" },
    "America/Managua" => TimezoneInfo { std_offset: "-06:00", dst_offset: "-06:00", std_abbr: "CST", dst_abbr: "" },
    "America/Manaus" => TimezoneInfo { std_offset: "-04:00", dst_offset: "-04:00", std_abbr: "-04", dst_abbr: "" },
    "America/Martinique" => TimezoneInfo { std_offset: "-04:00", dst_offset: "-04:00", std_abbr: "AST", dst_abbr: "" },
    "America/Matamoros" => TimezoneInfo { std_offset: "-06:00", dst_offset: "-05:00", std_abbr: "CST", dst_abbr: "CDT" },
    "America/Mazatlan" => TimezoneInfo { std_offset: "-07:00", dst_offset: "-07:00", std_abbr: "MST", dst_abbr: "" },
    "America/Menominee" => TimezoneInfo { std_offset: "-06:00", dst_offset: "-05:00", std_abbr: "CST", dst_abbr: "CDT" },
    "America/Merida" => TimezoneInfo { std_offset: "-06:00", dst_offset: "-06:00", std_abbr: "CST", dst_abbr: "" },
    "America/Metlakatla" => TimezoneInfo { std_offset: "-09:00", dst_offset: "-08:00", std_abbr: "AKST", dst_abbr: "AKDT" },
    "America/Mexico_City" => TimezoneInfo { std_offset: "-06:00", dst_offset: "-06:00", std_abbr: "CST", dst_abbr: "" },
    "America/Miquelon" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-02:00", std_abbr: "-03", dst_abbr: "-02" },
    "America/Moncton" => TimezoneInfo { std_offset: "-04:00", dst_offset: "-03:00", std_abbr: "AST", dst_abbr: "ADT" },
    "America/Monterrey" => TimezoneInfo { std_offset: "-06:00", dst_offset: "-06:00", std_abbr: "CST", dst_abbr: "" },
    "America/Montevideo" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-03:00", std_abbr: "-03", dst_abbr: "" },
    "America/New_York" => TimezoneInfo { std_offset: "-05:00", dst_offset: "-04:00", std_abbr: "EST", dst_abbr: "EDT" },
    "America/Nome" => TimezoneInfo { std_offset: "-09:00", dst_offset: "-08:00", std_abbr: "AKST", dst_abbr: "AKDT" },
    "America/Noronha" => TimezoneInfo { std_offset: "-02:00", dst_offset: "-02:00", std_abbr: "-02", dst_abbr: "" },
    "America/North_Dakota/Beulah" => TimezoneInfo { std_offset: "-06:00", dst_offset: "-05:00", std_abbr: "CST", dst_abbr: "CDT" },
    "America/North_Dakota/Center" => TimezoneInfo { std_offset: "-06:00", dst_offset: "-05:00", std_abbr: "CST", dst_abbr: "CDT" },
    "America/North_Dakota/New_Salem" => TimezoneInfo { std_offset: "-06:00", dst_offset: "-05:00", std_abbr: "CST", dst_abbr: "CDT" },
    "America/Nuuk" => TimezoneInfo { std_offset: "-02:00", dst_offset: "-02:00", std_abbr: "-02", dst_abbr: "-02" },
    "America/Ojinaga" => TimezoneInfo { std_offset: "-06:00", dst_offset: "-05:00", std_abbr: "CST", dst_abbr: "CDT" },
    "America/Panama" => TimezoneInfo { std_offset: "-05:00", dst_offset: "-05:00", std_abbr: "EST", dst_abbr: "" },
    "America/Paramaribo" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-03:00", std_abbr: "-03", dst_abbr: "" },
    "America/Phoenix" => TimezoneInfo { std_offset: "-07:00", dst_offset: "-07:00", std_abbr: "MST", dst_abbr: "" },
    "America/Port-au-Prince" => TimezoneInfo { std_offset: "-05:00", dst_offset: "-04:00", std_abbr: "EST", dst_abbr: "EDT" },
    "America/Porto_Velho" => TimezoneInfo { std_offset: "-04:00", dst_offset: "-04:00", std_abbr: "-04", dst_abbr: "" },
    "America/Puerto_Rico" => TimezoneInfo { std_offset: "-04:00", dst_offset: "-04:00", std_abbr: "AST", dst_abbr: "" },
    "America/Punta_Arenas" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-03:00", std_abbr: "-03", dst_abbr: "" },
    "America/Rankin_Inlet" => TimezoneInfo { std_offset: "-06:00", dst_offset: "-05:00", std_abbr: "CST", dst_abbr: "CDT" },
    "America/Recife" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-03:00", std_abbr: "-03", dst_abbr: "" },
    "America/Regina" => TimezoneInfo { std_offset: "-06:00", dst_offset: "-06:00", std_abbr: "CST", dst_abbr: "" },
    "America/Resolute" => TimezoneInfo { std_offset: "-06:00", dst_offset: "-05:00", std_abbr: "CST", dst_abbr: "CDT" },
    "America/Rio_Branco" => TimezoneInfo { std_offset: "-05:00", dst_offset: "-05:00", std_abbr: "-05", dst_abbr: "" },
    "America/Santarem" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-03:00", std_abbr: "-03", dst_abbr: "" },
    "America/Santiago" => TimezoneInfo { std_offset: "-04:00", dst_offset: "-03:00", std_abbr: "-04", dst_abbr: "-03" },
    "America/Santo_Domingo" => TimezoneInfo { std_offset: "-04:00", dst_offset: "-04:00", std_abbr: "AST", dst_abbr: "" },
    "America/Sao_Paulo" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-03:00", std_abbr: "-03", dst_abbr: "" },
    "America/Scoresbysund" => TimezoneInfo { std_offset: "-02:00", dst_offset: "-01:00", std_abbr: "-02", dst_abbr: "-01" },
    "America/Sitka" => TimezoneInfo { std_offset: "-09:00", dst_offset: "-08:00", std_abbr: "AKST", dst_abbr: "AKDT" },
    "America/St_Johns" => TimezoneInfo { std_offset: "-03:30", dst_offset: "-02:30", std_abbr: "NST", dst_abbr: "NDT" },
    "America/Swift_Current" => TimezoneInfo { std_offset: "-06:00", dst_offset: "-06:00", std_abbr: "CST", dst_abbr: "" },
    "America/Tegucigalpa" => TimezoneInfo { std_offset: "-06:00", dst_offset: "-06:00", std_abbr: "CST", dst_abbr: "" },
    "America/Thule" => TimezoneInfo { std_offset: "-04:00", dst_offset: "-03:00", std_abbr: "AST", dst_abbr: "ADT" },
    "America/Tijuana" => TimezoneInfo { std_offset: "-08:00", dst_offset: "-07:00", std_abbr: "PST", dst_abbr: "PDT" },
    "America/Toronto" => TimezoneInfo { std_offset: "-05:00", dst_offset: "-04:00", std_abbr: "EST", dst_abbr: "EDT" },
    "America/Vancouver" => TimezoneInfo { std_offset: "-08:00", dst_offset: "-07:00", std_abbr: "PST", dst_abbr: "PDT" },
    "America/Whitehorse" => TimezoneInfo { std_offset: "-07:00", dst_offset: "-07:00", std_abbr: "MST", dst_abbr: "" },
    "America/Winnipeg" => TimezoneInfo { std_offset: "-06:00", dst_offset: "-05:00", std_abbr: "CST", dst_abbr: "CDT" },
    "America/Yakutat" => TimezoneInfo { std_offset: "-09:00", dst_offset: "-08:00", std_abbr: "AKST", dst_abbr: "AKDT" },

    // Antarctica
    "Antarctica/Casey" => TimezoneInfo { std_offset: "+08:00", dst_offset: "+08:00", std_abbr: "+08", dst_abbr: "" },
    "Antarctica/Davis" => TimezoneInfo { std_offset: "+07:00", dst_offset: "+07:00", std_abbr: "+07", dst_abbr: "" },
    "Antarctica/Macquarie" => TimezoneInfo { std_offset: "+10:00", dst_offset: "+10:00", std_abbr: "AEST", dst_abbr: "AEST" },
    "Antarctica/Mawson" => TimezoneInfo { std_offset: "+05:00", dst_offset: "+05:00", std_abbr: "+05", dst_abbr: "" },
    "Antarctica/Palmer" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-03:00", std_abbr: "-03", dst_abbr: "" },
    "Antarctica/Rothera" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-03:00", std_abbr: "-03", dst_abbr: "" },
    "Antarctica/Troll" => TimezoneInfo { std_offset: "+00:00", dst_offset: "+02:00", std_abbr: "+00", dst_abbr: "+02" },
    "Antarctica/Vostok" => TimezoneInfo { std_offset: "+05:00", dst_offset: "+05:00", std_abbr: "+05", dst_abbr: "" },

    // Asia
    "Asia/Almaty" => TimezoneInfo { std_offset: "+05:00", dst_offset: "+05:00", std_abbr: "+05", dst_abbr: "" },
    "Asia/Amman" => TimezoneInfo { std_offset: "+03:00", dst_offset: "+03:00", std_abbr: "+03", dst_abbr: "" },
    "Asia/Anadyr" => TimezoneInfo { std_offset: "+12:00", dst_offset: "+12:00", std_abbr: "+12", dst_abbr: "" },
    "Asia/Aqtau" => TimezoneInfo { std_offset: "+05:00", dst_offset: "+05:00", std_abbr: "+05", dst_abbr: "" },
    "Asia/Aqtobe" => TimezoneInfo { std_offset: "+05:00", dst_offset: "+05:00", std_abbr: "+05", dst_abbr: "" },
    "Asia/Ashgabat" => TimezoneInfo { std_offset: "+05:00", dst_offset: "+05:00", std_abbr: "+05", dst_abbr: "" },
    "Asia/Atyrau" => TimezoneInfo { std_offset: "+05:00", dst_offset: "+05:00", std_abbr: "+05", dst_abbr: "" },
    "Asia/Baghdad" => TimezoneInfo { std_offset: "+03:00", dst_offset: "+03:00", std_abbr: "+03", dst_abbr: "" },
    "Asia/Baku" => TimezoneInfo { std_offset: "+04:00", dst_offset: "+04:00", std_abbr: "+04", dst_abbr: "" },
    "Asia/Bangkok" => TimezoneInfo { std_offset: "+07:00", dst_offset: "+07:00", std_abbr: "+07", dst_abbr: "" },
    "Asia/Barnaul" => TimezoneInfo { std_offset: "+07:00", dst_offset: "+07:00", std_abbr: "+07", dst_abbr: "" },
    "Asia/Beirut" => TimezoneInfo { std_offset: "+02:00", dst_offset: "+03:00", std_abbr: "EET", dst_abbr: "EEST" },
    "Asia/Bishkek" => TimezoneInfo { std_offset: "+06:00", dst_offset: "+06:00", std_abbr: "+06", dst_abbr: "" },
    "Asia/Chita" => TimezoneInfo { std_offset: "+09:00", dst_offset: "+09:00", std_abbr: "+09", dst_abbr: "" },
    "Asia/Colombo" => TimezoneInfo { std_offset: "+05:30", dst_offset: "+05:30", std_abbr: "IST", dst_abbr: "" },
    "Asia/Damascus" => TimezoneInfo { std_offset: "+03:00", dst_offset: "+03:00", std_abbr: "+03", dst_abbr: "" },
    "Asia/Dhaka" => TimezoneInfo { std_offset: "+06:00", dst_offset: "+06:00", std_abbr: "+06", dst_abbr: "" },
    "Asia/Dili" => TimezoneInfo { std_offset: "+09:00", dst_offset: "+09:00", std_abbr: "+09", dst_abbr: "" },
    "Asia/Dubai" => TimezoneInfo { std_offset: "+04:00", dst_offset: "+04:00", std_abbr: "+04", dst_abbr: "" },
    "Asia/Dushanbe" => TimezoneInfo { std_offset: "+05:00", dst_offset: "+05:00", std_abbr: "+05", dst_abbr: "" },
    "Asia/Famagusta" => TimezoneInfo { std_offset: "+02:00", dst_offset: "+03:00", std_abbr: "EET", dst_abbr: "EEST" },
    "Asia/Gaza" => TimezoneInfo { std_offset: "+02:00", dst_offset: "+03:00", std_abbr: "EET", dst_abbr: "EEST" },
    "Asia/Hebron" => TimezoneInfo { std_offset: "+02:00", dst_offset: "+03:00", std_abbr: "EET", dst_abbr: "EEST" },
    "Asia/Ho_Chi_Minh" => TimezoneInfo { std_offset: "+07:00", dst_offset: "+07:00", std_abbr: "+07", dst_abbr: "" },
    "Asia/Hong_Kong" => TimezoneInfo { std_offset: "+08:00", dst_offset: "+08:00", std_abbr: "HKT", dst_abbr: "" },
    "Asia/Hovd" => TimezoneInfo { std_offset: "+07:00", dst_offset: "+07:00", std_abbr: "+07", dst_abbr: "" },
    "Asia/Irkutsk" => TimezoneInfo { std_offset: "+08:00", dst_offset: "+08:00", std_abbr: "+08", dst_abbr: "" },
    "Asia/Jakarta" => TimezoneInfo { std_offset: "+07:00", dst_offset: "+07:00", std_abbr: "WIB", dst_abbr: "" },
    "Asia/Jayapura" => TimezoneInfo { std_offset: "+09:00", dst_offset: "+09:00", std_abbr: "WIT", dst_abbr: "" },
    "Asia/Jerusalem" => TimezoneInfo { std_offset: "+02:00", dst_offset: "+03:00", std_abbr: "IST", dst_abbr: "IDT" },
    "Asia/Kabul" => TimezoneInfo { std_offset: "+04:30", dst_offset: "+04:30", std_abbr: "+0430", dst_abbr: "" },
    "Asia/Kamchatka" => TimezoneInfo { std_offset: "+12:00", dst_offset: "+12:00", std_abbr: "+12", dst_abbr: "" },
    "Asia/Karachi" => TimezoneInfo { std_offset: "+05:00", dst_offset: "+05:00", std_abbr: "PKT", dst_abbr: "" },
    "Asia/Kathmandu" => TimezoneInfo { std_offset: "+05:45", dst_offset: "+05:45", std_abbr: "NPT", dst_abbr: "" },
    "Asia/Khandyga" => TimezoneInfo { std_offset: "+09:00", dst_offset: "+09:00", std_abbr: "+09", dst_abbr: "" },
    "Asia/Kolkata" => TimezoneInfo { std_offset: "+05:30", dst_offset: "+05:30", std_abbr: "IST", dst_abbr: "" },
    "Asia/Krasnoyarsk" => TimezoneInfo { std_offset: "+07:00", dst_offset: "+07:00", std_abbr: "+07", dst_abbr: "" },
    "Asia/Kuching" => TimezoneInfo { std_offset: "+08:00", dst_offset: "+08:00", std_abbr: "+08", dst_abbr: "" },
    "Asia/Macau" => TimezoneInfo { std_offset: "+08:00", dst_offset: "+08:00", std_abbr: "CST", dst_abbr: "" },
    "Asia/Magadan" => TimezoneInfo { std_offset: "+11:00", dst_offset: "+11:00", std_abbr: "+11", dst_abbr: "" },
    "Asia/Makassar" => TimezoneInfo { std_offset: "+08:00", dst_offset: "+08:00", std_abbr: "WITA", dst_abbr: "" },
    "Asia/Manila" => TimezoneInfo { std_offset: "+08:00", dst_offset: "+08:00", std_abbr: "PST", dst_abbr: "" },
    "Asia/Nicosia" => TimezoneInfo { std_offset: "+02:00", dst_offset: "+03:00", std_abbr: "EET", dst_abbr: "EEST" },
    "Asia/Novokuznetsk" => TimezoneInfo { std_offset: "+07:00", dst_offset: "+07:00", std_abbr: "+07", dst_abbr: "" },
    "Asia/Novosibirsk" => TimezoneInfo { std_offset: "+07:00", dst_offset: "+07:00", std_abbr: "+07", dst_abbr: "" },
    "Asia/Omsk" => TimezoneInfo { std_offset: "+06:00", dst_offset: "+06:00", std_abbr: "+06", dst_abbr: "" },
    "Asia/Oral" => TimezoneInfo { std_offset: "+05:00", dst_offset: "+05:00", std_abbr: "+05", dst_abbr: "" },
    "Asia/Pontianak" => TimezoneInfo { std_offset: "+07:00", dst_offset: "+07:00", std_abbr: "WIB", dst_abbr: "" },
    "Asia/Pyongyang" => TimezoneInfo { std_offset: "+09:00", dst_offset: "+09:00", std_abbr: "KST", dst_abbr: "" },
    "Asia/Qatar" => TimezoneInfo { std_offset: "+03:00", dst_offset: "+03:00", std_abbr: "+03", dst_abbr: "" },
    "Asia/Qostanay" => TimezoneInfo { std_offset: "+05:00", dst_offset: "+05:00", std_abbr: "+05", dst_abbr: "" },
    "Asia/Qyzylorda" => TimezoneInfo { std_offset: "+05:00", dst_offset: "+05:00", std_abbr: "+05", dst_abbr: "" },
    "Asia/Riyadh" => TimezoneInfo { std_offset: "+03:00", dst_offset: "+03:00", std_abbr: "+03", dst_abbr: "" },
    "Asia/Sakhalin" => TimezoneInfo { std_offset: "+11:00", dst_offset: "+11:00", std_abbr: "+11", dst_abbr: "" },
    "Asia/Samarkand" => TimezoneInfo { std_offset: "+05:00", dst_offset: "+05:00", std_abbr: "+05", dst_abbr: "" },
    "Asia/Seoul" => TimezoneInfo { std_offset: "+09:00", dst_offset: "+09:00", std_abbr: "KST", dst_abbr: "" },
    "Asia/Shanghai" => TimezoneInfo { std_offset: "+08:00", dst_offset: "+08:00", std_abbr: "CST", dst_abbr: "" },
    "Asia/Singapore" => TimezoneInfo { std_offset: "+08:00", dst_offset: "+08:00", std_abbr: "+08", dst_abbr: "" },
    "Asia/Srednekolymsk" => TimezoneInfo { std_offset: "+11:00", dst_offset: "+11:00", std_abbr: "+11", dst_abbr: "" },
    "Asia/Taipei" => TimezoneInfo { std_offset: "+08:00", dst_offset: "+08:00", std_abbr: "CST", dst_abbr: "" },
    "Asia/Tashkent" => TimezoneInfo { std_offset: "+05:00", dst_offset: "+05:00", std_abbr: "+05", dst_abbr: "" },
    "Asia/Tbilisi" => TimezoneInfo { std_offset: "+04:00", dst_offset: "+04:00", std_abbr: "+04", dst_abbr: "" },
    "Asia/Tehran" => TimezoneInfo { std_offset: "+03:30", dst_offset: "+03:30", std_abbr: "+0330", dst_abbr: "" },
    "Asia/Thimphu" => TimezoneInfo { std_offset: "+06:00", dst_offset: "+06:00", std_abbr: "+06", dst_abbr: "" },
    "Asia/Tokyo" => TimezoneInfo { std_offset: "+09:00", dst_offset: "+09:00", std_abbr: "JST", dst_abbr: "" },
    "Asia/Tomsk" => TimezoneInfo { std_offset: "+07:00", dst_offset: "+07:00", std_abbr: "+07", dst_abbr: "" },
    "Asia/Ulaanbaatar" => TimezoneInfo { std_offset: "+08:00", dst_offset: "+08:00", std_abbr: "+08", dst_abbr: "" },
    "Asia/Urumqi" => TimezoneInfo { std_offset: "+06:00", dst_offset: "+06:00", std_abbr: "+06", dst_abbr: "" },
    "Asia/Ust-Nera" => TimezoneInfo { std_offset: "+10:00", dst_offset: "+10:00", std_abbr: "+10", dst_abbr: "" },
    "Asia/Vladivostok" => TimezoneInfo { std_offset: "+10:00", dst_offset: "+10:00", std_abbr: "+10", dst_abbr: "" },
    "Asia/Yakutsk" => TimezoneInfo { std_offset: "+09:00", dst_offset: "+09:00", std_abbr: "+09", dst_abbr: "" },
    "Asia/Yangon" => TimezoneInfo { std_offset: "+06:30", dst_offset: "+06:30", std_abbr: "MMT", dst_abbr: "" },
    "Asia/Yekaterinburg" => TimezoneInfo { std_offset: "+05:00", dst_offset: "+05:00", std_abbr: "+05", dst_abbr: "" },
    "Asia/Yerevan" => TimezoneInfo { std_offset: "+04:00", dst_offset: "+04:00", std_abbr: "+04", dst_abbr: "" },

    // Atlantic
    "Atlantic/Azores" => TimezoneInfo { std_offset: "-01:00", dst_offset: "+00:00", std_abbr: "-01", dst_abbr: "+00" },
    "Atlantic/Bermuda" => TimezoneInfo { std_offset: "-04:00", dst_offset: "-03:00", std_abbr: "AST", dst_abbr: "ADT" },
    "Atlantic/Canary" => TimezoneInfo { std_offset: "+00:00", dst_offset: "+01:00", std_abbr: "WET", dst_abbr: "WEST" },
    "Atlantic/Cape_Verde" => TimezoneInfo { std_offset: "-01:00", dst_offset: "-01:00", std_abbr: "-01", dst_abbr: "" },
    "Atlantic/Faroe" => TimezoneInfo { std_offset: "+00:00", dst_offset: "+01:00", std_abbr: "WET", dst_abbr: "WEST" },
    "Atlantic/Madeira" => TimezoneInfo { std_offset: "+00:00", dst_offset: "+01:00", std_abbr: "WET", dst_abbr: "WEST" },
    "Atlantic/South_Georgia" => TimezoneInfo { std_offset: "-02:00", dst_offset: "-02:00", std_abbr: "-02", dst_abbr: "" },
    "Atlantic/Stanley" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-03:00", std_abbr: "-03", dst_abbr: "" },

    // Australia
    "Australia/Adelaide" => TimezoneInfo { std_offset: "+09:30", dst_offset: "+10:30", std_abbr: "ACST", dst_abbr: "ACDT" },
    "Australia/Brisbane" => TimezoneInfo { std_offset: "+10:00", dst_offset: "+10:00", std_abbr: "AEST", dst_abbr: "" },
    "Australia/Broken_Hill" => TimezoneInfo { std_offset: "+09:30", dst_offset: "+10:30", std_abbr: "ACST", dst_abbr: "ACDT" },
    "Australia/Darwin" => TimezoneInfo { std_offset: "+09:30", dst_offset: "+09:30", std_abbr: "ACST", dst_abbr: "" },
    "Australia/Eucla" => TimezoneInfo { std_offset: "+08:45", dst_offset: "+08:45", std_abbr: "+0845", dst_abbr: "" },
    "Australia/Hobart" => TimezoneInfo { std_offset: "+10:00", dst_offset: "+11:00", std_abbr: "AEST", dst_abbr: "AEDT" },
    "Australia/Lindeman" => TimezoneInfo { std_offset: "+10:00", dst_offset: "+10:00", std_abbr: "AEST", dst_abbr: "" },
    "Australia/Lord_Howe" => TimezoneInfo { std_offset: "+10:30", dst_offset: "+11:00", std_abbr: "+1030", dst_abbr: "+11" },
    "Australia/Melbourne" => TimezoneInfo { std_offset: "+10:00", dst_offset: "+11:00", std_abbr: "AEST", dst_abbr: "AEDT" },
    "Australia/Perth" => TimezoneInfo { std_offset: "+08:00", dst_offset: "+08:00", std_abbr: "AWST", dst_abbr: "" },
    "Australia/Sydney" => TimezoneInfo { std_offset: "+10:00", dst_offset: "+11:00", std_abbr: "AEST", dst_abbr: "AEDT" },

    // Etc
    "Etc/GMT" => TimezoneInfo { std_offset: "+00:00", dst_offset: "+00:00", std_abbr: "GMT", dst_abbr: "" },
    "Etc/GMT+1" => TimezoneInfo { std_offset: "-01:00", dst_offset: "-01:00", std_abbr: "-01", dst_abbr: "" },
    "Etc/GMT+10" => TimezoneInfo { std_offset: "-10:00", dst_offset: "-10:00", std_abbr: "-10", dst_abbr: "" },
    "Etc/GMT+11" => TimezoneInfo { std_offset: "-11:00", dst_offset: "-11:00", std_abbr: "-11", dst_abbr: "" },
    "Etc/GMT+12" => TimezoneInfo { std_offset: "-12:00", dst_offset: "-12:00", std_abbr: "-12", dst_abbr: "" },
    "Etc/GMT+2" => TimezoneInfo { std_offset: "-02:00", dst_offset: "-02:00", std_abbr: "-02", dst_abbr: "" },
    "Etc/GMT+3" => TimezoneInfo { std_offset: "-03:00", dst_offset: "-03:00", std_abbr: "-03", dst_abbr: "" },
    "Etc/GMT+4" => TimezoneInfo { std_offset: "-04:00", dst_offset: "-04:00", std_abbr: "-04", dst_abbr: "" },
    "Etc/GMT+5" => TimezoneInfo { std_offset: "-05:00", dst_offset: "-05:00", std_abbr: "-05", dst_abbr: "" },
    "Etc/GMT+6" => TimezoneInfo { std_offset: "-06:00", dst_offset: "-06:00", std_abbr: "-06", dst_abbr: "" },
    "Etc/GMT+7" => TimezoneInfo { std_offset: "-07:00", dst_offset: "-07:00", std_abbr: "-07", dst_abbr: "" },
    "Etc/GMT+8" => TimezoneInfo { std_offset: "-08:00", dst_offset: "-08:00", std_abbr: "-08", dst_abbr: "" },
    "Etc/GMT+9" => TimezoneInfo { std_offset: "-09:00", dst_offset: "-09:00", std_abbr: "-09", dst_abbr: "" },
    "Etc/GMT-1" => TimezoneInfo { std_offset: "+01:00", dst_offset: "+01:00", std_abbr: "+01", dst_abbr: "" },
    "Etc/GMT-10" => TimezoneInfo { std_offset: "+10:00", dst_offset: "+10:00", std_abbr: "+10", dst_abbr: "" },
    "Etc/GMT-11" => TimezoneInfo { std_offset: "+11:00", dst_offset: "+11:00", std_abbr: "+11", dst_abbr: "" },
    "Etc/GMT-12" => TimezoneInfo { std_offset: "+12:00", dst_offset: "+12:00", std_abbr: "+12", dst_abbr: "" },
    "Etc/GMT-13" => TimezoneInfo { std_offset: "+13:00", dst_offset: "+13:00", std_abbr: "+13", dst_abbr: "" },
    "Etc/GMT-14" => TimezoneInfo { std_offset: "+14:00", dst_offset: "+14:00", std_abbr: "+14", dst_abbr: "" },
    "Etc/GMT-2" => TimezoneInfo { std_offset: "+02:00", dst_offset: "+02:00", std_abbr: "+02", dst_abbr: "" },
    "Etc/GMT-3" => TimezoneInfo { std_offset: "+03:00", dst_offset: "+03:00", std_abbr: "+03", dst_abbr: "" },
    "Etc/GMT-4" => TimezoneInfo { std_offset: "+04:00", dst_offset: "+04:00", std_abbr: "+04", dst_abbr: "" },
    "Etc/GMT-5" => TimezoneInfo { std_offset: "+05:00", dst_offset: "+05:00", std_abbr: "+05", dst_abbr: "" },
    "Etc/GMT-6" => TimezoneInfo { std_offset: "+06:00", dst_offset: "+06:00", std_abbr: "+06", dst_abbr: "" },
    "Etc/GMT-7" => TimezoneInfo { std_offset: "+07:00", dst_offset: "+07:00", std_abbr: "+07", dst_abbr: "" },
    "Etc/GMT-8" => TimezoneInfo { std_offset: "+08:00", dst_offset: "+08:00", std_abbr: "+08", dst_abbr: "" },
    "Etc/GMT-9" => TimezoneInfo { std_offset: "+09:00", dst_offset: "+09:00", std_abbr: "+09", dst_abbr: "" },
    "Etc/UTC" => TimezoneInfo { std_offset: "+00:00", dst_offset: "+00:00", std_abbr: "UTC", dst_abbr: "" },

    // Europe
    "Europe/Andorra" => TimezoneInfo { std_offset: "+01:00", dst_offset: "+02:00", std_abbr: "CET", dst_abbr: "CEST" },
    "Europe/Astrakhan" => TimezoneInfo { std_offset: "+04:00", dst_offset: "+04:00", std_abbr: "+04", dst_abbr: "" },
    "Europe/Athens" => TimezoneInfo { std_offset: "+02:00", dst_offset: "+03:00", std_abbr: "EET", dst_abbr: "EEST" },
    "Europe/Belgrade" => TimezoneInfo { std_offset: "+01:00", dst_offset: "+02:00", std_abbr: "CET", dst_abbr: "CEST" },
    "Europe/Berlin" => TimezoneInfo { std_offset: "+01:00", dst_offset: "+02:00", std_abbr: "CET", dst_abbr: "CEST" },
    "Europe/Brussels" => TimezoneInfo { std_offset: "+01:00", dst_offset: "+02:00", std_abbr: "CET", dst_abbr: "CEST" },
    "Europe/Bucharest" => TimezoneInfo { std_offset: "+02:00", dst_offset: "+03:00", std_abbr: "EET", dst_abbr: "EEST" },
    "Europe/Budapest" => TimezoneInfo { std_offset: "+01:00", dst_offset: "+02:00", std_abbr: "CET", dst_abbr: "CEST" },
    "Europe/Chisinau" => TimezoneInfo { std_offset: "+02:00", dst_offset: "+03:00", std_abbr: "EET", dst_abbr: "EEST" },
    "Europe/Dublin" => TimezoneInfo { std_offset: "+00:00", dst_offset: "+01:00", std_abbr: "GMT", dst_abbr: "IST" },
    "Europe/Gibraltar" => TimezoneInfo { std_offset: "+01:00", dst_offset: "+02:00", std_abbr: "CET", dst_abbr: "CEST" },
    "Europe/Helsinki" => TimezoneInfo { std_offset: "+02:00", dst_offset: "+03:00", std_abbr: "EET", dst_abbr: "EEST" },
    "Europe/Istanbul" => TimezoneInfo { std_offset: "+03:00", dst_offset: "+03:00", std_abbr: "+03", dst_abbr: "" },
    "Europe/Kaliningrad" => TimezoneInfo { std_offset: "+02:00", dst_offset: "+02:00", std_abbr: "EET", dst_abbr: "" },
    "Europe/Kirov" => TimezoneInfo { std_offset: "+03:00", dst_offset: "+03:00", std_abbr: "MSK", dst_abbr: "" },
    "Europe/Kyiv" => TimezoneInfo { std_offset: "+02:00", dst_offset: "+03:00", std_abbr: "EET", dst_abbr: "EEST" },
    "Europe/Lisbon" => TimezoneInfo { std_offset: "+00:00", dst_offset: "+01:00", std_abbr: "WET", dst_abbr: "WEST" },
    "Europe/London" => TimezoneInfo { std_offset: "+00:00", dst_offset: "+01:00", std_abbr: "GMT", dst_abbr: "BST" },
    "Europe/Madrid" => TimezoneInfo { std_offset: "+01:00", dst_offset: "+02:00", std_abbr: "CET", dst_abbr: "CEST" },
    "Europe/Malta" => TimezoneInfo { std_offset: "+01:00", dst_offset: "+02:00", std_abbr: "CET", dst_abbr: "CEST" },
    "Europe/Minsk" => TimezoneInfo { std_offset: "+03:00", dst_offset: "+03:00", std_abbr: "+03", dst_abbr: "" },
    "Europe/Moscow" => TimezoneInfo { std_offset: "+03:00", dst_offset: "+03:00", std_abbr: "MSK", dst_abbr: "" },
    "Europe/Paris" => TimezoneInfo { std_offset: "+01:00", dst_offset: "+02:00", std_abbr: "CET", dst_abbr: "CEST" },
    "Europe/Prague" => TimezoneInfo { std_offset: "+01:00", dst_offset: "+02:00", std_abbr: "CET", dst_abbr: "CEST" },
    "Europe/Riga" => TimezoneInfo { std_offset: "+02:00", dst_offset: "+03:00", std_abbr: "EET", dst_abbr: "EEST" },
    "Europe/Rome" => TimezoneInfo { std_offset: "+01:00", dst_offset: "+02:00", std_abbr: "CET", dst_abbr: "CEST" },
    "Europe/Samara" => TimezoneInfo { std_offset: "+04:00", dst_offset: "+04:00", std_abbr: "+04", dst_abbr: "" },
    "Europe/Saratov" => TimezoneInfo { std_offset: "+04:00", dst_offset: "+04:00", std_abbr: "+04", dst_abbr: "" },
    "Europe/Simferopol" => TimezoneInfo { std_offset: "+03:00", dst_offset: "+03:00", std_abbr: "MSK", dst_abbr: "" },
    "Europe/Sofia" => TimezoneInfo { std_offset: "+02:00", dst_offset: "+03:00", std_abbr: "EET", dst_abbr: "EEST" },
    "Europe/Tallinn" => TimezoneInfo { std_offset: "+02:00", dst_offset: "+03:00", std_abbr: "EET", dst_abbr: "EEST" },
    "Europe/Tirane" => TimezoneInfo { std_offset: "+01:00", dst_offset: "+02:00", std_abbr: "CET", dst_abbr: "CEST" },
    "Europe/Ulyanovsk" => TimezoneInfo { std_offset: "+04:00", dst_offset: "+04:00", std_abbr: "+04", dst_abbr: "" },
    "Europe/Vienna" => TimezoneInfo { std_offset: "+01:00", dst_offset: "+02:00", std_abbr: "CET", dst_abbr: "CEST" },
    "Europe/Vilnius" => TimezoneInfo { std_offset: "+02:00", dst_offset: "+03:00", std_abbr: "EET", dst_abbr: "EEST" },
    "Europe/Volgograd" => TimezoneInfo { std_offset: "+03:00", dst_offset: "+03:00", std_abbr: "MSK", dst_abbr: "" },
    "Europe/Warsaw" => TimezoneInfo { std_offset: "+01:00", dst_offset: "+02:00", std_abbr: "CET", dst_abbr: "CEST" },
    "Europe/Zurich" => TimezoneInfo { std_offset: "+01:00", dst_offset: "+02:00", std_abbr: "CET", dst_abbr: "CEST" },

    // Factory
    "Factory" => TimezoneInfo { std_offset: "+00:00", dst_offset: "+00:00", std_abbr: "+00", dst_abbr: "" },

    // Indian
    "Indian/Chagos" => TimezoneInfo { std_offset: "+06:00", dst_offset: "+06:00", std_abbr: "+06", dst_abbr: "" },
    "Indian/Maldives" => TimezoneInfo { std_offset: "+05:00", dst_offset: "+05:00", std_abbr: "+05", dst_abbr: "" },
    "Indian/Mauritius" => TimezoneInfo { std_offset: "+04:00", dst_offset: "+04:00", std_abbr: "+04", dst_abbr: "" },

    // Pacific
    "Pacific/Apia" => TimezoneInfo { std_offset: "+13:00", dst_offset: "+13:00", std_abbr: "+13", dst_abbr: "" },
    "Pacific/Auckland" => TimezoneInfo { std_offset: "+12:00", dst_offset: "+13:00", std_abbr: "NZST", dst_abbr: "NZDT" },
    "Pacific/Bougainville" => TimezoneInfo { std_offset: "+11:00", dst_offset: "+11:00", std_abbr: "+11", dst_abbr: "" },
    "Pacific/Chatham" => TimezoneInfo { std_offset: "+12:45", dst_offset: "+13:45", std_abbr: "+1245", dst_abbr: "+1345" },
    "Pacific/Easter" => TimezoneInfo { std_offset: "-06:00", dst_offset: "-05:00", std_abbr: "-06", dst_abbr: "-05" },
    "Pacific/Efate" => TimezoneInfo { std_offset: "+11:00", dst_offset: "+11:00", std_abbr: "+11", dst_abbr: "" },
    "Pacific/Fakaofo" => TimezoneInfo { std_offset: "+13:00", dst_offset: "+13:00", std_abbr: "+13", dst_abbr: "" },
    "Pacific/Fiji" => TimezoneInfo { std_offset: "+12:00", dst_offset: "+12:00", std_abbr: "+12", dst_abbr: "" },
    "Pacific/Galapagos" => TimezoneInfo { std_offset: "-06:00", dst_offset: "-06:00", std_abbr: "-06", dst_abbr: "" },
    "Pacific/Gambier" => TimezoneInfo { std_offset: "-09:00", dst_offset: "-09:00", std_abbr: "-09", dst_abbr: "" },
    "Pacific/Guadalcanal" => TimezoneInfo { std_offset: "+11:00", dst_offset: "+11:00", std_abbr: "+11", dst_abbr: "" },
    "Pacific/Guam" => TimezoneInfo { std_offset: "+10:00", dst_offset: "+10:00", std_abbr: "ChST", dst_abbr: "" },
    "Pacific/Honolulu" => TimezoneInfo { std_offset: "-10:00", dst_offset: "-10:00", std_abbr: "HST", dst_abbr: "" },
    "Pacific/Kanton" => TimezoneInfo { std_offset: "+13:00", dst_offset: "+13:00", std_abbr: "+13", dst_abbr: "" },
    "Pacific/Kiritimati" => TimezoneInfo { std_offset: "+14:00", dst_offset: "+14:00", std_abbr: "+14", dst_abbr: "" },
    "Pacific/Kosrae" => TimezoneInfo { std_offset: "+11:00", dst_offset: "+11:00", std_abbr: "+11", dst_abbr: "" },
    "Pacific/Kwajalein" => TimezoneInfo { std_offset: "+12:00", dst_offset: "+12:00", std_abbr: "+12", dst_abbr: "" },
    "Pacific/Marquesas" => TimezoneInfo { std_offset: "-09:30", dst_offset: "-09:30", std_abbr: "-0930", dst_abbr: "" },
    "Pacific/Nauru" => TimezoneInfo { std_offset: "+12:00", dst_offset: "+12:00", std_abbr: "+12", dst_abbr: "" },
    "Pacific/Niue" => TimezoneInfo { std_offset: "-11:00", dst_offset: "-11:00", std_abbr: "-11", dst_abbr: "" },
    "Pacific/Norfolk" => TimezoneInfo { std_offset: "+11:00", dst_offset: "+11:00", std_abbr: "+11", dst_abbr: "+11" },
    "Pacific/Noumea" => TimezoneInfo { std_offset: "+11:00", dst_offset: "+11:00", std_abbr: "+11", dst_abbr: "" },
    "Pacific/Pago_Pago" => TimezoneInfo { std_offset: "-11:00", dst_offset: "-11:00", std_abbr: "SST", dst_abbr: "" },
    "Pacific/Palau" => TimezoneInfo { std_offset: "+09:00", dst_offset: "+09:00", std_abbr: "+09", dst_abbr: "" },
    "Pacific/Pitcairn" => TimezoneInfo { std_offset: "-08:00", dst_offset: "-08:00", std_abbr: "-08", dst_abbr: "" },
    "Pacific/Port_Moresby" => TimezoneInfo { std_offset: "+10:00", dst_offset: "+10:00", std_abbr: "+10", dst_abbr: "" },
    "Pacific/Rarotonga" => TimezoneInfo { std_offset: "-10:00", dst_offset: "-10:00", std_abbr: "-10", dst_abbr: "" },
    "Pacific/Tahiti" => TimezoneInfo { std_offset: "-10:00", dst_offset: "-10:00", std_abbr: "-10", dst_abbr: "" },
    "Pacific/Tarawa" => TimezoneInfo { std_offset: "+12:00", dst_offset: "+12:00", std_abbr: "+12", dst_abbr: "" },
    "Pacific/Tongatapu" => TimezoneInfo { std_offset: "+13:00", dst_offset: "+13:00", std_abbr: "+13", dst_abbr: "" },
};

/// Static map of timezone abbreviations to standard offset strings
///
/// Note: Some abbreviations are ambiguous (e.g., "CST" can mean Central Standard Time,
/// China Standard Time, or Cuba Standard Time). This map uses the most common interpretation.
/// For unambiguous results, use IANA timezone identifiers or direct offset strings.
#[cfg(feature = "datetime_ops")]
pub static ABBR_TO_OFFSET: Map<&'static str, &'static str> = phf_map! {
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
};

/// Looks up timezone information by IANA identifier, abbreviation, or direct offset
///
/// # Arguments
/// * `tz_str` - Timezone string (e.g., "Australia/Sydney", "AEST", "+10:00")
///
/// # Returns
/// Standard offset string in ±HH:MM format, or None if not found
///
/// # Examples
/// ```
/// use minarrow::structs::variants::datetime::tz::lookup_timezone;
///
/// # #[cfg(feature = "datetime_ops")]
/// # {
/// assert_eq!(lookup_timezone("Australia/Sydney"), Some("+10:00"));
/// assert_eq!(lookup_timezone("AEST"), Some("+10:00"));
/// assert_eq!(lookup_timezone("+10:00"), Some("+10:00"));
/// assert_eq!(lookup_timezone("UTC"), Some("+00:00"));
/// # }
/// ```
#[cfg(feature = "datetime_ops")]
pub fn lookup_timezone(tz_str: &str) -> Option<&str> {
    // Try IANA timezone identifier lookup
    if let Some(tz_info) = TZ_DATABASE.get(tz_str) {
        return Some(tz_info.std_offset);
    }

    // Try abbreviation lookup
    if let Some(offset) = ABBR_TO_OFFSET.get(tz_str) {
        return Some(offset);
    }

    // Check if it's already a direct offset string
    if is_offset_string(tz_str) {
        return Some(tz_str);
    }

    None
}

/// Checks if a string is a valid offset string (e.g., "+10:00", "-05:00")
#[cfg(feature = "datetime_ops")]
fn is_offset_string(s: &str) -> bool {
    if s.eq_ignore_ascii_case("UTC") || s.eq_ignore_ascii_case("Z") {
        return true;
    }

    let s = s.trim();
    if !s.starts_with('+') && !s.starts_with('-') {
        return false;
    }

    // Check for HH:MM format or HHMM format
    let rest = &s[1..];
    if rest.contains(':') {
        // HH:MM format
        rest.len() == 5 && rest.chars().all(|c| c.is_ascii_digit() || c == ':')
    } else {
        // HHMM format
        rest.len() == 4 && rest.chars().all(|c| c.is_ascii_digit())
    }
}

#[cfg(all(feature = "datetime", feature = "datetime_ops"))]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_iana_identifier() {
        assert_eq!(lookup_timezone("Australia/Sydney"), Some("+10:00"));
        assert_eq!(lookup_timezone("America/New_York"), Some("-05:00"));
        assert_eq!(lookup_timezone("Europe/London"), Some("+00:00"));
        assert_eq!(lookup_timezone("Asia/Tokyo"), Some("+09:00"));
        assert_eq!(lookup_timezone("Pacific/Auckland"), Some("+12:00"));
    }

    #[test]
    fn test_lookup_abbreviation() {
        assert_eq!(lookup_timezone("AEST"), Some("+10:00"));
        assert_eq!(lookup_timezone("EST"), Some("-05:00"));
        assert_eq!(lookup_timezone("GMT"), Some("+00:00"));
        assert_eq!(lookup_timezone("UTC"), Some("+00:00"));
        assert_eq!(lookup_timezone("PST"), Some("-08:00"));
        assert_eq!(lookup_timezone("JST"), Some("+09:00"));
    }

    #[test]
    fn test_lookup_direct_offset() {
        assert_eq!(lookup_timezone("+10:00"), Some("+10:00"));
        assert_eq!(lookup_timezone("-05:00"), Some("-05:00"));
        assert_eq!(lookup_timezone("+00:00"), Some("+00:00"));
        assert_eq!(lookup_timezone("+05:30"), Some("+05:30"));
        assert_eq!(lookup_timezone("-03:30"), Some("-03:30"));
    }

    #[test]
    fn test_lookup_invalid() {
        assert_eq!(lookup_timezone("Invalid/Timezone"), None);
        assert_eq!(lookup_timezone("INVALID"), None);
        assert_eq!(lookup_timezone("25:00"), None);
    }

    #[test]
    fn test_unusual_offsets() {
        // Test unusual offset timezones
        assert_eq!(lookup_timezone("Australia/Eucla"), Some("+08:45"));
        assert_eq!(lookup_timezone("Asia/Kathmandu"), Some("+05:45"));
        assert_eq!(lookup_timezone("Asia/Colombo"), Some("+05:30"));
        assert_eq!(lookup_timezone("America/St_Johns"), Some("-03:30"));
        assert_eq!(lookup_timezone("Pacific/Chatham"), Some("+12:45"));
    }

    #[test]
    fn test_all_timezones_have_valid_offsets() {
        for (tz_name, tz_info) in &TZ_DATABASE {
            assert!(
                is_offset_string(tz_info.std_offset),
                "Invalid std_offset for {}: {}",
                tz_name,
                tz_info.std_offset
            );
            if !tz_info.dst_offset.is_empty() {
                assert!(
                    is_offset_string(tz_info.dst_offset),
                    "Invalid dst_offset for {}: {}",
                    tz_name,
                    tz_info.dst_offset
                );
            }
        }
    }
}
