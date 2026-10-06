// Copyright 2025 Peter Garfield Bower
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Exact comparison and guard tests for the string to categorical conversion.
//!
//! `reference_string_to_cat` is the previous body of
//! `TryFrom<StringAVT<Off>> for CategoricalArray<Idx>`, which validated each
//! row with `str::from_utf8` and interned it through a SipHash `HashMap`. The
//! comparison tests run it and the current conversion on the same randomised
//! inputs and compare codes, dictionary contents and order, null masks,
//! errors and panics. The inputs cover all-distinct and heavy-duplicate
//! labels, multi-byte UTF-8, empty strings, null rows, both offset widths,
//! non-zero windows, invalid UTF-8, character boundaries split across rows,
//! malformed offsets and code overflow. The `TextArray::try_cat*` callers
//! are compared against the same reference.
//!
//! Run under each categorical configuration:
//!
//! ```ignore
//! cargo test --test string_to_categorical_hasher
//! cargo test --test string_to_categorical_hasher --features default_categorical_8
//! cargo test --test string_to_categorical_hasher --features extended_categorical
//! cargo test --test string_to_categorical_hasher --features extended_categorical,shared_dict
//! ```

use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::time::{Duration, Instant};

use minarrow::enums::error::MinarrowError;
use minarrow::{
    Bitmask, CategoricalArray, Integer, MaskedArray, StringAVT, StringArray, TextArray, Vec64,
};
use num_traits::FromPrimitive;

// The previous conversion body, kept unchanged apart from generic offset and
// code types in place of the macro parameters.
fn reference_string_to_cat<O: Integer, I: Integer + FromPrimitive>(
    src: &StringArray<O>,
    offset: usize,
    len: usize,
) -> Result<CategoricalArray<I>, MinarrowError> {
    let n_rows = src.offsets.len().saturating_sub(1);
    if offset + len > n_rows {
        return Err(MinarrowError::IndexError(format!(
            "String window {}..{} is out of bounds for an array of length {}",
            offset,
            offset + len,
            n_rows
        )));
    }

    let mut dict = HashMap::<&str, I>::new();
    let mut uniq = Vec64::new();
    let mut codes = Vec64::with_capacity(len);

    for win in src.offsets[offset..].windows(2).take(len) {
        let (start, end) = (Integer::to_usize(win[0]), Integer::to_usize(win[1]));
        let slice = &src.data[start..end];
        let s = std::str::from_utf8(slice).map_err(|e| MinarrowError::TypeError {
            from: "String",
            to: "Categorical",
            message: Some(e.to_string()),
        })?;

        let code = *dict.entry(s).or_insert_with(|| {
            let next = uniq.len();
            let idx_val: I = <I as FromPrimitive>::from_usize(next)
                .ok_or_else(|| MinarrowError::Overflow {
                    value: next.to_string(),
                    target: std::any::type_name::<I>(),
                })
                .unwrap();
            uniq.push(s.to_owned());
            idx_val
        });
        codes.push(code);
    }

    let null_mask = if offset == 0 && len == n_rows {
        src.null_mask.clone()
    } else {
        src.null_mask.as_ref().map(|mask| mask.slice_clone(offset, len))
    };

    Ok(CategoricalArray::from_parts(codes, uniq, null_mask))
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else {
        "non-string panic payload".to_string()
    }
}

// Asserts that two conversion outcomes are identical, including the panic
// message when both panic.
fn assert_same_outcome<I: Integer + Debug>(
    new: std::thread::Result<Result<CategoricalArray<I>, MinarrowError>>,
    old: std::thread::Result<Result<CategoricalArray<I>, MinarrowError>>,
    context: &str,
) {
    match (new, old) {
        (Ok(Ok(new)), Ok(Ok(old))) => {
            assert_eq!(&new.data[..], &old.data[..], "codes differ: {context}");
            assert_eq!(new.unique_values(), old.unique_values(), "dictionary differs: {context}");
            assert_eq!(new.null_mask, old.null_mask, "null mask differs: {context}");
            assert_eq!(new, old, "arrays differ: {context}");
        }
        (Ok(Err(new)), Ok(Err(old))) => assert_eq!(new, old, "errors differ: {context}"),
        (Err(new), Err(old)) => {
            assert_eq!(panic_message(new), panic_message(old), "panics differ: {context}")
        }
        (new, old) => panic!(
            "outcomes differ: {context}: new {}, old {}",
            describe(&new),
            describe(&old)
        ),
    }
}

fn describe<I: Integer>(
    outcome: &std::thread::Result<Result<CategoricalArray<I>, MinarrowError>>,
) -> &'static str {
    match outcome {
        Ok(Ok(_)) => "Ok",
        Ok(Err(_)) => "Err",
        Err(_) => "panic",
    }
}

fn assert_window_matches<O, I>(src: &StringArray<O>, offset: usize, len: usize, context: &str)
where
    O: Integer,
    I: Integer + FromPrimitive,
    CategoricalArray<I>: for<'a> TryFrom<StringAVT<'a, O>, Error = MinarrowError>,
{
    let new = catch_unwind(AssertUnwindSafe(|| CategoricalArray::<I>::try_from((src, offset, len))));
    let old = catch_unwind(AssertUnwindSafe(|| reference_string_to_cat::<O, I>(src, offset, len)));
    assert_same_outcome(new, old, &format!("{context}, window {offset}+{len}"));
}

fn assert_whole_matches<O, I>(src: &StringArray<O>, context: &str)
where
    O: Integer,
    I: Integer + FromPrimitive,
    CategoricalArray<I>: for<'a> TryFrom<&'a StringArray<O>, Error = MinarrowError>,
{
    let n_rows = src.offsets.len().saturating_sub(1);
    let new = catch_unwind(AssertUnwindSafe(|| CategoricalArray::<I>::try_from(src)));
    let old = catch_unwind(AssertUnwindSafe(|| reference_string_to_cat::<O, I>(src, 0, n_rows)));
    assert_same_outcome(new, old, &format!("{context}, whole array"));
}

// SplitMix64, so every case is reproducible from its seed.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

// Labels with equal 8-byte prefixes, lengths on and around the 8-byte word
// size, a trailing NUL and multi-byte characters.
const POOL: [&str; 14] = [
    "",
    "a",
    "ab",
    "ab\0",
    "abcdefgh",
    "abcdefghi",
    "12345678\0",
    "é",
    "日本語",
    "😀x",
    "prefix_shared_long_0001",
    "prefix_shared_long_0002",
    "ÄÖÜ äöü",
    "z",
];

#[derive(Clone, Copy, Debug)]
enum Distribution {
    AllDistinct,
    HeavyDuplicate,
    Mixed,
}

const DISTRIBUTIONS: [Distribution; 3] =
    [Distribution::AllDistinct, Distribution::HeavyDuplicate, Distribution::Mixed];

fn distinct_label(rng: &mut SplitMix64, i: usize) -> String {
    let suffix = match rng.below(4) {
        0 => "",
        1 => "é",
        2 => "_suffix_beyond_one_word",
        _ => "日",
    };
    format!("{i}{suffix}")
}

fn labels(rng: &mut SplitMix64, n: usize, dist: Distribution) -> Vec<String> {
    (0..n)
        .map(|i| match dist {
            Distribution::AllDistinct => distinct_label(rng, i),
            Distribution::HeavyDuplicate => POOL[rng.below(POOL.len() as u64) as usize].to_string(),
            Distribution::Mixed => {
                if rng.below(2) == 0 {
                    distinct_label(rng, i)
                } else {
                    POOL[rng.below(POOL.len() as u64) as usize].to_string()
                }
            }
        })
        .collect()
}

fn random_mask(rng: &mut SplitMix64, n: usize) -> Option<Bitmask> {
    match rng.below(3) {
        0 => None,
        1 => Some(Bitmask::from_bools(&vec![true; n])),
        _ => Some(Bitmask::from_bools(&(0..n).map(|_| rng.below(4) != 0).collect::<Vec<_>>())),
    }
}

fn random_window(rng: &mut SplitMix64, n: usize) -> (usize, usize) {
    let offset = rng.below(n as u64 + 1) as usize;
    let len = rng.below((n - offset) as u64 + 1) as usize;
    (offset, len)
}

const ROW_COUNTS: [usize; 7] = [0, 1, 2, 9, 100, 300, 2000];

// Runs randomised arrays of one offset width through one code width, as a
// whole array and as random windows.
fn run_randomised<O, I>(label: &str)
where
    O: Integer,
    I: Integer + FromPrimitive,
    CategoricalArray<I>: for<'a> TryFrom<StringAVT<'a, O>, Error = MinarrowError>
        + for<'a> TryFrom<&'a StringArray<O>, Error = MinarrowError>,
{
    for seed in 0..30u64 {
        let mut rng = SplitMix64(seed);
        for dist in DISTRIBUTIONS {
            let n = ROW_COUNTS[seed as usize % ROW_COUNTS.len()];
            let values = labels(&mut rng, n, dist);
            let mask = random_mask(&mut rng, n);
            let refs: Vec<&str> = values.iter().map(String::as_str).collect();
            let src = StringArray::<O>::from_vec(refs, mask);
            let context = format!("{label} seed {seed} {dist:?} rows {n}");

            assert_whole_matches::<O, I>(&src, &context);
            for _ in 0..4 {
                let (offset, len) = random_window(&mut rng, n);
                assert_window_matches::<O, I>(&src, offset, len, &context);
            }
            assert_window_matches::<O, I>(&src, n, 1, &context);
        }
    }
}

fn offsets<O: Integer>(values: &[usize]) -> Vec64<O> {
    values.iter().map(|&v| Integer::from_usize(v)).collect()
}

// Arrays whose bytes or offsets the conversion rejects, together with
// arrays at the edges of the window handling.
fn run_malformed<O, I>(label: &str)
where
    O: Integer,
    I: Integer + FromPrimitive,
    CategoricalArray<I>: for<'a> TryFrom<StringAVT<'a, O>, Error = MinarrowError>
        + for<'a> TryFrom<&'a StringArray<O>, Error = MinarrowError>,
{
    let cases: Vec<(&str, StringArray<O>)> = vec![
        (
            "invalid byte in the middle row",
            StringArray::new(Vec64::from_slice(b"ab\xFFcd"), None, offsets::<O>(&[0, 2, 3, 5])),
        ),
        (
            "invalid byte in the last row",
            StringArray::new(Vec64::from_slice(b"abc\xC3"), None, offsets::<O>(&[0, 2, 3, 4])),
        ),
        (
            "two-byte character split across rows",
            StringArray::new(Vec64::from_slice("xéy".as_bytes()), None, offsets::<O>(&[0, 2, 3, 4])),
        ),
        (
            "four-byte character split across rows",
            StringArray::new(Vec64::from_slice("😀".as_bytes()), None, offsets::<O>(&[0, 1, 3, 4])),
        ),
        (
            "decreasing offsets",
            StringArray::new(Vec64::from_slice(b"abcd"), None, offsets::<O>(&[0, 3, 1, 4])),
        ),
        (
            "offset past the data",
            StringArray::new(Vec64::from_slice(b"abcd"), None, offsets::<O>(&[0, 10, 4])),
        ),
        (
            "final offset past the data",
            StringArray::new(Vec64::from_slice(b"abcd"), None, offsets::<O>(&[0, 2, 9])),
        ),
        (
            "non-zero first offset",
            StringArray::new(Vec64::from_slice(b"..abcab"), None, offsets::<O>(&[2, 3, 4, 5, 7])),
        ),
        (
            "invalid bytes under a null row",
            StringArray::new(
                Vec64::from_slice(b"a\xFFb"),
                Some(Bitmask::from_bools(&[true, false, true])),
                offsets::<O>(&[0, 1, 2, 3]),
            ),
        ),
        ("empty array", StringArray::new(Vec64::<u8>::new(), None, offsets::<O>(&[0]))),
        ("default array", StringArray::<O>::default()),
    ];

    for (name, src) in &cases {
        let context = format!("{label} {name}");
        let n = src.offsets.len().saturating_sub(1);
        assert_whole_matches::<O, I>(src, &context);
        for offset in 0..=n {
            for len in 0..=(n - offset) {
                assert_window_matches::<O, I>(src, offset, len, &context);
            }
        }
        assert_window_matches::<O, I>(src, n, 1, &context);
    }
}

// More distinct labels than the code type holds. Both implementations panic
// with the same overflow message at the same row.
fn run_overflow<O, I>(label: &str, distinct: usize)
where
    O: Integer,
    I: Integer + FromPrimitive,
    CategoricalArray<I>: for<'a> TryFrom<StringAVT<'a, O>, Error = MinarrowError>
        + for<'a> TryFrom<&'a StringArray<O>, Error = MinarrowError>,
{
    let values: Vec<String> = (0..distinct).map(|i| format!("label_{i}")).collect();
    let refs: Vec<&str> = values.iter().map(String::as_str).collect();
    let src = StringArray::<O>::from_vec(refs, None);
    assert_whole_matches::<O, I>(&src, &format!("{label} overflow"));
    assert_window_matches::<O, I>(&src, 3, distinct - 3, &format!("{label} overflow"));
}

macro_rules! comparison_tests {
    ($cfg:meta, $idx:ty, $randomised:ident, $malformed:ident, $overflow:ident, $overflow_rows:expr) => {
        #[$cfg]
        #[test]
        fn $randomised() {
            run_randomised::<u32, $idx>(concat!("u32/", stringify!($idx)));
            #[cfg(feature = "large_string")]
            run_randomised::<u64, $idx>(concat!("u64/", stringify!($idx)));
        }

        #[$cfg]
        #[test]
        fn $malformed() {
            run_malformed::<u32, $idx>(concat!("u32/", stringify!($idx)));
            #[cfg(feature = "large_string")]
            run_malformed::<u64, $idx>(concat!("u64/", stringify!($idx)));
        }

        #[$cfg]
        #[test]
        fn $overflow() {
            run_overflow::<u32, $idx>(concat!("u32/", stringify!($idx)), $overflow_rows);
            #[cfg(feature = "large_string")]
            run_overflow::<u64, $idx>(concat!("u64/", stringify!($idx)), $overflow_rows);
        }
    };
}

comparison_tests!(
    cfg(feature = "default_categorical_8"),
    u8,
    u8_codes_match_reference,
    u8_codes_malformed_match_reference,
    u8_codes_overflow_matches_reference,
    300
);
comparison_tests!(
    cfg(feature = "extended_categorical"),
    u16,
    u16_codes_match_reference,
    u16_codes_malformed_match_reference,
    u16_codes_overflow_matches_reference,
    70_000
);
comparison_tests!(
    cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical")),
    u32,
    u32_codes_match_reference,
    u32_codes_malformed_match_reference,
    u32_codes_ok_beyond_u16_matches_reference,
    70_000
);
comparison_tests!(
    cfg(feature = "extended_categorical"),
    u64,
    u64_codes_match_reference,
    u64_codes_malformed_match_reference,
    u64_codes_ok_beyond_u16_matches_reference,
    70_000
);

// `TextArray::try_cat*` and the panicking `cat*` accessors convert string
// variants through the whole-array conversion.
fn text_arrays(seed: u64) -> Vec<(String, TextArray, StringArray<u32>, Option<StringArray<u64>>)> {
    let mut rng = SplitMix64(seed);
    let mut out = Vec::new();
    for dist in DISTRIBUTIONS {
        let n = 200;
        let values = labels(&mut rng, n, dist);
        let mask = random_mask(&mut rng, n);
        let refs: Vec<&str> = values.iter().map(String::as_str).collect();
        let s32 = StringArray::<u32>::from_vec(refs.clone(), mask.clone());
        out.push((
            format!("String32 {dist:?}"),
            TextArray::String32(Arc::new(s32.clone())),
            s32,
            None,
        ));
        #[cfg(feature = "large_string")]
        {
            let s64 = StringArray::<u64>::from_vec(refs, mask);
            out.push((
                format!("String64 {dist:?}"),
                TextArray::String64(Arc::new(s64.clone())),
                StringArray::<u32>::default(),
                Some(s64),
            ));
        }
    }
    out
}

fn assert_cast_matches<I: Integer + FromPrimitive + Debug>(
    cast: impl Fn(&TextArray) -> Result<Arc<CategoricalArray<I>>, MinarrowError>,
    seed: u64,
) {
    for (name, text, s32, s64) in text_arrays(seed) {
        let new = catch_unwind(AssertUnwindSafe(|| cast(&text).map(|a| (*a).clone())));
        let old = catch_unwind(AssertUnwindSafe(|| match &s64 {
            Some(s64) => reference_string_to_cat::<u64, I>(s64, 0, s64.len()),
            None => reference_string_to_cat::<u32, I>(&s32, 0, s32.len()),
        }));
        assert_same_outcome(new, old, &format!("try_cat {name} seed {seed}"));
    }
}

#[cfg(feature = "default_categorical_8")]
#[test]
fn try_cat8_matches_reference() {
    for seed in 0..10 {
        assert_cast_matches::<u8>(|t| t.try_cat8(), seed);
        assert_cast_matches::<u8>(|t| Ok(t.cat8()), seed);
    }
}

#[cfg(feature = "extended_categorical")]
#[test]
fn try_cat16_matches_reference() {
    for seed in 0..10 {
        assert_cast_matches::<u16>(|t| t.try_cat16(), seed);
        assert_cast_matches::<u16>(|t| Ok(t.cat16()), seed);
    }
}

#[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
#[test]
fn try_cat32_matches_reference() {
    for seed in 0..10 {
        assert_cast_matches::<u32>(|t| t.try_cat32(), seed);
        assert_cast_matches::<u32>(|t| Ok(t.cat32()), seed);
    }
}

#[cfg(feature = "extended_categorical")]
#[test]
fn try_cat64_matches_reference() {
    for seed in 0..10 {
        assert_cast_matches::<u64>(|t| t.try_cat64(), seed);
        assert_cast_matches::<u64>(|t| Ok(t.cat64()), seed);
    }
}

const GUARD_BOUND: Duration = Duration::from_secs(120);

// Four million rows of labels sharing a 28-byte prefix and differing only in
// their final digits.
fn guard_labels(cardinality: usize) -> StringArray<u32> {
    let values: Vec<String> = (0..4_000_000)
        .map(|i| format!("customer-account-identifier-{:08}", (i * 7919) % cardinality))
        .collect();
    let refs: Vec<&str> = values.iter().map(String::as_str).collect();
    StringArray::<u32>::from_vec(refs, None)
}

#[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
#[test]
fn string_to_categorical_u32_guard() {
    let src = guard_labels(200_000);
    let start = Instant::now();
    let cat = CategoricalArray::<u32>::try_from(&src).unwrap();
    assert!(start.elapsed() < GUARD_BOUND, "took {:?}", start.elapsed());
    assert_eq!(cat.unique_values().len(), 200_000);
}

#[cfg(feature = "default_categorical_8")]
#[test]
fn string_to_categorical_u8_guard() {
    let src = guard_labels(256);
    let start = Instant::now();
    let cat = CategoricalArray::<u8>::try_from(&src).unwrap();
    assert!(start.elapsed() < GUARD_BOUND, "took {:?}", start.elapsed());
    assert_eq!(cat.unique_values().len(), 256);
}
