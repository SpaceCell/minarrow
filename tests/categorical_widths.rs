// Copyright Peter Bower 2026. All Rights Reserved.

//! Verifies that `Categorical8` and `Categorical32` are both available under
//! `default_categorical_8`.
//!
//! Run with: `cargo test --test categorical_widths --features default_categorical_8`

#![cfg(feature = "default_categorical_8")]

use std::sync::Arc;

use minarrow::ffi::arrow_dtype::{ArrowType, CategoricalIndexType};
use minarrow::{Array, CategoricalArray, TextArray};

fn labels() -> [&'static str; 4] {
    ["a", "b", "a", "c"]
}

#[test]
fn builds_both_widths() {
    let cat8 = Array::from_categorical8(CategoricalArray::<u8>::from_values(labels()));
    let cat32 = Array::from_categorical32(CategoricalArray::<u32>::from_values(labels()));
    assert_eq!(cat8.len(), 4);
    assert_eq!(cat32.len(), 4);
}

#[test]
fn matches_both_widths() {
    let cat8 = Array::from_categorical8(CategoricalArray::<u8>::from_values(labels()));
    let cat32 = Array::from_categorical32(CategoricalArray::<u32>::from_values(labels()));

    match &cat8 {
        Array::TextArray(TextArray::Categorical8(a)) => assert_eq!(a.get_str(1), Some("b")),
        _ => panic!("expected Categorical8"),
    }
    match &cat32 {
        Array::TextArray(TextArray::Categorical32(a)) => assert_eq!(a.get_str(3), Some("c")),
        _ => panic!("expected Categorical32"),
    }
    assert_eq!(
        cat8.arrow_type(),
        ArrowType::Dictionary(CategoricalIndexType::UInt8)
    );
    assert_eq!(
        cat32.arrow_type(),
        ArrowType::Dictionary(CategoricalIndexType::UInt32)
    );
}

#[test]
fn converts_between_widths() {
    let cat8 = CategoricalArray::<u8>::from_values(labels());
    let wide: CategoricalArray<u32> = (&cat8).into();
    assert_eq!(wide.get_str(2), Some("a"));
    let narrow = CategoricalArray::<u8>::try_from(&wide).unwrap();
    assert_eq!(narrow.get_str(3), Some("c"));
    let _ = Arc::new(wide);
}
