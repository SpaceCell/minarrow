//! Unit tests for `delete_range` across minarrow's array types, the
//! dispatch enums, `Bitmask`, `FieldArray` and `Table`.
//!
//! Run with: `cargo test --test delete_range` and with `--features vmap64`
//! for the page-remapping fast path inside `Vec64::delete_range`.

use std::sync::Arc;

use minarrow::traits::masked_array::MaskedArray;
use minarrow::{
    Array, Bitmask, BooleanArray, FieldArray, FloatArray, IntegerArray, NumericArray, StringArray,
    Table, TextArray, arr_bool, arr_f64, arr_i64, arr_str32,
};

// Bitmask

#[test]
fn bitmask_delete_byte_aligned() {
    let mut mask = Bitmask::from_bools(&(0..32).map(|i| i % 3 == 0).collect::<Vec<_>>());
    let expected: Vec<bool> = (0..32)
        .filter(|i| !(8..16).contains(i))
        .map(|i| i % 3 == 0)
        .collect();
    mask.delete_range(8, 16);
    assert_eq!(mask.len(), 24);
    for (i, want) in expected.iter().enumerate() {
        assert_eq!(mask.get(i), *want, "bit {i}");
    }
}

#[test]
fn bitmask_delete_unaligned() {
    let bools: Vec<bool> = (0..67).map(|i| i % 5 == 0 || i % 3 == 0).collect();
    let mut mask = Bitmask::from_bools(&bools);
    mask.delete_range(3, 22);
    let expected: Vec<bool> = bools
        .iter()
        .enumerate()
        .filter(|(i, _)| !(3..22).contains(i))
        .map(|(_, b)| *b)
        .collect();
    assert_eq!(mask.len(), expected.len());
    for (i, want) in expected.iter().enumerate() {
        assert_eq!(mask.get(i), *want, "bit {i}");
    }
}

#[test]
fn bitmask_delete_tail() {
    let bools: Vec<bool> = (0..40).map(|i| i % 2 == 0).collect();
    let mut mask = Bitmask::from_bools(&bools);
    mask.delete_range(13, 40);
    assert_eq!(mask.len(), 13);
    for i in 0..13 {
        assert_eq!(mask.get(i), i % 2 == 0, "bit {i}");
    }
}

#[test]
fn bitmask_delete_empty_range_is_noop() {
    let mut mask = Bitmask::from_bools(&[true, false, true]);
    mask.delete_range(1, 1);
    assert_eq!(mask.len(), 3);
    assert!(mask.get(0) && !mask.get(1) && mask.get(2));
}

#[test]
#[should_panic]
fn bitmask_delete_out_of_bounds_panics() {
    let mut mask = Bitmask::from_bools(&[true, false]);
    mask.delete_range(0, 3);
}

// Typed arrays

#[test]
fn integer_array_delete_middle() {
    let mut arr = IntegerArray::<i64>::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7]);
    arr.delete_range(2, 5);
    assert_eq!(arr.data.as_slice(), &[0, 1, 5, 6, 7]);
}

#[test]
fn integer_array_delete_with_mask() {
    let mut arr = IntegerArray::<i64>::with_capacity(8, true);
    for i in 0..8i64 {
        if i % 3 == 0 {
            arr.push_null();
        } else {
            arr.push(i);
        }
    }
    // Values: null, 1, 2, null, 4, 5, null, 7
    arr.delete_range(1, 4);
    // Survivors: null, 4, 5, null, 7
    assert_eq!(arr.len(), 5);
    assert!(arr.is_null(0));
    assert_eq!(arr.get(1), Some(4));
    assert_eq!(arr.get(2), Some(5));
    assert!(arr.is_null(3));
    assert_eq!(arr.get(4), Some(7));
}

#[test]
fn float_array_delete_head() {
    let mut arr = FloatArray::<f64>::from_slice(&[1.5, 2.5, 3.5, 4.5]);
    arr.delete_range(0, 2);
    assert_eq!(arr.data.as_slice(), &[3.5, 4.5]);
}

#[test]
fn string_array_delete_middle() {
    let mut arr = StringArray::<u32>::from_slice(&["alpha", "bee", "see", "delta", "echo"]);
    arr.delete_range(1, 3);
    assert_eq!(arr.len(), 3);
    assert_eq!(arr.get_str(0), Some("alpha"));
    assert_eq!(arr.get_str(1), Some("delta"));
    assert_eq!(arr.get_str(2), Some("echo"));
}

#[test]
fn string_array_delete_head_rebases_offsets() {
    let mut arr = StringArray::<u32>::from_slice(&["abc", "de", "fghi", "j"]);
    arr.delete_range(0, 2);
    assert_eq!(arr.len(), 2);
    assert_eq!(arr.offsets[0], 0);
    assert_eq!(arr.get_str(0), Some("fghi"));
    assert_eq!(arr.get_str(1), Some("j"));
}

#[test]
fn string_array_delete_with_nulls() {
    let mut arr = StringArray::<u32>::default();
    arr.push_str("one");
    arr.push_null();
    arr.push_str("three");
    arr.push_str("four");
    arr.delete_range(2, 3);
    assert_eq!(arr.len(), 3);
    assert_eq!(arr.get_str(0), Some("one"));
    assert!(arr.is_null(1));
    assert_eq!(arr.get_str(2), Some("four"));
}

#[test]
fn boolean_array_delete_middle() {
    let mut arr = BooleanArray::<()>::from_slice(&[true, false, true, true, false, true]);
    arr.delete_range(1, 4);
    assert_eq!(arr.len(), 3);
    assert_eq!(arr.get(0), Some(true));
    assert_eq!(arr.get(1), Some(false));
    assert_eq!(arr.get(2), Some(true));
}

// Enum dispatch and copy-on-write

#[test]
fn array_delete_range_copy_on_write() {
    let inner = Arc::new(IntegerArray::<i64>::from_slice(&[10, 20, 30, 40]));
    let mut array = Array::NumericArray(NumericArray::Int64(inner.clone()));
    let snapshot = Array::NumericArray(NumericArray::Int64(inner.clone()));

    array.delete_range(1, 3);

    assert_eq!(array.len(), 2);
    // The reader's clone is untouched: make_mut forked before mutating.
    assert_eq!(snapshot.len(), 4);
    assert_eq!(inner.data.as_slice(), &[10, 20, 30, 40]);
}

#[test]
fn array_null_delete_range_empty_is_noop() {
    let mut array = Array::Null;
    array.delete_range(0, 0);
    assert_eq!(array.len(), 0);
}

#[test]
#[should_panic]
fn array_null_delete_range_nonempty_panics() {
    let mut array = Array::Null;
    array.delete_range(0, 1);
}

// FieldArray and Table

#[test]
fn field_array_delete_refreshes_null_count() {
    let mut arr = IntegerArray::<i64>::with_capacity(6, true);
    arr.push(1);
    arr.push_null();
    arr.push_null();
    arr.push(4);
    arr.push_null();
    arr.push(6);
    let mut fa = FieldArray::from_arr("a", arr);
    assert_eq!(fa.null_count, 3);

    fa.delete_range(1, 3);
    assert_eq!(fa.len(), 4);
    assert_eq!(fa.null_count, 1);
}

#[test]
fn table_delete_range_all_column_types() {
    let mut table = Table::new(
        "t",
        vec![
            FieldArray::from_arr("i", arr_i64![0, 1, 2, 3, 4, 5]),
            FieldArray::from_arr("f", arr_f64![0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
            FieldArray::from_arr("s", arr_str32!["r0", "r1", "r2", "r3", "r4", "r5"]),
            FieldArray::from_arr("b", arr_bool![true, false, true, false, true, false]),
        ]
        .into(),
    );
    assert_eq!(table.n_rows, 6);

    table.delete_range(2, 5);

    assert_eq!(table.n_rows, 3);
    let Array::NumericArray(NumericArray::Int64(i)) = &table.cols[0].array else {
        panic!("expected Int64 column");
    };
    assert_eq!(i.data.as_slice(), &[0, 1, 5]);
    let Array::NumericArray(NumericArray::Float64(f)) = &table.cols[1].array else {
        panic!("expected Float64 column");
    };
    assert_eq!(f.data.as_slice(), &[0.0, 0.1, 0.5]);
    let Array::TextArray(TextArray::String32(s)) = &table.cols[2].array else {
        panic!("expected String32 column");
    };
    assert_eq!(s.get_str(0), Some("r0"));
    assert_eq!(s.get_str(1), Some("r1"));
    assert_eq!(s.get_str(2), Some("r5"));
    let Array::BooleanArray(b) = &table.cols[3].array else {
        panic!("expected Boolean column");
    };
    assert_eq!(b.get(0), Some(true));
    assert_eq!(b.get(1), Some(false));
    assert_eq!(b.get(2), Some(false));
}

#[test]
fn table_delete_range_empty_is_noop() {
    let mut table = Table::new(
        "t",
        vec![FieldArray::from_arr("i", arr_i64![1, 2, 3])].into(),
    );
    table.delete_range(1, 1);
    assert_eq!(table.n_rows, 3);
}

#[test]
#[should_panic]
fn table_delete_range_out_of_bounds_panics() {
    let mut table = Table::new(
        "t",
        vec![FieldArray::from_arr("i", arr_i64![1, 2, 3])].into(),
    );
    table.delete_range(1, 4);
}

// Large-buffer pass: with vmap64 this drives the mmap splice inside
// Vec64::delete_range; without it, the same call drains. Either way the
// observable contents must be identical.

#[test]
fn large_i64_column_delete_page_multiple_span() {
    let n: usize = 512 * 1024;
    let mut data = minarrow::Vec64::with_capacity(n);
    for i in 0..n as i64 {
        data.push(i);
    }
    let mut arr = IntegerArray::<i64>::from_vec64(data, None);

    // 512-row multiple at a mid-page position.
    let start = 1037;
    let end = start + 512 * 64;
    arr.delete_range(start, end);

    assert_eq!(arr.len(), n - 512 * 64);
    for i in (start - 2)..(start + 2) {
        let expected = if i < start {
            i as i64
        } else {
            (i + 512 * 64) as i64
        };
        assert_eq!(arr.get(i), Some(expected), "row {i}");
    }
    assert_eq!(arr.get(arr.len() - 1), Some((n - 1) as i64));
}
