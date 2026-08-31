"""Tests for decimal array support in minarrow-py.

Run after `maturin develop`:
    python -m pytest tests/test_decimal.py
"""

import decimal

import pytest

import minarrow as mp


# --- Construction from int values with dtype string -------------------------


def test_decimal128_from_ints():
    a = mp.Array([12345, 67890], dtype="decimal128(10,2)")
    assert len(a) == 2
    assert a.dtype == mp.DType.Decimal
    assert a.dtype.group == mp.TypeClass.Numeric
    assert a.dtype.is_numeric
    assert a.precision == 10
    assert a.scale == 2
    assert a.bit_width == 128


def test_decimal64_from_ints():
    a = mp.Array([100, 200, 300], dtype="decimal64(18,4)")
    assert len(a) == 3
    assert a.precision == 18
    assert a.scale == 4
    assert a.bit_width == 64


def test_decimal32_from_ints():
    a = mp.Array([10, 20], dtype="decimal32(9,2)")
    assert len(a) == 2
    assert a.precision == 9
    assert a.scale == 2
    assert a.bit_width == 32


def test_decimal_alias_is_decimal128():
    a = mp.Array([100], dtype="decimal(10,2)")
    assert str(a.arrow_type) == "Decimal128(10, 2)"
    assert a.bit_width == 128


def test_decimal_with_nulls():
    a = mp.Array([12345, None, 67890], dtype="decimal128(10,2)")
    assert len(a) == 3
    assert a.null_count == 1


# --- Precision and scale on non-decimal types --------------------------------


def test_precision_for_int64():
    a = mp.Array([1, 2, 3])
    assert a.precision == 19
    assert a.scale == 0


def test_precision_for_float64():
    a = mp.Array([1.0, 2.0])
    assert a.precision == 15
    assert a.scale is None


# --- Element access ---------------------------------------------------------


def test_getitem_returns_decimal():
    a = mp.Array([12345, 67890], dtype="decimal128(10,2)")
    val = a[0]
    assert isinstance(val, decimal.Decimal)
    assert val == decimal.Decimal("123.45")


def test_getitem_null_is_none():
    a = mp.Array([12345, None], dtype="decimal128(10,2)")
    assert a[1] is None


def test_getitem_negative_index():
    a = mp.Array([100, 200, 300], dtype="decimal128(10,2)")
    assert a[-1] == decimal.Decimal("3.00")


# --- Display (repr) shows scale-formatted values ----------------------------


def test_repr_shows_decimal_values():
    a = mp.Array([12345, 67890], dtype="decimal128(10,2)")
    text = repr(a)
    assert "dtype: Decimal" in text
    assert "123.45" in text
    assert "678.90" in text


def test_repr_shows_null():
    a = mp.Array([12345, None], dtype="decimal128(10,2)")
    text = repr(a)
    assert "null" in text


# --- Slicing ----------------------------------------------------------------


def test_slice_preserves_decimal():
    a = mp.Array([100, 200, 300, 400], dtype="decimal128(10,2)")
    s = a[1:3]
    assert len(s) == 2
    assert s[0] == decimal.Decimal("2.00")
    assert s[1] == decimal.Decimal("3.00")


# --- Named column -----------------------------------------------------------


def test_named_decimal_column():
    a = mp.Array([12345, 67890], name="price", dtype="decimal128(10,2)")
    assert a.name == "price"
    assert a.precision == 10
    assert a.scale == 2


# --- ArrowType construction -------------------------------------------------


def test_arrow_type_decimal128():
    at = mp.ArrowType.Decimal128(precision=38, scale=10)
    assert str(at) == "Decimal128(38, 10)"


def test_arrow_type_decimal64():
    at = mp.ArrowType.Decimal64(precision=18, scale=4)
    assert str(at) == "Decimal64(18, 4)"


def test_arrow_type_decimal32():
    at = mp.ArrowType.Decimal32(precision=9, scale=2)
    assert str(at) == "Decimal32(9, 2)"


# --- Arrow interop (PyArrow round-trip) -------------------------------------


def test_from_arrow_decimal128():
    import pyarrow as pa

    pa_arr = pa.array(
        [decimal.Decimal("123.45"), decimal.Decimal("678.90"), None],
        type=pa.decimal128(10, 2),
    )
    a = mp.Array.from_arrow(pa_arr)
    assert a.dtype == mp.DType.Decimal
    assert a.precision == 10
    assert a.scale == 2
    assert len(a) == 3
    assert a.null_count == 1
    assert a[0] == decimal.Decimal("123.45")
    assert a[1] == decimal.Decimal("678.90")
    assert a[2] is None


def test_to_arrow_roundtrip():
    import pyarrow as pa

    pa_arr = pa.array(
        [decimal.Decimal("19.99"), decimal.Decimal("29.99")],
        type=pa.decimal128(10, 2),
    )
    a = mp.Array.from_arrow(pa_arr)
    out = a.to_arrow()
    assert out.to_pylist() == pa_arr.to_pylist()
    assert out.type == pa.decimal128(10, 2)


def test_pyarrow_roundtrip_with_nulls():
    import pyarrow as pa

    pa_arr = pa.array(
        [decimal.Decimal("1.23"), None, decimal.Decimal("-4.56")],
        type=pa.decimal128(10, 2),
    )
    a = mp.Array.from_arrow(pa_arr)
    out = a.to_arrow()
    assert out.to_pylist() == pa_arr.to_pylist()


def test_pyarrow_high_precision():
    import pyarrow as pa

    pa_arr = pa.array(
        [decimal.Decimal("12345678901234567890.1234567890")],
        type=pa.decimal128(38, 10),
    )
    a = mp.Array.from_arrow(pa_arr)
    assert a.precision == 38
    assert a.scale == 10
    out = a.to_arrow()
    assert out.to_pylist() == pa_arr.to_pylist()


# --- PyCapsule protocol (consumed by PyArrow) -------------------------------


def test_pycapsule_consumed_by_pyarrow():
    import pyarrow as pa

    a = mp.Array([12345, 67890], dtype="decimal128(10,2)")
    pa_arr = pa.array(a)
    assert pa_arr.type == pa.decimal128(10, 2)
    assert pa_arr.to_pylist() == [decimal.Decimal("123.45"), decimal.Decimal("678.90")]


def test_pycapsule_with_nulls():
    import pyarrow as pa

    a = mp.Array([12345, None, 67890], dtype="decimal128(10,2)")
    pa_arr = pa.array(a)
    assert pa_arr.to_pylist() == [
        decimal.Decimal("123.45"),
        None,
        decimal.Decimal("678.90"),
    ]
