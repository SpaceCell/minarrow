"""Tests for the native minarrow-py NdArray surface and its DLPack bridges.

Run after `maturin develop`:
    python -m pytest tests/test_ndarray.py

The framework bridge tests skip when the target library is not installed.
"""

import pytest

import minarrow as mp


# --- Construction and inspection -------------------------------------------


def test_construct_and_inspect():
    a = mp.NdArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    assert a.shape == (3, 2)
    assert a.strides == (1, 3)
    assert a.ndim == 2
    assert a.size == 6
    assert a.dtype == "float64"
    assert len(a) == 3
    assert repr(a) == "NdArray(shape=[3, 2], dtype=float64)"


def test_construct_1d_default_shape():
    a = mp.NdArray([1.0, 2.0, 3.0])
    assert a.shape == (3,)
    assert a[1] == 2.0


def test_construct_f32():
    a = mp.NdArray([1.0, 2.0, 3.0, 4.0], shape=[2, 2], dtype="float32")
    assert a.dtype == "float32"
    assert a[1, 1] == 4.0


def test_construct_shape_mismatch():
    with pytest.raises(ValueError):
        mp.NdArray([1.0, 2.0, 3.0], shape=[2, 2])


def test_construct_bad_dtype():
    with pytest.raises(ValueError):
        mp.NdArray([1.0], dtype="int64")


def test_getitem_column_major():
    a = mp.NdArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    assert a[0, 0] == 1.0
    assert a[2, 0] == 3.0
    assert a[0, 1] == 4.0
    assert a[2, 1] == 6.0


def test_getitem_out_of_bounds():
    a = mp.NdArray([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    with pytest.raises(IndexError):
        a[2, 0]
    row = a[0]
    assert isinstance(row, mp.NdArray)
    assert row.shape == (2,)
    assert row[1] == 3.0


def test_getitem_returns_zero_copy_views():
    a = mp.NdArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    v = a[1:, :]
    assert isinstance(v, mp.NdArray)
    assert v.is_view is True
    assert v.shape == (2, 2)
    assert v.strides == (1, 3)
    assert v[0, 0] == 2.0
    assert v[-1, -1] == 6.0


def test_transpose_is_same_python_type():
    a = mp.NdArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    t = a.T
    assert isinstance(t, mp.NdArray)
    assert t.is_view is True
    assert t.shape == (2, 3)
    assert t.strides == (3, 1)
    assert t[1, 2] == 6.0


# --- DLPack protocol --------------------------------------------------------


def test_dlpack_device():
    a = mp.NdArray([1.0, 2.0])
    assert a.__dlpack_device__() == (1, 0)


def test_dlpack_capsule_names():
    a = mp.NdArray([1.0, 2.0, 3.0])
    legacy = a.__dlpack__()
    versioned = a.__dlpack__(max_version=(1, 1))
    # repr quotes the exact capsule name, so match it in full to keep
    # "dltensor" from passing on a "dltensor_versioned" capsule.
    assert '"dltensor"' in repr(legacy)
    assert '"dltensor_versioned"' in repr(versioned)


def test_rank_zero_scalar_and_dlpack():
    a = mp.NdArray([5.0], shape=[])
    assert a.shape == ()
    assert a.strides == ()
    assert a.ndim == 0
    assert a.size == 1
    assert a[()] == 5.0
    assert a.T.shape == ()
    with pytest.raises(TypeError):
        len(a)

    b = mp.NdArray.from_dlpack(a)
    assert b.shape == ()
    assert b[()] == 5.0

    np = pytest.importorskip("numpy")
    n = np.from_dlpack(a)
    assert n.shape == ()
    assert n.item() == 5.0


def test_dlpack_rejects_foreign_device():
    a = mp.NdArray([1.0, 2.0])
    with pytest.raises(ValueError):
        a.__dlpack__(dl_device=(2, 0))


def test_from_dlpack_roundtrip_self():
    a = mp.NdArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    b = mp.NdArray.from_dlpack(a)
    assert b.shape == (3, 2)
    assert b[2, 1] == 6.0
    assert b.dtype == "float64"


def test_from_dlpack_rejects_non_producer():
    with pytest.raises((ValueError, AttributeError)):
        mp.NdArray.from_dlpack(object())


# --- NumPy bridge -----------------------------------------------------------


def test_numpy_roundtrip():
    np = pytest.importorskip("numpy")
    a = mp.NdArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    v = np.from_dlpack(a)
    assert v.shape == (3, 2)
    assert v[0, 0] == 1.0
    assert v[2, 1] == 6.0

    back = mp.NdArray.from_dlpack(np.arange(6, dtype="float64").reshape(2, 3))
    assert back.shape == (2, 3)
    assert back[0, 0] == 0.0
    assert back[1, 2] == 5.0


def test_to_numpy_method():
    np = pytest.importorskip("numpy")
    a = mp.NdArray([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    v = a.to_numpy()
    assert isinstance(v, np.ndarray)
    assert v.dtype == np.float64
    assert v[1, 1] == 4.0


def test_numpy_f32_roundtrip():
    np = pytest.importorskip("numpy")
    a = mp.NdArray([1.0, 2.0, 3.0], dtype="float32")
    v = a.to_numpy()
    assert v.dtype == np.float32
    back = mp.NdArray.from_dlpack(np.asarray([1.5, 2.5], dtype="float32"))
    assert back.dtype == "float32"
    assert back[1] == 2.5


def test_dlpack_copy_is_writable():
    np = pytest.importorskip("numpy")
    a = mp.NdArray([1.0, 2.0, 3.0])
    v = np.from_dlpack(a, copy=True)
    assert v.flags.writeable is True
    v[0] = 99.0
    # The copy is independent of the source tensor.
    assert a[0] == 1.0


def test_dlpack_shared_view_is_read_only():
    np = pytest.importorskip("numpy")
    a = mp.NdArray([1.0, 2.0, 3.0])
    v = np.from_dlpack(a)
    # The versioned capsule shares the buffer, so it arrives read-only.
    assert v.flags.writeable is False


def test_dlpack_ndarray_slice_preserves_shape_and_values():
    np = pytest.importorskip("numpy")
    a = mp.NdArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    v = np.from_dlpack(a[1:, :])
    assert v.shape == (2, 2)
    assert v[0, 0] == 2.0
    assert v[1, 1] == 6.0


def test_dlpack_export_shares_one_address():
    np = pytest.importorskip("numpy")
    a = mp.NdArray([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    p1 = np.from_dlpack(a).__array_interface__["data"][0]
    p2 = np.from_dlpack(a).__array_interface__["data"][0]
    assert p1 == p2
    # An explicit copy detaches to a fresh allocation.
    p3 = np.from_dlpack(a, copy=True).__array_interface__["data"][0]
    assert p3 != p1


def test_dlpack_aligned_import_wraps_source_memory():
    np = pytest.importorskip("numpy")
    # Carve a 64-byte aligned window out of an over-allocated buffer so
    # the import takes the zero-copy branch deterministically.
    raw = np.zeros(16 + 8, dtype=np.float64)
    base = raw.__array_interface__["data"][0]
    off = (-base) % 64 // 8
    src = raw[off:off + 16].reshape(4, 4)
    src[:] = np.arange(16, dtype=np.float64).reshape(4, 4)
    p_src = src.__array_interface__["data"][0]
    assert p_src % 64 == 0

    b = mp.NdArray.from_dlpack(src)
    p_rt = np.from_dlpack(b).__array_interface__["data"][0]
    assert p_rt == p_src

    # A misaligned window copies into a fresh 64-byte aligned buffer.
    mis = raw[off + 1:off + 9]
    c = mp.NdArray.from_dlpack(mis)
    p_c = np.from_dlpack(c).__array_interface__["data"][0]
    assert p_c != mis.__array_interface__["data"][0]
    assert p_c % 64 == 0


# --- Framework bridges (skip when the library is absent) --------------------


def test_to_pytorch():
    torch = pytest.importorskip("torch")
    a = mp.NdArray([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    t = a.to_pytorch()
    assert isinstance(t, torch.Tensor)
    assert t.shape == (2, 2)
    assert t[1, 1].item() == 4.0


def test_to_jax():
    pytest.importorskip("jax")
    a = mp.NdArray([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    t = a.to_jax()
    assert t.shape == (2, 2)
    assert float(t[1, 1]) == 4.0


def test_to_tensorflow():
    pytest.importorskip("tensorflow")
    a = mp.NdArray([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    t = a.to_tensorflow()
    assert tuple(t.shape) == (2, 2)


def test_to_cupy():
    cupy = pytest.importorskip("cupy")
    a = mp.NdArray([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    t = a.to_cupy()
    assert t.shape == (2, 2)


# --- Table interop ----------------------------------------------------------


def test_table_to_ndarray():
    t = mp.Table({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    a = t.to_ndarray()
    assert a.shape == (3, 2)
    assert a[0, 0] == 1.0
    assert a[2, 1] == 6.0


def test_ndarray_to_table():
    a = mp.NdArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    t = a.to_table()
    assert t.n_rows == 3
    assert t.n_cols == 2


# --- ChunkedNdArray ---------------------------------------------------------


def make_chunked_ndarray():
    a = mp.NdArray([1.0, 2.0, 10.0, 20.0], shape=[2, 2])
    b = mp.NdArray([3.0, 30.0], shape=[1, 2])
    return mp.ChunkedNdArray([a, b], name="readings")


def test_chunked_ndarray_construction_and_access():
    a = make_chunked_ndarray()
    assert a.shape == (3, 2)
    assert a.ndim == 2
    assert a.size == 6
    assert a.dtype == "float64"
    assert a.name == "readings"
    assert a.n_chunks == 2
    assert a.is_view is False
    assert isinstance(a.chunk(0), mp.NdArray)
    assert a.chunk(2) is None
    assert len(a) == 3
    assert a[2, 1] == 30.0
    row = a[1]
    assert isinstance(row, mp.NdArray)
    assert row.shape == (2,)
    assert row[1] == 20.0


def test_chunked_ndarray_rejects_rank_zero_pieces():
    with pytest.raises(ValueError, match="axis 0"):
        mp.ChunkedNdArray([mp.NdArray([5.0], shape=[])])


def test_chunked_ndarray_slice_stays_chunked():
    a = make_chunked_ndarray()
    v = a[1:]
    assert isinstance(v, mp.ChunkedNdArray)
    assert v.shape == (2, 2)
    assert v.n_chunks == 2
    assert v.is_view is True
    assert v[0, 0] == 2.0
    assert v[1, 1] == 30.0
    assert all(isinstance(chunk, mp.NdArray) for chunk in v.chunks)


def test_chunked_ndarray_materialises_in_logical_column_major_order():
    a = make_chunked_ndarray().to_ndarray()
    assert isinstance(a, mp.NdArray)
    assert a.shape == (3, 2)
    assert [a[i, 0] for i in range(3)] == [1.0, 2.0, 3.0]
    assert [a[i, 1] for i in range(3)] == [10.0, 20.0, 30.0]


def test_chunked_ndarray_dlpack_is_per_chunk():
    a = make_chunked_ndarray()
    assert not hasattr(a, "__dlpack__")

    chunks = a.chunks
    assert len(chunks) == 2
    assert all(hasattr(chunk, "__dlpack__") for chunk in chunks)

    arrays = a.to_numpy()
    assert [array.shape for array in arrays] == [(2, 2), (1, 2)]
    assert arrays[0].tolist() == [[1.0, 10.0], [2.0, 20.0]]
    assert arrays[1].tolist() == [[3.0, 30.0]]

    pointers = [array.__array_interface__["data"][0] for array in arrays]
    assert pointers[0] != pointers[1]


# --- XArray ----------------------------------------------------------------


def make_xarray():
    data = mp.NdArray([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], shape=[3, 2])
    return mp.XArray(
        data,
        dims=["time", "feature"],
        coords={
            "time": mp.Array([10, 20, 30]),
            "feature": mp.Array(["x", "y"]),
        },
    )


def test_xarray_construction_and_positional_selection():
    a = make_xarray()
    assert a.shape == (3, 2)
    assert a.dims == ["time", "feature"]
    assert set(a.coords) == {"time", "feature"}
    assert a.data.is_view is False
    assert a[2, 1] == 30.0
    row = a[1]
    assert isinstance(row, mp.XArray)
    assert row.shape == (2,)
    assert row.dims == ["feature"]
    assert row.data.is_view is True
    assert row.data[1] == 20.0


def test_rank_zero_xarray():
    a = mp.XArray(mp.NdArray([5.0], shape=[]), dims=[])
    assert a.shape == ()
    assert a.dims == []
    assert a.coords == {}
    assert a[()] == 5.0
    with pytest.raises(TypeError):
        len(a)


def test_xarray_coordinate_selection():
    a = make_xarray()
    exact = a.sel(time=20)
    assert exact.dims == ["feature"]
    assert exact.data[0] == 2.0
    assert exact.data[1] == 20.0

    window = a.between("time", 15, 30)
    assert window.shape == (2, 2)
    assert window.data[0, 0] == 2.0
    assert window.data[1, 1] == 30.0

    nearest = a.nearest("time", 19)
    assert nearest.dims == ["feature"]
    assert nearest.data[1] == 20.0


def test_xarray_dlpack_exports_data_only():
    a = make_xarray()
    assert '"dltensor_versioned"' in repr(a.__dlpack__(max_version=(1, 1)))
