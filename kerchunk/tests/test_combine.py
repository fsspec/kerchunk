import fsspec
import fsspec.utils
import numpy as np
import dask.array as da
import pytest
import xarray as xr
import zarr

import kerchunk.combine
from kerchunk.zarr import single_zarr
from kerchunk.combine import MultiZarrToZarr

fs = fsspec.filesystem("memory")
arr = np.random.rand(1, 10, 10)

static = xr.DataArray(data=np.random.rand(10, 10), dims=["x", "y"], name="static")
data = xr.DataArray(
    data=arr.squeeze(),
    dims=["x", "y"],
    name="data",
)
xr.Dataset({"data": data}, attrs={"attr0": 3}).to_zarr("memory://simple1.zarr")

data = xr.DataArray(
    data=arr.squeeze() + 1,
    dims=["x", "y"],
    name="data",
)
xr.Dataset({"data": data}, attrs={"attr0": 4}).to_zarr("memory://simple2.zarr")

data = xr.DataArray(
    data=arr.squeeze(),
    dims=["x", "y"],
    name="datum",
)
xr.Dataset({"datum": data}, attrs={"attr0": 3}).to_zarr("memory://simple_var1.zarr")

data = xr.DataArray(
    data=arr.squeeze() + 1,
    dims=["x", "y"],
    name="datum",
)
xr.Dataset({"datum": data}, attrs={"attr0": 4}).to_zarr("memory://simple_var2.zarr")

data = xr.DataArray(
    data=arr,
    coords={"time": np.array([1])},
    dims=["time", "x", "y"],
    name="data",
    attrs={"attr0": 3},
)
xr.Dataset({"data": data, "static": static}, attrs={"attr1": 5}).to_zarr(
    "memory://single1.zarr"
)

data = xr.DataArray(
    data=arr,
    coords={"time": np.array([2])},
    dims=["time", "x", "y"],
    name="data",
    attrs={"attr0": 4},
)
xr.Dataset({"data": data, "static": static}, attrs={"attr1": 6}).to_zarr(
    "memory://single2.zarr"
)

data = xr.DataArray(
    data=arr,
    coords={"time": np.array([3])},
    dims=["time", "x", "y"],
    name="data",
    attrs={"attr0": 4},
)
xr.Dataset({"data": data, "static": static}, attrs={"attr1": 6}).to_zarr(
    "memory://single3.zarr"
)

data = xr.DataArray(
    data=np.vstack([arr] * 4),
    coords={"time": np.array([1, 2, 3, 4])},
    dims=["time", "x", "y"],
    name="data",
    attrs={"attr0": 0},
)
xr.Dataset({"data": data}).to_zarr("memory://quad_nochunk1.zarr")
xr.Dataset({"data": data}).to_zarr("memory://group1.zarr", group="group")

data = xr.DataArray(
    data=np.vstack([arr] * 4),
    coords={"time": np.array([5, 6, 7, 8])},
    dims=["time", "x", "y"],
    name="data",
    attrs={"attr0": 0},
)
xr.Dataset({"data": data}).to_zarr("memory://quad_nochunk2.zarr")
xr.Dataset({"data": data}).to_zarr("memory://group2.zarr", group="group")

data = xr.DataArray(
    data=da.from_array(np.vstack([arr] * 4), chunks=(1, 10, 10)),
    coords={"time": np.array([1, 2, 3, 4])},
    dims=["time", "x", "y"],
    name="data",
    attrs={"attr0": 0},
)
xr.Dataset({"data": data}).to_zarr("memory://quad_1chunk1.zarr")

data = xr.DataArray(
    data=da.from_array(np.vstack([arr] * 4), chunks=(1, 10, 10)),
    coords={"time": np.array([5, 6, 7, 8])},
    dims=["time", "x", "y"],
    name="data",
    attrs={"attr0": 0},
)
xr.Dataset({"data": data}).to_zarr("memory://quad_1chunk2.zarr")

data = xr.DataArray(
    data=da.from_array(np.vstack([arr] * 4), chunks=(2, 10, 10)),
    coords={"time": np.array([1, 2, 3, 4])},
    dims=["time", "x", "y"],
    name="data",
    attrs={"attr0": 0},
)
xr.Dataset({"data": data}).to_zarr("memory://quad_2chunk1.zarr")

data = xr.DataArray(
    data=da.from_array(np.vstack([arr] * 4), chunks=(2, 10, 10)),
    coords={"time": np.array([5, 6, 7, 8])},
    dims=["time", "x", "y"],
    name="data",
    attrs={"attr0": 0},
)
xr.Dataset({"data": data}).to_zarr("memory://quad_2chunk2.zarr")

# simple time arrays - xarray can't make these!
m = fs.get_mapper("time1.zarr")
z = zarr.open(m, mode="w")
ar = z.create_dataset("time", data=np.array([1], dtype="M8[s]"))
ar.attrs.update({"_ARRAY_DIMENSIONS": ["time"]})
ar = z.create_dataset("data", data=arr)
ar.attrs.update({"_ARRAY_DIMENSIONS": ["time", "x", "y"]})

m = fs.get_mapper("time2.zarr")
z = zarr.open(m, mode="w")
ar = z.create_dataset("time", data=np.array([2], dtype="M8[s]"))
ar.attrs.update({"_ARRAY_DIMENSIONS": ["time"]})
ar = z.create_dataset("data", data=arr)
ar.attrs.update({"_ARRAY_DIMENSIONS": ["time", "x", "y"]})


# cftime arrays - standard
tdata1 = xr.DataArray(
    data=arr,
    coords={"time": np.array([1])},
    dims=["time", "x", "y"],
    name="data",
)
xr.Dataset({"data": tdata1}).to_zarr("memory://cfstdtime1.zarr")
fs.pipe(
    "cfstdtime1.zarr/time/.zattrs",
    b'{"_ARRAY_DIMENSIONS": ["time"], "units": "seconds since '
    b'1970-01-01T00:00:00"}',
)

tdata1 = xr.DataArray(
    data=arr,
    coords={"time": np.array([2])},
    dims=["time", "x", "y"],
    name="data",
)
xr.Dataset({"data": tdata1}).to_zarr("memory://cfstdtime2.zarr")
fs.pipe(
    "cfstdtime2.zarr/time/.zattrs",
    b'{"_ARRAY_DIMENSIONS": ["time"], "units": "seconds since '
    b'1970-01-01T00:00:00"}',
)

tdata1 = xr.DataArray(
    data=arr,
    coords={"time": np.array([3])},
    dims=["time", "x", "y"],
    name="data",
)
xr.Dataset({"data": tdata1}).to_zarr("memory://cfstdtime3.zarr")
fs.pipe(
    "cfstdtime3.zarr/time/.zattrs",
    b'{"_ARRAY_DIMENSIONS": ["time"], "units": "seconds since '
    b'1970-01-01T00:00:00"}',
)

# cftime arrays - non standard
tdata1 = xr.DataArray(
    data=arr,
    coords={"time": np.array([1])},
    dims=["time", "x", "y"],
    name="data",
    attrs={"units": "months since 1970-01-01", "calendar": "360_day"},
)
xr.Dataset({"data": tdata1}).to_zarr("memory://cfnontime1.zarr")
fs.pipe(
    "cfnontime1.zarr/time/.zattrs",
    b'{"_ARRAY_DIMENSIONS": ["time"], "units": "months since 1970-01-01", "calendar": "360_day"}',
)

tdata1 = xr.DataArray(
    data=arr,
    coords={"time": np.array([2])},
    dims=["time", "x", "y"],
    name="data",
    attrs={"units": "months since 1970-01-01", "calendar": "360_day"},
)
xr.Dataset({"data": tdata1}).to_zarr("memory://cfnontime2.zarr")
fs.pipe(
    "cfnontime2.zarr/time/.zattrs",
    b'{"_ARRAY_DIMENSIONS": ["time"], "units": "months since 1970-01-01", "calendar": "360_day"}',
)


@pytest.fixture(scope="module")
def refs():
    return {
        path.replace(".zarr", "").lstrip("/"): single_zarr(f"memory://{path}")
        for path in fs.ls("", detail=False)
    }


def test_fixture(refs):
    # effectively checks that single_zarr works
    assert "single1" in refs
    m = fsspec.get_mapper("reference://", fo=refs["single1"], remote_protocol="memory")
    g = xr.open_dataset(m, engine="zarr", backend_kwargs={"consolidated": False})
    assert g.time.values.tolist() == [1]
    assert (g.data.values == arr).all()
    assert g.attrs["attr1"] == 5
    assert g.data.attrs["attr0"] == 3


@pytest.mark.parametrize(
    "dataset,chunks",
    [
        ["quad_nochunk1", ["data/0.0.0"]],
        ["quad_1chunk1", ["data/0.0.0", "data/1.0.0", "data/2.0.0", "data/3.0.0"]],
        ["quad_2chunk1", ["data/0.0.0", "data/1.0.0"]],
    ],
)
def test_fixture_chunks(refs, dataset, chunks):
    # checks that the right number of chunks are made per dataset
    fs = fsspec.filesystem("reference", fo=refs[dataset], remote_protocol="memory")
    out = fs.ls("data", detail=False)
    assert out == ["data/.zarray", "data/.zattrs"] + chunks
    out = fs.ls("time", detail=False)
    assert out == ["time/.zarray", "time/.zattrs", "time/0"]


@pytest.mark.parametrize(
    "selector,expected",
    [
        ["data:time", [1, 2]],
        [[9, 10], [9, 10]],
        ["INDEX", [0, 1]],
        # [re.compile("simple(\\d)"), [1, 2]],  # we don't give path names here
        ["attr:attr1", [5, 6]],
        ["vattr:data:attr0", [3, 4]],
    ],
)
def test_get_coos(refs, selector, expected):
    mzz = MultiZarrToZarr(
        [refs["single1"], refs["single2"]],
        remote_protocol="memory",
        concat_dims=["time"],
        coo_map={"time": selector},
    )
    mzz.first_pass()
    assert mzz.coos["time"].tolist() == expected
    mzz.store_coords()
    g = zarr.open(mzz.out)
    assert g["time"][:].tolist() == expected
    assert dict(g.attrs)


def test_coo_vars(refs):
    mzz = MultiZarrToZarr(
        [refs["simple1"], refs["simple_var1"]],
        remote_protocol="memory",
        concat_dims=["var"],
    )
    mzz.first_pass()
    assert mzz.coos["var"].tolist() == ["data", "datum"]

    mzz = MultiZarrToZarr(
        [refs["simple1"], refs["simple2"], refs["simple_var1"], refs["simple_var2"]],
        remote_protocol="memory",
        concat_dims=["var", "time"],
        coo_map={"time": "attr:attr0"},
    )
    mzz.first_pass()
    assert mzz.coos["var"].tolist() == ["data", "datum"]
    assert mzz.coos["time"].tolist() == [3, 4]


def test_single(refs):
    mzz = MultiZarrToZarr(
        [refs["single1"], refs["single2"]],
        remote_protocol="memory",
        concat_dims=["time"],
        coo_dtypes={"time": "int16"},
    )
    mzz.first_pass()
    mzz.store_coords()
    mzz.second_pass()
    out = kerchunk.combine.consolidate(mzz.out)
    z = xr.open_dataset(
        "reference://",
        backend_kwargs={
            "storage_options": {"fo": out, "remote_protocol": "memory"},
            "consolidated": False,
        },
        engine="zarr",
        decode_cf=False,
    )
    # TODO: make some assert_eq style function
    assert (
        z.time.dtype == "int16"
    )  # default is int64, or float64 if decode_cf is True because or array special name
    assert z.time.values.tolist() == [1, 2]
    assert z.data.shape == (2, 10, 10)
    assert (z.data[0].values == arr).all()
    assert (z.data[1].values == arr).all()


def test_single_append(refs):
    mzz = MultiZarrToZarr(
        [refs["single1"], refs["single2"]],
        remote_protocol="memory",
        concat_dims=["time"],
        coo_dtypes={"time": "int16"},
    )
    out = mzz.translate()
    mzz = MultiZarrToZarr.append(
        [refs["single3"]],
        out,
        remote_protocol="memory",
        concat_dims=["time"],
        coo_dtypes={"time": "int16"},
    )
    out = mzz.translate()
    z = xr.open_dataset(
        "reference://",
        backend_kwargs={
            "storage_options": {"fo": out, "remote_protocol": "memory"},
            "consolidated": False,
        },
        engine="zarr",
        decode_cf=False,
    )
    assert z.data.shape == (3, 10, 10)
    assert out["refs"]["data/1.0.0"] == ["memory:///single2.zarr/data/0.0.0"]
    assert out["refs"]["data/2.0.0"] == ["memory:///single3.zarr/data/0.0.0"]
    assert z.time.values.tolist() == [1, 2, 3]


@pytest.mark.parametrize("mapper", [{}, {"time": "cf:time"}])
@pytest.mark.parametrize("dtype", [{"time": "M8[s]"}, {}])
def test_single_append_cf(refs, mapper, dtype):
    mzz = MultiZarrToZarr(
        [refs["cfstdtime1"], refs["cfstdtime2"]],
        remote_protocol="memory",
        concat_dims=["time"],
        coo_map=mapper,
        coo_dtypes=dtype,
    )
    out = mzz.translate()
    mzz = MultiZarrToZarr.append(
        [refs["cfstdtime3"]],
        out,
        remote_protocol="memory",
        concat_dims=["time"],
        coo_map=mapper,
        coo_dtypes=dtype,
    )
    out = mzz.translate()
    z = xr.open_dataset(
        "reference://",
        backend_kwargs={
            "storage_options": {"fo": out, "remote_protocol": "memory"},
            "consolidated": False,
        },
        engine="zarr",
    )
    assert z.data.shape == (3, 10, 10)
    assert out["refs"]["data/0.0.0"] == ["memory:///cfstdtime1.zarr/data/0.0.0"]
    assert out["refs"]["data/1.0.0"] == ["memory:///cfstdtime2.zarr/data/0.0.0"]
    assert out["refs"]["data/2.0.0"] == ["memory:///cfstdtime3.zarr/data/0.0.0"]
    np.testing.assert_equal(
        z.time.values,
        np.array(
            [
                "1970-01-01T00:00:01.000000000",
                "1970-01-01T00:00:02.000000000",
                "1970-01-01T00:00:03.000000000",
            ],
            dtype="datetime64[ns]",
        ),
    )


def test_single_append_parquet(refs):
    from fsspec.implementations.reference import LazyReferenceMapper

    m = fsspec.filesystem("memory")
    out = LazyReferenceMapper.create("memory://refs/out.parquet", fs=m)
    mzz = MultiZarrToZarr(
        [refs["single1"], refs["single2"]],
        remote_protocol="memory",
        concat_dims=["time"],
        coo_dtypes={"time": "int16"},
        out=out,
    )
    mzz.translate()

    # reload here due to unknown bug after flush
    out = LazyReferenceMapper("memory://refs/out.parquet", fs=m)
    mzz = MultiZarrToZarr.append(
        [refs["single3"]],
        out,
        remote_protocol="memory",
        concat_dims=["time"],
        coo_dtypes={"time": "int16"},
    )
    out = mzz.translate()

    z = xr.open_dataset(
        out,
        backend_kwargs={
            "storage_options": {"remote_protocol": "memory"},
        },
        engine="kerchunk",
        decode_cf=False,
    )
    assert z.data.shape == (3, 10, 10)
    assert out["data/1.0.0"] == ["memory:///single2.zarr/data/0.0.0"]
    assert out["data/2.0.0"] == ["memory:///single3.zarr/data/0.0.0"]
    assert z.time.values.tolist() == [1, 2, 3]


def test_lazy_filler(tmpdir, refs):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("fastparquet")
    from fsspec.implementations.reference import LazyReferenceMapper

    fs = fsspec.filesystem("file")
    tmpdir = str(tmpdir)
    out = LazyReferenceMapper.create(record_size=10, root=tmpdir, fs=fs)

    mzz = MultiZarrToZarr(
        [refs["single1"], refs["single2"]],
        remote_protocol="memory",
        concat_dims=["time"],
        coo_dtypes={"time": "int16"},
        out=out,
    )
    mzz.first_pass()
    mzz.store_coords()

    out.flush()
    allfiles = fs.find(tmpdir)
    assert allfiles == [f"{tmpdir}/{a}" for a in [".zmetadata", "time/refs.0.parq"]]
    df = pd.read_parquet(f"{tmpdir}/time/refs.0.parq", engine="fastparquet")
    assert df.iloc[:1].to_dict() == {
        "raw": {0: b"\x01\x00\x02\x00"},
        "offset": {0: 0},
        "size": {0: 0},
        "path": {0: None},
    }
    assert out["time/0"] == b"\x01\x00\x02\x00"
    mzz.second_pass()

    # actual references don't show
    assert set(out) == {
        ".zattrs",
        ".zgroup",
        ".zmetadata",
        "data/.zarray",
        "data/.zattrs",
        "static/.zarray",
        "static/.zattrs",
        "time/.zarray",
        "time/.zattrs",
    }
    out.flush()
    assert set(out) == {
        ".zattrs",
        ".zgroup",
        ".zmetadata",
        "data/.zarray",
        "data/.zattrs",
        "static/.zarray",
        "static/.zattrs",
        "time/.zarray",
        "time/.zattrs",
    }
    allfiles = fs.find(tmpdir)
    assert [
        f"{tmpdir}/{a}" in allfiles for a in ["static/refs.0.parq", "data/refs.0.parq"]
    ]

    z = xr.open_dataset(
        "reference://",
        backend_kwargs={
            "storage_options": {"fo": tmpdir, "remote_protocol": "memory"},
            "consolidated": False,
        },
        engine="zarr",
        decode_cf=False,
    )
    assert z.time.values.tolist() == [1, 2]
    assert z.data.shape == (2, 10, 10)
    assert (z.data[0].values == arr).all()
    assert (z.data[1].values == arr).all()


def test_run_twice(refs):
    mzz = MultiZarrToZarr(
        [refs["single1"], refs["single2"]],
        remote_protocol="memory",
        concat_dims=["time"],
        coo_dtypes={"time": "int16"},
    )
    mzz.translate()
    out = mzz.translate()
    z = xr.open_dataset(
        "reference://",
        backend_kwargs={
            "storage_options": {"fo": out, "remote_protocol": "memory"},
            "consolidated": False,
        },
        engine="zarr",
        decode_cf=False,
    )
    assert (
        z.time.dtype == "int16"
    )  # default is int64, or float64 if decode_cf is True because or array special name
    assert z.time.values.tolist() == [1, 2]
    assert z.data.shape == (2, 10, 10)
    assert (z.data[0].values == arr).all()
    assert (z.data[1].values == arr).all()


def test_outfile_postprocess(refs):
    def post_process(ref):
        # renamed "data" array to "a_data"; to rename a coordinate, one would nee
        # to alter attributes of arrays pointing to it
        return {("a_" + k if k.startswith("data") else k): v for k, v in ref.items()}

    mzz = MultiZarrToZarr(
        [refs["single1"], refs["single2"]],
        remote_protocol="memory",
        concat_dims=["time"],
        postprocess=post_process,
    )
    mzz.translate("memory://myout.json")
    z = xr.open_dataset(
        "reference://",
        backend_kwargs={
            "storage_options": {
                "fo": "memory://myout.json",
                "remote_protocol": "memory",
            },
            "consolidated": False,
        },
        engine="zarr",
    )
    assert z.time.values.tolist() == [1, 2]
    assert z.a_data.shape == (2, 10, 10)
    assert (z.a_data[0].values == arr).all()
    assert (z.a_data[1].values == arr).all()


@pytest.mark.parametrize(
    "inputs, chunks",
    [
        [["quad_nochunk1", "quad_nochunk2"], ((4, 4), (10,), (10,))],
        [["quad_1chunk1", "quad_1chunk2"], ((1,) * 8, (10,), (10,))],
        [["quad_2chunk1", "quad_2chunk2"], ((2,) * 4, (10,), (10,))],
        [["group1", "group2"], ((4, 4), (10,), (10,))],
    ],
)
def test_chunked(refs, inputs, chunks):
    mzz = MultiZarrToZarr(
        [refs[inputs[0]], refs[inputs[1]]],
        remote_protocol="memory",
        concat_dims=["time"],
        coo_map={"time": "data:group/time"} if "group" in inputs[0] else None,
    )
    out = mzz.translate()
    z = xr.open_dataset(
        "reference://",
        backend_kwargs={
            "storage_options": {"fo": out, "remote_protocol": "memory"},
            "consolidated": False,
        },
        engine="zarr",
        chunks={},
        group="group" if "group" in inputs[0] else None,
    )
    # TODO: make some assert_eq style function
    assert z.time.values.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert z.data.shape == (8, 10, 10)
    assert z.data.chunks == chunks
    for i in range(z.data.shape[0]):
        assert (z.data[i].values == arr).all()


def test_var(refs):
    mzz = MultiZarrToZarr(
        [refs["simple1"], refs["simple_var1"]],
        remote_protocol="memory",
        concat_dims=["var"],
    )
    out = mzz.translate()
    z = xr.open_dataset(
        "reference://",
        backend_kwargs={
            "storage_options": {"fo": out, "remote_protocol": "memory"},
            "consolidated": False,
        },
        engine="zarr",
        chunks={},
    )
    assert list(z) == ["data", "datum"]
    assert list(z.dims) == ["x", "y"]
    assert z.data.shape == z.datum.shape == (10, 10)
    assert (z.data.values == arr).all()
    assert (z.datum.values == arr).all()


def test_var_dimension(refs):
    mzz = MultiZarrToZarr(
        [refs["simple1"], refs["simple_var1"]],
        remote_protocol="memory",
        coo_map={"varname": "VARNAME", "var": "output"},
    )
    out = mzz.translate()
    z = xr.open_dataset(
        "reference://",
        backend_kwargs={
            "storage_options": {"fo": out, "remote_protocol": "memory"},
            "consolidated": False,
        },
        engine="zarr",
        chunks={},
    )
    assert list(z) == ["output"]
    assert list(z.dims) == ["varname", "x", "y"]
    assert list(z.varname) == ["data", "datum"]
    assert z.output.shape == (2, 10, 10)
    assert (z.output.values == arr).all()


def test_var_and_dim(refs):
    mzz = MultiZarrToZarr(
        [refs["simple1"], refs["simple2"], refs["simple_var1"], refs["simple_var2"]],
        remote_protocol="memory",
        concat_dims=["var", "dim"],
        coo_map={"dim": "attr:attr0"},
    )
    out = mzz.translate()
    z = xr.open_dataset(
        "reference://",
        backend_kwargs={
            "storage_options": {"fo": out, "remote_protocol": "memory"},
            "consolidated": False,
        },
        engine="zarr",
        chunks={},
    )
    assert list(z) == ["data", "datum"]
    assert list(z.dims) == ["dim", "x", "y"]
    assert z.data.shape == z.datum.shape == (2, 10, 10)
    assert z["dim"].values.tolist() == [3, 4]
    assert (z.data.values == np.vstack([arr, arr + 1])).all()


@pytest.mark.parametrize(
    "inps,expected",
    [
        ["time", ["1970-01-01T00:00:01", "1970-01-01T00:00:02"]],
        ["cfstdtime", ["1970-01-01T00:00:01", "1970-01-01T00:00:02"]],
        ["cfnontime", ["1970-02-01T00:00:00", "1970-03-01T00:00:00"]],
    ],
)
def test_times(refs, inps, expected):
    mapper = "data:time" if inps == "time" else "cf:time"
    mzz = MultiZarrToZarr(
        [refs[inps + "1"], refs[inps + "2"]],
        remote_protocol="memory",
        concat_dims=["time"],
        coo_map={"time": mapper},
    )
    out = mzz.translate()
    z = xr.open_dataset(
        "reference://",
        backend_kwargs={
            "storage_options": {"fo": out, "remote_protocol": "memory"},
            "consolidated": False,
        },
        engine="zarr",
        chunks={},
    )
    if inps == "cfnontime":
        assert [_.isoformat() for _ in z.time.values] == expected
    else:
        assert (z.time == np.array(expected, dtype="M8[ms]")).all()


def test_cftimes_to_normal(refs):
    mzz = MultiZarrToZarr(
        [refs["cfnontime1"], refs["cfnontime2"]],
        remote_protocol="memory",
        concat_dims=["time"],
        coo_map={"time": "cf:time"},
        coo_dtypes={"time": "M8[s]"},
    )
    out = mzz.translate()
    z = xr.open_dataset(
        "reference://",
        backend_kwargs={
            "storage_options": {"fo": out, "remote_protocol": "memory"},
            "consolidated": False,
        },
        engine="zarr",
        chunks={},
    )
    assert z.time.dtype == "M8[s]"
    assert (
        z.time.values
        == np.array(["1970-02-01T00:00:00", "1970-03-01T00:00:00"], dtype="M8[s]")
    ).all()


def test_inline(refs):
    mzz = MultiZarrToZarr(
        [refs["single1"], refs["single2"]],
        remote_protocol="memory",
        concat_dims=["time"],
        coo_dtypes={"time": "int16"},
        inline_threshold=50000,
    )
    out = mzz.translate()
    z = xr.open_dataset(
        "reference://",
        backend_kwargs={
            "storage_options": {"fo": out, "remote_protocol": "memory"},
            "consolidated": False,
        },
        engine="zarr",
        decode_cf=False,
    )
    ref = fsspec.filesystem("reference", fo=out, remote_protocol="memory")
    assert (
        z.time.dtype == "int16"
    )  # default is int64, or float64 if decode_cf is True because or array special name
    assert z.time.values.tolist() == [1, 2]
    assert z.data.shape == (2, 10, 10)
    assert (z.data[0].values == arr).all()
    assert (z.data[1].values == arr).all()

    assert isinstance(ref.references["data/0.0.0"], str)
    assert ref.references["data/0.0.0"].startswith("base64:")


def test_merge_vars():
    a = dict({"version": 1, "refs": dict({"item1": 1})})
    b = dict({"version": 1, "refs": dict({"item2": 2})})
    merge = kerchunk.combine.merge_vars([a, b])
    assert list(merge["refs"]) == ["item1", "item2"]
    fs = fsspec.filesystem("memory")
    fs.pipe("file1.json", b"""{"version": 1, "refs": {"item1": 1}}""")
    fs.pipe("file2.json", b"""{"version": 1, "refs": {"item2": 2}}""")
    merge = kerchunk.combine.merge_vars(["memory://file1.json", "memory://file2.json"])
    assert list(merge["refs"]) == ["item1", "item2"]


def test_bad_coo_warning(refs):
    def f(*_, **__):
        return 1

    mzz = MultiZarrToZarr(
        [refs["single1"], refs["single2"]],
        remote_protocol="memory",
        concat_dims=["time"],
        coo_map={"time": f},
    )
    with pytest.warns(match="contains less than expected"):
        mzz.first_pass()


def test_chunk_error(refs):
    refs2 = refs["single1"].copy()
    refs2.pop(".zmetadata")
    fs = fsspec.filesystem("reference", fo=refs2, remote_protocol="memory")
    refs2[
        "data/.zarray"
    ] = b"""
    {
    "chunks": [2, 3, 3],
    "zarr_format": 2,
    "shape": [1, 10, 10]
    }
    """
    mzz = MultiZarrToZarr(
        [refs["single1"], refs2],
        remote_protocol="memory",
        concat_dims=["time"],
    )
    with pytest.raises(ValueError) as e:
        mzz.translate()
    assert "chunk size mismatch" in str(e.value)
