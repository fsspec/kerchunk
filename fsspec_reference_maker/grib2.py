import base64
import logging
import os
import tempfile
try:
    import numcodecs.abc
    upclass = numcodecs.abc.Codec
except:
    upclass = object
logger = logging.getLogger("grib2-to-zarr")


def _split_file(f, pathspec="part.{}.grib", skip=0):
    if hasattr(f, "size"):
        size = f.size
    else:
        f.seek(0, 2)
        size = f.tell()
        f.seek(0)
    part = 0
    out = []
    offsets = []
    sizes = []

    while f.tell() < size:
        # TODO: move filtering to here
        logger.debug(f"extract part {part}")
        start = f.tell()
        offsets.append(start)
        f.seek(12, 1)
        part_size = int.from_bytes(f.read(4), "big")
        sizes.append(part_size)
        f.seek(start)
        data = f.read(part_size)
        assert data[:4] == b"GRIB"
        assert data[-4:] == b"7777"
        fn = pathspec.format(part)
        with open(fn, "wb") as fo:
            fo.write(data)
        out.append(fn)
        part += 1
        if skip and part > skip:
            break
    return out, offsets, sizes


def _store_array(store, z, data, var, inline_threshold, offset, size, attr):
    nbytes = data.dtype.itemsize
    for i in data.shape:
        nbytes *= i

    shape = tuple(data.shape or ())
    if nbytes < inline_threshold:
        logger.debug(f"Store {var} inline")
        d = z.create_dataset(
            name=var,
            shape=shape,
            chunks=shape,
            dtype=data.dtype,
            fill_value=getattr(data, "missing_value", 0),
            compressor=False,
        )
        if hasattr(data, "tobytes"):
            b = data.tobytes()
        else:
            b = data.build_array().tobytes()
        try:
            # easiest way to test if data is ascii
            b.decode('ascii')
        except UnicodeDecodeError:
            b = b"base64:" + base64.b64encode(data)
        store[f"{var}/0"] = b.decode('ascii')
    else:
        logger.debug(f"Store {var} reference")
        d = z.create_dataset(
            name=var,
            shape=shape,
            chunks=shape,
            dtype=data.dtype,
            fill_value=getattr(data, "missing_value", 0),
            filters=[GRIBCodec(var=var)],
            compressor=False,
        )
        store[f"{var}/" + ".".join(["0"] * len(shape))] = ["{{u}}", offset, size]
    d.attrs.update(attr)


def scan_grib(url, common_vars, storage_options, inline_threashold=100, skip=0, filter={}):
    import fsspec
    import zarr
    import cfgrib
    import numpy as np
    d = tempfile.mkdtemp()
    if filter:
        assert "typeOfLevel" in filter and "level" in filter
    logger.debug(f"Open {url}")
    with fsspec.open(url, "rb", **storage_options) as f:
        allfiles, offsets, sizes = _split_file(f, os.path.join(d, "part.{}.grib2"), skip=skip)

    store = {}
    z = zarr.open_group(store, mode='w')
    ds = cfgrib.open_file(allfiles[0])
    z.attrs.update(ds.attributes)
    # logger.debug("Dimensions")
    # for name in ds.dimensions:
    #     logger.debug(f"Dimension {name}")
    #     arr = np.arange(ds.dimensions[name], dtype='int32')
    #     z.create_dataset(name, dtype="int32", shape=arr.shape, compressor=False, chunks=False)
    #     store[f"{name}/0"] = (b"base64:" + base64.b64encode(arr.tobytes())).decode()
    logger.debug("Common variables")
    for var in common_vars:
        # assume grid, etc is the same across all messages
        attr = ds.variables[var].attributes or {}
        attr['_ARRAY_DIMENSIONS'] = ds.variables[var].dimensions
        _store_array(store, z, ds.variables[var].data, var, inline_threashold, 0, sizes[0],
                     attr)
    for fn, offset, size in zip(allfiles, offsets, sizes):
        logger.debug(f"File {fn}")
        ds = cfgrib.open_file(fn)
        if filter:
            var = filter["typeOfLevel"]
            if var not in ds.variables:
                continue
            if ds.variables[var].data != filter["level"]:
                continue
            if var not in z:
                attr = ds.variables[var].attributes or {}
                attr['_ARRAY_DIMENSIONS'] = []
                _store_array(store, z, np.array(filter["level"]), var, 100000, 0, 0,
                             attr)

        for var in ds.variables:
            if var not in common_vars and getattr(ds.variables[var].data, "shape", None):

                attr = ds.variables[var].attributes or {}
                attr['_ARRAY_DIMENSIONS'] = ds.variables[var].dimensions
                _store_array(store, z, ds.variables[var].data, var, inline_threashold, offset, size,
                     attr)
    logger.debug("Done")
    return {"version": 1,
            "refs": {k: v.decode() if isinstance(v, bytes) else v for k, v in store.items()},
            "templates": {"u": url}}


class GRIBCodec(upclass):
    """
    """

    codec_id = 'grib'

    def __init__(self, var):
        self.var = var

    def encode(self, buf):
        raise NotImplementedError("Read-only codec")

    def decode(self, buf, out=None):
        import cfgrib
        from numcodecs.compat import ndarray_copy, ensure_contiguous_ndarray

        # normalise inputs
        buf = ensure_contiguous_ndarray(buf)
        fn = tempfile.mktemp(suffix="grib2")
        buf.tofile(fn)

        # do decompression
        ds = cfgrib.open_file(fn)
        data = ds.variables[self.var].data.build_array()

        if out is not None:
            return ndarray_copy(data, out)
        else:
            return data


if upclass is not object:
    numcodecs.register_codec(GRIBCodec, "grib")


def example():
    import json
    url = "s3://noaa-hrrr-bdp-pds/hrrr.20190101/conus/hrrr.t10z.wrfsfcf01.grib2"
    so = {"anon": True, "default_cache_type": "readahead"}
    common = ['time', 'step', 'latitude', 'longitude', 'valid_time']
    filter = {'typeOfLevel': 'heightAboveGround', 'level': 2}
    out = scan_grib(url, common, so, inline_threashold=100, filter=filter)
    with open("hrrr.json", "w") as f:
        json.dump(out, f)
