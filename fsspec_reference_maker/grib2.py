import base64
import logging
import os
import tempfile

import cfgrib
import numcodecs.abc
from numcodecs.compat import ndarray_copy, ensure_contiguous_ndarray
import fsspec
import zarr
import numpy as np

logger = logging.getLogger("grib2-to-zarr")


def _split_file(f, skip=0):
    if hasattr(f, "size"):
        size = f.size
    else:
        f.seek(0, 2)
        size = f.tell()
        f.seek(0)
    part = 0

    while f.tell() < size:
        logger.debug(f"extract part {part}")
        start = f.tell()
        f.seek(12, 1)
        part_size = int.from_bytes(f.read(4), "big")
        f.seek(start)
        data = f.read(part_size)
        assert data[:4] == b"GRIB"
        assert data[-4:] == b"7777"
        fn = tempfile.mktemp(suffix="grib2")
        with open(fn, "wb") as fo:
            fo.write(data)
        yield fn, start, part_size
        part += 1
        if skip and part > skip:
            break


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
    if filter:
        assert "typeOfLevel" in filter and "level" in filter
    logger.debug(f"Open {url}")

    store = {}
    z = zarr.open_group(store, mode='w')
    common = False
    with fsspec.open(url, "rb", **storage_options) as f:
        for fn, offset, size in _split_file(f, skip=skip):
            logger.debug(f"File {fn}")
            ds = cfgrib.open_file(fn)
            if filter:
                var = filter["typeOfLevel"]
                if var not in ds.variables:
                    continue
                if ds.variables[var].data != filter["level"]:
                    continue
                if common is False:
                    # done for first valid message
                    logger.debug("Common variables")
                    z.attrs.update(ds.attributes)
                    for var in common_vars:
                        # assume grid, etc is the same across all messages
                        attr = ds.variables[var].attributes or {}
                        attr['_ARRAY_DIMENSIONS'] = ds.variables[var].dimensions
                        _store_array(store, z, ds.variables[var].data, var, inline_threashold, offset, size,
                                     attr)
                    common = True
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


class GRIBCodec(numcodecs.abc.Codec):
    """
    Read GRIB stream of bytes by writing to a temp file and calling cfgrib
    """

    codec_id = 'grib'

    def __init__(self, var):
        self.var = var

    def encode(self, buf):
        # on encode, pass through
        return buf

    def decode(self, buf, out=None):
        buf = ensure_contiguous_ndarray(buf)
        fn = tempfile.mktemp(suffix="grib2")
        buf.tofile(fn)

        # do decode
        ds = cfgrib.open_file(fn)
        data = ds.variables[self.var].data
        if hasattr(data, "build_array"):
            data = data.build_array()

        if out is not None:
            return ndarray_copy(data, out)
        else:
            return data


numcodecs.register_codec(GRIBCodec, "grib")


def example_multi():
    import json
    # 1GB of data files, forming a time-series
    files = ['s3://noaa-hrrr-bdp-pds/hrrr.20190101/conus/hrrr.t22z.wrfsfcf01.grib2',
             's3://noaa-hrrr-bdp-pds/hrrr.20190101/conus/hrrr.t23z.wrfsfcf01.grib2',
             's3://noaa-hrrr-bdp-pds/hrrr.20190102/conus/hrrr.t00z.wrfsfcf01.grib2',
             's3://noaa-hrrr-bdp-pds/hrrr.20190102/conus/hrrr.t01z.wrfsfcf01.grib2',
             's3://noaa-hrrr-bdp-pds/hrrr.20190102/conus/hrrr.t02z.wrfsfcf01.grib2',
             's3://noaa-hrrr-bdp-pds/hrrr.20190102/conus/hrrr.t03z.wrfsfcf01.grib2',
             's3://noaa-hrrr-bdp-pds/hrrr.20190102/conus/hrrr.t04z.wrfsfcf01.grib2',
             's3://noaa-hrrr-bdp-pds/hrrr.20190102/conus/hrrr.t05z.wrfsfcf01.grib2',
             's3://noaa-hrrr-bdp-pds/hrrr.20190102/conus/hrrr.t06z.wrfsfcf01.grib2']
    so = {"anon": True, "default_cache_type": "readahead"}
    common = ['time', 'step', 'latitude', 'longitude', 'valid_time']
    filter = {'typeOfLevel': 'heightAboveGround', 'level': 2}
    for url in files:
        out = scan_grib(url, common, so, inline_threashold=100, filter=filter)
        with open(os.path.basename(url).replace("grib2", "json"), "w") as f:
            json.dump(out, f)

    # stitch with
    # mzz = MultiZarrToZarr("hrrr.*.json", remote_protocol="s3", remote_options={"anon": True}, with_mf=False)
    # mzz.translate("hrrr.total.json")
