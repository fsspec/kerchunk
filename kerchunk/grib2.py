import base64
import logging
import os

import cfgrib
import eccodes
import numcodecs.abc
from numcodecs.compat import ndarray_copy
import fsspec
import zarr
import numpy as np

from kerchunk.utils import class_factory

logger = logging.getLogger("grib2-to-zarr")
fsspec.utils.setup_logging(logger=logger)


def _split_file(f, skip=0):
    if hasattr(f, "size"):
        size = f.size
    else:
        f.seek(0, 2)
        size = f.tell()
        f.seek(0)
    part = 0

    while f.tell() < size:
        logger.debug(f"extract part {part + 1}")
        start = f.tell()
        head = f.read(16)
        marker = head[:4]
        if not marker:
            break  # EOF
        assert head[:4] == b"GRIB", "Bad grib message start marker"
        part_size = int.from_bytes(head[12:], "big")
        f.seek(start)
        yield start, part_size, f.read(part_size)
        part += 1
        if skip and part >= skip:
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
            b.decode("ascii")
        except UnicodeDecodeError:
            b = b"base64:" + base64.b64encode(data)
        store[f"{var}/0"] = b.decode("ascii")
    else:
        logger.debug(f"Store {var} reference")
        d = z.create_dataset(
            name=var,
            shape=shape,
            chunks=shape,
            dtype=data.dtype,
            fill_value=getattr(data, "missing_value", 0),
            filters=[GRIBCodec(var=var, dtype=str(data.dtype))],
            compressor=False,
            overwrite=True,
        )
        store[f"{var}/" + ".".join(["0"] * len(shape))] = ["{{u}}", offset, size]
    d.attrs.update(attr)


def scan_grib(
    url,
    storage_options=None,
    inline_threashold=100,
    skip=0,
    filter={},
):
    """
    Generate references for a GRIB2 file

    Parameters
    ----------

    url: str
        File location
    common_vars: list[str]
        Names of variables that are common to multiple measurable (i.e., coordinates)
    storage_options: dict
        For accessing the data, passed to filesystem
    inline_threashold: int
        If given, store array data smaller than this value directly in the output
    skip: int
        If non-zero, stop processing the file after this many messages
    filter: dict
        cfgrib-style filter dictionary

    Returns
    -------

    dict: references dict in Version 1 format.
    """
    storage_options = storage_options or {}
    if filter:
        assert "typeOfLevel" in filter
    logger.debug(f"Open {url}")

    out = []
    with fsspec.open(url, "rb", **storage_options) as f:
        logger.debug(f"File {url}")
        for offset, size, data in _split_file(f, skip=skip):
            store = {}
            mid = eccodes.codes_new_from_message(data)
            m = cfgrib.cfmessage.CfMessage(mid)

            if filter:
                if filter["typeOfLevel"] != m.get("typeOfLevel", True):
                    continue
                if float(filter["level"]) != float(m.get("level", -99)):
                    continue

            z = zarr.open_group(store)
            global_attrs = {k: m[k] for k in cfgrib.dataset.GLOBAL_ATTRIBUTES_KEYS}
            z.attrs.update(global_attrs)

            vals = m["values"].reshape((m["Ny"], m["Nx"]))
            attrs = {
                k: m[k]
                for k in cfgrib.dataset.DATA_ATTRIBUTES_KEYS
                + cfgrib.dataset.DATA_TIME_KEYS
                + cfgrib.dataset.EXTRA_DATA_ATTRIBUTES_KEYS
                if k in m
            }
            _store_array(
                store, z, vals, m["shortName"], inline_threashold, offset, size, attrs
            )
            dims = (
                ["x", "y"]
                if m["gridType"] in cfgrib.dataset.GRID_TYPES_2D_NON_DIMENSION_COORDS
                else ["longitude", "latitude"]
            )
            z[m["shortName"]].attrs["_ARRAY_DIMENSIONS"] = dims
            if filter:
                z[m["shortName"]].attrs[filter["typeOfLevel"]] = m["level"]

            for coord in cfgrib.dataset.COORD_ATTRS:
                coord2 = {"latitude": "latitudes", "longitude": "longitudes"}.get(
                    coord, coord
                )
                if coord2 in m:
                    x = m[coord2]
                else:
                    continue
                if isinstance(x, np.ndarray) and x.size == vals.size:
                    x = x.reshape(vals.shape)
                    if (
                        m["gridType"]
                        in cfgrib.dataset.GRID_TYPES_2D_NON_DIMENSION_COORDS
                    ):
                        dims = ["x", "y"]
                    else:
                        dims = [coord]
                else:
                    x = np.array([x])
                    dims = [coord]
                attrs = cfgrib.dataset.COORD_ATTRS[coord]
                _store_array(store, z, x, coord, inline_threashold, offset, size, attrs)
                z[coord].attrs["_ARRAY_DIMENSIONS"] = dims

            out.append(
                {
                    "version": 1,
                    "refs": {
                        k: v.decode() if isinstance(v, bytes) else v
                        for k, v in store.items()
                    },
                    "templates": {"u": url},
                }
            )
    logger.debug("Done")
    return out


GribToZarr = class_factory(scan_grib)


class GRIBCodec(numcodecs.abc.Codec):
    """
    Read GRIB stream of bytes by writing to a temp file and calling cfgrib
    """

    codec_id = "grib"

    def __init__(self, var, dtype=None):
        self.var = var
        self.dtype = dtype

    def encode(self, buf):
        # on encode, pass through
        return buf

    def decode(self, buf, out=None):
        if self.var in ["latitude", "longitude"]:
            var = self.var + "s"
            dt = self.dtype or "float64"
        else:
            var = "values"
            dt = self.dtype or "float32"
        mid = eccodes.codes_new_from_message(bytes(buf))
        try:
            data = eccodes.codes_get_array(mid, var)
        finally:
            eccodes.codes_release(mid)

        if out is not None:
            return ndarray_copy(data, out)
        else:
            return data.astype(dt)


numcodecs.register_codec(GRIBCodec, "grib")


def example_multi(filter={"typeOfLevel": "heightAboveGround", "level": 2}):
    import json

    # 1GB of data files, forming a time-series
    files = [
        "s3://noaa-hrrr-bdp-pds/hrrr.20190101/conus/hrrr.t22z.wrfsfcf01.grib2",
        "s3://noaa-hrrr-bdp-pds/hrrr.20190101/conus/hrrr.t23z.wrfsfcf01.grib2",
        "s3://noaa-hrrr-bdp-pds/hrrr.20190102/conus/hrrr.t00z.wrfsfcf01.grib2",
        "s3://noaa-hrrr-bdp-pds/hrrr.20190102/conus/hrrr.t01z.wrfsfcf01.grib2",
        "s3://noaa-hrrr-bdp-pds/hrrr.20190102/conus/hrrr.t02z.wrfsfcf01.grib2",
        "s3://noaa-hrrr-bdp-pds/hrrr.20190102/conus/hrrr.t03z.wrfsfcf01.grib2",
        "s3://noaa-hrrr-bdp-pds/hrrr.20190102/conus/hrrr.t04z.wrfsfcf01.grib2",
        "s3://noaa-hrrr-bdp-pds/hrrr.20190102/conus/hrrr.t05z.wrfsfcf01.grib2",
        "s3://noaa-hrrr-bdp-pds/hrrr.20190102/conus/hrrr.t06z.wrfsfcf01.grib2",
    ]
    so = {"anon": True, "default_cache_type": "readahead"}
    common = ["time", "step", "latitude", "longitude", "valid_time"]
    for url in files:
        out = scan_grib(url, common, so, inline_threashold=100, filter=filter)
        with open(os.path.basename(url).replace("grib2", "json"), "w") as f:
            json.dump(out, f)


def example_combine():
    from kerchunk.combine import MultiZarrToZarr

    files = [
        "hrrr.t22z.wrfsfcf01.json",
        "hrrr.t23z.wrfsfcf01.json",
        "hrrr.t00z.wrfsfcf01.json",
        "hrrr.t01z.wrfsfcf01.json",
        "hrrr.t02z.wrfsfcf01.json",
        "hrrr.t03z.wrfsfcf01.json",
        "hrrr.t04z.wrfsfcf01.json",
        "hrrr.t05z.wrfsfcf01.json",
        "hrrr.t06z.wrfsfcf01.json",
    ]
    mzz = MultiZarrToZarr(
        files,
        remote_protocol="s3",
        remote_options={"anon": True},
        xarray_concat_args={"dim": "valid_time"},
    )
    mzz.translate("hrrr.total.json")
