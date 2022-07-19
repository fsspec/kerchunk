import attr
import base64
import io
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
        yield start, part_size
        f.seek(start + part_size)
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


@attr.attrs(auto_attribs=True)
class SingleMessagesStream(cfgrib.messages.FileStream):
    # override - an iterator that provides exactly one Message
    f: io.BytesIO = io.BytesIO()
    offset: int = 0
    size: int = 0

    def __iter__(self):
        self.f.seek(self.offset)
        data = self.f.read(self.size)
        yield cfgrib.messages.Message(eccodes.codes_new_from_message(data))


SingleMessagesStream.__eq__ = lambda *_, **__: True


def open_fileindex(
    path,
    f,
    offset,
    size,
    grib_errors: str = "warn",
    indexpath: str = "{path}.{short_hash}.idx",
    index_keys=cfgrib.dataset.INDEX_KEYS + ["time", "step"],
    filter_by_keys={},
):
    # override to read single message from open file-like
    index_keys = sorted(set(index_keys) | set(filter_by_keys))
    stream = SingleMessagesStream(
        path,
        f=f,
        offset=offset,
        size=size,
        message_class=cfgrib.dataset.cfmessage.CfMessage,
        errors=grib_errors,
    )
    index = stream.index(index_keys, indexpath=indexpath)
    return index.subindex(filter_by_keys)


def open_file(
    path,
    f,
    offset,
    size,
    grib_errors: str = "warn",
    indexpath: str = "{path}.{short_hash}.idx",
    filter_by_keys={},
    read_keys=(),
    time_dims=("time", "step"),
    extra_coords={},
    **kwargs,
):
    """take open file and make it into a dataset at the given offset"""
    index_keys = (
        cfgrib.dataset.INDEX_KEYS
        + list(filter_by_keys)
        + list(time_dims)
        + list(extra_coords.keys())
    )
    index = open_fileindex(
        path,
        f,
        offset,
        size,
        grib_errors,
        indexpath,
        index_keys,
        filter_by_keys=filter_by_keys,
    )
    return cfgrib.dataset.Dataset(
        *cfgrib.dataset.build_dataset_components(
            index,
            read_keys=read_keys,
            time_dims=time_dims,
            extra_coords=extra_coords,
            **kwargs,
        )
    )


def scan_grib(
    url,
    common_vars=None,
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
    common_vars = common_vars or []
    storage_options = storage_options or {}
    if filter:
        assert "typeOfLevel" in filter
    logger.debug(f"Open {url}")

    out = []
    common = False
    with fsspec.open(url, "rb", **storage_options) as f:
        logger.debug(f"File {url}")
        for offset, size in _split_file(f, skip=skip):
            store = {}
            z = zarr.open_group(store, mode="w")
            logger.debug(f"Bytes {offset}-{offset+size} of {f.size}")
            ds = open_file(url, f, offset, size)

            if filter:
                var = filter["typeOfLevel"]
                if var not in ds.variables:
                    continue
                if "level" in filter and ds.variables[var].data not in np.array(
                    filter["level"]
                ):
                    continue
                attr = ds.variables[var].attributes or {}
                attr["_ARRAY_DIMENSIONS"] = []
                if var not in z:
                    _store_array(
                        store,
                        z,
                        np.array(ds.variables[var].data),
                        var,
                        100000,
                        0,
                        0,
                        attr,
                    )
            if common is False:
                # done for first valid message
                logger.debug("Common variables")
                z.attrs.update(ds.attributes)
                for var in common_vars:
                    # assume grid, etc is the same across all messages
                    attr = ds.variables[var].attributes or {}
                    attr["_ARRAY_DIMENSIONS"] = ds.variables[var].dimensions
                    _store_array(
                        store,
                        z,
                        ds.variables[var].data,
                        var,
                        inline_threashold,
                        offset,
                        size,
                        attr,
                    )
                common = True

            for var in ds.variables:
                if (
                    var not in common_vars
                    and getattr(ds.variables[var].data, "shape", None)
                    and var != filter.get("typeOfLevel", "")
                ):

                    attr = ds.variables[var].attributes or {}
                    if "(deprecated)" in attr.get("GRIB_name", ""):
                        continue
                    attr["_ARRAY_DIMENSIONS"] = ds.variables[var].dimensions
                    _store_array(
                        store,
                        z,
                        ds.variables[var].data,
                        var,
                        inline_threashold,
                        offset,
                        size,
                        attr,
                    )
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

    def __init__(self, var, dtype="float32"):
        self.var = var
        self.dtype = dtype

    def encode(self, buf):
        # on encode, pass through
        return buf

    def decode(self, buf, out=None):
        if self.var in ["latitude", "longitude"]:
            var = self.var + "s"
        else:
            var = "values"
        mid = eccodes.codes_new_from_message(bytes(buf))
        try:
            data = eccodes.codes_get_array(mid, var)
        finally:
            eccodes.codes_release(mid)

        if out is not None:
            return ndarray_copy(data, out)
        else:
            return data.astype(self.dtype)


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
