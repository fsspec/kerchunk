import base64
import logging

try:
    import cfgrib
except ModuleNotFoundError as err:  # pragma: no cover
    if err.name == "cfgrib":
        raise ImportError(
            "cfgrib is needed to kerchunk GRIB2 files. Please install it with "
            "`conda install -c conda-forge cfgrib`. See https://github.com/ecmwf/cfgrib "
            "for more details."
        )

import fsspec
import zarr
import numpy as np

from kerchunk.utils import class_factory, _encode_for_JSON
from kerchunk.codecs import GRIBCodec


# cfgrib copies over certain GRIB attributes
# but renames them to CF-compliant values
ATTRS_TO_COPY_OVER = {
    "long_name": "GRIB_name",
    "units": "GRIB_units",
    "standard_name": "GRIB_cfName",
}

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
            fill_value=attr.get("missingValue", None),
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
            fill_value=attr.get("missingValue", None),
            filters=[GRIBCodec(var=var, dtype=str(data.dtype))],
            compressor=False,
            overwrite=True,
        )
        store[f"{var}/" + ".".join(["0"] * len(shape))] = ["{{u}}", offset, size]
    d.attrs.update(attr)


def scan_grib(
    url,
    common=None,
    storage_options=None,
    inline_threshold=100,
    skip=0,
    filter={},
):
    """
    Generate references for a GRIB2 file

    Parameters
    ----------

    url: str
        File location
    common_vars: (depr, do not use)
    storage_options: dict
        For accessing the data, passed to filesystem
    inline_threshold: int
        If given, store array data smaller than this value directly in the output
    skip: int
        If non-zero, stop processing the file after this many messages
    filter: dict
        keyword filtering. For each key, only messages where the key exists and has
        the exact value or is in the given set, are processed.
        E.g., the cf-style filter ``{'typeOfLevel': 'heightAboveGround', 'level': 2}``
        only keeps messages where heightAboveGround==2.

    Returns
    -------

    list(dict): references dicts in Version 1 format, one per message in the file
    """
    import eccodes

    storage_options = storage_options or {}
    logger.debug(f"Open {url}")

    # This is hardcoded a lot in cfgrib!
    # valid_time is added if "time" and "step" are present in time_dims
    # These are present by default
    # TIME_DIMS = ["step", "time", "valid_time"]

    out = []
    with fsspec.open(url, "rb", **storage_options) as f:
        logger.debug(f"File {url}")
        for offset, size, data in _split_file(f, skip=skip):
            store = {}
            mid = eccodes.codes_new_from_message(data)
            m = cfgrib.cfmessage.CfMessage(mid)

            # It would be nice to just have a list of valid keys
            # There does not seem to be a nice API for this
            # 1. message_grib_keys returns keys coded in the message
            # 2. There exist "computed" keys, that are functions applied on the data
            # 3. There are also aliases!
            #    e.g. "number" is an alias of "perturbationNumber", and cfgrib uses this alias
            # So we stick to checking membership in 'm', which ends up doing
            # a lot of reads.
            message_keys = set(m.message_grib_keys())
            # The choices here copy cfgrib :(
            # message_keys.update(cfgrib.dataset.INDEX_KEYS)
            # message_keys.update(TIME_DIMS)
            # print("totalNumber" in cfgrib.dataset.INDEX_KEYS)
            # Adding computed keys adds a lot that isn't added by cfgrib
            # message_keys.extend(m.computed_keys)

            shape = (m["Ny"], m["Nx"])
            # thank you, gribscan
            native_type = eccodes.codes_get_native_type(m.codes_id, "values")
            data_size = eccodes.codes_get_size(m.codes_id, "values")
            coordinates = []

            good = True
            for k, v in (filter or {}).items():
                if k not in m:
                    good = False
                elif isinstance(v, (list, tuple, set)):
                    if m[k] not in v:
                        good = False
                elif m[k] != v:
                    good = False
            if good is False:
                continue

            z = zarr.open_group(store)
            global_attrs = {
                f"GRIB_{k}": m[k]
                for k in cfgrib.dataset.GLOBAL_ATTRIBUTES_KEYS
                if k in m
            }
            if "GRIB_centreDescription" in global_attrs:
                # follow CF compliant renaming from cfgrib
                global_attrs["institution"] = global_attrs["GRIB_centreDescription"]
            z.attrs.update(global_attrs)

            if data_size < inline_threshold:
                # read the data
                vals = m["values"].reshape(shape)
            else:
                # dummy array to match the required interface
                vals = np.empty(shape, dtype=native_type)
                assert vals.size == data_size

            attrs = {
                # Follow cfgrib convention and rename key
                f"GRIB_{k}": m[k]
                for k in cfgrib.dataset.DATA_ATTRIBUTES_KEYS
                + cfgrib.dataset.EXTRA_DATA_ATTRIBUTES_KEYS
                + cfgrib.dataset.GRID_TYPE_MAP.get(m["gridType"], [])
                if k in m
            }
            for k, v in ATTRS_TO_COPY_OVER.items():
                if v in attrs:
                    attrs[k] = attrs[v]

            # try to use cfVarName if available,
            # otherwise use the grib shortName
            varName = m["cfVarName"]
            if varName in ("undef", "unknown"):
                varName = m["shortName"]
            _store_array(store, z, vals, varName, inline_threshold, offset, size, attrs)
            if "typeOfLevel" in message_keys and "level" in message_keys:
                name = m["typeOfLevel"]
                coordinates.append(name)
                # convert to numpy scalar, so that .tobytes can be used for inlining
                # dtype=float is hardcoded in cfgrib
                data = np.array(m["level"], dtype=float)[()]
                try:
                    attrs = cfgrib.dataset.COORD_ATTRS[name]
                except KeyError:
                    logger.debug(f"Couldn't find coord {name} in dataset")
                    attrs = {}
                attrs["_ARRAY_DIMENSIONS"] = []
                _store_array(
                    store, z, data, name, inline_threshold, offset, size, attrs
                )
            dims = (
                ["y", "x"]
                if m["gridType"] in cfgrib.dataset.GRID_TYPES_2D_NON_DIMENSION_COORDS
                else ["latitude", "longitude"]
            )
            z[varName].attrs["_ARRAY_DIMENSIONS"] = dims

            for coord in cfgrib.dataset.COORD_ATTRS:
                coord2 = {"latitude": "latitudes", "longitude": "longitudes"}.get(
                    coord, coord
                )
                try:
                    x = m.get(coord2)
                except eccodes.WrongStepUnitError as e:
                    logger.warning(
                        "Ignoring coordinate '%s' for varname '%s', raises: eccodes.WrongStepUnitError(%s)",
                        coord2,
                        varName,
                        e,
                    )
                    continue

                if x is None:
                    continue
                coordinates.append(coord)
                inline_extra = 0
                if isinstance(x, np.ndarray) and x.size == data_size:
                    if (
                        m["gridType"]
                        in cfgrib.dataset.GRID_TYPES_2D_NON_DIMENSION_COORDS
                    ):
                        dims = ["y", "x"]
                        x = x.reshape(vals.shape)
                    else:
                        dims = [coord]
                        if coord == "latitude":
                            x = x.reshape(vals.shape)[:, 0].copy()
                        elif coord == "longitude":
                            x = x.reshape(vals.shape)[0].copy()
                        # force inlining of x/y/latitude/longitude coordinates.
                        # since these are derived from analytic formulae
                        # and are not stored in the message
                        inline_extra = x.nbytes + 1
                elif np.isscalar(x):
                    # convert python scalars to numpy scalar
                    # so that .tobytes can be used for inlining
                    x = np.array(x)[()]
                    dims = []
                else:
                    x = np.array([x])
                    dims = [coord]
                attrs = cfgrib.dataset.COORD_ATTRS[coord]
                _store_array(
                    store,
                    z,
                    x,
                    coord,
                    inline_threshold + inline_extra,
                    offset,
                    size,
                    attrs,
                )
                z[coord].attrs["_ARRAY_DIMENSIONS"] = dims
            if coordinates:
                z.attrs["coordinates"] = " ".join(coordinates)

            out.append(
                {
                    "version": 1,
                    "refs": _encode_for_JSON(store),
                    "templates": {"u": url},
                }
            )
    logger.debug("Done")
    return out


GribToZarr = class_factory(scan_grib)


def example_combine(
    filter={"typeOfLevel": "heightAboveGround", "level": 2}
):  # pragma: no cover
    """Create combined dataset of weather measurements at 2m height

    Ten consecutive timepoints from ten 120MB files on s3.
    Example usage:

    >>> tot = example_combine()
    >>> ds = xr.open_dataset("reference://", engine="zarr", backend_kwargs={
    ...        "consolidated": False,
    ...        "storage_options": {"fo": tot, "remote_options": {"anon": True}}})
    """
    from kerchunk.combine import MultiZarrToZarr, drop

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

    out = [scan_grib(u, storage_options=so, filter=filter) for u in files]
    out = sum(out, [])
    mzz = MultiZarrToZarr(
        out,
        remote_protocol="s3",
        preprocess=drop(("valid_time", "step")),
        remote_options=so,
        concat_dims=["time", "var"],
        identical_dims=["heightAboveGround", "latitude", "longitude"],
    )
    return mzz.translate()
