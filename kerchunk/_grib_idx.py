"""
_grib_idx.py
============

This module provides utility functions for indexing and then utilizing
the index for the messages in GRIB2 files, accompanied by their idx files.

The module includes functions for extracting grib metadata, parsing the
attributes in idx files, one-to-one mapping between the grib metadata
and idx data, building the index for the grib files. It has been tested
with GFS, HRRR, GEFS model outputs.

Attributes
----------
(to be added)


Functions
---------
parse_grib_idx(basename, suffix, storage_options, validate)
    Parses the idx file with corresponding basename(grib file) with
    storage options, and returns the idx data as a Dataframe

extract_dataset_chunk_index(dset, ref_store, grib)
    Process and extracts the kerchunk index for an xarray dataset
    or datatree node and returns index meta data as a list of dictionary

extract_datatree_chunk_index(dtree, kerchunk_store, grib)
    Recursively iterate through the xarray datatree and with the
    help of extract_dataset_chunk_index, the index metadata as a Dataframe

build_idx_grib_mapping(basename, storage_options, suffix, mapper, validate)
    Creates a one-to-one mapping from the idx metadata and gribdata for a single
    file, which is needed to build the index. Mapping is returned as a Dataframe

map_from_index(run_time, mapping, idxdf, raw_merged)
    Builds the index for each grib file using the mapping made from
    building_idx_grib_mapping function and returns the index as a Dataframe.

Notes
-----
Indexing process should only happen for files belonging to the same horizon or else
the index won't work. This module is not be used directly but through kerchunk.grib2
module as the necessary functions for building the index are exposed there.


References
----------
For more information on indexing, check out this
link: https://github.com/asascience-open/nextgen-dmac/pull/57


Original license
----------------
MIT License Copyright (c) 2023 Camus Energy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""  # noqa: E501

import ujson
import itertools
import os
import gzip
import re
import logging
import fsspec
from enum import unique, Enum
from functools import cache
import warnings
from typing import Iterable, Dict, Optional, Callable, Any
import numpy as np
import pandas as pd
import xarray as xr
import base64
import copy


class DynamicZarrStoreError(ValueError):
    pass


logger = logging.getLogger("grib-indexing")

COORD_DIM_MAPPING: dict[str, str] = dict(
    time="run_times",
    valid_time="valid_times",
    step="model_horizons",
)

ZARR_TREE_STORE = "zarr_tree_store.json.gz"


def repeat_steps(step_index: pd.TimedeltaIndex, to_length: int) -> np.array:
    return np.tile(step_index.to_numpy(), int(np.ceil(to_length / len(step_index))))[
        :to_length
    ]


def create_steps(steps_index: pd.Index, to_length) -> np.array:
    return np.vstack([repeat_steps(si, to_length) for si in steps_index])


def store_coord_var(key: str, zstore: dict, coords: tuple[str, ...], data: np.array):
    if np.isnan(data).any():
        if f"{key}/.zarray" not in zstore:
            logger.debug("Skipping nan coordinate with no variable %s", key)
            return
        else:
            logger.info("Trying to add coordinate var %s with nan value!", key)

    zattrs = ujson.loads(zstore[f"{key}/.zattrs"])
    zarray = ujson.loads(zstore[f"{key}/.zarray"])
    # Use list not tuple
    zarray["chunks"] = [*data.shape]
    zarray["shape"] = [*data.shape]

    if key.endswith("step"):
        zarray["dtype"] = "<f8"
        zattrs["units"] = "hours"

    zattrs["_ARRAY_DIMENSIONS"] = [
        COORD_DIM_MAPPING[v] if v in COORD_DIM_MAPPING else v for v in coords
    ]

    zstore[f"{key}/.zarray"] = ujson.dumps(zarray)
    zstore[f"{key}/.zattrs"] = ujson.dumps(zattrs)

    vkey = ".".join(["0" for _ in coords])
    logger.info("COOORD %s DATA!: %s", key, data.dtype)
    data_bytes = data.tobytes()
    try:
        enocded_val = data_bytes.decode("ascii")
    except UnicodeDecodeError:
        enocded_val = (b"base64:" + base64.b64encode(data_bytes)).decode("ascii")
    zstore[f"{key}/{vkey}"] = enocded_val


def store_data_var(
    key: str,
    zstore: dict,
    dims: dict[str, int],
    coords: dict[str, tuple[str, ...]],
    data: pd.DataFrame,
    steps: np.array,
    times: np.array,
    lvals: Optional[np.array],
):
    zattrs = ujson.loads(zstore[f"{key}/.zattrs"])
    zarray = ujson.loads(zstore[f"{key}/.zarray"])

    dcoords = coords["datavar"]

    # The lat/lon y/x coordinates are always the last two
    lat_lon_dims = {
        k: v for k, v in zip(zattrs["_ARRAY_DIMENSIONS"][-2:], zarray["shape"][-2:])
    }
    full_coords = dcoords + tuple(lat_lon_dims.keys())
    full_dims = dict(**dims, **lat_lon_dims)

    # all chunk dimensions are 1 except for lat/lon or x/y
    zarray["chunks"] = [
        1 if c not in lat_lon_dims else lat_lon_dims[c] for c in full_coords
    ]
    zarray["shape"] = [full_dims[k] for k in full_coords]
    if zarray["fill_value"] is None:
        # Check dtype first?
        zarray["fill_value"] = np.nan

    zattrs["_ARRAY_DIMENSIONS"] = [
        COORD_DIM_MAPPING[v] if v in COORD_DIM_MAPPING else v for v in full_coords
    ]

    zstore[f"{key}/.zarray"] = ujson.dumps(zarray)
    zstore[f"{key}/.zattrs"] = ujson.dumps(zattrs)

    idata = data.set_index(["time", "step", "level"]).sort_index()

    for idx in itertools.product(*[range(dims[k]) for k in dcoords]):
        # Build an iterator over each of the single dimension chunks
        # TODO if the .loc call is slow inside the loop replace with a reindex operation and iterate that.
        dim_idx = {k: v for k, v in zip(dcoords, idx)}

        iloc: tuple[Any, ...] = (
            times[tuple([dim_idx[k] for k in coords["time"]])],
            steps[tuple([dim_idx[k] for k in coords["step"]])],
        )
        if lvals is not None:
            iloc = iloc + (lvals[idx[-1]],)  # type:ignore[assignment]

        try:
            # Squeeze if needed to get a series. Noop if already a series Df has multiple rows
            dval = idata.loc[iloc].squeeze()
        except KeyError:
            logger.info(f"Error getting vals {iloc} for in path {key}")
            continue

        assert isinstance(
            dval, pd.Series
        ), f"Got multiple values for iloc {iloc} in key {key}: {dval}"

        if pd.isna(dval.inline_value):
            # List of [URI(Str), offset(Int), length(Int)] using python (not numpy) types.
            record = [dval.uri, dval.offset.item(), dval.length.item()]
        else:
            record = dval.inline_value
        # lat/lon y/x have only the zero chunk
        vkey = ".".join([str(v) for v in (idx + (0, 0))])
        zstore[f"{key}/{vkey}"] = record


@unique
class AggregationType(Enum):
    """
    ENUM for aggregation types
    TODO is this useful elsewhere?
    """

    HORIZON = "horizon"
    VALID_TIME = "valid_time"
    RUN_TIME = "run_time"
    BEST_AVAILABLE = "best_available"


def reinflate_grib_store(
    axes: list[pd.Index],
    aggregation_type: AggregationType,
    chunk_index: pd.DataFrame,
    zarr_ref_store: dict,
) -> dict:
    """
    Given a zarr_store hierarchy, pull out the variables present in the chunks dataframe and reinflate the
    zarr variables adding any needed dimensions. This is a select operation - based on the time axis
    provided.
    Assumes everything is stored in hours per grib convention.

    :param axes: a list of new axes for aggregation
    :param aggregation_type: the type of fmrc aggregation
    :param chunk_index: a dataframe containing the kerchunk index
    :param zarr_ref_store: the deflated (chunks removed) zarr store
    :return: the inflated zarr store
    """
    # Make a deep copy to avoid modifying the input
    zstore = copy.deepcopy(zarr_ref_store["refs"])

    axes_by_name: dict[str, pd.Index] = {pdi.name: pdi for pdi in axes}
    # Validate axis names
    time_dims: dict[str, int] = {}
    time_coords: dict[str, tuple[str, ...]] = {}
    # TODO: add a data class or other method of typing and validating the variables created in this if block
    if aggregation_type.value == AggregationType.HORIZON.value:
        # Use index length horizons containing timedelta ranges for the set of steps
        time_dims["step"] = len(axes_by_name["step"])
        time_dims["valid_time"] = len(axes_by_name["valid_time"])

        time_coords["step"] = ("step", "valid_time")
        time_coords["valid_time"] = ("step", "valid_time")
        time_coords["time"] = ("step", "valid_time")
        time_coords["datavar"] = ("step", "valid_time")

        steps = create_steps(axes_by_name["step"], time_dims["valid_time"])
        valid_times = np.tile(
            axes_by_name["valid_time"].to_numpy(), (time_dims["step"], 1)
        )
        times = valid_times - steps

    elif aggregation_type.value == AggregationType.VALID_TIME.value:
        # Provide an index of steps and an index of valid times
        time_dims["step"] = len(axes_by_name["step"])
        time_dims["valid_time"] = len(axes_by_name["valid_time"])

        time_coords["step"] = ("step",)
        time_coords["valid_time"] = ("valid_time",)
        time_coords["time"] = ("valid_time", "step")
        time_coords["datavar"] = ("valid_time", "step")

        steps = axes_by_name["step"].to_numpy()
        valid_times = axes_by_name["valid_time"].to_numpy()

        steps2d = np.tile(axes_by_name["step"], (time_dims["valid_time"], 1))
        valid_times2d = np.tile(
            np.reshape(axes_by_name["valid_time"], (-1, 1)), (1, time_dims["step"])
        )
        times = valid_times2d - steps2d

    elif aggregation_type.value == AggregationType.RUN_TIME.value:
        # Provide an index of steps and an index of run times.
        time_dims["step"] = len(axes_by_name["step"])
        time_dims["time"] = len(axes_by_name["time"])

        time_coords["step"] = ("step",)
        time_coords["valid_time"] = ("time", "step")
        time_coords["time"] = ("time",)
        time_coords["datavar"] = ("time", "step")

        steps = axes_by_name["step"].to_numpy()
        times = axes_by_name["time"].to_numpy()

        # The valid times will be runtimes by steps
        steps2d = np.tile(axes_by_name["step"], (time_dims["time"], 1))
        times2d = np.tile(
            np.reshape(axes_by_name["time"], (-1, 1)), (1, time_dims["step"])
        )
        valid_times = times2d + steps2d

    elif aggregation_type.value == AggregationType.BEST_AVAILABLE.value:
        time_dims["valid_time"] = len(axes_by_name["valid_time"])
        assert (
            len(axes_by_name["time"]) == 1
        ), "The time axes must describe a single 'as of' date for best available"
        reference_time = axes_by_name["time"].to_numpy()[0]

        time_coords["step"] = ("valid_time",)
        time_coords["valid_time"] = ("valid_time",)
        time_coords["time"] = ("valid_time",)
        time_coords["datavar"] = ("valid_time",)

        valid_times = axes_by_name["valid_time"].to_numpy()
        times = np.where(valid_times <= reference_time, valid_times, reference_time)
        steps = valid_times - times
    else:
        raise RuntimeError(f"Invalid aggregation_type argument: {aggregation_type}")

    # Copy all the groups that contain variables in the chunk dataset
    unique_groups = chunk_index.set_index(
        ["varname", "stepType", "typeOfLevel"]
    ).index.unique()

    # Drop keys not in the unique groups
    for key in list(zstore.keys()):
        # Separate the key as a path keeping only: varname, stepType and typeOfLevel
        # Treat root keys like ".zgroup" as special and return an empty tuple
        lookup = tuple(
            [val for val in os.path.dirname(key).split("/")[:3] if val != ""]
        )
        if lookup not in unique_groups:
            del zstore[key]

    # Now update the zstore for each variable.
    for key, group in chunk_index.groupby(["varname", "stepType", "typeOfLevel"]):
        base_path = "/".join(key)
        lvals = group.level.unique()
        dims = time_dims.copy()
        coords = time_coords.copy()
        if len(lvals) == 1:
            lvals = lvals.squeeze()
            dims[key[2]] = 0
        elif len(lvals) > 1:
            lvals = np.sort(lvals)
            # multipel levels
            dims[key[2]] = len(lvals)
            coords["datavar"] += (key[2],)
        else:
            raise ValueError("")

        # Convert to floating point seconds
        # td.astype("timedelta64[s]").astype(float) / 3600  # Convert to floating point hours
        store_coord_var(
            key=f"{base_path}/time",
            zstore=zstore,
            coords=time_coords["time"],
            data=times.astype("datetime64[s]"),
        )

        store_coord_var(
            key=f"{base_path}/valid_time",
            zstore=zstore,
            coords=time_coords["valid_time"],
            data=valid_times.astype("datetime64[s]"),
        )

        store_coord_var(
            key=f"{base_path}/step",
            zstore=zstore,
            coords=time_coords["step"],
            data=steps.astype("timedelta64[s]").astype("float64") / 3600.0,
        )

        store_coord_var(
            key=f"{base_path}/{key[2]}",
            zstore=zstore,
            coords=(key[2],) if lvals.shape else (),
            data=lvals,  # all grib levels are floats
        )

        store_data_var(
            key=f"{base_path}/{key[0]}",
            zstore=zstore,
            dims=dims,
            coords=coords,
            data=group,
            steps=steps,
            times=times,
            lvals=lvals if lvals.shape else None,
        )

    return dict(refs=zstore, version=1)


def strip_datavar_chunks(
    kerchunk_store: dict, keep_list: tuple[str, ...] = ("latitude", "longitude")
) -> None:
    """
    Modify in place a kerchunk reference store to strip the kerchunk references for variables not in the
    keep list.
    :param kerchunk_store: a kerchunk ref spec store
    :param keep_list: the list of variables to keep references
    """
    zarr_store = kerchunk_store["refs"]

    zchunk_matcher = re.compile(r"^(?P<name>.*)\/(?P<zchunk>\d+[\.\d+]*)$")
    for key in list(zarr_store.keys()):
        matched = zchunk_matcher.match(key)
        if matched:
            logger.debug("Matched! %s", matched)
            if any([matched.group("name").endswith(keeper) for keeper in keep_list]):
                logger.debug("Skipping key %s", matched.group("name"))
                continue
            del zarr_store[key]


def write_store(metadata_path: str, store: dict):
    fpath = os.path.join(metadata_path, ZARR_TREE_STORE)
    compressed = gzip.compress(ujson.dumps(store).encode())
    with fsspec.open(fpath, "wb") as f:
        f.write(compressed)
    logger.info("Wrote %d bytes to %s", len(compressed), fpath)


@cache
def read_store(metadata_path: str) -> dict:
    """
    Cached method for loading the static zarr store from a metadata path
    :param metadata_path: the path (usually gcs) to the metadata directory
    :return: a kerchunk zarr store reference spec dictionary (defalated)
    """
    fpath = os.path.join(metadata_path, ZARR_TREE_STORE)
    with fsspec.open(fpath, "rb") as f:
        compressed = f.read()
    logger.info("Read %d bytes from %s", len(compressed), fpath)
    zarr_store = ujson.loads(gzip.decompress(compressed).decode())
    return zarr_store


def grib_coord(name: str) -> str:
    """
    Take advantage of gribs strict coordinate name structure
    Use the known level names and assume everything else is "level" which is described by the
    gribLevelAttribute
    This is helpful because there are a lot of named levels but they are all float values
    and each variable can have only one level coordinate.
    By mapping all of these levels to level the sparse chunk index becomes dense.
    :param name:
    :return:
    """

    if name in ("valid_time", "time", "step", "latitude", "longitude"):
        return name
    else:
        return "level"


def parse_grib_idx(
    basename: str,
    suffix: str = "idx",
    storage_options: Optional[Dict] = None,
    validate: bool = False,
) -> "pd.DataFrame":
    """
    Parses per-message metadata from a grib2.idx file (text-type) to a dataframe of attributes

    The function uses the idx file, extracts the metadata known as attrs (variables with
    level and forecast time) from each idx entry and converts it into pandas
    DataFrame. The dataframe is later to build the one-to-one mapping to the grib file metadata.

    Parameters
    ----------
    basename : str
        The base name is the full path to the grib file.
    suffix : str
        The suffix is the ending for the idx file.
    storage_options: dict
        For accessing the data, passed to filesystem
    validate : bool
        The validation if the metadata table has duplicate attrs.

    Returns
    -------
    pandas.DataFrame : The data frame containing the results.
    """
    import pandas as pd

    fs, _ = fsspec.core.url_to_fs(basename, **(storage_options or {}))

    fname = f"{basename}.{suffix}"

    baseinfo = fs.info(basename)

    with fs.open(fname) as f:
        result = pd.read_csv(f, header=None, names=["raw_data"])
        result[["idx", "offset", "date", "attrs"]] = result["raw_data"].str.split(
            ":", expand=True, n=3
        )
        result["offset"] = result["offset"].astype(int)
        result["idx"] = result["idx"].astype(int)

        # dropping the original single "raw_data" column after formatting
        result.drop(columns=["raw_data"], inplace=True)

    result = result.assign(
        length=(
            result.offset.shift(periods=-1, fill_value=baseinfo["size"]) - result.offset
        ),
        idx_uri=fname,
        grib_uri=basename,
    )

    if validate and not result["attrs"].is_unique:
        raise ValueError(f"Attribute mapping for grib file {basename} is not unique")

    return result.set_index("idx")


def build_path(path: Iterable[str | None], suffix: Optional[str] = None) -> str:
    """
    Returns the path to access the values in a zarr store without a leading "/"

    Parameters
    ----------
    path : Iterable[str | None]
        The path is the list of values to the element in zarr store
    suffix : str
        Last element if any

    Returns
    -------
    str : returns the path as a string
    """
    return "/".join([val for val in [*path, suffix] if val is not None]).lstrip("/")


def extract_dataset_chunk_index(
    dset: "xr.DataTree",
    ref_store: Dict,
    grib: bool = False,
) -> list[dict]:
    """
    Process and extracts the kerchunk index for an xarray dataset or datatree node.

    The data_vars from the dataset will be indexed.
    The coordinate vars for each dataset will be used for indexing.
    Datatrees generated by grib_tree have some nice properties which allow a denser index.

    Parameters
    ----------
    dset : datatree.DataTree
        The datatree node from the datatree instance
    ref_store : Dict
        The zarr store dictionary backed by the gribtree
    grib : bool
        boolean for treating coordinates as grib levels

    Returns
    -------
    list[dict] : returns the extracted grib metadata in the form of key-value pairs inside a list
    """

    result: list[dict] = []
    attributes = dset.attrs.copy()

    dpath = None
    if isinstance(dset, xr.DataTree):
        dpath = dset.path
        walk_group = dset.parent
        while walk_group:
            attributes.update(walk_group.attrs)
            walk_group = walk_group.parent

    for dname, dvar in dset.data_vars.items():
        # Get the chunk size - `chunks` property only works for xarray native
        zarray = ujson.loads(ref_store[build_path([dpath, dname], suffix=".zarray")])
        dchunk = zarray["chunks"]
        dshape = dvar.shape

        index_dims = {}
        for ddim_nane, ddim_size, dchunk_size in zip(dvar.dims, dshape, dchunk):
            if dchunk_size == 1:
                index_dims[ddim_nane] = ddim_size
            elif dchunk_size != ddim_size:
                # Must be able to get a single coordinate value for each chunk to index it.
                raise ValueError(
                    "Can not extract chunk index for dimension %s with non singleton chunk dimensions"
                    % ddim_nane
                )
            # Drop the dim where each chunk covers the whole dimension - no indexing needed!

        for idx in itertools.product(*[range(v) for v in index_dims.values()]):
            # Build an iterator over each of the single dimension chunks
            dim_idx = {key: val for key, val in zip(index_dims.keys(), idx)}

            coord_vals = {}
            for cname, cvar in dvar.coords.items():
                if grib:
                    # Grib data has only one level coordinate
                    cname = (
                        cname
                        if cname
                        in ("valid_time", "time", "step", "latitude", "longitude")
                        else "level"
                    )

                if all([dim_name in dim_idx for dim_name in cvar.dims]):
                    coord_index = tuple([dim_idx[dim_name] for dim_name in cvar.dims])
                    try:
                        coord_vals[cname] = cvar.to_numpy()[coord_index]
                    except Exception:
                        raise DynamicZarrStoreError(
                            f"Error reading coords for {dpath}/{dname} coord {cname} with index {coord_index}"
                        )

            whole_dim_cnt = len(dvar.dims) - len(dim_idx)
            chunk_idx = map(str, [*idx, *[0] * whole_dim_cnt])
            chunk_key = build_path([dpath, dname], suffix=".".join(chunk_idx))
            chunk_ref = ref_store.get(chunk_key)

            if chunk_ref is None:
                logger.warning("Chunk not found: %s", chunk_key)
                continue

            elif isinstance(chunk_ref, list) and len(chunk_ref) == 3:
                chunk_data = dict(
                    uri=chunk_ref[0],
                    offset=chunk_ref[1],
                    length=chunk_ref[2],
                    inline_value=None,
                )
            elif isinstance(chunk_ref, (bytes, str)):
                chunk_data = dict(inline_value=chunk_ref, offset=-1, length=-1)
            else:
                raise ValueError(f"Key {chunk_key} has bad value '{chunk_ref}'")
            result.append(dict(varname=dname, **attributes, **coord_vals, **chunk_data))

    return result


def extract_datatree_chunk_index(
    dtree: "xr.DataTree", kerchunk_store: dict, grib: bool = False
) -> "pd.DataFrame":
    """
    Recursive method to iterate over the data tree and extract the data variable chunks with index metadata

    Parameters
    ----------
    dtree : datatree.DataTree
        The xarray datatree representation of the reference filesystem
    kerchunk_store : dict
        the grib_tree output for a single grib file
    grib : bool
        boolean for treating coordinates as grib levels

    Returns
    -------
    pandas.Dataframe : The dataframe constructed from the grib metadata
    """
    import pandas as pd

    result: list[dict] = []

    for node in dtree.subtree:
        if node.has_data:
            result += extract_dataset_chunk_index(
                node, kerchunk_store["refs"], grib=grib
            )

    return pd.DataFrame.from_records(result)


def _map_grib_file_by_group(
    fname: str,
    mapper: Optional[Callable] = None,
    storage_options: Optional[Dict] = None,
) -> "pd.DataFrame":
    """
    Helper method used to read the cfgrib metadata associated with each message (group) in the grib file
    This method does not add metadata

    Parameters
    ----------
    fname : str
        the file name to read with scan_grib
    mapper : Optional[Callable]
        the mapper if any to apply (used for hrrr subhf)

    Returns
    -------
    pandas.Dataframe : The intermediate dataframe constructed from the grib metadata
    """
    import pandas as pd
    from kerchunk.grib2 import scan_grib

    mapper = (lambda x: x) if mapper is None else mapper
    references = scan_grib(fname, storage_options=storage_options)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.concat(
            # grib idx is fortran indexed (from one not zero)
            list(
                filter(
                    # filtering out the cfgrib metadata dataframe in case it is None
                    lambda item: item is not None,
                    [
                        # extracting the metadata from a single message
                        _extract_single_group(mapper(group), i, storage_options)
                        for i, group in enumerate(references, start=1)
                    ],
                )
            )
        ).set_index("idx")


def _extract_single_group(grib_group: dict, idx: int, storage_options: Dict):
    from kerchunk.grib2 import grib_tree

    grib_tree_store = grib_tree(
        [
            grib_group,
        ],
        storage_options,
    )

    if len(grib_tree_store["refs"]) <= 1:
        logger.info("Empty DT: %s", grib_tree_store)
        return None

    dt = xr.open_datatree(
        "reference://",
        storage_options={"fo": grib_tree_store},
        engine="zarr",
        consolidated=False,
    )

    k_ind = extract_datatree_chunk_index(dt, grib_tree_store, grib=True)
    if k_ind.empty:
        logger.warning("Empty Kind: %s", grib_tree_store)
        return None

    assert (
        len(k_ind) == 1
    ), f"expected a single variable grib group but produced: {k_ind}"
    k_ind.loc[:, "idx"] = idx
    return k_ind


def build_idx_grib_mapping(
    basename: str,
    storage_options: Optional[Dict] = None,
    suffix: str = "idx",
    mapper: Optional[Callable] = None,
    validate: bool = True,
) -> "pd.DataFrame":
    """
    Mapping method combines the idx and grib metadata to make a mapping from
    one to the other for a particular model horizon file. This should be generally
    applicable to all forecasts for the given horizon.

    Parameters
    ----------
    basename : str
        the full path for the grib2 file
    storage_options: dict
        For accessing the data, passed to filesystem
    suffix : str
        The suffix is the ending for the idx file.
    mapper : Optional[Callable]
        the mapper if any to apply (used for hrrr subhf)
    validate : bool
        to assert the mapping is correct or fail before returning

    Returns
    -------
    pandas.Dataframe : The merged dataframe with the results of the two operations
    joined on the grib message (group) number
    """
    import pandas as pd

    grib_file_index = _map_grib_file_by_group(
        fname=basename,
        mapper=mapper,
        storage_options=storage_options,
    )
    idx_file_index = parse_grib_idx(
        basename=basename, suffix=suffix, storage_options=storage_options
    )
    result = idx_file_index.merge(
        # Left merge because the idx file should be authoritative - one record per grib message
        grib_file_index,
        on="idx",
        how="left",
        suffixes=("_idx", "_grib"),
    )

    if validate:
        # If any of these conditions fail - inspect the result manually on colab.
        all_match_offset = (
            (result.loc[:, "offset_idx"] == result.loc[:, "offset_grib"])
            | pd.isna(result.loc[:, "offset_grib"])
            | ~pd.isna(result.loc[:, "inline_value"])
        )
        all_match_length = (
            (result.loc[:, "length_idx"] == result.loc[:, "length_grib"])
            | pd.isna(result.loc[:, "length_grib"])
            | ~pd.isna(result.loc[:, "inline_value"])
        )

        if not all_match_offset.all():
            vcs = all_match_offset.value_counts()
            raise ValueError(
                f"Failed to match message offset mapping for grib file {basename}: "
                f"{vcs[True]} matched, {vcs[False]} didn't"
            )

        if not all_match_length.all():
            vcs = all_match_length.value_counts()
            raise ValueError(
                f"Failed to match message offset mapping for grib file {basename}: "
                f"{vcs[True]} matched, {vcs[False]} didn't"
            )

        if not result["attrs"].is_unique:
            dups = result.loc[result["attrs"].duplicated(keep=False), :]
            logger.warning(
                "The idx attribute mapping for %s is not unique for %d variables: %s",
                basename,
                len(dups),
                dups.varname.tolist(),
            )

        r_index = result.set_index(
            ["varname", "typeOfLevel", "stepType", "level", "valid_time"]
        )
        if not r_index.index.is_unique:
            dups = r_index.loc[r_index.index.duplicated(keep=False), :]
            logger.warning(
                "The grib hierarchy in %s is not unique for %d variables: %s",
                basename,
                len(dups),
                dups.index.get_level_values("varname").tolist(),
            )

    return result


def map_from_index(
    run_time: "pd.Timestamp",
    mapping: "pd.DataFrame",
    idxdf: "pd.DataFrame",
    raw_merged: bool = False,
) -> "pd.DataFrame":
    """
    Main method used for building index dataframes from parsed IDX files
    merged with the correct mapping for the horizon

    Parameters
    ----------

    run_time : pd.Timestamp
        the run time timestamp of the idx data
    mapping : pd.DataFrame
        the mapping data derived from comparing the idx attributes to the
        CFGrib attributes for a given horizon
    idxdf : pd.DataFrame
        the dataframe of offsets and lengths for each grib message and its
        attributes derived from an idx file
    raw_merged : bool
        Used for debugging to see all the columns in the merge. By default,
        it returns the index columns with the corrected time values plus
        the index metadata

    Returns
    -------

    pd.Dataframe : the index dataframe that will be used to read variable data from the grib file
    """

    idxdf = idxdf.reset_index().set_index("attrs")
    mapping = mapping.reset_index().set_index("attrs")
    mapping.drop(columns="uri", inplace=True)  # Drop the URI column from the mapping

    if not idxdf.index.is_unique:
        raise ValueError("Parsed idx data must have unique attrs to merge on!")

    if not mapping.index.is_unique:
        raise ValueError("Mapping data must have unique attrs to merge on!")

    # Merge the offset and length from the idx file with the varname, step and level from the mapping

    result = idxdf.merge(mapping, on="attrs", how="left", suffixes=("", "_mapping"))

    if raw_merged:
        return result
    else:
        # Get the grib_uri column from the idxdf and ignore the uri column from the mapping
        # We want the offset, length and uri of the index file with the varname, step and level of the mapping
        selected_results = result.rename(columns=dict(grib_uri="uri"))[
            [
                "varname",
                "typeOfLevel",
                "stepType",
                "name",
                "step",
                "level",
                "time",
                "valid_time",
                "uri",
                "offset",
                "length",
                "inline_value",
            ]
        ]
    # Drop the inline values from the mapping data
    selected_results.loc[:, "inline_value"] = None
    selected_results.loc[:, "time"] = run_time
    selected_results.loc[:, "valid_time"] = (
        selected_results.time + selected_results.step
    )
    logger.info("Dropping %d nan varnames", selected_results.varname.isna().sum())
    selected_results = selected_results.loc[~selected_results.varname.isna(), :]
    return selected_results.reset_index(drop=True)
