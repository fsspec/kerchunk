import base64
import logging

import numpy as np
import ujson
import pandas as pd
import fsspec
import zarr.convenience
import zarr


# example from preffs's README'
df = pd.DataFrame(
    {
        "key": ["a/b", "a/b", "b"],
        "path": ["a.dat", "b.dat", None],
        "offset": [123, 12, 0],
        "size": [12, 17, 0],
        "raw": [None, None, b"data"],
    }
)
logger = logging.getLogger("kerchunk.df")


def _proc_raw(r):
    if not isinstance(r, bytes):
        r = r.encode()
    if r.startswith(b"base64:"):
        return base64.b64decode(r[7:])
    return r


def get_variables(refs, consolidated=True):
    """Get list of variable names from references.

    Finds the top-level prefixes in a reference set, corresponding to
    the directory listing of the root for zarr.

    Parameters
    ----------
    refs : dict
        kerchunk references keys
    consolidated : bool
        Whether or not to add consolidated metadata key to references. (default True)

    Returns
    -------
    fields : list of str
        List of variable names.
    """
    fields = []
    meta = {}
    for k in refs:
        if ".z" in k and consolidated:
            meta[k] = refs[k]
        if "/" in k:
            name, chunk = k.split("/")
            if name not in fields:
                fields.append(name)
            else:
                continue
        else:
            fields.append(k)
    if consolidated and ".zmetadata" not in fields:
        zarr.consolidate_metadata(meta)
        refs[".zmetadata"] = meta[".zmetadata"]
        fields.append(".zmetadata")
    return fields


def _normalize_json(json_obj):
    """Normalize json representation as bytes

    Parameters
    ----------
    json_obj : str, bytes, dict, list
        JSON data for parquet file to be written.
    """
    if not isinstance(json_obj, str) and not isinstance(json_obj, bytes):
        json_obj = ujson.dumps(json_obj)
    if not isinstance(json_obj, bytes):
        json_obj = json_obj.encode()
    return json_obj


def _write_json(fname, json_obj, storage_options):
    """Write references into a parquet file.

    Parameters
    ----------
    fname : str
        Output filename.
    json_obj : str, bytes, dict, list
        JSON data for parquet file to be written.
    """
    json_obj = _normalize_json(json_obj)
    with fsspec.open(fname, "wb", **storage_options) as f:
        f.write(json_obj)


def refs_to_dataframe(
    refs,
    url,
    consolidated=True,
    storage_options=None,
    # need to balance speed to read one RG versus latency in many reads and size of
    # parquet metadata. We should benchmark to remote storage to decide a good number
    row_group_size=10_000,
    **kwargs,
):
    """Write references as a store of parquet files with multiple row groups.

    The directory structure should mimic a normal zarr store but instead of standard chunk
    keys, references are saved as parquet dataframes with multiple row groups.

    Parameters
    ----------
    refs: str | dict
        Location of a JSON file containing references or a reference set already loaded
        into memory. It will get processed by the standard referenceFS, to normalise
        any templates, etc., it might contain.
    url: str
        Location for the output, together with protocol. This must be a writable
        directory.
    consolidated : bool
        Whether or not to add consolidated metadata key to references. (default True)
    storage_options: dict | None
        Passed to fsspec when for writing the parquet.
    row_group_size : int
        Number of references to store in each reference file (default 1000)
    **kwargs : dict
        Additional keyword arguments passed to parquet engine of choice.
    """
    consolidated = True  # not even an argument, let's just use it
    if "refs" in refs:
        refs = refs["refs"]
    if consolidated:
        # because all metadata are embedded
        refs = zarr.convenience.consolidate_metadata(refs)

    # write into .zmetadata at top level, one fewer read on access
    refs[".row_group_size"] = '{"row_group_size": %i}' % row_group_size
    # _write_json(
    #    "/".join([url, ]), dict(row_group_size=row_group_size),
    #    storage_options=storage_options
    # )

    fs, _ = fsspec.core.url_to_fs(url)
    fs.makedirs(url, exist_ok=True)
    fields = get_variables(refs, consolidated=consolidated)
    for field in fields:
        field_path = "/".join([url, field])
        if field.startswith("."):
            # zarr metadata keys (.zgroup, .zmetadata, etc)
            _write_json(field_path, refs[field], storage_options=storage_options)
            continue

        fs.makedirs(field_path, exist_ok=True)
        # Read the variable zarray metadata to determine number of chunks
        zarray = ujson.loads(refs[f"{field}/.zarray"])
        chunk_sizes = np.ceil(np.array(zarray["shape"]) / np.array(zarray["chunks"]))
        if chunk_sizes.size == 0:
            chunk_sizes = np.array([0])
        nchunks = int(np.product(chunk_sizes))
        extra_rows = row_group_size - nchunks % row_group_size
        output_size = nchunks + extra_rows

        paths = np.full(output_size, np.nan, dtype="O")
        offsets = np.zeros(output_size, dtype="int64")
        sizes = np.zeros(output_size, dtype="int64")
        raws = np.full(output_size, np.nan, dtype="O")
        nmissing = 0

        for metakey in [".zarray", ".zattrs"]:
            # skip when consolidated?
            key = f"{field}/{metakey}"
            _write_json(
                "/".join([field_path, metakey]),
                refs[key],
                storage_options=storage_options,
            )
        for i, ind in enumerate(np.ndindex(tuple(chunk_sizes.astype(int)))):
            chunk_id = ".".join([str(ix) for ix in ind])
            key = f"{field}/{chunk_id}"
            # Make note if expected number of chunks differs from actual
            # number found in references
            if key in refs:
                data = refs[key]
                if isinstance(data, list):
                    paths[i] = data[0]
                    offsets[i] = data[1]
                    sizes[i] = data[2]
                else:
                    raws[i] = _proc_raw(data)
            else:
                nmissing += 1

        if nmissing:
            # comment: missing keys are fine, so long as they are not a large fraction.
            #  Does referenceFS successfully give FileNotFound for them?
            logger.warning(
                f"Warning: Chunks missing for field {field}. "
                f"Expected: {nchunks}, Found: {nchunks - nmissing}"
            )
        # The convention for parquet files is
        # <url>/<field_name>/refs.parq
        out_path = "/".join([field_path, "refs.parq"])
        df = pd.DataFrame(
            dict(path=paths, offset=offsets, size=sizes, raw=raws), copy=False
        )

        # value of 10 to be configurable
        if df.paths.nunique() and ((~df.path.isna()).sum() / df.paths.nunique() > 10):
            df["paths"] = df.astype("category")

        kwargs.update(
            row_group_offsets=row_group_size,
            object_encoding=dict(raw="bytes", path="utf8"),
            has_nulls=["path", "raw"],
            stats=False,
        )
        df.to_parquet(
            out_path,
            engine="fastparquet",
            storage_options=storage_options,
            compression="zstd",
            index=False,
            **kwargs,
        )
