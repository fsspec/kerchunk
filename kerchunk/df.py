import base64
import logging

import numpy as np
import ujson
import pandas as pd
import fsspec
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
        refs[".zmetadata"] = ujson.loads(meta[".zmetadata"])
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


def _write_json(fname, json_obj, storage_options=None):
    """Write references into a parquet file.
    
    Parameters
    ----------
    fname : str
        Output filename.
    json_obj : str, bytes, dict, list
        JSON data for parquet file to be written.
    """
    storage_options = {} if storage_options is None else storage_options
    json_obj = _normalize_json(json_obj)
    with fsspec.open(fname, "wb", **storage_options) as f:
        f.write(json_obj)


def refs_to_dataframe(
    refs,
    url,
    storage_options=None,
    record_size=10_000,
    categorical_threshold=10,
    **kwargs,
):
    """Write references as a parquet files store.

    The directory structure should mimic a normal zarr store but instead of standard chunk
    keys, references are saved as parquet dataframes.

    Parameters
    ----------
    refs: str | dict
        Location of a JSON file containing references or a reference set already loaded
        into memory.
    url: str
        Location for the output, together with protocol. This must be a writable
        directory.
    storage_options: dict | None
        Passed to fsspec when for writing the parquet.
    record_size : int
        Number of references to store in each reference file (default 10000). Bigger values
        mean fewer read requests but larger memory footprint.
    categorical_threshold : int
        Encode urls as pandas.Categorical to reduce memory footprint if the ratio
        of the number of unique urls to total number of refs for each variable
        is greater than or equal to this number. (default 10)
    **kwargs : dict
        Additional keyword arguments passed to fastparquet.
    """
    if "refs" in refs:
        refs = refs["refs"]

    fs, _ = fsspec.core.url_to_fs(url)
    fs.makedirs(url, exist_ok=True)
    fields = get_variables(refs, consolidated=True)
    # write into .zmetadata at top level, one fewer read on access
    refs[".zmetadata"]["record_size"] = record_size

    # Initialize arrays
    paths = np.full(record_size, np.nan, dtype="O")
    offsets = np.zeros(record_size, dtype="int64")
    sizes = np.zeros(record_size, dtype="int64")
    raws = np.full(record_size, np.nan, dtype="O")
    for field in fields:
        field_path = "/".join([url, field])
        if field.startswith("."):
            # zarr metadata keys (.zgroup, .zmetadata, etc)
            # only need to save .zmetadata
            if field == ".zmetadata":
                _write_json(field_path, refs[field], storage_options=storage_options)
            continue

        fs.makedirs(field_path, exist_ok=True)
        # Read the variable zarray metadata to determine number of chunks
        zarray = ujson.loads(refs[f"{field}/.zarray"])
        chunk_sizes = np.ceil(np.array(zarray["shape"]) / np.array(zarray["chunks"]))
        if chunk_sizes.size == 0:
            chunk_sizes = np.array([0])
        nchunks = int(np.product(chunk_sizes))
        nrec = nchunks // record_size
        rem = nchunks % record_size
        if rem != 0:
            nrec += 1
        nmissing = 0
        nraw = 0
        npath = 0
        irec = 0
        for i, ind in enumerate(np.ndindex(tuple(chunk_sizes.astype(int)))):
            chunk_id = ".".join([str(ix) for ix in ind])
            key = f"{field}/{chunk_id}"
            # Last parquet record can be smaller than record_size
            output_size = record_size if irec < nrec - 1 else rem
            if output_size == 0:
                continue
            j = i % record_size
            # Make note if expected number of chunks differs from actual
            # number found in references
            if key in refs:
                data = refs[key]
                if isinstance(data, list):
                    npath += 1
                    paths[j] = data[0]
                    if len(data) > 1:
                        offsets[j] = data[1]
                        sizes[j] = data[2]
                else:
                    nraw += 1
                    raws[j] = _proc_raw(data)
            else:
                nmissing += 1
            if j == output_size - 1:
                # The convention for parquet files is
                # <url>/<field_name>/refs.<rec_num>.parq
                out_path = "/".join([field_path, f"refs.{irec}.parq"])
                if nraw == output_size:
                    # All raw refs, so we can drop path/offset/size
                    df = pd.DataFrame(dict(raw=raws), copy=False)
                    object_encoding = dict(raw="bytes")
                    has_nulls = False
                else:
                    paths_maybe_cat = pd.Series(paths)
                    nunique = paths_maybe_cat.nunique()
                    if nunique and npath / nunique >= categorical_threshold:
                        paths_maybe_cat = paths_maybe_cat.astype("category")
                    if nraw == 0:
                        # No raw refs
                        df = pd.DataFrame(
                            dict(path=paths_maybe_cat, offset=offsets, size=sizes),
                            copy=False,
                        )
                        object_encoding = dict(path="utf8")
                        has_nulls = ["path"] if npath != output_size else False
                    else:
                        df = pd.DataFrame(
                            dict(
                                path=paths_maybe_cat,
                                offset=offsets,
                                size=sizes,
                                raw=raws,
                            ),
                            copy=False,
                        )
                        object_encoding = dict(raw="bytes", path="utf8")
                        has_nulls = ["path", "raw"]

                # Subset df if selection is smaller than record size
                if output_size != record_size:
                    df = df.iloc[:output_size]

                df.to_parquet(
                    out_path,
                    engine="fastparquet",
                    storage_options=storage_options,
                    compression="zstd",
                    index=False,
                    stats=False,
                    object_encoding=object_encoding,
                    has_nulls=has_nulls,
                    **kwargs,
                )
                # Reinitialize arrays for next batch of refs to process.
                paths[:] = np.nan
                offsets[:] = 0
                sizes[:] = 0
                raws[:] = np.nan
                irec += 1
                nraw = 0
                npath = 0

        if nmissing:
            logger.warning(
                f"Warning: Chunks missing for field {field}. "
                f"Expected: {nchunks}, Found: {nchunks - nmissing}"
            )
