import base64
import logging

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
    fo,
    url,
    target_protocol=None,
    target_options=None,
    storage_options=None,
    record_size=100_000,
    categorical_threshold=10,
):
    """Write references as a parquet files store.

    The directory structure should mimic a normal zarr store but instead of standard chunk
    keys, references are saved as parquet dataframes.

    Parameters
    ----------
    fo : str | dict
        Location of a JSON file containing references or a reference set already loaded
        into memory.
    url : str
        Location for the output, together with protocol. This must be a writable
        directory.
    target_protocol : str
        Used for loading the reference file, if it is a path. If None, protocol
        will be derived from the given path
    target_options : dict
        Extra FS options for loading the reference file ``fo``, if given as a path
    storage_options: dict | None
        Passed to fsspec when for writing the parquet.
    record_size : int
        Number of references to store in each reference file (default 10000). Bigger values
        mean fewer read requests but larger memory footprint.
    categorical_threshold : int
        Encode urls as pandas.Categorical to reduce memory footprint if the ratio
        of the number of unique urls to total number of refs for each variable
        is greater than or equal to this number. (default 10)
    """
    from fsspec.implementations.reference import LazyReferenceMapper

    if isinstance(fo, str):
        # JSON file
        dic = dict(**(target_options or {}), protocol=target_protocol)
        with fsspec.open(fo, "rb", **dic) as f:
            logger.info("Read reference from URL %s", fo)
            refs = ujson.load(f)
    else:
        # Mapping object
        refs = fo

    if "refs" in refs:
        refs = refs["refs"]

    fs, _ = fsspec.core.url_to_fs(url, **(storage_options or {}))
    out = LazyReferenceMapper.create(
        record_size=record_size,
        root=url,
        fs=fs,
        categorical_threshold=categorical_threshold,
    )

    for k in sorted(refs):
        out[k] = refs[k]
    out.flush()
