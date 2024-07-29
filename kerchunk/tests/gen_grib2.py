import fsspec
import os
from kerchunk.grib2 import _split_file
from typing import Optional, Dict


def make_test_grib_idx_files(
    basename: str,
    suffix: str = "idx",
    limit: int = 10,
    local: bool = False,
    storage_options: Optional[Dict] = {},
):
    """
    Copy the first n (limit) messages from an existing grib2 file to a new file with an appended suffix.
    The idx files is also copied with only the first n entries.
    You can copy the files locally or to the cloud.

    Parameters
    ----------
    basename : str
        The base name is the full path to the grib file.
    suffix : str
        The suffix is the ending for the idx file.
    limit : int
        Number of grib messages(groups) you can want to copy
    local : bool
        Copy the files to local system if True
    storage_options: dict
        For accessing the data, passed to filesystem

    """
    fs, _ = fsspec.core.url_to_fs(basename, **(storage_options or {}))
    fw = fsspec.filesystem("") if local else fs
    wpath = os.path.basename(basename) if local else basename

    with fs.open(basename, "rb") as gf:
        with fw.open(f"{wpath}.test-limit-{limit}", "wb") as tf:
            for _, _, data in _split_file(gf, skip=limit):
                tf.write(data)

    with fs.open(f"{basename}.{suffix}", "rt") as idxf:
        with fw.open(f"{wpath}.test-limit-{limit}.{suffix}", "wt") as tidxf:
            for cnt, line in enumerate(idxf.readlines()):
                if cnt > limit:
                    break
                tidxf.write(line)
