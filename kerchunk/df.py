import pandas as pd
import numpy as np
import fastparquet
import fsspec

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
fastparquet.write(
    "preffs.parq",
    df,
    object_encoding={"raw": "bytes", "key": "utf8", "path": "utf8"},
    stats=["key"],
    has_nulls=["path", "raw"],
    compression="zstd",
)

out = {
    "raw": np.empty(3, dtype=object),
    "path": np.empty(3, dtype=object),
    "key": np.empty(3, dtype=object),
    "offset": np.empty(3, dtype="int64"),
    "size": np.empty(3, dtype="int64"),
}
pf = fastparquet.ParquetFile("preffs.parq")
with open("preffs.parq", "rb") as f:
    fastparquet.core.read_row_group_arrays(
        f, pf.row_groups[0], list(out), [], pf.schema, [], assign=out
    )


def refs_to_dataframe(refs, url, storage_options=None, partition=False):
    # normalise refs (e.g., for templates)
    fs = fsspec.filesystem("reference", fo=refs)
    refs = fs.references

    df = pd.DataFrame(
        {
            "key": list(refs),
            "path": [r[0] if isinstance(r, list) else None for r in refs.values()],
            "offset": [
                r[1] if isinstance(r, list) and len(r) > 1 else 0 for r in refs.values()
            ],
            "size": [
                r[2] if isinstance(r, list) and len(r) > 1 else 0 for r in refs.values()
            ],
            "raw": [
                (r if isinstance(r, bytes) else r.encode())
                if not isinstance(r, list)
                else None
                for r in refs.values()
            ],
        }
    )
    if partition is False:
        df.to_parquet(
            url,
            storage_options=storage_options,
            index=False,
            object_encoding={"raw": "bytes", "key": "utf8", "path": "utf8"},
            stats=["key"],
            has_nulls=["path", "raw"],
            compression="zstd",
            engine="fastparquet",
        )
    else:
        ismeta = df.key.str.contains(".z")
        df[ismeta].to_parquet(
            f"{url}/metadata.parq",
            storage_options=storage_options,
            index=False,
            object_encoding={"raw": "bytes", "key": "utf8", "path": "utf8"},
            stats=["key"],
            has_nulls=["path", "raw"],
            compression="zstd",
            engine="fastparquet",
        )
