import pandas as pd
import fsspec

from kerchunk.utils import templateize

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


def refs_to_dataframe(
    refs,
    url,
    storage_options=None,
    partition=False,
    template_length=10,
    dict_fraction=0.1,
):
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
    # recoup memory
    fs.clear_instance_cache()
    del fs, refs

    if partition is False:
        templates = None
        haspath = ~df["path"].isna()
        nhaspath = haspath.sum()
        if template_length:
            templates, urls = templateize(
                df["path"][haspath], min_length=template_length
            )
            df.loc[haspath, "path"] = urls
        if (
            dict_fraction
            and nhaspath
            and (df["path"][haspath].nunique() / haspath.sum()) < dict_fraction
        ):
            df["path"] = df["path"].astype("category")
        df.to_parquet(
            url,
            storage_options=storage_options,
            index=False,
            object_encoding={"raw": "bytes", "key": "utf8", "path": "utf8"},
            stats=["key"],
            has_nulls=["path", "raw"],
            compression="zstd",
            engine="fastparquet",
            custom_metadata=templates or None,
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
        gb = df[~ismeta].groupby(df.key.map(lambda s: s.split("/", 1)[0]))
        for prefix, subdf in gb:
            subdf["key"] = subdf.key.str.slice(len(prefix) + 1, None)
            templates = None
            haspath = ~subdf["path"].isna()
            nhaspath = haspath.sum()
            if template_length:
                templates, urls = templateize(
                    subdf["path"][haspath], min_length=template_length
                )
                subdf.loc[haspath, "path"] = urls
            if (
                dict_fraction
                and nhaspath
                and (subdf["path"][haspath].nunique() / haspath.sum()) < dict_fraction
            ):
                subdf["path"] = subdf["path"].astype("category")

            subdf.to_parquet(
                f"{url}/{prefix}.parq",
                storage_options=storage_options,
                index=False,
                object_encoding={"raw": "bytes", "key": "utf8", "path": "utf8"},
                stats=["key"],
                has_nulls=["path", "raw"],
                compression="zstd",
                engine="fastparquet",
                custom_metadata=templates or None,
            )
