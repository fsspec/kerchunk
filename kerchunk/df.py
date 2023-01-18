import base64

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


def _proc_raw(r):
    if not isinstance(r, bytes):
        r = r.encode()
    if r.startswith(b"base64:"):
        return base64.b64decode(r[7:])
    return r


def refs_to_dataframe(
    refs,
    url,
    storage_options=None,
    partition=False,
    template_length=10,
    dict_fraction=0.1,
    min_refs=100,
):
    # normalise refs (e.g., for templates)
    fs = fsspec.filesystem("reference", fo=refs)
    refs = fs.references

    df = pd.DataFrame(
        {
            "key": list(refs),
            # TODO: could get unique values using set() here and make categorical
            #  columns with pd.Categorical.from_codes if it meets criterion
            "path": [r[0] if isinstance(r, list) else None for r in refs.values()],
            "offset": [
                r[1] if isinstance(r, list) and len(r) > 1 else 0 for r in refs.values()
            ],
            "size": [
                r[2] if isinstance(r, list) and len(r) > 1 else 0 for r in refs.values()
            ],
            "raw": [
                _proc_raw(r) if not isinstance(r, list) else None for r in refs.values()
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
        extra_inds = []
        gb = df[~ismeta].groupby(df.key.map(lambda s: s.split("/", 1)[0]))
        prefs = {"metadata"}
        for prefix, subdf in gb:
            if len(subdf) < min_refs:
                ind = ismeta[~ismeta].iloc[gb.indices[prefix]].index
                extra_inds.extend(ind.tolist())
                prefs.add(prefix)
                continue
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
        ismeta[extra_inds] = True
        df[ismeta].to_parquet(
            f"{url}/metadata.parq",
            storage_options=storage_options,
            index=False,
            object_encoding={"raw": "bytes", "key": "utf8", "path": "utf8"},
            stats=["key"],
            has_nulls=["path", "raw"],
            compression="zstd",
            engine="fastparquet",
            custom_metadata={"prefs": str(prefs)},
        )
