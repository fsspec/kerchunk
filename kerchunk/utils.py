import base64

import ujson

import fsspec
import zarr


def class_factory(func):
    """Experimental uniform API across function-based file scanners"""

    class FunctionWrapper:
        __doc__ = func.__doc__
        __module__ = func.__module__

        def __init__(self, url, storage_options=None, inline_threshold=100, **kwargs):
            self.url = url
            self.storage_options = storage_options
            self.inline = inline_threshold
            self.kwargs = kwargs

        def translate(self):
            return func(
                self.url,
                inline_threshold=self.inline,
                storage_options=self.storage_options,
                **self.kwargs,
            )

        def __str__(self):
            return f"<Single file to zarr processor using {func.__module__}.{func.__qualname__}>"

        __repr__ = __str__

    return FunctionWrapper


def consolidate(refs):
    """Turn raw references into output"""
    out = {}
    for k, v in refs.items():
        if isinstance(v, bytes):
            try:
                # easiest way to test if data is ascii
                out[k] = v.decode("ascii")
            except UnicodeDecodeError:
                out[k] = (b"base64:" + base64.b64encode(v)).decode()
        else:
            out[k] = v
    return {"version": 1, "refs": out}


def rename_target(refs, renames):
    """Utility to change URLs in a reference set in a predictable way

    For reference sets including templates, this is more easily done by
    using template overrides at access time; but rewriting the references
    and saving a new file means not having to do that every time.

    Parameters
    ----------
    refs: dict
        Reference set
    renames: dict[str, str]
        Mapping from the old URL (including protocol, if this is how they appear
        in the original) to new URL

    Returns
    -------
    dict: the altered reference set, which can be saved
    """
    fs = fsspec.filesystem("reference", fo=refs)  # to produce normalised refs
    refs = fs.references
    out = {}
    for k, v in refs.items():
        if isinstance(v, list) and v[0] in renames:
            out[k] = [renames[v[0]]] + v[1:]
        else:
            out[k] = v
    return consolidate(out)


def rename_target_files(
    url_in, renames, url_out=None, storage_options_in=None, storage_options_out=None
):
    """Perform URL renames on a reference set - read and write from JSON

    Parameters
    ----------
    url_in: str
        Original JSON reference set
    renames: dict
        URL renamings to perform (see ``renate_target``)
    url_out: str | None
        Where to write to. If None, overwrites original
    storage_options_in: dict | None
        passed to fsspec for opening url_in
    storage_options_out: dict | None
        passed to fsspec for opening url_out. If None, storage_options_in is used.

    Returns
    -------
    None
    """
    with fsspec.open(url_in, **(storage_options_in or {})) as f:
        old = ujson.load(f)
    new = rename_target(old, renames)
    if url_out is None:
        url_out = url_in
    if storage_options_out is None:
        storage_options_out = storage_options_in
    with fsspec.open(url_out, mode="wt", **(storage_options_out or {})) as f:
        ujson.dump(new, f)


def _encode_for_JSON(store):
    """Make store JSON encodable"""
    for k, v in store.copy().items():
        if isinstance(v, list):
            continue
        else:
            try:
                # minify JSON
                v = ujson.dumps(ujson.loads(v))
            except (ValueError, TypeError):
                pass
            try:
                store[k] = v.decode() if isinstance(v, bytes) else v
            except UnicodeDecodeError:
                store[k] = "base64:" + base64.b64encode(v).decode()
    return store


def _do_inline(store, threshold, remote_options=None):
    """Replace short chunks with the value of that chunk

    The chunk may need encoding with base64 if not ascii, so actual
    length may be larger than threshold.
    """
    fs = fsspec.filesystem("reference", fo=store, **(remote_options or {}))
    out = fs.references.copy()
    get_keys = [
        k
        for k, v in out.items()
        if isinstance(v, list) and len(v) == 3 and v[2] < threshold
    ]
    values = fs.cat(get_keys)
    for k, v in values.items():
        try:
            # easiest way to test if data is ascii
            v.decode("ascii")
        except UnicodeDecodeError:
            v = b"base64:" + base64.b64encode(v)
        out[k] = v
    return out


def _inline_array(group, threshold, names, prefix=""):
    for name, thing in group.items():
        if prefix:
            prefix1 = f"{prefix}.{name}"
        else:
            prefix1 = name
        if isinstance(thing, zarr.Group):
            _inline_array(thing, threshold=threshold, prefix=prefix1, names=names)
        else:
            cond1 = threshold and thing.nbytes < threshold and thing.nchunks > 1
            cond2 = prefix1 in names
            print(name, prefix1, cond1, cond2)
            if cond1 or cond2:
                print("inline")
                group.create_dataset(
                    name=name,
                    dtype=thing.dtype,
                    shape=thing.shape,
                    data=thing[:],
                    chunks=thing.shape,
                    compression=None,
                    overwrite=True,
                )


def inline_array(store, threshold=1000, names=None, remote_options=None):
    """Inline whole arrays by threshold or name, repace with a single metadata chunk

    Inlining whole arrays results in fewer keys. If the constituent keys were
    already inlined, this also results in a smaller file overall. No action is taken
    for arrays that are already of one chunk (they should be in

    Parameters
    ----------
    store: dict/JSON file
        reference set
    threshold: int
        Size in bytes below which to inline. Set to 0 to prevent inlining by size
    names: list[str] | None
        It the array name (as a dotted full path) appears in this list, it will
        be inlined irrespective of the threshold size. Useful for coordinates.
    remote_options: dict | None
        Needed to fetch data, if the required keys are not already individually inlined
        in the data.

    Returns
    -------
    amended references set (simple style)
    """
    fs = fsspec.filesystem(
        "reference", fo=store, **(remote_options or {}), skip_instance_cache=True
    )
    g = zarr.open_group(fs.get_mapper(), mode="r+")
    _inline_array(g, threshold, names=names or [])
    return fs.references
