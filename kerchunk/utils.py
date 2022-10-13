import ujson

import fsspec


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
    refs = fs.refs
    out = {}
    for k, v in refs.items():
        if isinstance(v, list) and v[0] in renames:
            out[k] = [renames[v[0]]] + v[1:]
        else:
            out[k] = v
    return out


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
        old = ujson.loads(f)
    new = rename_target(old, renames)
    if url_out is None:
        url_out = url_in
    if storage_options_out is None:
        storage_options_out = storage_options_in
    with fsspec.open(url_out, mode="wb", **(storage_options_out or {})) as f:
        ujson.dump(new, f)
