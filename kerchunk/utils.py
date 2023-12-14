import base64
import copy
import itertools
import warnings

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


def do_inline(store, threshold, remote_options=None, remote_protocol=None):
    """Replace short chunks with the value of that chunk and inline metadata

    The chunk may need encoding with base64 if not ascii, so actual
    length may be larger than threshold.
    """
    fs = fsspec.filesystem(
        "reference",
        fo=store,
        remote_options=remote_options,
        remote_protocol=remote_protocol,
    )
    out = fs.references.copy()

    # Inlining is done when one of two conditions are satisfied:
    # 1. The item is small enough, i.e. smaller than the threshold specified in the function call
    # 2. The item is a metadata file, i.e. a .z* file
    get_keys = [
        k
        for k, v in out.items()
        if (isinstance(v, list) and len(v) == 3 and v[2] < threshold)
        or (
            isinstance(v, list)
            and len(v) == 1
            and isinstance(v[0], str)
            and v[0].split("/")[-1].startswith(".z")
        )
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
            if cond1 or cond2:
                original_attrs = dict(thing.attrs)
                arr = group.create_dataset(
                    name=name,
                    dtype=thing.dtype,
                    shape=thing.shape,
                    data=thing[:],
                    chunks=thing.shape,
                    compression=None,
                    overwrite=True,
                )
                arr.attrs.update(original_attrs)


def inline_array(store, threshold=1000, names=None, remote_options=None):
    """Inline whole arrays by threshold or name, replace with a single metadata chunk

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


def subchunk(store, variable, factor):
    """
    Split uncompressed chunks into integer subchunks on the largest axis

    Parameters
    ----------
    store: dict
        reference set
    variable: str
        the named zarr variable (give as /-separated path if deep)
    factor: int
        the number of chunks each input chunk turns into. Must be an exact divisor
        of the original largest dimension length.

    Returns
    -------
    modified store
    """
    fs = fsspec.filesystem("reference", fo=store)
    store = copy.deepcopy(store)
    meta_file = f"{variable}/.zarray"
    meta = ujson.loads(fs.cat(meta_file))
    if meta["compressor"] is not None:
        raise ValueError("Can only subchunk an uncompressed array")
    chunks_orig = meta["chunks"]
    if chunks_orig[0] % factor == 0:
        chunk_new = [chunks_orig[0] // factor] + chunks_orig[1:]
    else:
        raise ValueError("Must subchunk by exact integer factor")

    meta["chunks"] = chunk_new
    store[meta_file] = ujson.dumps(meta)

    for k, v in store.copy().items():
        if k.startswith(f"{variable}/"):
            kpart = k[len(variable) + 1 :]
            if kpart.startswith(".z"):
                continue
            sep = "." if "." in k else "/"
            chunk_index = [int(_) for _ in kpart.split(sep)]
            if len(v) > 1:
                url, offset, size = v
            else:
                (url,) = v
                offset = 0
                size = fs.size(k)
            for subpart in range(factor):
                new_index = [chunk_index[0] * factor + subpart] + chunk_index[1:]
                newpart = sep.join(str(_) for _ in new_index)
                newv = [url, offset + subpart * size // factor, size // factor]
                store[f"{variable}/{newpart}"] = newv
    return store


def dereference_archives(references, remote_options=None):
    """Directly point to uncompressed byte ranges in ZIP/TAR archives

    If a set of references have been made for files contained within ZIP or
    (uncompressed) TAR archives, the "zip://..." and "tar://..." URLs should
    be converted to byte ranges in the overall file.

    Parameters
    ----------
    references: dict
        a simple reference set
    remote_options: dict or None
        For opening the archives
    """
    import zipfile
    import tarfile

    if "version" in references and references["version"] == 1:
        references = references["refs"]

    target_files = [l[0] for l in references.values() if isinstance(l, list)]
    target_files = {
        (t.split("::", 1)[1], t[:3])
        for t in target_files
        if t.startswith(("tar://", "zip://"))
    }

    # find all member file offsets in all archives
    offsets = {}
    for target, tar_or_zip in target_files:
        with fsspec.open(target, **(remote_options or {})) as tf:
            if tar_or_zip == "tar":
                tar = tarfile.TarFile(fileobj=tf)
                offsets[target] = {
                    ti.name: {"offset": ti.offset_data, "size": ti.size, "comp": False}
                    for ti in tar.getmembers()
                    if ti.isfile()
                }
            elif tar_or_zip == "zip":
                zf = zipfile.ZipFile(file=tf)
                offsets[target] = {}
                for zipinfo in zf.filelist:
                    if zipinfo.is_dir():
                        continue
                    # if uncompressed, include only the buffer. In compressed (DEFLATE), include
                    # also the header, and must use DeflateCodec
                    if zipinfo.compress_type == zipfile.ZIP_DEFLATED:
                        # TODO: find relevant .zarray and add filter directly
                        header = 0
                        warnings.warn(
                            "ZIP file contains compressed files, must use DeflateCodec"
                        )
                        tail = len(zipinfo.FileHeader())
                    elif zipinfo.compress_type == zipfile.ZIP_STORED:
                        header = len(zipinfo.FileHeader())
                        tail = 0
                    else:
                        comp = zipfile.compressor_names[zipinfo.compress_type]
                        raise ValueError(
                            f"ZIP compression method not supported: {comp}"
                        )
                    offsets[target][zipinfo.filename] = {
                        "offset": zipinfo.header_offset + header,
                        "size": zipinfo.compress_size + tail,
                        "comp": zipinfo.compress_type != zipfile.ZIP_STORED,
                    }

    # modify references
    mods = copy.deepcopy(references)
    for k, v in mods.items():
        if not isinstance(v, list):
            continue
        target = v[0].split("::", 1)[1]
        infile = v[0].split("::", 1)[0][6:]  # strip "zip://" or "tar://"
        if target not in offsets:
            continue
        detail = offsets[target][infile]
        if detail["comp"]:
            # leave compressed member file alone
            pass
        v[0] = target
        if len(v) == 1:
            v.append(detail["offset"])
            v.append(detail["size"])
        else:
            v[1] += detail["offset"]
    return mods


def _max_prefix(*strings):
    # https://stackoverflow.com/a/6719272/3821154
    def all_same(x):
        return all(x[0] == y for y in x)

    char_tuples = zip(*strings)
    prefix_tuples = itertools.takewhile(all_same, char_tuples)
    return "".join(x[0] for x in prefix_tuples)


def templateize(strings, min_length=10, template_name="u"):
    """Make prefix template for a set of strings

    Useful for condensing strings by extracting out a common prefix.
    If the common prefix is shorted than ``min_length``, the original
    strings are returned and the output templates are empty.

    Parameters
    ----------
    strings: List[str]
        inputs
    min_length: int
        Only perform transformm if the common prefix is at least this long.
    template_name: str
        The placeholder string, should be short.

    Returns
    -------
    templates: Dict[str, str], strings: List[str]
    Such that [s.format(**templates) for s in strings] recreates original strings list
    """
    prefix = _max_prefix(*strings)
    lpref = len(prefix)
    if lpref >= min_length:
        template = {template_name: prefix}
        strings = [("{%s}" % template_name) + s[lpref:] for s in strings]
    else:
        template = {}
    return template, strings
