import base64
import copy
import itertools
import fsspec.asyn
from typing import Any, cast
import warnings

import ujson

import fsspec.implementations.asyn_wrapper
import numpy as np
import zarr.storage


def dict_to_store(store_dict: dict):
    """Create an in memory zarr store backed by the given dictionary"""
    return zarr.storage.MemoryStore(read_only=False, store_dict=store_dict)


def refs_as_fs(
    refs,
    fs=None,
    remote_protocol=None,
    remote_options=None,
    asynchronous=True,
    **kwargs,
):
    """Convert a reference set to an fsspec filesystem"""
    fs = fsspec.filesystem(
        "reference",
        fo=refs,
        fs=fs,
        remote_protocol=remote_protocol,
        remote_options=remote_options,
        **kwargs,
        asynchronous=asynchronous,
    )
    return fs


def refs_as_store(
    refs, read_only=False, fs=None, remote_protocol=None, remote_options=None
):
    """Convert a reference set to a zarr store"""
    remote_options = remote_options or {}
    remote_options["asynchronous"] = True

    fss = refs_as_fs(
        refs,
        fs=fs,
        remote_protocol=remote_protocol,
        remote_options=remote_options,
    )
    return fs_as_store(fss, read_only=read_only)


def fs_as_store(fs: fsspec.asyn.AsyncFileSystem, read_only=False):
    """Open the refs as a zarr store

    Parameters
    ----------
    fs: fsspec.async.AsyncFileSystem
    read_only: bool

    Returns
    -------
    zarr.storage.Store or zarr.storage.Mapper, fsspec.AbstractFileSystem
    """
    if not fs.async_impl:
        try:
            from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper

            fs = AsyncFileSystemWrapper(fs)
        except ImportError:
            raise ImportError(
                "Only fsspec>2024.10.0 supports the async filesystem wrapper "
                "required for working with reference filesystems. "
            )
    fs.asynchronous = True
    return zarr.storage.FsspecStore(fs, read_only=read_only)


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
        if hasattr(v, "to_bytes"):
            v = v.to_bytes()
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
    fs = refs_as_fs(refs)  # to produce normalised refs
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


def encode_fill_value(v: Any, dtype: np.dtype, compressor: Any = None) -> Any:
    # early out
    if v is None:
        return v
    if dtype.kind == "V" and dtype.hasobject:
        if compressor is None:
            raise ValueError("missing compressor for object array")
        v = compressor.encode(v)
        v = str(base64.standard_b64encode(v), "ascii")
        return v
    if dtype.kind == "f":
        if np.isnan(v):
            return "NaN"
        elif np.isposinf(v):
            return "Infinity"
        elif np.isneginf(v):
            return "-Infinity"
        else:
            return float(v)
    elif dtype.kind in "ui":
        return int(v)
    elif dtype.kind == "b":
        return bool(v)
    elif dtype.kind in "c":
        c = cast(np.complex128, np.dtype(complex).type())
        v = (
            encode_fill_value(v.real, c.real.dtype, compressor),
            encode_fill_value(v.imag, c.imag.dtype, compressor),
        )
        return v
    elif dtype.kind in "SV":
        v = str(base64.standard_b64encode(v), "ascii")
        return v
    elif dtype.kind == "U":
        return v
    elif dtype.kind in "mM":
        return int(v.view("i8"))
    else:
        return v


def do_inline(store, threshold, remote_options=None, remote_protocol=None):
    """Replace short chunks with the value of that chunk and inline metadata

    The chunk may need encoding with base64 if not ascii, so actual
    length may be larger than threshold.
    """
    fs = refs_as_fs(
        store,
        remote_protocol=remote_protocol,
        remote_options=remote_options,
        asynchronous=False,
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
    for name, thing in group.members():
        if prefix:
            prefix1 = f"{prefix}.{name}"
        else:
            prefix1 = name
        if isinstance(thing, zarr.Group):
            _inline_array(thing, threshold=threshold, prefix=prefix1, names=names)
        else:
            cond1 = threshold and thing.nbytes < threshold
            cond2 = prefix1 in names
            if cond1 or cond2:
                original_attrs = dict(thing.attrs)
                arr = group.create_array(
                    name=name,
                    dtype=thing.dtype,
                    shape=thing.shape,
                    chunks=thing.shape,
                    fill_value=thing.fill_value,
                    overwrite=True,
                )
                arr[:] = thing[:]
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
    fs = refs_as_fs(store, remote_options=remote_options or {})
    zarr_store = fs_as_store(fs, read_only=False)
    g = zarr.open_group(zarr_store, zarr_format=2)
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
    store = fs.references
    meta_file = f"{variable}/.zarray"
    meta = ujson.loads(fs.cat(meta_file))
    if meta["compressor"] is not None:
        raise ValueError("Can only subchunk an uncompressed array")
    chunks_orig = meta["chunks"]
    chunk_new = []
    # plan
    multi = None
    for ind, this_chunk in enumerate(chunks_orig):
        if this_chunk == 1:
            chunk_new.append(1)
            continue
        elif this_chunk % factor == 0:
            chunk_new.extend([this_chunk // factor] + chunks_orig[ind + 1 :])
            break
        elif factor % this_chunk == 0:
            # if factor // chunks_orig[0] > 1:
            chunk_new.append(1)
            if multi is None:
                multi = this_chunk
            factor //= this_chunk
        else:
            raise ValueError("Must subchunk by exact integer factor")

    if multi:
        # TODO: this reloads the referenceFS; *maybe* reuses it
        return subchunk(store, variable, multi)

    # execute
    meta["chunks"] = chunk_new
    store = copy.deepcopy(store)
    store[meta_file] = ujson.dumps(meta)

    for k, v in store.copy().items():
        if k.startswith(f"{variable}/"):
            kpart = k[len(variable) + 1 :]
            if kpart.startswith(".z"):
                continue
            sep = "." if "." in kpart else "/"
            chunk_index = [int(_) for _ in kpart.split(sep)]
            if isinstance(v, (str, bytes)):
                # TODO: check this early, as some refs might have been edited already
                raise ValueError("Refusing to sub-chunk inlined data")
            if len(v) > 1:
                url, offset, size = v
            else:
                (url,) = v
                offset = 0
                size = fs.info(k)["size"]
            for subpart in range(factor):
                new_index = (
                    chunk_index[:ind]
                    + [chunk_index[ind] * factor + subpart]
                    + chunk_index[ind + 1 :]
                )
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


def translate_refs_serializable(refs: dict):
    """Translate a reference set to a serializable form, given that zarr
    v3 memory stores store data in buffers by default. This modifies the
    input dictionary in place, and returns a reference to it.

    It also fixes keys that have a leading slash, which is not appropriate for
    zarr v3 keys

    Parameters
    ----------
    refs: dict
        The reference set

    Returns
    -------
    dict
        A serializable form of the reference set
    """
    keys_to_remove = []
    new_keys = {}
    for k, v in refs.items():
        if isinstance(v, zarr.core.buffer.cpu.Buffer):
            key = k.removeprefix("/")
            new_keys[key] = v.to_bytes()
            keys_to_remove.append(k)
    for k in keys_to_remove:
        del refs[k]
    refs.update(new_keys)
    return refs
