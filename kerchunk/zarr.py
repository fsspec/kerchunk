from packaging.version import Version

import fsspec
from fsspec.implementations.reference import LazyReferenceMapper
import zarr

import kerchunk.utils


def is_zarr3():
    """Check if the installed zarr version is version 3"""
    return Version(zarr.__version__) >= Version("3.0.0.a0")


def dict_to_store(store_dict: dict):
    """Create an in memory zarr store backed by the given dictionary"""
    if is_zarr3():
        return zarr.storage.MemoryStore(mode="a", store_dict=store_dict)
    else:
        return zarr.storage.KVStore(store_dict)


def fs_as_store(fs, mode='r', remote_protocol=None, remote_options=None):
    """Open the refs as a zarr store
    
    Parameters
    ----------
    refs: dict-like
        the references to open
    mode: str
    
    Returns
    -------
    zarr.storage.Store or zarr.storage.Mapper, fsspec.AbstractFileSystem
    """
    if is_zarr3():
        return zarr.storage.RemoteStore(fs, mode=mode)
    else:
        return fs.get_mapper()


def single_zarr(
    uri_or_store,
    storage_options=None,
    inline_threshold=100,
    inline=None,
    out=None,
):
    """kerchunk-style view on zarr mapper

    This is a similar process to zarr's consolidate_metadata, but does not
    need to be held in the original file tree. You do not need zarr itself
    to do this.

    This is useful for testing, so that we can pass hand-made zarrs to combine.

    Parameters
    ----------
    uri_or_store: str or dict-like
    storage_options: dict or None
        given to fsspec
    out: dict-like or None
        This allows you to supply an fsspec.implementations.reference.LazyReferenceMapper
        to write out parquet as the references get filled, or some other dictionary-like class
        to customise how references get stored

    Returns
    -------
    reference dict like
    """
    if isinstance(uri_or_store, str):
        mapper = fsspec.get_mapper(uri_or_store, **(storage_options or {}))
    else:
        mapper = uri_or_store
        if isinstance(mapper, fsspec.FSMap) and storage_options is None:
            storage_options = mapper.fs.storage_options

    refs = out or {}
    for k in mapper:
        if k.startswith("."):
            refs[k] = mapper[k]
        else:
            refs[k] = [fsspec.utils._unstrip_protocol(mapper._key_to_str(k), mapper.fs)]
    from kerchunk.utils import do_inline

    inline_threshold = inline or inline_threshold
    if inline_threshold:
        refs = do_inline(refs, inline_threshold, remote_options=storage_options)
    if isinstance(refs, LazyReferenceMapper):
        refs.flush()
    refs = kerchunk.utils.consolidate(refs)
    return refs


ZarrToZarr = kerchunk.utils.class_factory(single_zarr)
