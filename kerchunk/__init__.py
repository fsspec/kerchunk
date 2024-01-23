from . import codecs

from importlib.metadata import version as _version

try:
    __version__ = _version("kerchunk")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "9999"

__all__ = ["__version__"]


def set_reference_filesystem_cachable(cachable=True):
    """While experimenting with kerchunk and referenceFS, it can be convenient to not cache FS instances

    You may wish to call this function with ``False`` before any kerchunking session; leaving
    the instances cachable (the default) is what end-users will want, since it will be
    more efficient.
    """
    import fsspec

    fsspec.get_filesystem_class("reference").cachable = cachable
