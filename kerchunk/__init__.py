from . import codecs

from importlib.metadata import version as _version

try:
    __version__ = _version("kerchunk")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "9999"

__all__ = ["__version__"]
