import numcodecs
from numcodecs.abc import Codec
import numpy as np


class FillStringsCodec(Codec):
    """Sets fixed-length string fields to empty

    To be used with HDF fields of strings, to fill in the valules of the opaque
    16-byte string IDs.
    """

    codec_id = "fill_hdf_strings"

    def __init__(self, dtype, id_map=None):
        """
        Note: we must pass id_map using strings, because this is JSON-encoded
        by zarr.

        Parameters
        ----------
        id_map: None | str | dict(str, str)
        """
        self.dtype = dtype
        self.id_map = id_map

    def encode(self, buf):
        raise NotImplementedError

    def decode(self, buf, out=None):
        arr = np.frombuffer(buf, dtype=self.dtype)
        if arr.dtype.kind in "SU":
            if self.id_map is None:
                arr = np.zeros(arr.shape, dtype="O")
            elif isinstance(self.id_map, (str, bytes)):
                arr = np.full(arr.shape, self.id_map, dtype="O")
            else:
                arr = np.array([self.id_map[_.decode()] for _ in arr], dtype="O")
        elif arr.dtype.kind == "V":
            for name in arr.dtype.names:
                if arr[name].dtype.kind in "SU":
                    arr[name][:] = ""
        return arr


numcodecs.register_codec(FillStringsCodec, "fill_hdf_strings")
