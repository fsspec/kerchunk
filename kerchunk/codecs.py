import ast
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
        if "[" in dtype:
            self.dtype = ast.literal_eval(dtype)
        else:
            self.dtype = dtype
        self.id_map = id_map

    def encode(self, buf):
        raise NotImplementedError

    def decode(self, buf, out=None):
        if isinstance(self.dtype, list):
            dt = [tuple(_) for _ in self.dtype]
        else:
            dt = self.dtype
        arr = np.frombuffer(buf, dtype=dt).copy()
        if arr.dtype.kind in "SU":
            if isinstance(self.id_map, dict):
                arr = np.array([self.id_map[_.decode()] for _ in arr], dtype="O")
            else:
                arr = np.full(arr.shape, self.id_map, dtype="O")
            return arr
        elif arr.dtype.kind == "V":
            dt2 = []
            for name in arr.dtype.names:
                if arr.dtype[name].kind in "SU":
                    dt2.append((name, "O"))
                else:
                    dt2.append((name, arr.dtype[name]))
            arr2 = np.empty(arr.shape, dtype=dt2)
            for name in arr.dtype.names:
                if arr[name].dtype.kind in "SU":
                    if isinstance(self.id_map, dict):
                        arr2[name][:] = [self.id_map[_.decode()] for _ in arr[name]]
                    else:
                        arr2[name][:] = self.id_map
                else:
                    arr2[name][:] = arr[name]

            return arr2


numcodecs.register_codec(FillStringsCodec, "fill_hdf_strings")


class GRIBCodec(numcodecs.abc.Codec):
    """
    Read GRIB stream of bytes as a message using eccodes
    """

    codec_id = "grib"

    def __init__(self, var, dtype=None):
        self.var = var
        self.dtype = dtype

    def encode(self, buf):
        # on encode, pass through
        return buf

    def decode(self, buf, out=None):
        import eccodes

        if self.var in ["latitude", "longitude"]:
            var = self.var + "s"
            dt = self.dtype or "float64"
        else:
            var = "values"
            dt = self.dtype or "float32"
        mid = eccodes.codes_new_from_message(bytes(buf))
        try:
            data = eccodes.codes_get_array(mid, var)
        finally:
            eccodes.codes_release(mid)

        if out is not None:
            return numcodecs.compat.ndarray_copy(data, out)
        else:
            return data.astype(dt)


numcodecs.register_codec(GRIBCodec, "grib")


class AsciiTableCodec(numcodecs.abc.Codec):

    codec_id = "FITSAscii"

    def __init__(self, indtypes, outdtypes):
        self.indtypes = indtypes
        self.outdtypes = outdtypes

    def decode(self, buf, out=None):
        indtypes = np.dtype([tuple(d) for d in self.indtypes])
        outdtypes = np.dtype([tuple(d) for d in self.outdtypes])
        arr = np.fromstring(buf, dtype=indtypes)
        return arr.astype(outdtypes)

    def encode(self, _):
        pass


numcodecs.register_codec(AsciiTableCodec, "FITSAscii")


class RecordArrayMember(Codec):
    """Read components of a record array (complex dtype)"""

    codec_id = "record_member"

    def __init__(self, member, dtype):
        """
        Parameters
        ----------
        member: str
            name of desired subarray
        dtype: list of lists
            description of the complex dtype of the overall record array. Must be both
            parsable by ``np.dtype()`` and also be JSON serialisable
        """
        self.member = member
        self.dtype = dtype

    def decode(self, buf, out=None):
        arr = np.frombuffer(buf, dtype=np.dtype(self.dtype))
        return arr[self.member].copy()

    def encode(self, buf):
        raise NotImplementedError
