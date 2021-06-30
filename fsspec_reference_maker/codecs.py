import numcodecs.abc
from numcodecs.compat import ndarray_copy, ensure_contiguous_ndarray
import numpy as np


class StringDictCodec(numcodecs.abc.Codec):

    codec_id = "dict_string"

    def __init__(self, d):
        """

        :param d: dict(16-byte-string, bytestring)
        """
        self.d = d

    def encode(self, buf):
        # on encode, pass through
        return buf

    def decode(self, buf, out=None):
        out = ensure_contiguous_ndarray(out)
        ids = np.frombuffer(buf, dtype="S16")
        out[:] = [self.d[i] for i in ids]
        return out


class GRIBCodec(numcodecs.abc.Codec):
    """
    Read GRIB stream of bytes by writing to a temp file and calling cfgrib
    """

    codec_id = 'grib'

    def __init__(self, var):
        self.var = var

    def encode(self, buf):
        # on encode, pass through
        return buf

    def decode(self, buf, out=None):
        import cfgrib
        buf = ensure_contiguous_ndarray(buf)
        fn = tempfile.mktemp(suffix="grib2")
        buf.tofile(fn)

        # do decode
        ds = cfgrib.open_file(fn)
        data = ds.variables[self.var].data
        if hasattr(data, "build_array"):
            data = data.build_array()

        if out is not None:
            return ndarray_copy(data, out)
        else:
            return data


numcodecs.register_codec(GRIBCodec, "grib")
numcodecs.register_codec(StringDictCodec, "dict_string")
