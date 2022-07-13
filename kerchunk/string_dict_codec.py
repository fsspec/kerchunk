from numcodecs.abc import Codec
import numpy as np
import numcodecs


class StringDict(Codec):
    # similar to "categorize", but the keys are preassigned

    codec_id = "string_dict"

    def __init__(self, mapping, dtype=None):
        self.mapping = mapping
        self.dtype = dtype

    def encode(self, buf):
        # read-only codec
        raise NotImplementedError

    def decode(self, buf, out=None):
        values = [self.mapping[k] for k in buf]
        if out is None:
            out = np.array(values, dtype=self.dtype)
        else:
            out[:] = values
        return out


class ComplexStringDict(Codec):
    # specifically for HDF's tables (compound dtypes) with embedded string pointers

    codec_id = "comlpex_string_dict"

    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, buf):
        # read-only codec
        raise NotImplementedError

    def decode(self, buf, out=None):
        return buf
        values = [self.mapping[k] for k in buf]
        if out is None:
            out = np.array(values, dtype=self.dtype)
        else:
            out[:] = values
        return out


class DebugCodec(Codec):

    codec_id = "debug"

    def encode(self, buf):
        return buf

    def decode(self, buf, out=None):
        import pdb

        pdb.set_trace()
        return buf


numcodecs.register_codec(DebugCodec, "debug")
