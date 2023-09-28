import ast
import io

import numcodecs
from numcodecs.abc import Codec
import numpy as np
import threading


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

    eclock = threading.RLock()

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
        with self.eclock:
            mid = eccodes.codes_new_from_message(bytes(buf))
            try:
                data = eccodes.codes_get_array(mid, var)
                missingValue = eccodes.codes_get_string(mid, "missingValue")
                if var == "values" and missingValue:
                    data[data == float(missingValue)] = np.nan
                if out is not None:
                    return numcodecs.compat.ndarray_copy(data, out)
                else:
                    return data.astype(dt, copy=False)

            finally:
                eccodes.codes_release(mid)


numcodecs.register_codec(GRIBCodec, "grib")


class AsciiTableCodec(numcodecs.abc.Codec):
    """Decodes ASCII-TABLE extensions in FITS files"""

    codec_id = "FITSAscii"

    def __init__(self, indtypes, outdtypes):
        """

        Parameters
        ----------
        indtypes: list[str]
            dtypes of the fields as in the table
        outdtypes: list[str]
            requested final dtypes
        """
        self.indtypes = indtypes
        self.outdtypes = outdtypes

    def decode(self, buf, out=None):
        indtypes = np.dtype([tuple(d) for d in self.indtypes])
        outdtypes = np.dtype([tuple(d) for d in self.outdtypes])
        arr = np.frombuffer(buf, dtype=indtypes)
        return arr.astype(outdtypes)

    def encode(self, _):
        pass


class VarArrCodec(numcodecs.abc.Codec):
    """Variable length arrays in a FITS BINTABLE extension"""

    codec_id = "FITSVarBintable"
    # https://heasarc.gsfc.nasa.gov/docs/software/fitsio/quick/node10.html
    ftypes = {"B": "uint8", "I": ">i2", "J": ">i4"}

    def __init__(self, dt_in, dt_out, nrow, types):
        self.dt_in = dt_in
        self.dt_out = dt_out
        self.nrow = nrow
        self.types = types

    def encode(self, _):
        raise NotImplementedError

    def decode(self, buf, out=None):
        dt_in = np.dtype(ast.literal_eval(self.dt_in))
        dt_out = np.dtype(ast.literal_eval(self.dt_out))
        arr = np.frombuffer(buf, dtype=dt_in, count=self.nrow)
        arr2 = np.empty((self.nrow,), dtype=dt_out)
        heap = buf[arr.nbytes :]
        for name in dt_out.names:

            if dt_out[name] == "O":
                dt = np.dtype(self.ftypes[self.types[name]])
                counts = arr[name][:, 0]
                offs = arr[name][:, 1]
                for i, (off, count) in enumerate(zip(offs, counts)):
                    arr2[name][i] = np.frombuffer(
                        heap[off : off + count * dt.itemsize], dtype=dt
                    )
            else:
                arr2[name][:] = arr[name][:]
        return arr2


numcodecs.register_codec(AsciiTableCodec, "FITSAscii")
numcodecs.register_codec(VarArrCodec, "FITSVarBintable")


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


class DeflateCodec(Codec):
    """As implemented for members of zip-files

    The input buffer contains the file header as well as the compressed bytes
    """

    codec_id = "deflate"

    def decode(self, buf, out=None):
        import zipfile
        import struct

        head = buf[: zipfile.sizeFileHeader]
        *_, csize, usize, fnsize, extra_size = struct.unpack(
            zipfile.structFileHeader, head
        )

        zi = zipfile.ZipInfo()
        zi.compress_size = csize
        zi.file_size = usize
        zi.compress_type = zipfile.ZIP_DEFLATED

        b = io.BytesIO(buf)
        b.seek(zipfile.sizeFileHeader + fnsize + extra_size)
        zf = zipfile.ZipExtFile(b, mode="r", zipinfo=zi)
        return zf.read()

    def encode(self, buf):
        raise NotImplementedError
