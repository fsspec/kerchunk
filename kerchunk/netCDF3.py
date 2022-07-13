from functools import reduce
from operator import mul

from numcodecs.abc import Codec
import numpy as np
try:
    # causes warning on newer scipy, all moved to scipy.io
    # TODO: this construct is only here to make the codec importable without
    #  scipy. Should instead move to separate module.
    from scipy.io.netcdf import ZERO, NC_VARIABLE, netcdf_file, netcdf_variable
except ImportError:

    netcdf_file = object

import fsspec


class NetCDF3ToZarr(netcdf_file):
    """Generate references for a netCDF3 file

    Uses scipy's netCDF3 reader, but only reads the metadata. Note that instances
    do behave like actual scipy netcdf files, but contain no valid data.
    """

    def __init__(self, filename, *args, storage_options=None,
                 inline_threshold=100, **kwargs):
        """
        Parameters
        ----------
        filename: str
            location of the input
        storage_options: dict
            passed to fsspec when opening filename
        inline_threshold: int
            Byte size below which an array will be embedded in the output
        args, kwargs: passed to scipy superclass ``scipy.io.netcdf.netcdf_file``
        """
        if netcdf_file is object:
            raise ImportError("scipy was not imported, and is required for netCDF3")
        assert kwargs.pop("mmap", False) is False
        assert kwargs.pop("mode", "r") == "r"
        assert kwargs.pop("maskandscale", False) is False
        self.chunks = {}
        self.inline = inline_threshold
        with fsspec.open(filename, **(storage_options or {})) as fp:
            super().__init__(fp, *args, mmap=False, mode="r", maskandscale=False, **kwargs)
        self.filename = filename

    def _read_var_array(self):
        header = self.fp.read(4)
        if header not in [ZERO, NC_VARIABLE]:
            raise ValueError("Unexpected header.")

        begin = 0
        dtypes = {'names': [], 'formats': []}
        rec_vars = []
        count = self._unpack_int()
        for var in range(count):
            (name, dimensions, shape, attributes,
             typecode, size, dtype_, begin_, vsize) = self._read_var()
            if shape and shape[0] is None:  # record variable
                rec_vars.append(name)
                # The netCDF "record size" is calculated as the sum of
                # the vsize's of all the record variables.
                self.__dict__['_recsize'] += vsize
                if begin == 0:
                    begin = begin_
                dtypes['names'].append(name)
                dtypes['formats'].append(str(shape[1:]) + dtype_)

                # Handle padding with a virtual variable.
                if typecode in 'bch':
                    actual_size = reduce(mul, (1,) + shape[1:]) * size
                    padding = -actual_size % 4
                    if padding:
                        dtypes['names'].append('_padding_%d' % var)
                        dtypes['formats'].append('(%d,)>b' % padding)

                # Data will be set later.
                data = None
            else:  # not a record variable
                # Calculate size to avoid problems with vsize (above)
                a_size = reduce(mul, shape, 1) * size
                self.chunks[name] = [begin_, a_size, dtype_, shape]
                if name in ["latitude", "longitude", "time"]:
                    pos = self.fp.tell()
                    self.fp.seek(begin_)
                    data = np.frombuffer(self.fp.read(a_size), dtype=dtype_
                                         ).copy()
                    # data.shape = shape
                    self.fp.seek(pos)
                else:
                    data = np.empty(1, dtype=dtype_)

            # Add variable.
            self.variables[name] = netcdf_variable(
                data, typecode, size, shape, dimensions, attributes,
                maskandscale=self.maskandscale)

        if rec_vars:
            # Remove padding when only one record variable.
            if len(rec_vars) == 1:
                dtypes['names'] = dtypes['names'][:1]
                dtypes['formats'] = dtypes['formats'][:1]

            pos = self.fp.tell()
            self.fp.seek(begin)
            self.chunks.setdefault("record_array", []).append([begin, self._recs*self._recsize, dtypes])
            # rec_array = frombuffer(self.fp.read(self._recs*self._recsize),
            #                        dtype=dtypes).copy()
            # rec_array.shape = (self._recs,)
            self.fp.seek(pos)

    def translate(self, max_chunk_size=0):
        """
        Produce references dictionary

        Parameters
        ----------
        max_chunk_size: int
            How big a chunk can be before triggering subchunking. If 0, there is no
            subchunking, and there is never subchunking for coordinate/dimension arrays.
            E.g., if an array contains 10,000bytes, and this value is 6000, there will
            be two output chunks, split on the biggest available dimension.
        """
        import zarr
        if threshold or max_chunk_size:
            raise NotImplementedError
        out = {}
        z = zarr.open(out, mode='w')
        for dim, var in self.variables.items():
            if dim in self.dimensions:
                shape = self.dimensions[dim]
            else:
                shape = self.chunks[dim][-1]
            if isinstance(shape, int):
                shape = shape,
            if shape is None or (len(shape) and shape[0] is None):
                # record array: either simple chunks, or use codec
                data = var[:]
                arr = z.create_dataset(name=dim, data=data, chunks=data.shape, compression=None)
            else:
                # simple array block
                arr = z.empty(name=dim, shape=shape, dtype=var.data.dtype, chunks=shape,
                              compression=None)
                part = ".".join(["0"] * len(shape)) or "0"
                out[f"{dim}/{part}"] = [self.filename] + [int(self.chunks[dim][0]), int(self.chunks[dim][1])]
            arr.attrs.update(
                {k: v.decode() if isinstance(v, bytes) else str(v)
                 for k, v in var._attributes.items()}
            )
            arr.attrs['_ARRAY_DIMENSIONS'] = list(var.dimensions)
        return out


netcdf_recording_file = NetCDF3ToZarr


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

