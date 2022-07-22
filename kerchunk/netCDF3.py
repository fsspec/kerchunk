from functools import reduce
from operator import mul

import numpy as np

from scipy.io.netcdf import ZERO, NC_VARIABLE, netcdf_file, netcdf_variable

import fsspec


class NetCDF3ToZarr(netcdf_file):
    """Generate references for a netCDF3 file

    Uses scipy's netCDF3 reader, but only reads the metadata. Note that instances
    do behave like actual scipy netcdf files, but contain no valid data.
    """

    def __init__(
        self,
        filename,
        *args,
        storage_options=None,
        inline_threshold=100,
        max_chunk_size=0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        filename: str
            location of the input
        storage_options: dict
            passed to fsspec when opening filename
        inline_threshold: int
            Byte size below which an array will be embedded in the output [TBC]
        max_chunk_size: int
            How big a chunk can be before triggering subchunking. If 0, there is no
            subchunking, and there is never subchunking for coordinate/dimension arrays.
            E.g., if an array contains 10,000bytes, and this value is 6000, there will
            be two output chunks, split on the biggest available dimension. [TBC]
        args, kwargs: passed to scipy superclass ``scipy.io.netcdf.netcdf_file``
        """
        assert kwargs.pop("mmap", False) is False
        assert kwargs.pop("mode", "r") == "r"
        assert kwargs.pop("maskandscale", False) is False

        # attributes set before super().__init__ don' accidentally turn into
        # dataset attribues
        self.chunks = {}
        self.threshold = inline_threshold
        self.max_chunk_size = max_chunk_size
        self.out = {}
        with fsspec.open(filename, **(storage_options or {})) as fp:
            super().__init__(
                fp, *args, mmap=False, mode="r", maskandscale=False, **kwargs
            )
        self.filename = filename  # this becomes an attribute, so must ignore on write

    def _read_var_array(self):
        header = self.fp.read(4)
        if header not in [ZERO, NC_VARIABLE]:
            raise ValueError("Unexpected header.")

        begin = 0
        dtypes = {"names": [], "formats": []}
        rec_vars = []
        count = self._unpack_int()
        for var in range(count):
            (
                name,
                dimensions,
                shape,
                attributes,
                typecode,
                size,
                dtype_,
                begin_,
                vsize,
            ) = self._read_var()
            if shape and shape[0] is None:  # record variable
                rec_vars.append(name)
                # The netCDF "record size" is calculated as the sum of
                # the vsize's of all the record variables.
                self.__dict__["_recsize"] += vsize
                if begin == 0:
                    begin = begin_
                dtypes["names"].append(name)
                dtypes["formats"].append(str(shape[1:]) + dtype_)

                # Handle padding with a virtual variable.
                if typecode in "bch":
                    actual_size = reduce(mul, (1,) + shape[1:]) * size
                    padding = -actual_size % 4
                    if padding:
                        dtypes["names"].append("_padding_%d" % var)
                        dtypes["formats"].append("(%d,)>b" % padding)

                # Data will be set later.
                data = None
            else:  # not a record variable
                # Calculate size to avoid problems with vsize (above)
                a_size = reduce(mul, shape, 1) * size
                self.chunks[name] = [begin_, a_size, dtype_, shape]
                if name in ["latitude", "longitude", "time"]:
                    pos = self.fp.tell()
                    self.fp.seek(begin_)
                    data = np.frombuffer(self.fp.read(a_size), dtype=dtype_).copy()
                    # data.shape = shape
                    self.fp.seek(pos)
                else:
                    data = np.empty(1, dtype=dtype_)

            # Add variable.
            self.variables[name] = netcdf_variable(
                data,
                typecode,
                size,
                shape,
                dimensions,
                attributes,
                maskandscale=self.maskandscale,
            )

        if rec_vars:
            # Remove padding when only one record variable.
            if len(rec_vars) == 1:
                dtypes["names"] = dtypes["names"][:1]
                dtypes["formats"] = dtypes["formats"][:1]

            pos = self.fp.tell()
            self.fp.seek(begin)
            self.chunks.setdefault("record_array", []).append(
                [begin, self._recs * self._recsize, dtypes]
            )
            self.fp.seek(pos)

    def translate(self):
        """
        Produce references dictionary

        Parameters
        ----------
        """
        import zarr

        out = self.out
        z = zarr.open(out, mode="w")
        for dim, var in self.variables.items():
            if dim in self.dimensions:
                shape = self.dimensions[dim]
            elif dim in self.chunks:
                shape = self.chunks[dim][-1]
            else:
                # defer record array
                continue
            if isinstance(shape, int):
                shape = (shape,)
            if shape is None or (len(shape) and shape[0] is None):
                # defer record array
                continue
            else:
                # simple array block
                # TODO: chance to sub-chunk
                fill = var._attributes.get("missing_value", None)
                if fill is not None and var.data.dtype.kind == "f":
                    fill = float(fill)
                if fill is not None and var.data.dtype.kind == "i":
                    fill = int(fill)
                arr = z.create_dataset(
                    name=dim,
                    shape=shape,
                    dtype=var.data.dtype,
                    fill_value=fill,
                    chunks=shape,
                    compression=None,
                )
                part = ".".join(["0"] * len(shape)) or "0"
                out[f"{dim}/{part}"] = [self.filename] + [
                    int(self.chunks[dim][0]),
                    int(self.chunks[dim][1]),
                ]
            arr.attrs.update(
                {
                    k: v.decode() if isinstance(v, bytes) else str(v)
                    for k, v in var._attributes.items()
                    if k not in ["_FillValue", "missing_value"]
                }
            )
            arr.attrs["_ARRAY_DIMENSIONS"] = list(var.dimensions)
        if "record_array" in self.chunks:
            # native chunks version (no codec, no options)
            start, size, dt = self.chunks["record_array"][0]
            dt = np.dtype(dt)
            outer_shape = size // dt.itemsize
            offset = start
            for name in dt.names:
                # the order of the names if fixed and important!
                var = self.variables[name]
                dtype = dt[name]
                base = dtype.base  # actual dtype
                shape = (outer_shape,) + dtype.shape

                # TODO: avoid this code repeat
                fill = var._attributes.get("missing_value", None)
                if fill is not None and base.kind == "f":
                    fill = float(fill)
                if fill is not None and base.kind == "i":
                    fill = int(fill)
                arr = z.create_dataset(
                    name=name,
                    shape=shape,
                    dtype=base,
                    fill_value=fill,
                    chunks=(1,) + dtype.shape,
                    compression=None,
                )
                arr.attrs.update(
                    {
                        k: v.decode() if isinstance(v, bytes) else str(v)
                        for k, v in var._attributes.items()
                        if k not in ["_FillValue", "missing_value"]
                    }
                )
                arr.attrs["_ARRAY_DIMENSIONS"] = list(var.dimensions)

                suffix = (
                    ("." + ".".join("0" for _ in dtype.shape)) if dtype.shape else ""
                )
                for i in range(outer_shape):
                    out[f"{name}/{i}{suffix}"] = [
                        self.filename,
                        int(offset + i * dt.itemsize),
                        int(dtype.itemsize),
                    ]

                offset += dtype.itemsize
        z.attrs.update(
            {
                k: v.decode() if isinstance(v, bytes) else str(v)
                for k, v in self._attributes.items()
                if k != "filename"  # special "attribute"
            }
        )

        # remove bytes
        out = {
            k: (v.decode() if isinstance(v, bytes) else v) for k, v in self.out.items()
        }

        return {"version": 1, "refs": out}


netcdf_recording_file = NetCDF3ToZarr
