import numpy as np
from scipy.io.netcdf import ZERO, NC_VARIABLE, netcdf_file, reduce, mul, netcdf_variable, frombuffer

import fsspec


class netcdf_recording_file(netcdf_file):

    def __init__(self, filename, *args, storage_options=None, **kwargs):
        self.chunks = {}
        with fsspec.open(filename, **(storage_options or {})) as fp:
            super().__init__(fp, *args, **kwargs)

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
                pos = self.fp.tell()
                self.fp.seek(begin_)
                self.chunks.setdefault(name, []).append((begin_, a_size, dtype_, shape))
                # data = frombuffer(self.fp.read(a_size), dtype=dtype_
                #                   ).copy()
                # data.shape = shape
                data = np.empty(1, dtype=dtype_)
                self.fp.seek(pos)

            # Add variable.
            self.variables[name] = netcdf_variable(
                data, typecode, size, shape, dimensions, attributes,
                maskandscale=self.maskandscale)

        if rec_vars:
            # Remove padding when only one record variable.
            if len(rec_vars) == 1:
                dtypes['names'] = dtypes['names'][:1]
                dtypes['formats'] = dtypes['formats'][:1]

            # Build rec array.
            if self.use_mmap:
                rec_array = self._mm_buf[begin:begin+self._recs*self._recsize].view(dtype=dtypes)
                rec_array.shape = (self._recs,)
            else:
                pos = self.fp.tell()
                self.fp.seek(begin)
                self.chunks.setdefault(var, []).append([begin, self._recs*self._recsize, dtypes])
                # rec_array = frombuffer(self.fp.read(self._recs*self._recsize),
                #                        dtype=dtypes).copy()
                # rec_array.shape = (self._recs,)
                self.fp.seek(pos)

            # for var in rec_vars:
            #    self.variables[var].__dict__['data'] = rec_array[var]
