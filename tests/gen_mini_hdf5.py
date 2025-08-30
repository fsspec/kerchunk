import numpy
import h5py
import hdf5plugin


def _dict_add(a, b):
    c = a.copy()
    c.update(b)
    return c

compressors = dict(
    zstd=hdf5plugin.Zstd(),
    shuffle_zstd=_dict_add(dict(shuffle=True), hdf5plugin.Zstd()), # Test for two filters
    blosc=hdf5plugin.Blosc(),
)

for c in compressors:
    f = h5py.File(f"hdf5_mini_{c}.h5", "w")
    f.create_dataset("data", (4,), dtype=numpy.int32, **compressors[c]
                     ).write_direct(numpy.arange(4, dtype=numpy.int32))
    f.close()

from kerchunk.hdf import SingleHdf5ToZarr
for c in compressors:
    with open(f'hdf5_mini_{c}.h5', 'rb') as f:
        SingleHdf5ToZarr(f, None).translate()
