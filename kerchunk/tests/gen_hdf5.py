import numpy
import h5py
import hdf5plugin

compressors = dict(
    zstd=hdf5plugin.Zstd(),
    bitshuffle=hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
    lz4=hdf5plugin.LZ4(nbytes=0),
    blosc_lz4_bitshuffle=hdf5plugin.Blosc(
        cname="blosclz", clevel=9, shuffle=hdf5plugin.Blosc.BITSHUFFLE
    ),
)

for c in compressors:
    f = h5py.File(f"hdf5_compression_{c}.h5", "w")
    f.create_dataset("data", data=numpy.arange(100), **compressors[c])
    f.close()
