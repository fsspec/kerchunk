import numpy
import h5py
import hdf5plugin
import ujson
import numcodecs
import zarr
from fsspec.implementations.reference import ReferenceFileSystem


def _dict_add(a, b):
    c = a.copy()
    c.update(b)
    return c

compressors = dict(
    zstd=hdf5plugin.Zstd(),
    shuffle_zstd=_dict_add(dict(shuffle=True), hdf5plugin.Zstd()), # Test for two filters
    blosc=hdf5plugin.Blosc(),
    shuffle_blosc=_dict_add(dict(shuffle=True), hdf5plugin.Blosc()),
    lz4=hdf5plugin.LZ4(),
)

for c in compressors:
    f = h5py.File(f"hdf5_mini_{c}.h5", "w")
    f.create_dataset("data", (3, 2), dtype=numpy.int32, **compressors[c]
                     ).write_direct(numpy.arange(6, dtype=numpy.int32).reshape((3, 2)))
    f.close()

f = h5py.File(f"hdf5_mali_chunk.h5", "w")
f.create_dataset("data", (8,), dtype=numpy.int32, chunks=(7,), **hdf5plugin.Zstd()
                 )
f['data'][0:7] = numpy.arange(7, dtype=numpy.int32)
f['data'].id.write_direct_chunk((7,),numpy.array([7]+[0]*6, dtype=numpy.int32), 1)
print(f['data'][:])
f.close()

f = h5py.File(f"hdf5_mali_chunk2.h5", "w")
f.create_dataset("data", (8,), dtype=numpy.int32, chunks=(3,), shuffle=True, **hdf5plugin.Zstd()
                 )
f['data'].id.write_direct_chunk((0,),numcodecs.Shuffle(4).encode(numpy.array([0, 1, 2], dtype=numpy.int32)), 2)
f['data'].id.write_direct_chunk((3,),numpy.array([3, 4, 5], dtype=numpy.int32), 3)
f['data'][6:] = numpy.array([6, 7,], dtype=numpy.int32)
print(f['data'][:])
f.close()

# from kerchunk.hdf import SingleHdf5ToZarr
# for c in compressors:
#     # if c=='lz4':
#     #     continue
#     with open(f'hdf5_mini_{c}.h5', 'rb') as f:
#         print(ujson.dumps(SingleHdf5ToZarr(f, None, inline_threshold=0, unsupported_inline_threshold=1).translate(), indent=4))
# with open(f'hdf5_mali_chunk.h5', 'rb') as f:
#     ref = SingleHdf5ToZarr(f, None).translate()
#     print(ujson.dumps(ref, indent=4))
# print(zarr.group(ReferenceFileSystem(ref, target=f'hdf5_mali_chunk.h5').get_mapper())['data'][:])
# with open(f'hdf5_mali_chunk2.h5', 'rb') as f:
#     ref = SingleHdf5ToZarr(f, None).translate()
#     print(ujson.dumps(ref, indent=4))
# print(zarr.group(ReferenceFileSystem(ref, target=f'hdf5_mali_chunk2.h5').get_mapper())['data'][:])
