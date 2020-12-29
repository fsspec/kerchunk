from typing import Union, BinaryIO
import logging
import numpy as np
import h5py
import zarr
from zarr.meta import encode_fill_value, json_dumps
from numcodecs import Zlib
import fsspec

lggr = logging.getLogger('h5-to-zarr')
chunks_meta_key = ".zchunkstore"


def _path_to_prefix(path):
    # assume path already normalized
    if path:
        prefix = path + '/'
    else:
        prefix = ''
    return prefix


def chunks_info(zarray, chunks_loc):
    """Store chunks location information for a Zarr array.

    Parameters
    ----------
    zarray : zarr.core.Array
        Zarr array that will use the chunk data.
    chunks_loc : dict
        File storage information for the chunks belonging to the Zarr array.
    """
    if 'source' not in chunks_loc:
        raise ValueError('Chunk source information missing')
    if any([k not in chunks_loc['source'] for k in ('uri', 'array_name')]):
        raise ValueError(
            f'{chunks_loc["source"]}: Chunk source information incomplete')

    key = _path_to_prefix(zarray.path) + chunks_meta_key
    chunks_meta = dict()
    for k, v in chunks_loc.items():
        if k != 'source':
            k = zarray._chunk_key(k)
            if any([a not in v for a in ('offset', 'size')]):
                raise ValueError(
                    f'{k}: Incomplete chunk location information')
        chunks_meta[k] = v

    # Store Zarr array chunk location metadata...
    zarray.store[key] = json_dumps(chunks_meta)


class Hdf5ToZarr:
    """Translate the content of one HDF5 file into Zarr metadata.

    HDF5 groups become Zarr groups. HDF5 datasets become Zarr arrays. Zarr array
    chunks remain in the HDF5 file.

    Parameters
    ----------
    h5f : file-like or str
        Input HDF5 file as a string or file-like Python object.
    url : str
        URI of the HDF5 file.
    xarray : bool, optional
        Produce attributes required by the `xarray <http://xarray.pydata.org>`_
        package to correctly identify dimensions (HDF5 dimension scales) of a
        Zarr array. Default is ``False``.
    """

    def __init__(self, h5f: Union[str, BinaryIO], url: str,
                 xarray: bool = False):
        # Open HDF5 file in read mode...
        lggr.debug(f'HDF5 file: {h5f}')
        lggr.debug(f'xarray: {xarray}')
        self._h5f = h5py.File(h5f, mode='r')
        self._xr = xarray

        self.store = {}
        self._zroot = zarr.group(store=self.store, overwrite=True)

        self._uri = url
        lggr.debug(f'HDF5 file URI: {self._uri}')

    def translate(self):
        """Translate content of one HDF5 file into Zarr storage format.

        No data is copied out of the HDF5 file.
        """
        import json
        lggr.debug('Translation begins')
        self.transfer_attrs(self._h5f, self._zroot)
        self._h5f.visititems(self.translator)
        ref = {}
        for key, value in self.store.items():
            if key.endswith(".zchunkstore"):
                value = json.loads(value)
                source = value.pop("source")["uri"]
                for k, v in value.items():
                    ref[k] = (source, v["offset"], v["size"])
            else:
                ref[key] = value.decode()
        return ref

    def transfer_attrs(self, h5obj: Union[h5py.Dataset, h5py.Group],
                       zobj: Union[zarr.Array, zarr.Group]):
        """Transfer attributes from an HDF5 object to its equivalent Zarr object.

        Parameters
        ----------
        h5obj : h5py.Group or h5py.Dataset
            An HDF5 group or dataset.
        zobj : zarr.hierarchy.Group or zarr.core.Array
            An equivalent Zarr group or array to the HDF5 group or dataset with
            attributes.
        """
        for n, v in h5obj.attrs.items():
            if n in ('REFERENCE_LIST', 'DIMENSION_LIST'):
                continue

            # Fix some attribute values to avoid JSON encoding exceptions...
            if isinstance(v, bytes):
                v = v.decode('utf-8')
            elif isinstance(v, (np.ndarray, np.number)):
                if v.dtype.kind == 'S':
                    v = v.astype(str)
                if n == '_FillValue':
                    v = encode_fill_value(v, v.dtype)
                elif v.size == 1:
                    v = v.flatten()[0].tolist()
                else:
                    v = v.tolist()
            if self._xr and v == 'DIMENSION_SCALE':
                continue
            try:
                zobj.attrs[n] = v
            except TypeError:
                lggr.exception(
                    f'Caught TypeError: {n}@{h5obj.name} = {v} ({type(v)})')

    def translator(self, name: str, h5obj: Union[h5py.Dataset, h5py.Group]):
        """Produce Zarr metadata for all groups and datasets in the HDF5 file.
        """
        if isinstance(h5obj, h5py.Dataset):
            lggr.debug(f'HDF5 dataset: {h5obj.name}')
            if h5obj.id.get_create_plist().get_layout() == h5py.h5d.COMPACT:
                RuntimeError(
                    f'Compact HDF5 datasets not yet supported: <{h5obj.name} '
                    f'{h5obj.shape} {h5obj.dtype} {h5obj.nbytes} bytes>')
                return

            if (h5obj.scaleoffset or h5obj.fletcher32 or h5obj.shuffle or
                    h5obj.compression in ('szip', 'lzf')):
                raise RuntimeError(
                    f'{h5obj.name} uses unsupported HDF5 filters')
            if h5obj.compression == 'gzip':
                compression = Zlib(level=h5obj.compression_opts)
            else:
                compression = None

            # Get storage info of this HDF5 dataset...
            cinfo = self.storage_info(h5obj)
            if self._xr and h5py.h5ds.is_scale(h5obj.id) and not cinfo:
                return

            # Create a Zarr array equivalent to this HDF5 dataset...
            za = self._zroot.create_dataset(h5obj.name, shape=h5obj.shape,
                                            dtype=h5obj.dtype,
                                            chunks=h5obj.chunks or False,
                                            fill_value=h5obj.fillvalue,
                                            compression=compression,
                                            overwrite=True)
            lggr.debug(f'Created Zarr array: {za}')
            self.transfer_attrs(h5obj, za)

            if self._xr:
                # Do this for xarray...
                adims = self._get_array_dims(h5obj)
                if adims:
                    za.attrs['_ARRAY_DIMENSIONS'] = adims
                    lggr.debug(f'_ARRAY_DIMENSIONS = {adims}')

            # Store chunk location metadata...
            if cinfo:
                cinfo['source'] = {'uri': self._uri,
                                   'array_name': h5obj.name}
                chunks_info(za, cinfo)

        elif isinstance(h5obj, h5py.Group):
            lggr.debug(f'HDF5 group: {h5obj.name}')
            zgrp = self._zroot.create_group(h5obj.name)
            self.transfer_attrs(h5obj, zgrp)

    def _get_array_dims(self, dset):
        """Get a list of dimension scale names attached to input HDF5 dataset.

        This is required by the xarray package to work with Zarr arrays. Only
        one dimension scale per dataset dimension is allowed. If dataset is
        dimension scale, it will be considered as the dimension to itself.

        Parameters
        ----------
        dset : h5py.Dataset
            HDF5 dataset.

        Returns
        -------
        list
            List with HDF5 path names of dimension scales attached to input
            dataset.
        """
        dims = list()
        rank = len(dset.shape)
        if rank:
            for n in range(rank):
                num_scales = len(dset.dims[n])
                if num_scales == 1:
                    dims.append(dset.dims[n][0].name[1:])
                elif h5py.h5ds.is_scale(dset.id):
                    dims.append(dset.name[1:])
                elif num_scales > 1:
                    raise RuntimeError(
                        f'{dset.name}: {len(dset.dims[n])} '
                        f'dimension scales attached to dimension #{n}')
        return dims

    def storage_info(self, dset: h5py.Dataset) -> dict:
        """Get storage information of an HDF5 dataset in the HDF5 file.

        Storage information consists of file offset and size (length) for every
        chunk of the HDF5 dataset.

        Parameters
        ----------
        dset : h5py.Dataset
            HDF5 dataset for which to collect storage information.

        Returns
        -------
        dict
            HDF5 dataset storage information. Dict keys are chunk array offsets
            as tuples. Dict values are pairs with chunk file offset and size
            integers.
        """
        # Empty (null) dataset...
        if dset.shape is None:
            return dict()

        dsid = dset.id
        if dset.chunks is None:
            # Contiguous dataset...
            if dsid.get_offset() is None:
                # No data ever written...
                return dict()
            else:
                key = (0,) * (len(dset.shape) or 1)
                return {key: {'offset': dsid.get_offset(),
                              'size': dsid.get_storage_size()}}
        else:
            # Chunked dataset...
            num_chunks = dsid.get_num_chunks()
            if num_chunks == 0:
                # No data ever written...
                return dict()

            # Go over all the dataset chunks...
            stinfo = dict()
            chunk_size = dset.chunks
            for index in range(num_chunks):
                blob = dsid.get_chunk_info(index)
                key = tuple(
                    [a // b for a, b in zip(blob.chunk_offset, chunk_size)])
                stinfo[key] = {'offset': blob.byte_offset, 'size': blob.size}
            return stinfo


def run(url, **storage_options):
    lggr.setLevel(logging.DEBUG)
    lggr_handler = logging.StreamHandler()
    lggr_handler.setFormatter(logging.Formatter(
        '%(levelname)s:%(name)s:%(funcName)s:%(message)s')
    )
    lggr.handlers.clear()
    lggr.addHandler(lggr_handler)

    with fsspec.open(url, **storage_options) as f:
        h5chunks = Hdf5ToZarr(f, url, xarray=True)
        return h5chunks.translate()


def example():
    return run(
        's3://pangeo-data-uswest2/esip/adcirc/adcirc_01d.nc',
        mode='rb', anon=False, requester_pays=True,
        default_fill_cache=False, default_cache_type='none'
    )
