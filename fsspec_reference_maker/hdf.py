import base64
from typing import Union, BinaryIO
import logging
import numpy as np
import h5py
import zarr
from zarr.meta import encode_fill_value
import numcodecs

lggr = logging.getLogger('h5-to-zarr')
_HIDDEN_ATTRS = {  # from h5netcdf.attrs
    "REFERENCE_LIST",
    "CLASS",
    "DIMENSION_LIST",
    "NAME",
    "_Netcdf4Dimid",
    "_Netcdf4Coordinates",
    "_nc3_strict",
    "_NCProperties",
}


class SingleHdf5ToZarr:
    """Translate the content of one HDF5 file into Zarr metadata.

    HDF5 groups become Zarr groups. HDF5 datasets become Zarr arrays. Zarr array
    chunks remain in the HDF5 file.

    Parameters
    ----------
    h5f : file-like
        Input HDF5 file as a binary Python file-like object (duck-typed, adhering
        to BinaryIO is optional)
    url : string
        URI of the HDF5 file.
    spec : int
        The version of output to produce (see README of this repo)
    inline_threshold : int
        Include chunks smaller than this value directly in the output. Zero or negative
        to disable
    """

    def __init__(self, h5f: BinaryIO, url: str,
                 spec=1, inline_threshold=100):
        # Open HDF5 file in read mode...
        lggr.debug(f'HDF5 file: {h5f}')
        self.input_file = h5f
        self.spec = spec
        self.inline = inline_threshold
        self._h5f = h5py.File(h5f, mode='r')

        self.store = {}
        self._zroot = zarr.group(store=self.store, overwrite=True)

        self._uri = url
        lggr.debug(f'HDF5 file URI: {self._uri}')

    def translate(self):
        """Translate content of one HDF5 file into Zarr storage format.

        This method is the main entry point to execute the workflow, and
        returns a "reference" structure to be used with zarr/fsspec-reference-maker

        No data is copied out of the HDF5 file.

        Returns
        -------
        dict
            Dictionary containing reference structure. 
        """
        lggr.debug('Translation begins')
        self._transfer_attrs(self._h5f, self._zroot)
        self._h5f.visititems(self._translator)
        if self.inline > 0:
            self._do_inline(self.inline)
        if self.spec < 1:
            return self.store
        else:
            for k, v in self.store.copy().items():
                if isinstance(v, list):
                    self.store[k][0] = "{{u}}"
                else:
                    self.store[k] = v.decode()
            return {
                "version": 1,
                "templates": {
                    "u": self._uri
                },
                "refs": self.store
            }

    def _do_inline(self, threshold):
        """Replace short chunks with the value of that chunk

        The chunk may need encoding with base64 if not ascii, so actual
        length may be larger than threshold.
        """
        for k, v in self.store.copy().items():
            if isinstance(v, list) and v[2] < threshold:
                self.input_file.seek(v[1])
                data = self.input_file.read(v[2])
                try:
                    # easiest way to test if data is ascii
                    data.decode('ascii')
                except UnicodeDecodeError:
                    data = b"base64:" + base64.b64encode(data)
                self.store[k] = data

    def _transfer_attrs(self, h5obj: Union[h5py.Dataset, h5py.Group],
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
            if n in _HIDDEN_ATTRS:
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
                    v = v.flatten()[0]
                    if isinstance(v, (np.ndarray, np.number)):
                        v = v.tolist()
                else:
                    v = v.tolist()
            elif isinstance(v, h5py._hl.base.Empty):
                v = ""
            if v == 'DIMENSION_SCALE':
                continue
            try:
                zobj.attrs[n] = v
            except TypeError:
                lggr.exception(
                    f'Caught TypeError: {n}@{h5obj.name} = {v} ({type(v)})')

    def _translator(self, name: str, h5obj: Union[h5py.Dataset, h5py.Group]):
        """Produce Zarr metadata for all groups and datasets in the HDF5 file.
        """
        refs = {}
        if isinstance(h5obj, h5py.Dataset):
            lggr.debug(f'HDF5 dataset: {h5obj.name}')
            if h5obj.id.get_create_plist().get_layout() == h5py.h5d.COMPACT:
                RuntimeError(
                    f'Compact HDF5 datasets not yet supported: <{h5obj.name} '
                    f'{h5obj.shape} {h5obj.dtype} {h5obj.nbytes} bytes>')
                return

            #
            # check for unsupported HDF encoding/filters
            #
            if h5obj.scaleoffset:
                raise RuntimeError(
                    f'{h5obj.name} uses HDF5 scaleoffset filter - not supported by reference-maker')
            if h5obj.compression in ('szip', 'lzf'):
                raise RuntimeError(
                    f'{h5obj.name} uses szip or lzf compression - not supported by reference-maker')
            if h5obj.compression == 'gzip':
                compression = numcodecs.Zlib(level=h5obj.compression_opts)
            else:
                compression = None
            
            # Add filter for shuffle
            filters = []
            if h5obj.shuffle:
                filters.append(numcodecs.Shuffle(elementsize=h5obj.dtype.itemsize))

            # Get storage info of this HDF5 dataset...
            cinfo = self._storage_info(h5obj)
            if h5py.h5ds.is_scale(h5obj.id) and not cinfo:
                return

            # Create a Zarr array equivalent to this HDF5 dataset...
            za = self._zroot.create_dataset(h5obj.name, shape=h5obj.shape,
                                            dtype=h5obj.dtype,
                                            chunks=h5obj.chunks or False,
                                            fill_value=h5obj.fillvalue,
                                            compression=compression,
                                            filters=filters,
                                            overwrite=True)
            lggr.debug(f'Created Zarr array: {za}')
            self._transfer_attrs(h5obj, za)

            adims = self._get_array_dims(h5obj)
            za.attrs['_ARRAY_DIMENSIONS'] = adims
            lggr.debug(f'_ARRAY_DIMENSIONS = {adims}')

            # Store chunk location metadata...
            if cinfo:
                for k, v in cinfo.items():
                    if h5obj.fletcher32:
                        logging.info("Discarding fletcher32 checksum")
                        v['size'] -= 4
                    self.store[za._chunk_key(k)] = [self._uri, v['offset'], v['size']]

        elif isinstance(h5obj, h5py.Group):
            lggr.debug(f'HDF5 group: {h5obj.name}')
            zgrp = self._zroot.create_group(h5obj.name)
            self._transfer_attrs(h5obj, zgrp)

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
                elif num_scales == 0:
                    # Some HDF5 files do not have dimension scales. 
                    # If this is the case, `num_scales` will be 0.
                    # In this case, we mimic netCDF4 and assign phony dimension names.
                    # See https://github.com/intake/fsspec-reference-maker/issues/41

                    dims.append(f"phony_dim_{n}")
        return dims

    def _storage_info(self, dset: h5py.Dataset) -> dict:
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
