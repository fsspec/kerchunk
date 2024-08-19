import base64
import io
import logging
from typing import Union, BinaryIO

import fsspec.core
from fsspec.implementations.reference import LazyReferenceMapper
import numpy as np
from numba.core.pythonapi import reflect

import zarr
import numcodecs

from .codecs import FillStringsCodec
from .utils import _encode_for_JSON

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover
    raise ImportError(
        "h5py is required for kerchunking HDF5/NetCDF4 files. Please install with "
        "`pip/conda install h5py`. See https://docs.h5py.org/en/latest/build.html "
        "for more details."
    )

try:
    from zarr.meta import encode_fill_value
except ModuleNotFoundError:
    # https://github.com/zarr-developers/zarr-python/issues/2021
    from zarr.v2.meta import encode_fill_value

lggr = logging.getLogger("h5-to-zarr")
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
    h5f : file-like or str
        Input HDF5 file. Can be a binary Python file-like object (duck-typed, adhering
        to BinaryIO is optional), in which case must also provide url. If a str,
        file will be opened using fsspec and storage_options.
    url : string
        URI of the HDF5 file, if passing a file-like object or h5py File/Group
    spec : int
        The version of output to produce (see README of this repo)
    inline_threshold : int
        Include chunks smaller than this value directly in the output. Zero or negative
        to disable
    storage_options: dict
        passed to fsspec if h5f is a str
    error: "warn" (default) | "pdb" | "ignore" | "raise"
    vlen_encode: ["embed", "null", "leave", "encode"]
        What to do with VLEN string variables or columns of tabular variables
        leave: pass through the 16byte garbage IDs unaffected, but requires no codec
        null: set all the strings to None or empty; required that this library is available
        at read time
        embed: include all the values in the output JSON (should not be used for large tables)
        encode: save the ID-to-value mapping in a codec, to produce the real values at read
        time; requires this library to be available. Can be efficient storage where there
        are few unique values.
    out: dict-like or None
        This allows you to supply an fsspec.implementations.reference.LazyReferenceMapper
        to write out parquet as the references get filled, or some other dictionary-like class
        to customise how references get stored
    """

    def __init__(
        self,
        h5f: "BinaryIO | str | h5py.File | h5py.Group",
        url: str = None,
        spec=1,
        inline_threshold=500,
        storage_options=None,
        error="warn",
        vlen_encode="embed",
        out=None,
    ):

        # Open HDF5 file in read mode...
        lggr.debug(f"HDF5 file: {h5f}")
        if isinstance(h5f, str):
            fs, path = fsspec.core.url_to_fs(h5f, **(storage_options or {}))
            self.input_file = fs.open(path, "rb")
            url = h5f
            self._h5f = h5py.File(self.input_file, mode="r")
        elif isinstance(h5f, io.IOBase):
            self.input_file = h5f
            self._h5f = h5py.File(self.input_file, mode="r")
        elif isinstance(h5f, (h5py.File, h5py.Group)):
            # assume h5py object (File or group/dataset)
            self._h5f = h5f
            fs, path = fsspec.core.url_to_fs(url, **(storage_options or {}))
            self.input_file = fs.open(path, "rb")
        else:
            raise ValueError("type of input `h5f` not recognised")
        self.spec = spec
        self.inline = inline_threshold
        if vlen_encode not in ["embed", "null", "leave", "encode"]:
            raise NotImplementedError
        self.vlen = vlen_encode
        self.store = out or {}
        self._zroot = zarr.group(store=self.store, overwrite=True)

        self._uri = url
        self.error = error
        lggr.debug(f"HDF5 file URI: {self._uri}")

    def translate(self, preserve_linked_dsets=False):
        """Translate content of one HDF5 file into Zarr storage format.

        This method is the main entry point to execute the workflow, and
        returns a "reference" structure to be used with zarr/kerchunk

        No data is copied out of the HDF5 file.

        Parameters
        ----------
        preserve_linked_dsets : bool (optional, default False)
            If True, translate HDF5 soft and hard links for each `h5py.Dataset`
            into the reference structure. Requires h5py version 3.11.0 or later.
            Will not translate external links or links to `h5py.Group` objects.

        Returns
        -------
        dict
            Dictionary containing reference structure.
        """
        lggr.debug("Translation begins")
        self._transfer_attrs(self._h5f, self._zroot)

        self._h5f.visititems(self._translator)

        if preserve_linked_dsets:
            if not has_visititems_links():
                raise RuntimeError(
                    "'preserve_linked_dsets' kwarg requires h5py 3.11.0 or later "
                    f"is installed, found {h5py.__version__}"
                )
            self._h5f.visititems_links(self._translator)

        if self.spec < 1:
            return self.store
        elif isinstance(self.store, LazyReferenceMapper):
            self.store.flush()
            return self.store
        else:
            store = _encode_for_JSON(self.store)
            return {"version": 1, "refs": store}

    def _unref(self, ref):
        name = h5py.h5r.get_name(ref, self._h5f.id)
        return self._h5f[name]

    def _transfer_attrs(
        self,
        h5obj: Union[h5py.Dataset, h5py.Group],
        zobj: Union[zarr.Array, zarr.Group],
    ):
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
                v = v.decode("utf-8") or " "
            elif isinstance(v, (np.ndarray, np.number, np.bool_)):
                if v.dtype.kind == "S":
                    v = v.astype(str)
                if n == "_FillValue":
                    continue  # strip it out!
                elif v.size == 1:
                    v = v.flatten()[0]
                    if isinstance(v, (np.ndarray, np.number, np.bool_)):
                        v = v.tolist()
                else:
                    v = v.tolist()
            elif isinstance(v, h5py._hl.base.Empty):
                v = ""
            if v == "DIMENSION_SCALE":
                continue
            try:
                zobj.attrs[n] = v
            except TypeError:
                lggr.debug(
                    f"TypeError transferring attr, skipping:\n {n}@{h5obj.name} = {v} ({type(v)})"
                )

    def _decode_filters(self, h5obj: Union[h5py.Dataset, h5py.Group]):
        if h5obj.scaleoffset:
            raise RuntimeError(
                f"{h5obj.name} uses HDF5 scaleoffset filter - not supported by kerchunk"
            )
        if h5obj.compression in ("szip", "lzf"):
            raise RuntimeError(
                f"{h5obj.name} uses szip or lzf compression - not supported by kerchunk"
            )
        filters = []
        if h5obj.shuffle and h5obj.dtype.kind != "O":
            # cannot use shuffle if we materialised objects
            filters.append(numcodecs.Shuffle(elementsize=h5obj.dtype.itemsize))
        for filter_id, properties in h5obj._filters.items():
            if str(filter_id) == "32001":
                blosc_compressors = (
                    "blosclz",
                    "lz4",
                    "lz4hc",
                    "snappy",
                    "zlib",
                    "zstd",
                )
                (
                    _1,
                    _2,
                    bytes_per_num,
                    total_bytes,
                    clevel,
                    shuffle,
                    compressor,
                ) = properties
                pars = dict(
                    blocksize=total_bytes,
                    clevel=clevel,
                    shuffle=shuffle,
                    cname=blosc_compressors[compressor],
                )
                filters.append(numcodecs.Blosc(**pars))
            elif str(filter_id) == "32015":
                filters.append(numcodecs.Zstd(level=properties[0]))
            elif str(filter_id) == "gzip":
                filters.append(numcodecs.Zlib(level=properties))
            elif str(filter_id) == "32004":
                raise RuntimeError(
                    f"{h5obj.name} uses lz4 compression - not supported by kerchunk"
                )
            elif str(filter_id) == "32008":
                raise RuntimeError(
                    f"{h5obj.name} uses bitshuffle compression - not supported by kerchunk"
                )
            elif str(filter_id) == "shuffle":
                # already handled before this loop
                pass
            else:
                raise RuntimeError(
                    f"{h5obj.name} uses filter id {filter_id} with properties {properties},"
                    f" not supported by kerchunk."
                )
        return filters

    def _translator(
        self,
        name: str,
        h5obj: Union[
            h5py.Dataset, h5py.Group, h5py.SoftLink, h5py.HardLink, h5py.ExternalLink
        ],
    ):
        """Produce Zarr metadata for all groups and datasets in the HDF5 file."""
        try:  # method must not raise exception
            kwargs = {}

            if isinstance(h5obj, (h5py.SoftLink, h5py.HardLink)):
                h5obj = self._h5f[name]
                if isinstance(h5obj, h5py.Group):
                    # continues iteration of visititems_links
                    lggr.debug(
                        f"Skipping translation of HDF5 linked group: '{h5obj.name}'"
                    )
                    return None

            if isinstance(h5obj, h5py.Dataset):
                lggr.debug(f"HDF5 dataset: {h5obj.name}")
                lggr.debug(f"HDF5 compression: {h5obj.compression}")
                if h5obj.id.get_create_plist().get_layout() == h5py.h5d.COMPACT:
                    # Only do if h5obj.nbytes < self.inline??
                    kwargs["data"] = h5obj[:]
                    filters = []
                else:
                    filters = self._decode_filters(h5obj)
                dt = None
                # Get storage info of this HDF5 dataset...
                cinfo = self._storage_info(h5obj)

                if "data" in kwargs:
                    fill = None
                else:
                    # encodings
                    if h5obj.dtype.kind in "US":
                        fill = h5obj.fillvalue or " "  # cannot be None
                    elif h5obj.dtype.kind == "O":
                        if self.vlen == "embed":
                            if np.isscalar(h5obj):
                                out = str(h5obj)
                            elif h5obj.ndim == 0:
                                out = np.array(h5obj).tolist().decode()
                            else:
                                out = h5obj[:]
                                out2 = out.ravel()
                                for i, val in enumerate(out2):
                                    if isinstance(val, bytes):
                                        out2[i] = val.decode()
                                    elif isinstance(val, str):
                                        out2[i] = val
                                    elif isinstance(val, h5py.h5r.Reference):
                                        # TODO: recursively recreate references
                                        out2[i] = None
                                    else:
                                        out2[i] = [
                                            v.decode() if isinstance(v, bytes) else v
                                            for v in val
                                        ]
                            kwargs["data"] = out
                            kwargs["object_codec"] = numcodecs.JSON()
                            fill = None
                        elif self.vlen == "null":
                            dt = "O"
                            kwargs["object_codec"] = FillStringsCodec(dtype="S16")
                            fill = " "
                        elif self.vlen == "leave":
                            dt = "S16"
                            fill = " "
                        elif self.vlen == "encode":
                            assert len(cinfo) == 1
                            v = list(cinfo.values())[0]
                            data = _read_block(self.input_file, v["offset"], v["size"])
                            indexes = np.frombuffer(data, dtype="S16")
                            labels = h5obj[:]
                            mapping = {
                                index.decode(): label.decode()
                                for index, label in zip(indexes, labels)
                            }
                            kwargs["object_codec"] = FillStringsCodec(
                                dtype="S16", id_map=mapping
                            )
                            fill = " "
                        else:
                            raise NotImplementedError
                    elif _is_netcdf_datetime(h5obj) or _is_netcdf_variable(h5obj):
                        fill = None
                    else:
                        fill = h5obj.fillvalue
                    if h5obj.dtype.kind == "V":
                        fill = None
                        if self.vlen == "encode":
                            assert len(cinfo) == 1
                            v = list(cinfo.values())[0]
                            dt = [
                                (
                                    v,
                                    (
                                        "S16"
                                        if h5obj.dtype[v].kind == "O"
                                        else str(h5obj.dtype[v])
                                    ),
                                )
                                for v in h5obj.dtype.names
                            ]
                            data = _read_block(self.input_file, v["offset"], v["size"])
                            labels = h5obj[:]
                            arr = np.frombuffer(data, dtype=dt)
                            mapping = {}
                            for field in labels.dtype.names:
                                if labels[field].dtype == "O":
                                    mapping.update(
                                        {
                                            index.decode(): label.decode()
                                            for index, label in zip(
                                                arr[field], labels[field]
                                            )
                                        }
                                    )
                            kwargs["object_codec"] = FillStringsCodec(
                                dtype=str(dt), id_map=mapping
                            )
                            dt = [
                                (
                                    v,
                                    (
                                        "O"
                                        if h5obj.dtype[v].kind == "O"
                                        else str(h5obj.dtype[v])
                                    ),
                                )
                                for v in h5obj.dtype.names
                            ]
                        elif self.vlen == "null":
                            dt = [
                                (
                                    v,
                                    (
                                        "S16"
                                        if h5obj.dtype[v].kind == "O"
                                        else str(h5obj.dtype[v])
                                    ),
                                )
                                for v in h5obj.dtype.names
                            ]
                            kwargs["object_codec"] = FillStringsCodec(dtype=str(dt))
                            dt = [
                                (
                                    v,
                                    (
                                        "O"
                                        if h5obj.dtype[v].kind == "O"
                                        else str(h5obj.dtype[v])
                                    ),
                                )
                                for v in h5obj.dtype.names
                            ]
                        elif self.vlen == "leave":
                            dt = [
                                (
                                    v,
                                    (
                                        "S16"
                                        if h5obj.dtype[v].kind == "O"
                                        else h5obj.dtype[v]
                                    ),
                                )
                                for v in h5obj.dtype.names
                            ]
                        elif self.vlen == "embed":
                            # embed fails due to https://github.com/zarr-developers/numcodecs/issues/333
                            data = h5obj[:].tolist()
                            data2 = []
                            for d in data:
                                data2.append(
                                    [
                                        (
                                            _.decode(errors="ignore")
                                            if isinstance(_, bytes)
                                            else _
                                        )
                                        for _ in d
                                    ]
                                )
                            dt = "O"
                            kwargs["data"] = data2
                            kwargs["object_codec"] = numcodecs.JSON()
                            fill = None
                        else:
                            raise NotImplementedError

                    if h5py.h5ds.is_scale(h5obj.id) and not cinfo:
                        return
                    if h5obj.attrs.get("_FillValue") is not None:
                        fill = encode_fill_value(
                            h5obj.attrs.get("_FillValue"), dt or h5obj.dtype
                        )

                # Create a Zarr array equivalent to this HDF5 dataset...
                za = self._zroot.require_dataset(
                    h5obj.name,
                    shape=h5obj.shape,
                    dtype=dt or h5obj.dtype,
                    chunks=h5obj.chunks or False,
                    fill_value=fill,
                    compression=None,
                    filters=filters,
                    overwrite=True,
                    **kwargs,
                )
                lggr.debug(f"Created Zarr array: {za}")
                self._transfer_attrs(h5obj, za)
                adims = self._get_array_dims(h5obj)
                za.attrs["_ARRAY_DIMENSIONS"] = adims
                lggr.debug(f"_ARRAY_DIMENSIONS = {adims}")

                if "data" in kwargs:
                    return  # embedded bytes, no chunks to copy

                # Store chunk location metadata...
                if cinfo:
                    for k, v in cinfo.items():
                        if h5obj.fletcher32:
                            logging.info("Discarding fletcher32 checksum")
                            v["size"] -= 4
                        if (
                            self.inline
                            and isinstance(v, dict)
                            and v["size"] < self.inline
                        ):
                            self.input_file.seek(v["offset"])
                            data = self.input_file.read(v["size"])
                            try:
                                # easiest way to test if data is ascii
                                data.decode("ascii")
                            except UnicodeDecodeError:
                                data = b"base64:" + base64.b64encode(data)
                            self.store[za._chunk_key(k)] = data
                        else:
                            self.store[za._chunk_key(k)] = [
                                self._uri,
                                v["offset"],
                                v["size"],
                            ]

            elif isinstance(h5obj, h5py.Group):
                lggr.debug(f"HDF5 group: {h5obj.name}")
                zgrp = self._zroot.require_group(h5obj.name)
                self._transfer_attrs(h5obj, zgrp)
        except Exception as e:
            import traceback

            msg = "\n".join(
                [
                    "The following excepion was caught and quashed while traversing HDF5",
                    str(e),
                    traceback.format_exc(limit=5),
                ]
            )
            if self.error == "ignore":
                return
            elif self.error == "pdb":
                print(msg)
                import pdb

                pdb.post_mortem()
            elif self.error == "raise":
                raise
            else:
                # "warn" or anything else, the default
                import warnings

                warnings.warn(msg)
            del e  # garbage collect

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
                        f"{dset.name}: {len(dset.dims[n])} "
                        f"dimension scales attached to dimension #{n}"
                    )
                elif num_scales == 0:
                    # Some HDF5 files do not have dimension scales.
                    # If this is the case, `num_scales` will be 0.
                    # In this case, we mimic netCDF4 and assign phony dimension names.
                    # See https://github.com/fsspec/kerchunk/issues/41

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
                return {
                    key: {"offset": dsid.get_offset(), "size": dsid.get_storage_size()}
                }
        else:
            # Chunked dataset...
            num_chunks = dsid.get_num_chunks()
            if num_chunks == 0:
                # No data ever written...
                return dict()

            # Go over all the dataset chunks...
            stinfo = dict()
            chunk_size = dset.chunks

            def get_key(blob):
                return tuple([a // b for a, b in zip(blob.chunk_offset, chunk_size)])

            def store_chunk_info(blob):
                stinfo[get_key(blob)] = {"offset": blob.byte_offset, "size": blob.size}

            has_chunk_iter = callable(getattr(dsid, "chunk_iter", None))

            if has_chunk_iter:
                dsid.chunk_iter(store_chunk_info)
            else:
                for index in range(num_chunks):
                    store_chunk_info(dsid.get_chunk_info(index))

            return stinfo


def _simple_type(x):
    if isinstance(x, bytes):
        return x.decode()
    if isinstance(x, np.number):
        if x.dtype.kind == "i":
            return int(x)
        return float(x)
    return x


def _read_block(open_file, offset, size):
    place = open_file.tell()
    open_file.seek(offset)
    data = open_file.read(size)
    open_file.seek(place)
    return data


def _is_netcdf_datetime(dataset: h5py.Dataset):
    units = dataset.attrs.get("units")
    if isinstance(units, bytes):
        units = units.decode("utf-8")
    # This is the same heuristic used by xarray
    # https://github.com/pydata/xarray/blob/f8bae5974ee2c3f67346298da12621af4cae8cf8/xarray/coding/times.py#L670
    return units and "since" in units


def _is_netcdf_variable(dataset: h5py.Dataset):
    return any("_Netcdf4" in _ for _ in dataset.attrs)


def has_visititems_links():
    return hasattr(h5py.Group, "visititems_links")


decoders = {}


def reg(name):
    def f(func):
        decoders[name] = func
        return func

    return f


class HDF4ToZarr:
    def __init__(
        self,
        path,
        storage_options=None,
        inline_threshold=100,
        out=None,
        remote_protocol=None,
        remote_options=None,
    ):
        self.path = path
        self.st = storage_options
        self.thresh = inline_threshold
        self.out = out or {}
        self.remote_protocol = remote_protocol
        self.remote_options = remote_options

    def read_int(self, n):
        return int.from_bytes(self.f.read(n), "big")

    def read_ddh(self):
        return {"ndd": self.read_int(2), "next": self.read_int(4)}

    def read_dd(self):
        loc = self.f.tell()
        i = int.from_bytes(self.f.read(2), "big")
        if i & 0x4000:
            extended = True
            i = i - 0x4000
        else:
            extended = False
        tag = tags.get(i, i)
        no_data = tag not in {"NULL"}
        ref = (tag, int.from_bytes(self.f.read(2), "big"))
        info = {
            "offset": int.from_bytes(self.f.read(4), "big") * no_data,
            "length": int.from_bytes(self.f.read(4), "big") * no_data,
            "extended": extended,
            "loc": loc,
        }
        return ref, info

    def decode(self, tag, info):
        self.f.seek(info["offset"])
        ident = lambda _, __: info
        return decoders.get(tag, ident)(self, info)

    def translate(self):
        import zarr

        self.f = fsspec.open(self.path, **(self.st or {})).open()
        fs = fsspec.filesystem(
            "reference",
            fo=self.out,
            remote_protocol=self.remote_protocol,
            remote_options=self.remote_options,
        )
        g = zarr.open_group("reference://", storage_options=dict(fs=fs))
        assert self.f.read(4) == b"\x0e\x03\x13\x01"
        self.tags = {}
        while True:
            ddh = self.read_ddh()

            for _ in range(ddh["ndd"]):
                ident, info = self.read_dd()
                self.tags[ident] = info
            if ddh["next"] == 0:
                # "finished" sentry
                break
            # or continue
            self.f.seek(ddh["next"])

        for tag, ref in self.tags:
            self._dec(tag, ref)
        return self.tags

    def _dec(self, tag, ref):
        info = self.tags[(tag, ref)]
        if not set(info) - {"length", "offset", "extended", "loc"}:
            self.f.seek(info["offset"])
            if info["extended"]:
                info["data"] = self._dec_extended()
            else:
                info.update(self.decode(tag, info))
        return info

    def _dec_extended(self):
        ext_type = spec[self.read_int(2)]
        if ext_type == "CHUNKED":
            return self._dec_chunked()
        elif ext_type == "LINKED":
            return self._dec_linked_header()
        elif ext_type == "COMP":
            return self._dec_comp()

    def _dec_linked_header(self):
        # get the bytes of a linked set - these will always be inlined
        length = self.read_int(4)
        blk_len = self.read_int(4)
        num_blk = self.read_int(4)
        next_ref = self.read_int(2)
        out = []
        while next_ref:
            next_ref, data = self._dec_linked_block(self.tags[("LINKED", next_ref)])
            out.extend([d for d in data if d])
        bits = []
        for ref in out:
            info = self.tags[("LINKED", ref)]
            self.f.seek(info["offset"])
            bits.append(self.f.read(info["length"]))
        return b"".join(bits)

    def _dec_linked_block(self, block):
        self.f.seek(block["offset"])
        next_ref = self.read_int(2)
        refs = [self.read_int(2) for _ in range((block["length"] // 2) - 1)]
        return next_ref, refs

    def _dec_chunked(self):
        # we want to turn the chunks table into references
        tag_head_len = self.read_int(4)
        version = self.f.read(1)[0]
        flag = self.read_int(4)
        elem_tot_len = self.read_int(4)
        chunk_size = self.read_int(4)
        nt_size = self.read_int(4)
        chk_tbl_tag = tags[self.read_int(2)]  # should be VH
        chk_tbl_ref = self.read_int(2)
        sp_tag = tags[self.read_int(2)]
        sp_ref = self.read_int(2)
        ndims = self.read_int(4)
        dims = [
            {
                "flag": self.read_int(4),
                "dim_length": self.read_int(4),
                "chunk_length": self.read_int(4),
            }
            for _ in range(ndims)
        ]
        fill_value = self.f.read(
            self.read_int(4)
        )  # to be interpreted as a number later
        header = self._dec(chk_tbl_tag, chk_tbl_ref)
        data = self._dec("VS", chk_tbl_ref)["data"]  # corresponding table
        # header gives the field pattern for the rows of data, one per chunk
        dt = [(f"ind{i}", ">u4") for i in range(len(dims))] + [
            ("tag", ">u2"),
            ("ref", ">u2"),
        ]
        rows = np.frombuffer(data, dtype=dt, count=header["nvert"])
        refs = []
        for *ind, tag, ref in rows:
            # maybe ind needs reversing since everything is FORTRAN
            chunk_tag = self.tags[(tags[tag], ref)]
            if chunk_tag["extended"]:
                self.f.seek(chunk_tag["offset"])
                # these are always COMP?
                ctype, offset, length = self._dec_extended()
                refs.append([".".join(str(_) for _ in ind), offset, length, ctype])
            else:
                refs.append(
                    [
                        ".".join(str(_) for _ in ind),
                        chunk_tag["offset"],
                        chunk_tag["length"],
                    ]
                )
        # ref["tag"] should always be 61 -> CHUNK
        return refs

    def _dec_comp(self):
        version = self.read_int(2)  # always 0
        len_uncomp = self.read_int(4)
        data_ref = self.read_int(2)
        model = self.read_int(2)  # always 0
        ctype = comp[self.read_int(2)]
        tag = self.tags[("COMPRESSED", data_ref)]
        offset = tag["offset"]
        length = tag["length"]
        return ctype, offset, length


@reg("VERSION")
def _dec_version(self, info):
    return {
        "major": self.read_int(4),
        "minor": self.read_int(4),
        "release": self.read_int(4),
        "string:": _null_str(self.f.read(info["length"] - 10).decode()),
    }


@reg("VH")
def _dec_vh(self, info):
    # virtual group ("table") header
    interface = self.read_int(2)
    nvert = self.read_int(4)
    ivsize = self.read_int(2)
    nfields = self.read_int(2)
    types = [self.read_int(2) for _ in range(nfields)]
    isize = [self.read_int(2) for _ in range(nfields)]
    offset = [self.read_int(2) for _ in range(nfields)]
    order = [self.read_int(2) for _ in range(nfields)]
    names = [self.f.read(self.read_int(2)).decode() for _ in range(nfields)]
    namelen = self.read_int(2)
    name = self.f.read(namelen).decode()
    classlen = self.read_int(2)
    cls = self.f.read(classlen).decode()
    ref = (self.read_int(2), self.read_int(2))
    return _pl(locals())


def _null_str(s):
    return s.split("\00", 1)[0]


def _pl(l):
    return {k: v for k, v in l.items() if k not in {"info", "f", "self"}}


# hdf/src/htags.h
tags = {
    1: "NULL",
    20: "LINKED",
    30: "VERSION",
    40: "COMPRESSED",
    50: "VLINKED",
    51: "VLINKED_DATA",
    60: "CHUNKED",
    61: "CHUNK",
    100: "FID",
    101: "FD",
    102: "TID",
    103: "TD",
    104: "DIL",
    105: "DIA",
    106: "NT",
    107: "MT",
    108: "FREE",
    200: "ID8",
    201: "IP8",
    202: "RI8",
    203: "CI8",
    204: "II8",
    300: "ID",
    301: "LUT",
    302: "RI",
    303: "CI",
    304: "NRI",
    306: "RIG",
    307: "LD",
    308: "MD",
    309: "MA",
    310: "CCN",
    311: "CFM",
    312: "AR",
    400: "DRAW",
    401: "RUN",
    500: "XYP",
    501: "MTO",
    602: "T14",
    603: "T105",
    700: "SDG",
    701: "SDD",
    702: "SD",
    703: "SDS",
    704: "SDL",
    705: "SDU",
    706: "SDF",
    707: "SDM",
    708: "SDC",
    709: "SDT",
    710: "SDLNK",
    720: "NDG",
    731: "CAL",
    732: "FV",
    799: "BREQ",
    781: "SDRAG",
    780: "EREQ",
    1965: "VG",
    1962: "VH",
    1963: "VS",
    11: "RLE",
    12: "IMCOMP",
    13: "JPEG",
    14: "GREYJPEG",
    15: "JPEG5",
    16: "GREYJPEG5",
}
spec = {
    1: "LINKED",
    2: "EXT",
    3: "COMP",
    4: "VLINKED",
    5: "CHUNKED",
    6: "BUFFERED",
    7: "COMPRAS",
}

# hdf4/hdf/src/hntdefs.h
dtypes = {
    5: "f4",
    6: "f8",
    20: "i1",  # = CHAR, 3?
    21: "u1",  # = UCHAR, 4?
    22: "i2",
    23: "u2",
    24: "i4",
    25: "u4",
    26: "i8",
    27: "u8",
}

# hdf4/hdf/src/hcomp.h
comp = {
    0: "NONE",
    1: "RLE",
    2: "NBIT",
    3: "SKPHUFF",
    4: "DEFLATE",
    5: "SZIP",
    7: "JPEG",
}
