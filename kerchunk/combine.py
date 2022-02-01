import base64
import collections
import logging
import re

import fsspec
import numpy as np
import ujson
import zarr
logger = logging.getLogger("kerchunk.combine")


class MultiZarrToZarr:

    def __init__(self, path, coo_map, concat_dims=None, coo_dtypes=None,
                 target_options=None, remote_protocol=None, remote_options=None,
                 cfcombine=False):
        """

        Selectors ("how to get coordinate values from a dataset") can be:

        - a constant value (usually str for a var name, number for a coordinate)
        - a compiled regex ``re.Pattern``, which will be applied to the filename. Should
          return exactly one value
        - a string beginning "attr:" which will fetch this attribute from the
          zarr dataset of each path
        - a string beginning "vattr:{var}:" as above, but the attribute is taken from
          the array named var
        - "VARNAME" special value where a dataset contains multiple variables, just
          use the variable names as given
        - "INDEX" special value for the index of how far through the list of inputs
          we are so far
        - a string beginning "data:{var}" which will get the appropriate zarr array from
          each input dataset.
        - a list with the values that are known beforehand
        - a function with signature (index, fs, var, fn) -> value, where index is an
          int counter, fs is the file system made for the current input, var is the
          variable we are probing (may be "var") and fn is the filename or None if
          dicts were used as input

        :param path: str, list(str) or list(dict)
            Local paths, each containing a references JSON; or a list of references dicts
        :param concat_dims: str or list(str)
            Names of the dimensions to expand with
        :param coo_map: dict(str, selector)
            The special key "var" means the variable name int he output, which will be
            "VARNAME" by default (i.e., variable names are the same as in the input
            datasets)
        :param coo_dtypes: map(str, str|np.dtype)
            Coerce the final type of coordinate arrays (otherwise use numpy default)
        :param target_options: dict
            Storage options for opening ``path``
        :param remote_protocol: str
            The protocol of the original data
        :param remote_options: dict
        :param cfcombine: bool
        """
        self._fss = None
        self._paths = None
        self.ds = None
        self.path = path
        if concat_dims is None:
            self.concat_dims = list(coo_map)
        elif isinstance(concat_dims, str):
            self.concat_dims = [concat_dims]
        else:
            self.concat_dims = concat_dims
        logger.debug("Concat dims: %s", self.concat_dims)
        self.coo_map = coo_map
        if "var" not in coo_map:
            self.coo_map["var"] = "VARNAME"
        self.coo_dtypes = coo_dtypes or {}
        self.cf = cfcombine
        self.target_options = target_options or {}
        self.remote_protocol = remote_protocol
        self.remote_options = remote_options or {}
        self.out = {}

    @property
    def fss(self):
        """filesystem instances being analysed"""
        import collections.abc
        if self._fss is None:
            logger.debug("setup filesystems")
            if isinstance(self.path[0], collections.abc.Mapping):
                fo_list = self.path
                self._paths = [None] * len(fo_list)
            else:
                self._paths = []
                fo_list = []
                for of in fsspec.open_files(self.path, **self.storage_options):
                    fo_list = of
                    self._paths.append(of.full_name)

            self._fss = [
                fsspec.filesystem(
                    "reference", fo=fo,
                    remote_protocol=self.remote_protocol,
                    remote_options=self.remote_options
                ) for fo in fo_list
            ]
        return self._fss

    def _get_value(self, index, fs, var, arrname, fn=None):
        """derive coordinate values for given piece"""
        selector = self.coo_map[var]
        if isinstance(selector, collections.abc.Callable):
            o = selector(index, fs, var, fn)
        elif isinstance(selector, list):
            o = selector[index]
        elif isinstance(selector, re.Pattern):
            o = selector.match(fn).groups[0]  # may raise
        elif not isinstance(selector, str):
            # constant, should be int or float
            o = selector
        elif selector == "VARMANE":
            o = arrname or var
        elif selector == "INDEX":
            o = index
        else:
            z = zarr.open_group(fs.get_mapper(""))
            if selector.startswith("attr:"):
                o = z.attrs[selector.split(":", 1)[1]]
            elif selector.startswith("vattr:"):
                _, var, item = selector.split(":", 3)
                o = z[var].attrs[item]
            elif selector.startswith("data:"):
                o = z[selector.split(":", 1)[1]][...]
            else:
                o = selector  # must be a constant
        logger.debug("Decode: %s -> %s", (selector, index, var, arrname), o)
        return selector

    def first_pass(self):
        """Accumulate the set of concat coords values across all inputs"""
        coos = {c: set() for c in self.coo_map}
        chunks = {}
        for i, fs in enumerate(self.fss):
            logger.debug("First pass: %s", i)
            for var in self.concat_dims:
                value = self._get_value(i, fs, var, arrname=None, fn=self._paths[i])
                if isinstance(value, np.ndarray):
                    coos[var].update(value.ravel())
                    if var not in chunks:
                        chunks[var] = value.size
                elif isinstance(value, collections.abc.Iterable):
                    coos[var].update(value)
                    if var not in chunks:
                        chunks[var] = len(value)
                else:
                    coos[var].add(value)
                    if var not in chunks:
                        chunks[var] = 1
        coos = {c: np.atleast_1d(np.array(list(v), dtype=self.coo_dtypes.get(c, None)))
                for c, v in coos.items()}

        for k, arr in coos.copy().items():
            if arr.ndim == 1:
                arr.sort()
                continue
            for i in range(arr.ndim):
                slices = [slice(None, None)] * arr.ndim
                slices[i] = 0
                if (arr[slices] == all).all():
                    arr = arr[slices]
                if arr.ndim == 1:
                    # we reduced coord dimension to 1
                    arr.sort()
                    coos[k] = arr
                    continue
        self.chunks = chunks
        self.coos = coos
        logger.debug("Created coordinates")
        return coos

    def store_coords(self, z=None, coo_attrs=None):
        """
        :param z: zarr group
            Probably the first dataset, with original attrs
        :param coo_attrs: dict
            Extra attributes to attach to coordinates, each value of the dict should
            also be a dict with string keys and json-serializable values.
        """
        group = zarr.open(self.out)
        for k, v in self.coos.items():
            if k == "var":
                # The names of the variables to write in the second pass, not a coordinate
                continue
            # data may need preparation, e.g., cftime - use xarray Variable here?
            # override compression?
            arr = group.create_dataset(name=k, data=v, overwrite=True)
            if z is not None and k in z:
                arr.attrs.update(z[k].attrs)
            if coo_attrs:
                arr.attrs.update(coo_attrs.get(k, {}))
            arr.attrs["_ARRAY_DIMENSIONS"] = [k]

    def second_pass(self):
        for i, fs in enumerate(self.fss):
            m = fs.get_mapper("")

            for v in fs.ls("", detail=False):
                if v.startswith(".z"):
                    continue
                logger.debug("Second pass: %s, %s", i, v)
                import pdb
                pdb.set_trace()

                cvalues = {c: self._get_value(i, fs, c, arrname=v, fn=self._paths[i])
                           for c in self.coo_map}
                var = cvalues.pop("var")
                var = var[0] if isinstance(var, list) else var
                zarray = ujson.loads(m[f"{v}/.zarray"])
                zattrs = ujson.loads(m[f"{v}/.zattrs"])
                coords = zattrs["_ARRAY_DIMENSIONS"]
                coord_order = self.concat_dims + [c for c in coords if c not in self.concat_dims]
                for fn in fs.ls(var, detail=False):
                    if ".z" in fn:
                        continue
                    key_parts = fn.split("/", 1)[1].split(".")
                    key = f"{var}/"
                    for c in coord_order:
                        if c in self.coos:
                            i = np.searchsorted(self.coos[c], cvalues[c])
                            key += str(i // self.chunks[c])
                        else:
                            key += key_parts.pop(0)
                        key += "."
                    key = key.rstrip(".")
                    self.out[key] = fs.references[fn]

                if f"{var}/.zarray" not in self.out:
                    for k in reversed(self.concat_dims):
                        shape = zarray["shape"]
                        chunks = zarray["chunks"]
                        coords.insert(0, k)
                        shape.insert(0, self.coos[k].size)
                        chunks.insert(0, self.chunks[k])
                    self.out[f"{var}/.zarray"] = ujson.dumps(zarray)
                    self.out[f"{var}/.zattrs"] = ujson.dumps(zattrs)

    def consolidate(self, inline_threshold=500):
        """Turn raw references into output"""
        out = {}
        for k, v in self.out.items():
            if isinstance(v, list) and v[2] < inline_threshold:
                v = self.fs.cat_file(v[0], start=v[1], end=v[1] + v[2])
            if isinstance(v, bytes):
                try:
                    # easiest way to test if data is ascii
                    out[k] = v.decode('ascii')
                except UnicodeDecodeError:
                    out[k] = (b"base64:" + base64.b64encode(v)).decode()
            else:
                out[k] = v
        return {"version": 1, "refs": out}
