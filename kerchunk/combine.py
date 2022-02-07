import base64
import collections.abc
import logging
import re

import fsspec
import fsspec.utils
import numpy as np
import numcodecs
import ujson
import zarr
logger = logging.getLogger("kerchunk.combine")
# fsspec.utils.setup_logging(logger)


def drop(fields):
    """Generate example preprocessor removing given fields"""
    def preproc(refs):
        for k in list(refs):
            if k.startswith(fields):
                refs.pop(k)
        return refs
    return preproc


class MultiZarrToZarr:
    """
    :param path: str, list(str) or list(dict)
        Local paths, each containing a references JSON; or a list of references dicts
    :param concat_dims: str or list(str)
        Names of the dimensions to expand with
    :param coo_map: dict(str, selector)
        The special key "var" means the variable name in the output, which will be
        "VARNAME" by default (i.e., variable names are the same as in the input
        datasets). The default for any other coordinate is data:varname, i.e., look
        for an array with that name. 
        
        Selectors ("how to get coordinate values from a dataset") can be:
            - a constant value (usually str for a var name, number for a coordinate)
            - a compiled regex ``re.Pattern``, which will be applied to the filename. Should return exactly one value
            - a string beginning "attr:" which will fetch this attribute from the zarr dataset of each path
            - a string beginning "vattr:{var}:" as above, but the attribute is taken from the array named var
            - "VARNAME" special value where a dataset contains multiple variables, just use the variable names as given
            - "INDEX" special value for the index of how far through the list of inputs we are so far
            - a string beginning "data:{var}" which will get the appropriate zarr array from each input dataset.
            - a list with the values that are known beforehand
            - a function with signature (index, fs, var, fn) -> value, where index is an int counter, fs is the file system made for the current input, var is the variable we are probing may be "var") and fn is the filename or None if dicts were used as input

    :param coo_dtypes: map(str, str|np.dtype)
        Coerce the final type of coordinate arrays (otherwise use numpy default)
    :param target_options: dict
        Storage options for opening ``path``
    :param remote_protocol: str
        The protocol of the original data
    :param remote_options: dict
    :param inline_threshold: int
        Size below which binary blocks are included directly in the output
    :param preprocess: callable
        Acts on the references dict of al inputs before processing. See ``drop()``
        for an example.
    """

    def __init__(self, path, coo_map=None, concat_dims=None, coo_dtypes=None,
                 target_options=None, remote_protocol=None, remote_options=None,
                 inline_threshold=500, preprocess=None):
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
        self.coo_map = coo_map or {}
        self.coo_map.update({
            c: "VARNAME" if c == "var" else f"data:{c}"
            for c in concat_dims if c not in self.coo_map
        })
        logger.debug("Concat dims: %s", self.concat_dims)
        logger.debug("Coord map: %s", self.coo_map)
        self.coo_dtypes = coo_dtypes or {}
        self.target_options = target_options or {}
        self.remote_protocol = remote_protocol
        self.remote_options = remote_options or {}
        self.inline = inline_threshold
        self.preprocess = preprocess
        self.out = {}

    @property
    def fss(self):
        """filesystem instances being analysed, one per input dataset"""
        import collections.abc
        if self._fss is None:
            logger.debug("setup filesystems")
            if isinstance(self.path[0], collections.abc.Mapping):
                fo_list = self.path
                self._paths = [None] * len(fo_list)
            else:
                self._paths = []
                fo_list = []
                for of in fsspec.open_files(self.path, **self.target_options):
                    fo_list.append(of.open())
                    self._paths.append(of.full_name)

            self._fss = [
                fsspec.filesystem(
                    "reference", fo=fo,
                    remote_protocol=self.remote_protocol,
                    remote_options=self.remote_options
                ) for fo in fo_list
            ]
        return self._fss

    def _get_value(self, index, z, var, fn=None):
        """derive coordinate value(s) for given input dataset

        How to map from input to

        index: int
            Current place in the list of inputs
        z: zarr group
            Open for the current input
        var: str
            name of value to extract.
        fn: str
            filename
        """
        selector = self.coo_map[var]
        if isinstance(selector, collections.abc.Callable):
            o = selector(index, z, var, fn)
        elif isinstance(selector, list):
            o = selector[index]
        elif isinstance(selector, re.Pattern):
            o = selector.match(fn).groups[0]  # may raise
        elif not isinstance(selector, str):
            # constant, should be int or float
            o = selector
        elif selector == "VARNAME":
            # used for merging variable names across datasets
            o = [_ for _ in z if _ not in self.concat_dims]
            if len(o) > 1:
                raise ValueError("Multiple varnames found in dataset, please "
                                 "provide a more specific selector")
            o = o[0]
        elif selector == "INDEX":
            o = index
        elif selector.startswith("attr:"):
            o = z.attrs[selector.split(":", 1)[1]]
        elif selector.startswith("vattr:"):
            _, var, item = selector.split(":", 3)
            o = z[var].attrs[item]
        elif selector.startswith("data:"):
            o = z[selector.split(":", 1)[1]][...]
        else:
            o = selector  # must be a non-number constant - error?
        logger.debug("Decode: %s -> %s", (selector, index, var, fn), o)
        return o

    def first_pass(self):
        """Accumulate the set of concat coords values across all inputs"""

        coos = {c: set() for c in self.coo_map}
        for i, fs in enumerate(self.fss):
            if self.preprocess:
                self.preprocess(fs.references)
                # reset this to force references to update
                fs.dircache = None
                fs._dircache_from_items()

            logger.debug("First pass: %s", i)
            z = zarr.open_group(fs.get_mapper(""))
            for var in self.concat_dims:
                value = self._get_value(i, z, var, fn=self._paths[i])
                if isinstance(value, np.ndarray):
                    value = value.ravel()
                if isinstance(value, (np.ndarray, tuple, list)):
                    coos[var].update(value)
                else:
                    coos[var].add(value)

        self.coos = _reorganise(coos)
        logger.debug("Created coordinates map")
        return coos

    def store_coords(self):
        """
        Write coordinate arrays into the output
        """
        group = zarr.open(self.out)
        m = self.fss[0].get_mapper("")
        z = zarr.open(m)
        for k, v in self.coos.items():
            if k == "var":
                # The names of the variables to write in the second pass, not a coordinate
                continue
            # parametrize the threshold value below?
            compression = numcodecs.Zstd() if len(v) > 100 else None
            if all([isinstance(_, (tuple, list)) for _ in v]):
                v = sum([list(_) if isinstance(_, tuple) else _ for _ in v], [])
                data = np.array(v, dtype=self.coo_dtypes.get(k))
            else:
                data = np.concatenate([np.atleast_1d(np.array(_, dtype=self.coo_dtypes.get(k)))
                                       for _ in v])
            arr = group.create_dataset(name=k, data=np.array(v).ravel(),
                                       overwrite=True, compressor=compression,
                                       dtype=self.coo_dtypes.get(k, data.dtype))
            if k in z:
                # copy attributes if values came from an original variable
                arr.attrs.update(z[k].attrs)
            arr.attrs["_ARRAY_DIMENSIONS"] = [k]
        logger.debug("Written coordinates")
        for fn in [".zgroup", ".zattrs"]:
            # top-level group attributes from first input
            if fn in m:
                self.out[fn] = ujson.dumps(ujson.loads(m[fn]))
        logger.debug("Written global metadata")

    def second_pass(self):
        """map every input chunk to the output"""
        chunk_sizes = {}  #
        skip = set()
        dont_skip = set()
        no_deps = None

        for i, fs in enumerate(self.fss):
            m = fs.get_mapper("")
            z = zarr.open(m)

            if no_deps is None:
                # done first time only
                deps = [z[_].attrs["_ARRAY_DIMENSIONS"] for _ in z]
                all_deps = set(sum(deps, []))
                no_deps = set(self.coo_map) - all_deps

            # Coordinate values for the whole of this dataset
            cvalues = {c: self._get_value(i, z, c, fn=self._paths[i])
                       for c in self.coo_map}
            var = cvalues.get("var", None)

            for v in fs.ls("", detail=False):
                if v in self.coo_map or v in skip or v.startswith(".z"):
                    # already made coordinate variables and metadata
                    continue
                logger.debug("Second pass: %s, %s", i, v)

                zarray = ujson.loads(m[f"{v}/.zarray"])
                if v not in chunk_sizes:
                    chunk_sizes[v] = zarray["chunks"]
                else:
                    assert chunk_sizes[v] == zarray["chunks"], "Found chunk size mismatch"
                chunks = chunk_sizes[v]
                zattrs = ujson.loads(m[f"{v}/.zattrs"])
                coords = zattrs["_ARRAY_DIMENSIONS"]

                if v not in dont_skip and v in all_deps:
                    # this is an input coordinate
                    # a coordinate is any array appearing in its own or other array's _ARRAY_DIMENSIONS
                    skip.add(v)
                    for k in fs.ls(v, detail=False):
                        self.out[k] = fs.references[k]
                    continue

                # TODO: identical check here, if array does not depend on a concat dim
                #  This is only for the case that none of the data arrays depends on a concat dim
                # if IKnowItsIdentical:
                #     skip.add(v)
                #     continue

                dont_skip.add(v)  # don't check for coord or identical again

                coord_order = [c for c in self.concat_dims if c not in coords and c != "var"] + coords

                # create output array, accounting for shape, chunks and dim dependencies
                if f"{var or v}/.zarray" not in self.out:
                    shape = []
                    ch = []
                    for c in coord_order:
                        if c in self.coos:
                            shape.append(self.coos[c].size if isinstance(self.coos[c], np.ndarray)
                                         else len(self.coos[c]))
                        else:
                            shape.append(zarray["shape"][coords.index(c)])
                        ch.append(chunks[coords.index(c)] if c in coords else 1)

                    zarray['shape'] = shape
                    zarray['chunks'] = ch
                    zattrs["_ARRAY_DIMENSIONS"] = coord_order
                    self.out[f"{var or v}/.zarray"] = ujson.dumps(zarray)
                    # other attributes copied as-is from first occurrence of this array
                    self.out[f"{var or v}/.zattrs"] = ujson.dumps(zattrs)

                for fn in fs.ls(v, detail=False):
                    # loop over the chunks and copy the references
                    if ".z" in fn:
                        continue
                    key_parts = fn.split("/", 1)[1].split(".")
                    key = f"{var or v}/"
                    for loc, c in enumerate(coord_order):
                        if c in self.coos:
                            cv = cvalues[c]
                            if isinstance(cv, np.ndarray):
                                cv = cv.ravel()
                            if isinstance(cv, (np.ndarray, list, tuple)):
                                cv = tuple(sorted(set(cv)))[0]
                            ind = self.coos[c].index(cv)
                            if key_parts == ["10", "3", "3"] and i > 1:
                                import pdb
                                #pdb.set_trace()
                            if c in coords:
                                key += str(ind // ch[loc] + int(key_parts[coords.index(c)]))
                            else:
                                key += str(ind // ch[loc])
                        else:
                            key += key_parts[coords.index(c)]
                        key += "."
                    key = key.rstrip(".")
                    self.out[key] = fs.references[fn]

    def consolidate(self):
        """Turn raw references into output, inlining as needed"""
        out = {}
        for k, v in self.out.items():
            if isinstance(v, list) and len(v) > 1 and v[2] < self.inline:
                # TODO: check original file's size if not a range
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

    def translate(self):
        """Perform all stages and return the resultant references dict"""
        self.first_pass()
        self.store_coords()
        self.second_pass()
        return self.consolidate()


def _reorganise(coos):
    # reorganise and sort coordinate values
    # extracted here to enable testing
    out = {}
    for k, arr in coos.items():
        out[k] = list(sorted(arr))
    return out

