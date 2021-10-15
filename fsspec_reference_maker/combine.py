import base64
from collections import Counter
import ujson as json
from packaging import version
import logging
import os

import fsspec
import numcodecs
import numpy as np
import xarray as xr
import zarr
logger = logging.getLogger('reference-combine')


class MultiZarrToZarr:
    """Combine multiple reference files into one

    Parameters
    ----------
    path: string or list of strings
        List of JSON paths or a URL containing multiple JSONs
    
    remote_protocol: string
        Protocol used to access remote files (e.g. 's3', 'az', etc)

    remote_options : dict
        Dictionary of args to pass to ``fsspec.filesystem()``

    xarray_open_kwargs : dict
        Dictionary of args to pass to ``xr.open_dataset()``

    xarray_concat_args : dict
        Dictionary of args to pass to ``xr.concat()``

    preprocess : function
        Function take takes in/returns a ``xr.Dataset`` to be processed before dataset concatenation

    storage_options : dict
        Dictionary of args to pass to ``fsspec.open_files()``
    """

    def __init__(self, path, remote_protocol,
                 remote_options=None, xarray_open_kwargs=None, xarray_concat_args=None,
                 preprocess=None, storage_options=None):
        self.path = path
        self.xr_kwargs = xarray_open_kwargs or {}
        self.concat_kwargs = xarray_concat_args or {}
        self.storage_options = storage_options or {}
        self.preprocess = preprocess
        self.remote_protocol = remote_protocol
        self.remote_options = remote_options or {}

    def translate(self, outpath=None, template_count=5):
        """
        Translate the combined reference files and write to new file

        Parameters
        ----------
        outpath : String (optional)
            Path of file to be written. If left blank, ``MultiZarrToZarr.translate()`` returns a dict.

        template_count : int or None (optional, default=5)
            Set to ``None`` to disable templates

        """


        ds, ds0, fss = self._determine_dims()
        out = self._build_output(ds, ds0, fss)
        self.output = self._consolidate(out, template_count=template_count)

        if outpath:
            self._write(self.output, outpath)
        else:
            return self.output
        # TODO: return new zarr dataset?

    @staticmethod
    def _write(refs, outpath, filetype=None):
        types = {
            "json": "json",
            "parquet": "parquet",
            "zarr": "zarr"
        }
        if filetype is None:
            ext = os.path.splitext(outpath)[1].lstrip(".")
            filetype = types[ext]
        elif filetype not in types:
            raise KeyError
        if filetype == "json":
            with open(outpath, "w") as f:
                json.dump(refs, f)
            return
        import pandas as pd
        references2 = {
            k: {"data": v.encode('ascii') if not isinstance(v, list) else None,
                "url": v[0] if isinstance(v, list) else None,
                "offset": v[1] if isinstance(v, list) else None,
                "size": v[2] if isinstance(v, list) else None}
            for k, v in refs['refs'].items()}
        # use pandas for sorting
        df = pd.DataFrame(references2.values(), index=list(references2)).sort_values("offset")

        if filetype == "zarr":
            # compression should be NONE, if intent is to store in single zip
            g = zarr.open_group(outpath, mode='w')
            g.attrs.update({k: v for k, v in refs.items() if k in ['version', "templates", "gen"]})
            g.array(name="key", data=df.index.values, dtype="object", compression="zstd",
                    object_codec=numcodecs.VLenUTF8())
            g.array(name="offset", data=df.offset.values, dtype="uint32", compression="zstd")
            g.array(name="size", data=df['size'].values, dtype="uint32", compression="zstd")
            g.array(name="data", data=df.data.values, dtype="object",
                    object_codec=numcodecs.VLenBytes(), compression="gzip")
            # may be better as fixed length
            g.array(name="url", data=df.url.values, dtype="object",
                    object_codec=numcodecs.VLenUTF8(), compression='gzip')
        if filetype == "parquet":
            import fastparquet
            metadata = {k: v for k, v in refs.items() if k in ['version', "templates", "gen"]}
            fastparquet.write(
                outpath,
                df,
                custom_metadata=metadata,
                compression="ZSTD"
            )

    def _consolidate(self, mapping, inline_threshold=100, template_count=5):
        counts = Counter(v[0] for v in mapping.values() if isinstance(v, list))

        def letter_sets():
            import string
            import itertools
            i = 1
            while True:
                for tup in itertools.product(string.ascii_letters + string.digits, repeat=i):
                    if tup[0].isdigit():
                        continue
                    yield "".join(tup)
                i += 1

        templates = {i: u for i, (u, v) in zip(letter_sets(), counts.items())
                     if v > template_count} if template_count is not None else {}
        inv = {v: k for k, v in templates.items()}

        out = {}
        for k, v in mapping.items():
            if isinstance(v, list) and v[2] < inline_threshold:
                v = self.fs.cat_file(v[0], start=v[1], end=v[1] + v[2])
            if isinstance(v, bytes):
                try:
                    # easiest way to test if data is ascii
                    out[k] = v.decode('ascii')
                    try:
                        # minify json
                        out[k] = json.dumps(json.loads(out[k]))
                    except:
                        pass
                except UnicodeDecodeError:
                    out[k] = (b"base64:" + base64.b64encode(v)).decode()
            else:
                if v[0] in inv:
                    out[k] = ["{{" + inv[v[0]] + "}}"] + v[1:]
                else:
                    out[k] = v
        return {"version": 1, "templates": templates, "refs": out}

    def _build_output(self, ds, ds0, fss):
        out = {}
        logger.debug("write zarr metadata")
        ds.to_zarr(out, chunk_store={}, compute=False,
                   consolidated=False)  # fills in metadata&coords
        z = zarr.open_group(out, mode='a')
        accum_dim = list(self.concat_dims.union(self.extra_dims))[0]  # only ever one dim for now

        acc_len = make_coord(fss, z, accum_dim)

        for variable in ds.variables:
            logger.debug("considering %s", variable)

            # cases
            # a) this is accum_dim -> note values, dealt with above
            # b) this is a dimension that didn't change -> copy (once)
            # c) this is a normal var, without accum_dim, var.shape == var0.shape -> copy (once)
            # d) this is var needing reshape -> each dataset's keys get new names, update shape
            # e) this is a dimension that DOES change

            if variable == accum_dim:
                logger.debug("a)")
                continue

            var, var0 = ds[variable], ds0[variable]
            if variable in ds.dims or accum_dim not in var.dims:
                # b) and c)
                logger.debug(f"b) c) copy variable: {variable}")
                out.update({k: v for k, v in fss[0].references.items() if k.startswith(variable + "/")})
                continue

            if variable in ds.coords:
                logger.debug("e)")
                make_coord(fss, z, variable)
                continue

            logger.debug(f"d) process variable: {variable}")
            # update shape
            shape = list(var.shape)
            bit = json.loads(out[f"{variable}/.zarray"])
            if accum_dim in var0.dims:
                chunks_per_part = len(var0.chunks[var.dims.index(accum_dim)])
            else:
                chunks_per_part = 1
            shape[var.dims.index(accum_dim)] = acc_len
            bit["shape"] = shape
            out[f"{variable}/.zarray"] = json.dumps(bit)

            # handle components chunks
            for i, fs in enumerate(fss):
                for k, v in fs.references.items():
                    start, part = os.path.split(k)
                    if start != variable or part in ['.zgroup', '.zarray', '.zattrs']:
                        # OK, so we go through all the keys multiple times
                        continue
                    elif var.dims == var0.dims:
                        # concat only
                        parts = {d: c for d, c in zip(var.dims, part.split("."))}
                        parts = [parts[d] if d in self.same_dims else str(i * chunks_per_part + int(parts[d]))
                                 for d in var.dims]
                        out[f"{start}/{'.'.join(parts)}"] = v
                    else:
                        # merge with new coordinate
                        # i.e., self.extra_dims applies
                        out[f"{start}/{i}.{part}"] = v
        return out

    def _determine_dims(self):
        logger.debug("open mappers")

        # If self.path is a list of dictionaries, pass them directly to fsspec.filesystem
        import collections.abc
        if isinstance(self.path[0], collections.abc.Mapping):
            fo_list = self.path
        
        # If self.path is list of files, open the files and load the json as a dictionary
        else:
            with fsspec.open_files(self.path, **self.storage_options) as ofs:
                fo_list = [json.load(of) for of in ofs]

        fss = [
            fsspec.filesystem(
                "reference", fo=fo,
                remote_protocol=self.remote_protocol,
                remote_options=self.remote_options
            ) for fo in fo_list
        ]
        self.fs = fss[0].fs
        mappers = [fs.get_mapper("") for fs in fss]

        logger.debug("open first two datasets")
        xr_kwargs_copy = self.xr_kwargs.copy()
        
        # Add consolidated=False to xr kwargs if not explictly given by user
        # needed to suppress zarr open warnings
        if (version.parse(xr.__version__) >= version.parse("0.19.0")
                and 'consolidated' not in xr_kwargs_copy):
            xr_kwargs_copy['consolidated'] = False

        dss = [xr.open_dataset(m, engine="zarr", chunks={},  **xr_kwargs_copy)
               for m in mappers[:2]]

        if self.preprocess:
            logger.debug("preprocess")
            dss = [self.preprocess(d) for d in dss]
        logger.debug("concat")
        ds = xr.concat(dss, **self.concat_kwargs)
        ds0 = dss[0]
        self.extra_dims = set(ds.dims) - set(ds0.dims)
        self.concat_dims = set(
            k for k, v in ds.dims.items()
           if k in ds0.dims and v / ds0.dims[k] == 2
        )
        self.same_dims = set(ds.dims) - self.extra_dims - self.concat_dims
        return ds, ds0, fss


def make_coord(fss, z, accum_dim):
    # a)
    accum = []
    logger.debug("accumulate coords array %s", accum_dim)
    times = False
    for fs in fss:
        zz = zarr.open_array(fs.get_mapper(accum_dim))

        try:
            import cftime
            if not isinstance(zz, cftime.real_datetime):

                # Try and get the calendar attribute from "calendar" attribute
                # If it doesn't exist, assume a standard calendar
                if zz.attrs.get("calendar") is not None:
                    calendar = zz.attrs.get("calendar")
                else:
                    calendar = 'standard'

                    # Update attrs in z[accum_dim]
                    zattr = dict(z[accum_dim].attrs)
                    zattr['calendar'] = 'standard'
                    z[accum_dim].attrs.put(zattr)            
                
                zz = cftime.num2pydate(zz[...], units=zz.attrs["units"],
                                       calendar=calendar)
                times = True
                logger.debug("converted times")
                accum.append(zz)
            else:
                accum.append(zz)
        except Exception as e:
            ex = e
            accum.append(zz[...].copy())
    attr = dict(z[accum_dim].attrs)
    if times:
        accum = [np.array(a, dtype="M8") for a in accum]
        attr.pop('units', None)
    
    attr.pop('calendar', None)

    acc = np.concatenate([np.atleast_1d(a) for a in accum]).squeeze()

    logger.debug("write coords array")
    arr = z.create_dataset(name=accum_dim,
                           data=acc,
                           overwrite=True)
    arr.attrs.update(attr)
    return len(acc)
