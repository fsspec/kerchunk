import base64
from collections import Counter
import json
import logging
import os

import fsspec
import numcodecs
import xarray as xr
import zarr
logging = logging.getLogger('reference-combine')


class MultiZarrToZarr:

    def __init__(self, path, remote_protocol,
                 remote_options=None, xarray_kwargs=None, storage_options=None,
                 with_mf=True):
        """

        :param path: a URL containing multiple JSONs
        :param xarray_kwargs:
        :param storage_options:
        """
        xarray_kwargs = xarray_kwargs or {}
        self.path = path
        self.with_mf = with_mf
        self.xr_kwargs = xarray_kwargs
        self.storage_options = storage_options or {}
        self.remote_protocol = remote_protocol
        self.remote_options = remote_options or {}

    def translate(self, outpath):
        ds, ds0, fss = self._determine_dims()
        out = self._build_output(ds, ds0, fss)
        self.output = self._consolidate(out)

        self._write(self.output, outpath)

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

    def _consolidate(self, mapping, inline_threashold=100, template_count=5):
        counts = Counter(v[0] for v in mapping.values() if isinstance(v, list))

        def letter_sets():
            import string
            import itertools
            yield from string.ascii_letters
            for a, b in itertools.combinations(string.ascii_letters, 2):
                yield a + b
            for a, b, c in itertools.combinations(string.ascii_letters, 3):
                yield a + b + c

        templates = {i: u for i, (u, v) in zip(letter_sets(), counts.items())
                     if v > template_count}
        inv = {v: k for k, v in templates.items()}

        out = {}
        for k, v in mapping.items():
            if isinstance(v, list) and v[2] < inline_threashold:
                v = self.fs.cat_file(v[0], start=v[1], end=v[1] + v[2])
            if isinstance(v, bytes):
                try:
                    # easiest way to test if data is ascii
                    out[k] = v.decode('ascii')
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
        ds.to_zarr(out, chunk_store={}, compute=False)  # fills in metadata&coords
        z = zarr.open_group(out, mode='a')
        for dim in self.concat_dims:
            # concatenated dims stored as absolute data
            z[dim][:] = ds[dim].values
        for dim in self.same_dims:
            # duplicated coordinates stored as references just once
            out.update({k: v for k, v in fss[0].references.items() if k.startswith(dim)})
        for variable in ds.variables:
            if variable in ds.dims or variable in self.same_dims or variable in self.concat_dims:
                # already handled above
                continue
            var, var0 = ds[variable], ds0[variable]
            for i, fs in enumerate(fss):
                for k, v in fs.references.items():
                    start, part = os.path.split(k)
                    if start != variable or part in ['.zgroup', '.zarray', '.zattrs']:
                        # OK, so we go through all the keys multiple times
                        continue
                    if var.shape == var0.shape:
                        out[k] = v  # copy
                    else:
                        out[f"{start}/{i}.{part}"] = v
        return out

    def _determine_dims(self):
        with fsspec.open_files(self.path, **self.storage_options) as ofs:
            fss = [
                fsspec.filesystem(
                    "reference", fo=json.load(of),
                    remote_protocol=self.remote_protocol,
                    remote_options=self.remote_options
                ) for of in ofs
            ]
            self.fs = fss[0].fs
            mappers = [fs.get_mapper("") for fs in fss]

        if self.with_mf is True:
            ds = xr.open_mfdataset(mappers, engine="zarr", chunks={}, **self.xr_kwargs)
            ds0 = xr.open_mfdataset(mappers[:1], engine="zarr", chunks={}, **self.xr_kwargs)
        else:
            dss = [xr.open_dataset(m, engine="zarr", chunks={}, **self.xr_kwargs) for m in mappers]
            ds = xr.concat(dss, dim=self.with_mf)
            ds0 = dss[0]
        self.extra_dims = set(ds.dims) - set(ds0.dims)
        self.concat_dims = set(k for k, v in ds.dims.items() if k in ds0.dims and v / ds0.dims[k] == len(mappers))
        self.same_dims = set(ds.dims) - self.extra_dims - self.concat_dims
        return ds, ds0, fss


def example_ensamble():
    """Scan the set of URLs and create a single reference output

    This example uses the output of hdf.example_multiple
    """
    def drop_coords(ds):
        ds = ds.drop(['reference_time', 'crs'])
        return ds.reset_coords(drop=True)

    mzz = MultiZarrToZarr(
        "zip://*.json::out.zip",
        remote_protocol="s3",
        remote_options={'anon': True},
        xarray_kwargs={
            "preprocess": drop_coords,
            "decode_cf": False,
            "mask_and_scale": False,
            "decode_times": False,
            "decode_timedelta": False,
            "use_cftime": False,
            "decode_coords": False
        },
    )
    mzz.translate("output.zarr")

