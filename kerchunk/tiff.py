import json
import io
import fsspec
import zarr
import tifffile
import enum
import ujson


def tiff_to_zarr(urlpath, remote_options=None, inline_threshold=0, target=None, target_options=None):
    with fsspec.open(urlpath, **(remote_options or {})) as of:
        url, name = urlpath.rsplit('/', 1)

        with tifffile.TiffFile(of, name=name) as tif:
            with tif.series[0].aszarr() as store:
                of2 = io.StringIO()
                store.write_fsspec(of2, url=url)
                out = ujson.loads(of2.getvalue())

                meta = ujson.loads(out[".zattrs"])
                for k in dir(tif):
                    if not k.endswith("metadata"):
                        continue
                    meta.update(getattr(tif, k) or {})
                for k, v in meta.copy().items():
                    # deref enums
                    if isinstance(v, enum.EnumMeta):
                        meta[k] = v._name_
                out[".zattrs"] = ujson.dumps(meta)
    if target is not None:
        with fsspec.open(target, **(target_options or {})) as of:
            ujson.dump(out, of)
    return out
