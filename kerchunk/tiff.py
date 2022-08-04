import io
import fsspec
import enum
import ujson


def tiff_to_zarr(urlpath, remote_options=None, target=None, target_options=None):
    """
    Wraps TIFFFile's fsspec writer to extract metadata as attributes

    Parameters
    ----------
    urlpath: str
        Location of input TIFF
    remote_options: dict
        pass these to fsspec when opening urlpath
    target: str
        Write JSON to this location. If not given, no file is output
    target_options: dict
        pass these to fsspec when opening target

    Returns
    -------
    references dict
    """
    import tifffile

    with fsspec.open(urlpath, **(remote_options or {})) as of:
        url, name = urlpath.rsplit("/", 1)

        with tifffile.TiffFile(of, name=name) as tif:
            with tif.series[0].aszarr() as store:
                of2 = io.StringIO()
                store.write_fsspec(of2, url=url)
                out = ujson.loads(of2.getvalue())

                meta = ujson.loads(out[".zattrs"])
                for k in dir(tif):
                    if not k.endswith("metadata"):
                        continue
                    met = getattr(tif, k, None)
                    try:
                        d = dict(met or {})
                    except ValueError:
                        # newer tifffile exposes xml structured tags
                        from xml.etree import ElementTree

                        e = ElementTree.fromstring(met)
                        d = {i.get("name"): i.text for i in e}
                    meta.update(d)
                for k, v in meta.copy().items():
                    # deref enums
                    if isinstance(v, enum.EnumMeta):
                        meta[k] = v._name_
                out[".zattrs"] = ujson.dumps(meta)
    if target is not None:
        with fsspec.open(target, **(target_options or {})) as of:
            ujson.dump(out, of)
    return out
