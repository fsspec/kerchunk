import base64
import fsspec
import logging
import numcodecs
import numpy as np
import zarr

from fsspec_reference_maker.utils import _unstrip_protocol
logger = logging.getLogger("fits-to-zarr")


BITPIX2DTYPE = {8: 'uint8', 16: '>i2', 32: '>i4', 64: '>i8',
                -32: 'float32', -64: 'float64'}


def process_file(url, storage_options=None, extension=None, ext_coords=False,
                 primary_attr_to_group=False, chunksize=None):
    """

    :param url: str
        Where the files are
    :param storage_options: dict
        How to load them
    :param extension: int, list(int), None
        Which extensons to include. If an integer, just use that extension. If a list, those extension
        indexes. If None, use the first data extension only.
    :param primary_attr_to_group: bool
        Whether the output top-level group contains the attributes of the primary extension
        (which often contains no data, just a general description)
    """
    from astropy.io import fits
    storage_options = storage_options or {}
    out = {}
    g = zarr.open(out)
    if ext_coords or chunksize is not None:
        raise NotImplementedError

    with fsspec.open(url, mode="rb", **storage_options) as f:
        infile = fits.open(f, do_not_scale_image_data=True)
        if extension is None:
            found = False
            for i, hdu in enumerate(infile):
                if hdu.header.get("NAXIS", 0) > 0:
                    extension = [i]
                    found = True
                    break
            if not found:
                raise ValueError("No data extensions")
        elif isinstance(extension, int):
            extension = [extension]

        for ext in extension:
            hdu = infile[ext]
            hdu.header.__str__()  # causes fixing of invalid cards

            # for images/cubes (i.e., ndarrays with simple type)
            assert hdu.header["NAXIS"] > 1
            shape = [hdu.header[f"NAXIS{i + 1}"] for i in range(hdu.header["NAXIS"])]
            dtype = BITPIX2DTYPE[hdu.header['BITPIX']]
            size = np.dtype(dtype).itemsize
            for s in shape:
                size *= s
            attrs = dict(hdu.header)

            if 'BSCALE' in hdu.header or 'BZERO' in hdu.header:
                filter = [
                    numcodecs.FixedScaleOffset(
                        offset=float(hdu.header.get("BZERO", 0)),
                        scale=float(hdu.header.get("BSCALE", 1)),
                        astype=dtype,
                        dtype=float
                    )
                ]
            else:
                filter=None
            # TODO: if chunksize is not None, calculate chunks here
            chunks = shape
            arr = g.empty(hdu.name, dtype=dtype, shape=shape, chunks=chunks, compression=None,
                          filters=filter)
            arr.attrs.update({k: str(v) if not isinstance(v, (int, float, str)) else v
                              for k, v in attrs.items() if k != "COMMENT"})
            arr.attrs["_ARRAY_DIMENSIONS"] = ["z", "y", "x"][-len(shape):]
            loc = hdu.fileinfo()["datLoc"]
            if chunksize is None:
                # one chunk for the whole thing
                parts = ".".join(["0"] * len(shape))
                out[f"{hdu.name}/{parts}"] = [url, loc, loc + arr.nbytes]
        if primary_attr_to_group:
            hdu.infile[0]
            hdu.header.__str__()
            g.attrs.update({k: str(v) if not isinstance(v, (int, float, str)) else v
                            for k, v in dict(hdu.header).items() if k != "COMMENT"})
    return out


def add_wcs_coords(hdu, shape, zarr_group=None, dataset=None, dtype=float):
    from astropy.wcs import WCS
    from astropy.io import fits

    if zarr_group is None and dataset is None:
        raise ValueError("please provide a zarr group or xarray dataset")

    if not isinstance(hdu, fits.hdu.base._BaseHDU):
        # assume dict-like
        head = fits.Header()
        hdu2 = hdu.copy()
        hdu2.pop("COMMENT", None)  # comment fields can be non-standard
        head.update(hdu2)
        hdu = fits.PrimaryHDU(header=head)

    wcs = WCS(hdu)
    coords = [coo.ravel() for coo in np.meshgrid(*(np.arange(sh) for sh in shape))]  # ?[::-1]
    world_coords = wcs.pixel_to_world(*coords)
    for i, (name, world_coord) in enumerate(zip(wcs.axis_type_names, world_coords)):
        dims = ['z', 'y', 'x'][3 - len(shape):]
        attrs = {"unit": world_coord.unit.name,
                 "type": hdu.header[f"CTYPE{i + 1}"],
                 "_ARRAY_DIMENSIONS": dims}
        if zarr_group is not None:
            arr = zarr_group.empty(name, shape=shape,
                                   chunks=shape, overwrite=True, dtype=dtype)
            arr.attrs.update(attrs)
            arr[:] = world_coord.value.reshape(shape)
        if dataset is not None:
            import xarray as xr
            coo = xr.Coordinate(data=world_coord.value.reshape(shape),
                                dims=dims, attrs=attrs)
            dataset = dataset.assign_coordinates(name=coo)
    if dataset is not None:
        return dataset
