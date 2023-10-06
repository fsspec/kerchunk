from xarray.backends import BackendEntrypoint
import xarray as xr
import os
from fsspec.implementations.reference import ReferenceFileSystem


class KerchunkBackend(BackendEntrypoint):
    def open_dataset(
        self,
        filename_or_obj,
        **kwargs,
    ):
        return my_open_dataset(filename_or_obj, **kwargs)

    open_dataset_parameters = ["filename_or_obj"]

    def guess_can_open(self, filename_or_obj):
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        return ext in {".json", ".parquet"}

    description = "Open Kerchunk References with Xarray"
    url = "https://fsspec.github.io/kerchunk/"


def my_open_dataset(filename_or_obj, **kwargs):
    fs = ReferenceFileSystem(fo=filename_or_obj, **kwargs)
    m = fs.get_mapper()
    return xr.open_dataset(m, engine="zarr", consolidated=False)
