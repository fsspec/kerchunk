from xarray.backends import BackendEntrypoint
import xarray as xr
import os
from fsspec.implementations.reference import ReferenceFileSystem


class KerchunkBackend(BackendEntrypoint):
    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        storage_options=None,
    ):

        ref_ds = open_reference_dataset(
            filename_or_obj,
            storage_options=storage_options,
        )
        if drop_variables is not None:
            ref_ds = ref_ds.drop_vars(drop_variables)
        return ref_ds

    open_dataset_parameters = [
        "filename_or_obj",
        "storage_options",
    ]

    def guess_can_open(self, filename_or_obj):
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        return ext in {".json", ".parquet"}

    description = "Open Kerchunk References with Xarray"
    url = "https://fsspec.github.io/kerchunk/"


def open_reference_dataset(filename_or_obj, *, storage_options=None):
    fs = ReferenceFileSystem(fo=filename_or_obj, **storage_options)
    m = fs.get_mapper()
    return xr.open_dataset(m, engine="zarr", consolidated=False)
