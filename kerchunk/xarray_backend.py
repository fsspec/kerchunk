from xarray.backends import BackendEntrypoint
import xarray as xr
import os
import fsspec


class KerchunkBackend(BackendEntrypoint):
    def open_dataset(
        self, filename_or_obj, *, storage_options=None, open_dataset_options=None, **kw
    ):
        open_dataset_options = (open_dataset_options or {}) | kw
        ref_ds = open_reference_dataset(
            filename_or_obj,
            storage_options=storage_options,
            open_dataset_options=open_dataset_options,
        )
        return ref_ds

    open_dataset_parameters = [
        "filename_or_obj",
        "storage_options",
        "open_dataset_options",
    ]

    def guess_can_open(self, filename_or_obj):
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        return ext in {".json", ".json.zstd", ".parquet"}

    description = "Open Kerchunk References with Xarray"
    url = "https://fsspec.github.io/kerchunk/"


def open_reference_dataset(
    filename_or_obj, *, storage_options=None, open_dataset_options=None
):
    if storage_options is None:
        storage_options = {}
    if open_dataset_options is None:
        open_dataset_options = {}

    m = fsspec.get_mapper("reference://", fo=filename_or_obj, **storage_options)

    return xr.open_dataset(m, engine="zarr", consolidated=False, **open_dataset_options)
