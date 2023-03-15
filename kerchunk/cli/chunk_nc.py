import json
import logging
from pathlib import Path
from typing import List

import click
import fsspec

from kerchunk.hdf import SingleHdf5ToZarr
from kerchunk.combine import MultiZarrToZarr

logger = logging.getLogger("kercli")


class NetcdfChunker:
    dataset_name: str = "mydataset"
    input: List[str]
    input_format: str = "nc"
    input_fs_args: dict = {"anon": True}
    json_dir: Path = Path("json")
    zarr_output: Path = Path("zarr")
    force_scan: bool = False

    def __init__(self, *args, **kwargs):
        self.__dict__.update(**kwargs)
        self.json_dir = Path(self.json_dir)
        self.zarr_output = Path(self.zarr_output)

    @property
    def json_dataset_dir(self) -> Path:
        fp = self.json_dir / self.dataset_name
        if not fp.exists(): fp.mkdir(parents=True)
        return fp

    def run(self):
        self.scan_them_all()
        self.consolidate()

    def scan_them_all(self):
        # Scan all input file
        for fp in self.input:
            fp_json = self.json_dataset_dir / Path(fp).parent.as_posix() / Path(fp).with_suffix(".json").name
            fp_json.parent.mkdir(parents=True, exist_ok=True)

            if fp_json.exists() and not self.force_scan:
                logger.info(f"Scanning {fp} : {fp_json} already exists, skipping")
                continue

            logger.info(f"Scanning {fp} ...")
            with fsspec.open(fp, **self.input_fs_args) as fd:
                try:
                    h5chunk = SingleHdf5ToZarr(fd, fp, inline_threshold=100)
                    json_data = h5chunk.translate()
                except OSError as e:
                    logger.error(str(e))
                    continue

            logger.info(f"Saving to {fp_json}")
            with fp_json.open("w") as fd:
                json.dump(json_data, fd, indent=2)

    def consolidate(self):
        fp_json_list = [str(fp) for fp in self.json_dataset_dir.glob("**/*.json")]
        logger.info(f"Data loaded from {self.json_dataset_dir} : {len(fp_json_list)} found")
        mzz = MultiZarrToZarr(
            fp_json_list,
            concat_dims=["time0"]
        )

        fp_zarr = self.zarr_output / f"{self.dataset_name}.zarr"
        logger.info(f"Consolidating to {fp_zarr} ...")
        mzz.translate(filename=str(fp_zarr))


def str_to_json(ctx, param, value):
    if value and not isinstance(value, dict):
        value = json.loads(value)
    return value

@click.command()
@click.option("--name", help="Dataset name", default="mydataset", show_default=True)
@click.option("--input", "-i",
              help="Input file url, readable by fsspec", required=True,
              multiple=True)
@click.option("--input-format", default="nc")
@click.option("--input-fs-args", help="Arguments that will be passed to fsspec.open()",
              type=click.UNPROCESSED, callback=str_to_json, default={'anon': True}, show_default=True)
@click.option("--json-dir", help="Where to store scan output as json", default=Path("json"))
@click.option("--zarr-output", help="Output of fully merged kerchunk zarr file", default=Path("zarr"))
@click.option("--force-scan",
              help="Force scanning input file, even if json file exists",
              is_flag=True, default=False)
@click.option('--verbose', '-v', is_flag=True)
def cli(verbose: bool, **kwargs):
    """ Cli for ker-chunking local or remote NetCDF files"""
    logging.basicConfig(level=logging.INFO)
    if verbose:
        # Show what's going on with fsspec
        _logger = logging.getLogger("fsspec")
        _logger.setLevel(logging.DEBUG)
    c = NetcdfChunker(**kwargs)
    c.run()


if __name__ == "__main__":
    cli()
