from os.path import exists
from setuptools import setup
import versioneer

setup(
    name="kerchunk",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=["kerchunk"],
    url="https://github.com/fsspec/kerchunk",
    license="MIT",
    author="Martin Durant",
    author_email="martin.durant@alumni.utoronto.ca",
    description="Functions to make reference descriptions for ReferenceFileSystem",
    python_requires=">=3.7",
    long_description=(open("README.md").read() if exists("README.md") else ""),
    long_description_content_type="text/markdown",
    install_requires=list(open("requirements.txt").read().strip().split("\n")),
    extras_require={
        "cftime": ["cftime"],
        "fits": ["xarray"],
        "hdf": ["h5py", "xarray"],
        "grib2": ["cfgrib"],
        "netcdf3": ["scipy"],
    },
    entry_points={
        "xarray.backends": ["kerchunk=kerchunk.xarray_backend:KerchunkBackend"],
        "numcodecs.codecs": [
            "grib = kerchunk.codecs:GRIBCodec",
            "fill_hdf_strings = kerchunk.codecs:FillStringsCodec",
            "FITSAscii = kerchunk.codecs:AsciiTableCodec",
            "FITSVarBintable = kerchunk.codecs:VarArrCodec",
            "record_member = kerchunk.codecs:RecordArrayMember",
        ],
        'console_scripts': [
            'kerchunk-nc = kerchunk.cli.chunk_nc:cli',
        ],
    },
    zip_safe=False,
)
