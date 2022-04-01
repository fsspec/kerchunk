from os.path import exists
from setuptools import setup
import versioneer

setup(
    name='kerchunk',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=['kerchunk', 'fsspec_reference_maker'],
    url='https://github.com/fsspec/kerchunk',
    license='MIT',
    author='Martin Durant',
    author_email='martin.durant@alumni.utoronto.ca',
    description='Functions to make reference descriptions for ReferenceFileSystem',
    python_requires='>=3.7',
    long_description=(open('README.md').read() if exists('README.md') else ''),
    long_description_content_type='text/markdown',
    install_requires=list(open('requirements.txt').read().strip().split('\n')),
    extras_require={
        "cftime": ["cftime"],
        "fits": ["xarray"],
        "hdf": ["h5py", "xarray"],
        "grib2": ["cfgrib"],
    },
    entry_points={
        'numcodecs.codecs': [
            'grib = kerchunk.grib2:GRIBCodec',
            'FITSAscii = kerchunk.fits:AsciiTableCodec'
        ],
    },
    zip_safe=False
)
