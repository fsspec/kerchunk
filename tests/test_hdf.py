import os.path

from fsspec_reference_maker.hdf import SingleHdf5ToZarr
from fsspec_reference_maker.combine import MultiZarrToZarr
import fsspec
import json
import numpy as np
import pytest
import xarray as xr

here = os.path.abspath(os.path.dirname(__file__))


def test_single():
    """Test creating references for a single HDF file"""
    url = 's3://noaa-nwm-retro-v2.0-pds/full_physics/2017/201704010000.CHRTOUT_DOMAIN1.comp'
    so = dict(
        mode='rb', anon=True, default_fill_cache=False, default_cache_type="none"
    )
    with fsspec.open(url, **so) as f:
        h5chunks = SingleHdf5ToZarr(f, url)

        # Get output from hdf.py example_single()
        test_dict = h5chunks.translate()

    # Compare to output from file
    with open(os.path.join(here, 'example_jsons/single_example.json')) as inf:
        file_dict = json.load(inf)
    
    assert(test_dict == file_dict)


@pytest.mark.parametrize("templates", [True, False])
def test_multizarr(templates):
    """Test creating a combined reference file with MultiZarrToZarr"""
    urls = ["s3://" + p for p in [
        'noaa-nwm-retro-v2.0-pds/full_physics/2017/201704010000.CHRTOUT_DOMAIN1.comp',
        'noaa-nwm-retro-v2.0-pds/full_physics/2017/201704010100.CHRTOUT_DOMAIN1.comp',
        'noaa-nwm-retro-v2.0-pds/full_physics/2017/201704010200.CHRTOUT_DOMAIN1.comp',
        'noaa-nwm-retro-v2.0-pds/full_physics/2017/201704010300.CHRTOUT_DOMAIN1.comp',
        'noaa-nwm-retro-v2.0-pds/full_physics/2017/201704010400.CHRTOUT_DOMAIN1.comp',
        'noaa-nwm-retro-v2.0-pds/full_physics/2017/201704010500.CHRTOUT_DOMAIN1.comp',
        'noaa-nwm-retro-v2.0-pds/full_physics/2017/201704010600.CHRTOUT_DOMAIN1.comp',
        'noaa-nwm-retro-v2.0-pds/full_physics/2017/201704010700.CHRTOUT_DOMAIN1.comp',
        'noaa-nwm-retro-v2.0-pds/full_physics/2017/201704010800.CHRTOUT_DOMAIN1.comp',
        'noaa-nwm-retro-v2.0-pds/full_physics/2017/201704010900.CHRTOUT_DOMAIN1.comp'
    ]]
    so = dict(
        anon=True, default_fill_cache=False, default_cache_type='first'
    )

    dict_list = []

    for u in urls:
        with fsspec.open(u, **so) as inf:
            h5chunks = SingleHdf5ToZarr(inf, u, inline_threshold=100)
            dict_list.append(h5chunks.translate())

    if templates:
        mzz = generate_mzz(dict_list)
    else:
        mzz = generate_mzz(dict_list, template_count=None)

    test_dict = mzz.translate()

    ds = xr.open_dataset(
        "reference://", engine="zarr",
        backend_kwargs={
            "consolidated": False,
            "storage_options": {
                "fo": test_dict,
                "target_protocol": "s3",
                "target_options": {"anon": True}
            }
        }
    )
    assert ds.dims == {"time": 10, "feature_id": 2729077}
    assert ds.time.values == np.array(
        ["2017-04-01T00:00:00", "2017-04-01T01:00:00", "2017-04-01T02:00:00", "2017-04-01T03:00:00",
         "2017-04-01T04:00:00", "2017-04-01T05:00:00", "2017-04-01T06:00:00", "2017-04-01T07:00:00",
         "2017-04-08T00:00:00", "2017-04-09T01:00:00"], dtype="M8")
    assert (ds.velocity[0, 0].values - 0.04) < 0.01


def generate_mzz(dict_list):
    """This function generates a MultiZarrToZarr class for use with the ``example_multizarr*.py`` testss"""
    def drop_coords(ds):
        ds = ds.drop_vars(['reference_time', 'crs'])
        return ds.reset_coords(drop=True)

    xarray_open_kwargs = {
        "decode_cf": False,
        "mask_and_scale": False,
        "decode_times": False,
        "decode_timedelta": False,
        "use_cftime": False,
        "decode_coords": False
    }

    concat_kwargs = {
        "dim": "time"
    }

    mzz = MultiZarrToZarr(
        dict_list,
        remote_protocol="s3",
        remote_options={'anon': True, 'simple_templates': True},
        preprocess=drop_coords,
        xarray_open_kwargs=xarray_open_kwargs,
        xarray_concat_args=concat_kwargs,
    )

    return mzz
