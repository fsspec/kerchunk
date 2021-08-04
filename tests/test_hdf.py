from fsspec_reference_maker.hdf import SingleHdf5ToZarr
from fsspec_reference_maker.combine import MultiZarrToZarr
import fsspec
import json

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
    with open('./example_jsons/single_example.json') as inf:
        file_dict = json.load(inf)
    
    assert(test_dict == file_dict)



def test_multizarr():
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
        remote_options={'anon': True, 'simple_templates' : True},
        preprocess=drop_coords,
        xarray_open_kwargs=xarray_open_kwargs,
        xarray_concat_args=concat_kwargs,
    )

    test_dict = mzz.translate()

    with open('./example_jsons/multizarr_example.json','r') as inf:
        file_dict = json.load(inf)

    assert(test_dict == file_dict)
