from fsspec_reference_maker.hdf import SingleHdf5ToZarr
import fsspec
import json

def test_single():

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