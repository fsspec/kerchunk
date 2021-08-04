from fsspec_reference_maker.hdf import example_single

import fsspec
import json

def test_single():

    # Get output from hdf.py example_single()
    test_dict = example_single()

    # Compare to output from file
    with open('./example_jsons/single_example.json') as inf:
        file_dict = json.load(inf)
    
    assert(test_dict == file_dict)