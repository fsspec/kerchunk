Example Tests
*************

These are some reproducible examples of how to use ``fsspec-reference-maker``.

Single file JSONs
=================

This will create a ``.json`` file for each of the files defined in ``urllist`` in a file called 
``out.zip``.

.. code-block:: python

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
    zf = zipfile.ZipFile("out.zip", mode="w")
    for u in urls:
        with fsspec.open(u, **so) as inf:
            h5chunks = SingleHdf5ToZarr(inf, u, xarray=True, inline_threshold=100)
            with zf.open(os.path.basename(u) + ".json", 'w') as outf:
                outf.write(json.dumps(h5chunks.translate()).encode())


Multi-file JSONs
================

This code will read the ``out.zip`` file generated above and create a single ``.zarr`` 
or ``.json`` reference file that points to all the individual files as a single dataset.

.. code-block:: python

    mzz = MultiZarrToZarr(
        "zip://*.json::out.zip",
        remote_protocol="s3",
        remote_options={'anon': True},
        xarray_kwargs={
            "preprocess": drop_coords,
            "decode_cf": False,
            "mask_and_scale": False,
            "decode_times": False,
            "decode_timedelta": False,
            "use_cftime": False,
            "decode_coords": False
        },
    )


    mzz.translate("output.zarr")

    # This can also be written as a json
    mzz.translate("output.json")
