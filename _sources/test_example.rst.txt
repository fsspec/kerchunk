Quick Start
************

This is a run-through example for how to use this package. We scan a set of netCDF4/HDF5 files,
and create a single emsemble, virtual dataset, which can be read in parallel from remote
using ``zarr``.

Single file JSONs
=================

This will create a ``.json`` file for each of the files defined in ``urllist``. In this case,
we simply keep the resultant reference sets in memory, but we could have written them into
JSON files. Writing to files is useful, so that we can access the individual datasets, or
redo the combine (which is the next step, below).

.. code-block:: python

    import kerchunk.hdf
    import fsspec

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
    singles = []
    for u in urls:
        with fsspec.open(u, **so) as inf:
            h5chunks = kerchunk.hdf.SingleHdf5ToZarr(inf, u, inline_threshold=100)
            singles.append(h5chunks.translate())


Multi-file JSONs
================

This code uses the output generated above to create a single ensemble dataset, with
one set of references pointing to all of the chunks in the individual files.

.. code-block:: python

    from kerchunk.combine import MultiZarrToZarr
    mzz = MultiZarrToZarr(
        singles,
        remote_protocol="s3",
        remote_options={'anon': True},
        concat_dims=["time"]
    )

    out = mzz.translate()

Again, ``out`` could be written to a JSON file by providing arguments to
``translate()``. Crucially, there is no restriction on where
this lives, it can be anywhere that fsspec can read from.

Using the output
================

This is what a user of the generated dataset would do. This person does not need to have
``kerchunk`` installed, or even ``h5py`` (the library we used to initially scan the files).

.. code-block:: python

    import xarray as xr
    ds = xr.open_dataset(
        "reference://", engine="zarr",
        backend_kwargs={
            "storage_options": {
                "fo": out,
                "remote_protocol": "s3",
                "remote_options": {"anon": True}
            },
            "consolidated": False
        }
    )
    # do analysis...
    ds.velocity.mean()

Since the invocation for xarray to read this data is a little involved, we recommend
declaring the data set in an ``intake`` catalog. Alternatively, you might split the command
into multiple lines by first constructing the filesystem or mapper (you will see this in some
examples).

Note that, if the combining was done previously and saved to a JSON file, then the path to
it should replace ``out``, above, along with a ``target_options`` for any additional
arguments fsspec might to access it

Example/Tutorial Notebook
=========================

A set of tutorials notebooks, presented at the Earth Science Information Partners 2022 Winter Meeting, can be found at the following link, along with links to run the code on free cloud-based notebook environments: https://github.com/lsterzinger/2022-esip-kerchunk-tutorial

.. raw:: html

    <script data-goatcounter="https://kerchunk.goatcounter.com/count"
            async src="//gc.zgo.at/count.js"></script>
