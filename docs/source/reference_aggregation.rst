Other Methods of Aggregations
=============================

As we have already seen in this `page <https://fsspec.github.io/kerchunk/test_example.html#multi-file-jsons>`_,
that the main purpose of ``kerchunk`` it to generate references, to view whole archive of files like
GRIB2, NetCDF etc. allowing us for *in-situ* access to the data. In this part of the documentation,
we will see some other efficient ways of combining references.

GRIB Aggregations
-----------------

This new method for reference aggregation, developed by **Camus Energy**, is based on GRIB2 files. Utilizing
this method can significantly reduce the time required to combine references, cutting it down to
a fraction of the previous duration. In reality, this approach builds upon consolidating references
with ``kerchunk.combine.MultiZarrtoZarr``, making it faster. The functions and operations used in this
will be a part of the ``kerchunk``'s API.

*How is it faster*

Every GRIB file stored on cloud platforms such as **AWS** and **GCP** is accompanied by its
corresponding ``.idx`` file. This file otherwise known as an *index* file contains the key
metadata of the messages in the GRIB files. These metadata include `index`, `offset`, `datetime`,
`variable` and `forecast time` for their respective messages stored in the files.

These metadata will be used to build an k_index for every GRIB message that we will be
indexing. The indexing process primarily involves the `pandas <https://pandas.pydata.org/>`_ library.

.. list-table:: k_index for a single GRIB file
   :header-rows: 1
   :widths: 5 10 15 10 20 15 10 20 20 30 10 10 10

   * -
     - varname
     - typeOfLevel
     - stepType
     - name
     - step
     - level
     - time
     - valid_time
     - uri
     - offset
     - length
     - inline_value
   * - 0
     - gh
     - isobaricInhPa
     - instant
     - Geopotential height
     - 0 days 06:00:00
     - 0.0
     - 2017-01-01 06:00:00
     - 2017-01-01 12:00:00
     - s3://noaa-gefs-pds/gefs.20170101/06/gec00.t06z...
     - 0
     - 47493
     - None
   * - 1
     - t
     - isobaricInhPa
     - instant
     - Temperature
     - 0 days 06:00:00
     - 0.0
     - 2017-01-01 06:00:00
     - 2017-01-01 12:00:00
     - s3://noaa-gefs-pds/gefs.20170101/06/gec00.t06z...
     - 47493
     - 19438
     - None
   * - 2
     - r
     - isobaricInhPa
     - instant
     - Relative humidity
     - 0 days 06:00:00
     - 0.0
     - 2017-01-01 06:00:00
     - 2017-01-01 12:00:00
     - s3://noaa-gefs-pds/gefs.20170101/06/gec00.t06z...
     - 66931
     - 10835
     - None
   * - 3
     - u
     - isobaricInhPa
     - instant
     - U component of wind
     - 0 days 06:00:00
     - 0.0
     - 2017-01-01 06:00:00
     - 2017-01-01 12:00:00
     - s3://noaa-gefs-pds/gefs.20170101/06/gec00.t06z...
     - 77766
     - 22625
     - None
   * - 4
     - v
     - isobaricInhPa
     - instant
     - V component of wind
     - 0 days 06:00:00
     - 0.0
     - 2017-01-01 06:00:00
     - 2017-01-01 12:00:00
     - s3://noaa-gefs-pds/gefs.20170101/06/gec00.t06z...
     - 100391
     - 20488
     - None


.. note::
    The index in ``idx`` file indexes the GRIB messages where as the ``k_index`` (kerchunk index)
    we build as part of this workflow index the variables in those messages.

*What now*

After creating the k_index as per the desired duration, we will use the ``DataTree`` model
from the `xarray-datatree <https://xarray-datatree.readthedocs.io/en/latest/>`_ to view a
part of the aggregation or the whole. Below is a tree model made from an aggregation of
GRIB files produced from **GEFS** model hosted in AWS S3 bucket.

.. code-block:: bash

    DataTree('None', parent=None)
    ├── DataTree('prmsl')
    │   │   Dimensions:  ()
    │   │   Data variables:
    │   │       *empty*
    │   │   Attributes:
    │   │       name:     Pressure reduced to MSL
    │   └── DataTree('instant')
    │       │   Dimensions:  ()
    │       │   Data variables:
    │       │       *empty*
    │       │   Attributes:
    │       │       stepType:  instant
    │       └── DataTree('meanSea')
    │               Dimensions:     (latitude: 181, longitude: 360, time: 1, step: 1,
    │                                model_horizons: 1, valid_times: 237)
    │               Coordinates:
    │                 * latitude    (latitude) float64 1kB 90.0 89.0 88.0 87.0 ... -88.0 -89.0 -90.0
    │                 * longitude   (longitude) float64 3kB 0.0 1.0 2.0 3.0 ... 357.0 358.0 359.0
    │                   meanSea     float64 8B ...
    │                   number      (time, step) int64 8B ...
    │                   step        (model_horizons, valid_times) timedelta64[ns] 2kB ...
    │                   time        (model_horizons, valid_times) datetime64[ns] 2kB ...
    │                   valid_time  (model_horizons, valid_times) datetime64[ns] 2kB ...
    │               Dimensions without coordinates: model_horizons, valid_times
    │               Data variables:
    │                   prmsl       (model_horizons, valid_times, latitude, longitude) float64 124MB ...
    │               Attributes:
    │                   typeOfLevel:  meanSea
    └── DataTree('ulwrf')
        │   Dimensions:  ()
        │   Data variables:
        │       *empty*
        │   Attributes:
        │       name:     Upward long-wave radiation flux
        └── DataTree('avg')
            │   Dimensions:  ()
            │   Data variables:
            │       *empty*
            │   Attributes:
            │       stepType:  avg
            └── DataTree('nominalTop')
                    Dimensions:     (latitude: 181, longitude: 360, time: 1, step: 1,
                                        model_horizons: 1, valid_times: 237)
                    Coordinates:
                        * latitude    (latitude) float64 1kB 90.0 89.0 88.0 87.0 ... -88.0 -89.0 -90.0
                        * longitude   (longitude) float64 3kB 0.0 1.0 2.0 3.0 ... 357.0 358.0 359.0
                        nominalTop  float64 8B ...
                        number      (time, step) int64 8B ...
                        step        (model_horizons, valid_times) timedelta64[ns] 2kB ...
                        time        (model_horizons, valid_times) datetime64[ns] 2kB ...
                        valid_time  (model_horizons, valid_times) datetime64[ns] 2kB ...
                    Dimensions without coordinates: model_horizons, valid_times
                    Data variables:
                        ulwrf       (model_horizons, valid_times, latitude, longitude) float64 124MB ...
                    Attributes:
                        typeOfLevel:  nominalTop

.. tip::
    For a full tutorial on this workflow, refer this `kerchunk cookbook <https://projectpythia.org/kerchunk-cookbook/README.html>`_
    in `Project Pythia <https://projectpythia.org/>`_.

.. raw:: html

    <script data-goatcounter="https://kerchunk.goatcounter.com/count"
            async src="//gc.zgo.at/count.js"></script>
