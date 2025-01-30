"""
# Extract, Load and Build Kerchunk aggregations on the fly from hierarchical metadata and kerchunk indexes.

MIT License Copyright (c) 2023 Camus Energy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import copy
import re
import unittest
import logging
import gzip
import os
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from kerchunk.grib2 import (
    scan_grib,
    grib_tree,
    correct_hrrr_subhf_step,
    extract_datatree_chunk_index,
    strip_datavar_chunks,
    reinflate_grib_store,
    AggregationType,
    build_idx_grib_mapping,
    parse_grib_idx,
    map_from_index,
    read_store,
    write_store,
)
from kerchunk.utils import refs_as_store
import fsspec
import zarr
import ujson
import tempfile
import typing
import io

# https://github.com/pydata/xarray/issues/9984
# until datatree/xarray supports zarr3
pytest.skip(allow_module_level=True)
logger = logging.getLogger(__name__)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class DataExtractorTests(unittest.TestCase):
    def test_integration(self):
        # Small test file...
        uri = f"{THIS_DIR}/grib_idx_fixtures/hrrr.wrfsubhf.sample.grib2"

        scanned_msg_groups = scan_grib(uri)
        corrected_msg_groups = [
            correct_hrrr_subhf_step(msg) for msg in scanned_msg_groups
        ]
        grib_tree_store = grib_tree(corrected_msg_groups)
        zg = zarr.open_group("reference://", storage_options={"fo": grib_tree_store})
        self.assertIsInstance(zg["refc/instant/atmosphere/refc"], zarr.Array)
        self.assertIsInstance(zg["vbdsf/avg/surface/vbdsf"], zarr.Array)
        self.assertEqual(
            zg["vbdsf/avg/surface"].attrs["coordinates"],
            "surface latitude longitude step time valid_time",
        )
        self.assertEqual(
            zg["refc/instant/atmosphere"].attrs["coordinates"],
            "atmosphere latitude longitude step time valid_time",
        )
        # Assert that the fill value is set correctly
        self.assertIs(zg["refc"]["instant"]["atmosphere"]["step"].fill_value, np.nan)

        np.testing.assert_array_equal(
            zg["refc"]["instant"]["atmosphere"]["time"][:], np.array([1665709200])
        )

        # Read it with data tree and assert the same...
        dt = xr.open_datatree(
            "reference://",
            storage_options={"fo": grib_tree_store},
            engine="zarr",
            consolidated=False,
        )
        # Assert a few things... but if it loads we are mostly done.
        np.testing.assert_array_equal(
            dt["refc"]["instant"]["atmosphere"]["time"].values[:],
            np.array([np.datetime64("2022-10-14T01:00:00")]),
        )
        self.assertDictEqual(
            dt["refc"].attrs, dict(name="Maximum/Composite radar reflectivity")
        )

        # Now try the extract and reinflate methods
        self.assertEqual(len(grib_tree_store["refs"]), 55)
        deflated_grib_tree = copy.deepcopy(grib_tree_store)
        kind = extract_datatree_chunk_index(dt, deflated_grib_tree, grib=True)
        strip_datavar_chunks(deflated_grib_tree)
        self.assertEqual(len(deflated_grib_tree["refs"]), 45)

        axes = [
            pd.Index(
                [
                    pd.timedelta_range(
                        start="0 minutes",
                        end="60 minutes",
                        freq="15min",
                        closed="left",
                        name="001 hour",
                    ),
                ],
                name="step",
            ),
            pd.date_range(
                "2022-10-14T00:00:00",
                "2022-10-14T02:00:00",
                freq="15min",
                name="valid_time",
            ),
        ]

        zstore = reinflate_grib_store(
            axes=axes,
            aggregation_type=AggregationType.HORIZON,
            chunk_index=kind,
            zarr_ref_store=deflated_grib_tree,
        )
        # Back to the same number of keys!
        self.assertEqual(len(zstore["refs"]), 55)

        dt = xr.open_datatree(
            "reference://",
            storage_options={"fo": zstore},
            engine="zarr",
            consolidated=False,
        )
        for node in dt.subtree:
            if node.has_data:
                with self.subTest(node_path=node.path):
                    np.testing.assert_array_equal(
                        node.time.values, node.valid_time.values - node.step.values
                    )

                    npath = node.path.lstrip("/")

                    # Assert the values that should be nonnull based on where the chunk is in the time slice
                    expected_null = np.full((1, 9, 1059, 1799), True)
                    expected_null[0, 5, :, :] = False
                    for var in node.data_vars.values():
                        vpath = f"{npath}/{var.name}"
                        with self.subTest(var_name=var.name):
                            key_set = sorted(
                                [
                                    key
                                    for key in zstore["refs"].keys()
                                    if key.startswith(vpath)
                                ]
                            )
                            expected_keys = [
                                f"{vpath}/.zarray",
                                f"{vpath}/.zattrs",
                                f"{vpath}/0.5.0.0",
                            ]
                            self.assertListEqual(key_set, expected_keys)

                            np.testing.assert_array_equal(
                                np.isnan(var.values[:, :, :, :]), expected_null
                            )

    def test_build_idx_grib_mapping(self):
        """
        This test builds a mapping between idx and grib metadata from one runtime and applies it in another, asserting the
        mapped index matches the direct index (fixture from the next test).
        :return:
        """
        grib_uri = os.path.join(THIS_DIR, "grib_idx_fixtures")

        # Define pairs of files with the same horizon but from different runtimes to validate idx mapping
        datasets = {
            "hrrr.wrfsubhf": [
                {
                    "20221014": "hrrr.t01z.wrfsubhf00.grib2.test-limit-10",
                    "20231104": "hrrr.t01z.wrfsubhf00.grib2.test-limit-10",
                },
                {
                    "20221014": "hrrr.t03z.wrfsubhf09.grib2.test-limit-10",
                    "20231104": "hrrr.t01z.wrfsubhf09.grib2.test-limit-10",
                },
            ],
            "gfs.pgrb2.0p25": [
                {
                    "20221014": "gfs.t00z.pgrb2.0p25.f000.test-limit-10",
                    "20231104": "gfs.t00z.pgrb2.0p25.f000.test-limit-10",
                },
                {
                    "20221014": "gfs.t06z.pgrb2.0p25.f021.test-limit-10",
                    "20231104": "gfs.t00z.pgrb2.0p25.f021.test-limit-10",
                },
            ],
            "hrrr.wrfsfcf": [
                {
                    "20221014": "hrrr.t01z.wrfsfcf00.grib2.test-limit-10",
                    "20231104": "hrrr.t01z.wrfsfcf00.grib2.test-limit-10",
                },
                {
                    "20221014": "hrrr.t09z.wrfsfcf05.grib2.test-limit-10",
                    "20231104": "hrrr.t01z.wrfsfcf05.grib2.test-limit-10",
                },
            ],
        }
        for sample_prefix, input_pairs in datasets.items():
            for fnames in input_pairs:
                # First, build a mapping from the 2022 10 14 data
                mapping_fname = fnames["20221014"]
                basename = os.path.join(grib_uri, "20221014", mapping_fname)
                with self.subTest(sample_prefix=sample_prefix):
                    mapping = build_idx_grib_mapping(
                        basename=basename,
                        mapper=correct_hrrr_subhf_step,
                    )

                    mapping = mapping.assign(
                        uri=lambda x: x.uri.str.replace(THIS_DIR, ""),
                        grib_uri=lambda x: x.uri.str.replace(THIS_DIR, ""),
                        idx_uri=lambda x: x.uri.str.replace(THIS_DIR, ""),
                    )

                    # # To update the test fixture
                    # write_path = os.path.join(
                    #     THIS_DIR,
                    #     "grib_idx_fixtures",
                    #     sample_prefix,
                    #     "20221014",
                    #     f"{mapping_fname}.idx_grib_mapping.parquet",
                    # )
                    # mapping.to_parquet(write_path)

                    test_path = os.path.join(
                        THIS_DIR,
                        "grib_idx_fixtures",
                        sample_prefix,
                        "20221014",
                        f"{mapping_fname}.idx_grib_mapping.parquet",
                    )
                    # Build the mapping from idx to cfgrib metadata and assert it matches the fixture
                    expected = pd.read_parquet(test_path, engine="fastparquet")
                    pd.testing.assert_frame_equal(mapping, expected)

                    # parse the idx files for 20231104 and compare the mapped result to the direct indexed result
                    test_name = fnames["20231104"]
                    basename = os.path.join(grib_uri, "20231104", test_name)

                    idxdf = parse_grib_idx(
                        basename=basename,
                    )

                    # Get the runtime hour from the filename as we would in prod
                    matcher = re.compile(r"\w*\.t(?P<hour>\d{2})z\..*")
                    matched = matcher.match(test_name)
                    if not matched:
                        raise ValueError(f"test name {test_name} hour not matched")
                    runtime_hour = matched.groupdict()["hour"]

                    # hard code the runtime date for now and build the index using the
                    # 2022-10-14 mapping for 2023-11-04 idx file!
                    mapped_index = map_from_index(
                        pd.Timestamp(f"2023-11-04T{runtime_hour}"), mapping, idxdf
                    )

                    mapped_index = mapped_index.assign(
                        uri=lambda x: x.uri.str.replace(THIS_DIR, "")
                    )

                    # Read the expected fixture - created by test_kerchunk_indexing
                    kindex_test_path = os.path.join(
                        THIS_DIR,
                        "grib_idx_fixtures",
                        sample_prefix,
                        "20231104",
                        f"{test_name}.kindex.parquet",
                    )
                    expected = pd.read_parquet(kindex_test_path, engine="fastparquet")

                    expected = expected.set_index(
                        ["varname", "typeOfLevel", "stepType", "step", "level"]
                    ).sort_index()
                    mapped_index = mapped_index.set_index(
                        ["varname", "typeOfLevel", "stepType", "step", "level"]
                    ).sort_index()

                    pd.testing.assert_index_equal(expected.index, mapped_index.index)

                    pd.testing.assert_frame_equal(
                        expected[["time", "valid_time"]],
                        mapped_index[["time", "valid_time"]],
                    )

                    # The grib index can build inline values so only compare the offset, length, and uri where
                    # the inline value is not present

                    expected_subset = expected.loc[
                        expected.inline_value.isna(), ["offset", "length", "uri"]
                    ]

                    mapped_index_subset = mapped_index.loc[
                        expected.inline_value.isna(), ["offset", "length", "uri"]
                    ]
                    pd.testing.assert_frame_equal(expected_subset, mapped_index_subset)

    def test_kerchunk_indexing(self):
        """
        This test builds the grib metadata index for a set of forecasts and asserts it has not changed from what is
        recorded in the fixture files
        :return:
        """
        TEST_DATE = "20231104"
        grib_uri = os.path.join(THIS_DIR, "grib_idx_fixtures", TEST_DATE)

        datasets = {
            "hrrr.wrfsubhf": [
                "hrrr.t01z.wrfsubhf00.grib2.test-limit-10",
                "hrrr.t01z.wrfsubhf09.grib2.test-limit-10",
            ],
            "gfs.pgrb2.0p25": [
                "gfs.t00z.pgrb2.0p25.f000.test-limit-10",
                "gfs.t00z.pgrb2.0p25.f021.test-limit-10",
            ],
            "hrrr.wrfsfcf": [
                "hrrr.t01z.wrfsfcf00.grib2.test-limit-10",
                "hrrr.t01z.wrfsfcf05.grib2.test-limit-10",
            ],
        }
        for sample_prefix, fnames in datasets.items():
            for fname in fnames:
                with self.subTest(sample_prefix=sample_prefix, fname=fname):

                    basename = os.path.join(grib_uri, fname)
                    if "hrrr.wrfsubhf" == sample_prefix:
                        grib_tree_store = grib_tree(
                            [
                                correct_hrrr_subhf_step(scan)
                                for scan in scan_grib(basename)
                            ]
                        )
                    else:
                        grib_tree_store = grib_tree(scan_grib(basename))

                    dt = xr.open_datatree(
                        "reference://",
                        storage_options={"fo": grib_tree_store},
                        engine="zarr",
                        consolidated=False,
                    )

                    kindex = extract_datatree_chunk_index(
                        dt, grib_tree_store, grib=True
                    )

                    # Strip the current working directory
                    kindex = kindex.assign(
                        uri=lambda x: x.uri.str.replace(THIS_DIR, "")
                    )

                    # # To update the test fixture
                    # write_path = os.path.join(
                    #     THIS_DIR,
                    #     "grib_idx_fixtures",
                    #     sample_prefix,
                    #     TEST_DATE,
                    #     f"{fname}.kindex.parquet",
                    # )
                    # kindex.to_parquet(write_path)

                    test_path = os.path.join(
                        THIS_DIR,
                        "grib_idx_fixtures",
                        sample_prefix,
                        TEST_DATE,
                        f"{fname}.kindex.parquet",
                    )
                    expected = pd.read_parquet(test_path, engine="fastparquet")
                    pd.testing.assert_frame_equal(kindex, expected)

    def test_index_extraction(self):
        for sample_prefix in ["hrrr.wrfsubhf", "hrrr.wrfsfcf", "gfs.pgrb2.0p25"]:
            with self.subTest(sample_prefix):
                self._test_index_extraction(sample_prefix)

    def _read_sample_prefix(self, sample_prefix: str) -> tuple[xr.DataTree, dict]:
        read_path = os.path.join(
            THIS_DIR, "grib_idx_fixtures", sample_prefix, "subset.json.gz"
        )
        with gzip.open(read_path, "rt") as f:
            scanned_msgs = ujson.load(typing.cast(io.TextIOWrapper, f))

        if "subhf" in sample_prefix:
            scanned_msgs = [correct_hrrr_subhf_step(msg) for msg in scanned_msgs]
        grib_tree_store = grib_tree(scanned_msgs)

        dt = xr.open_datatree(
            "reference://",
            storage_options={"fo": grib_tree_store},
            engine="zarr",
            consolidated=False,
        )
        return dt, grib_tree_store

    def _test_index_extraction(self, sample_prefix: str):
        dt, grib_tree_store = self._read_sample_prefix(sample_prefix)

        k_index = extract_datatree_chunk_index(dt, grib_tree_store, grib=True)

        # # To update the test fixture
        # write_path = os.path.join(
        #     THIS_DIR, "grib_idx_fixtures", sample_prefix, "kerchunk_index.parquet"
        # )
        # k_index.to_parquet(write_path)

        test_path = os.path.join(
            THIS_DIR, "grib_idx_fixtures", sample_prefix, "kerchunk_index.parquet"
        )
        expected = pd.read_parquet(test_path, engine="fastparquet")
        pd.testing.assert_frame_equal(k_index, expected)

    def test_strip_datavar_chunks(self):
        for sample_prefix, pre, post in [
            ("hrrr.wrfsubhf", 104, 65),
            ("hrrr.wrfsfcf", 108, 63),
            ("gfs.pgrb2.0p25", 917, 205),
        ]:
            with self.subTest(sample_prefix=sample_prefix, pre=pre, post=post):
                self._test_strip_datavar_chunks(sample_prefix, pre, post)

    def _test_strip_datavar_chunks(self, sample_prefix: str, pre: int, post: int):
        dt, grib_tree_store = self._read_sample_prefix(sample_prefix)

        self.assertEqual(len(grib_tree_store["refs"]), pre)
        strip_datavar_chunks(grib_tree_store)
        self.assertEqual(len(grib_tree_store["refs"]), post)

        # # To update the test grib_idx_fixtures
        # write_path = os.path.join(
        #     TESTS_DIR, "grib_idx_fixtures", sample_prefix
        # )
        # dynamic_zarr_store.write_store(write_path, grib_tree_store)

        test_path = os.path.join(THIS_DIR, "grib_idx_fixtures", sample_prefix)
        expected = read_store(test_path)
        self.assertDictEqual(grib_tree_store, expected)

    def test_read_write_store(self):
        data = {
            "version": 1,
            "refs": {
                ".zgroup": '{"zarr_format":2}',
                ".zattrs": '{"GRIB_centre":"kwbc","GRIB_centreDescription":"US National Weather Service - NCEP ","GRIB_edition":2,"GRIB_subCentre":0,"coordinates":"heightAboveGround latitude longitude step time valid_time","institution":"US National Weather Service - NCEP "}',
                "u/.zarray": '{"chunks":[1059,1799],"compressor":null,"dtype":"<f8","fill_value":null,"filters":[{"dtype":"float64","id":"grib","var":"u"}],"order":"C","shape":[1059,1799],"zarr_format":2}',
                "u/0.0": ["{{u}}", 3653893, 1088857],
                "u/.zattrs": '{"GRIB_DxInMetres":3000.0,"GRIB_DyInMetres":3000.0,"GRIB_LaDInDegrees":38.5,"GRIB_Latin1InDegrees":38.5,"GRIB_Latin2InDegrees":38.5,"GRIB_LoVInDegrees":262.5,"GRIB_NV":0,"GRIB_Nx":1799,"GRIB_Ny":1059,"GRIB_cfName":"eastward_wind","GRIB_cfVarName":"u","GRIB_dataType":"fc","GRIB_gridDefinitionDescription":"Lambert Conformal can be secant or tangent, conical or bipolar","GRIB_gridType":"lambert","GRIB_iScansNegatively":0,"GRIB_jPointsAreConsecutive":0,"GRIB_jScansPositively":1,"GRIB_latitudeOfFirstGridPointInDegrees":21.138123,"GRIB_latitudeOfSouthernPoleInDegrees":0.0,"GRIB_longitudeOfFirstGridPointInDegrees":237.280472,"GRIB_longitudeOfSouthernPoleInDegrees":0.0,"GRIB_missingValue":3.4028234663852886e+38,"GRIB_name":"U component of wind","GRIB_numberOfPoints":1905141,"GRIB_paramId":131,"GRIB_shortName":"u","GRIB_stepType":"instant","GRIB_stepUnits":1,"GRIB_typeOfLevel":"heightAboveGround","GRIB_units":"m s**-1","_ARRAY_DIMENSIONS":["y","x"],"long_name":"U component of wind","standard_name":"eastward_wind","units":"m s**-1"}',
                "heightAboveGround/.zarray": '{"chunks":[],"compressor":null,"dtype":"<f8","fill_value":null,"filters":null,"order":"C","shape":[],"zarr_format":2}',
                "heightAboveGround/0": "\x00\x00\x00\x00\x00\x00T@",
                "heightAboveGround/.zattrs": '{"_ARRAY_DIMENSIONS":[],"long_name":"height above the surface","positive":"up","standard_name":"height","units":"m"}',
                "latitude/.zarray": '{"chunks":[1059,1799],"compressor":null,"dtype":"<f8","fill_value":null,"filters":[{"dtype":"float64","id":"grib","var":"latitude"}],"order":"C","shape":[1059,1799],"zarr_format":2}',
                "latitude/0.0": ["{{u}}", 3653893, 1088857],
                "latitude/.zattrs": '{"_ARRAY_DIMENSIONS":["y","x"],"long_name":"latitude","standard_name":"latitude","units":"degrees_north"}',
                "longitude/.zarray": '{"chunks":[1059,1799],"compressor":null,"dtype":"<f8","fill_value":null,"filters":[{"dtype":"float64","id":"grib","var":"longitude"}],"order":"C","shape":[1059,1799],"zarr_format":2}',
                "longitude/0.0": ["{{u}}", 3653893, 1088857],
                "longitude/.zattrs": '{"_ARRAY_DIMENSIONS":["y","x"],"long_name":"longitude","standard_name":"longitude","units":"degrees_east"}',
                "step/.zarray": '{"chunks":[],"compressor":null,"dtype":"<f8","fill_value":null,"filters":null,"order":"C","shape":[],"zarr_format":2}',
                "step/0": "\x00\x00\x00\x00\x00\x00\x00\x00",
                "step/.zattrs": '{"_ARRAY_DIMENSIONS":[],"long_name":"time since forecast_reference_time","standard_name":"forecast_period","units":"hours"}',
                "time/.zarray": '{"chunks":[],"compressor":null,"dtype":"<i8","fill_value":null,"filters":null,"order":"C","shape":[],"zarr_format":2}',
                "time/0": "base64:ENAUZQAAAAA=",
                "time/.zattrs": '{"_ARRAY_DIMENSIONS":[],"calendar":"proleptic_gregorian","long_name":"initial time of forecast","standard_name":"forecast_reference_time","units":"seconds since 1970-01-01T00:00:00"}',
                "valid_time/.zarray": '{"chunks":[],"compressor":null,"dtype":"<i8","fill_value":null,"filters":null,"order":"C","shape":[],"zarr_format":2}',
                "valid_time/0": "base64:ENAUZQAAAAA=",
                "valid_time/.zattrs": '{"_ARRAY_DIMENSIONS":[],"calendar":"proleptic_gregorian","long_name":"time","standard_name":"time","units":"seconds since 1970-01-01T00:00:00"}',
            },
            "templates": {"u": "testdata/hrrr.t01z.wrfsubhf00.grib2"},
        }
        with tempfile.TemporaryDirectory(suffix=".test") as ntd:
            write_store(ntd, data)

            result = read_store(ntd)
        self.assertDictEqual(data, result)

    def _reinflate_grib_store_dataset(self):
        datasets = [
            "hrrr.wrfsfcf",
            "gfs.pgrb2.0p25",
            "hrrr.wrfsubhf",
        ]
        for name in datasets:
            yield name

    def _reinflate_grib_store_aggregation(self):
        # Provide some general axes - this will subset the dataset for the selected times
        aggregations = {
            AggregationType.HORIZON: [
                pd.Index(
                    [
                        pd.timedelta_range(
                            start="0 minutes",
                            end="60 minutes",
                            freq="60min",
                            closed="left",
                            name="000 hour",
                        ),
                        pd.timedelta_range(
                            start="60 minutes",
                            end="120 minutes",
                            freq="60min",
                            closed="left",
                            name="001 hour",
                        ),
                    ],
                    name="step",
                ),
                pd.date_range(
                    "2023-09-28T00:00",
                    "2023-09-28T03:00",
                    freq="60min",
                    name="valid_time",
                ),
            ],
            AggregationType.VALID_TIME: [
                pd.timedelta_range("0 min", "300 min", freq="60 min", name="step"),
                pd.DatetimeIndex(
                    ["2023-09-28T02:00", "2023-09-28T04:00"], name="valid_time"
                ),
            ],
            AggregationType.RUN_TIME: [
                pd.timedelta_range("0 min", "120 min", freq="15 min", name="step"),
                pd.DatetimeIndex(["2023-09-28T00:00", "2023-09-28T02:00"], name="time"),
            ],
            AggregationType.BEST_AVAILABLE: [
                pd.date_range(
                    "2023-09-28T00:00",
                    "2023-09-28T10:00",
                    freq="60min",
                    name="valid_time",
                ),
                pd.DatetimeIndex(["2023-09-28T02:00"], name="time"),
            ],
        }

        for aggregation, axes in aggregations.items():
            yield aggregation, axes

    def _reinflate_grib_store(
        self,
        dataset: str,
        aggregation: AggregationType,
        axes: list[pd.Index],
    ):
        kind = pd.read_parquet(
            os.path.join(
                THIS_DIR, "grib_idx_fixtures", dataset, "test_reinflate.parquet"
            ),
            engine="fastparquet",
        )

        zstore = reinflate_grib_store(
            axes=axes,
            aggregation_type=aggregation,
            chunk_index=kind,
            zarr_ref_store=read_store(
                os.path.join(THIS_DIR, "grib_idx_fixtures", dataset)
            ),
        )
        dt = xr.open_datatree(refs_as_store(zstore), engine="zarr")
        for node in dt.subtree:
            if not node.has_data:
                continue
            with self.subTest(node_path=node.path):

                match aggregation:
                    case AggregationType.HORIZON | AggregationType.BEST_AVAILABLE:

                        self.assertEqual(node.time.dims, node.valid_time.dims)
                        self.assertEqual(node.time.dims, node.step.dims)
                        np.testing.assert_array_equal(
                            node.time.values, node.valid_time.values - node.step.values
                        )
                    case AggregationType.VALID_TIME:
                        steps2d = np.tile(
                            node.step.values, (node.valid_time.shape[0], 1)
                        )
                        valid_times2d = np.tile(
                            np.reshape(node.valid_time.values, (-1, 1)),
                            (1, node.step.shape[0]),
                        )

                        np.testing.assert_array_equal(
                            node.time.values, valid_times2d - steps2d
                        )
                    case AggregationType.RUN_TIME:
                        steps2d = np.tile(node.step.values, (node.time.shape[0], 1))
                        times2d = np.tile(
                            np.reshape(node.time.values, (-1, 1)),
                            (1, node.step.shape[0]),
                        )
                        np.testing.assert_array_equal(
                            times2d, node.valid_time.values - steps2d
                        )
                    case _:
                        raise RuntimeError("uhoh - unknown aggregation!")

                npath = node.path.lstrip("/")
                # Can't read nonlocal data here, but we can assert the correct keys are present
                for var in node.data_vars.values():
                    vpath = f"{npath}/{var.name}"
                    with self.subTest(var_name=var.name):
                        key_set = sorted(
                            [
                                key
                                for key in zstore["refs"].keys()
                                if key.startswith(vpath)
                            ]
                        )

                        # # To update test grib_idx_fixtures
                        # write_path = os.path.join(THIS_DIR, "grib_idx_fixtures", dataset, "reinflate", aggregation.value, f"{vpath}_chunks.json")
                        # with fsspec.open(write_path, "w",) as f:
                        #     f.write(ujson.dumps(key_set, indent=2))

                        test_path = os.path.join(
                            THIS_DIR,
                            "grib_idx_fixtures",
                            dataset,
                            "reinflate",
                            aggregation.value,
                            f"{vpath}_chunks.json",
                        )
                        with fsspec.open(test_path, "r") as f:
                            expected_keys = ujson.loads(f.read())

                        self.assertListEqual(key_set, expected_keys)

    def test_reinflate_grib_store(self):
        for dataset in self._reinflate_grib_store_dataset():
            for aggregation, axes in self._reinflate_grib_store_aggregation():
                with self.subTest(dataset=dataset, aggregation=aggregation):
                    self._reinflate_grib_store(dataset, aggregation, axes)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
