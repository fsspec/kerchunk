# Fixture data

To examine json.gz fixture files on the command line use *zcat* & *jq*
```console
zcat tests/fixtures/hrrr.wrfsubhf/zarr_tree_store_v1.json.gz | jq .
```
To examine the parquet files you need to use jupyter notebook or a debugger


HRRR Sub hourly test data
```
gs://high-resolution-rapid-refresh/hrrr.20210928/conus/hrrr.t01z.wrfsubhf00.grib2
gs://high-resolution-rapid-refresh/hrrr.20210928/conus/hrrr.t01z.wrfsubhf01.grib2
```

HRRR Surface 2d test data
```
gs://high-resolution-rapid-refresh/hrrr.20210928/conus/hrrr.t01z.wrfsfcf00.grib2
gs://high-resolution-rapid-refresh/hrrr.20210928/conus/hrrr.t01z.wrfsfcf01.grib2
```

GFS pgrb2 0p25 test data
```
gs://global-forecast-system/gfs.20230928/00/atmos/gfs.t00z.pgrb2.0p25.f000
gs://global-forecast-system/gfs.20230928/00/atmos/gfs.t00z.pgrb2.0p25.f001
gs://global-forecast-system/gfs.20230928/00/atmos/gfs.t00z.pgrb2.0p25.f002
gs://global-forecast-system/gfs.20230928/06/atmos/gfs.t06z.pgrb2.0p25.f000
gs://global-forecast-system/gfs.20230928/06/atmos/gfs.t06z.pgrb2.0p25.f001
gs://global-forecast-system/gfs.20230928/06/atmos/gfs.t06z.pgrb2.0p25.f002
```

## To make more fixture data

Copy data with a command like:
```console
gsutil -m cp gs://high-resolution-rapid-refresh/hrrr.20230928/conus/hrrr.t00z.wrfsfcf* testdata/.
```

Call scan_grib to read the files and filter the results message groups for interesting keys e.g. starting with "dswrf" or "u". Even a single grib file will be huge for all messages.

```python
scans = scan_grib("testdata/hrrr.t01z.wrfsubhf00.grib2")
scans += scan_grib("testdata/hrrr.t01z.wrfsubhf01.grib2")
jsets = []
vname = "dswrf"
for gg in scans:
    if "dswrf/.zattrs" in gg['refs'] or "u/.zattrs" in gg['refs']:
        jsets.append(gg)

with gzip.open("ingestion/noaa_nwp/tests/fixtures/hrrr.wrfsubhf.subset.json.gz", 'w') as f:
    ss = ujson.dumps(jsets)
    f.write(ss)
```

To make reinflate test parquet chunk indexes... load data from ingestion extract a single parquet file
```python
gfs_base_path = "gs://dev.camus-infra.camus.store/davetest/gfs"
gfs_kind = dd.read_parquet(
    [f.full_name for f in fsspec.open_files(os.path.join(gfs_base_path, "data_index/**.parquet"))],
    index=False
).compute()
gfs_kind.loc[
    gfs_kind.varname.isin(["u", "dswrf"]) &
    (gfs_kind.valid_time	<= "2023-09-28 04:00:00")
].to_parquet("/home/builder/bando/ingestion/noaa_nwp/tests/fixtures/gfs.pgrb2.0p25/test_reinflate.parquet")
```


Make truncated grib files and idx files with make_test_grib_idx_files
```python
fs = fsspec.filesystem("gcs")
dynamic_zarr_store.make_test_grib_idx_files(
fs=fs,
basename="gs://camus-infra.camus.store/circleci_test_data/bando/ingestion/noaa_nwp/tests/fixtures/20221014/hrrr.t01z.wrfsubhf00.grib2"
)
```
Copy the files from the NODD bucket first, then truncate, so we keep the original artifact too.
e.g. `gsutil cp gs://global-forecast-system/gfs.20221014/00/atmos/gfs.t00z.pgrb2.0p25.f000* gs://camus-infra.camus.store/circleci_test_data/bando/ingestion/noaa_nwp/tests/fixtures/20221014/`
