
def test_read_parquet_td(tmp_path):
    import pandas as pd
    import numpy as np


    df = pd.DataFrame(
        data=dict(
            ints=np.linspace(1,25, num=25, dtype="int64"),
            floats=np.linspace(1,25, num=25, dtype="float64"),
            dates=pd.date_range("2024-01-01T00", "2024-01-02T00", freq="h"),
            deltas=pd.timedelta_range(start="1 h", end="25 h", freq="h"),
        )
    )

    fpath = tmp_path / "fixture.parquet"
    df.to_parquet(fpath)

    result = pd.read_parquet(fpath)

    pd.testing.assert_frame_equal(result, df)




