import fsspec
import kerchunk.utils


def test_rename():

    old = {"version": 1, "refs": {"v0": ["oldpath", 0, 0], "bin": "data"}}
    new = kerchunk.utils.rename_target(old, {"oldpath": "newpath"})
    assert new == {"version": 1, "refs": {"v0": ["newpath", 0, 0], "bin": "data"}}


def test_rename_files(m):
    m.pipe(
        "ref.json", b'{"version": 1, "refs": {"v0": ["oldpath", 0, 0], "bin": "data"}}'
    )
    kerchunk.utils.rename_target_files("memory://ref.json", {"oldpath": "newpath"})
    out = m.cat("ref.json")
    assert out == b'{"version":1,"refs":{"v0":["newpath",0,0],"bin":"data"}}'
    kerchunk.utils.rename_target_files(
        "memory://ref.json", {"newpath": "newerpath"}, url_out="memory://ref2.json"
    )
    out = m.cat("ref.json")
    assert out == b'{"version":1,"refs":{"v0":["newpath",0,0],"bin":"data"}}'
    out = m.cat("ref2.json")
    assert out == b'{"version":1,"refs":{"v0":["newerpath",0,0],"bin":"data"}}'


def test_inline(m):
    m.pipe("data", b"stuff")
    refs = {
        "key0": b"00",
        "key1": ["memory://data"],
        "key2": ["memory://data", 1, 1],
        "key1": ["memory://data", 2, 4],
    }
    out = kerchunk.utils._do_inline(refs, 2)
    expected = {
        "key0": b"00",
        "key1": ["memory://data"],
        "key2": b"t",
        "key1": ["memory://data", 2, 4],
    }
    assert out == expected
