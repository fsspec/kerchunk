import fsspec
import numpy as np
import ujson

from kerchunk.utils import refs_as_store


decoders = {}


def reg(name):
    def f(func):
        decoders[name] = func
        return func

    return f


class HDF4ToZarr:
    """Experimental: interface to HDF4 archival files"""

    def __init__(
        self,
        path,
        storage_options=None,
        inline_threshold=100,
        out=None,
    ):
        self.path = path
        self.st = storage_options
        self.thresh = inline_threshold
        self.out = out or {}

    def read_int(self, n):
        return int.from_bytes(self.f.read(n), "big")

    def read_ddh(self):
        return {"ndd": self.read_int(2), "next": self.read_int(4)}

    def read_dd(self):
        loc = self.f.tell()
        i = int.from_bytes(self.f.read(2), "big")
        if i & 0x4000:
            extended = True
            i = i - 0x4000
        else:
            extended = False
        tag = tags.get(i, i)
        no_data = tag not in {"NULL"}
        ref = (tag, int.from_bytes(self.f.read(2), "big"))
        info = {
            "offset": int.from_bytes(self.f.read(4), "big") * no_data,
            "length": int.from_bytes(self.f.read(4), "big") * no_data,
            "extended": extended,
            "loc": loc,
        }
        return ref, info

    def decode(self, tag, info):
        self.f.seek(info["offset"])
        ident = lambda _, __: info
        return decoders.get(tag, ident)(self, info)

    def translate(self, filename=None, storage_options=None):
        """Scan and return references

        Parameters
        ----------
        filename: if given, write to this as JSON
        storage_options: to interpret filename

        Returns
        -------
        references
        """
        import zarr
        from kerchunk.codecs import ZlibCodec

        fo = fsspec.open(self.path, **(self.st or {}))
        self.f = fo.open()

        # magic header
        assert self.f.read(4) == b"\x0e\x03\x13\x01"

        # all the data descriptors in a linked list
        self.tags = {}
        while True:
            ddh = self.read_ddh()

            for _ in range(ddh["ndd"]):
                ident, info = self.read_dd()
                self.tags[ident] = info
            if ddh["next"] == 0:
                # "finished" sentry
                break
            # or continue
            self.f.seek(ddh["next"])

        # basic decode
        for tag, ref in self.tags:
            self._dec(tag, ref)

        # global attributes
        attrs = {}
        for (tag, ref), info in self.tags.items():
            if tag == "VH" and info["names"][0].upper() == "VALUES":
                # dtype = dtypes[info["types"][0]]
                inf2 = self.tags[("VS", ref)]
                self.f.seek(inf2["offset"])
                # remove zero padding
                data = self.f.read(inf2["length"]).split(b"\x00", 1)[0]
                # NASA conventions
                if info["name"].startswith(
                    ("CoreMetadata.", "ArchiveMetadata.", "StructMetadata.")
                ):
                    obj = None
                    for line in data.decode().split("\n"):
                        if "OBJECT" in line:
                            obj = line.split()[-1]
                        if "VALUE" in line:
                            attrs[obj] = line.split()[-1].lstrip('"').rstrip('"')

        # there should be only one root, and it's probably the last VG
        # so maybe this loop isn't needed
        roots = set()
        children = set()
        child = {}
        for (tag, ref), info in self.tags.items():
            if tag == "VG":
                here = child.setdefault((tag, ref), set())
                for t, r in zip(info["tag"], info["refs"]):
                    if t == "VG":
                        children.add((t, r))
                        roots.discard((t, r))
                        here.add((t, r))
                if tag not in children:
                    roots.add((tag, ref))

        # hierarchical output
        output = self._descend_vg(*sorted(roots, key=lambda t: t[1])[-1])
        prot = fo.fs.protocol
        prot = prot[0] if isinstance(prot, tuple) else prot
        store = refs_as_store(self.out, remote_protocol=prot, remote_options=self.st)
        g = zarr.open_group(store, zarr_format=2, use_consolidated=False)
        refs = {}
        for k, v in output.items():
            if isinstance(v, dict):
                compressor = ZlibCodec() if "refs" in v else None
                arr = g.require_array(
                    name=k,
                    shape=v["dims"],
                    dtype=v["dtype"],
                    chunks=v.get("chunks", v["dims"]),
                    compressor=compressor,
                )
                arr.attrs.update(
                    dict(
                        _ARRAY_DIMENSIONS=(
                            [f"{k}_x", f"{k}_y"][: len(v["dims"])]
                            if "refs" in v
                            else ["0"]
                        ),
                        **{
                            i: j.tolist() if isinstance(j, np.generic) else j
                            for i, j in v.items()
                            if i not in {"chunk", "dims", "dtype", "refs"}
                        },
                    )
                )
                for r in v.get("refs", []):
                    if r[0] == "DEFLATE":
                        continue
                    refs[f"{k}/{r[0]}"] = [self.path, r[1], r[2]]
            else:
                if not k.startswith(
                    ("CoreMetadata.", "ArchiveMetadata.", "StructMetadata.")
                ):
                    attrs[k] = v.tolist() if isinstance(v, np.generic) else v
        store.fs.references.update(refs)
        g.attrs.update(attrs)

        if filename is None:
            return store.fs.references
        with fsspec.open(filename, **(storage_options or {})) as f:
            ujson.dumps(dict(store.fs.references), f)

    def _descend_vg(self, tag, ref):
        info = self.tags[(tag, ref)]
        out = {}
        for t, r in zip(info["tag"], info["refs"]):
            inf2 = self.tags[(t, r)]
            if t == "VG":
                tmp = self._descend_vg(t, r)
                if tmp and list(tmp)[0] == inf2["name"]:
                    tmp = tmp[inf2["name"]]
                out[inf2["name"]] = tmp
            elif t == "VH":
                if len(inf2["names"]) == 1 and inf2["names"][0].lower() == "values":
                    dtype = dtypes[inf2["types"][0]]
                    name = inf2["name"]
                    inf2 = self.tags[("VS", r)]
                    self.f.seek(inf2["offset"])
                    data = self.f.read(inf2["length"])
                    if dtype == "str":
                        out[name] = (
                            data.split(b"\x00", 1)[0].decode().lstrip('"').rstrip('"')
                        )  # decode() ?
                    else:
                        out[name] = np.frombuffer(data, dtype)[0]
            elif t == "NT":
                out["dtype"] = inf2["typ"]
            elif t == "SD":
                if isinstance(inf2["data"][-1], (tuple, list)):
                    out["refs"] = inf2["data"][:-1]
                    out["chunks"] = [_["chunk_length"] for _ in inf2["data"][-1]]
                else:
                    out["refs"] = [inf2["data"]]
                    out["chunks"] = True
            elif t == "SDD":
                out["dims"] = inf2["dims"]
            elif t == "NDG":
                pass  # out.setdefault("extra", []).append(_dec_ndg(self, inf2))
        if out.get("chunks") is True:
            out["chunks"] = out["dims"]
            out["refs"] = [
                [".".join(["0"] * len(out["dims"]))]
                + [out["refs"][0][1], out["refs"][0][2], out["refs"][0][0]]
            ]
        return out

    def _dec(self, tag, ref):
        info = self.tags[(tag, ref)]
        if not set(info) - {"length", "offset", "extended", "loc"}:
            self.f.seek(info["offset"])
            if info["extended"]:
                info["data"] = self._dec_extended()
            else:
                info.update(self.decode(tag, info))
        return info

    def _dec_extended(self):
        ext_type = spec[self.read_int(2)]
        if ext_type == "CHUNKED":
            return self._dec_chunked()
        elif ext_type == "LINKED":
            return self._dec_linked_header()
        elif ext_type == "COMP":
            return self._dec_comp()

    def _dec_linked_header(self):
        # get the bytes of a linked set - these will always be inlined
        self.read_int(4)  # length
        self.read_int(4)  # blk_len
        self.read_int(4)  # num_blk
        next_ref = self.read_int(2)
        out = []
        while next_ref:
            next_ref, data = self._dec_linked_block(self.tags[("LINKED", next_ref)])
            out.extend([d for d in data if d])
        bits = []
        for ref in out:
            info = self.tags[("LINKED", ref)]
            self.f.seek(info["offset"])
            bits.append(self.f.read(info["length"]))
        return b"".join(bits)

    def _dec_linked_block(self, block):
        self.f.seek(block["offset"])
        next_ref = self.read_int(2)
        refs = [self.read_int(2) for _ in range((block["length"] // 2) - 1)]
        return next_ref, refs

    def _dec_chunked(self):
        # we want to turn the chunks table into references
        # tag_head_len = self.read_int(4)
        # version = self.f.read(1)[0]
        # flag = self.read_int(4)
        # elem_tot_len = self.read_int(4)
        # chunk_size = self.read_int(4)
        # nt_size = self.read_int(4)
        self.f.seek(21, 1)
        chk_tbl_tag = tags[self.read_int(2)]  # should be VH
        chk_tbl_ref = self.read_int(2)
        self.read_int(2)  # sp_tab = tags[self.read_int(2)]
        self.read_int(2)  # sp_ref
        ndims = self.read_int(4)

        dims = [  # we don't use these, could skip
            {
                "flag": self.read_int(4),
                "dim_length": self.read_int(4),
                "chunk_length": self.read_int(4),
            }
            for _ in range(ndims)
        ]
        self.f.read(  # fill_value
            self.read_int(4)
        )  # to be interpreted as a number later; but chunk table probs has no fill
        # self.f.seek(12*ndims + 4, 1)  # if skipping

        header = self._dec(chk_tbl_tag, chk_tbl_ref)
        data = self._dec("VS", chk_tbl_ref)["data"]  # corresponding table

        # header gives the field pattern for the rows of data, one per chunk
        # maybe faster to use struct and iter than numpy, since we iterate anyway
        dt = [(f"ind{i}", ">u4") for i in range(ndims)] + [
            ("tag", ">u2"),
            ("ref", ">u2"),
        ]
        rows = np.frombuffer(data, dtype=dt, count=header["nvert"])
        # rows["tag"] should always be 61 -> CHUNK
        refs = []
        for *ind, tag, ref in rows:
            # maybe ind needs reversing since everything is FORTRAN
            chunk_tag = self.tags[("CHUNK", ref)]
            if chunk_tag["extended"]:
                self.f.seek(chunk_tag["offset"])
                # these are always COMP?
                ctype, offset, length = self._dec_extended()
                refs.append([".".join(str(_) for _ in ind), offset, length, ctype])
            else:
                refs.append(
                    [
                        ".".join(str(_) for _ in ind),
                        chunk_tag["offset"],
                        chunk_tag["length"],
                    ]
                )
        refs.append(dims)
        return refs

    def _dec_comp(self):
        # version = self.read_int(2)  # always 0
        # len_uncomp = self.read_int(4)
        self.f.seek(6, 1)

        data_ref = self.read_int(2)
        # model = self.read_int(2)  # always 0
        ctype = "DEFLATE"  # comp[self.read_int(2)]
        tag = self.tags[("COMPRESSED", data_ref)]
        return ctype, tag["offset"], tag["length"]


@reg("NDG")
def _dec_ndg(self, info):
    if "tags" not in info:
        return {
            "tags": [
                (tags[self.read_int(2)], self.read_int(2))
                for _ in range(0, info["length"], 4)
            ]
        }
    return info["tags"]


@reg("SDD")
def _dec_sdd(self, info):
    rank = self.read_int(2)
    dims = [self.read_int(4) for _ in range(rank)]
    data_tag = (tags[self.read_int(2)], self.read_int(2))
    scale_tags = [(tags[self.read_int(2)], self.read_int(2)) for _ in range(rank)]
    return _pl(locals())


@reg("VERSION")
def _dec_version(self, info):
    return {
        "major": self.read_int(4),
        "minor": self.read_int(4),
        "release": self.read_int(4),
        "string:": _null_str(self.f.read(info["length"] - 10).decode()),
    }


@reg("VH")
def _dec_vh(self, info):
    # virtual group ("table") header
    interface = self.read_int(2)
    nvert = self.read_int(4)
    ivsize = self.read_int(2)
    nfields = self.read_int(2)
    types = [self.read_int(2) for _ in range(nfields)]
    isize = [self.read_int(2) for _ in range(nfields)]
    offsets = [self.read_int(2) for _ in range(nfields)]
    order = [self.read_int(2) for _ in range(nfields)]
    names = [
        self.f.read(self.read_int(2)).split(b"\x00", 1)[0].decode()
        for _ in range(nfields)
    ]
    namelen = self.read_int(2)
    name = self.f.read(namelen).split(b"\x00", 1)[0].decode()
    classlen = self.read_int(2)
    cls = self.f.read(classlen).split(b"\x00", 1)[0].decode()
    ref = (self.read_int(2), self.read_int(2))
    return _pl(locals())


@reg("VG")
def _dec_vg(self, info):
    nelt = self.read_int(2)
    tag = [tags[self.read_int(2)] for _ in range(nelt)]
    refs = [self.read_int(2) for _ in range(nelt)]
    name = self.f.read(self.read_int(2)).split(b"\x00", 1)[0].decode()
    cls = self.f.read(self.read_int(2)).split(b"\x00", 1)[0].decode()
    return _pl(locals())


@reg("NT")
def _dec_nt(self, info):
    version, typ, width, cls = list(self.f.read(4))
    typ = dtypes[typ]
    return _pl(locals())


def _null_str(s):
    return s.split("\00", 1)[0]


def _pl(l):
    return {k: v for k, v in l.items() if k not in {"info", "f", "self"}}


# hdf/src/htags.h
tags = {
    1: "NULL",
    20: "LINKED",
    30: "VERSION",
    40: "COMPRESSED",
    50: "VLINKED",
    51: "VLINKED_DATA",
    60: "CHUNKED",
    61: "CHUNK",
    100: "FID",
    101: "FD",
    102: "TID",
    103: "TD",
    104: "DIL",
    105: "DIA",
    106: "NT",
    107: "MT",
    108: "FREE",
    200: "ID8",
    201: "IP8",
    202: "RI8",
    203: "CI8",
    204: "II8",
    300: "ID",
    301: "LUT",
    302: "RI",
    303: "CI",
    304: "NRI",
    306: "RIG",
    307: "LD",
    308: "MD",
    309: "MA",
    310: "CCN",
    311: "CFM",
    312: "AR",
    400: "DRAW",
    401: "RUN",
    500: "XYP",
    501: "MTO",
    602: "T14",
    603: "T105",
    700: "SDG",
    701: "SDD",
    702: "SD",
    703: "SDS",
    704: "SDL",
    705: "SDU",
    706: "SDF",
    707: "SDM",
    708: "SDC",
    709: "SDT",
    710: "SDLNK",
    720: "NDG",
    721: "RESERVED",
    # "Objects of tag 721 are never actually written to the file. The tag is
    # needed to make things easier mixing DFSD and SD style objects in the same file"
    731: "CAL",
    732: "FV",
    799: "BREQ",
    781: "SDRAG",
    780: "EREQ",
    1965: "VG",
    1962: "VH",
    1963: "VS",
    11: "RLE",
    12: "IMCOMP",
    13: "JPEG",
    14: "GREYJPEG",
    15: "JPEG5",
    16: "GREYJPEG5",
}
spec = {
    1: "LINKED",
    2: "EXT",
    3: "COMP",
    4: "VLINKED",
    5: "CHUNKED",
    6: "BUFFERED",
    7: "COMPRAS",
}

# hdf4/hdf/src/hntdefs.h
dtypes = {
    5: "f4",
    6: "f8",
    20: "i1",
    21: "u1",
    4: "str",  # special case, size given in header
    22: ">i2",
    23: ">u2",
    24: ">i4",
    25: ">u4",
    26: ">i8",
    27: ">u8",
}

# hdf4/hdf/src/hcomp.h
comp = {
    0: "NONE",
    1: "RLE",
    2: "NBIT",
    3: "SKPHUFF",
    4: "DEFLATE",  # called deflate, but code says "gzip" and doc says "GNU zip"; actually zlib?
    # see codecs.ZlibCodec
    5: "SZIP",
    7: "JPEG",
}
