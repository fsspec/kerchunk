import io
import numpy as np
import fsspec

import cfgrib
import eccodes
import numcodecs.abc
import imagecodecs.numcodecs
# imagecodecs.numcodecs.register_codecs()


# https://old.wmo.int/extranet/pages/prog/www/DPS/FM92-GRIB2-11-2003.pdf
def go(fn, storage_options=None):
    with fsspec.open(fn, "rb", **(storage_options or {})) as f:
        while True:  # loop over messages

            msg_start = f.tell()
            indicator = f.read(16)
            if len(indicator) < 16:
                print("EOF")
                return
            assert indicator[:4] == b"GRIB"
            assert indicator[7] == 2  # GRIB2
            size = int.from_bytes(indicator[8:], "big")
            data = indicator + f.read(size - 16)
            # read same buffer with eccodes to find the shape, attributes
            # and coordinates
            mid = eccodes.codes_new_from_message(data)
            m = cfgrib.cfmessage.CfMessage(mid)

            buffer = io.BytesIO(data)
            buffer.seek(16)
            bitmap_start = None
            while True:  # loop over sections
                section_start = buffer.tell()
                header = buffer.read(4)
                if header == b"7777":
                    print("END")
                    break
                length = int.from_bytes(header, "big")
                section = buffer.read(1)[0]
                buffer.seek(section_start)
                rest = buffer.read(length)
                print(section_start, length, section)
                if section == 5:
                    num_points = int.from_bytes(rest[5:9], "big")
                    template_number = int.from_bytes(rest[9:11], "big")
                    R = np.frombuffer(rest[11:15], dtype=">f4")[0]
                    E = int.from_bytes(rest[15:17], "big")
                    D = int.from_bytes(rest[17:19], "big")
                    bitpix = rest[19]
                    final_dtype = ["float32", "int32"][rest[20]]

                    print("Representation", template_number, R, E, D)
                    # Y * 10**D = R + (X1 + X2) * 2**E
                    # Y is decoded value, X2 is encoded value
                    # X1 is local reference value (complex packing) or 0
                    scale = {
                        "id": "fixedscaleoffset",
                        "offset": R and R/10**D,
                        "scale": 10**D / 2**E,
                        "astype": "uint8",
                        "dtype": final_dtype
                    }
                    if template_number == 40:
                        encoding = imagecodecs.numcodecs.Jpeg2k()
                    else:
                        encoding = None

                elif section == 6:
                    if rest[5]:
                        bitmap_start = section_start + msg_start + 6
                        bitmap_length = len(rest) - 6
                elif section == 7:
                    data_start = section_start + msg_start + 5
                    data_length = length - 5
                    if bitmap_start:
                        ref = [fn, bitmap_start, bitmap_length + data_length + 5]
                        compression = BitmapPlus(1, [scale, encoding])
                        filters = None
                    else:
                        ref = [fn, data_start, data_length]
                        compression = encoding
                        filters = [scale]


class BitPack(numcodecs.abc.Codec):
    codec_id = "bitpack"  # actually, this is a compressor

    def __init__(self, bitwidth):
        self.bitwidth = bitwidth

    def encode(self, buf):
        raise NotImplemented  # but not too hard!

    def decode(self, buf, out=None):
        if out is not None:
            raise NotImplemented
        lout = (len(buf) * 8) // self.bitwidth
        if (len(buf) * 8) % self.bitwidth:
            lout += 8
        if self.bitwidth < 9:
            dt = "uint8"
        elif self.bitwidth < 17:
            dt = "uint16"
        else:
            dt = "uint32"
        out = np.empty(lout, dt)
        mask = (2**32 - 1) >> (32 - self.bitwidth)

        big = np.sum([buf[i::self.bitwidth] for i in range(self.bitwidth)], axis=0,
                     dtype="int32")
        for i in range(8):
            out[i::8] = (big >> (i * self.bitwidth)) & mask
        return out


class BitmapPlus(numcodecs.abc.Codec):
    """Apply bitmap mask after decoding data with other codecs"""

    codec_id = "bitmap_plus"

    def __init__(self, npoints, encoding_args):
        self.npoints = npoints
        self.encoding_args = encoding_args  # list of dicts

    def encode(self, buf):
        raise NotImplemented

    def decode(self, buf, out=None):
        nbytes = self.npoints // 8
        if self.npoints % 8:
            nbytes += 1
        mask_bytes = np.frombuffer(buf[:nbytes], "uint8")
        mask = np.unpackbits(mask_bytes)

        data = buf[nbytes + 5:]  # 5 bytes for the data section header
        for args in self.encoding_args:
            data = numcodecs.registry.get_codec(args).decode(data)

        data[mask] = np.nan  # could make np masked array
        return data
