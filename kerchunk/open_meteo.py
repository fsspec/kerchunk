import base64
import datetime
import io
import json
import logging
import math
import os
import pathlib
from dataclasses import dataclass
from enum import Enum

import fsspec.core
import numcodecs
import numcodecs.abc
import numcodecs.compat
import numpy as np

from .utils import (
    _encode_for_JSON,
    dict_to_store,
    translate_refs_serializable,
)

try:
    import omfiles
except ModuleNotFoundError:  # pragma: no cover
    raise ImportError(
        "omfiles is required for kerchunking Open-Meteo files. Please install with "
        "`pip install omfiles`."
    )

logger = logging.getLogger("kerchunk.open_meteo")

class SupportedDomain(Enum):
    dwd_icon = "dwd_icon"

    # Potentially all this information should be available in the
    # meta.json file under /data/{domain}/{static}/meta.json
    def s3_chunk_size(self) -> int:
        """Defines how many timesteps in native model-dt are in the om-files on AWS S3"""
        if self == SupportedDomain.dwd_icon:
            return 253
        else:
            raise ValueError(f"Unsupported domain {self}")


@dataclass
class StartLengthStepSize:
    start: datetime.datetime
    length: int
    step_dt: int # in seconds


def chunk_number_to_start_time(domain: SupportedDomain, chunk_no: int) -> StartLengthStepSize:
    meta_file = f"openmeteo/data/{domain.value}/static/meta.json"
    # Load metadata from S3
    fs = fsspec.filesystem(protocol="s3", anon=True)
    with fs.open(meta_file, mode="r") as f:
        metadata = json.load(f)

    dt_seconds = metadata["temporal_resolution_seconds"]
    om_file_length = domain.s3_chunk_size()

    seconds_since_epoch = chunk_no * om_file_length * dt_seconds

    epoch = datetime.datetime(1970, 1, 1)
    chunk_start = epoch + datetime.timedelta(seconds=seconds_since_epoch)
    print("chunk_start", chunk_start)
    return StartLengthStepSize(start=chunk_start, length=om_file_length, step_dt=dt_seconds)


class Reshape(numcodecs.abc.Codec):
    """Codec to reshape data between encoding and decoding.

    This codec reshapes the data to a specific shape before passing it to the next
    filter in the pipeline, which is particularly useful for filters like Delta2D
    that expect 2D data.

    Parameters
    ----------
    shape : tuple
        Shape to reshape the data to during decoding
    """

    codec_id = 'reshape'

    def __init__(self, shape):
        self.shape = tuple(shape)

    def encode(self, buf):
        # For encoding, we flatten back to 1D
        arr = numcodecs.compat.ensure_ndarray(buf)
        return arr.reshape(-1)

    def decode(self, buf, out=None):
        print(f"Reshape decode to {self.shape}")
        # Reshape to the specified 2D shape for delta2d
        arr = numcodecs.compat.ensure_ndarray(buf)

        # Check if total elements match
        expected_size = np.prod(self.shape)
        if arr.size != expected_size:
            raise ValueError(f"Buffer size {arr.size} doesn't match expected size {expected_size}")

        # Reshape
        arr = arr.reshape(self.shape)
        print("arr.shape", arr.shape)
        return arr

    def get_config(self):
        return {'id': self.codec_id, 'shape': self.shape}

    def __repr__(self):
        return f'{type(self).__name__}(shape={self.shape!r})'

class Delta2D(numcodecs.abc.Codec):
    """Codec to encode data as the difference between adjacent rows in a 2D array."""

    codec_id = 'delta2d'

    def __init__(self, dtype, astype=None):
        self.dtype = np.dtype(dtype)
        if astype is None:
            self.astype = self.dtype
        else:
            self.astype = np.dtype(astype)
        if self.dtype == np.dtype(object) or self.astype == np.dtype(object):
            raise ValueError('object arrays are not supported')

    def encode(self, buf):
        arr = numcodecs.compat.ensure_ndarray(buf).view(self.dtype)
        if arr.ndim != 2:
            raise ValueError("Delta2D only works with 2D arrays")
        enc = arr.astype(self.astype, copy=True)
        if enc.shape[0] > 1:
            for d0 in range(enc.shape[0]-1, 0, -1):
                enc[d0] -= enc[d0-1]
        return enc

    def decode(self, buf, out=None):
        print("Delta2D decode")
        print("buf.shape", buf.shape)
        enc = numcodecs.compat.ensure_ndarray(buf).view(self.astype)
        if enc.ndim != 2:
            raise ValueError("Delta2D only works with 2D arrays")
        if out is not None:
            dec = out.view(self.dtype)
            if dec.shape != enc.shape:
                raise ValueError("Output array has wrong shape")
        else:
            dec = np.empty_like(enc, dtype=self.dtype)
        dec[0] = enc[0]
        if enc.shape[0] > 1:
            for d0 in range(1, enc.shape[0]):
                dec[d0] = enc[d0] + dec[d0-1]
        return dec if out is None else out

    def get_config(self):
        return {'id': self.codec_id, 'dtype': self.dtype.str, 'astype': self.astype.str}

    def __repr__(self):
        r = f'{type(self).__name__}(dtype={self.dtype.str!r}'
        if self.astype != self.dtype:
            r += f', astype={self.astype.str!r}'
        r += ')'
        return r

# Register codecs
# NOTE: TurboPfor is register as `turbo_pfor` by omfiles already
numcodecs.register_codec(Delta2D, "delta2d")
numcodecs.register_codec(Reshape, "reshape")


class SingleOmToZarr:
    """Translate a .om file into Zarr metadata"""

    def __init__(
        self,
        om_file,
        url=None,
        spec=1,
        inline_threshold=500,
        storage_options=None,
        chunk_no=None,
        domain=None,
        reference_time=None,
        time_step=3600,
    ):
        # Initialize a reader for your om file
        if isinstance(om_file, (pathlib.Path, str)):
            fs, path = fsspec.core.url_to_fs(om_file, **(storage_options or {}))
            self.input_file = fs.open(path, "rb")
            url = om_file
            self.reader = omfiles.OmFilePyReader(self.input_file)
        elif isinstance(om_file, io.IOBase):
            self.input_file = om_file
            self.reader = omfiles.OmFilePyReader(self.input_file)
        else:
            raise ValueError("type of input `om_file` not recognized")

        self.url = url if url else om_file
        self.spec = spec
        self.inline = inline_threshold
        self.store_dict = {}
        self.store = dict_to_store(self.store_dict)
        self.name = "data" # FIXME: This should be the name from om-variable

        if domain is not None and chunk_no is not None:
            start_step = chunk_number_to_start_time(domain=domain, chunk_no=chunk_no)
            # Store time parameters
            self.reference_time = start_step.start
            self.time_step = start_step.step_dt
        else:
            self.reference_time = None
            self.time_step = None

    def translate(self):
        """Main method to create the kerchunk references"""
        # 1. Extract metadata about shape, dtype, chunks, etc.
        shape = self.reader.shape
        dtype = self.reader.dtype
        chunks = self.reader.chunk_dimensions
        scale_factor = self.reader.scale_factor
        add_offset = self.reader.add_offset
        lut = self.reader.get_complete_lut()

        # Get dimension names if available, otherwise use defaults
        # FIXME: Currently we don't have dimension names exposed by the reader (or even necessarily in the file)
        dim_names = getattr(self.reader, "dimension_names", ["x", "y", "time"])

        # Calculate number of chunks in each dimension
        chunks_per_dim = [math.ceil(s/c) for s, c in zip(shape, chunks)]

        # 2. Create Zarr array metadata (.zarray)
        blocksize = chunks[0] * chunks[1] * chunks[2] if len(chunks) >= 3 else chunks[0] * chunks[1]

        zarray = {
            "zarr_format": 2,
            "shape": shape,
            "chunks": chunks,
            "dtype": str(dtype),
            "compressor": {"id": "turbo_pfor", "chunk_elements": blocksize},  # As main compressor
            "fill_value": None,
            "order": "C",
            "filters": [
                {"id": "fixedscaleoffset", "scale": scale_factor, "offset": add_offset, "dtype": "f4", "astype": "i2"},
                {"id": "delta2d", "dtype": "<i2"},
                {"id": "reshape", "shape": [chunks[1], chunks[2]]},  # Reshape to 2D
            ]
        }

        # 3. Add metadata to store
        self.store_dict[".zgroup"] = json.dumps({"zarr_format": 2})
        self.store_dict[f"{self.name}/.zarray"] = json.dumps(zarray)
        self.store_dict[f"{self.name}/.zattrs"] = json.dumps({
            "_ARRAY_DIMENSIONS": dim_names,
            "scale_factor": scale_factor,
            "add_offset": add_offset
        })

        # 4. Add chunk references
        for chunk_idx in range(len(lut) - 1):
            # Calculate chunk coordinates (i,j,k) from linear index
            chunk_coords = self._get_chunk_coords(chunk_idx, chunks_per_dim)

            # Calculate chunk size.
            # Loop index is defined so this is safe!
            chunk_size = lut[chunk_idx + 1] - lut[chunk_idx]

            # Add to references
            key = self.name + "/" + ".".join(map(str, chunk_coords))

            # Check if chunk is small enough to inline
            if self.inline > 0 and chunk_size < self.inline:
                # Read the chunk data and inline it
                self.input_file.seek(lut[chunk_idx])
                data = self.input_file.read(chunk_size)
                try:
                    # Try to decode as ASCII
                    self.store_dict[key] = data.decode('ascii')
                except UnicodeDecodeError:
                    # If not ASCII, encode as base64
                    self.store_dict[key] = b"base64:" + base64.b64encode(data)
            else:
                # Otherwise store as reference
                self.store_dict[key] = [self.url, lut[chunk_idx], chunk_size]

        # 5. Create coordinate arrays. TODO: This needs to be improved
        # Add coordinate arrays for ALL dimensions
        for i, dim_name in enumerate(dim_names):
            dim_size = shape[i]
            if dim_name == "time":
                # Special handling for time dimension
                self._add_time_coordinate(dim_size, i)
            else:
                print(f"No coordinates for dimension {dim_name}")
                continue

        # Convert to proper format for return
        if self.spec < 1:
            print("self.spec < 1")
            return self.store
        else:
            print("translate_refs_serializable")
            translate_refs_serializable(self.store_dict)
            store = _encode_for_JSON(self.store_dict)
            return {"version": 1, "refs": store}

    def _add_time_coordinate(self, time_dim, time_axis=0):
        """Add a time coordinate array following CF conventions"""

        # Always use standard CF epoch reference
        units = "seconds since 1970-01-01T00:00:00Z"

        # Create CF-compliant units string from reference_time
        if self.reference_time is not None and self.time_step is not None:
            ref_time = self.reference_time

            # Format the reference time as CF-compliant string
            if isinstance(ref_time, datetime.datetime):
                # Calculate hours since epoch (1970-01-01)
                epoch = datetime.datetime(1970, 1, 1, 0, 0, 0)
                seconds_since_epoch = int((ref_time - epoch).total_seconds())

                # Generate time values with integer seconds
                time_values = np.arange(
                    seconds_since_epoch,
                    seconds_since_epoch + time_dim * self.time_step,
                    self.time_step,
                    dtype='i8'
                )
            else:
                raise TypeError("expected datetime.datetime object as self.reference_time")
        else:
            raise TypeError("expected datetime.datetime object as self.reference_time")

        # Create time array metadata
        time_zarray = {
            "zarr_format": 2,
            "shape": [time_dim],
            "chunks": [time_dim],
            "dtype": "<i8",
            "compressor": None,
            "fill_value": None,
            "order": "C",
            "filters": None
        }

        # Get the dimension name for time based on the axis
        dim_names = json.loads(self.store_dict[f"{self.name}/.zattrs"]).get("_ARRAY_DIMENSIONS", [])
        time_dim_name = dim_names[time_axis] if dim_names and time_axis < len(dim_names) else "time"

        # Add CF-compliant attributes
        time_zattrs = {
            "_ARRAY_DIMENSIONS": [time_dim_name],
            "units": units,
            "calendar": "proleptic_gregorian",
            "standard_name": "time",
            "long_name": "time",
        }

        # Add to zarr store
        self.store_dict[f"{time_dim_name}/.zarray"] = json.dumps(time_zarray)
        self.store_dict[f"{time_dim_name}/.zattrs"] = json.dumps(time_zattrs)

        # Add time values inline (they're small)
        self.store_dict[f"{time_dim_name}/0"] = time_values.tobytes()

        # Debug info
        print(f"Created time coordinate '{time_dim_name}' with {time_dim} values")
        print(f"Time units: {units}")
        if time_dim > 0:
            print(f"First timestamp: {time_values[0]} seconds since 1970-01-01, Last: {time_values[-1]}")

    def _get_chunk_coords(self, idx, chunks_per_dim):
        """Convert linear chunk index to multidimensional coordinates

        Parameters
        ----------
        idx : int
            Linear chunk index
        chunks_per_dim : list
            Number of chunks in each dimension

        Returns
        -------
        list
            Chunk coordinates in multidimensional space
        """
        coords = []
        remaining = idx

        # Start from the fastest-changing dimension (C-order)
        for chunks_in_dim in reversed(chunks_per_dim):
            coords.insert(0, remaining % chunks_in_dim)
            remaining //= chunks_in_dim

        return coords

    def _get_file_size(self):
        """Get the total file size"""
        current_pos = self.input_file.tell()
        self.input_file.seek(0, os.SEEK_END)
        size = self.input_file.tell()
        self.input_file.seek(current_pos)
        return size

    def close(self):
        """Close the reader"""
        if hasattr(self, 'reader'):
            self.reader.close()
