"""Raw CRY 1.7 showers from the C++ reference implementation, through a small C ABI shim and ctypes.

The shim (``cryjax/native/cry_shim.cc``, built against the CRY 1.7 sources) wraps CRYSetup / CRYGenerator behind
four C functions; :class:`CrySource` loads it, seeds an independent ``mt19937_64`` per source, and returns whole
showers with CRY's cumulative live time. No projection, cuts or acceptance loop: those belong to the experiment.
It needs no JAX, art, Geant4, ROOT or CLHEP.

The shim library comes from ``library=`` or ``$CRYJAX_SHIM``; the CRY data directory from ``data=`` or
``$CRYDATAPATH`` (CRY's own variable). There are no built-in default paths.
"""
import argparse
import ctypes as ct
from dataclasses import dataclass
from datetime import date
import json
import os
from pathlib import Path
import threading

import numpy as np


class _Particle(ct.Structure):
    _fields_ = [("shower", ct.c_uint64), ("pdg", ct.c_int32), ("reserved", ct.c_int32),
                *[(key, ct.c_double) for key in ("ke", "x", "y", "z", "u", "v", "w", "t")]]


PARTICLE_DTYPE = np.dtype(_Particle)      # shower, pdg, reserved, ke (MeV), x, y, z (m), u, v, w, t (s)
ABI_VERSION = 1


@dataclass(frozen=True)
class CryBatch:
    """Raw correlated showers: metres, MeV, seconds; shower IDs local to the batch."""
    particles: np.ndarray
    cumulative_seconds: np.ndarray
    before_seconds: float

    @property
    def live_seconds(self):
        return float(self.cumulative_seconds[-1] - self.before_seconds) if len(self.cumulative_seconds) else 0.0


def _library(path):
    lib = ct.CDLL(str(path))
    lib.m2cry_abi_version.argtypes = []
    lib.m2cry_abi_version.restype = ct.c_int
    if lib.m2cry_abi_version() != ABI_VERSION:
        raise RuntimeError("unsupported CRY shim ABI")
    lib.m2cry_last_error.argtypes = []
    lib.m2cry_last_error.restype = ct.c_char_p
    lib.m2cry_create.argtypes = [ct.c_char_p, ct.c_uint64, ct.c_double, ct.c_double,
                                ct.c_int, ct.c_int, ct.c_int, ct.c_int]
    lib.m2cry_create.restype = ct.c_void_p
    lib.m2cry_generate.argtypes = [ct.c_void_p, ct.c_uint64, ct.POINTER(ct.POINTER(_Particle)),
                                  ct.POINTER(ct.c_uint64), ct.POINTER(ct.POINTER(ct.c_double)),
                                  ct.POINTER(ct.c_double)]
    lib.m2cry_generate.restype = ct.c_int
    lib.m2cry_destroy.argtypes = [ct.c_void_p]
    lib.m2cry_destroy.restype = None
    return lib


def _setting(value, variable, what):
    value = value or os.environ.get(variable)
    if not value:
        raise FileNotFoundError(f"{what}: pass it as an argument or set ${variable}")
    return value


class CrySource:
    """Host CPU CRY source; independent seeded RNG, complete showers and cumulative live time.

    Use one long-lived source per worker. CRY 1.7 itself lacks complete destructors
    for its internal tables/samplers; closing frees the shim buffers but does not
    promise recovery of every upstream allocation. Invalid upstream table contents
    can assert/abort: use only the checksum-verified release tables.
    """

    def __init__(self, *, seed=1, data=None, library=None, subbox_m=300.0,
                 latitude_deg=41.8, altitude_m=0, when="2021-06-21"):
        self._handle = None
        self._lock = threading.Lock()
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**64:
            raise ValueError("seed must be a uint64 integer")
        if not np.isfinite(subbox_m) or not 0 < subbox_m <= 300:
            raise ValueError("subbox_m must be in (0,300]")
        if not np.isfinite(latitude_deg) or not -90 <= latitude_deg <= 90:
            raise ValueError("latitude_deg must be in [-90,90]")
        if altitude_m not in (0, 2100, 11300):
            raise ValueError("altitude_m must be 0,2100,11300")
        day = date.fromisoformat(when)
        if day.year < 1900:
            raise ValueError("CRY dates before 1900 are not supported")
        data = Path(_setting(data, "CRYDATAPATH", "CRY data directory"))
        for height in (0, 2100, 11300):
            if not (data / f"cosmics_{height}.data").is_file():
                raise FileNotFoundError(f"missing CRY table: {data / f'cosmics_{height}.data'}")
        self._lib = _library(_setting(library, "CRYJAX_SHIM", "CRY shim library"))
        self._handle = self._lib.m2cry_create(os.fsencode(data), seed, subbox_m, latitude_deg,
                                             altitude_m, day.month, day.day, day.year)
        if not self._handle:
            raise RuntimeError(self._lib.m2cry_last_error().decode(errors="replace"))
        self.settings = {"seed": seed, "subbox_m": float(subbox_m), "latitude": float(latitude_deg),
                         "altitude_m": altitude_m, "date": when}

    def next_showers(self, count):
        """Generate raw CRY showers in C++; no truncation, cuts, projection or acceptance loop."""
        if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= 100000:
            raise ValueError("count must be an integer in [0,100000]; use batches for larger samples")
        with self._lock:
            if not self._handle:
                raise RuntimeError("CRY source is closed")
            particles, times = ct.POINTER(_Particle)(), ct.POINTER(ct.c_double)()
            size, before = ct.c_uint64(), ct.c_double()
            rc = self._lib.m2cry_generate(self._handle, count, ct.byref(particles), ct.byref(size),
                                         ct.byref(times), ct.byref(before))
            if rc:
                raise RuntimeError(self._lib.m2cry_last_error().decode(errors="replace"))
            rows = (np.ctypeslib.as_array(particles, shape=(size.value,)).copy()
                    if size.value else np.empty(0, dtype=PARTICLE_DTYPE))
            clock = (np.ctypeslib.as_array(times, shape=(count,)).copy() if count else np.empty(0))
            return CryBatch(rows, clock, before.value)

    def provenance(self):
        return {"generator": "cryjax-native", "model": "CRY 1.7 (C++ via shim ABI %d)" % ABI_VERSION,
                "rng": "mt19937_64 per source", **getattr(self, "settings", {})}

    def close(self):
        """Release shim-owned buffers; call once the worker no longer needs this source."""
        with self._lock:
            if self._handle:
                self._lib.m2cry_destroy(self._handle)
                self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--showers", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    with CrySource(seed=args.seed) as source:
        batch = source.next_showers(args.showers)
    ids, counts = np.unique(batch.particles["pdg"], return_counts=True)
    print(json.dumps({"raw_showers": args.showers, "particles": len(batch.particles),
                      "live_seconds": batch.live_seconds,
                      "pdg_counts": dict(zip(map(str, ids), map(int, counts)))}, indent=2))


if __name__ == "__main__":
    main()
