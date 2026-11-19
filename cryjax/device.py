"""JAX re-implementation of CRY 1.7's correlated shower sampler (raw showers, no detector projection).

Host code parses and uploads the verified tables (:mod:`cryjax.tables`); it never draws showers. :func:`raw` draws,
on the device and in one jitted program, everything CRY's generator draws for a batch of showers: primary energy and
the live time per shower, the multiplicity conditioned on the primary energy bin, and for every particle its species,
kinetic energy, time delay, direction cosine, charge, azimuth and lateral position. One primary energy bin conditions
all secondaries of a shower, as in CRY; within-bin sampling is logarithmic or linear as each table declares.

Precision is explicit: every floating array is created in the ``dtype`` passed (float32 or float64), nothing depends
on global JAX dtype defaults, and 64-bit JAX (``jax_enable_x64``) is required for the integer and key arithmetic.
Shapes are fixed per call: the flat particle pool has ``capacity`` lanes and an overflow is reported, never truncated.

Model reference: LLNL UCRL-CODE-227323; notices in docs/third-party-cry.txt.
"""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from . import tables as cry_tables
from .native import CryBatch, PARTICLE_DTYPE

DTYPES = ("float32", "float64")


def _dtype(dtype):
    name = np.dtype(dtype).name
    if name not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, not {name}")
    return name


def require_x64():
    if not jax.config.jax_enable_x64:
        raise RuntimeError("cryjax.device needs 64-bit JAX: jax.config.update('jax_enable_x64', True) before use")


class DevicePdf(NamedTuple):
    cdf: object
    valid: object
    low: object
    high: object
    mode: object


def pdf_to_device(pdf, dtype):
    """Row-normalised CDFs of a table PDF; zero-weight rows get a safe padding and are marked invalid."""
    dtype = _dtype(dtype)
    values = pdf.weights
    if np.any(values < 0):
        raise ValueError(cry_tables.NEGATIVE_PDF)
    total = values.sum(axis=-1, keepdims=True)
    safe = np.where(total > 0, values / np.maximum(total, 1e-300), 1/values.shape[-1])
    cdf = np.cumsum(safe, axis=-1)
    cdf[:, -1] = 1
    lo, hi = np.log10([pdf.low, pdf.high]) if pdf.kind == "log" else (pdf.low, pdf.high)
    return DevicePdf(jnp.asarray(cdf, dtype), jnp.asarray(total[:, 0] > 0), jnp.asarray(lo, dtype),
                     jnp.asarray(hi, dtype), jnp.asarray({"dis": 0, "lin": 1, "log": 2}[pdf.kind]))


def stack(pdfs, dtype):
    fields = list(zip(*[pdf_to_device(pdf, dtype) for pdf in pdfs]))
    cdf, valid, lo, hi, mode = (jnp.stack(x) for x in fields)
    return DevicePdf(cdf.reshape(-1, cdf.shape[-1]), valid.reshape(-1), lo, hi, mode)


def inverse(cdf, row, uniform):
    """Per-lane upper-bound search without materialising a PDF row per particle."""
    n = cdf.shape[-1]
    def step(_, state):
        lo, hi = state
        mid = (lo+hi)//2
        less = cdf[row, jnp.minimum(mid, n-1)] <= uniform
        return jnp.where(less, mid+1, lo), jnp.where(less, hi, mid)
    lo = jnp.zeros_like(row)
    hi = jnp.full_like(row, n)
    return lax.fori_loop(0, n.bit_length(), step, (lo, hi))[0]


def draw(pdf, row, u, within, species=None):
    index = inverse(pdf.cdf, row, u)
    lo, hi, mode = (x if species is None else x[species] for x in (pdf.low, pdf.high, pdf.mode))
    n = pdf.cdf.shape[-1]
    linear = lo + (index+within)*(hi-lo)/n
    value = jnp.where(mode == 0, lo + index*(hi-lo)/max(1, n-1),
                      jnp.where(mode == 2, 10.0**linear, linear))
    return value, pdf.valid[row] & (index < n)


def device_tables(path=None, *, dtype="float64", **settings):
    """(arrays, sampling metadata) for :func:`raw`; `settings` are those of :func:`cryjax.tables.load`."""
    require_x64()
    dtype = _dtype(dtype)
    prepared = cry_tables.load(path, **settings)
    t = prepared["tables"]
    arrays = {"primary": pdf_to_device(prepared["primary_pdf"], dtype),
              "multiplicity": pdf_to_device(t["nParticles300"], dtype),
              "species": pdf_to_device(t["particleFractions"], dtype),
              "primary_edges": jnp.asarray(t["primaryBins"], dtype),
              "secondary_edges": jnp.asarray(t["secondaryBins"], dtype),
              "lifetime": jnp.asarray(prepared["lifetime_s"], dtype), "pdg": jnp.asarray(cry_tables.PDG)}
    for key, suffix in (("energy", "KEDist"), ("time", "TimeDist"), ("angle", "CosThetaDist"), ("charge", "ChargeDist")):
        arrays[key] = stack([t[s+suffix] for s in cry_tables.SPECIES], dtype)
    meta = {k: prepared[k] for k in ("table_sha256", "date", "latitude", "subbox_m", "lifetime_s")}
    if prepared["altitude_m"] != 0:      # sea-level metadata keeps its historical keys exactly
        meta["altitude_m"] = prepared["altitude_m"]
    return arrays, meta


def uniform(key, tag, shape, dtype):
    return jax.random.uniform(jax.random.fold_in(key, tag), shape, dtype=dtype, minval=2.0**-53, maxval=1.0)


@partial(jax.jit, static_argnames=("showers", "capacity", "dtype"))
def raw(t, key, before, *, showers, capacity, dtype="float64"):
    """All raw-shower sampling on the device; pool overflow or an undefined PDF row is returned, never hidden.

    Returns per particle lane (``capacity`` lanes, ``mask`` marks the filled ones): ``shower`` (index in the batch),
    ``pdg`` (signed), ``ke`` (MeV), ``position`` (m, CRY frame, z = 0 plane), ``direction`` (unit vector), ``t``
    (s, absolute: the shower's clock plus its delay), ``delay`` (s, the drawn delay itself: independent of ``before``
    and free of the rounding of ``t - clock`` in float32); per shower: ``clocks`` (cumulative live time, s),
    ``primary_bin`` and ``multiplicity``; and the scalars ``particles``, ``overflow``, ``invalid_pdf``. Stream tags
    0-14 of ``key`` are used; callers that draw more from the same key start at tag 15.
    """
    before = jnp.asarray(before, dtype)
    primary, good = draw(t["primary"], jnp.zeros(showers, jnp.int32), uniform(key, 0, (showers,), dtype),
                         uniform(key, 1, (showers,), dtype))
    pbin = jnp.searchsorted(t["primary_edges"], primary, side="right")-1
    mult, ok = draw(t["multiplicity"], pbin, uniform(key, 2, (showers,), dtype), jnp.zeros(showers, dtype))
    mult = mult.astype(jnp.int32)
    ends = jnp.cumsum(mult)
    count = ends[-1]
    lane = jnp.arange(capacity)
    shower = jnp.minimum(jnp.searchsorted(ends, lane, side="right"), showers-1)
    mask = lane < count
    pb = pbin[shower]
    species, valid = draw(t["species"], pb, uniform(key, 3, (capacity,), dtype), jnp.zeros(capacity, dtype))
    species = species.astype(jnp.int32)-1
    ke, energy_ok = draw(t["energy"], species*(t["primary_edges"].size-1)+pb,
                         uniform(key, 4, (capacity,), dtype), uniform(key, 5, (capacity,), dtype), species)
    sb = jnp.searchsorted(t["secondary_edges"], ke, side="right")-1
    time_row = species*(t["secondary_edges"].size-1)+sb
    delay, time_ok = draw(t["time"], time_row, uniform(key, 6, (capacity,), dtype),
                          uniform(key, 7, (capacity,), dtype), species)
    w, angle_ok = draw(t["angle"], time_row, uniform(key, 8, (capacity,), dtype),
                       uniform(key, 9, (capacity,), dtype), species)
    charge, charge_ok = draw(t["charge"], time_row, uniform(key, 10, (capacity,), dtype),
                             jnp.zeros(capacity, dtype), species)
    phi = 2*jnp.pi*uniform(key, 11, (capacity,), dtype)
    transverse = jnp.sqrt(jnp.maximum(1-w*w, 0))
    direction = jnp.stack([transverse*jnp.cos(phi), transverse*jnp.sin(phi), w], axis=-1)
    position = jnp.stack([(uniform(key, 12, (capacity,), dtype)-.5)*300,
                          (uniform(key, 13, (capacity,), dtype)-.5)*300, jnp.zeros(capacity, dtype)], axis=-1)
    unsigned = t["pdg"][species]
    negative = jnp.where((unsigned == 11) | (unsigned == 13), charge > 0, charge < 0)
    pdg = jnp.where(negative, -unsigned, unsigned)
    clocks = before + jnp.cumsum(-t["lifetime"]*jnp.log(uniform(key, 14, (showers,), dtype)))
    bad = jnp.any(~good | ~ok) | jnp.any(mask & ~(valid & energy_ok & time_ok & angle_ok & charge_ok))
    return {"shower": shower, "mask": mask, "pdg": pdg, "ke": ke, "position": position,
            "direction": direction, "t": clocks[shower]+delay, "delay": delay, "clocks": clocks,
            "primary_bin": pbin, "multiplicity": mult, "particles": count,
            "overflow": count > capacity, "invalid_pdf": bad}


def make_key(seed):
    """threefry2x32 key from all 64 bits of a uint64 seed."""
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be a uint64 integer")
    return jax.random.fold_in(jax.random.key(seed & 0xffffffff, impl="threefry2x32"), seed >> 32)


_SERVED = ("shower", "pdg", "ke", "position", "direction", "t", "clocks", "particles", "overflow", "invalid_pdf")


class DeviceSource:
    """Raw CRY showers generated on a JAX device, served in the layout of :class:`cryjax.native.CrySource`.

    ``next_showers(count)`` returns a :class:`cryjax.native.CryBatch` (metres, MeV, seconds; shower ids local to
    the batch), so code written for the C++ backend runs unchanged. Showers are drawn in device batches of
    ``batch`` and buffered on the host; a batch that overflows its particle pool or reaches an undefined PDF row
    fails the source (no resampling). Not bit-identical to the C++ backend: same model, independent random stream.
    """

    def __init__(self, *, seed=1, batch=100000, capacity=None, dtype="float64", table=None, require_gpu=False,
                 **settings):
        if type(batch) is not int or not 1 <= batch <= 1000000:
            raise ValueError("batch must be an integer in [1, 1000000]")
        if require_gpu and jax.default_backend() != "gpu":
            raise ValueError("GPU backend requested but JAX's default backend is " + jax.default_backend())
        self.batch, self.dtype = batch, _dtype(dtype)
        self.capacity = 4*batch if capacity is None else capacity
        if type(self.capacity) is not int or not 1 <= self.capacity < 2**31:
            raise ValueError("capacity must be a positive int32")
        self.key = make_key(seed)
        self.seed = seed
        self.tables, self.sampling = device_tables(table, dtype=self.dtype, **settings)
        self.before = 0.0           # device clock at the end of the last device batch
        self.served_until = 0.0     # cumulative live time at the end of the last served shower
        self.batches = 0
        self.rows = np.empty(0, PARTICLE_DTYPE)
        self.clocks = np.empty(0)
        self.failed = self.closed = False

    def _fill(self):
        # Each device batch starts its clock at 0; the running total is kept on the host in float64, so a float32
        # sampling dtype never accumulates live time across batches (where it would lose resolution).
        out = raw(self.tables, jax.random.fold_in(self.key, self.batches), 0.0,
                  showers=self.batch, capacity=self.capacity, dtype=self.dtype)
        host = jax.device_get({k: out[k] for k in _SERVED})
        offset = self.before
        self.batches += 1
        if host["overflow"] or host["invalid_pdf"]:
            self.failed = True
            raise RuntimeError("device CRY batch overflowed its particle pool or reached an undefined PDF row")
        n = int(host["particles"])
        rows = np.empty(n, PARTICLE_DTYPE)
        rows["shower"] = host["shower"][:n] + len(self.clocks)
        rows["pdg"] = host["pdg"][:n]
        rows["reserved"] = 0
        rows["ke"] = host["ke"][:n]
        for j, a in enumerate("xyz"):
            rows[a] = host["position"][:n, j]
        for j, a in enumerate("uvw"):
            rows[a] = host["direction"][:n, j]
        rows["t"] = offset + np.asarray(host["t"][:n], np.float64)
        clocks = offset + np.asarray(host["clocks"], np.float64)
        self.rows = np.concatenate([self.rows, rows])
        self.clocks = np.concatenate([self.clocks, clocks])
        self.before = float(clocks[-1])

    def next_showers(self, count):
        if self.closed or self.failed:
            raise RuntimeError("device CRY source is closed or failed")
        if type(count) is not int or count < 0:
            raise ValueError("count must be a non-negative integer")
        while len(self.clocks) < count:
            self._fill()
        before = self.served_until
        take = self.rows["shower"] < count
        rows, self.rows = self.rows[take].copy(), self.rows[~take]
        self.rows["shower"] -= count
        clocks, self.clocks = self.clocks[:count].copy(), self.clocks[count:]
        if count:
            self.served_until = float(clocks[-1])
        return CryBatch(rows, clocks, before)

    def provenance(self):
        return {"generator": "cryjax-device", "model": "CRY 1.7", "seed": self.seed, "dtype": self.dtype,
                "batch": self.batch, "capacity": self.capacity, "device_batches": self.batches,
                "backend": jax.default_backend(), "rng": "jax threefry2x32; streams by batch and quantity",
                "sampling": self.sampling}

    def close(self):
        self.closed = True
        self.tables = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
