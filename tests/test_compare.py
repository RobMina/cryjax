"""Shard comparison on synthetic batches (no CRY tables needed)."""
import numpy as np
import pytest

pytest.importorskip("scipy")

from cryjax import compare as C
from cryjax.native import PARTICLE_DTYPE, CryBatch


def synthetic(seed, showers=4000, muon_fraction=0.7):
    rng = np.random.default_rng(seed)
    mult = rng.integers(1, 4, showers)
    n = int(mult.sum())
    rows = np.zeros(n, PARTICLE_DTYPE)
    rows["shower"] = np.repeat(np.arange(showers), mult)
    rows["pdg"] = np.where(rng.random(n) < muon_fraction, rng.choice([13, -13], n), 22)
    rows["ke"] = 10**rng.uniform(1, 4, n)
    w = -rng.random(n)
    phi = rng.uniform(-np.pi, np.pi, n)
    s = np.sqrt(1-w*w)
    rows["u"], rows["v"], rows["w"] = s*np.cos(phi), s*np.sin(phi), w
    rows["x"], rows["y"] = rng.uniform(-150, 150, n), rng.uniform(-150, 150, n)
    clocks = np.cumsum(rng.exponential(5e-8, showers))
    rows["t"] = clocks[rows["shower"]] + 10**rng.uniform(-8, -5, n)
    return CryBatch(rows, clocks, 0.0)


def test_identical_models_agree():
    a = [C.summarise(synthetic(i)) for i in range(6)]
    b = [C.summarise(synthetic(100 + i)) for i in range(6)]
    report = C.compare(a, b)
    assert report["family"]["tested_bins"] > 50
    assert report["family"]["bonferroni_min_p"] > 1e-3
    assert report["metrics"]["particles_per_shower"]["ratio"] == pytest.approx(1, abs=0.02)


def test_a_species_change_is_flagged():
    a = [C.summarise(synthetic(i)) for i in range(6)]
    b = [C.summarise(synthetic(100 + i, muon_fraction=0.6)) for i in range(6)]
    report = C.compare(a, b)
    assert report["family"]["bonferroni_min_p"] < 1e-6
    assert report["metrics"]["rate_pdg_22"]["ratio"] < 0.9


def test_needs_two_shards():
    s = [C.summarise(synthetic(0))]
    with pytest.raises(ValueError, match="two shards"):
        C.compare(s, s)
