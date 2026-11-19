"""Device sampler contracts on the JAX CPU backend (GPU runs are separate, sized jobs)."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from cryjax import device as D, tables as T


@pytest.fixture(scope="module", params=["float64", "float32"])
def setup(request, data_dir):
    return request.param, D.device_tables(data_dir / "cosmics_0.data", dtype=request.param)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_pdf_inverse_and_within_bin_semantics(dtype):
    pdf = T.Pdf("", 1, 100, "log", np.array([[1., 2., 0., 1.]]))
    p = D.pdf_to_device(pdf, dtype)
    assert p.cdf.dtype == np.dtype(dtype)
    u = jnp.array([0., .25, .5, .75, .9999], dtype)
    row = jnp.zeros(5, jnp.int32)
    result, valid = jax.jit(D.draw)(p, row, u, jnp.full(5, .5, dtype))
    np.testing.assert_array_equal(D.inverse(p.cdf, row, u), [0, 1, 1, 3, 3])
    np.testing.assert_allclose(result, 10**((np.array([0, 1, 1, 3, 3])+.5)/2), rtol=1e-6)
    assert np.all(valid)


def test_dtype_is_checked():
    with pytest.raises(ValueError, match="dtype"):
        D._dtype("float16")


def test_table_dimensions_and_dtype(setup):
    dtype, (tables, meta) = setup
    assert meta["lifetime_s"] == pytest.approx(5.0136379137012805e-8, rel=1e-12)
    assert tables["primary"].cdf.shape == (1, 10000) and tables["primary"].cdf.dtype == np.dtype(dtype)
    assert tables["multiplicity"].cdf.shape == (33, 1000)
    assert tables["energy"].cdf.shape == (7*33, 85)
    assert tables["time"].cdf.shape == (7*85, 45)
    assert tables["lifetime"].dtype == np.dtype(dtype)


def test_raw_repeatability_units_and_overflow(setup):
    dtype, (tables, _) = setup
    key = jax.random.key(144)
    a = D.raw(tables, key, 0., showers=4096, capacity=20000, dtype=dtype)
    b = D.raw(tables, key, 0., showers=4096, capacity=20000, dtype=dtype)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
    assert not a["overflow"] and not a["invalid_pdf"]
    for name in ("ke", "position", "direction", "t", "delay", "clocks"):
        assert a[name].dtype == np.dtype(dtype), name
    mask = np.asarray(a["mask"])
    assert int(a["particles"]) == int(np.sum(a["multiplicity"])) == np.sum(mask)
    tol = 1e-12 if dtype == "float64" else 1e-5
    np.testing.assert_allclose(np.sum(np.asarray(a["direction"])[mask]**2, axis=1), 1, atol=tol)
    assert np.all(np.abs(np.asarray(a["position"])[mask, :2]) <= 150)
    assert np.all(np.diff(a["clocks"]) > 0)
    assert set(np.abs(np.asarray(a["pdg"])[mask])) <= set(T.PDG)
    small = D.raw(tables, key, 0., showers=4096, capacity=1, dtype=dtype)
    assert small["overflow"] and int(small["particles"]) == int(a["particles"])


@pytest.mark.parametrize("species", range(7))
def test_conditional_energy_sampler_matches_table(data_dir, species):
    tables, _ = D.device_tables(data_dir / "cosmics_0.data")
    primary_bin = 15
    pdf = T.load(data_dir / "cosmics_0.data")["tables"][T.SPECIES[species]+"KEDist"]
    row = jnp.full(20000, species*33+primary_bin, jnp.int32)
    energy, valid = D.draw(tables["energy"], row, D.uniform(jax.random.key(species), 0, (20000,), "float64"),
                           D.uniform(jax.random.key(species), 1, (20000,), "float64"), jnp.full(20000, species))
    observed = np.histogram(np.log10(energy), np.linspace(-9, 8, 86))[0]
    probability = pdf.weights[primary_bin]/pdf.weights[primary_bin].sum()
    variance = 20000*probability*(1-probability)
    np.testing.assert_array_less(np.abs(observed-20000*probability), 7*np.sqrt(variance)+4)
    assert np.all(valid)


def test_raw_delay_is_independent_of_the_clock_start(setup):
    dtype, (tables, _) = setup
    key = jax.random.key(145)
    a = D.raw(tables, key, 0., showers=4096, capacity=20000, dtype=dtype)
    b = D.raw(tables, key, 1000., showers=4096, capacity=20000, dtype=dtype)
    mask = np.asarray(a["mask"])
    np.testing.assert_array_equal(a["delay"], b["delay"])
    assert np.all(np.isfinite(np.asarray(a["delay"])[mask]))
    np.testing.assert_array_equal(np.asarray(a["t"])[mask],
                                  (np.asarray(a["clocks"])[np.asarray(a["shower"])] + np.asarray(a["delay"]))[mask])


def test_raw_program_has_no_host_callback(setup):
    dtype, (tables, _) = setup
    jaxpr = str(jax.make_jaxpr(lambda key: D.raw(tables, key, 0., showers=64, capacity=512, dtype=dtype))(
        jax.random.key(10)))
    for forbidden in ("pure_callback", "io_callback", "host_callback"):
        assert forbidden not in jaxpr


def test_source_batches_are_seamless(data_dir):
    with D.DeviceSource(seed=7, batch=1000, table=data_dir / "cosmics_0.data") as one:
        whole = one.next_showers(2500)
    with D.DeviceSource(seed=7, batch=1000, table=data_dir / "cosmics_0.data") as two:
        parts = [two.next_showers(1000), two.next_showers(1500)]
    joined = np.concatenate([parts[0].particles, parts[1].particles.copy()])
    joined["shower"][len(parts[0].particles):] += 1000
    np.testing.assert_array_equal(whole.particles, joined)
    np.testing.assert_array_equal(whole.cumulative_seconds,
                                  np.concatenate([p.cumulative_seconds for p in parts]))
    assert parts[1].before_seconds == parts[0].cumulative_seconds[-1]
    assert whole.live_seconds == pytest.approx(sum(p.live_seconds for p in parts))
    assert np.all(np.diff(whole.cumulative_seconds) > 0)
    t_rel = whole.particles["t"] - whole.cumulative_seconds[whole.particles["shower"].astype(int)]
    assert np.all(np.isfinite(t_rel))


def test_float32_clock_is_accumulated_in_float64_on_the_host(data_dir):
    with D.DeviceSource(seed=3, batch=500, dtype="float32", table=data_dir / "cosmics_0.data") as s:
        batch = s.next_showers(5000)
    assert batch.cumulative_seconds.dtype == np.float64
    assert np.all(np.diff(batch.cumulative_seconds) > 0)


def test_seed_admission_and_all_bits():
    with pytest.raises(ValueError):
        D.make_key(-1)
    with pytest.raises(ValueError):
        D.make_key(2**64)
    a = jax.random.key_data(D.make_key(5))
    b = jax.random.key_data(D.make_key(5 + 2**32))
    assert not np.array_equal(a, b)


def test_negative_weight_pdf_is_refused_on_device():
    pdf = T.parse("pdf a::b[0,1,lin] = {{-1 2} {1 1}};")["a"]
    for dtype in ("float32", "float64"):
        with pytest.raises(ValueError, match="negative"):
            D.pdf_to_device(pdf, dtype)


def test_every_altitude_samples(data_dir):
    for altitude in T.ALTITUDES:
        t, meta = D.device_tables(T.table_path(altitude), dtype="float64")
        out = D.raw(t, D.make_key(altitude + 1), 0.0, showers=2000, capacity=20000)
        assert not bool(out["overflow"]) and not bool(out["invalid_pdf"]) and int(out["particles"]) >= 2000
