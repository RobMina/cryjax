"""Parser and setup contracts of cryjax.tables."""
import numpy as np
import pytest

from cryjax import tables as T


def test_strict_parser_pdf_mean():
    data = T.parse("% comment\npdf a::b[1,3,dis] = {{1 2 1} {0 0 0}}; binning b={1 2 3};")
    np.testing.assert_allclose(data["a"].mean(), [2, 0])
    assert data["a"].weights.shape == (2, 3)
    with pytest.raises(ValueError):
        data["a"].cdf()
    for text in ("pdf a::b[0,1] = {1 1};", "parameter a={nan};", "unknown a={1};",
                 "parameter a={1}; parameter a={2};"):
        with pytest.raises(ValueError):
            T.parse(text)


def test_negative_weights_parse_but_never_sample():
    data = T.parse("pdf a::b[0,1,lin] = {{-1 2} {1 1}}; pdf c::b[0,1,lin] = {1 1};")
    assert T.negative_pdfs(data) == ["a"]
    with pytest.raises(ValueError, match="negative"):
        data["a"].cdf()
    np.testing.assert_allclose(data["c"].cdf(), [[.5, 1]])


def test_table_signature_fail_closed(tmp_path):
    path = tmp_path / "table"
    path.write_text("pdf a::b[0,1,lin]={1};")
    with pytest.raises(ValueError, match="signature"):
        T.load(path)


def test_no_default_directory(monkeypatch):
    monkeypatch.delenv("CRYDATAPATH", raising=False)
    with pytest.raises(FileNotFoundError, match="CRYDATAPATH"):
        T.load()
    with pytest.raises(ValueError, match="altitude"):
        T.table_path(500, "/x")


def test_sea_level_setup(data_dir):
    s = T.load(data_dir / "cosmics_0.data")
    assert s["lifetime_s"] == pytest.approx(5.0136379137012805e-8, rel=1e-12)   # the value fastsim pinned
    assert s["altitude_m"] == 0 and s["table_sha256"] == T.SEA_LEVEL_SHA256
    assert s["primary_pdf"].weights.shape == (1, 10000)
    with pytest.raises(ValueError, match="2100"):
        T.load(data_dir / "cosmics_0.data", altitude_m=2100)
    with pytest.raises(ValueError, match="subbox"):
        T.load(data_dir / "cosmics_0.data", subbox_m=100.0)
    with pytest.raises(ValueError, match="latitude"):
        T.load(data_dir / "cosmics_0.data", latitude=95)


@pytest.mark.parametrize("altitude", T.ALTITUDES)
def test_every_release_table_has_the_sampled_structure(data_dir, altitude):
    s = T.load(altitude_m=altitude)
    t = s["tables"]
    assert s["altitude_m"] == altitude and s["lifetime_s"] > 0
    n_primary = len(t["primaryBins"]) - 1
    n_secondary = len(t["secondaryBins"]) - 1
    assert t["nParticles300"].weights.shape[0] == n_primary
    assert t["particleFractions"].weights.shape == (n_primary, len(T.SPECIES))
    for species in T.SPECIES:
        assert t[species + "KEDist"].weights.shape[0] == n_primary
        for suffix in ("TimeDist", "CosThetaDist", "ChargeDist"):
            assert t[species + suffix].weights.shape[0] == n_secondary


def test_latitude_and_date_move_the_rate(data_dir):
    base = T.load(data_dir / "cosmics_0.data")["lifetime_s"]
    assert T.load(data_dir / "cosmics_0.data", latitude=0.0)["lifetime_s"] != base     # geomagnetic cutoff
    assert T.load(data_dir / "cosmics_0.data", when="2014-04-01")["lifetime_s"] != base  # solar cycle


def test_release_tables_negative_pdfs(data_dir):
    """The 11300 m release table has negative multiplicity weights for the small subboxes only."""
    small = ["nParticles001", "nParticles003", "nParticles010", "nParticles030"]
    assert T.load(altitude_m=0)["negative_pdfs"] == []
    assert T.load(altitude_m=2100)["negative_pdfs"] == []
    assert T.load(altitude_m=11300)["negative_pdfs"] == small
