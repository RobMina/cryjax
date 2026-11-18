"""Load the published CRY 1.7 distribution tables; host setup only, no shower draws.

Model reference: CRYPrimary / CRYGenerator / CRYPdf, LLNL UCRL-CODE-227323 (notices in docs/third-party-cry.txt).

:func:`parse` reads the semicolon/braced table format strictly, without executing anything from the file.
:func:`load` checks the table's SHA-256 against the three tables of the CRY 1.7 release (one per altitude), then
reproduces CRYPrimary's setup for a latitude and date: the solar-cycle mix of the primary spectrum, the geomagnetic
cutoff, the 1000-point left-Riemann partial rate per primary bin, the 10,000-bin primary energy PDF and the live time
per shower. Unknown tables, other subbox sizes and malformed contents fail closed.
"""
from dataclasses import dataclass
from datetime import date
import hashlib
import os
from pathlib import Path
import re

import numpy as np

SPECIES = ("neutron", "gamma", "muon", "electron", "proton", "pion", "kaon")
PDG = (2112, 22, 13, 11, 2212, 211, 321)

# SHA-256 of the distribution tables shipped in cry_v1.7.tar.gz (data/cosmics_<altitude>.data)
SEA_LEVEL_SHA256 = "b5227d8c1d38ab249a3b238456a67a60c9a9a5b6f91f748562f4a517390dca75"
TABLE_SHA256 = {
    0: SEA_LEVEL_SHA256,
    2100: "86cc99fb87fad93896d307fdbe8a90c63855e02c788bfb1eafee77c4c8254e95",
    11300: "8d583055d1c38a941c1e84a7843337ef4e00bce7d95df7dbb71f2947e2aec9e8",
}
ALTITUDES = tuple(TABLE_SHA256)
SUBBOX_M = 300.0        # the only subbox the device sampler supports (multiplicity table nParticles300)
NEGATIVE_PDF = "CRY PDF with negative weights (see cryjax.tables.negative_pdfs); refusing to build its CDF"


@dataclass(frozen=True)
class Pdf:
    binning: str
    low: float
    high: float
    kind: str
    weights: np.ndarray

    def cdf(self):
        if np.any(self.weights < 0):
            raise ValueError(NEGATIVE_PDF)
        total = self.weights.sum(axis=-1, keepdims=True)
        if np.any(total <= 0):
            raise ValueError("unreachable zero-weight PDF row must be handled explicitly")
        cdf = np.cumsum(self.weights / total, axis=-1)
        cdf[:, -1] = 1.0
        return cdf

    def mean(self):
        n = self.weights.shape[1]
        if self.kind == "dis":
            centers = self.low + np.arange(n) * (self.high-self.low) / max(1, n-1)
        else:
            lo, hi = (np.log10([self.low, self.high]) if self.kind == "log" else (self.low, self.high))
            centers = lo + (np.arange(n)+.5) * (hi-lo) / n
            if self.kind == "log":
                centers = 10**centers
        return (self.weights * centers).sum(axis=-1) / np.maximum(self.weights.sum(axis=-1), 1e-300)


def parse(text):
    """Strict parser for the published semicolon/braced format, without executing file contents."""
    joined = " ".join(line.split("%", 1)[0] for line in text.splitlines())
    result = {}
    for statement in joined.split(";"):
        if not statement.strip():
            continue
        lhs, rhs = statement.strip().split("=", 1)
        kind, name = lhs.split(None, 1)
        values = [np.fromstring(row, sep=" ") for row in re.findall(r"\{([^{}]*)\}", rhs)]
        if not values or len({len(row) for row in values}) != 1 or not len(values[0]):
            raise ValueError("malformed CRY table array")
        array = np.stack(values)
        if not np.all(np.isfinite(array)):
            raise ValueError("nonfinite CRY table value")
        if kind == "pdf":
            match = re.fullmatch(r"(\w+)::(\w+)\[([^,]+),([^,]+),(dis|lin|log)\]", name.strip())
            if not match:
                raise ValueError("malformed CRY PDF")
            key, binning, low, high, mode = match.groups()
            low, high = float(low), float(high)
            if not low < high or mode == "log" and low <= 0:
                raise ValueError("invalid CRY PDF range")
            value = Pdf(binning, low, high, mode, array)
        else:
            key = name.strip()
            if kind == "function":
                key = key.split("::")[0]
            if kind not in ("function", "parameter", "paramInt", "binning") or len(values) != 1:
                raise ValueError("unsupported CRY table declaration")
            value = array[0]
        if key in result:
            raise ValueError("duplicate CRY table declaration")
        result[key] = value
    return result


def negative_pdfs(tables):
    """Names of the PDFs of a parsed table with negative weights, sorted.

    The CRY 1.7 release table ``cosmics_11300.data`` has negative entries in its multiplicity PDFs for the 1, 3, 10
    and 30 m subboxes (``nParticles001`` ... ``nParticles030``). CRY's C++ normalises each row by its signed sum and
    builds a non-monotone cumulative sum from it; cryjax does not reproduce that and refuses to sample such a PDF.
    The 300 m subbox PDFs it samples are not affected.
    """
    return sorted(k for k, v in tables.items() if isinstance(v, Pdf) and np.any(v.weights < 0))


def _spectrum(energy, params):
    a, b, c, d = params
    return a * (energy+b)**c * energy**d


def table_path(altitude_m=0, data=None):
    """Path of the release table for an altitude: `data` or $CRYDATAPATH; no built-in default directory."""
    if altitude_m not in TABLE_SHA256:
        raise ValueError(f"CRY altitude must be one of {ALTITUDES} m")
    data = data or os.environ.get("CRYDATAPATH")
    if not data:
        raise FileNotFoundError("CRY tables: pass data=<dir> or set CRYDATAPATH to the release's data directory")
    return Path(data) / f"cosmics_{altitude_m}.data"


def load(path=None, *, altitude_m=None, latitude=41.8, when="2021-06-21", subbox_m=SUBBOX_M):
    """Verified CRY setup for one release table, latitude and date, all seven species, 300 m subbox.

    `path` is a table file (its hash decides the altitude; `altitude_m`, if given, must agree); without it the table
    of `altitude_m` (default sea level) is taken from $CRYDATAPATH.
    """
    if path is None:
        path = table_path(0 if altitude_m is None else altitude_m)
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    found = [h for h, sha in TABLE_SHA256.items() if sha == digest]
    if not found:
        raise ValueError("unverified CRY table signature (not a CRY 1.7 release table)")
    if altitude_m is not None and altitude_m != found[0]:
        raise ValueError(f"table is the {found[0]} m release table, not {altitude_m} m")
    if subbox_m != SUBBOX_M:
        raise ValueError("only the 300 m subbox is supported")
    if isinstance(latitude, bool) or not np.isfinite(latitude) or not -90 <= latitude <= 90:
        raise ValueError("invalid CRY latitude")
    day = date.fromisoformat(when)
    decimal_year = day.year + (day-date(day.year, 1, 1)).days / (date(day.year+1, 1, 1)-date(day.year, 1, 1)).days
    t = parse(data.decode("ascii"))
    primary, secondary = t["primaryBins"], t["secondaryBins"]
    cycle = abs(np.sin(np.pi*(decimal_year-t["solarMinDate"][0])/t["solarCycleLength"][0]))
    cutoff = t["bfieldCorr"][0] * np.cos(np.deg2rad(latitude))**t["bfieldCorr"][1]
    minimum, maximum = max(primary[0], cutoff), primary[-1]
    if not minimum < maximum:
        raise ValueError("empty CRY primary spectrum")
    flux = lambda e: (1-cycle)*_spectrum(e, t["primarySpectrumSolarMin"]) + cycle*_spectrum(e, t["primarySpectrumSolarMax"])
    n_pdf, fractions = t["nParticles300"], t["particleFractions"]
    mean = n_pdf.mean()
    weight = np.divide(fractions.weights.sum(axis=1), mean, out=np.zeros_like(mean), where=mean>0)
    points = primary[:-1, None] + np.arange(1000)[None, :] * .001 * np.diff(primary)[:, None]
    partial = (flux(points) * (points >= minimum)).sum(axis=1) * .001 * np.diff(primary)
    lifetime = 1 / (np.dot(partial, weight) * np.pi * 300**2)
    edges = np.logspace(np.log10(minimum), np.log10(maximum), 10001)
    centers = 10**((np.log10(edges[:-1])+np.log10(edges[1:]))/2)
    primary_weights = flux(centers) * weight[np.searchsorted(primary, centers, side="right")-1] * np.diff(edges)
    p_pdf = Pdf("", minimum, maximum, "log", primary_weights[None, :])
    for name in ("latDist",):
        pdf = t[name]
        if pdf.kind != "lin" or not np.all(pdf.weights == 1) or pdf.weights.shape != (len(primary)-1, 1):
            raise ValueError("device setup expects the verified uniform lateral distribution")
    if not np.all(np.diff(primary)>0) or not np.all(np.diff(secondary)>0):
        raise ValueError("unordered CRY energy bins")
    needed = ["KEDist", "TimeDist", "CosThetaDist", "ChargeDist"]
    negative = negative_pdfs(t)
    used = ["nParticles300", "particleFractions", "latDist"] + [s+x for x in needed for s in SPECIES]
    if any(name in negative for name in used):
        raise ValueError("a PDF the sampler uses has negative weights: " + ", ".join(n for n in used if n in negative))
    zero_rows = {}
    for suffix in needed:
        for species in SPECIES:
            pdf = t[species+suffix]
            zero_rows[species+suffix] = np.flatnonzero(pdf.weights.sum(axis=1) == 0).tolist()
    return {"tables": t, "primary_pdf": p_pdf, "lifetime_s": float(lifetime), "primary_bin_weights": weight,
            "table_sha256": digest, "altitude_m": found[0], "latitude": latitude, "date": when,
            "zero_rows": zero_rows, "negative_pdfs": negative, "subbox_m": SUBBOX_M}


def main(argv=None):
    import argparse
    import json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", help="table file (default: $CRYDATAPATH/cosmics_<altitude>.data)")
    p.add_argument("--altitude", type=int, choices=ALTITUDES)
    p.add_argument("--latitude", type=float, default=41.8)
    p.add_argument("--date", default="2021-06-21")
    a = p.parse_args(argv)
    result = load(a.table, altitude_m=a.altitude, latitude=a.latitude, when=a.date)
    print(json.dumps({k: v for k, v in result.items() if k not in ("tables", "primary_pdf", "primary_bin_weights")}, indent=2))
    for name in ("nParticles300", "particleFractions", *[s+x for s in SPECIES for x in ("KEDist", "TimeDist", "CosThetaDist", "ChargeDist")]):
        pdf = result["tables"][name]
        print(name, pdf.weights.shape, pdf.kind, pdf.low, pdf.high)
    print("raw mean multiplicities", result["tables"]["nParticles300"].mean())


if __name__ == "__main__":
    main()
