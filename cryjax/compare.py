"""Compare two raw-shower backends (device vs native, or two settings) on independent shards.

Each arm generates K independent shards (distinct seeds) of N raw showers. Per shard, rates per live second, the
multiplicity and per-species spectra (kinetic energy, cos theta, azimuth, lateral position, delay) are summarised as
counts per shower. The two arms are compared shard-mean against shard-mean: ratios with standard errors from the
shard spread, and a Welch t-test in every bin populated by at least ``min_count`` particles in both arms, with a
Bonferroni correction over all tested bins. The arms are independent samples, not paired draws: this measures
agreement of the models, it does not prove equivalence, and absent rare species certify nothing.

    python -m cryjax.compare --a device --b native --shards 10 --showers 100000 --out report.json
"""
import argparse
import json
import time

import numpy as np

KEYS = (2112, 22, 13, -13, 11, -11, 2212, 211, -211, 321, -321)
MULTIPLICITY_EDGES = [1, 2, 3, 4, 6, 10, 20, 50, 100, 300, 1000]


def _hist(values, edges):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.histogram(values, np.r_[-np.inf, edges, np.inf])[0]


def summarise(batch):
    """Per-shard metrics and histograms (counts per shower) of a :class:`cryjax.native.CryBatch`."""
    rows, clocks = batch.particles, batch.cumulative_seconds
    showers, live = len(clocks), batch.live_seconds
    if not showers or live <= 0:
        raise ValueError("empty shard")
    counts = np.bincount(rows["shower"].astype(np.int64), minlength=showers)
    metrics = {"showers_per_s": showers / live, "particles_per_shower": len(rows) / showers,
               "multi_particle_fraction": float(np.mean(counts > 1))}
    hists = {"multiplicity": (MULTIPLICITY_EDGES, _hist(counts, MULTIPLICITY_EDGES))}
    delay = rows["t"] - clocks[rows["shower"].astype(np.int64)]
    energy = np.bincount(rows["shower"].astype(np.int64), weights=rows["ke"], minlength=showers)
    with np.errstate(divide="ignore"):
        hists["shower_ke_log10"] = (np.linspace(0, 7, 36).tolist(), _hist(np.log10(energy), np.linspace(0, 7, 36)))
    for pdg in KEYS:
        sel = rows["pdg"] == pdg
        metrics[f"rate_pdg_{pdg}"] = float(np.sum(sel)) / live
        if not np.any(sel):
            continue
        r = rows[sel]
        with np.errstate(divide="ignore"):
            spectra = {"ke_log10": (np.log10(r["ke"]), np.linspace(-3, 6, 46)),
                       "w": (r["w"], np.linspace(-1, 1, 41)),
                       "azimuth": (np.arctan2(r["v"], r["u"]), np.linspace(-np.pi, np.pi, 37)),
                       "x": (r["x"], np.linspace(-150, 150, 31)),
                       "y": (r["y"], np.linspace(-150, 150, 31)),
                       "delay_log10": (np.log10(delay[sel]), np.linspace(-10, -2, 41))}
        for name, (values, edges) in spectra.items():
            hists[f"{pdg}:{name}"] = (edges.tolist(), _hist(values, edges))
    return {"showers": showers, "particles": len(rows), "live_seconds": live, "metrics": metrics,
            "histograms": {k: {"edges": list(e), "counts_per_shower": (h / showers).tolist(),
                               "counts": h.tolist()} for k, (e, h) in hists.items()}}


def compare(a, b, *, min_count=20):
    """Ratios and per-bin Welch tests of two lists of shard summaries (see the module docstring)."""
    from scipy.stats import t as student_t
    if len(a) < 2 or len(b) < 2:
        raise ValueError("need at least two shards per arm for shard-spread errors")
    metrics = {}
    for key in sorted(set.intersection(*[set(s["metrics"]) for s in a + b])):
        x = np.array([s["metrics"][key] for s in a])
        y = np.array([s["metrics"][key] for s in b])
        ex, ey = x.std(ddof=1) / np.sqrt(len(x)), y.std(ddof=1) / np.sqrt(len(y))
        ratio = float(x.mean() / y.mean()) if y.mean() else None
        ratio_se = float(np.hypot(ex / y.mean(), x.mean() * ey / y.mean()**2)) if y.mean() else None
        metrics[key] = {"a": float(x.mean()), "b": float(y.mean()), "a_se": float(ex), "b_se": float(ey),
                        "ratio": ratio, "ratio_se": ratio_se}
    spectra, pvalues = {}, []
    for key in sorted(set.intersection(*[set(s["histograms"]) for s in a + b])):
        x = np.array([s["histograms"][key]["counts_per_shower"] for s in a])
        y = np.array([s["histograms"][key]["counts_per_shower"] for s in b])
        nx = np.sum([s["histograms"][key]["counts"] for s in a], axis=0)
        ny = np.sum([s["histograms"][key]["counts"] for s in b], axis=0)
        vx, vy = x.var(axis=0, ddof=1) / len(x), y.var(axis=0, ddof=1) / len(y)
        use = (nx >= min_count) & (ny >= min_count) & ((vx + vy) > 0)
        diff = x.mean(axis=0) - y.mean(axis=0)
        stat = np.abs(diff[use]) / np.sqrt((vx + vy)[use])
        dof = (vx + vy)[use]**2 / (vx[use]**2 / (len(x) - 1) + vy[use]**2 / (len(y) - 1))
        p = 2 * student_t.sf(stat, dof)
        pvalues.extend(p.tolist())
        spectra[key] = {"edges": a[0]["histograms"][key]["edges"], "a": x.mean(axis=0).tolist(),
                        "b": y.mean(axis=0).tolist(), "tested_bins": int(use.sum()),
                        "max_abs_t": float(stat.max()) if len(stat) else None,
                        "min_p": float(p.min()) if len(p) else None}
    family = len(pvalues)
    for s in spectra.values():
        s["bonferroni_min_p"] = min(1.0, s["min_p"] * family) if s["min_p"] is not None else None
    return {"metrics": metrics, "spectra": spectra,
            "family": {"tested_bins": family, "bonferroni_min_p": min(1.0, min(pvalues) * family) if pvalues else None,
                       "method": f"Welch t across independent shards; bins with >= {min_count} particles in both "
                                 "arms; Bonferroni over all tested bins; descriptive, no admission band"}}


def _source(kind, seed, settings, args):
    if kind == "native":
        from .native import CrySource
        return CrySource(seed=seed, library=args.library, data=args.data, altitude_m=settings["altitude_m"],
                         latitude_deg=settings["latitude"], when=settings["when"])
    from . import device
    from .tables import table_path
    return device.DeviceSource(seed=seed, batch=args.batch, dtype=args.dtype, require_gpu=args.require_gpu,
                               table=table_path(settings["altitude_m"], args.data),
                               latitude=settings["latitude"], when=settings["when"])


def run_arm(kind, shards, showers, seed, settings, args):
    out, receipts = [], []
    for i in range(shards):
        start = time.perf_counter()
        with _source(kind, seed + i, settings, args) as source:
            parts = []
            left = showers
            while left:
                n = min(left, 100000)
                parts.append(source.next_showers(n))
                left -= n
            receipts.append({**source.provenance(), "wall_s": time.perf_counter() - start})
        rows = np.concatenate([p.particles for p in parts])
        offsets = np.cumsum([0] + [len(p.cumulative_seconds) for p in parts])[:-1]
        shower = np.concatenate([p.particles["shower"] + o for p, o in zip(parts, offsets)])
        rows["shower"] = shower
        from .native import CryBatch
        batch = CryBatch(rows, np.concatenate([p.cumulative_seconds for p in parts]), parts[0].before_seconds)
        out.append(summarise(batch))
    return out, receipts


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", choices=("device", "native"), default="device")
    p.add_argument("--b", choices=("device", "native"), default="native")
    p.add_argument("--shards", type=int, default=10)
    p.add_argument("--showers", type=int, default=100000, help="raw showers per shard")
    p.add_argument("--seed", type=int, default=20261118, help="arm a uses seed+i, arm b seed+10**6+i")
    p.add_argument("--altitude", type=int, default=0, choices=(0, 2100, 11300))
    p.add_argument("--latitude", type=float, default=41.8)
    p.add_argument("--date", default="2021-06-21")
    p.add_argument("--dtype", default="float64", choices=("float32", "float64"), help="device sampling dtype")
    p.add_argument("--batch", type=int, default=100000, help="device batch (showers)")
    p.add_argument("--require-gpu", action="store_true")
    p.add_argument("--library", help="CRY shim (default $CRYJAX_SHIM)")
    p.add_argument("--data", help="CRY data directory (default $CRYDATAPATH)")
    p.add_argument("--min-count", type=int, default=20)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if "device" in (args.a, args.b):
        import jax
        jax.config.update("jax_enable_x64", True)
    settings = {"altitude_m": args.altitude, "latitude": args.latitude, "when": args.date}
    a, ra = run_arm(args.a, args.shards, args.showers, args.seed, settings, args)
    b, rb = run_arm(args.b, args.shards, args.showers, args.seed + 10**6, settings, args)
    report = {"arms": {"a": args.a, "b": args.b}, "settings": settings, "shards": args.shards,
              "showers_per_shard": args.showers, **compare(a, b, min_count=args.min_count),
              "receipts": {"a": ra, "b": rb}}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({"family": report["family"],
                      "metrics": {k: (v["ratio"], v["ratio_se"]) for k, v in report["metrics"].items()}}, indent=1))


if __name__ == "__main__":
    main()
