# Validation: device CRY against the C++ CRY 1.7 reference

Public inputs only: the CRY 1.7 release (sources and tables, fetched and checked by `cryjax/native/build-shim.sh`),
this package, and seeds. No experiment geometry, projection or selection is involved: both arms are raw showers in
CRY's frame.

## Design

- Arms: `cryjax.device.DeviceSource` (JAX on CPU) against `cryjax.native.CrySource` (the C++ CRY 1.7 through the
  shim), each with its own seeds.
- Settings: latitude 41.8, date 2021-06-21, 300 m subbox, at the three altitudes the release tables cover (0, 2100
  and 11300 m); device sampling in float64 at all three and in float32 at sea level.
- Sample: 40 independent shards of 250,000 raw showers per arm and setting (10 M showers per arm).
- Tool: `python -m cryjax.compare --a device --b native` (rate and per-species ratios with shard standard errors;
  per-bin Welch tests over the energy, angle, multiplicity and position spectra; Bonferroni over all bins).
- Code: cryjax `3094804` (0.1.0). 0.1.1 only adds the `delay` output of `raw`; every existing output and the draw
  streams are unchanged.

## Result (2026-11-18): the arms agree at every setting

Ratios device / native with shard standard errors:

| Setting | showers per live second | mu- | mu+ | e- | e+ | gamma | n | p | smallest Bonferroni p (bins) |
|---|---|---|---|---|---|---|---|---|---|
| 0 m, float64 | 0.9999 +- 0.0005 | 1.0011 +- 0.0012 | 0.9996 +- 0.0014 | 0.9991 +- 0.0019 | 1.0000 +- 0.0024 | 0.9994 +- 0.0010 | 0.9982 +- 0.0017 | 1.0038 +- 0.0048 | 1.0 (1300) |
| 2100 m, float64 | 0.9999 +- 0.0005 | 1.0001 +- 0.0016 | 0.9988 +- 0.0016 | 0.9997 +- 0.0019 | 0.9977 +- 0.0024 | 0.9991 +- 0.0012 | 0.9998 +- 0.0009 | 1.0015 +- 0.0035 | 0.35 (1609) |
| 11300 m, float64 | 0.9999 +- 0.0004 | 0.9994 +- 0.0035 | 1.0034 +- 0.0036 | 1.0029 +- 0.0016 | 0.9989 +- 0.0019 | 0.9992 +- 0.0009 | 0.9987 +- 0.0009 | 1.0004 +- 0.0029 | 1.0 (1603) |
| 0 m, float32 | 0.9994 +- 0.0004 | 0.9998 +- 0.0012 | 0.9979 +- 0.0013 | 1.0004 +- 0.0019 | 1.0029 +- 0.0022 | 0.9991 +- 0.0011 | 1.0013 +- 0.0017 | 1.0057 +- 0.0051 | 1.0 (1302) |

Charged pions agree within their errors (largest pull 1.8). The mean multiplicity and the multi-particle fraction
agree to 0.1 %. No spectrum bin is flagged at any setting. Warm CPU time per 250,000-shower shard: device 0.8-0.9 s,
native 0.5 s (one core; the device arm is meant for accelerators).

## Release-table finding

`cosmics_11300.data` of CRY 1.7 has negative weights in the multiplicity PDFs of its 1, 3, 10 and 30 m subboxes
(`nParticles001`, `003`, `010`, `030`; from primary-bin row 18 on). CRY normalises such a row by its signed sum and
scans a non-monotone CDF. cryjax parses these PDFs, lists them in `load()["negative_pdfs"]` and refuses to build a
CDF from them; none of the PDFs it samples (the 300 m subbox) is affected. The 0 m and 2100 m tables are clean.

## Not covered

- Other latitudes and dates than the one above (the table setup for them is checked by the unit tests only).
- GPU arms: the device program is the same, but no GPU comparison is recorded here.
- Detector projections and selections, which belong to the experiment using cryjax.
