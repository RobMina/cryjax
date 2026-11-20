# cryjax

The CRY 1.7 cosmic-ray shower model ([LLNL](https://nuclear.llnl.gov/simulations)) on JAX devices, next to the
C++ reference behind a four-function C ABI, with a tool that measures how well the two agree.

| Module | What it does | Needs |
|---|---|---|
| `cryjax.tables` | strict parser of the release tables; checks their SHA-256; reproduces CRYPrimary's setup for a latitude and date (solar cycle, geomagnetic cutoff, partial rates, primary spectrum, live time per shower) | NumPy |
| `cryjax.device` | `raw(...)`: one jitted program that draws whole raw showers on a JAX device (primary energy and live time, multiplicity, species, energies, delays, angles, charges, lateral positions); `DeviceSource`: the same showers served batch by batch | JAX (CPU or GPU), 64-bit mode |
| `cryjax.native` | `CrySource`: raw showers from the C++ CRY 1.7 through the shim (`cryjax/native/cry_shim.cc`) | NumPy, ctypes, a built shim |
| `cryjax.compare` | device vs native (or any two settings) on independent shards: rate and spectrum ratios with shard errors, per-bin Welch tests, Bonferroni over all bins | SciPy |

**Raw showers only.** Both backends return showers in CRY's frame (metres, MeV, seconds; positions on the z = 0
plane of the 300 m subbox; cumulative live time per shower). Projecting them onto a detector, energy cuts and the
acceptance loop of an experiment's generator belong to the experiment's code. `mu2e-fastsim` is the first user: it
keeps the Mu2e projection and selection and imports the sampler from here.

## Install

```bash
python -m pip install -e '.[device,compare]'      # or '.[test]' for development
bash cryjax/native/build-shim.sh /some/dir          # downloads CRY 1.7 (checksum), builds libcryshim.so, prints:
export CRYDATAPATH=/some/dir/cry_v1.7/data          # the release tables (the loader accepts only these, by hash)
export CRYJAX_SHIM=/some/dir/libcryshim.so          # only for cryjax.native
```

The CRY sources and tables are not redistributed here; the script fetches the official release and checks its
checksum. There are no built-in default paths.

## Use

```python
import jax
jax.config.update("jax_enable_x64", True)
from cryjax.device import DeviceSource
with DeviceSource(seed=1, dtype="float64") as source:      # altitude_m, latitude, when: see cryjax.tables.load
    batch = source.next_showers(100000)                      # CryBatch: particles, cumulative_seconds, live_seconds
```

Lower level, inside your own jitted program: `tables, meta = cryjax.device.device_tables(dtype=...)`, then
`cryjax.device.raw(tables, key, before, showers=N, capacity=M, dtype=...)`. Stream tags 0-14 of the key are used;
draw your own quantities from tag 15 on. `raw` returns each particle's absolute time `t` (the shower clock, started
at `before`, plus the delay) and the drawn `delay` itself: in float32, take the delay from `delay`, not from
`t - clock`, and keep any running clock in float64 on the host, as `DeviceSource` does (start each batch at 0).

Compare the backends:

```bash
python -m cryjax.compare --a device --b native --shards 10 --showers 100000 --out report.json
```

## Design rules

- **Fail closed.** Unknown tables, other subbox sizes, malformed contents, a PDF row with zero weight that is
  actually reached, and a particle-pool overflow all raise; nothing is clipped, resampled or replaced by a fallback.
- **Explicit precision.** Every floating array is created in the `dtype` passed (float32 or float64); nothing relies
  on global JAX dtype defaults. `DeviceSource` keeps the running live time in float64 on the host in either mode.
- **Reproducible per seed and shape.** threefry2x32 keys use all 64 seed bits; batch index and quantity give
  independent streams. Changing the batch or pool shape can change the stream. The device stream is not the C++
  stream: the backends agree statistically, not bit for bit.

## Validation status

| Setting | Device vs | Result |
|---|---|---|
| sea level, 300 m, latitude 41.8, 2021-06-21 | an experiment's stock CRY generator, after its detector projection (mu2e-fastsim, 2026-11-03: 100 + 100 shards of 50,000 accepted events) | accepted rate per live second 1.00014 +- 0.00063, species rates within errors, no bin flagged among 2,534 (Bonferroni) |
| sea level, raw showers | the C++ shim (`cryjax.compare`) | to be run for the first release |
| 2100 m, 11300 m, other latitudes and dates | the C++ shim | to be run; the table structure is checked by the tests |

## Tests

```bash
python -m pytest -q        # table, device and native tests skip unless CRYDATAPATH / CRYJAX_SHIM are set
```

CI (`.github/workflows/ci.yml`) downloads the release, builds the shim and runs everything on the JAX CPU backend,
Python 3.10 to 3.13.

## Licence

Apache-2.0 for this package; CRY's BSD-style licence and the additional DOE notice (`docs/third-party-cry.txt`)
apply to CRY and to the parts here that reproduce it. See `NOTICE`. cryjax is not produced or endorsed by LLNL.

## Citing

Please cite the release you used: `CITATION.cff` holds the metadata (GitHub's "Cite this repository" renders it), and
each tagged release is archived on Zenodo with its own DOI.
