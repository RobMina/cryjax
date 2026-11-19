"""cryjax: the CRY 1.7 cosmic-ray shower model on JAX devices, with the C++ reference behind a small C ABI.

    cryjax.tables   verified release tables and CRYPrimary's setup (NumPy only)
    cryjax.native   raw showers from the C++ CRY 1.7 through the shim (NumPy + ctypes)
    cryjax.device   raw showers sampled on a JAX device (CPU, GPU), same batch layout as native
    cryjax.compare  backend-vs-backend agreement on independent shards

Raw showers only: projection onto a detector, energy cuts and acceptance loops are the experiment's.
"""
__version__ = "0.1.1"

from .tables import ALTITUDES, PDG, SPECIES, TABLE_SHA256  # noqa: F401
