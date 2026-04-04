"""Simple examples for the UniTensor-based MPS modules.

Run from the project root with:

    LD_LIBRARY_PATH=/home/chiamin/miniconda3/lib:$LD_LIBRARY_PATH \
    PYTHONPATH=/home/chiamin/.local \
    python3.11 MPS/example_mps_uniTensor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cytnx
import numpy as np


# Make imports robust whether this file is launched from `Uni10/` or `Uni10/MPS/`.
cwd = Path.cwd().resolve()
for path in [cwd, cwd.parent]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from MPS.mps import *
from MPS.mps_operations import inner
from unitensor.core import *
from unitensor.utils import *


def section(title: str) -> None:
    """Pretty section header for terminal output."""
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def make_site(dl: int, d: int, dr: int, start: float = 0.0) -> "cytnx.UniTensor":
    """Create one deterministic MPS site tensor with labels l/i/r."""
    arr = np.arange(start, start + dl * d * dr, dtype=float).reshape(dl, d, dr)
    u = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=2)
    u.set_labels(["l", "i", "r"])
    return u


def main() -> None:
    """Run a compact walkthrough of the UniTensor MPS API."""
    # Build a tiny 3-site open-boundary chain:
    # site0: (1,2,3), site1: (3,2,2), site2: (2,2,1)
    a0 = make_site(1, 2, 3, start=1.0)
    a1 = make_site(3, 2, 2, start=101.0)
    a2 = make_site(2, 2, 1, start=201.0)

    section("uniTensor_utils.py")
    print("1) numpy -> UniTensor -> numpy")
    arr = np.arange(12, dtype=float).reshape(3, 2, 2)
    u = to_uniTensor(arr)
    arr_back = to_numpy_array(u)
    print("to_numpy_array shape:", arr_back.shape)
    print("round-trip equal:", np.allclose(arr, arr_back))

    print()
    print("2) quick structural debug prints")
    print_bond(a0.bond("i"))
    print_bonds(a0)

    section("uniTensor_core.py")
    print("1) scalar_from_uniTensor")
    s = cytnx.UniTensor(cytnx.from_numpy(np.array([3.5])), rowrank=0)
    print("scalar_from_uniTensor:", scalar_from_uniTensor(s))

    print()
    print("2) assert_bond_match")
    assert_bond_match(a0.bond("i"), make_site(1, 2, 1).bond("i"))
    print("assert_bond_match passed")

    print()
    print("3) qr_by_labels")
    q, r = qr_by_labels(a1, row_labels=["l", "i"], aux_label="x")
    print("qr_by_labels labels:", q.labels(), r.labels())

    print()
    print("4) svd_by_labels")
    t = cytnx.UniTensor(
        cytnx.from_numpy(np.arange(2 * 3 * 2 * 2, dtype=float).reshape(2, 3, 2, 2)),
        rowrank=2,
    )
    left, right, dw = svd_by_labels(t, row_labels=["0", "1"], absorb="right", aux_label="x")
    print("svd_by_labels done:", isinstance(left, cytnx.UniTensor), isinstance(right, cytnx.UniTensor), "discarded:", dw)

    section("mps_uniTensor.py")
    print("1) Validate one site contract")
    from MPS.mps import _check_labels
    _check_labels(a0, 0)

    print()
    print("2) Construct MPS and inspect basic metadata")
    mps = MPS([a0, a1, a2])
    print("len:", len(mps))
    print("repr:", mps)
    print("phys_dims:", mps.phys_dims)
    print("bond_dims:", mps.bond_dims)
    print("max_dim:", mps.max_dim)
    print("getitem site0 labels:", mps[0].labels())

    print()
    print("3) Iterate through sites")
    for i, tensor in enumerate(mps):
        print(i, tensor.shape())

    print()
    print("4) Replace one site with a compatible tensor")
    new_mid = make_site(3, 2, 2, start=999.0)
    mps[1] = new_mid

    print()
    print("5) Copy and run consistency checks")
    mps2 = mps.copy()
    mps.check_site_labels()
    mps._validate_bonds()
    mps._check_compatible(mps2)
    print("copy/checks passed")

    print()
    print("6) Overlap and norm workflow")
    print("inner:", inner(mps, mps2))
    print("norm before:", mps.norm())
    mps.normalize()
    print("norm after:", mps.norm())

    print()
    print("7) Gauge/canonical operations")
    mps.move_center(2)
    print("center after move_center(2):", mps.center)
    mps.orthogonalize()
    print("center after orthogonalize():", mps.center)

    print()
    print("8) MPS compression")
    from MPS.mps_compression import svd_compress_mps
    phi_svd = svd_compress_mps(mps, max_dim=2)
    print("svd_compress_mps bond_dims:", phi_svd.bond_dims)
    # denmat_compress_mps is commented out until cytnx fixes "svd-aux-qnums".
    # See _internal/CYTNX_BUGS.md and _internal/TODO.md.


if __name__ == "__main__":
    main()
