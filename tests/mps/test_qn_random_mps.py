"""Unit tests for `random_u1_sz_mps` and `allowed_cumulative_nup_after_site`."""

from __future__ import annotations

import itertools
import sys
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent.parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

import numpy as np

try:
    import cytnx
except ImportError:
    cytnx = None

if cytnx is not None:
    from MPS.mps_operations import inner
    from MPS.physical_sites.spin_half import spin_half
    from tests.helpers.mps_test_cases import (
        allowed_cumulative_nup_after_site,
        random_u1_sz_mps,
    )


def _brute_allowed_prefix_charges(site_index: int, num_sites: int, n_up_total: int) -> set[int]:
    """Reference: enumerate all length-``num_sites`` bit strings with ``n_up_total`` ones."""
    out: set[int] = set()
    for bits in itertools.product([0, 1], repeat=num_sites):
        if sum(bits) != n_up_total:
            continue
        c = sum(bits[: site_index + 1])
        out.add(c)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@unittest.skipIf(cytnx is None, "cytnx is required")
class TestAllowedCumulativeNup(unittest.TestCase):
    def test_matches_brute_force_small_chains(self):
        for N in range(1, 7):
            for n_up in range(N + 1):
                for s in range(N):
                    ref = _brute_allowed_prefix_charges(s, N, n_up)
                    got = set(allowed_cumulative_nup_after_site(s, N, n_up))
                    self.assertEqual(
                        got,
                        ref,
                        msg=f"N={N} n_up={n_up} site={s}: got {got} ref {ref}",
                    )

    def test_last_bond_unique_sector(self):
        self.assertEqual(allowed_cumulative_nup_after_site(4, 5, 3), [3])


@unittest.skipIf(cytnx is None, "cytnx is required")
class TestRandomU1SzMPS(unittest.TestCase):
    def test_invalid_args(self):
        with self.assertRaises(ValueError):
            random_u1_sz_mps(0, 0)
        with self.assertRaises(ValueError):
            random_u1_sz_mps(3, 4)
        with self.assertRaises(IndexError):
            random_u1_sz_mps(3, 1, center=3)

    def test_structure_and_blockform(self):
        mps = random_u1_sz_mps(5, 2, seed=0, center=0)
        self.assertEqual(len(mps), 5)
        self.assertEqual(mps.center, 0)
        mps._validate_bonds()
        for k, t in enumerate(mps):
            self.assertTrue(t.is_blockform(), f"site {k} not blockform")
            self.assertEqual(list(t.labels()), ["l", "i", "r"])

    def test_neighbor_bonds_match_dims(self):
        mps = random_u1_sz_mps(6, 3, seed=1)
        for k in range(len(mps) - 1):
            self.assertEqual(
                mps[k].bond("r").dim(),
                mps[k + 1].bond("l").dim(),
            )

    def test_normalize_finite_norm(self):
        mps = random_u1_sz_mps(4, 2, seed=2)
        n0 = abs(complex(inner(mps, mps))) ** 0.5
        self.assertGreater(n0, 1e-30)
        mps.normalize()
        n1 = abs(complex(inner(mps, mps))) ** 0.5
        self.assertAlmostEqual(float(n1), 1.0, places=8)

    def test_complex_dtype(self):
        mps = random_u1_sz_mps(4, 2, seed=3, dtype=complex)
        mps._validate_bonds()
        self.assertEqual(mps[0].dtype(), cytnx.Type.ComplexDouble)

    def test_inner_with_matching_product_state(self):
        """Same symmetry sector → overlap with a product state in that sector is finite."""
        N, n_up = 5, 2
        mps = random_u1_sz_mps(N, n_up, seed=4)
        # Two up-spins at the first two sites (indices 1,1,0,0,0) — total N_up = 2
        seq = [1, 1] + [0] * (N - 2)
        psi = spin_half(qn="Sz").product_state(seq, center=0)
        ov = complex(inner(psi, mps))
        self.assertTrue(np.isfinite(ov.real) and np.isfinite(ov.imag))


if __name__ == "__main__":
    unittest.main()
