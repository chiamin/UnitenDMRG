"""Unit tests for MPS/mps_operations.py.

Coverage
--------
1. Real-valued tests (TestMPSOperations)
   - inner: self-overlap equals 1 for normalised MPS, both real and imag checked
   - expectation: identity MPO gives same result as inner

2. Complex-valued tests (TestInnerComplex, TestExpectationComplex)
   - inner: self-overlap is real and positive, conjugate symmetry <psi|phi>=conj(<phi|psi>)
   - expectation: Hermitian MPO gives real expectation, conjugate symmetry
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent.parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

try:
    import cytnx
except ImportError:  # pragma: no cover
    cytnx = None

if cytnx is not None:
    from MPS.mps import MPS
    from MPS.mps_init import random_mps
    from MPS.mps_operations import expectation, inner
else:  # pragma: no cover
    MPS = object

    def _missing_cytnx(*_args, **_kwargs):
        raise RuntimeError("cytnx is required")

    random_mps = _missing_cytnx
    expectation = _missing_cytnx
    inner = _missing_cytnx


@unittest.skipIf(cytnx is None, "cytnx is required for mps_operations tests")
class TestMPSOperations(unittest.TestCase):
    """Tests for inner and expectation in mps_operations."""

    def test_inner_normalized_self_overlap(self) -> None:
        psi = random_mps(num_sites=3, phys_dim=2, bond_dim=3, seed=1, normalize=True)
        val = inner(psi, psi)
        self.assertAlmostEqual(float(val.real), 1.0, places=10)
        self.assertAlmostEqual(float(val.imag), 0.0, places=10)

    def test_expectation_identity(self) -> None:
        psi = random_mps(num_sites=3, phys_dim=2, bond_dim=3, seed=1, normalize=True)
        I = []
        for _ in range(3):
            arr = np.eye(2, dtype=float).reshape(1, 2, 2, 1)
            ut = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=3)
            ut.set_labels(["l", "ip", "i", "r"])
            I.append(ut)
        self.assertAlmostEqual(
            expectation(psi, I, psi),
            inner(psi, psi),
            places=10,
        )


@unittest.skipIf(cytnx is None, "cytnx is required for mps_operations tests")
class TestInnerComplex(unittest.TestCase):
    """Complex-valued inner product tests.

    These tests verify that the bra is correctly conjugated.  For real MPS,
    conj(psi) == psi so the real tests cannot catch missing-conjugation bugs.
    """

    def setUp(self):
        self.psi = random_mps(num_sites=4, phys_dim=2, bond_dim=3,
                              dtype=complex, seed=10, normalize=True)
        self.phi = random_mps(num_sites=4, phys_dim=2, bond_dim=3,
                              dtype=complex, seed=11, normalize=True)

    def test_self_overlap_is_real_and_positive(self):
        """<psi|psi> is real and positive for a complex normalised MPS."""
        val = complex(inner(self.psi, self.psi))
        self.assertAlmostEqual(val.imag, 0.0, places=10)
        self.assertGreater(val.real, 0.0)

    def test_conjugate_symmetry(self):
        """<psi|phi> = conj(<phi|psi>) for complex MPS."""
        psi_phi = complex(inner(self.psi, self.phi))
        phi_psi = complex(inner(self.phi, self.psi))
        self.assertAlmostEqual(psi_phi.real,  phi_psi.real,  places=10)
        self.assertAlmostEqual(psi_phi.imag, -phi_psi.imag,  places=10)


@unittest.skipIf(cytnx is None, "cytnx is required for mps_operations tests")
class TestExpectationComplex(unittest.TestCase):
    """Complex-valued expectation value tests.

    Uses a real Hermitian MPO (identity) and complex MPS to verify that
    the bra is correctly conjugated inside expectation().
    """

    def setUp(self):
        N = 4
        self.psi = random_mps(num_sites=N, phys_dim=2, bond_dim=3,
                              dtype=complex, seed=20, normalize=True)
        self.phi = random_mps(num_sites=N, phys_dim=2, bond_dim=3,
                              dtype=complex, seed=21, normalize=True)
        self.I_mpo = []
        for _ in range(N):
            arr = np.eye(2, dtype=float).reshape(1, 2, 2, 1)
            ut = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=3)
            ut.set_labels(["l", "ip", "i", "r"])
            self.I_mpo.append(ut)

    def test_hermitian_mpo_gives_real_expectation(self):
        """<psi|H|psi> is real and positive for complex MPS and Hermitian H."""
        val = complex(expectation(self.psi, self.I_mpo, self.psi))
        self.assertAlmostEqual(val.imag, 0.0, places=10)
        self.assertGreater(val.real, 0.0)

    def test_conjugate_symmetry(self):
        """<psi|H|phi> = conj(<phi|H|psi>) for Hermitian H."""
        psi_phi = complex(expectation(self.psi, self.I_mpo, self.phi))
        phi_psi = complex(expectation(self.phi, self.I_mpo, self.psi))
        self.assertAlmostEqual(psi_phi.real,  phi_psi.real,  places=10)
        self.assertAlmostEqual(psi_phi.imag, -phi_psi.imag,  places=10)


if __name__ == "__main__":
    unittest.main()
