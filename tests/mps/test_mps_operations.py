"""Unit tests for MPS/mps_operations.py.

Coverage
--------
1. Dense MPS-MPS inner product — all 4 dtype combinations
   <real|real>, <real|complex>, <complex|real>, <complex|complex>

2. Dense MPS-MPO-MPS expectation — all 5 dtype combinations
   <real|realH|real>, <real|realH|complex>, <complex|realH|real>,
   <complex|realH|complex>, <complex|complexH|complex>

3. QN MPS-MPS inner product — all 4 dtype combinations

4. QN MPS-MPO-MPS expectation — all 5 dtype combinations

5. mps_sum — dense (real, complex) + QN (real; complex skipped: contract-mixed-dtype)

6. mpo_sum — dense (real, complex) + QN (real; complex skipped: contract-mixed-dtype)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import cytnx
except ImportError:  # pragma: no cover
    cytnx = None

THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent.parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

if cytnx is not None:
    from MPS.mps import MPS
    from MPS.mpo import MPO
    from MPS.mps_init import random_mps
    from MPS.mps_operations import expectation, inner, mps_sum, mpo_sum
    from MPS.physical_sites.spin_half import spin_half
    from MPS.auto_mpo import AutoMPO
    from tests.helpers.mps_test_cases import random_u1_sz_mps


# ===========================================================================
# Helpers
# ===========================================================================

N, D, d = 4, 3, 2


def _dense_mps(dtype=float, seed=0):
    """Normalised random dense MPS."""
    return random_mps(N, d, D, seed=seed, dtype=dtype, normalize=True)


def _dense_identity_mpo(dtype=float):
    """Dense identity MPO (bond dim 1)."""
    np_dtype = np.complex128 if np.issubdtype(np.dtype(dtype), np.complexfloating) else np.float64
    tensors = []
    for _ in range(N):
        arr = np.zeros((1, d, d, 1), dtype=np_dtype)
        for j in range(d):
            arr[0, j, j, 0] = 1.0
        ut = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=3)
        ut.set_labels(["l", "ip", "i", "r"])
        tensors.append(ut)
    return MPO(tensors)


def _qn_mps(dtype=float, seed=0):
    """Normalised random QN MPS (spin-1/2, 4 sites, n_up=2)."""
    return random_u1_sz_mps(
        num_sites=N, n_up_total=2, seed=seed, dtype=dtype,
        center=0, normalize=True,
    )


def _qn_heisenberg_mpo():
    """QN Heisenberg MPO via AutoMPO."""
    site = spin_half(qn="Sz")
    ampo = AutoMPO(N, site)
    for i in range(N - 1):
        ampo.add(1.0, "Sz", i, "Sz", i + 1)
        ampo.add(0.5, "Sp", i, "Sm", i + 1)
        ampo.add(0.5, "Sm", i, "Sp", i + 1)
    return ampo.to_mpo()


# ===========================================================================
# 1. Dense MPS-MPS inner product — 4 dtype combinations
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx is required")
class TestInnerDense(unittest.TestCase):
    """Dense inner product: all 4 bra/ket dtype combinations."""

    # -- <real | real> --
    def test_real_bra_real_ket(self):
        psi = _dense_mps(float, seed=1)
        phi = _dense_mps(float, seed=2)
        val = complex(inner(psi, phi))
        # Conjugate symmetry: <psi|phi> = conj(<phi|psi>)
        rev = complex(inner(phi, psi))
        self.assertAlmostEqual(val.real, rev.real, places=10)
        self.assertAlmostEqual(val.imag, -rev.imag, places=10)

    # -- <real | complex> --
    def test_real_bra_complex_ket(self):
        bra = _dense_mps(float, seed=3)
        ket = _dense_mps(complex, seed=4)
        val = complex(inner(bra, ket))
        rev = complex(inner(ket, bra))
        self.assertAlmostEqual(val.real, rev.real, places=10)
        self.assertAlmostEqual(val.imag, -rev.imag, places=10)

    # -- <complex | real> --
    def test_complex_bra_real_ket(self):
        bra = _dense_mps(complex, seed=5)
        ket = _dense_mps(float, seed=6)
        val = complex(inner(bra, ket))
        rev = complex(inner(ket, bra))
        self.assertAlmostEqual(val.real, rev.real, places=10)
        self.assertAlmostEqual(val.imag, -rev.imag, places=10)

    # -- <complex | complex> --
    def test_complex_bra_complex_ket(self):
        psi = _dense_mps(complex, seed=7)
        val = complex(inner(psi, psi))
        self.assertAlmostEqual(val.imag, 0.0, places=10)
        self.assertGreater(val.real, 0.0)


# ===========================================================================
# 2. Dense MPS-MPO-MPS expectation — 5 dtype combinations
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx is required")
class TestExpectationDense(unittest.TestCase):
    """Dense expectation: all 5 bra/H/ket dtype combinations."""

    # -- <real | real H | real> --
    def test_real_bra_real_H_real_ket(self):
        psi = _dense_mps(float, seed=10)
        mpo = _dense_identity_mpo(float)
        val = complex(expectation(psi, mpo, psi))
        ref = complex(inner(psi, psi))
        self.assertAlmostEqual(val.real, ref.real, places=10)
        self.assertAlmostEqual(val.imag, ref.imag, places=10)

    # -- <real | real H | complex> --
    def test_real_bra_real_H_complex_ket(self):
        bra = _dense_mps(float, seed=11)
        ket = _dense_mps(complex, seed=12)
        mpo = _dense_identity_mpo(float)
        val = complex(expectation(bra, mpo, ket))
        ref = complex(inner(bra, ket))
        self.assertAlmostEqual(val.real, ref.real, places=10)
        self.assertAlmostEqual(val.imag, ref.imag, places=10)

    # -- <complex | real H | real> --
    def test_complex_bra_real_H_real_ket(self):
        bra = _dense_mps(complex, seed=13)
        ket = _dense_mps(float, seed=14)
        mpo = _dense_identity_mpo(float)
        val = complex(expectation(bra, mpo, ket))
        ref = complex(inner(bra, ket))
        self.assertAlmostEqual(val.real, ref.real, places=10)
        self.assertAlmostEqual(val.imag, ref.imag, places=10)

    # -- <complex | real H | complex> --
    def test_complex_bra_real_H_complex_ket(self):
        psi = _dense_mps(complex, seed=15)
        mpo = _dense_identity_mpo(float)
        val = complex(expectation(psi, mpo, psi))
        self.assertAlmostEqual(val.imag, 0.0, places=10)
        self.assertGreater(val.real, 0.0)

    # -- <complex | complex H | complex> --
    def test_complex_bra_complex_H_complex_ket(self):
        psi = _dense_mps(complex, seed=16)
        mpo = _dense_identity_mpo(complex)
        val = complex(expectation(psi, mpo, psi))
        self.assertAlmostEqual(val.imag, 0.0, places=10)
        self.assertGreater(val.real, 0.0)


# ===========================================================================
# 3. QN MPS-MPS inner product — 4 dtype combinations
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx is required")
class TestInnerQN(unittest.TestCase):
    """QN inner product: all 4 bra/ket dtype combinations."""

    # -- <real | real> --
    def test_real_bra_real_ket(self):
        psi = _qn_mps(float, seed=100)
        phi = _qn_mps(float, seed=101)
        val = complex(inner(psi, phi))
        rev = complex(inner(phi, psi))
        self.assertAlmostEqual(val.real, rev.real, places=10)
        self.assertAlmostEqual(val.imag, -rev.imag, places=10)

    # -- <real | complex> --
    def test_real_bra_complex_ket(self):
        bra = _qn_mps(float, seed=102)
        ket = _qn_mps(complex, seed=103)
        val = complex(inner(bra, ket))
        rev = complex(inner(ket, bra))
        self.assertAlmostEqual(val.real, rev.real, places=10)
        self.assertAlmostEqual(val.imag, -rev.imag, places=10)

    # -- <complex | real> --
    def test_complex_bra_real_ket(self):
        bra = _qn_mps(complex, seed=104)
        ket = _qn_mps(float, seed=105)
        val = complex(inner(bra, ket))
        rev = complex(inner(ket, bra))
        self.assertAlmostEqual(val.real, rev.real, places=10)
        self.assertAlmostEqual(val.imag, -rev.imag, places=10)

    # -- <complex | complex> --
    def test_complex_bra_complex_ket(self):
        psi = _qn_mps(complex, seed=106)
        val = complex(inner(psi, psi))
        self.assertAlmostEqual(val.imag, 0.0, places=10)
        self.assertGreater(val.real, 0.0)


# ===========================================================================
# 4. QN MPS-MPO-MPS expectation — 5 dtype combinations
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx is required")
class TestExpectationQN(unittest.TestCase):
    """QN expectation: all 5 bra/H/ket dtype combinations."""

    # -- <real | real H | real> --
    def test_real_bra_real_H_real_ket(self):
        psi = _qn_mps(float, seed=110)
        mpo = _qn_heisenberg_mpo()
        val = complex(expectation(psi, mpo, psi))
        self.assertAlmostEqual(val.imag, 0.0, places=10)
        self.assertTrue(np.isfinite(val.real))

    # -- <real | real H | complex> --
    def test_real_bra_real_H_complex_ket(self):
        bra = _qn_mps(float, seed=111)
        ket = _qn_mps(complex, seed=112)
        mpo = _qn_heisenberg_mpo()
        val = complex(expectation(bra, mpo, ket))
        rev = complex(expectation(ket, mpo, bra))
        self.assertAlmostEqual(val.real, rev.real, places=10)
        self.assertAlmostEqual(val.imag, -rev.imag, places=10)

    # -- <complex | real H | real> --
    def test_complex_bra_real_H_real_ket(self):
        bra = _qn_mps(complex, seed=113)
        ket = _qn_mps(float, seed=114)
        mpo = _qn_heisenberg_mpo()
        val = complex(expectation(bra, mpo, ket))
        rev = complex(expectation(ket, mpo, bra))
        self.assertAlmostEqual(val.real, rev.real, places=10)
        self.assertAlmostEqual(val.imag, -rev.imag, places=10)

    # -- <complex | real H | complex> --
    def test_complex_bra_real_H_complex_ket(self):
        psi = _qn_mps(complex, seed=115)
        mpo = _qn_heisenberg_mpo()
        val = complex(expectation(psi, mpo, psi))
        self.assertAlmostEqual(val.imag, 0.0, places=10)
        self.assertTrue(np.isfinite(val.real))

    # -- <complex | complex H | complex> --
    def test_complex_bra_complex_H_complex_ket(self):
        psi = _qn_mps(complex, seed=116)
        mpo = _qn_heisenberg_mpo()
        mpo_c = MPO([t.astype(cytnx.Type.ComplexDouble) for t in mpo.tensors])
        val = complex(expectation(psi, mpo_c, psi))
        self.assertAlmostEqual(val.imag, 0.0, places=10)
        self.assertTrue(np.isfinite(val.real))


# ===========================================================================
# 5. mps_sum — dense + QN
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx is required")
class TestMPSSumDense(unittest.TestCase):
    """Dense tests for mps_sum."""

    N, d, D = 4, 2, 3

    def _check_factorization(self, dtype):
        """inner(sum(α,β), sum(φ,χ)) == inner(α,φ) + inner(α,χ) + inner(β,φ) + inner(β,χ)."""
        alpha = random_mps(self.N, self.d, self.D, seed=1, dtype=dtype, normalize=True)
        beta  = random_mps(self.N, self.d, self.D, seed=2, dtype=dtype, normalize=True)
        phi   = random_mps(self.N, self.d, self.D, seed=3, dtype=dtype, normalize=True)
        chi   = random_mps(self.N, self.d, self.D, seed=4, dtype=dtype, normalize=True)

        sumAB = mps_sum(alpha, beta)
        sumPC = mps_sum(phi, chi)

        lhs = complex(inner(sumAB, sumPC))
        rhs = complex(inner(alpha, phi) + inner(alpha, chi)
                       + inner(beta, phi) + inner(beta, chi))
        self.assertAlmostEqual(lhs.real, rhs.real, places=10)
        self.assertAlmostEqual(lhs.imag, rhs.imag, places=10)

    def test_real_factorization(self):
        """Dense real mps_sum: inner product factorizes."""
        self._check_factorization(float)

    def test_complex_factorization(self):
        """Dense complex mps_sum: inner product factorizes."""
        self._check_factorization(complex)

    def test_bond_dims(self):
        """Virtual bond dims of mps_sum equal the sum of the two MPS bond dims."""
        psi = random_mps(self.N, self.d, self.D, seed=5)
        phi = random_mps(self.N, self.d, self.D, seed=6)
        result = mps_sum(psi, phi)
        for k in range(len(psi) - 1):
            expected = psi[k].bond("r").dim() + phi[k].bond("r").dim()
            self.assertEqual(result[k].bond("r").dim(), expected)


@unittest.skipIf(cytnx is None, "cytnx is required")
class TestMPSSumQN(unittest.TestCase):
    """QN tests for mps_sum."""

    def test_real_inner_product_factorizes(self):
        """inner(sum(α,β), sum(φ,χ)) == inner(α,φ) + inner(β,χ) for orthogonal QN product states."""
        site = spin_half(qn="Sz")
        alpha = site.product_state([1, 0, 1, 0])
        phi   = site.product_state([1, 0, 1, 0])
        beta  = site.product_state([0, 1, 0, 1])
        chi   = site.product_state([0, 1, 0, 1])

        sumAB = mps_sum(alpha, beta)
        sumPC = mps_sum(phi, chi)

        self.assertAlmostEqual(
            inner(sumAB, sumPC),
            inner(alpha, phi) + inner(beta, chi),
            places=10,
        )

    @unittest.skip("cytnx bug contract-mixed-dtype: direct_sum expand tensor is "
                   "real, Contract(qn_real, qn_complex) fails. "
                   "See _internal/CYTNX_BUGS.md.")
    def test_complex_inner_product_factorizes(self):
        """QN complex mps_sum: inner product factorizes."""
        psi1 = random_u1_sz_mps(4, 2, seed=10, dtype=complex, normalize=True)
        psi2 = random_u1_sz_mps(4, 2, seed=11, dtype=complex, normalize=True)
        phi1 = random_u1_sz_mps(4, 2, seed=12, dtype=complex, normalize=True)
        phi2 = random_u1_sz_mps(4, 2, seed=13, dtype=complex, normalize=True)

        sumA = mps_sum(psi1, psi2)
        sumB = mps_sum(phi1, phi2)

        lhs = complex(inner(sumA, sumB))
        rhs = complex(inner(psi1, phi1) + inner(psi1, phi2)
                       + inner(psi2, phi1) + inner(psi2, phi2))
        self.assertAlmostEqual(lhs.real, rhs.real, places=10)
        self.assertAlmostEqual(lhs.imag, rhs.imag, places=10)

    def test_bond_dims(self):
        """Virtual bond dims of mps_sum equal the sum of the two MPS bond dims."""
        site = spin_half(qn="Sz")
        psi = site.product_state([1, 0, 1, 0])
        phi = site.product_state([0, 1, 0, 1])
        result = mps_sum(psi, phi)
        for k in range(len(psi) - 1):
            expected = psi[k].bond("r").dim() + phi[k].bond("r").dim()
            self.assertEqual(result[k].bond("r").dim(), expected)


# ===========================================================================
# 6. mpo_sum — dense + QN
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx is required")
class TestMPOSumDense(unittest.TestCase):
    """Dense tests for mpo_sum."""

    def _check_factorization(self, dtype):
        """expectation(ψ, H1+H2, ψ) == expectation(ψ, H1, ψ) + expectation(ψ, H2, ψ)."""
        psi = _dense_mps(dtype, seed=1)
        H1 = _dense_identity_mpo(dtype)
        H2 = _dense_identity_mpo(dtype)
        H_sum = mpo_sum(H1, H2)

        lhs = complex(expectation(psi, H_sum, psi))
        rhs = complex(expectation(psi, H1, psi) + expectation(psi, H2, psi))
        self.assertAlmostEqual(lhs.real, rhs.real, places=10)
        self.assertAlmostEqual(lhs.imag, rhs.imag, places=10)

    def test_real_factorization(self):
        """Dense real mpo_sum: expectation factorizes."""
        self._check_factorization(float)

    def test_complex_factorization(self):
        """Dense complex mpo_sum: expectation factorizes."""
        self._check_factorization(complex)


@unittest.skipIf(cytnx is None, "cytnx is required")
class TestMPOSumQN(unittest.TestCase):
    """QN tests for mpo_sum."""

    def test_real_expectation_factorizes(self):
        """expectation(ψ, mpo_sum(H1, H2), ψ) == expectation(ψ, H1, ψ) + expectation(ψ, H2, ψ)."""
        site = spin_half(qn="Sz")
        J = 1.0

        ampo_ising = AutoMPO(N, site)
        ampo_xy    = AutoMPO(N, site)
        for i in range(N - 1):
            ampo_ising.add(J,       "Sz", i, "Sz", i + 1)
            ampo_xy.add(J / 2, "Sp", i, "Sm", i + 1)
            ampo_xy.add(J / 2, "Sm", i, "Sp", i + 1)
        H_ising = ampo_ising.to_mpo()
        H_xy    = ampo_xy.to_mpo()
        H_sum   = mpo_sum(H_ising, H_xy)

        psi = site.product_state([1, 0, 1, 0])

        self.assertAlmostEqual(
            expectation(psi, H_sum, psi),
            expectation(psi, H_ising, psi) + expectation(psi, H_xy, psi),
            places=10,
        )

    @unittest.skip("cytnx bug contract-mixed-dtype: direct_sum expand tensor is "
                   "real, Contract(qn_real, qn_complex) fails. "
                   "See _internal/CYTNX_BUGS.md.")
    def test_complex_expectation_factorizes(self):
        """QN complex mpo_sum: expectation factorizes."""
        psi = _qn_mps(complex, seed=20)

        site = spin_half(qn="Sz")
        ampo_ising = AutoMPO(N, site)
        ampo_xy    = AutoMPO(N, site)
        for i in range(N - 1):
            ampo_ising.add(1.0, "Sz", i, "Sz", i + 1)
            ampo_xy.add(0.5, "Sp", i, "Sm", i + 1)
            ampo_xy.add(0.5, "Sm", i, "Sp", i + 1)
        H_ising = ampo_ising.to_mpo()
        H_xy    = ampo_xy.to_mpo()
        H_sum   = mpo_sum(H_ising, H_xy)

        lhs = complex(expectation(psi, H_sum, psi))
        rhs = complex(expectation(psi, H_ising, psi) + expectation(psi, H_xy, psi))
        self.assertAlmostEqual(lhs.real, rhs.real, places=10)
        self.assertAlmostEqual(lhs.imag, rhs.imag, places=10)


if __name__ == "__main__":
    unittest.main()
