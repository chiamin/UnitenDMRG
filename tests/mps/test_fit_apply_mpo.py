"""Unit tests for `fit_apply_mpo` in mps_operations.py.

Coverage
--------
1. Input validation  (TestFitApplyMpoValidation)
   - length mismatch → ValueError
   - num_center not 1 or 2 → ValueError
   - fitmps.center != 0 → ValueError

2. Output structure  (TestFitApplyMpoStructure)
   - center is 0 after a complete sweep, num_center=1 and num_center=2
   - max_dim is respected: all bond dimensions ≤ max_dim
   - normalize=True: ‖fitmps‖ ≈ 1

3. Identity MPO  (TestFitApplyMpoIdentityMpo)
   - 1-site: I|ψ> ≈ |ψ>, |⟨fitmps|mps_input⟩| / ‖fitmps‖ ≈ 1
   - 2-site: same

4. Scalar MPO  (TestFitApplyMpoScalarMpo)
   - (c*I)|ψ> = c^N |ψ>: direction preserved and scale correct
   - 1-site and 2-site

5. Heisenberg MPO  (TestFitApplyMpoHeisenberg)
   - Consistency identity: expectation(fitmps, H, mps_input) ≈ inner(fitmps, fitmps)
     (holds exactly when |fitmps> = H|mps_input>)
   - num_center=1 and num_center=2

6. Dense MPS — all dtype combinations  (TestFitApplyMpoComplex)
   real H:    <real|real H|real>, <real|real H|complex>,
              <complex|real H|real>, <complex|real H|complex>
   complex H: <complex|complex H|complex>

7. QN (U1 Sz) MPS via mps_sum  (TestFitApplyMpoQN)
   - Real QN MPS + QN MPO: consistency identity
   - Complex QN MPS + QN MPO: consistency identity (skipped — cytnx bug contract-mixed-dtype)

8. QN (U1 Sz) MPS via random_u1_sz_mps — all dtype combinations  (TestFitApplyMpoQNDirect)
   real H:    <real|real H|real>, <real|real H|complex>,
              <complex|real H|real>, <complex|real H|complex>
   complex H: <complex|complex H|complex>
"""

from __future__ import annotations

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
    from MPS.mps import MPS
    from MPS.mpo import MPO
    from MPS.mps_init import random_mps
    from MPS.mps_operations import expectation, fit_apply_mpo, inner, mps_sum
    from MPS.auto_mpo import AutoMPO
    from MPS.physical_sites.spin_half import spin_half
    from tests.helpers.heisenberg import heisenberg_mpo
    from tests.helpers.mps_test_cases import random_u1_sz_mps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_mpo(N: int, d: int, dtype=float) -> "MPO":
    """Identity MPO with virtual bond dimension 1."""
    np_dtype = np.complex128 if np.issubdtype(np.dtype(dtype), np.complexfloating) else np.float64
    tensors = []
    for _ in range(N):
        arr = np.eye(d, dtype=np_dtype).reshape(1, d, d, 1)
        ut = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=2)
        ut.set_labels(["l", "ip", "i", "r"])
        tensors.append(ut)
    return MPO(tensors)


def _scalar_mpo(N: int, d: int, c: float) -> "MPO":
    """MPO equal to c * I at every site (virtual bond dimension 1).

    Applying this to |ψ> gives c^N |ψ>, since each site contributes a
    factor of c independently.
    """
    tensors = []
    for _ in range(N):
        arr = (c * np.eye(d, dtype=float)).reshape(1, d, d, 1)
        ut = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=2)
        ut.set_labels(["l", "ip", "i", "r"])
        tensors.append(ut)
    return MPO(tensors)


def _cosine_similarity(a: "MPS", b: "MPS") -> float:
    """Return |⟨a|b⟩| / (‖a‖ * ‖b‖) as a float in [0, 1]."""
    ov    = abs(complex(inner(a, b)))
    nrm_a = abs(complex(inner(a, a))) ** 0.5
    nrm_b = abs(complex(inner(b, b))) ** 0.5
    return float(ov / (nrm_a * nrm_b))


def _qn_heisenberg_mpo(N: int) -> "MPO":
    """QN Heisenberg MPO built via AutoMPO."""
    site = spin_half(qn="Sz")
    ampo = AutoMPO(N, site)
    for i in range(N - 1):
        ampo.add(1.0, "Sz", i, "Sz", i + 1)
        ampo.add(0.5, "Sp", i, "Sm", i + 1)
        ampo.add(0.5, "Sm", i, "Sp", i + 1)
    return ampo.to_mpo()


def _qn_heisenberg_mpo_complex(N: int) -> "MPO":
    """QN Heisenberg MPO cast to ComplexDouble."""
    H = _qn_heisenberg_mpo(N)
    return MPO([t.astype(cytnx.Type.ComplexDouble) for t in H.tensors])


def _consistency_check(self, fitmps, H, mps_input, places=6):
    """Assert ⟨fitmps|H|mps_input⟩ ≈ ⟨fitmps|fitmps⟩."""
    lhs = complex(expectation(fitmps, H, mps_input))
    rhs = complex(inner(fitmps, fitmps))
    self.assertAlmostEqual(lhs.real, rhs.real, places=places)
    self.assertAlmostEqual(lhs.imag, 0.0,      places=places)


# ===========================================================================
# 1. Input validation
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx is required for fit_apply_mpo tests")
class TestFitApplyMpoValidation(unittest.TestCase):
    """Correct errors are raised for invalid arguments."""

    def setUp(self):
        N, d, D = 4, 2, 3
        self.mps_input = random_mps(N, d, D, seed=0, normalize=True)
        self.fitmps    = random_mps(N, d, D, seed=1, normalize=True)
        self.mpo       = _identity_mpo(N, d)

    def test_length_mismatch_raises(self):
        """MPO shorter than MPS → ValueError."""
        short_mpo = _identity_mpo(3, 2)
        with self.assertRaises(ValueError):
            fit_apply_mpo(short_mpo, self.mps_input, self.fitmps)

    def test_invalid_num_center_raises(self):
        """num_center=3 → ValueError."""
        with self.assertRaises(ValueError):
            fit_apply_mpo(self.mpo, self.mps_input, self.fitmps, num_center=3)

    def test_fitmps_center_not_zero_raises(self):
        """fitmps.center != 0 → ValueError."""
        fitmps = random_mps(4, 2, 3, seed=2, normalize=True)
        fitmps.move_center(2)
        with self.assertRaises(ValueError):
            fit_apply_mpo(self.mpo, self.mps_input, fitmps)


# ===========================================================================
# 2. Output structure
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx is required for fit_apply_mpo tests")
class TestFitApplyMpoStructure(unittest.TestCase):
    """Structural properties of fitmps after the sweep."""

    def _run(self, num_center, max_dim=None, normalize=False):
        N, d, D = 4, 2, 3
        mps_input = random_mps(N, d, D, seed=10, normalize=True)
        fitmps    = random_mps(N, d, D, seed=11, normalize=True)
        mpo       = _identity_mpo(N, d)
        fit_apply_mpo(mpo, mps_input, fitmps, num_center=num_center,
                      nsweep=1, max_dim=max_dim, normalize=normalize)
        return fitmps

    def test_center_restored_1site(self):
        """After one sweep with num_center=1, fitmps.center is 0."""
        fitmps = self._run(num_center=1)
        self.assertEqual(fitmps.center, 0)

    def test_center_restored_2site(self):
        """After one sweep with num_center=2, fitmps.center is 0."""
        fitmps = self._run(num_center=2)
        self.assertEqual(fitmps.center, 0)

    def test_max_dim_respected(self):
        """All bond dimensions are ≤ max_dim after the sweep."""
        fitmps = self._run(num_center=2, max_dim=2)
        self.assertTrue(
            all(d <= 2 for d in fitmps.bond_dims),
            f"bond_dims exceeded max_dim=2: {fitmps.bond_dims}",
        )

    def test_normalize_flag(self):
        """normalize=True: ‖fitmps‖² ≈ 1 after the sweep."""
        fitmps = self._run(num_center=2, normalize=True)
        norm_sq = float(complex(inner(fitmps, fitmps)).real)
        self.assertAlmostEqual(norm_sq, 1.0, places=10)


# ===========================================================================
# 3. Identity MPO
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx is required for fit_apply_mpo tests")
class TestFitApplyMpoIdentityMpo(unittest.TestCase):
    """I|ψ> = |ψ>: fitmps must align with mps_input after fitting."""

    def _cosine(self, num_center):
        N, d, D = 4, 2, 3
        mps_input = random_mps(N, d, D, seed=20, normalize=True)
        fitmps    = random_mps(N, d, D, seed=21, normalize=True)
        mpo       = _identity_mpo(N, d)
        fit_apply_mpo(mpo, mps_input, fitmps, num_center=num_center,
                      nsweep=2, max_dim=None)
        return _cosine_similarity(fitmps, mps_input)

    def test_identity_1site(self):
        """1-site fit: fitmps aligns with mps_input (cosine similarity ≈ 1)."""
        self.assertAlmostEqual(self._cosine(num_center=1), 1.0, places=6)

    def test_identity_2site(self):
        """2-site fit: fitmps aligns with mps_input (cosine similarity ≈ 1)."""
        self.assertAlmostEqual(self._cosine(num_center=2), 1.0, places=6)


# ===========================================================================
# 4. Scalar MPO
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx is required for fit_apply_mpo tests")
class TestFitApplyMpoScalarMpo(unittest.TestCase):
    """(c*I)|ψ> = c^N |ψ>: direction preserved, scale correct."""

    _C = 2.5
    _N = 4
    _D = 3

    def _run(self, num_center):
        mps_input = random_mps(self._N, 2, self._D, seed=30, normalize=True)
        fitmps    = random_mps(self._N, 2, self._D, seed=31, normalize=True)
        mpo       = _scalar_mpo(self._N, 2, self._C)
        fit_apply_mpo(mpo, mps_input, fitmps, num_center=num_center,
                      nsweep=2, max_dim=None)
        return fitmps, mps_input

    def test_direction_1site(self):
        """1-site: fitmps is parallel to mps_input (cosine similarity ≈ 1)."""
        fitmps, mps_input = self._run(num_center=1)
        self.assertAlmostEqual(_cosine_similarity(fitmps, mps_input), 1.0, places=6)

    def test_scale_1site(self):
        """1-site: ‖fitmps‖ ≈ c^N (mps_input is normalized)."""
        fitmps, _ = self._run(num_center=1)
        nrm = abs(complex(inner(fitmps, fitmps))) ** 0.5
        self.assertAlmostEqual(float(nrm), self._C ** self._N, places=5)

    def test_direction_2site(self):
        """2-site: fitmps is parallel to mps_input (cosine similarity ≈ 1)."""
        fitmps, mps_input = self._run(num_center=2)
        self.assertAlmostEqual(_cosine_similarity(fitmps, mps_input), 1.0, places=6)

    def test_scale_2site(self):
        """2-site: ‖fitmps‖ ≈ c^N (mps_input is normalized)."""
        fitmps, _ = self._run(num_center=2)
        nrm = abs(complex(inner(fitmps, fitmps))) ** 0.5
        self.assertAlmostEqual(float(nrm), self._C ** self._N, places=5)


# ===========================================================================
# 5. Heisenberg MPO
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx is required for fit_apply_mpo tests")
class TestFitApplyMpoHeisenberg(unittest.TestCase):
    """Heisenberg MPO: consistency identity expectation(fitmps, H, psi) ≈ inner(fitmps, fitmps).

    When |fitmps> = H|mps_input> exactly, ⟨fitmps|H|mps_input⟩ = ⟨fitmps|fitmps⟩
    by definition.  We use a bond dimension large enough that the variational
    fit has no approximation error for N=3 (exact bond dim ≤ 2 after SVD).
    """

    def _run(self, num_center):
        N, d, D_in, D_fit = 3, 2, 2, 10
        mps_input = random_mps(N, d, D_in,  seed=40, normalize=True)
        fitmps    = random_mps(N, d, D_fit, seed=41, normalize=True)
        H         = heisenberg_mpo(N)
        fit_apply_mpo(H, mps_input, fitmps, num_center=num_center,
                      nsweep=5, max_dim=None)
        return fitmps, mps_input, H

    def test_consistency_1site(self):
        """⟨fitmps|H|mps_input⟩ ≈ ⟨fitmps|fitmps⟩ for num_center=1."""
        fitmps, mps_input, H = self._run(num_center=1)
        _consistency_check(self, fitmps, H, mps_input)

    def test_consistency_2site(self):
        """⟨fitmps|H|mps_input⟩ ≈ ⟨fitmps|fitmps⟩ for num_center=2."""
        fitmps, mps_input, H = self._run(num_center=2)
        _consistency_check(self, fitmps, H, mps_input)


# ===========================================================================
# 6. Dense MPS — all dtype combinations
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx is required for fit_apply_mpo tests")
class TestFitApplyMpoComplex(unittest.TestCase):
    """Dense MPS: all bra/H/ket dtype combinations for fit_apply_mpo.

    Bra = fitmps, ket = mps_input.  Uses the Heisenberg MPO consistency
    identity ⟨fitmps|H|mps_input⟩ ≈ ⟨fitmps|fitmps⟩ (exact when bond
    dim is large enough for N=3).
    """

    N, d, D_in, D_fit = 3, 2, 2, 10

    def _run(self, fitmps_dtype, mps_input_dtype, H):
        mps_input = random_mps(self.N, self.d, self.D_in,
                               dtype=mps_input_dtype, seed=50, normalize=True)
        fitmps    = random_mps(self.N, self.d, self.D_fit,
                               dtype=fitmps_dtype,   seed=51, normalize=True)
        fit_apply_mpo(H, mps_input, fitmps, num_center=2, nsweep=5, max_dim=None)
        return fitmps, mps_input

    # -- real H --

    def test_real_fitmps_real_H_real_input(self):
        """<real|real H|real>: consistency identity holds."""
        H = heisenberg_mpo(self.N)
        fitmps, mps_input = self._run(float, float, H)
        _consistency_check(self, fitmps, H, mps_input)

    def test_real_fitmps_real_H_complex_input(self):
        """<real|real H|complex>: fitmps is real but target H|input> is complex."""
        H = heisenberg_mpo(self.N)
        fitmps, mps_input = self._run(float, complex, H)
        _consistency_check(self, fitmps, H, mps_input)

    def test_complex_fitmps_real_H_real_input(self):
        """<complex|real H|real>: fitmps complex, target real — imaginary part vanishes."""
        H = heisenberg_mpo(self.N)
        fitmps, mps_input = self._run(complex, float, H)
        _consistency_check(self, fitmps, H, mps_input)

    def test_complex_fitmps_real_H_complex_input(self):
        """<complex|real H|complex>: standard complex case."""
        H = heisenberg_mpo(self.N)
        fitmps, mps_input = self._run(complex, complex, H)
        _consistency_check(self, fitmps, H, mps_input)

    # -- complex H --

    def test_complex_fitmps_complex_H_complex_input(self):
        """<complex|complex H|complex>: all complex."""
        H = heisenberg_mpo(self.N, dtype=complex)
        fitmps, mps_input = self._run(complex, complex, H)
        _consistency_check(self, fitmps, H, mps_input)


# ===========================================================================
# 7. QN (U1 Sz) MPS via mps_sum
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx is required for fit_apply_mpo tests")
class TestFitApplyMpoQN(unittest.TestCase):
    """QN (U1 Sz symmetry) MPS and MPO tests via mps_sum.

    `test_complex_qn_consistency` is skipped due to cytnx bug
    contract-mixed-dtype: `mps_sum` → `direct_sum` →
    `Contract(qn_real, qn_complex)` raises dtype=3 error.
    See `_internal/CYTNX_BUGS.md`.
    """

    @staticmethod
    def _heisenberg_qn_mpo(N: int):
        return _qn_heisenberg_mpo(N)

    @staticmethod
    def _qn_superposition(states1, states2, dtype=float):
        """Build a normalized QN MPS as a superposition of two product states."""
        site = spin_half(qn="Sz")
        psi1 = site.product_state(states1, dtype=dtype)
        psi2 = site.product_state(states2, dtype=dtype)
        mps  = mps_sum(psi1, psi2)
        mps.move_center(0)
        mps.normalize()
        return mps

    def test_real_qn_consistency(self):
        """Real QN MPS + QN MPO: ⟨fitmps|H|mps_input⟩ ≈ ⟨fitmps|fitmps⟩."""
        N = 3
        mps_input = self._qn_superposition([1, 0, 1], [0, 1, 1])
        fitmps    = self._qn_superposition([1, 1, 0], [1, 0, 1])
        H = self._heisenberg_qn_mpo(N)
        fit_apply_mpo(H, mps_input, fitmps, num_center=2, nsweep=5, max_dim=None)
        _consistency_check(self, fitmps, H, mps_input)

    @unittest.skip("cytnx bug contract-mixed-dtype: Contract(qn_real, qn_complex) "
                   "fails in mps_sum/direct_sum. See _internal/CYTNX_BUGS.md.")
    def test_complex_qn_consistency(self):
        """Complex QN MPS + QN MPO: ⟨fitmps|H|mps_input⟩ ≈ ⟨fitmps|fitmps⟩."""
        N = 3
        mps_input = self._qn_superposition([1, 0, 1], [0, 1, 1], dtype=complex)
        fitmps    = self._qn_superposition([1, 1, 0], [1, 0, 1], dtype=complex)
        H = self._heisenberg_qn_mpo(N)
        fit_apply_mpo(H, mps_input, fitmps, num_center=2, nsweep=5, max_dim=None)
        _consistency_check(self, fitmps, H, mps_input)


# ===========================================================================
# 8. QN (U1 Sz) MPS via random_u1_sz_mps — all dtype combinations
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx is required for fit_apply_mpo tests")
class TestFitApplyMpoQNDirect(unittest.TestCase):
    """QN MPS: all bra/H/ket dtype combinations, using random_u1_sz_mps.

    Uses random_u1_sz_mps directly to avoid the mps_sum / move_center issues
    in TestFitApplyMpoQN.  N=4, n_up=2, bond dim=1 per sector.
    """

    N, n_up = 4, 2

    def _run(self, fitmps_dtype, mps_input_dtype, H):
        mps_input = random_u1_sz_mps(self.N, self.n_up,
                                     seed=60, dtype=mps_input_dtype)
        fitmps    = random_u1_sz_mps(self.N, self.n_up,
                                     seed=61, dtype=fitmps_dtype)
        fit_apply_mpo(H, mps_input, fitmps, num_center=2, nsweep=5, max_dim=None)
        return fitmps, mps_input

    # -- real H --

    def test_real_fitmps_real_H_real_input(self):
        """<real|real H|real>: QN consistency identity."""
        H = _qn_heisenberg_mpo(self.N)
        fitmps, mps_input = self._run(float, float, H)
        _consistency_check(self, fitmps, H, mps_input)

    def test_real_fitmps_real_H_complex_input(self):
        """<real|real H|complex>: fitmps real, input complex."""
        H = _qn_heisenberg_mpo(self.N)
        fitmps, mps_input = self._run(float, complex, H)
        _consistency_check(self, fitmps, H, mps_input)

    def test_complex_fitmps_real_H_real_input(self):
        """<complex|real H|real>: fitmps complex, input real."""
        H = _qn_heisenberg_mpo(self.N)
        fitmps, mps_input = self._run(complex, float, H)
        _consistency_check(self, fitmps, H, mps_input)

    def test_complex_fitmps_real_H_complex_input(self):
        """<complex|real H|complex>: standard complex QN case."""
        H = _qn_heisenberg_mpo(self.N)
        fitmps, mps_input = self._run(complex, complex, H)
        _consistency_check(self, fitmps, H, mps_input)

    # -- complex H --

    def test_complex_fitmps_complex_H_complex_input(self):
        """<complex|complex H|complex>: all complex QN."""
        H = _qn_heisenberg_mpo_complex(self.N)
        fitmps, mps_input = self._run(complex, complex, H)
        _consistency_check(self, fitmps, H, mps_input)


if __name__ == "__main__":
    unittest.main()
