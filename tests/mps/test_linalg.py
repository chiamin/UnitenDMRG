"""Unit tests for MPS/linalg.py.

Coverage
--------
1. Real-valued tests (currently enabled)
   - inner: norm, symmetry, linearity
   - lanczos: energy/eigenvector/canonical convergence checks on real-symmetric H

2. Complex-valued tests (kept separate, currently disabled)
   - inner: conjugate symmetry and sesquilinearity
   - lanczos: complex-Hermitian operator checks
"""

from __future__ import annotations

import os
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
except ImportError:
    cytnx = None

if cytnx is not None:
    from MPS.linalg import inner, lanczos

RUN_COMPLEX_TESTS = os.environ.get("RUN_COMPLEX_TESTS", "").lower() in {
    "1", "true", "yes", "on"
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vec_real(arr: "np.ndarray") -> "cytnx.UniTensor":
    """Wrap a real 1-D numpy array into a rank-1 UniTensor with label 'x'."""
    t = cytnx.UniTensor(cytnx.from_numpy(arr.astype(float)), rowrank=1)
    t.set_labels(["x"])
    return t


def _vec_complex(arr: "np.ndarray") -> "cytnx.UniTensor":
    """Wrap a complex 1-D numpy array into a rank-1 UniTensor with label 'x'."""
    t = cytnx.UniTensor(cytnx.from_numpy(arr.astype(complex)), rowrank=1)
    t.set_labels(["x"])
    return t


def _random_symmetric(n: int, seed: int = 0) -> "np.ndarray":
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return (A + A.T) / 2.0


def _random_hermitian(n: int, seed: int = 0) -> "np.ndarray":
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return (A + A.conj().T) / 2.0


def _matvec_func(H_np: "np.ndarray", vec_builder):
    """Return a callable v -> H @ v that works on rank-1 UniTensors."""
    def _apply(v: "cytnx.UniTensor") -> "cytnx.UniTensor":
        x = v.get_block().numpy().ravel()
        y = H_np @ x
        return vec_builder(y)
    return _apply


# ===========================================================================
# 1. inner (real)
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestInnerReal(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)
        self.v_np = rng.standard_normal(8)
        self.w_np = rng.standard_normal(8)
        self.v = _vec_real(self.v_np)
        self.w = _vec_real(self.w_np)

    def test_self_inner_equals_norm_squared(self):
        """<v|v> = ||v||^2."""
        result = inner(self.v, self.v)
        expected = float(np.dot(self.v_np.conj(), self.v_np).real)
        self.assertAlmostEqual(result.real, expected, places=10)
        self.assertAlmostEqual(abs(result.imag), 0.0, places=10)

    def test_symmetry(self):
        """For real vectors, <v|w> = <w|v>."""
        vw = inner(self.v, self.w)
        wv = inner(self.w, self.v)
        self.assertAlmostEqual(vw, wv, places=10)

    def test_linearity_in_second_arg(self):
        """<v|alpha*w> = alpha * <v|w>."""
        alpha = 2.5
        scaled_w = self.w * alpha
        lhs = inner(self.v, scaled_w)
        rhs = alpha * inner(self.v, self.w)
        self.assertAlmostEqual(lhs, rhs, places=10)

    def test_linearity_in_first_arg_for_real_scalars(self):
        """For real alpha, <alpha*v|w> = alpha * <v|w>."""
        alpha = 2.5
        scaled_v = self.v * alpha
        lhs = inner(scaled_v, self.w)
        rhs = alpha * inner(self.v, self.w)
        self.assertAlmostEqual(lhs, rhs, places=10)


# ===========================================================================
# 2. lanczos (real)
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestLanczosReal(unittest.TestCase):

    def _ground_state_numpy(self, H_np):
        evals, evecs = np.linalg.eigh(H_np)
        return float(evals[0].real), evecs[:, 0]

    def test_energy_matches_numpy(self):
        """Lanczos ground-state energy matches numpy eigensolver (real symmetric H)."""
        n = 10
        H_np = _random_symmetric(n, seed=1)
        apply = _matvec_func(H_np, _vec_real)
        rng = np.random.default_rng(7)
        v0 = _vec_real(rng.standard_normal(n))

        E0_lanczos, _ = lanczos(apply, v0, k=n)
        E0_numpy, _ = self._ground_state_numpy(H_np)

        self.assertAlmostEqual(E0_lanczos, E0_numpy, places=8)

    def test_eigenvector_is_normalised(self):
        """Returned eigenvector has unit norm."""
        n = 8
        H_np = _random_symmetric(n, seed=2)
        apply = _matvec_func(H_np, _vec_real)
        v0 = _vec_real(np.ones(n, dtype=float))

        _, psi = lanczos(apply, v0, k=n)
        norm = psi.Norm().item()
        self.assertAlmostEqual(norm, 1.0, places=8)

    def test_eigenvector_close_to_numpy(self):
        """Returned eigenvector matches numpy's (up to global phase)."""
        n = 10
        H_np = _random_symmetric(n, seed=3)
        apply = _matvec_func(H_np, _vec_real)
        rng = np.random.default_rng(99)
        v0 = _vec_real(rng.standard_normal(n))

        _, psi_l = lanczos(apply, v0, k=n)
        _, psi_np = self._ground_state_numpy(H_np)

        psi_l_np = psi_l.get_block().numpy().ravel()
        # |<psi_l | psi_np>| should be 1 up to numerical precision
        overlap = abs(np.dot(psi_l_np.conj(), psi_np))
        self.assertAlmostEqual(overlap, 1.0, places=6)

    def test_early_convergence(self):
        """When Krylov space exhausts the subspace, result is still correct."""
        # Use a diagonal matrix: Krylov subspace saturates after 1 step
        # if v0 is aligned with an eigenvector.
        n = 6
        diag = np.arange(1.0, n + 1.0)
        H_np = np.diag(diag)
        # Initial vector: pure ground-state component
        v0_np = np.zeros(n, dtype=float)
        v0_np[0] = 1.0
        apply = _matvec_func(H_np, _vec_real)
        v0 = _vec_real(v0_np)

        E0, psi = lanczos(apply, v0, k=n)
        self.assertAlmostEqual(E0, 1.0, places=10)
        self.assertAlmostEqual(psi.Norm().item(), 1.0, places=10)

    def test_eigenvector_satisfies_eigenvalue_equation(self):
        """H|psi> ≈ E0 * |psi>  (psi is a true eigenvector, not just energy-correct)."""
        n = 8
        H_np = _random_symmetric(n, seed=5)
        apply = _matvec_func(H_np, _vec_real)
        rng = np.random.default_rng(50)
        v0 = _vec_real(rng.standard_normal(n))

        E0, psi = lanczos(apply, v0, k=n)

        H_psi    = apply(psi)
        H_psi_np = H_psi.get_block().numpy().ravel()
        psi_np   = psi.get_block().numpy().ravel()

        # H|psi> should be E0 * |psi> up to numerical precision
        np.testing.assert_allclose(H_psi_np, E0 * psi_np, atol=1e-6)

    def test_k_equals_one(self):
        """k=1 Lanczos (one step) does not crash and returns a scalar."""
        n = 5
        H_np = _random_symmetric(n, seed=4)
        apply = _matvec_func(H_np, _vec_real)
        v0 = _vec_real(np.ones(n, dtype=float))

        # Must not raise
        E0, psi = lanczos(apply, v0, k=1)
        self.assertIsInstance(E0, float)
        self.assertAlmostEqual(psi.Norm().item(), 1.0, places=8)


# ===========================================================================
# 3. complex (separated; currently disabled)
# ===========================================================================

@unittest.skipUnless(
    RUN_COMPLEX_TESTS,
    "Set RUN_COMPLEX_TESTS=1 to enable complex-valued tests.",
)
@unittest.skipIf(cytnx is None, "cytnx not available")
class TestInnerComplex(unittest.TestCase):
    """Complex sesquilinear checks live separately from real-only suite."""

    def test_placeholder(self):
        rng = np.random.default_rng(123)
        v_np = rng.standard_normal(4) + 1j * rng.standard_normal(4)
        w_np = rng.standard_normal(4) + 1j * rng.standard_normal(4)
        v = _vec_complex(v_np)
        w = _vec_complex(w_np)
        self.assertAlmostEqual(inner(v, w), inner(w, v).conjugate(), places=10)


@unittest.skipUnless(
    RUN_COMPLEX_TESTS,
    "Set RUN_COMPLEX_TESTS=1 to enable complex-valued tests.",
)
@unittest.skipIf(cytnx is None, "cytnx not available")
class TestLanczosComplex(unittest.TestCase):
    """Complex Hermitian checks live separately from real-only suite."""

    def test_placeholder(self):
        n = 4
        H_np = _random_hermitian(n, seed=8)
        apply = _matvec_func(H_np, _vec_complex)
        v0 = _vec_complex(np.ones(n, dtype=complex))
        E0, _ = lanczos(apply, v0, k=n)
        E0_numpy = float(np.linalg.eigh(H_np)[0][0].real)
        self.assertAlmostEqual(E0, E0_numpy, places=8)


if __name__ == "__main__":
    unittest.main()
