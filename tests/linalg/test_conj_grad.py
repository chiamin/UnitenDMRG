"""Unit tests for linalg.conj_grad (cg).

Coverage
--------
1. Real symmetric positive-definite A: matches numpy.linalg.solve
2. Complex Hermitian positive-definite A: catches missing .Dagger()
3. b = 0 → x = 0 early exit
4. Initial guess close to solution
5. Insufficient k → warning is emitted
6. Indefinite A → warning is emitted (sanity check on the PD guard)
"""

from __future__ import annotations

import unittest
import warnings

import numpy as np

from linalg import cg

from ._helpers import vec, to_np, make_apply


class TestCG(unittest.TestCase):

    def _check_solves(self, M, b_np, k=200, tol=1.e-10):
        x, res = cg(make_apply(M), vec(b_np), k=k, tol=tol)
        x_np = to_np(x)
        x_ref = np.linalg.solve(M, b_np)
        self.assertLess(res, 1.e-8, f"reported residual {res:.3e}")
        np.testing.assert_allclose(x_np, x_ref, atol=1.e-7, rtol=1.e-7)

    # ------------------------------------------------------------------
    # 1. Real SPD
    # ------------------------------------------------------------------
    def test_real_spd(self):
        rng = np.random.default_rng(0)
        n = 8
        A = rng.standard_normal((n, n))
        M = A.T @ A + np.eye(n)        # SPD
        b = rng.standard_normal(n)
        self._check_solves(M.astype(complex), b.astype(complex))

    # ------------------------------------------------------------------
    # 2. Complex Hermitian PD — catches missing Dagger
    # ------------------------------------------------------------------
    def test_complex_hermitian_pd(self):
        rng = np.random.default_rng(1)
        n = 8
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        M = A.conj().T @ A + np.eye(n)
        b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        self._check_solves(M, b)

    # ------------------------------------------------------------------
    # 3. b = 0 → x = 0
    # ------------------------------------------------------------------
    def test_zero_rhs(self):
        n = 6
        M = np.eye(n, dtype=complex) * 2.
        b = np.zeros(n, dtype=complex)
        x, res = cg(make_apply(M), vec(b))
        np.testing.assert_allclose(to_np(x), np.zeros(n), atol=1.e-14)
        self.assertEqual(res, 0.)

    # ------------------------------------------------------------------
    # 4. Initial guess close to solution
    # ------------------------------------------------------------------
    def test_with_initial_guess(self):
        rng = np.random.default_rng(2)
        n = 8
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        M = A.conj().T @ A + np.eye(n)
        b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x_true = np.linalg.solve(M, b)
        x0_np = x_true + 1.e-3 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        x, res = cg(make_apply(M), vec(b), x0=vec(x0_np), k=200, tol=1.e-10)
        np.testing.assert_allclose(to_np(x), x_true, atol=1.e-7, rtol=1.e-7)
        self.assertLess(res, 1.e-8)

    # ------------------------------------------------------------------
    # 5. Insufficient k → warning
    # ------------------------------------------------------------------
    def test_insufficient_k_warns(self):
        rng = np.random.default_rng(3)
        n = 50
        # Make ill-conditioned SPD: large condition number
        U, _ = np.linalg.qr(rng.standard_normal((n, n)))
        evals = np.linspace(1.e-4, 1., n)
        M = (U * evals) @ U.T
        b = rng.standard_normal(n)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cg(make_apply(M.astype(complex)), vec(b.astype(complex)),
               k=3, tol=1.e-12)
            self.assertTrue(any("did not converge" in str(wi.message) for wi in w),
                            "expected non-convergence warning")

    # ------------------------------------------------------------------
    # 6. Indefinite A → PD guard fires
    # ------------------------------------------------------------------
    def test_indefinite_warns(self):
        # Diagonal with mixed signs.
        n = 5
        M = np.diag([1., -2., 3., -4., 5.]).astype(complex)
        b = np.ones(n, dtype=complex)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cg(make_apply(M), vec(b), k=20, tol=1.e-12)
            self.assertTrue(
                any("positive definite" in str(wi.message)
                    or "did not converge" in str(wi.message) for wi in w),
                "expected PD-violation or non-convergence warning",
            )


if __name__ == "__main__":
    unittest.main()
