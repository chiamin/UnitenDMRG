"""Unit tests for linalg.gmres.

Coverage
--------
1. Real non-symmetric A: solution matches numpy.linalg.solve
2. Complex non-Hermitian A: catches missing .Dagger() in the solver
3. Hermitian positive-definite (complex) A
4. b = 0 → x = 0 early exit
5. Initial guess close to solution → near-zero work
6. Low-rank / small dimension exact convergence (h_next < tol break path)
7. Insufficient k → warning is emitted
"""

from __future__ import annotations

import unittest
import warnings

import numpy as np

from linalg import gmres

from ._helpers import vec, to_np, make_apply


class TestGMRES(unittest.TestCase):

    def _check_solves(self, M, b_np, k=60, tol=1.e-10):
        x, res = gmres(make_apply(M), vec(b_np), k=k, tol=tol)
        x_np = to_np(x)
        x_ref = np.linalg.solve(M, b_np)
        self.assertLess(res, 1.e-8, f"reported residual {res:.3e}")
        np.testing.assert_allclose(x_np, x_ref, atol=1.e-8, rtol=1.e-8)

    # ------------------------------------------------------------------
    # 1. Real non-symmetric
    # ------------------------------------------------------------------
    def test_real_nonsymmetric(self):
        rng = np.random.default_rng(0)
        n = 8
        M = rng.standard_normal((n, n)) + 4. * np.eye(n)   # well-conditioned
        b = rng.standard_normal(n)
        self._check_solves(M.astype(complex), b.astype(complex))

    # ------------------------------------------------------------------
    # 2. Complex non-Hermitian — catches missing Dagger
    # ------------------------------------------------------------------
    def test_complex_nonhermitian(self):
        rng = np.random.default_rng(1)
        n = 8
        M = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
             + 5. * np.eye(n))
        b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        self._check_solves(M, b)

    # ------------------------------------------------------------------
    # 3. Hermitian positive definite (complex)
    # ------------------------------------------------------------------
    def test_hermitian_positive_definite(self):
        rng = np.random.default_rng(2)
        n = 8
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        M = A.conj().T @ A + np.eye(n)
        b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        self._check_solves(M, b)

    # ------------------------------------------------------------------
    # 4. b = 0 → x = 0 early exit
    # ------------------------------------------------------------------
    def test_zero_rhs(self):
        n = 6
        M = np.eye(n, dtype=complex) * 2.
        b = np.zeros(n, dtype=complex)
        x, res = gmres(make_apply(M), vec(b))
        np.testing.assert_allclose(to_np(x), np.zeros(n), atol=1.e-14)
        self.assertEqual(res, 0.)

    # ------------------------------------------------------------------
    # 5. Initial guess already close — should still produce correct x
    # ------------------------------------------------------------------
    def test_with_initial_guess(self):
        rng = np.random.default_rng(3)
        n = 8
        M = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
             + 5. * np.eye(n))
        b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x_true = np.linalg.solve(M, b)
        # perturb x_true slightly
        x0_np = x_true + 1.e-3 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        x, res = gmres(make_apply(M), vec(b), x0=vec(x0_np), k=60, tol=1.e-10)
        np.testing.assert_allclose(to_np(x), x_true, atol=1.e-8, rtol=1.e-8)
        self.assertLess(res, 1.e-8)

    # ------------------------------------------------------------------
    # 6. Exact convergence in <= n steps for small system
    # ------------------------------------------------------------------
    def test_small_dim_exact(self):
        # n=4 GMRES should converge exactly within 4 iterations.
        rng = np.random.default_rng(4)
        n = 4
        M = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
             + 3. * np.eye(n))
        b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x, res = gmres(make_apply(M), vec(b), k=n, tol=1.e-12)
        np.testing.assert_allclose(to_np(x), np.linalg.solve(M, b),
                                   atol=1.e-10, rtol=1.e-10)
        self.assertLess(res, 1.e-10)

    # ------------------------------------------------------------------
    # 7. Insufficient k triggers warning
    # ------------------------------------------------------------------
    def test_insufficient_k_warns(self):
        rng = np.random.default_rng(5)
        n = 30
        # ill-conditioned: needs many iterations
        M = rng.standard_normal((n, n)) + 0.1 * np.eye(n)
        b = rng.standard_normal(n)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gmres(make_apply(M.astype(complex)), vec(b.astype(complex)),
                  k=2, tol=1.e-12)
            self.assertTrue(any("did not converge" in str(wi.message) for wi in w),
                            "expected non-convergence warning")


if __name__ == "__main__":
    unittest.main()
