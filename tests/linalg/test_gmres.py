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

from ._helpers import (vec, to_np, make_apply,
                       make_qn_vector, make_qn_general_apply,
                       make_qn_hermitian_apply, qn_to_np, qn_dense_matrix)

try:
    import cytnx
except ImportError:
    cytnx = None


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


# ======================================================================
# QN: general non-Hermitian A (real and complex)
#
# The real-QN case is a regression test for the dtype up-cast fix in gmres:
# the complex Hessenberg coefficients must be cast back to real before
# multiplying the (real) Krylov vectors, otherwise a later
# Contract(A_real, x_complex) hits the cytnx QN mixed-dtype bug.
# ======================================================================

L_SEC, L_DEG = [0, 1, 2], [3, 3, 2]
I_SEC, I_DEG = [0, 1], [2, 2]


@unittest.skipIf(cytnx is None, "cytnx not available")
class TestGMRESQN(unittest.TestCase):

    def _solve_and_check(self, apply, b, expect_complex_x):
        M = qn_dense_matrix(apply, b)
        x, res = gmres(apply, b, k=M.shape[0] + 2, tol=1e-12)
        x_np = qn_to_np(x)
        x_ref = np.linalg.solve(M, qn_to_np(b))
        self.assertLess(res, 1e-8, f"reported residual {res:.3e}")
        np.testing.assert_allclose(x_np, x_ref, atol=1e-7, rtol=1e-7)
        # dtype must follow the input: a real problem stays real (the fix)
        is_complex = x.dtype() == cytnx.Type.ComplexDouble
        self.assertEqual(is_complex, expect_complex_x)
        self.assertEqual(list(x.labels()), list(b.labels()))
        self.assertEqual(x.Nblocks(), b.Nblocks())

    def test_real_nonhermitian(self):
        b = make_qn_vector(L_SEC, L_DEG, I_SEC, I_DEG, "real", seed=10)
        apply, _ = make_qn_general_apply(L_SEC, L_DEG, I_SEC, I_DEG,
                                         "real", seed=11, shift=4.0)
        self._solve_and_check(apply, b, expect_complex_x=False)

    def test_complex_nonhermitian(self):
        b = make_qn_vector(L_SEC, L_DEG, I_SEC, I_DEG, "complex", seed=20)
        apply, _ = make_qn_general_apply(L_SEC, L_DEG, I_SEC, I_DEG,
                                         "complex", seed=21, shift=4.0)
        self._solve_and_check(apply, b, expect_complex_x=True)

    def test_real_hermitian_pd(self):
        b = make_qn_vector(L_SEC, L_DEG, I_SEC, I_DEG, "real", seed=30)
        apply, _ = make_qn_hermitian_apply(L_SEC, L_DEG, I_SEC, I_DEG,
                                           "real", seed=31, shift=1.0)
        self._solve_and_check(apply, b, expect_complex_x=False)

    def test_complex_hermitian_pd(self):
        b = make_qn_vector(L_SEC, L_DEG, I_SEC, I_DEG, "complex", seed=40)
        apply, _ = make_qn_hermitian_apply(L_SEC, L_DEG, I_SEC, I_DEG,
                                           "complex", seed=41, shift=1.0)
        self._solve_and_check(apply, b, expect_complex_x=True)


if __name__ == "__main__":
    unittest.main()
