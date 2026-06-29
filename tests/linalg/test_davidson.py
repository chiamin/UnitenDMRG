"""Unit tests for linalg.davidson (dense and QN, real and complex).

Covers, independently for dense and QN:
- ground-state energy matches numpy eigh (without and with Jacobi precond)
- eigenvector is normalised and satisfies H|psi> = E0|psi>
- (QN) the eigenvector preserves the BlockUniTensor structure / labels

The Jacobi preconditioner t_i = r_i / (theta - H_ii) is built from the dense
diagonal of the operator (obtained via qn_dense_matrix for the QN case), then
applied entry-wise in the flattened basis and scattered back.
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    import cytnx
except ImportError:
    cytnx = None

from linalg import davidson

from ._helpers import (vec, to_np, make_apply,
                       make_qn_vector, make_qn_hermitian_apply,
                       qn_to_np, qn_from_np, qn_dense_matrix)


def _random_hermitian(n, seed, complex_):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    if complex_:
        A = A + 1j * rng.standard_normal((n, n))
    return (A + A.conj().T) / 2.0


L_SEC, L_DEG = [0, 1, 2], [3, 3, 2]
I_SEC, I_DEG = [0, 1], [2, 2]


# ===========================================================================
# davidson — dense
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestDavidsonDense(unittest.TestCase):

    def _check(self, complex_, seed, use_precond):
        n = 10
        M = _random_hermitian(n, seed, complex_)
        apply = make_apply(M)
        rng = np.random.default_rng(seed + 100)
        v0_np = rng.standard_normal(n)
        if complex_:
            v0_np = v0_np + 1j * rng.standard_normal(n)
        v0 = vec(v0_np if complex_ else v0_np.astype(float))

        precond = None
        if use_precond:
            diag = np.diag(M).real if not complex_ else np.diag(M)
            def precond(r, theta, _diag=diag):
                denom = theta - _diag
                safe = np.where(np.abs(denom) > 1e-12, denom, 1.0)
                return vec(to_np(r) / safe)

        E0, psi = davidson(apply, v0, precond=precond, k=n)
        self.assertAlmostEqual(E0, float(np.linalg.eigvalsh(M)[0]), places=8)
        self.assertAlmostEqual(psi.Norm().item(), 1.0, places=8)
        np.testing.assert_allclose(to_np(apply(psi)), E0 * to_np(psi), atol=1e-6)

    def test_real(self):
        self._check(False, 1, use_precond=False)

    def test_complex(self):
        self._check(True, 2, use_precond=False)

    def test_real_precond(self):
        self._check(False, 3, use_precond=True)

    def test_complex_precond(self):
        self._check(True, 4, use_precond=True)


# ===========================================================================
# davidson — QN
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestDavidsonQN(unittest.TestCase):

    def _setup(self, dtype, seed):
        v0 = make_qn_vector(L_SEC, L_DEG, I_SEC, I_DEG, dtype, seed=seed)
        apply, _ = make_qn_hermitian_apply(L_SEC, L_DEG, I_SEC, I_DEG,
                                           dtype, seed=seed + 1, shift=1.0)
        M = qn_dense_matrix(apply, v0)
        return apply, M, v0

    def _check(self, dtype, seed, use_precond):
        apply, M, v0 = self._setup(dtype, seed)

        precond = None
        if use_precond:
            diag = np.diag(M)
            if dtype == "real":
                diag = diag.real
            def precond(r, theta, _diag=diag, _tmpl=v0):
                denom = theta - _diag
                safe = np.where(np.abs(denom) > 1e-12, denom, 1.0)
                return qn_from_np(qn_to_np(r) / safe, _tmpl)

        E0, psi = davidson(apply, v0, precond=precond, k=M.shape[0])
        self.assertAlmostEqual(E0, float(np.linalg.eigvalsh(M)[0]), places=8)
        self.assertAlmostEqual(psi.Norm().item(), 1.0, places=8)
        np.testing.assert_allclose(qn_to_np(apply(psi)), E0 * qn_to_np(psi),
                                   atol=1e-6)
        self.assertEqual(list(psi.labels()), list(v0.labels()))
        self.assertEqual(psi.Nblocks(), v0.Nblocks())

    def test_real(self):
        self._check("real", seed=10, use_precond=False)

    def test_complex(self):
        self._check("complex", seed=20, use_precond=False)

    def test_real_precond(self):
        self._check("real", seed=30, use_precond=True)

    def test_complex_precond(self):
        self._check("complex", seed=40, use_precond=True)


if __name__ == "__main__":
    unittest.main()
