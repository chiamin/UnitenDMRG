"""Unit tests for linalg.lanczos and linalg.lanczos_expm_multiply.

Covers dense and QN, real and complex, independently.

lanczos
-------
- ground-state energy matches numpy eigh
- eigenvector is normalised and satisfies H|psi> = E0|psi>
- (QN) the eigenvector preserves the BlockUniTensor structure / labels

lanczos_expm_multiply
---------------------
- exp(dt*H)|v> matches numpy expm (eigh-based) for full Krylov
- real-time step preserves norm (unitary)
- dt = 0 is the identity
- labels/structure preserved
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    import cytnx
except ImportError:
    cytnx = None

from linalg import lanczos, lanczos_expm_multiply

from ._helpers import (vec, to_np, make_apply,
                       make_qn_vector, make_qn_hermitian_apply,
                       qn_to_np, qn_dense_matrix)


def _random_hermitian(n, seed, complex_):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    if complex_:
        A = A + 1j * rng.standard_normal((n, n))
    return (A + A.conj().T) / 2.0


def _expm_eigh(M, dt):
    """exp(dt*M) for Hermitian M via eigendecomposition."""
    w, U = np.linalg.eigh(M)
    return (U * np.exp(dt * w)) @ U.conj().T


# QN problem size (shared)
L_SEC, L_DEG = [0, 1, 2], [3, 3, 2]
I_SEC, I_DEG = [0, 1], [2, 2]


# ===========================================================================
# lanczos — dense
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestLanczosDense(unittest.TestCase):

    def _run(self, complex_, seed):
        n = 10
        M = _random_hermitian(n, seed, complex_)
        apply = make_apply(M)
        rng = np.random.default_rng(seed + 100)
        v0_np = rng.standard_normal(n)
        if complex_:
            v0_np = v0_np + 1j * rng.standard_normal(n)
        v0 = vec(v0_np if complex_ else v0_np.astype(float))
        E0, psi = lanczos(apply, v0, k=n)
        return M, E0, psi

    def test_real_energy_and_eigvec(self):
        M, E0, psi = self._run(False, 1)
        self.assertAlmostEqual(E0, float(np.linalg.eigvalsh(M)[0]), places=8)
        self.assertAlmostEqual(psi.Norm().item(), 1.0, places=8)
        np.testing.assert_allclose(to_np(make_apply(M)(psi)),
                                   E0 * to_np(psi), atol=1e-6)

    def test_complex_energy_and_eigvec(self):
        M, E0, psi = self._run(True, 2)
        self.assertAlmostEqual(E0, float(np.linalg.eigvalsh(M)[0]), places=8)
        self.assertAlmostEqual(psi.Norm().item(), 1.0, places=8)
        np.testing.assert_allclose(to_np(make_apply(M)(psi)),
                                   E0 * to_np(psi), atol=1e-6)

    def test_eigenvector_overlap_with_numpy(self):
        """|<psi_lanczos | psi_numpy>| = 1 up to global phase."""
        n = 10
        M = _random_hermitian(n, 7, False)
        rng = np.random.default_rng(77)
        _, psi = lanczos(make_apply(M), vec(rng.standard_normal(n)), k=n)
        psi_np = to_np(psi)
        ref = np.linalg.eigh(M)[1][:, 0]
        self.assertAlmostEqual(abs(np.dot(psi_np.conj(), ref)), 1.0, places=6)

    def test_early_convergence_diagonal(self):
        """Krylov saturates after 1 step when v0 is an exact eigenvector."""
        n = 6
        M = np.diag(np.arange(1.0, n + 1.0))
        v0_np = np.zeros(n); v0_np[0] = 1.0
        E0, psi = lanczos(make_apply(M), vec(v0_np), k=n)
        self.assertAlmostEqual(E0, 1.0, places=10)
        self.assertAlmostEqual(psi.Norm().item(), 1.0, places=10)

    def test_k_equals_one(self):
        """k=1 (single step) returns a scalar energy and unit-norm vector."""
        n = 5
        M = _random_hermitian(n, 4, False)
        E0, psi = lanczos(make_apply(M), vec(np.ones(n)), k=1)
        self.assertIsInstance(E0, float)
        self.assertAlmostEqual(psi.Norm().item(), 1.0, places=8)


# ===========================================================================
# lanczos — QN
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestLanczosQN(unittest.TestCase):

    def _run(self, dtype, seed):
        v0 = make_qn_vector(L_SEC, L_DEG, I_SEC, I_DEG, dtype, seed=seed)
        apply, _ = make_qn_hermitian_apply(L_SEC, L_DEG, I_SEC, I_DEG,
                                           dtype, seed=seed + 1, shift=1.0)
        M = qn_dense_matrix(apply, v0)
        E0, psi = lanczos(apply, v0, k=M.shape[0])
        return apply, M, E0, psi, v0

    def _check(self, dtype, seed):
        apply, M, E0, psi, v0 = self._run(dtype, seed)
        # energy matches dense block-diagonal reference
        self.assertAlmostEqual(E0, float(np.linalg.eigvalsh(M)[0]), places=8)
        # normalised
        self.assertAlmostEqual(psi.Norm().item(), 1.0, places=8)
        # eigenvector equation, compared in the flattened basis
        np.testing.assert_allclose(qn_to_np(apply(psi)), E0 * qn_to_np(psi),
                                   atol=1e-6)
        # QN structure preserved
        self.assertEqual(list(psi.labels()), list(v0.labels()))
        self.assertEqual(psi.Nblocks(), v0.Nblocks())

    def test_real(self):
        self._check("real", seed=10)

    def test_complex(self):
        self._check("complex", seed=20)


# ===========================================================================
# lanczos_expm_multiply — dense
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestExpmDense(unittest.TestCase):

    def test_real_matches_expm(self):
        n = 10
        M = _random_hermitian(n, 3, False)
        rng = np.random.default_rng(33)
        v0 = vec(rng.standard_normal(n))
        dt = -0.2                                 # imaginary-time-like
        out = to_np(lanczos_expm_multiply(make_apply(M), v0, dt, k=n))
        ref = _expm_eigh(M, dt) @ to_np(v0)
        np.testing.assert_allclose(out, ref, atol=1e-8)

    def test_complex_matches_expm(self):
        n = 10
        M = _random_hermitian(n, 4, True)
        rng = np.random.default_rng(44)
        v0 = vec(rng.standard_normal(n) + 1j * rng.standard_normal(n))
        dt = 0.3 + 0.1j
        out = to_np(lanczos_expm_multiply(make_apply(M), v0, dt, k=n))
        ref = _expm_eigh(M, dt) @ to_np(v0)
        np.testing.assert_allclose(out, ref, atol=1e-8)

    def test_real_time_norm_preserved(self):
        n = 8
        M = _random_hermitian(n, 5, True)
        rng = np.random.default_rng(55)
        v0 = vec(rng.standard_normal(n) + 1j * rng.standard_normal(n))
        out = lanczos_expm_multiply(make_apply(M), v0, -1j * 0.1, k=n)
        self.assertAlmostEqual(out.Norm().item(), v0.Norm().item(), places=6)

    def test_zero_dt_is_identity(self):
        n = 6
        M = _random_hermitian(n, 6, True)
        rng = np.random.default_rng(66)
        v0 = vec(rng.standard_normal(n) + 1j * rng.standard_normal(n))
        out = lanczos_expm_multiply(make_apply(M), v0, 0.0, k=n)
        np.testing.assert_allclose(to_np(out), to_np(v0), atol=1e-10)

    def test_known_1x1_matrix(self):
        """exp(dt*a)*v for a 1x1 matrix [[a]] equals e^(a*dt) * v."""
        a, dt = 3.5 + 0.0j, 0.2
        out = lanczos_expm_multiply(make_apply(np.array([[a]])),
                                    vec(np.array([1.0 + 0.0j])), dt, k=4)
        self.assertAlmostEqual(complex(to_np(out)[0]), np.exp(a * dt), places=10)

    def test_short_time_taylor_agreement(self):
        """exp(dt*H)|v> ~ |v> + dt*H|v> for small real dt."""
        n = 12
        M = _random_hermitian(n, 8, True)
        rng = np.random.default_rng(88)
        v0_np = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        v0 = vec(v0_np)
        dt = 1e-4
        out = to_np(lanczos_expm_multiply(make_apply(M), v0, dt, k=n))
        # first-order Taylor; the truncation error is O(dt^2) = O(1e-8) times
        # ||H^2 v||, so compare at a tolerance comfortably above that.
        np.testing.assert_allclose(out, v0_np + dt * (M @ v0_np),
                                   atol=1e-6, rtol=1e-6)

    def test_backward_evolution_increases_norm(self):
        """exp(+dt*H), dt>0, H positive-definite, increases the norm."""
        rng = np.random.default_rng(99)
        n = 8
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        M = A.conj().T @ A + np.eye(n)            # positive definite
        v0 = vec(rng.standard_normal(n) + 1j * rng.standard_normal(n))
        out = lanczos_expm_multiply(make_apply(M), v0, 0.1, k=n)
        self.assertGreater(out.Norm().item(), v0.Norm().item())


# ===========================================================================
# lanczos_expm_multiply — QN
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestExpmQN(unittest.TestCase):

    def _setup(self, dtype, seed):
        v0 = make_qn_vector(L_SEC, L_DEG, I_SEC, I_DEG, dtype, seed=seed)
        apply, _ = make_qn_hermitian_apply(L_SEC, L_DEG, I_SEC, I_DEG,
                                           dtype, seed=seed + 1, shift=1.0)
        M = qn_dense_matrix(apply, v0)
        return apply, M, v0

    def test_real_matches_expm(self):
        apply, M, v0 = self._setup("real", seed=10)
        dt = -0.2
        out = qn_to_np(lanczos_expm_multiply(apply, v0, dt, k=M.shape[0]))
        ref = _expm_eigh(M, dt) @ qn_to_np(v0)
        np.testing.assert_allclose(out, ref, atol=1e-8)

    def test_complex_matches_expm(self):
        apply, M, v0 = self._setup("complex", seed=20)
        dt = 0.3 + 0.1j
        out = qn_to_np(lanczos_expm_multiply(apply, v0, dt, k=M.shape[0]))
        ref = _expm_eigh(M, dt) @ qn_to_np(v0)
        np.testing.assert_allclose(out, ref, atol=1e-8)

    def test_real_time_norm_preserved_and_structure(self):
        apply, M, v0 = self._setup("complex", seed=30)
        out = lanczos_expm_multiply(apply, v0, -1j * 0.1, k=M.shape[0])
        self.assertAlmostEqual(out.Norm().item(), v0.Norm().item(), places=6)
        self.assertEqual(list(out.labels()), list(v0.labels()))
        self.assertEqual(out.Nblocks(), v0.Nblocks())


if __name__ == "__main__":
    unittest.main()
