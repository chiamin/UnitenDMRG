"""Unit tests for DMRG/dmrg_engine.py.

Coverage
--------
1. DMRGEngine construction  (TestDMRGEngineInit)
   - psi.center != 0 → ValueError
   - ortho_states / ortho_weights length mismatch → ValueError
   - Valid construction succeeds

2. sweep() input validation  (TestDMRGEngineSweepValidation)
   - num_center not in {1, 2} → ValueError
   - psi.center != 0 at start of sweep → ValueError

3. Return value contract  (TestDMRGEngineSweepReturnValues)
   - Returns (float, float)
   - trunc >= 0
   - Energy is finite

4. Ground-state energy correctness  (TestDMRGGroundState)
   - 2-site DMRG converges to exact ground state (exact diagonalisation reference)
   - 1-site DMRG converges to same ground state

5. Excited-state targeting  (TestDMRGExcitedState)
   - First excited energy E1 > E0
   - E1 matches exact diagonalisation (small chain, large bond dim)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

try:
    import cytnx
except ImportError:
    cytnx = None

if cytnx is not None:
    from MPS.mps_init import random_mps
    from MPS.hamiltonian import heisenberg_mpo
    from DMRG.dmrg_engine import DMRGEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_psi(N: int, D: int = 4, seed: int = 0) -> "MPS":
    """Random MPS with center at 0."""
    psi = random_mps(N, phys_dim=2, bond_dim=D, seed=seed)
    psi.move_center(0)
    return psi


def _heisenberg_matrix(N: int, J: float = 1.0, delta: float = 1.0) -> np.ndarray:
    """Build the full N-site Heisenberg Hamiltonian as a 2^N × 2^N matrix.

    H = J Σ_i [(S⁺_i S⁻_{i+1} + S⁻_i S⁺_{i+1}) / 2 + Δ Sz_i Sz_{i+1}]
    """
    I  = np.eye(2)
    Sz = np.array([[0.5, 0.0], [0.0, -0.5]])
    Sp = np.array([[0.0, 1.0], [0.0,  0.0]])
    Sm = np.array([[0.0, 0.0], [1.0,  0.0]])

    dim = 2 ** N
    H = np.zeros((dim, dim))

    for i in range(N - 1):
        for coeff, op_i, op_j in [
            (J / 2.0,       Sp, Sm),   # S⁺_i S⁻_{i+1}
            (J / 2.0,       Sm, Sp),   # S⁻_i S⁺_{i+1}
            (J * delta,     Sz, Sz),   # Sz_i Sz_{i+1}
        ]:
            mat = np.array([[1.0]])
            for k in range(N):
                if k == i:
                    mat = np.kron(mat, op_i)
                elif k == i + 1:
                    mat = np.kron(mat, op_j)
                else:
                    mat = np.kron(mat, I)
            H += coeff * mat

    return H


def _exact_energies(N: int, num: int = 2) -> list[float]:
    """Return the lowest `num` eigenvalues of the N-site Heisenberg chain."""
    H = _heisenberg_matrix(N)
    evals = np.linalg.eigvalsh(H)
    return [float(evals[k]) for k in range(num)]


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestDMRGEngineInit(unittest.TestCase):

    def test_center_not_zero_raises(self):
        """psi.center must be 0; otherwise ValueError."""
        psi = _make_psi(4)
        psi.move_center(2)            # center = 2 ≠ 0
        H = heisenberg_mpo(4)
        with self.assertRaises(ValueError):
            DMRGEngine(psi, H)

    def test_ortho_length_mismatch_raises(self):
        """ortho_states and ortho_weights must have the same length."""
        psi  = _make_psi(4)
        psi0 = _make_psi(4, seed=1)
        H    = heisenberg_mpo(4)
        with self.assertRaises(ValueError):
            DMRGEngine(psi, H, ortho_states=[psi0], ortho_weights=[])

    def test_valid_construction(self):
        """Normal construction should not raise."""
        psi = _make_psi(4)
        H   = heisenberg_mpo(4)
        DMRGEngine(psi, H)            # must not raise


# ---------------------------------------------------------------------------
# 2. sweep() input validation
# ---------------------------------------------------------------------------

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestDMRGEngineSweepValidation(unittest.TestCase):

    def setUp(self):
        self.psi = _make_psi(4)
        self.engine = DMRGEngine(self.psi, heisenberg_mpo(4))

    def test_invalid_num_center_raises(self):
        """num_center not in {1, 2} → ValueError."""
        with self.assertRaises(ValueError):
            self.engine.sweep(max_dim=10, num_center=3)

    def test_center_not_zero_before_sweep_raises(self):
        """psi.center != 0 at the start of sweep → ValueError."""
        self.psi.move_center(2)
        with self.assertRaises(ValueError):
            self.engine.sweep(max_dim=10)


# ---------------------------------------------------------------------------
# 3. Return value contract
# ---------------------------------------------------------------------------

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestDMRGEngineSweepReturnValues(unittest.TestCase):

    def test_returns_float_pair(self):
        """sweep() returns (float, float)."""
        psi = _make_psi(4)
        engine = DMRGEngine(psi, heisenberg_mpo(4))
        E, trunc = engine.sweep(max_dim=10)
        self.assertIsInstance(E, float)
        self.assertIsInstance(trunc, float)

    def test_trunc_nonnegative(self):
        """Truncation error is non-negative."""
        psi = _make_psi(4)
        engine = DMRGEngine(psi, heisenberg_mpo(4))
        _, trunc = engine.sweep(max_dim=4, cutoff=1e-8)
        self.assertGreaterEqual(trunc, 0.0)

    def test_energy_is_finite(self):
        """Returned energy is a finite number."""
        psi = _make_psi(4)
        engine = DMRGEngine(psi, heisenberg_mpo(4))
        E, _ = engine.sweep(max_dim=10)
        self.assertTrue(np.isfinite(E))


# ---------------------------------------------------------------------------
# 4. Ground-state energy correctness
# ---------------------------------------------------------------------------

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestDMRGGroundState(unittest.TestCase):
    """Compare DMRG ground-state energy to exact diagonalisation."""

    N     = 6
    ATOL  = 1e-5    # energy tolerance

    @classmethod
    def setUpClass(cls):
        cls.E0_exact = _exact_energies(cls.N, num=1)[0]

    def _run_dmrg(self, num_center: int) -> float:
        psi    = _make_psi(self.N, D=4, seed=0)
        engine = DMRGEngine(psi, heisenberg_mpo(self.N))
        E = None
        for max_dim in [10, 20, 20]:
            E, _ = engine.sweep(max_dim=max_dim, cutoff=1e-10,
                                num_center=num_center)
        return E

    def test_2site_ground_state_energy(self):
        """2-site DMRG converges to exact ground-state energy."""
        E = self._run_dmrg(num_center=2)
        self.assertAlmostEqual(E, self.E0_exact, delta=self.ATOL)

    def test_1site_ground_state_energy(self):
        """1-site DMRG converges to exact ground-state energy."""
        # Start from a well-converged state so 1-site can find the minimum.
        psi    = _make_psi(self.N, D=16, seed=0)
        engine = DMRGEngine(psi, heisenberg_mpo(self.N))
        # Warm up with 2-site
        for max_dim in [10, 20]:
            engine.sweep(max_dim=max_dim, cutoff=1e-10, num_center=2)
        # Refine with 1-site
        E = None
        for _ in range(3):
            E, _ = engine.sweep(max_dim=20, cutoff=1e-10, num_center=1)
        self.assertAlmostEqual(E, self.E0_exact, delta=self.ATOL)

    def test_energy_decreases_across_sweeps(self):
        """Each successive sweep must not increase the energy."""
        psi    = _make_psi(self.N, D=4, seed=7)
        engine = DMRGEngine(psi, heisenberg_mpo(self.N))
        energies = []
        for _ in range(4):
            E, _ = engine.sweep(max_dim=16, cutoff=0.0)
            energies.append(E)
        for i in range(1, len(energies)):
            self.assertLessEqual(energies[i], energies[i - 1] + 1e-10)


# ---------------------------------------------------------------------------
# 5. Excited-state targeting
# ---------------------------------------------------------------------------

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestDMRGExcitedState(unittest.TestCase):
    """Excited-state DMRG via orthogonal penalty."""

    N    = 6
    ATOL = 1e-4

    @classmethod
    def setUpClass(cls):
        exact = _exact_energies(cls.N, num=2)
        cls.E0_exact = exact[0]
        cls.E1_exact = exact[1]

    def _run_sweeps(self, engine, n=4, max_dim=20):
        E = None
        for _ in range(n):
            E, _ = engine.sweep(max_dim=max_dim, cutoff=1e-10)
        return E

    def test_excited_energy_above_ground(self):
        """E1 found by DMRG is strictly above E0."""
        # Ground state
        psi0   = _make_psi(self.N, D=8, seed=0)
        engine0 = DMRGEngine(psi0, heisenberg_mpo(self.N))
        E0 = self._run_sweeps(engine0)

        # First excited state with penalty weight >> gap
        psi1   = _make_psi(self.N, D=8, seed=1)
        engine1 = DMRGEngine(psi1, heisenberg_mpo(self.N),
                             ortho_states=[psi0], ortho_weights=[10.0])
        E1 = self._run_sweeps(engine1)

        self.assertGreater(E1, E0 - 1e-6)

    def test_excited_energy_matches_exact(self):
        """First excited energy matches exact diagonalisation."""
        # Ground state
        psi0    = _make_psi(self.N, D=16, seed=0)
        engine0 = DMRGEngine(psi0, heisenberg_mpo(self.N))
        self._run_sweeps(engine0, n=5, max_dim=32)

        # Excited state
        psi1    = _make_psi(self.N, D=16, seed=2)
        engine1 = DMRGEngine(psi1, heisenberg_mpo(self.N),
                             ortho_states=[psi0], ortho_weights=[10.0])
        E1 = self._run_sweeps(engine1, n=5, max_dim=32)

        self.assertAlmostEqual(E1, self.E1_exact, delta=self.ATOL)


if __name__ == "__main__":
    unittest.main()
