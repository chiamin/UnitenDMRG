"""Small-system exact benchmark for DMRG.

This benchmark is intentionally isolated from existing DMRG tests.
It compares DMRG ground-state results against exact diagonalization on N=4.
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
    from DMRG.dmrg_engine import DMRGEngine
    from MPS.mps_init import random_mps
    from tests.helpers.heisenberg import heisenberg_mpo


def _heisenberg_matrix(N: int, J: float = 1.0, delta: float = 1.0) -> np.ndarray:
    """Build dense Heisenberg Hamiltonian for N spin-1/2 sites."""
    I = np.eye(2, dtype=complex)
    Sz = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=complex)
    Sp = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    Sm = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)

    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(N - 1):
        for coeff, op_i, op_j in (
            (J / 2.0, Sp, Sm),
            (J / 2.0, Sm, Sp),
            (J * delta, Sz, Sz),
        ):
            mat = np.array([[1.0]], dtype=complex)
            for k in range(N):
                if k == i:
                    mat = np.kron(mat, op_i)
                elif k == i + 1:
                    mat = np.kron(mat, op_j)
                else:
                    mat = np.kron(mat, I)
            H += coeff * mat
    return H


def _mps_to_state(mps) -> np.ndarray:
    """Convert a dense open-boundary MPS to a full state vector."""
    tensors = [mps[p].get_block().numpy() for p in range(len(mps))]
    state = tensors[0]  # (1, d, D1)
    for A in tensors[1:]:
        state = np.tensordot(state, A, axes=([-1], [0]))
    state = np.squeeze(state, axis=(0, -1))
    return state.reshape(-1)


def _energy_variance(psi_vec: np.ndarray, H: np.ndarray) -> float:
    """Return variance <H^2>-<H>^2 for normalized (or unnormalized) psi."""
    norm = np.vdot(psi_vec, psi_vec)
    e = np.vdot(psi_vec, H @ psi_vec) / norm
    e2 = np.vdot(psi_vec, (H @ (H @ psi_vec))) / norm
    return float(np.real(e2 - e * e.conjugate()))


@unittest.skipIf(cytnx is None, "cytnx not available")
class TestDMRGExactBenchmark(unittest.TestCase):
    """Exact benchmark for DMRG on a small chain."""

    def test_ground_state_energy_and_variance_vs_exact(self):
        N = 4
        H_dense = _heisenberg_matrix(N)
        E_exact = float(np.linalg.eigvalsh(H_dense)[0].real)

        psi = random_mps(N, phys_dim=2, bond_dim=8, dtype=complex, seed=7, normalize=True)
        psi.move_center(0)
        H_mpo = heisenberg_mpo(N, dtype=complex)
        engine = DMRGEngine(psi, H_mpo)

        E = None
        for max_dim in (8, 16, 16):
            E, _ = engine.sweep(max_dim=max_dim, cutoff=1e-12, num_center=2)

        self.assertIsNotNone(E)
        self.assertAlmostEqual(E, E_exact, delta=1e-5)

        psi_vec = _mps_to_state(psi)
        var = _energy_variance(psi_vec, H_dense)
        self.assertLess(var, 1e-6)


if __name__ == "__main__":
    unittest.main()

