"""Small-system exact benchmark for real-time TDVP.

This is an integration-style regression test:
- Evolve a small dense MPS with TDVP real-time.
- Compare local observables against exact dense evolution.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import cytnx
except ImportError:
    cytnx = None

THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

if cytnx is not None:
    from MPS.mps_init import random_mps
    from TDVP.tdvp_engine import TDVPEngine
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


def _local_sz_op(N: int, site: int) -> np.ndarray:
    """Return dense operator I..I*Sz*I..I acting on one site."""
    I = np.eye(2, dtype=complex)
    Sz = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=complex)
    op = np.array([[1.0]], dtype=complex)
    for k in range(N):
        op = np.kron(op, Sz if k == site else I)
    return op


def _mps_to_state(mps) -> np.ndarray:
    """Convert a dense open-boundary MPS to a full state vector."""
    tensors = [mps[p].get_block().numpy() for p in range(len(mps))]
    state = tensors[0]  # (1, d, D1)
    for A in tensors[1:]:
        # Contract previous right virtual with next left virtual.
        state = np.tensordot(state, A, axes=([-1], [0]))
    # Final shape: (1, d, d, ..., d, 1) -> (2^N,)
    state = np.squeeze(state, axis=(0, -1))
    return state.reshape(-1)


def _expect(psi: np.ndarray, op: np.ndarray) -> complex:
    """Return <psi|op|psi>/<psi|psi> for dense vectors."""
    denom = np.vdot(psi, psi)
    return np.vdot(psi, op @ psi) / denom


@unittest.skipIf(cytnx is None, "cytnx not available")
class TestTDVPRealTimeExactBenchmark(unittest.TestCase):
    """Compare TDVP real-time against exact dense evolution on N=4."""

    def test_real_time_matches_exact_local_sz(self):
        N = 4
        dt = 1j * 0.02
        n_sweeps = 4
        total_t = 0.08

        # TDVP evolution
        psi = random_mps(N, phys_dim=2, bond_dim=8, dtype=complex, seed=123, normalize=True)
        psi.move_center(0)
        psi_init = psi.copy()
        H_mpo = heisenberg_mpo(N, dtype=complex)
        engine = TDVPEngine(psi, H_mpo)
        for _ in range(n_sweeps):
            engine.sweep(dt=dt, num_center=2, max_dim=16, cutoff=1e-12)
        psi_tdvp = _mps_to_state(psi)

        # Exact dense evolution
        H = _heisenberg_matrix(N)
        evals, evecs = np.linalg.eigh(H)
        U = evecs @ np.diag(np.exp(-1j * total_t * evals)) @ evecs.conj().T
        psi0 = _mps_to_state(psi_init)
        psi_exact = U @ psi0

        # Compare local magnetizations <Sz_i>.
        for i in range(N):
            op = _local_sz_op(N, i)
            sz_tdvp = _expect(psi_tdvp, op)
            sz_exact = _expect(psi_exact, op)
            self.assertAlmostEqual(sz_tdvp.real, sz_exact.real, delta=2e-2)
            self.assertAlmostEqual(sz_tdvp.imag, sz_exact.imag, delta=2e-2)


if __name__ == "__main__":
    unittest.main()

