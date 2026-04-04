"""
Hubbard model example: 1D chain with N=4 sites.

Model: H = -t sum_{i,sigma} (c†_{i,sigma} c_{i+1,sigma} + h.c.)
           + U sum_i n_{i,up} n_{i,dn}

Builds the MPO via AutoMPO with automatic Jordan-Wigner strings,
then compares the full many-body spectrum against exact diagonalization.
"""

import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import numpy as np

from MPS.physical_sites import electron
from MPS.auto_mpo import AutoMPO
from tests.helpers.mpo_utils import mpo_full_matrix
from tests.auto_mpo.test_hubbard import hubbard_ed


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N = 4
t = 1.0
U = 4.0

print(f"1D Hubbard chain: N={N}, t={t}, U={U}")
print(f"Hilbert space dim = 4^{N} = {4**N}")
print()


# ---------------------------------------------------------------------------
# Build MPO via AutoMPO
# ---------------------------------------------------------------------------

site = electron()
ampo = AutoMPO(N, site)
for i in range(N - 1):
    for op_dag, op in [("Cupdag", "Cup"), ("Cdndag", "Cdn")]:
        ampo.add(-t, op_dag, i, op, i + 1)       # c†_{i,sigma} c_{i+1,sigma}
        ampo.add(-t, op_dag, i + 1, op, i)       # h.c.
for i in range(N):
    ampo.add(U, "Nup", i, "Ndn", i)              # U n_up n_dn (same site)

H_mpo = ampo.to_mpo()
print("MPO bond dimensions:", [H_mpo[p].shape()[0] for p in range(N)])
print()


# ---------------------------------------------------------------------------
# Compare MPO vs ED
# ---------------------------------------------------------------------------

H_mat = mpo_full_matrix(H_mpo)
H_ed = hubbard_ed(N, t, U)

max_diff = np.max(np.abs(H_mat - H_ed))
print(f"Max |H_mpo - H_ed| = {max_diff:.2e}")
assert max_diff < 1e-12, "MPO and ED matrices differ!"
print("MPO matches exact diagonalization.")
print()


# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------

evals = np.linalg.eigvalsh(H_mat)

print(f"Ground state energy: E0 = {evals[0]:.8f}")
print(f"First gap:           dE = {evals[1] - evals[0]:.8f}")
print()
print(f"Lowest 10 eigenvalues:")
for k, e in enumerate(evals[:10]):
    print(f"  E_{k:2d} = {e:+.8f}")
