"""
Fermionic 2D tight-binding example: 3x3 square lattice.

Model: H = -t sum_{<i,j>} (c+_i c_j + c+_j c_i)

Builds the MPO via AutoMPO (with automatic Jordan-Wigner strings),
then compares the full many-body spectrum against exact diagonalization
to verify correctness.

Single-particle energies for OBC:
    e_{nx,ny} = -2t [cos(pi nx/(Lx+1)) + cos(pi ny/(Ly+1))]
The many-body ground state fills the lowest single-particle levels.
"""

import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import numpy as np

from lattice import SquareLattice
from MPS.physical_sites import spinless_fermion
from MPS.auto_mpo import AutoMPO


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

Lx, Ly = 3, 3
t = 1.0

lat = SquareLattice(Lx, Ly)
N = lat.N
dim = 2**N

print(f"Lattice: {lat}")
print(f"N = {N} sites, Hilbert space dim = {dim}")
print(f"Bonds: {lat.bonds()}")
print()


# ---------------------------------------------------------------------------
# Build MPO via AutoMPO
# ---------------------------------------------------------------------------

site = spinless_fermion()
ampo = AutoMPO(N, site)
for i, j in lat.bonds():
    ampo.add(-t, "Cdag", i, "C", j)
    ampo.add(-t, "Cdag", j, "C", i)
H_mpo = ampo.to_mpo()

print("MPO bond dimensions:", [H_mpo[p].shape()[0] for p in range(N)])
print()


# ---------------------------------------------------------------------------
# MPO -> full dense matrix
# ---------------------------------------------------------------------------

def mpo_full_matrix(mpo):
    from itertools import product as iproduct
    N = len(mpo)
    d = mpo.phys_dims[0]
    W_arrs = []
    for p in range(N):
        w = mpo[p].permute(["l", "ip", "i", "r"])
        W_arrs.append(w.get_block().numpy())
    total_dim = d**N
    mat = np.zeros((total_dim, total_dim))
    for bra in iproduct(range(d), repeat=N):
        for ket in iproduct(range(d), repeat=N):
            vec = np.array([1.0])
            for p in range(N):
                vec = W_arrs[p][:, bra[p], ket[p], :].T @ vec
            bra_flat = sum(bra[p] * d**(N-1-p) for p in range(N))
            ket_flat = sum(ket[p] * d**(N-1-p) for p in range(N))
            mat[bra_flat, ket_flat] = vec[0]
    return mat


H_mat = mpo_full_matrix(H_mpo)


# ---------------------------------------------------------------------------
# Exact diagonalization with explicit Jordan-Wigner
# ---------------------------------------------------------------------------

def tb_ed_matrix(lat, t):
    N = lat.N
    dim = 2**N
    H = np.zeros((dim, dim))
    for state in range(dim):
        bits = [(state >> (N - 1 - k)) & 1 for k in range(N)]
        for i, j in lat.bonds():
            if bits[j] == 1 and bits[i] == 0:
                new_bits = list(bits)
                new_bits[j] = 0
                new_bits[i] = 1
                new_state = sum(new_bits[k] << (N - 1 - k) for k in range(N))
                sign = (-1) ** sum(bits[k] for k in range(i + 1, j))
                H[new_state, state] += -t * sign
                H[state, new_state] += -t * sign
    return H


H_ed = tb_ed_matrix(lat, t)


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

max_diff = np.max(np.abs(H_mat - H_ed))
print(f"Max |H_mpo - H_ed| = {max_diff:.2e}")
assert max_diff < 1e-12, "MPO and ED matrices differ!"
print("MPO matches exact diagonalization.\n")


# ---------------------------------------------------------------------------
# Eigenvalues and single-particle energies
# ---------------------------------------------------------------------------

evals = np.linalg.eigvalsh(H_mat)

# Single-particle energies (OBC)
sp_energies = sorted(
    -2.0 * t * (np.cos(np.pi * nx / (Lx + 1)) + np.cos(np.pi * ny / (Ly + 1)))
    for nx in range(1, Lx + 1)
    for ny in range(1, Ly + 1)
)

print("Single-particle energies:")
for k, e in enumerate(sp_energies):
    print(f"  e_{k} = {e:+.6f}")
print()

# Ground state: fill lowest levels (try all fillings)
E0_exact = min(sum(sp_energies[:Nf]) for Nf in range(N + 1))
Nf_best = min(range(N + 1), key=lambda Nf: sum(sp_energies[:Nf]))

print(f"Many-body ground state energy (filling {Nf_best} particles):")
print(f"  E0 (from single-particle) = {E0_exact:.8f}")
print(f"  E0 (from MPO eigvalsh)    = {evals[0]:.8f}")
print(f"  Difference                = {abs(evals[0] - E0_exact):.2e}")
