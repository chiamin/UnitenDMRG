"""
DMRG ground-state example: spin-1/2 Heisenberg chain.

Model: H = J Σ_i (S⁺_i S⁻_{i+1} + S⁻_i S⁺_{i+1}) / 2 + Δ Σ_i Sz_i Sz_{i+1}

The exact ground-state energy per site for the isotropic (Δ=1) chain is
given by the Bethe ansatz:

    E0/N → J (1/4 - ln 2) ≈ -0.4431 J  (thermodynamic limit)
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly.
root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from MPS.mps_init import random_mps
from MPS.hamiltonian import heisenberg_mpo
from DMRG.dmrg_engine import DMRGEngine


# Parameters
N     = 10      # number of sites
J     = 1.0     # exchange coupling
delta = 1.0     # anisotropy (1 = isotropic Heisenberg)
h     = 0.0     # longitudinal field
seed  = 42      # RNG seed for reproducibility

# DMRG schedule: (max_bond_dim, SVD_cutoff)
schedule = [
    (10,  0.0),
    (20,  0.0),
    (40,  1e-8),
    (40,  1e-8),
    (40,  1e-8),
]


print(f"Heisenberg chain: N={N}, J={J}, Δ={delta}, h={h}")
print()

H = heisenberg_mpo(N, J=J, delta=delta, h=h)

# Initial state: random MPS with small bond dim, right-canonical (center=0)
psi = random_mps(N, phys_dim=2, bond_dim=4, seed=seed)
psi.move_center(0)

print(f"Initial state: {psi}")
print()

engine = DMRGEngine(psi, H)


# ---------------------------------------------------------------------------
# DMRG sweeps
# ---------------------------------------------------------------------------

print(f"{'sweep':>5}  {'max_dim':>7}  {'cutoff':>8}  {'E':>14}  {'E/N':>10}  {'trunc':>10}")
print("-" * 65)

E, trunc = None, None
for sweep_idx, (max_dim, cutoff) in enumerate(schedule):
    E, trunc = engine.sweep(max_dim=max_dim, cutoff=cutoff, num_center=2)
    print(f"{sweep_idx+1:>5}  {max_dim:>7}  {cutoff:>8.1e}  {E:>14.8f}  {E/N:>10.6f}  {trunc:>10.2e}")

print()
print(f"Final energy        E   = {E:.8f}")
print(f"Energy per site     E/N = {E/N:.8f}")
print(f"Bethe ansatz (TDL)  E/N ≈ {J*(0.25 - 0.6931471805599453):.6f}  (thermodynamic limit)")
print(f"Final MPS: {psi}")
