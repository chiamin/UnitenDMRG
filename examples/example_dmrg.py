"""
DMRG ground-state example: spin-1/2 Heisenberg chain.

Model: H = J Σ_i (S⁺_i S⁻_{i+1} + S⁻_i S⁺_{i+1}) / 2 + Δ Σ_i Sz_i Sz_{i+1}

Runs both a dense (no QN) and a QN-symmetric (U1 N_up) DMRG sweep for comparison.

The exact ground-state energy per site for the isotropic (Δ=1) chain is
given by the Bethe ansatz:

    E0/N → J (1/4 - ln 2) ≈ -0.4431 J  (thermodynamic limit)
"""

import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from MPS.mps_init import random_mps
from MPS.physical_sites import spin_half
from MPS.auto_mpo import AutoMPO
from DMRG.dmrg_engine import DMRGEngine


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N     = 10      # number of sites
J     = 1.0     # exchange coupling
delta = 1.0     # anisotropy (1 = isotropic Heisenberg)
h     = 0.0     # longitudinal field
seed  = 42

# DMRG schedule: (max_bond_dim, rho_eigenvalue_cutoff)
schedule = [
    (10,  0.0),
    (20,  0.0),
    (40,  1e-8),
    (40,  1e-8),
    (40,  1e-8),
]

BETHE = J * (0.25 - 0.6931471805599453)   # Bethe ansatz TDL value


def build_heisenberg(site, N, J, delta, h):
    ampo = AutoMPO(N, site)
    for i in range(N - 1):
        ampo.add(J * delta, "Sz", i, "Sz", i + 1)
        ampo.add(J / 2,     "Sp", i, "Sm", i + 1)
        ampo.add(J / 2,     "Sm", i, "Sp", i + 1)
    if h != 0.0:
        for i in range(N):
            ampo.add(h, "Sz", i)
    return ampo.to_mpo()


def run_dmrg(label, psi, H):
    engine = DMRGEngine(psi, H)
    print(f"\n{'':=<65}")
    print(f"  {label}")
    print(f"{'':=<65}")
    print(f"{'sweep':>5}  {'max_dim':>7}  {'cutoff':>8}  {'E':>14}  {'E/N':>10}  {'trunc':>10}")
    print("-" * 65)
    E, trunc = None, None
    for idx, (max_dim, cutoff) in enumerate(schedule):
        E, trunc = engine.sweep(max_dim=max_dim, cutoff=cutoff, num_center=2)
        print(f"{idx+1:>5}  {max_dim:>7}  {cutoff:>8.1e}  {E:>14.8f}  {E/N:>10.6f}  {trunc:>10.2e}")
    print()
    print(f"  Final E     = {E:.8f}")
    print(f"  Final E/N   = {E/N:.8f}")
    print(f"  Bethe (TDL) = {BETHE:.8f}")
    print(f"  Final MPS   = {psi}")
    return E


# ---------------------------------------------------------------------------
# Dense (no QN)
# ---------------------------------------------------------------------------

site_dense = spin_half()
H_dense    = build_heisenberg(site_dense, N, J, delta, h)
psi_dense  = random_mps(N, phys_dim=2, bond_dim=4, seed=seed)
psi_dense.move_center(0)

E_dense = run_dmrg("Dense (no QN)", psi_dense, H_dense)


# ---------------------------------------------------------------------------
# QN-symmetric (U1 N_up), Neel initial state at half-filling (N_up = N/2)
# ---------------------------------------------------------------------------

site_qn = spin_half(qn="Sz")
H_qn    = build_heisenberg(site_qn, N, J, delta, h)

# Neel state: |dn up dn up ...> — N_up = N/2 for even N
neel = [i % 2 for i in range(N)]
psi_qn = site_qn.product_state(neel, center=0)

E_qn = run_dmrg("QN-symmetric (U1 N_up)", psi_qn, H_qn)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'':=<65}")
print("  Summary")
print(f"{'':=<65}")
print(f"  Dense E/N  = {E_dense/N:.8f}")
print(f"  QN    E/N  = {E_qn/N:.8f}")
print(f"  Bethe TDL  = {BETHE:.8f}")
