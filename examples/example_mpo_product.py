"""
MPO product and compression example: H² for spin-1/2 Heisenberg chain.

Demonstrates three methods to compute and compress the product of two MPOs:

1. Exact product + SVD compression  (mpo_product + svd_compress_mpo)
2. Variational fitting              (fit_mpo_product)
3. Both methods compared against the exact H² full matrix

The Heisenberg Hamiltonian is:
    H = Σ_i [ Sz_i Sz_{i+1} + (S⁺_i S⁻_{i+1} + S⁻_i S⁺_{i+1}) / 2 ]
"""

import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from MPS.physical_sites import spin_half
from MPS.auto_mpo import AutoMPO
from MPS.mps_operations import (
    mpo_product,
    fit_mpo_product,
    expectation,
)
from MPS.mpo_compression import svd_compress_mpo, _svd_two_mpo_sites


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N       = 10
max_dim = 12      # target bond dimension for the compressed H²
nsweep  = 4       # number of fitting sweeps
seed    = 42

# ---------------------------------------------------------------------------
# Build Hamiltonian
# ---------------------------------------------------------------------------

site = spin_half(qn="Sz")

ampo = AutoMPO(N, site)
for i in range(N - 1):
    ampo.add(1.0, "Sz", i, "Sz", i + 1)
    ampo.add(0.5, "Sp", i, "Sm", i + 1)
    ampo.add(0.5, "Sm", i, "Sp", i + 1)
H = ampo.to_mpo()

print(f"H bond dims:  {H.mpo_dims}")

# ---------------------------------------------------------------------------
# Method 1: Exact product + SVD compression
# ---------------------------------------------------------------------------

print("\n--- Method 1: exact product + SVD compression ---")

H2_exact = mpo_product(H, H)
print(f"H² exact bond dims:  {H2_exact.mpo_dims}")

H2_svd = svd_compress_mpo(H2_exact, max_dim=max_dim, cutoff=1e-14)
print(f"H² SVD compressed:   {H2_svd.mpo_dims}")

# ---------------------------------------------------------------------------
# Method 2: Variational fitting (never builds D² product)
# ---------------------------------------------------------------------------

print("\n--- Method 2: variational fitting ---")

# Initial guess: rough SVD compression with small bond dimension.
fitmpo = svd_compress_mpo(H2_exact, max_dim=max_dim // 2)

# Right-canonicalize fitmpo before fitting.
for p in range(N - 2, -1, -1):
    l, r, _ = _svd_two_mpo_sites(
        fitmpo.tensors[p], fitmpo.tensors[p + 1],
        absorb="left", dim=sys.maxsize, cutoff=0.0,
    )
    fitmpo.tensors[p] = l
    fitmpo.tensors[p + 1] = r

H2_fit = fit_mpo_product(
    H, H, fitmpo,
    nsweep=nsweep, max_dim=max_dim, cutoff=1e-14,
)
print(f"H² fit compressed:   {H2_fit.mpo_dims}")

# ---------------------------------------------------------------------------
# Validate: compare <psi|H²|psi> across all three
# ---------------------------------------------------------------------------

print("\n--- Validation: <psi|H²|psi> with QN product states ---")
print(f"{'state':>20}  {'exact':>16}  {'SVD':>16}  {'fit':>16}  {'SVD err':>10}  {'fit err':>10}")

states_list = [
    [0, 1] * (N // 2),             # Neel
    [1, 0] * (N // 2),             # reversed Neel
    [0, 0, 1, 1] * (N // 4) + [0, 1] * (N % 4 // 2),  # domain walls
]

max_svd_err = 0.0
max_fit_err = 0.0
for states in states_list:
    psi = site.product_state(states)
    psi.orthogonalize(center=0)
    v_exact = expectation(psi, H2_exact, psi)
    v_svd   = expectation(psi, H2_svd, psi)
    v_fit   = expectation(psi, H2_fit, psi)
    err_svd = abs(v_exact - v_svd)
    err_fit = abs(v_exact - v_fit)
    max_svd_err = max(max_svd_err, err_svd)
    max_fit_err = max(max_fit_err, err_fit)
    tag = "".join("↑" if s == 0 else "↓" for s in states)
    print(f"{tag:>20}  {v_exact:16.10f}  {v_svd:16.10f}  {v_fit:16.10f}  {err_svd:10.2e}  {err_fit:10.2e}")

print(f"\nMax error — SVD: {max_svd_err:.2e},  fit: {max_fit_err:.2e}")
