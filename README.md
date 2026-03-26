# Uni10 — DMRG with Cytnx UniTensor

A Python implementation of the **Density Matrix Renormalization Group (DMRG)** algorithm for 1D quantum many-body systems, built on top of [Cytnx](https://github.com/Cytnx-dev/Cytnx) UniTensor as the tensor backend.

## Features

- **Matrix Product States (MPS)** — open-boundary, labeled-index site tensors with gauge/canonical form management
- **Matrix Product Operators (MPO)** — open-boundary operators following the upper-triangular MPO convention
- **Two-site DMRG sweep** — ground-state and excited-state optimization with SVD-based bond compression
- **Lanczos eigensolver** — type-agnostic Krylov-subspace solver for local eigenvalue problems
- **Environment caching** — lazy left/right environment tensor cache with automatic stale-window invalidation
- **Heisenberg XXZ Hamiltonian** — built-in spin-1/2 chain MPO builder

## Quick Start

```python
from MPS.mps_init import random_mps
from MPS.hamiltonian import heisenberg_mpo
from DMRG.dmrg_engine import DMRGEngine

N = 10  # number of sites

H   = heisenberg_mpo(N, J=1.0, delta=1.0, h=0.0)
psi = random_mps(N, phys_dim=2, bond_dim=4, seed=42)
psi.move_center(0)

engine = DMRGEngine(psi, H)

# DMRG schedule: (max_bond_dim, SVD_cutoff)
schedule = [(10, 0.0), (20, 0.0), (40, 1e-8), (40, 1e-8)]

for max_dim, cutoff in schedule:
    E, trunc = engine.sweep(max_dim=max_dim, cutoff=cutoff, num_center=2)
    print(f"E = {E:.8f}  trunc = {trunc:.2e}")

# Bethe ansatz (thermodynamic limit): E/N ≈ -0.4431
print(f"E/N = {E/N:.6f}")
```

See [examples/example_dmrg.py](examples/example_dmrg.py) for a complete runnable script.

## Installation

### Requirements

- Python ≥ 3.8
- [Cytnx](https://github.com/Cytnx-dev/Cytnx) — see the [installation guide](https://kaihsinwu.gitlab.io/Cytnx_doc/install.html)
- NumPy

To verify that Cytnx is installed correctly:

```bash
python test_install.py
```

