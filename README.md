# Uni10DMRG — DMRG with Cytnx UniTensor

A Python implementation of the **Density Matrix Renormalization Group (DMRG)** algorithm for 1D quantum many-body systems, built on top of [Cytnx](https://github.com/Cytnx-dev/Cytnx) UniTensor as the tensor backend.

## Quick Start

```python
from MPS.mps_init import random_mps
from MPS.physical_sites import spin_half
from MPS.auto_mpo import AutoMPO
from DMRG.dmrg_engine import DMRGEngine

N = 10  # number of sites

site = spin_half()
ampo = AutoMPO(N, site)
for i in range(N - 1):
    ampo.add(1.0, "Sz", i, "Sz", i + 1)
    ampo.add(0.5, "Sp", i, "Sm", i + 1)
    ampo.add(0.5, "Sm", i, "Sp", i + 1)
H = ampo.to_mpo()

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

See [examples/](examples/) for more runnable scripts.

## Installation

### Requirements

- Python ≥ 3.8
- [Cytnx](https://github.com/Cytnx-dev/Cytnx) — see the [installation guide](https://kaihsinwu.gitlab.io/Cytnx_doc/install.html)
- NumPy

To verify that Cytnx is installed correctly:

```bash
python tests/smoke/test_install.py
```

