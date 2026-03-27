"""
Product-state MPS example using PhysicalSite.

Demonstrates how to create a product-state MPS for a spin-1/2 chain,
both with and without U(1) Sz quantum number symmetry.

Spin-1/2 basis (spin_half convention):
    index 0 = |dn>  (N_up = 0)
    index 1 = |up>  (N_up = 1)
"""

import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from MPS.physical_sites import spin_half

N = 6

# Neel state: |dn up dn up dn up>
neel = [0, 1, 0, 1, 0, 1]

# ---------------------------------------------------------------------------
# Dense (no QN)
# ---------------------------------------------------------------------------
site = spin_half()
psi = site.product_state(neel)
print("Dense product state:")
print(f"  {psi}")
print(f"  bond dims = {psi.bond_dims}")
print(f"  norm      = {psi.norm():.6f}")
print()

# ---------------------------------------------------------------------------
# U(1) Sz symmetric
# ---------------------------------------------------------------------------
site_qn = spin_half(qn="Sz")
psi_qn = site_qn.product_state(neel)
print("QN (U1 Sz) product state:")
print(f"  {psi_qn}")
print(f"  bond dims = {psi_qn.bond_dims}")
print(f"  norm      = {psi_qn.norm():.6f}")
print()

# The virtual bond QNs accumulate N_up from the left.
# For neel = [0,1,0,1,0,1] (N_up = 0,1,0,1,0,1):
#   bond 0-1: N_up = 0
#   bond 1-2: N_up = 1
#   bond 2-3: N_up = 1
#   bond 3-4: N_up = 2
#   bond 4-5: N_up = 2
#   right boundary: N_up = 3  (total up spins = 3)
print("Virtual bond QNs (N_up = number of up spins from the left):")
for i in range(N - 1):
    b = psi_qn[i + 1].bond("l")
    print(f"  bond {i}-{i+1}: QN = {b.qnums()[0]}")
