"""
TDVP QN example: real-time evolution on a Heisenberg chain.

Shows:
1) U1-symmetric (Sz) tensors with fixed quantum-number sector.
2) Real-time TDVP (dt = 1j * delta_t): norm/energy should be conserved.
"""

import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from MPS.physical_sites import spin_half
from MPS.auto_mpo import AutoMPO
from MPS.measure import expectation, inner
from TDVP.tdvp_engine import TDVPEngine


def measure_energy(psi, H_mpo):
    """Compute <psi|H|psi> / <psi|psi> by direct tensor contraction."""
    numer = expectation(psi, H_mpo, psi)
    denom = inner(psi, psi)
    return numer / denom


def build_heisenberg(site, N, J=1.0, delta=1.0, h=0.0):
    """Build Heisenberg MPO from AutoMPO for the given physical site."""
    ampo = AutoMPO(N, site)
    Jc = complex(J)
    deltac = complex(delta)
    hc = complex(h)
    for i in range(N - 1):
        ampo.add(Jc * deltac, "Sz", i, "Sz", i + 1)
        ampo.add(Jc / 2.0, "Sp", i, "Sm", i + 1)
        ampo.add(Jc / 2.0, "Sm", i, "Sp", i + 1)
    if h != 0.0:
        for i in range(N):
            ampo.add(hc, "Sz", i)
    return ampo.to_mpo()


def run_real_time(psi, H_mpo, n_sweeps=20, delta_t=0.1, max_dim=32):
    """Run real-time TDVP and print energy/norm trend.

    Note: with current backend, QN TDVP may fail in cytnx during environment
    contractions. We catch and report it to keep this example runnable.
    """
    engine = TDVPEngine(psi, H_mpo)

    print("\nReal-time TDVP (num_center=2)")
    print(f"{'step':>4}  {'E':>14}  {'norm':>12}  {'trunc':>10}")
    print("-" * 48)
    for step in range(1, n_sweeps + 1):
        try:
            trunc = engine.sweep(
                dt=1j * delta_t,
                num_center=2,
                max_dim=max_dim,
                cutoff=1e-10,
            )
            energy = measure_energy(psi, H_mpo)
            norm = psi.norm()
            print(f"{step:>4}  {energy:>14.8f}  {norm:>12.8f}  {trunc:>10.2e}")
        except RuntimeError as err:
            print("\nQN TDVP is currently not supported by this backend build.")
            print("Encountered runtime error during sweep:")
            print(f"  {err}")
            print("\nTip: use dense TDVP for now, or run imaginary-time/QN with DMRG.")
            break


def main():
    N = 8
    site = spin_half(qn="Sz")
    H = build_heisenberg(site, N, J=1.0, delta=1.0, h=0.0)

    # Fixed-Sz sector initial state (N_up = N/2): Neel product state.
    neel = [i % 2 for i in range(N)]  # [0,1,0,1,...]
    psi0 = site.product_state(neel, center=0, dtype=complex)
    psi_it = psi0.copy()

    print("=" * 60)
    print("TDVP QN example: Heisenberg chain")
    print("=" * 60)
    print(f"N = {N}")
    print(f"QN sector (total N_up) = {psi0.total_qn}")
    print(f"Initial energy = {measure_energy(psi0, H):.8f}")
    print(f"Initial norm   = {psi0.norm():.8f}")

    run_real_time(psi_it, H, n_sweeps=20, delta_t=0.1, max_dim=32)


if __name__ == "__main__":
    main()
