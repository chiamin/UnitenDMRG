"""Heisenberg spin-1/2 chain MPO.

Model
-----
    H = J Σ_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Δ Sz_i Sz_{i+1}) + h Σ_i Sz_i

Using S⁺/S⁻ form: Sx Sx + Sy Sy = (S⁺ S⁻ + S⁻ S⁺) / 2

Upper-triangular MPO with virtual bond dimension 5:

    W = | I        0        0        0    0 |
        | S⁺       0        0        0    0 |
        | S⁻       0        0        0    0 |
        | Sz       0        0        0    0 |
        | hSz    J/2 S⁻   J/2 S⁺   JΔSz  I |

    Left  boundary L0:  selects the last row  → L0[4, 0, 0] = 1
    Right boundary R0:  selects the first col → R0[0, 0, 0] = 1

MPO site tensor has shape [d_mpo, d_phys, d_phys, d_mpo] with
labels ["l", "ip", "i", "r"].
    l  : left  MPO virtual bond
    ip : outgoing (bra) physical index
    i  : incoming (ket) physical index
    r  : right MPO virtual bond

Physical basis (d=2): |0⟩ = |↑⟩, |1⟩ = |↓⟩

    Sz = diag(+1/2, -1/2)
    S⁺ |↓⟩ = |↑⟩  → S⁺[0,1] = 1
    S⁻ |↑⟩ = |↓⟩  → S⁻[1,0] = 1
    I  = identity
"""

from __future__ import annotations

import numpy as np

try:
    import cytnx
except ImportError as exc:
    raise ImportError("cytnx is required for heisenberg.py.") from exc

import sys
from pathlib import Path
# Allow running standalone: add project root to sys.path
for _p in [Path(__file__).resolve().parents[2]]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from MPS.mpo import MPO


# ---------------------------------------------------------------------------
# Local spin-1/2 operators (2x2 numpy arrays, basis: |↑⟩, |↓⟩)
# ---------------------------------------------------------------------------

def _spin_operators() -> dict[str, np.ndarray]:
    """Return Sz, S+, S-, I as numpy arrays."""
    I  = np.eye(2, dtype=float)
    Sz = np.array([[ 0.5,  0.0],
                   [ 0.0, -0.5]], dtype=float)
    Sp = np.array([[0.0, 1.0],    # S⁺|↓⟩ = |↑⟩
                   [0.0, 0.0]], dtype=float)
    Sm = np.array([[0.0, 0.0],    # S⁻|↑⟩ = |↓⟩
                   [1.0, 0.0]], dtype=float)
    return {"I": I, "Sz": Sz, "Sp": Sp, "Sm": Sm}


# ---------------------------------------------------------------------------
# Single MPO site tensor
# ---------------------------------------------------------------------------

def _make_w(ops: dict, J: float, delta: float, h: float) -> np.ndarray:
    """Build the 5x2x2x5 bulk MPO tensor W.

    Index order: [l, ip, i, r]  →  W[row, ip, i, col]

    The 5x5 operator matrix is:

        row\col   0      1        2       3      4
          0       I      0        0       0      0
          1       S⁺     0        0       0      0
          2       S⁻     0        0       0      0
          3       Sz     0        0       0      0
          4       hSz  J/2 S⁻   J/2 S⁺  JΔSz   I

    W[row, ip, i, col] = ops[row→col][ip, i]
    """
    d = 2           # physical dimension
    D = 5           # MPO virtual dimension
    I, Sz, Sp, Sm = ops["I"], ops["Sz"], ops["Sp"], ops["Sm"]

    W = np.zeros((D, d, d, D), dtype=float)

    # Column 0 (rightmost non-zero block in each row accumulates interactions)
    W[0, :, :, 0] = I
    W[1, :, :, 0] = Sp
    W[2, :, :, 0] = Sm
    W[3, :, :, 0] = Sz
    W[4, :, :, 0] = h * Sz

    # Row 4 (bottom row completes the hopping terms)
    W[4, :, :, 1] = (J / 2.0) * Sm    # pairs with S⁺ on left → S⁺ S⁻ term
    W[4, :, :, 2] = (J / 2.0) * Sp    # pairs with S⁻ on left → S⁻ S⁺ term
    W[4, :, :, 3] = J * delta * Sz    # pairs with Sz on left  → Sz Sz term
    W[4, :, :, 4] = I

    return W


def _numpy_to_mpo_site(arr: np.ndarray, site: int) -> "cytnx.UniTensor":
    """Convert [l, ip, i, r] numpy array to a labelled UniTensor MPO site."""
    # arr shape: [D, d, d, D]
    ut = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=2)
    ut.set_labels(["l", "ip", "i", "r"])
    return ut


# ---------------------------------------------------------------------------
# Public constructor
# ---------------------------------------------------------------------------

def heisenberg_mpo(N: int, J: float = 1.0, delta: float = 1.0, h: float = 0.0) -> MPO:
    """Build the spin-1/2 Heisenberg chain MPO.

    Parameters
    ----------
    N     : number of sites.
    J     : exchange coupling  (J > 0 → antiferromagnetic).
    delta : Ising anisotropy   (delta=1 → isotropic Heisenberg).
    h     : longitudinal field  (h * Σ Sz_i).

    Returns
    -------
    MPO object with site labels ["l", "ip", "i", "r"] and
    boundary tensors L0, R0 set automatically.
    """
    if N < 2:
        raise ValueError("Need at least 2 sites for a spin chain.")

    ops = _spin_operators()
    W   = _make_w(ops, J, delta, h)
    tensors = [_numpy_to_mpo_site(W, i) for i in range(N)]
    return MPO(tensors)
