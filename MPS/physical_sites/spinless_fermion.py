"""Factory function for spinless fermion physical site."""

from __future__ import annotations

import numpy as np

import cytnx

from .site import PhysicalSite
from unitensor.core import derive_delta_qn


def spinless_fermion(qn: str | None = None) -> PhysicalSite:
    """Create a spinless fermion PhysicalSite with operators I, N, C, Cdag, F.

    Basis (fixed):
        index 0 = |0⟩  (empty,    N = 0)
        index 1 = |1⟩  (occupied, N = 1)

    Parameters
    ----------
    qn : None, "N", or "parity"
        None      → dense bond, no symmetry.
        "N"       → U(1) particle-number-symmetric bond (QN = 0, 1).
        "parity"  → Z2 fermion-parity-symmetric bond (QN = 0 even, 1 odd).
    """
    if qn is None:
        bond = cytnx.Bond(2, cytnx.BD_IN)
    elif qn == "N":
        sym = cytnx.Symmetry.U1()
        bond = cytnx.Bond(cytnx.BD_IN, [[0], [1]], [1, 1], [sym])
    elif qn == "parity":
        sym = cytnx.Symmetry.Zn(2)
        bond = cytnx.Bond(cytnx.BD_IN, [[0], [1]], [1, 1], [sym])
    else:
        raise ValueError(f"Unknown qn='{qn}'. Supported values: None, 'N', 'parity'.")

    I    = np.eye(2, dtype=float)
    N    = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=float)
    C    = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=float)   # c|1>=|0>: C[0,1]=1
    Cdag = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)   # c+|0>=|1>: Cdag[1,0]=1
    F    = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float)   # (-1)^N

    ops = {
        "I":    (I,    derive_delta_qn(I,    bond), False),
        "N":    (N,    derive_delta_qn(N,    bond), False),
        "C":    (C,    derive_delta_qn(C,    bond), True),
        "Cdag": (Cdag, derive_delta_qn(Cdag, bond), True),
        "F":    (F,    derive_delta_qn(F,    bond), False),
    }
    return PhysicalSite(bond, type_name="SpinlessFermion", ops=ops)
