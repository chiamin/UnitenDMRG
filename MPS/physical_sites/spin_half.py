"""Factory function for spin-1/2 physical site."""

from __future__ import annotations

import numpy as np

try:
    import cytnx
except ImportError as exc:
    raise ImportError("cytnx is required for spin_half.py.") from exc

from .site import PhysicalSite
from ..uniTensor_core import derive_delta_qn


def spin_half(qn: str | None = None) -> PhysicalSite:
    """Create a spin-1/2 PhysicalSite with operators I, Sz, Sp, Sm registered.

    Basis (fixed):
        index 0 = |dn⟩  (N_up = 0)
        index 1 = |up⟩  (N_up = 1)

    Parameters
    ----------
    qn : None or "Sz"
        None  → dense bond, no symmetry.
        "Sz"  → U(1) N_up-symmetric bond (QN = 0 for |dn⟩, 1 for |up⟩).
    """
    if qn is None:
        bond = cytnx.Bond(2, cytnx.BD_IN)
    elif qn == "Sz":
        sym  = cytnx.Symmetry.U1()
        bond = cytnx.Bond(cytnx.BD_IN, [[0], [1]], [1, 1], [sym])
    else:
        raise ValueError(f"Unknown qn='{qn}'. Supported values: None, 'Sz'.")

    I  = np.eye(2, dtype=float)
    Sz = np.array([[ 0.5,  0.0], [ 0.0, -0.5]], dtype=float)
    Sp = np.array([[ 0.0,  0.0], [ 1.0,  0.0]], dtype=float)  # Sp|dn>=|up>: [1,0]=1
    Sm = np.array([[ 0.0,  1.0], [ 0.0,  0.0]], dtype=float)  # Sm|up>=|dn>: [0,1]=1

    ops = {
        "I":  (I,  derive_delta_qn(I,  bond)),
        "Sz": (Sz, derive_delta_qn(Sz, bond)),
        "Sp": (Sp, derive_delta_qn(Sp, bond)),
        "Sm": (Sm, derive_delta_qn(Sm, bond)),
    }
    return PhysicalSite(bond, type_name="SpinHalf", ops=ops)
