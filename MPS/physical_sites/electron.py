"""Factory function for spin-1/2 electron physical site (Hubbard model)."""

from __future__ import annotations

import numpy as np

import cytnx

from .site import PhysicalSite
from unitensor.core import derive_delta_qn


def electron(qn: str | None = None) -> PhysicalSite:
    """Create a spin-1/2 electron PhysicalSite.

    Basis (fixed):
        index 0 = |0⟩    (empty)
        index 1 = |↑⟩    (spin up)
        index 2 = |↓⟩    (spin down)
        index 3 = |↑↓⟩   (doubly occupied)

    Convention: |↑↓⟩ = c†_↑ c†_↓ |0⟩  (c†_↓ acts first).

    Operators
    ---------
    I       identity
    Cup     c_↑  (annihilate spin-up)
    Cupdag  c†_↑ (create spin-up)
    Cdn     c_↓  (annihilate spin-down; picks up sign from crossing c†_↑)
    Cdndag  c†_↓ (create spin-down; picks up sign from crossing c†_↑)
    Nup     n_↑ = c†_↑ c_↑
    Ndn     n_↓ = c†_↓ c_↓
    Ntot    n_↑ + n_↓
    Sz      (n_↑ - n_↓) / 2
    F       (-1)^Ntot  (Jordan-Wigner parity)

    Parameters
    ----------
    qn : None, "Ntot", "Sz", "Ntot,Sz", or "Nup,Ndn"
        None        → dense bond, no symmetry.
        "Ntot"      → U(1) total particle number.
        "Sz"        → U(1) spin-z.
        "Ntot,Sz"   → U(1) × U(1).
        "Nup,Ndn"   → U(1) × U(1).
    """
    # ------------------------------------------------------------------
    # Physical bond
    # ------------------------------------------------------------------
    #           |0⟩   |↑⟩   |↓⟩   |↑↓⟩
    # Ntot:      0     1     1     2
    # Sz:        0    +1    -1     0
    # Nup:       0     1     0     1
    # Ndn:       0     0     1     1

    if qn is None:
        bond = cytnx.Bond(4, cytnx.BD_IN)
    elif qn == "Ntot":
        sym = cytnx.Symmetry.U1()
        bond = cytnx.Bond(cytnx.BD_IN,
                          [[0], [1], [1], [2]], [1, 1, 1, 1], [sym])
    elif qn == "Sz":
        sym = cytnx.Symmetry.U1()
        bond = cytnx.Bond(cytnx.BD_IN,
                          [[0], [1], [-1], [0]], [1, 1, 1, 1], [sym])
    elif qn == "Ntot,Sz":
        sym_n = cytnx.Symmetry.U1()
        sym_s = cytnx.Symmetry.U1()
        bond = cytnx.Bond(cytnx.BD_IN,
                          [[0, 0], [1, 1], [1, -1], [2, 0]],
                          [1, 1, 1, 1], [sym_n, sym_s])
    elif qn == "Nup,Ndn":
        sym_up = cytnx.Symmetry.U1()
        sym_dn = cytnx.Symmetry.U1()
        bond = cytnx.Bond(cytnx.BD_IN,
                          [[0, 0], [1, 0], [0, 1], [1, 1]],
                          [1, 1, 1, 1], [sym_up, sym_dn])
    else:
        raise ValueError(
            f"Unknown qn='{qn}'. "
            "Supported: None, 'Ntot', 'Sz', 'Ntot,Sz', 'Nup,Ndn'."
        )

    # ------------------------------------------------------------------
    # Operator matrices (basis order: |0⟩, |↑⟩, |↓⟩, |↑↓⟩)
    # ------------------------------------------------------------------

    I = np.eye(4, dtype=float)

    # Cup: c_↑ |↑⟩=|0⟩, c_↑ |↑↓⟩=+|↓⟩
    Cup = np.zeros((4, 4), dtype=float)
    Cup[0, 1] = 1.0    # |0⟩ ← |↑⟩
    Cup[2, 3] = 1.0    # |↓⟩ ← |↑↓⟩  (no sign: c_↑ is outermost)

    # Cupdag: transpose
    Cupdag = Cup.T.copy()

    # Cdn: c_↓ |↓⟩=|0⟩, c_↓ |↑↓⟩=-|↑⟩  (crosses c†_↑)
    Cdn = np.zeros((4, 4), dtype=float)
    Cdn[0, 2] = 1.0    # |0⟩ ← |↓⟩
    Cdn[1, 3] = -1.0   # |↑⟩ ← |↑↓⟩  (sign from anti-commuting past c†_↑)

    # Cdndag: transpose
    Cdndag = Cdn.T.copy()

    # Number operators
    Nup  = np.diag([0.0, 1.0, 0.0, 1.0])
    Ndn  = np.diag([0.0, 0.0, 1.0, 1.0])
    Ntot = np.diag([0.0, 1.0, 1.0, 2.0])
    Sz   = np.diag([0.0, 0.5, -0.5, 0.0])

    # Jordan-Wigner parity: (-1)^Ntot
    F = np.diag([1.0, -1.0, -1.0, 1.0])

    # ------------------------------------------------------------------
    # Register
    # ------------------------------------------------------------------

    ops = {
        "I":       (I,       derive_delta_qn(I,       bond), False),
        "Cup":     (Cup,     derive_delta_qn(Cup,     bond), True),
        "Cupdag":  (Cupdag,  derive_delta_qn(Cupdag,  bond), True),
        "Cdn":     (Cdn,     derive_delta_qn(Cdn,     bond), True),
        "Cdndag":  (Cdndag,  derive_delta_qn(Cdndag,  bond), True),
        "Nup":     (Nup,     derive_delta_qn(Nup,     bond), False),
        "Ndn":     (Ndn,     derive_delta_qn(Ndn,     bond), False),
        "Ntot":    (Ntot,    derive_delta_qn(Ntot,    bond), False),
        "Sz":      (Sz,      derive_delta_qn(Sz,      bond), False),
        "F":       (F,       derive_delta_qn(F,       bond), False),
    }
    return PhysicalSite(bond, type_name="Electron", ops=ops)
