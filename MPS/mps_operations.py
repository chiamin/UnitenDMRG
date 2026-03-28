"""High-level MPS/MPO operations.

This module collects standalone operations on MPS and MPO objects that do not
require a persistent environment cache.  It is intended to grow as new
operations are added.

Current contents
----------------
- `inner(psi, phi)`            -> `<psi|phi>`
- `expectation(psi, mpo, phi)` -> `<psi|mpo|phi>`
- `mps_sum(psi, phi)`          -> MPS representing |psi> + |phi>
- `mpo_sum(mpo1, mpo2)`        -> MPO representing mpo1 + mpo2
"""

from __future__ import annotations

import cytnx

from .mps import MPS
from .mpo import MPO
from .uniTensor_core import assert_bond_match, direct_sum, scalar_from_uniTensor
from .uniTensor_utils import any_complex_tensors


def inner(psi, phi) -> float | complex:
    """Return `<psi|phi>` via direct left-to-right contraction.

    Args:
        psi: Bra MPS.
        phi: Ket MPS.
    """
    if len(psi) != len(phi):
        raise ValueError(f"MPS length mismatch: {len(psi)} != {len(phi)}.")
    for a_psi, a_phi in zip(psi, phi):
        assert_bond_match(a_psi.bond("i"), a_phi.bond("i"))

    use_complex = any_complex_tensors(psi) or any_complex_tensors(phi)
    ut_dtype = cytnx.Type.ComplexDouble if use_complex else cytnx.Type.Double

    # Boundary legs:
    #   dn -> ket(phi) chain
    #   up -> bra(psi) chain
    b_dn = phi[0].bond("l").redirect()
    b_up = psi[0].bond("l")
    env = cytnx.UniTensor([b_dn, b_up], labels=["dn", "up"], dtype=ut_dtype)
    env.at([0, 0]).value = 1.0

    for a_phi, a_psi in zip(phi, psi):
        e = env.relabels(["dn", "up"], ["_dn", "_up"])
        ket = a_phi.relabels(["l", "i", "r"], ["_dn", "_i", "dn"])
        bra = a_psi.Dagger().relabels(["l", "i", "r"], ["_up", "_i", "up"])
        tmp = cytnx.Contract(e, ket)
        env = cytnx.Contract(tmp, bra)

    return scalar_from_uniTensor(env)


def expectation(psi, mpo, phi) -> float | complex:
    """Return `<psi|mpo|phi>` via direct left-to-right contraction.

    Args:
        psi: Bra MPS.
        mpo: Operator MPO.
        phi: Ket MPS.
    """
    if not len(psi) == len(phi) == len(mpo):
        raise ValueError(f"Length mismatch: len(psi)={len(psi)}, len(phi)={len(phi)}, len(mpo)={len(mpo)}.")
    for a_psi, w, a_phi in zip(psi, mpo, phi):
        assert_bond_match(a_phi.bond("i"), w.bond("i"))
        assert_bond_match(a_psi.bond("i"), w.bond("ip"))

    use_complex = (
        any_complex_tensors(psi)
        or any_complex_tensors(phi)
        or any_complex_tensors(mpo)
    )
    ut_dtype = cytnx.Type.ComplexDouble if use_complex else cytnx.Type.Double

    # Boundary legs:
    #   mid -> MPO chain
    #   dn  -> ket(phi) chain
    #   up  -> bra(psi) chain
    b_mid = mpo[0].bond("l").redirect()
    b_dn = phi[0].bond("l").redirect()
    b_up = psi[0].bond("l")
    env = cytnx.UniTensor([b_mid, b_dn, b_up], labels=["mid", "dn", "up"], dtype=ut_dtype)
    env.at([0, 0, 0]).value = 1.0

    for a_phi, w, a_psi in zip(phi, mpo, psi):
        e = env.relabels(["mid", "dn", "up"], ["_mid", "_dn", "_up"])
        ket = a_phi.relabels(["l", "i", "r"], ["_dn", "_i", "dn"])
        op = w.relabels(["l", "ip", "i", "r"], ["_mid", "_ip", "_i", "mid"])
        bra = a_psi.Dagger().relabels(["l", "i", "r"], ["_up", "_ip", "up"])

        tmp = cytnx.Contract(e, ket)
        tmp = cytnx.Contract(tmp, op)
        env = cytnx.Contract(tmp, bra)

    return scalar_from_uniTensor(env)


def mps_sum(psi: MPS, phi: MPS) -> MPS:
    """Return the MPS representing |psi> + |phi> via virtual-bond direct sum.

    For site 0:         direct-sum on "r" only (left boundary bond is shared).
    For interior sites: direct-sum on "l" and "r".
    For site N-1:       direct-sum on "l" only (right boundary bond is shared).

    Physical bonds at each site must match between psi and phi.
    Requires N >= 2.
    """
    N = len(psi)
    if N != len(phi):
        raise ValueError(f"MPS length mismatch: {len(psi)} != {len(phi)}.")
    if N < 2:
        raise ValueError("mps_sum requires at least 2 sites.")
    tensors = []
    for k in range(N):
        A, B = psi[k], phi[k]
        if k == 0:
            C = direct_sum(A, B, ["r"], ["r"], ["r"])
        elif k == N - 1:
            C = direct_sum(A, B, ["l"], ["l"], ["l"])
        else:
            C = direct_sum(A, B, ["l", "r"], ["l", "r"], ["l", "r"])
        tensors.append(C)
    result = MPS(tensors)
    result.center_left = 0
    result.center_right = N - 1
    return result


def mpo_sum(mpo1: MPO, mpo2: MPO) -> MPO:
    """Return the MPO representing mpo1 + mpo2 via virtual-bond direct sum.

    For site 0:         direct-sum on "r" only (left boundary bond is shared).
    For interior sites: direct-sum on "l" and "r".
    For site N-1:       direct-sum on "l" only (right boundary bond is shared).

    Physical bonds ("i", "ip") at each site must match between mpo1 and mpo2.
    Requires N >= 2.
    """
    N = len(mpo1)
    if N != len(mpo2):
        raise ValueError(f"MPO length mismatch: {len(mpo1)} != {len(mpo2)}.")
    if N < 2:
        raise ValueError("mpo_sum requires at least 2 sites.")
    tensors = []
    for k in range(N):
        A, B = mpo1[k], mpo2[k]
        if k == 0:
            C = direct_sum(A, B, ["r"], ["r"], ["r"])
        elif k == N - 1:
            C = direct_sum(A, B, ["l"], ["l"], ["l"])
        else:
            C = direct_sum(A, B, ["l", "r"], ["l", "r"], ["l", "r"])
        tensors.append(C)
    return MPO(tensors)
