"""Direct measurement helpers for MPS/MPO contractions.

This module provides plain contractions for

- ``inner(psi, phi)``        -> ``<psi|phi>``
- ``expectation(psi, mpo, phi)`` -> ``<psi|mpo|phi>``

without constructing environment cache objects.
"""

from __future__ import annotations

try:
    import cytnx
except ImportError as exc:
    raise ImportError("cytnx is required for measure.py.") from exc

from .uniTensor_core import scalar_from_uniTensor
from .uniTensor_utils import any_complex_tensors


def _assert_same_length(psi, phi, *, with_mpo=None) -> None:
    if with_mpo is None:
        if len(psi) != len(phi):
            raise ValueError(
                f"MPS length mismatch: len(psi)={len(psi)} != len(phi)={len(phi)}."
            )
        return

    if not (len(psi) == len(phi) == len(with_mpo)):
        raise ValueError(
            "Length mismatch: len(psi), len(phi), len(mpo) must be equal; got "
            f"{len(psi)}, {len(phi)}, {len(with_mpo)}."
        )


def _assert_phys_compatible(b1, b2, *, where: str) -> None:
    mismatch = (
        b1.dim() != b2.dim()
        or b1.Nsym() != b2.Nsym()
        or b1.qnums() != b2.qnums()
        or b1.getDegeneracies() != b2.getDegeneracies()
    )
    if mismatch:
        raise ValueError(f"Physical bond mismatch at {where}.")


def inner(psi, phi) -> float | complex:
    """Return ``<psi|phi>`` via direct left-to-right contraction.

    Args:
        psi: Bra MPS.
        phi: Ket MPS.
    """
    _assert_same_length(psi, phi)
    for site, (a_psi, a_phi) in enumerate(zip(psi, phi)):
        _assert_phys_compatible(
            a_psi.bond("i"),
            a_phi.bond("i"),
            where=f"site {site} between psi and phi",
        )

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
    """Return ``<psi|mpo|phi>`` via direct left-to-right contraction.

    Args:
        psi: Bra MPS.
        mpo: Operator MPO.
        phi: Ket MPS.
    """
    _assert_same_length(psi, phi, with_mpo=mpo)
    for site, (a_psi, w, a_phi) in enumerate(zip(psi, mpo, phi)):
        _assert_phys_compatible(
            a_phi.bond("i"),
            w.bond("i"),
            where=f"site {site} between phi and mpo.i",
        )
        _assert_phys_compatible(
            a_psi.bond("i"),
            w.bond("ip"),
            where=f"site {site} between psi and mpo.ip",
        )

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
