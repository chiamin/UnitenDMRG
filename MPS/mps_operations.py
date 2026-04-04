"""High-level MPS/MPO operations.

This module collects standalone operations on MPS and MPO objects that do not
require a persistent environment cache.  It is intended to grow as new
operations are added.

Current contents
----------------
- `inner(psi, phi)`                                -> `<psi|phi>`
- `expectation(psi, mpo, phi)`                     -> `<psi|mpo|phi>`
- `mps_sum(psi, phi)`                              -> MPS representing |psi> + |phi>
- `mpo_sum(mpo1, mpo2)`                            -> MPO representing mpo1 + mpo2
- `mpo_product(mpo1, mpo2)`                        -> MPO representing mpo1 @ mpo2
- `fit_apply_mpo(mpo, mps_input, fitmps, ...)`     -> approximate MPO|mps_input> by fitting
- `fit_mpo_product(mpo1, mpo2, fitmpo, ...)`       -> approximate mpo1 @ mpo2 by fitting
"""

from __future__ import annotations

import sys

import cytnx

from .mps import MPS
from .mpo import MPO
from unitensor.core import assert_bond_match, direct_sum, scalar_from_uniTensor
from unitensor.utils import any_complex_tensors


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


def fit_apply_mpo(
    mpo,
    mps_input: MPS,
    fitmps: MPS,
    *,
    num_center: int = 2,
    nsweep: int = 1,
    max_dim: int | None = None,
    cutoff: float = 0.0,
    normalize: bool = False,
) -> MPS:
    """Approximate `|fitmps> ≈ MPO|mps_input>` by variational fitting.

    Minimizes `‖MPO|mps_input> - |fitmps>‖` by sweeping over the MPS sites and
    solving each local subspace problem exactly (one contraction step, no
    eigensolver).

    `fitmps` is modified in-place and also returned.  The caller is responsible
    for providing a reasonable initial `fitmps`; a closer initial guess
    converges faster (often one sweep suffices).

    Parameters
    ----------
    mpo       : MPO — operator to apply.
    mps_input : MPS — input ket state |ψ>.
    fitmps    : MPS — initial guess for the output state; modified in-place.
                Must have a single orthogonality center at site 0.
    num_center : 1 or 2 — number of sites optimised per local step.
    nsweep    : int — number of full (right + left) sweeps to perform.
    max_dim   : int | None — maximum bond dimension to keep per bond.
    cutoff    : float — discard Schmidt components whose normalised rho
                eigenvalue is below this threshold.
    normalize : bool — if True, normalize `fitmps` after all sweeps.

    Returns
    -------
    fitmps : the updated MPS (same object that was passed in).

    Raises
    ------
    ValueError : if lengths mismatch, `num_center` is not 1 or 2, or
                 `fitmps.center` is not 0 at entry.
    """
    from DMRG.environment import OperatorEnv
    from DMRG.effective_operators import EffOperator

    N = len(mps_input)
    if len(mpo) != N or len(fitmps) != N:
        raise ValueError(
            f"Length mismatch: mpo={len(mpo)}, mps_input={N}, fitmps={len(fitmps)}."
        )
    if num_center not in (1, 2):
        raise ValueError(f"num_center must be 1 or 2; got {num_center}.")
    if fitmps.center != 0:
        raise ValueError(
            f"fitmps.center must be 0 at entry; got {fitmps.center}."
        )

    n = num_center

    # Build environment for <fitmps|MPO|mps_input>.
    # bra=fitmps, ket=mps_input.  Observer callbacks are registered on both
    # so that fitmps.update_sites() automatically invalidates stale envs.
    op_env = OperatorEnv(mps_input, fitmps, mpo, init_center=0)

    def _local_update(p: int, absorb: str) -> None:
        op_env.update_envs(p, p + n - 1)
        mpo_tensors = [mpo[p + k] for k in range(n)]
        effH = EffOperator(op_env[p - 1], op_env[p + n], *mpo_tensors)
        phi_in = mps_input.make_phi(p, n)
        phi_new = effH.apply(phi_in)
        fitmps.update_sites(p, phi_new, max_dim=max_dim, cutoff=cutoff, absorb=absorb)

    for _ in range(nsweep):
        # Sweep right: p = 0 … N-2 (same for 1-site and 2-site).
        # After this loop the orthogonality center of fitmps is at N-1.
        for p in range(N - 1):
            _local_update(p, absorb="right")

        # Sweep left.
        # 2-site: p = N-2 … 0,  1-site: p = N-1 … 1.
        # After this loop the center is back at 0.
        left_start = N - n
        left_stop  = 0 if n == 2 else 1
        for p in range(left_start, left_stop - 1, -1):
            _local_update(p, absorb="left")

    if normalize:
        fitmps.normalize()

    return fitmps


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


def mpo_product(mpo1: MPO, mpo2: MPO) -> MPO:
    """Return the MPO representing mpo1 @ mpo2 (mpo1 applied after mpo2).

    At each site, contracts `mpo1[k].bond("i")` with `mpo2[k].bond("ip")`,
    then combines the virtual bonds via `combineBonds`.  The result has
    `bond("ip")` from mpo1 and `bond("i")` from mpo2.

    Virtual bond dimensions become the product of the input dimensions:
    `D_out[k] = D1[k] * D2[k]`.

    Physical bonds at each site must match: `mpo1[k].bond("i")` must be
    compatible with `mpo2[k].bond("ip")`.
    """
    N = len(mpo1)
    if N != len(mpo2):
        raise ValueError(f"MPO length mismatch: {len(mpo1)} != {len(mpo2)}.")
    tensors = []
    for k in range(N):
        upper = mpo1[k].clone().relabels(
            ["l", "ip", "i", "r"], ["l1", "ip", "_s", "r1"],
        )
        lower = mpo2[k].clone().relabels(
            ["l", "ip", "i", "r"], ["l2", "_s", "i", "r2"],
        )
        c = cytnx.Contract(upper, lower)
        c = c.permute(["l1", "l2", "ip", "i", "r1", "r2"])
        c.combineBonds(["l1", "l2"])
        c.combineBonds(["r1", "r2"])
        c.relabels_(["l1", "ip", "i", "r1"], ["l", "ip", "i", "r"])
        tensors.append(c)
    return MPO(tensors)


def fit_mpo_product(
    mpo1: MPO,
    mpo2: MPO,
    fitmpo: MPO,
    *,
    nsweep: int = 1,
    max_dim: int | None = None,
    cutoff: float = 0.0,
) -> MPO:
    """Approximate `fitmpo ≈ mpo1 @ mpo2` by variational fitting.

    Minimizes `‖mpo1·mpo2 - fitmpo‖_F` by sweeping over MPO sites and
    solving each local problem exactly (one contraction step, no eigensolver).
    Never constructs the exact D² product.

    `fitmpo` is modified in-place and also returned.

    Parameters
    ----------
    mpo1    : MPO — first (upper) operator.
    mpo2    : MPO — second (lower) operator.
    fitmpo  : MPO — initial guess, modified in-place.  Should be
              right-canonicalized (orthogonality center at site 0).
    nsweep  : int — number of full (right + left) sweeps.
    max_dim : int | None — maximum bond dimension per bond.
    cutoff  : float — discard singular values below this threshold
              (normalized rho eigenvalue).

    Returns
    -------
    fitmpo : the updated MPO (same object that was passed in).
    """
    from DMRG.environment import MPOProductEnv
    from .mpo_compression import _svd_two_mpo_sites

    N = len(mpo1)
    if len(mpo2) != N or len(fitmpo) != N:
        raise ValueError(
            f"Length mismatch: mpo1={len(mpo1)}, mpo2={len(mpo2)}, fitmpo={len(fitmpo)}."
        )

    dim = sys.maxsize if max_dim is None else max_dim

    # Build environment for Tr[fitmpo† · mpo1 · mpo2].
    env = MPOProductEnv(mpo1, mpo2, fitmpo, init_center=0)

    def _contract_local(p: int) -> "cytnx.UniTensor":
        """Contract L[p-1], W1[p], W2[p], R[p+1] → optimal 1-site M_fit[p].

        Open legs: L.up→l, W1.ip→ip, W2.i→i, R.up→r
        """
        Lr = env[p - 1].relabels(["up", "mid", "dn"], ["l", "_mid", "_dn"])
        W1 = mpo1[p].relabels(
            ["l", "ip", "i", "r"], ["_mid", "ip", "_i1", "_mid2"],
        )
        W2 = mpo2[p].relabels(
            ["l", "ip", "i", "r"], ["_dn", "_i1", "i", "_dn2"],
        )
        Rr = env[p + 1].relabels(["up", "mid", "dn"], ["r", "_mid2", "_dn2"])
        tmp = cytnx.Contract(Lr, W1)
        tmp = cytnx.Contract(tmp, W2)
        return cytnx.Contract(tmp, Rr)  # labels: ["l", "ip", "i", "r"]

    def _contract_local_2site(p: int) -> "cytnx.UniTensor":
        """Contract L[p-1], W1[p], W2[p], W1[p+1], W2[p+1], R[p+2]
        → optimal 2-site merged tensor with labels [l, ip0, i0, ip1, i1, r].
        """
        Lr = env[p - 1].relabels(["up", "mid", "dn"], ["l", "_mid", "_dn"])
        W1a = mpo1[p].relabels(
            ["l", "ip", "i", "r"], ["_mid", "ip0", "_i1a", "_m1"],
        )
        W2a = mpo2[p].relabels(
            ["l", "ip", "i", "r"], ["_dn", "_i1a", "i0", "_d1"],
        )
        W1b = mpo1[p + 1].relabels(
            ["l", "ip", "i", "r"], ["_m1", "ip1", "_i1b", "_mid2"],
        )
        W2b = mpo2[p + 1].relabels(
            ["l", "ip", "i", "r"], ["_d1", "_i1b", "i1", "_dn2"],
        )
        Rr = env[p + 2].relabels(["up", "mid", "dn"], ["r", "_mid2", "_dn2"])
        tmp = cytnx.Contract(Lr, W1a)
        tmp = cytnx.Contract(tmp, W2a)
        tmp = cytnx.Contract(tmp, W1b)
        tmp = cytnx.Contract(tmp, W2b)
        return cytnx.Contract(tmp, Rr)

    def _update_1site(p: int, absorb: str) -> None:
        env.update_envs(p, p)
        new_tensor = _contract_local(p)
        # SVD to shift the orthogonality center.
        if absorb == "right" and p < N - 1:
            left_new, right_new, _ = _svd_two_mpo_sites(
                new_tensor, fitmpo.tensors[p + 1],
                absorb="right", dim=dim, cutoff=cutoff,
            )
            fitmpo.tensors[p] = left_new
            fitmpo.tensors[p + 1] = right_new
        elif absorb == "left" and p > 0:
            left_new, right_new, _ = _svd_two_mpo_sites(
                fitmpo.tensors[p - 1], new_tensor,
                absorb="left", dim=dim, cutoff=cutoff,
            )
            fitmpo.tensors[p - 1] = left_new
            fitmpo.tensors[p] = right_new
        else:
            fitmpo.tensors[p] = new_tensor
        # Notify the environment that fitmpo changed.
        env.delete(p)

    def _update_2site(p: int, absorb: str) -> None:
        env.update_envs(p, p + 1)
        merged = _contract_local_2site(p)
        # SVD split: row = [l, ip0, i0], col = [ip1, i1, r]
        from unitensor.core import svd_by_labels
        left_new, right_new, _ = svd_by_labels(
            merged,
            row_labels=["l", "ip0", "i0"],
            absorb=absorb,
            dim=dim, cutoff=cutoff, aux_label="_aux",
        )
        left_new.relabels_(["ip0", "i0", "_aux"], ["ip", "i", "r"])
        right_new.relabels_(["_aux", "ip1", "i1"], ["l", "ip", "i"])
        fitmpo.tensors[p] = left_new
        fitmpo.tensors[p + 1] = right_new
        env.delete(p)
        env.delete(p + 1)

    for _ in range(nsweep):
        # Sweep right: p = 0 … N-2
        for p in range(N - 1):
            _update_2site(p, absorb="right")

        # Sweep left: p = N-2 … 0
        for p in range(N - 2, -1, -1):
            _update_2site(p, absorb="left")

    return fitmpo
