"""Test-only helpers for building MPS (and matching MPO) across all required dtype combinations.

Not part of the public library API.  Import from this module in test files.
"""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

import numpy as np

try:
    import cytnx
except ImportError:
    cytnx = None

if cytnx is not None:
    from MPS.mps import MPS
    from MPS.physical_sites.spin_half import spin_half


# ---------------------------------------------------------------------------
# QN MPS helpers
# ---------------------------------------------------------------------------

def allowed_cumulative_nup_after_site(site_index: int, num_sites: int, n_up_total: int) -> list[int]:
    """Allowed cumulative N_up on the right bond of `site_index` (inclusive prefix).

    Only charges that can still reach `n_up_total` at the last bond are kept:

        max(0, n_up_total - (num_sites - 1 - site_index))
            <= q <=
        min(site_index + 1, n_up_total)
    """
    lo = max(0, n_up_total - (num_sites - 1 - site_index))
    hi = min(site_index + 1, n_up_total)
    if lo > hi:
        return []
    return list(range(lo, hi + 1))


def random_u1_sz_mps(
    num_sites: int,
    n_up_total: int,
    *,
    seed: int | None = None,
    dtype: np.dtype | type = float,
    center: int = 0,
    normalize: bool = False,
) -> "MPS":
    """Random open MPS: spin-1/2 U(1) N_up, fixed total `n_up_total`.

    Each admissible virtual QN sector has bond dimension 1.  Virtual sectors
    are exactly those cumulative charges that can still reach `n_up_total` at
    the right boundary.

    If `normalize` is True, the MPS is orthogonalized (center moved to
    `center`) and normalized to unit norm.  Otherwise only `center_left` /
    `center_right` are set to `center` without any canonicalization.
    """
    if cytnx is None:
        raise RuntimeError("cytnx is required.")
    if num_sites < 1:
        raise ValueError("num_sites must be >= 1.")
    if not 0 <= n_up_total <= num_sites:
        raise ValueError(
            f"n_up_total must be in [0, num_sites]; got n_up_total={n_up_total}, "
            f"num_sites={num_sites}."
        )
    if not 0 <= center < num_sites:
        raise IndexError(f"center={center} out of range [0, {num_sites - 1}].")

    if seed is not None:
        np.random.seed(seed)

    site = spin_half(qn="Sz")
    phys_bond = site.bond
    sym = cytnx.Symmetry.U1()
    syms = [sym]

    out_dtype = np.dtype(dtype)
    if np.issubdtype(out_dtype, np.complexfloating):
        ut_dtype = cytnx.Type.ComplexDouble
    else:
        ut_dtype = cytnx.Type.Double

    tensors: list[cytnx.UniTensor] = []

    for k in range(num_sites):
        q_list = allowed_cumulative_nup_after_site(k, num_sites, n_up_total)
        if not q_list:
            raise RuntimeError(
                f"Empty bond sector list at site {k} (num_sites={num_sites}, "
                f"n_up_total={n_up_total})."
            )

        b_r = cytnx.Bond(
            cytnx.BD_OUT,
            [[q] for q in q_list],
            [1] * len(q_list),
            syms,
        )

        if k == 0:
            b_l = cytnx.Bond(cytnx.BD_IN, [[0]], [1], syms)
        else:
            b_l = tensors[k - 1].bond("r").redirect()

        ut = cytnx.UniTensor([b_l, phys_bond, b_r], rowrank=2, dtype=ut_dtype)
        ut.set_labels(["l", "i", "r"])

        cytnx.random.uniform_(ut, -1.0, 1.0)
        tensors.append(ut)

    mps = MPS(tensors)
    mps.center_left = 0
    mps.center_right = len(tensors) - 1
    if normalize:
        mps.orthogonalize(center)
        mps.normalize()
    else:
        mps.move_center(center)
    return mps


def product_u1_sz_mps(
    config: list[int],
    *,
    dtype: np.dtype | type = float,
) -> "MPS":
    """Product-state MPS with U(1) N_up symmetry.

    Each site is in a definite spin state (0 = down, 1 = up).
    Bond dimension is 1 everywhere. Every site is orthonormal,
    so the MPS is canonical with any choice of center.

    Parameters
    ----------
    config : list[int]
        Spin configuration, one entry per site: 0 = |dn>, 1 = |up>.
    dtype : float or complex
        Tensor dtype.

    Returns
    -------
    MPS with `center_left = 0`, `center_right = 0`.
    """
    if cytnx is None:
        raise RuntimeError("cytnx is required.")
    num_sites = len(config)
    if num_sites < 1:
        raise ValueError("config must have at least one site.")
    for k, s in enumerate(config):
        if s not in (0, 1):
            raise ValueError(f"config[{k}] must be 0 or 1; got {s}.")

    site = spin_half(qn="Sz")
    phys_bond = site.bond
    sym = cytnx.Symmetry.U1()
    syms = [sym]

    out_dtype = np.dtype(dtype)
    ut_dtype = (cytnx.Type.ComplexDouble
                if np.issubdtype(out_dtype, np.complexfloating)
                else cytnx.Type.Double)

    tensors: list[cytnx.UniTensor] = []
    cumulative_nup = 0

    for k in range(num_sites):
        # Left bond: cumulative N_up before this site
        b_l = cytnx.Bond(cytnx.BD_IN, [[cumulative_nup]], [1], syms)

        cumulative_nup += config[k]

        # Right bond: cumulative N_up after this site
        b_r = cytnx.Bond(cytnx.BD_OUT, [[cumulative_nup]], [1], syms)

        ut = cytnx.UniTensor([b_l, phys_bond, b_r], rowrank=2, dtype=ut_dtype)
        ut.set_labels(["l", "i", "r"])

        # Set the single nonzero element: config[k] selects which physical
        # index is 1.  The block that matches the QN sector (qn_l, qn_phys,
        # qn_r) has shape (1, 1, 1).
        for blk_idx in range(ut.Nblocks()):
            blk = ut.get_block(blk_idx)
            arr = np.zeros(blk.shape(), dtype=out_dtype)
            ut.put_block(cytnx.from_numpy(arr), blk_idx)

        # Find the correct block for config[k] and set it to 1.
        # Physical qnums: index 0 = qn 0 (dn), index 1 = qn 1 (up).
        # The valid block has qn_l + qn_phys = qn_r.
        # qn_l = cumulative_nup - config[k], qn_phys = config[k], qn_r = cumulative_nup.
        # There is exactly one block satisfying this.
        for blk_idx in range(ut.Nblocks()):
            blk = ut.get_block(blk_idx)
            if blk.shape() == [1, 1, 1]:
                one = np.ones([1, 1, 1], dtype=out_dtype)
                ut.put_block(cytnx.from_numpy(one), blk_idx)
                break

        tensors.append(ut)

    mps = MPS(tensors)
    mps.center_left = 0
    mps.center_right = 0
    return mps
