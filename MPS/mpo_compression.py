"""MPO compression algorithms for open-chain MPO."""

from __future__ import annotations

import sys

import cytnx

from unitensor.core import svd_by_labels
from .mpo import MPO


def _svd_two_mpo_sites(
    left: "cytnx.UniTensor",
    right: "cytnx.UniTensor",
    *,
    absorb: str,
    dim: int,
    cutoff: float,
) -> tuple["cytnx.UniTensor", "cytnx.UniTensor", float]:
    """Merge two adjacent MPO site tensors, SVD-truncate, and split back.

    Args:
        left:   Site tensor with labels `l`, `ip`, `i`, `r`.
        right:  Site tensor with labels `l`, `ip`, `i`, `r`.
        absorb: `"left"` or `"right"` — which side absorbs the singular values.
        dim:    Maximum number of singular values to keep.
        cutoff: Discard components whose normalized rho eigenvalue is below this.

    Returns:
        Tuple of `(left_new, right_new, discarded_weight)`.
    """
    a1 = left.relabels(["i", "ip", "r"], ["i1", "ip1", "_"])
    a2 = right.relabels(["i", "ip", "l"], ["i2", "ip2", "_"])
    aa = cytnx.Contract(a1, a2)
    left_new, right_new, discarded = svd_by_labels(
        aa,
        row_labels=["l", "ip1", "i1"],
        absorb=absorb,
        dim=dim,
        cutoff=cutoff,
        aux_label="aux",
    )
    left_new.relabels_(["ip1", "i1", "aux"], ["ip", "i", "r"])
    right_new.relabels_(["ip2", "i2", "aux"], ["ip", "i", "l"])
    return left_new, right_new, discarded


def svd_compress_mpo(
    H: "MPO",
    *,
    max_dim: int | None = None,
    cutoff: float = 0.0,
) -> "MPO":
    """Compress an MPO via SVD truncation (two-pass sweep).

    First sweeps left-to-right without truncation (left-canonicalize), then
    sweeps right-to-left with SVD truncation so that singular values on each
    bond equal the Schmidt values.

    Parameters
    ----------
    H       : Input MPO.  Not modified.
    max_dim : Maximum bond dimension to keep per bond.  `None` = no limit.
    cutoff  : Discard Schmidt values whose normalized rho eigenvalue is below
              this threshold.

    Returns
    -------
    MPO with reduced bond dimensions.
    """
    mpo = H.copy()
    N = len(mpo)
    if N <= 1:
        return mpo

    dim = sys.maxsize if max_dim is None else max_dim

    # Left-to-right: no truncation, absorb right (left-canonicalize).
    for p in range(N - 1):
        left_new, right_new, _ = _svd_two_mpo_sites(
            mpo.tensors[p], mpo.tensors[p + 1],
            absorb="right", dim=sys.maxsize, cutoff=0.0,
        )
        mpo.tensors[p] = left_new
        mpo.tensors[p + 1] = right_new

    # Right-to-left: truncate.
    for p in range(N - 2, -1, -1):
        left_new, right_new, _ = _svd_two_mpo_sites(
            mpo.tensors[p], mpo.tensors[p + 1],
            absorb="left", dim=dim, cutoff=cutoff,
        )
        mpo.tensors[p] = left_new
        mpo.tensors[p + 1] = right_new

    return mpo
