"""MPS compression algorithms for open-chain MPS."""

from __future__ import annotations

import sys

import cytnx

from .uniTensor_core import svd_by_labels
from .mps import MPS


def _svd_two_sites(
    left: "cytnx.UniTensor",
    right: "cytnx.UniTensor",
    *,
    absorb: str,
    dim: int,
    cutoff: float,
) -> tuple["cytnx.UniTensor", "cytnx.UniTensor", float]:
    """Merge two adjacent MPS site tensors, SVD-truncate, and split back.

    Args:
        left:   Site tensor with labels `l`, `i`, `r`.
        right:  Site tensor with labels `l`, `i`, `r`.
        absorb: `"left"` or `"right"` — which side absorbs the singular values.
        dim:    Maximum number of singular values to keep.
        cutoff: Discard components whose normalized rho eigenvalue is below this.

    Returns:
        Tuple of `(left_new, right_new, discarded_weight)`.
    """
    a1 = left.relabels(["i", "r"], ["i1", "_"])
    a2 = right.relabels(["i", "l"], ["i2", "_"])
    aa = cytnx.Contract(a1, a2)   # labels: [l, i1, i2, r]
    left_new, right_new, discarded = svd_by_labels(
        aa,
        row_labels=["l", "i1"],
        absorb=absorb,
        dim=dim,
        cutoff=cutoff,
        aux_label="aux",
    )
    left_new.relabels_(["i1", "aux"], ["i", "r"])
    right_new.relabels_(["i2", "aux"], ["i", "l"])
    return left_new, right_new, discarded


def svd_compress_mps(
    psi: "MPS",
    *,
    max_dim: int | None = None,
    cutoff: float = 0.0,
) -> "MPS":
    """Compress an MPS via SVD truncation (right-to-left sweep).

    Moves the orthogonality center to site N-1 via QR (no truncation), so that
    singular values on each bond equal the Schmidt values.  Then performs a
    single right-to-left SVD sweep with truncation.

    Parameters
    ----------
    psi     : Input MPS.  Not modified.
    max_dim : Maximum bond dimension to keep per bond.  `None` = no limit.
    cutoff  : Discard Schmidt values whose normalized rho eigenvalue is below
              this threshold.

    Returns
    -------
    phi : New MPS with reduced bond dimensions.  Center at site 0.
    """
    phi = psi.copy()
    phi.move_center(len(phi) - 1)
    N   = len(phi)
    dim = sys.maxsize if max_dim is None else max_dim

    # Right-to-left: truncate each bond; singular values equal Schmidt values
    # because phi is left-canonical (center at N-1).
    for p in range(N - 2, -1, -1):
        left_new, right_new, _ = _svd_two_sites(
            phi.tensors[p], phi.tensors[p + 1],
            absorb="left", dim=dim, cutoff=cutoff,
        )
        phi.tensors[p]     = left_new
        phi.tensors[p + 1] = right_new
        phi.center_left  = p
        phi.center_right = p

    return phi


def denmat_compress_mps(
    psi: "MPS",
    *,
    max_dim: int | None = None,
    cutoff: float = 0.0,
) -> "MPS":
    """Compress an MPS via the single-sweep density-matrix (optimal) algorithm.

    At each site i (right to left), the reduced density matrix is formed
    using the exact left environments of `psi` and the already-chosen
    compressed tensors on the right.  The optimal compressed site tensor is
    the top singular vectors of that density matrix.

    This single sweep is globally optimal: the left environments are computed
    from the original (uncompressed) `psi`, so the truncation at each site is
    conditioned on the exact left context.

    Parameters
    ----------
    psi     : Input MPS.  Not modified.
    max_dim : Maximum bond dimension to keep per bond.  `None` = no limit.
    cutoff  : Discard density-matrix eigenvalues below this threshold
              (normalized: eigenvalue / total weight).

    Returns
    -------
    phi : New MPS approximating `psi` with reduced bond dimensions.
          Sites 1..N-1 are right-orthonormal; center is set to site 0.
    """
    N        = len(psi)
    dim      = sys.maxsize if max_dim is None else max_dim
    ut_dtype = cytnx.Type.ComplexDouble if psi.is_complex else cytnx.Type.Double

    # ------------------------------------------------------------------
    # Build left environments  Ls[p]  for  p = 0 .. N-2
    #
    # Ls[p] contracts sites 0..p of psi with their conjugates.
    # Labels: ["dn", "up"]
    #   "dn" BD_OUT — will connect to psi[p+1]["l"] (BD_IN)
    #   "up" BD_IN  — will connect to psi[p+1].Dagger()["l"] (BD_OUT)
    # ------------------------------------------------------------------
    Ls = [None] * (N + 1)

    b_dn  = psi[0].bond("l").redirect()   # BD_IN → BD_OUT
    b_up  = psi[0].bond("l")              # BD_IN
    Ls[-1] = cytnx.UniTensor([b_dn, b_up], labels=["dn", "up"], dtype=ut_dtype)
    Ls[-1].at([0, 0]).value = 1.0

    for p in range(N - 1):
        L      = Ls[p - 1].relabels(["dn", "up"], ["_dn", "_up"])
        A1     = psi[p].relabels(["l", "i", "r"], ["_dn", "_i", "dn"])
        A2     = psi[p].Dagger().relabels(["l", "i", "r"], ["_up", "_i", "up"])
        tmp    = cytnx.Contract(L, A1)
        Ls[p]  = cytnx.Contract(tmp, A2)

    # ------------------------------------------------------------------
    # Right-to-left sweep
    # ------------------------------------------------------------------

    # Right boundary R (1×1 identity-like)
    b_dn_r = psi[-1].bond("r").redirect()   # BD_OUT → BD_IN
    b_up_r = psi[-1].bond("r")              # BD_OUT
    R = cytnx.UniTensor([b_dn_r, b_up_r], labels=["dn", "up"], dtype=ut_dtype)
    R.at([0, 0]).value = 1.0

    new_tensors = [None] * N

    for i in range(N - 1, -1, -1):
        # Effective tensor  A = psi[i] * R
        # Labels after contract: ["l_k", "i_k", "_up"]
        #   "l_k" BD_IN  (left bond of psi[i])
        #   "i_k" BD_IN  (physical)
        #   "_up" BD_OUT (compressed right bond)
        E   = R.relabels(["dn", "up"], ["_dn", "_up"])
        ket = psi[i].relabels(["l", "i", "r"], ["l_k", "i_k", "_dn"])
        A   = cytnx.Contract(ket, E)

        if i > 0:
            # Density matrix  rho = A * Ls[i-1] * A†
            # Row side: ("i_k" BD_IN, "r_k" BD_OUT)
            # Col side: ("i_b" BD_BRA, "r_b" BD_BRA)
            L       = Ls[i - 1].relabels(["dn", "up"], ["_dn", "_up"])
            ket_rho = A.relabels(["l_k", "i_k", "_up"], ["_dn", "i_k", "r_k"])
            bra_rho = A.Dagger().relabels(["l_k", "i_k", "_up"], ["_up", "i_b", "r_b"])
            tmp     = cytnx.Contract(L, ket_rho)
            rho     = cytnx.Contract(tmp, bra_rho)
            rho.permute_(["i_k", "r_k", "i_b", "r_b"])
            rho.set_rowrank_(2)

            _, Ui, _ = svd_by_labels(
                rho,
                row_labels=["i_k", "r_k"],
                absorb="left",
                dim=dim,
                cutoff=cutoff,
                aux_label="_s",
            )
            Ui.relabels_(["_s", "i_b", "r_b"], ["l", "i", "r"])
            Ui.set_rowrank_(2)
            new_tensors[i] = Ui

            # Update R for site i-1 using psi[i] (ket side) and Ui (bra side)
            E2   = R.relabels(["dn", "up"], ["_dn", "_up"])
            A1   = psi[i].relabels(["l", "i", "r"], ["dn", "_i", "_dn"])
            Au   = Ui.Dagger().relabels(["l", "i", "r"], ["up", "_i", "_up"])
            tmp2 = cytnx.Contract(E2, A1)
            R    = cytnx.Contract(tmp2, Au)

        else:
            # First site: left bond of psi[0] is dim 1; no truncation needed.
            A.relabels_(["l_k", "i_k", "_up"], ["l", "i", "r"])
            A.set_rowrank_(2)
            new_tensors[0] = A

    phi = MPS(new_tensors)
    phi.center_left  = 0
    phi.center_right = 0
    return phi
