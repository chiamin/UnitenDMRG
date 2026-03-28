"""Core UniTensor kernels for MPS operations."""

from __future__ import annotations

import sys

import numpy as np
import cytnx


def _split_row_col_labels(
    tensor: "cytnx.UniTensor",
    *,
    row_labels: list[str] | tuple[str, ...] | None = None,
    col_labels: list[str] | tuple[str, ...] | None = None,
) -> tuple[list[str], list[str]]:
    """Normalize and validate row/column label groups."""
    labels = list(tensor.labels())
    if row_labels is None and col_labels is None:
        raise ValueError("Either row_labels or col_labels must be provided.")

    if row_labels is not None:
        row_labels = list(row_labels)
        col_labels = [label for label in labels if label not in row_labels]
    else:
        col_labels = list(col_labels)
        row_labels = [label for label in labels if label not in col_labels]

    if not row_labels or not col_labels:
        raise ValueError("Both row and column label groups must be non-empty.")
    if set(row_labels) & set(col_labels):
        raise ValueError("row_labels and col_labels must be disjoint.")
    if set(row_labels) | set(col_labels) != set(labels):
        raise ValueError("row_labels and col_labels must cover all tensor labels.")
    return row_labels, col_labels


def qr_by_labels(
    tensor: "cytnx.UniTensor",
    *,
    row_labels: list[str] | tuple[str, ...] | None = None,
    col_labels: list[str] | tuple[str, ...] | None = None,
    aux_label: str = "aux",
) -> tuple["cytnx.UniTensor", "cytnx.UniTensor"]:
    """QR-decompose a UniTensor by specifying which labels belong to rows/columns.

    Returns:
        q: labels `row_labels + [aux_label]`
        r: labels `[aux_label] + col_labels`
    """
    row_labels, col_labels = _split_row_col_labels(
        tensor, row_labels=row_labels, col_labels=col_labels
    )
    ordered_labels = row_labels + col_labels
    ordered = tensor.permute(ordered_labels)
    ordered.set_rowrank_(len(row_labels))

    q, r = cytnx.linalg.Qr(ordered)

    q.set_labels(row_labels + [aux_label])
    q.set_rowrank_(len(row_labels))

    r.set_labels([aux_label] + col_labels)
    r.set_rowrank_(1)
    return q, r


def svd_by_labels(
    tensor: "cytnx.UniTensor",
    *,
    row_labels: list[str] | tuple[str, ...] | None = None,
    col_labels: list[str] | tuple[str, ...] | None = None,
    absorb: str | None = None,
    dim: int = sys.maxsize,
    cutoff: float = 0.0,
    aux_label: str = "aux",
):
    """SVD-decompose a UniTensor by specifying row/column label groups.

    Args:
        absorb: Where to absorb the singular values.
            `"left"` or `"right"` absorbs `s` into that tensor and returns
            `(left, right, discarded_weight)`.
            `None` (default) keeps `s` separate and returns
            `(left, s, right, discarded_weight)`.
            In the `None` case, the bond between `left` and `s` is labelled
            `aux_label`, and the bond between `s` and `right` is labelled
            `aux_label + "_r"` (e.g. `"aux"` and `"aux_r"` by default).
        cutoff: Truncation threshold on normalized rho eigenvalues
            `lambda_i = |s_i|^2 / sum_j |s_j|^2`.
            Singular values with `lambda_i < cutoff` are discarded.
            At least one singular value is always kept (cytnx guarantee).
            `cutoff=0` is allowed.
    """
    if absorb not in {"left", "right", None}:
        raise ValueError("absorb must be 'left', 'right', or None.")
    if cutoff < 0.0:
        raise ValueError(f"cutoff must be >= 0; got {cutoff}.")

    row_labels, col_labels = _split_row_col_labels(
        tensor, row_labels=row_labels, col_labels=col_labels
    )
    ordered_labels = row_labels + col_labels
    ordered = tensor.permute(ordered_labels)
    ordered.set_rowrank_(len(row_labels))

    total_sq = float(ordered.Norm().item()) ** 2
    if total_sq == 0.0:
        raise ValueError(
            "All singular values are zero; cannot normalize rho eigenvalues for truncation."
        )

    # Convert rho-eigenvalue cutoff to a singular-value threshold:
    # lambda_i = |s_i|^2 / total_sq >= cutoff  <=>  |s_i| >= sqrt(cutoff * total_sq)
    # cytnx.Svd_truncate always keeps >= 1 singular value even if all fall below err.
    sv_cutoff = float(np.sqrt(cutoff * total_sq))
    s, left, right = cytnx.linalg.Svd_truncate(ordered, keepdim=dim, err=sv_cutoff)
    kept_sq = _kept_weight(s)
    # Return discarded weight in normalized-rho units.
    discarded = max(0.0, 1.0 - kept_sq / total_sq)

    # SVD convention: aux bond of u is its last index; aux bond of vt is its first index.
    aux_in_left = left.labels()[-1]
    aux_in_right = right.labels()[0]

    # s always has real dtype; left/right may be complex (e.g. after Lanczos).
    # Cast s to match so that Contract does not fail with "real += complex".
    if s.dtype() != left.dtype():
        s = s.astype(left.dtype())

    if absorb == "left":
        left = cytnx.Contract(left, s)
    elif absorb == "right":
        right = cytnx.Contract(s, right)
    else:  # absorb is None: relabel s so it can be contracted with left and right
        s.relabel_(aux_in_left, aux_label)
        s.relabel_(aux_in_right, aux_label + "_r")

    left.set_labels(row_labels + [aux_label])
    left.set_rowrank_(len(row_labels))

    aux_right_label = aux_label + "_r" if absorb is None else aux_label
    right.set_labels([aux_right_label] + col_labels)
    right.set_rowrank_(1)

    if absorb is None:
        return left, s, right, discarded
    return left, right, discarded


def scalar_from_uniTensor(u: "cytnx.UniTensor") -> float | complex:
    """Read one scalar from a 1-element UniTensor; float if real dtype, else complex."""
    block = u.get_block()
    raw = block.item() if hasattr(block, "item") else block[0].item()
    if isinstance(raw, complex):
        return complex(raw)
    return float(raw)


def assert_bond_match(b1: "cytnx.Bond", b2: "cytnx.Bond") -> None:
    """Validate two cytnx bonds have identical content (dim, symmetry, qnums, degeneracies).

    Direction (BD_IN / BD_OUT) is intentionally not checked so the function
    can be used for both virtual-bond and physical-bond comparisons.
    """
    if b1.dim() != b2.dim():
        raise ValueError("Bond dimensions mismatch.")
    if b1.Nsym() != b2.Nsym():
        raise ValueError("Bond symmetry count mismatch.")
    if b1.qnums() != b2.qnums():
        raise ValueError("Bond qnums mismatch.")
    if b1.getDegeneracies() != b2.getDegeneracies():
        raise ValueError("Bond degeneracies mismatch.")


def _make_expand(
    bond1: "cytnx.Bond",
    bond2: "cytnx.Bond",
    label1: str,
    label2: str,
    re_label: str,
) -> tuple["cytnx.UniTensor", "cytnx.UniTensor"]:
    """Build embedding matrices for the direct sum of bond1 and bond2.

    bond_sum = bond1's sectors || bond2's sectors.
    Returns (exp1, exp2):
      exp1: embeds bond1 into bond_sum — identity block at (k, k)
      exp2: embeds bond2 into bond_sum — identity block at (k, k + nsec1)
    exp1 has labels [label1, re_label]; exp2 has labels [label2, re_label].
    """
    qnums    = list(bond1.qnums()) + list(bond2.qnums())
    degs     = list(bond1.getDegeneracies()) + list(bond2.getDegeneracies())
    bond_sum = cytnx.Bond(bond1.type(), qnums, degs, list(bond1.syms()))
    nsec1    = len(bond1.qnums())

    exp1 = cytnx.UniTensor([bond1.redirect(), bond_sum], labels=[label1, re_label], rowrank=1)
    for k, d in enumerate(bond1.getDegeneracies()):
        exp1.put_block_(cytnx.eye(d), [label1, re_label], [k, k])

    exp2 = cytnx.UniTensor([bond2.redirect(), bond_sum], labels=[label2, re_label], rowrank=1)
    for k, d in enumerate(bond2.getDegeneracies()):
        exp2.put_block_(cytnx.eye(d), [label2, re_label], [k, k + nsec1])

    return exp1, exp2


def direct_sum(
    A: "cytnx.UniTensor",
    B: "cytnx.UniTensor",
    sum_labels_A: list[str],
    sum_labels_B: list[str],
    re_labels: list[str],
) -> "cytnx.UniTensor":
    """Compute C = A ⊕ B, direct-summing along bond pairs (sum_labels_A[k], sum_labels_B[k]).

    The enlarged bond for each pair is named re_labels[k].
    All non-summed labels must be identical between A and B with matching bonds.
    re_labels must not clash with any non-summed label.
    """
    if not len(sum_labels_A) == len(sum_labels_B) == len(re_labels):
        raise ValueError(
            f"sum_labels_A, sum_labels_B, re_labels must have equal length; "
            f"got {len(sum_labels_A)}, {len(sum_labels_B)}, {len(re_labels)}."
        )

    non_sum_A = [l for l in A.labels() if l not in sum_labels_A]
    non_sum_B = [l for l in B.labels() if l not in sum_labels_B]
    if set(non_sum_A) != set(non_sum_B):
        raise ValueError(
            f"Non-summed labels mismatch: A has {non_sum_A}, B has {non_sum_B}."
        )
    for lab in non_sum_A:
        assert_bond_match(A.bond(lab), B.bond(lab))
    for re in re_labels:
        if re in non_sum_A:
            raise ValueError(
                f"re_label '{re}' clashes with non-summed label '{re}'."
            )

    TA, TB = A, B
    for lA, lB, re in zip(sum_labels_A, sum_labels_B, re_labels):
        tmp = f"__ds_{re}__"   # avoids clash when re == lA or lB
        exp1, exp2 = _make_expand(TA.bond(lA), TB.bond(lB), lA, lB, tmp)
        TA = cytnx.Contract(exp1, TA)
        TB = cytnx.Contract(exp2, TB)
        TA.relabel_(tmp, re)
        TB.relabel_(tmp, re)

    labels = list(TA.labels())
    C = TA + TB
    C.set_labels(labels)
    return C


def _bond_sector_at(bond: "cytnx.Bond", idx: int) -> tuple[int, int]:
    """Return (sector_index, offset_within_sector) for flat basis index idx.

    Iterates through sectors in order, accumulating dims, until the sector
    containing idx is found.

    Raises IndexError if idx >= bond.dim().
    """
    offset = idx
    for sector_idx, deg in enumerate(bond.getDegeneracies()):
        if offset < deg:
            return sector_idx, offset
        offset -= deg
    raise IndexError(f"Index {idx} out of range for bond of dim {bond.dim()}.")


def bond_qnums_at(bond: "cytnx.Bond", idx: int) -> list[int]:
    """Return the QN list for flat basis index idx in a bond.

    Returns [] for dense (no-QN) bonds.
    The returned list has one integer per symmetry in the bond.
    """
    if bond.Nsym() == 0:
        return []
    sector_idx, _ = _bond_sector_at(bond, idx)
    return list(bond.qnums()[sector_idx])


def derive_delta_qn(matrix: "np.ndarray", bond: "cytnx.Bond") -> int:
    """Derive the QN charge of a pure-charge operator from its matrix and physical bond.

    A pure-charge operator has the same QN(ip) - QN(i) for every nonzero element.
    Returns 0 for dense (no-QN) bonds.

    Raises:
        ValueError: If the operator mixes different QN charges.
    """
    if bond.Nsym() == 0:
        return 0
    delta = None
    for ip in range(matrix.shape[0]):
        for i in range(matrix.shape[1]):
            if abs(matrix[ip, i]) > 1e-14:
                qn_ip = bond_qnums_at(bond, ip)
                qn_i  = bond_qnums_at(bond, i)
                # Only support single-symmetry bonds for now
                d = qn_ip[0] - qn_i[0]
                if delta is None:
                    delta = d
                elif delta != d:
                    raise ValueError(
                        "Operator is not pure-charge: found mixed QN charges "
                        f"{delta} and {d}. Decompose into pure-charge terms first."
                    )
    return delta if delta is not None else 0


def _kept_weight(s_ut: "cytnx.UniTensor") -> float:
    """Sum of |s_k|^2 over all (kept) singular values in s_ut."""
    block = s_ut.get_block()
    shape = block.shape()
    if len(shape) == 1:
        arr = block.numpy()
        return float(np.sum(np.abs(arr) ** 2))
    if len(shape) == 2:
        arr = np.diag(block.numpy())
        return float(np.sum(np.abs(arr) ** 2))
    raise ValueError(f"Unexpected singular-value block rank: {len(shape)}")


