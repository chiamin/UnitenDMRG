"""Open-chain MPS backed by Cytnx UniTensor.

UniTensor label convention (every site tensor MUST follow this):

- `"l"`: left virtual bond (connects to site `site - 1`). Open chain: site 0 has
  `dim(l) == 1`.
- `"i"`: physical index at this site.
- `"r"`: right virtual bond (connects to site `site + 1`). Open chain: last site has
  `dim(r) == 1`.

Each site is rank 3. Internal order of axes may vary; the label set must be exactly
`{"l", "i", "r"}`. Rowrank is not part of the public contract (this module still sets
`rowrank` to 2 when constructing tensors for Cytnx SVD/contract routines).

`center` is stored for gauge bookkeeping; how it is maintained may be refined later.
"""

from __future__ import annotations

import math
import sys
import weakref
from typing import Iterable, Iterator

import numpy as np

try:
    import cytnx
except ImportError as exc:
    raise ImportError(
        "cytnx is required for mps.py. Install/import cytnx first."
    ) from exc

from .uniTensor_core import assert_bond_match, qr_by_labels, scalar_from_uniTensor, svd_by_labels
from .uniTensor_utils import to_numpy_array

MPS_SITE_LABELS = frozenset({"l", "i", "r"})


def assert_mps_site_uniTensor_labels(tensor: "cytnx.UniTensor", site: int) -> None:
    """Raise ValueError if `tensor` is not rank-3 or labels are not exactly l, i, r."""
    if len(tensor.shape()) != 3:
        raise ValueError(
            f"Site {site} must be rank-3 (l, i, r); got rank {len(tensor.shape())}."
        )
    if set(tensor.labels()) != MPS_SITE_LABELS:
        raise ValueError(
            f"Site {site} labels must be exactly l, i, r; got {tensor.labels()}."
        )


class MPS:
    """Open-boundary MPS with UniTensor site tensors (labels `l`, `i`, `r`)."""

    def __init__(self, tensors: Iterable):
        """Load site tensors left to right; validate labels and open-chain bonds."""
        self.tensors = list(tensors)
        if not self.tensors:
            raise ValueError("An MPS must contain at least one site tensor.")
        for i, tensor in enumerate(self.tensors):
            if not isinstance(tensor, cytnx.UniTensor):
                raise TypeError(
                    f"Site {i} must be cytnx.UniTensor; got {type(tensor).__name__}."
                )
            assert_mps_site_uniTensor_labels(tensor, i)
        # Canonical window [center_left, center_right]:
        # - sites < center_left are expected left-orthonormal
        # - sites > center_right are expected right-orthonormal
        self.center_left = 0
        self.center_right = len(self.tensors) - 1
        self._validate_bonds()
        # Observer callbacks: list of (weakref.ref(obj), method_name).
        # Populated via register_callback(); fired on every successful __setitem__.
        # Weak references ensure that registered observers (e.g. LR environment
        # objects) do not prevent garbage collection once they go out of scope.
        self._callbacks: list = []

    def __len__(self) -> int:
        """Number of sites."""
        return len(self.tensors)

    def __iter__(self) -> Iterator["cytnx.UniTensor"]:
        """Site tensors, left to right."""
        return iter(self.tensors)

    def __getitem__(self, site: int) -> "cytnx.UniTensor":
        """Tensor at `site`."""
        return self.tensors[site]

    def __setitem__(self, site: int, tensor) -> None:
        """Replace one site; rollback if bonds no longer match.

        After a successful update, all registered Observer callbacks are fired
        with the updated site index.  Dead callbacks (whose owner was garbage
        collected) are removed automatically.
        """
        if not isinstance(tensor, cytnx.UniTensor):
            raise TypeError(
                f"Site {site} must be cytnx.UniTensor; got {type(tensor).__name__}."
            )
        assert_mps_site_uniTensor_labels(tensor, site)
        old_tensor = self.tensors[site]
        old_left = self.center_left
        old_right = self.center_right
        self.tensors[site] = tensor
        self.center_left = min(self.center_left, site)
        self.center_right = max(self.center_right, site)
        try:
            self._validate_bonds()
        except Exception:
            self.tensors[site] = old_tensor
            self.center_left = old_left
            self.center_right = old_right
            raise
        # Tensor update succeeded: notify observers and prune dead weak-refs.
        live = []
        for obj_ref, method_name in self._callbacks:
            obj = obj_ref()
            if obj is not None:
                getattr(obj, method_name)(site)
                live.append((obj_ref, method_name))
        self._callbacks = live

    def register_callback(self, obj, method_name: str = "delete") -> None:
        """Register *obj.method_name* as an Observer callback.

        The callback is stored as a weak reference so that *obj* can be garbage
        collected when it goes out of scope even if it is still registered here.
        Each call to ``__setitem__`` fires ``obj.method_name(site)`` for every
        live registered observer.

        Args:
            obj: Observer object that has a method named *method_name*.
            method_name: Name of the method to call on *obj* (default ``"delete"``).

        Raises:
            AssertionError: If *obj* does not have a method called *method_name*.
        """
        assert hasattr(obj, method_name), (
            f"{type(obj).__name__} has no method '{method_name}'"
        )
        self._callbacks.append((weakref.ref(obj), method_name))

    @property
    def center(self) -> int | None:
        """Single-site center index, or None when canonical window spans multiple sites."""
        if self.center_left == self.center_right:
            return self.center_left
        return None

    def __repr__(self) -> str:
        """Short summary."""
        center_text = (
            str(self.center)
            if self.center is not None
            else f"[{self.center_left}, {self.center_right}]"
        )
        return (
            f"MPS(num_sites={len(self)}, phys_dim={self.phys_dims[0]}, "
            f"max_bond_dim={self.max_dim}, center={center_text})"
        )

    def check_site_labels(self) -> None:
        """Assert every site satisfies rank-3 and labels `l`, `i`, `r`."""
        for i, tensor in enumerate(self.tensors):
            assert_mps_site_uniTensor_labels(tensor, i)

    @property
    def phys_dims(self) -> list[int]:
        """Local physical dimension at each site."""
        return [tensor.bond("i").dim() for tensor in self.tensors]

    @property
    def bond_dims(self) -> list[int]:
        """Virtual bond dimensions `[D0, D1, ...]` (open ends: `D0` and last are 1)."""
        dims = [self.tensors[0].bond("l").dim()]
        dims.extend(tensor.bond("r").dim() for tensor in self.tensors)
        return dims

    @property
    def max_dim(self) -> int:
        """Largest bond dimension."""
        return max(self.bond_dims)

    def copy(self) -> "MPS":
        """Deep copy via `UniTensor.clone()`."""
        cloned = [tensor.clone() for tensor in self.tensors]
        copied = MPS(cloned)
        copied.center_left = self.center_left
        copied.center_right = self.center_right
        return copied

    def inner(self, other: "MPS") -> float | complex:
        """Overlap `<self|other>` via left environment contraction (`other` daggered per site)."""
        self._check_compatible(other)
        l1 = self.tensors[0].bond("l").redirect()
        l2 = other.tensors[0].bond("l")
        env = cytnx.UniTensor([l1, l2], labels=["dn", "up"])
        env.at([0, 0]).value = 1.0

        for tensor_self, tensor_other in zip(self.tensors, other.tensors):
            a1 = env.relabels(["up", "dn"], ["_up", "_dn"])
            a2 = tensor_self.relabels(["l", "i", "r"], ["_dn", "i", "dn"])
            a3 = tensor_other.Dagger().relabels(["l", "i", "r"], ["_up", "i", "up"])
            tmp = cytnx.Contract(a1, a2)
            env = cytnx.Contract(tmp, a3)
        return scalar_from_uniTensor(env)

    def norm(self) -> float:
        """`sqrt(real(<psi|psi>))`."""
        val = self.inner(self)
        return math.sqrt(float(val.real))

    def normalize(self) -> "MPS":
        """Scale the orthogonality center so norm is 1.

        Raises:
            ValueError: If the MPS has no single center (call `orthogonalize()` first).
            ValueError: If the MPS norm is zero.
        """
        if self.center is None:
            raise ValueError(
                "MPS has no single orthogonality center; call orthogonalize() first."
            )
        nrm = self.norm()
        if nrm < 1e-14:
            raise ValueError("Cannot normalize a zero MPS.")
        self.tensors[self.center] = self.tensors[self.center] * (1.0 / nrm)
        return self

    def move_center(self, site: int) -> "MPS":
        """Move orthogonality center to `site`.

        If `center` is `None`, first canonicalize the active window with
        `orthogonalize()` so the center is initialized at `center_left`, then
        move to the requested site.
        """
        if not 0 <= site < len(self):
            raise IndexError(f"Center site {site} is outside [0, {len(self) - 1}].")

        if self.center is None:
            self.orthogonalize()

        while self.center_left < site:
            self._shift_center_right_qr(self.center_left)

        while self.center_left > site:
            self._shift_center_left_qr(self.center_left)
        return self

    def orthogonalize(self, center: int | None = None) -> "MPS":
        """Canonicalize the current center window and optionally set center site.

        The tensors outside `[center_left, center_right]` are assumed to already
        satisfy left/right orthonormal conditions, so only the active window is
        swept. After this routine, the window collapses to a single center at
        `center_left`. If `center` is provided, the orthogonality center is
        then moved to that site.
        """
        if center is not None and not 0 <= center < len(self):
            raise IndexError(f"Center site {center} is outside [0, {len(self) - 1}].")

        for site in range(self.center_right, self.center_left, -1):
            self._shift_center_left_qr(site)
        self._validate_bonds()

        if center is not None and self.center_left != center:
            self.move_center(center)
        return self

    def check_left_right_orthonormal(self, *, atol: float = 1e-10) -> None:
        """Validate orthonormal constraints outside the current center window.

        Sites strictly left of `center_left` must be left-orthonormal.
        Sites strictly right of `center_right` must be right-orthonormal.
        Raises `ValueError` on the first violation.
        """
        for site, tensor in enumerate(self.tensors):
            arr = to_numpy_array(tensor)
            left_dim, phys_dim, right_dim = arr.shape

            if site < self.center_left:
                left_mat = arr.reshape(left_dim * phys_dim, right_dim)
                left_gram = left_mat.conj().T @ left_mat
                if not np.allclose(left_gram, np.eye(right_dim, dtype=left_gram.dtype), atol=atol):
                    raise ValueError(
                        f"Site {site} is not left-orthonormal "
                        f"(required for sites < center_left={self.center_left})."
                    )

            if site > self.center_right:
                right_mat = arr.reshape(left_dim, phys_dim * right_dim)
                right_gram = right_mat @ right_mat.conj().T
                if not np.allclose(right_gram, np.eye(left_dim, dtype=right_gram.dtype), atol=atol):
                    raise ValueError(
                        f"Site {site} is not right-orthonormal "
                        f"(required for sites > center_right={self.center_right})."
                    )

    def _shift_center_right_qr(self, site: int) -> None:
        """One QR step that moves the orthogonality center from `site` to `site + 1`."""
        q, r = qr_by_labels(self.tensors[site], row_labels=["l", "i"], aux_label="nr")
        q.relabel_("nr", "r")
        r.relabel_("r", "l")
        right_new = cytnx.Contract(r, self.tensors[site + 1])
        right_new.relabel_("nr", "l")

        self.tensors[site] = q
        self.tensors[site + 1] = right_new
        self.center_left = site + 1
        self.center_right = site + 1

    def _shift_center_left_qr(self, site: int) -> None:
        """One QR step that moves the orthogonality center from `site` to `site - 1`."""
        q, r = qr_by_labels(self.tensors[site], row_labels=["i", "r"], aux_label="nr")
        q.permute_(["nr", "i", "r"])
        q.set_labels(["l", "i", "r"])
        q.set_rowrank_(2)

        r.permute_(["l", "nr"])
        r.set_labels(["r", "nr"])
        left_new = cytnx.Contract(self.tensors[site - 1], r)
        left_new.relabel_("nr", "r")

        self.tensors[site - 1] = left_new
        self.tensors[site] = q
        self.center_left = site - 1
        self.center_right = site - 1

    def _validate_bonds(self) -> None:
        """Labels, open boundaries, neighbor bond match, and center window range."""
        self.check_site_labels()
        if self.tensors[0].bond("l").dim() != 1:
            raise ValueError("First tensor left bond must be 1.")
        if self.tensors[-1].bond("r").dim() != 1:
            raise ValueError("Last tensor right bond must be 1.")

        for site in range(len(self.tensors) - 1):
            if self.tensors[site].bond("r").dim() != self.tensors[site + 1].bond("l").dim():
                raise ValueError(f"Bond mismatch between sites {site} and {site + 1}.")

        if not 0 <= self.center_left <= self.center_right < len(self):
            raise ValueError(
                "center window must satisfy "
                f"0 <= center_left <= center_right < {len(self)}; "
                f"got ({self.center_left}, {self.center_right})."
            )

    def compress_bond(
        self,
        bond: int,
        *,
        max_dim: int | None = None,
        cutoff: float = 0.0,
        absorb: str = "right",
    ) -> tuple[int, float]:
        """Compress one bond in place via SVD truncation.

        Args:
            bond: Bond index to compress (0 to num_sites - 2).
            max_dim: Maximum bond dimension to keep. None means no limit.
            cutoff: Discard singular values below this threshold.
            absorb: Which side to absorb the singular values into.

        Returns:
            Tuple of `(kept_dim, discarded_weight)`.
        """
        if not 0 <= bond < len(self) - 1:
            raise IndexError(f"Bond {bond} is outside [0, {len(self) - 2}].")
        if absorb not in {"left", "right"}:
            raise ValueError("absorb must be 'left' or 'right'.")

        left_new, right_new, kept_dim, discarded = compress_bond_tensors(
            self.tensors[bond],
            self.tensors[bond + 1],
            absorb=absorb,
            max_dim=max_dim,
            cutoff=cutoff,
        )
        self.tensors[bond] = left_new
        self.tensors[bond + 1] = right_new
        center_site = bond if absorb == "left" else bond + 1
        self.center_left = center_site
        self.center_right = center_site
        self._validate_bonds()
        return kept_dim, discarded

    def _check_compatible(self, other: "MPS") -> None:
        """Same length and `phys_dims`."""
        if len(self) != len(other):
            raise ValueError("Both MPS objects must have the same number of sites.")
        for site, (tensor_self, tensor_other) in enumerate(zip(self.tensors, other.tensors)):
            try:
                assert_bond_match(tensor_self.bond("i"), tensor_other.bond("i"))
            except ValueError as exc:
                raise ValueError(f"Physical bond mismatch at site {site}.") from exc

    # ------------------------------------------------------------------
    # φ label convention  (single source of truth)
    # ------------------------------------------------------------------

    @staticmethod
    def _phi_label(k: int) -> str:
        """Canonical physical label for site k in a merged φ tensor.

        Used by make_phi, update_sites, and EffOperator._apply_operator.
        Changing this one line updates the convention everywhere.
        """
        return f"i{k}"

    # ------------------------------------------------------------------
    # make_phi / update_sites
    # ------------------------------------------------------------------

    def make_phi(self, p: int, n: int = 1) -> "cytnx.UniTensor":
        """Merge n site tensors starting at p into a single φ.

        Diagram (n=2):

            A[p] ─── A[p+1]
             │            │
             l   i0  i1   r

        Parameters
        ----------
        p : int — start site.
        n : int — number of sites to merge (1 or 2).

        Returns
        -------
        phi : UniTensor with labels ``["l", "i0", ..., "i{n-1}", "r"]``.
        """
        if not 0 <= p < len(self):
            raise IndexError(f"Site {p} out of range [0, {len(self)-1}].")
        if n < 1:
            raise ValueError("n must be >= 1.")
        if p + n > len(self):
            raise IndexError(f"Sites {p}..{p+n-1} exceed MPS length {len(self)}.")

        tensors = []
        for k in range(n):
            A = self.tensors[p + k]
            i_label = MPS._phi_label(k)
            if k == 0:
                l_label = "l"
                r_label = "r" if n == 1 else "_bond0"
            elif k == n - 1:
                l_label = f"_bond{k-1}"
                r_label = "r"
            else:
                l_label = f"_bond{k-1}"
                r_label = f"_bond{k}"
            tensors.append(A.relabels(["l", "i", "r"], [l_label, i_label, r_label]))

        phi = tensors[0]
        for t in tensors[1:]:
            phi = cytnx.Contract(phi, t)
        return phi

    def update_sites(
        self,
        p: int,
        phi: "cytnx.UniTensor",
        max_dim: int | None = None,
        cutoff: float = 0.0,
        absorb: str = "right",
    ) -> float:
        """Decompose φ back into site tensors and update psi in-place.

        Inverse of make_phi.  Updates psi[p] (and psi[p+1] for 2-site),
        sets the orthogonality center, and fires observer callbacks.

        1-site: QR when max_dim=None and cutoff=0 (fast, no truncation);
                SVD otherwise (truncation possible).
        2-site: always SVD.

        Parameters
        ----------
        p       : int — start site (same p passed to make_phi).
        phi     : UniTensor — merged tensor from make_phi.
        max_dim : int | None — max bond dimension (None = no limit).
        cutoff  : float — discard singular values below this.
        absorb  : ``"right"`` → sweep →, centre moves right;
                  ``"left"``  → sweep ←, centre moves left.

        Returns
        -------
        discarded : float — truncation error.
        """
        if absorb not in ("left", "right"):
            raise ValueError("absorb must be 'left' or 'right'.")

        raw = [lab for lab in phi.labels() if lab not in ("l", "r")]
        n = len(raw)
        expected = set(MPS._phi_label(k) for k in range(n))
        if set(raw) != expected:
            raise ValueError(
                f"phi physical labels must be i0,i1,...; got {raw}"
            )

        if n == 1:
            return self._update_1site(p, phi, max_dim, cutoff, absorb)
        elif n == 2:
            return self._update_2site(p, phi, max_dim, cutoff, absorb)
        else:
            raise NotImplementedError(
                f"update_sites supports n=1 or 2; got n={n}."
            )

    def _update_1site(self, p, phi, max_dim, cutoff, absorb):
        """1-site update: QR (no truncation) or SVD (with truncation).

        absorb='right':                absorb='left':

          φ                              φ
          │                             │
          ├─ A[p] ─── (into A[p+1])    (into A[p-1]) ─── A[p]
          l   i    r                   l               i    r
        """
        i0 = MPS._phi_label(0)
        use_svd = (max_dim is not None) or (cutoff != 0.0)
        discarded = 0.0

        if absorb == "right":
            # Left-canonicalise site p; absorb residual into p+1.
            if use_svd:
                A, Vt, discarded = svd_by_labels(
                    phi,
                    row_labels=["l", i0], col_labels=["r"],
                    absorb="right",
                    dim=max_dim if max_dim is not None else sys.maxsize,
                    cutoff=cutoff, aux_label="_s",
                )
                # Vt: (_s, r) → absorb into psi[p+1]
                Vt.relabel_("r", "l")
                next_new = cytnx.Contract(Vt, self.tensors[p + 1])
                next_new.relabel_("_s", "l")
            else:
                A, Vt = qr_by_labels(phi, row_labels=["l", i0], aux_label="_s")
                # A: (l, i0, _s),  Vt: (_s, r) → absorb into psi[p+1]
                Vt.relabel_("r", "l")
                next_new = cytnx.Contract(Vt, self.tensors[p + 1])
                next_new.relabel_("_s", "l")

            A.relabels_(["l", i0, "_s"], ["l", "i", "r"])
            self.tensors[p]     = A
            self.tensors[p + 1] = next_new
            self.center_left  = p + 1
            self.center_right = p + 1
            self._validate_bonds()
            self._notify_sites(p, p + 1)

        else:  # absorb == "left"
            # Right-canonicalise site p; absorb residual into p-1.
            if use_svd:
                US, A, discarded = svd_by_labels(
                    phi,
                    row_labels=["l"], col_labels=[i0, "r"],
                    absorb="left",
                    dim=max_dim if max_dim is not None else sys.maxsize,
                    cutoff=cutoff, aux_label="_s",
                )
                # A (=Vt): (_s, i0, r) → set as psi[p]
                # US: (l, _s) → absorb into psi[p-1]
                US.permute_(["_s", "l"])
                US.set_labels(["_sr", "r"])
                prev_new = cytnx.Contract(self.tensors[p - 1], US)
                prev_new.relabel_("_sr", "r")
            else:
                A, US = qr_by_labels(phi, row_labels=[i0, "r"], aux_label="_s")
                # A: (i0, r, _s),  US (=R): (_s, l) → absorb into psi[p-1]
                US.permute_(["l", "_s"])
                US.set_labels(["r", "_sr"])
                prev_new = cytnx.Contract(self.tensors[p - 1], US)
                prev_new.relabel_("_sr", "r")

            A.permute_(["_s", i0, "r"])
            A.set_labels(["l", "i", "r"])
            A.set_rowrank_(2)
            self.tensors[p - 1] = prev_new
            self.tensors[p]     = A
            self.center_left  = p - 1
            self.center_right = p - 1
            self._validate_bonds()
            self._notify_sites(p - 1, p)

        return discarded

    def _update_2site(self, p, phi, max_dim, cutoff, absorb):
        """2-site update: always SVD.

        Diagram:

          │    │      │    │      ← output: l, i0_out, i1_out, r
          A[p] ─── A[p+1]
          │              │
          └───── φ ──────┘

        absorb='right': A[p] left-ortho,  centre at p+1.
        absorb='left' : A[p+1] right-ortho, centre at p.
        """
        i0  = MPS._phi_label(0)
        i1  = MPS._phi_label(1)
        dim = max_dim if max_dim is not None else sys.maxsize

        A0, A1, discarded = svd_by_labels(
            phi,
            row_labels=["l", i0], col_labels=[i1, "r"],
            absorb=absorb,
            dim=dim, cutoff=cutoff, aux_label="_s",
        )
        A0.relabels_(["l", i0, "_s"], ["l", "i", "r"])
        A1.relabels_(["_s", i1, "r"], ["l", "i", "r"])

        self.tensors[p]     = A0
        self.tensors[p + 1] = A1
        center = p + 1 if absorb == "right" else p
        self.center_left  = center
        self.center_right = center
        self._validate_bonds()
        self._notify_sites(p, p + 1)
        return discarded

    def _notify_sites(self, *sites: int) -> None:
        """Fire observer delete(site) for each site; prune dead refs."""
        live = []
        for obj_ref, method_name in self._callbacks:
            obj = obj_ref()
            if obj is not None:
                for site in sites:
                    getattr(obj, method_name)(site)
                live.append((obj_ref, method_name))
        self._callbacks = live


def svd_bond(
    left: "cytnx.UniTensor",
    right: "cytnx.UniTensor",
    *,
    absorb: str,
    dim: int,
    cutoff: float,
) -> tuple["cytnx.UniTensor", "cytnx.UniTensor", float]:
    """SVD a merged two-site MPS tensor and split back into site tensors.

    Args:
        left: Left site tensor with labels `l`, `i`, `r`.
        right: Right site tensor with labels `l`, `i`, `r`.
        absorb: Which side to absorb the singular values into (`"left"` or `"right"`).
        dim: Maximum number of singular values to keep.
        cutoff: Discard singular values below this threshold.

    Returns:
        Tuple of `(left_new, right_new, discarded_weight)`.
    """
    if absorb not in {"left", "right"}:
        raise ValueError("absorb must be 'left' or 'right'.")

    a1 = left.relabels(["i", "r"], ["i1", "_"])
    a2 = right.relabels(["i", "l"], ["i2", "_"])
    aa = cytnx.Contract(a1, a2)
    # aa has labels: [l, i1, i2, r]

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


def compress_bond_tensors(
    left: "cytnx.UniTensor",
    right: "cytnx.UniTensor",
    *,
    absorb: str,
    max_dim: int | None,
    cutoff: float,
) -> tuple["cytnx.UniTensor", "cytnx.UniTensor", int, float]:
    """Compress one MPS bond and return updated tensors and truncation info.

    Returns:
        Tuple of `(left_new, right_new, kept_dim, discarded_weight)`.
    """
    keep = sys.maxsize if max_dim is None else max_dim
    if keep <= 0:
        raise ValueError("max_dim must be positive.")
    left_new, right_new, discarded = svd_bond(
        left, right, absorb=absorb, dim=keep, cutoff=cutoff,
    )
    kept_dim = left_new.bond("r").dim()
    return left_new, right_new, kept_dim, discarded
