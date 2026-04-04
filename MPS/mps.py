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

import cytnx

from unitensor.core import assert_bond_match, scalar_from_uniTensor, svd_by_labels
from unitensor.utils import any_complex_tensors, to_numpy_array

MPS_SITE_LABELS = frozenset({"l", "i", "r"})


def _check_labels(tensor: "cytnx.UniTensor", site: int) -> None:
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
            _check_labels(tensor, i)
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
        """Replace one site; rollback if validation fails.

        After a successful update, all registered Observer callbacks are fired
        with the updated site index.  Dead callbacks (whose owner was garbage
        collected) are removed automatically.
        """
        if not isinstance(tensor, cytnx.UniTensor):
            raise TypeError(
                f"Site {site} must be cytnx.UniTensor; got {type(tensor).__name__}."
            )
        old_tensor = self.tensors[site]
        old_left = self.center_left
        old_right = self.center_right
        self.tensors[site] = tensor
        self.center_left = min(self.center_left, site)
        self.center_right = max(self.center_right, site)
        try:
            self.check_mps_tensor(site)
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

    def set_sites(self, updates: dict[int, "cytnx.UniTensor"]) -> None:
        """Replace multiple site tensors atomically.

        All tensors are written first, then validated and notified once.
        This avoids the intermediate bond-mismatch that occurs when
        updating adjacent sites one at a time via `__setitem__`.

        Parameters
        ----------
        updates : dict[int, UniTensor]
            Mapping from site index to new tensor.
        """
        sites = sorted(updates.keys())
        for site in sites:
            tensor = updates[site]
            if not isinstance(tensor, cytnx.UniTensor):
                raise TypeError(
                    f"Site {site} must be cytnx.UniTensor; "
                    f"got {type(tensor).__name__}."
                )
            self.tensors[site] = tensor

        for site in sites:
            self.center_left = min(self.center_left, site)
            self.center_right = max(self.center_right, site)

        for site in sites:
            self.check_mps_tensor(site)

        self._notify_sites(*sites)

    def register_callback(self, obj, method_name: str = "delete") -> None:
        """Register *obj.method_name* as an Observer callback.

        The callback is stored as a weak reference so that *obj* can be garbage
        collected when it goes out of scope even if it is still registered here.
        Each call to `__setitem__` fires `obj.method_name(site)` for every
        live registered observer.

        Args:
            obj: Observer object that has a method named *method_name*.
            method_name: Name of the method to call on *obj* (default `"delete"`).

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
            _check_labels(tensor, i)

    def check_mps_tensor(self, site: int, tensor: "cytnx.UniTensor" | None = None) -> None:
        """Validate a tensor as an MPS site tensor at the given position.

        Checks:
        1. Rank-3 with labels exactly `{l, i, r}`.
        2. Bond directions: `l` = BD_IN, `i` = BD_IN, `r` = BD_OUT
           (only for QN/block-form tensors; dense tensors skip this).
        3. Neighbour bond match: `tensor.bond("l")` must match
           `self[site-1].bond("r")` (if site > 0), and `tensor.bond("r")`
           must match `self[site+1].bond("l")` (if site < N-1).

        If `tensor` is None, checks `self.tensors[site]`.
        """
        if tensor is None:
            tensor = self.tensors[site]

        _check_labels(tensor, site)

        # Bond direction check (QN tensors only).
        if tensor.is_blockform():
            if tensor.bond("l").type() != cytnx.bondType.BD_KET:
                raise ValueError(
                    f"Site {site} bond \"l\" must be BD_IN; "
                    f"got {tensor.bond('l').type()}."
                )
            if tensor.bond("i").type() != cytnx.bondType.BD_KET:
                raise ValueError(
                    f"Site {site} bond \"i\" must be BD_IN; "
                    f"got {tensor.bond('i').type()}."
                )
            if tensor.bond("r").type() != cytnx.bondType.BD_BRA:
                raise ValueError(
                    f"Site {site} bond \"r\" must be BD_OUT; "
                    f"got {tensor.bond('r').type()}."
                )

        # Neighbour bond match.
        if site > 0:
            try:
                assert_bond_match(
                    self.tensors[site - 1].bond("r"), tensor.bond("l")
                )
            except ValueError:
                raise ValueError(
                    f"Bond mismatch between sites {site - 1} and {site}."
                )
        if site < len(self) - 1:
            try:
                assert_bond_match(
                    tensor.bond("r"), self.tensors[site + 1].bond("l")
                )
            except ValueError:
                raise ValueError(
                    f"Bond mismatch between sites {site} and {site + 1}."
                )

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

    @property
    def is_complex(self) -> bool:
        """Whether any site tensor uses a complex dtype."""
        return any_complex_tensors(self.tensors)

    @property
    def total_qn(self) -> list[int]:
        """Total quantum number of the MPS.

        Returns the QN of the last site's right virtual bond, which equals the
        sum of all physical QNs (Convention B: QN accumulates left-to-right).
        Returns [] for dense (no-symmetry) MPS.
        """
        b = self.tensors[-1].bond("r")
        if b.Nsym() == 0:
            return []
        return list(b.qnums()[0])

    def copy(self) -> "MPS":
        """Deep copy via `UniTensor.clone()`."""
        cloned = [tensor.clone() for tensor in self.tensors]
        copied = MPS(cloned)
        copied.center_left = self.center_left
        copied.center_right = self.center_right
        return copied

    def norm(self) -> float:
        """`sqrt(real(<psi|psi>))`."""
        from .mps_operations import inner
        val = inner(self, self)
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
        self[self.center] = self.tensors[self.center] * (1.0 / nrm)
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
            self._shift_center_right(self.center_left)

        while self.center_left > site:
            self._shift_center_left(self.center_left)
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
            self._shift_center_left(site)
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

    def _shift_center_right(self, site: int) -> None:
        """One SVD step that moves the orthogonality center from `site` to `site + 1`.

        SVD with row=["l","i"] (all BD_IN), absorb singular values right.
        """
        u, vt, _ = svd_by_labels(
            self.tensors[site],
            row_labels=["l", "i"], col_labels=["r"],
            absorb="right", aux_label="_s",
        )

        # u → new A[site]: (l, i, _s) → (l, i, r)
        u.relabels_(["l", "i", "_s"], ["l", "i", "r"])

        # vt (with s absorbed) → (_s, r), contract into A[site+1]
        vt.relabel_("r", "l")
        right_new = cytnx.Contract(vt, self.tensors[site + 1])
        right_new.relabel_("_s", "l")

        self.set_sites({site: u, site + 1: right_new})
        self.center_left = site + 1
        self.center_right = site + 1

    def _shift_center_left(self, site: int) -> None:
        """One SVD step that moves the orthogonality center from `site` to `site - 1`.

        SVD with row=["l"] (all BD_IN), absorb singular values left.
        """
        u, vt, _ = svd_by_labels(
            self.tensors[site],
            row_labels=["l"], col_labels=["i", "r"],
            absorb="left", aux_label="_s",
        )

        # vt → new A[site]: (_s, i, r) → (l, i, r)
        vt.relabels_(["_s", "i", "r"], ["l", "i", "r"])

        # u (with s absorbed) → (l, _s), contract into A[site-1]
        u.permute_(["_s", "l"])
        u.set_labels(["_sr", "r"])
        left_new = cytnx.Contract(self.tensors[site - 1], u)
        left_new.relabel_("_sr", "r")

        self.set_sites({site - 1: left_new, site: vt})
        self.center_left = site - 1
        self.center_right = site - 1

    def _validate_bonds(self) -> None:
        """Full MPS validation: labels, directions, boundaries, neighbours, center."""
        for site in range(len(self)):
            self.check_mps_tensor(site)

        if self.tensors[0].bond("l").dim() != 1:
            raise ValueError("First tensor left bond must be 1.")
        if self.tensors[-1].bond("r").dim() != 1:
            raise ValueError("Last tensor right bond must be 1.")

        if not 0 <= self.center_left <= self.center_right < len(self):
            raise ValueError(
                "center window must satisfy "
                f"0 <= center_left <= center_right < {len(self)}; "
                f"got ({self.center_left}, {self.center_right})."
            )

    def _check_compatible(self, other: "MPS") -> None:
        """Same length and physical bond structure at every site."""
        if len(self) != len(other):
            raise ValueError("Both MPS objects must have the same number of sites.")
        for site, (tensor_self, tensor_other) in enumerate(zip(self.tensors, other.tensors)):
            try:
                assert_bond_match(tensor_self.bond("i"), tensor_other.bond("i"))
            except ValueError:
                raise ValueError(f"Physical bond mismatch at site {site}.")

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
        phi : UniTensor with labels `["l", "i0", ..., "i{n-1}", "r"]`.
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
        cutoff  : float — discard Schmidt components whose normalized rho
                  eigenvalue is below this threshold.
        absorb  : `"right"` → sweep →, centre moves right;
                  `"left"`  → sweep ←, centre moves left.

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
        """1-site update via SVD.

        absorb='right':                absorb='left':

          φ                              φ
          │                             │
          ├─ A[p] ─── (into A[p+1])    (into A[p-1]) ─── A[p]
          l   i    r                   l               i    r
        """
        i0 = MPS._phi_label(0)
        discarded = 0.0

        if absorb == "right":
            # Left-canonicalise site p; absorb residual into p+1.
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

            A.relabels_(["l", i0, "_s"], ["l", "i", "r"])
            self.set_sites({p: A, p + 1: next_new})
            self.center_left  = p + 1
            self.center_right = p + 1

        else:  # absorb == "left"
            # Right-canonicalise site p; absorb residual into p-1.
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

            A.permute_(["_s", i0, "r"])
            A.set_labels(["l", "i", "r"])
            A.set_rowrank_(2)
            self.set_sites({p - 1: prev_new, p: A})
            self.center_left  = p - 1
            self.center_right = p - 1

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

        self.set_sites({p: A0, p + 1: A1})
        center = p + 1 if absorb == "right" else p
        self.center_left  = center
        self.center_right = center
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
