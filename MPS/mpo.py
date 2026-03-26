"""Open-chain MPO backed by Cytnx UniTensor.

UniTensor label convention (every site tensor MUST follow this):

- `"l"`: left MPO virtual bond (connects to site `site - 1`).
- `"r"`: right MPO virtual bond (connects to site `site + 1`).
- `"i"`: incoming physical index (contracted with the MPS ket).
- `"ip"`: outgoing physical index (contracted with the MPS bra, already conjugated).

Each site is rank 4. Internal axis order may vary; the label set must be exactly
`{"l", "ip", "i", "r"}`.

Boundary tensors `L0` and `R0` have labels `["mid", "up", "dn"]` and shape
`[d_mpo, 1, 1]`, where `d_mpo` is the MPO virtual bond dimension at that end.
They are constructed automatically from the MPO's endpoint bonds following the
standard upper-triangular MPO convention:

    L0[d_mpo - 1, 0, 0] = 1.0   (selects the last MPO virtual state on the left)
    R0[0,         0, 0] = 1.0   (selects the first MPO virtual state on the right)

Custom boundary tensors can be supplied via the `L0` / `R0` keyword arguments of
`__init__`.
"""

from __future__ import annotations

import sys
import weakref
from typing import Iterable, Iterator

import numpy as np

try:
    import cytnx
except ImportError as exc:
    raise ImportError(
        "cytnx is required for mpo.py. Install/import cytnx first."
    ) from exc

from .uniTensor_core import assert_bond_match, svd_by_labels

MPO_SITE_LABELS = frozenset({"l", "ip", "i", "r"})
MPO_BOUNDARY_LABELS = frozenset({"mid", "up", "dn"})


def assert_mpo_site_labels(tensor: "cytnx.UniTensor", site: int) -> None:
    """Raise ValueError if `tensor` is not rank-4 or labels are not exactly l, ip, i, r."""
    if len(tensor.shape()) != 4:
        raise ValueError(
            f"Site {site} must be rank-4 (l, ip, i, r); got rank {len(tensor.shape())}."
        )
    if set(tensor.labels()) != MPO_SITE_LABELS:
        raise ValueError(
            f"Site {site} labels must be exactly l, ip, i, r; got {tensor.labels()}."
        )


def _assert_boundary_tensor(tensor: "cytnx.UniTensor", name: str) -> None:
    """Raise ValueError if boundary tensor labels are not exactly mid, up, dn."""
    if set(tensor.labels()) != MPO_BOUNDARY_LABELS:
        raise ValueError(
            f"{name} labels must be exactly mid, up, dn; got {tensor.labels()}."
        )
    if tensor.bond("up").dim() != 1 or tensor.bond("dn").dim() != 1:
        raise ValueError(
            f"{name} bonds 'up' and 'dn' must each have dimension 1."
        )


class MPO:
    """Open-boundary MPO with UniTensor site tensors (labels `l`, `ip`, `i`, `r`).

    Boundary tensors `L0` and `R0` are stored as attributes and are constructed
    automatically unless overridden.
    """

    def __init__(
        self,
        tensors: Iterable,
        *,
        L0: "cytnx.UniTensor | None" = None,
        R0: "cytnx.UniTensor | None" = None,
    ) -> None:
        """Load site tensors left to right; validate labels and bonds.

        Args:
            tensors: Iterable of cytnx.UniTensor with labels {l, ip, i, r}.
            L0: Optional custom left boundary tensor (labels mid, up, dn).
                If None, constructed automatically.
            R0: Optional custom right boundary tensor (labels mid, up, dn).
                If None, constructed automatically.

        Raises:
            ValueError: If tensors is empty, labels are wrong, physical bonds are
                inconsistent, or neighbor virtual bonds do not match.
            TypeError: If any element is not cytnx.UniTensor.
        """
        self.tensors = list(tensors)
        if not self.tensors:
            raise ValueError("An MPO must contain at least one site tensor.")
        for i, tensor in enumerate(self.tensors):
            if not isinstance(tensor, cytnx.UniTensor):
                raise TypeError(
                    f"Site {i} must be cytnx.UniTensor; got {type(tensor).__name__}."
                )
            assert_mpo_site_labels(tensor, i)
        self._validate_bonds()

        if L0 is None and R0 is None:
            self.L0, self.R0 = self._make_boundary_tensors()
        else:
            if L0 is None or R0 is None:
                raise ValueError(
                    "Both L0 and R0 must be provided together, or neither."
                )
            _assert_boundary_tensor(L0, "L0")
            _assert_boundary_tensor(R0, "R0")
            self.L0 = L0
            self.R0 = R0
        # Observer callbacks: same mechanism as MPS.  See MPS.register_callback.
        self._callbacks: list = []

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------

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
        """Replace one site tensor; fire Observer callbacks on success."""
        if not isinstance(tensor, cytnx.UniTensor):
            raise TypeError(
                f"Site {site} must be cytnx.UniTensor; got {type(tensor).__name__}."
            )
        assert_mpo_site_labels(tensor, site)
        self.tensors[site] = tensor
        live = []
        for obj_ref, method_name in self._callbacks:
            obj = obj_ref()
            if obj is not None:
                getattr(obj, method_name)(site)
                live.append((obj_ref, method_name))
        self._callbacks = live

    def register_callback(self, obj, method_name: str = "delete") -> None:
        """Register *obj.method_name* as an Observer callback.

        Identical contract to ``MPS.register_callback``.  The callback fires
        with the site index whenever a tensor is updated via ``__setitem__``.
        """
        assert hasattr(obj, method_name), (
            f"{type(obj).__name__} has no method '{method_name}'"
        )
        self._callbacks.append((weakref.ref(obj), method_name))

    def __repr__(self) -> str:
        return (
            f"MPO(num_sites={len(self)}, phys_dims={self.phys_dims}, "
            f"mpo_dims={self.mpo_dims})"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def phys_dims(self) -> list[int]:
        """Incoming physical dimension at each site (from bond 'i')."""
        return [tensor.bond("i").dim() for tensor in self.tensors]

    @property
    def mpo_dims(self) -> list[int]:
        """MPO virtual bond dimensions including both endpoints.

        Length is `num_sites + 1`: first element is `bond('l')` of site 0,
        last element is `bond('r')` of the last site.
        """
        dims = [self.tensors[0].bond("l").dim()]
        dims.extend(tensor.bond("r").dim() for tensor in self.tensors)
        return dims

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def check_site_labels(self) -> None:
        """Assert every site satisfies rank-4 and labels `l`, `ip`, `i`, `r`."""
        for i, tensor in enumerate(self.tensors):
            assert_mpo_site_labels(tensor, i)

    def _validate_bonds(self) -> None:
        """Validate labels, physical bond consistency, and neighbor virtual bonds.

        Checks:
        1. Every site has rank-4 and labels exactly {l, ip, i, r}.
        2. Physical bond consistency: bond('i') matches bond('ip') up to direction.
        3. Neighbor virtual bonds: mpo[p].bond('r') matches mpo[p+1].bond('l').

        Note: Unlike MPS, the MPO endpoint bonds are NOT required to have dim == 1.
        Open-boundary conditions are enforced via L0/R0 (whose 'up'/'dn' bonds have
        dim == 1), not by restricting the MPO virtual bond dimensions.
        """
        self.check_site_labels()
        for site, tensor in enumerate(self.tensors):
            try:
                assert_bond_match(tensor.bond("i"), tensor.bond("ip"))
            except ValueError as exc:
                raise ValueError(
                    f"Site {site}: physical bonds 'i' and 'ip' are inconsistent."
                ) from exc

        for site in range(len(self.tensors) - 1):
            try:
                assert_bond_match(
                    self.tensors[site].bond("r"),
                    self.tensors[site + 1].bond("l"),
                )
            except ValueError as exc:
                raise ValueError(
                    f"Virtual bond mismatch between sites {site} and {site + 1}."
                ) from exc

    # ------------------------------------------------------------------
    # Boundary tensors
    # ------------------------------------------------------------------

    def _make_boundary_tensors(
        self,
    ) -> tuple["cytnx.UniTensor", "cytnx.UniTensor"]:
        """Construct L0 and R0 from the MPO endpoint bonds.

        Uses the standard upper-triangular MPO convention:
            L0[d_l - 1, 0, 0] = 1.0
            R0[0,        0, 0] = 1.0
        """
        d_l = self.tensors[0].bond("l").dim()
        d_r = self.tensors[-1].bond("r").dim()

        arr_l = np.zeros((d_l, 1, 1), dtype=float)
        arr_l[d_l - 1, 0, 0] = 1.0
        L0 = cytnx.UniTensor(cytnx.from_numpy(arr_l), rowrank=1)
        L0.set_labels(["mid", "up", "dn"])

        arr_r = np.zeros((d_r, 1, 1), dtype=float)
        arr_r[0, 0, 0] = 1.0
        R0 = cytnx.UniTensor(cytnx.from_numpy(arr_r), rowrank=1)
        R0.set_labels(["mid", "up", "dn"])

        return L0, R0

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------

    def copy(self) -> "MPO":
        """Deep copy via `UniTensor.clone()`; L0/R0 are reconstructed."""
        cloned = [tensor.clone() for tensor in self.tensors]
        return MPO(cloned)

    # ------------------------------------------------------------------
    # Bond compression
    # ------------------------------------------------------------------

    def compress_bond(
        self,
        bond: int,
        *,
        max_dim: int | None = None,
        cutoff: float = 0.0,
        absorb: str = "right",
    ) -> tuple[int, float]:
        """Compress one MPO virtual bond in place via SVD truncation.

        Merges site tensors at `bond` and `bond + 1`, SVD-truncates the shared
        virtual bond, and writes the result back.  The row partition used is
        `["l", "ip1", "i1"]` (rowrank-3 semantics), matching the old
        ``svd_bond_mpo`` convention.

        Args:
            bond: Bond index to compress (0 to num_sites - 2).
            max_dim: Maximum bond dimension to keep; None means no limit.
            cutoff: Discard singular values below this threshold.
            absorb: Which side absorbs the singular values ('left' or 'right').

        Returns:
            Tuple of `(kept_dim, discarded_weight)`.
        """
        if not 0 <= bond < len(self) - 1:
            raise IndexError(f"Bond {bond} is outside [0, {len(self) - 2}].")
        if absorb not in {"left", "right"}:
            raise ValueError("absorb must be 'left' or 'right'.")

        keep = sys.maxsize if max_dim is None else max_dim
        if keep <= 0:
            raise ValueError("max_dim must be positive.")

        a1 = self.tensors[bond].relabels(["i", "ip", "r"], ["i1", "ip1", "_"])
        a2 = self.tensors[bond + 1].relabels(["i", "ip", "l"], ["i2", "ip2", "_"])
        aa = cytnx.Contract(a1, a2)
        aa.permute_(["l", "ip1", "i1", "ip2", "i2", "r"])

        left_new, right_new, discarded = svd_by_labels(
            aa,
            row_labels=["l", "ip1", "i1"],
            absorb=absorb,
            dim=keep,
            cutoff=cutoff,
            aux_label="aux",
        )
        left_new.relabels_(["ip1", "i1", "aux"], ["ip", "i", "r"])
        right_new.relabels_(["ip2", "i2", "aux"], ["ip", "i", "l"])

        kept_dim = left_new.bond("r").dim()
        self.tensors[bond] = left_new
        self.tensors[bond + 1] = right_new
        # Recompute L0/R0 in case endpoint bond dims changed.
        self.L0, self.R0 = self._make_boundary_tensors()
        return kept_dim, discarded
