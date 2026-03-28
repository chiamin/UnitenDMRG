"""PhysicalSite: local Hilbert space for one lattice site."""

from __future__ import annotations

import numpy as np

import cytnx

from ..mps import MPS
from ..uniTensor_core import bond_qnums_at, derive_delta_qn


class PhysicalSite:
    """Encapsulates the local Hilbert space for one lattice site.

    The physical bond (BD_IN) encodes the index→QN mapping via its sectors.
    Factory functions (e.g. spin_half) construct the appropriate instance.

    Operators are stored in a registry mapping name → (d×d numpy matrix, delta_qn).

    Parameters
    ----------
    bond : cytnx.Bond
        Physical bond, must be BD_IN.
    type_name : str
        Optional label, e.g. "SpinHalf", "Electron".
    ops : dict[str, tuple[np.ndarray, int]] | None
        Operator definitions: name → (matrix, delta_qn).
        delta_qn is derived automatically if not provided via register_op.
    """

    def __init__(
        self,
        bond: "cytnx.Bond",
        type_name: str = "",
        ops: "dict[str, tuple[np.ndarray, int]] | None" = None,
    ) -> None:
        if not isinstance(bond, cytnx.Bond):
            raise TypeError(f"bond must be cytnx.Bond; got {type(bond).__name__}.")
        if bond.type() != cytnx.BD_IN:
            raise ValueError("Physical bond must be BD_IN.")
        self._bond = bond
        self.type_name = type_name
        self._ops: dict[str, np.ndarray] = {}
        self._delta_qn: dict[str, int] = {}
        if ops:
            for name, (matrix, delta_qn) in ops.items():
                self.register_op(name, matrix, delta_qn)

    # ------------------------------------------------------------------
    # Operator registry
    # ------------------------------------------------------------------

    def register_op(self, name: str, matrix: np.ndarray, delta_qn: int) -> None:
        """Register a local operator by name.

        Parameters
        ----------
        name : str
            Operator name, e.g. "Sz", "Sp".
        matrix : np.ndarray
            d×d matrix in the physical basis (row=ip/bra, col=i/ket).
        delta_qn : int
            QN charge of this operator: QN(ip) - QN(i) for every nonzero element.
            Use derive_delta_qn(matrix, bond) to compute this automatically.
        """
        matrix = np.asarray(matrix)
        d = self._bond.dim()
        if matrix.shape != (d, d):
            raise ValueError(
                f"Operator '{name}' must be ({d},{d}); got {matrix.shape}."
            )
        self._ops[name] = matrix
        self._delta_qn[name] = delta_qn

    def op(self, name: str) -> np.ndarray:
        """Return the d×d numpy matrix for operator `name`."""
        if name not in self._ops:
            raise KeyError(
                f"Operator '{name}' not registered in {self.type_name or 'PhysicalSite'}. "
                f"Available: {list(self._ops.keys())}"
            )
        return self._ops[name]

    def op_delta_qn(self, name: str) -> int:
        """Return the QN charge of operator `name`."""
        if name not in self._delta_qn:
            raise KeyError(f"Operator '{name}' not registered.")
        return self._delta_qn[name]

    def has_qn(self) -> bool:
        """Return True if this site uses QN symmetry."""
        return self._bond.Nsym() > 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def bond(self) -> "cytnx.Bond":
        """Physical bond (BD_IN)."""
        return self._bond

    @property
    def dim(self) -> int:
        """Physical Hilbert space dimension."""
        return self._bond.dim()

    # ------------------------------------------------------------------
    # Product state
    # ------------------------------------------------------------------

    def product_state(
        self,
        states: list[int],
        center: int = 0,
        dtype: np.dtype | type = float,
    ) -> MPS:
        """Create a product-state MPS for a uniform chain.

        Parameters
        ----------
        states : list[int]
            Physical index at each site (0-based).  All sites use this
            PhysicalSite's bond.
        center : int
            Orthogonality center of the returned MPS (default 0).
            Only sets center_left/center_right; does not affect tensor values.
        dtype : np.dtype | type
            Tensor dtype for the returned MPS tensors (default float).

        Returns
        -------
        MPS with center_left = center_right = center.
        """
        N = len(states)
        if N == 0:
            raise ValueError("states must be non-empty.")
        if not 0 <= center < N:
            raise IndexError(f"center={center} out of range [0, {N-1}].")
        d = self._bond.dim()
        for i, s in enumerate(states):
            if not isinstance(s, int):
                raise TypeError(f"states[{i}] must be int; got {type(s).__name__}.")
            if not 0 <= s < d:
                raise ValueError(f"states[{i}]={s} out of range [0, {d-1}].")

        out_dtype = np.dtype(dtype)
        if self._bond.Nsym() == 0:
            tensors = self._make_dense(states, out_dtype)
        else:
            tensors = self._make_qn(states, out_dtype)

        mps = MPS(tensors)
        mps.center_left = mps.center_right = center
        return mps

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_dense(
        self, states: list[int], out_dtype: np.dtype
    ) -> list["cytnx.UniTensor"]:
        """Build dense (no-QN) product-state tensors."""
        d = self._bond.dim()
        tensors = []
        for s in states:
            arr = np.zeros((1, d, 1), dtype=out_dtype)
            arr[0, s, 0] = 1.0
            ut = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=2)
            ut.set_labels(["l", "i", "r"])
            tensors.append(ut)
        return tensors

    def _make_qn(
        self, states: list[int], out_dtype: np.dtype
    ) -> list["cytnx.UniTensor"]:
        """Build QN-symmetric product-state tensors."""
        phys_bond = self._bond
        syms = list(phys_bond.syms())
        nsym = phys_bond.Nsym()
        acc_qn = [0] * nsym   # cumulative QN; becomes left bond QN of each site
        tensors = []
        ut_dtype = (
            cytnx.Type.ComplexDouble
            if np.issubdtype(out_dtype, np.complexfloating)
            else cytnx.Type.Double
        )

        for s in states:
            phys_qn = bond_qnums_at(phys_bond, s)   # QN of chosen physical state

            # Left bond carries accumulated QN from all previous sites.
            b_left = cytnx.Bond(cytnx.BD_IN, [acc_qn], [1], syms)
            # Right bond QN = left + physical (QN conservation).
            new_acc_qn = [syms[k].combine_rule(acc_qn[k], phys_qn[k])
                          for k in range(nsym)]
            b_right = cytnx.Bond(cytnx.BD_OUT, [new_acc_qn], [1], syms)

            ut = cytnx.UniTensor(
                [b_left, phys_bond, b_right], rowrank=2, dtype=ut_dtype
            )
            ut.set_labels(["l", "i", "r"])
            ut.at([0, s, 0]).value = 1.0   # s is the flat physical index

            acc_qn = new_acc_qn
            tensors.append(ut)

        return tensors
