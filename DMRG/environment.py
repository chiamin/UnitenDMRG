"""Left/right environment tensor cache for DMRG and related algorithms.

Concepts
--------
For an N-site system, environment tensors are indexed from -1 to N:

    LR[-1]          : left boundary (constructed from MPS and MPO edge bonds)
    LR[0..N-1]      : site environments (left or right, depending on position)
    LR[N]           : right boundary (constructed from MPS and MPO edge bonds)

Left environment LR[p] is computed by contracting bra, MPO, ket from site 0 up to
and including site p (reading left-to-right).  Right environment LR[p] contracts
from site N-1 down to and including site p.

Stale window [centerL, centerR]
--------------------------------
Only environments *outside* [centerL, centerR] are guaranteed up-to-date:

    valid left envs  : LR[-1], LR[0], ..., LR[centerL - 1]
    valid right envs : LR[centerR + 1], ..., LR[N-1], LR[N]
    stale (invalid)  : LR[centerL], ..., LR[centerR]

Call `update_envs(new_centerL, new_centerR)` to shrink the stale window
(i.e. compute missing environments).  Call `delete(i)` to mark LR[i] as
stale and expand the stale window.

Observer pattern
-----------------
Each subclass registers itself as a callback on the MPS/MPO objects it was
constructed with.  Whenever a tensor is updated via `MPS.__setitem__` or
`MPO.__setitem__`, `delete(site)` is called automatically, keeping the
stale window accurate without any manual bookkeeping.

The callbacks are stored as *weak references* so that LR objects are garbage
collected normally when they go out of scope (e.g. when a DMRG function
returns), without needing an explicit `unregister` call.
"""

from __future__ import annotations

import cytnx

from unitensor.utils import any_complex_tensors
from MPS.mpo import MPO


# ---------------------------------------------------------------------------
# Bond direction convention for environment tensors
# ---------------------------------------------------------------------------
#
# Convention (derived from MPS bond directions l=BD_IN, r=BD_OUT):
#
#   Left env  (L):  "dn" = BD_OUT,  "up" = BD_IN
#   Right env (R):  "dn" = BD_IN,   "up" = BD_OUT
#
# "dn" carries the ket (mps1) virtual bond.
# "up" carries the bra (mps2, Daggered) virtual bond.

def assert_left_env_dirs(env: "cytnx.UniTensor") -> None:
    """Assert that a left environment tensor has the expected bond directions.

    Skipped for dense (non-QN) tensors where bond direction is irrelevant.
    """
    if not env.is_blockform():
        return
    assert env.bond("dn").type() == cytnx.bondType.BD_BRA, (
        f'Left env "dn" must be BD_OUT, got {env.bond("dn").type()}'
    )
    assert env.bond("up").type() == cytnx.bondType.BD_KET, (
        f'Left env "up" must be BD_IN, got {env.bond("up").type()}'
    )


def assert_right_env_dirs(env: "cytnx.UniTensor") -> None:
    """Assert that a right environment tensor has the expected bond directions.

    Skipped for dense (non-QN) tensors where bond direction is irrelevant.
    """
    if not env.is_blockform():
        return
    assert env.bond("dn").type() == cytnx.bondType.BD_KET, (
        f'Right env "dn" must be BD_IN, got {env.bond("dn").type()}'
    )
    assert env.bond("up").type() == cytnx.bondType.BD_BRA, (
        f'Right env "up" must be BD_OUT, got {env.bond("up").type()}'
    )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class LREnv:
    """Abstract base: stale-window bookkeeping for left/right environment tensors.

    Subclasses must implement `_grow_left(p, prev_env)` and
    `_grow_right(p, next_env)` to define the actual tensor contraction.

    Parameters
    ----------
    N : int
        Number of sites.
    L0 : cytnx.UniTensor
        Left boundary tensor (stored at index -1).
    R0 : cytnx.UniTensor
        Right boundary tensor (stored at index N).
    tensors_to_watch : list
        MPS/MPO objects whose `register_callback` will be called so that
        `delete` fires automatically on tensor updates.
    init_center : int
        Site around which environments are initialised.  After `__init__`
        all environments outside `[init_center, init_center]` are valid.
    """

    def __init__(
        self,
        N: int,
        L0: "cytnx.UniTensor",
        R0: "cytnx.UniTensor",
        tensors_to_watch: list,
        init_center: int = 0,
    ) -> None:
        assert 0 <= init_center <= N - 1, (
            f"init_center={init_center} out of range [0, {N-1}]"
        )
        self.N = N
        # Stale window: LR[centerL..centerR] is invalid.
        self.centerL = 0
        self.centerR = N - 1
        # Dict keyed by site index (-1 to N).  None means "not yet computed".
        self.LR: dict = {i: None for i in range(-1, N + 1)}
        self.LR[-1] = L0
        self.LR[N] = R0

        # Register this object as a callback observer on all given tensors.
        # Weak references inside MPS/MPO ensure we don't prevent GC.
        for obj in tensors_to_watch:
            obj.register_callback(self)

        # Compute all right environments (and any left envs to the left of
        # init_center).  After this call the stale window is [init_center, init_center].
        self.update_envs(init_center, init_center)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __getitem__(self, i: int) -> "cytnx.UniTensor":
        """Return a validated environment tensor.

        Raises
        ------
        ValueError  : if `i` is out of the valid range [-1, N].
        RuntimeError: if LR[i] is inside the stale window.
        AssertionError: if LR[i] is in the valid zone but has not been computed
                        (should not happen in normal usage).
        """
        if not (-1 <= i <= self.N):
            raise ValueError(
                f"Environment index {i} out of range [-1, {self.N}]."
            )
        if self.centerL <= i <= self.centerR:
            raise RuntimeError(
                f"LR[{i}] is in the stale window [{self.centerL}, {self.centerR}]."
                " Call update_envs() first."
            )
        result = self.LR[i]
        assert result is not None, (
            f"LR[{i}] is marked valid but has not been computed. "
            "This should not happen; please report a bug."
        )
        return result

    def delete(self, i: int) -> None:
        """Mark LR[i] as stale by expanding the stale window to include i.

        Called automatically via the Observer pattern whenever a MPS/MPO tensor
        at site i is updated.  May also be called manually.

        LR[i] (whether it is a left or right environment) always depends on the
        tensor at site i, so marking exactly LR[i] as stale is precise.

        Parameters
        ----------
        i : int
            Site index.  Must be in [-1, N].
        """
        assert -1 <= i <= self.N, (
            f"delete({i}): index out of range [-1, {self.N}]."
        )
        self.centerL = min(self.centerL, i)
        self.centerR = max(self.centerR, i)

    def update_envs(self, centerL: int, centerR: int | None = None) -> None:
        """Compute missing environments to shrink the stale window.

        After this call the stale window becomes `[centerL, centerR]`.
        All environments outside this window are guaranteed up-to-date.

        The method only computes what is missing:

        * Left  loop: sites `self.centerL` → `centerL - 1` (growing rightward).
        * Right loop: sites `self.centerR` → `centerR + 1` (growing leftward).

        Parameters
        ----------
        centerL : int
            Left edge of the new stale window.
        centerR : int, optional
            Right edge of the new stale window.  Defaults to `centerL`.

        Raises
        ------
        ValueError : if centerL > centerR + 1 (invalid window), or if either
                     index is out of range [-1, N].
        """
        if centerR is None:
            centerR = centerL
        if centerL > centerR + 1:
            raise ValueError(
                f"centerL={centerL} > centerR+1={centerR+1}: invalid stale window."
            )
        if not (-1 <= centerL <= self.N) or not (-1 <= centerR <= self.N):
            raise ValueError(
                f"centerL={centerL} or centerR={centerR} out of range [-1, {self.N}]."
            )

        # Grow left environments: compute LR[self.centerL], ..., LR[centerL - 1]
        for p in range(self.centerL, centerL):
            self.LR[p] = self._grow_left(p, self.LR[p - 1])

        # Grow right environments: compute LR[self.centerR], ..., LR[centerR + 1]
        for p in range(self.centerR, centerR, -1):
            self.LR[p] = self._grow_right(p, self.LR[p + 1])

        self.centerL = centerL
        self.centerR = centerR

    # ------------------------------------------------------------------
    # Abstract methods (subclasses implement the actual contractions)
    # ------------------------------------------------------------------

    def _grow_left(self, p: int, prev_env: "cytnx.UniTensor") -> "cytnx.UniTensor":
        """Contract one site into the left environment.

        Parameters
        ----------
        p        : site index being incorporated.
        prev_env : left environment just to the left of site p (i.e. LR[p-1]).

        Returns
        -------
        New left environment LR[p] with labels appropriate for the subclass.
        """
        raise NotImplementedError

    def _grow_right(self, p: int, next_env: "cytnx.UniTensor") -> "cytnx.UniTensor":
        """Contract one site into the right environment.

        Parameters
        ----------
        p        : site index being incorporated.
        next_env : right environment just to the right of site p (i.e. LR[p+1]).

        Returns
        -------
        New right environment LR[p] with labels appropriate for the subclass.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# MPO environment  <bra| H |ket>
# ---------------------------------------------------------------------------

class OperatorEnv(LREnv):
    """Environment tensors for <mps2| MPO |mps1>.

    Used to project an operator (e.g. Hamiltonian) into the effective local
    subspace for EffOperator.

    Environment tensor labels: `["mid", "dn", "up"]`
        mid : MPO virtual bond
        dn  : ket (mps1) virtual bond
        up  : bra (mps2, Daggered) virtual bond

    The boundary tensors L0 / R0 are constructed from the MPS and MPO edge bonds
    via `_make_op_boundaries`.

    Parameters
    ----------
    mps1        : MPS — ket state.
    mps2        : MPS — bra state (use same object as mps1 for ground-state DMRG).
    mpo         : MPO — Hamiltonian.
    init_center : int — initial stale-window center (default 0).
    """

    def __init__(self, mps1, mps2, mpo, init_center: int = 0) -> None:
        assert len(mps1) == len(mps2) == len(mpo), (
            f"mps1 ({len(mps1)}), mps2 ({len(mps2)}), mpo ({len(mpo)}) "
            "must all have the same number of sites."
        )
        # Store tensor references.  These are live references, so _grow_left /
        # _grow_right always see the current tensors without being re-passed.
        self.mps1 = mps1
        self.mps2 = mps2
        self.mpo = mpo
        L0, R0 = OperatorEnv._make_op_boundaries(mpo, mps1, mps2)
        super().__init__(
            N=len(mps1),
            L0=L0,
            R0=R0,
            # Register callbacks on all three objects (mps1, mps2, mpo).
            # Double registration when mps1 is mps2 is harmless because
            # delete() is idempotent (min/max on the same value has no effect).
            tensors_to_watch=[mps1, mps2, mpo],
            init_center=init_center,
        )

    @staticmethod
    def _make_op_boundaries(mpo, mps1, mps2):
        """Construct rank-3 left and right boundary tensors from MPO and MPS edge bonds.

        Bond directions are chosen to match the contractions in _grow_left/_grow_right:
            L0["mid"] BD_OUT contracts with W[0]["l"]  BD_IN
            L0["dn"]  BD_OUT contracts with mps1[0]["l"] BD_IN
            L0["up"]  BD_IN  contracts with mps2[0].Dagger()["l"] BD_OUT
            R0["mid"] BD_IN  contracts with W[-1]["r"] BD_OUT
            R0["dn"]  BD_IN  contracts with mps1[-1]["r"] BD_OUT
            R0["up"]  BD_OUT contracts with mps2[-1].Dagger()["r"] BD_IN
        """
        use_complex = (
            any_complex_tensors(mpo)
            or any_complex_tensors(mps1)
            or any_complex_tensors(mps2)
        )
        ut_dtype = cytnx.Type.ComplexDouble if use_complex else cytnx.Type.Double

        b_mid = mpo[0].bond("l").redirect()
        b_dn  = mps1[0].bond("l").redirect()
        b_up  = mps2[0].bond("l")
        L0 = cytnx.UniTensor(
            [b_mid, b_dn, b_up], labels=["mid", "dn", "up"], dtype=ut_dtype
        )
        L0.at([0, 0, 0]).value = 1.0

        b_mid = mpo[-1].bond("r").redirect()
        b_dn  = mps1[-1].bond("r").redirect()
        b_up  = mps2[-1].bond("r")
        R0 = cytnx.UniTensor(
            [b_mid, b_dn, b_up], labels=["mid", "dn", "up"], dtype=ut_dtype
        )
        R0.at([0, 0, 0]).value = 1.0
        assert_left_env_dirs(L0)
        assert_right_env_dirs(R0)
        return L0, R0

    def _grow_left(self, p: int, prev_env: "cytnx.UniTensor") -> "cytnx.UniTensor":
        """Extend left environment by one site to the right (incorporating site p).

        Contraction diagram (left-to-right):

              dn ---A1[p]---
               |      |
            E[p-1]--W[p]----  =  E[p]
               |      |
              up ---A2†[p]--

            E[p-1] legs: dn → A1 l,  mid → W l,  up → A2† l
            W legs: i → A1 i (ket physical),  ip → A2† i (bra physical)
            result labels: ["mid", "dn", "up"]
        """
        # Rename shared indices so cytnx.Contract contracts them.
        E  = prev_env.relabels(["mid", "dn", "up"], ["_mid", "_dn", "_up"])
        A1 = self.mps1[p].relabels(["l", "i", "r"], ["_dn", "_i", "dn"])
        W  = self.mpo[p].relabels(["l", "ip", "i", "r"], ["_mid", "_ip", "_i", "mid"])
        A2 = self.mps2[p].Dagger().relabels(["l", "i", "r"], ["_up", "_ip", "up"])
        tmp = cytnx.Contract(E, A1)
        tmp = cytnx.Contract(tmp, W)
        tmp = cytnx.Contract(tmp, A2)
        return tmp  # labels: ["mid", "dn", "up"]

    def _grow_right(self, p: int, next_env: "cytnx.UniTensor") -> "cytnx.UniTensor":
        """Extend right environment by one site to the left (incorporating site p).

        Contraction diagram (right-to-left):

            ---A1[p]--- dn
                  |     |
            ---W[p]---E[p+1]  =  E[p]
                  |     |
            ---A2†[p]-- up

            E[p+1] legs: dn → A1 r,  mid → W r,  up → A2† r
            W legs: i → A1 i (ket physical),  ip → A2† i (bra physical)
            result labels: ["mid", "dn", "up"]
        """
        E  = next_env.relabels(["mid", "dn", "up"], ["_mid", "_dn", "_up"])
        A1 = self.mps1[p].relabels(["r", "i", "l"], ["_dn", "_i", "dn"])
        W  = self.mpo[p].relabels(["l", "ip", "i", "r"], ["mid", "_ip", "_i", "_mid"])
        A2 = self.mps2[p].Dagger().relabels(["r", "i", "l"], ["_up", "_ip", "up"])
        tmp = cytnx.Contract(E, A1)
        tmp = cytnx.Contract(tmp, W)
        tmp = cytnx.Contract(tmp, A2)
        return tmp  # labels: ["mid", "dn", "up"]


# ---------------------------------------------------------------------------
# MPS-only environment  <bra|ket>  (used for excited-state penalty)
# ---------------------------------------------------------------------------

class VectorEnv(LREnv):
    """Environment tensors for the overlap <mps2|mps1>.

    Used to project a state vector (e.g. a reference MPS) into the effective
    local subspace for EffVector.

    Environment tensor labels: `["dn", "up"]`
        dn : ket (mps1) virtual bond
        up : bra (mps2, Daggered) virtual bond

    Boundary tensors are constructed automatically from the MPS boundary bonds.

    Parameters
    ----------
    mps1        : MPS — ket state.
    mps2        : MPS — bra state (the fixed orthogonal reference state).
    init_center : int — initial stale-window center (default 0).
    """

    def __init__(self, mps1, mps2, init_center: int = 0) -> None:
        assert len(mps1) == len(mps2), (
            f"mps1 ({len(mps1)}) and mps2 ({len(mps2)}) must have the same length."
        )
        self.mps1 = mps1
        self.mps2 = mps2
        L0, R0 = self._make_boundaries(mps1, mps2)
        super().__init__(
            N=len(mps1),
            L0=L0,
            R0=R0,
            tensors_to_watch=[mps1, mps2],
            init_center=init_center,
        )

    @staticmethod
    def _make_boundaries(mps1, mps2):
        """Construct scalar (1x1) left and right boundary tensors from MPS bonds."""
        use_complex = any_complex_tensors(mps1) or any_complex_tensors(mps2)
        ut_dtype = cytnx.Type.ComplexDouble if use_complex else cytnx.Type.Double

        # Left boundary: dimension-1 bonds from site 0's left bond of each MPS.
        # redirect() flips bond direction so the two bonds can be paired.
        l1 = mps1[0].bond("l").redirect()
        l2 = mps2[0].bond("l")
        L0 = cytnx.UniTensor([l1, l2], labels=["dn", "up"], dtype=ut_dtype)
        L0.at([0, 0]).value = 1.0

        r1 = mps1[-1].bond("r").redirect()
        r2 = mps2[-1].bond("r")
        R0 = cytnx.UniTensor([r1, r2], labels=["dn", "up"], dtype=ut_dtype)
        R0.at([0, 0]).value = 1.0
        assert_left_env_dirs(L0)
        assert_right_env_dirs(R0)
        return L0, R0

    def _grow_left(self, p: int, prev_env: "cytnx.UniTensor") -> "cytnx.UniTensor":
        """Extend left overlap environment by one site to the right.

        Contraction diagram (left-to-right):

              dn ---A1[p]---
              |       |
            E[p-1]    |      =  E[p]
              |       |
              up ---A2†[p]--

            E[p-1] legs: dn → A1 l,  up → A2† l
            A1 and A2† share the same physical index i (bra = ket)
            result labels: ["dn", "up"]
        """
        E  = prev_env.relabels(["dn", "up"], ["_dn", "_up"])
        A1 = self.mps1[p].relabels(["l", "i", "r"], ["_dn", "i", "dn"])
        A2 = self.mps2[p].Dagger().relabels(["l", "i", "r"], ["_up", "i", "up"])
        tmp = cytnx.Contract(E, A1)
        tmp = cytnx.Contract(tmp, A2)
        return tmp  # labels: ["dn", "up"]

    def _grow_right(self, p: int, next_env: "cytnx.UniTensor") -> "cytnx.UniTensor":
        """Extend right overlap environment by one site to the left.

        Contraction diagram (right-to-left):

            ---A1[p]--- dn
                  |     |
                  |   E[p+1]  =  E[p]
                  |     |
            ---A2†[p]-- up

            E[p+1] legs: dn → A1 r,  up → A2† r
            A1 and A2† share the same physical index i (bra = ket)
            result labels: ["dn", "up"]
        """
        E  = next_env.relabels(["dn", "up"], ["_dn", "_up"])
        A1 = self.mps1[p].relabels(["r", "i", "l"], ["_dn", "i", "dn"])
        A2 = self.mps2[p].Dagger().relabels(["r", "i", "l"], ["_up", "i", "up"])
        tmp = cytnx.Contract(E, A1)
        tmp = cytnx.Contract(tmp, A2)
        return tmp  # labels: ["dn", "up"]


# ---------------------------------------------------------------------------
# MPO product environment  Tr[M_fit† · MPO1 · MPO2]
# ---------------------------------------------------------------------------

class MPOProductEnv(LREnv):
    """Environment tensors for Tr[fitmpo† · mpo1 · mpo2].

    Used to variationally fit a compressed MPO to the product of two MPOs.

    Environment tensor labels: `["up", "mid", "dn"]`
        up  : fitmpo (Daggered bra) virtual bond
        mid : mpo1 virtual bond
        dn  : mpo2 virtual bond

    Contraction diagram (left-to-right, absorbing site p):

          up ---[A†]--- up'        fitmpo site (Daggered)
               |ip  |i
         mid --[W1]-- mid'         mpo1 site
               |ip  |i
          dn --[W2]-- dn'          mpo2 site

        Physical contractions:
            A†.ip ↔ W1.ip      (top)
            W1.i  ↔ W2.ip      (middle)
            A†.i  ↔ W2.i       (bottom)

    Parameters
    ----------
    mpo1       : MPO — first operator (upper).
    mpo2       : MPO — second operator (lower).
    fitmpo     : MPO — the MPO being fitted (modified in-place by the caller).
    init_center : int — initial stale-window center (default 0).
    """

    def __init__(self, mpo1: MPO, mpo2: MPO, fitmpo: MPO,
                 init_center: int = 0) -> None:
        assert len(mpo1) == len(mpo2) == len(fitmpo), (
            f"mpo1 ({len(mpo1)}), mpo2 ({len(mpo2)}), fitmpo ({len(fitmpo)}) "
            "must all have the same number of sites."
        )
        self.mpo1 = mpo1
        self.mpo2 = mpo2
        self.fitmpo = fitmpo
        L0, R0 = MPOProductEnv._make_boundaries(mpo1, mpo2, fitmpo)
        super().__init__(
            N=len(mpo1),
            L0=L0,
            R0=R0,
            tensors_to_watch=[fitmpo],
            init_center=init_center,
        )

    @staticmethod
    def _make_boundaries(mpo1, mpo2, fitmpo):
        """Construct rank-3 left and right boundary tensors.

        Bond directions follow the same logic as OperatorEnv: the boundary
        tensor legs must contract with the corresponding edge bonds.

            L0["mid"] contracts with W1[0]["l"]         → redirect of W1[0]["l"]
            L0["dn"]  contracts with W2[0]["l"]         → redirect of W2[0]["l"]
            L0["up"]  contracts with fitmpo[0].Dag["l"] → fitmpo[0]["l"] as-is
            (analogous for R0 with the last site's "r" bonds)
        """
        use_complex = (
            any_complex_tensors(mpo1)
            or any_complex_tensors(mpo2)
            or any_complex_tensors(fitmpo)
        )
        ut_dtype = cytnx.Type.ComplexDouble if use_complex else cytnx.Type.Double

        b_mid = mpo1[0].bond("l").redirect()
        b_dn = mpo2[0].bond("l").redirect()
        b_up = fitmpo[0].bond("l")
        L0 = cytnx.UniTensor(
            [b_up, b_mid, b_dn], labels=["up", "mid", "dn"], dtype=ut_dtype,
        )
        L0.at([0, 0, 0]).value = 1.0

        b_mid = mpo1[-1].bond("r").redirect()
        b_dn = mpo2[-1].bond("r").redirect()
        b_up = fitmpo[-1].bond("r")
        R0 = cytnx.UniTensor(
            [b_up, b_mid, b_dn], labels=["up", "mid", "dn"], dtype=ut_dtype,
        )
        R0.at([0, 0, 0]).value = 1.0
        return L0, R0

    def _grow_left(self, p: int, prev_env: "cytnx.UniTensor") -> "cytnx.UniTensor":
        """Extend left environment by absorbing site p (left-to-right).

        Contraction order:
            1. E  + A†  → contract "up" ↔ A†.l
            2. tmp + W1 → contract "mid" ↔ W1.l,  A†.ip ↔ W1.ip
            3. tmp + W2 → contract "dn" ↔ W2.l,  W1.i ↔ W2.ip,  A†.i ↔ W2.i
        Result labels: ["up", "mid", "dn"]
        """
        E = prev_env.relabels(["up", "mid", "dn"], ["_up", "_mid", "_dn"])
        A = self.fitmpo[p].Dagger().relabels(
            ["l", "ip", "i", "r"], ["_up", "_ip", "_i", "up"],
        )
        W1 = self.mpo1[p].relabels(
            ["l", "ip", "i", "r"], ["_mid", "_ip", "_i1", "mid"],
        )
        W2 = self.mpo2[p].relabels(
            ["l", "ip", "i", "r"], ["_dn", "_i1", "_i", "dn"],
        )
        tmp = cytnx.Contract(E, A)
        tmp = cytnx.Contract(tmp, W1)
        tmp = cytnx.Contract(tmp, W2)
        return tmp  # labels: ["up", "mid", "dn"]

    def _grow_right(self, p: int, next_env: "cytnx.UniTensor") -> "cytnx.UniTensor":
        """Extend right environment by absorbing site p (right-to-left).

        Same contraction pattern as `_grow_left` but using the "r" bonds.
        """
        E = next_env.relabels(["up", "mid", "dn"], ["_up", "_mid", "_dn"])
        A = self.fitmpo[p].Dagger().relabels(
            ["l", "ip", "i", "r"], ["up", "_ip", "_i", "_up"],
        )
        W1 = self.mpo1[p].relabels(
            ["l", "ip", "i", "r"], ["mid", "_ip", "_i1", "_mid"],
        )
        W2 = self.mpo2[p].relabels(
            ["l", "ip", "i", "r"], ["dn", "_i1", "_i", "_dn"],
        )
        tmp = cytnx.Contract(E, A)
        tmp = cytnx.Contract(tmp, W1)
        tmp = cytnx.Contract(tmp, W2)
        return tmp  # labels: ["up", "mid", "dn"]
