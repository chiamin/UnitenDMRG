"""Effective operators for DMRG local subspace optimisation.

Given an MPS |ψ⟩ being optimised and a set of centre sites to update, the
tensors *outside* those sites act as projectors mapping the full Hilbert space
down to an effective (local) subspace.  This module provides two classes that
live in that subspace:

EffOperator
    Projects a linear operator (e.g. a Hamiltonian given as an MPO) into the
    effective subspace.  Provides:
        make_phi(*mps_tensors)          – merge site tensors into a single φ
        apply(phi)                      – H_eff |φ⟩  (used by Lanczos)
        add_term(eff_vec, weight)       – add a weighted rank-1 term
                                          w |Φ_0⟩⟨Φ_0| to H_eff
                                          (weight can be positive or negative)
        split_phi(phi, max_dim, cutoff, absorb)  – SVD φ back into site tensors

EffVector
    Projects a reference state vector (e.g. a previously found eigenstate)
    into the effective subspace.  Provides:
        inner(phi)                      – ⟨Φ_0|φ⟩  scalar overlap

Both classes receive their left/right environment tensors explicitly (Option A
design): the caller is responsible for calling OperatorEnv.update_envs /
VectorEnv.update_envs before constructing these objects.
"""

from __future__ import annotations

try:
    import cytnx
except ImportError as exc:
    raise ImportError("cytnx is required for effective_operators.py.") from exc

from MPS.mps import MPS


# ---------------------------------------------------------------------------
# EffVector
# ---------------------------------------------------------------------------

class EffVector:
    """Effective vector  ⟨φ| projected into the local subspace.

    ``EffVector`` represents a reference MPS |φ⟩ projected into the same
    local subspace as the optimisation variable φ.  The projected vector
    |Φ_0⟩ is pre-computed in ``__init__`` by contracting the left/right
    overlap environments with the reference site tensors.

    Parameters
    ----------
    L : UniTensor
        Left overlap environment, labels ``["dn", "up"]``.
        Typically ``VectorEnv[p-1]``.
    R : UniTensor
        Right overlap environment, labels ``["dn", "up"]``.
        Typically ``VectorEnv[p+n]`` where n = number of sites.
    *mps_tensors : UniTensor
        Site tensors of the reference state |φ⟩, in site order.
        Labels ``["l", "i", "r"]``.  Typically 1 or 2 tensors.

    Notes
    -----
    The contraction order is sequential left-to-right:
    L × A[0] × A[1] × … × R → |Φ_0⟩

    ``|Φ_0⟩`` has the same labels as the optimisation variable φ produced by
    ``EffOperator.make_phi``: ``["l", "i0", "i1", …, "r"]``.

    ``inner(phi)`` then computes ⟨Φ_0|φ⟩ as a direct contraction (no Dagger
    needed because the bond directions in L/R already handle bra/ket).
    """

    def __init__(self, L: "cytnx.UniTensor", R: "cytnx.UniTensor",
                 *mps_tensors: "cytnx.UniTensor") -> None:
        self.tensor = self._precompute(L, R, mps_tensors)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def inner(self, phi: "cytnx.UniTensor") -> complex:
        """Compute ⟨Φ_0|φ⟩ → scalar.

        Notes
        -----
        Diagram (2-site example):

          |Φ_0⟩:   l ─── i0 ─── i1 ─── r
          φ    :   l ─── i0 ─── i1 ─── r

          Contract all matching labels → scalar.

        The bond directions set by VectorEnv already encode the bra side, so
        no explicit Dagger() is needed here.
        """
        return cytnx.Contract(self.tensor, phi).item()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _precompute(self, L, R, mps_tensors):
        """Contract L + A[0] + … + A[n-1] + R → |Φ_0⟩.

        Diagram (2-site):

          L["dn","up"] ── A0["l"→"up","i"→"i0","r"→"_b0"] ──
                          A1["l"→"_b0","i"→"i1","r"→"up_r"] ── R["dn","up"]

          Free indices: L["dn"]="l", i0, i1, R["dn"]="r"
          Result labels: ["l", "i0", "i1", "r"]
        """
        n = len(mps_tensors)

        if n == 0:
            # 0-site: contract L["up"] with R["up"] (shared virtual bond at cut).
            # L["dn"] → "l" (output bra bond), R["dn"] → "r" (output bra bond).
            # (rare; mainly for TDVP bond step)
            tmp  = L.relabels(["dn", "up"], ["l",  "_bond"])
            tmp2 = R.relabels(["dn", "up"], ["r",  "_bond"])
            return cytnx.Contract(tmp, tmp2)

        # Connect L["up"] → A[0]["l"]
        tmp = L.relabels(["dn", "up"], ["l", "_up"])
        for k, A in enumerate(mps_tensors):
            i_label = f"i{k}"
            if k < n - 1:
                r_label = f"_b{k}"
            else:
                r_label = "_up_r"      # will connect to R["up"]
            l_label = "_up" if k == 0 else f"_b{k-1}"
            Ak = A.relabels(["l", "i", "r"], [l_label, i_label, r_label])
            tmp = cytnx.Contract(tmp, Ak)

        # Connect last bond → R["up"]
        tmp2 = R.relabels(["dn", "up"], ["r", "_up_r"])
        tmp = cytnx.Contract(tmp, tmp2)
        return tmp


# ---------------------------------------------------------------------------
# EffOperator
# ---------------------------------------------------------------------------

class EffOperator:
    """Effective Hamiltonian (or general operator) projected into local subspace.

    Supports 0-, 1-, and 2-site DMRG via a variable number of MPO tensors.

    Parameters
    ----------
    L : UniTensor
        Left operator environment, labels ``["mid", "dn", "up"]``.
        Typically ``OperatorEnv[p-1]``.
    R : UniTensor
        Right operator environment, labels ``["mid", "dn", "up"]``.
        Typically ``OperatorEnv[p+n]`` where n = number of sites.
    *mpo_tensors : UniTensor
        MPO site tensors in site order, labels ``["l", "ip", "i", "r"]``.
        0 tensors → 0-site (bond-only), 1 → single-site, 2 → two-site DMRG.
    """

    def __init__(self, L: "cytnx.UniTensor", R: "cytnx.UniTensor",
                 *mpo_tensors: "cytnx.UniTensor") -> None:
        self._L = L
        self._R = R
        self._mpo_tensors = list(mpo_tensors)

        # Weighted rank-1 terms:  list of (EffVector, weight)
        # Used for excited-state targeting; weight can be positive or negative.
        # Conceptually: H_eff → H_eff + Σ_k weight_k |Φ_k⟩⟨Φ_k|
        self._terms: list[tuple["EffVector", complex]] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def apply(self, phi: "cytnx.UniTensor") -> "cytnx.UniTensor":
        """Compute H_eff |φ⟩  (including any added rank-1 terms).

        Parameters
        ----------
        phi : UniTensor
            The optimisation variable.  Must have labels ``"l"`` and ``"r"``
            for the virtual bonds; all other labels are treated as physical
            indices (in MPO site order).

        Returns
        -------
        result : UniTensor   same shape and labels as phi.
        """
        result = self._apply_operator(phi)
        _labels = list(result.labels())

        # Add rank-1 terms:  Σ_k w_k |Φ_k⟩ ⟨Φ_k|φ⟩
        # weight can be positive (penalty) or negative; see add_term().
        # Cytnx addition drops labels; restore them after each accumulation.
        for eff_vec, weight in self._terms:
            overlap = eff_vec.inner(phi)        # ⟨Φ_k|φ⟩  scalar
            result = result + (weight * overlap) * eff_vec.tensor
            result.set_labels(_labels)

        return result

    def add_term(self, eff_vec: "EffVector", weight: complex) -> None:
        """Add a weighted rank-1 term  w |Φ_0⟩⟨Φ_0|  to H_eff.

        Typical use: excited-state targeting in DMRG, where ``weight`` is a
        positive energy penalty that pushes the optimisation away from a
        previously found eigenstate.  A negative ``weight`` would instead
        *attract* the solution toward |Φ_0⟩.

        Parameters
        ----------
        eff_vec : EffVector
            Projected reference vector |Φ_0⟩.
        weight : complex
            Coefficient w.  Positive → penalty; negative → reward.
        """
        self._terms.append((eff_vec, weight))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_operator(self, phi: "cytnx.UniTensor") -> "cytnx.UniTensor":
        """Contract L + φ + M[0] + M[1] + … + R → H_eff|φ⟩.

        Index convention
        ----------------
        L   : ["mid", "dn", "up"]
        phi : ["l", "i0", ..., "i{n-1}", "r"]  (canonical label convention)
        M[k]: ["l", "ip", "i", "r"]
        R   : ["mid", "dn", "up"]
        output: same labels as phi.

        Contraction diagram (2-site):

          │    │      │    │      ← output: l, i0_out, i1_out, r
          L ── M0 ── M1 ── R
          │    │      │    │
          └──────── φ ────┘

        The contraction proceeds left to right:
          step 1:  L + φ         (contract ldn)
          step 2:  + M[0]        (contract lmid, i0)
          step 3:  + M[1]        (contract mid0, i1)   [2-site only]
          step 4:  + R           (contract rmid, rdn)
        """
        n = len(self._mpo_tensors)
        # Physical labels from canonical convention — order is guaranteed.
        phys_in  = [MPS._phi_label(k) for k in range(n)]
        phys_out = [MPS._phi_label(k) + "_out" for k in range(n)]

        # -- 0-site special case --
        # No MPO tensors: L["mid"] and R["mid"] must contract directly with
        # each other (they represent the same MPO virtual bond at the cut).
        # The generic code below leaves "lmid" and "rmid" uncontracted,
        # producing a 4-index result instead of 2-index.
        if n == 0:
            #
            #   L: ["mid","dn","up"] → ["_mid","ldn","lup"]
            #   R: ["mid","dn","up"] → ["_mid","rdn","rup"]
            #   phi: ["l","r"]       → ["ldn","rdn"]
            #
            #   Contract order: (L + phi) contracts "ldn",
            #                   then + R   contracts "_mid" and "rdn".
            #   Output: ["lup","rup"] → ["l","r"]
            #
            L    = self._L.relabels(["mid", "dn", "up"], ["_mid", "ldn", "lup"])
            R    = self._R.relabels(["mid", "dn", "up"], ["_mid", "rdn", "rup"])
            phi_r = phi.relabels(["l", "r"], ["ldn", "rdn"])
            tmp  = cytnx.Contract(L, phi_r)
            tmp  = cytnx.Contract(tmp, R)
            return tmp.relabels(["lup", "rup"], ["l", "r"])

        # -- Relabel L --
        #
        #   L: ["mid","dn","up"] → ["lmid","ldn","lup"]
        #
        L = self._L.relabels(["mid", "dn", "up"], ["lmid", "ldn", "lup"])

        # -- Relabel phi --
        #
        #   phi: ["l", p0, ..., "r"] → ["ldn", p0, ..., "rdn"]
        #
        phi_r = phi.relabels(["l", "r"], ["ldn", "rdn"])

        # step 1: L + phi  →  contracts ldn
        #
        #   ----
        #   |  |── lup
        #   |  |── lmid
        #   |  |── ldn ──── p0 ──── p1 ──── rdn
        #   ----
        #
        tmp = cytnx.Contract(L, phi_r)

        # step 2..n+1: contract each MPO tensor
        prev_mid = "lmid"
        for k, M in enumerate(self._mpo_tensors):
            next_mid = f"mid{k}" if k < n - 1 else "rmid"
            #
            #   M[k]: ["l","ip","i","r"] → [prev_mid, pk_out, pk_in, next_mid]
            #
            #   Contracts: prev_mid (MPO virtual) and pk_in (physical input)
            #
            M_r = M.relabels(["l", "ip", "i", "r"],
                              [prev_mid, phys_out[k], phys_in[k], next_mid])
            tmp = cytnx.Contract(tmp, M_r)
            prev_mid = next_mid

        # step n+2: contract R  →  contracts rmid, rdn
        #
        #   R: ["mid","dn","up"] → ["rmid","rdn","rup"]
        #
        R = self._R.relabels(["mid", "dn", "up"], ["rmid", "rdn", "rup"])
        tmp = cytnx.Contract(tmp, R)

        # -- Relabel output back to phi's original labels --
        #
        #   output has: lup, p0_out, p1_out, ..., rup
        #   rename to:  l,   p0,     p1,     ..., r
        #
        out_labels_now  = ["lup"] + phys_out + ["rup"]
        out_labels_want = ["l"]   + phys_in  + ["r"]
        return tmp.relabels(out_labels_now, out_labels_want)
