"""AutoMPO: construct general MPOs from sums of operator strings.

Usage
-----
    from MPS.physical_sites import spin_half
    from MPS.auto_mpo import AutoMPO

    site = spin_half(qn="Sz")
    ampo = AutoMPO(N, site)
    for i in range(N - 1):
        ampo.add(J * delta, "Sz", i, "Sz", i + 1)
        ampo.add(J / 2,     "Sp", i, "Sm", i + 1)
        ampo.add(J / 2,     "Sm", i, "Sp", i + 1)
    for i in range(N):
        ampo.add(h, "Sz", i)
    H = ampo.to_mpo()

Interface
---------
    ampo.add(coeff, "Op1", site1, "Op2", site2, ...)
        coeff   : scalar (real or complex)
        "OpK"   : operator name registered in PhysicalSite
        siteK   : int, 0-based site index (must be strictly increasing)

Algorithm
---------
    Finite-state-machine (FSM) construction with partial-string merging.

    Each term is a list of (site, op_name) pairs with a coefficient.
    The automaton state at each bond encodes which partial operator strings
    are "in progress" (applied on the left, waiting for completion on the right).

    States are identified by their left-operator-sequence prefix so that
    terms sharing the same prefix share the same automaton state, keeping
    bond dimension minimal.

    Virtual bond QN for a state = total delta_qn of the remaining
    (right-side) operators still to be applied.

    W matrix convention (upper-triangular):
        State 0      = "done"  (QN = +total_charge)
        State D-1    = "start" (QN = 0)
        States 1..D-2 = in-progress partial strings

    Boundary conditions are absorbed directly into W[0] and W[-1]:
        W[0]["l"]  has dim=1, containing only the "start" state.
        W[-1]["r"] has dim=1, containing only the "done"  state.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import cytnx
except ImportError as exc:
    raise ImportError("cytnx is required for auto_mpo.py.") from exc

from .mpo import MPO
from .physical_sites.site import PhysicalSite


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _Term:
    """A single term: coeff * O_{s0} O_{s1} ... O_{sk}."""
    coeff: complex | float
    ops: list[tuple[int, str]]   # [(site, op_name), ...] sorted by site

    @property
    def total_delta_qn(self) -> int:
        return sum(dq for _, dq in self._op_delta_qns)

    def delta_qns(self, site_obj: PhysicalSite) -> list[int]:
        return [site_obj.op_delta_qn(name) for _, name in self.ops]

    def total_delta_qn_from(self, site_obj: PhysicalSite) -> int:
        return sum(self.delta_qns(site_obj))


# ---------------------------------------------------------------------------
# Automaton state key
# ---------------------------------------------------------------------------

# A partial-string key identifies which partial operator string has been applied.
# It is a tuple of (site, op_name) pairs for the already-applied operators.
# Terms that share the same prefix share the same automaton state.
_PartialKey = tuple[tuple[int, str], ...]


# ---------------------------------------------------------------------------
# AutoMPO
# ---------------------------------------------------------------------------

class AutoMPO:
    """Collect operator terms and build an MPO via the FSM algorithm."""

    def __init__(self, N: int, site: PhysicalSite) -> None:
        if N < 1:
            raise ValueError("N must be >= 1.")
        if not isinstance(site, PhysicalSite):
            raise TypeError("site must be a PhysicalSite.")
        self._N = N
        self._site = site
        self._terms: list[_Term] = []
        self._total_charge: int | None = None   # set on first add()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add(self, coeff: complex | float, *args) -> None:
        """Add one term to the Hamiltonian.

        Signature:
            add(coeff, "Op1", site1, "Op2", site2, ...)

        Sites must be strictly increasing.
        """
        ops = self._parse_args(args)
        coeff_arr = np.asarray(coeff)
        if coeff_arr.ndim != 0:
            raise TypeError("coeff must be a scalar.")
        term = _Term(coeff=coeff_arr.item(), ops=ops)
        term_charge = term.total_delta_qn_from(self._site)

        if self._total_charge is None:
            self._total_charge = term_charge
        elif term_charge != self._total_charge:
            raise ValueError(
                f"Inconsistent total delta_qn: existing terms have "
                f"{self._total_charge}, new term has {term_charge}. "
                "All terms must carry the same total QN charge."
            )
        self._terms.append(term)

    def to_mpo(self) -> MPO:
        """Build and return the MPO."""
        if not self._terms:
            raise RuntimeError("No terms have been added.")
        return _FSMBuilder(self._N, self._site, self._terms,
                           self._total_charge).build()

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_args(self, args) -> list[tuple[int, str]]:
        """Parse ("Op", site, "Op", site, ...) into [(site, op), ...]."""
        if len(args) % 2 != 0:
            raise ValueError(
                "add() expects pairs of (op_name, site): "
                "add(coeff, 'Op', site, 'Op', site, ...)"
            )
        ops = []
        for k in range(0, len(args), 2):
            op_name, site_idx = args[k], args[k + 1]
            if not isinstance(op_name, str):
                raise TypeError(f"Expected operator name (str), got {type(op_name)}.")
            if not isinstance(site_idx, int):
                raise TypeError(f"Expected site index (int), got {type(site_idx)}.")
            if not 0 <= site_idx < self._N:
                raise ValueError(f"Site {site_idx} out of range [0, {self._N - 1}].")
            # Validate operator exists
            self._site.op(op_name)
            ops.append((site_idx, op_name))
        # Must be strictly increasing in site
        for k in range(len(ops) - 1):
            if ops[k][0] >= ops[k + 1][0]:
                raise ValueError(
                    f"Operator sites must be strictly increasing; "
                    f"got site {ops[k][0]} before site {ops[k+1][0]}."
                )
        return ops


# ---------------------------------------------------------------------------
# FSM builder
# ---------------------------------------------------------------------------

class _FSMBuilder:
    """Build the MPO W tensors using the finite-state-machine algorithm."""

    # Special state keys
    _DONE  = "done"
    _START = "start"

    def __init__(
        self,
        N: int,
        site: PhysicalSite,
        terms: list[_Term],
        total_charge: int,
    ) -> None:
        self._N = N
        self._site = site
        self._terms = terms
        self._total_charge = total_charge
        self._d = site.dim                    # physical dimension
        self._use_qn = site.has_qn()
        self._out_dtype = self._resolve_out_dtype()

        # done state QN = +total_charge  (derived from QN conservation)
        self._done_qn  = +total_charge
        self._start_qn = 0

    def _resolve_out_dtype(self) -> np.dtype:
        """Decide a single global dtype for all MPO tensors.

        Rule:
          - If any used operator matrix is complex, OR
          - If any term coefficient is complex,
            then use complex.
          - Otherwise use float.
        """
        is_complex = False

        # Identity is always used by _fill_identity.
        if np.iscomplexobj(self._site.op("I")):
            is_complex = True

        for term in self._terms:
            if np.iscomplexobj(term.coeff):
                is_complex = True
                break
            for _, op_name in term.ops:
                if np.iscomplexobj(self._site.op(op_name)):
                    is_complex = True
                    break
            if is_complex:
                break

        return np.dtype(complex if is_complex else float)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def build(self) -> MPO:
        # Step 1: enumerate all automaton states at each bond
        bond_states = self._enumerate_states()   # list[list[state_key]], len=N+1
        # Absorb boundary conditions into W[0] and W[-1]:
        # truncate to a single state at each endpoint so that dim=1.
        bond_states[0]       = [self._START]
        bond_states[self._N] = [self._DONE]
        # Step 2: build W numpy arrays
        W_arrays = self._build_W_arrays(bond_states)
        # Step 3: convert to UniTensors
        tensors = self._make_tensors(W_arrays, bond_states)
        return MPO(tensors)

    # ------------------------------------------------------------------
    # Step 1: enumerate automaton states at each bond
    # ------------------------------------------------------------------

    def _enumerate_states(self) -> list[list[Any]]:
        """Return bond_states[p] = ordered list of state keys at bond p.

        bond_states[0]   = states entering site 0  (left  boundary bond)
        bond_states[N]   = states leaving  site N-1 (right boundary bond)

        Each list always starts with DONE (index 0) and ends with START (index D-1).
        In-progress partial-string keys fill positions 1..D-2.
        """
        # Collect all partial-string keys that cross each bond.
        # A term [(s0,O0),(s1,O1),...,(sk,Ok)] contributes partial key
        # (s0,O0),...,(sj,Oj) to bonds p where sj < p <= s_{j+1}.
        in_progress: list[set[_PartialKey]] = [set() for _ in range(self._N + 1)]

        for term in self._terms:
            ops = term.ops
            for j in range(len(ops) - 1):
                # After applying ops[0..j], before applying ops[j+1]:
                partial_key = tuple(ops[:j + 1])
                left_site  = ops[j][0]
                right_site = ops[j + 1][0]
                # This partial string is in-progress at bonds left_site+1 .. right_site
                for p in range(left_site + 1, right_site + 1):
                    in_progress[p].add(partial_key)

        bond_states: list[list[Any]] = []
        for p in range(self._N + 1):
            partials = sorted(in_progress[p])
            all_states = [self._DONE] + partials + [self._START]
            # Sort by (QN, str(key)) so that states with the same QN are adjacent.
            # This ensures states-list index == QN-bond flat index, which is
            # required for convert_from to place values in the correct blocks.
            all_states.sort(key=lambda k: (self._state_qn(k), str(k)))
            bond_states.append(all_states)
        return bond_states

    # ------------------------------------------------------------------
    # Step 2: build dense W numpy arrays
    # ------------------------------------------------------------------

    def _build_W_arrays(
        self, bond_states: list[list[Any]]
    ) -> list[np.ndarray]:
        """Build a dense numpy W array for each site."""
        W_arrays = []
        for p in range(self._N):
            left_states  = bond_states[p]
            right_states = bond_states[p + 1]
            Dl = len(left_states)
            Dr = len(right_states)
            d  = self._d
            W  = np.zeros((Dl, d, d, Dr), dtype=self._out_dtype)

            # --- diagonal identity: done→done, start→start, partial→partial ---
            self._fill_identity(W, left_states, right_states)

            # --- term contributions ---
            for term in self._terms:
                self._fill_term(W, p, term, left_states, right_states)

            W_arrays.append(W)
        return W_arrays

    def _state_index(self, states: list[Any], key: Any) -> int | None:
        """Return index of key in states list, or None if absent."""
        try:
            return states.index(key)
        except ValueError:
            return None

    def _fill_identity(
        self,
        W: np.ndarray,
        left_states: list[Any],
        right_states: list[Any],
    ) -> None:
        """Fill diagonal identity blocks: done→done, start→start, partial→partial."""
        I_mat = self._site.op("I")
        for key in (self._DONE, self._START):
            l = self._state_index(left_states, key)
            r = self._state_index(right_states, key)
            if l is not None and r is not None:
                W[l, :, :, r] += I_mat
        # in-progress partials that survive unchanged across this site
        for key in left_states:
            if key in (self._DONE, self._START):
                continue
            r = self._state_index(right_states, key)
            l = self._state_index(left_states, key)
            if l is not None and r is not None:
                W[l, :, :, r] += I_mat

    def _fill_term(
        self,
        W: np.ndarray,
        p: int,
        term: _Term,
        left_states: list[Any],
        right_states: list[Any],
    ) -> None:
        """Add contributions of one term to W at site p."""
        ops = term.ops
        # Find which operators of this term act at site p
        acting = [(j, site, name) for j, (site, name) in enumerate(ops) if site == p]
        if not acting:
            return   # identity at this site handled by _fill_identity

        for j, site, name in acting:
            op_mat = self._site.op(name)
            partial_left  = tuple(ops[:j])          # prefix before this op
            partial_right = tuple(ops[:j + 1])       # prefix including this op

            # Determine left and right automaton states for this transition
            if j == 0:
                l_key = self._START
            else:
                l_key = partial_left

            if j == len(ops) - 1:
                r_key = self._DONE
            else:
                r_key = partial_right

            l = self._state_index(left_states, l_key)
            r = self._state_index(right_states, r_key)
            if l is None or r is None:
                continue   # this state doesn't exist at this bond (shouldn't happen)

            coeff = term.coeff if (j == 0) else 1.0
            W[l, :, :, r] += coeff * op_mat

    # ------------------------------------------------------------------
    # Step 3: convert numpy arrays to UniTensors
    # ------------------------------------------------------------------

    def _make_tensors(
        self, W_arrays: list[np.ndarray], bond_states: list[list[Any]]
    ) -> list["cytnx.UniTensor"]:
        tensors = []
        for p in range(self._N):
            W = W_arrays[p]
            if self._use_qn:
                ut = self._numpy_to_qn_tensor(W, bond_states[p], bond_states[p + 1])
            else:
                ut = cytnx.UniTensor(cytnx.from_numpy(W), rowrank=2)
                ut.set_labels(["l", "ip", "i", "r"])
            tensors.append(ut)
        return tensors

    def _state_qn(self, key: Any) -> int:
        """Return the virtual bond QN for an automaton state key."""
        if key == self._DONE:
            return self._done_qn
        if key == self._START:
            return self._start_qn
        # Partial string: QN = sum of delta_qn already applied from the left
        applied = sum(self._site.op_delta_qn(name) for _, name in key)
        return +applied

    def _make_virtual_bond(
        self, states: list[Any], direction: int
    ) -> "cytnx.Bond":
        """Build a cytnx.Bond for a virtual bond from a list of automaton states.

        Groups states by QN into sectors.
        direction: cytnx.BD_IN or cytnx.BD_OUT
        """
        sym = cytnx.Symmetry.U1()
        # Collect (qn, count) preserving state order
        qn_order: list[int] = []
        qn_count: dict[int, int] = {}
        for key in states:
            qn = self._state_qn(key)
            if qn not in qn_count:
                qn_order.append(qn)
                qn_count[qn] = 0
            qn_count[qn] += 1
        qnums = [[q] for q in qn_order]
        degs  = [qn_count[q] for q in qn_order]
        return cytnx.Bond(direction, qnums, degs, [sym])

    def _numpy_to_qn_tensor(
        self,
        W: np.ndarray,
        left_states: list[Any],
        right_states: list[Any],
    ) -> "cytnx.UniTensor":
        """Convert dense W array to a QN UniTensor via convert_from."""
        phys_ket = self._site.bond        # BD_IN  (KET) for "ip"
        phys_bra = phys_ket.redirect()    # BD_OUT (BRA) for "i"
        l_bond = self._make_virtual_bond(left_states,  cytnx.BD_IN)
        r_bond = self._make_virtual_bond(right_states, cytnx.BD_OUT)

        # Dense wrapper
        ut_dense = cytnx.UniTensor(cytnx.from_numpy(W), rowrank=2)
        ut_dense.set_labels(["l", "ip", "i", "r"])

        # Target QN tensor: ip=BD_IN (KET), i=BD_OUT (BRA)
        # so W["i"] (BD_OUT/BRA) can contract with MPS["i"] (BD_IN/KET)
        ut_dtype = (
            cytnx.Type.ComplexDouble
            if np.issubdtype(W.dtype, np.complexfloating)
            else cytnx.Type.Double
        )
        ut_qn = cytnx.UniTensor(
            [l_bond, phys_ket, phys_bra, r_bond], rowrank=2, dtype=ut_dtype
        )
        ut_qn.set_labels(["l", "ip", "i", "r"])
        ut_qn.convert_from(ut_dense, True)
        return ut_qn

