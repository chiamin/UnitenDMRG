"""AutoMPO: construct general MPOs from sums of operator strings.

Usage
-----
    Bosonic (Heisenberg):

        site = spin_half(qn="Sz")
        ampo = AutoMPO(N, site)
        for i in range(N - 1):
            ampo.add(J / 2, "Sp", i, "Sm", i + 1)
            ampo.add(J / 2, "Sm", i, "Sp", i + 1)
        H = ampo.to_mpo()

    Fermionic (tight-binding):

        site = spinless_fermion(qn="N")
        ampo = AutoMPO(N, site)
        for i in range(N - 1):
            ampo.add(-t, "Cdag", i,     "C", i + 1)   # c+_i c_{i+1}
            ampo.add(-t, "Cdag", i + 1, "C", i)       # h.c. = c+_{i+1} c_i
        H = ampo.to_mpo()

Interface
---------
    ampo.add(coeff, "Op1", site1, "Op2", site2, ...)
        coeff   : scalar (real or complex)
        "OpK"   : operator name registered in PhysicalSite
        siteK   : int, 0-based site index

    Operators may be given in any order and on the same site.
    The product is interpreted left-to-right: add(c, "A", i, "B", j)
    means c * A_i B_j (B_j acts on the state first, then A_i).

    For fermionic operators (marked ``fermionic=True`` in PhysicalSite),
    Jordan-Wigner strings are inserted automatically.

Jordan-Wigner handling
----------------------
    Fermionic operators in PhysicalSite are "bare" (local) operators a, a+.
    The physical fermion operators include a JW string:

        c_k  = F_0 F_1 ... F_{k-1} a_k
        c+_k = F_0 F_1 ... F_{k-1} a+_k

    where F = (-1)^n is the on-site parity operator.

    When the user writes add(coeff, "Cdag", i, "C", j), AutoMPO interprets
    this as the *physical* operator product c+_i c_j and expands the JW
    strings to determine the correct on-site operator at each site.

    Example: c+_i c_j  (i < j) expands to

        site i  : a+_i F_i   (the F comes from c_j's JW string passing site i)
        site i+1..j-1 : F    (JW string from c_j)
        site j  : a_j        (bare operator)

    The operator ordering matters: c+_i c_j != c_j c+_i (they differ by -1).
    AutoMPO preserves the user-specified ordering when expanding JW strings.

    Implementation: _preprocess_term() computes the on-site operator for
    each operator site.  Intermediate sites (no user operator) are handled
    by _fill_identity(), which inserts F or I based on the parity of
    fermionic operators already applied (tracked in the partial key).

FSM algorithm
-------------
    Finite-state-machine construction with partial-string merging.

    Each term's operators (sorted by site after preprocessing) define a
    sequence of transitions through automaton states.  The automaton state
    at each bond encodes which partial operator strings are "in progress"
    (applied on the left, waiting for completion on the right).

    A partial key is a tuple of (site, symbolic_key) pairs, where
    symbolic_key records the operator names at that site (including any
    inserted F's).  Terms sharing the same partial key prefix share the
    same automaton state — this is prefix merging, which keeps the MPO
    bond dimension minimal.

    W matrix at each site has two types of entries:
      - Diagonal (state passes through unchanged):
          DONE->DONE, START->START: identity I
          partial->partial: F if partial has odd fermionic count, else I
      - Off-diagonal (transition at an operator site):
          Uses the pre-computed on-site matrix from _preprocess_term.
          The coefficient is placed on the last operator of each term.

    Boundary conditions are absorbed into W[0] and W[-1]:
        W[0]["l"]  has dim=1, containing only the "start" state.
        W[-1]["r"] has dim=1, containing only the "done"  state.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import cytnx

from .mpo import MPO
from .physical_sites.site import PhysicalSite
from unitensor.core import normalize_qn


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _Term:
    """A single term: coeff * O_{s0} O_{s1} ... O_{sk}.

    After preprocessing, *ops* holds the JW-expanded on-site operators
    sorted by site:
        [(site, symbolic_key, matrix), ...]
    where *symbolic_key* is a tuple of op-name strings (e.g. ("Cdag","F"))
    and *matrix* is the corresponding d×d numpy array.
    """
    coeff: complex | float
    ops: list[tuple[int, tuple[str, ...], "np.ndarray"]]

    def total_delta_qn_from(self, site_obj: PhysicalSite) -> list[int]:
        """Sum of delta_qn for the user operators (excluding inserted F).

        Returns a list with one int per symmetry.
        """
        total: list[int] | None = None
        for _, sym_key, _ in self.ops:
            for name in sym_key:
                if name != "F":
                    dq = site_obj.op_delta_qn(name)
                    if total is None:
                        total = list(dq)
                    else:
                        for k in range(len(total)):
                            total[k] += dq[k]
        if total is None:
            nsym = max(site_obj.bond.Nsym(), 1)
            total = [0] * nsym
        return total


# ---------------------------------------------------------------------------
# Automaton state key
# ---------------------------------------------------------------------------

# A partial-string key identifies which partial operator string has been applied.
# It is a tuple of (site, symbolic_key) pairs for the already-applied operators.
# Terms that share the same prefix share the same automaton state.
_PartialKey = tuple[tuple[int, tuple[str, ...]], ...]


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
        self._total_charge: list[int] | None = None   # set on first add()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add(self, coeff: complex | float, *args) -> None:
        """Add one term to the Hamiltonian.

        Signature:
            add(coeff, "Op1", site1, "Op2", site2, ...)

        Operators may be given in any site order; same-site operators
        are allowed.  The JW expansion is computed automatically.
        """
        user_ops = self._parse_args(args)
        coeff_arr = np.asarray(coeff)
        if coeff_arr.ndim != 0:
            raise TypeError("coeff must be a scalar.")
        processed_ops = self._preprocess_term(user_ops)
        term = _Term(coeff=coeff_arr.item(), ops=processed_ops)
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
        """Parse ("Op", site, "Op", site, ...) into [(site, op), ...].

        Returns ops in user order (not sorted).  Allows same-site and
        arbitrary site ordering; the caller is responsible for
        preprocessing (JW expansion) and sorting before passing to the
        FSM builder.
        """
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
        return ops

    # ------------------------------------------------------------------
    # JW preprocessing
    # ------------------------------------------------------------------

    def _preprocess_term(
        self, user_ops: list[tuple[int, str]]
    ) -> list[tuple[int, tuple[str, ...], "np.ndarray"]]:
        """Expand JW strings and compute on-site operators.

        Each fermionic operator c_k has a JW string F_0 F_1 ... F_{k-1}.
        For each operator site s, we scan user_ops left-to-right and
        collect the contributions from every operator in the term:

          - s_k == s            : append the bare op name (the operator
                                  acts on this site)
          - s_k > s and ferm.   : append "F" (the JW string of operator
                                  at s_k passes through site s)
          - otherwise           : identity, skip

        Consecutive F's cancel (F²=I): if the last entry in the sequence
        is already "F", appending another "F" pops it instead.

        The symbolic sequence is converted to a matrix by multiplying the
        corresponding operator matrices left-to-right.

        Returns [(site, symbolic_key, matrix), ...] sorted by site.
        """
        operator_sites = sorted(set(s for s, _ in user_ops))
        result = []
        for s in operator_sites:
            seq: list[str] = []
            for s_k, op_name in user_ops:
                if s_k == s:
                    seq.append(op_name)
                elif s_k > s and self._site.op_is_fermionic(op_name):
                    # FF cancellation
                    if seq and seq[-1] == "F":
                        seq.pop()
                    else:
                        seq.append("F")
            sym_key = tuple(seq)
            mat = np.eye(self._site.dim)
            for name in seq:
                mat = mat @ self._site.op(name)
            result.append((s, sym_key, mat))
        return result


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
        total_charge: list[int],
    ) -> None:
        self._N = N
        self._site = site
        self._terms = terms
        self._total_charge = total_charge
        self._d = site.dim                    # physical dimension
        self._use_qn = site.has_qn()
        self._nsym = max(site.bond.Nsym(), 1)
        self._out_dtype = self._resolve_out_dtype()
        self._has_fermion = self._check_fermion()

        # done state QN = +total_charge  (derived from QN conservation)
        if self._use_qn:
            syms = list(site.bond.syms())
            self._done_qn = tuple(normalize_qn(total_charge[k], syms[k])
                                  for k in range(self._nsym))
        else:
            self._done_qn = tuple(total_charge)
        self._start_qn = tuple([0] * self._nsym)

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
            for _, _, mat in term.ops:
                if np.iscomplexobj(mat):
                    is_complex = True
                    break
            if is_complex:
                break

        return np.dtype(complex if is_complex else float)

    def _check_fermion(self) -> bool:
        """Check whether any term contains fermionic operators.

        If so, validate that the site has an "F" operator registered.
        """
        for term in self._terms:
            for _, sym_key, _ in term.ops:
                for name in sym_key:
                    if name != "F" and self._site.op_is_fermionic(name):
                        self._site.op("F")   # raises KeyError if missing
                        return True
        return False

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
                partial_key = tuple((s, sk) for s, sk, _ in ops[:j + 1])
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
            seen: set[tuple[int, int]] = set()   # dedup shared-prefix transitions
            for term in self._terms:
                self._fill_term(W, p, term, left_states, right_states, seen)

            W_arrays.append(W)
        return W_arrays

    def _state_index(self, states: list[Any], key: Any) -> int | None:
        """Return index of key in states list, or None if absent."""
        try:
            return states.index(key)
        except ValueError:
            return None

    def _partial_is_fermionic(self, key: _PartialKey) -> bool:
        """Return True if the partial string has an odd number of fermionic operators.

        This determines what to fill at intermediate sites (sites between
        operator sites where the partial state passes through unchanged):
          - odd  → fill F (the JW string of remaining operators)
          - even → fill I

        Counts only user operators (C, Cdag, etc.), not inserted F's.
        """
        count = 0
        for _, sym_key in key:
            count += sum(1 for name in sym_key
                         if name != "F" and self._site.op_is_fermionic(name))
        return count % 2 == 1

    def _fill_identity(
        self,
        W: np.ndarray,
        left_states: list[Any],
        right_states: list[Any],
    ) -> None:
        """Fill diagonal blocks where a state passes through a site unchanged.

        - DONE→DONE, START→START: always identity I.
        - partial→partial (in-progress string, no operator at this site):
          If the partial string has applied an odd number of fermionic
          operators so far, the remaining operators' JW strings require
          an F at this site.  Otherwise, identity I.
        """
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
                if self._partial_is_fermionic(key):
                    W[l, :, :, r] += self._site.op("F")
                else:
                    W[l, :, :, r] += I_mat

    def _fill_term(
        self,
        W: np.ndarray,
        p: int,
        term: _Term,
        left_states: list[Any],
        right_states: list[Any],
        seen: set[tuple[int, int]],
    ) -> None:
        """Add contributions of one term to W at site p.

        Each operator site in the term produces a state transition in the
        automaton.  The on-site matrix (from _preprocess_term, which
        includes JW-inserted F's) is placed into the W matrix at the
        corresponding (left_state, right_state) entry.

        The coefficient is placed on the **last** operator of each term so
        that shared-prefix transitions carry only the operator matrix.
        Non-terminal transitions that have already been written (tracked by
        *seen*) are skipped to avoid double-counting shared prefixes.
        """
        ops = term.ops
        # Find which operators of this term act at site p
        acting = [(j, s, sk, mat) for j, (s, sk, mat) in enumerate(ops) if s == p]
        if not acting:
            return   # identity at this site handled by _fill_identity

        for j, site, sym_key, op_mat in acting:
            partial_left  = tuple((s, sk) for s, sk, _ in ops[:j])
            partial_right = tuple((s, sk) for s, sk, _ in ops[:j + 1])

            # Determine left and right automaton states for this transition
            if j == 0:
                l_key = self._START
            else:
                l_key = partial_left

            is_terminal = (j == len(ops) - 1)
            if is_terminal:
                r_key = self._DONE
            else:
                r_key = partial_right

            l = self._state_index(left_states, l_key)
            r = self._state_index(right_states, r_key)
            if l is None or r is None:
                continue

            if is_terminal:
                W[l, :, :, r] += term.coeff * op_mat
            else:
                lr = (l, r)
                if lr not in seen:
                    seen.add(lr)
                    W[l, :, :, r] += op_mat

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

    def _state_qn(self, key: Any) -> tuple[int, ...]:
        """Return the virtual bond QN for an automaton state key.

        Returns a tuple with one int per symmetry.
        For Zn symmetries each component is normalized to {0, ..., n-1}.
        """
        if key == self._DONE:
            return self._done_qn
        if key == self._START:
            return self._start_qn
        # Partial string: QN = sum of delta_qn of user operators (excluding F)
        applied = [0] * self._nsym
        for _, sym_key in key:
            for name in sym_key:
                if name != "F":
                    dq = self._site.op_delta_qn(name)
                    for k in range(self._nsym):
                        applied[k] += dq[k]
        if self._use_qn:
            syms = list(self._site.bond.syms())
            applied = [normalize_qn(applied[k], syms[k])
                       for k in range(self._nsym)]
        return tuple(applied)

    def _make_virtual_bond(
        self, states: list[Any], direction: int
    ) -> "cytnx.Bond":
        """Build a cytnx.Bond for a virtual bond from a list of automaton states.

        Groups states by QN into sectors.
        direction: cytnx.BD_IN or cytnx.BD_OUT
        """
        syms = list(self._site.bond.syms())
        # Collect (qn_tuple, count) preserving state order
        qn_order: list[tuple[int, ...]] = []
        qn_count: dict[tuple[int, ...], int] = {}
        for key in states:
            qn = self._state_qn(key)
            if qn not in qn_count:
                qn_order.append(qn)
                qn_count[qn] = 0
            qn_count[qn] += 1
        qnums = [list(q) for q in qn_order]
        degs  = [qn_count[q] for q in qn_order]
        return cytnx.Bond(direction, qnums, degs, syms)

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

