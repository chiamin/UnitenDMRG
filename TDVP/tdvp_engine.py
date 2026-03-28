"""TDVPEngine: time-dependent variational principle (TDVP) for MPS.

Implements 1-site and 2-site TDVP sweeps.  psi is modified in-place.
Environments are built once in __init__ and reused across sweeps.

Time convention
---------------
One full sweep (right then left) applies  exp(-dt * H)  to psi.

    Forward propagation  at each site:  exp(-dt/2 * H_eff)
    Backward propagation at each bond:  exp(+dt/2 * H_eff_0site)

For real-time evolution by physical step Δt:  pass  dt = 1j * delta_t
For imaginary-time evolution by step Δτ    :  pass  dt = delta_tau  (real)

Usage
-----
    engine = TDVPEngine(psi, H)
    for _ in range(n_steps):
        trunc = engine.sweep(dt)
"""

from __future__ import annotations

import cytnx

from MPS.mps import MPS
from MPS.linalg import lanczos_expm_multiply
from MPS.uniTensor_core import qr_by_labels
from DMRG.environment import OperatorEnv
from DMRG.effective_operators import EffOperator


class TDVPEngine:
    """TDVP sweep engine for a single MPS/MPO pair.

    Parameters
    ----------
    psi : MPS — state to evolve (modified in-place).
              Must have center == 0 (right-canonical form).
    H   : MPO — Hamiltonian.
    """

    def __init__(self, psi: MPS, H) -> None:
        if psi.center != 0:
            raise ValueError(
                f"psi.center must be 0 (right-canonical); got {psi.center}."
            )
        self.psi = psi
        self.H   = H
        self._op_env = OperatorEnv(psi, psi, H, init_center=0)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def sweep(
        self,
        dt: complex,
        max_dim: int | None = None,
        cutoff: float = 0.0,
        num_center: int = 1,
    ) -> float:
        """Perform one full sweep (right then left), applying exp(-dt*H).

        Parameters
        ----------
        dt         : complex — time step (see module docstring for convention).
        max_dim    : int | None — max bond dimension (2-site only).
        cutoff     : float — cutoff on normalized rho eigenvalues during
                     2-site splits.
        num_center : 1 or 2 — 1-site (fixed bond dim) or 2-site (can grow).

        Returns
        -------
        avg_trunc : float — average truncation error (0 for 1-site).
        """
        if num_center not in (1, 2):
            raise ValueError("num_center must be 1 or 2.")
        if self.psi.center != 0:
            raise ValueError(
                f"psi.center must be 0 at the start of each sweep; "
                f"got {self.psi.center}."
            )

        N = len(self.psi)
        truncs = []

        if num_center == 1:
            # Right sweep: p = 0 .. N-1
            for p in range(N):
                self._update_1site(p, dt, absorb="right")
            # Left sweep: p = N-1 .. 0
            for p in range(N - 1, -1, -1):
                self._update_1site(p, dt, absorb="left")

        else:  # num_center == 2
            # Right sweep: p = 0 .. N-2
            for p in range(N - 1):
                trunc = self._update_2site(p, dt, max_dim, cutoff, absorb="right")
                truncs.append(trunc)
            # Left sweep: p = N-2 .. 0
            for p in range(N - 2, -1, -1):
                trunc = self._update_2site(p, dt, max_dim, cutoff, absorb="left")
                truncs.append(trunc)

        return float(sum(truncs) / len(truncs)) if truncs else 0.0

    # ------------------------------------------------------------------
    # 1-site local update
    # ------------------------------------------------------------------

    def _update_1site(self, p: int, dt: complex, absorb: str) -> None:
        """1-site TDVP local update at site p.

        absorb='right': forward + QR + backward into p+1  (right sweep)
        absorb='left' : forward + QR + backward into p-1  (left sweep)

        Boundary sites (p=N-1 for right, p=0 for left) only do forward.
        """
        psi = self.psi
        N   = len(psi)
        _i0 = MPS._phi_label(0)

        # ── 1. Prepare environments ────────────────────────────────────
        # Need op_env[p-1] (left) and op_env[p+1] (right).
        self._op_env.update_envs(p, p)

        # ── 2. Forward propagation ─────────────────────────────────────
        # phi = exp(-dt/2 * H_eff_1site) |psi[p]>
        effH = EffOperator(self._op_env[p - 1], self._op_env[p + 1], self.H[p])
        phi  = psi.make_phi(p, 1)                              # ["l","i0","r"]
        phi  = lanczos_expm_multiply(effH.apply, phi, -0.5 * dt)

        # ── Boundary: no backward propagation ─────────────────────────
        is_boundary = (absorb == "right" and p == N - 1) or \
                      (absorb == "left"  and p == 0)
        if is_boundary:
            phi.relabels_([_i0], ["i"])
            psi[p] = phi
            psi.center_left = psi.center_right = p
            return

        # ── 3. QR decompose ────────────────────────────────────────────
        if absorb == "right":
            # Left-orthogonalise: row = ["l","i0"]
            #   A : ["l","i0","_s"]
            #   C : ["_s","r"]           ← bond tensor
            A, C = qr_by_labels(phi, row_labels=["l", _i0], aux_label="_s")
            A.relabels_(["l", _i0, "_s"], ["l", "i", "r"])
            # C["_s"] → new bond (A's right), C["r"] → psi[p+1]'s left bond.
            # Relabel so C["r"] matches psi[p+1]["l"] on contraction:
            #   "_s" → "_new_l",  "r" → "l"  (mirrors _shift_center_right_qr)
            C.relabels_(["_s", "r"], ["_new_l", "l"])

        else:  # absorb == "left"
            # Right-orthogonalise: row = ["i0","r"]
            #   A : ["i0","r","_s"]
            #   C : ["_s","l"]           ← bond tensor
            A, C = qr_by_labels(phi, row_labels=[_i0, "r"], aux_label="_s")
            A.permute_([_i0, "r", "_s"])
            A.set_labels(["i", "r", "_s"])
            A.permute_(["_s", "i", "r"])
            A.set_labels(["l", "i", "r"])
            A.set_rowrank_(2)
            # C["_s"] → new bond (A's left), C["l"] → psi[p-1]'s right bond.
            # Relabel so C["l"] matches psi[p-1]["r"] on contraction:
            #   "_s" → "_new_r",  "l" → "r"
            C.relabels_(["_s", "l"], ["_new_r", "r"])

        # ── 4. Update psi[p] ───────────────────────────────────────────
        # Bypass __setitem__ bond validation: QR may reduce the right bond when
        # D_l*d < D_r (over-specified state).  psi[p+1] will be fixed in step 6.
        psi.tensors[p] = A
        psi.center_left = min(psi.center_left, p)
        psi.center_right = max(psi.center_right, p)
        self._op_env.delete(p)   # fire the callback that __setitem__ would fire

        if absorb == "right":
            psi.center_left = psi.center_right = p + 1
            # update_envs(p+1, p): computes LR[p] with new A;
            # stale window → [p+1, p] = EMPTY → both LR[p] and LR[p+1] valid.
            self._op_env.update_envs(p + 1, p)
        else:
            psi.center_left = psi.center_right = p - 1
            # update_envs(p, p-1): computes LR[p] with new A;
            # stale window → [p, p-1] = EMPTY → both LR[p-1] and LR[p] valid.
            self._op_env.update_envs(p, p - 1)

        # ── 5. Backward propagation ────────────────────────────────────
        # C' = exp(+dt/2 * H_eff_0site) |C>
        if absorb == "right":
            effH_0 = EffOperator(self._op_env[p], self._op_env[p + 1])
        else:
            effH_0 = EffOperator(self._op_env[p - 1], self._op_env[p])

        # Rename C to canonical ["l","r"] for EffOperator.apply()
        if absorb == "right":
            C.relabels_(["_new_l", "l"], ["l", "r"])
        else:
            C.relabels_(["_new_r", "r"], ["r", "l"])
            C.permute_(["l", "r"])
            C.set_labels(["l", "r"])

        C = lanczos_expm_multiply(effH_0.apply, C, +0.5 * dt)

        # ── 6. Absorb C into neighbour ─────────────────────────────────
        if absorb == "right":
            # Contract C["r"] with psi[p+1]["l"]
            # C currently has labels ["l","r"]; rename "r"→"l" to match psi[p+1]["l"]
            C.relabels_(["l", "r"], ["_new_l", "l"])
            new_nb = cytnx.Contract(C, psi[p + 1])   # contracts "l"
            new_nb.relabels_(["_new_l", "i", "r"], ["l", "i", "r"])
            psi[p + 1] = new_nb   # fires op_env.delete(p+1)

        else:  # absorb == "left"
            # Contract psi[p-1]["r"] with C["l"]
            # C currently has labels ["l","r"]; rename "l"→"r" to match psi[p-1]["r"]
            C.relabels_(["l", "r"], ["r", "_new_r"])
            new_nb = cytnx.Contract(psi[p - 1], C)   # contracts "r"
            new_nb.relabels_(["l", "i", "_new_r"], ["l", "i", "r"])
            psi[p - 1] = new_nb   # fires op_env.delete(p-1)

    # ------------------------------------------------------------------
    # 2-site local update
    # ------------------------------------------------------------------

    def _update_2site(
        self, p: int, dt: complex,
        max_dim: int | None, cutoff: float, absorb: str,
    ) -> float:
        """2-site TDVP local update at sites p, p+1.

        absorb='right': SVD → psi[p] left-ortho, centre at p+1.
        absorb='left' : SVD → psi[p+1] right-ortho, centre at p.

        Backward propagation on the resulting centre tensor (1-site),
        except at sweep boundaries.

        Returns truncation error from SVD.
        """
        psi = self.psi
        N   = len(psi)
        _i0 = MPS._phi_label(0)

        # ── 1. Prepare environments ────────────────────────────────────
        # Need op_env[p-1] (left) and op_env[p+2] (right).
        self._op_env.update_envs(p, p + 1)

        # ── 2. Forward propagation ─────────────────────────────────────
        # phi = exp(-dt/2 * H_eff_2site) |psi[p:p+2]>
        effH = EffOperator(
            self._op_env[p - 1], self._op_env[p + 2],
            self.H[p], self.H[p + 1],
        )
        phi   = psi.make_phi(p, 2)
        phi   = lanczos_expm_multiply(effH.apply, phi, -0.5 * dt)

        # ── 3. SVD + update psi ────────────────────────────────────────
        # update_sites fires observers and sets center.
        trunc = psi.update_sites(p, phi, max_dim=max_dim, cutoff=cutoff,
                                 absorb=absorb)

        # ── Boundary: no backward propagation ─────────────────────────
        is_boundary = (absorb == "right" and p == N - 2) or \
                      (absorb == "left"  and p == 0)
        if is_boundary:
            return trunc

        # ── 4. Backward propagation (1-site) ──────────────────────────
        # After SVD with absorb="right": centre is psi[p+1] (=SV†).
        # After SVD with absorb="left" : centre is psi[p]   (=US).
        # Evolve the centre tensor with 1-site H_eff.
        if absorb == "right":
            q_back = p + 1
            # Recompute op_env[p] (left env) with newly updated psi[p].
            # Stale window after update_sites fires: includes p and p+1.
            # update_envs(p+1, p+1) → computes LR[p], window → [p+1,p+1].
            self._op_env.update_envs(p + 1, p + 1)
            effH_1 = EffOperator(
                self._op_env[p], self._op_env[p + 2], self.H[p + 1]
            )
        else:  # absorb == "left"
            q_back = p
            # Recompute op_env[p+1] (right env) with newly updated psi[p+1].
            # update_envs(p, p) → computes LR[p+1], window → [p,p].
            self._op_env.update_envs(p, p)
            effH_1 = EffOperator(
                self._op_env[p - 1], self._op_env[p + 1], self.H[p]
            )

        phi_back = psi.make_phi(q_back, 1)                    # ["l","i0","r"]
        phi_back = lanczos_expm_multiply(effH_1.apply, phi_back, +0.5 * dt)
        phi_back.relabels_([_i0], ["i"])
        psi[q_back] = phi_back
        psi.center_left = psi.center_right = q_back

        return trunc
