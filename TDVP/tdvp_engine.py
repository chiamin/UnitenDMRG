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

import sys

import cytnx

from MPS.mps import MPS
from MPS.linalg import lanczos_expm_multiply
from MPS.uniTensor_core import svd_by_labels
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

        absorb='right': forward + SVD + backward + absorb into p+1
        absorb='left' : forward + SVD + backward + absorb into p-1

        Boundary sites (p=N-1 for right, p=0 for left) only do forward.

        Steps:
            1. Forward propagation: phi = exp(-dt/2 * H_eff) |psi[p]>
            2. SVD: split phi into A (isometry) and C (bond tensor)
            3. Build 0-site env from op_env and A (without updating psi)
            4. Backward propagation: C' = exp(+dt/2 * H_eff_0site) |C>
            5. Absorb C' into neighbour, update both psi[p] and neighbour
        """
        psi = self.psi
        N   = len(psi)
        _i0 = MPS._phi_label(0)

        # ── 1. Prepare environments ────────────────────────────────────
        self._op_env.update_envs(p, p)

        # ── 2. Forward propagation ─────────────────────────────────────
        effH = EffOperator(self._op_env[p - 1], self._op_env[p + 1], self.H[p])
        phi  = psi.make_phi(p, 1)
        phi  = lanczos_expm_multiply(effH.apply, phi, -0.5 * dt)

        # ── Boundary: no backward propagation ─────────────────────────
        is_boundary = (absorb == "right" and p == N - 1) or \
                      (absorb == "left"  and p == 0)
        if is_boundary:
            phi.relabels_([_i0], ["i"])
            psi[p] = phi
            psi.center_left = psi.center_right = p
            return

        # ── 3. SVD decompose ───────────────────────────────────────────
        if absorb == "right":
            # row=["l","i0"] (all BD_IN), col=["r"]
            A, C, _ = svd_by_labels(
                phi, row_labels=["l", _i0], col_labels=["r"],
                absorb="right", aux_label="_s",
            )
            A.relabels_(["l", _i0, "_s"], ["l", "i", "r"])
            # C: (_s, r) — bond tensor with labels ["_s", "r"]
        else:
            # row=["l"] (all BD_IN), col=["i0","r"]
            C, A, _ = svd_by_labels(
                phi, row_labels=["l"], col_labels=[_i0, "r"],
                absorb="left", aux_label="_s",
            )
            A.relabels_(["_s", _i0, "r"], ["l", "i", "r"])
            # C: (l, _s) — bond tensor with labels ["l", "_s"]

        # ── 4. Build 0-site env from op_env and A ──────────────────────
        # Grow the environment by one site using A, without modifying
        # self._op_env.  This gives us the 0-site left/right env pair.
        if absorb == "right":
            # Grow left env: op_env[p-1] + A + W[p] + A† → new_L (= env at p)
            L_prev = self._op_env[p - 1]
            E  = L_prev.relabels(["mid", "dn", "up"], ["_mid", "_dn", "_up"])
            A1 = A.relabels(["l", "i", "r"], ["_dn", "_i", "dn"])
            W  = self.H[p].relabels(["l", "ip", "i", "r"], ["_mid", "_ip", "_i", "mid"])
            A2 = A.Dagger().relabels(["l", "i", "r"], ["_up", "_ip", "up"])
            tmp = cytnx.Contract(E, A1)
            tmp = cytnx.Contract(tmp, W)
            new_L = cytnx.Contract(tmp, A2)
            effH_0 = EffOperator(new_L, self._op_env[p + 1])
        else:
            # Grow right env: op_env[p+1] + A + W[p] + A† → new_R (= env at p)
            R_next = self._op_env[p + 1]
            E  = R_next.relabels(["mid", "dn", "up"], ["_mid", "_dn", "_up"])
            A1 = A.relabels(["r", "i", "l"], ["_dn", "_i", "dn"])
            W  = self.H[p].relabels(["l", "ip", "i", "r"], ["mid", "_ip", "_i", "_mid"])
            A2 = A.Dagger().relabels(["r", "i", "l"], ["_up", "_ip", "up"])
            tmp = cytnx.Contract(E, A1)
            tmp = cytnx.Contract(tmp, W)
            new_R = cytnx.Contract(tmp, A2)
            effH_0 = EffOperator(self._op_env[p - 1], new_R)

        # ── 5. Backward propagation ────────────────────────────────────
        # Rename C to canonical ["l","r"] for EffOperator.apply()
        if absorb == "right":
            C.relabels_(["_s", "r"], ["l", "r"])
        else:
            C.relabels_(["l", "_s"], ["l", "r"])

        C = lanczos_expm_multiply(effH_0.apply, C, +0.5 * dt)

        # ── 6. Absorb C into neighbour and update psi ──────────────────
        if absorb == "right":
            C.relabels_(["l", "r"], ["_new_l", "l"])
            new_nb = cytnx.Contract(C, psi[p + 1])
            new_nb.relabels_(["_new_l", "i", "r"], ["l", "i", "r"])
            psi.set_sites({p: A, p + 1: new_nb})
            psi.center_left = psi.center_right = p + 1
        else:
            C.relabels_(["l", "r"], ["r", "_new_r"])
            new_nb = cytnx.Contract(psi[p - 1], C)
            new_nb.relabels_(["l", "i", "_new_r"], ["l", "i", "r"])
            psi.set_sites({p - 1: new_nb, p: A})
            psi.center_left = psi.center_right = p - 1

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
