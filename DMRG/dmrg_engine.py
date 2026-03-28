"""DMRGEngine: iterative ground-state (and excited-state) DMRG sweeper.

Usage
-----
    engine = DMRGEngine(psi, H)
    for max_dim, cutoff in zip([20, 50, 100], [0.0, 0.0, 1e-8]):
        E, trunc = engine.sweep(max_dim, cutoff)

psi is modified in-place.  Environments are built once in __init__ and
reused across sweeps — rebuilding is expensive and unnecessary.

Excited states
--------------
    engine = DMRGEngine(psi, H,
                        ortho_states=[psi0],
                        ortho_weights=[100.0])

A penalty term  w * |Φ_k><Φ_k|  is added to H_eff for each reference
state, pushing the optimisation away from previously found eigenstates.
"""

from __future__ import annotations

import cytnx

from MPS.mps import MPS
from MPS.linalg import lanczos
from DMRG.environment import OperatorEnv, VectorEnv
from DMRG.effective_operators import EffOperator, EffVector


class DMRGEngine:
    """DMRG sweep engine for a single MPS/MPO pair.

    Parameters
    ----------
    psi           : MPS — state to optimise (modified in-place).
                    Must have center == 0 (right-canonical form).
    H             : MPO — Hamiltonian.
    ortho_states  : list[MPS] | None — previously found eigenstates.
    ortho_weights : list[float] | None — penalty weights (one per ortho state).
    """

    def __init__(
        self,
        psi: MPS,
        H,
        ortho_states: list | None = None,
        ortho_weights: list | None = None,
    ) -> None:
        if psi.center != 0:
            raise ValueError(
                f"psi.center must be 0 (right-canonical); got {psi.center}."
            )

        self.psi = psi
        self.H   = H
        self._ortho_states  = ortho_states  or []
        self._ortho_weights = ortho_weights or []

        if len(self._ortho_states) != len(self._ortho_weights):
            raise ValueError(
                "ortho_states and ortho_weights must have the same length."
            )

        # Build operator environment once; reused across sweeps.
        # LREnv.__init__ calls update_envs(0, 0) internally, which builds
        # all right environments before returning.
        self._op_env = OperatorEnv(psi, psi, H, init_center=0)

        # One VectorEnv per orthogonal reference state.
        self._vec_envs = [
            VectorEnv(psi, s, init_center=0)
            for s in self._ortho_states
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def sweep(
        self,
        max_dim: int | None = None,
        cutoff: float = 0.0,
        num_center: int = 2,
    ) -> tuple[float, float]:
        """Perform one full sweep (right then left).

        Parameters
        ----------
        max_dim    : int | None — max bond dimension to keep.
        cutoff     : float — cutoff on normalized rho eigenvalues at each
                     local split.
        num_center : 1 or 2 — single-site or two-site DMRG.

        Returns
        -------
        E     : float — energy after the sweep (last local optimisation).
        trunc : float — average truncation error over all local steps.
        """
        if num_center not in (1, 2):
            raise ValueError("num_center must be 1 or 2.")
        if self.psi.center != 0:
            raise ValueError(
                f"psi.center must be 0 at the start of each sweep; "
                f"got {self.psi.center}."
            )

        N = len(self.psi)
        n = num_center
        energies = []
        truncs   = []

        def record(E, trunc):
            energies.append(E)
            truncs.append(trunc)

        # ── sweep right: p = 0 … N-2 (both 1-site and 2-site) ───────
        for p in range(N - 1):
            record(*self._local_update(p, n, max_dim, cutoff, absorb="right"))
        # after right sweep: center = N-1

        # ── sweep left ───────────────────────────────────────────────
        # 2-site: p = N-2 … 0   (absorb="left" at p=0 safe: touches only p, p+1)
        # 1-site: p = N-1 … 1   (absorb="left" at p=1 absorbs into psi[0])
        left_start = N - n   # 2-site: N-2,  1-site: N-1
        left_stop  = 0 if n == 2 else 1
        for p in range(left_start, left_stop - 1, -1):
            record(*self._local_update(p, n, max_dim, cutoff, absorb="left"))

        avg_trunc = sum(truncs) / len(truncs) if truncs else 0.0
        return energies[-1], avg_trunc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _local_update(self, p, num_center, max_dim, cutoff, absorb):
        """Optimise the local subspace at site(s) p..p+num_center-1."""
        psi = self.psi
        N   = len(psi)

        # Prepare environments: need op_env[p-1] and op_env[p+num_center].
        self._op_env.update_envs(p, p + num_center - 1)
        for vec_env in self._vec_envs:
            vec_env.update_envs(p, p + num_center - 1)

        # Build effective Hamiltonian.
        mpo_tensors = [self.H[p + k] for k in range(num_center)]
        effH = EffOperator(
            self._op_env[p - 1],
            self._op_env[p + num_center],
            *mpo_tensors,
        )

        # Add excited-state penalty terms.
        for vec_env, ortho_state, weight in zip(self._vec_envs,
                                                 self._ortho_states,
                                                 self._ortho_weights):
            mps_tensors = [ortho_state[p + k] for k in range(num_center)]
            eff_vec = EffVector(
                vec_env[p - 1],
                vec_env[p + num_center],
                *mps_tensors,
            )
            effH.add_term(eff_vec, weight)

        # Merge site tensors into φ, optimise with Lanczos, split back.
        phi = psi.make_phi(p, num_center)
        E, phi = lanczos(effH.apply, phi)
        trunc = psi.update_sites(p, phi, max_dim=max_dim, cutoff=cutoff,
                                 absorb=absorb)
        return float(E), float(trunc)
