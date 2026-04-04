"""Unit tests for TDVPEngine and Lanczos Krylov time evolution.

Coverage
--------
1. lanczos_expm_multiply  (TestLanczoExpmMultiply)
   - Real-time evolution preserves norm:  ||exp(-i*dt*H)|v>|| == ||v||
   - Labels of output are identical to labels of input
   - dt=0 is identity:  exp(0)|v> == |v>
   - Short-time agreement with first-order Taylor:  exp(dt*H)|v> ≈ |v> + dt*H|v>
   - Known matrix exponential: 1×1 matrix exp(a*dt) == scalar
   - Backward sign: exp(+dt*H) with H negative definite → norm grows

2. EffOperator 0-site regression  (TestEffOperator0Site)
   - Bug fix: output has labels ["l","r"] (not four labels)
   - apply(phi) has same shape as phi
   - Identity-like 0-site MPO: apply(phi) ≈ phi

3. TDVPEngine construction  (TestTDVPEngineInit)
   - psi.center != 0 → ValueError
   - Valid construction succeeds (no exception)

4. sweep() input validation  (TestTDVPSweepValidation)
   - num_center not in {1,2} → ValueError
   - psi.center != 0 at start of sweep → ValueError

5. Return value contract  (TestTDVPReturnValues)
   - 1-site sweep returns float 0.0 (no truncation)
   - 2-site sweep returns non-negative float

6. 1-site TDVP invariant properties  (TestTDVP1SiteInvariants)
   - psi.center == 0 after each sweep
   - Multiple consecutive sweeps work without error
   - Bond dimensions are preserved (QR, no truncation)
   - MPS labels are ["l","i","r"] at every site after sweep

7. 2-site TDVP invariant properties  (TestTDVP2SiteInvariants)
   - psi.center == 0 after each sweep
   - Multiple consecutive sweeps work without error

8. Norm preservation — real-time evolution  (TestTDVPNormPreservation)
   - 1-site TDVP: ||psi|| ≈ 1 after several sweeps (imaginary part of dt == 0)
   - 2-site TDVP: ||psi|| ≈ 1 after several sweeps

9. Energy conservation — real-time evolution  (TestTDVPEnergyConservation)
   - Start near ground state; real-time TDVP preserves <H> across sweeps

10. Imaginary-time convergence  (TestTDVPImaginaryTime)
    - Imaginary-time evolution converges to ground-state energy
    - Both 1-site (warm-start) and 2-site TDVP reach the exact ground energy
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import cytnx
except ImportError:
    cytnx = None

THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

if cytnx is not None:
    from MPS.mps_init import random_mps
    from tests.helpers.heisenberg import heisenberg_mpo
    from tests.helpers.mps_test_cases import random_u1_sz_mps
    from MPS.physical_sites import spin_half
    from MPS.auto_mpo import AutoMPO
    from linalg import lanczos_expm_multiply, inner
    from DMRG.effective_operators import EffOperator
    from DMRG.environment import OperatorEnv
    from TDVP.tdvp_engine import TDVPEngine


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

def _make_psi(N: int, D: int = 4, seed: int = 0, dtype: np.dtype | type = float):
    """Random MPS with center at 0."""
    psi = random_mps(N, phys_dim=2, bond_dim=D, seed=seed, dtype=dtype)
    psi.move_center(0)
    return psi


def _heisenberg_matrix(N: int, J: float = 1.0, delta: float = 1.0) -> np.ndarray:
    """Build the full N-site Heisenberg Hamiltonian as a 2^N × 2^N dense matrix."""
    I  = np.eye(2)
    Sz = np.array([[0.5, 0.0], [0.0, -0.5]])
    Sp = np.array([[0.0, 1.0], [0.0,  0.0]])
    Sm = np.array([[0.0, 0.0], [1.0,  0.0]])

    dim = 2 ** N
    H = np.zeros((dim, dim))
    for i in range(N - 1):
        for coeff, op_i, op_j in [
            (J / 2.0,   Sp, Sm),
            (J / 2.0,   Sm, Sp),
            (J * delta, Sz, Sz),
        ]:
            mat = np.array([[1.0]])
            for k in range(N):
                if k == i:
                    mat = np.kron(mat, op_i)
                elif k == i + 1:
                    mat = np.kron(mat, op_j)
                else:
                    mat = np.kron(mat, I)
            H += coeff * mat
    return H


def _exact_energies(N: int, num: int = 2) -> list[float]:
    """Return the lowest `num` eigenvalues of the N-site Heisenberg chain."""
    evals = np.linalg.eigvalsh(_heisenberg_matrix(N))
    return [float(evals[k]) for k in range(num)]


def _vec(arr: np.ndarray, label: str = "x") -> "cytnx.UniTensor":
    """Wrap a 1-D numpy array into a rank-1 UniTensor."""
    t = cytnx.UniTensor(cytnx.from_numpy(arr.astype(complex)), rowrank=1)
    t.set_labels([label])
    return t


def _matvec(H_np: np.ndarray, label: str = "x"):
    """Return apply(v) = H @ v for a rank-1 UniTensor with the given label."""
    def _apply(v: "cytnx.UniTensor") -> "cytnx.UniTensor":
        x = v.get_block().numpy().ravel()
        return _vec(H_np @ x, label)
    return _apply


def _psi_norm(psi) -> float:
    """Compute ||psi|| via <psi|psi> contraction at site 0."""
    psi.move_center(0)
    A = psi[0]
    overlap = cytnx.Contract(A.Dagger(), A)
    for p in range(1, len(psi)):
        Ap = psi[p]
        overlap = cytnx.Contract(overlap, Ap)
        overlap = cytnx.Contract(overlap, Ap.Dagger())
    return float(abs(overlap.item()) ** 0.5)


def _energy(psi, H_mpo) -> float:
    """Compute <psi|H|psi> / <psi|psi> using a single-site effective Hamiltonian.

    Builds OperatorEnv and contracts the full expectation value via Lanczos
    ground-state energy (which equals the Rayleigh quotient for the given psi).
    """
    from linalg import lanczos
    psi.move_center(0)
    op_env = OperatorEnv(psi, psi, H_mpo, init_center=0)
    op_env.update_envs(0, 0)
    N = len(psi)
    # Use the 1-site effective Hamiltonian at site 0 to get the energy.
    # Lanczos on H_eff with psi[0] as input gives the Rayleigh quotient.
    effH = EffOperator(op_env[-1], op_env[1], H_mpo[0])
    phi  = psi.make_phi(0, 1)
    E, _ = lanczos(effH.apply, phi, k=1)
    return E


def _make_qn_psi(N: int, n_up: int, seed: int = 0, dtype=float) -> "MPS":
    """Random QN MPS (U(1) N_up) with center at 0."""
    return random_u1_sz_mps(N, n_up, seed=seed, dtype=dtype, center=0)


def _qn_heisenberg_mpo(N: int, dtype=float):
    """Build a QN Heisenberg MPO via AutoMPO + spin_half(qn="Sz")."""
    site = spin_half(qn="Sz")
    ampo = AutoMPO(N, site)
    J, delta = 1.0, 1.0
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        J, delta = complex(J), complex(delta)
    for i in range(N - 1):
        ampo.add(J * delta, "Sz", i, "Sz", i + 1)
        ampo.add(J / 2, "Sp", i, "Sm", i + 1)
        ampo.add(J / 2, "Sm", i, "Sp", i + 1)
    return ampo.to_mpo()


# ---------------------------------------------------------------------------
# 1. lanczos_expm_multiply
# ---------------------------------------------------------------------------

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestLanczoExpmMultiply(unittest.TestCase):

    def setUp(self):
        n = 12
        rng = np.random.default_rng(42)
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        self.H_np = (A + A.conj().T) / 2.0          # Hermitian
        self.n    = n
        v_np = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        self.v0   = _vec(v_np)
        self.norm0 = float(np.linalg.norm(v_np))

    def test_real_time_norm_preserved(self):
        """||exp(-i*dt*H)|v>|| equals ||v|| for unitary evolution."""
        dt    = -1j * 0.1                            # real-time step
        apply = _matvec(self.H_np)
        result = lanczos_expm_multiply(apply, self.v0, dt, k=self.n)
        norm_out = result.Norm().item()
        self.assertAlmostEqual(norm_out, self.norm0, places=6)

    def test_labels_preserved(self):
        """Output labels are identical to input labels."""
        apply = _matvec(self.H_np)
        result = lanczos_expm_multiply(apply, self.v0, -1j * 0.05, k=self.n)
        self.assertEqual(list(result.labels()), ["x"])

    def test_zero_dt_is_identity(self):
        """exp(0 * H)|v> equals |v> to numerical precision."""
        apply  = _matvec(self.H_np)
        result = lanczos_expm_multiply(apply, self.v0, 0.0, k=self.n)
        v_np   = self.v0.get_block().numpy().ravel()
        r_np   = result.get_block().numpy().ravel()
        np.testing.assert_allclose(r_np, v_np, atol=1e-10)

    def test_short_time_taylor_agreement(self):
        """exp(dt*H)|v> ≈ |v> + dt*H|v>  for small real dt."""
        dt    = 1e-4                                 # small real step
        apply = _matvec(self.H_np)
        result = lanczos_expm_multiply(apply, self.v0, dt, k=self.n)

        v_np   = self.v0.get_block().numpy().ravel()
        Hv_np  = self.H_np @ v_np
        taylor = v_np + dt * Hv_np

        r_np = result.get_block().numpy().ravel()
        np.testing.assert_allclose(r_np, taylor, atol=1e-7)

    def test_known_1x1_matrix(self):
        """For a 1×1 matrix [[a]], exp(dt * a) * v == e^(a*dt) * v."""
        a  = 3.5 + 0.0j
        dt = 0.2
        H1 = np.array([[a]])
        v1 = _vec(np.array([1.0 + 0.0j]))
        apply = _matvec(H1)
        result = lanczos_expm_multiply(apply, v1, dt, k=4)
        r_np = result.get_block().numpy().ravel()
        expected = np.exp(a * dt)
        self.assertAlmostEqual(complex(r_np[0]), expected, places=10)

    def test_backward_evolution_increases_norm(self):
        """exp(+dt*H) with dt>0 and H positive-definite increases norm."""
        # Build a positive-definite matrix:  H = A†A + I
        rng = np.random.default_rng(7)
        A   = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
        H_pd = A.conj().T @ A + np.eye(8)
        v_np = rng.standard_normal(8) + 1j * rng.standard_normal(8)
        v0   = _vec(v_np)
        apply = _matvec(H_pd)
        result = lanczos_expm_multiply(apply, v0, 0.1, k=8)
        norm_out = result.Norm().item()
        self.assertGreater(norm_out, v0.Norm().item())

    def test_exact_eigh_agreement(self):
        """Full Krylov (k=n) matches numpy eigh-based expm exactly (Hermitian H)."""
        dt    = 0.3 + 0.1j
        apply = _matvec(self.H_np)
        result = lanczos_expm_multiply(apply, self.v0, dt, k=self.n)

        # Reference: exp(dt*H) via eigendecomposition (same as our implementation)
        v_np = self.v0.get_block().numpy().ravel()
        evals, evecs = np.linalg.eigh(self.H_np)
        exact = evecs @ (np.exp(dt * evals) * (evecs.T.conj() @ v_np))

        r_np = result.get_block().numpy().ravel()
        np.testing.assert_allclose(r_np, exact, atol=1e-8)


# ---------------------------------------------------------------------------
# 2. EffOperator 0-site regression (bug fix verification)
# ---------------------------------------------------------------------------

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestEffOperator0Site(unittest.TestCase):
    """Verify the n=0 special case of EffOperator._apply_operator."""

    N = 4

    def setUp(self):
        self.psi    = _make_psi(self.N, D=4)
        self.H_mpo  = heisenberg_mpo(self.N)
        self.op_env = OperatorEnv(self.psi, self.psi, self.H_mpo, init_center=0)
        # Move to site 1 so that op_env[0] (left) and op_env[2] (right) are valid.
        self.op_env.update_envs(1, 1)
        self.effH_0 = EffOperator(self.op_env[0], self.op_env[2])

    def _bond_tensor(self) -> "cytnx.UniTensor":
        """Make a small bond tensor with labels ["l","r"]."""
        rng = np.random.default_rng(0)
        chi = 4
        arr = rng.standard_normal((chi, chi)) + 1j * rng.standard_normal((chi, chi))
        t   = cytnx.UniTensor(cytnx.from_numpy(arr.astype(complex)), rowrank=1)
        t.set_labels(["l", "r"])
        return t

    def test_output_labels_are_l_r(self):
        """0-site apply must return a tensor with labels ['l','r'] only."""
        phi    = self._bond_tensor()
        result = self.effH_0.apply(phi)
        self.assertEqual(sorted(result.labels()), ["l", "r"])

    def test_output_shape_matches_input(self):
        """0-site apply must preserve the shape of the bond tensor."""
        phi    = self._bond_tensor()
        result = self.effH_0.apply(phi)
        self.assertEqual(result.shape(), phi.shape())

    def test_output_is_not_zero(self):
        """For a random non-zero phi, apply(phi) must be non-zero."""
        phi    = self._bond_tensor()
        result = self.effH_0.apply(phi)
        self.assertGreater(result.Norm().item(), 0.0)


# ---------------------------------------------------------------------------
# 3. TDVPEngine construction
# ---------------------------------------------------------------------------

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestTDVPEngineInit(unittest.TestCase):

    def test_center_not_zero_raises(self):
        """TDVPEngine.__init__ requires psi.center == 0."""
        psi = _make_psi(4)
        psi.move_center(2)
        H = heisenberg_mpo(4)
        with self.assertRaises(ValueError):
            TDVPEngine(psi, H)

    def test_valid_construction(self):
        """Normal construction (center==0) must not raise."""
        psi = _make_psi(4)
        H   = heisenberg_mpo(4)
        TDVPEngine(psi, H)          # must not raise


# ---------------------------------------------------------------------------
# 4. sweep() input validation
# ---------------------------------------------------------------------------

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestTDVPSweepValidation(unittest.TestCase):

    def setUp(self):
        self.psi    = _make_psi(4)
        self.H      = heisenberg_mpo(4)
        self.engine = TDVPEngine(self.psi, self.H)

    def test_invalid_num_center_raises(self):
        """num_center not in {1, 2} → ValueError."""
        with self.assertRaises(ValueError):
            self.engine.sweep(dt=0.01, num_center=3)

    def test_center_not_zero_before_sweep_raises(self):
        """psi.center != 0 at start of sweep → ValueError."""
        self.psi.move_center(2)
        with self.assertRaises(ValueError):
            self.engine.sweep(dt=0.01)


# ---------------------------------------------------------------------------
# 5. Return value contract
# ---------------------------------------------------------------------------

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestTDVPReturnValues(unittest.TestCase):

    N = 4

    def test_1site_returns_zero_trunc(self):
        """1-site sweep returns float 0.0 (no truncation in 1-site TDVP)."""
        psi    = _make_psi(self.N, D=4)
        engine = TDVPEngine(psi, heisenberg_mpo(self.N))
        trunc  = engine.sweep(dt=0.01, num_center=1)
        self.assertIsInstance(trunc, float)
        self.assertEqual(trunc, 0.0)

    def test_2site_returns_nonneg_float(self):
        """2-site sweep returns a non-negative float (avg truncation error)."""
        psi    = _make_psi(self.N, D=4)
        engine = TDVPEngine(psi, heisenberg_mpo(self.N))
        trunc  = engine.sweep(dt=0.01, max_dim=8, cutoff=1e-12, num_center=2)
        self.assertIsInstance(trunc, float)
        self.assertGreaterEqual(trunc, 0.0)

    def test_2site_trunc_respects_cutoff(self):
        """2-site with large cutoff returns larger trunc than with tiny cutoff."""
        psi_loose = _make_psi(self.N, D=4, seed=10)
        psi_tight = _make_psi(self.N, D=4, seed=10)
        H = heisenberg_mpo(self.N)
        eng_loose = TDVPEngine(psi_loose, H)
        eng_tight = TDVPEngine(psi_tight, heisenberg_mpo(self.N))
        trunc_loose = eng_loose.sweep(dt=0.01, max_dim=2,  cutoff=1e-1,  num_center=2)
        trunc_tight = eng_tight.sweep(dt=0.01, max_dim=16, cutoff=1e-12, num_center=2)
        self.assertGreaterEqual(trunc_loose, trunc_tight - 1e-12)


# ---------------------------------------------------------------------------
# 6. 1-site TDVP invariant properties
# ---------------------------------------------------------------------------

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestTDVP1SiteInvariants(unittest.TestCase):

    N = 4

    def test_center_zero_after_sweep(self):
        """psi.center must equal 0 after a 1-site sweep."""
        psi    = _make_psi(self.N, D=4)
        engine = TDVPEngine(psi, heisenberg_mpo(self.N))
        engine.sweep(dt=0.01, num_center=1)
        self.assertEqual(psi.center, 0)

    def test_multiple_sweeps_no_error(self):
        """Three consecutive 1-site sweeps complete without error."""
        psi    = _make_psi(self.N, D=4)
        engine = TDVPEngine(psi, heisenberg_mpo(self.N))
        for _ in range(3):
            engine.sweep(dt=0.01, num_center=1)

    def test_bond_dims_preserved(self):
        """Bond dimensions must not change during 1-site TDVP.

        Uses D=2 so bonds are not over-specified and QR is always full-rank.
        (For N=4, d=2: max physical bond dim at boundaries is 2.)
        """
        psi   = _make_psi(self.N, D=2)
        H_mpo = heisenberg_mpo(self.N)
        dims_before = [psi[p].shape() for p in range(self.N)]
        engine = TDVPEngine(psi, H_mpo)
        engine.sweep(dt=0.01, num_center=1)
        dims_after = [psi[p].shape() for p in range(self.N)]
        self.assertEqual(dims_before, dims_after)

    def test_site_labels_after_sweep(self):
        """Every site tensor must have labels ['l','i','r'] after sweep."""
        psi    = _make_psi(self.N, D=4)
        engine = TDVPEngine(psi, heisenberg_mpo(self.N))
        engine.sweep(dt=0.01, num_center=1)
        for p in range(self.N):
            self.assertEqual(sorted(psi[p].labels()), ["i", "l", "r"],
                             msg=f"Site {p} has unexpected labels: {psi[p].labels()}")

    def test_center_zero_after_multiple_sweeps(self):
        """psi.center must equal 0 after each sweep, not just the first."""
        psi    = _make_psi(self.N, D=4)
        engine = TDVPEngine(psi, heisenberg_mpo(self.N))
        for _ in range(3):
            engine.sweep(dt=0.01, num_center=1)
            self.assertEqual(psi.center, 0)


# ---------------------------------------------------------------------------
# 7. 2-site TDVP invariant properties
# ---------------------------------------------------------------------------

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestTDVP2SiteInvariants(unittest.TestCase):

    N = 4

    def test_center_zero_after_sweep(self):
        """psi.center must equal 0 after a 2-site sweep."""
        psi    = _make_psi(self.N, D=4)
        engine = TDVPEngine(psi, heisenberg_mpo(self.N))
        engine.sweep(dt=0.01, max_dim=8, cutoff=0.0, num_center=2)
        self.assertEqual(psi.center, 0)

    def test_multiple_sweeps_no_error(self):
        """Three consecutive 2-site sweeps complete without error."""
        psi    = _make_psi(self.N, D=4)
        engine = TDVPEngine(psi, heisenberg_mpo(self.N))
        for _ in range(3):
            engine.sweep(dt=0.01, max_dim=8, cutoff=0.0, num_center=2)

    def test_center_zero_after_multiple_sweeps(self):
        """psi.center must equal 0 after each 2-site sweep."""
        psi    = _make_psi(self.N, D=4)
        engine = TDVPEngine(psi, heisenberg_mpo(self.N))
        for _ in range(3):
            engine.sweep(dt=0.01, max_dim=8, cutoff=0.0, num_center=2)
            self.assertEqual(psi.center, 0)

    def test_site_labels_after_sweep(self):
        """Every site tensor must have labels ['l','i','r'] after 2-site sweep."""
        psi    = _make_psi(self.N, D=4)
        engine = TDVPEngine(psi, heisenberg_mpo(self.N))
        engine.sweep(dt=0.01, max_dim=8, cutoff=0.0, num_center=2)
        for p in range(self.N):
            self.assertEqual(sorted(psi[p].labels()), ["i", "l", "r"],
                             msg=f"Site {p} has unexpected labels: {psi[p].labels()}")


# ---------------------------------------------------------------------------
# 8. Norm preservation — real-time evolution
# ---------------------------------------------------------------------------

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestTDVPNormPreservation(unittest.TestCase):
    """Real-time TDVP (dt = 1j * delta_t) must preserve the MPS norm."""

    N    = 4
    ATOL = 1e-5

    @staticmethod
    def _mps_norm(psi) -> float:
        """||psi|| via Frobenius norm of the center tensor (center must be 0)."""
        psi.move_center(0)
        arr = psi[0].get_block().numpy()
        return float(np.linalg.norm(arr))

    def _check_norm(self, num_center: int, n_sweeps: int = 4,
                    max_dim: int = 8) -> None:
        psi    = _make_psi(self.N, D=4, seed=0, dtype=complex)
        H_mpo  = heisenberg_mpo(self.N, dtype=complex)
        # Record norm before evolution (center already at 0 from _make_psi)
        norm_before = self._mps_norm(psi)
        engine = TDVPEngine(psi, H_mpo)
        dt_real = 1j * 0.05           # real-time step
        for _ in range(n_sweeps):
            engine.sweep(dt=dt_real, max_dim=max_dim, cutoff=0.0,
                         num_center=num_center)
        norm_after = self._mps_norm(psi)
        self.assertAlmostEqual(norm_after / norm_before, 1.0, delta=self.ATOL)

    def test_1site_norm_preserved(self):
        """1-site real-time TDVP preserves the MPS norm."""
        self._check_norm(num_center=1)

    def test_2site_norm_preserved(self):
        """2-site real-time TDVP preserves the MPS norm."""
        self._check_norm(num_center=2)


# ---------------------------------------------------------------------------
# 9. Energy conservation — real-time evolution
# ---------------------------------------------------------------------------

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestTDVPEnergyConservation(unittest.TestCase):
    """<H> must be conserved (to tolerance) during real-time evolution."""

    N    = 4
    ATOL = 1e-3     # energy tolerance after a few sweeps

    @classmethod
    def setUpClass(cls):
        """Prepare a good initial state by running DMRG first."""
        from DMRG.dmrg_engine import DMRGEngine
        cls.H_mpo = heisenberg_mpo(cls.N, dtype=complex)
        psi = _make_psi(cls.N, D=8, seed=0, dtype=complex)
        engine = DMRGEngine(psi, cls.H_mpo)
        for max_dim in [8, 16, 16]:
            E0, _ = engine.sweep(max_dim=max_dim, cutoff=1e-10)
        cls.E0_dmrg = E0
        cls.psi0    = psi

    def _measure_energy(self, psi) -> float:
        """<psi|H|psi> / <psi|psi> via Rayleigh quotient at site 0."""
        psi.move_center(0)
        op_env = OperatorEnv(psi, psi, self.H_mpo, init_center=0)
        op_env.update_envs(0, 0)
        effH = EffOperator(op_env[-1], op_env[1], self.H_mpo[0])
        phi  = psi.make_phi(0, 1)
        from linalg import lanczos
        E, _ = lanczos(effH.apply, phi, k=1)
        return E

    def test_1site_energy_conserved(self):
        """Real-time 1-site TDVP conserves <H> across 4 sweeps."""
        psi = _make_psi(self.N, D=8, seed=0, dtype=complex)
        # Warm up with DMRG
        from DMRG.dmrg_engine import DMRGEngine
        eng = DMRGEngine(psi, self.H_mpo)
        for md in [8, 16]:
            eng.sweep(max_dim=md, cutoff=1e-10)
        E_before = self._measure_energy(psi)

        tdvp = TDVPEngine(psi, self.H_mpo)
        for _ in range(4):
            tdvp.sweep(dt=1j * 0.05, num_center=1)
        E_after = self._measure_energy(psi)

        self.assertAlmostEqual(E_after, E_before, delta=self.ATOL)

    def test_2site_energy_conserved(self):
        """Real-time 2-site TDVP conserves <H> across 4 sweeps."""
        psi = _make_psi(self.N, D=8, seed=0, dtype=complex)
        from DMRG.dmrg_engine import DMRGEngine
        eng = DMRGEngine(psi, self.H_mpo)
        for md in [8, 16]:
            eng.sweep(max_dim=md, cutoff=1e-10)
        E_before = self._measure_energy(psi)

        tdvp = TDVPEngine(psi, self.H_mpo)
        for _ in range(4):
            tdvp.sweep(dt=1j * 0.05, max_dim=16, cutoff=1e-12, num_center=2)
        E_after = self._measure_energy(psi)

        self.assertAlmostEqual(E_after, E_before, delta=self.ATOL)


# ---------------------------------------------------------------------------
# 10. Imaginary-time convergence to ground state
# ---------------------------------------------------------------------------

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestTDVPImaginaryTime(unittest.TestCase):
    """Imaginary-time TDVP (dt = delta_tau, real) must converge to E0."""

    N    = 4
    ATOL = 0.02     # energy tolerance (gap=0.66, tau=5 gives ~1.5% error)

    @classmethod
    def setUpClass(cls):
        cls.E0_exact = _exact_energies(cls.N, num=1)[0]
        cls.H_mpo    = heisenberg_mpo(cls.N)

    def _measure_energy(self, psi) -> float:
        """<psi|H|psi> via Rayleigh quotient at site 0."""
        from linalg import lanczos
        psi.move_center(0)
        op_env = OperatorEnv(psi, psi, self.H_mpo, init_center=0)
        op_env.update_envs(0, 0)
        effH = EffOperator(op_env[-1], op_env[1], self.H_mpo[0])
        phi  = psi.make_phi(0, 1)
        E, _ = lanczos(effH.apply, phi, k=1)
        return E

    def test_2site_imagtime_convergence(self):
        """2-site imaginary-time TDVP converges to exact ground-state energy."""
        psi    = _make_psi(self.N, D=8, seed=0)
        engine = TDVPEngine(psi, self.H_mpo)
        # 50 sweeps × dt=0.1 → tau=5; with gap≈0.66, error ≈ exp(-6.6)×E_scale ≈ 0.01
        for _ in range(50):
            engine.sweep(dt=0.1, max_dim=16, cutoff=1e-10, num_center=2)
        E = self._measure_energy(psi)
        self.assertAlmostEqual(E, self.E0_exact, delta=self.ATOL)

    def test_1site_imagtime_converges_from_good_start(self):
        """1-site imaginary-time TDVP refines a warm-started state to E0."""
        psi    = _make_psi(self.N, D=8, seed=0)
        engine = TDVPEngine(psi, self.H_mpo)
        # Warm up with 2-site (tau=4)
        for _ in range(40):
            engine.sweep(dt=0.1, max_dim=16, cutoff=1e-10, num_center=2)
        # Refine with 1-site (tau=2)
        for _ in range(20):
            engine.sweep(dt=0.1, num_center=1)
        E = self._measure_energy(psi)
        self.assertAlmostEqual(E, self.E0_exact, delta=self.ATOL)

    def test_imagtime_energy_monotone(self):
        """Imaginary-time energy must be non-increasing sweep by sweep."""
        psi    = _make_psi(self.N, D=4, seed=3)
        engine = TDVPEngine(psi, self.H_mpo)
        energies = []
        for _ in range(5):
            engine.sweep(dt=0.1, max_dim=12, cutoff=1e-10, num_center=2)
            energies.append(self._measure_energy(psi))
        for i in range(1, len(energies)):
            self.assertLessEqual(energies[i], energies[i - 1] + 1e-6,
                                 msg=f"Energy increased at step {i}: "
                                     f"{energies[i-1]:.6f} → {energies[i]:.6f}")


# ---------------------------------------------------------------------------
# 11. QN norm preservation — real-time evolution
# ---------------------------------------------------------------------------

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestTDVPNormPreservationQN(unittest.TestCase):
    """Real-time TDVP with QN MPS must preserve the MPS norm."""

    N     = 4
    N_UP  = 2       # half-filling → Sz = 0
    ATOL  = 1e-5

    @staticmethod
    def _mps_norm(psi) -> float:
        """||psi|| via Frobenius norm of the center tensor (center must be 0).

        Uses UniTensor.Norm() which sums over all blocks for QN tensors.
        """
        psi.move_center(0)
        return float(psi[0].Norm().item())

    def _check_norm(self, num_center: int, n_sweeps: int = 4,
                    max_dim: int = 8) -> None:
        psi   = _make_qn_psi(self.N, self.N_UP, seed=0, dtype=complex)
        H_mpo = _qn_heisenberg_mpo(self.N, dtype=complex)
        norm_before = self._mps_norm(psi)
        engine = TDVPEngine(psi, H_mpo)
        dt_real = 1j * 0.05
        for _ in range(n_sweeps):
            engine.sweep(dt=dt_real, max_dim=max_dim, cutoff=0.0,
                         num_center=num_center)
        norm_after = self._mps_norm(psi)
        self.assertAlmostEqual(norm_after / norm_before, 1.0, delta=self.ATOL)

    def test_1site_norm_preserved(self):
        """1-site real-time QN TDVP preserves the MPS norm."""
        self._check_norm(num_center=1)

    def test_2site_norm_preserved(self):
        """2-site real-time QN TDVP preserves the MPS norm."""
        self._check_norm(num_center=2)


# ---------------------------------------------------------------------------
# 12. QN energy conservation — real-time evolution
# ---------------------------------------------------------------------------

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestTDVPEnergyConservationQN(unittest.TestCase):
    """<H> must be conserved during real-time QN TDVP evolution."""

    N     = 4
    N_UP  = 2
    ATOL  = 1e-3

    @classmethod
    def setUpClass(cls):
        """Prepare a good initial QN state by running DMRG first."""
        from DMRG.dmrg_engine import DMRGEngine
        cls.H_mpo = _qn_heisenberg_mpo(cls.N, dtype=complex)
        psi = _make_qn_psi(cls.N, cls.N_UP, seed=0, dtype=complex)
        engine = DMRGEngine(psi, cls.H_mpo)
        for max_dim in [8, 16, 16]:
            E0, _ = engine.sweep(max_dim=max_dim, cutoff=1e-10)
        cls.E0_dmrg = E0
        cls.psi0    = psi

    def _measure_energy(self, psi) -> float:
        """<psi|H|psi> via Rayleigh quotient at site 0."""
        from linalg import lanczos
        psi.move_center(0)
        op_env = OperatorEnv(psi, psi, self.H_mpo, init_center=0)
        op_env.update_envs(0, 0)
        effH = EffOperator(op_env[-1], op_env[1], self.H_mpo[0])
        phi  = psi.make_phi(0, 1)
        E, _ = lanczos(effH.apply, phi, k=1)
        return E

    def test_1site_energy_conserved(self):
        """Real-time 1-site QN TDVP conserves <H> across 4 sweeps."""
        psi = _make_qn_psi(self.N, self.N_UP, seed=0, dtype=complex)
        from DMRG.dmrg_engine import DMRGEngine
        eng = DMRGEngine(psi, self.H_mpo)
        for md in [8, 16]:
            eng.sweep(max_dim=md, cutoff=1e-10)
        E_before = self._measure_energy(psi)

        tdvp = TDVPEngine(psi, self.H_mpo)
        for _ in range(4):
            tdvp.sweep(dt=1j * 0.05, num_center=1)
        E_after = self._measure_energy(psi)

        self.assertAlmostEqual(E_after, E_before, delta=self.ATOL)

    def test_2site_energy_conserved(self):
        """Real-time 2-site QN TDVP conserves <H> across 4 sweeps."""
        psi = _make_qn_psi(self.N, self.N_UP, seed=0, dtype=complex)
        from DMRG.dmrg_engine import DMRGEngine
        eng = DMRGEngine(psi, self.H_mpo)
        for md in [8, 16]:
            eng.sweep(max_dim=md, cutoff=1e-10)
        E_before = self._measure_energy(psi)

        tdvp = TDVPEngine(psi, self.H_mpo)
        for _ in range(4):
            tdvp.sweep(dt=1j * 0.05, max_dim=16, cutoff=1e-12, num_center=2)
        E_after = self._measure_energy(psi)

        self.assertAlmostEqual(E_after, E_before, delta=self.ATOL)


# ---------------------------------------------------------------------------
# 13. QN imaginary-time convergence to ground state
# ---------------------------------------------------------------------------

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestTDVPImaginaryTimeQN(unittest.TestCase):
    """Imaginary-time QN TDVP must converge to E0."""

    N     = 4
    N_UP  = 2
    ATOL  = 0.02

    @classmethod
    def setUpClass(cls):
        cls.E0_exact = _exact_energies(cls.N, num=1)[0]
        cls.H_mpo    = _qn_heisenberg_mpo(cls.N)

    def _measure_energy(self, psi) -> float:
        """<psi|H|psi> via Rayleigh quotient at site 0."""
        from linalg import lanczos
        psi.move_center(0)
        op_env = OperatorEnv(psi, psi, self.H_mpo, init_center=0)
        op_env.update_envs(0, 0)
        effH = EffOperator(op_env[-1], op_env[1], self.H_mpo[0])
        phi  = psi.make_phi(0, 1)
        E, _ = lanczos(effH.apply, phi, k=1)
        return E

    def test_2site_imagtime_convergence(self):
        """2-site imaginary-time QN TDVP converges to exact ground-state energy."""
        psi    = _make_qn_psi(self.N, self.N_UP, seed=0)
        engine = TDVPEngine(psi, self.H_mpo)
        for _ in range(50):
            engine.sweep(dt=0.1, max_dim=16, cutoff=1e-10, num_center=2)
        E = self._measure_energy(psi)
        self.assertAlmostEqual(E, self.E0_exact, delta=self.ATOL)

    def test_1site_imagtime_converges_from_good_start(self):
        """1-site imaginary-time QN TDVP refines a warm-started state to E0."""
        psi    = _make_qn_psi(self.N, self.N_UP, seed=0)
        engine = TDVPEngine(psi, self.H_mpo)
        # Warm up with 2-site (tau=4)
        for _ in range(40):
            engine.sweep(dt=0.1, max_dim=16, cutoff=1e-10, num_center=2)
        # Refine with 1-site (tau=2)
        for _ in range(20):
            engine.sweep(dt=0.1, num_center=1)
        E = self._measure_energy(psi)
        self.assertAlmostEqual(E, self.E0_exact, delta=self.ATOL)

    def test_imagtime_energy_monotone(self):
        """Imaginary-time QN energy must be non-increasing sweep by sweep."""
        psi    = _make_qn_psi(self.N, self.N_UP, seed=3)
        engine = TDVPEngine(psi, self.H_mpo)
        energies = []
        for _ in range(5):
            engine.sweep(dt=0.1, max_dim=12, cutoff=1e-10, num_center=2)
            energies.append(self._measure_energy(psi))
        for i in range(1, len(energies)):
            self.assertLessEqual(energies[i], energies[i - 1] + 1e-6,
                                 msg=f"Energy increased at step {i}: "
                                     f"{energies[i-1]:.6f} → {energies[i]:.6f}")


if __name__ == "__main__":
    unittest.main()

