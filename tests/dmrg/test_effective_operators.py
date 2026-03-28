"""Unit tests for EffOperator and EffVector.

Coverage
--------
1. EffVector construction  (TestEffVectorConstruction)
   - 1-site and 2-site: result has labels ["l","i0","r"] / ["l","i0","i1","r"]
   - 0-site: result has labels ["l","r"]

2. EffVector.inner  (TestEffVectorInner)
   - inner(phi) returns a scalar
   - inner(Φ_0) ≈ ||Φ_0||^2  (self-overlap, real MPS)
   - inner(Φ_0) ≈ ||Φ_0||^2  (self-overlap, genuinely complex MPS)
   - linearity: inner(alpha*phi) = alpha * inner(phi)

3. EffOperator construction  (TestEffOperatorConstruction)
   - accepts 0/1/2 MPO tensors
   - MPS.make_phi 1-site: labels ["l","i0","r"]
   - MPS.make_phi 2-site: labels ["l","i0","i1","r"]
   - MPS.make_phi out-of-range checks

4. EffOperator.apply  (TestEffOperatorApply)
   - identity MPO (dim-1 boundaries): apply(phi) == phi,  1-site and 2-site
   - scalar MPO (c * identity): apply(phi) == c * phi  (non-trivial MPO test)
   - apply returns tensors with same labels as phi

5. EffOperator.add_term  (TestEffOperatorAddTerm)
   - With term weight=0: result unchanged
   - With nonzero term: result differs from H-only result
   - Rank-1 contribution formula: apply_with - apply_without = w * inner * |Φ_0⟩
   - Two add_term calls: both rank-1 terms applied and summed correctly

6. MPS.update_sites  (TestEffOperatorSplitPhi)
   - 1-site: returns single tensor, labels ["l","i","r"]
   - 2-site absorb="right": left tensor is isometric
   - 2-site absorb="left":  right tensor is isometric
   - update then make_phi reconstructs original phi (no truncation)
   - update with max_dim < chi: bond dimension is actually truncated
   - 0-site phi update: NotImplementedError

7. Integration with OperatorEnv / VectorEnv  (TestEffOperatorIntegration)
   - Identity MPO: <ψ|H_eff(p,p+1)|ψ> / <φ|φ> = 1 for normalized product-state MPS
   - VectorEnv + EffVector.inner self-overlap == ||Φ_0||^2
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent.parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

try:
    import cytnx
except ImportError:
    cytnx = None

if cytnx is not None:
    from MPS.mps import MPS
    from MPS.mpo import MPO
    from MPS.mps_init import random_mps
    from DMRG.environment import OperatorEnv, VectorEnv
    from DMRG.effective_operators import EffVector, EffOperator
    from MPS.linalg import inner as linalg_inner


# ===========================================================================
# Helpers
# ===========================================================================

def _make_mps_site(dl: int, d: int, dr: int,
                   arr: "np.ndarray | None" = None) -> "cytnx.UniTensor":
    """Create a rank-3 MPS site tensor with labels [l, i, r]."""
    if arr is None:
        arr = np.ones((dl, d, dr), dtype=float)
    u = cytnx.UniTensor(cytnx.from_numpy(arr.astype(complex)), rowrank=2)
    u.set_labels(["l", "i", "r"])
    return u


def _make_mpo_site(dl: int, d: int, dr: int,
                   arr: "np.ndarray | None" = None) -> "cytnx.UniTensor":
    """Create a rank-4 MPO site tensor with labels [l, ip, i, r]."""
    if arr is None:
        arr = np.zeros((dl, d, d, dr), dtype=float)
        for j in range(d):
            arr[0, j, j, 0] = 1.0    # identity by default
    u = cytnx.UniTensor(cytnx.from_numpy(arr.astype(complex)), rowrank=2)
    u.set_labels(["l", "ip", "i", "r"])
    return u


def _make_identity_mpo(N: int, d: int) -> "MPO":
    """Identity MPO with virtual bond dim 1."""
    tensors = []
    for _ in range(N):
        tensors.append(_make_mpo_site(1, d, 1))
    return MPO(tensors)


def _make_random_mps(N: int, d: int = 2, D: int = 3, seed: int = 0) -> "MPS":
    return random_mps(N, d, D, seed=seed, normalize=True)


def _make_phi_from_sites(*mps_tensors: "cytnx.UniTensor") -> "cytnx.UniTensor":
    """Build phi with the current API: MPS.make_phi(p, n)."""
    if len(mps_tensors) == 0:
        raise ValueError("Need at least one MPS tensor to build phi.")
    return MPS(list(mps_tensors)).make_phi(0, len(mps_tensors))


def _trivial_op_env_LR(d: int, Dmid: int = 1):
    """Minimal scalar (dim-1) L and R for OperatorEnv tests.

    L shape: [Dmid, 1, 1], labels ["mid","dn","up"], all zeros except L[0,0,0]=1
    R shape: [Dmid, 1, 1], labels ["mid","dn","up"], all zeros except R[0,0,0]=1
    """
    L_arr = np.zeros((Dmid, 1, 1), dtype=complex)
    L_arr[0, 0, 0] = 1.0
    L = cytnx.UniTensor(cytnx.from_numpy(L_arr), rowrank=2)
    L.set_labels(["mid", "dn", "up"])

    R_arr = np.zeros((Dmid, 1, 1), dtype=complex)
    R_arr[0, 0, 0] = 1.0
    R = cytnx.UniTensor(cytnx.from_numpy(R_arr), rowrank=2)
    R.set_labels(["mid", "dn", "up"])
    return L, R


def _vec_env_LR(mps: "MPS", site: int):
    """Build VectorEnv for mps with itself and extract L, R for 1-site at `site`.

    Returns (L, R) = (vec_env[site-1], vec_env[site+1]).
    """
    vec_env = VectorEnv(mps, mps, init_center=site)
    return vec_env[site - 1], vec_env[site + 1]


def _phi_labels(phi: "cytnx.UniTensor") -> list[str]:
    return list(phi.labels())


# ===========================================================================
# 1. EffVector construction
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestEffVectorConstruction(unittest.TestCase):

    def setUp(self):
        self.mps = _make_random_mps(4, d=2, D=2, seed=10)

    def test_1site_labels(self):
        """EffVector with 1 MPS tensor has labels ['l','i0','r']."""
        L, R = _vec_env_LR(self.mps, site=1)
        ev = EffVector(L, R, self.mps[1])
        labels = set(_phi_labels(ev.tensor))
        self.assertEqual(labels, {"l", "i0", "r"})

    def test_2site_labels(self):
        """EffVector with 2 MPS tensors has labels ['l','i0','i1','r']."""
        vec_env = VectorEnv(self.mps, self.mps, init_center=1)
        vec_env.update_envs(1, 2)   # stale window = [1,2]
        L = vec_env[0]
        R = vec_env[3]
        ev = EffVector(L, R, self.mps[1], self.mps[2])
        labels = set(_phi_labels(ev.tensor))
        self.assertEqual(labels, {"l", "i0", "i1", "r"})

    def test_0site_labels(self):
        """EffVector with 0 MPS tensors has labels ['l','r']."""
        L, R = _vec_env_LR(self.mps, site=1)
        ev = EffVector(L, R)
        labels = set(_phi_labels(ev.tensor))
        self.assertEqual(labels, {"l", "r"})


# ===========================================================================
# 2. EffVector.inner
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestEffVectorInner(unittest.TestCase):

    def setUp(self):
        """1-site setup using real VectorEnv for correct bond directions."""
        self.mps = _make_random_mps(4, d=2, D=2, seed=42)
        L, R = _vec_env_LR(self.mps, site=1)
        self.ev = EffVector(L, R, self.mps[1])
        # Build a second random phi with the same structure
        self.phi = EffVector(L, R, self.mps[2]).tensor
        # Also store Φ_0 itself for self-overlap test
        self.phi0 = self.ev.tensor

    def test_inner_returns_scalar(self):
        """inner() returns a Python scalar (complex)."""
        result = self.ev.inner(self.phi)
        self.assertIsInstance(result, (complex, float, int))

    def test_inner_linearity(self):
        """inner(alpha * phi) = alpha * inner(phi)."""
        alpha = 3.0 + 1.5j
        scaled = self.phi * alpha
        self.assertAlmostEqual(
            self.ev.inner(scaled),
            alpha * self.ev.inner(self.phi),
            places=10,
        )

    def test_self_inner_equals_norm_squared(self):
        """inner(Φ_0) == ||Φ_0||^2  (using Φ_0 itself as phi)."""
        result = self.ev.inner(self.phi0)
        norm2 = self.phi0.Norm().item() ** 2
        self.assertAlmostEqual(result.real, norm2, places=6)

    def test_self_inner_complex_mps(self):
        """inner on genuinely complex MPS: inner(Φ_0) is real and equals ||Φ_0||^2."""
        rng = np.random.default_rng(seed=77)
        sites = [
            _make_mps_site(1, 2, 2, rng.random((1, 2, 2)) + 1j * rng.random((1, 2, 2))),
            _make_mps_site(2, 2, 2, rng.random((2, 2, 2)) + 1j * rng.random((2, 2, 2))),
            _make_mps_site(2, 2, 2, rng.random((2, 2, 2)) + 1j * rng.random((2, 2, 2))),
            _make_mps_site(2, 2, 1, rng.random((2, 2, 1)) + 1j * rng.random((2, 2, 1))),
        ]
        mps_c = MPS(sites)
        L, R = _vec_env_LR(mps_c, site=1)
        ev = EffVector(L, R, mps_c[1])
        result = ev.inner(ev.tensor)
        norm2 = ev.tensor.Norm().item() ** 2
        self.assertAlmostEqual(result.real, norm2, places=6)
        self.assertAlmostEqual(result.imag, 0.0, places=6)


# ===========================================================================
# 3. EffOperator construction
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestEffOperatorConstruction(unittest.TestCase):

    def _LR(self):
        return _trivial_op_env_LR(d=2)

    def test_num_sites_1(self):
        L, R = self._LR()
        M = _make_mpo_site(1, 2, 1)
        eff = EffOperator(L, R, M)
        self.assertEqual(len(eff._mpo_tensors), 1)

    def test_num_sites_2(self):
        L, R = self._LR()
        M0 = _make_mpo_site(1, 2, 1)
        M1 = _make_mpo_site(1, 2, 1)
        eff = EffOperator(L, R, M0, M1)
        self.assertEqual(len(eff._mpo_tensors), 2)

    def test_num_sites_0(self):
        L, R = self._LR()
        eff = EffOperator(L, R)
        self.assertEqual(len(eff._mpo_tensors), 0)

    def test_make_phi_1site_labels(self):
        """MPS.make_phi with 1 site → labels ['l','i0','r']."""
        A = _make_mps_site(1, 2, 1)
        phi = _make_phi_from_sites(A)
        self.assertEqual(set(_phi_labels(phi)), {"l", "i0", "r"})

    def test_make_phi_2site_labels(self):
        """MPS.make_phi with 2 sites → labels ['l','i0','i1','r']."""
        A0 = _make_mps_site(1, 2, 2)
        A1 = _make_mps_site(2, 2, 1)
        phi = _make_phi_from_sites(A0, A1)
        self.assertEqual(set(_phi_labels(phi)), {"l", "i0", "i1", "r"})

    def test_make_phi_wrong_count(self):
        """MPS.make_phi with out-of-range window raises IndexError."""
        A0 = _make_mps_site(1, 2, 2)
        A1 = _make_mps_site(2, 2, 1)
        mps = MPS([A0, A1])
        with self.assertRaises(IndexError):
            mps.make_phi(1, 2)

    def test_make_phi_0site_raises(self):
        """MPS.make_phi with n=0 raises ValueError."""
        A = _make_mps_site(1, 2, 1)
        mps = MPS([A])
        with self.assertRaises(ValueError):
            mps.make_phi(0, 0)


# ===========================================================================
# 4. EffOperator.apply
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestEffOperatorApply(unittest.TestCase):
    """Correctness of EffOperator._apply_operator."""

    def test_1site_apply_identity(self):
        """1-site identity MPO + trivial boundaries: apply(phi) == phi."""
        d = 2
        L, R = _trivial_op_env_LR(d=d, Dmid=1)
        M = _make_mpo_site(1, d, 1)
        eff = EffOperator(L, R, M)

        rng = np.random.default_rng(10)
        arr = rng.standard_normal((1, d, 1)).astype(complex)
        A = _make_mps_site(1, d, 1, arr=arr)
        phi = _make_phi_from_sites(A)

        result = eff.apply(phi)

        phi_np    = phi.get_block().numpy().ravel()
        result_np = result.get_block().numpy().ravel()
        np.testing.assert_allclose(result_np, phi_np, atol=1e-10)

    def test_2site_apply_identity(self):
        """2-site identity MPO + trivial boundaries: apply(phi) == phi."""
        d = 2
        L, R = _trivial_op_env_LR(d=d, Dmid=1)
        M0 = _make_mpo_site(1, d, 1)
        M1 = _make_mpo_site(1, d, 1)
        eff = EffOperator(L, R, M0, M1)

        rng = np.random.default_rng(11)
        arr0 = rng.standard_normal((1, d, 1)).astype(complex)
        arr1 = rng.standard_normal((1, d, 1)).astype(complex)
        A0 = _make_mps_site(1, d, 1, arr=arr0)
        A1 = _make_mps_site(1, d, 1, arr=arr1)
        phi = _make_phi_from_sites(A0, A1)

        result = eff.apply(phi)

        phi_np    = phi.get_block().numpy().ravel()
        result_np = result.get_block().numpy().ravel()
        np.testing.assert_allclose(result_np, phi_np, atol=1e-10)

    def test_1site_apply_scalar_mpo(self):
        """Scalar MPO c * I: apply(phi) == c * phi  (non-trivial MPO values).

        Each site matrix element W[0,j,k,0] = c * delta_{j,k}.
        With trivial boundaries, _apply_operator must return c * phi.
        This verifies that MPO matrix elements are actually contracted and
        not accidentally dropped.
        """
        d, c = 3, 2.5
        L, R = _trivial_op_env_LR(d=d, Dmid=1)
        arr = np.zeros((1, d, d, 1), dtype=float)
        for j in range(d):
            arr[0, j, j, 0] = c
        M = _make_mpo_site(1, d, 1, arr=arr)
        eff = EffOperator(L, R, M)

        rng = np.random.default_rng(12)
        A = _make_mps_site(1, d, 1,
                           arr=rng.standard_normal((1, d, 1)).astype(complex))
        phi = _make_phi_from_sites(A)

        result = eff.apply(phi)

        phi_np    = phi.get_block().numpy().ravel()
        result_np = result.get_block().numpy().ravel()
        np.testing.assert_allclose(result_np, c * phi_np, atol=1e-10)

    def test_2site_apply_scalar_mpo(self):
        """2-site scalar MPO c * I: apply(phi) == c^2 * phi.

        Two sites each with scalar c → total factor c^2 on phi.
        """
        d, c = 2, 3.0
        L, R = _trivial_op_env_LR(d=d, Dmid=1)
        arr = np.zeros((1, d, d, 1), dtype=float)
        for j in range(d):
            arr[0, j, j, 0] = c
        M0 = _make_mpo_site(1, d, 1, arr=arr)
        M1 = _make_mpo_site(1, d, 1, arr=arr)
        eff = EffOperator(L, R, M0, M1)

        rng = np.random.default_rng(13)
        A0 = _make_mps_site(1, d, 1,
                             arr=rng.standard_normal((1, d, 1)).astype(complex))
        A1 = _make_mps_site(1, d, 1,
                             arr=rng.standard_normal((1, d, 1)).astype(complex))
        phi = _make_phi_from_sites(A0, A1)

        result = eff.apply(phi)

        phi_np    = phi.get_block().numpy().ravel()
        result_np = result.get_block().numpy().ravel()
        np.testing.assert_allclose(result_np, c ** 2 * phi_np, atol=1e-10)

    def test_apply_returns_same_labels(self):
        """apply(phi) has the same labels as phi."""
        d = 3
        L, R = _trivial_op_env_LR(d=d, Dmid=1)
        M = _make_mpo_site(1, d, 1)
        eff = EffOperator(L, R, M)
        A = _make_mps_site(1, d, 1)
        phi = _make_phi_from_sites(A)
        result = eff.apply(phi)
        self.assertEqual(set(_phi_labels(result)), set(_phi_labels(phi)))


# ===========================================================================
# 5. EffOperator.add_term
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestEffOperatorAddTerm(unittest.TestCase):

    def _setup(self, d=2):
        # Use a 4-site D=1 MPS so that environments are trivial scalars.
        mps_ref = _make_random_mps(4, d=d, D=1, seed=20)
        mps_phi = _make_random_mps(4, d=d, D=1, seed=21)
        mpo     = _make_identity_mpo(4, d)

        op_env = OperatorEnv(mps_phi, mps_phi, mpo, init_center=1)
        op_env.update_envs(1, 1)   # stale=[1,1]
        L = op_env[0]
        R = op_env[2]
        M = mpo[1]
        eff = EffOperator(L, R, M)

        vec_env = VectorEnv(mps_ref, mps_phi, init_center=1)
        L_v = vec_env[0]
        R_v = vec_env[2]
        ev = EffVector(L_v, R_v, mps_ref[1])

        phi = mps_phi.make_phi(1, 1)
        return eff, ev, phi

    def test_weight_zero_unchanged(self):
        """add_term with weight=0 leaves apply result unchanged."""
        eff, ev, phi = self._setup()
        result_before = eff.apply(phi).get_block().numpy().ravel().copy()
        eff.add_term(ev, 0.0)
        result_after = eff.apply(phi).get_block().numpy().ravel()
        np.testing.assert_allclose(result_after, result_before, atol=1e-12)

    def test_add_term_changes_result(self):
        """add_term with nonzero weight changes apply result."""
        eff, ev, phi = self._setup()
        result_before = eff.apply(phi).get_block().numpy().ravel().copy()
        eff.add_term(ev, 10.0)
        result_after = eff.apply(phi).get_block().numpy().ravel()
        self.assertFalse(np.allclose(result_after, result_before),
                         "add_term with nonzero weight must change the result.")

    def test_add_term_rank1_contribution(self):
        """apply_with_term(phi) - apply_plain(phi) = w * inner(phi) * |Φ_0>."""
        d = 2
        eff_plain, ev, phi = self._setup(d=d)
        eff_with, _, _ = self._setup(d=d)
        weight = 5.0
        eff_with.add_term(ev, weight)

        H_phi      = eff_plain.apply(phi).get_block().numpy().ravel()
        H_plus_phi = eff_with.apply(phi).get_block().numpy().ravel()
        overlap = ev.inner(phi)
        Phi0 = ev.tensor.get_block().numpy().ravel()

        expected_diff = weight * overlap * Phi0
        np.testing.assert_allclose(
            H_plus_phi - H_phi, expected_diff, atol=1e-10
        )

    def test_multiple_add_terms(self):
        """Two add_term calls: both rank-1 terms are applied and summed correctly.

        apply(phi) - H|phi> = w1 * <Φ1|phi> * |Φ1> + w2 * <Φ2|phi> * |Φ2>
        """
        d = 2
        # Two independent EffVectors from two different reference MPS
        mps_ref1 = _make_random_mps(4, d=d, D=1, seed=22)
        mps_ref2 = _make_random_mps(4, d=d, D=1, seed=23)
        mps_phi  = _make_random_mps(4, d=d, D=1, seed=21)
        mpo      = _make_identity_mpo(4, d)

        op_env = OperatorEnv(mps_phi, mps_phi, mpo, init_center=1)
        op_env.update_envs(1, 1)
        L, R, M = op_env[0], op_env[2], mpo[1]

        eff_plain = EffOperator(L, R, M)
        eff_with  = EffOperator(L, R, M)

        def _make_ev(mps_ref):
            ve = VectorEnv(mps_ref, mps_phi, init_center=1)
            return EffVector(ve[0], ve[2], mps_ref[1])

        ev1, ev2 = _make_ev(mps_ref1), _make_ev(mps_ref2)
        w1, w2 = 3.0, -1.5
        eff_with.add_term(ev1, w1)
        eff_with.add_term(ev2, w2)

        phi = mps_phi.make_phi(1, 1)
        H_phi      = eff_plain.apply(phi).get_block().numpy().ravel()
        H_plus_phi = eff_with.apply(phi).get_block().numpy().ravel()

        Phi1 = ev1.tensor.get_block().numpy().ravel()
        Phi2 = ev2.tensor.get_block().numpy().ravel()
        expected_diff = (w1 * ev1.inner(phi) * Phi1
                         + w2 * ev2.inner(phi) * Phi2)
        np.testing.assert_allclose(
            H_plus_phi - H_phi, expected_diff, atol=1e-10
        )


# ===========================================================================
# 6. EffOperator.split_phi
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestEffOperatorSplitPhi(unittest.TestCase):

    def test_1site_split_labels(self):
        """split_phi on 1-site phi → labels ['l','i','r']."""
        d = 3
        L, R = _trivial_op_env_LR(d=d)
        M = _make_mpo_site(1, d, 1)
        eff = EffOperator(L, R, M)
        A = _make_mps_site(1, d, 1)
        mps = MPS([A, _make_mps_site(1, d, 1)])
        phi = mps.make_phi(0, 1)
        mps.update_sites(0, phi, max_dim=10, cutoff=0.0, absorb="right")
        A_out = mps[0]
        self.assertEqual(set(_phi_labels(A_out)), {"l", "i", "r"})

    def test_2site_split_right_left_isometry(self):
        """split_phi absorb='right': left tensor is isometry (A† A = I)."""
        d = 2
        chi = 3   # non-trivial bond dim
        L, R = _trivial_op_env_LR(d=d)
        M0 = _make_mpo_site(1, d, 1)
        M1 = _make_mpo_site(1, d, 1)
        eff = EffOperator(L, R, M0, M1)

        rng = np.random.default_rng(30)
        arr0 = rng.standard_normal((1, d, chi)).astype(complex)
        arr1 = rng.standard_normal((chi, d, 1)).astype(complex)
        A0 = _make_mps_site(1, d, chi, arr=arr0)
        A1 = _make_mps_site(chi, d, 1, arr=arr1)
        mps = MPS([A0, A1])
        phi = mps.make_phi(0, 2)
        mps.update_sites(0, phi, max_dim=chi, cutoff=0.0, absorb="right")
        A_left, A_right = mps[0], mps[1]

        # Left tensor is isometry: A† A ≈ I  (contract l,i bonds)
        A_left_dag = A_left.Dagger()
        # rename so the contraction works on l and i
        At = A_left_dag.relabels(["l", "i", "r"], ["_l", "_i", "s"])
        Af = A_left.relabels(["l", "i", "r"], ["_l", "_i", "s2"])
        prod = cytnx.Contract(At, Af)  # contracts _l and _i → square matrix in s,s2
        prod_np = prod.get_block().numpy()
        s_dim = prod_np.shape[0]
        np.testing.assert_allclose(prod_np, np.eye(s_dim), atol=1e-8)

    def test_2site_split_left_right_isometry(self):
        """split_phi absorb='left': right tensor is isometry (B B† = I)."""
        d = 2
        chi = 3
        L, R = _trivial_op_env_LR(d=d)
        M0 = _make_mpo_site(1, d, 1)
        M1 = _make_mpo_site(1, d, 1)
        eff = EffOperator(L, R, M0, M1)

        rng = np.random.default_rng(31)
        arr0 = rng.standard_normal((1, d, chi)).astype(complex)
        arr1 = rng.standard_normal((chi, d, 1)).astype(complex)
        A0 = _make_mps_site(1, d, chi, arr=arr0)
        A1 = _make_mps_site(chi, d, 1, arr=arr1)
        mps = MPS([A0, A1])
        phi = mps.make_phi(0, 2)
        mps.update_sites(0, phi, max_dim=chi, cutoff=0.0, absorb="left")
        A_left, A_right = mps[0], mps[1]

        # Right tensor is isometry: B B† ≈ I
        Bf = A_right.relabels(["l", "i", "r"], ["s", "_i", "_r"])
        Bt = A_right.Dagger().relabels(["l", "i", "r"], ["s2", "_i", "_r"])
        prod = cytnx.Contract(Bf, Bt)
        prod_np = prod.get_block().numpy()
        s_dim = prod_np.shape[0]
        np.testing.assert_allclose(prod_np, np.eye(s_dim), atol=1e-8)

    def test_2site_split_reconstruct(self):
        """split then make_phi reconstructs original phi (no truncation)."""
        d = 2
        chi = 2
        L, R = _trivial_op_env_LR(d=d)
        M0 = _make_mpo_site(1, d, 1)
        M1 = _make_mpo_site(1, d, 1)
        eff = EffOperator(L, R, M0, M1)

        rng = np.random.default_rng(32)
        arr0 = rng.standard_normal((1, d, chi)).astype(complex)
        arr1 = rng.standard_normal((chi, d, 1)).astype(complex)
        A0 = _make_mps_site(1, d, chi, arr=arr0)
        A1 = _make_mps_site(chi, d, 1, arr=arr1)
        mps = MPS([A0, A1])
        phi = mps.make_phi(0, 2)
        mps.update_sites(0, phi, max_dim=chi, cutoff=0.0, absorb="right")
        A_left, A_right = mps[0], mps[1]

        # Rebuild phi from split tensors
        phi_rebuilt = mps.make_phi(0, 2)

        phi_np       = phi.get_block().numpy().ravel()
        phi_rebuilt_np = phi_rebuilt.get_block().numpy().ravel()
        np.testing.assert_allclose(phi_rebuilt_np, phi_np, atol=1e-8)

    def test_2site_split_truncation(self):
        """split_phi with max_dim < full rank truncates the shared bond.

        Build phi from (1, d, chi) × (chi, d, 1) tensors so the full rank
        is chi.  Requesting max_dim = chi//2 must produce tensors whose
        shared bond dimension is at most max_dim.
        """
        d   = 2
        chi = 4          # full rank
        keep = chi // 2  # request truncation to 2
        L, R = _trivial_op_env_LR(d=d)
        M0 = _make_mpo_site(1, d, 1)
        M1 = _make_mpo_site(1, d, 1)
        eff = EffOperator(L, R, M0, M1)

        rng = np.random.default_rng(33)
        A0 = _make_mps_site(1, d, chi,
                             arr=rng.standard_normal((1, d, chi)).astype(complex))
        A1 = _make_mps_site(chi, d, 1,
                             arr=rng.standard_normal((chi, d, 1)).astype(complex))
        mps = MPS([A0, A1])
        phi = mps.make_phi(0, 2)
        mps.update_sites(0, phi, max_dim=keep, cutoff=0.0, absorb="right")
        A_left, A_right = mps[0], mps[1]

        # The shared bond ("r" of A_left = "l" of A_right) must be ≤ keep.
        shape_left  = {l: s for l, s in zip(A_left.labels(),  A_left.shape())}
        shape_right = {l: s for l, s in zip(A_right.labels(), A_right.shape())}
        self.assertLessEqual(shape_left["r"],  keep,
                             "Left tensor's 'r' bond must be truncated.")
        self.assertLessEqual(shape_right["l"], keep,
                             "Right tensor's 'l' bond must be truncated.")

    def test_0site_split_raises(self):
        """MPS.update_sites with 0-site phi raises NotImplementedError."""
        mps = MPS([_make_mps_site(1, 2, 1)])
        arr = np.ones((1, 1), dtype=complex)
        phi = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=1)
        phi.set_labels(["l", "r"])
        with self.assertRaises(NotImplementedError):
            mps.update_sites(0, phi, max_dim=1, cutoff=0.0, absorb="right")


# ===========================================================================
# 7. Integration with OperatorEnv / VectorEnv
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestEffOperatorIntegration(unittest.TestCase):
    """End-to-end: build eff operator from environment objects, check energy."""

    def setUp(self):
        """4-site, d=2, D=1 (product state), identity MPO.

        With D=1 the MPS is a product state and all environments are trivial
        scalars, so <φ|H_eff|φ> / <φ|φ> = 1 exactly for the identity MPO.
        """
        self.N = 4
        self.d = 2
        self.D = 1
        self.mps  = _make_random_mps(self.N, d=self.d, D=self.D, seed=5)
        self.mpo  = _make_identity_mpo(self.N, self.d)
        # OperatorEnv centered at site 0 (all right envs computed)
        self.op_env = OperatorEnv(self.mps, self.mps, self.mpo, init_center=0)

    def test_rayleigh_quotient_identity_mpo(self):
        """<φ|H_eff|φ> / <φ|φ> = 1 for identity MPO, product-state MPS."""
        p = 1   # 2-site block: sites [1, 2]
        self.op_env.update_envs(p, p + 1)   # stale window = [1, 2]

        L = self.op_env[p - 1]
        R = self.op_env[p + 2]

        eff = EffOperator(L, R, self.mpo[p], self.mpo[p + 1])
        phi = self.mps.make_phi(p, 2)

        H_phi = eff.apply(phi)

        energy = linalg_inner(phi, H_phi)
        norm2  = linalg_inner(phi, phi)

        rayleigh = energy.real / norm2.real
        self.assertAlmostEqual(rayleigh, 1.0, places=6)

    def test_effvector_inner_with_vec_env(self):
        """VectorEnv + EffVector.inner gives ||Φ_0||^2 for self-overlap."""
        p = 1
        vec_env = VectorEnv(self.mps, self.mps, init_center=p)
        L_v = vec_env[p - 1]
        R_v = vec_env[p + 1]

        ev = EffVector(L_v, R_v, self.mps[p])
        phi = ev.tensor   # Φ_0 itself

        inner_result = ev.inner(phi)
        norm2 = phi.Norm().item() ** 2
        self.assertAlmostEqual(inner_result.real, norm2, places=8)


if __name__ == "__main__":
    unittest.main()
