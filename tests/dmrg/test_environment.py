"""Unit tests for the DMRG environment module.

Coverage
--------
1. Observer pattern on MPS  (TestMPSObserver)
   - register_callback: callback fires on __setitem__ with correct site index
   - Rollback (invalid tensor): callback must NOT fire
   - Dead weak-references are pruned from the list

2. Observer pattern on MPO  (TestMPOObserver)
   - register_callback: callback fires on __setitem__
   - Dead weak-references are pruned

3. OperatorEnv initialisation  (TestLRMPOInit)
   - Mismatched MPS/MPO lengths → AssertionError
   - init_center out of range → AssertionError
   - Stale window is exactly [init_center, init_center] after __init__
   - Boundary tensors LR[-1]/LR[N] are rank-3 with labels [mid, dn, up]
   - Valid environments outside the stale window are not None

4. OperatorEnv access & bookkeeping  (TestLRMPOAccess)
   - __getitem__ on stale index → RuntimeError
   - __getitem__ out of range → ValueError
   - __getitem__ on valid left/right env succeeds
   - delete(p): stale window expands to include p
   - update_envs: stale window shrinks; previously-stale envs become accessible
   - update_envs with centerL > centerR+1 → ValueError
   - update_envs with out-of-range indices → ValueError
   - update_envs(p, p-1): empty stale window — all sites accessible

5. OperatorEnv ↔ Observer integration  (TestLRMPOObserver)
   - Updating mps1[p] via __setitem__ automatically calls delete(p) on LR
   - Updating mpo[p] via __setitem__ automatically calls delete(p) on LR
   - When LR object is garbage-collected the dead callback is pruned from MPS

6. OperatorEnv mathematical correctness  (TestLRMPOMath)
   - Identity MPO: full left sweep gives <psi|I|psi> = ||psi||^2
   - Consistency: left-env contracted from the left vs right-env from the right
     agree when both reach the middle of the chain

7. VectorEnv initialisation  (TestLRMPSInit)
   - Length mismatch → AssertionError
   - Stale window is [init_center, init_center] after __init__
   - Boundary tensors L0/R0 are rank-2 with dims [1,1]

8. VectorEnv ↔ Observer integration  (TestLRMPSObserver)
   - Updating mps1[p] automatically makes VectorEnv[p] stale
   - Updating mps2[p] automatically makes VectorEnv[p] stale
   - When VectorEnv is garbage-collected, dead callback pruned from MPS

9. VectorEnv mathematical correctness  (TestLRMPSMath)
   - Full left sweep gives <mps1|mps2> matching inner(mps1, mps2)
   - Overlap with itself gives 1.0 for a normalised MPS

10. Observer notifications from gauge operations  (TestNotifyFromGaugeOps)
    - move_center rightward: valid right envs become stale automatically
    - move_center leftward:  valid left  envs become stale automatically
    - Each QR step notifies both the source site and its neighbour
    - normalize() fires delete(center) on registered envs
    - update_sites 1-site: fires delete on both updated sites
    - update_sites 2-site: fires delete on both updated sites
    - move_center + full env rebuild gives mathematically correct result
"""

from __future__ import annotations

import sys
import unittest
import weakref
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent.parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

import numpy as np

try:
    import cytnx
except ImportError:
    cytnx = None

if cytnx is not None:
    from MPS.mps import MPS
    from MPS.mpo import MPO
    from MPS.mps_init import random_mps
    from MPS.mps_operations import inner, expectation
    from unitensor.utils import to_numpy_array
    from MPS.physical_sites.spin_half import spin_half
    from MPS.auto_mpo import AutoMPO
    from DMRG.environment import (
        OperatorEnv,
        VectorEnv,
    )
    from tests.helpers.mps_test_cases import random_u1_sz_mps


# ===========================================================================
# Helpers shared across test classes
# ===========================================================================

def _make_mps_site(dl: int, d: int, dr: int) -> "cytnx.UniTensor":
    """Create a rank-3 MPS site tensor with labels [l, i, r], filled with ones."""
    arr = np.ones((dl, d, dr), dtype=float)
    u = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=2)
    u.set_labels(["l", "i", "r"])
    return u


def _make_mpo_site(dl: int, d: int, dr: int, arr: "np.ndarray | None" = None,
                   dtype: "np.dtype | type" = float) -> "cytnx.UniTensor":
    """Create a rank-4 MPO site tensor with labels [l, ip, i, r].

    If *arr* is None, fills with ones.  Shape must be (dl, d, d, dr).
    """
    np_dtype = np.complex128 if np.issubdtype(np.dtype(dtype), np.complexfloating) else np.float64
    if arr is None:
        arr = np.ones((dl, d, d, dr), dtype=np_dtype)
    u = cytnx.UniTensor(cytnx.from_numpy(arr.astype(np_dtype)), rowrank=2)
    u.set_labels(["l", "ip", "i", "r"])
    return u


def _make_identity_mpo(N: int, d: int, dtype: "np.dtype | type" = float) -> "MPO":
    """Build an identity MPO with virtual bond dimension 1.

    Each site tensor W[0, j, k, 0] = delta_{j,k}, so <psi|MPO|psi> = <psi|psi>.
    """
    np_dtype = np.complex128 if np.issubdtype(np.dtype(dtype), np.complexfloating) else np.float64
    tensors = []
    for _ in range(N):
        arr = np.zeros((1, d, d, 1), dtype=np_dtype)
        for j in range(d):
            arr[0, j, j, 0] = 1.0
        tensors.append(_make_mpo_site(1, d, 1, arr=arr, dtype=dtype))
    return MPO(tensors)


def _make_random_mps(
    N: int,
    d: int = 2,
    D: int = 4,
    seed: int = 0,
    dtype: np.dtype | type = float,
) -> "MPS":
    """Return a normalised random MPS."""
    return random_mps(N, d, D, seed=seed, normalize=True, dtype=dtype)


# Simple counter class used as a callback observer in tests.
class _Counter:
    """Records (site) arguments from every callback invocation."""

    def __init__(self):
        self.calls: list[int] = []

    def delete(self, site: int) -> None:
        self.calls.append(site)


# ===========================================================================
# 1. Observer pattern on MPS
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestMPSObserver(unittest.TestCase):
    """Observer / callback behaviour of MPS.__setitem__ and register_callback."""

    def setUp(self):
        # 4-site MPS, phys_dim=2, bond_dim=3
        self.mps = _make_random_mps(4, d=2, D=3, seed=1)

    def test_register_and_fire_correct_site(self):
        """Updating mps[p] fires the callback with the correct site index."""
        counter = _Counter()
        self.mps.register_callback(counter)
        # Replace site 2 with a valid new tensor (same shape).
        new_tensor = self.mps.tensors[2].clone()
        self.mps[2] = new_tensor
        self.assertEqual(counter.calls, [2])

    def test_callback_fires_for_every_update(self):
        """Each __setitem__ call fires the callback once."""
        counter = _Counter()
        self.mps.register_callback(counter)
        for site in [0, 1, 2]:
            self.mps[site] = self.mps.tensors[site].clone()
        self.assertEqual(counter.calls, [0, 1, 2])

    def test_invalid_tensor_does_not_fire_callback(self):
        """A failing setitem (wrong labels) triggers rollback and must NOT fire the callback."""
        counter = _Counter()
        self.mps.register_callback(counter)
        # Build a tensor with wrong labels so MPS raises on validation.
        arr = np.ones((1, 2, 1), dtype=float)
        bad = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=2)
        bad.set_labels(["a", "b", "c"])  # wrong labels
        with self.assertRaises(Exception):
            self.mps[0] = bad
        self.assertEqual(counter.calls, [], "Callback must not fire on rollback.")

    def test_dead_weakref_pruned(self):
        """Once the observer is garbage-collected its weak-ref is removed from the list."""
        counter = _Counter()
        self.mps.register_callback(counter)
        self.assertEqual(len(self.mps._callbacks), 1)
        # Delete the only reference to *counter* and force GC.
        del counter
        # Trigger a setitem so pruning runs.
        self.mps[0] = self.mps.tensors[0].clone()
        self.assertEqual(len(self.mps._callbacks), 0, "Dead weak-ref should be pruned.")

    def test_multiple_observers(self):
        """Multiple registered observers all fire on a single __setitem__."""
        c1, c2 = _Counter(), _Counter()
        self.mps.register_callback(c1)
        self.mps.register_callback(c2)
        self.mps[1] = self.mps.tensors[1].clone()
        self.assertEqual(c1.calls, [1])
        self.assertEqual(c2.calls, [1])


# ===========================================================================
# 2. Observer pattern on MPO
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestMPOObserver(unittest.TestCase):
    """Observer / callback behaviour of MPO.__setitem__ and register_callback."""

    def setUp(self):
        self.mpo = _make_identity_mpo(4, d=2)

    def test_register_and_fire_correct_site(self):
        """Updating mpo[p] fires the callback with the correct site index p."""
        counter = _Counter()
        self.mpo.register_callback(counter)
        new_tensor = self.mpo.tensors[1].clone()
        self.mpo[1] = new_tensor
        self.assertEqual(counter.calls, [1])

    def test_dead_weakref_pruned(self):
        """Garbage-collected observer's weak-ref is cleaned up on next __setitem__."""
        counter = _Counter()
        self.mpo.register_callback(counter)
        del counter
        self.mpo[0] = self.mpo.tensors[0].clone()
        self.assertEqual(len(self.mpo._callbacks), 0)

    def test_multiple_observers(self):
        """All registered MPO observers fire on a single setitem."""
        c1, c2 = _Counter(), _Counter()
        self.mpo.register_callback(c1)
        self.mpo.register_callback(c2)
        self.mpo[2] = self.mpo.tensors[2].clone()
        self.assertEqual(c1.calls, [2])
        self.assertEqual(c2.calls, [2])


# ===========================================================================
# 3. OperatorEnv initialisation
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestLRMPOInit(unittest.TestCase):
    """Initialisation checks for OperatorEnv."""

    def _make(self, N: int = 4, d: int = 2, D: int = 3,
              init_center: int = 0) -> "OperatorEnv":
        mps = _make_random_mps(N, d, D, seed=42)
        mpo = _make_identity_mpo(N, d)
        return OperatorEnv(mps, mps, mpo, init_center=init_center)

    def test_length_mismatch_raises(self):
        """MPS and MPO with different lengths must raise AssertionError."""
        mps4 = _make_random_mps(4, d=2, D=3, seed=0)
        mps3 = _make_random_mps(3, d=2, D=3, seed=0)
        mpo4 = _make_identity_mpo(4, d=2)
        mpo3 = _make_identity_mpo(3, d=2)
        with self.assertRaises(AssertionError):
            OperatorEnv(mps4, mps3, mpo4)   # mps2 too short
        with self.assertRaises(AssertionError):
            OperatorEnv(mps4, mps4, mpo3)   # mpo too short

    def test_init_center_out_of_range_raises(self):
        """init_center outside [0, N-1] must raise AssertionError."""
        mps = _make_random_mps(4, d=2, D=3, seed=0)
        mpo = _make_identity_mpo(4, d=2)
        with self.assertRaises(AssertionError):
            OperatorEnv(mps, mps, mpo, init_center=-1)
        with self.assertRaises(AssertionError):
            OperatorEnv(mps, mps, mpo, init_center=4)  # N=4, valid is 0..3

    def test_stale_window_equals_init_center(self):
        """After __init__ with init_center=p, stale window is exactly [p, p]."""
        for center in [0, 1, 2, 3]:
            lr = self._make(N=4, init_center=center)
            self.assertEqual(lr.centerL, center,
                             f"centerL should be {center}, got {lr.centerL}")
            self.assertEqual(lr.centerR, center,
                             f"centerR should be {center}, got {lr.centerR}")

    def test_boundary_tensors_stored_correctly(self):
        """LR[-1] and LR[N] must be rank-3 tensors with labels [mid, dn, up]."""
        mps = _make_random_mps(4, d=2, D=3, seed=0)
        mpo = _make_identity_mpo(4, d=2)
        lr = OperatorEnv(mps, mps, mpo)
        L0 = lr.LR[-1]
        R0 = lr.LR[4]
        self.assertEqual(set(L0.labels()), {"mid", "dn", "up"})
        self.assertEqual(set(R0.labels()), {"mid", "dn", "up"})
        self.assertEqual(len(L0.shape()), 3)
        self.assertEqual(len(R0.shape()), 3)

    def test_boundary_dtype_promotes_to_complex(self):
        """Boundary tensors become complex if any input tensor is complex."""
        mps = _make_random_mps(4, d=2, D=3, seed=0, dtype=complex)
        mpo = _make_identity_mpo(4, d=2)
        lr = OperatorEnv(mps, mps, mpo)
        self.assertIn("Complex", lr.LR[-1].dtype_str())
        self.assertIn("Complex", lr.LR[4].dtype_str())

    def test_left_environments_valid_after_init(self):
        """LR[p] for p < init_center must be non-None (computed during init)."""
        lr = self._make(N=4, init_center=2)
        # Valid left envs: LR[-1], LR[0], LR[1]
        for p in [-1, 0, 1]:
            self.assertIsNotNone(lr.LR[p],
                                 f"LR[{p}] should be computed (left of center=2)")

    def test_right_environments_valid_after_init(self):
        """LR[p] for p > init_center must be non-None (computed during init)."""
        lr = self._make(N=4, init_center=1)
        # Valid right envs: LR[2], LR[3], LR[4]
        for p in [2, 3, 4]:
            self.assertIsNotNone(lr.LR[p],
                                 f"LR[{p}] should be computed (right of center=1)")

    def test_N_attribute_matches_chain_length(self):
        """lr.N must equal the number of MPS sites."""
        lr = self._make(N=5)
        self.assertEqual(lr.N, 5)


# ===========================================================================
# 4. OperatorEnv access & bookkeeping
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestLRMPOAccess(unittest.TestCase):
    """__getitem__, delete, and update_envs bookkeeping for OperatorEnv."""

    def setUp(self):
        N, d, D = 5, 2, 3
        self.N = N
        self.mps = _make_random_mps(N, d, D, seed=7)
        self.mpo = _make_identity_mpo(N, d)
        # init at center=2 so there are valid envs on both sides.
        self.lr = OperatorEnv(self.mps, self.mps, self.mpo, init_center=2)

    def test_getitem_stale_raises_RuntimeError(self):
        """Accessing LR[centerL] (inside stale window) raises RuntimeError."""
        with self.assertRaises(RuntimeError):
            _ = self.lr[self.lr.centerL]

    def test_getitem_out_of_range_raises_ValueError(self):
        """Accessing LR[-2] or LR[N+1] raises ValueError."""
        with self.assertRaises(ValueError):
            _ = self.lr[-2]
        with self.assertRaises(ValueError):
            _ = self.lr[self.N + 1]

    def test_getitem_left_boundary_valid(self):
        """LR[-1] (left boundary) is always accessible and non-None."""
        env = self.lr[-1]
        self.assertIsNotNone(env)

    def test_getitem_right_boundary_valid(self):
        """LR[N] (right boundary) is always accessible and non-None."""
        env = self.lr[self.N]
        self.assertIsNotNone(env)

    def test_getitem_valid_left_env(self):
        """LR[p] for p < centerL is valid and accessible after init."""
        # With init_center=2, LR[0] and LR[1] are valid left envs.
        for p in [0, 1]:
            env = self.lr[p]
            self.assertIsNotNone(env)

    def test_getitem_valid_right_env(self):
        """LR[p] for p > centerR is valid and accessible after init."""
        # With N=5 and init_center=2, LR[3] and LR[4] are valid right envs.
        for p in [3, 4]:
            env = self.lr[p]
            self.assertIsNotNone(env)

    def test_delete_expands_stale_window_left(self):
        """delete(p) for p < centerL expands the stale window to include p."""
        original_left = self.lr.centerL  # == 2
        # Mark LR[1] as stale (it was valid before).
        self.lr.delete(1)
        self.assertLessEqual(self.lr.centerL, 1,
                             "centerL should shrink to include the deleted site")
        self.assertEqual(self.lr.centerR, original_left,
                         "centerR should not change when deleting a left site")

    def test_delete_expands_stale_window_right(self):
        """delete(p) for p > centerR expands the stale window to include p."""
        original_right = self.lr.centerR  # == 2
        self.lr.delete(3)
        self.assertGreaterEqual(self.lr.centerR, 3)
        self.assertEqual(self.lr.centerL, original_right)

    def test_delete_out_of_range_raises(self):
        """delete with an index outside [-1, N] raises AssertionError."""
        with self.assertRaises(AssertionError):
            self.lr.delete(-2)
        with self.assertRaises(AssertionError):
            self.lr.delete(self.N + 1)

    def test_update_envs_shrinks_stale_window(self):
        """update_envs(new_left, new_right) shrinks [centerL, centerR] as requested."""
        # Current stale window: [2, 2].  Grow it first by deleting more sites.
        self.lr.delete(0)
        self.lr.delete(4)
        # Now stale window is [0, 4] (whole chain plus boundaries of each side).
        # Update to shrink stale window to [2, 2].
        self.lr.update_envs(2, 2)
        self.assertEqual(self.lr.centerL, 2)
        self.assertEqual(self.lr.centerR, 2)
        # LR[0] and LR[1] should now be recomputed (valid).
        env_left = self.lr[0]
        self.assertIsNotNone(env_left)
        # LR[3] and LR[4] should now be recomputed (valid).
        env_right = self.lr[3]
        self.assertIsNotNone(env_right)

    def test_update_envs_previously_stale_now_accessible(self):
        """After update_envs, sites that were stale and are now outside the window are accessible."""
        # Expand stale window to cover site 1.
        self.lr.delete(1)
        # Rebuild: shrink stale window back to [2, 2].
        self.lr.update_envs(2, 2)
        # LR[1] should now be a freshly computed valid env.
        env = self.lr[1]
        self.assertIsNotNone(env)

    def test_update_envs_invalid_window_raises(self):
        """update_envs with centerL > centerR + 1 raises ValueError."""
        with self.assertRaises(ValueError):
            self.lr.update_envs(4, 2)  # centerL=4 > centerR+1=3

    def test_update_envs_out_of_range_raises(self):
        """update_envs with out-of-range index raises ValueError."""
        with self.assertRaises(ValueError):
            self.lr.update_envs(-2, 2)
        with self.assertRaises(ValueError):
            self.lr.update_envs(1, self.N + 1)

    def test_update_envs_single_site(self):
        """update_envs(p) with one argument sets centerL = centerR = p."""
        self.lr.delete(0)
        self.lr.delete(4)
        self.lr.update_envs(2)  # should default centerR to centerL = 2
        self.assertEqual(self.lr.centerL, 2)
        self.assertEqual(self.lr.centerR, 2)

    def test_update_envs_empty_stale_window(self):
        """update_envs(p, p-1) creates an empty stale window (centerL = centerR + 1).

        After this call, every environment from LR[-1] to LR[N] must be
        accessible (no sites are stale).
        """
        # First expand the stale window to cover the full chain.
        self.lr.delete(0)
        self.lr.delete(self.N - 1)
        # Now shrink to an empty stale window: centerL = N, centerR = N-1
        # i.e. no site is stale.
        self.lr.update_envs(self.N, self.N - 1)
        self.assertEqual(self.lr.centerL, self.N)
        self.assertEqual(self.lr.centerR, self.N - 1)
        # Every environment must now be accessible.
        for i in range(-1, self.N + 1):
            env = self.lr[i]
            self.assertIsNotNone(env,
                                 f"LR[{i}] must be accessible with empty stale window.")


# ===========================================================================
# 5. OperatorEnv ↔ Observer integration
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestLRMPOObserver(unittest.TestCase):
    """Integration: MPS/MPO tensor updates automatically call LR.delete."""

    def setUp(self):
        N, d, D = 4, 2, 3
        self.N = N
        self.mps = _make_random_mps(N, d, D, seed=10)
        self.mpo = _make_identity_mpo(N, d)
        # Start with stale window at center [2, 2].
        self.lr = OperatorEnv(self.mps, self.mps, self.mpo, init_center=2)

    def test_mps_setitem_fires_delete_on_lr(self):
        """Updating mps[p] via __setitem__ makes LR[p] stale automatically."""
        # LR[0] is currently valid (left of center 2).
        _ = self.lr[0]  # should not raise
        # Update mps[0].
        self.mps[0] = self.mps.tensors[0].clone()
        # Now LR[0] must be in the stale window.
        with self.assertRaises(RuntimeError):
            _ = self.lr[0]

    def test_mpo_setitem_fires_delete_on_lr(self):
        """Updating mpo[p] via __setitem__ makes LR[p] stale automatically."""
        # LR[0] is currently valid.
        _ = self.lr[0]
        # Update mpo[0].
        self.mpo[0] = self.mpo.tensors[0].clone()
        with self.assertRaises(RuntimeError):
            _ = self.lr[0]

    def test_mps_setitem_fires_delete_for_right_env(self):
        """Updating mps[p] where LR[p] is a right env also expands the stale window."""
        # LR[3] is a valid right env (right of center 2).
        _ = self.lr[3]
        self.mps[3] = self.mps.tensors[3].clone()
        with self.assertRaises(RuntimeError):
            _ = self.lr[3]

    def test_gc_of_lr_removes_callback_from_mps(self):
        """When an LR object is GC'd, its dead weak-ref is eventually pruned from MPS._callbacks."""
        initial_count = len(self.mps._callbacks)
        # Create and immediately discard a new LR object.
        _ = OperatorEnv(self.mps, self.mps, self.mpo, init_center=1)
        del _
        # Trigger pruning by performing a setitem on mps.
        self.mps[0] = self.mps.tensors[0].clone()
        # The dead weak-ref from the deleted LR must have been pruned.
        self.assertEqual(len(self.mps._callbacks), initial_count,
                         "Dead weak-refs from GC'd LR should be pruned on next setitem.")


# ===========================================================================
# 6. OperatorEnv mathematical correctness
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestLRMPOMath(unittest.TestCase):
    """Mathematical correctness of LR environment contractions for MPO."""

    def _full_left_sweep(self, mps, mpo, *, mps2=None):
        """Build LR with init_center=0, sweep fully left to expose LR[N-1].

        If `mps2` is given it is used as the bra; otherwise bra = ket = `mps`.
        """
        if mps2 is None:
            mps2 = mps
        N = len(mps)
        lr = OperatorEnv(mps, mps2, mpo, init_center=0)
        lr.delete(N - 1)
        lr.update_envs(N, N)
        return lr

    def _scalar_from_env(self, t: "cytnx.UniTensor") -> complex:
        """Extract a scalar from a rank-3 env tensor with shape [1,1,1]."""
        arr = to_numpy_array(t)
        return complex(arr.flat[0])

    N, d, D = 4, 2, 3

    # -- <real | real H | real> -------------------------------------------

    def test_real_bra_real_H_real_ket_norm_squared(self):
        """<psi|I|psi> computed via full left sweep == psi.norm()^2."""
        mps = _make_random_mps(self.N, self.d, self.D, seed=99)
        mpo = _make_identity_mpo(self.N, self.d)
        lr = self._full_left_sweep(mps, mpo)
        val = self._scalar_from_env(lr[self.N - 1])
        expected = mps.norm() ** 2
        self.assertAlmostEqual(val.real, expected, places=10)

    def test_real_bra_real_H_real_ket_normalised(self):
        """For a normalised MPS, <psi|I|psi> = 1.0."""
        mps = random_mps(self.N, self.d, self.D, seed=17, normalize=True)
        mpo = _make_identity_mpo(self.N, self.d)
        lr = self._full_left_sweep(mps, mpo)
        val = self._scalar_from_env(lr[self.N - 1])
        self.assertAlmostEqual(val.real, 1.0, places=10)

    def test_real_bra_real_H_real_ket_left_right_consistency(self):
        """Left env and right env contract to give the same observable."""
        p = 2
        mps = _make_random_mps(self.N, self.d, self.D, seed=55)
        mpo = _make_identity_mpo(self.N, self.d)
        lr = OperatorEnv(mps, mps, mpo, init_center=p)
        L = lr[p - 1]
        R = lr[p + 1]
        A  = mps.tensors[p].relabels(["l", "i", "r"], ["_dn", "_i", "dn"])
        W  = mpo.tensors[p].relabels(["l", "ip", "i", "r"], ["_mid", "_ip", "_i", "mid"])
        Ad = mps.tensors[p].Dagger().relabels(["l", "i", "r"], ["_up", "_ip", "up"])
        E = L.relabels(["mid", "dn", "up"], ["_mid", "_dn", "_up"])
        tmp = cytnx.Contract(E, A)
        tmp = cytnx.Contract(tmp, W)
        tmp = cytnx.Contract(tmp, Ad)
        result = cytnx.Contract(tmp, R)
        val = complex(to_numpy_array(result).flat[0])
        expected = mps.norm() ** 2
        self.assertAlmostEqual(val.real, expected, places=9)

    # -- <real | real H | complex> ----------------------------------------

    def test_real_bra_real_H_complex_ket(self):
        """<real|I|complex> via OperatorEnv matches expectation()."""
        bra = _make_random_mps(self.N, self.d, self.D, seed=81)
        ket = _make_random_mps(self.N, self.d, self.D, seed=82, dtype=complex)
        mpo = _make_identity_mpo(self.N, self.d)
        lr = self._full_left_sweep(ket, mpo, mps2=bra)
        val = self._scalar_from_env(lr[self.N - 1])
        ref = complex(expectation(bra, mpo, ket))
        self.assertAlmostEqual(val.real, ref.real, places=10)
        self.assertAlmostEqual(val.imag, ref.imag, places=10)

    # -- <complex | real H | real> ----------------------------------------

    def test_complex_bra_real_H_real_ket(self):
        """<complex|I|real> via OperatorEnv matches expectation()."""
        bra = _make_random_mps(self.N, self.d, self.D, seed=83, dtype=complex)
        ket = _make_random_mps(self.N, self.d, self.D, seed=84)
        mpo = _make_identity_mpo(self.N, self.d)
        lr = self._full_left_sweep(ket, mpo, mps2=bra)
        val = self._scalar_from_env(lr[self.N - 1])
        ref = complex(expectation(bra, mpo, ket))
        self.assertAlmostEqual(val.real, ref.real, places=10)
        self.assertAlmostEqual(val.imag, ref.imag, places=10)

    # -- <complex | real H | complex> -------------------------------------

    def test_complex_bra_real_H_complex_ket_self(self):
        """<psi|I|psi> is real and positive for complex MPS and real identity MPO."""
        mps = random_mps(self.N, self.d, self.D, dtype=complex, seed=85, normalize=True)
        mpo = _make_identity_mpo(self.N, self.d)
        lr = self._full_left_sweep(mps, mpo)
        val = self._scalar_from_env(lr[self.N - 1])
        self.assertAlmostEqual(val.imag, 0.0, places=10)
        self.assertGreater(val.real, 0.0)

    # -- <complex | complex H | complex> ----------------------------------

    def test_complex_bra_complex_H_complex_ket(self):
        """<psi|H|psi> via OperatorEnv is real for complex MPS and Hermitian complex H."""
        mps = random_mps(self.N, self.d, self.D, dtype=complex, seed=80, normalize=True)
        mpo = _make_identity_mpo(self.N, self.d, dtype=complex)
        lr = self._full_left_sweep(mps, mpo)
        val = self._scalar_from_env(lr[self.N - 1])
        self.assertAlmostEqual(val.imag, 0.0, places=10)
        self.assertGreater(val.real, 0.0)


# ===========================================================================
# 7. VectorEnv initialisation
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestLRMPSInit(unittest.TestCase):
    """Initialisation checks for VectorEnv."""

    def test_length_mismatch_raises(self):
        """mps1 and mps2 with different lengths raise AssertionError."""
        mps4 = _make_random_mps(4, d=2, D=3, seed=0)
        mps5 = _make_random_mps(5, d=2, D=3, seed=0)
        with self.assertRaises(AssertionError):
            VectorEnv(mps4, mps5)

    def test_stale_window_after_init(self):
        """After __init__ with init_center=p, stale window is [p, p]."""
        mps = _make_random_mps(4, d=2, D=3, seed=1)
        for center in range(4):
            lr = VectorEnv(mps, mps, init_center=center)
            self.assertEqual(lr.centerL, center)
            self.assertEqual(lr.centerR, center)

    def test_boundary_tensors_are_rank2_dim1(self):
        """L0 and R0 must be rank-2 tensors with both bond dimensions equal to 1."""
        mps = _make_random_mps(4, d=2, D=3, seed=2)
        lr = VectorEnv(mps, mps)
        L0 = lr.LR[-1]
        R0 = lr.LR[4]
        # Check rank and dimension.
        self.assertEqual(len(L0.shape()), 2, "L0 must be rank 2")
        self.assertEqual(len(R0.shape()), 2, "R0 must be rank 2")
        self.assertEqual(L0.shape()[0], 1, "L0 dim[0] must be 1")
        self.assertEqual(L0.shape()[1], 1, "L0 dim[1] must be 1")
        self.assertEqual(R0.shape()[0], 1, "R0 dim[0] must be 1")
        self.assertEqual(R0.shape()[1], 1, "R0 dim[1] must be 1")

    def test_boundary_tensor_labels(self):
        """L0 and R0 must carry labels [dn, up]."""
        mps = _make_random_mps(4, d=2, D=3, seed=3)
        lr = VectorEnv(mps, mps)
        self.assertEqual(set(lr.LR[-1].labels()), {"dn", "up"})
        self.assertEqual(set(lr.LR[4].labels()), {"dn", "up"})

    def test_boundary_dtype_promotes_to_complex(self):
        """VectorEnv boundaries become complex if either MPS is complex."""
        mps1 = _make_random_mps(4, d=2, D=3, seed=4, dtype=complex)
        mps2 = _make_random_mps(4, d=2, D=3, seed=5)
        lr = VectorEnv(mps1, mps2)
        self.assertIn("Complex", lr.LR[-1].dtype_str())
        self.assertIn("Complex", lr.LR[4].dtype_str())

    def test_valid_envs_outside_stale_window(self):
        """Environments outside [centerL, centerR] must be non-None after init."""
        mps = _make_random_mps(5, d=2, D=3, seed=4)
        lr = VectorEnv(mps, mps, init_center=2)
        # Valid left envs: -1, 0, 1
        for p in [-1, 0, 1]:
            self.assertIsNotNone(lr.LR[p])
        # Valid right envs: 3, 4, 5
        for p in [3, 4, 5]:
            self.assertIsNotNone(lr.LR[p])


# ===========================================================================
# 8. VectorEnv ↔ Observer integration
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestLRMPSObserver(unittest.TestCase):
    """Integration: MPS tensor updates automatically call VectorEnv.delete."""

    def setUp(self):
        N, d, D = 4, 2, 3
        self.N = N
        self.mps1 = _make_random_mps(N, d, D, seed=70)
        self.mps2 = _make_random_mps(N, d, D, seed=71)
        # init_center=2: LR[0], LR[1] are valid left envs; LR[3] valid right env.
        self.lr = VectorEnv(self.mps1, self.mps2, init_center=2)

    def test_mps1_setitem_fires_delete_on_lr(self):
        """Updating mps1[p] makes VectorEnv[p] stale automatically."""
        # LR[1] is currently valid (left of center 2).
        _ = self.lr[1]
        self.mps1[1] = self.mps1.tensors[1].clone()
        with self.assertRaises(RuntimeError):
            _ = self.lr[1]

    def test_mps2_setitem_fires_delete_on_lr(self):
        """Updating mps2[p] makes VectorEnv[p] stale automatically."""
        _ = self.lr[1]
        self.mps2[1] = self.mps2.tensors[1].clone()
        with self.assertRaises(RuntimeError):
            _ = self.lr[1]

    def test_mps1_setitem_right_env_fires_delete(self):
        """Updating mps1[p] where LR[p] is a right env also expands the stale window."""
        _ = self.lr[3]
        self.mps1[3] = self.mps1.tensors[3].clone()
        with self.assertRaises(RuntimeError):
            _ = self.lr[3]

    def test_gc_of_vec_env_removes_callback_from_mps1(self):
        """When VectorEnv is GC'd, its dead weak-ref is pruned from mps1._callbacks."""
        initial_count = len(self.mps1._callbacks)
        _ = VectorEnv(self.mps1, self.mps2, init_center=1)
        del _
        self.mps1[0] = self.mps1.tensors[0].clone()
        self.assertEqual(len(self.mps1._callbacks), initial_count,
                         "Dead weak-ref from GC'd VectorEnv must be pruned.")

    def test_gc_of_vec_env_removes_callback_from_mps2(self):
        """When VectorEnv is GC'd, its dead weak-ref is pruned from mps2._callbacks."""
        initial_count = len(self.mps2._callbacks)
        _ = VectorEnv(self.mps1, self.mps2, init_center=1)
        del _
        self.mps2[0] = self.mps2.tensors[0].clone()
        self.assertEqual(len(self.mps2._callbacks), initial_count,
                         "Dead weak-ref from GC'd VectorEnv must be pruned.")


# ===========================================================================
# 9. VectorEnv mathematical correctness
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestLRMPSMath(unittest.TestCase):
    """Mathematical correctness of LR overlap environment contractions."""

    def _full_left_sweep(self, mps1, mps2):
        """Return a VectorEnv with all left envs computed."""
        N = len(mps1)
        lr = VectorEnv(mps1, mps2, init_center=0)
        lr.delete(N - 1)     # stale window now covers entire chain
        lr.update_envs(N, N)   # recompute all left envs
        return lr

    def _scalar_from_1x1_tensor(self, t: "cytnx.UniTensor") -> complex:
        """Extract a scalar from a rank-2 tensor with shape [1, 1]."""
        arr = to_numpy_array(t)
        return complex(arr.flat[0])

    def test_self_overlap_gives_one(self):
        """<psi|psi> via full left sweep equals 1.0 for a normalised MPS."""
        N, d, D = 4, 2, 3
        mps = random_mps(N, d, D, seed=20, normalize=True)
        lr = self._full_left_sweep(mps, mps)
        env = lr[N - 1]   # shape [1, 1] after incorporating all N sites
        val = self._scalar_from_1x1_tensor(env)
        self.assertAlmostEqual(val.real, 1.0, places=10,
                               msg="<psi|psi> should be 1 for normalised psi")
        self.assertAlmostEqual(val.imag, 0.0, places=10)

    def test_overlap_matches_mps_inner(self):
        """<mps2|mps1> via VectorEnv matches inner(mps2, mps1)."""
        N, d, D = 4, 2, 3
        mps1 = _make_random_mps(N, d, D, seed=30)
        mps2 = _make_random_mps(N, d, D, seed=31)
        lr = self._full_left_sweep(mps1, mps2)
        env = lr[N - 1]
        val_lr = self._scalar_from_1x1_tensor(env)
        val_ref = complex(inner(mps2, mps1))
        self.assertAlmostEqual(val_lr.real, val_ref.real, places=10)
        self.assertAlmostEqual(val_lr.imag, val_ref.imag, places=10)

    def test_overlap_conjugate_symmetry(self):
        """<psi1|psi2>* == <psi2|psi1> for real-valued MPS."""
        N, d, D = 4, 2, 3
        mps1 = _make_random_mps(N, d, D, seed=40)
        mps2 = _make_random_mps(N, d, D, seed=41)
        lr12 = self._full_left_sweep(mps1, mps2)
        lr21 = self._full_left_sweep(mps2, mps1)
        val12 = self._scalar_from_1x1_tensor(lr12[N - 1])
        val21 = self._scalar_from_1x1_tensor(lr21[N - 1])
        # For real-valued MPS, <psi1|psi2> = <psi2|psi1> (both real).
        self.assertAlmostEqual(val12.real, val21.real, places=10)

    def test_right_env_contraction_matches_inner(self):
        """Full right sweep: LR[0] (incorporating sites N-1..0) matches inner(mps2, mps1).

        After `update_envs(-1, -1)`, LR[0] is the right environment built from the
        full chain.  Contracting it with L0 gives the overlap scalar.
        """
        N, d, D = 4, 2, 3
        mps1 = _make_random_mps(N, d, D, seed=50)
        mps2 = _make_random_mps(N, d, D, seed=51)
        lr = VectorEnv(mps1, mps2, init_center=N - 1)
        # Expand stale window to entire chain, then shrink to [-1, -1].
        lr.delete(0)
        lr.update_envs(-1, -1)   # compute all right envs; stale = [-1, -1]
        env = lr[0]   # right env incorporating sites 0..N-1, shape [1,1]
        val_lr = self._scalar_from_1x1_tensor(env)
        val_ref = complex(inner(mps2, mps1))
        self.assertAlmostEqual(val_lr.real, val_ref.real, places=10)
        self.assertAlmostEqual(val_lr.imag, val_ref.imag, places=10)

    def test_complex_self_overlap_is_real(self):
        """<psi|psi> via VectorEnv is real and positive for a complex normalised MPS.

        If the bra were not conjugated, imaginary parts would not cancel → imag != 0.
        """
        N, d, D = 4, 2, 3
        mps = random_mps(N, d, D, dtype=complex, seed=70, normalize=True)
        lr = self._full_left_sweep(mps, mps)
        val = self._scalar_from_1x1_tensor(lr[N - 1])
        self.assertAlmostEqual(val.imag, 0.0, places=10)
        self.assertGreater(val.real, 0.0)

    def test_complex_conjugate_symmetry(self):
        """<psi1|psi2> = conj(<psi2|psi1>) for complex MPS via VectorEnv."""
        N, d, D = 4, 2, 3
        mps1 = random_mps(N, d, D, dtype=complex, seed=71, normalize=True)
        mps2 = random_mps(N, d, D, dtype=complex, seed=72, normalize=True)
        lr12 = self._full_left_sweep(mps1, mps2)
        lr21 = self._full_left_sweep(mps2, mps1)
        val12 = self._scalar_from_1x1_tensor(lr12[N - 1])
        val21 = self._scalar_from_1x1_tensor(lr21[N - 1])
        self.assertAlmostEqual(val12.real,  val21.real,  places=10)
        self.assertAlmostEqual(val12.imag, -val21.imag,  places=10)

    # -- <real | complex> --------------------------------------------------

    def test_real_bra_complex_ket_matches_inner(self):
        """<real|complex> via VectorEnv matches inner(real, complex)."""
        N, d, D = 4, 2, 3
        mps1 = _make_random_mps(N, d, D, seed=73, dtype=complex)   # ket
        mps2 = _make_random_mps(N, d, D, seed=74)                  # bra (real)
        lr = self._full_left_sweep(mps1, mps2)
        val_lr = self._scalar_from_1x1_tensor(lr[N - 1])
        val_ref = complex(inner(mps2, mps1))
        self.assertAlmostEqual(val_lr.real, val_ref.real, places=10)
        self.assertAlmostEqual(val_lr.imag, val_ref.imag, places=10)

    # -- <complex | real> -------------------------------------------------

    def test_complex_bra_real_ket_matches_inner(self):
        """<complex|real> via VectorEnv matches inner(complex, real)."""
        N, d, D = 4, 2, 3
        mps1 = _make_random_mps(N, d, D, seed=75)                  # ket (real)
        mps2 = _make_random_mps(N, d, D, seed=76, dtype=complex)   # bra
        lr = self._full_left_sweep(mps1, mps2)
        val_lr = self._scalar_from_1x1_tensor(lr[N - 1])
        val_ref = complex(inner(mps2, mps1))
        self.assertAlmostEqual(val_lr.real, val_ref.real, places=10)
        self.assertAlmostEqual(val_lr.imag, val_ref.imag, places=10)

    # -- observer integration ---------------------------------------------

    def test_observer_integration_invalidates_cached_overlap(self):
        """Modifying mps1 via __setitem__ invalidates the cached LR env at that site."""
        N, d, D = 4, 2, 3
        mps1 = _make_random_mps(N, d, D, seed=60)
        mps2 = _make_random_mps(N, d, D, seed=61)
        lr = VectorEnv(mps1, mps2, init_center=2)
        # LR[1] is valid (left of center).
        _ = lr[1]
        # Modify mps1[1] → LR[1] should become stale automatically.
        mps1[1] = mps1.tensors[1].clone()
        with self.assertRaises(RuntimeError):
            _ = lr[1]


# ===========================================================================
# 10. Observer notifications from gauge operations
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestNotifyFromGaugeOps(unittest.TestCase):
    """Gauge-transforming MPS methods must notify registered environment observers.

    Strategy: prime a *valid* LR environment slot, then perform the gauge
    operation.  The slot must become stale automatically (RuntimeError on
    __getitem__) because the underlying MPS tensor changed.

    Covers notification paths that were added as bug fixes:
    - _shift_center_right/left_qr  (called by move_center and orthogonalize)
    - normalize
    - update_sites  (1-site and 2-site variants)
    """

    N, d, D = 4, 2, 3

    def _make_psi(self, seed: int, center: int = 0) -> "MPS":
        psi = _make_random_mps(self.N, self.d, self.D, seed=seed)
        psi.move_center(center)
        return psi

    def _make_lr(self, psi: "MPS", init_center: int = 0) -> "OperatorEnv":
        mpo = _make_identity_mpo(self.N, self.d)
        return OperatorEnv(psi, psi, mpo, init_center=init_center)

    # ------------------------------------------------------------------
    # move_center (→ _shift_center_{right,left}_qr)
    # ------------------------------------------------------------------

    def test_move_center_right_fires_delete_on_right_envs(self):
        """move_center(k) stales all right envs at sites < k.

        After init_center=0, LR[1] and LR[2] are valid right envs.
        move_center(2) performs two QR steps: notifies (0,1) then (1,2),
        expanding the stale window to [0, 2].  LR[3] is untouched.
        """
        psi = self._make_psi(seed=100, center=0)
        lr = self._make_lr(psi, init_center=0)
        _ = lr[1]   # confirm initially accessible
        _ = lr[2]

        psi.move_center(2)

        with self.assertRaises(RuntimeError):
            _ = lr[1]
        with self.assertRaises(RuntimeError):
            _ = lr[2]
        _ = lr[3]   # right of the new center; not touched — must NOT raise

    def test_move_center_left_fires_delete_on_left_envs(self):
        """move_center(k) leftward stales valid left envs between k+1 and N-1.

        After init_center=3, LR[1] and LR[2] are valid left envs.
        move_center(1) performs two QR steps: notifies (2,3) then (1,2),
        expanding the stale window to [1, 3].  LR[0] is untouched.
        """
        psi = self._make_psi(seed=101, center=3)
        lr = self._make_lr(psi, init_center=3)
        _ = lr[1]
        _ = lr[2]

        psi.move_center(1)

        with self.assertRaises(RuntimeError):
            _ = lr[1]
        with self.assertRaises(RuntimeError):
            _ = lr[2]
        _ = lr[0]   # left of the new center; not touched — must NOT raise

    def test_move_center_notifies_both_sites_per_qr_step(self):
        """Each QR step notifies both the source site and its neighbour.

        A single-step move_center(0 → 1) must call delete(0) *and* delete(1).
        """
        psi = self._make_psi(seed=102, center=0)
        counter = _Counter()
        psi.register_callback(counter)

        psi.move_center(1)   # one QR step only

        self.assertIn(0, counter.calls, "source site 0 was not notified")
        self.assertIn(1, counter.calls, "neighbour site 1 was not notified")

    # ------------------------------------------------------------------
    # normalize
    # ------------------------------------------------------------------

    def test_normalize_fires_delete_on_center_site(self):
        """normalize() modifies tensors[center] → delete(center) must fire.

        With init_center=0, LR[1] is a valid right env (right of centerR=0).
        After normalize() with psi.center=1, delete(1) expands centerR to 1,
        making LR[1] stale.
        """
        psi = self._make_psi(seed=103, center=1)
        lr = self._make_lr(psi, init_center=0)   # LR[1] is a valid right env
        _ = lr[1]

        psi.normalize()   # modifies tensors[1]

        with self.assertRaises(RuntimeError):
            _ = lr[1]

    # ------------------------------------------------------------------
    # update_sites (1-site and 2-site)
    # ------------------------------------------------------------------

    def test_update_sites_1site_fires_delete(self):
        """1-site update_sites stales LR[p] and LR[p+1].

        update_sites(0, phi, absorb='right') writes tensors[0] and tensors[1],
        so delete(1) must expand the stale window to include LR[1].
        """
        psi = self._make_psi(seed=104, center=0)
        lr = self._make_lr(psi, init_center=0)   # LR[1] is a valid right env
        _ = lr[1]

        phi = psi.make_phi(0, n=1)
        psi.update_sites(0, phi, absorb="right")

        with self.assertRaises(RuntimeError):
            _ = lr[1]

    def test_update_sites_2site_fires_delete(self):
        """2-site update_sites stales LR[p] and LR[p+1].

        update_sites(0, phi, absorb='right') for a 2-site phi writes both
        tensors[0] and tensors[1], so LR[1] must become stale.
        """
        psi = self._make_psi(seed=105, center=0)
        lr = self._make_lr(psi, init_center=0)   # LR[1] is a valid right env
        _ = lr[1]

        phi = psi.make_phi(0, n=2)
        psi.update_sites(0, phi, absorb="right")

        with self.assertRaises(RuntimeError):
            _ = lr[1]

    # ------------------------------------------------------------------
    # Correctness after invalidation + rebuild
    # ------------------------------------------------------------------

    def test_move_center_then_rebuild_gives_correct_result(self):
        """<psi|I|psi> via OperatorEnv stays 1.0 after move_center + full rebuild.

        move_center changes gauge only; the quantum state is unchanged, so
        ||psi||^2 = 1.  After rebuilding the stale envs from the new tensors,
        the identity-MPO expectation must still equal 1.0.
        """
        psi = self._make_psi(seed=106, center=0)
        lr = self._make_lr(psi, init_center=0)

        # Move center to 3: notifies sites (0,1), (1,2), (2,3) → stale = [0, 3].
        psi.move_center(3)

        # Shrink stale window to [3, 3]: builds LR[0], LR[1], LR[2].
        lr.update_envs(3, 3)
        # Shrink to empty sentinel: builds LR[3].
        lr.update_envs(self.N, self.N - 1)

        # LR[N-1] is the left env through all N sites → scalar <psi|I|psi>.
        env = lr[self.N - 1]
        val = float(to_numpy_array(env).flat[0].real)
        self.assertAlmostEqual(val, 1.0, places=9)


# ===========================================================================
# 11. QN OperatorEnv mathematical correctness
# ===========================================================================


def _qn_heisenberg_mpo(N: int) -> "MPO":
    """QN Heisenberg MPO built via AutoMPO."""
    site = spin_half(qn="Sz")
    ampo = AutoMPO(N, site)
    for i in range(N - 1):
        ampo.add(1.0, "Sz", i, "Sz", i + 1)
        ampo.add(0.5, "Sp", i, "Sm", i + 1)
        ampo.add(0.5, "Sm", i, "Sp", i + 1)
    return ampo.to_mpo()


def _qn_mps(dtype, seed, normalize=True):
    """QN MPS helper: spin-1/2, 4 sites, n_up=2."""
    return random_u1_sz_mps(
        num_sites=4, n_up_total=2, seed=seed, dtype=dtype,
        center=0, normalize=normalize,
    )


@unittest.skipIf(cytnx is None, "cytnx not available")
class TestLRMPOQN(unittest.TestCase):
    """OperatorEnv mathematical correctness with QN tensors."""

    N = 4

    def _full_left_sweep(self, mps, mpo, *, mps2=None):
        """Build OperatorEnv with init_center=0, sweep fully left to expose LR[N-1].

        If `mps2` is given it is used as the bra; otherwise bra = ket = `mps`.
        """
        if mps2 is None:
            mps2 = mps
        N = len(mps)
        lr = OperatorEnv(mps, mps2, mpo, init_center=0)
        lr.delete(N - 1)
        lr.update_envs(N, N)
        return lr

    def _scalar_from_env(self, t: "cytnx.UniTensor") -> complex:
        """Extract a scalar from a rank-3 env tensor with shape [1,1,1]."""
        arr = to_numpy_array(t)
        return complex(arr.flat[0])

    # -- <real | real H | real> -------------------------------------------

    def test_qn_real_bra_real_H_real_ket(self):
        """<psi|H|psi> via OperatorEnv is real for real QN MPS + Heisenberg H."""
        mps = _qn_mps(float, seed=200)
        mpo = _qn_heisenberg_mpo(self.N)
        lr = self._full_left_sweep(mps, mpo)
        val = self._scalar_from_env(lr[self.N - 1])
        self.assertAlmostEqual(val.imag, 0.0, places=10)
        self.assertTrue(np.isfinite(val.real))

    # -- <real | real H | complex> ----------------------------------------

    def test_qn_real_bra_real_H_complex_ket(self):
        """<real|H|complex> via OperatorEnv matches expectation()."""
        bra = _qn_mps(float,   seed=202)
        ket = _qn_mps(complex, seed=203)
        mpo = _qn_heisenberg_mpo(self.N)
        lr = self._full_left_sweep(ket, mpo, mps2=bra)
        val = self._scalar_from_env(lr[self.N - 1])
        ref = complex(expectation(bra, mpo, ket))
        self.assertAlmostEqual(val.real, ref.real, places=10)
        self.assertAlmostEqual(val.imag, ref.imag, places=10)

    # -- <complex | real H | real> ----------------------------------------

    def test_qn_complex_bra_real_H_real_ket(self):
        """<complex|H|real> via OperatorEnv matches expectation()."""
        bra = _qn_mps(complex, seed=204)
        ket = _qn_mps(float,   seed=205)
        mpo = _qn_heisenberg_mpo(self.N)
        lr = self._full_left_sweep(ket, mpo, mps2=bra)
        val = self._scalar_from_env(lr[self.N - 1])
        ref = complex(expectation(bra, mpo, ket))
        self.assertAlmostEqual(val.real, ref.real, places=10)
        self.assertAlmostEqual(val.imag, ref.imag, places=10)

    # -- <complex | real H | complex> -------------------------------------

    def test_qn_complex_bra_real_H_complex_ket(self):
        """<psi|H|psi> is real for complex QN MPS + real Heisenberg H."""
        mps = _qn_mps(complex, seed=210)
        mpo = _qn_heisenberg_mpo(self.N)
        lr = self._full_left_sweep(mps, mpo)
        val = self._scalar_from_env(lr[self.N - 1])
        self.assertAlmostEqual(val.imag, 0.0, places=10)
        self.assertTrue(np.isfinite(val.real))

    # -- <complex | complex H | complex> ----------------------------------

    def test_qn_complex_bra_complex_H_complex_ket(self):
        """<psi|H|psi> is real for complex QN MPS + complex Hermitian H."""
        mps = _qn_mps(complex, seed=212)
        mpo = _qn_heisenberg_mpo(self.N)
        # Cast MPO to complex
        mpo_c = MPO([t.astype(cytnx.Type.ComplexDouble) for t in mpo.tensors])
        lr = self._full_left_sweep(mps, mpo_c)
        val = self._scalar_from_env(lr[self.N - 1])
        self.assertAlmostEqual(val.imag, 0.0, places=10)
        self.assertTrue(np.isfinite(val.real))

    # -- left vs right env consistency ------------------------------------

    def test_qn_left_right_env_consistency(self):
        """Left-env and right-env agree at a center site for QN tensors."""
        p = 2
        mps = _qn_mps(float, seed=220)
        mpo = _qn_heisenberg_mpo(self.N)

        lr_full = self._full_left_sweep(mps, mpo)
        ref_val = self._scalar_from_env(lr_full[self.N - 1])

        lr = OperatorEnv(mps, mps, mpo, init_center=p)
        L = lr[p - 1]
        R = lr[p + 1]

        A  = mps.tensors[p].relabels(["l", "i", "r"], ["_dn", "_i", "dn"])
        W  = mpo.tensors[p].relabels(["l", "ip", "i", "r"], ["_mid", "_ip", "_i", "mid"])
        Ad = mps.tensors[p].Dagger().relabels(["l", "i", "r"], ["_up", "_ip", "up"])

        E = L.relabels(["mid", "dn", "up"], ["_mid", "_dn", "_up"])
        tmp = cytnx.Contract(E, A)
        tmp = cytnx.Contract(tmp, W)
        tmp = cytnx.Contract(tmp, Ad)
        result = cytnx.Contract(tmp, R)

        val = complex(to_numpy_array(result).flat[0])
        self.assertAlmostEqual(val.real, ref_val.real, places=9)
        self.assertAlmostEqual(val.imag, ref_val.imag, places=9)


# ===========================================================================
# 12. QN VectorEnv mathematical correctness
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestLRMPSQN(unittest.TestCase):
    """VectorEnv mathematical correctness with QN tensors."""

    N = 4

    def _full_left_sweep(self, mps1, mps2):
        """Return a VectorEnv with all left envs computed."""
        N = len(mps1)
        lr = VectorEnv(mps1, mps2, init_center=0)
        lr.delete(N - 1)
        lr.update_envs(N, N)
        return lr

    def _scalar_from_1x1_tensor(self, t: "cytnx.UniTensor") -> complex:
        """Extract a scalar from a rank-2 tensor with shape [1, 1]."""
        arr = to_numpy_array(t)
        return complex(arr.flat[0])

    # -- real QN --

    def test_qn_real_self_overlap_gives_one(self):
        """<psi|psi> via VectorEnv equals 1.0 for a normalised real QN MPS."""
        mps = _qn_mps(float, seed=300)
        lr = self._full_left_sweep(mps, mps)
        val = self._scalar_from_1x1_tensor(lr[self.N - 1])
        self.assertAlmostEqual(val.real, 1.0, places=10)
        self.assertAlmostEqual(val.imag, 0.0, places=10)

    def test_qn_real_overlap_matches_inner(self):
        """<mps2|mps1> via VectorEnv matches inner(mps2, mps1) for real QN MPS."""
        mps1 = _qn_mps(float, seed=301)
        mps2 = _qn_mps(float, seed=302)
        lr = self._full_left_sweep(mps1, mps2)
        val_lr = self._scalar_from_1x1_tensor(lr[self.N - 1])
        val_ref = complex(inner(mps2, mps1))
        self.assertAlmostEqual(val_lr.real, val_ref.real, places=10)
        self.assertAlmostEqual(val_lr.imag, val_ref.imag, places=10)

    # -- complex QN --

    def test_qn_complex_self_overlap_is_real_positive(self):
        """<psi|psi> via VectorEnv is real and positive for a complex QN MPS."""
        mps = _qn_mps(complex, seed=310)
        lr = self._full_left_sweep(mps, mps)
        val = self._scalar_from_1x1_tensor(lr[self.N - 1])
        self.assertAlmostEqual(val.imag, 0.0, places=10)
        self.assertGreater(val.real, 0.0)

    def test_qn_complex_overlap_matches_inner(self):
        """<mps2|mps1> via VectorEnv matches inner(mps2, mps1) for complex QN MPS."""
        mps1 = _qn_mps(complex, seed=311)
        mps2 = _qn_mps(complex, seed=312)
        lr = self._full_left_sweep(mps1, mps2)
        val_lr = self._scalar_from_1x1_tensor(lr[self.N - 1])
        val_ref = complex(inner(mps2, mps1))
        self.assertAlmostEqual(val_lr.real, val_ref.real, places=10)
        self.assertAlmostEqual(val_lr.imag, val_ref.imag, places=10)

    def test_qn_complex_conjugate_symmetry(self):
        """<psi1|psi2> = conj(<psi2|psi1>) for complex QN MPS via VectorEnv."""
        mps1 = _qn_mps(complex, seed=320)
        mps2 = _qn_mps(complex, seed=321)
        lr12 = self._full_left_sweep(mps1, mps2)
        lr21 = self._full_left_sweep(mps2, mps1)
        val12 = self._scalar_from_1x1_tensor(lr12[self.N - 1])
        val21 = self._scalar_from_1x1_tensor(lr21[self.N - 1])
        self.assertAlmostEqual(val12.real,  val21.real,  places=10)
        self.assertAlmostEqual(val12.imag, -val21.imag,  places=10)

    # -- <real | complex> (bra=complex, ket=real) --

    def test_qn_real_ket_complex_bra_matches_inner(self):
        """<complex|real> via VectorEnv matches inner(complex, real) for QN MPS."""
        mps1 = _qn_mps(float,   seed=330)   # ket
        mps2 = _qn_mps(complex, seed=331)   # bra
        lr = self._full_left_sweep(mps1, mps2)
        val_lr = self._scalar_from_1x1_tensor(lr[self.N - 1])
        val_ref = complex(inner(mps2, mps1))
        self.assertAlmostEqual(val_lr.real, val_ref.real, places=10)
        self.assertAlmostEqual(val_lr.imag, val_ref.imag, places=10)

    # -- <complex | real> (bra=real, ket=complex) --

    def test_qn_complex_ket_real_bra_matches_inner(self):
        """<real|complex> via VectorEnv matches inner(real, complex) for QN MPS."""
        mps1 = _qn_mps(complex, seed=340)   # ket
        mps2 = _qn_mps(float,   seed=341)   # bra
        lr = self._full_left_sweep(mps1, mps2)
        val_lr = self._scalar_from_1x1_tensor(lr[self.N - 1])
        val_ref = complex(inner(mps2, mps1))
        self.assertAlmostEqual(val_lr.real, val_ref.real, places=10)
        self.assertAlmostEqual(val_lr.imag, val_ref.imag, places=10)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main()
