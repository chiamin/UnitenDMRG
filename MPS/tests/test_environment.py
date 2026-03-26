"""Unit tests for MPS/dmrg/environment.py.

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
   - Boundary tensors LR[-1]/LR[N] match mpo.L0 / mpo.R0
   - Valid environments outside the stale window are not None

4. OperatorEnv access & bookkeeping  (TestLRMPOAccess)
   - __getitem__ on stale index → RuntimeError
   - __getitem__ out of range → ValueError
   - __getitem__ on valid left/right env succeeds
   - delete(p): stale window expands to include p
   - update_LR: stale window shrinks; previously-stale envs become accessible
   - update_LR with centerL > centerR+1 → ValueError
   - update_LR with out-of-range indices → ValueError
   - update_LR(p, p-1): empty stale window — all sites accessible

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
   - Full left sweep gives <mps1|mps2> matching mps1.inner(mps2)
   - Overlap with itself gives 1.0 for a normalised MPS
"""

from __future__ import annotations

import sys
import unittest
import weakref
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

try:
    import cytnx
except ImportError:
    cytnx = None

if cytnx is not None:
    from MPS.mps import MPS
    from MPS.mpo import MPO
    from MPS.mps_init import random_mps, product_state
    from MPS.uniTensor_utils import to_numpy_array
    from MPS.dmrg.environment import (
        OperatorEnv,
        VectorEnv,
    )


# ===========================================================================
# Helpers shared across test classes
# ===========================================================================

def _make_mps_site(dl: int, d: int, dr: int) -> "cytnx.UniTensor":
    """Create a rank-3 MPS site tensor with labels [l, i, r], filled with ones."""
    arr = np.ones((dl, d, dr), dtype=float)
    u = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=2)
    u.set_labels(["l", "i", "r"])
    return u


def _make_mpo_site(dl: int, d: int, dr: int, arr: "np.ndarray | None" = None
                   ) -> "cytnx.UniTensor":
    """Create a rank-4 MPO site tensor with labels [l, ip, i, r].

    If *arr* is None, fills with ones.  Shape must be (dl, d, d, dr).
    """
    if arr is None:
        arr = np.ones((dl, d, d, dr), dtype=float)
    u = cytnx.UniTensor(cytnx.from_numpy(arr.astype(float)), rowrank=2)
    u.set_labels(["l", "ip", "i", "r"])
    return u


def _make_identity_mpo(N: int, d: int) -> "MPO":
    """Build an identity MPO with virtual bond dimension 1.

    Each site tensor W[0, j, k, 0] = delta_{j,k}, so <psi|MPO|psi> = <psi|psi>.
    """
    tensors = []
    for _ in range(N):
        arr = np.zeros((1, d, d, 1), dtype=float)
        for j in range(d):
            arr[0, j, j, 0] = 1.0
        tensors.append(_make_mpo_site(1, d, 1, arr=arr))
    return MPO(tensors)


def _make_random_mps(N: int, d: int = 2, D: int = 4, seed: int = 0) -> "MPS":
    """Return a normalised random MPS."""
    return random_mps(N, d, D, seed=seed, normalize=True)


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
        """LR[-1] must be mpo.L0 and LR[N] must be mpo.R0 after init."""
        mps = _make_random_mps(4, d=2, D=3, seed=0)
        mpo = _make_identity_mpo(4, d=2)
        lr = OperatorEnv(mps, mps, mpo)
        # Identity check via object identity (same Python object)
        self.assertIs(lr.LR[-1], mpo.L0)
        self.assertIs(lr.LR[4], mpo.R0)

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
    """__getitem__, delete, and update_LR bookkeeping for OperatorEnv."""

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

    def test_update_LR_shrinks_stale_window(self):
        """update_LR(new_left, new_right) shrinks [centerL, centerR] as requested."""
        # Current stale window: [2, 2].  Grow it first by deleting more sites.
        self.lr.delete(0)
        self.lr.delete(4)
        # Now stale window is [0, 4] (whole chain plus boundaries of each side).
        # Update to shrink stale window to [2, 2].
        self.lr.update_LR(2, 2)
        self.assertEqual(self.lr.centerL, 2)
        self.assertEqual(self.lr.centerR, 2)
        # LR[0] and LR[1] should now be recomputed (valid).
        env_left = self.lr[0]
        self.assertIsNotNone(env_left)
        # LR[3] and LR[4] should now be recomputed (valid).
        env_right = self.lr[3]
        self.assertIsNotNone(env_right)

    def test_update_LR_previously_stale_now_accessible(self):
        """After update_LR, sites that were stale and are now outside the window are accessible."""
        # Expand stale window to cover site 1.
        self.lr.delete(1)
        # Rebuild: shrink stale window back to [2, 2].
        self.lr.update_LR(2, 2)
        # LR[1] should now be a freshly computed valid env.
        env = self.lr[1]
        self.assertIsNotNone(env)

    def test_update_LR_invalid_window_raises(self):
        """update_LR with centerL > centerR + 1 raises ValueError."""
        with self.assertRaises(ValueError):
            self.lr.update_LR(4, 2)  # centerL=4 > centerR+1=3

    def test_update_LR_out_of_range_raises(self):
        """update_LR with out-of-range index raises ValueError."""
        with self.assertRaises(ValueError):
            self.lr.update_LR(-2, 2)
        with self.assertRaises(ValueError):
            self.lr.update_LR(1, self.N + 1)

    def test_update_LR_single_site(self):
        """update_LR(p) with one argument sets centerL = centerR = p."""
        self.lr.delete(0)
        self.lr.delete(4)
        self.lr.update_LR(2)  # should default centerR to centerL = 2
        self.assertEqual(self.lr.centerL, 2)
        self.assertEqual(self.lr.centerR, 2)

    def test_update_LR_empty_stale_window(self):
        """update_LR(p, p-1) creates an empty stale window (centerL = centerR + 1).

        After this call, every environment from LR[-1] to LR[N] must be
        accessible (no sites are stale).
        """
        # First expand the stale window to cover the full chain.
        self.lr.delete(0)
        self.lr.delete(self.N - 1)
        # Now shrink to an empty stale window: centerL = N, centerR = N-1
        # i.e. no site is stale.
        self.lr.update_LR(self.N, self.N - 1)
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

    def _full_left_sweep(self, mps, mpo, *, seed=42):
        """Build LR with init_center=0, then sweep fully left to expose LR[N-1]."""
        N = len(mps)
        lr = OperatorEnv(mps, mps, mpo, init_center=0)
        # Expand stale window to entire chain, then shrink to [N, N] so that
        # all left-environments LR[-1..N-1] are computed.
        lr.delete(N - 1)     # stale window now covers [0, N-1]
        lr.update_LR(N, N)   # recompute left envs; stale window shrinks to [N, N]
        return lr

    def _scalar_from_1x1x1_tensor(self, t: "cytnx.UniTensor") -> float:
        """Extract a real scalar from a rank-3 tensor with shape [1,1,1]."""
        arr = to_numpy_array(t)
        return float(arr.flat[0].real)

    def test_identity_mpo_gives_norm_squared(self):
        """<psi|I|psi> computed via full left sweep == psi.norm()^2.

        After a full left sweep, LR[N-1] has labels [mid, dn, up] and shape
        [D_mpo, 1, 1] = [1, 1, 1] for the identity MPO.  Its single element
        must equal ||psi||^2.
        """
        N, d, D = 4, 2, 3
        mps = _make_random_mps(N, d, D, seed=99)
        mpo = _make_identity_mpo(N, d)
        lr = self._full_left_sweep(mps, mpo)
        env = lr[N - 1]                      # LR[3]: shape [1, 1, 1]
        val = self._scalar_from_1x1x1_tensor(env)
        expected = mps.norm() ** 2
        self.assertAlmostEqual(val, expected, places=10,
                               msg="<psi|I|psi> must equal ||psi||^2")

    def test_identity_mpo_normalised_mps_gives_one(self):
        """For a normalised MPS, <psi|I|psi> = 1.0 (within floating-point tolerance)."""
        N, d, D = 5, 2, 4
        mps = random_mps(N, d, D, seed=17, normalize=True)
        mpo = _make_identity_mpo(N, d)
        lr = self._full_left_sweep(mps, mpo)
        env = lr[N - 1]
        val = self._scalar_from_1x1x1_tensor(env)
        self.assertAlmostEqual(val, 1.0, places=10,
                               msg="Normalised MPS: <psi|I|psi> must be 1.0")

    def test_left_and_right_envs_consistent_at_center(self):
        """Left env LR[p-1] and right env LR[p+1] contract to give the same observable.

        For an identity MPO, we can check:
            contract(LR[p-1], psi[p], psi†[p], LR[p+1]) == ||psi||^2
        This verifies that _grow_left and _grow_right produce compatible tensors.
        """
        N, d, D = 4, 2, 3
        p = 2   # center site
        mps = _make_random_mps(N, d, D, seed=55)
        mpo = _make_identity_mpo(N, d)
        lr = OperatorEnv(mps, mps, mpo, init_center=p)

        # Left env at p-1, right env at p+1 — both computed during __init__.
        L = lr[p - 1]   # LR[1]: labels [mid, dn, up]
        R = lr[p + 1]   # LR[3]: labels [mid, dn, up]

        # Build the two-site effective Hamiltonian contraction manually.
        # Contract: L -- psi[p] -- W[p] -- psi†[p] -- R
        A  = mps.tensors[p].relabels(["l", "i", "r"], ["_dn", "_i", "dn"])
        W  = mpo.tensors[p].relabels(["l", "ip", "i", "r"], ["_mid", "_ip", "_i", "mid"])
        Ad = mps.tensors[p].Dagger().relabels(["l", "i", "r"], ["_up", "_ip", "up"])

        E = L.relabels(["mid", "dn", "up"], ["_mid", "_dn", "_up"])
        tmp = cytnx.Contract(E, A)
        tmp = cytnx.Contract(tmp, W)
        tmp = cytnx.Contract(tmp, Ad)
        # tmp has labels [dn, up, mid]; R has labels [mid, dn, up]
        result = cytnx.Contract(tmp, R)

        arr = to_numpy_array(result)
        val = float(arr.flat[0].real)
        expected = mps.norm() ** 2
        self.assertAlmostEqual(val, expected, places=9,
                               msg="Manual two-site contraction should give ||psi||^2")


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
        lr.update_LR(N, N)   # recompute all left envs
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
        """<psi1|psi2> via VectorEnv matches MPS.inner()."""
        N, d, D = 4, 2, 3
        mps1 = _make_random_mps(N, d, D, seed=30)
        mps2 = _make_random_mps(N, d, D, seed=31)
        lr = self._full_left_sweep(mps1, mps2)
        env = lr[N - 1]
        val_lr = self._scalar_from_1x1_tensor(env)
        val_ref = complex(mps1.inner(mps2))
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
        """Full right sweep: LR[0] (incorporating sites N-1..0) matches mps1.inner(mps2).

        After `update_LR(-1, -1)`, LR[0] is the right environment built from the
        full chain.  Contracting it with L0 gives the overlap scalar.
        """
        N, d, D = 4, 2, 3
        mps1 = _make_random_mps(N, d, D, seed=50)
        mps2 = _make_random_mps(N, d, D, seed=51)
        lr = VectorEnv(mps1, mps2, init_center=N - 1)
        # Expand stale window to entire chain, then shrink to [-1, -1].
        lr.delete(0)
        lr.update_LR(-1, -1)   # compute all right envs; stale = [-1, -1]
        env = lr[0]   # right env incorporating sites 0..N-1, shape [1,1]
        val_lr = self._scalar_from_1x1_tensor(env)
        val_ref = complex(mps1.inner(mps2))
        self.assertAlmostEqual(val_lr.real, val_ref.real, places=10)
        self.assertAlmostEqual(val_lr.imag, val_ref.imag, places=10)

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
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main()
