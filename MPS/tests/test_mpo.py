"""Unit tests for MPO module.

Coverage:
- `mpo.py`: MPO API, label validation, bond checks, boundary tensors, compression

Tests are skipped automatically if `cytnx` is unavailable.
"""

import sys
import unittest
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
    from MPS.mpo import MPO, MPO_SITE_LABELS, assert_mpo_site_labels
    from MPS.uniTensor_utils import to_numpy_array
else:
    cytnx = None
    MPO = object

    def assert_mpo_site_labels(*_args, **_kwargs):
        raise RuntimeError("cytnx is required")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mpo_site(
    dl: int, d: int, dr: int, start: float = 1.0
) -> "cytnx.UniTensor":
    """Rank-4 MPO site tensor with labels l/ip/i/r.

    Physical bonds 'i' and 'ip' both have dimension d.
    Shape: (dl, d, d, dr) → axes order [l, ip, i, r].
    """
    arr = np.arange(start, start + dl * d * d * dr, dtype=float).reshape(dl, d, d, dr)
    u = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=2)
    u.set_labels(["l", "ip", "i", "r"])
    return u


def _make_mpo(num_sites: int = 3, d: int = 2, D: int = 3) -> "MPO":
    """Valid open-boundary MPO with uniform physical and virtual dimensions."""
    tensors = [
        _make_mpo_site(D, d, D, start=float(p * 100 + 1))
        for p in range(num_sites)
    ]
    return MPO(tensors)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@unittest.skipIf(cytnx is None, "cytnx is required for MPO tests")
class TestMPOSiteLabelValidator(unittest.TestCase):
    """Tests for assert_mpo_site_labels."""

    def test_accepts_valid_rank4_labels(self) -> None:
        t = _make_mpo_site(3, 2, 3)
        assert_mpo_site_labels(t, 0)  # should not raise

    def test_rejects_wrong_rank(self) -> None:
        arr = np.ones((3, 2, 3), dtype=float)
        u = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=2)
        u.set_labels(["l", "i", "r"])
        with self.assertRaises(ValueError):
            assert_mpo_site_labels(u, 0)

    def test_rejects_wrong_labels(self) -> None:
        t = _make_mpo_site(3, 2, 3)
        t.set_labels(["a", "b", "c", "d"])
        with self.assertRaises(ValueError):
            assert_mpo_site_labels(t, 0)


@unittest.skipIf(cytnx is None, "cytnx is required for MPO tests")
class TestMPOInit(unittest.TestCase):
    """Tests for MPO.__init__ and boundary tensor construction."""

    def test_rejects_empty_list(self) -> None:
        with self.assertRaises(ValueError):
            MPO([])

    def test_rejects_wrong_type(self) -> None:
        with self.assertRaises(TypeError):
            MPO([np.zeros((3, 2, 2, 3))])

    def test_accepts_valid_mpo(self) -> None:
        mpo = _make_mpo()
        self.assertEqual(len(mpo), 3)
        self.assertIsNotNone(mpo.L0)
        self.assertIsNotNone(mpo.R0)

    def test_l0_r0_labels(self) -> None:
        mpo = _make_mpo(D=3)
        self.assertEqual(set(mpo.L0.labels()), {"mid", "up", "dn"})
        self.assertEqual(set(mpo.R0.labels()), {"mid", "up", "dn"})

    def test_l0_r0_up_dn_dim_one(self) -> None:
        mpo = _make_mpo(D=3)
        self.assertEqual(mpo.L0.bond("up").dim(), 1)
        self.assertEqual(mpo.L0.bond("dn").dim(), 1)
        self.assertEqual(mpo.R0.bond("up").dim(), 1)
        self.assertEqual(mpo.R0.bond("dn").dim(), 1)

    def test_l0_boundary_value(self) -> None:
        D = 3
        mpo = _make_mpo(D=D)
        arr = to_numpy_array(mpo.L0)
        self.assertAlmostEqual(float(arr[D - 1, 0, 0]), 1.0)
        self.assertAlmostEqual(float(np.sum(arr)), 1.0)

    def test_r0_boundary_value(self) -> None:
        D = 3
        mpo = _make_mpo(D=D)
        arr = to_numpy_array(mpo.R0)
        self.assertAlmostEqual(float(arr[0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(np.sum(arr)), 1.0)

    def test_l0_mid_dim_matches_mpo_left_bond(self) -> None:
        D = 4
        mpo = _make_mpo(D=D)
        self.assertEqual(mpo.L0.bond("mid").dim(), D)

    def test_r0_mid_dim_matches_mpo_right_bond(self) -> None:
        D = 4
        mpo = _make_mpo(D=D)
        self.assertEqual(mpo.R0.bond("mid").dim(), D)

    def test_custom_l0_r0_accepted(self) -> None:
        D = 3
        tensors = [_make_mpo_site(D, 2, D, start=1.0)]
        arr_l = np.zeros((D, 1, 1), dtype=float)
        arr_l[0, 0, 0] = 1.0  # lower-triangular convention
        L0 = cytnx.UniTensor(cytnx.from_numpy(arr_l), rowrank=1)
        L0.set_labels(["mid", "up", "dn"])
        arr_r = np.zeros((D, 1, 1), dtype=float)
        arr_r[D - 1, 0, 0] = 1.0
        R0 = cytnx.UniTensor(cytnx.from_numpy(arr_r), rowrank=1)
        R0.set_labels(["mid", "up", "dn"])
        mpo = MPO(tensors, L0=L0, R0=R0)
        self.assertAlmostEqual(float(to_numpy_array(mpo.L0)[0, 0, 0]), 1.0)

    def test_partial_custom_boundary_raises(self) -> None:
        D = 3
        tensors = [_make_mpo_site(D, 2, D)]
        arr_l = np.zeros((D, 1, 1), dtype=float)
        arr_l[D - 1, 0, 0] = 1.0
        L0 = cytnx.UniTensor(cytnx.from_numpy(arr_l), rowrank=1)
        L0.set_labels(["mid", "up", "dn"])
        with self.assertRaises(ValueError):
            MPO(tensors, L0=L0)  # R0 missing


@unittest.skipIf(cytnx is None, "cytnx is required for MPO tests")
class TestMPOValidateBonds(unittest.TestCase):
    """Tests for MPO._validate_bonds."""

    def test_valid_mpo_passes(self) -> None:
        mpo = _make_mpo()
        mpo._validate_bonds()  # should not raise

    def test_physical_bond_mismatch_raises(self) -> None:
        # Build a site where 'i' and 'ip' have different dims.
        arr = np.ones((3, 2, 3, 3), dtype=float)  # ip=2, i=3 → mismatch
        u = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=2)
        u.set_labels(["l", "ip", "i", "r"])
        with self.assertRaises(ValueError):
            MPO([u])

    def test_virtual_bond_mismatch_raises(self) -> None:
        t0 = _make_mpo_site(3, 2, 3)   # right bond dim = 3
        t1 = _make_mpo_site(2, 2, 2)   # left bond dim = 2  ← mismatch
        with self.assertRaises(ValueError):
            MPO([t0, t1])


@unittest.skipIf(cytnx is None, "cytnx is required for MPO tests")
class TestMPOSequenceProtocol(unittest.TestCase):
    """Tests for MPO sequence interface and repr."""

    def test_len(self) -> None:
        self.assertEqual(len(_make_mpo(num_sites=4)), 4)

    def test_iter(self) -> None:
        mpo = _make_mpo(num_sites=3)
        self.assertEqual(sum(1 for _ in mpo), 3)

    def test_getitem_returns_unitensor_with_correct_labels(self) -> None:
        mpo = _make_mpo()
        t = mpo[0]
        self.assertIsInstance(t, cytnx.UniTensor)
        self.assertEqual(set(t.labels()), {"l", "ip", "i", "r"})

    def test_repr_contains_expected_fields(self) -> None:
        mpo = _make_mpo()
        r = repr(mpo)
        self.assertIn("phys_dims", r)
        self.assertIn("mpo_dims", r)


@unittest.skipIf(cytnx is None, "cytnx is required for MPO tests")
class TestMPOProperties(unittest.TestCase):
    """Tests for MPO.phys_dims, mpo_dims, and copy."""

    def test_phys_dims(self) -> None:
        mpo = _make_mpo(num_sites=3, d=2, D=3)
        self.assertEqual(mpo.phys_dims, [2, 2, 2])

    def test_mpo_dims_length_and_values(self) -> None:
        mpo = _make_mpo(num_sites=3, d=2, D=3)
        dims = mpo.mpo_dims
        self.assertEqual(len(dims), 4)       # num_sites + 1
        self.assertEqual(dims[0], 3)          # bond('l') of site 0
        self.assertEqual(dims[-1], 3)         # bond('r') of last site

    def test_copy_tensors_are_independent(self) -> None:
        mpo = _make_mpo()
        mpo2 = mpo.copy()
        self.assertIsNot(mpo2[0], mpo[0])

    def test_copy_has_same_phys_and_mpo_dims(self) -> None:
        mpo = _make_mpo(num_sites=4, d=3, D=2)
        mpo2 = mpo.copy()
        self.assertEqual(mpo2.phys_dims, mpo.phys_dims)
        self.assertEqual(mpo2.mpo_dims, mpo.mpo_dims)


@unittest.skipIf(cytnx is None, "cytnx is required for MPO tests")
class TestMPOCompressBond(unittest.TestCase):
    """Tests for MPO.compress_bond."""

    def test_compress_bond_reduces_dim(self) -> None:
        mpo = _make_mpo(num_sites=3, d=2, D=4)
        kept, dw = mpo.compress_bond(0, max_dim=2, cutoff=0.0, absorb="right")
        self.assertLessEqual(kept, 2)
        self.assertGreaterEqual(dw, 0.0)

    def test_compress_bond_neighbor_dims_match(self) -> None:
        mpo = _make_mpo(num_sites=3, d=2, D=4)
        mpo.compress_bond(0, max_dim=2, cutoff=0.0)
        self.assertEqual(mpo[0].bond("r").dim(), mpo[1].bond("l").dim())

    def test_compress_bond_preserves_site_labels(self) -> None:
        mpo = _make_mpo(num_sites=3, d=2, D=4)
        mpo.compress_bond(1, max_dim=2, cutoff=0.0)
        self.assertEqual(set(mpo[1].labels()), {"l", "ip", "i", "r"})
        self.assertEqual(set(mpo[2].labels()), {"l", "ip", "i", "r"})

    def test_compress_bond_absorb_left(self) -> None:
        mpo = _make_mpo(num_sites=3, d=2, D=4)
        kept, dw = mpo.compress_bond(0, max_dim=2, cutoff=0.0, absorb="left")
        self.assertLessEqual(kept, 2)
        self.assertEqual(mpo[0].bond("r").dim(), mpo[1].bond("l").dim())

    def test_compress_bond_out_of_range(self) -> None:
        mpo = _make_mpo(num_sites=3)
        with self.assertRaises(IndexError):
            mpo.compress_bond(-1, max_dim=2)
        with self.assertRaises(IndexError):
            mpo.compress_bond(len(mpo) - 1, max_dim=2)

    def test_compress_bond_invalid_absorb(self) -> None:
        mpo = _make_mpo()
        with self.assertRaises(ValueError):
            mpo.compress_bond(0, max_dim=2, absorb="both")

    def test_compress_bond_zero_max_dim_raises(self) -> None:
        mpo = _make_mpo()
        with self.assertRaises(ValueError):
            mpo.compress_bond(0, max_dim=0)

    def test_compress_bond_updates_l0_r0(self) -> None:
        mpo = _make_mpo(num_sites=2, d=2, D=4)
        mpo.compress_bond(0, max_dim=2, cutoff=0.0)
        # L0 mid dim should still match site-0 left bond
        self.assertEqual(mpo.L0.bond("mid").dim(), mpo[0].bond("l").dim())
        self.assertEqual(mpo.R0.bond("mid").dim(), mpo[-1].bond("r").dim())


if __name__ == "__main__":
    unittest.main()
