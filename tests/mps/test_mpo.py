"""Unit tests for the MPO module.

Coverage:
- MPO: API, label validation, bond checks, endpoint dim=1 enforcement, compression

Tests are skipped automatically if `cytnx` is unavailable.
"""

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
    """Valid open-boundary MPO with endpoint bonds having dim=1.

    Site 0 has left bond dim=1; site num_sites-1 has right bond dim=1;
    bulk bonds have dimension D.
    """
    tensors = []
    for p in range(num_sites):
        dl = 1 if p == 0 else D
        dr = 1 if p == num_sites - 1 else D
        tensors.append(_make_mpo_site(dl, d, dr, start=float(p * 100 + 1)))
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
    """Tests for MPO.__init__ and endpoint bond dim=1 enforcement."""

    def test_rejects_empty_list(self) -> None:
        with self.assertRaises(ValueError):
            MPO([])

    def test_rejects_wrong_type(self) -> None:
        with self.assertRaises(TypeError):
            MPO([np.zeros((3, 2, 2, 3))])

    def test_accepts_valid_mpo(self) -> None:
        mpo = _make_mpo()
        self.assertEqual(len(mpo), 3)

    def test_endpoint_bonds_dim_one(self) -> None:
        mpo = _make_mpo(D=3)
        self.assertEqual(mpo[0].bond("l").dim(), 1)
        self.assertEqual(mpo[-1].bond("r").dim(), 1)

    def test_non_dim1_left_endpoint_raises(self) -> None:
        """W[0]['l'] with dim > 1 must raise ValueError at construction."""
        t = _make_mpo_site(2, 2, 1)   # left dim=2, right dim=1
        with self.assertRaises(ValueError):
            MPO([t])

    def test_non_dim1_right_endpoint_raises(self) -> None:
        """W[-1]['r'] with dim > 1 must raise ValueError at construction."""
        t = _make_mpo_site(1, 2, 2)   # left dim=1, right dim=2
        with self.assertRaises(ValueError):
            MPO([t])


@unittest.skipIf(cytnx is None, "cytnx is required for MPO tests")
class TestMPOValidateBonds(unittest.TestCase):
    """Tests for MPO._validate_bonds."""

    def test_valid_mpo_passes(self) -> None:
        mpo = _make_mpo()
        mpo._validate_bonds()  # should not raise

    def test_physical_bond_mismatch_raises(self) -> None:
        # Build a site where "i" and "ip" have different dims.
        arr = np.ones((3, 2, 3, 3), dtype=float)  # ip=2, i=3 → mismatch
        u = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=2)
        u.set_labels(["l", "ip", "i", "r"])
        with self.assertRaises(ValueError):
            MPO([u])

    def test_virtual_bond_mismatch_raises(self) -> None:
        t0 = _make_mpo_site(1, 2, 3)   # left dim=1 (valid endpoint), right dim=3
        t1 = _make_mpo_site(2, 2, 1)   # left dim=2 (mismatch), right dim=1 (valid endpoint)
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
        self.assertEqual(dims[0], 1)          # bond("l") of site 0 must be dim=1
        self.assertEqual(dims[-1], 1)         # bond("r") of last site must be dim=1
        self.assertEqual(dims[1], 3)          # bulk bond dim = D

    def test_copy_tensors_are_independent(self) -> None:
        mpo = _make_mpo()
        mpo2 = mpo.copy()
        self.assertIsNot(mpo2[0], mpo[0])

    def test_copy_has_same_phys_and_mpo_dims(self) -> None:
        mpo = _make_mpo(num_sites=4, d=3, D=2)
        mpo2 = mpo.copy()
        self.assertEqual(mpo2.phys_dims, mpo.phys_dims)
        self.assertEqual(mpo2.mpo_dims, mpo.mpo_dims)

    def test_is_complex_false_for_real_mpo(self) -> None:
        """Real-valued MPO should report is_complex == False."""
        mpo = _make_mpo()
        self.assertFalse(mpo.is_complex)

    def test_is_complex_true_for_complex_mpo(self) -> None:
        """Complex-valued MPO should report is_complex == True."""
        mpo = _make_mpo()
        tensors = []
        for t in mpo:
            arr = t.get_block().numpy().astype(complex, copy=True)
            u = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=2)
            u.set_labels(["l", "ip", "i", "r"])
            tensors.append(u)
        mpo_c = MPO(tensors)
        self.assertTrue(mpo_c.is_complex)


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


if __name__ == "__main__":
    unittest.main()
