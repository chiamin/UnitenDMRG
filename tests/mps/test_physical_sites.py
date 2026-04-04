"""Tests for PhysicalSite, spin_half, and bond_qnums_at."""

import sys
import unittest
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

try:
    import cytnx
except ImportError:
    cytnx = None

if cytnx is not None:
    from MPS.physical_sites import PhysicalSite, spin_half
    from unitensor.core import bond_qnums_at, _bond_sector_at


SKIP = unittest.skipIf(cytnx is None, "cytnx not available")


# ---------------------------------------------------------------------------
# bond_qnums_at / _bond_sector_at
# ---------------------------------------------------------------------------

@SKIP
class TestBondHelpers(unittest.TestCase):

    def _u1_bond(self):
        # sectors: QN=-1 (dim=1, idx=0), QN=+1 (dim=1, idx=1)
        return cytnx.Bond(cytnx.BD_IN, [[-1], [1]], [1, 1], [cytnx.Symmetry.U1()])

    def _z2_bond(self):
        # sectors: QN=0 (dim=3, idx=0,1,2), QN=1 (dim=1, idx=3)
        return cytnx.Bond(cytnx.BD_IN, [[0], [1]], [3, 1], [cytnx.Symmetry.Zn(2)])

    def _dense_bond(self):
        return cytnx.Bond(2, cytnx.BD_IN)

    def test_bond_qnums_at_u1(self):
        b = self._u1_bond()
        self.assertEqual(bond_qnums_at(b, 0), [-1])
        self.assertEqual(bond_qnums_at(b, 1), [1])

    def test_bond_qnums_at_z2_multidim_sector(self):
        b = self._z2_bond()
        # indices 0,1,2 are in sector QN=0; index 3 in sector QN=1
        self.assertEqual(bond_qnums_at(b, 0), [0])
        self.assertEqual(bond_qnums_at(b, 1), [0])
        self.assertEqual(bond_qnums_at(b, 2), [0])
        self.assertEqual(bond_qnums_at(b, 3), [1])

    def test_bond_qnums_at_dense_returns_empty(self):
        b = self._dense_bond()
        self.assertEqual(bond_qnums_at(b, 0), [])
        self.assertEqual(bond_qnums_at(b, 1), [])

    def test_bond_sector_at_offset(self):
        b = self._z2_bond()
        # sector 0 covers indices 0,1,2 with offsets 0,1,2
        self.assertEqual(_bond_sector_at(b, 0), (0, 0))
        self.assertEqual(_bond_sector_at(b, 1), (0, 1))
        self.assertEqual(_bond_sector_at(b, 2), (0, 2))
        # sector 1 covers index 3 with offset 0
        self.assertEqual(_bond_sector_at(b, 3), (1, 0))

    def test_bond_qnums_at_out_of_range(self):
        b = self._u1_bond()
        with self.assertRaises(IndexError):
            bond_qnums_at(b, 2)


# ---------------------------------------------------------------------------
# PhysicalSite construction
# ---------------------------------------------------------------------------

@SKIP
class TestPhysicalSite(unittest.TestCase):

    def test_rejects_non_bd_in(self):
        bond = cytnx.Bond(cytnx.BD_OUT, [[-1], [1]], [1, 1], [cytnx.Symmetry.U1()])
        with self.assertRaises(ValueError):
            PhysicalSite(bond)

    def test_rejects_non_bond(self):
        with self.assertRaises(TypeError):
            PhysicalSite("not a bond")

    def test_bond_property(self):
        sym = cytnx.Symmetry.U1()
        bond = cytnx.Bond(cytnx.BD_IN, [[-1], [1]], [1, 1], [sym])
        site = PhysicalSite(bond)
        self.assertIs(site.bond, bond)

    def test_type_name(self):
        bond = cytnx.Bond(2, cytnx.BD_IN)
        site = PhysicalSite(bond, type_name="Test")
        self.assertEqual(site.type_name, "Test")


# ---------------------------------------------------------------------------
# spin_half factory
# ---------------------------------------------------------------------------

@SKIP
class TestSpinHalf(unittest.TestCase):

    def test_dense_bond_dim(self):
        site = spin_half()
        self.assertEqual(site.bond.dim(), 2)
        self.assertEqual(site.bond.Nsym(), 0)
        self.assertEqual(site.type_name, "SpinHalf")

    def test_sz_bond(self):
        site = spin_half(qn="Sz")
        self.assertEqual(site.bond.dim(), 2)
        self.assertEqual(site.bond.Nsym(), 1)
        # index 0 = dn = N_up = 0, index 1 = up = N_up = 1
        self.assertEqual(bond_qnums_at(site.bond, 0), [0])
        self.assertEqual(bond_qnums_at(site.bond, 1), [1])

    def test_unknown_qn_raises(self):
        with self.assertRaises(ValueError):
            spin_half(qn="N")

    def test_ops_registered_dense(self):
        import numpy as np
        site = spin_half()
        for name in ["I", "Sz", "Sp", "Sm"]:
            op = site.op(name)
            self.assertIsInstance(op, np.ndarray)
            self.assertEqual(op.shape, (2, 2))

    def test_ops_registered_qn(self):
        # Operators are stored as dense numpy matrices regardless of QN mode.
        import numpy as np
        site = spin_half(qn="Sz")
        for name in ["I", "Sz", "Sp", "Sm"]:
            op = site.op(name)
            self.assertIsInstance(op, np.ndarray)
            self.assertEqual(op.shape, (2, 2))

    def test_delta_qn_dense(self):
        site = spin_half()
        self.assertEqual(site.op_delta_qn("I"),  [0])
        self.assertEqual(site.op_delta_qn("Sz"), [0])
        self.assertEqual(site.op_delta_qn("Sp"), [0])
        self.assertEqual(site.op_delta_qn("Sm"), [0])

    def test_delta_qn_qn(self):
        site = spin_half(qn="Sz")
        self.assertEqual(site.op_delta_qn("I"),  [0])
        self.assertEqual(site.op_delta_qn("Sz"), [0])
        self.assertEqual(site.op_delta_qn("Sp"), [1])
        self.assertEqual(site.op_delta_qn("Sm"), [-1])

    def test_op_missing_raises(self):
        site = spin_half()
        with self.assertRaises(KeyError):
            site.op("Sx")

    def test_op_values_dense(self):
        import numpy as np
        site = spin_half()
        Sz = site.op("Sz")
        np.testing.assert_allclose(Sz, np.diag([0.5, -0.5]))
        Sp = site.op("Sp")
        self.assertAlmostEqual(Sp[1, 0], 1.0)  # Sp|dn>=|up>
        self.assertAlmostEqual(Sp[0, 1], 0.0)

    def test_derive_delta_qn_mixed_raises(self):
        import numpy as np
        from unitensor.core import derive_delta_qn
        site = spin_half(qn="Sz")
        Sx = np.array([[0, 0.5], [0.5, 0]])  # mixes delta_qn +1 and -1
        with self.assertRaises(ValueError):
            derive_delta_qn(Sx, site.bond)


# ---------------------------------------------------------------------------
# product_state — dense
# ---------------------------------------------------------------------------

@SKIP
class TestProductStateDense(unittest.TestCase):

    def setUp(self):
        self.site = spin_half()

    def test_basic_shape(self):
        psi = self.site.product_state([0, 1, 0, 1])
        self.assertEqual(len(psi), 4)
        self.assertEqual(psi.bond_dims, [1, 1, 1, 1, 1])

    def test_norm_is_one(self):
        psi = self.site.product_state([0, 1, 0])
        self.assertAlmostEqual(psi.norm(), 1.0)

    def test_correct_element_set(self):
        import numpy as np
        from unitensor.utils import to_numpy_array
        psi = self.site.product_state([1, 0])
        # site 0: state=1 -> arr[0,1,0]=1
        arr0 = to_numpy_array(psi[0])
        self.assertAlmostEqual(arr0[0, 1, 0], 1.0)
        self.assertAlmostEqual(arr0[0, 0, 0], 0.0)
        # site 1: state=0 -> arr[0,0,0]=1
        arr1 = to_numpy_array(psi[1])
        self.assertAlmostEqual(arr1[0, 0, 0], 1.0)

    def test_center_is_set(self):
        psi = self.site.product_state([0, 1, 0], center=1)
        self.assertEqual(psi.center, 1)

    def test_empty_states_raises(self):
        with self.assertRaises(ValueError):
            self.site.product_state([])

    def test_out_of_range_state_raises(self):
        with self.assertRaises(ValueError):
            self.site.product_state([0, 2])

    def test_invalid_center_raises(self):
        with self.assertRaises(IndexError):
            self.site.product_state([0, 1], center=5)

    def test_dtype_complex(self):
        psi = self.site.product_state([0, 1, 0], dtype=complex)
        for tensor in psi:
            self.assertIn("Complex", tensor.dtype_str())


# ---------------------------------------------------------------------------
# product_state — QN (U1 Sz)
# ---------------------------------------------------------------------------

@SKIP
class TestProductStateQN(unittest.TestCase):

    def setUp(self):
        self.site = spin_half(qn="Sz")

    def test_basic_shape(self):
        psi = self.site.product_state([0, 1, 0, 1])
        self.assertEqual(len(psi), 4)
        self.assertEqual(psi.bond_dims, [1, 1, 1, 1, 1])

    def test_is_blockform(self):
        psi = self.site.product_state([0, 1])
        for tensor in psi:
            self.assertTrue(tensor.is_blockform())

    def test_norm_is_one(self):
        psi = self.site.product_state([0, 1, 0, 1])
        self.assertAlmostEqual(psi.norm(), 1.0)

    def test_virtual_bond_qns_neel(self):
        # Neel [0,1,0,1]: N_up = 0,1,0,1  -> cumulative: 0,1,1,2
        psi = self.site.product_state([0, 1, 0, 1])
        expected = [0, 1, 1, 2]
        for i, exp in enumerate(expected):
            qn = psi[i].bond("r").qnums()[0][0]
            self.assertEqual(qn, exp, f"site {i} right bond QN")

    def test_virtual_bond_qns_all_up(self):
        # All up [1,1,1]: cumulative Sz = +1,+2,+3
        psi = self.site.product_state([1, 1, 1])
        expected = [1, 2, 3]
        for i, exp in enumerate(expected):
            qn = psi[i].bond("r").qnums()[0][0]
            self.assertEqual(qn, exp, f"site {i} right bond QN")

    def test_center_is_set(self):
        psi = self.site.product_state([0, 1, 0], center=2)
        self.assertEqual(psi.center, 2)

    def test_total_qn_neel(self):
        # Neel [0,1,0,1]: total N_up = 0+1+0+1 = 2
        psi = self.site.product_state([0, 1, 0, 1])
        self.assertEqual(psi.total_qn, [2])

    def test_total_qn_all_up(self):
        # All up [1,1,1]: total Sz*2 = 3
        psi = self.site.product_state([1, 1, 1])
        self.assertEqual(psi.total_qn, [3])

    def test_total_qn_dense_returns_empty(self):
        psi = spin_half().product_state([0, 1, 0])
        self.assertEqual(psi.total_qn, [])

    def test_dtype_complex(self):
        psi = self.site.product_state([0, 1, 0], dtype=complex)
        for tensor in psi:
            self.assertIn("Complex", tensor.dtype_str())


if __name__ == "__main__":
    unittest.main()

