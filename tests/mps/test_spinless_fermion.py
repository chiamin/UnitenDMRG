"""Tests for spinless_fermion site factory."""

import sys
import unittest
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

try:
    import cytnx
    import numpy as np
except ImportError:
    cytnx = None

if cytnx is not None:
    from MPS.physical_sites import spinless_fermion
    from unitensor.core import bond_qnums_at

SKIP = unittest.skipIf(cytnx is None, "cytnx not available")


@SKIP
class TestSpinlessFermionDense(unittest.TestCase):

    def setUp(self):
        self.site = spinless_fermion()

    def test_bond_dim(self):
        self.assertEqual(self.site.dim, 2)
        self.assertEqual(self.site.bond.Nsym(), 0)

    def test_type_name(self):
        self.assertEqual(self.site.type_name, "SpinlessFermion")

    def test_operators_registered(self):
        for name in ["I", "N", "C", "Cdag", "F"]:
            op = self.site.op(name)
            self.assertEqual(op.shape, (2, 2))

    def test_operator_values(self):
        I = self.site.op("I")
        np.testing.assert_allclose(I, np.eye(2))

        N = self.site.op("N")
        np.testing.assert_allclose(N, np.diag([0, 1]))

        C = self.site.op("C")
        self.assertAlmostEqual(C[0, 1], 1.0)   # c|1>=|0>
        self.assertAlmostEqual(C[1, 0], 0.0)

        Cdag = self.site.op("Cdag")
        self.assertAlmostEqual(Cdag[1, 0], 1.0)   # c+|0>=|1>
        self.assertAlmostEqual(Cdag[0, 1], 0.0)

        F = self.site.op("F")
        np.testing.assert_allclose(F, np.diag([1, -1]))

    def test_fermionic_flags(self):
        self.assertFalse(self.site.op_is_fermionic("I"))
        self.assertFalse(self.site.op_is_fermionic("N"))
        self.assertTrue(self.site.op_is_fermionic("C"))
        self.assertTrue(self.site.op_is_fermionic("Cdag"))
        self.assertFalse(self.site.op_is_fermionic("F"))

    def test_delta_qn_all_zero(self):
        for name in ["I", "N", "C", "Cdag", "F"]:
            self.assertEqual(self.site.op_delta_qn(name), [0])


@SKIP
class TestSpinlessFermionU1(unittest.TestCase):

    def setUp(self):
        self.site = spinless_fermion(qn="N")

    def test_bond_qn(self):
        self.assertEqual(self.site.bond.Nsym(), 1)
        self.assertEqual(bond_qnums_at(self.site.bond, 0), [0])
        self.assertEqual(bond_qnums_at(self.site.bond, 1), [1])

    def test_delta_qn(self):
        self.assertEqual(self.site.op_delta_qn("I"), [0])
        self.assertEqual(self.site.op_delta_qn("N"), [0])
        self.assertEqual(self.site.op_delta_qn("C"), [-1])
        self.assertEqual(self.site.op_delta_qn("Cdag"), [1])
        self.assertEqual(self.site.op_delta_qn("F"), [0])


@SKIP
class TestSpinlessFermionZ2(unittest.TestCase):

    def setUp(self):
        self.site = spinless_fermion(qn="parity")

    def test_bond_qn(self):
        self.assertEqual(self.site.bond.Nsym(), 1)
        self.assertEqual(bond_qnums_at(self.site.bond, 0), [0])
        self.assertEqual(bond_qnums_at(self.site.bond, 1), [1])

    def test_delta_qn_z2_normalized(self):
        self.assertEqual(self.site.op_delta_qn("I"), [0])
        self.assertEqual(self.site.op_delta_qn("N"), [0])
        self.assertEqual(self.site.op_delta_qn("C"), [1])     # -1 mod 2 = 1
        self.assertEqual(self.site.op_delta_qn("Cdag"), [1])
        self.assertEqual(self.site.op_delta_qn("F"), [0])

    def test_unknown_qn_raises(self):
        with self.assertRaises(ValueError):
            spinless_fermion(qn="Sz")


if __name__ == "__main__":
    unittest.main()
