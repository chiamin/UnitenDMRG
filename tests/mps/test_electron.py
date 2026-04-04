"""Tests for electron (spin-1/2 fermion) site factory."""

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
    from MPS.physical_sites import electron
    from unitensor.core import bond_qnums_at

SKIP = unittest.skipIf(cytnx is None, "cytnx not available")


# ---------------------------------------------------------------------------
# Dense (no QN) factory tests
# ---------------------------------------------------------------------------

@SKIP
class TestElectronDense(unittest.TestCase):

    def setUp(self):
        self.site = electron()

    def test_dim(self):
        self.assertEqual(self.site.dim, 4)
        self.assertEqual(self.site.bond.Nsym(), 0)

    def test_type_name(self):
        self.assertEqual(self.site.type_name, "Electron")

    def test_operators_registered(self):
        for name in ["I", "Cup", "Cupdag", "Cdn", "Cdndag",
                      "Nup", "Ndn", "Ntot", "Sz", "F"]:
            op = self.site.op(name)
            self.assertEqual(op.shape, (4, 4))

    def test_identity(self):
        np.testing.assert_allclose(self.site.op("I"), np.eye(4))

    def test_number_operators(self):
        np.testing.assert_allclose(self.site.op("Nup"),  np.diag([0, 1, 0, 1]))
        np.testing.assert_allclose(self.site.op("Ndn"),  np.diag([0, 0, 1, 1]))
        np.testing.assert_allclose(self.site.op("Ntot"), np.diag([0, 1, 1, 2]))
        np.testing.assert_allclose(self.site.op("Sz"),   np.diag([0, 0.5, -0.5, 0]))

    def test_F_operator(self):
        np.testing.assert_allclose(self.site.op("F"), np.diag([1, -1, -1, 1]))

    def test_Cup_values(self):
        C = self.site.op("Cup")
        # c_up |up> = |0>
        self.assertAlmostEqual(C[0, 1], 1.0)
        # c_up |updn> = +|dn>  (outermost, no sign)
        self.assertAlmostEqual(C[2, 3], 1.0)
        # zeros elsewhere
        self.assertAlmostEqual(C[0, 0], 0.0)
        self.assertAlmostEqual(C[0, 2], 0.0)
        self.assertAlmostEqual(C[1, 3], 0.0)

    def test_Cdn_values(self):
        C = self.site.op("Cdn")
        # c_dn |dn> = |0>
        self.assertAlmostEqual(C[0, 2], 1.0)
        # c_dn |updn> = -|up>  (crosses c†_up)
        self.assertAlmostEqual(C[1, 3], -1.0)

    def test_Cupdag_is_transpose_of_Cup(self):
        np.testing.assert_allclose(self.site.op("Cupdag"),
                                   self.site.op("Cup").T)

    def test_Cdndag_is_transpose_of_Cdn(self):
        np.testing.assert_allclose(self.site.op("Cdndag"),
                                   self.site.op("Cdn").T)

    def test_fermionic_flags(self):
        for name in ["Cup", "Cupdag", "Cdn", "Cdndag"]:
            self.assertTrue(self.site.op_is_fermionic(name), f"{name} should be fermionic")
        for name in ["I", "Nup", "Ndn", "Ntot", "Sz", "F"]:
            self.assertFalse(self.site.op_is_fermionic(name), f"{name} should be bosonic")

    def test_delta_qn_all_zero(self):
        for name in ["I", "Cup", "Cupdag", "Cdn", "Cdndag",
                      "Nup", "Ndn", "Ntot", "Sz", "F"]:
            self.assertEqual(self.site.op_delta_qn(name), [0])


# ---------------------------------------------------------------------------
# On-site anti-commutation relations
# ---------------------------------------------------------------------------

@SKIP
class TestElectronOnSiteAntiCommutation(unittest.TestCase):
    """Verify on-site anti-commutation: {c_s, c†_s'} = delta_{s,s'} I."""

    def setUp(self):
        self.site = electron()

    def _anticommutator(self, A, B):
        return A @ B + B @ A

    def test_Cup_Cupdag_anticommute_to_I(self):
        """{ c_up, c†_up } = I"""
        result = self._anticommutator(self.site.op("Cup"), self.site.op("Cupdag"))
        np.testing.assert_allclose(result, np.eye(4), atol=1e-14)

    def test_Cdn_Cdndag_anticommute_to_I(self):
        """{ c_dn, c†_dn } = I"""
        result = self._anticommutator(self.site.op("Cdn"), self.site.op("Cdndag"))
        np.testing.assert_allclose(result, np.eye(4), atol=1e-14)

    def test_Cup_Cdndag_anticommute_to_zero(self):
        """{ c_up, c†_dn } = 0"""
        result = self._anticommutator(self.site.op("Cup"), self.site.op("Cdndag"))
        np.testing.assert_allclose(result, np.zeros((4, 4)), atol=1e-14)

    def test_Cdn_Cupdag_anticommute_to_zero(self):
        """{ c_dn, c†_up } = 0"""
        result = self._anticommutator(self.site.op("Cdn"), self.site.op("Cupdag"))
        np.testing.assert_allclose(result, np.zeros((4, 4)), atol=1e-14)

    def test_Cup_Cup_anticommute_to_zero(self):
        """{ c_up, c_up } = 0"""
        result = self._anticommutator(self.site.op("Cup"), self.site.op("Cup"))
        np.testing.assert_allclose(result, np.zeros((4, 4)), atol=1e-14)

    def test_Cdn_Cdn_anticommute_to_zero(self):
        """{ c_dn, c_dn } = 0"""
        result = self._anticommutator(self.site.op("Cdn"), self.site.op("Cdn"))
        np.testing.assert_allclose(result, np.zeros((4, 4)), atol=1e-14)

    def test_Cup_Cdn_anticommute_to_zero(self):
        """{ c_up, c_dn } = 0"""
        result = self._anticommutator(self.site.op("Cup"), self.site.op("Cdn"))
        np.testing.assert_allclose(result, np.zeros((4, 4)), atol=1e-14)

    def test_Nup_from_CdagC(self):
        """c†_up c_up = Nup"""
        result = self.site.op("Cupdag") @ self.site.op("Cup")
        np.testing.assert_allclose(result, self.site.op("Nup"), atol=1e-14)

    def test_Ndn_from_CdagC(self):
        """c†_dn c_dn = Ndn"""
        result = self.site.op("Cdndag") @ self.site.op("Cdn")
        np.testing.assert_allclose(result, self.site.op("Ndn"), atol=1e-14)


# ---------------------------------------------------------------------------
# QN mode tests
# ---------------------------------------------------------------------------

@SKIP
class TestElectronQNNtot(unittest.TestCase):

    def setUp(self):
        self.site = electron(qn="Ntot")

    def test_bond(self):
        self.assertEqual(self.site.bond.Nsym(), 1)
        self.assertEqual(bond_qnums_at(self.site.bond, 0), [0])
        self.assertEqual(bond_qnums_at(self.site.bond, 1), [1])
        self.assertEqual(bond_qnums_at(self.site.bond, 2), [1])
        self.assertEqual(bond_qnums_at(self.site.bond, 3), [2])

    def test_delta_qn(self):
        self.assertEqual(self.site.op_delta_qn("Cup"),    [-1])
        self.assertEqual(self.site.op_delta_qn("Cupdag"), [1])
        self.assertEqual(self.site.op_delta_qn("Cdn"),    [-1])
        self.assertEqual(self.site.op_delta_qn("Cdndag"), [1])
        self.assertEqual(self.site.op_delta_qn("Nup"),    [0])
        self.assertEqual(self.site.op_delta_qn("F"),      [0])


@SKIP
class TestElectronQNSz(unittest.TestCase):

    def setUp(self):
        self.site = electron(qn="Sz")

    def test_bond(self):
        self.assertEqual(self.site.bond.Nsym(), 1)
        self.assertEqual(bond_qnums_at(self.site.bond, 0), [0])
        self.assertEqual(bond_qnums_at(self.site.bond, 1), [1])
        self.assertEqual(bond_qnums_at(self.site.bond, 2), [-1])
        self.assertEqual(bond_qnums_at(self.site.bond, 3), [0])

    def test_delta_qn(self):
        self.assertEqual(self.site.op_delta_qn("Cup"),    [-1])
        self.assertEqual(self.site.op_delta_qn("Cupdag"), [1])
        self.assertEqual(self.site.op_delta_qn("Cdn"),    [1])
        self.assertEqual(self.site.op_delta_qn("Cdndag"), [-1])


@SKIP
class TestElectronQNNtotSz(unittest.TestCase):

    def setUp(self):
        self.site = electron(qn="Ntot,Sz")

    def test_bond(self):
        self.assertEqual(self.site.bond.Nsym(), 2)
        self.assertEqual(bond_qnums_at(self.site.bond, 0), [0, 0])
        self.assertEqual(bond_qnums_at(self.site.bond, 1), [1, 1])
        self.assertEqual(bond_qnums_at(self.site.bond, 2), [1, -1])
        self.assertEqual(bond_qnums_at(self.site.bond, 3), [2, 0])

    def test_delta_qn(self):
        self.assertEqual(self.site.op_delta_qn("Cup"),    [-1, -1])
        self.assertEqual(self.site.op_delta_qn("Cupdag"), [1, 1])
        self.assertEqual(self.site.op_delta_qn("Cdn"),    [-1, 1])
        self.assertEqual(self.site.op_delta_qn("Cdndag"), [1, -1])
        self.assertEqual(self.site.op_delta_qn("Nup"),    [0, 0])
        self.assertEqual(self.site.op_delta_qn("F"),      [0, 0])


@SKIP
class TestElectronQNNupNdn(unittest.TestCase):

    def setUp(self):
        self.site = electron(qn="Nup,Ndn")

    def test_bond(self):
        self.assertEqual(self.site.bond.Nsym(), 2)
        self.assertEqual(bond_qnums_at(self.site.bond, 0), [0, 0])
        self.assertEqual(bond_qnums_at(self.site.bond, 1), [1, 0])
        self.assertEqual(bond_qnums_at(self.site.bond, 2), [0, 1])
        self.assertEqual(bond_qnums_at(self.site.bond, 3), [1, 1])

    def test_delta_qn(self):
        self.assertEqual(self.site.op_delta_qn("Cup"),    [-1, 0])
        self.assertEqual(self.site.op_delta_qn("Cupdag"), [1, 0])
        self.assertEqual(self.site.op_delta_qn("Cdn"),    [0, -1])
        self.assertEqual(self.site.op_delta_qn("Cdndag"), [0, 1])


@SKIP
class TestElectronUnknownQNRaises(unittest.TestCase):

    def test_unknown_qn(self):
        with self.assertRaises(ValueError):
            electron(qn="Sx")


if __name__ == "__main__":
    unittest.main()
