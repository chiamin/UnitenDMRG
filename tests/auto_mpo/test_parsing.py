"""Tests for AutoMPO argument parsing and validation."""

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
    from MPS.physical_sites import spin_half
    from MPS.auto_mpo import AutoMPO

SKIP = unittest.skipIf(cytnx is None, "cytnx not available")


@SKIP
class TestAutoMPOParsing(unittest.TestCase):

    def setUp(self):
        self.site = spin_half()
        self.ampo = AutoMPO(4, self.site)

    def test_add_single_site(self):
        self.ampo.add(1.0, "Sz", 0)   # should not raise

    def test_add_two_site(self):
        self.ampo.add(1.0, "Sz", 0, "Sz", 1)

    def test_add_three_site(self):
        self.ampo.add(1.0, "Sz", 0, "Sz", 1, "Sz", 2)

    def test_descending_site_order_accepted(self):
        """Descending site order should be accepted."""
        self.ampo.add(1.0, "Sz", 1, "Sz", 0)   # should not raise

    def test_same_site_accepted(self):
        """Multiple operators on the same site should be accepted."""
        self.ampo.add(1.0, "Sz", 0, "Sz", 0)   # should not raise

    def test_out_of_range_site(self):
        with self.assertRaises(ValueError):
            self.ampo.add(1.0, "Sz", 5)

    def test_unknown_op_raises(self):
        with self.assertRaises(KeyError):
            self.ampo.add(1.0, "Sx", 0)

    def test_inconsistent_charge_raises(self):
        ampo = AutoMPO(4, spin_half(qn="Sz"))
        ampo.add(1.0, "Sp", 0)          # charge +1
        with self.assertRaises(ValueError):
            ampo.add(1.0, "Sm", 1)      # charge -1 != +1

    def test_no_terms_raises(self):
        with self.assertRaises(RuntimeError):
            self.ampo.to_mpo()


if __name__ == "__main__":
    unittest.main()
