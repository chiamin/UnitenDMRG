"""Unit tests for MPS compression algorithms."""

import sys
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent.parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

try:
    import cytnx
except ImportError:  # pragma: no cover
    cytnx = None

if cytnx is not None:
    from MPS.mps_compression import svd_compress_mps, denmat_compress_mps
    from MPS.mps_init import random_mps
    from MPS.mps_operations import inner
else:  # pragma: no cover
    def _missing_cytnx(*_args, **_kwargs):
        raise RuntimeError("cytnx is required")

    svd_compress_mps = _missing_cytnx
    denmat_compress_mps = _missing_cytnx
    random_mps = _missing_cytnx
    inner = _missing_cytnx


@unittest.skipIf(cytnx is None, "cytnx is required for UniTensor tests")
class TestMPSCompression(unittest.TestCase):
    """Tests for `svd_compress_mps` and `denmat_compress_mps`."""

    def test_svd_compress_mps_no_truncation(self) -> None:
        """svd_compress_mps with no limit should preserve the state up to machine precision."""
        psi = random_mps(num_sites=4, phys_dim=2, bond_dim=4, seed=0, normalize=True)
        phi = svd_compress_mps(psi)
        self.assertAlmostEqual(abs(inner(phi, psi)), 1.0, places=10)

    def test_svd_compress_mps_reduces_bond_dim(self) -> None:
        """svd_compress_mps should enforce max_dim on all bonds."""
        psi = random_mps(num_sites=4, phys_dim=2, bond_dim=4, seed=1, normalize=True)
        phi = svd_compress_mps(psi, max_dim=2)
        self.assertTrue(all(d <= 2 for d in phi.bond_dims))

    def test_svd_compress_mps_returns_new_mps(self) -> None:
        """svd_compress_mps should not modify the input MPS."""
        psi = random_mps(num_sites=4, phys_dim=2, bond_dim=4, seed=2, normalize=True)
        orig_dims = list(psi.bond_dims)
        svd_compress_mps(psi, max_dim=2)
        self.assertEqual(psi.bond_dims, orig_dims)

    def test_svd_compress_mps_center(self) -> None:
        """svd_compress_mps should set center to site 0."""
        psi = random_mps(num_sites=4, phys_dim=2, bond_dim=4, seed=3, normalize=True)
        phi = svd_compress_mps(psi, max_dim=2)
        self.assertEqual(phi.center, 0)

    def test_svd_compress_mps_complex(self) -> None:
        """svd_compress_mps should work on complex MPS and preserve the state when no truncation."""
        psi = random_mps(num_sites=4, phys_dim=2, bond_dim=4, seed=4, dtype=complex, normalize=True)
        phi = svd_compress_mps(psi)
        self.assertAlmostEqual(abs(inner(phi, psi)), 1.0, places=10)

    def test_svd_compress_mps_single_site(self) -> None:
        """svd_compress_mps on a single-site MPS should return an equivalent MPS."""
        psi = random_mps(num_sites=1, phys_dim=2, bond_dim=1, seed=5, normalize=True)
        phi = svd_compress_mps(psi, max_dim=2)
        self.assertEqual(len(phi), 1)
        self.assertAlmostEqual(abs(inner(phi, psi)), 1.0, places=10)

    def test_denmat_compress_mps_no_truncation(self) -> None:
        """denmat_compress_mps with no limit should preserve the state up to machine precision."""
        psi = random_mps(num_sites=4, phys_dim=2, bond_dim=4, seed=0, normalize=True)
        phi = denmat_compress_mps(psi)
        self.assertAlmostEqual(abs(inner(phi, psi)), 1.0, places=10)

    def test_denmat_compress_mps_reduces_bond_dim(self) -> None:
        """denmat_compress_mps should enforce max_dim on all bonds."""
        psi = random_mps(num_sites=4, phys_dim=2, bond_dim=4, seed=1, normalize=True)
        phi = denmat_compress_mps(psi, max_dim=2)
        self.assertTrue(all(d <= 2 for d in phi.bond_dims))

    def test_denmat_compress_mps_returns_new_mps(self) -> None:
        """denmat_compress_mps should not modify the input MPS."""
        psi = random_mps(num_sites=4, phys_dim=2, bond_dim=4, seed=2, normalize=True)
        orig_dims = list(psi.bond_dims)
        denmat_compress_mps(psi, max_dim=2)
        self.assertEqual(psi.bond_dims, orig_dims)

    def test_denmat_compress_mps_center(self) -> None:
        """denmat_compress_mps should set center to site 0."""
        psi = random_mps(num_sites=4, phys_dim=2, bond_dim=4, seed=3, normalize=True)
        phi = denmat_compress_mps(psi, max_dim=2)
        self.assertEqual(phi.center, 0)

    def test_denmat_compress_mps_complex(self) -> None:
        """denmat_compress_mps should work on complex MPS and preserve the state when no truncation."""
        psi = random_mps(num_sites=4, phys_dim=2, bond_dim=4, seed=4, dtype=complex, normalize=True)
        phi = denmat_compress_mps(psi)
        self.assertAlmostEqual(abs(inner(phi, psi)), 1.0, places=10)

    def test_denmat_compress_mps_single_site(self) -> None:
        """denmat_compress_mps on a single-site MPS should return an equivalent MPS."""
        psi = random_mps(num_sites=1, phys_dim=2, bond_dim=1, seed=5, normalize=True)
        phi = denmat_compress_mps(psi, max_dim=2)
        self.assertEqual(len(phi), 1)
        self.assertAlmostEqual(abs(inner(phi, psi)), 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
