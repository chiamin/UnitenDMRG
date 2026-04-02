"""Unit tests for MPS compression algorithms."""

import sys
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent.parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

import numpy as np

try:
    import cytnx
except ImportError:  # pragma: no cover
    cytnx = None

if cytnx is not None:
    from MPS.mps_compression import svd_compress_mps
    from MPS.mps_init import random_mps
    from MPS.mps_operations import inner
    from tests.helpers.mps_test_cases import random_u1_sz_mps
else:  # pragma: no cover
    def _missing_cytnx(*_args, **_kwargs):
        raise RuntimeError("cytnx is required")

    svd_compress_mps = _missing_cytnx
    random_mps = _missing_cytnx
    inner = _missing_cytnx
    random_u1_sz_mps = _missing_cytnx


@unittest.skipIf(cytnx is None, "cytnx is required for UniTensor tests")
class TestMPSCompression(unittest.TestCase):
    """Tests for `svd_compress_mps`."""

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


@unittest.skipIf(cytnx is None, "cytnx is required for UniTensor tests")
class TestQNSVDCompression(unittest.TestCase):
    """QN MPS tests for `svd_compress_mps`."""

    # Use 8-site spin-1/2 U(1) with n_up=4 → max bond dim 5, enough to truncate.
    NUM_SITES = 8
    N_UP = 4

    def test_svd_compress_qn_real_no_truncation(self) -> None:
        """svd_compress_mps preserves a QN real MPS when no truncation is applied."""
        psi = random_u1_sz_mps(self.NUM_SITES, self.N_UP, seed=100, dtype=float, center=0, normalize=True)
        phi = svd_compress_mps(psi)
        self.assertAlmostEqual(abs(inner(phi, psi)), 1.0, places=10)

    def test_svd_compress_qn_real_reduces_bond_dim(self) -> None:
        """svd_compress_mps enforces max_dim on QN real MPS."""
        psi = random_u1_sz_mps(self.NUM_SITES, self.N_UP, seed=101, dtype=float, center=0, normalize=True)
        phi = svd_compress_mps(psi, max_dim=2)
        self.assertTrue(all(d <= 2 for d in phi.bond_dims))

    def test_svd_compress_qn_real_returns_new_mps(self) -> None:
        """svd_compress_mps does not modify the input QN real MPS."""
        psi = random_u1_sz_mps(self.NUM_SITES, self.N_UP, seed=102, dtype=float, center=0, normalize=True)
        orig_dims = list(psi.bond_dims)
        svd_compress_mps(psi, max_dim=2)
        self.assertEqual(psi.bond_dims, orig_dims)

    def test_svd_compress_qn_real_center(self) -> None:
        """svd_compress_mps sets center to site 0 for QN real MPS."""
        psi = random_u1_sz_mps(self.NUM_SITES, self.N_UP, seed=103, dtype=float, center=0, normalize=True)
        phi = svd_compress_mps(psi, max_dim=2)
        self.assertEqual(phi.center, 0)

    def test_svd_compress_qn_complex_no_truncation(self) -> None:
        """svd_compress_mps preserves a QN complex MPS when no truncation is applied."""
        psi = random_u1_sz_mps(self.NUM_SITES, self.N_UP, seed=104, dtype=complex, center=0, normalize=True)
        phi = svd_compress_mps(psi)
        self.assertAlmostEqual(abs(inner(phi, psi)), 1.0, places=10)

    def test_svd_compress_qn_complex_reduces_bond_dim(self) -> None:
        """svd_compress_mps enforces max_dim on QN complex MPS."""
        psi = random_u1_sz_mps(self.NUM_SITES, self.N_UP, seed=105, dtype=complex, center=0, normalize=True)
        phi = svd_compress_mps(psi, max_dim=2)
        self.assertTrue(all(d <= 2 for d in phi.bond_dims))

    def test_svd_compress_qn_complex_center(self) -> None:
        """svd_compress_mps sets center to site 0 for QN complex MPS."""
        psi = random_u1_sz_mps(self.NUM_SITES, self.N_UP, seed=106, dtype=complex, center=0, normalize=True)
        phi = svd_compress_mps(psi, max_dim=2)
        self.assertEqual(phi.center, 0)


# denmat_compress_mps is commented out until cytnx fixes the "svd-aux-qnums"
# bug: linalg.Svd produces wrong auxiliary bond qnums when row bonds have
# mixed directions.  The density matrix built by denmat_compress_mps has mixed
# bond directions; SVD on rho.Dagger() (needed for correct output directions)
# triggers the bug and segfaults.  The function code is correct — only the
# cytnx backend is broken.
# See _internal/CYTNX_BUGS.md (svd-aux-qnums) and _internal/TODO.md.
# Uncomment denmat_compress_mps and remove this skip once cytnx fixes the bug.
@unittest.skipIf(cytnx is None, "cytnx is required for UniTensor tests")
@unittest.skip("denmat_compress_mps disabled: cytnx bug svd-aux-qnums")
class TestDenmatCompression(unittest.TestCase):
    """Tests for `denmat_compress_mps` (dense + QN)."""

    # -- dense -----------------------------------------------------------------

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

    # -- QN --------------------------------------------------------------------

    NUM_SITES = 8
    N_UP = 4

    def test_denmat_compress_qn_real_no_truncation(self) -> None:
        """denmat_compress_mps preserves a QN real MPS when no truncation is applied."""
        psi = random_u1_sz_mps(self.NUM_SITES, self.N_UP, seed=110, dtype=float, center=0, normalize=True)
        phi = denmat_compress_mps(psi)
        self.assertAlmostEqual(abs(inner(phi, psi)), 1.0, places=10)

    def test_denmat_compress_qn_real_reduces_bond_dim(self) -> None:
        """denmat_compress_mps enforces max_dim on QN real MPS."""
        psi = random_u1_sz_mps(self.NUM_SITES, self.N_UP, seed=111, dtype=float, center=0, normalize=True)
        phi = denmat_compress_mps(psi, max_dim=2)
        self.assertTrue(all(d <= 2 for d in phi.bond_dims))

    def test_denmat_compress_qn_real_returns_new_mps(self) -> None:
        """denmat_compress_mps does not modify the input QN real MPS."""
        psi = random_u1_sz_mps(self.NUM_SITES, self.N_UP, seed=112, dtype=float, center=0, normalize=True)
        orig_dims = list(psi.bond_dims)
        denmat_compress_mps(psi, max_dim=2)
        self.assertEqual(psi.bond_dims, orig_dims)

    def test_denmat_compress_qn_real_center(self) -> None:
        """denmat_compress_mps sets center to site 0 for QN real MPS."""
        psi = random_u1_sz_mps(self.NUM_SITES, self.N_UP, seed=113, dtype=float, center=0, normalize=True)
        phi = denmat_compress_mps(psi, max_dim=2)
        self.assertEqual(phi.center, 0)

    def test_denmat_compress_qn_complex_no_truncation(self) -> None:
        """denmat_compress_mps preserves a QN complex MPS when no truncation is applied."""
        psi = random_u1_sz_mps(self.NUM_SITES, self.N_UP, seed=114, dtype=complex, center=0, normalize=True)
        phi = denmat_compress_mps(psi)
        self.assertAlmostEqual(abs(inner(phi, psi)), 1.0, places=10)

    def test_denmat_compress_qn_complex_reduces_bond_dim(self) -> None:
        """denmat_compress_mps enforces max_dim on QN complex MPS."""
        psi = random_u1_sz_mps(self.NUM_SITES, self.N_UP, seed=115, dtype=complex, center=0, normalize=True)
        phi = denmat_compress_mps(psi, max_dim=2)
        self.assertTrue(all(d <= 2 for d in phi.bond_dims))

    def test_denmat_compress_qn_complex_center(self) -> None:
        """denmat_compress_mps sets center to site 0 for QN complex MPS."""
        psi = random_u1_sz_mps(self.NUM_SITES, self.N_UP, seed=116, dtype=complex, center=0, normalize=True)
        phi = denmat_compress_mps(psi, max_dim=2)
        self.assertEqual(phi.center, 0)


if __name__ == "__main__":
    unittest.main()
