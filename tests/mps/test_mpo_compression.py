"""Unit tests for MPO compression algorithms."""

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
    from MPS.mpo import MPO
    from MPS.mpo_compression import svd_compress_mpo
    from MPS.auto_mpo import AutoMPO
    from MPS.physical_sites import spin_half
    from tests.helpers.mpo_utils import mpo_full_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mpo_site(dl, d, dr, start=1.0):
    """Rank-4 dense MPO site tensor with labels l/ip/i/r."""
    arr = np.arange(start, start + dl * d * d * dr, dtype=float).reshape(dl, d, d, dr)
    u = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=2)
    u.set_labels(["l", "ip", "i", "r"])
    return u


def _make_mpo(num_sites=4, d=2, D=5):
    """Dense open-boundary MPO with large bond dim (easy to compress)."""
    tensors = []
    for p in range(num_sites):
        dl = 1 if p == 0 else D
        dr = 1 if p == num_sites - 1 else D
        tensors.append(_make_mpo_site(dl, d, dr, start=float(p * 100 + 1)))
    return MPO(tensors)


def _heisenberg_mpo_qn(N):
    """QN Heisenberg MPO via AutoMPO + spin_half(qn="Sz")."""
    site = spin_half(qn="Sz")
    ampo = AutoMPO(N, site)
    for i in range(N - 1):
        ampo.add(0.5, "Sp", i, "Sm", i + 1)
        ampo.add(0.5, "Sm", i, "Sp", i + 1)
        ampo.add(1.0, "Sz", i, "Sz", i + 1)
    return ampo.to_mpo()


# ---------------------------------------------------------------------------
# Dense tests
# ---------------------------------------------------------------------------

@unittest.skipIf(cytnx is None, "cytnx is required")
class TestSVDCompressMPO(unittest.TestCase):
    """Tests for `svd_compress_mpo` with dense MPO."""

    def test_no_truncation_preserves_mpo(self):
        """Without truncation, the full matrix should be preserved."""
        H = _make_mpo(num_sites=4, d=2, D=5)
        H2 = svd_compress_mpo(H)
        np.testing.assert_allclose(mpo_full_matrix(H2), mpo_full_matrix(H), atol=1e-10)

    def test_reduces_bond_dim(self):
        """max_dim should limit all internal bond dimensions."""
        H = _make_mpo(num_sites=4, d=2, D=5)
        H2 = svd_compress_mpo(H, max_dim=2)
        for d in H2.mpo_dims[1:-1]:
            self.assertLessEqual(d, 2)

    def test_does_not_modify_input(self):
        """The input MPO must not be changed."""
        H = _make_mpo(num_sites=4, d=2, D=5)
        orig_dims = list(H.mpo_dims)
        svd_compress_mpo(H, max_dim=2)
        self.assertEqual(H.mpo_dims, orig_dims)

    def test_preserves_site_labels(self):
        """Every site in the result must have labels {l, ip, i, r}."""
        H = _make_mpo(num_sites=4, d=2, D=5)
        H2 = svd_compress_mpo(H, max_dim=2)
        for p in range(len(H2)):
            self.assertEqual(set(H2[p].labels()), {"l", "ip", "i", "r"})

    def test_endpoint_bonds_dim_one(self):
        """Endpoint bonds must remain dim=1 after compression."""
        H = _make_mpo(num_sites=4, d=2, D=5)
        H2 = svd_compress_mpo(H, max_dim=2)
        self.assertEqual(H2[0].bond("l").dim(), 1)
        self.assertEqual(H2[-1].bond("r").dim(), 1)

    def test_single_site_mpo(self):
        """Single-site MPO should be returned unchanged."""
        t = _make_mpo_site(1, 2, 1)
        H = MPO([t])
        H2 = svd_compress_mpo(H)
        np.testing.assert_allclose(mpo_full_matrix(H2), mpo_full_matrix(H), atol=1e-10)

    def test_complex_mpo_no_truncation(self):
        """Complex MPO should be preserved without truncation."""
        H = _make_mpo(num_sites=4, d=2, D=5)
        tensors = []
        for t in H:
            arr = t.get_block().numpy().astype(complex) + 0.1j
            u = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=2)
            u.set_labels(["l", "ip", "i", "r"])
            tensors.append(u)
        Hc = MPO(tensors)
        Hc2 = svd_compress_mpo(Hc)
        np.testing.assert_allclose(mpo_full_matrix(Hc2), mpo_full_matrix(Hc), atol=1e-10)


# ---------------------------------------------------------------------------
# QN tests
# ---------------------------------------------------------------------------

@unittest.skipIf(cytnx is None, "cytnx is required")
class TestSVDCompressMPOQN(unittest.TestCase):
    """Tests for `svd_compress_mpo` with QN (U(1) Sz) MPO."""

    NUM_SITES = 6

    def test_qn_no_truncation_preserves_mpo(self):
        """QN MPO without truncation should give the same full matrix."""
        H = _heisenberg_mpo_qn(self.NUM_SITES)
        H2 = svd_compress_mpo(H)
        np.testing.assert_allclose(mpo_full_matrix(H2), mpo_full_matrix(H), atol=1e-10)

    def test_qn_reduces_bond_dim(self):
        """max_dim should reduce QN MPO bond dimensions."""
        H = _heisenberg_mpo_qn(self.NUM_SITES)
        orig_max = max(H.mpo_dims[1:-1])
        H2 = svd_compress_mpo(H, max_dim=3)
        new_max = max(H2.mpo_dims[1:-1])
        self.assertLess(new_max, orig_max)

    def test_qn_does_not_modify_input(self):
        """The input QN MPO must not be changed."""
        H = _heisenberg_mpo_qn(self.NUM_SITES)
        orig_dims = list(H.mpo_dims)
        svd_compress_mpo(H, max_dim=2)
        self.assertEqual(H.mpo_dims, orig_dims)

    def test_qn_preserves_site_labels(self):
        """Every site in the QN result must have labels {l, ip, i, r}."""
        H = _heisenberg_mpo_qn(self.NUM_SITES)
        H2 = svd_compress_mpo(H, max_dim=2)
        for p in range(len(H2)):
            self.assertEqual(set(H2[p].labels()), {"l", "ip", "i", "r"})

    def test_qn_endpoint_bonds_dim_one(self):
        """QN endpoint bonds must remain dim=1."""
        H = _heisenberg_mpo_qn(self.NUM_SITES)
        H2 = svd_compress_mpo(H, max_dim=2)
        self.assertEqual(H2[0].bond("l").dim(), 1)
        self.assertEqual(H2[-1].bond("r").dim(), 1)

    def test_qn_complex_no_truncation(self):
        """Complex QN MPO should be preserved without truncation."""
        H = _heisenberg_mpo_qn(self.NUM_SITES)
        Hc = MPO([t.astype(cytnx.Type.ComplexDouble) for t in H.tensors])
        Hc2 = svd_compress_mpo(Hc)
        np.testing.assert_allclose(mpo_full_matrix(Hc2), mpo_full_matrix(Hc), atol=1e-10)

    def test_qn_complex_reduces_bond_dim(self):
        """max_dim should reduce complex QN MPO bond dimensions."""
        H = _heisenberg_mpo_qn(self.NUM_SITES)
        Hc = MPO([t.astype(cytnx.Type.ComplexDouble) for t in H.tensors])
        orig_max = max(Hc.mpo_dims[1:-1])
        Hc2 = svd_compress_mpo(Hc, max_dim=3)
        new_max = max(Hc2.mpo_dims[1:-1])
        self.assertLess(new_max, orig_max)


if __name__ == "__main__":
    unittest.main()
